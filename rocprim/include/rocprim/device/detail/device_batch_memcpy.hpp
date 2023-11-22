/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2023, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_BATCH_MEMCPY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_BATCH_MEMCPY_HPP_

#include "rocprim/device/config_types.hpp"
#include "rocprim/device/detail/device_scan_common.hpp"
#include "rocprim/device/detail/lookback_scan_state.hpp"
#include "rocprim/device/device_scan.hpp"

#include "rocprim/block/block_exchange.hpp"
#include "rocprim/block/block_load.hpp"
#include "rocprim/block/block_load_func.hpp"
#include "rocprim/block/block_run_length_decode.hpp"
#include "rocprim/block/block_scan.hpp"
#include "rocprim/block/block_store.hpp"
#include "rocprim/block/block_store_func.hpp"

#include "rocprim/thread/thread_load.hpp"
#include "rocprim/thread/thread_search.hpp"
#include "rocprim/thread/thread_store.hpp"

#include "rocprim/detail/temp_storage.hpp"
#include "rocprim/detail/various.hpp"
#include "rocprim/functional.hpp"
#include "rocprim/intrinsics.hpp"
#include "rocprim/intrinsics/thread.hpp"

#include "rocprim/config.hpp"

#include <hip/hip_runtime.h>

#include <stdint.h>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
namespace batch_memcpy
{
enum class size_class
{
    tlev = 0,
    wlev = 1,
    blev = 2,
    num_size_classes,
};

template<uint32_t MaxItemValue, typename BackingUnitType = uint32_t>
struct counter
{
private:
    static constexpr int32_t num_items = static_cast<int32_t>(size_class::num_size_classes);
    BackingUnitType          data[num_items];

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE uint32_t get(size_class index) const
    {
        return data[static_cast<uint32_t>(index)];
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void add(size_class index, uint32_t value)
    {
        data[static_cast<uint32_t>(index)] += value;
    }

    ROCPRIM_DEVICE counter operator+(const counter& other) const
    {
        counter result{};

#pragma unroll
        for(uint32_t i = 0; i < num_items; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }
};

template<class Offset>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static uint8_t read_byte(void* buffer_src, Offset offset)
{
    return rocprim::thread_load<rocprim::cache_load_modifier::load_cs>(
        reinterpret_cast<uint8_t*>(buffer_src) + offset);
}

template<class Offset>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    write_byte(void* buffer_dst, Offset offset, uint8_t value)
{
    rocprim::thread_store<rocprim::cache_store_modifier::store_cs>(
        reinterpret_cast<uint8_t*>(buffer_dst) + offset,
        value);
}

template<class VectorType>
struct aligned_ranges
{
    VectorType* out_begin;
    VectorType* out_end;

    const uint8_t* in_begin;
    const uint8_t* in_end;
};

/// \brief Gives a a pair of ranges (in_* and out_*) that are contained in in_begin and
/// out_begin of a given size such that the returned out range aligns with the given vector
/// type.
template<class VectorType>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static aligned_ranges<VectorType>
               get_aligned_ranges(const void* in_begin, void* out_begin, size_t num_bytes)
{
    uint8_t*       out_ptr = static_cast<uint8_t*>(out_begin);
    const uint8_t* in_ptr  = static_cast<const uint8_t*>(in_begin);

    auto* out_aligned_begin = detail::cast_align_up<VectorType*>(out_ptr);
    auto* out_aligned_end   = detail::cast_align_down<VectorType*>(out_ptr + num_bytes);

    auto           begin_offset     = reinterpret_cast<uint8_t*>(out_aligned_begin) - out_ptr;
    auto           end_offset       = reinterpret_cast<uint8_t*>(out_aligned_end) - out_ptr;
    const uint8_t* in_aligned_begin = in_ptr + begin_offset;
    const uint8_t* in_aligned_end   = in_ptr + end_offset;

    return aligned_ranges<VectorType>{out_aligned_begin,
                                      out_aligned_end,
                                      in_aligned_begin,
                                      in_aligned_end};
}

template<class T, class S>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static T funnel_shift_r(T lo, T hi, S shift)
{
    constexpr uint32_t bit_size = sizeof(T) * 8;
    return (hi << (bit_size - shift)) | lo >> shift;
}

template<class Offset>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void vectorized_copy_bytes(const void* input_buffer,
                                                                      void*       output_buffer,
                                                                      Offset      num_bytes,
                                                                      Offset      offset = 0)
{
    using vector_type                      = uint4;
    constexpr uint32_t ints_in_vector_type = sizeof(uint4) / sizeof(uint32_t);

    constexpr auto warp_size = rocprim::device_warp_size();
    const auto     rank      = rocprim::detail::block_thread_id<0>() % warp_size;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(input_buffer) + offset;
    uint8_t*       dst = reinterpret_cast<uint8_t*>(output_buffer) + offset;

    const uint8_t* in_ptr  = src;
    uint8_t*       out_ptr = dst;

    const auto aligned = batch_memcpy::get_aligned_ranges<vector_type>(src, dst, num_bytes);

    // If no aligned range, copy byte-by-byte and early exit
    if(aligned.out_end <= aligned.out_begin)
    {
        for(uint32_t i = rank; i < num_bytes; i += warp_size)
        {
            out_ptr[i] = in_ptr[i];
        }
        return;
    }

    out_ptr += rank;
    in_ptr += rank;

    // Ensure that all pointers are in aligned range
    while(out_ptr < reinterpret_cast<uint8_t*>(aligned.out_begin))
    {
        *out_ptr = *in_ptr;
        out_ptr += warp_size;
        in_ptr += warp_size;
    }

    // This can be outside the while block since 'warp_size % ints_in_vector_type' always is '0'
    static_assert(warp_size % ints_in_vector_type == 0, "Warp size is not a multiple of 4");

    in_ptr                          = aligned.in_begin + rank * sizeof(vector_type);
    const uint32_t  in_offset       = (reinterpret_cast<size_t>(in_ptr) % ints_in_vector_type);
    vector_type*    aligned_out_ptr = aligned.out_begin + rank;
    const uint32_t* aligned_in_ptr  = reinterpret_cast<const uint32_t*>(in_ptr - in_offset);

    // Copy elements in aligned range
    if(in_offset == 0)
    {
        // No offset, can do cacheline-aligned to cacheline-aligned copy
        while(aligned_out_ptr < aligned.out_end)
        {
            vector_type data = vector_type{aligned_in_ptr[0],
                                           aligned_in_ptr[1],
                                           aligned_in_ptr[2],
                                           aligned_in_ptr[3]};
            *aligned_out_ptr = data;
            aligned_in_ptr += warp_size * sizeof(vector_type) / sizeof(uint32_t);
            aligned_out_ptr += warp_size;
        }
    }
    else
    {
        while(aligned_out_ptr < aligned.out_end)
        {
            union
            {
                vector_type result;
                uint32_t    bytes[5];
            } data;

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < 5; ++i)
            {
                data.bytes[i] = aligned_in_ptr[i];
            }

            // Reads are offset to our cache aligned writes so we need to shift bytes over.
            // AMD has no intrinsic for funner shift, hence the manual implementation.
            // Perhaps a better cacheline-aligned to byte-aligned copy method can be used here.
            const uint32_t shift = in_offset * 8 /* bits per byte */;
            data.result.x        = funnel_shift_r(data.bytes[0], data.bytes[1], shift);
            data.result.y        = funnel_shift_r(data.bytes[1], data.bytes[2], shift);
            data.result.z        = funnel_shift_r(data.bytes[2], data.bytes[3], shift);
            data.result.w        = funnel_shift_r(data.bytes[3], data.bytes[4], shift);

            *aligned_out_ptr = data.result;
            aligned_in_ptr += warp_size * sizeof(vector_type) / sizeof(uint32_t);
            aligned_out_ptr += warp_size;
        }
    }

    out_ptr = reinterpret_cast<uint8_t*>(aligned.out_end) + rank;
    in_ptr  = aligned.in_end + rank;

    // Copy non-aligned tail
    while(out_ptr < dst + num_bytes)
    {
        *out_ptr = *in_ptr;
        out_ptr += warp_size;
        in_ptr += warp_size;
    }
}
} // namespace batch_memcpy

template<class Config, class InputBufferItType, class OutputBufferItType, class BufferSizeItType>
struct batch_memcpy_impl
{
    using input_buffer_type  = typename std::iterator_traits<InputBufferItType>::value_type;
    using output_buffer_type = typename std::iterator_traits<OutputBufferItType>::value_type;
    using buffer_size_type   = typename std::iterator_traits<BufferSizeItType>::value_type;

    using input_type = typename std::iterator_traits<input_buffer_type>::value_type;

    // top level policy
    static constexpr uint32_t block_size            = Config::non_blev_block_size;
    static constexpr uint32_t buffers_per_thread    = Config::non_blev_buffers_per_thread;
    static constexpr uint32_t tlev_bytes_per_thread = Config::tlev_bytes_per_thread;

    static constexpr uint32_t blev_block_size       = Config::blev_block_size;
    static constexpr uint32_t blev_bytes_per_thread = Config::blev_bytes_per_thread;

    static constexpr uint32_t wlev_size_threshold = Config::wlev_size_threshold;
    static constexpr uint32_t blev_size_threshold = Config::blev_size_threshold;

    static constexpr uint32_t tlev_buffers_per_thread = buffers_per_thread;
    static constexpr uint32_t blev_buffers_per_thread = buffers_per_thread;

    static constexpr uint32_t buffers_per_block = buffers_per_thread * block_size;

    // Offset over buffers.
    using buffer_offset_type = uint32_t;

    // Offset over tiles.
    using tile_offset_type = uint32_t;

    // The byte offset within a thread-level buffer. Must fit at least `wlev_size_threshold`.
    static_assert(wlev_size_threshold < std::numeric_limits<uint16_t>::max(),
                  "wlev_size_threshhold too large (should fit in 16 bits)");
    using tlev_byte_offset_type =
        typename std::conditional<(wlev_size_threshold < 256), uint8_t, uint16_t>::type;

    struct copyable_buffers
    {
        InputBufferItType  srcs;
        OutputBufferItType dsts;
        BufferSizeItType   sizes;
    };

    struct copyable_blev_buffers
    {
        InputBufferItType  srcs;
        OutputBufferItType dsts;
        BufferSizeItType   sizes;
        tile_offset_type*  offsets;
    };

private:
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static batch_memcpy::size_class
                   get_size_class(buffer_size_type size)
    {
        auto size_class = batch_memcpy::size_class::tlev;
        size_class      = size > wlev_size_threshold ? batch_memcpy::size_class::wlev : size_class;
        size_class      = size > blev_size_threshold ? batch_memcpy::size_class::blev : size_class;
        return size_class;
    }

    struct zipped_tlev_byte_assignment
    {
        buffer_offset_type    tile_buffer_id;
        tlev_byte_offset_type buffer_byte_offset;
    };

    struct buffer_tuple
    {
        tlev_byte_offset_type size;
        buffer_offset_type    buffer_id;
    };

    using size_class_counter = batch_memcpy::counter<buffers_per_block>;

    using blev_buffer_scan_state_type = rocprim::detail::lookback_scan_state<buffer_offset_type>;
    using blev_block_scan_state_type  = rocprim::detail::lookback_scan_state<tile_offset_type>;

    using block_size_scan_type            = rocprim::block_scan<size_class_counter, block_size>;
    using block_blev_tile_count_scan_type = rocprim::block_scan<tile_offset_type, block_size>;

    using block_run_length_decode_type = rocprim::block_run_length_decode<buffer_offset_type,
                                                                          block_size,
                                                                          tlev_buffers_per_thread,
                                                                          tlev_bytes_per_thread>;

    using block_exchange_tlev_type
        = rocprim::block_exchange<zipped_tlev_byte_assignment, block_size, tlev_bytes_per_thread>;

    using buffer_load_type = rocprim::block_load<buffer_size_type,
                                                 block_size,
                                                 buffers_per_thread,
                                                 rocprim::block_load_method::block_load_striped>;

    using blev_buffer_scan_prefix_callback_type
        = rocprim::detail::offset_lookback_scan_prefix_op<buffer_offset_type,
                                                          blev_buffer_scan_state_type,
                                                          rocprim::plus<buffer_offset_type>>;

    using blev_block_scan_prefix_callback_type
        = rocprim::detail::offset_lookback_scan_prefix_op<tile_offset_type,
                                                          blev_block_scan_state_type,
                                                          rocprim::plus<tile_offset_type>>;

    struct non_blev_memcpy
    {
        struct storage
        {
            buffer_tuple buffers_by_size_class[buffers_per_block];

            // This value is passed from analysis to prepare_blev.
            buffer_offset_type blev_buffer_offset;

            union shared_t
            {
                union analysis_t
                {
                    typename buffer_load_type::storage_type     load_storage;
                    typename block_size_scan_type::storage_type size_scan_storage;
                    typename blev_buffer_scan_prefix_callback_type::storage_type
                        buffer_scan_callback;
                } analysis;

                struct prepare_blev_t
                {
                    typename blev_block_scan_prefix_callback_type::storage_type block_scan_callback;
                    typename block_blev_tile_count_scan_type::storage_type      block_scan_storage;
                } prepare_blev;

                struct copy_tlev_t
                {
                    typename block_run_length_decode_type::storage_type rld_storage;
                    typename block_exchange_tlev_type::storage_type     block_exchange_storage;
                } copy_tlev;
            } shared;
        };

        using storage_type = rocprim::detail::raw_storage<storage>;

        ROCPRIM_DEVICE ROCPRIM_INLINE non_blev_memcpy() {}

        ROCPRIM_DEVICE ROCPRIM_INLINE static size_class_counter get_buffer_size_class_histogram(
            const buffer_size_type (&buffer_sizes)[buffers_per_thread])
        {
            size_class_counter counters{};

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < buffers_per_thread; ++i)
            {
                auto size_class = get_size_class(buffer_sizes[i]);
                counters.add(size_class, buffer_sizes[i] > 0 ? 1 : 0);
            }
            return counters;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE void
            partition_buffer_by_size(const buffer_size_type (&buffer_sizes)[buffers_per_thread],
                                     size_class_counter counters,
                                     buffer_tuple (&buffers_by_size_class)[buffers_per_block])
        {
            const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();

            buffer_offset_type           buffer_id     = flat_block_thread_id;
            constexpr buffer_offset_type buffer_stride = block_size;

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < buffers_per_thread; ++i, buffer_id += buffer_stride)
            {
                if(buffer_sizes[i] <= 0)
                {
                    continue;
                }

                const auto     size_class   = get_size_class(buffer_sizes[i]);
                const uint32_t write_offset = counters.get(size_class);
                buffers_by_size_class[write_offset]
                    = buffer_tuple{static_cast<tlev_byte_offset_type>(buffer_sizes[i]), buffer_id};

                counters.add(size_class, 1);
            }
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE void
            prepare_blev_buffers(typename storage::shared_t::prepare_blev_t& blev_storage,
                                 buffer_tuple*                               buffer_by_size_class,
                                 copyable_buffers                            buffers,
                                 buffer_offset_type                          num_blev_buffers,
                                 copyable_blev_buffers                       blev_buffers,
                                 buffer_offset_type                          tile_buffer_offset,
                                 blev_block_scan_state_type                  blev_block_scan_state,
                                 buffer_offset_type                          tile_id)
        {
            const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();

            tile_offset_type tile_offsets[blev_buffers_per_thread];
            auto             blev_buffer_offset = flat_block_thread_id * blev_buffers_per_thread;

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < blev_buffers_per_thread; ++i)
            {
                if(blev_buffer_offset < num_blev_buffers)
                {
                    auto tile_buffer_id = buffer_by_size_class[blev_buffer_offset].buffer_id;
                    tile_offsets[i]
                        = rocprim::detail::ceiling_div(buffers.sizes[tile_buffer_id],
                                                       blev_block_size * blev_bytes_per_thread);
                }
                else
                {
                    tile_offsets[i] = 0;
                }
                ++blev_buffer_offset;
            }

            // Convert tile counts into tile offsets.
            if(tile_id == 0)
            {
                tile_offset_type tile_aggregate{};
                block_blev_tile_count_scan_type{}.exclusive_scan(tile_offsets,
                                                                 tile_offsets,
                                                                 tile_aggregate,
                                                                 tile_aggregate,
                                                                 blev_storage.block_scan_storage,
                                                                 rocprim::plus<tile_offset_type>{});
                if(flat_block_thread_id == 0)
                {
                    blev_block_scan_state.set_complete(0, tile_aggregate);
                }
            }
            else
            {
                blev_block_scan_prefix_callback_type blev_tile_prefix_op{
                    tile_id,
                    blev_block_scan_state,
                    blev_storage.block_scan_callback,
                    rocprim::plus<tile_offset_type>{}};
                block_blev_tile_count_scan_type{}.exclusive_scan(tile_offsets,
                                                                 tile_offsets,
                                                                 blev_storage.block_scan_storage,
                                                                 blev_tile_prefix_op,
                                                                 rocprim::plus<tile_offset_type>{});
            }
            rocprim::syncthreads();

            blev_buffer_offset = flat_block_thread_id * blev_buffers_per_thread;

            // For each buffer this thread processes...
            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < blev_buffers_per_thread; ++i, ++blev_buffer_offset)
            {
                if(blev_buffer_offset >= num_blev_buffers)
                {
                    continue;
                }

                // If this thread has any blev buffers to process...
                const auto tile_buffer_id = buffer_by_size_class[blev_buffer_offset].buffer_id;
                const auto blev_index     = tile_buffer_offset + blev_buffer_offset;

                blev_buffers.srcs[blev_index]    = buffers.srcs[tile_buffer_id];
                blev_buffers.dsts[blev_index]    = buffers.dsts[tile_buffer_id];
                blev_buffers.sizes[blev_index]   = buffers.sizes[tile_buffer_id];
                blev_buffers.offsets[blev_index] = tile_offsets[i];
            }
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE void copy_wlev_buffers(buffer_tuple*    buffers_by_size_class,
                                                             copyable_buffers tile_buffers,
                                                             buffer_offset_type num_wlev_buffers)
        {
            const uint32_t warp_id = rocprim::warp_id();
            const uint32_t warps_per_block
                = rocprim::flat_block_size() / rocprim::device_warp_size();

            for(buffer_offset_type buffer_offset = warp_id; buffer_offset < num_wlev_buffers;
                buffer_offset += warps_per_block)
            {
                const auto buffer_id = buffers_by_size_class[buffer_offset].buffer_id;

                batch_memcpy::vectorized_copy_bytes(tile_buffers.srcs[buffer_id],
                                                    tile_buffers.dsts[buffer_id],
                                                    tile_buffers.sizes[buffer_id]);
            }
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE void
            copy_tlev_buffers(typename storage::shared_t::copy_tlev_t& tlev_storage,
                              buffer_tuple*                            buffers_by_size_class,
                              copyable_buffers                         tile_buffers,
                              buffer_offset_type                       num_tlev_buffers)
        {
            const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();

            buffer_offset_type    tlev_buffer_ids[tlev_buffers_per_thread];
            tlev_byte_offset_type tlev_buffer_sizes[tlev_buffers_per_thread];

            static_assert(
                tlev_buffers_per_thread >= buffers_per_thread,
                "Unsupported configuration: The number of 'thread-level buffers' must be at "
                "least as large as the number of overall buffers being processed by each "
                "thread.");

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < tlev_buffers_per_thread; ++i)
            {
                tlev_buffer_sizes[i] = 0;
            }

            uint32_t tlev_buffer_offset = flat_block_thread_id * tlev_buffers_per_thread;

            ROCPRIM_UNROLL
            for(uint32_t i = 0; i < tlev_buffers_per_thread; ++i)
            {
                if(tlev_buffer_offset < num_tlev_buffers)
                {
                    const auto buffer_info = buffers_by_size_class[tlev_buffer_offset];

                    tlev_buffer_ids[i]   = buffer_info.buffer_id;
                    tlev_buffer_sizes[i] = buffer_info.size;
                }
                ++tlev_buffer_offset;
            }

            // Total number of bytes in this block.
            uint32_t num_total_tlev_bytes = 0;

            block_run_length_decode_type block_run_length_decode{tlev_storage.rld_storage,
                                                                 tlev_buffer_ids,
                                                                 tlev_buffer_sizes,
                                                                 num_total_tlev_bytes};

            // Run-length decode the buffers' sizes into a window buffer of limited size. This is repeated
            // until we were able to cover all the bytes of TLEV buffers
            uint32_t decoded_window_offset = 0;
            while(decoded_window_offset < num_total_tlev_bytes)
            {
                buffer_offset_type    buffer_id[tlev_bytes_per_thread];
                tlev_byte_offset_type buffer_byte_offset[tlev_bytes_per_thread];

                // Now we have a balanced assignment: buffer_id[i] will hold the tile's buffer id and
                // buffer_byte_offset[i] that buffer's byte that this thread supposed to copy
                block_run_length_decode.run_length_decode(buffer_id,
                                                          buffer_byte_offset,
                                                          decoded_window_offset);

                // Zip from SoA to AoS
                zipped_tlev_byte_assignment zipped_byte_assignment[tlev_bytes_per_thread];

                ROCPRIM_UNROLL
                for(uint32_t i = 0; i < tlev_bytes_per_thread; ++i)
                {
                    zipped_byte_assignment[i]
                        = zipped_tlev_byte_assignment{buffer_id[i], buffer_byte_offset[i]};
                }

                // Exchange from blocked to striped arrangement for coalesced memory reads and writes
                block_exchange_tlev_type{}.blocked_to_striped(zipped_byte_assignment,
                                                              zipped_byte_assignment,
                                                              tlev_storage.block_exchange_storage);

                // Read in the bytes that this thread is assigned to
                constexpr auto window_size = tlev_bytes_per_thread * block_size;

                const bool is_full_window
                    = decoded_window_offset + window_size < num_total_tlev_bytes;

                if(is_full_window)
                {
                    uint8_t src_byte[tlev_bytes_per_thread];

                    ROCPRIM_UNROLL
                    for(uint32_t i = 0; i < tlev_bytes_per_thread; ++i)
                    {
                        src_byte[i] = batch_memcpy::read_byte(
                            tile_buffers.srcs[zipped_byte_assignment[i].tile_buffer_id],
                            zipped_byte_assignment[i].buffer_byte_offset);
                    }

                    ROCPRIM_UNROLL
                    for(uint32_t i = 0; i < tlev_bytes_per_thread; ++i)
                    {
                        batch_memcpy::write_byte(
                            tile_buffers.dsts[zipped_byte_assignment[i].tile_buffer_id],
                            zipped_byte_assignment[i].buffer_byte_offset,
                            src_byte[i]);
                    }
                }
                else
                {
                    // Read in the bytes that this thread is assigned to
                    uint32_t absolute_tlev_byte_offset
                        = decoded_window_offset + flat_block_thread_id;
                    for(uint32_t i = 0; i < tlev_bytes_per_thread; ++i)
                    {
                        if(absolute_tlev_byte_offset < num_total_tlev_bytes)
                        {
                            const auto buffer_id     = zipped_byte_assignment[i].tile_buffer_id;
                            const auto buffer_offset = zipped_byte_assignment[i].buffer_byte_offset;

                            const auto src_byte
                                = batch_memcpy::read_byte(tile_buffers.srcs[buffer_id],
                                                          buffer_offset);
                            batch_memcpy::write_byte(tile_buffers.dsts[buffer_id],
                                                     buffer_offset,
                                                     src_byte);
                        }
                        absolute_tlev_byte_offset += block_size;
                    }
                }

                decoded_window_offset += window_size;

                // Ensure all threads finished collaborative BlockExchange so temporary storage can be reused
                // with next iteration
                rocprim::syncthreads();
            }
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE void copy(storage&                    temp_storage,
                                                copyable_buffers            buffers,
                                                uint32_t                    num_buffers,
                                                copyable_blev_buffers       blev_buffers,
                                                blev_buffer_scan_state_type blev_buffer_scan_state,
                                                blev_block_scan_state_type  blev_block_scan_state,
                                                const buffer_offset_type    tile_id)
        {
            const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();

            // Offset into this tile's buffers
            const buffer_offset_type buffer_offset = tile_id * buffers_per_block;

            // Indicates whether all of this tile's items are within bounds
            bool is_full_tile = buffer_offset + buffers_per_block < num_buffers;

            // Load the buffer sizes of this tile's buffers
            auto tile_buffer_sizes = buffers.sizes + buffer_offset;

            // Sizes of the buffers this thread should work on.
            buffer_size_type buffer_sizes[buffers_per_thread];
            if(is_full_tile)
            {
                buffer_load_type{}.load(tile_buffer_sizes,
                                        buffer_sizes,
                                        temp_storage.shared.analysis.load_storage);
            }
            else
            {
                buffer_load_type{}.load(tile_buffer_sizes,
                                        buffer_sizes,
                                        num_buffers - buffer_offset,
                                        0,
                                        temp_storage.shared.analysis.load_storage);
            }

            // Ensure we can repurpose the scan's temporary storage for scattering the buffer ids
            rocprim::syncthreads();

            // Count how many buffers fall into each size-class
            auto size_class_histogram = get_buffer_size_class_histogram(buffer_sizes);

            // Prefix sum over size_class_histogram.
            size_class_counter size_class_agg{};
            block_size_scan_type{}.exclusive_scan(size_class_histogram /* input     */,
                                                  size_class_histogram /* output    */,
                                                  size_class_counter{} /* initial   */,
                                                  size_class_agg /* aggregate */,
                                                  temp_storage.shared.analysis.size_scan_storage,
                                                  rocprim::plus<size_class_counter>{});

            rocprim::syncthreads();

            uint32_t buffer_count = 0;

            // Factor in the per-size-class counts / offsets
            // That is, WLEV buffer offset has to be offset by the TLEV buffer count and BLEV buffer offset
            // has to be offset by the TLEV+WLEV buffer count
            for(const auto size_class : {batch_memcpy::size_class::tlev,
                                         batch_memcpy::size_class::wlev,
                                         batch_memcpy::size_class::blev})
            {
                size_class_histogram.add(size_class, buffer_count);
                buffer_count += size_class_agg.get(size_class);
            }

            // Signal the number of BLEV buffers we're planning to write out
            // Aggregate the count of blev buffers across threads.
            buffer_offset_type buffer_exclusive_prefix{};
            if(tile_id == 0)
            {
                if(flat_block_thread_id == 0)
                {
                    blev_buffer_scan_state.set_complete(
                        tile_id,
                        size_class_agg.get(batch_memcpy::size_class::blev));
                }
                buffer_exclusive_prefix = 0;
            }
            else
            {
                blev_buffer_scan_prefix_callback_type blev_buffer_prefix_op{
                    tile_id,
                    blev_buffer_scan_state,
                    temp_storage.shared.analysis.buffer_scan_callback,
                    rocprim::plus<buffer_offset_type>{}};

                buffer_exclusive_prefix
                    = blev_buffer_prefix_op(size_class_agg.get(batch_memcpy::size_class::blev));
            }
            if(flat_block_thread_id == 0)
            {
                temp_storage.blev_buffer_offset = buffer_exclusive_prefix;
            }

            rocprim::syncthreads();

            // Write partitions to shared memory.
            partition_buffer_by_size(buffer_sizes,
                                     size_class_histogram,
                                     temp_storage.buffers_by_size_class);
            rocprim::syncthreads();

            // Get buffers for this tile.
            copyable_buffers tile_buffer = copyable_buffers{
                buffers.srcs + buffer_offset,
                buffers.dsts + buffer_offset,
                buffers.sizes + buffer_offset,
            };

            auto num_blev_buffers = size_class_agg.get(batch_memcpy::size_class::blev);
            auto num_wlev_buffers = size_class_agg.get(batch_memcpy::size_class::wlev);
            auto num_tlev_buffers = size_class_agg.get(batch_memcpy::size_class::tlev);

            // BLEV buffers are copied in a seperate kernel. We need to prepare global memory
            // to pass what needs to be copied where that kernel.
            prepare_blev_buffers(
                temp_storage.shared.prepare_blev,
                &temp_storage
                     .buffers_by_size_class[size_class_agg.get(batch_memcpy::size_class::tlev)
                                            + size_class_agg.get(batch_memcpy::size_class::wlev)],
                tile_buffer,
                num_blev_buffers,
                blev_buffers,
                temp_storage.blev_buffer_offset,
                blev_block_scan_state,
                tile_id);

            rocprim::syncthreads();

            copy_wlev_buffers(
                &temp_storage
                     .buffers_by_size_class[size_class_agg.get(batch_memcpy::size_class::tlev)],
                tile_buffer,
                num_wlev_buffers);

            copy_tlev_buffers(temp_storage.shared.copy_tlev,
                              temp_storage.buffers_by_size_class,
                              tile_buffer,
                              num_tlev_buffers);
        }
    };

public:
    __global__ static void init_tile_state_kernel(blev_buffer_scan_state_type buffer_scan_state,
                                                  blev_block_scan_state_type  block_scan_state,
                                                  tile_offset_type            num_tiles)
    {
        const uint32_t block_id        = rocprim::detail::block_id<0>();
        const uint32_t block_size      = rocprim::detail::block_size<0>();
        const uint32_t block_thread_id = rocprim::detail::block_thread_id<0>();
        const uint32_t flat_thread_id  = (block_id * block_size) + block_thread_id;

        buffer_scan_state.initialize_prefix(flat_thread_id, num_tiles);

        block_scan_state.initialize_prefix(flat_thread_id, num_tiles);
    }

    __global__ static void
        non_blev_memcpy_kernel(copyable_buffers            buffers,
                               buffer_offset_type          num_buffers,
                               copyable_blev_buffers       blev_buffers,
                               blev_buffer_scan_state_type blev_buffer_scan_state,
                               blev_block_scan_state_type  blev_block_scan_state)
    {
        ROCPRIM_SHARED_MEMORY typename non_blev_memcpy::storage_type temp_storage;
        non_blev_memcpy{}.copy(temp_storage.get(),
                               buffers,
                               num_buffers,
                               blev_buffers,
                               blev_buffer_scan_state,
                               blev_block_scan_state,
                               rocprim::flat_block_id());
    }

    __global__ static void blev_memcpy_kernel(copyable_blev_buffers       blev_buffers,
                                              blev_buffer_scan_state_type buffer_offset_tile,
                                              tile_offset_type            last_tile_offset)
    {
        const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
        const auto flat_block_id        = ::rocprim::detail::block_id<0>();
        const auto flat_grid_size       = ::rocprim::detail::grid_size<0>();

        constexpr auto blev_tile_size   = blev_block_size * blev_bytes_per_thread;
        const auto     num_blev_buffers = buffer_offset_tile.get_complete_value(last_tile_offset);

        if(num_blev_buffers == 0)
        {
            return;
        }

        uint32_t tile_id = flat_block_id;
        while(true)
        {
            __shared__ buffer_offset_type shared_buffer_id;

            rocprim::syncthreads();

            if(flat_block_thread_id == 0)
            {
                shared_buffer_id
                    = rocprim::upper_bound(blev_buffers.offsets, num_blev_buffers, tile_id) - 1;
            }

            rocprim::syncthreads();

            const buffer_offset_type buffer_id = shared_buffer_id;

            // The relative offset of this tile within the buffer it's assigned to
            const buffer_size_type tile_offset_within_buffer
                = (tile_id - blev_buffers.offsets[buffer_id]) * blev_tile_size;

            // If the tile has already reached beyond the work of the end of the last buffer
            if(buffer_id >= num_blev_buffers - 1
               && tile_offset_within_buffer > blev_buffers.sizes[buffer_id])
            {
                return;
            }

            // Tiny remainders are copied without vectorizing loads
            if(blev_buffers.sizes[buffer_id] - tile_offset_within_buffer <= 32)
            {
                buffer_size_type thread_offset = tile_offset_within_buffer + flat_block_thread_id;
                for(uint32_t i = 0; i < blev_buffers_per_thread;
                    ++i, thread_offset += blev_block_size)
                {
                    if(thread_offset < blev_buffers.sizes[buffer_id])
                    {
                        uint8_t item
                            = batch_memcpy::read_byte(blev_buffers.srcs[buffer_id], thread_offset);
                        batch_memcpy::write_byte(blev_buffers.dsts[buffer_id], thread_offset, item);
                    }
                }
                tile_id += flat_grid_size;
                continue;
            }

            const buffer_size_type items_to_copy
                = rocprim::min(static_cast<buffer_size_type>(blev_buffers.sizes[buffer_id]
                                                             - tile_offset_within_buffer),
                               static_cast<buffer_size_type>(blev_tile_size));

            batch_memcpy::vectorized_copy_bytes(blev_buffers.srcs[buffer_id],
                                                blev_buffers.dsts[buffer_id],
                                                items_to_copy,
                                                tile_offset_within_buffer);

            tile_id += flat_grid_size;
        }
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif
