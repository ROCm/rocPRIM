// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/radix_sort.hpp"

#include "block_scan.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

template<unsigned int BlockSizeX,
         unsigned int RadixBits,
         bool         MemoizeOuterScan = false,
         unsigned int BlockSizeY       = 1,
         unsigned int BlockSizeZ       = 1>
class block_radix_rank
{
    using digit_counter_type  = unsigned short;
    using packed_counter_type = unsigned int;

    using block_scan_type = ::rocprim::block_scan<packed_counter_type,
                                                  BlockSizeX,
                                                  ::rocprim::block_scan_algorithm::using_warp_scan,
                                                  BlockSizeY,
                                                  BlockSizeZ>;

    static constexpr unsigned int block_size   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int radix_digits = 1 << RadixBits;
    static constexpr unsigned int packing_ratio
        = sizeof(packed_counter_type) / sizeof(digit_counter_type);
    static constexpr unsigned int log_packing_ratio = ::rocprim::Log2<packing_ratio>::VALUE;
    static constexpr unsigned int column_size       = radix_digits / packing_ratio;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        union
        {
            digit_counter_type  digit_counters[block_size * radix_digits];
            packed_counter_type packed_counters[block_size * column_size];
        };

        typename block_scan_type::storage_type block_scan;
        packed_counter_type                    column_prefix;
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE void reset_counters(const unsigned int flat_id,
                                                      storage_type_&     storage)
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            storage.packed_counters[i * block_size + flat_id] = 0;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void
        scan_block_counters(const unsigned int         flat_id,
                            storage_type_&             storage,
                            packed_counter_type* const packed_counters)
    {
        packed_counter_type block_reduction = 0;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            block_reduction += packed_counters[i];
        }

        packed_counter_type exclusive_prefix = 0;
        block_scan_type{}.exclusive_scan(block_reduction, exclusive_prefix, 0, storage.block_scan);

        if(flat_id == block_size - 1)
        {
            packed_counter_type totals        = exclusive_prefix + block_reduction;
            packed_counter_type column_prefix = 0;
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < packing_ratio; i <<= 1)
            {
                column_prefix += totals << (sizeof(digit_counter_type) * 8 * i);
            }
            storage.column_prefix = column_prefix;
        }

        ::rocprim::syncthreads();

        exclusive_prefix += storage.column_prefix;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            packed_counter_type counter = packed_counters[i];
            packed_counters[i]          = exclusive_prefix;
            exclusive_prefix += counter;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void scan_counters(const unsigned int flat_id,
                                                     storage_type_&     storage)
    {
        packed_counter_type* const shared_counters
            = &storage.packed_counters[flat_id * column_size];

        if ROCPRIM_IF_CONSTEXPR(MemoizeOuterScan)
        {
            packed_counter_type local_counters[column_size];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                local_counters[i] = shared_counters[i];
            }

            scan_block_counters(flat_id, storage, local_counters);

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                shared_counters[i] = local_counters[i];
            }
        }
        else
        {
            scan_block_counters(flat_id, storage, shared_counters);
        }
    }

    template<typename KeyCodec, typename BitKey, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void rank_bit_keys_impl(const BitKey (&bit_keys)[ItemsPerThread],
                                           unsigned int (&ranks)[ItemsPerThread],
                                           storage_type_&     storage,
                                           const unsigned int begin_bit,
                                           const unsigned int pass_bits)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        reset_counters(flat_id, storage);

        digit_counter_type thread_prefixes[ItemsPerThread];
        unsigned int       digit_counters[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int digit = KeyCodec::extract_digit(bit_keys[i], begin_bit, pass_bits);
            const unsigned int column_counter = digit % (radix_digits / packing_ratio);
            const unsigned int sub_counter    = digit / (radix_digits / packing_ratio);
            const unsigned int counter
                = (column_counter * block_size + flat_id) * packing_ratio + sub_counter;

            digit_counters[i]  = counter;
            thread_prefixes[i] = storage.digit_counters[counter]++;
        }

        ::rocprim::syncthreads();

        scan_counters(flat_id, storage);

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            ranks[i] = thread_prefixes[i] + storage.digit_counters[digit_counters[i]];
        }
    }

    template<bool Descending, typename Key, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_&     storage,
                                       const unsigned int begin_bit,
                                       const unsigned int pass_bits)
    {
        using key_codec    = ::rocprim::detail::radix_key_codec<Key, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        bit_key_type bit_keys[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            bit_keys[i] = key_codec::encode(keys[i]);
        }

        rank_bit_keys_impl<key_codec>(bit_keys, ranks, storage, begin_bit, pass_bits);
    }

public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = detail::raw_storage<storage_type_>;
#else
    using storage_type = storage_type_; // only for Doxygen
#endif

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type& storage,
                                  unsigned int  begin_bit = 0,
                                  unsigned int  pass_bits = RadixBits)
    {
        rank_keys_impl<false>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  unsigned int begin_bit = 0,
                                  unsigned int pass_bits = RadixBits)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        rank_keys(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type& storage,
                                       unsigned int  begin_bit = 0,
                                       unsigned int  pass_bits = RadixBits)
    {
        rank_keys_impl<true>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       unsigned int begin_bit = 0,
                                       unsigned int pass_bits = RadixBits)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        rank_keys_desc(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename KeyCodec, typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_bit_keys(const Key (&keys)[ItemsPerThread],
                                      unsigned int (&ranks)[ItemsPerThread],
                                      storage_type& storage,
                                      unsigned int  begin_bit = 0,
                                      unsigned int  pass_bits = RadixBits)
    {
        rank_bit_keys_impl<KeyCodec>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename KeyCodec, typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_bit_keys(const Key (&keys)[ItemsPerThread],
                                      unsigned int (&ranks)[ItemsPerThread],
                                      unsigned int begin_bit = 0,
                                      unsigned int pass_bits = RadixBits)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        rank_bit_keys(keys, ranks, storage.get(), begin_bit, pass_bits);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
