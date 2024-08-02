// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
#define ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_

#include <cstring>
#include <type_traits>

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"
#include "../../types.hpp"

#include "../../warp/detail/warp_reduce_crosslane.hpp"
#include "../../warp/detail/warp_scan_crosslane.hpp"

#include "../../detail/binary_op_wrappers.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../detail/various.hpp"

#include "../config_types.hpp"
#include "rocprim/config.hpp"

// This version is specific for devices with slow __threadfence ("agent" fence which does
// L2 cache flushing and invalidation).
// Fences with "workgroup" scope are used instead to ensure ordering only but not coherence,
// they do not flush and invalidate cache.
// Global coherence of prefixes_*_values is ensured by atomic_load/atomic_store that bypass
// cache.
#ifndef ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
    #if defined(__HIP_DEVICE_COMPILE__) \
        && (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
        #define ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES 1
    #else
        #define ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES 0
    #endif
#endif // ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES

extern "C" {
void __builtin_amdgcn_s_sleep(int);
}
BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

enum prefix_flag
{
    // flag for padding, values should be discarded
    PREFIX_INVALID = -1,
    // initialized, not result in value
    PREFIX_EMPTY = 0,
    // partial prefix value (from single block)
    PREFIX_PARTIAL = 1,
    // final prefix value
    PREFIX_COMPLETE = 2
};

// In the original implementation, lookback scan is not deterministic
// for non-associative operations: This is because the number of lookback
// steps may vary depending on the algorithm that its used in, scanned
// operator, and the hardware of the device running on it. Usually, the
// trade-off is worth the extra speed bonus, but sometimes bitwise
// reproducibility is more important. This enum may be used to tune the
// lookback scan implementation to favor one or the other.
enum class lookback_scan_determinism
{
    // Allow the implementation to produce non-deterministic results.
    nondeterministic,
    // Do not allow the implementation to produce non-deterministic results.
    // This may come at a performance penalty, depending on algorithm and device.
    deterministic,
    // By default, prefer the speedy option.
    default_determinism = nondeterministic,
};

// lookback_scan_state object keeps track of prefixes status for
// a look-back prefix scan. Initially every prefix can be either
// invalid (padding values) or empty. One thread in a block should
// later set it to partial, and later to complete.
template<class T, bool UseSleep = false, bool IsSmall = (sizeof(T) <= 4)>
struct lookback_scan_state;

// Packed flag and prefix value are loaded/stored in one atomic operation.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, true>
{
private:
    using flag_type_ = char;

    // Type which is used in store/load operations of block prefix (flag and value).
    // It is 32-bit or 64-bit int and can be loaded/stored using single atomic instruction.
    using prefix_underlying_type =
        typename std::conditional<(sizeof(T) > 2), unsigned long long, unsigned int>::type;

    // Helper struct
    struct alignas(sizeof(prefix_underlying_type)) prefix_type
    {
        flag_type_ flag;
        T          value;
    };

    static_assert(sizeof(prefix_underlying_type) == sizeof(prefix_type), "");

public:
    // Type used for flag/flag of block prefix
    using flag_type  = flag_type_;
    using value_type = T;

    static constexpr bool use_sleep = UseSleep;

    // temp_storage must point to allocation of get_storage_size(number_of_blocks) bytes
    ROCPRIM_HOST static inline hipError_t create(lookback_scan_state& state,
                                                 void*                temp_storage,
                                                 const unsigned int   number_of_blocks,
                                                 const hipStream_t /*stream*/)
    {
        (void)number_of_blocks;
        state.prefixes = reinterpret_cast<prefix_underlying_type*>(temp_storage);
        return hipSuccess;
    }

    [[deprecated(
        "Please use the overload returns an error code, this function assumes the default"
        " stream and silently ignores errors.")]] ROCPRIM_HOST static inline lookback_scan_state
        create(void* temp_storage, const unsigned int number_of_blocks)
    {
        lookback_scan_state result;
        (void)create(result, temp_storage, number_of_blocks, /*default stream*/ 0);
        return result;
    }

    ROCPRIM_HOST static inline hipError_t get_storage_size(const unsigned int number_of_blocks,
                                                           const hipStream_t  stream,
                                                           size_t&            storage_size)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);

        storage_size = sizeof(prefix_underlying_type) * (warp_size + number_of_blocks);

        return error;
    }

    [[deprecated("Please use the overload returns an error code, this function assumes the default"
                 " stream and silently ignores errors.")]] ROCPRIM_HOST static inline size_t
        get_storage_size(const unsigned int number_of_blocks)
    {
        size_t result;
        (void)get_storage_size(number_of_blocks, /*default stream*/ 0, result);
        return result;
    }

    ROCPRIM_HOST static inline hipError_t
        get_temp_storage_layout(const unsigned int            number_of_blocks,
                                const hipStream_t             stream,
                                detail::temp_storage::layout& layout)
    {
        size_t     storage_size = 0;
        hipError_t error        = get_storage_size(number_of_blocks, stream, storage_size);
        layout = detail::temp_storage::layout{storage_size, alignof(prefix_underlying_type)};
        return error;
    }

    [[deprecated("Please use the overload returns an error code, this function assumes the default"
                 " stream and silently ignores errors.")]] ROCPRIM_HOST static inline detail::
        temp_storage::layout
        get_temp_storage_layout(const unsigned int number_of_blocks)
    {
        detail::temp_storage::layout result;
        (void)get_temp_storage_layout(number_of_blocks, /*default stream*/ 0, result);
        return result;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void initialize_prefix(const unsigned int block_id,
                                                         const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        if(block_id < number_of_blocks)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_EMPTY;
            prefix_underlying_type p;
            memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[padding + block_id] = p;
        }
        if(block_id < padding)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_INVALID;
            prefix_underlying_type p;
            memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[block_id] = p;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_PARTIAL, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_COMPLETE, value);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE ROCPRIM_INLINE void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type prefix;

        const unsigned int SLEEP_MAX     = 32;
        unsigned int       times_through = 1;

        prefix_underlying_type p = ::rocprim::detail::atomic_load(&prefixes[padding + block_id]);
        memcpy(&prefix, &p, sizeof(prefix_type));
        while(prefix.flag == PREFIX_EMPTY)
        {
            if(UseSleep)
            {
                for(unsigned int j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if(times_through < SLEEP_MAX)
                    times_through++;
            }
            prefix_underlying_type p
                = ::rocprim::detail::atomic_load(&prefixes[padding + block_id]);
            memcpy(&prefix, &p, sizeof(prefix_type));
        }

        // return
        flag  = prefix.flag;
        value = prefix.value;
    }

    /// \brief Gets the prefix value for a block. Should only be called after all
    /// blocks/prefixes are completed.
    ROCPRIM_DEVICE ROCPRIM_INLINE T get_complete_value(const unsigned int block_id)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        auto        p = prefixes[padding + block_id];
        prefix_type prefix{};
        memcpy(&prefix, &p, sizeof(prefix_type));
        assert(prefix.flag == PREFIX_COMPLETE);
        return prefix.value;
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        set(const unsigned int block_id, const flag_type flag, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type            prefix = {flag, value};
        prefix_underlying_type p;
        memcpy(&p, &prefix, sizeof(prefix_type));
        ::rocprim::detail::atomic_store(&prefixes[padding + block_id], p);
    }

    prefix_underlying_type* prefixes;
};

// Flag, partial and final prefixes are stored in separate arrays.
// Consistency ensured by memory fences between flag and prefixes load/store operations.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, false>
{

public:
    using flag_type  = unsigned int;
    using value_type = T;

    static constexpr bool use_sleep = UseSleep;

    // temp_storage must point to allocation of get_storage_size(number_of_blocks) bytes
    ROCPRIM_HOST static inline hipError_t create(lookback_scan_state& state,
                                                 void*                temp_storage,
                                                 const unsigned int   number_of_blocks,
                                                 const hipStream_t    stream)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);

        const auto n = warp_size + number_of_blocks;

        auto ptr = static_cast<char*>(temp_storage);

        state.prefixes_flags = reinterpret_cast<flag_type*>(ptr);
        ptr += ::rocprim::detail::align_size(n * sizeof(flag_type));

        state.prefixes_partial_values = ptr;
        ptr += ::rocprim::detail::align_size(n * sizeof(value_underlying_type));

        state.prefixes_complete_values = ptr;
        return error;
    }

    [[deprecated(
        "Please use the overload returns an error code, this function assumes the default"
        " stream and silently ignores errors.")]] ROCPRIM_HOST static inline lookback_scan_state
        create(void* temp_storage, const unsigned int number_of_blocks)
    {
        lookback_scan_state result;
        (void)create(result, temp_storage, number_of_blocks, /*default stream*/ 0);
        return result;
    }

    ROCPRIM_HOST static inline hipError_t get_storage_size(const unsigned int number_of_blocks,
                                                           const hipStream_t  stream,
                                                           size_t&            storage_size)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);
        const auto   n     = warp_size + number_of_blocks;
        storage_size       = ::rocprim::detail::align_size(n * sizeof(flag_type));
        // Always use sizeof(value_underlying_type) instead of sizeof(T) because storage is
        // allocated by host so it can hold both types no matter what device is used.
        storage_size += 2 * ::rocprim::detail::align_size(n * sizeof(value_underlying_type));
        return error;
    }

    [[deprecated("Please use the overload returns an error code, this function assumes the default"
                 " stream and silently ignores errors.")]] ROCPRIM_HOST static inline size_t
        get_storage_size(const unsigned int number_of_blocks)
    {
        size_t result;
        (void)get_storage_size(number_of_blocks, /*default stream*/ 0, result);
        return result;
    }

    ROCPRIM_HOST static inline hipError_t
        get_temp_storage_layout(const unsigned int            number_of_blocks,
                                const hipStream_t             stream,
                                detail::temp_storage::layout& layout)
    {
        size_t storage_size = 0;
        size_t alignment
            = std::max({alignof(flag_type), alignof(T), alignof(value_underlying_type)});
        hipError_t error = get_storage_size(number_of_blocks, stream, storage_size);
        layout           = detail::temp_storage::layout{storage_size, alignment};
        return error;
    }

    [[deprecated("Please use the overload returns an error code, this function assumes the default"
                 " stream and silently ignores errors.")]] ROCPRIM_HOST static inline detail::
        temp_storage::layout
        get_temp_storage_layout(const unsigned int number_of_blocks)
    {
        detail::temp_storage::layout result;
        (void)get_temp_storage_layout(number_of_blocks, /*default stream*/ 0, result);
        return result;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void initialize_prefix(const unsigned int block_id,
                                                         const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();
        if(block_id < number_of_blocks)
        {
            prefixes_flags[padding + block_id] = PREFIX_EMPTY;
        }
        if(block_id < padding)
        {
            prefixes_flags[block_id] = PREFIX_INVALID;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_PARTIAL, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_COMPLETE, value);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE ROCPRIM_INLINE void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        const unsigned int SLEEP_MAX     = 32;
        unsigned int       times_through = 1;

        flag = ::rocprim::detail::atomic_load(&prefixes_flags[padding + block_id]);
        while(flag == PREFIX_EMPTY)
        {
            if(UseSleep)
            {
                for(unsigned int j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if(times_through < SLEEP_MAX)
                    times_through++;
            }

            flag = ::rocprim::detail::atomic_load(&prefixes_flags[padding + block_id]);
        }
#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        rocprim::detail::atomic_fence_acquire_order_only();

        const auto* values = static_cast<const value_underlying_type*>(
            flag == PREFIX_PARTIAL ? prefixes_partial_values : prefixes_complete_values);
        value_underlying_type v;
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            v.words[i] = ::rocprim::detail::atomic_load(&values[padding + block_id].words[i]);
        }
        __builtin_memcpy(&value, &v, sizeof(value));
#else
        ::rocprim::detail::memory_fence_device();

        const auto* values = static_cast<const T*>(
            flag == PREFIX_PARTIAL ? prefixes_partial_values : prefixes_complete_values);
        value = values[padding + block_id];
#endif
    }

    /// \brief Gets the prefix value for a block. Should only be called after all
    /// blocks/prefixes are completed.
    ROCPRIM_DEVICE ROCPRIM_INLINE T get_complete_value(const unsigned int block_id)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        T           value;
        const auto* values = static_cast<const value_underlying_type*>(prefixes_complete_values);
        value_underlying_type v;
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            v.words[i] = ::rocprim::detail::atomic_load(&values[padding + block_id].words[i]);
        }
        __builtin_memcpy(&value, &v, sizeof(value));
        return value;
#else
        assert(prefixes_flags[padding + block_id] == PREFIX_COMPLETE);
        const auto* values = static_cast<const T*>(prefixes_complete_values);
        return values[padding + block_id];
#endif
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        set(const unsigned int block_id, const flag_type flag, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        auto* values = static_cast<value_underlying_type*>(
            flag == PREFIX_PARTIAL ? prefixes_partial_values : prefixes_complete_values);
        value_underlying_type v;
        __builtin_memcpy(&v, &value, sizeof(value));
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            ::rocprim::detail::atomic_store(&values[padding + block_id].words[i], v.words[i]);
        }
        // Wait for all atomic stores of prefixes_*_values before signaling complete / partial state
        rocprim::detail::atomic_fence_release_vmem_order_only();
#else
        auto* values = static_cast<T*>(flag == PREFIX_PARTIAL ? prefixes_partial_values
                                                              : prefixes_complete_values);
        values[padding + block_id] = value;
        ::rocprim::detail::memory_fence_device();
#endif

        ::rocprim::detail::atomic_store(&prefixes_flags[padding + block_id], flag);
    }

    struct value_underlying_type
    {
        static constexpr int words_no = ceiling_div(sizeof(T), sizeof(unsigned int));

        unsigned int words[words_no];
    };

    flag_type* prefixes_flags;
    // We need to separate arrays for partial and final prefixes, because
    // value can be overwritten before flag is changed (flag and value are
    // not stored in single instruction).
    void* prefixes_partial_values;
    void* prefixes_complete_values;
};

template<class T,
         class BinaryFunction,
         class LookbackScanState,
         lookback_scan_determinism Determinism = lookback_scan_determinism::default_determinism>
class lookback_scan_prefix_op
{
    using flag_type = typename LookbackScanState::flag_type;
    static_assert(std::is_same<T, typename LookbackScanState::value_type>::value,
                  "T must be LookbackScanState::value_type");

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE lookback_scan_prefix_op(unsigned int       block_id,
                                                          BinaryFunction     scan_op,
                                                          LookbackScanState& scan_state)
        : block_id_(block_id), scan_op_(scan_op), scan_state_(scan_state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE ~lookback_scan_prefix_op() = default;

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        reduce_partial_prefixes(unsigned int block_id, flag_type& flag, T& partial_prefix)
    {
        // Order of reduction must be reversed, because 0th thread has
        // prefix from the (block_id_ - 1) block, 1st thread has prefix
        // from (block_id_ - 2) block etc.
        using headflag_scan_op_type = reverse_binary_op_wrapper<BinaryFunction, T, T>;
        using warp_reduce_prefix_type
            = warp_reduce_crosslane<T, ::rocprim::device_warp_size(), false>;

        T block_prefix;
        scan_state_.get(block_id, flag, block_prefix);

        auto headflag_scan_op = headflag_scan_op_type(scan_op_);
        warp_reduce_prefix_type().tail_segmented_reduce(block_prefix,
                                                        partial_prefix,
                                                        (flag == PREFIX_COMPLETE),
                                                        headflag_scan_op);
    }

    /// Reduce lanes `0-valid_items` and return the result in lane `device_warp_size() - 1`.
    ROCPRIM_DEVICE ROCPRIM_INLINE T reduce_forward_init(T block_prefix, unsigned int valid_items)
    {
        block_prefix = warp_shuffle_up(block_prefix, device_warp_size() - 1 - valid_items);
        T prefix     = block_prefix;
        for(int i = valid_items - 1; i >= 0; --i)
        {
#ifdef ROCPRIM_DETAIL_HAS_DPP_WF_ROTATE
            block_prefix = warp_move_dpp<T, 0x13C>(block_prefix);
#else
            block_prefix = warp_shuffle_up(block_prefix, 1);
#endif
            prefix = scan_op_(prefix, block_prefix);
        }
        return prefix;
    }

    /// Reduce all lanes with the `prefix`, which is taken from the `device_warp_size() - 1`th
    /// lane, and return the result in lane `device_warp_size() - 1`.
    ROCPRIM_DEVICE ROCPRIM_INLINE T reduce_forward(T prefix, T block_prefix)
    {
#ifdef ROCPRIM_DETAIL_HAS_DPP_WF
        ROCPRIM_UNROLL
        for(int i = device_warp_size() - 1; i >= 0; --i)
        {
            prefix       = scan_op_(prefix, block_prefix);
            block_prefix = warp_move_dpp<T, 0x13C /* DPP_WF_RR1 */>(block_prefix);
        }
#elif ROCPRIM_DETAIL_USE_DPP == 1
        // If we can't rotate or shift the entire wavefront in one instruction,
        // iterate over rows of 16 lanes and use warp_readlane to communicate across rows.
        constexpr const int row_size = 16;
        ROCPRIM_UNROLL
        for(int j = device_warp_size() - 1; j > 0; j -= row_size)
        {
            ROCPRIM_UNROLL
            for(int i = 0; i < row_size; ++i)
            {
                prefix       = scan_op_(prefix, block_prefix);
                block_prefix = warp_move_dpp<T, 0x121 /* DPP_ROW_RR1 */>(block_prefix);
            }

            prefix = warp_readlane(prefix, j);
        }
#else
        // If no DPP available at all, fall back to shuffles.
        ROCPRIM_UNROLL
        for(int i = device_warp_size() - 1; i >= 0; --i)
        {
            prefix       = scan_op_(prefix, block_prefix);
            block_prefix = warp_shuffle(block_prefix, lane_id() - 1);
        }
#endif
        return prefix;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE T get_prefix()
    {
        if ROCPRIM_IF_CONSTEXPR(Determinism == lookback_scan_determinism::nondeterministic)
        {
            flag_type    flag;
            T            partial_prefix;
            unsigned int previous_block_id = block_id_ - ::rocprim::lane_id() - 1;

            // reduce last warp_size() number of prefixes to
            // get the complete prefix for this block.
            reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
            T prefix = partial_prefix;

            // while we don't load a complete prefix, reduce partial prefixes
            while(::rocprim::detail::warp_all(flag != PREFIX_COMPLETE))
            {
                previous_block_id -= ::rocprim::device_warp_size();
                reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
                prefix = scan_op_(partial_prefix, prefix);
            }
            return prefix;
        }
        else /* Determinism == lookback_scan_state::deterministic */
        {
            unsigned int lookback_block_id = block_id_ - lane_id() - 1;

            // There is one lookback scan per block, though a lookback scan is done by a single warp.
            // Because every lane of the warp checks a different lookback scan state value,
            // we need space for at least ceil(CUs / device_warp_size()) items in the cache,
            // assuming that only one block is active per CU (assumes low occupancy).
            // For MI300, with 304 CUs, we have 304 / 64 = 5 items for the lookback cache.
            // Note that one item is kept in the `block_prefix` register, so we only need to
            // cache 4 values here in the worst case.
            constexpr int max_lookback_per_thread = 4;

            T   cache[max_lookback_per_thread];
            int cache_offset = 0;

            flag_type flag;
            T         block_prefix;
            scan_state_.get(lookback_block_id, flag, block_prefix);

            while(warp_all(flag != PREFIX_COMPLETE && flag != PREFIX_INVALID)
                  && cache_offset < max_lookback_per_thread)
            {
                cache[cache_offset++] = block_prefix;
                lookback_block_id -= device_warp_size();
                scan_state_.get(lookback_block_id, flag, block_prefix);
            }

            // If no flags are complete, we have hit either of the following edge cases:
            // - The lookback_block_id is < 0 for all lanes. In this case, we need to go
            //   forward one block and pop one invalid item off the cache.
            // - We have run out of available space in the cache. In this case, wait until
            //   any of the current lookback flags pointed to by lookback_block_id changes
            //   to complete.
            if(warp_all(flag != PREFIX_COMPLETE))
            {
                if(warp_all(flag == PREFIX_INVALID))
                {
                    // All invalid, so we have to move one block back to
                    // get back to known civilization.
                    // Don't forget to pop one item off the cache too.
                    lookback_block_id += device_warp_size();
                    --cache_offset;
                }

                do
                {
                    scan_state_.get(lookback_block_id, flag, block_prefix);
                }
                while(warp_all(flag != PREFIX_COMPLETE));
            }

            // Now just sum all these values to get the prefix
            // Note that the values are striped across the threads.
            // In the first iteration, the current prefix is at the value cache for the current
            // offset at the lowest warp number that has PREFIX_COMPLETE set.
            const auto bits   = ballot(flag == PREFIX_COMPLETE);
            const auto lowest = ctz(bits);

            // Now sum all the values from block_prefix that are lower than the current prefix.
            T prefix = reduce_forward_init(block_prefix, lowest);

            // Now sum all from the prior cache.
            // These are all guaranteed to be PARTIAL
            while(cache_offset > 0)
            {
                block_prefix = cache[--cache_offset];
                prefix       = reduce_forward(prefix, block_prefix);
            }

            return warp_readlane(prefix, device_warp_size() - 1);
        }
    }

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE T operator()(T reduction)
    {
        // Set partial prefix for next block
        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_partial(block_id_, reduction);
        }

        // Get prefix
        auto prefix = get_prefix();

        // Set complete prefix for next block
        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_complete(block_id_, scan_op_(prefix, reduction));
        }
        return prefix;
    }

protected:
    unsigned int       block_id_;
    BinaryFunction     scan_op_;
    LookbackScanState& scan_state_;
};

// It is known that early revisions of MI100 (gfx908) hang in the wait loop of
// lookback_scan_state::get() without sleeping (s_sleep).
// is_sleep_scan_state_used() checks the architecture/revision of the device on host in runtime,
// to select the corresponding kernel (with or without sleep). However, since the check is runtime,
// both versions of the kernel must be compiled for all architectures.
// is_lookback_kernel_runnable() can be used in device code to prevent compilation of the version
// with sleep on all device arhitectures except gfx908.

ROCPRIM_HOST ROCPRIM_INLINE hipError_t is_sleep_scan_state_used(const hipStream_t stream,
                                                                bool&             use_sleep)
{
    hipDeviceProp_t prop;
    int             device_id;
    if(const hipError_t error = get_device_from_stream(stream, device_id))
    {
        return error;
    }
    else if(const hipError_t error = hipGetDeviceProperties(&prop, device_id))
    {
        return error;
    }
#if HIP_VERSION >= 307
    const int asicRevision = prop.asicRevision;
#else
    const int asicRevision = 0;
#endif
    use_sleep = std::string(prop.gcnArchName).find("908") != std::string::npos && asicRevision < 2;
    return hipSuccess;
}

template<typename LookbackScanState>
constexpr bool is_lookback_kernel_runnable()
{
    if(device_target_arch() == target_arch::gfx908)
    {
        // For gfx908 kernels with both version of lookback_scan_state can run: with and without
        // sleep
        return true;
    }
    // For other GPUs only a kernel without sleep can run
    return !LookbackScanState::use_sleep;
}

template<typename T>
class offset_lookback_scan_factory
{
private:
    struct storage_type_
    {
        T block_reduction;
        T prefix;
    };

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<typename PrefixOp>
    static ROCPRIM_DEVICE auto create(PrefixOp& prefix_op, storage_type& storage)
    {
        return [&](T reduction) mutable
        {
            auto prefix = prefix_op(reduction);
            if(::rocprim::lane_id() == 0)
            {
                storage.get().block_reduction = std::move(reduction);
                storage.get().prefix          = prefix;
            }
            return prefix;
        };
    }

    static ROCPRIM_DEVICE T get_reduction(const storage_type& storage)
    {
        return storage.get().block_reduction;
    }

    static ROCPRIM_DEVICE T get_prefix(const storage_type& storage)
    {
        return storage.get().prefix;
    }
};

template<class T,
         class LookbackScanState,
         class BinaryOp                        = ::rocprim::plus<T>,
         lookback_scan_determinism Determinism = lookback_scan_determinism::default_determinism>
class offset_lookback_scan_prefix_op
    : public lookback_scan_prefix_op<T, BinaryOp, LookbackScanState, Determinism>
{
private:
    using base_type = lookback_scan_prefix_op<T, BinaryOp, LookbackScanState, Determinism>;
    using factory   = detail::offset_lookback_scan_factory<T>;

    ROCPRIM_DEVICE ROCPRIM_INLINE base_type& base()
    {
        return *this;
    }

public:
    using storage_type = typename factory::storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE offset_lookback_scan_prefix_op(unsigned int       block_id,
                                                                 LookbackScanState& state,
                                                                 storage_type&      storage,
                                                                 BinaryOp binary_op = BinaryOp())
        : base_type(block_id, BinaryOp(std::move(binary_op)), state), storage(storage)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE T operator()(T reduction)
    {
        return factory::create(base(), storage)(reduction);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE T get_reduction() const
    {
        return factory::get_reduction(storage);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE T get_prefix() const
    {
        return factory::get_prefix(storage);
    }

    // rocThrust uses this implementation detail of rocPRIM, required for backwards compatibility
    ROCPRIM_DEVICE ROCPRIM_INLINE T get_exclusive_prefix() const
    {
        return get_prefix();
    }

private:
    storage_type& storage;
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
