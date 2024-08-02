// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_

#include "../../block/block_load.hpp"
#include "../../block/block_radix_rank.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_sort.hpp"
#include "../../block/block_store.hpp"

#include "../../config.hpp"
#include "../../intrinsics.hpp"

#include "device_config_helper.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/config.hpp>

#include <iostream>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

struct onesweep_lookback_state
{
    // The two most significant bits are used to indicate the status of the prefix - leaving the other 30 bits for the
    // counter value.
    using underlying_type = uint32_t;

    static constexpr unsigned int state_bits = 8u * sizeof(underlying_type);

    enum prefix_flag : underlying_type
    {
        EMPTY    = 0,
        PARTIAL  = 1u << (state_bits - 2),
        COMPLETE = 2u << (state_bits - 2)
    };

    static constexpr underlying_type status_mask = 3u << (state_bits - 2);
    static constexpr underlying_type value_mask  = ~status_mask;

    underlying_type state;

    ROCPRIM_DEVICE ROCPRIM_INLINE explicit onesweep_lookback_state(underlying_type state)
        : state(state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE onesweep_lookback_state(prefix_flag status, underlying_type value)
        : state(static_cast<underlying_type>(status) | value)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE underlying_type value() const
    {
        return this->state & value_mask;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE prefix_flag status() const
    {
        return static_cast<prefix_flag>(this->state & status_mask);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static onesweep_lookback_state load(onesweep_lookback_state* ptr)
    {
        underlying_type state = ::rocprim::detail::atomic_load(&ptr->state);
        return onesweep_lookback_state(state);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void store(onesweep_lookback_state* ptr) const
    {
        ::rocprim::detail::atomic_store(&ptr->state, this->state);
    }
};

struct nth_element_data_type
{
    size_t nth_element;
    size_t offset;
    size_t size;
    bool   equality_bucket;
};

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void
    kernel_block_sort(KeysIterator keys, const size_t size, BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int stop_recursion_size = params.stop_recursion_size;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, stop_recursion_size>;

    ROCPRIM_SHARED_MEMORY typename block_sort_key::storage_type storage;

    size_t idx = threadIdx.x;
    Key    sample_buffer;

    if(idx < size)
    {
        sample_buffer = keys[idx];
    }

    block_sort_key().sort(sample_buffer, storage, size, compare_function);

    if(idx < size)
    {
        keys[idx] = sample_buffer;
    }
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_find_splitters(KeysIterator   keys,
                                          KeysIterator   tree,
                                          bool*          equality_buckets,
                                          const size_t   size,
                                          BinaryFunction compare_function)
{
    constexpr nth_element_config_params params        = device_params<config>();
    constexpr unsigned int              num_splitters = params.number_of_buckets - 1;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, num_splitters>;

    ROCPRIM_SHARED_MEMORY typename block_sort_key::storage_type storage;

    const auto offset = size / num_splitters;
    auto       idx    = threadIdx.x;

    // Find values to split data in buckets
    auto sample_buffer = keys[offset + offset * idx];
    // Sort the splitters
    block_sort_key().sort(sample_buffer, storage, compare_function);

    tree[idx] = sample_buffer;

    rocprim::syncthreads();

    bool equality_bucket = false;
    if(idx > 0)
    {
        // Check if the splitter value before has the same value and the value after is different
        // If so use the bucket for items equal to the splitter value.
        equality_bucket
            = tree[idx - 1] == sample_buffer
              && (idx == num_splitters - 1 || compare_function(sample_buffer, tree[idx + 1]));
    }

    equality_buckets[idx] = equality_bucket;
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_count_bucket_sizes(KeysIterator   keys,
                                              KeysIterator   tree,
                                              const size_t   size,
                                              size_t*        buckets,
                                              unsigned char* oracles,
                                              bool*          equality_buckets,
                                              const size_t   tree_depth,
                                              BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_threads = params.kernel_config.items_per_thread;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key = rocprim::block_load<Key, num_threads_per_block, num_items_per_threads>;
    using block_store_oracle
        = rocprim::block_store<unsigned char, num_threads_per_block, num_items_per_threads>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_key::storage_type     load;
        typename block_store_oracle::storage_type store;
    } storage;

    struct storage_type_
    {
        Key search_tree[num_splitters];
    };
    using storage_type = detail::raw_storage<storage_type_>;

    ROCPRIM_SHARED_MEMORY storage_type raw_storage;
    ROCPRIM_SHARED_MEMORY unsigned int shared_buckets[num_buckets];
    ROCPRIM_SHARED_MEMORY bool         shared_equality_buckets[num_buckets];

    if(threadIdx.x < num_buckets)
    {
        shared_buckets[threadIdx.x]          = 0;
        shared_equality_buckets[threadIdx.x] = equality_buckets[threadIdx.x];
    }

    storage_type_& raw_storage_ = raw_storage.get();
    if(threadIdx.x < num_splitters)
    {
        raw_storage_.search_tree[threadIdx.x] = tree[threadIdx.x];
    }

    Key          elements[num_items_per_threads];
    const size_t offset         = blockIdx.x * num_items_per_block;
    const bool   complete_block = offset + num_items_per_block <= size;

    if(complete_block)
    {
        block_load_key().load(keys + offset, elements, storage.load);
    }
    else
    {
        block_load_key().load(keys + offset, elements, size - offset, storage.load);
    }

    rocprim::syncthreads();

    unsigned char local_oracles[num_items_per_threads];
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        auto idx = offset + threadIdx.x * num_items_per_threads + item;
        if(idx < size)
        {
            auto         element = elements[item];
            unsigned int bucket  = num_splitters / 2;
            auto         diff    = num_buckets / 2;
            // Binary search through splitters to put in bucket
            for(unsigned int i = 0; i < tree_depth - 1; i++)
            {
                diff = diff / 2;
                bucket
                    += compare_function(element, raw_storage_.search_tree[bucket]) ? -diff : diff;
            }

            if(!compare_function(element, raw_storage_.search_tree[bucket]))
            {
                bucket++;
            }

            // Checks if the bucket before is an equality bucket for the current value
            if(bucket > 0 && shared_equality_buckets[bucket - 1]
               && element == raw_storage_.search_tree[bucket - 1])
            {
                bucket = bucket - 1;
            }

            local_oracles[item] = bucket;

            detail::atomic_add(&shared_buckets[bucket], 1);
        }
    }

    rocprim::syncthreads();

    if(complete_block)
    {
        block_store_oracle().store(oracles + offset, local_oracles, storage.store);
    }
    else
    {
        block_store_oracle().store(oracles + offset, local_oracles, size - offset, storage.store);
    }

    if(threadIdx.x < num_buckets)
    {
        detail::atomic_add(&buckets[threadIdx.x], shared_buckets[threadIdx.x]);
    }
}

template<class config>
ROCPRIM_KERNEL void kernel_find_nth_element_bucket(size_t*                buckets,
                                                   nth_element_data_type* nth_element_data,
                                                   bool*                  equality_buckets,
                                                   const size_t           rank)

{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets = params.number_of_buckets;

    using block_scan_buckets = rocprim::block_scan<size_t, num_buckets>;

    ROCPRIM_SHARED_MEMORY typename block_scan_buckets::storage_type storage;

    ROCPRIM_SHARED_MEMORY size_t bucket_offsets[num_buckets];

    size_t bucket_offset;
    size_t bucket_size = buckets[threadIdx.x];
    // Calculate the global offset of the buckets based on bucket sizes
    block_scan_buckets().exclusive_scan(bucket_size, bucket_offset, 0, storage);

    bucket_offsets[threadIdx.x] = bucket_offset;

    rocprim::syncthreads();

    size_t num_buckets_before;

    // Find in which bucket the nth element sits
    bool in_nth = bucket_offset <= rank;
    block_scan_buckets().inclusive_scan(in_nth, num_buckets_before, storage);

    if(threadIdx.x == (num_buckets - 1))
    {
        // Store nth_element data
        size_t nth_element                = num_buckets_before - 1;
        nth_element_data->offset          = bucket_offsets[nth_element];
        nth_element_data->size            = buckets[nth_element];
        nth_element_data->equality_bucket = equality_buckets[nth_element];
        nth_element_data->nth_element     = nth_element;
    }
}

template<class config,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type>
ROCPRIM_KERNEL void kernel_copy_buckets(KeysIterator             keys,
                                        const size_t             size,
                                        unsigned char*           oracles,
                                        onesweep_lookback_state* lookback_states,
                                        nth_element_data_type*   nth_element_data,
                                        KeysIterator             output)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_threads = params.kernel_config.items_per_thread;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;
    constexpr unsigned int usefull_buckets       = 3;

    using block_load_bucket_t
        = rocprim::block_load<unsigned char, num_threads_per_block, num_items_per_threads>;
    using block_rank_t
        = rocprim::block_radix_rank<num_threads_per_block, 2, params.radix_rank_algorithm>;
    using block_load_element_t
        = rocprim::block_load<Key, num_threads_per_block, num_items_per_threads>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_bucket_t::storage_type  load_bucket;
        typename block_rank_t::storage_type         rank;
        typename block_load_element_t::storage_type load_element;
    } storage;

    ROCPRIM_SHARED_MEMORY size_t buckets_block_offsets_shared[usefull_buckets];
    ROCPRIM_SHARED_MEMORY size_t global_offset[usefull_buckets];

    unsigned char buckets[num_items_per_threads];

    const size_t nth_element = nth_element_data->nth_element;

    // The global offsets based on the nth element bucket.
    if(threadIdx.x < usefull_buckets)
    {
        global_offset[threadIdx.x] = 0;

        if(threadIdx.x > 0)
        {
            const size_t nth_bucket_offset = nth_element_data->offset;
            global_offset[threadIdx.x] += nth_bucket_offset;
        }

        if((threadIdx.x > 0 && nth_element == 0) || threadIdx.x > 1)
        {
            const size_t nth_bucket_size = nth_element_data->size;
            global_offset[threadIdx.x] += nth_bucket_size;
        }
    }

    const size_t offset         = blockIdx.x * num_items_per_block;
    const bool   complete_block = offset + num_items_per_block <= size;

    if(complete_block)
    {
        block_load_bucket_t().load(oracles + offset, buckets, storage.load_bucket);
    }
    else
    {
        const size_t valid = size - offset;
        block_load_bucket_t().load(oracles + offset,
                                   buckets,
                                   valid,
                                   usefull_buckets,
                                   storage.load_bucket);
    }

    const size_t thread_id = (threadIdx.x * num_items_per_threads) + offset;

    ROCPRIM_UNROLL
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        auto bucket = buckets[item];
        if(complete_block)
        {
            // All buckets before the nth element go in bucket 0 in nth bucket 1 and after nth element go in bucket 2
            buckets[item] = ((nth_element > 0) && (bucket >= nth_element))
                            + ((nth_element < num_splitters) && (bucket > nth_element));
        }
        else
        {
            const size_t idx = item + thread_id;
            if(idx < size)
            {
                buckets[item] = ((nth_element > 0) && (bucket >= nth_element))
                                + ((nth_element < num_splitters) && (bucket > nth_element));
            }
        }
    }

    unsigned int ranks[num_items_per_threads];
    unsigned int digit_prefix[block_rank_t::digits_per_thread];
    unsigned int digit_counts[block_rank_t::digits_per_thread];
    block_rank_t().rank_keys(
        buckets,
        ranks,
        storage.rank,
        [](const int& key) { return key; },
        digit_prefix,
        digit_counts);

    rocprim::syncthreads();

    const unsigned int digit = threadIdx.x;
    if(digit < usefull_buckets)
    {
        onesweep_lookback_state* block_state
            = &lookback_states[blockIdx.x * usefull_buckets + digit];
        onesweep_lookback_state(onesweep_lookback_state::PARTIAL, digit_counts[0])
            .store(block_state);

        unsigned int exclusive_prefix  = 0;
        unsigned int lookback_block_id = blockIdx.x;
        // The main back tracking loop.
        while(lookback_block_id > 0)
        {
            --lookback_block_id;
            onesweep_lookback_state* lookback_state_ptr
                = &lookback_states[lookback_block_id * usefull_buckets + digit];
            onesweep_lookback_state lookback_state
                = onesweep_lookback_state::load(lookback_state_ptr);
            while(lookback_state.status() == onesweep_lookback_state::EMPTY)
            {
                lookback_state = onesweep_lookback_state::load(lookback_state_ptr);
            }

            exclusive_prefix += lookback_state.value();
            if(lookback_state.status() == onesweep_lookback_state::COMPLETE)
            {
                break;
            }
        }

        // Update the state for the current block.
        const unsigned int inclusive_digit_prefix = exclusive_prefix + digit_counts[0];
        // Note that this should not deadlock, as HSA guarantees that blocks with a lower block ID launch before
        // those with a higher block id.
        onesweep_lookback_state(onesweep_lookback_state::COMPLETE, inclusive_digit_prefix)
            .store(block_state);

        // Subtract the exclusive digit prefix from the global offset here, since we already ordered the keys in shared
        // memory.
        buckets_block_offsets_shared[threadIdx.x]
            = global_offset[digit] - digit_prefix[0] + exclusive_prefix;
    }

    rocprim::syncthreads();

    Key elements[num_items_per_threads];
    if(complete_block)
    {
        block_load_element_t().load(keys + offset, elements, storage.load_element);
    }
    else
    {
        block_load_element_t().load(keys + offset,
                                    elements,
                                    size - offset,
                                    0,
                                    storage.load_element);
    }

    // Scatter the keys based on their placement in the buckets
    ROCPRIM_UNROLL
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        const size_t idx = item + thread_id;
        if(idx < size)
        {
            const auto   bucket = buckets[item];
            const size_t index  = buckets_block_offsets_shared[bucket] + ranks[item];
            output[index]       = elements[item];
        }
    }
}

template<class config,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction>
ROCPRIM_INLINE hipError_t nth_element_keys_impl(KeysIterator             keys,
                                                KeysIterator             output,
                                                KeysIterator             tree,
                                                const size_t             rank,
                                                const size_t             size,
                                                size_t*                  buckets,
                                                bool*                    equality_buckets,
                                                unsigned char*           oracles,
                                                onesweep_lookback_state* lookback_states,
                                                const unsigned int       num_buckets,
                                                const unsigned int       stop_recursion_size,
                                                const unsigned int       num_threads_per_block,
                                                const unsigned int       num_items_per_threads,
                                                const unsigned int       tree_depth,
                                                nth_element_data_type*   nth_element_data,
                                                BinaryFunction           compare_function,
                                                hipStream_t              stream,
                                                bool                     debug_synchronous,
                                                const size_t             recursion)
{
    const unsigned int num_splitters       = num_buckets - 1;
    const unsigned int num_items_per_block = num_threads_per_block * num_items_per_threads;
    const unsigned int num_blocks          = (size + num_items_per_block - 1) / num_items_per_block;

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << std::endl;
        std::cout << "rank: " << rank << std::endl;
        std::cout << "recursion level: " << recursion << std::endl;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if(size < stop_recursion_size)
    {
        if(debug_synchronous)
        {
            start = std::chrono::high_resolution_clock::now();
        }
        kernel_block_sort<config>
            <<<1, stop_recursion_size, 0, stream>>>(keys, size, compare_function);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_block_sort", size, start);
        return hipSuccess;
    }

    hipError_t error = hipMemsetAsync(buckets, 0, sizeof(size_t) * num_buckets, stream);

    if(error != hipSuccess)
    {
        return error;
    }

    error = hipMemsetAsync(equality_buckets, 0, sizeof(bool) * num_buckets, stream);

    if(error != hipSuccess)
    {
        return error;
    }

    // Reset lookback scan states to zero, indicating empty prefix.
    error = hipMemsetAsync(lookback_states,
                           0,
                           sizeof(onesweep_lookback_state) * 3 * num_blocks,
                           stream);
    if(error != hipSuccess)
    {
        return error;
    }

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_find_splitters<config>
        <<<1, num_splitters, 0, stream>>>(keys, tree, equality_buckets, size, compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_splitters", size, start);
    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_count_bucket_sizes<config>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                           tree,
                                                           size,
                                                           buckets,
                                                           oracles,
                                                           equality_buckets,
                                                           tree_depth,
                                                           compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_count_bucket_sizes", size, start);
    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_find_nth_element_bucket<config>
        <<<1, num_buckets, 0, stream>>>(buckets, nth_element_data, equality_buckets, rank);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_nth_element_bucket", size, start);

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_copy_buckets<config><<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                                                  size,
                                                                                  oracles,
                                                                                  lookback_states,
                                                                                  nth_element_data,
                                                                                  output);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_copy_buckets", size, start);

    error = hipMemcpyAsync(keys, output, sizeof(Key) * size, hipMemcpyDeviceToDevice);

    nth_element_data_type h_nth_element_data;
    error = hipMemcpyAsync(&h_nth_element_data,
                           nth_element_data,
                           sizeof(nth_element_data_type),
                           hipMemcpyDeviceToHost);

    hipDeviceSynchronize();

    if(error != hipSuccess)
    {
        return error;
    }

    size_t offset          = h_nth_element_data.offset;
    size_t bucket_size     = h_nth_element_data.size;
    bool   equality_bucket = h_nth_element_data.equality_bucket;

    // If all values are the same it is already sorted
    if(equality_bucket)
    {
        return hipSuccess;
    }

    return nth_element_keys_impl<config>(keys + offset,
                                         output,
                                         tree,
                                         rank - offset,
                                         bucket_size,
                                         buckets,
                                         equality_buckets,
                                         oracles,
                                         lookback_states,
                                         num_buckets,
                                         stop_recursion_size,
                                         num_threads_per_block,
                                         num_items_per_threads,
                                         tree_depth,
                                         nth_element_data,
                                         compare_function,
                                         stream,
                                         debug_synchronous,
                                         recursion + 1);
}
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
