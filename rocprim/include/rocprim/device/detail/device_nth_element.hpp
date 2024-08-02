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
#include "../../type_traits.hpp"

#include "device_config_helper.hpp"

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

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

struct nth_element_onesweep_lookback_state
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

    ROCPRIM_DEVICE ROCPRIM_INLINE explicit nth_element_onesweep_lookback_state(
        underlying_type state)
        : state(state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE nth_element_onesweep_lookback_state(prefix_flag     status,
                                                                      underlying_type value)
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

    ROCPRIM_DEVICE ROCPRIM_INLINE static nth_element_onesweep_lookback_state
        load(nth_element_onesweep_lookback_state* ptr)
    {
        underlying_type state = atomic_load(&ptr->state);
        return nth_element_onesweep_lookback_state(state);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void store(nth_element_onesweep_lookback_state* ptr) const
    {
        atomic_store(&ptr->state, this->state);
    }

    static hipError_t reset(nth_element_onesweep_lookback_state* states,
                            unsigned int                         num_total_states,
                            hipStream_t                          stream)
    {
        // All zeroes is equivalent to the empty prefix
        return hipMemsetAsync(states, 0, num_total_states * sizeof(*states), stream);
    }
};

struct n_th_element_iteration_data
{
    size_t bucket_idx;
    size_t offset;
    size_t size;
    bool   equality_bucket;
};

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    kernel_block_sort_impl(KeysIterator keys, const size_t size, BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int stop_recursion_size = params.stop_recursion_size;

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key  = block_load<key_type, stop_recursion_size, 1>;
    using block_sort_key  = block_sort<key_type, stop_recursion_size>;
    using block_store_key = block_store<key_type, stop_recursion_size, 1>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_key::storage_type  load;
        typename block_sort_key::storage_type  sort;
        typename block_store_key::storage_type store;
    } storage;

    key_type sample_buffer[1];

    block_load_key().load(keys, sample_buffer, size, storage.load);

    syncthreads();

    block_sort_key().sort(sample_buffer, storage.sort, size, compare_function);

    syncthreads();

    block_store_key().store(keys, sample_buffer, size, storage.load);
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<config>().stop_recursion_size) void kernel_block_sort(
        KeysIterator keys, const size_t size, BinaryFunction compare_function)
{
    kernel_block_sort_impl<config>(keys, size, compare_function);
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void kernel_find_splitters_impl(KeysIterator   keys,
                                                                    KeysIterator   tree,
                                                                    bool*          equality_buckets,
                                                                    const size_t   size,
                                                                    BinaryFunction compare_function)
{
    constexpr nth_element_config_params params        = device_params<config>();
    constexpr unsigned int              num_splitters = params.number_of_buckets - 1;

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = block_sort<key_type, num_splitters>;

    ROCPRIM_SHARED_MEMORY typename block_sort_key::storage_type storage;

    const auto stride = size / num_splitters;
    auto       idx    = threadIdx.x;

    // Find values to split data in buckets
    auto sample_buffer = keys[stride * idx];
    // Sort the splitters
    block_sort_key().sort(sample_buffer, storage, compare_function);

    tree[idx] = sample_buffer;

    syncthreads();

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
ROCPRIM_KERNEL __launch_bounds__(device_params<config>().number_of_buckets
                                 - 1) void kernel_find_splitters(KeysIterator   keys,
                                                                 KeysIterator   tree,
                                                                 bool*          equality_buckets,
                                                                 const size_t   size,
                                                                 BinaryFunction compare_function)
{
    kernel_find_splitters_impl<config>(keys, tree, equality_buckets, size, compare_function);
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    kernel_count_bucket_sizes_impl(KeysIterator   keys,
                                   KeysIterator   tree,
                                   const size_t   size,
                                   size_t*        buckets,
                                   uint8_t*       oracles,
                                   bool*          equality_buckets,
                                   BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_threads = params.kernel_config.items_per_thread;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;

    // It needs enough splitters to choose from the input
    static_assert(params.stop_recursion_size >= num_buckets,
                  "stop_recursion_size should be larger or equal than the number_of_buckets");
    // It loads in shared memory for the number_of_buckets for every thread
    static_assert(num_threads_per_block >= num_buckets,
                  "num_threads_per_block should be larger or equal than the number_of_buckets");
    // It assumes the number_of_buckets of buckets is a power of two in the traversal
    static_assert(detail::is_power_of_two(num_buckets),
                  "number_of_buckets should be a power of two");
    // The buckets are stored in a uint8_t
    static_assert(num_buckets <= 256, "number_of_buckets should be smaller or equal than 256");

    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key     = block_load<key_type, num_threads_per_block, num_items_per_threads>;
    using block_store_oracle = block_store<uint8_t, num_threads_per_block, num_items_per_threads>;

    ROCPRIM_SHARED_MEMORY struct
    {
        union
        {
            typename block_load_key::storage_type        load;
            typename block_store_oracle::storage_type    store;
            uninitialized_array<key_type, num_splitters> buffer;
        };
        unsigned int shared_buckets[num_buckets];
        bool         shared_equality_buckets[num_buckets];
    } storage;

    if(threadIdx.x < num_buckets)
    {
        storage.shared_buckets[threadIdx.x]          = 0;
        storage.shared_equality_buckets[threadIdx.x] = equality_buckets[threadIdx.x];
    }

    if(threadIdx.x < num_splitters)
    {
        storage.buffer.emplace(threadIdx.x, tree[threadIdx.x]);
    }
    const auto& search_tree = storage.buffer.get_unsafe_array();

    key_type     elements[num_items_per_threads];
    const size_t offset            = blockIdx.x * num_items_per_block;
    const bool   is_complete_block = offset + num_items_per_block <= size;

    if(is_complete_block)
    {
        block_load_key().load(keys + offset, elements, storage.load);
    }
    else
    {
        block_load_key().load(keys + offset, elements, size - offset, storage.load);
    }

    syncthreads();

    uint8_t local_oracles[num_items_per_threads];
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        auto idx = offset + threadIdx.x * num_items_per_threads + item;
        if(idx < size)
        {
            const auto   element = elements[item];
            unsigned int left    = 0;
            unsigned int right   = num_splitters;
            unsigned int bucket;
            // Binary search through splitters to put in bucket
            for(unsigned int i = 0; i < Log2<num_buckets>::VALUE; i++)
            {
                const unsigned int mid  = (left + right) >> 1;
                const bool         comp = compare_function(element, search_tree[mid]);
                right                   = comp ? mid : right;
                left                    = comp ? left : mid;
                bucket                  = right;
            }

            // Checks if the bucket before is an equality bucket for the current value
            if(bucket > 0 && storage.shared_equality_buckets[bucket - 1]
               && element == search_tree[bucket - 1])
            {
                bucket = bucket - 1;
            }

            local_oracles[item] = bucket;

            atomic_add(&storage.shared_buckets[bucket], 1);
        }
    }

    syncthreads();

    if(is_complete_block)
    {
        block_store_oracle().store(oracles + offset, local_oracles, storage.store);
    }
    else
    {
        block_store_oracle().store(oracles + offset, local_oracles, size - offset, storage.store);
    }

    if(threadIdx.x < num_buckets)
    {
        atomic_add(&buckets[threadIdx.x], storage.shared_buckets[threadIdx.x]);
    }
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<config>()
        .kernel_config.block_size) void kernel_count_bucket_sizes(KeysIterator   keys,
                                                                  KeysIterator   tree,
                                                                  const size_t   size,
                                                                  size_t*        buckets,
                                                                  uint8_t*       oracles,
                                                                  bool*          equality_buckets,
                                                                  BinaryFunction compare_function)
{
    kernel_count_bucket_sizes_impl<config>(keys,
                                           tree,
                                           size,
                                           buckets,
                                           oracles,
                                           equality_buckets,
                                           compare_function);
}

template<class config>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    kernel_find_nth_element_bucket_impl(size_t*                      buckets,
                                        n_th_element_iteration_data* nth_element_data,
                                        bool*                        equality_buckets,
                                        const size_t                 rank)

{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets = params.number_of_buckets;

    using block_scan_buckets = block_scan<size_t, num_buckets>;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename block_scan_buckets::storage_type scan;
        size_t                                    bucket_offsets[num_buckets];
    } storage;

    size_t bucket_size = buckets[threadIdx.x];
    size_t bucket_offset;
    // Calculate the global offset of the buckets based on bucket sizes
    block_scan_buckets().exclusive_scan(bucket_size, bucket_offset, 0, storage.scan);

    storage.bucket_offsets[threadIdx.x] = bucket_offset;

    syncthreads();

    size_t num_buckets_before;

    // Find in which bucket the nth element sits
    bool in_nth = storage.bucket_offsets[threadIdx.x] <= rank;
    block_scan_buckets().inclusive_scan(in_nth, num_buckets_before, storage.scan);

    if(threadIdx.x == (num_buckets - 1))
    {
        // Store nth_element data
        size_t nth_element                = num_buckets_before - 1;
        nth_element_data->offset          = storage.bucket_offsets[nth_element];
        nth_element_data->size            = buckets[nth_element];
        nth_element_data->equality_bucket = equality_buckets[nth_element];
        nth_element_data->bucket_idx      = nth_element;
    }
}

template<class config>
ROCPRIM_KERNEL __launch_bounds__(
    device_params<config>()
        .number_of_buckets) void kernel_find_nth_element_bucket(size_t* buckets,
                                                                n_th_element_iteration_data*
                                                                             nth_element_data,
                                                                bool*        equality_buckets,
                                                                const size_t rank)

{
    kernel_find_nth_element_bucket_impl<config>(buckets, nth_element_data, equality_buckets, rank);
}

template<class config, unsigned int NumPartitions, class KeysIterator>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    kernel_copy_buckets_impl(KeysIterator                         keys,
                             const size_t                         size,
                             uint8_t*                             oracles,
                             nth_element_onesweep_lookback_state* lookback_states,
                             n_th_element_iteration_data*         nth_element_data,
                             KeysIterator                         keys_buffer)
{
    using key_type = typename std::iterator_traits<KeysIterator>::value_type;

    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.number_of_buckets;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_threads_per_block = params.kernel_config.block_size;
    constexpr unsigned int num_items_per_threads = params.kernel_config.items_per_thread;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;
    constexpr unsigned int num_partitions        = NumPartitions;

    using block_load_bucket_t  = block_load<uint8_t, num_threads_per_block, num_items_per_threads>;
    using block_rank_t         = rocprim::block_radix_rank<num_threads_per_block,
                                                   Log2<num_partitions + 1>::VALUE,
                                                   params.radix_rank_algorithm>;
    using block_load_element_t = block_load<key_type, num_threads_per_block, num_items_per_threads>;

    static_assert(block_rank_t::digits_per_thread == 1,
                  "The digits_per_thread is assumed to be one.");

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_bucket_t::storage_type  load_bucket;
        typename block_rank_t::storage_type         rank;
        typename block_load_element_t::storage_type load_element;
        size_t                                      buckets_block_offsets_shared[num_partitions];
    } storage;

    uint8_t buckets[num_items_per_threads];

    const size_t nth_element = nth_element_data->bucket_idx;

    const size_t offset            = blockIdx.x * num_items_per_block;
    const bool   is_complete_block = offset + num_items_per_block <= size;

    if(is_complete_block)
    {
        block_load_bucket_t().load(oracles + offset, buckets, storage.load_bucket);
    }
    else
    {
        const size_t valid = size - offset;
        block_load_bucket_t().load(oracles + offset,
                                   buckets,
                                   valid,
                                   num_partitions,
                                   storage.load_bucket);
    }

    const size_t thread_id = (threadIdx.x * num_items_per_threads) + offset;

    if(is_complete_block)
    {
        ROCPRIM_UNROLL
        for(size_t item = 0; item < num_items_per_threads; item++)
        {
            auto bucket = buckets[item];
            // All buckets before the nth element go in bucket 0 in nth bucket 1 and after nth element go in bucket 2
            buckets[item] = ((nth_element > 0) && (bucket >= nth_element))
                            + ((nth_element < num_splitters) && (bucket > nth_element));
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(size_t item = 0; item < num_items_per_threads; item++)
        {
            auto         bucket = buckets[item];
            const size_t idx    = item + thread_id;
            if(idx < size)
            {
                buckets[item] = ((nth_element > 0) && (bucket >= nth_element))
                                + ((nth_element < num_splitters) && (bucket > nth_element));
            }
        }
    }

    syncthreads();

    unsigned int ranks[num_items_per_threads];
    unsigned int partition_prefix[block_rank_t::digits_per_thread];
    unsigned int partition_counts[block_rank_t::digits_per_thread];
    block_rank_t().rank_keys(
        buckets,
        ranks,
        storage.rank,
        [](const uint8_t& key) { return key; },
        partition_prefix,
        partition_counts);

    syncthreads();

    const unsigned int partition = threadIdx.x;
    if(partition < num_partitions)
    {
        nth_element_onesweep_lookback_state* block_state
            = &lookback_states[blockIdx.x * num_partitions + partition];
        nth_element_onesweep_lookback_state(nth_element_onesweep_lookback_state::PARTIAL,
                                            partition_counts[0])
            .store(block_state);

        unsigned int exclusive_prefix  = 0;
        unsigned int lookback_block_id = blockIdx.x;
        // The main back tracking loop.
        while(lookback_block_id > 0)
        {
            --lookback_block_id;
            nth_element_onesweep_lookback_state* lookback_state_ptr
                = &lookback_states[lookback_block_id * num_partitions + partition];
            nth_element_onesweep_lookback_state lookback_state
                = nth_element_onesweep_lookback_state::load(lookback_state_ptr);
            while(lookback_state.status() == nth_element_onesweep_lookback_state::EMPTY)
            {
                lookback_state = nth_element_onesweep_lookback_state::load(lookback_state_ptr);
            }

            exclusive_prefix += lookback_state.value();
            if(lookback_state.status() == nth_element_onesweep_lookback_state::COMPLETE)
            {
                break;
            }
        }

        // Update the state for the current block.
        const unsigned int inclusive_digit_prefix = exclusive_prefix + partition_counts[0];
        // Note that this should not deadlock, as HSA guarantees that blocks with a lower block ID launch before
        // those with a higher block id.
        nth_element_onesweep_lookback_state(nth_element_onesweep_lookback_state::COMPLETE,
                                            inclusive_digit_prefix)
            .store(block_state);

        size_t global_offset = 0;

        // The global offsets based on the nth element bucket.
        if(partition > 0)
        {
            const size_t nth_bucket_offset = nth_element_data->offset;
            global_offset += nth_bucket_offset;
        }

        if((partition > 0 && nth_element == 0) || partition > 1)
        {
            const size_t nth_bucket_size = nth_element_data->size;
            global_offset += nth_bucket_size;
        }

        // Subtract the exclusive digit prefix from the global offset here, since we already ordered the keys in shared
        // memory.
        storage.buckets_block_offsets_shared[threadIdx.x]
            = global_offset - partition_prefix[0] + exclusive_prefix;
    }

    syncthreads();

    key_type elements[num_items_per_threads];
    if(is_complete_block)
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

    syncthreads();

    // Scatter the keys based on their placement in the buckets
    ROCPRIM_UNROLL
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        const size_t idx = item + thread_id;
        if(idx < size)
        {
            const auto   bucket = buckets[item];
            const size_t index  = storage.buckets_block_offsets_shared[bucket] + ranks[item];
            keys_buffer[index]  = elements[item];
        }
    }
}

template<class config, unsigned int NumPartitions, class KeysIterator>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<config>().kernel_config.block_size) void kernel_copy_buckets(
        KeysIterator                         keys,
        const size_t                         size,
        uint8_t*                             oracles,
        nth_element_onesweep_lookback_state* lookback_states,
        n_th_element_iteration_data*         nth_element_data,
        KeysIterator                         keys_buffer)
{
    kernel_copy_buckets_impl<config, NumPartitions>(keys,
                                                    size,
                                                    oracles,
                                                    lookback_states,
                                                    nth_element_data,
                                                    keys_buffer);
}

template<class config, unsigned int NumPartitions, class KeysIterator, class BinaryFunction>
ROCPRIM_INLINE hipError_t
    nth_element_keys_impl(KeysIterator                         keys,
                          KeysIterator                         keys_buffer,
                          KeysIterator                         tree,
                          size_t                               rank,
                          size_t                               size,
                          size_t*                              buckets,
                          bool*                                equality_buckets,
                          uint8_t*                             oracles,
                          nth_element_onesweep_lookback_state* lookback_states,
                          const unsigned int                   num_buckets,
                          const unsigned int                   stop_recursion_size,
                          const unsigned int                   num_threads_per_block,
                          const unsigned int                   num_items_per_threads,
                          n_th_element_iteration_data*         nth_element_data,
                          BinaryFunction                       compare_function,
                          hipStream_t                          stream,
                          bool                                 debug_synchronous)
{
    constexpr unsigned int num_partitions      = NumPartitions;
    const unsigned int     num_splitters       = num_buckets - 1;
    const unsigned int     num_items_per_block = num_threads_per_block * num_items_per_threads;
    size_t                 iteration           = 0;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    const auto start_timer = [&start, debug_synchronous]()
    {
        if(debug_synchronous)
        {
            start = std::chrono::high_resolution_clock::now();
        }
    };

    while(size >= stop_recursion_size)
    {
        const unsigned int num_blocks = ceiling_div(size, num_items_per_block);

        if(debug_synchronous)
        {
            std::cout << "-----" << '\n';
            std::cout << "size: " << size << '\n';
            std::cout << "rank: " << rank << '\n';
            std::cout << "iteration: " << iteration++ << '\n';
        }

        hipError_t error = hipMemsetAsync(buckets, 0, sizeof(*buckets) * num_buckets, stream);

        if(error != hipSuccess)
        {
            return error;
        }

        error
            = hipMemsetAsync(equality_buckets, 0, sizeof(*equality_buckets) * num_buckets, stream);

        if(error != hipSuccess)
        {
            return error;
        }

        // Reset lookback scan states to zero, indicating empty prefix.
        error = nth_element_onesweep_lookback_state::reset(lookback_states,
                                                           num_partitions * num_blocks,
                                                           stream);

        if(error != hipSuccess)
        {
            return error;
        }

        start_timer();
        kernel_find_splitters<config>
            <<<1, num_splitters, 0, stream>>>(keys, tree, equality_buckets, size, compare_function);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_splitters", size, start);

        start_timer();
        kernel_count_bucket_sizes<config>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                               tree,
                                                               size,
                                                               buckets,
                                                               oracles,
                                                               equality_buckets,
                                                               compare_function);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_count_bucket_sizes", size, start);

        start_timer();
        kernel_find_nth_element_bucket<config>
            <<<1, num_buckets, 0, stream>>>(buckets, nth_element_data, equality_buckets, rank);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_nth_element_bucket", size, start);

        start_timer();
        kernel_copy_buckets<config, num_partitions>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                               size,
                                                               oracles,
                                                               lookback_states,
                                                               nth_element_data,
                                                               keys_buffer);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_copy_buckets", size, start);

        // Copy the results in keys_buffer back to the keys
        error = hipMemcpyAsync(keys,
                               keys_buffer,
                               sizeof(*keys) * size,
                               hipMemcpyDeviceToDevice,
                               stream);

        if(error != hipSuccess)
        {
            return error;
        }

        n_th_element_iteration_data h_nth_element_data;
        error = hipMemcpyAsync(&h_nth_element_data,
                               nth_element_data,
                               sizeof(h_nth_element_data),
                               hipMemcpyDeviceToHost,
                               stream);
        if(error != hipSuccess)
        {
            return error;
        }

        error = hipStreamSynchronize(stream);

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

        size = bucket_size;
        // rank is the n from the nth-element, but it reduces based on the previous iteration
        rank = rank - offset;
        keys = keys + offset;
    }

    start_timer();
    kernel_block_sort<config><<<1, stop_recursion_size, 0, stream>>>(keys, size, compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_block_sort", size, start);
    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
