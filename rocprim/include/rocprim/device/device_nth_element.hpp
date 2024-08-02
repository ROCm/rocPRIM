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

#ifndef ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_

#include <cstddef>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <iostream>
#include <iterator>
#include <rocprim/intrinsics/thread.hpp>

#include "../detail/temp_storage.hpp"
#include "../intrinsics.hpp"

#include "../config.hpp"

#include "../block/block_scan.hpp"
#include "../block/block_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<size_t min_size, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_block_sort(KeysIterator   keys,
                                      const size_t   size,
                                      BinaryFunction compare_function,
                                      hipStream_t    stream,
                                      bool           debug_synchronous)
{
    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, min_size>;

    __shared__ typename block_sort_key::storage_type storage;

    auto idx = threadIdx.x;
    Key sample_buffer;
    if (idx < size)
    {
        sample_buffer = keys[idx];
    }

    block_sort_key().sort(sample_buffer, storage, size, compare_function);

    rocprim::syncthreads();
    if (idx < size)
    {
        keys[idx] = sample_buffer;
    }
}

template<size_t num_buckets, size_t num_threads_per_block, class KeysIterator>
ROCPRIM_KERNEL void kernel_copy_buckets(KeysIterator   keys,
                                        const size_t   rank,
                                        const size_t   size,
                                        size_t*        buckets,
                                        size_t*        buckets_per_block,
                                        bool*          equality_buckets,
                                        unsigned char* oracles,
                                        size_t*        nth_element_data)
{
    using block_scan_offsets = rocprim::block_scan<size_t, num_threads_per_block>;

    __shared__ typename block_scan_offsets::storage_type storage;
    __shared__ size_t bucket_offsets[num_buckets];
    __shared__ size_t block_bucket_offsets[num_buckets];

    auto idx = threadIdx.x + (blockDim.x * blockIdx.x);

    size_t bucket_offset;
    size_t bucket_size = threadIdx.x < num_buckets ? buckets[threadIdx.x] : 0;
    size_t init        = 0;
    block_scan_offsets().exclusive_scan(bucket_size, bucket_offset, init, storage);

    if (threadIdx.x < num_buckets)
    {
        bucket_offsets[threadIdx.x] = bucket_offset;
        block_bucket_offsets[threadIdx.x] = 0;
        for (size_t i = 0; i < blockIdx.x; i++)
        {
            block_bucket_offsets[threadIdx.x] += buckets_per_block[threadIdx.x + i * num_buckets];
        }
    }

    rocprim::syncthreads();

    size_t num_buckets_before;
    // Find offset and bucket size of nth element
    using block_scan_find_nth = rocprim::block_scan<size_t, num_threads_per_block>;

    __shared__ typename block_scan_offsets::storage_type storage_find;

    bool in_nth = threadIdx.x < num_buckets ? (bucket_offsets[idx] <= rank) : 0;

    block_scan_find_nth().inclusive_scan(in_nth, num_buckets_before, storage_find);

    if(idx == (num_buckets - 1))
    {
        auto nth_element = num_buckets_before - 1;
        nth_element_data[0]   = bucket_offsets[nth_element];
        nth_element_data[1]   = buckets[nth_element];
        nth_element_data[2]   = equality_buckets[nth_element];
    }

    if (idx >= size)
    {
        return;
    }
    auto bucket  = oracles[idx];
    auto element = keys[idx];

    using block_scan_local_offsets = rocprim::block_scan<size_t, num_threads_per_block>;

    __shared__ typename block_scan_local_offsets::storage_type storage_bucket;
    size_t index = bucket_offsets[bucket] + block_bucket_offsets[bucket];
    for (int i = 0; i < num_buckets; i++)
    {
        size_t temp;
        size_t current_bucket = bucket == i;
        block_scan_local_offsets().exclusive_scan(current_bucket, temp, 0, storage_bucket);

        if (current_bucket)
        {
            index += temp;
        }
    }
    keys[index] = element;
}

template<size_t num_splitters, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_build_searchtree(KeysIterator   keys,
                                            KeysIterator   tree,
                                            bool*          equality_buckets,
                                            const size_t   rank,
                                            const size_t   size,
                                            BinaryFunction compare_function,
                                            size_t         recursion)
{
    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, num_splitters>;
    
    __shared__ typename block_sort_key::storage_type storage;
    // allocate storage in shared memory
    const auto offset = size / num_splitters;
    auto idx = threadIdx.x;

    // Change offset based on recursion level
    auto index = (recursion + offset + offset * idx) % size;
    auto sample_buffer = keys[offset + offset * idx];
    block_sort_key().sort(sample_buffer, storage, compare_function);

    rocprim::syncthreads();

    tree[idx] = sample_buffer;

    rocprim::syncthreads();

    bool equality_bucket = false;
    if(idx > 0)
    {
        equality_bucket
            = tree[idx - 1] == tree[idx]
              && (idx == num_splitters-1 || tree[idx] < tree[idx + 1]);
    }

    equality_buckets[idx] = equality_bucket;
}

template<size_t num_buckets, size_t num_threads_per_block, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_traversal_searchtree(KeysIterator   keys,
                                                KeysIterator   tree,
                                                const size_t   size,
                                                size_t*        buckets,
                                                size_t*        buckets_per_block,
                                                bool*          equality_buckets,
                                                unsigned char* oracles,
                                                const size_t   tree_depth,
                                                BinaryFunction compare_function)
{
    constexpr size_t num_splitters = num_buckets - 1;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;
    
    __shared__ size_t shared_buckets[num_buckets];
    

    auto idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (threadIdx.x < num_buckets)
    {
        
        shared_buckets[threadIdx.x] = 0;
    }

    __shared__ Key search_tree[num_splitters];
    if (threadIdx.x < num_splitters)
    {
        search_tree[threadIdx.x] = tree[threadIdx.x];
    }

    Key element;
    if (idx < size)
    {
        element = keys[idx];
    }
    
    rocprim::syncthreads();
    
    if (idx < size)
    {

        size_t bucket = num_splitters;
        for (size_t i = 0; i < num_splitters; i++)
        {
            if (compare_function(element, search_tree[i]))
            {
                bucket = i;
                break;
            }
        }

        // TODO make working binary search
        // size_t bucket = num_splitters / 2;
        // auto diff = num_buckets / 2;
        // for (size_t i = 0; i < tree_depth-1; i++)
        // {
        //     diff = diff / 2;
        //     bucket += compare_function(element, search_tree[bucket]) ? -diff : diff;
        // }

        if (idx > 0 && equality_buckets[bucket - 1] && element == search_tree[bucket - 1])
        {
            bucket = bucket - 1;
        }

        oracles[idx] = bucket;

        detail::atomic_add(&shared_buckets[bucket], 1);
    }

    rocprim::syncthreads();

    if (threadIdx.x < num_buckets)
    {
        buckets_per_block[threadIdx.x + blockIdx.x * num_buckets] = shared_buckets[threadIdx.x];
        detail::atomic_add(&buckets[threadIdx.x], shared_buckets[threadIdx.x]);
    }
}

template<size_t num_buckets,
         size_t min_size,
         size_t num_threads_per_block,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction>
ROCPRIM_INLINE hipError_t nth_element_keys_impl(KeysIterator   keys,
                                                KeysIterator   tree,
                                                const size_t   rank,
                                                const size_t   size,
                                                size_t*        buckets,
                                                size_t*        buckets_per_block,
                                                bool*          equality_buckets,
                                                unsigned char* oracles,
                                                const size_t   tree_depth,
                                                size_t*        nth_element_data,
                                                BinaryFunction compare_function,
                                                hipStream_t    stream,
                                                bool           debug_synchronous,
                                                const size_t   recursion)
{
    constexpr size_t num_splitters = num_buckets - 1;
    const size_t num_blocks = (size / num_threads_per_block) + 1;

    if (debug_synchronous)
    {
        std::cout << "Size: " << size << std::endl;
        std::cout << "Rank: " << rank << std::endl;
        std::cout << "Recursion level: " << recursion << std::endl;
    }

    if(size < min_size)
    {
        kernel_block_sort<min_size>
            <<<1, min_size, 0, stream>>>(keys, size, compare_function, stream, debug_synchronous);
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

    // Currently launches power of 2 minus 1 threads
    kernel_build_searchtree<num_splitters>
        <<<1, num_splitters, 0, stream>>>(keys, tree, equality_buckets, rank, size, compare_function, recursion);
    kernel_traversal_searchtree<num_buckets, num_threads_per_block>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys, tree, size, buckets, buckets_per_block, equality_buckets, oracles, tree_depth, compare_function);
    kernel_copy_buckets<num_buckets, num_threads_per_block>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys, rank, size, buckets, buckets_per_block, equality_buckets, oracles, nth_element_data);

    size_t h_nth_element_data[3];
    error = hipMemcpy(&h_nth_element_data, nth_element_data,3 * sizeof(size_t), hipMemcpyDeviceToHost);

    if (error != hipSuccess)
    {
        return error;
    }

    size_t offset = h_nth_element_data[0];
    size_t bucket_size = h_nth_element_data[1];
    bool equality_bucket = h_nth_element_data[2];

    if(false)
    {
        std::cout << "Keys: " << std::endl;
        auto new_keys = keys + offset;
        for(int i = 0; i < bucket_size; i++)
        {
            std::cout << new_keys[i] << ", ";
        }
        std::cout << std::endl;
        Key h_tree[num_splitters];
        hipMemcpy(&h_tree, tree,num_splitters * sizeof(Key), hipMemcpyDeviceToHost);
        std::cout << "Tree: " << std::endl;
        for(int i = 0; i < num_splitters; i++)
        {
            std::cout << h_tree[i] << ", ";
        }
        std::cout << std::endl;
        size_t h_buckets[num_buckets];
        hipMemcpy(&h_buckets, buckets,num_buckets * sizeof(size_t), hipMemcpyDeviceToHost);
        std::cout << "Buckets: " << std::endl;
        for(int i = 0; i < num_buckets; i++)
        {
            std::cout << h_buckets[i] << ", ";
        }
        std::cout << std::endl;
         bool h_equal[num_buckets];
        hipMemcpy(&h_equal, equality_buckets,num_buckets * sizeof(bool), hipMemcpyDeviceToHost);
        std::cout << "Equality buckets: " << std::endl;
        for(int i = 0; i < num_buckets; i++)
        {
            std::cout << h_equal[i] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << equality_bucket << std::endl;

    if (equality_bucket)
    {
        return hipSuccess;
    }

    return nth_element_keys_impl<num_buckets, min_size, num_threads_per_block>(keys + offset,
                          tree,
                          rank - offset,
                          bucket_size,
                          buckets,
                          buckets_per_block,
                          equality_buckets,
                          oracles,
                          tree_depth,
                          nth_element_data,
                          compare_function,
                          stream,
                          debug_synchronous,
                          recursion + 1);
}

template<class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysIterator>::value_type>>
ROCPRIM_INLINE hipError_t nth_element_keys(void*          temporary_storage,
                                                          size_t&        storage_size,
                                                          KeysIterator   keys,
                                                          size_t         nth,
                                                          size_t         size,
                                                          BinaryFunction compare_function
                                                          = BinaryFunction(),
                                                          hipStream_t stream            = 0,
                                                          bool        debug_synchronous = false)
{
    constexpr size_t num_buckets = 64;
    constexpr size_t num_splitters = num_buckets - 1;
    constexpr size_t min_size = num_buckets;

    Key*    tree           = nullptr;
    size_t* buckets        = nullptr;
    size_t* buckets_per_block = nullptr;
    size_t* nth_element_data    = nullptr;
    // Maximum of 256 buckets
    unsigned char* oracles       = nullptr;
    bool*          equality_buckets = nullptr;

    // Find the maximum number of threads in one block
    constexpr size_t num_threads_per_block = 1024;
    const size_t num_blocks = (size / num_threads_per_block) + 1;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&tree, num_splitters),
            detail::temp_storage::ptr_aligned_array(&equality_buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&buckets_per_block, num_buckets * num_blocks),
            detail::temp_storage::ptr_aligned_array(&oracles, size),
            detail::temp_storage::ptr_aligned_array(&nth_element_data, 3)));

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(size == 0 || nth >= size)
    {
        return hipSuccess;
    }

    const size_t tree_depth = std::log2(num_buckets);
    nth_element_keys_impl<num_buckets, min_size, num_threads_per_block>(keys,
                                                 tree,
                                                 nth,
                                                 size,
                                                 buckets,
                                                 buckets_per_block,
                                                 equality_buckets,
                                                 oracles,
                                                 tree_depth,
                                                 nth_element_data,
                                                 compare_function,
                                                 stream,
                                                 debug_synchronous,
                                                 0);

    return hipSuccess;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_