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

#include <cassert>
#include <cstddef>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <iostream>
#include <iterator>
#include <rocprim/intrinsics/thread.hpp>
#include <type_traits>
#include <utility>

#include "../detail/temp_storage.hpp"
#include "../intrinsics.hpp"

#include "../config.hpp"

#include "../block/block_scan.hpp"
#include "../block/block_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<size_t num_buckets, class KeysIterator>
ROCPRIM_KERNEL void kernel_copy_buckets(KeysIterator   keys,
                                        const size_t   rank,
                                        const size_t   size,
                                        size_t*        buckets,
                                        unsigned char* oracles,
                                        size_t*        offset_rank)
{
    using block_scan_offsets = rocprim::block_scan<size_t, num_buckets>;

    __shared__ typename block_scan_offsets::storage_type storage;
    __shared__ size_t bucket_offsets[num_buckets];

    auto idx = threadIdx.x + (threadIdx.y * blockDim.x);

    if (threadIdx.x < num_buckets)
    {
        size_t bucket_size = buckets[threadIdx.x];
        size_t init        = 0;
        size_t bucket_offset;
        block_scan_offsets().exclusive_scan(bucket_size, bucket_offset, init, storage);
        bucket_offsets[threadIdx.x] = bucket_offset;
    }
    rocprim::syncthreads();
    // TODO very naive implementation
    if (idx == 0)
    {
        offset_rank[0] = 0;
        offset_rank[1] = buckets[0];
        size_t stored_offset = 0;
        size_t bucket_size;
        for(int i = 1; i < num_buckets; i++)
        {
            auto offset = bucket_offsets[i];
            bucket_size = buckets[i];
            if (bucket_size == 0)
            {
                continue;
            }
            if (rank < offset)
            {
                offset_rank[0] = stored_offset;
                offset_rank[1] = bucket_size;
                break;
            }
            stored_offset = offset;
        }
    }

    rocprim::syncthreads();

    auto bucket = oracles[idx];
    auto element = keys[idx];

    // Find the maximum number of threads in one block
    constexpr size_t max_num_threads = 1024;
    using block_scan_local_offsets = rocprim::block_scan<size_t, max_num_threads>;

    __shared__ typename block_scan_local_offsets::storage_type storage_bucket;
    size_t index = bucket_offsets[bucket];
    for (int i = 0; i < num_buckets; i++)
    {
        size_t temp;
        size_t current_bucket = bucket == i;
        block_scan_local_offsets().exclusive_scan(current_bucket, temp, 0);
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
                                       const size_t   rank,
                                       const size_t   size,
                                       BinaryFunction compare_function)
{
    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, num_splitters>;
    
    __shared__ typename block_sort_key::storage_type storage;
    // allocate storage in shared memory
    const auto offset = size / num_splitters;
    auto idx = threadIdx.x;

    auto sample_buffer = keys[offset + offset * idx];
    block_sort_key().sort(sample_buffer, storage, compare_function);

    rocprim::syncthreads();

    // Set it in an order easier traversable for binary search
    constexpr size_t tree_width    = num_splitters + 1;
    const size_t     tree_lvl      = 31 - __clz(idx + 1);
    const size_t     tree_step     = tree_width >> tree_lvl;
    const size_t     tree_lvl_id   = idx - (1 << tree_lvl) + 1;
    const size_t     tree_entry_id = (tree_lvl_id * tree_step + tree_step / 2) - 1;

    tree[tree_entry_id] = sample_buffer;
}

template<size_t num_buckets, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_traversal_searchtree(KeysIterator   keys,
                                                KeysIterator   tree,
                                                const size_t   size,
                                                size_t*        buckets,
                                                unsigned char* oracles,
                                                const size_t   tree_depth,
                                                BinaryFunction compare_function)
{
    using Key = typename std::iterator_traits<KeysIterator>::value_type;
    __shared__ size_t shared_buckets[num_buckets];

    auto idx = threadIdx.x + (threadIdx.y * blockDim.x);

    if (threadIdx.x < num_buckets)
    {
        shared_buckets[threadIdx.x] = 0;
    }

    Key element = keys[idx];

    rocprim::syncthreads();
    
    size_t index = 0;
    for (size_t i = 0; i < tree_depth; i++)
    {
        index = 2 * index + (compare_function(element, tree[index]) ? 1 : 2);
    }

    constexpr size_t num_splitters = num_buckets - 1;
    auto bucket = index - num_splitters;

    oracles[idx] = bucket;

    detail::atomic_add(&shared_buckets[bucket], 1);

    rocprim::syncthreads();

    if (threadIdx.x < num_buckets)
    {
        detail::atomic_add(&buckets[threadIdx.x], shared_buckets[threadIdx.x]);
    }
}

template<size_t num_buckets,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction>
ROCPRIM_INLINE hipError_t nth_element_keys_impl(KeysIterator   keys,
                                                KeysIterator   tree,
                                                const size_t   rank,
                                                const size_t   size,
                                                size_t*        buckets,
                                                unsigned char* oracles,
                                                const size_t   tree_depth,
                                                size_t* offset_rank,
                                                BinaryFunction compare_function,
                                                hipStream_t    stream,
                                                bool           debug_synchronous)
{
    constexpr size_t num_splitters = num_buckets - 1;
    constexpr size_t min_size = num_buckets;
    if (size < min_size)
    {
        //TODO do sort here
        return hipSuccess;
    }

    kernel_build_searchtree<num_splitters><<<1, num_splitters, 0, stream>>>(keys, tree, rank, size, compare_function);
    kernel_traversal_searchtree<num_buckets><<<1, size, 0, stream>>>(keys, tree, size, buckets, oracles, tree_depth, compare_function);
    kernel_copy_buckets<num_buckets><<<1, size, 0, stream>>>(keys, rank, size, buckets, oracles, offset_rank);
    size_t h_offset_rank[2];
    hipError_t error = hipMemcpy(&h_offset_rank, offset_rank, 2 * sizeof(size_t), hipMemcpyDeviceToHost);

    if (error != hipSuccess)
    {
        return error;
    }

    size_t offset = h_offset_rank[0];
    std::cout << offset << std::endl;
    size_t bucket_size = h_offset_rank[1];
    std::cout << bucket_size << std::endl;

    size_t h_buckets[num_buckets];
    error = hipMemcpy(&h_buckets, buckets, num_buckets * sizeof(size_t), hipMemcpyDeviceToHost);
    for(int i = 0; i < num_buckets; i++)
    {
        std::cout << h_buckets[i] << ' ';
    }
    std::cout << std::endl;

    error = hipMemsetAsync(buckets, 0, sizeof(size_t) * num_buckets, stream);

    if (error != hipSuccess)
    {
        return error;
    }

    return nth_element_keys_impl<num_buckets>(keys + offset,
                          tree,
                          rank - offset,
                          bucket_size,
                          buckets,
                          oracles,
                          tree_depth,
                          offset_rank,
                          compare_function,
                          stream,
                          debug_synchronous);
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
    constexpr size_t num_buckets = 4;
    constexpr size_t num_splitters = num_buckets - 1;

    KeysIterator tree;
    size_t*      buckets;
    size_t*      bucket_offsets;
    size_t*      offset_rank;
    // Maximum of 256 buckets
    unsigned char* oracles;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&tree, num_splitters),
            detail::temp_storage::ptr_aligned_array(&buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&oracles, size),
            detail::temp_storage::ptr_aligned_array(&offset_rank, 2)));

    hipError_t error = hipMemsetAsync(buckets, 0, sizeof(size_t) * num_buckets, stream);
    if(error != hipSuccess)
    {
        return error;
    }

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(size == 0 || nth >= size)
    {
        return hipSuccess;
    }
    const size_t tree_depth = std::log2(num_buckets);
    nth_element_keys_impl<num_buckets>(keys,
                          tree,
                          nth,
                          size,
                          buckets,
                          oracles,
                          tree_depth,
                          offset_rank,
                          compare_function,
                          stream,
                          debug_synchronous);

    return hipSuccess;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_