// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_BLOCK_SORT_KERNELS_HPP_
#define TEST_BLOCK_SORT_KERNELS_HPP_

constexpr bool is_buildable(unsigned int                  BlockSize,
                            unsigned int                  ItemsPerThread,
                            rocprim::block_sort_algorithm algorithm)
{
    switch(algorithm)
    {
        case rocprim::block_sort_algorithm::merge_sort:
        case rocprim::block_sort_algorithm::stable_merge_sort:
            return (rocprim::detail::is_power_of_two(ItemsPerThread)
                    && rocprim::detail::is_power_of_two(BlockSize));
        case rocprim::block_sort_algorithm::bitonic_sort:
            return ItemsPerThread == 1u
                   || (ItemsPerThread > 1u && rocprim::detail::is_power_of_two(ItemsPerThread)
                       && rocprim::detail::is_power_of_two(BlockSize));
    }
    return false;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeyIterator,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<typename std::iterator_traits<KeyIterator>::value_type>,
    class OffsetT>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(KeyIterator device_key_output,
                                                              OffsetT     size)
{
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + threadIdx.x;
    if(size % ItemsPerBlock == 0)
    {
        key_type key = device_key_output[index];
        rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type, algorithm>
            bsort;
        bsort.sort(key, BinaryOp());
        device_key_output[index] = key;
    }
    else
    {
        key_type key  = device_key_output[index];
        using st_type = typename rocprim::
            block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type>::storage_type;
        ROCPRIM_SHARED_MEMORY st_type                                                 storage;
        rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type> bsort;
        bsort.sort(key,
                   storage,
                   std::min(static_cast<size_t>(ItemsPerBlock), size - block_offset),
                   BinaryOp());
        device_key_output[index] = key;
    }
}

#endif // TEST_BLOCK_SORT_KERNELS_HPP_
