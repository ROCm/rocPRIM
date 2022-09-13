// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
#define ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_

#include "../detail/device_radix_sort.hpp"
#include "../device_merge_sort.hpp"
#include "../specialization/device_radix_single_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator>
inline hipError_t radix_sort_merge_impl(
    void*                                                           temporary_storage,
    size_t&                                                         storage_size,
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_buffer,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_buffer,
    ValuesOutputIterator                                            values_output,
    unsigned int                                                    size,
    bool&                                                           is_result_in_output,
    unsigned int                                                    bit,
    unsigned int                                                    end_bit,
    hipStream_t                                                     stream,
    bool                                                            debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    // TODO: Will be tunable soon
    using sort_config = kernel_config<1024, 1>;

    static constexpr unsigned int sort_block_size       = sort_config::block_size;
    static constexpr unsigned int sort_items_per_thread = sort_config::items_per_thread;
    static constexpr unsigned int sort_items_per_block  = sort_block_size * sort_items_per_thread;
    static_assert(rocprim::detail::is_power_of_two(sort_items_per_block),
                  "For device_radix_merge_sort: sort_items_per_block must be a power of two");

    const unsigned int current_radix_bits = end_bit - bit;

    const unsigned int sort_number_of_blocks = ceiling_div(size, sort_items_per_block);

    // TODO: Will be tunable soon
    using block_merge_config = merge_sort_block_merge_config<1024, 1, 128, 256, 8, 1024 * 512>;

    if(temporary_storage == nullptr)
    {
        if(sort_number_of_blocks > 1)
        {
            return merge_sort_block_merge<sort_items_per_block, block_merge_config>(
                temporary_storage,
                storage_size,
                keys_output,
                values_output,
                size,
                radix_merge_compare<Descending, false, key_type>(),
                stream,
                debug_synchronous);
        }
        else
        {
            storage_size = 1;
            return hipSuccess;
        }
    }

    if(size == size_t(0))
        return hipSuccess;

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << '\n';
        std::cout << "sort_block_size: " << sort_block_size << '\n';
        std::cout << "sort_items_per_thread: " << sort_items_per_thread << '\n';
        std::cout << "sort_items_per_block: " << sort_items_per_block << '\n';
        std::cout << "sort_number_of_blocks: " << sort_number_of_blocks << '\n';
        std::cout << "bit: " << bit << '\n';
        std::cout << "current_radix_bits: " << current_radix_bits << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
        start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sort_single_kernel<sort_block_size, sort_items_per_thread, Descending>),
        dim3(sort_number_of_blocks),
        dim3(sort_block_size),
        0,
        stream,
        keys_input,
        keys_output,
        values_input,
        values_output,
        size,
        bit,
        current_radix_bits);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_block_sort_kernel", size, start);

    if(sort_number_of_blocks > 1)
    {
        if(current_radix_bits == sizeof(key_type) * 8)
        {
            return merge_sort_block_merge<sort_items_per_block, block_merge_config>(
                temporary_storage,
                storage_size,
                keys_output,
                values_output,
                size,
                radix_merge_compare<Descending, false, key_type>(),
                stream,
                debug_synchronous,
                keys_buffer,
                values_buffer);
        }
        else
        {
            return merge_sort_block_merge<sort_items_per_block, block_merge_config>(
                temporary_storage,
                storage_size,
                keys_output,
                values_output,
                size,
                radix_merge_compare<Descending, true, key_type>(bit, current_radix_bits),
                stream,
                debug_synchronous,
                keys_buffer,
                values_buffer);
        }
    }
    is_result_in_output = true;
    return hipSuccess;
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
