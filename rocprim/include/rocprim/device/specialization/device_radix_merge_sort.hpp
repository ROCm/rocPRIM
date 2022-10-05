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
#include "../specialization/device_radix_block_sort.hpp"

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
    unsigned int                                                    bit,
    unsigned int                                                    end_bit,
    hipStream_t                                                     stream,
    bool                                                            debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    const unsigned int current_radix_bits = end_bit - bit;
    if(current_radix_bits == sizeof(key_type) * 8)
    {
        return merge_sort_impl<Config>(temporary_storage,
                                       storage_size,
                                       keys_input,
                                       keys_output,
                                       values_input,
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
        return merge_sort_impl<Config>(
            temporary_storage,
            storage_size,
            keys_input,
            keys_output,
            values_input,
            values_output,
            size,
            radix_merge_compare<Descending, true, key_type>(bit, current_radix_bits),
            stream,
            debug_synchronous,
            keys_buffer,
            values_buffer);
    }
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
