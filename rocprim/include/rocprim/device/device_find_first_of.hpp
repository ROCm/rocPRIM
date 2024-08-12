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

#ifndef ROCPRIM_DEVICE_DEVICE_FIND_FIRST_OF_HPP_
#define ROCPRIM_DEVICE_DEVICE_FIND_FIRST_OF_HPP_

#include "../detail/temp_storage.hpp"

#include "../config.hpp"

#include "../iterator/transform_iterator.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../iterator/counting_iterator.hpp"
#include "config_types.hpp"
#include "device_reduce.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction>
ROCPRIM_INLINE
hipError_t find_first_of_impl(void*          temporary_storage,
                              size_t&        storage_size,
                              InputIterator1 input,
                              InputIterator2 keys,
                              OutputIterator output,
                              size_t         size,
                              size_t         keys_size,
                              BinaryFunction compare_function,
                              hipStream_t    stream,
                              bool           debug_synchronous)
{
    return reduce<Config>(
        temporary_storage, storage_size,
        rocprim::make_transform_iterator(
            rocprim::make_zip_iterator(rocprim::make_tuple(rocprim::make_counting_iterator<size_t>(0), input)),
            [keys, keys_size, size, compare_function] ROCPRIM_DEVICE (const auto& index_value)
            {
                for(size_t i = 0; i < keys_size; ++i)
                {
                    if(compare_function(keys[i], get<1>(index_value))) 
                    {
                        return get<0>(index_value);
                    }
                }
                return size;
            }
        ),
        output,
        size,
        rocprim::minimum<size_t>(),
        stream,
        debug_synchronous
    );
}

} // namespace detail

/// \addtogroup devicemodule
/// @{

template<class Config = default_config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction
         = ::rocprim::equal_to<typename std::iterator_traits<InputIterator1>::value_type>>
ROCPRIM_INLINE
hipError_t find_first_of(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator1 input,
                         InputIterator2 keys,
                         OutputIterator output,
                         size_t         size,
                         size_t         keys_size,
                         BinaryFunction compare_function  = BinaryFunction(),
                         hipStream_t    stream            = 0,
                         bool           debug_synchronous = false)
{
    return detail::find_first_of_impl<Config>(
        temporary_storage, storage_size,
        input, keys, output, size, keys_size, compare_function,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_FIND_FIRST_OF_HPP_
