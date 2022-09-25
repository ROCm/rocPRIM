// Copyright (c) 2018-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_reduce.hpp"

#include "config_types.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE


namespace detail
{
// Default config for any GPU arch that doesnt have any specializations
template<unsigned int arch, class Value> struct default_reduce_config  : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };

/*********************************BEGIN gfx803 CONFIG**************************/

/*********************************END gfx803 CONFIG**************************/

/*********************************BEGIN gfx900 CONFIG**************************/

/*********************************END gfx900 CONFIG**************************/

/*********************************BEGIN gfx906 CONFIG**************************/
template<class Value> struct default_reduce_config<906, Value>  : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, double> : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, float> : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, int> : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, int64_t> : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, int8_t> : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
template<> struct default_reduce_config<906, rocprim::half> : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce> { };
/*********************************END gfx906 CONFIG**************************/

/*********************************BEGIN gfx908 CONFIG**************************/

/*********************************END gfx908 CONFIG**************************/

/*********************************BEGIN gfx90a CONFIG**************************/

/*********************************END gfx90a CONFIG**************************/

/*********************************BEGIN gfx1030 CONFIG**************************/

/*********************************END gfx1030 CONFIG**************************/


} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_
