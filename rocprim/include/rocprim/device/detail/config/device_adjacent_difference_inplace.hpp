// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_DIFFERENCE_INPLACE_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_DIFFERENCE_INPLACE_HPP_

#include "../../../type_traits.hpp"
#include "../device_config_helper.hpp"
#include <type_traits>

/* DO NOT EDIT THIS FILE
 * This file is automatically generated by `/scripts/autotune/create_optimization.py`.
 * so most likely you want to edit rocprim/device/device_(algo)_config.hpp
 */

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int arch, class value_type, class enable = void>
struct default_adjacent_difference_inplace_config
    : default_adjacent_difference_config_base<value_type>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<128, 16>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<256, 16>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<512, 16>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<128, 16>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<256, 16>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<256, 32>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<512, 32>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 4>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<1024, 8>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 4>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<1024, 8>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<32, 64>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<256, 16>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<128, 64>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<256, 64>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<256, 16>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<128, 64>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<256, 64>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx900),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<512, 16>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 8>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<256, 16>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<512, 16>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<256, 16>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx906),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<64, 16>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<512, 8>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<64, 32>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx908),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<64, 16>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<512, 8>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<64, 32>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::unknown),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<64, 16>
{};

// Based on value_type = double
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = float
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = rocprim::half
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2))>> : adjacent_difference_config<512, 8>
{};

// Based on value_type = int64_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 8) && (sizeof(value_type) > 4))>>
    : adjacent_difference_config<512, 2>
{};

// Based on value_type = int
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2))>>
    : adjacent_difference_config<1024, 4>
{};

// Based on value_type = short
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 2) && (sizeof(value_type) > 1))>>
    : adjacent_difference_config<64, 32>
{};

// Based on value_type = int8_t
template<class value_type>
struct default_adjacent_difference_inplace_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    value_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<value_type>::value)
                      && (sizeof(value_type) <= 1))>> : adjacent_difference_config<64, 16>
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_DIFFERENCE_INPLACE_HPP_