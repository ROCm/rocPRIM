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

#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_load.hpp"
#include "../block/block_scan.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level select operation.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam KeyBlockLoadMethod - method for loading input keys.
/// \tparam ValueBlockLoadMethod - method for loading input values.
/// \tparam FlagBlockLoadMethod - method for loading flag values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single select kernel launch.
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    ::rocprim::block_load_method KeyBlockLoadMethod,
    ::rocprim::block_load_method ValueBlockLoadMethod,
    ::rocprim::block_load_method FlagBlockLoadMethod,
    ::rocprim::block_scan_algorithm BlockScanMethod,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct select_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Method for loading input keys.
    static constexpr block_load_method key_block_load_method = KeyBlockLoadMethod;
    /// \brief Method for loading input values.
    static constexpr block_load_method value_block_load_method = ValueBlockLoadMethod;
    /// \brief Method for loading flag values.
    static constexpr block_load_method flag_block_load_method = FlagBlockLoadMethod;
    /// \brief Algorithm for block scan.
    static constexpr block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single select kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<class Key>
struct select_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 13u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Key>
struct select_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct select_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct select_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};


template<unsigned int TargetArch, class Key, class /*Value*/>
struct default_select_config
    : select_arch<
        TargetArch,
        select_arch_case<803, select_config_803<Key>>,
        select_arch_case<900, select_config_900<Key>>,
        select_arch_case<ROCPRIM_ARCH_90a, select_config_90a<Key>>,
        select_arch_case<1030, select_config_1030<Key>>,
        select_config_803<Key>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_SELECT_CONFIG_HPP_
