// Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../functional.hpp"

#include "../block/block_reduce.hpp"

#include "config_types.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE


namespace detail
{

template<class Value>
struct reduce_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

template<class Value>
struct reduce_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

// TODO: We need to update these parameters
template<class Value>
struct reduce_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

// TODO: We need to update these parameters
template<class Value>
struct reduce_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

template<unsigned int TargetArch, class Value>
struct default_reduce_config
    : select_arch<
        TargetArch,
        select_arch_case<803, reduce_config_803<Value>>,
        select_arch_case<900, reduce_config_900<Value>>,
        select_arch_case<ROCPRIM_ARCH_90a, reduce_config_90a<Value>>,
        select_arch_case<1030, reduce_config_1030<Value>>,
        reduce_config_900<Value>
    > { };

struct reduce_config_params
{
    unsigned int           block_size;
    unsigned int           items_per_thread;
    block_reduce_algorithm block_reduce_method;
    unsigned int           size_limit;
};

template<typename ReduceConfig>
constexpr reduce_config_params wrap_reduce_config()
{
    return reduce_config_params{ReduceConfig::block_size,
                                ReduceConfig::items_per_thread,
                                ReduceConfig::block_reduce_method,
                                ReduceConfig::size_limit};
}

template<typename ReduceConfig, typename>
struct wrapped_reduce_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_config_params params = wrap_reduce_config<ReduceConfig>();
    };
};

template<typename Value>
struct wrapped_reduce_config<default_config, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_config_params params
            = wrap_reduce_config<default_reduce_config<static_cast<unsigned int>(Arch), Value>>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename ReduceConfig, typename Value>
template<target_arch Arch>
constexpr reduce_config_params
    wrapped_reduce_config<ReduceConfig, Value>::architecture_config<Arch>::params;

template<typename Value>
template<target_arch Arch>
constexpr reduce_config_params
    wrapped_reduce_config<default_config, Value>::architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_
