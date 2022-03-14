// Copyright (c) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of the warp sort part of the device segmented radix sort operation.
/// Short enough segments are processed on warp level.
///
/// \tparam LogicalWarpSize - number of threads in the logical warp.
/// \tparam ItemsPerThread - number of items processed by a thread.
template<
    unsigned int LogicalWarpSize,
    unsigned int ItemsPerThread
>
struct WarpSortConfig
{
    /// \brief The number of threads in the logical warp.
    static constexpr unsigned int logical_warp_size = LogicalWarpSize;
    /// \brief The number of items processed by a thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

/// \brief Indicates if the warp level sorting is disabled in the
/// device segmented radix sort configuration.
struct DisabledWarpSortConfig
{
};

/// \brief Selects the appropriate \p WarpSortConfig based on the size of the key type.
///
/// \tparam Key - the type of the sorted keys.
template<class Key>
using select_warp_sort_config_t =
    std::conditional_t<
        sizeof(Key) < 4,
        DisabledWarpSortConfig,
        WarpSortConfig<32, 4>
    >;

/// \brief Configuration of device-level segmented radix sort operation.
///
/// Radix sort is excecuted in a few iterations (passes) depending on total number of bits to be sorted
/// (\p begin_bit and \p end_bit), each iteration sorts either \p LongRadixBits or \p ShortRadixBits bits
/// choosen to cover whole bit range in optimal way.
///
/// For example, if \p LongRadixBits is 7, \p ShortRadixBits is 6, \p begin_bit is 0 and \p end_bit is 32
/// there will be 5 iterations: 7 + 7 + 6 + 6 + 6 = 32 bits.
///
/// If a segment's element count is low ( <= warp_sort_config::items_per_thread * warp_sort_config::logical_warp_size ),
/// it is sorted by a special warp-level sorting method.
///
/// \tparam LongRadixBits - number of bits in long iterations.
/// \tparam ShortRadixBits - number of bits in short iterations, must be equal to or less than \p LongRadixBits.
/// \tparam SortConfig - configuration of radix sort kernel. Must be \p kernel_config.
/// \tparam WarpSortConfig - configuration of the warp sort that is used on the short segments.
template<
    unsigned int LongRadixBits,
    unsigned int ShortRadixBits,
    class SortConfig,
    class WarpSortConfig
>
struct segmented_radix_sort_config
{
    /// \brief Number of bits in long iterations.
    static constexpr unsigned int long_radix_bits = LongRadixBits;
    /// \brief Number of bits in short iterations
    static constexpr unsigned int short_radix_bits = ShortRadixBits;
    /// \brief Configuration of radix sort kernel.
    using sort = SortConfig;
    /// \brief Configuration of the warp sort method.
    using warp_sort_config = WarpSortConfig;
};

namespace detail
{

template<class Key, class Value>
struct segmented_radix_sort_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 13>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 9>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 7>, select_warp_sort_config_t<Key> > >
    > { };

template<class Key, class Value>
struct segmented_radix_sort_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4, 4, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_900<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<4, 3, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 17>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> > >
    > { };

template<class Key, class Value>
struct segmented_radix_sort_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4, 4, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_90a<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<4, 3, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 17>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> > >
    > { };

template<class Key, class Value>
struct segmented_radix_sort_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4, 4, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<4, 3, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 17>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> > >
    > { };

template<unsigned int TargetArch, class Key, class Value>
struct default_segmented_radix_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, detail::segmented_radix_sort_config_803<Key, Value> >,
        select_arch_case<900, detail::segmented_radix_sort_config_900<Key, Value> >,
        select_arch_case<ROCPRIM_ARCH_90a, detail::segmented_radix_sort_config_90a<Key, Value> >,
        select_arch_case<1030, detail::segmented_radix_sort_config_900<Key, Value> >,
        detail::segmented_radix_sort_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
