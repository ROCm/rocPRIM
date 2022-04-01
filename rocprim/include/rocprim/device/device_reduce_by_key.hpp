// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_

#include "config_types.hpp"
#include "device_reduce_by_key_config.hpp"
#include "device_transform.hpp"

#include "detail/device_reduce_by_key.hpp"
#include "detail/device_scan_common.hpp"
#include "detail/lookback_scan_state.hpp"

#include "../config.hpp"
#include "../detail/match_result_type.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../iterator/constant_iterator.hpp"

#include <chrono>
#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

namespace reduce_by_key
{

template<typename Config,
         typename KeyIterator,
         typename ValueIterator,
         typename UniqueIterator,
         typename ReductionIterator,
         typename UniqueCountIterator,
         typename CompareFunction,
         typename BinaryOp,
         typename LookBackScanState>
ROCPRIM_KERNEL __launch_bounds__(Config::block_size) void kernel(
    const KeyIterator                    keys_input,
    const ValueIterator                  values_input,
    const UniqueIterator                 unique_keys,
    const ReductionIterator              reductions,
    const UniqueCountIterator            unique_count,
    const BinaryOp                       reduce_op,
    const CompareFunction                compare,
    const LookBackScanState              scan_state,
    const ordered_block_id<unsigned int> ordered_bid,
    const std::size_t                    starting_block,
    const std::size_t                    number_of_blocks,
    const std::size_t                    size)
{
    reduce_by_key::kernel_impl<Config>(keys_input,
                                       values_input,
                                       unique_keys,
                                       reductions,
                                       unique_count,
                                       reduce_op,
                                       compare,
                                       scan_state,
                                       ordered_bid,
                                       starting_block,
                                       number_of_blocks,
                                       size);
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    do                                                                                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }                                                                                            \
    while(false)

template<class Config,
         class KeysInputIterator,
         class ValuesInputIterator,
         class UniqueOutputIterator,
         class AggregatesOutputIterator,
         class UniqueCountOutputIterator,
         class BinaryFunction,
         class KeyCompareFunction>
hipError_t reduce_by_key_impl(void*                     temporary_storage,
                              size_t&                   storage_size,
                              KeysInputIterator         keys_input,
                              ValuesInputIterator       values_input,
                              const unsigned int        size,
                              UniqueOutputIterator      unique_output,
                              AggregatesOutputIterator  aggregates_output,
                              UniqueCountOutputIterator unique_count_output,
                              BinaryFunction            reduce_op,
                              KeyCompareFunction        key_compare_op,
                              const hipStream_t         stream,
                              const bool                debug_synchronous)
{
    using key_type         = typename std::iterator_traits<KeysInputIterator>::value_type;
    using accumulator_type = typename ::rocprim::detail::match_result_type<
        typename std::iterator_traits<ValuesInputIterator>::value_type,
        BinaryFunction>::type;
    using wrapped_type = rocprim::tuple<accumulator_type, unsigned int>;

    using config = default_or_custom_config<
        Config,
        default_reduce_by_key_config<ROCPRIM_TARGET_ARCH, key_type, accumulator_type>>;

    constexpr unsigned int block_size      = config::block_size;
    constexpr unsigned int items_per_block = block_size * config::items_per_thread;

    using scan_state_type = detail::lookback_scan_state<wrapped_type, false>;
    //using scan_state_with_sleep_type = detail::lookback_scan_state<wrapped_type, true>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    const std::size_t number_of_blocks = detail::ceiling_div(size, items_per_block);

    // Calculate required temporary storage
    // This is valid even with scan_state_with_sleep_type
    const std::size_t scan_state_bytes
        = detail::align_size(scan_state_type::get_storage_size(number_of_blocks));
    if(temporary_storage == nullptr)
    {
        const std::size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
        // storage_size is never zero
        storage_size = scan_state_bytes + ordered_block_id_bytes;

        return hipSuccess;
    }

    auto scan_state = scan_state_type::create(temporary_storage, number_of_blocks);
    // TODO: FIX for gfx908 with asicRevision < 2 (scan_state with sleep)
    //auto scan_state_with_sleep
    //    = scan_state_with_sleep_type::create(temporary_storage, number_of_blocks);
    auto* temp_storage_ptr = static_cast<char*>(temporary_storage);
    auto  ordered_bid      = ordered_block_id_type::create(
        reinterpret_cast<ordered_block_id_type::id_type*>(temp_storage_ptr + scan_state_bytes));

    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
    {
        std::cout << "size:             " << size << '\n';
        std::cout << "block_size:       " << block_size << '\n';
        std::cout << "number of blocks: " << number_of_blocks << '\n';
        std::cout << "items_per_block:  " << items_per_block << '\n';
        start = std::chrono::high_resolution_clock::now();
    }

    if(size == 0)
    {
        // Fill out unique_count_output with zero
        return rocprim::transform(rocprim::constant_iterator<std::size_t>(0),
                                  unique_count_output,
                                  1,
                                  rocprim::identity<std::size_t>{},
                                  stream,
                                  debug_synchronous);
    }

    // TODO: Large indices support
    const unsigned int init_grid_size = detail::ceiling_div(number_of_blocks, block_size);
    hipLaunchKernelGGL(init_lookback_scan_state_kernel,
                       dim3(init_grid_size),
                       dim3(block_size),
                       0,
                       stream,
                       scan_state,
                       number_of_blocks,
                       ordered_bid);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel",
                                                number_of_blocks,
                                                start);

    hipLaunchKernelGGL(reduce_by_key::kernel<config>,
                       dim3(number_of_blocks),
                       dim3(block_size),
                       0,
                       stream,
                       keys_input,
                       values_input,
                       unique_output,
                       aggregates_output,
                       unique_count_output,
                       reduce_op,
                       key_compare_op,
                       scan_state,
                       ordered_bid,
                       0,
                       number_of_blocks,
                       size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key_kernel", size, start);

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // namespace reduce_by_key

} // namespace detail

/// \brief Parallel reduce-by-key primitive for device level.
///
/// reduce_by_key function performs a device-wide reduction operation of groups
/// of consecutive values having the same key using binary \p reduce_op operator. The first key of each group
/// is copied to \p unique_output and reduction of the group is written to \p aggregates_output.
/// The total number of group is written to \p unique_count_output.
///
/// \par Overview
/// * Supports non-commutative reduction operators. However, a reduction operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input and \p values_input must have at least \p size elements.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * Ranges specified by \p unique_output and \p aggregates_output must have at least
/// <tt>*unique_count_output</tt> (i.e. the number of unique keys) elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p reduce_by_key_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam UniqueOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam AggregatesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam UniqueCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p ValuesInputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to reduce.
/// \param [in] size - number of element in the input range.
/// \param [out] unique_output - iterator to the first element in the output range of unique keys.
/// \param [out] aggregates_output - iterator to the first element in the output range of reductions.
/// \param [out] unique_count_output - iterator to total number of groups.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level sum operation is performed on an array of
/// integer values and integer keys.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * keys_input;           // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * values_input;         // e.g., [1, 2, 3, 4,  5,  6,  7,  8]
/// int * unique_output;        // empty array of at least 4 elements
/// int * aggregates_output;    // empty array of at least 4 elements
/// int * unique_count_output;  // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform reduction
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
/// // unique_output:       [1, 2, 10, 88]
/// // aggregates_output:   [6, 4, 18,  8]
/// // unique_count_output: [4]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t reduce_by_key(void * temporary_storage,
                         size_t& storage_size,
                         KeysInputIterator keys_input,
                         ValuesInputIterator values_input,
                         unsigned int size,
                         UniqueOutputIterator unique_output,
                         AggregatesOutputIterator aggregates_output,
                         UniqueCountOutputIterator unique_count_output,
                         BinaryFunction reduce_op = BinaryFunction(),
                         KeyCompareFunction key_compare_op = KeyCompareFunction(),
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    return detail::reduce_by_key::reduce_by_key_impl<Config>(temporary_storage,
                                                             storage_size,
                                                             keys_input,
                                                             values_input,
                                                             size,
                                                             unique_output,
                                                             aggregates_output,
                                                             unique_count_output,
                                                             reduce_op,
                                                             key_compare_op,
                                                             stream,
                                                             debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
