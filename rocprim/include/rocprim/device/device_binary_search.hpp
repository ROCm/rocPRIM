// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_binary_search.hpp"
#include "device_binary_search_config.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    class Config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class SearchFunction,
    class CompareFunction
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         SearchFunction search_op,
                         CompareFunction compare_op,
                         hipStream_t stream,
                         bool debug_synchronous)
{
    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return hipSuccess;
    }

    return transform<Config>(
        needles, output,
        needles_size,
        [haystack, haystack_size, search_op, compare_op]
        ROCPRIM_DEVICE
        (const value_type& value)
        {
            return search_op(haystack, haystack_size, value, compare_op);
        },
        stream, debug_synchronous
    );
}

template<class Config, class Tag>
struct is_default_or_has_tag
{
    static constexpr bool value
        = std::integral_constant<bool, std::is_same<typename Config::tag, Tag>::value>::value;
};

template<class Tag>
struct is_default_or_has_tag<default_config, Tag>
{
    static constexpr bool value = true;
};

} // end of detail namespace

/// @brief Parallel primitive that uses binary search for computing a lower bound on a given ordered
/// range for each element of a given input.
///
/// The \p lower_bound function determines for each element \p e of a given input the greatest index
/// \p i in a given ordered range \p haystack such that <tt>!compare_op(e, haystack[i])</tt> is
/// \p true.
/// It uses the search function \p detail::lower_bound_search_op, which in turn uses a binary
/// operator \p compare_op for comparing the given value with the haystack ones.
///
/// \par Overview
/// * When a null pointer is passed as \p temporary_storage, the required allocation size (in bytes)
/// is written to \p storage_size and the function returns without performing the search operation.
/// * If used along with \p rocprim::upper_bound, the ith element of the given input must be located
/// in the semi-open interval <tt>[lower_output[i], upper_output[i])</tt> of \p haystack, in case of
/// being present at all.
///
/// @tparam Config - [optional] Configuration of the primitive. It has to be \p lower_bound_config or
/// a class derived from it. Default is \p default_config.
/// @tparam HaystackIterator - [inferred] Random-access iterator type of the search range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// @tparam NeedlesIterator - [inferred] Random-access iterator type of the input range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type. Elements of
/// the type pointed by it must be comparable to elements of the type pointed by HaystackIterator
/// as either operand of \p compare_op.
/// @tparam OutputIterator - [inferred] Random-access iterator type of the output range. Must meet
/// the requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// @tparam CompareFunction - [inferred] Type of binary function that accepts two arguments of the
/// types pointed by \p HaystackIterator and \p NeedlesIterator, and returns a value convertible
/// to bool. Default type is \p ::rocprim::less<>.
/// @param [in] temporary_storage - Pointer to a device-accessible temporary storage.
/// @param [in,out] storage_size - Reference to the size (in bytes) of \p temporary_storage.
/// @param [in] haystack - Iterator to the first element in the search range. Elements of this
/// range must be sorted.
/// @param [in] needles - Iterator to the first element in the range of values to search for on
/// \p haystack.
/// @param [out] output - Iterator to the first element in the output range.
/// @param [in] haystack_size - Number of elements in the search range \p haystack.
/// @param [in] needles_size - Number of elements in the input range \p needles.
/// @param [in] compare_op - Binary operation function object that is used to compare values. The
/// signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const U &b);</tt>. It does not need to have <tt>const &</tt>, but the
/// function object must not modify the objects passed to it. Default is \p CompareFunction().
/// @param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// @param [in] debug_synchronous - [optional] If true, synchronization after every kernel launch is
/// forced in order to check for errors.
/// @return \p hipSuccess (\p 0) after a successful search; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level lower bound computation on a haystack of double precision type
/// values is performed on an input array of integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.).
/// size_t          haystack_size;    // e.g. 7
/// double *        haystack;         // e.g. {0, 1.5, 3, 4.5, 6, 7.5, 9}
/// size_t          needles_size;     // e.g. 5
/// int *           needles;          // e.g. {1, 2, 3, 4, 5}
/// compare_op_type compare_op;       // e.g. compare_op_type = rocprim::less<>
/// size_t *        output;           // empty array of \p needles_size elements
///
/// // Get required size of the temporary storage.
/// void * temporary_storage = nullptr;
/// size_t temporary_storage_bytes;
/// rocprim::lower_bound<config>(temporary_storage,
///                              temporary_storage_bytes,
///                              haystack,
///                              needles,
///                              output,
///                              haystack_size,
///                              needles_size,
///                              compare_op,
///                              stream,
///                              debug_synchronous);
///
/// // Allocate temporary storage.
/// hipMalloc(&temporary_storage, temporary_storage_bytes);
///
/// // Perform binary search.
/// rocprim::lower_bound<config>(temporary_storage,
///                              temporary_storage_bytes,
///                              haystack,
///                              needles,
///                              output,
///                              haystack_size,
///                              needles_size,
///                              compare_op,
///                              stream,
///                              debug_synchronous);
///
/// // \p output = {0, 1, 2, 2, 3}
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t lower_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    static_assert(detail::is_default_or_has_tag<Config, detail::lower_bound_config_tag>::value,
                  "Config must be a specialization of struct template lower_bound_config");

    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_lower_bound<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::lower_bound_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// @brief Parallel primitive that uses binary search for computing an upper bound on a given ordered
/// range for each element of a given input.
///
/// The \p upper_bound function determines for each element \p e of a given input the lowest index
/// \p i in a given ordered range \p haystack such that <tt>compare_op(e, haystack[i])</tt> is
/// \p true.
/// It uses the search function \p detail::upper_bound_search_op, which in turn uses a binary
/// operator \p compare_op for comparing the input values with the haystack ones.
///
/// \par Overview
/// * When a null pointer is passed as \p temporary_storage, the required allocation size (in bytes)
/// is written to \p storage_size and the function returns without performing the search operation.
/// * If used along with \p rocprim::lower_bound, the ith element of the given input must be located
/// in the semi-open interval <tt>[lower_output[i], upper_output[i])</tt> of \p haystack, in case of
/// being present at all.
///
/// @tparam Config - [optional] Configuration of the primitive. It can be \p upper_bound_config or
/// a class derived from it. Default is \p default_config.
/// @tparam HaystackIterator - [inferred] Random-access iterator type of the search range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// @tparam NeedlesIterator - [inferred] Random-access iterator type of the input range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type. Elements of
/// the type pointed by it must be comparable to elements of the type pointed by HaystackIterator
/// as either operand of \p compare_op.
/// @tparam OutputIterator - [inferred] Random-access iterator type of the output range. Must meet
/// the requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// @tparam CompareFunction - [inferred] Type of binary function that accepts two arguments of the
/// types pointed by \p HaystackIterator and \p NeedlesIterator, and returns a value convertible
/// to bool. Default type is \p ::rocprim::less<>.
/// @param [in] temporary_storage - Pointer to a device-accessible temporary storage.
/// @param [in,out] storage_size - Reference to the size (in bytes) of \p temporary_storage.
/// @param [in] haystack - Iterator to the first element in the search range. Elements of this
/// range must be sorted.
/// @param [in] needles - Iterator to the first element in the range of values to search for on
/// \p haystack.
/// @param [out] output - Iterator to the first element in the output range.
/// @param [in] haystack_size - Number of elements in the search range \p haystack.
/// @param [in] needles_size - Number of elements in the input range \p needles.
/// @param [in] compare_op - Binary operation function object that is used to compare values. The
/// signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const U &b);</tt>. It does not need to have <tt>const &</tt>, but the
/// function object must not modify the objects passed to it. Default is \p CompareFunction().
/// @param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// @param [in] debug_synchronous - [optional] If true, synchronization after every kernel launch is
/// forced in order to check for errors.
/// @return \p hipSuccess (\p 0) after a successful search; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level upper bound computation on a haystack of double precision type
/// values is performed on an input array of integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.).
/// size_t          haystack_size;    // e.g. 7
/// double *        haystack;         // e.g. {0, 1.5, 3, 4.5, 6, 7.5, 9}
/// size_t          needles_size;     // e.g. 5
/// int *           needles;          // e.g. {1, 2, 3, 4, 5}
/// compare_op_type compare_op;       // e.g. compare_op_type = rocprim::less<>
/// size_t *        output;           // empty array of \p needles_size elements
///
/// // Get required size of the temporary storage.
/// void * temporary_storage = nullptr;
/// size_t temporary_storage_bytes;
/// rocprim::upper_bound<config>(temporary_storage,
///                              temporary_storage_bytes,
///                              haystack,
///                              needles,
///                              output,
///                              haystack_size,
///                              needles_size,
///                              compare_op,
///                              stream,
///                              debug_synchronous);
///
/// // Allocate temporary storage.
/// hipMalloc(&temporary_storage, temporary_storage_bytes);
///
/// // Perform binary search.
/// rocprim::upper_bound<config>(temporary_storage,
///                              temporary_storage_bytes,
///                              haystack,
///                              needles,
///                              output,
///                              haystack_size,
///                              needles_size,
///                              compare_op,
///                              stream,
///                              debug_synchronous);
///
/// // \p output = {1, 2, 3, 3, 4}
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t upper_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    static_assert(detail::is_default_or_has_tag<Config, detail::upper_bound_config_tag>::value,
                  "Config must be a specialization of struct template upper_bound_config");
    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_upper_bound<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::upper_bound_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// @brief Parallel primitive for performing a binary search (on a sorted range) of a given input.
///
/// The \p binary_search function determines for each element of a given input if it's present
/// in a given ordered range \p haystack. It uses the search function \p detail::binary_search_op
/// which in turn uses a binary operator \p compare_op for comparing the input values with the
/// haystack ones.
///
/// \par Overview
/// * When a null pointer is passed as \p temporary_storage, the required allocation size (in bytes)
/// is written to \p storage_size and the function returns without performing the search operation.
///
/// @tparam Config - [optional] Configuration of the primitive. It can be \p binary_search_config or
/// a class derived from it. Default is \p default_config.
/// @tparam HaystackIterator - [inferred] Random-access iterator type of the search range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// @tparam NeedlesIterator - [inferred] Random-access iterator type of the input range. Must meet
/// the requirements of a C++ InputIterator concept. It can be a simple pointer type. Elements of
/// the type pointed by it must be comparable to elements of the type pointed by \p HaystackIterator
/// as either operand of \p compare_op.
/// @tparam OutputIterator - [inferred] Random-access iterator type of the output range. Must meet
/// the requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// @tparam CompareFunction - [inferred] Type of binary function that accepts two arguments of the
/// types pointed by \p HaystackIterator and \p NeedlesIterator, and returns a value convertible to
/// bool. Default type is \p ::rocprim::less<>.
/// @param [in] temporary_storage - Pointer to a device-accessible temporary storage.
/// @param [in,out] storage_size - Reference to the size (in bytes) of \p temporary_storage.
/// @param [in] haystack - Iterator to the first element in the search range. Elements of this
/// range must be sorted.
/// @param [in] needles - Iterator to the first element in the range of values to search for on
/// \p haystack.
/// @param [out] output - Iterator to the first element in the output range of boolean values.
/// @param [in] haystack_size - Number of elements in the search range \p haystack.
/// @param [in] needles_size - Number of elements in the input range \p needles.
/// @param [in] compare_op - Binary operation function object that is used to compare values. The
/// signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const U &b);</tt>. It does not need to have <tt>const &</tt>, but the
/// function object must not modify the objects passed to it. Default is \p CompareFunction().
/// @param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// @param [in] debug_synchronous - [optional] If true, synchronization after every kernel launch is
/// forced in order to check for errors.
/// @return \p hipSuccess (\p 0) after a successful search; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level binary search on a haystack of integer values is performed on an
/// input array of integer values too.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.).
/// size_t          haystack_size;    // e.g. 10
/// int *           haystack;         // e.g. {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
/// size_t          needles_size;     // e.g. 8
/// int *           needles;          // e.g. {0, 2, 12, 4, 14, 6, 8, 10}
/// compare_op_type compare_op;       // e.g. compare_op_type = rocprim::less<int>
/// size_t *        output;           // empty array of \p needles_size elements
///
/// // Get required size of the temporary storage.
/// void * temporary_storage = nullptr;
/// size_t temporary_storage_bytes;
/// rocprim::binary_search<config>(temporary_storage,
///                                temporary_storage_bytes,
///                                haystack,
///                                needles,
///                                output,
///                                haystack_size,
///                                needles_size,
///                                compare_op,
///                                stream,
///                                debug_synchronous);
///
/// // Allocate temporary storage.
/// hipMalloc(&temporary_storage, temporary_storage_bytes);
///
/// // Perform binary search.
/// rocprim::binary_search<config>(temporary_storage,
///                                temporary_storage_bytes,
///                                haystack,
///                                needles,
///                                output,
///                                haystack_size,
///                                needles_size,
///                                compare_op,
///                                stream,
///                                debug_synchronous);
///
/// // \p output = {1, 1, 0, 1, 0, 1, 1, 0}
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         CompareFunction compare_op = CompareFunction(),
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    static_assert(detail::is_default_or_has_tag<Config, detail::binary_search_config_tag>::value,
                  "Config must be a specialization of struct template binary_search_config");
    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_binary_search<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::binary_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
