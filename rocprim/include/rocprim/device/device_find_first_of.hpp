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

#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "config_types.hpp"
#include "device_find_first_of_config.hpp"
#include "device_transform.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    do                                                                                           \
    {                                                                                            \
        hipError_t _error = hipGetLastError();                                                   \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            hipError_t __error = hipStreamSynchronize(stream);                                   \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::steady_clock::now();                                        \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }                                                                                            \
    while(0)

#define RETURN_ON_ERROR(...)              \
    do                                    \
    {                                     \
        hipError_t error = (__VA_ARGS__); \
        if(error != hipSuccess)           \
        {                                 \
            return error;                 \
        }                                 \
    }                                     \
    while(0)

template<class Config, class InputIterator1, class InputIterator2, class BinaryFunction>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void find_first_of_kernel(InputIterator1 input,
                          InputIterator2 keys,
                          size_t*        output,
                          size_t         size,
                          size_t         keys_size,
                          BinaryFunction compare_function)
{
    constexpr find_first_of_config_params params = device_params<Config>();

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;
    constexpr unsigned int identity         = std::numeric_limits<unsigned int>::max();

    using type     = typename std::iterator_traits<InputIterator1>::value_type;
    using key_type = typename std::iterator_traits<InputIterator2>::value_type;

    const unsigned int thread_id        = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id         = ::rocprim::detail::block_id<0>();
    const unsigned int number_of_blocks = ::rocprim::detail::grid_size<0>();

    ROCPRIM_SHARED_MEMORY struct
    {
        unsigned int block_first_index;
        size_t       grid_first_index;
    } storage;

    if(thread_id == 0)
    {
        storage.block_first_index = identity;
    }
    syncthreads();

    size_t block_offset = block_id * items_per_block;
    for(; block_offset < size; block_offset += number_of_blocks * items_per_block)
    {
        if(thread_id == 0)
        {
            storage.grid_first_index = atomic_load(output);
        }
        syncthreads();
        if(storage.grid_first_index < block_offset)
        {
            // No need to continue if one of previous blocks (or this one) has found a match
            break;
        }

        type items[items_per_thread];

        unsigned int thread_first_index = identity;

        if(block_offset + items_per_block <= size)
        {
            block_load_direct_striped<block_size>(thread_id, input + block_offset, items);
            for(size_t key_index = 0; key_index < keys_size; ++key_index)
            {
                const key_type key = keys[key_index];
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i)
                {
                    if(compare_function(key, items[i]))
                    {
                        thread_first_index = min(thread_first_index, i);
                    }
                }
            }
        }
        else
        {
            const unsigned int valid = size - block_offset;
            block_load_direct_striped<block_size>(thread_id, input + block_offset, items, valid);
            for(size_t key_index = 0; key_index < keys_size; ++key_index)
            {
                const key_type key = keys[key_index];
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i)
                {
                    if(i * block_size + thread_id < valid && compare_function(key, items[i]))
                    {
                        thread_first_index = min(thread_first_index, i);
                    }
                }
            }
        }

        if(thread_first_index != identity)
        {
            // This happens to some blocks rarely so it is not beneficial to avoid atomic conflicts
            // with block_reduce which needs to be computed even if no threads have a match.
            atomic_min(&storage.block_first_index, thread_first_index * block_size + thread_id);
        }
        syncthreads();
        if(thread_id == 0 && storage.block_first_index != identity)
        {
            atomic_min(output, block_offset + storage.block_first_index);
        }
    }
}

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
    using type   = typename std::iterator_traits<InputIterator1>::value_type;
    using config = wrapped_find_first_of_config<Config, type>;

    target_arch target_arch;
    hipError_t  result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const find_first_of_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    if(temporary_storage == nullptr)
    {
        storage_size = sizeof(size_t);
        return hipSuccess;
    }

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    RETURN_ON_ERROR(
        hipMemcpyAsync(tmp_output, &size, sizeof(*tmp_output), hipMemcpyHostToDevice, stream));

    if(size > 0)
    {
        std::chrono::steady_clock::time_point start;
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }

        const size_t shared_memory_size = 0;
        auto kernel = find_first_of_kernel<config, InputIterator1, InputIterator2, BinaryFunction>;

        // Choose minimum grid size needed to achieve the highest occupancy
        int min_grid_size, max_block_size;
        RETURN_ON_ERROR(hipOccupancyMaxPotentialBlockSize(&min_grid_size,
                                                          &max_block_size,
                                                          kernel,
                                                          shared_memory_size,
                                                          int(block_size)));
        const size_t num_blocks
            = std::min(size_t(min_grid_size), ceiling_div(size, items_per_block));

        kernel<<<num_blocks, block_size, shared_memory_size, stream>>>(input,
                                                                       keys,
                                                                       tmp_output,
                                                                       size,
                                                                       keys_size,
                                                                       compare_function);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("find_first_of_kernel", size, start);
    }

    RETURN_ON_ERROR(
        transform(tmp_output, output, 1, ::rocprim::identity<void>(), stream, debug_synchronous));

    return hipSuccess;
}

} // namespace detail

/// \addtogroup devicemodule
/// @{

/// \brief Searches the range [input, input + size) for any of the elements in the range
///   [keys, keys + keys_size).
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size` if `temporary_storage` is
//    a null pointer.
/// * Accepts custom compare_function.
///
/// \tparam Config [optional] configuration of the primitive. It has to be `find_first_of_config`.
/// \tparam InputIterator1 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator [inferred] random-access iterator type of the output range. Must meet
///   the requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts two arguments of the
///   type `InputIterator1` and returns a value convertible to bool. Default type is
///   `::rocprim::equal_to<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the search.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] input iterator to the range of elements to examine.
/// \param [in] keys iterator to the range of elements to search for.
/// \param [out] output iterator to the output range. `output` should be able to be written for 1
///   element. `*output` constains the position of the first element in the range
///   [input, input + size) that is equal to an element from the range [keys, keys + keys_size).
//    If no such element is found, `*output` contains `size`.
/// \param [in] size number of elements to examine.
/// \param [in] keys_size number of elements to search for.
/// \param [in] compare_function binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ named requirement Compare.
///   The default value is `BinaryFunction()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after successful search; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level find_first_of is performed where inputs and keys are
///   represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                // e.g., 8
/// size_t keys_size;           // e.g., 2
/// unsigned int* input;        // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
/// unsigned int* keys;         // e.g., [ 10, 5 ]
/// unsigned int* keys_output;  // 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::find_first_of(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, keys, output, size, keys_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform find_first_of
/// rocprim::find_first_of(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, keys, output, size, keys_size
/// );
/// // possible output: [ 2 ]
/// \endcode
/// \endparblock
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
    return detail::find_first_of_impl<Config>(temporary_storage,
                                              storage_size,
                                              input,
                                              keys,
                                              output,
                                              size,
                                              keys_size,
                                              compare_function,
                                              stream,
                                              debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_FIND_FIRST_OF_HPP_
