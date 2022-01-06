// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_WARP_EXCHANGE_HPP_
#define ROCPRIM_WARP_WARP_EXCHANGE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p warp_exchange class is a warp level parallel primitive which provides
/// methods for rearranging items partitioned across threads in a warp.
///
/// \tparam T - the input type.
/// \tparam ItemsPerThread - the number of items contributed by each thread.
/// \tparam WarpSize - the number of threads in a warp. It must be a divisor of the
/// kernel block size.
///
/// \par Overview
/// * The \p warp_exchange class supports the following rearrangement methods:
///   * Transposing a blocked arrangement to a striped arrangement.
///   * Transposing a striped arrangement to a blocked arrangement.
/// * Data is automatically padded to ensure zero bank conflicts.
///
/// \par Examples
/// \parblock
/// In the example an exchange operation is performed on a warp of 8 threads, using type
/// \p int with 4 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     constexpr unsigned int threads_per_block = 128;
///     constexpr unsigned int threads_per_warp  =   8;
///     constexpr unsigned int items_per_thread  =   4;
///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
///     // allocate storage in shared memory
///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
///
///     int items[items_per_thread];
///     ...
///     warp_exchange_int w_exchange;
///     w_exchange.blocked_to_striped(items, items, storage[warp_id]);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size()
>
class warp_exchange
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

    // Minimize LDS bank conflicts for power-of-two strides, i.e. when items accessed
    // using `thread_id * ItemsPerThread` pattern where ItemsPerThread is power of two
    static constexpr bool has_bank_conflicts =
        ItemsPerThread >= 2 && ::rocprim::detail::is_power_of_two(ItemsPerThread);
    static constexpr unsigned int banks_no = ::rocprim::detail::get_lds_banks_no();
    static constexpr unsigned int bank_conflicts_padding =
        has_bank_conflicts ? (WarpSize * ItemsPerThread / banks_no) : 0;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        T buffer[WarpSize * ItemsPerThread + bank_conflicts_padding];
    };

public:

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by the related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.blocked_to_striped(items, items, storage[warp_id]);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(flat_id * ItemsPerThread + i)] = input[i];
        }
        ::rocprim::wave_barrier();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(i * WarpSize + flat_id)];
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, threads_per_warp, items_per_thread>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.striped_to_blocked(items, items, storage[warp_id]);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(i * WarpSize + flat_id)] = input[i];
        }
        ::rocprim::wave_barrier();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

private:
    // Change index to minimize LDS bank conflicts if necessary
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int index(unsigned int n)
    {
        // Move every 32-bank wide "row" (32 banks * 4 bytes) by one item
        return has_bank_conflicts ? (n + n / banks_no) : n;
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_EXCHANGE_HPP_
