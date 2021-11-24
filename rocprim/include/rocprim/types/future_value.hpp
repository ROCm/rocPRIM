// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TYPES_FUTURE_VALUE_HPP_
#define ROCPRIM_TYPES_FUTURE_VALUE_HPP_

#include "../config.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/**
 * \brief Allows passing values that are not yet known at launch time as paramters to device algorithms.
 *
 * \note It is the users responsibility to ensure that value is available when the algorithm executes.
 * This can be guaranteed with stream dependencies or explicit external synchronization.
 *
 * \code
 * int* intermediate_result = nullptr;
 * hipMalloc(reinterpret_cast<void**>(&intermediate_result), sizeof(intermediate_result));
 * hipLaunchKernelGGL(compute_intermediate, blocks, threads, 0, stream, arg1, arg2, itermediate_result);
 * const auto initial_value = rocprim::future_value<int>{intermediate_result};
 * rocprim::exclusive_scan(temporary_storage,
 *                         storage_size,
 *                         input,
 *                         output,
 *                         initial_value,
 *                         size);
 * hipFree(intermediate_result)
 * \endcode
 *
 * \tparam T
 * \tparam Iter
 */
template <typename T, typename Iter = T*>
class future_value
{
public:
    using value_type    = T;
    using iterator_type = Iter;

    explicit ROCPRIM_HOST_DEVICE future_value(const Iter iter)
        : iter_ {iter}
    {
    }

    ROCPRIM_HOST_DEVICE operator T()
    {
        return *iter_;
    }

    ROCPRIM_HOST_DEVICE operator T() const
    {
        return *iter_;
    }
private:
    Iter iter_;
};

namespace detail
{
    /**
     * \brief Allows a single kernel to support both future and immediate values.
     */
    template <typename T, typename Iter = T*>
    class input_value
    {
    public:
        using value_type    = T;
        using iterator_type = Iter;

        ROCPRIM_HOST_DEVICE operator T()
        {
            return is_future_ ? future_value_ : immediate_value_;
        }

        ROCPRIM_HOST_DEVICE operator T() const
        {
            return is_future_ ? future_value_ : immediate_value_;
        }

        ROCPRIM_HOST_DEVICE T get()
        {
            return operator T();
        }

        ROCPRIM_HOST_DEVICE T get() const
        {
            return operator T();
        }

        explicit ROCPRIM_HOST_DEVICE input_value(const T immediate)
            : immediate_value_ {immediate}
            , is_future_ {false}
        {
        }

        explicit ROCPRIM_HOST_DEVICE input_value(const future_value<T, Iter> future)
            : future_value_ {future}
            , is_future_ {true}
        {
        }

        ROCPRIM_HOST_DEVICE input_value(const input_value& rhs)
            : is_future_ {rhs.is_future_}
        {
            if(is_future_)
            {
                future_value_ = rhs.future_value_;
            }
            else
            {
                immediate_value_ = rhs.immediate_value_;
            }
        }

        ROCPRIM_HOST_DEVICE input_value& operator=(const input_value& rhs)
        {
            is_future_ = rhs.is_future_;
            if(is_future_)
            {
                future_value_ = rhs.future_value_;
            }
            else
            {
                immediate_value_ = rhs.immediate_value_;
            }
        }

        ROCPRIM_HOST_DEVICE ~input_value()
        {
            if(is_future_) {
                future_value_.~future_value<T, Iter>();
            } else {
                immediate_value_.~T();
            }
        }

    private:
        union
        {
            future_value<T, Iter> future_value_;
            T                     immediate_value_;
        };
        bool is_future_;
    };

    template <typename T>
    struct future_value_traits
    {
        using value_type    = T;
        using iterator_type = T*;
    };

    template <typename T, typename Iter>
    struct future_value_traits<::rocprim::future_value<T, Iter>>
    {
        using value_type    = T;
        using iterator_type = Iter;
    };

    template <typename T>
    using future_type_t = typename future_value_traits<T>::value_type;

    template <typename T>
    using future_iterator_t = typename future_value_traits<T>::iterator_type;

    template <typename T, typename Iter>
    input_value<T, Iter> make_input_value(const ::rocprim::future_value<T, Iter> future) {
        return input_value<T, Iter>{future};
    }

    template <typename T>
    input_value<T, T*> make_input_value(const T immediate_value) {
        return input_value<T, T*>(immediate_value);
    }
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif
