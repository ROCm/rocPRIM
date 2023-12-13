// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_WEIRD_ITERATOR_HPP_
#define TEST_WEIRD_ITERATOR_HPP_

#include <functional>
#include <type_traits>

namespace test_utils
{

namespace detail
{
template<class T>
constexpr T& FUN(T& t) noexcept
{
    return t;
}
template<class T>
void FUN(T&&) = delete;

template<class T>
ROCPRIM_HOST_DEVICE
    typename std::enable_if<std::is_object<typename std::remove_reference<T>::type>::value,
                            T*>::type
    addressof(T& arg) noexcept
{
    return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char&>(arg)));
}
} // namespace detail

// assign-through reference_wrapper implementation
template<class T>
class reference_wrapper
{
public:
    // types
    using type = T;

    // construct/copy/destroy
    template<class U,
             class = decltype(detail::FUN<T>(std::declval<U>()),
                              std::enable_if_t<!std::is_same<
                                  reference_wrapper,
                                  std::remove_cv_t<std::remove_reference_t<U>>>::value>())>
    constexpr reference_wrapper(U&& u) noexcept(noexcept(detail::FUN<T>(std::forward<U>(u))))
        : _ptr(detail::addressof(detail::FUN<T>(std::forward<U>(u))))
    {}

    reference_wrapper(const reference_wrapper&) noexcept = default;

    // assignment
    reference_wrapper& operator=(const T& x) noexcept
    {
        *_ptr = x;
    }

    // access
    constexpr operator T&() const noexcept
    {
        return *_ptr;
    }
    constexpr T& get() const noexcept
    {
        return *_ptr;
    }

private:
    T* _ptr;
};

// Iterator used in tests to check situtations when value_type of the
// iterator is not the same as the return type of operator[] in the
// device_adjacent_difference API. This doesn't work for in_place version.
template<class T>
class weird_iterator
{
public:
    // Iterator traits
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using reference       = T&;

    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline weird_iterator(T* ptr) : ptr_(ptr) {}

    ROCPRIM_HOST_DEVICE inline ~weird_iterator() = default;

    ROCPRIM_HOST_DEVICE inline weird_iterator& operator++()
    {
        ptr_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator operator++(int)
    {
        weird_iterator old = *this;
        ptr_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator& operator--()
    {
        ptr_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator operator--(int)
    {
        weird_iterator old = *this;
        ptr_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline reference operator*() const
    {
        return *ptr_;
    }

    ROCPRIM_HOST_DEVICE inline reference_wrapper<T> operator[](difference_type n) const
    {
        return *(ptr_ + n);
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator operator+(difference_type distance) const
    {
        auto i = ptr_ + distance;
        return weird_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator& operator+=(difference_type distance)
    {
        ptr_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator operator-(difference_type distance) const
    {
        auto i = ptr_ - distance;
        return weird_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline weird_iterator& operator-=(difference_type distance)
    {
        ptr_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline difference_type operator-(weird_iterator other) const
    {
        return ptr_ - other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline bool operator==(weird_iterator other) const
    {
        return ptr_ == other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline bool operator!=(weird_iterator other) const
    {
        return ptr_ != other.ptr_;
    }

private:
    T* ptr_;
};

template<bool Wrap, class T>
inline auto wrap_in_weird_iterator(T* ptr) -> typename std::enable_if<Wrap, weird_iterator<T>>::type
{
    return weird_iterator<T>(ptr);
}

template<bool Wrap, class T>
inline auto wrap_in_weird_iterator(T* ptr) -> typename std::enable_if<!Wrap, T*>::type
{
    return ptr;
}

} // namespace test_utils

#endif // TEST_IDENTITY_ITERATOR_HPP_
