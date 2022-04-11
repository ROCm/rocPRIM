// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_UTILS_HPP_
#define ROCPRIM_BENCHMARK_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>

#ifdef WIN32
#include <numeric>
#endif

#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

// Support half operators on host side

ROCPRIM_HOST inline
rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half *>(&x);
}

ROCPRIM_HOST inline
rocprim::half native_to_half(const rocprim::native_half& x)
{
    return *reinterpret_cast<const rocprim::half *>(&x);
}

struct half_less
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return half_to_native(a) < half_to_native(b);
        #endif
    }
};

struct half_plus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_half(half_to_native(a) + half_to_native(b));
        #endif
    }
};

struct half_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return half_to_native(a) == half_to_native(b);
        #endif
    }
};

// std::uniform_int_distribution is undefined for anything other than:
// short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
template <typename T>
struct is_valid_for_int_distribution :
    std::integral_constant<bool,
        std::is_same<short, T>::value ||
        std::is_same<unsigned short, T>::value ||
        std::is_same<int, T>::value ||
        std::is_same<unsigned int, T>::value ||
        std::is_same<long, T>::value ||
        std::is_same<unsigned long, T>::value ||
        std::is_same<long long, T>::value ||
        std::is_same<unsigned long long, T>::value
    > {};

using engine_type = std::default_random_engine;

// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    engine_type gen{std::random_device{}()};
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value,
            int,
            unsigned int>::type
        >::type;
    std::uniform_int_distribution<dis_type> distribution((T)min, (T)max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    engine_type gen{std::random_device{}()};
    // Generate floats when T is half
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution((dis_type)min, (dis_type)max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, size_t max_random_size = 1024 * 1024)
{
    engine_type gen{std::random_device{}()};
    std::bernoulli_distribution distribution(p);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline T get_random_value(T min, T max)
{
    return get_random_data(1, min, max)[0];
}

template<class T, class U = T>
struct custom_type
{
    using first_type = T;
    using second_type = U;

    T x;
    U y;

    ROCPRIM_HOST_DEVICE inline
    custom_type(T xx = 0, U yy = 0) : x(xx), y(yy)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~custom_type() = default;

    ROCPRIM_HOST_DEVICE inline
    custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_type& rhs) const
    {
        return (x < rhs.x || (x == rhs.x && y < rhs.y));
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }
};

template<typename>
struct is_custom_type : std::false_type {};

template<class T, class U>
struct is_custom_type<custom_type<T,U>> : std::true_type {};

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<is_custom_type<T>::value, std::vector<T>>::type
{
    using first_type = typename T::first_type;
    using second_type = typename T::second_type;
    std::vector<T> data(size);
    auto fdata = get_random_data<first_type>(size, min.x, max.x, max_random_size);
    auto sdata = get_random_data<second_type>(size, min.y, max.y, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(fdata[i], sdata[i]);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<!is_custom_type<T>::value && !std::is_same<decltype(max.x), void>::value, std::vector<T>>::type
{
    // NOTE 1: post-increment operator required, because HIP has different typedefs for vector field types
    //         when using HCC or HIP-Clang. Using HIP-Clang members are accessed as fields of a struct via
    //         a union, but in HCC mode they are proxy types (Scalar_accessor). STL algorithms don't
    //         always tolerate proxies. Unfortunately, Scalar_accessor doesn't have any member typedefs to
    //         conveniently obtain the inner stored type. All operations on it (operator+, operator+=,
    //         CTOR, etc.) return a reference to an accessor, it is only the post-increment operator that
    //         returns a copy of the stored type, hence we take the decltype of that.
    //
    // NOTE 2: decltype() is unevaluated context. We don't really modify max, just compute the type of the
    //         expression if we were to actually call it.
    using field_type = decltype(max.x++);
    std::vector<T> data(size);
    auto field_data = get_random_data<field_type>(size, min.x, max.x, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(field_data[i]);
    }
    return data;
}

bool is_warp_size_supported(const unsigned int required_warp_size)
{
    return ::rocprim::host_warp_size() >= required_warp_size;
}

template<unsigned int LogicalWarpSize>
struct DeviceSelectWarpSize
{
    static constexpr unsigned int value = ::rocprim::device_warp_size() >= LogicalWarpSize
        ? LogicalWarpSize
        : ::rocprim::device_warp_size();
};

template<typename T>
std::vector<T> get_random_segments(const size_t size,
                                   const size_t max_segment_length,
                                   const int seed_value)
{
    static_assert(std::is_arithmetic<T>::value, "Key type must be arithmetic");

    std::default_random_engine prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    using key_distribution_type = std::conditional_t<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>
    >;
    key_distribution_type key_distribution(std::numeric_limits<T>::max());
    std::vector<T> keys(size);

    size_t keys_start_index = 0;
    while (keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end = std::min(size, keys_start_index + new_segment_length);
        const T key = key_distribution(prng);
        std::fill(
            keys.begin() + keys_start_index,
            keys.begin() + new_segment_end,
            key
        );
        keys_start_index += new_segment_length;
    }
    return keys;
}

template <typename T, T, typename>
struct make_index_range_impl;

template <typename T, T Start, T... I>
struct make_index_range_impl<T, Start, std::integer_sequence<T, I...>>
{
    using type = std::integer_sequence<T, (Start + I)...>;
};

// make a std::integer_sequence with values from Start to End inclusive
template <typename T, T Start, T End>
using make_index_range =
    typename make_index_range_impl<T, Start, std::make_integer_sequence<T, End - Start + 1>>::type;

template <typename T, template <T> class Function, T... I, typename... Args>
void static_for_each_impl(std::integer_sequence<T, I...>, Args... args)
{
    int a[] = {(Function<I> {}(args...), 0)...};
    static_cast<void>(a);
}

// call the supplied template with all values of the std::integer_sequence Indices
template <typename Indices,
          template <typename Indices::value_type> class Function,
          typename... Args>
void static_for_each(Args... args)
{
    static_for_each_impl<typename Indices::value_type, Function>(Indices {}, args...);
}

#endif // ROCPRIM_BENCHMARK_UTILS_HPP_
