// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"
#include "rocprim/types.hpp"

#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>

#include <type_traits>

template<class T>
struct device_plus
{
    __device__ inline constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<class InputType, class Function, class ExpectedType = InputType>
struct RocprimTypeInvokeResultParams
{
    using input_type    = InputType;
    using function      = Function;
    using expected_type = ExpectedType;
};

template<class Params>
class RocprimInvokeResultBinOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

typedef ::testing::Types<
    RocprimTypeInvokeResultParams<unsigned short, rocprim::plus<unsigned short>>,
    RocprimTypeInvokeResultParams<int, rocprim::plus<int>>,
    RocprimTypeInvokeResultParams<float, rocprim::plus<float>>,
    RocprimTypeInvokeResultParams<int, device_plus<int>>,
    RocprimTypeInvokeResultParams<rocprim::bfloat16, device_plus<rocprim::bfloat16>>,
    RocprimTypeInvokeResultParams<rocprim::half, device_plus<rocprim::half>>,
    RocprimTypeInvokeResultParams<int, rocprim::equal_to<int>, bool>,
    RocprimTypeInvokeResultParams<rocprim::bfloat16, rocprim::equal_to<rocprim::bfloat16>, bool>,
    RocprimTypeInvokeResultParams<rocprim::half, rocprim::equal_to<rocprim::half>, bool>>
    RocprimInvokeResultBinOpTestsParams;

TYPED_TEST_SUITE(RocprimInvokeResultBinOpTests, RocprimInvokeResultBinOpTestsParams);

TYPED_TEST(RocprimInvokeResultBinOpTests, HostInvokeResult)
{
    using input_type      = typename TestFixture::input_type;
    using binary_function = typename TestFixture::function;
    using expected_type   = typename TestFixture::expected_type;

    using resulting_type = rocprim::invoke_result_binary_op_t<input_type, binary_function>;

    // Compile and check on host
    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}

template<typename FromType, typename ToType>
struct static_cast_op
{
    __device__ inline constexpr ToType operator()(FromType a) const
    {
        return static_cast<ToType>(a);
    }
};

template<class Params>
class RocprimInvokeResultUnOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

typedef ::testing::Types<
    RocprimTypeInvokeResultParams<ushort, static_cast_op<ushort, float>, float>,
    RocprimTypeInvokeResultParams<double,
                                  static_cast_op<double, rocprim::bfloat16>,
                                  rocprim::bfloat16>,
    RocprimTypeInvokeResultParams<char, rocprim::identity<char>>>
    RocprimInvokeResultUnOpTestsParams;

TYPED_TEST_SUITE(RocprimInvokeResultUnOpTests, RocprimInvokeResultUnOpTestsParams);

TYPED_TEST(RocprimInvokeResultUnOpTests, HostInvokeResult)
{
    using input_type     = typename TestFixture::input_type;
    using unary_function = typename TestFixture::function;
    using expected_type  = typename TestFixture::expected_type;

    using resulting_type = rocprim::invoke_result_t<unary_function, input_type>;

    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}
