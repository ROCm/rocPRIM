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

template<class InputType, class Function, class ExpectedType = InputType>
struct RocprimTypeTraitsOpParams
{
    using input_type    = InputType;
    using function      = Function;
    using expected_type = ExpectedType;
};

template<class Params>
class RocprimTypeTraitsOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

typedef ::testing::Types<
    RocprimTypeTraitsOpParams<ushort, rocprim::plus<ushort>>,
    RocprimTypeTraitsOpParams<int, rocprim::plus<int>>,
    RocprimTypeTraitsOpParams<float, rocprim::plus<float>>,
    RocprimTypeTraitsOpParams<rocprim::bfloat16, rocprim::plus<rocprim::bfloat16>>,
    RocprimTypeTraitsOpParams<rocprim::half, rocprim::plus<rocprim::half>>,
    RocprimTypeTraitsOpParams<int, rocprim::equal_to<int>, bool>,
    RocprimTypeTraitsOpParams<rocprim::bfloat16, rocprim::equal_to<rocprim::bfloat16>, bool>,
    RocprimTypeTraitsOpParams<rocprim::half, rocprim::equal_to<rocprim::half>, bool>>
    RocprimTypeTraitsBinOpTestsParams;

TYPED_TEST_SUITE(RocprimTypeTraitsOpTests, RocprimTypeTraitsBinOpTestsParams);

TYPED_TEST(RocprimTypeTraitsOpTests, BinaryOpTraits)
{
    using input_type      = typename TestFixture::input_type;
    using binary_function = typename TestFixture::function;
    using expected_type   = typename TestFixture::expected_type;

    using resulting_type =
        typename rocprim::invoke_result_binary_op<input_type, binary_function>::type;

    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}

template<typename FromType, typename ToType>
struct static_cast_op
{
    ROCPRIM_HOST_DEVICE inline constexpr ToType operator()(FromType a) const
    {
        return static_cast<ToType>(a);
    }
};

template<class Params>
class RocprimTypeTraitsUnOpTests : public ::testing::Test
{
public:
    using input_type    = typename Params::input_type;
    using function      = typename Params::function;
    using expected_type = typename Params::expected_type;
};

typedef ::testing::Types<
    RocprimTypeTraitsOpParams<ushort, static_cast_op<ushort, float>, float>,
    RocprimTypeTraitsOpParams<double, static_cast_op<double, rocprim::bfloat16>, rocprim::bfloat16>,
    RocprimTypeTraitsOpParams<char, rocprim::identity<char>>>
    RocprimTypeTraitsUnOpTestsParams;

TYPED_TEST_SUITE(RocprimTypeTraitsUnOpTests, RocprimTypeTraitsUnOpTestsParams);

TYPED_TEST(RocprimTypeTraitsUnOpTests, UnaryOpTraits)
{
    using input_type     = typename TestFixture::input_type;
    using unary_function = typename TestFixture::function;
    using expected_type  = typename TestFixture::expected_type;

    using resulting_type = typename rocprim::invoke_result<unary_function, input_type>::type;

    static_assert(std::is_same<resulting_type, expected_type>::value,
                  "Resulting type is not equal to expected type!");
}
