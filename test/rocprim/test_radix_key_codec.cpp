// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprim/detail/radix_sort.hpp>
#include <rocprim/types/tuple.hpp>

struct extract_digit_params
{
    unsigned int start;
    unsigned int radix_bits;
    unsigned int expected_result;
};

class RadixKeyCodecTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(RocprimBlockRadixSort,
                         RadixKeyCodecTest,
                         ::testing::Values(extract_digit_params{7, 11, 0b01'1110'1111'0},
                                           extract_digit_params{0, 1, 1},
                                           extract_digit_params{1, 1, 0},
                                           extract_digit_params{0, 32, 0xabcdef01},
                                           extract_digit_params{1, 31, 0xabcdef01 >> 1},
                                           extract_digit_params{8, 12, 0xdef},
                                           extract_digit_params{12, 12, 0xcde},
                                           extract_digit_params{12, 13, 0x1cde},
                                           extract_digit_params{12, 20, 0xabcde}));

struct custom_key
{
    uint8_t  a;
    uint16_t b;
    uint8_t  c;
};

struct custom_key_decomposer
{
    ::rocprim::tuple<uint8_t&, uint16_t&, uint8_t&> operator()(custom_key& value) const
    {
        return ::rocprim::tuple<uint8_t&, uint16_t&, uint8_t&>{value.a, value.b, value.c};
    }
};

TEST_P(RadixKeyCodecTest, TestExtractDigit)
{
    using codec = rocprim::detail::radix_key_codec_inplace<custom_key>;

    const custom_key key{0xab, 0xcdef, 0x01};
    const auto       digit = codec::extract_digit(key,
                                            GetParam().start,
                                            GetParam().radix_bits,
                                            custom_key_decomposer{});

    ASSERT_EQ(digit, GetParam().expected_result);
}
