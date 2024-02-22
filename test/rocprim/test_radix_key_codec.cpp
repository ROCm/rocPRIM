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

#include <ios>
#include <ostream>
#include <rocprim/detail/radix_sort.hpp>
#include <rocprim/test_utils_custom_test_types.hpp>
#include <rocprim/test_utils_sort_comparator.hpp>
#include <rocprim/types/tuple.hpp>
#include <sstream>

struct extract_digit_params
{
    unsigned int start;
    unsigned int radix_bits;
    unsigned int expected_result;
};

std::ostream& operator<<(std::ostream& os, const extract_digit_params& params)
{
    std::stringstream sstream;
    sstream << "{ start: " << params.start << ", radix_bits: " << params.radix_bits
            << ", expected_result: 0x" << std::hex << params.expected_result << " }";
    return os << sstream.str();
}

class RadixKeyCodecTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(RocprimBlockRadixSort,
                         RadixKeyCodecTest,
                         ::testing::Values(extract_digit_params{0, 8, 0x01},
                                           extract_digit_params{8, 16, 0xcdef},
                                           extract_digit_params{24, 8, 0xab},
                                           extract_digit_params{7, 11, 0b01'1110'1111'0},
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
    auto operator()(custom_key& value) const
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

class RadixKeyCodecUnusedTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(RocprimBlockRadixSort,
                         RadixKeyCodecUnusedTest,
                         ::testing::Values(extract_digit_params{0, 16, 0xab01},
                                           extract_digit_params{0, 8, 0x01},
                                           extract_digit_params{8, 8, 0xab},
                                           extract_digit_params{1, 14, 0b010'1011'0000'000},
                                           extract_digit_params{14, 2, 0b10}));

struct custom_key_decomposer_with_unused
{
    auto operator()(custom_key& value) const
    {
        return ::rocprim::tuple<uint8_t&, uint8_t&>{value.a, value.c};
    }
};

TEST_P(RadixKeyCodecUnusedTest, TestExtractDigitUnused)
{
    using codec = rocprim::detail::radix_key_codec_inplace<custom_key>;

    const custom_key key{0xab, 0xcdef, 0x01};
    const auto       digit = codec::extract_digit(key,
                                            GetParam().start,
                                            GetParam().radix_bits,
                                            custom_key_decomposer_with_unused{});

    ASSERT_EQ(digit, GetParam().expected_result);
}

TEST(RadixKeyCodecTest, ExtractCustomTestType)
{
    using T       = test_utils::custom_test_type<int>;
    using codec_t = rocprim::detail::radix_key_codec_inplace<T, true>;

    T value{12, 34};

    test_utils::custom_test_type_decomposer<T> decomposer;
    codec_t::encode_inplace(value, decomposer);

    ASSERT_EQ(0x7FFFFFDD, codec_t::extract_digit(value, 0, 32, decomposer));
    ASSERT_EQ(0x7FFFFFF3, codec_t::extract_digit(value, 32, 32, decomposer));
}
