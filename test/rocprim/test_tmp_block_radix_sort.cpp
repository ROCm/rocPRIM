#include "../common_test_header.hpp"

#include "rocprim/block/block_radix_sort.hpp"
#include "rocprim/detail/radix_sort.hpp"

#include "rocprim/types/tuple.hpp"
#include "test_utils.hpp"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <ostream>
#include <vector>

// template<class T>
// __global__ void kernel(T* d_out)
// {
//     using tuple_t = rocprim::tuple<T&, T&>;
//     rocprim::detail::radix_key_codec<tuple_t> codec;
//     tuple_t tuple{ d_out[0], d_out[1] };
//     codec.encode_inplace(tuple);
//     codec.decode_inplace(tuple);
// }

// TEST(TmpBlockRadixSortTest, Test)
// {
//     using T = int;

//     const std::vector<T> h_in{ 1, -1 };
//     const size_t size = h_in.size();
//     std::vector<T> h_out(size);

//     T* d_out;
//     HIP_CHECK(hipMalloc(&d_out, size * sizeof(*d_out)));
//     HIP_CHECK(hipMemcpy(d_out, h_in.data(), size * sizeof(*d_out), hipMemcpyHostToDevice));

//     kernel<<<dim3(1), dim3(1), 0, 0>>>(d_out);

//     HIP_CHECK(hipMemcpy(h_out.data(), d_out, size * sizeof(*d_out), hipMemcpyDeviceToHost));
//     HIP_CHECK(hipFree(d_out));

//     for (auto x : h_out) {
//         std::cout << x << ", ";
//     }
//     std::cout << std::endl;
// }

struct extract_digit_params
{
    unsigned int start;
    unsigned int radix_bits;
    unsigned int expected_result;
};

class TmpBlockRadixSortTest : public ::testing::TestWithParam<extract_digit_params>
{};

INSTANTIATE_TEST_SUITE_P(TmpBlockRadixSortTest,
                         TmpBlockRadixSortTest,
                         ::testing::Values(extract_digit_params{7, 11, 0b01'1110'1111'0},
                                           extract_digit_params{0, 1, 1},
                                           extract_digit_params{1, 1, 0},
                                           extract_digit_params{0, 32, 0xabcdef01},
                                           extract_digit_params{1, 31, 0xabcdef01 >> 1},
                                           extract_digit_params{8, 12, 0xdef},
                                           extract_digit_params{12, 12, 0xcde},
                                           extract_digit_params{12, 13, 0x1cde},
                                           extract_digit_params{12, 20, 0xabcde}));

TEST_P(TmpBlockRadixSortTest, TestExtractDigit)
{
    using tuple_t = rocprim::tuple<char, short, char>;
    using codec   = rocprim::detail::radix_key_codec<tuple_t>;

    const tuple_t tuple{0xab, 0xcdef, 0x01};
    const auto    digit
        = codec::extract_digit_from_key(tuple, GetParam().start, GetParam().radix_bits);

    ASSERT_EQ(digit, GetParam().expected_result);
}
