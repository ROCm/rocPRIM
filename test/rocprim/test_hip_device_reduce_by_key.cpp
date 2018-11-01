// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

template<
    class Key,
    class Value,
    class ReduceOp,
    unsigned int MinSegmentLength,
    unsigned int MaxSegmentLength,
    class Aggregate = Value,
    class KeyCompareFunction = ::rocprim::equal_to<Key>,
    // Tests output iterator with void value_type (OutputIterator concept)
    bool UseIdentityIterator = false
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    using reduce_op_type = ReduceOp;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    using aggregate_type = Aggregate;
    using key_compare_op = KeyCompareFunction;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceReduceByKey : public ::testing::Test {
public:
    using params = Params;
};

struct custom_reduce_op1
{
    template<class T>
    ROCPRIM_HOST_DEVICE
    T operator()(T a, T b)
    {
        return a + b;
    }
};

template<class T>
struct custom_key_compare_op1
{
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return static_cast<int>(a / 10) == static_cast<int>(b / 10);
    }
};

using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    params<int, int, rp::plus<int>, 1, 1, int, rp::equal_to<int>, true>,
    params<double, int, rp::plus<int>, 3, 5, long long, custom_key_compare_op1<double>>,
    params<float, custom_double2, rp::minimum<custom_double2>, 1, 10000>,
    params<custom_double2, custom_int2, rp::plus<custom_int2>, 1, 10>,
    params<unsigned long long, float, rp::minimum<float>, 1, 30>,
    params<int, rp::half, test_utils::half_minimum, 15, 100>,
    params<int, unsigned int, rp::maximum<unsigned int>, 20, 100>,
    params<float, long long, rp::maximum<unsigned long long>, 100, 400, long long, custom_key_compare_op1<float>>,
    params<unsigned int, unsigned char, rp::plus<unsigned char>, 200, 600>,
    params<double, int, rp::plus<int>, 100, 2000, double, custom_key_compare_op1<double>>,
    params<char, rp::half, test_utils::half_maximum, 123, 1234>,
    params<custom_int2, unsigned int, rp::plus<unsigned int>, 1000, 5000>,
    params<unsigned int, int, rp::plus<int>, 2048, 2048>,
    params<long long, short, rp::plus<long long>, 1000, 10000, long long>,
    params<unsigned int, double, rp::minimum<double>, 1000, 50000>,
    params<unsigned long long, unsigned long long, rp::plus<unsigned long long>, 100000, 100000>
> Params;

TYPED_TEST_CASE(RocprimDeviceReduceByKey, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        100000,
        (1 << 16) - 1220, (1 << 23) - 76543
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(10, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(RocprimDeviceReduceByKey, ReduceByKey)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    using aggregate_type = typename TestFixture::params::aggregate_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;
    using key_compare_op_type = typename TestFixture::params::key_compare_op;
    using key_inner_type = typename test_utils::inner_type<key_type>::type;
    using key_distribution_type = typename std::conditional<
        std::is_floating_point<key_inner_type>::value,
        std::uniform_real_distribution<key_inner_type>,
        std::uniform_int_distribution<key_inner_type>
    >::type;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool debug_synchronous = false;

    reduce_op_type reduce_op;
    key_compare_op_type key_compare_op;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    for(size_t size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<key_type> unique_expected;
        std::vector<aggregate_type> aggregates_expected;
        size_t unique_count_expected = 0;

        std::vector<key_type> keys_input(size);
        key_distribution_type key_delta_dis(1, 5);
        std::uniform_int_distribution<size_t> key_count_dis(
            TestFixture::params::min_segment_length,
            TestFixture::params::max_segment_length
        );
        std::vector<value_type> values_input = test_utils::get_random_data<value_type>(size, 0, 100);

        size_t offset = 0;
        key_type current_key = key_distribution_type(0, 100)(gen);
        key_type prev_key = current_key;
        while(offset < size)
        {
            const size_t key_count = key_count_dis(gen);
            current_key = current_key + key_delta_dis(gen);

            const size_t end = std::min(size, offset + key_count);
            for(size_t i = offset; i < end; i++)
            {
                keys_input[i] = current_key;
            }
            aggregate_type aggregate = values_input[offset];
            for(size_t i = offset + 1; i < end; i++)
            {
                aggregate = reduce_op(aggregate, static_cast<aggregate_type>(values_input[i]));
            }

            // The first key of the segment must be written into unique
            // (it may differ from other keys in case of custom key compraison operators)
            if(unique_count_expected == 0 || !key_compare_op(prev_key, current_key))
            {
                unique_expected.push_back(current_key);
                unique_count_expected++;
                aggregates_expected.push_back(aggregate);
            }
            else
            {
                aggregates_expected.back() = reduce_op(aggregates_expected.back(), aggregate);
            }

            prev_key = current_key;
            offset += key_count;
        }

        key_type * d_keys_input;
        value_type * d_values_input;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
        HIP_CHECK(
            hipMemcpy(
                d_keys_input, keys_input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(value_type),
                hipMemcpyHostToDevice
            )
        );

        key_type * d_unique_output;
        aggregate_type * d_aggregates_output;
        unsigned int * d_unique_count_output;
        HIP_CHECK(hipMalloc(&d_unique_output, unique_count_expected * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_aggregates_output, unique_count_expected * sizeof(aggregate_type)));
        HIP_CHECK(hipMalloc(&d_unique_count_output, sizeof(unsigned int)));

        size_t temporary_storage_bytes;

        HIP_CHECK(
            rp::reduce_by_key(
                nullptr, temporary_storage_bytes,
                d_keys_input, d_values_input, size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unique_output),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_aggregates_output),
                d_unique_count_output,
                reduce_op, key_compare_op,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            rp::reduce_by_key(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_values_input, size,
                d_unique_output, d_aggregates_output,
                d_unique_count_output,
                reduce_op, key_compare_op,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<key_type> unique_output(unique_count_expected);
        std::vector<aggregate_type> aggregates_output(unique_count_expected);
        std::vector<unsigned int> unique_count_output(1);
        HIP_CHECK(
            hipMemcpy(
                unique_output.data(), d_unique_output,
                unique_count_expected * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                aggregates_output.data(), d_aggregates_output,
                unique_count_expected * sizeof(aggregate_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                unique_count_output.data(), d_unique_count_output,
                sizeof(unsigned int),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_aggregates_output));
        HIP_CHECK(hipFree(d_unique_count_output));

        ASSERT_EQ(unique_count_output[0], unique_count_expected);

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(unique_output, unique_expected));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output, aggregates_expected));
    }
}
