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
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error), hipSuccess)

namespace rp = rocprim;

template<
    class Key,
    class Value,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending = false,
    bool ToStriped = false,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(Key) * 8
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool descending = Descending;
    static constexpr bool to_striped = ToStriped;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
};

template<class Params>
class RocprimBlockRadixSort : public ::testing::Test {
public:
    using params = Params;
};

using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1>,
    params<int, int, 128U, 1>,
    params<unsigned int, int, 256U, 1>,
    params<unsigned short, char, 1024U, 1, true>,
    params<rp::half, int, 128U, 1>,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1>,
    params<float, int, 37U, 1>,
    params<long long, char, 510U, 1, true>,
    params<unsigned int, long long, 162U, 1, false, true>,
    params<unsigned char, float, 255U, 1>,
    params<rp::half, float, 113U, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<float, char, 64U, 2, true>,
    params<int, rp::half, 128U, 4>,
    params<unsigned short, char, 256U, 7>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5>,
    params<char, double, 464U, 2, true, true>,
    params<unsigned short, custom_int2, 100U, 3>,
    params<rp::half, int, 234U, 9>,

    // StartBit and EndBit
    params<unsigned long long, char, 64U, 1, false, false, 8, 20>,
    params<unsigned short, int, 102U, 3, true, false, 4, 10>,
    params<unsigned int, short, 162U, 2, true, true, 3, 12>,

    // Stability (a number of key values is lower than BlockSize * ItemsPerThread: some keys appear
    // multiple times with different values or key parts outside [StartBit, EndBit))
    params<unsigned char, int, 512U, 2, false, true>,
    params<unsigned short, custom_double2, 60U, 1, true, false, 8, 11>
> Params;

TYPED_TEST_CASE(RocprimBlockRadixSort, Params);

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator
{
    static_assert(rp::is_unsigned<Key>::value, "Test supports start and end bits only for unsigned integers");

    bool operator()(const Key& lhs, const Key& rhs)
    {
        auto mask = (1ull << (EndBit - StartBit)) - 1;
        auto l = (static_cast<unsigned long long>(lhs) >> StartBit) & mask;
        auto r = (static_cast<unsigned long long>(rhs) >> StartBit) & mask;
        return Descending ? (r < l) : (l < r);
    }
};

template<class Key, bool Descending>
struct key_comparator<Key, Descending, 0, sizeof(Key) * 8>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        return Descending ? (rhs < lhs) : (lhs < rhs);
    }
};

template<bool Descending>
struct key_comparator<rp::half, Descending, 0, sizeof(rp::half) * 8>
{
    bool operator()(const rp::half& lhs, const rp::half& rhs)
    {
        // HIP's half doesn't have __host__ comparison operators, use floats instead
        return key_comparator<float, Descending, 0, sizeof(float) * 8>()(lhs, rhs);
    }
};

template<class Key, class Value, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return key_comparator<Key, Descending, StartBit, EndBit>()(lhs.first, rhs.first);
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type
>
__global__
void sort_key_kernel(
    key_type* device_keys_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    rp::block_load_direct_blocked(lid, device_keys_output + block_offset, keys);

    rp::block_radix_sort<key_type, BlockSize, ItemsPerThread> bsort;

    if(to_striped)
    {
        if(descending)
            bsort.sort_desc_to_striped(keys, start_bit, end_bit);
        else
            bsort.sort_to_striped(keys, start_bit, end_bit);

        rp::block_store_direct_striped<BlockSize>(lid, device_keys_output + block_offset, keys);
    }
    else
    {
        if(descending)
            bsort.sort_desc(keys, start_bit, end_bit);
        else
            bsort.sort(keys, start_bit, end_bit);

        rp::block_store_direct_blocked(lid, device_keys_output + block_offset, keys);
    }
}

TYPED_TEST(RocprimBlockRadixSort, SortKeys)
{
    using key_type = typename TestFixture::params::key_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<key_type> keys_output;
    if(rp::is_floating_point<key_type>::value)
    {
        keys_output = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_output = test_utils::get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
    }

    // Calculate expected results on host
    std::vector<key_type> expected(keys_output);
    for(size_t i = 0; i < size / items_per_block; i++)
    {
        std::stable_sort(
            expected.begin() + (i * items_per_block),
            expected.begin() + ((i + 1) * items_per_block),
            key_comparator<key_type, descending, start_bit, end_bit>()
        );
    }

    // Preparing device
    key_type* device_keys_output;
    HIP_CHECK(hipMalloc(&device_keys_output, keys_output.size() * sizeof(key_type)));

    HIP_CHECK(
        hipMemcpy(
            device_keys_output, keys_output.data(),
            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sort_key_kernel<block_size, items_per_thread, key_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_keys_output, to_striped, descending, start_bit, end_bit
    );

    // Getting results to host
    HIP_CHECK(
        hipMemcpy(
            keys_output.data(), device_keys_output,
            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    // Verifying results
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));

    HIP_CHECK(hipFree(device_keys_output));
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type
>
__global__
void sort_key_value_kernel(
    key_type* device_keys_output,
    value_type* device_values_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    rp::block_load_direct_blocked(lid, device_keys_output + block_offset, keys);
    rp::block_load_direct_blocked(lid, device_values_output + block_offset, values);

    rp::block_radix_sort<key_type, BlockSize, ItemsPerThread, value_type> bsort;
    if(to_striped)
    {
        if(descending)
            bsort.sort_desc_to_striped(keys, values, start_bit, end_bit);
        else
            bsort.sort_to_striped(keys, values, start_bit, end_bit);

        rp::block_store_direct_striped<BlockSize>(lid, device_keys_output + block_offset, keys);
        rp::block_store_direct_striped<BlockSize>(lid, device_values_output + block_offset, values);
    }
    else
    {
        if(descending)
            bsort.sort_desc(keys, values, start_bit, end_bit);
        else
            bsort.sort(keys, values, start_bit, end_bit);

        rp::block_store_direct_blocked(lid, device_keys_output + block_offset, keys);
        rp::block_store_direct_blocked(lid, device_values_output + block_offset, values);
    }
}


TYPED_TEST(RocprimBlockRadixSort, SortKeysValues)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<key_type> keys_output;
    if(rp::is_floating_point<key_type>::value)
    {
        keys_output = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_output = test_utils::get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
    }

    std::vector<value_type> values_output = test_utils::get_random_data<value_type>(size, 0, 100);

    using key_value = std::pair<key_type, value_type>;

    // Calculate expected results on host
    std::vector<key_value> expected(size);
    for(size_t i = 0; i < size; i++)
    {
        expected[i] = key_value(keys_output[i], values_output[i]);
    }

    for(size_t i = 0; i < size / items_per_block; i++)
    {
        std::stable_sort(
            expected.begin() + (i * items_per_block),
            expected.begin() + ((i + 1) * items_per_block),
            key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
        );
    }

    std::vector<key_type> keys_expected(size);
    std::vector<value_type> values_expected(size);
    for(size_t i = 0; i < size; i++)
    {
        keys_expected[i] = expected[i].first;
        values_expected[i] = expected[i].second;
    }

    key_type* device_keys_output;
    HIP_CHECK(hipMalloc(&device_keys_output, keys_output.size() * sizeof(key_type)));
    value_type* device_values_output;
    HIP_CHECK(hipMalloc(&device_values_output, values_output.size() * sizeof(value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_keys_output, keys_output.data(),
            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_values_output, values_output.data(),
            values_output.size() * sizeof(typename decltype(values_output)::value_type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sort_key_value_kernel<block_size, items_per_thread, key_type, value_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_keys_output, device_values_output, to_striped, descending, start_bit, end_bit
    );

    // Getting results to host
    HIP_CHECK(
        hipMemcpy(
            keys_output.data(), device_keys_output,
            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            values_output.data(), device_values_output,
            values_output.size() * sizeof(typename decltype(values_output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));

    HIP_CHECK(hipFree(device_keys_output));
    HIP_CHECK(hipFree(device_values_output));
}

