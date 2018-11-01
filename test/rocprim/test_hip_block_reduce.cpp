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

#include <iostream>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error), hipSuccess)

namespace rp = rocprim;

template<class T, class BinaryOp>
T apply(BinaryOp binary_op, const T& a, const T& b)
{
    return binary_op(a, b);
}

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rp::block_reduce_algorithm Algorithm = rp::block_reduce_algorithm::using_warp_reduce,
    class BinaryOp = rocprim::plus<T>
>
struct params
{
    using type = T;
    using binary_op_type = BinaryOp;
    static constexpr rp::block_reduce_algorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimBlockReduceSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    using binary_op_type = typename Params::binary_op_type;
    static constexpr rp::block_reduce_algorithm algorithm = Params::algorithm;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::using_warp_reduce
    // -----------------------------------------------------------------------
    params<int, 64U>,
    params<int, 128U>,
    params<int, 192U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 1024U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 129U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // char tests
    params<char, 64U>,
    params<char, 256U>,
    params<char, 377U>,
    // half tests
    params<rp::half, 64U, 1, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    params<rp::half, 256U, 1, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    params<rp::half, 377U, 1, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>,
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::raking_reduce
    // -----------------------------------------------------------------------
    params<int, 64U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 128U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 192U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 256U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 512U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 1024U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<char, 64U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<char, 128U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<char, 192U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<char, 256U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<rp::half, 64U, 1, rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<rp::half, 128U, 1, rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<rp::half, 192U, 1, rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<rp::half, 256U, 1, rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<unsigned long, 65U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<long, 37U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<short, 162U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<unsigned int, 255U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<int, 377U, 1, rp::block_reduce_algorithm::raking_reduce>,
    params<unsigned char, 377U, 1, rp::block_reduce_algorithm::raking_reduce>
> SingleValueTestParams;

TYPED_TEST_CASE(RocprimBlockReduceSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rp::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, BinaryOp());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TYPED_TEST(RocprimBlockReduceSingleValueTests, Reduce)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename TestFixture::binary_op_type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            value = apply(binary_op, value, output[idx]);
        }
        expected_reductions[i] = value;
    }

    // Preparing device
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
    T* device_output_reductions;
    HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_kernel<block_size, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Verifying results
    test_utils::assert_eq(output_reductions, expected_reductions);

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T
>
__global__
void reduce_multiplies_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rp::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, rocprim::multiplies<T>());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceMultiplies)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Half not tested here
    if(std::is_same<T, rp::half>::value)
    {
        return;
    }

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output(size, 1);
    auto two_places = test_utils::get_random_data<unsigned int>(size/32, 0, size-1);
    for(auto i : two_places)
    {
        output[i] = T(2);
    }
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 1;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            value *= output[idx];
        }
        expected_reductions[i] = value;
    }

    // Preparing device
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
    T* device_output_reductions;
    HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_multiplies_kernel<block_size, algorithm, T>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Verifying results
    test_utils::assert_eq(output_reductions, expected_reductions);

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}

TYPED_TEST_CASE(RocprimBlockReduceSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void reduce_valid_kernel(T* device_output, T* device_output_reductions, const unsigned int valid_items)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rp::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, valid_items, BinaryOp());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceValid)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename TestFixture::binary_op_type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    const unsigned int valid_items = test_utils::get_random_value(block_size - 10, block_size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < valid_items; j++)
        {
            auto idx = i * block_size + j;
            value = apply(binary_op, value, output[idx]);
        }
        expected_reductions[i] = value;
    }

    // Preparing device
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
    T* device_output_reductions;
    HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_valid_kernel<block_size, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions, valid_items
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Verifying results
    test_utils::assert_eq(output_reductions, expected_reductions);

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}


template<class Params>
class RocprimBlockReduceInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    using binary_op_type = typename Params::binary_op_type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr rocprim::block_reduce_algorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::using_warp_reduce
    // -----------------------------------------------------------------------
    params<float, 6U,   32>,
    params<float, 32,   2>,
    params<unsigned int, 256,  3>,
    params<int, 512,  4>,
    params<float, 1024, 1>,
    params<float, 37,   2>,
    params<float, 65,   5>,
    params<float, 162,  7>,
    params<float, 255,  15>,
    params<char, 1024, 1>,
    params<char, 37,   2>,
    params<char, 65,   5>,
    params<rp::half, 1024, 1, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    params<rp::half, 37,   2, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    params<rp::half, 65,   5, rp::block_reduce_algorithm::using_warp_reduce, test_utils::half_maximum>,
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::raking_reduce
    // -----------------------------------------------------------------------
    params<float, 6U,   32, rp::block_reduce_algorithm::raking_reduce>,
    params<float, 32,   2,  rp::block_reduce_algorithm::raking_reduce>,
    params<int, 256,  3,  rp::block_reduce_algorithm::raking_reduce>,
    params<unsigned int, 512,  4,  rp::block_reduce_algorithm::raking_reduce>,
    params<float, 1024, 1,  rp::block_reduce_algorithm::raking_reduce>,
    params<float, 37,   2,  rp::block_reduce_algorithm::raking_reduce>,
    params<float, 65,   5,  rp::block_reduce_algorithm::raking_reduce>,
    params<float, 162,  7,  rp::block_reduce_algorithm::raking_reduce>,
    params<float, 255,  15, rp::block_reduce_algorithm::raking_reduce>,
    params<char, 1024, 1,  rp::block_reduce_algorithm::raking_reduce>,
    params<char, 37,   2,  rp::block_reduce_algorithm::raking_reduce>,
    params<char, 65,   5,  rp::block_reduce_algorithm::raking_reduce>,
    params<rp::half, 1024, 1,  rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<rp::half, 37,   2,  rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>,
    params<rp::half, 65,   5,  rp::block_reduce_algorithm::raking_reduce, test_utils::half_maximum>
> InputArrayTestParams;

TYPED_TEST_CASE(RocprimBlockReduceInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
void reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rp::block_reduce<T, BlockSize, Algorithm> breduce;
    T reduction;
    breduce.reduce(in_out, reduction, BinaryOp());

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

TYPED_TEST(RocprimBlockReduceInputArrayTests, Reduce)
{
    using T = typename TestFixture::type;
    using binary_op_type = typename TestFixture::binary_op_type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size, 0);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        T value = 0;
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            value = apply(binary_op, value, output[idx]);
        }
        expected_reductions[i] = value;
    }

    // Preparing device
    T* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
    T* device_output_reductions;
    HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

    HIP_CHECK(
        hipMemcpy(
            device_output, output.data(),
            output.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_output_reductions, output_reductions.data(),
            output_reductions.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_output, device_output_reductions
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_reductions.data(), device_output_reductions,
            output_reductions.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    // Verifying results
    test_utils::assert_near(output_reductions, expected_reductions, 0.05);

    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_reductions));
}
