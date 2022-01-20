// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

using test_common_utils::hipMallocHelper;

namespace
{

template <class T>
struct PinnedAllocator
{
    using value_type = T;

    T* allocate(const size_t size)
    {
        T* ptr{};
        HIP_CHECK(hipHostMalloc(&ptr, size * sizeof(T), hipHostMallocDefault));
        return ptr;
    }

    void deallocate(T* ptr, const size_t)
    {
        HIP_CHECK(hipHostFree(ptr));
    }

    bool operator==(const PinnedAllocator&)
    {
        return true;
    }

    bool operator!=(const PinnedAllocator& other)
    {
        return !(*this == other);
    }
};

std::vector<size_t> get_sizes()
{
    return std::vector<size_t>{
        0, 1, 10, 53, 211, 1024, 2345, 4096, 34567,
        (1 << 16) - 1220, (1 << 22) - 76543
    };
}

template<class T>
__global__
void increment_kernel(T* const d_input, const size_t size)
{
    const unsigned int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(id < size)
    {
        d_input[id] += static_cast<T>(1);
    }
}

}

TEST(HipAsyncCopyTests, AsyncCopySerially)
{
    using T = int;
    using vector_type = std::vector<T, PinnedAllocator<T>>;
    static constexpr int seed = 543897;
    static constexpr unsigned int block_size = 1024;

    std::default_random_engine prng(seed);
    std::uniform_int_distribution<T> dist;

    for(size_t stream_idx = 0; stream_idx < 2; stream_idx++)
    {
        hipStream_t stream = hipStreamDefault;
        if(stream_idx > 0)
        {
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }
        SCOPED_TRACE(testing::Message() << "with stream= " << stream);

        for(const auto size : get_sizes())
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            vector_type input;
            std::generate_n(std::back_inserter(input), size, [&](){ return dist(prng); });
            vector_type expected;
            std::transform(input.begin(), input.end(), std::back_inserter(expected),
                [](const auto& val) { return val + static_cast<T>(1); });

            T* d_input{};
            const auto size_bytes = size * sizeof(T);
            HIP_CHECK(hipMallocHelper(&d_input, size_bytes));
            HIP_CHECK(hipMemcpyAsync(d_input, input.data(), size_bytes, hipMemcpyHostToDevice, stream));

            const unsigned int grid_size = (size + block_size - 1) / block_size;
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(increment_kernel),
                dim3(grid_size), dim3(block_size), 0, stream,
                d_input, size
            );

            vector_type output(size);
            HIP_CHECK(hipMemcpyAsync(output.data(), d_input, size_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipFree(d_input));

            ASSERT_EQ(output, expected);
        }

        if(stream_idx > 0)
        {
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

TEST(HipAsyncCopyTests, AsyncCopyConcurrently)
{
    using T = int;
    using vector_type = std::vector<T, PinnedAllocator<T>>;
    static constexpr int seed = 543897;
    static constexpr unsigned int block_size = 1024;

    std::default_random_engine prng(seed);
    std::uniform_int_distribution<T> dist;

    const auto sizes = get_sizes();
    std::vector<vector_type> inputs(sizes.size());
    std::vector<vector_type> expecteds(sizes.size());
    std::vector<T*> d_inputs(sizes.size());
    std::vector<vector_type> outputs(sizes.size());
    std::vector<hipStream_t> streams(sizes.size());

    for(size_t i = 0; i < sizes.size(); i++)
    {
        const auto size = sizes[i];
        std::generate_n(std::back_inserter(inputs[i]), size, [&](){ return dist(prng); });
        std::transform(inputs[i].begin(), inputs[i].end(), std::back_inserter(expecteds[i]),
            [](const auto& val) { return val + static_cast<T>(1); });
        HIP_CHECK(hipMallocHelper(d_inputs.data() + i, size * sizeof(T)));
        outputs[i].resize(size);
        if(i == 0)
        {
            streams[i] = hipStreamDefault;
        }
        else
        {
            HIP_CHECK(hipStreamCreateWithFlags(streams.data() + i, hipStreamNonBlocking));
        }
    }

    for(size_t i = 0; i < sizes.size(); i++)
    {
        const auto size_bytes = sizes[i] * sizeof(T);
        HIP_CHECK(hipMemcpyAsync(d_inputs[i], inputs[i].data(), size_bytes, hipMemcpyHostToDevice, streams[i]));
        const unsigned int grid_size = (sizes[i] + block_size - 1) / block_size;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(increment_kernel),
            dim3(grid_size), dim3(block_size), 0, streams[i],
            d_inputs[i], sizes[i]
        );
        HIP_CHECK(hipMemcpyAsync(outputs[i].data(), d_inputs[i], size_bytes, hipMemcpyDeviceToHost, streams[i]));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(size_t i = 0; i < sizes.size(); i++)
    {
        if(i > 0)
        {
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
        HIP_CHECK(hipFree(d_inputs[i]));
    }

    ASSERT_EQ(expecteds, outputs);
}

TEST(HipAsyncCopyTests, StreamInStruct)
{
    struct StreamWrapper
    {
        hipStream_t stream;
    };
    using T = int;
    using vector_type = std::vector<T, PinnedAllocator<T>>;
    static constexpr int seed = 543897;
    static constexpr unsigned int block_size = 1024;

    const size_t size = get_sizes().back();
    std::default_random_engine prng(seed);
    std::uniform_int_distribution<T> dist;

    StreamWrapper stream_wrapper;
    HIP_CHECK(hipStreamCreateWithFlags(&stream_wrapper.stream, hipStreamNonBlocking));
    vector_type input;
    std::generate_n(std::back_inserter(input), size, [&](){ return dist(prng); });
    vector_type expected;
    std::transform(input.begin(), input.end(), std::back_inserter(expected),
        [](const auto& val){ return val + static_cast<T>(1); });

    T* d_input{};
    const auto size_bytes = size * sizeof(T);
    HIP_CHECK(hipMallocHelper(&d_input, size_bytes));
    HIP_CHECK(hipMemcpyAsync(d_input, input.data(), size_bytes, hipMemcpyHostToDevice, stream_wrapper.stream));

    const unsigned int grid_size = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(increment_kernel),
        dim3(grid_size), dim3(block_size), 0, stream_wrapper.stream,
        d_input, size
    );

    vector_type output(size);
    HIP_CHECK(hipMemcpyAsync(output.data(), d_input, size_bytes, hipMemcpyDeviceToHost, stream_wrapper.stream));
    HIP_CHECK(hipStreamSynchronize(stream_wrapper.stream));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipStreamDestroy(stream_wrapper.stream));

    ASSERT_EQ(output, expected);
}
