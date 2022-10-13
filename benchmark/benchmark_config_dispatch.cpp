// MIT License
//
// Copyright(C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprim/device/config_types.hpp>

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

#include <iostream>

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

enum class stream_kind
{
    default_stream,
    per_thread_stream,
    explicit_stream,
    async_stream
};

static void BM_host_target_arch(benchmark::State& state, const stream_kind stream_kind)
{
    const hipStream_t stream = [stream_kind]() -> hipStream_t
    {
        hipStream_t stream = 0;
        switch(stream_kind)
        {
            case stream_kind::default_stream: return stream;
            case stream_kind::per_thread_stream: return hipStreamPerThread;
            case stream_kind::explicit_stream: HIP_CHECK(hipStreamCreate(&stream)); return stream;
            case stream_kind::async_stream:
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                return stream;
        }
    }();

    for(auto _ : state)
    {
        rocprim::detail::target_arch target_arch;
        HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
        benchmark::DoNotOptimize(target_arch);
    }

    if(stream_kind != stream_kind::default_stream && stream_kind != stream_kind::per_thread_stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

__global__ void empty_kernel() {}

// An empty kernel launch for baseline
static void BM_kernel_launch(benchmark::State& state)
{
    static constexpr hipStream_t stream = 0;
    for(auto _ : state)
    {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipGetLastError());
    }
    hipStreamSynchronize(stream);
}

BENCHMARK_CAPTURE(BM_host_target_arch, default_stream, stream_kind::default_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, per_thread_stream, stream_kind::per_thread_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, explicit_stream, stream_kind::explicit_stream);
BENCHMARK_CAPTURE(BM_host_target_arch, async_stream, stream_kind::async_stream);
BENCHMARK(BM_kernel_launch);

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}