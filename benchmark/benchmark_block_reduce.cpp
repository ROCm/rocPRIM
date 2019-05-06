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
#include <chrono>
#include <vector>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

template<
    class Runner,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
__global__
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(input, output);
}

template<rocprim::block_reduce_algorithm algorithm>
struct reduce
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        T reduced_value;
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using breduce_t = rp::block_reduce<T, BlockSize, algorithm>;
        __shared__ typename breduce_t::storage_type storage;

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            breduce_t().reduce(values, reduced_value, storage);
            values[0] = reduced_value;
        }

        if(hipThreadIdx_x == 0)
        {
            output[hipBlockIdx_x] = reduced_value;
        }
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, 1.0f);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

// IPT - items per thread
#define CREATE_BENCHMARK(T, BS, IPT) \
    benchmark::RegisterBenchmark( \
        (std::string("block_reduce<"#T", "#BS", "#IPT", " + algorithm_name + ">.") + method_name).c_str(), \
        run_benchmark<Benchmark, T, BS, IPT>, \
        stream, size \
    )

template<class Benchmark>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& method_name,
                    const std::string& algorithm_name,
                    hipStream_t stream,
                    size_t size)
{
    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        CREATE_BENCHMARK(float, 256, 1),
        CREATE_BENCHMARK(float, 256, 2),
        CREATE_BENCHMARK(float, 256, 3),
        CREATE_BENCHMARK(float, 256, 4),
        CREATE_BENCHMARK(float, 256, 8),
        CREATE_BENCHMARK(float, 256, 11),
        CREATE_BENCHMARK(float, 256, 16),

        CREATE_BENCHMARK(int, 256, 1),
        CREATE_BENCHMARK(int, 256, 2),
        CREATE_BENCHMARK(int, 256, 3),
        CREATE_BENCHMARK(int, 256, 4),
        CREATE_BENCHMARK(int, 256, 8),
        CREATE_BENCHMARK(int, 256, 11),
        CREATE_BENCHMARK(int, 256, 16),

        CREATE_BENCHMARK(int, 320, 1),
        CREATE_BENCHMARK(int, 320, 2),
        CREATE_BENCHMARK(int, 320, 3),
        CREATE_BENCHMARK(int, 320, 4),
        CREATE_BENCHMARK(int, 320, 8),
        CREATE_BENCHMARK(int, 320, 11),
        CREATE_BENCHMARK(int, 320, 16),

        CREATE_BENCHMARK(double, 256, 1),
        CREATE_BENCHMARK(double, 256, 2),
        CREATE_BENCHMARK(double, 256, 3),
        CREATE_BENCHMARK(double, 256, 4),
        CREATE_BENCHMARK(double, 256, 8),
        CREATE_BENCHMARK(double, 256, 11),
        CREATE_BENCHMARK(double, 256, 16),

        CREATE_BENCHMARK(custom_double2, 256, 1),
        CREATE_BENCHMARK(custom_double2, 256, 4),
        CREATE_BENCHMARK(custom_double2, 256, 8),

        CREATE_BENCHMARK(custom_int_double, 256, 1),
        CREATE_BENCHMARK(custom_int_double, 256, 4),
        CREATE_BENCHMARK(custom_int_double, 256, 8)
    };
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    // using_warp_scan
    using reduce_uwr_t = reduce<rocprim::block_reduce_algorithm::using_warp_reduce>;
    add_benchmarks<reduce_uwr_t>(
        benchmarks, "reduce", "using_warp_reduce", stream, size
    );
    // reduce then scan
    using reduce_rr_t = reduce<rocprim::block_reduce_algorithm::raking_reduce>;
    add_benchmarks<reduce_rr_t>(
        benchmarks, "reduce", "raking_reduce", stream, size
    );

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
