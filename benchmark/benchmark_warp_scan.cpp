// MIT License
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

template<class T, unsigned int WarpSize, unsigned int Trials>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void warp_inclusive_scan_kernel(const T* input, T* output)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    auto value = input[i];

    using wscan_t = rp::warp_scan<T, WarpSize>;
    __shared__ typename wscan_t::storage_type storage;
    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wscan_t().inclusive_scan(value, value, storage);
    }

    output[i] = value;
}

template<class T, unsigned int WarpSize, unsigned int Trials>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void warp_exclusive_scan_kernel(const T* input, T* output, const T init)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    auto value = input[i];

    using wscan_t = rp::warp_scan<T, WarpSize>;
    __shared__ typename wscan_t::storage_type storage;
    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wscan_t().exclusive_scan(value, value, init, storage);
    }

    output[i] = value;
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int WarpSize,
    bool Inclusive = true,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, (T)1);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
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
        if(Inclusive)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, WarpSize, Trials>),
                dim3(size/BlockSize), dim3(BlockSize), 0, stream,
                d_input, d_output
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, WarpSize, Trials>),
                dim3(size/BlockSize), dim3(BlockSize), 0, stream,
                d_input, d_output, input[0]
            );
        }
        HIP_CHECK(hipGetLastError());
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

#define CREATE_BENCHMARK(T, BS, WS, INCLUSIVE) \
    benchmark::RegisterBenchmark( \
        (std::string("warp_scan<"#T", "#BS", "#WS">.") + method_name).c_str(), \
        run_benchmark<T, BS, WS, INCLUSIVE>, \
        stream, size \
    )

#define BENCHMARK_TYPE(type) \
    CREATE_BENCHMARK(type, 64, 64, Inclusive), \
    CREATE_BENCHMARK(type, 128, 64, Inclusive), \
    CREATE_BENCHMARK(type, 256, 64, Inclusive), \
    CREATE_BENCHMARK(type, 256, 32, Inclusive), \
    CREATE_BENCHMARK(type, 256, 16, Inclusive), \
    CREATE_BENCHMARK(type, 63, 63, Inclusive), \
    CREATE_BENCHMARK(type, 62, 31, Inclusive), \
    CREATE_BENCHMARK(type, 60, 15, Inclusive)

template<bool Inclusive>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& method_name,
                    hipStream_t stream,
                    size_t size)
{
    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half),
        BENCHMARK_TYPE(custom_double2),
        BENCHMARK_TYPE(custom_int_double)
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

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<true>(benchmarks, "inclusive_scan", stream, size);
    add_benchmarks<false>(benchmarks, "exclusive_scan", stream, size);

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
