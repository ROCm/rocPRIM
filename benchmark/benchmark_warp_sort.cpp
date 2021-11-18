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

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

template<class K, unsigned int BlockSize, unsigned int WarpSize, unsigned int ItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void warp_sort_kernel(K* input_keys, K* output_keys)
{
    const unsigned int flat_tid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread; 
    const unsigned int block_offset = blockIdx.x * items_per_block;

    K keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(flat_tid, input_keys + block_offset, keys);

    rp::warp_sort<K, WarpSize> wsort;
    wsort.sort(keys);

    rp::block_store_direct_blocked(flat_tid, output_keys + block_offset, keys);
}

template<class K, class V, unsigned int BlockSize, unsigned int WarpSize, unsigned int ItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void warp_sort_by_key_kernel(K* input_keys, V* input_values, K* output_keys, V* output_values)
{
    const unsigned int flat_tid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread; 
    const unsigned int block_offset = blockIdx.x * items_per_block;

    K keys[ItemsPerThread];
    V values[ItemsPerThread];

    rp::block_load_direct_striped<BlockSize>(flat_tid, input_keys + block_offset, keys);
    rp::block_load_direct_striped<BlockSize>(flat_tid, input_values + block_offset, values);

    rp::warp_sort<K, WarpSize, V> wsort;
    wsort.sort(keys, values);

    rp::block_store_direct_blocked(flat_tid, output_keys + block_offset, keys);
    rp::block_store_direct_blocked(flat_tid, output_values + block_offset, values);
}

template<class Value>
Value get_max_value()
{
    return Value(10000);
}

template<>
char get_max_value()
{
    return std::numeric_limits<char>::max();
}

template<>
custom_type<char, double> get_max_value()
{
    return custom_type<char, double>(std::numeric_limits<char>::max());
}

template<
    class Key,
    unsigned int BlockSize,
    unsigned int WarpSize,
    unsigned int ItemsPerThread = 1,
    class Value = Key,
    bool SortByKey = false,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    // Make sure size is a multiple of items_per_block
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    size = BlockSize * ((size + items_per_block - 1) / items_per_block);
    // Allocate and fill memory
    std::vector<Key> input_key = get_random_data<Key>(size, 0, get_max_value<Key>());
    std::vector<Value> input_value(size_t(1));
    if(SortByKey) input_value = get_random_data<Value>(size, 0, get_max_value<Value>());
    Key * d_input_key = nullptr;
    Key * d_output_key = nullptr;
    Value * d_input_value = nullptr;
    Value * d_output_value = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_key), size * sizeof(Key)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_key), size * sizeof(Key)));
    if(SortByKey) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_value), size * sizeof(Value)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_value), size * sizeof(Value)));
    }
    HIP_CHECK(
        hipMemcpy(
            d_input_key, input_key.data(),
            size * sizeof(Key),
            hipMemcpyHostToDevice
        )
    );
    if(SortByKey) HIP_CHECK(
        hipMemcpy(
            d_input_value, input_value.data(),
            size * sizeof(Value),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if(SortByKey)
        {
            ROCPRIM_NO_UNROLL
            for(unsigned int trial = 0; trial < Trials; ++trial) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_sort_by_key_kernel<Key, Value, BlockSize, WarpSize, ItemsPerThread>),
                    dim3(size/items_per_block), dim3(BlockSize), 0, stream,
                    d_input_key, d_input_value, d_output_key, d_output_value
                );
            }
        }
        else
        {
            ROCPRIM_NO_UNROLL
            for(unsigned int trial = 0; trial < Trials; ++trial) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_sort_kernel<Key, BlockSize, WarpSize, ItemsPerThread>),
                    dim3(size/items_per_block), dim3(BlockSize), 0, stream,
                    d_input_key, d_output_key
                );
            }
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    // SortByKey also transfers values
    auto sorted_type_size = sizeof(Key);
    if(SortByKey) sorted_type_size += sizeof(Value);
    state.SetBytesProcessed(state.iterations() * size * sorted_type_size * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input_key));
    HIP_CHECK(hipFree(d_output_key));
    HIP_CHECK(hipFree(d_input_value));
    HIP_CHECK(hipFree(d_output_value));
}

#define CREATE_SORT_BENCHMARK(K, BS, WS, IPT) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS", "#IPT">.sort(only keys)", \
        run_benchmark<K, BS, WS, IPT>, \
        stream, size \
    )

#define CREATE_SORTBYKEY_BENCHMARK(K, V, BS, WS, IPT) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS", "#IPT", "#V">.sort", \
        run_benchmark<K, BS, WS, IPT, V, true>, \
        stream, size \
    )

#define BENCHMARK_TYPE(type) \
    CREATE_SORT_BENCHMARK(type,  64, 64, 1), \
    CREATE_SORT_BENCHMARK(type,  64, 64, 2), \
    CREATE_SORT_BENCHMARK(type,  64, 64, 4), \
    CREATE_SORT_BENCHMARK(type, 128, 64, 1), \
    CREATE_SORT_BENCHMARK(type, 128, 64, 2), \
    CREATE_SORT_BENCHMARK(type, 128, 64, 4), \
    CREATE_SORT_BENCHMARK(type, 256, 64, 1), \
    CREATE_SORT_BENCHMARK(type, 256, 64, 2), \
    CREATE_SORT_BENCHMARK(type, 256, 64, 4), \
    CREATE_SORT_BENCHMARK(type,  64, 32, 1), \
    CREATE_SORT_BENCHMARK(type,  64, 32, 2), \
    CREATE_SORT_BENCHMARK(type,  64, 16, 1), \
    CREATE_SORT_BENCHMARK(type,  64, 16, 2), \
    CREATE_SORT_BENCHMARK(type,  64, 16, 4)

#define BENCHMARK_KEY_TYPE(type, value) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64,  1), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64,  2), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64,  4), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 1), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 2), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 4)

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

    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    using custom_int2            = custom_type<int, int>;
    using custom_char_double     = custom_type<char, double>;
    using custom_longlong_double = custom_type<long long, double>;

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half),

        BENCHMARK_KEY_TYPE(float, float),
        BENCHMARK_KEY_TYPE(unsigned int, int),
        BENCHMARK_KEY_TYPE(int, custom_double2),
        BENCHMARK_KEY_TYPE(int, custom_int_double),
        BENCHMARK_KEY_TYPE(custom_int2, custom_double2),
        BENCHMARK_KEY_TYPE(custom_int2, custom_char_double),
        BENCHMARK_KEY_TYPE(custom_int2, custom_longlong_double),
        BENCHMARK_KEY_TYPE(int8_t, int8_t),
        BENCHMARK_KEY_TYPE(uint8_t, uint8_t),
        BENCHMARK_KEY_TYPE(rocprim::half, rocprim::half)
    };

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
