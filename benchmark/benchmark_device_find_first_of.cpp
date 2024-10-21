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

#include "benchmark_device_find_first_of.hpp"
#include "benchmark_utils.hpp"

// CmdParser
#include "cmdparser.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

#define CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, FIRST_OCCURENCE)                 \
    {                                                                                    \
        const device_find_first_of_benchmark<TYPE> instance(KEYS_SIZE, FIRST_OCCURENCE); \
        REGISTER_BENCHMARK(benchmarks, size, seed, stream, instance);                    \
    }

// clang-format off
#define CREATE_BENCHMARK0(TYPE, KEYS_SIZE)                   \
    {                                                        \
        CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 0.1) \
        CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 0.5) \
        CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 1.0) \
    }

#define CREATE_BENCHMARK(TYPE)         \
    {                                  \
        CREATE_BENCHMARK0(TYPE, 10)    \
        CREATE_BENCHMARK0(TYPE, 128)   \
        CREATE_BENCHMARK0(TYPE, 1024)  \
        CREATE_BENCHMARK0(TYPE, 10000) \
    }
// clang-format on

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{};
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int64_t)

    using custom_int2            = custom_type<int, int>;
    using custom_longlong_double = custom_type<long long, double>;

    CREATE_BENCHMARK(custom_int2)
    CREATE_BENCHMARK(custom_longlong_double)

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
