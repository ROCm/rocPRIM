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

#ifndef ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_find_first_of.hpp>

#include <cstddef>
#include <string>
#include <vector>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_find_first_of_benchmark : public config_autotune_interface
{
    size_t keys_size;
    double first_occurence;

    device_find_first_of_benchmark(size_t keys_size, double first_occurence)
        : keys_size(keys_size), first_occurence(first_occurence)
    {}

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:find_first_of,keys_size:" + std::to_string(keys_size)
            + ",first_occurence:" + std::to_string(first_occurence)
            + ",key_type:" + std::string(Traits<Key>::name()) + ",cfg:default_config}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using type        = Key;
        using key_type    = Key;
        using output_type = size_t;

        // Generate data
        std::vector<key_type> key_input
            = get_random_data<key_type>(keys_size, 0, 100, seed.get_0());
        std::vector<type> input
            = get_random_data<type>(size, 101, generate_limits<type>::max(), seed.get_0());

        // Set the first occurence of keys in input
        const size_t p = static_cast<size_t>(size * first_occurence);
        if(p < size)
        {
            input[p] = key_input[keys_size / 2];
        }

        type*        d_input;
        key_type*    d_key_input;
        output_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMalloc(&d_key_input, size * sizeof(*d_key_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));

        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(*d_input),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_key_input,
                            key_input.data(),
                            key_input.size() * sizeof(*d_key_input),
                            hipMemcpyHostToDevice));

        ::rocprim::equal_to<type> compare_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        auto run = [&]()
        {
            HIP_CHECK(rocprim::find_first_of(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_input,
                                             d_key_input,
                                             d_output,
                                             input.size(),
                                             key_input.size(),
                                             compare_op,
                                             stream));
        };

        run();
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            run();
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                run();
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        // Only a part of data (before the first occurence) must be actually processed
        const size_t effective_size = static_cast<size_t>(size * first_occurence);
        state.SetBytesProcessed(state.iterations() * batch_size * effective_size
                                * sizeof(*d_input));
        state.SetItemsProcessed(state.iterations() * batch_size * effective_size);
        // All threads of all blocks read the same keys so this value is limited by cache bandwidth
        state.counters["bytes_per_second_keys"] = benchmark::Counter(
            static_cast<double>(state.iterations() * batch_size * effective_size * keys_size
                                * sizeof(*d_key_input)),
            benchmark::Counter::kIsRate,
            benchmark::Counter::kIs1024);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_key_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temporary_storage));
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_HPP_
