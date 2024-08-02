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

#ifndef ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_

#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <rocprim/rocprim.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

constexpr int warmup_iter = 5;
constexpr int batch_size  = 10;

template<class DataType,
         class Config    = rocprim::default_config,
         class FlagType  = char,
         int Probability = 50>
struct device_partition_flag_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition,subalgo:flag,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",flag_type:" + std::string(Traits<FlagType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             const hipStream_t   stream) const override
    {
        const float           probability = Probability / float{100};
        std::vector<DataType> input;
        std::vector<FlagType> flags = get_random_data01<FlagType>(size, probability, seed.get_0());
        if(std::is_floating_point<DataType>::value)
        {
            input = get_random_data<DataType>(size, DataType(-1000), DataType(1000), seed.get_1());
        }
        else
        {
            input = get_random_data<DataType>(size,
                                              std::numeric_limits<DataType>::min(),
                                              std::numeric_limits<DataType>::max(),
                                              seed.get_1());
        }

        DataType*     d_input{};
        FlagType*     d_flags{};
        DataType*     d_output{};
        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_flags), flags.size() * sizeof(FlagType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(DataType)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_flags,
                            flags.data(),
                            flags.size() * sizeof(FlagType),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        HIP_CHECK(rocprim::partition<Config>(nullptr,
                                             temp_storage_size_bytes,
                                             d_input,
                                             d_flags,
                                             d_output,
                                             d_selected_count_output,
                                             input.size(),
                                             stream));
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
                                                 temp_storage_size_bytes,
                                                 d_input,
                                                 d_flags,
                                                 d_output,
                                                 d_selected_count_output,
                                                 input.size(),
                                                 stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
                                                     temp_storage_size_bytes,
                                                     d_input,
                                                     d_flags,
                                                     d_output,
                                                     d_selected_count_output,
                                                     input.size(),
                                                     stream));
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        hipFree(d_flags);
        hipFree(d_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

template<class DataType, class Config = rocprim::default_config, int Probability = 50>
struct device_partition_predicate_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition,subalgo:predicate,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             const hipStream_t   stream) const override
    {
        float probability = Probability / float{100};
        auto  predicate   = [probability] __device__(const DataType& value) -> bool
        { return value < static_cast<DataType>(127 * probability); };

        std::vector<DataType> input
            = get_random_data<DataType>(size, DataType(0), DataType(127), seed.get_0());
        DataType*     d_input{};
        DataType*     d_output{};
        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(DataType)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        HIP_CHECK(rocprim::partition<Config>(nullptr,
                                             temp_storage_size_bytes,
                                             d_input,
                                             d_output,
                                             d_selected_count_output,
                                             input.size(),
                                             predicate,
                                             stream));
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        for(int i = 0; i < warmup_iter; ++i)
        {
            HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
                                                 temp_storage_size_bytes,
                                                 d_input,
                                                 d_output,
                                                 d_selected_count_output,
                                                 input.size(),
                                                 predicate,
                                                 stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::partition<Config>(d_temp_storage,
                                                     temp_storage_size_bytes,
                                                     d_input,
                                                     d_output,
                                                     d_selected_count_output,
                                                     input.size(),
                                                     predicate,
                                                     stream));
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

template<class DataType,
         class Config    = rocprim::default_config,
         class FlagType  = char,
         int Probability = 50>
struct device_partition_two_way_flag_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:partition_two_way,subalgo:flag,data_type:"
            + std::string(Traits<DataType>::name()) + ",flag_type:"
            + std::string(Traits<FlagType>::name()) + ",probability:" + std::to_string(Probability)
            + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             const hipStream_t   stream) const override
    {
        float                 probability = Probability / float{100};
        std::vector<DataType> input;
        std::vector<FlagType> flags = get_random_data01<FlagType>(size, probability, seed.get_0());
        if(std::is_floating_point<DataType>::value)
        {
            input = get_random_data<DataType>(size, DataType(-1000), DataType(1000), seed.get_1());
        }
        else
        {
            input = get_random_data<DataType>(size,
                                              std::numeric_limits<DataType>::min(),
                                              std::numeric_limits<DataType>::max(),
                                              seed.get_1());
        }

        DataType*     d_input{};
        FlagType*     d_flags{};
        DataType*     d_output_selected{};
        DataType*     d_output_rejected{};
        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_flags), flags.size() * sizeof(FlagType)));
        HIP_CHECK(hipMalloc(&d_output_selected, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_rejected, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_flags,
                            flags.data(),
                            flags.size() * sizeof(FlagType),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes = 0;
        HIP_CHECK(rocprim::partition_two_way<Config>(nullptr,
                                                     temp_storage_size_bytes,
                                                     d_input,
                                                     d_flags,
                                                     d_output_selected,
                                                     d_output_rejected,
                                                     d_selected_count_output,
                                                     input.size(),
                                                     stream));
        void* d_temp_storage = nullptr;
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_input,
                                                         d_flags,
                                                         d_output_selected,
                                                         d_output_rejected,
                                                         d_selected_count_output,
                                                         input.size(),
                                                         stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_input,
                                                             d_flags,
                                                             d_output_selected,
                                                             d_output_rejected,
                                                             d_selected_count_output,
                                                             input.size(),
                                                             stream));
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        hipFree(d_flags);
        hipFree(d_output_selected);
        hipFree(d_output_rejected);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

template<class DataType, class Config = rocprim::default_config, int Probability = 50>
struct device_partition_two_way_predicate_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:partition_two_way,subalgo:predicate,data_type:"
            + std::string(Traits<DataType>::name()) + ",probability:" + std::to_string(Probability)
            + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             const hipStream_t   stream) const override
    {
        float probability = Probability / float{100};
        auto  predicate   = [probability] __device__(const DataType& value)
        { return value < static_cast<DataType>(127 * probability); };

        std::vector<DataType> input
            = get_random_data<DataType>(size, DataType(0), DataType(127), seed.get_0());
        DataType*     d_input;
        DataType*     d_output_selected;
        DataType*     d_output_rejected;
        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_selected, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_rejected, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        HIP_CHECK(rocprim::partition_two_way<Config>(nullptr,
                                                     temp_storage_size_bytes,
                                                     d_input,
                                                     d_output_selected,
                                                     d_output_rejected,
                                                     d_selected_count_output,
                                                     input.size(),
                                                     predicate,
                                                     stream));
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         d_input,
                                                         d_output_selected,
                                                         d_output_rejected,
                                                         d_selected_count_output,
                                                         input.size(),
                                                         predicate,
                                                         stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::partition_two_way<Config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             d_input,
                                                             d_output_selected,
                                                             d_output_rejected,
                                                             d_selected_count_output,
                                                             input.size(),
                                                             predicate,
                                                             stream));
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        hipFree(d_output_selected);
        hipFree(d_output_rejected);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

template<class DataType,
         class Config       = rocprim::default_config,
         int ProbabilityOne = 33,
         int ProbabilityTwo = 66>
struct device_partition_three_way_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:partition_three_way,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability_one:" + std::to_string(ProbabilityOne)
                                         + ",probability_two:" + std::to_string(ProbabilityTwo)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             const hipStream_t   stream) const override
    {
        float probability_one = ProbabilityOne / float{100};
        auto  predicate_one   = [probability_one] __device__(const DataType& value)
        { return value < DataType(127 * probability_one); };
        float probability_two = ProbabilityTwo / float{100};
        auto  predicate_two   = [probability_two] __device__(const DataType& value)
        { return value < DataType(127 * probability_two); };

        std::vector<DataType> input
            = get_random_data<DataType>(size, DataType(0), DataType(127), seed.get_0());
        DataType*     d_input{};
        DataType*     d_output_first{};
        DataType*     d_output_second{};
        DataType*     d_output_unselected{};
        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_first, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_second, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_output_unselected, input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, 2 * sizeof(unsigned int)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        HIP_CHECK(rocprim::partition_three_way<Config>(nullptr,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output_first,
                                                       d_output_second,
                                                       d_output_unselected,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       predicate_one,
                                                       predicate_two,
                                                       stream));
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            HIP_CHECK(rocprim::partition_three_way<Config>(d_temp_storage,
                                                           temp_storage_size_bytes,
                                                           d_input,
                                                           d_output_first,
                                                           d_output_second,
                                                           d_output_unselected,
                                                           d_selected_count_output,
                                                           input.size(),
                                                           predicate_one,
                                                           predicate_two,
                                                           stream));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::partition_three_way<Config>(d_temp_storage,
                                                               temp_storage_size_bytes,
                                                               d_input,
                                                               d_output_first,
                                                               d_output_second,
                                                               d_output_unselected,
                                                               d_selected_count_output,
                                                               input.size(),
                                                               predicate_one,
                                                               predicate_two,
                                                               stream));
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(DataType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        hipFree(d_output_first);
        hipFree(d_output_second);
        hipFree(d_output_unselected);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename DataType, int BlockSize>
struct device_partition_benchmark_generator
{
    template<int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            using config = rocprim::select_config<BlockSize, ItemsPerThread>;
            storage.emplace_back(
                std::make_unique<device_partition_flag_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_predicate_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_two_way_flag_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_two_way_predicate_benchmark<DataType, config>>());
            storage.emplace_back(
                std::make_unique<device_partition_three_way_benchmark<DataType, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr int max_items_per_thread = std::min(64 / sizeof(DataType), size_t{32});
        static_for_each<make_index_range<int, 4, max_items_per_thread>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_PARTITION_PARALLEL_HPP_
