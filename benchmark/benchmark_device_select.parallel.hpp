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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_

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
struct device_select_flag_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:flag,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",flag_type:" + std::string(Traits<FlagType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             hipStream_t         stream) const override
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
        rocprim::select<Config>(nullptr,
                                temp_storage_size_bytes,
                                d_input,
                                d_flags,
                                d_output,
                                d_selected_count_output,
                                input.size(),
                                stream);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; i++)
        {
            rocprim::select<Config>(d_temp_storage,
                                    temp_storage_size_bytes,
                                    d_input,
                                    d_flags,
                                    d_output,
                                    d_selected_count_output,
                                    input.size(),
                                    stream);
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
                rocprim::select<Config>(d_temp_storage,
                                        temp_storage_size_bytes,
                                        d_input,
                                        d_flags,
                                        d_output,
                                        d_selected_count_output,
                                        input.size(),
                                        stream);
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
struct device_select_predicate_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:predicate,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        const float           probability = Probability / float{100};
        std::vector<DataType> input
            = get_random_data<DataType>(size, DataType(0), DataType(1000), seed.get_0());

        auto select_op = [probability] __device__(const DataType& value) -> bool
        { return value < DataType(1000 * probability); };

        DataType*     d_input;
        DataType*     d_output;
        unsigned int* d_selected_count_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(DataType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(DataType)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(DataType),
                            hipMemcpyHostToDevice));

        size_t temp_storage_size_bytes{};
        rocprim::select<Config>(nullptr,
                                temp_storage_size_bytes,
                                d_input,
                                d_output,
                                d_selected_count_output,
                                input.size(),
                                select_op,
                                stream);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            rocprim::select<Config>(d_temp_storage,
                                    temp_storage_size_bytes,
                                    d_input,
                                    d_output,
                                    d_selected_count_output,
                                    input.size(),
                                    select_op,
                                    stream);
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
                rocprim::select<Config>(d_temp_storage,
                                        temp_storage_size_bytes,
                                        d_input,
                                        d_output,
                                        d_selected_count_output,
                                        input.size(),
                                        select_op,
                                        stream);
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

template<class DataType, class Config = rocprim::default_config, int Probability = 50>
struct device_select_unique_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:unique,data_type:"
                                         + std::string(Traits<DataType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using op_type = typename std::conditional<std::is_same<DataType, rocprim::half>::value,
                                                  half_plus,
                                                  rocprim::plus<DataType>>::type;
        op_type op;

        std::vector<DataType> input(size);
        {
            const float probability = Probability / float{100};
            auto        input01     = get_random_data01<DataType>(size, probability, seed.get_0());
            auto        acc         = input01[0];
            input[0]                = acc;
            for(size_t i = 1; i < input01.size(); i++)
            {
                input[i] = op(acc, input01[i]);
            }
        }
        auto equality_op = rocprim::equal_to<DataType>();

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
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        rocprim::unique<Config>(nullptr,
                                temp_storage_size_bytes,
                                d_input,
                                d_output,
                                d_selected_count_output,
                                input.size(),
                                equality_op,
                                stream);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            rocprim::unique<Config>(d_temp_storage,
                                    temp_storage_size_bytes,
                                    d_input,
                                    d_output,
                                    d_selected_count_output,
                                    input.size(),
                                    equality_op,
                                    stream);
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
                rocprim::unique<Config>(d_temp_storage,
                                        temp_storage_size_bytes,
                                        d_input,
                                        d_output,
                                        d_selected_count_output,
                                        input.size(),
                                        equality_op,
                                        stream);
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

template<class KeyType,
         class ValueType,
         class Config    = rocprim::default_config,
         int Probability = 50>
struct device_select_unique_by_key_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name("{lvl:device,algo:select,subalgo:unique_by_key,key_type:"
                                         + std::string(Traits<KeyType>::name())
                                         + ",value_type:" + std::string(Traits<ValueType>::name())
                                         + ",probability:" + std::to_string(Probability)
                                         + ",cfg:" + partition_config_name<Config>() + "}");
    }

    void run(benchmark::State&   state,
             size_t              size,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using op_type = typename std::conditional_t<std::is_same<KeyType, rocprim::half>::value,
                                                    half_plus,
                                                    rocprim::plus<KeyType>>;
        op_type op;

        std::vector<KeyType> input_keys(size);
        {
            const float probability = Probability / float{100};
            auto        input01     = get_random_data01<KeyType>(size, probability, seed.get_0());
            auto        acc         = input01[0];
            input_keys[0]           = acc;
            for(size_t i = 1; i < input01.size(); i++)
            {
                input_keys[i] = op(acc, input01[i]);
            }
        }
        const auto input_values = get_random_data<ValueType>(size, -1000, 1000, seed.get_1());
        std::vector<unsigned int> selected_count_output(1);
        auto                      equality_op = rocprim::equal_to<KeyType>();

        KeyType*      d_keys_input{};
        ValueType*    d_values_input{};
        KeyType*      d_keys_output{};
        ValueType*    d_values_output{};
        unsigned int* d_selected_count_output{};
        HIP_CHECK(hipMalloc(&d_keys_input, input_keys.size() * sizeof(input_keys[0])));
        HIP_CHECK(hipMalloc(&d_values_input, input_values.size() * sizeof(input_values[0])));
        HIP_CHECK(hipMalloc(&d_keys_output, input_keys.size() * sizeof(input_keys[0])));
        HIP_CHECK(hipMalloc(&d_values_output, input_values.size() * sizeof(input_values[0])));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(selected_count_output[0])));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            input_keys.data(),
                            input_keys.size() * sizeof(input_keys[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_values_input,
                            input_values.data(),
                            input_values.size() * sizeof(input_values[0]),
                            hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes{};
        rocprim::unique_by_key<Config>(nullptr,
                                       temp_storage_size_bytes,
                                       d_keys_input,
                                       d_values_input,
                                       d_keys_output,
                                       d_values_output,
                                       d_selected_count_output,
                                       input_keys.size(),
                                       equality_op,
                                       stream);
        void* d_temp_storage{};
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        for(int i = 0; i < warmup_iter; ++i)
        {
            rocprim::unique_by_key<Config>(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_keys_input,
                                           d_values_input,
                                           d_keys_output,
                                           d_values_output,
                                           d_selected_count_output,
                                           input_keys.size(),
                                           equality_op,
                                           stream);
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
                rocprim::unique_by_key<Config>(d_temp_storage,
                                               temp_storage_size_bytes,
                                               d_keys_input,
                                               d_values_input,
                                               d_keys_output,
                                               d_values_output,
                                               d_selected_count_output,
                                               input_keys.size(),
                                               equality_op,
                                               stream);
            }
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds{};
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(KeyType) + sizeof(ValueType)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_keys_input);
        hipFree(d_values_input);
        hipFree(d_keys_output);
        hipFree(d_values_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename Config, typename KeyType, typename ValueType>
struct create_benchmark
{
    void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        storage.emplace_back(
            std::make_unique<device_select_unique_by_key_benchmark<KeyType, ValueType, Config>>());
    }
};

template<typename Config, typename KeyType>
struct create_benchmark<Config, KeyType, rocprim::empty_type>
{
    void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        storage.emplace_back(std::make_unique<device_select_flag_benchmark<KeyType, Config>>());
        storage.emplace_back(
            std::make_unique<device_select_predicate_benchmark<KeyType, Config>>());
        storage.emplace_back(std::make_unique<device_select_unique_benchmark<KeyType, Config>>());
    }
};

template<typename KeyType, typename ValueType, int BlockSize>
struct device_select_benchmark_generator
{
    template<int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            using config = rocprim::select_config<BlockSize, ItemsPerThread>;
            create_benchmark<config, KeyType, ValueType>{}(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr int max_items_per_thread
            = std::min(64 / std::max(sizeof(KeyType), sizeof(ValueType)), size_t{32});
        static_for_each<make_index_range<int, 4, max_items_per_thread>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SELECT_PARALLEL_HPP_
