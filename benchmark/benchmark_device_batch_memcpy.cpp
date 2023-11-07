// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark/benchmark.h"
#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

#include <rocprim/rocprim.hpp>

#include <hip/hip_runtime.h>

#include <iostream>
#include <stdint.h>
#include <vector>

constexpr uint32_t warmup_size   = 5;
constexpr int32_t  max_size      = 1024 * 1024;
constexpr int32_t  wlev_min_size = rocprim::batch_memcpy_config<>::wlev_size_threshold;
constexpr int32_t  blev_min_size = rocprim::batch_memcpy_config<>::blev_size_threshold;

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_memcpy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_memcpy(
//    [&a0 , &b0 , &c0 , &d0 ], // from (note the order is still just a, b, c, d!)
//    [&a0', &b0', &c0', &d0'], // to   (order is the same as above too!)
//    [2   , 3   , 1   , 2   ]) // size
//
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │b0 │b1 │b0 │a1 │a2 │d0 │d1 │c0 │ buffer x contains buffers a, b, c, d
// └───┴───┴───┴───┴───┴───┴───┴───┘ note that the order of buffers is shuffled!
//  ───┬─── ─────┬───── ───┬─── ───
//     └─────────┼─────────┼───┐
//           ┌───┘     ┌───┘   │ what batch_memcpy does
//           ▼         ▼       ▼
//  ─── ─────────── ─────── ───────
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │c0'│a0'│a1'│a2'│d0'│d1'│b0'│b1'│ buffer y contains buffers a', b', c', d'
// └───┴───┴───┴───┴───┴───┴───┴───┘
template<class T, class S, class RandomGenerator>
std::vector<T> shuffled_exclusive_scan(const std::vector<S>& input, RandomGenerator& rng)
{
    const auto n = input.size();
    assert(n > 0);

    std::vector<T> result(n);
    std::vector<T> permute(n);

    std::iota(permute.begin(), permute.end(), 0);
    std::shuffle(permute.begin(), permute.end(), rng);

    for(T i = 0, sum = 0; i < n; ++i)
    {
        result[permute[i]] = sum;
        sum += input[permute[i]];
    }

    return result;
}

template<class ValueType, class BufferSizeType>
void run_benchmark(benchmark::State& state,
                   hipStream_t       stream,
                   const int32_t     num_tlev_buffers = 1024,
                   const int32_t     num_wlev_buffers = 1024,
                   const int32_t     num_blev_buffers = 1024)
{
    const bool    shuffle_buffers = false;
    const int32_t num_buffers     = num_tlev_buffers + num_wlev_buffers + num_blev_buffers;

    using offset_type = size_t;

    // Generate data

    // Number of elements in each buffer.
    std::vector<BufferSizeType> h_buffer_num_elements(num_buffers);

    // Total number of bytes.
    offset_type total_num_bytes    = 0;
    offset_type total_num_elements = 0;

    std::default_random_engine rng{0};

    // Get random buffer sizes

    for(BufferSizeType i = 0; i < num_buffers; ++i)
    {
        BufferSizeType size;
        if(i < num_tlev_buffers)
        {
            size = get_random_value<BufferSizeType>(1, wlev_min_size - 1);
        }
        else if(i < num_wlev_buffers)
        {
            size = get_random_value<BufferSizeType>(wlev_min_size, blev_min_size - 1);
        }
        else
        {
            size = get_random_value<BufferSizeType>(blev_min_size, max_size);
        }

        // convert from number of bytes to number of elements
        size = max(1, size / sizeof(ValueType));

        h_buffer_num_elements[i] = size;
        total_num_elements += size;
    }

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // Get the byte size of each buffer
    std::vector<BufferSizeType> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(ValueType);
    }

    // And the total byte size
    total_num_bytes = total_num_elements * sizeof(ValueType);

    // Device pointers
    ValueType*      d_input{};
    ValueType*      d_output{};
    ValueType**     d_buffer_srcs{};
    ValueType**     d_buffer_dsts{};
    BufferSizeType* d_buffer_sizes{};

    size_t temp_storage_bytes = 0;
    HIP_CHECK(rocprim::batch_memcpy(nullptr,
                                    temp_storage_bytes,
                                    d_buffer_srcs,
                                    d_buffer_dsts,
                                    d_buffer_sizes,
                                    num_buffers));

    void* d_temp_storage{};

    // Generate data.
    std::vector<char> h_input = get_random_data<char>(total_num_bytes,
                                                      std::numeric_limits<char>::min(),
                                                      std::numeric_limits<char>::max(),
                                                      rng());

    HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
    HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

    HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(BufferSizeType)));

    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Generate the source and shuffled destination offsets.
    std::vector<offset_type> src_offsets;
    std::vector<offset_type> dst_offsets;

    if(shuffle_buffers)
    {
        src_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
    }
    else
    {
        src_offsets = std::vector<offset_type>(num_buffers);
        dst_offsets = std::vector<offset_type>(num_buffers);

        // Consecutive offsets (no shuffling).
        // src/dst offsets first element is 0, so skip that!
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         src_offsets.begin() + 1);
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         dst_offsets.begin() + 1);
    }

    // Generate the source and destination pointers.
    std::vector<ValueType*> h_buffer_srcs(num_buffers);
    std::vector<ValueType*> h_buffer_dsts(num_buffers);

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = d_input + src_offsets[i];
        h_buffer_dsts[i] = d_output + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), total_num_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(BufferSizeType),
                        hipMemcpyHostToDevice));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                        temp_storage_bytes,
                                        d_buffer_srcs,
                                        d_buffer_dsts,
                                        d_buffer_sizes,
                                        num_buffers,
                                        hipStreamDefault));
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

        HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                        temp_storage_bytes,
                                        d_buffer_srcs,
                                        d_buffer_dsts,
                                        d_buffer_sizes,
                                        num_buffers,
                                        hipStreamDefault));

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }
    state.SetBytesProcessed(state.iterations() * total_num_bytes);
    state.SetItemsProcessed(state.iterations() * total_num_elements);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_buffer_dsts));
    HIP_CHECK(hipFree(d_buffer_srcs));
    HIP_CHECK(hipFree(d_buffer_sizes));
}

// Naive implementation used for comparison
#define BENCHMARK_BATCH_MEMCPY_NAIVE
#ifdef BENCHMARK_BATCH_MEMCPY_NAIVE

template<typename OffsetType, int32_t BlockSize>
__launch_bounds__(BlockSize) __global__
    void naive_kernel(void** in_ptr, void** out_ptr, const OffsetType* sizes)
{
    using underlying_type              = unsigned char;
    constexpr int32_t items_per_thread = 4;
    constexpr int32_t tile_size        = items_per_thread * BlockSize;

    const int32_t buffer_id = rocprim::flat_block_id();
    auto          in        = reinterpret_cast<underlying_type*>(in_ptr[buffer_id]);
    auto          out       = reinterpret_cast<underlying_type*>(out_ptr[buffer_id]);

    const auto size             = sizes[buffer_id];
    const auto size_in_elements = size / sizeof(underlying_type);
    const auto tiles            = size_in_elements / tile_size;

    auto num_items_to_copy = size;

    for(int32_t i = 0; i < tiles; ++i)
    {
        underlying_type data[items_per_thread];
        rocprim::block_load_direct_blocked(rocprim::flat_block_thread_id(),
                                           in,
                                           data,
                                           num_items_to_copy);
        rocprim::block_store_direct_blocked(rocprim::flat_block_thread_id(),
                                            out,
                                            data,
                                            num_items_to_copy);

        in += tile_size;
        out += tile_size;
        num_items_to_copy -= tile_size;
    }
}

template<class ValueType, class BufferSizeType>
void run_naive_benchmark(benchmark::State& state,
                         hipStream_t       stream,
                         const int32_t     num_tlev_buffers = 1024,
                         const int32_t     num_wlev_buffers = 1024,
                         const int32_t     num_blev_buffers = 1024)
{
    const bool    shuffle_buffers = false;
    const int32_t num_buffers     = num_tlev_buffers + num_wlev_buffers + num_blev_buffers;

    using offset_type = size_t;

    // Generate data

    // Number of elements in each buffer.
    std::vector<BufferSizeType> h_buffer_num_elements(num_buffers);

    // Total number of bytes.
    offset_type total_num_bytes    = 0;
    offset_type total_num_elements = 0;

    std::default_random_engine rng{0};

    // Get random buffer sizes

    for(BufferSizeType i = 0; i < num_buffers; ++i)
    {
        BufferSizeType size;
        if(i < num_tlev_buffers)
        {
            size = get_random_value<BufferSizeType>(1, wlev_min_size - 1);
        }
        else if(i < num_wlev_buffers)
        {
            size = get_random_value<BufferSizeType>(wlev_min_size, blev_min_size - 1);
        }
        else
        {
            size = get_random_value<BufferSizeType>(blev_min_size, max_size);
        }

        // convert from number of bytes to number of elements
        size = max(1, size / sizeof(ValueType));

        h_buffer_num_elements[i] = size;
        total_num_elements += size;
    }

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // Get the byte size of each buffer
    std::vector<BufferSizeType> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(ValueType);
    }

    // And the total byte size
    total_num_bytes = total_num_elements * sizeof(ValueType);

    // Device pointers
    ValueType*      d_input{};
    ValueType*      d_output{};
    ValueType**     d_buffer_srcs{};
    ValueType**     d_buffer_dsts{};
    BufferSizeType* d_buffer_sizes{};

    // Generate data.
    std::vector<char> h_input = get_random_data<char>(total_num_bytes,
                                                      std::numeric_limits<char>::min(),
                                                      std::numeric_limits<char>::max(),
                                                      rng());

    // Allocate memory.
    HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
    HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

    HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(ValueType*)));
    HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(BufferSizeType)));

    // Generate the source and shuffled destination offsets.
    std::vector<offset_type> src_offsets;
    std::vector<offset_type> dst_offsets;

    if(shuffle_buffers)
    {
        src_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<offset_type>(h_buffer_num_elements, rng);
    }
    else
    {
        src_offsets = std::vector<offset_type>(num_buffers);
        dst_offsets = std::vector<offset_type>(num_buffers);

        // Consecutive offsets (no shuffling).
        // src/dst offsets first element is 0, so skip that!
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         src_offsets.begin() + 1);
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         dst_offsets.begin() + 1);
    }

    // Generate the source and destination pointers.
    std::vector<ValueType*> h_buffer_srcs(num_buffers);
    std::vector<ValueType*> h_buffer_dsts(num_buffers);

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = d_input + src_offsets[i];
        h_buffer_dsts[i] = d_output + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), total_num_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(ValueType*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(BufferSizeType),
                        hipMemcpyHostToDevice));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        naive_kernel<BufferSizeType, 256><<<num_buffers, 256, 0, stream>>>((void**)d_buffer_srcs,
                                                                           (void**)d_buffer_dsts,
                                                                           d_buffer_sizes);
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

        naive_kernel<BufferSizeType, 256><<<num_buffers, 256, 0, stream>>>((void**)d_buffer_srcs,
                                                                           (void**)d_buffer_dsts,
                                                                           d_buffer_sizes);

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }
    state.SetBytesProcessed(state.iterations() * total_num_bytes);
    state.SetItemsProcessed(state.iterations() * total_num_elements);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_buffer_dsts));
    HIP_CHECK(hipFree(d_buffer_srcs));
    HIP_CHECK(hipFree(d_buffer_sizes));
}

    #define CREATE_NAIVE_BENCHMARK(item_size,                                                   \
                                   item_alignment,                                              \
                                   size_type,                                                   \
                                   num_tlev,                                                    \
                                   num_wlev,                                                    \
                                   num_blev)                                                    \
        benchmark::RegisterBenchmark(                                                           \
            bench_naming::format_name(                                                          \
                "{lvl:device,item_size:" #item_size ",item_alignment:" #item_alignment          \
                ",size_type:" #size_type ",algo:naive_memcpy,num_tlev:" #num_tlev               \
                ",num_wlev:" #num_wlev ",num_blev:" #num_blev ",cfg:default_config}")           \
                .c_str(),                                                                       \
            [=](benchmark::State& state)                                                        \
            {                                                                                   \
                run_naive_benchmark<custom_aligned_type<item_size, item_alignment>, size_type>( \
                    state,                                                                      \
                    stream,                                                                     \
                    num_tlev,                                                                   \
                    num_wlev,                                                                   \
                    num_blev);                                                                  \
            })

#endif

#define CREATE_BENCHMARK(item_size, item_alignment, size_type, num_tlev, num_wlev, num_blev)      \
    benchmark::RegisterBenchmark(                                                                 \
        bench_naming::format_name("{lvl:device,item_size:" #item_size                             \
                                  ",item_alignment:" #item_alignment ",size_type:" #size_type     \
                                  ",algo:batch_memcpy,num_tlev:" #num_tlev ",num_wlev:" #num_wlev \
                                  ",num_blev:" #num_blev ",cfg:default_config}")                  \
            .c_str(),                                                                             \
        [=](benchmark::State& state)                                                              \
        {                                                                                         \
            run_benchmark<custom_aligned_type<item_size, item_alignment>, size_type>(state,       \
                                                                                     stream,      \
                                                                                     num_tlev,    \
                                                                                     num_wlev,    \
                                                                                     num_blev);   \
        })

#ifndef BENCHMARK_BATCH_MEMCPY_NAIVE
    #define BENCHMARK_TYPE(item_size, item_alignment)                            \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0),     \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0), \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000),   \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)
#else
    #define BENCHMARK_TYPE(item_size, item_alignment)                                  \
        CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0),           \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0),       \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000),         \
            CREATE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000),   \
            CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0), \
            CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0), \
            CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000),   \
            CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)
#endif

int32_t main(int32_t argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", 1024, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");

    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t  size   = parser.get<size_t>("size");
    const int32_t trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = hipStreamDefault; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;

    benchmarks = {BENCHMARK_TYPE(1, 1),
                  BENCHMARK_TYPE(1, 2),
                  BENCHMARK_TYPE(1, 4),
                  BENCHMARK_TYPE(1, 8),
                  BENCHMARK_TYPE(2, 2),
                  BENCHMARK_TYPE(4, 4),
                  BENCHMARK_TYPE(8, 8)};

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
