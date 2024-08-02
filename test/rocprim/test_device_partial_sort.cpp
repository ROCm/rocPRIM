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

// required test headers
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_partial_sort.hpp>
#include <rocprim/functional.hpp>

#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class KeyType,
         class CompareFunction    = ::rocprim::less<KeyType>,
         class Config             = ::rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DevicePartialSortParams
{
    using key_type                              = KeyType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDevicePartialSortTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

// TODO add custom config
// TODO no graph support
using RocprimDevicePartialSortTestsParams = ::testing::Types<
    DevicePartialSortParams<unsigned short>,
    DevicePartialSortParams<char>,
    DevicePartialSortParams<int>,
    DevicePartialSortParams<test_utils::custom_test_type<int>>,
    DevicePartialSortParams<unsigned long>,
    DevicePartialSortParams<long long, ::rocprim::greater<long long>>,
    DevicePartialSortParams<float>,
    DevicePartialSortParams<int8_t>,
    DevicePartialSortParams<uint8_t>,
    DevicePartialSortParams<rocprim::half>,
    DevicePartialSortParams<rocprim::bfloat16>,
    DevicePartialSortParams<double>,
    DevicePartialSortParams<test_utils::custom_test_type<float>>,
    DevicePartialSortParams<test_utils::custom_float_type>,
    DevicePartialSortParams<test_utils::custom_test_array_type<int, 4>>,
    DevicePartialSortParams<int, ::rocprim::less<int>, rocprim::default_config, false, true>,
    DevicePartialSortParams<
        int,
        ::rocprim::less<int>,
        rocprim::partial_sort_config<
            rocprim::
                nth_element_config<128, 4, 32, 16, rocprim::block_radix_rank_algorithm::basic>>>>;

TYPED_TEST_SUITE(RocprimDevicePartialSortTests, RocprimDevicePartialSortTestsParams);

TYPED_TEST(RocprimDevicePartialSortTests, PartialSort)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                              = typename TestFixture::key_type;
    using compare_function                      = typename TestFixture::compare_function;
    using config                                = typename TestFixture::config;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    bool in_place = false;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<size_t> middles = {0};
            if(size > 0)
            {
                middles.push_back(size);
            }
            if(size > 1)
            {
                middles.push_back(test_utils::get_random_value<size_t>(1, size - 1, seed_value));
            }

            for(size_t middle : middles)
            {
                SCOPED_TRACE(testing::Message() << "with middle = " << middle);

                hipStream_t stream = 0; // default
                if(TestFixture::use_graphs)
                {
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                std::vector<key_type> input;
                if(rocprim::is_floating_point<key_type>::value)
                {
                    input = test_utils::get_random_data<key_type>(size, -1000, 1000, seed_value);
                }
                else
                {
                    input = test_utils::get_random_data<key_type>(
                        size,
                        test_utils::numeric_limits<key_type>::min(),
                        test_utils::numeric_limits<key_type>::max(),
                        seed_value);
                }
                key_type* d_input;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(key_type)));
                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    size * sizeof(key_type),
                                    hipMemcpyHostToDevice));

                key_type* d_output;
                if(in_place)
                {
                    d_output = d_input;
                }
                else
                {
                    HIP_CHECK(
                        test_common_utils::hipMallocHelper(&d_output, size * sizeof(key_type)));
                }

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);

                compare_function compare_op;

                // Allocate temporary storage
                size_t temp_storage_size_bytes{};
                if(in_place)
                {
                    HIP_CHECK(rocprim::partial_sort<config>(nullptr,
                                                            temp_storage_size_bytes,
                                                            input_it,
                                                            middle,
                                                            size,
                                                            compare_op,
                                                            stream,
                                                            debug_synchronous));
                }
                else
                {
                    HIP_CHECK(rocprim::partial_sort_copy<config>(nullptr,
                                                                 temp_storage_size_bytes,
                                                                 input_it,
                                                                 d_output,
                                                                 middle,
                                                                 size,
                                                                 compare_op,
                                                                 stream,
                                                                 debug_synchronous));
                }

                ASSERT_GT(temp_storage_size_bytes, 0);
                void* d_temp_storage{};
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                hipGraph_t graph;
                if(TestFixture::use_graphs)
                {
                    graph = test_utils::createGraphHelper(stream);
                }
                if(in_place)
                {
                    HIP_CHECK(rocprim::partial_sort<config>(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            input_it,
                                                            middle,
                                                            size,
                                                            compare_op,
                                                            stream,
                                                            debug_synchronous));
                }
                else
                {
                    HIP_CHECK(rocprim::partial_sort_copy<config>(d_temp_storage,
                                                                 temp_storage_size_bytes,
                                                                 input_it,
                                                                 d_output,
                                                                 middle,
                                                                 size,
                                                                 compare_op,
                                                                 stream,
                                                                 debug_synchronous));
                }

                HIP_CHECK(hipGetLastError());

                hipGraphExec_t graph_instance;
                if(TestFixture::use_graphs)
                {
                    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
                }

                // The algorithm sorted [first, middle). Since the order of [middle, last) is not specified,
                //   sort [middle, last) to compare with expected values.
                std::vector<key_type> output(size);
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    size * sizeof(key_type),
                                    hipMemcpyDeviceToHost));
                std::sort(output.begin() + middle, output.begin() + size, compare_op);

                // Sort input fully to compare
                std::sort(input.begin(), input.end(), compare_op);

                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, input));

                HIP_CHECK(hipFree(d_input));
                if(!in_place)
                {
                    hipFree(d_output);
                }
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    test_utils::cleanupGraphHelper(graph, graph_instance);
                    HIP_CHECK(hipStreamDestroy(stream));
                }

                in_place = !in_place;
            }
        }
    }
}
