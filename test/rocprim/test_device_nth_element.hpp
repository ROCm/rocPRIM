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

#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

#include <algorithm>
#include <cinttypes>
#include <rocprim/functional.hpp>

#include <rocprim/device/device_nth_element.hpp>

#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class KeyType,
         class ValueType       = KeyType,
         class CompareFunction = ::rocprim::less<KeyType>,
         bool UseGraphs        = false>
struct DeviceNthelementParams
{
    using key_type                   = KeyType;
    using value_type                 = ValueType;
    using compare_function           = CompareFunction;
    static constexpr bool use_graphs = UseGraphs;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceNthelementTests : public ::testing::Test
{
public:
    using key_type               = typename Params::key_type;
    using value_type             = typename Params::value_type;
    using compare_function       = typename Params::compare_function;
    const bool debug_synchronous = false;
    bool       use_graphs        = Params::use_graphs;
};

using RocprimDeviceNthelementTestsParams = ::testing::Types<
    DeviceNthelementParams<unsigned short, int>,
    DeviceNthelementParams<signed char, test_utils::custom_test_type<float>>,
    DeviceNthelementParams<int>,
    DeviceNthelementParams<test_utils::custom_test_type<int>>,
    DeviceNthelementParams<unsigned long>,
    DeviceNthelementParams<long long>,
    DeviceNthelementParams<float, double>,
    DeviceNthelementParams<int8_t, int8_t>,
    DeviceNthelementParams<uint8_t, uint8_t>,
    DeviceNthelementParams<rocprim::half, rocprim::half, rocprim::less<rocprim::half>>,
    DeviceNthelementParams<rocprim::bfloat16, rocprim::bfloat16, rocprim::less<rocprim::bfloat16>>,
    DeviceNthelementParams<short, test_utils::custom_test_type<int>>,
    DeviceNthelementParams<double, test_utils::custom_test_type<double>>,
    DeviceNthelementParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<double>>,
    DeviceNthelementParams<int, test_utils::custom_float_type>,
    DeviceNthelementParams<test_utils::custom_test_array_type<int, 4>>,
    // DeviceNthelementParams<int, int, ::rocprim::less<int>, true>, // Bug with graphs
    DeviceNthelementParams<int, float, ::rocprim::greater<int>>>;

TYPED_TEST_SUITE(RocprimDeviceNthelementTests, RocprimDeviceNthelementTestsParams);

TYPED_TEST(RocprimDeviceNthelementTests, NthelementKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type               = typename TestFixture::key_type;
    using compare_function       = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            auto nth_element = size / 2;
            SCOPED_TRACE(testing::Message() << "with nth_element = " << nth_element);

            // Generate data
            std::vector<key_type> input
                = test_utils::get_random_data<key_type>(size,
                                                        -100,
                                                        100,
                                                        seed_value); // float16 can't exceed 65504
            std::vector<key_type> output(size);

            key_type* d_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(key_type)));

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                input.size() * sizeof(key_type),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // compare function
            compare_function compare_op;

            // Calculate sorted input results on host
            std::vector<key_type> sorted_input(input);
            std::stable_sort(sorted_input.begin(), sorted_input.end(), compare_op);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           nth_element,
                                           input.size(),
                                           compare_op,
                                           stream,
                                           debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            hipGraph_t graph;
            if(TestFixture::use_graphs)
            {
                graph = test_utils::createGraphHelper(stream);
            }

            // Run
            HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           nth_element,
                                           input.size(),
                                           compare_op,
                                           stream,
                                           debug_synchronous));

            hipGraphExec_t graph_instance;
            if(TestFixture::use_graphs)
            {
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(key_type),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate sorted output results on host
            std::vector<key_type> sorted_output(output);

            // Sort numbers before nth element
            std::stable_sort(sorted_output.begin(),
                             sorted_output.begin() + nth_element,
                             compare_op);

            // Sort numbers after nth element
            if(size > 0)
            {
                std::stable_sort(sorted_output.begin() + nth_element + 1,
                                 sorted_output.end(),
                                 compare_op);
            }

            // Check if the values are the same
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input));

            hipFree(d_input);
            hipFree(d_temp_storage);

            if(TestFixture::use_graphs)
            {
                test_utils::cleanupGraphHelper(graph, graph_instance);
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceNthelementTests, NthelementKeySame)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type               = typename TestFixture::key_type;
    using compare_function       = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    unsigned int seed_value = rand();
    for(size_t size : test_utils::get_sizes(seed_value))
    {
        hipStream_t stream = 0; // default
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto nth_element = size / 2;
        SCOPED_TRACE(testing::Message() << "with nth_element = " << nth_element);

        // Generate data
        std::vector<key_type> input(size);
        std::fill(input.begin(), input.end(), 8);
        std::vector<key_type> output(size);

        key_type* d_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(key_type),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // compare function
        compare_function compare_op;

        // temp storage
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       nth_element,
                                       input.size(),
                                       compare_op,
                                       stream,
                                       debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       nth_element,
                                       input.size(),
                                       compare_op,
                                       stream,
                                       debug_synchronous));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(hipMemcpy(output.data(),
                            d_input,
                            output.size() * sizeof(key_type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Check if the values are the same
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(input, output));

        hipFree(d_input);
        hipFree(d_temp_storage);
    }
}
