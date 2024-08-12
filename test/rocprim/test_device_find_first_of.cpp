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
// #include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_find_first_of.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class Type,
         class KeyType            = Type,
         class CompareFunction    = rocprim::equal_to<Type>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceFindFirstOfParams
{
    using type                                  = Type;
    using key_type                              = KeyType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

// std::find_first_of is available since C++17
template<class InputIt, class ForwardIt, class BinaryPred>
InputIt
    find_first_of_(InputIt first, InputIt last, ForwardIt s_first, ForwardIt s_last, BinaryPred p)
{
    for(; first != last; ++first)
    {
        for(ForwardIt it = s_first; it != s_last; ++it)
        {
            if(p(*first, *it))
            {
                return first;
            }
        }
    }
    return last;
}

template<class Params>
class RocprimDeviceFindFirstOfTests : public ::testing::Test
{
public:
    using type                                  = typename Params::type;
    using key_type                              = typename Params::key_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceFindFirstOfTestsParams = ::testing::Types<DeviceFindFirstOfParams<int>,
                                                             DeviceFindFirstOfParams<uint8_t>,
                                                             DeviceFindFirstOfParams<size_t>>;

TYPED_TEST_SUITE(RocprimDeviceFindFirstOfTests, RocprimDeviceFindFirstOfTestsParams);

TYPED_TEST(RocprimDeviceFindFirstOfTests, FindFirstOf)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type                           = typename TestFixture::type;
    using key_type                       = typename TestFixture::key_type;
    using output_type                    = size_t;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        std::cout << "with seed = " << seed_value << std::endl;

        for(size_t size : test_utils::get_sizes(seed_value))
        {

            const size_t keys_size
                = std::sqrt(test_utils::get_random_value<size_t>(0, size, seed_value));

            for(double starting_point : {0.0, 0.2, 0.8, 1.0})
            {
                std::cout << "with size = " << size << ", keys_size = " << keys_size
                          << ", starting_point = " << starting_point << std::endl;

                hipStream_t stream = 0; // default
                if(TestFixture::use_graphs)
                {
                    // Default stream does not support hipGraph stream capture, so create one
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                SCOPED_TRACE(testing::Message() << "with size = " << size);

                // Generate data
                std::vector<type>     input;
                std::vector<key_type> key_input;
                if ROCPRIM_IF_CONSTEXPR(rocprim::is_floating_point<key_type>::value)
                {
                    // input = test_utils::get_random_data<type>(size, -1000, 1000, seed_value);
                    // key_input
                    //     = test_utils::get_random_data<key_type>(keys_size, -1000, 1000, seed_value);
                }
                else
                {
                    key_input = test_utils::get_random_data<key_type>(
                        keys_size,
                        0,
                        10,
                        // test_utils::numeric_limits<key_type>::min(),
                        // test_utils::numeric_limits<key_type>::max(),
                        seed_value);
                    input = test_utils::get_random_data<type>(
                        size,
                        0,
                        1000,
                        //   test_utils::numeric_limits<type>::min(),
                        //   test_utils::numeric_limits<type>::max(),
                        seed_value);

                    if(size > 0 && keys_size > 0)
                    {
                        // Change the input range before starting_point to ensure that it does not contain
                        // any values from keys
                        auto minmax_key_input
                            = std::minmax_element(key_input.begin(), key_input.end());
                        const auto min_key_input = *minmax_key_input.first;
                        const auto max_key_input = *minmax_key_input.second;
                        const auto max_input
                            = *std::minmax_element(input.begin(), input.end()).second;
                        // std::cout << "min_key_input = " << min_key_input << ", max_key_input = " << max_key_input << ", max_input = " << max_input << std::endl;
                        for(size_t i = 0; i < size * starting_point; ++i)
                        {
                            if(min_key_input <= input[i] && input[i] <= max_key_input)
                            {
                                input[i] = max_input;
                            }
                        }
                    }
                }

                type*        d_input;
                key_type*    d_key_input;
                output_type* d_output;
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_key_input,
                                                       key_input.size() * sizeof(*d_key_input)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));

                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    input.size() * sizeof(*d_input),
                                    hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_key_input,
                                    key_input.data(),
                                    key_input.size() * sizeof(*d_key_input),
                                    hipMemcpyHostToDevice));

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);
                const auto key_input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_key_input);

                // compare function
                compare_function compare_op;

                // temp storage
                size_t temp_storage_size_bytes;
                void*  d_temp_storage = nullptr;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         input_it,
                                                         key_input_it,
                                                         d_output,
                                                         input.size(),
                                                         key_input.size(),
                                                         compare_op,
                                                         stream,
                                                         debug_synchronous));

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                hipGraph_t graph;
                if(TestFixture::use_graphs)
                {
                    graph = test_utils::createGraphHelper(stream);
                }

                // Run
                HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         input_it,
                                                         key_input_it,
                                                         d_output,
                                                         input.size(),
                                                         key_input.size(),
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

                output_type output;

                // Copy output to host
                HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));

                // Check
                auto expected = find_first_of_(input.begin(),
                                               input.end(),
                                               key_input.begin(),
                                               key_input.end(),
                                               compare_op)
                                - input.begin();

                std::cout << "expected = " << expected << ", output = " << output << ", "
                          << double(expected) / double(size) << std::endl;

                ASSERT_EQ(output, expected);

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_key_input));
                HIP_CHECK(hipFree(d_output));
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    test_utils::cleanupGraphHelper(graph, graph_instance);
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}
