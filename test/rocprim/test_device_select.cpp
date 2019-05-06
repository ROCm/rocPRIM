// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class FlagType = unsigned int,
    bool UseIdentityIterator = false
>
struct DeviceSelectParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceSelectTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DeviceSelectParams<int, long>,
    DeviceSelectParams<unsigned char, float, int, true>,
    DeviceSelectParams<double, double, int, true>,
    DeviceSelectParams<test_utils::custom_test_type<double>, test_utils::custom_test_type<double>, int, true>
> RocprimDeviceSelectTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        2, 32, 64, 256,
        1024, 2048,
        3072, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceSelectTests, RocprimDeviceSelectTestsParams);

TYPED_TEST(RocprimDeviceSelectTests, Flagged)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);
        std::vector<F> flags = test_utils::get_random_data<F>(size, 0, 1);

        T * d_input;
        F * d_flags;
        U * d_output;
        unsigned int * d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(F)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_flags, flags.data(),
                flags.size() * sizeof(F),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<U> expected;
        expected.reserve(input.size());
        for(size_t i = 0; i < input.size(); i++)
        {
            if(flags[i] != 0)
            {
                expected.push_back(input[i]);
            }
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::select(
                nullptr,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void * d_temp_storage = nullptr;
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        unsigned int selected_count_output = 0;
        HIP_CHECK(
            hipMemcpy(
                &selected_count_output, d_selected_count_output,
                sizeof(unsigned int),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());
        ASSERT_EQ(selected_count_output, expected.size());

        // Check if output values are as expected
        std::vector<U> output(input.size());
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(U),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());
        for(size_t i = 0; i < expected.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }

        hipFree(d_input);
        hipFree(d_flags);
        hipFree(d_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(RocprimDeviceSelectTests, SelectOp)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    auto select_op = [] __host__ __device__ (const T& value) -> bool
        {
            if(value < T(50)) return true;
            return false;
        };

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100);

        T * d_input;
        U * d_output;
        unsigned int * d_selected_count_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
        HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<U> expected;
        expected.reserve(input.size());
        for(size_t i = 0; i < input.size(); i++)
        {
            if(select_op(input[i]))
            {
                expected.push_back(input[i]);
            }
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::select(
                nullptr,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                select_op,
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void * d_temp_storage = nullptr;
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                select_op,
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        unsigned int selected_count_output = 0;
        HIP_CHECK(
            hipMemcpy(
                &selected_count_output, d_selected_count_output,
                sizeof(unsigned int),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());
        ASSERT_EQ(selected_count_output, expected.size());

        // Check if output values are as expected
        std::vector<U> output(input.size());
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(U),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());
        for(size_t i = 0; i < expected.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
}

std::vector<float> get_discontinuity_probabilities()
{
    std::vector<float> probabilities = {
        0.05, 0.25, 0.5, 0.75, 0.95, 1
    };
    return probabilities;
}

TYPED_TEST(RocprimDeviceSelectTests, UniqueEmptyInput)
{
    using T = typename TestFixture::input_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    // Allocate and copy to device
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));

    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::unique(
            nullptr,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(123),
            rocprim::make_discard_iterator(),
            d_selected_count_output,
            0,
            ::rocprim::equal_to<T>(),
            stream,
            debug_synchronous
        )
    );

    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::unique(
            d_temp_storage,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(123),
            rocprim::make_discard_iterator(),
            d_selected_count_output,
            0,
            ::rocprim::equal_to<T>(),
            stream,
            debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Check if number of selected value is 0
    unsigned int selected_count_output = 0;
    HIP_CHECK(
        hipMemcpy(
            &selected_count_output, d_selected_count_output,
            sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    ASSERT_EQ(selected_count_output, 0);

    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

TYPED_TEST(RocprimDeviceSelectTests, Unique)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const auto sizes = get_sizes();
    const auto probabilities = get_discontinuity_probabilities();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto p : probabilities)
        {
            SCOPED_TRACE(testing::Message() << "with p = " << p);

            // Generate data
            std::vector<T> input(size);
            {
                std::vector<T> input01 = test_utils::get_random_data01<T>(size, p);
                test_utils::host_inclusive_scan(
                    input01.begin(), input01.end(), input.begin(), rocprim::plus<T>()
                );
            }

            // Allocate and copy to device
            T * d_input;
            U * d_output;
            unsigned int * d_selected_count_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            expected.push_back(input[0]);
            for(size_t i = 1; i < input.size(); i++)
            {
                if(!(input[i-1] == input[i]))
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::unique(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    ::rocprim::equal_to<T>(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::unique(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    ::rocprim::equal_to<T>(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(
                hipMemcpy(
                    &selected_count_output, d_selected_count_output,
                    sizeof(unsigned int),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            for(size_t i = 0; i < expected.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
            }

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }
}
