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


#include "test_utils_data_generation.hpp"
#include <cstddef>
#include <rocprim/device/device_nth_element.hpp>

#include <iostream>

TEST(RocprimDeviceNthElementTests, BasicTest)
{
    using engine_type = std::default_random_engine;
    engine_type gen{std::random_device{}()};

    size_t storage_size;
    size_t size = 8;
    int    input[size];
    size_t nth = 4;
    int*   d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(int)));

    test_utils::generate_random_data_n(input, size, -10, 10, gen);

    for(int i = 0; i < size; i++)
    {
        std::cout << input[i] << ' ';
    }
    std::cout << std::endl;
    HIP_CHECK(hipMemcpy(d_input, input, size * sizeof(int), hipMemcpyHostToDevice));

    rocprim::nth_element_keys(nullptr, storage_size, d_input, nth, size);

    std::cout << "Size: " << storage_size << std::endl;
    void* d_temp_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, storage_size));

    rocprim::nth_element_keys(d_temp_storage, storage_size, d_input, nth, size);

    int output[size];

    HIP_CHECK(hipMemcpy(output, d_input, size * sizeof(int), hipMemcpyHostToDevice));

    for(int i = 0; i < size; i++)
    {
        std::cout << output[i] << ' ';
    }
    std::cout << std::endl;

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
}