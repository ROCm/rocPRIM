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

#include "../common_test_header.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/device/device_nth_element.hpp>

#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

TEST(RocprimDeviceNthElementTests, BasicTest)
{
    using engine_type = std::default_random_engine;
    engine_type gen{std::random_device{}()};

    size_t storage_size;
    size_t size = 40000;
    std::vector<int> input(size);
    size_t nth = size/2;
    int*   d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(int)));

    test_utils::generate_random_data_n(input.begin(), size, -1000, 1000, gen);
    // int input[] = {-38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -38, -23, 68, 87, 24, -61, 73, -39, -98, 0, -40, 65, -62, -19, 45, -65, -76, 6, -66, 27, -64, -56, -77, -15, -88, 89, -30, -44, -85, -19, 89, -73, 24, 59, -25, 12, 36, -3, 12, -23, -75, 63, 24, 4, 36, 1, -68, 77, -16, 9, -71, 14, 62, 75, -95, -85, -39, 26, -99, -43, -58, 98, 54, -79, 93, -99, -49, -91, 32, -29, -36, -64, -83, -97, 79, 89, -86, 13, -52, 100, -56, 82, 33, 42, 92, 81, -73, -29, -79, -95, 89, -10, 30, -36, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, -10, -75, -7, 79, 84, -81, -6, -23, 40, -73, -36, -44, 11, 46, -93, 37, -60, -59, -78, -38, -37, -81, -8, 84, 93, -67, -15, -96, -90, -56, 48, -65, -68, -36, -46, -43, 32, 35, 91, -61, 10, -43, -42, 81, -53, 8, -22, -41, 94, -26, 45, -59, -43, -85, -82, 62, 88, -67, -36, 1, -36, -11, 38, -53, 62, -99, 77, -57, -100, -28, -77, -37, -23, -74, -9, 70, 62, -13, -87, -51, -94, -10, -32, -77, -79, 64, -85, -85, 52, 54, -29, 50, 11, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, 89, 21, 48, 83, 8, 62, 92, 73, -13, 3, 31, 0, -10, -68, 12, -61, -29, -10, -65, 69, -57, 16, -31, 69, 42, 45, -10, 0, -84, -28, -29, 17, 37, -21, -59, 42, -89, 94, -75, -65, 89, -63, 99, -14, 93, 70, 55, -77, -40, 14, -54, 7, 82, -69, 0, 60, 87, -18, -60, -33, -14, 38, -38, -75, 66, 19, 12, 74, -11, 7, 18, -81, 72, 41, -88, -57, 8, -79, 94, 23, -71, 66, 86, -77, 19, -11, -71, -71, 54, 89, 62, -22, -47, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 26, -94, -85, -55, 48, 50, 16, -63, 78, -29, 57, 95, 25, -27, -29, -42, 74, 93, -85, -68, 99, 91, -72, 100, -57, -87, 86, 45, 74, -8, -32, -28, 49, -18, 74, -25, 16, -65, 15, 23, 55, 13, -86, 81, -34, -57, 16, 1, 95, -81, -51, 42, -18, 81, -14, -96, 93, 60, 82, 49, 71, -23, -53, 51, -37, -85, 74, -7, 92, -88, -74, -72, -42, -12, 61, 26, -50, 32, 78, -38, -11, -56, -17, -93, 7, 40, -35, -69, -69, 63, 95, -40, -90, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -81, -9, -15, -46, -89, -63, -20, -99, -38, -27, -83, -36, 42, -72, 81, -40, -24, 55, -62, -48, 75, 32, 1, -15, 56, 37, 96, -7, 28, 90, -37, 4, -36, -94, -54, 58, 54, 40, 35, 34, -39, -7, 27, -87, -77, 42, 65, -91, -64, -54, 48, -78, -12, -94, 7, -77, -21, 58, 70, -85, 6, -77, -66, 6, -96, 89, 44, 43, -39, -39, 27, -74, 0, 25, -65, 55, -31, 96, -18, -30, 16, -76, -64, 85, 13, -30, -17, 97, -78, 17, -45, 74, -89, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, -36, 42, -48, -74, 89, 22, -6, 99, -78, 43, 2, 47, -97, 91, -53, 7, -40, 70, -49, -46, -39, -70, 58, 87, 58, -30, -3, 62, -30, 77, 87, -46, -69, 35, 92, -54, 40, -88, -20, 69, 17, 4, -15, -67, 96, 55, 31, -73, -40, -35, -87, -64, 91, -37, -87, -16, 26, -27, 97, -34, 9, -30, -49, -83, -76, -14, 50, 45, 80, -72, 28, 86, -6, -15, -6, -8, 65, 3, -94, 94, 91, -17, -21, 53, 1, 40, -40, 10, -13, -49, -42, -57, -49, 94, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -68, -23, -79, -36, 3, -85, -79, -99, -57, -54, 95, -63, -56, -95, -20, 6, 71, 69, 12, 52, -19, 59, -70, 77, 66, 67, 9, 16, 51, 21, 1, 4, -93, 66, -64, 35, -35, -27, -23, 65, 26, -25, 3, -27, 64, 5, 85, -66, -4, 51, 6, -22, 27, -70, -58, 59, 44, 29, 95, 100, 55, 12, 20, 52, -20, 43, -32, 87, -96, 2, -86, -31, -69, -84, 96, -98, 21, -82, -75, 99, 21, -52, 92, 74, 49, -25, -69, -92, 90, -23, -83, 19, 34, 33, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, -85, 28, -35, 56, -80, 42, 97, -27, -71, 13, 55, 4, -90, 3, 54, 96, 65, 44, -74, -50, 94, 1, -75, 87, 23, -84, -23, -86, -90, -82, -18, -53, 18, 68, 61, -23, -64, 18, 25, 62, -14, 32, -64, -50, -22, 53, 61, -44, 68, -78, 90, 23, 96, 27, -86, 25, -1, -78, 1, -72, -45, 62, -87, 12, -5, 5, 35, -98, -18, -18, -68, -86, 23, 78, 14, -23, -94, 71, 76, 80, 87, -74, 58, -57, 48, -33, 84, 57, -55, -83, -35, -20, -59, 2122870, 0, -1865260656, 32219, 11, 0, 2226071, 0};

    // std::cout << "Input: " << std::endl;
    // for(int i = 0; i < size; i++)
    // {
    //     std::cout << input[i] << ", ";
    // }
    // std::cout << std::endl;

    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(int), hipMemcpyHostToDevice));
    
    rocprim::nth_element_keys(nullptr, storage_size, d_input, nth, size);

    std::cout << "Size: " << storage_size << std::endl;

    void* d_temp_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, storage_size));

    rocprim::nth_element_keys(d_temp_storage, storage_size, d_input, nth, size);

    std::vector<int> output(size);

    HIP_CHECK(hipMemcpy(output.data(), d_input, size * sizeof(int), hipMemcpyHostToDevice));

    // std::cout << "Output: " << std::endl;
    // for(int i = 0; i < size; i++)
    // {
    //     std::cout << output[i] << ", ";
    // }
    // std::cout << std::endl;

    // Check if values did not change
    std::vector<int> sorted_in(input);
    std::stable_sort(sorted_in.begin(), sorted_in.end());
    std::vector<int> sorted_out(output);
    std::stable_sort(sorted_out.begin(), sorted_out.end());
    test_utils::assert_eq(sorted_in, sorted_out);

    // Check if nth element holds
    auto value_nth = output[nth];
    for(int i = 0; i < size; i++)
    {
        if (i < nth)
        {
            ASSERT_LE(output[i], value_nth);
        }
        if (i > nth)
        {
            ASSERT_GE(output[i], value_nth);
        }
        if (i == nth)
        {
            ASSERT_EQ(output[i], value_nth);
        }
    }

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
}