// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_COMMON_TEST_HEADER
#define ROCPRIM_COMMON_TEST_HEADER

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <vector>
#include <utility>
#include <random>
#include <cmath>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hip_ext.h>

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

#include <cstdlib>
#include <string>
#include <cctype>

namespace test_common_utils
{

int obtain_device_from_ctest()
{
    static const std::string rg0 = "CTEST_RESOURCE_GROUP_0";
    if (std::getenv(rg0.c_str()) != nullptr)
    {
        std::string amdgpu_target = std::getenv(rg0.c_str());
        std::transform(amdgpu_target.cbegin(), amdgpu_target.cend(), amdgpu_target.begin(), ::toupper);
        std::string reqs = std::getenv((rg0 + "_" + amdgpu_target).c_str());
        return std::atoi(reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
    }
    else
        return 0;
}

bool supports_hmm()
{
    hipDeviceProp_t device_prop;
    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&device_prop, device_id));
    printf("managed memory support: %d\n", device_prop.managedMemory);
    if (device_prop.managedMemory == 1) return true;

    return false;
}

bool use_hmm()
{
    return std::getenv("ROCPRIM_USE_HMM");
}

// Helper for HMM allocations: if device supports managedMemory, and HMM is requested through
// ROCPRIM_USE_HMM environment variable
template <class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if (supports_hmm() && use_hmm())
    {
        return hipMallocManaged((void**)devPtr, size);
    }
    else
    {
        return hipMalloc((void**)devPtr, size);
    }
    return hipSuccess;
}

}

#endif
