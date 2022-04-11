// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

template<class T>
__device__ T ax(const T a, const T x)
{
    return x * a;
}

template <class T>
__global__
void saxpy_kernel(const T * x, T * y, const T a, const size_t size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
    {
        y[i] += ax(a, x[i]);
    }
}

TEST(HIPTests, Saxpy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t N = 100;

    const float a = 100.0f;
    std::vector<float> x(N, 2.0f);
    std::vector<float> y(N, 1.0f);

    float * d_x;
    float * d_y;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_x, N * sizeof(float)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_y, N * sizeof(float)));
    HIP_CHECK(
        hipMemcpy(
            d_x, x.data(),
            N * sizeof(float),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_y, y.data(),
            N * sizeof(float),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(saxpy_kernel<float>),
        dim3((N + 255)/256), dim3(256), 0, 0,
        d_x, d_y, a, N
    );
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(
        hipMemcpy(
            y.data(), d_y,
            N * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    for(size_t i = 0; i < N; i++)
    {
        ASSERT_NEAR(y[i], 201.0f, 0.1f);
    }
}
