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

// Google Test
#include <gtest/gtest.h>

// rocPRIM HIP API
#include <rocprim/rocprim.hpp>

#include "detail/get_rocprim_version.hpp"

TEST(RocprimBasicTests, GetVersion)
{
    auto version = rocprim::version();
    ASSERT_EQ(version, ROCPRIM_VERSION);
}

// get_rocprim_version_on_device is compiled in a separate source,
// that way we can be sure that all rocPRIM functions are inline
// and there won't be any multiple definitions error
TEST(RocprimBasicTests, GetVersionOnDevice)
{
    auto version = get_rocprim_version_on_device();
    ASSERT_EQ(version, ROCPRIM_VERSION);
}
