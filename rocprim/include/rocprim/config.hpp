// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_CONFIG_HPP_
#define ROCPRIM_CONFIG_HPP_

#define BEGIN_ROCPRIM_NAMESPACE \
    namespace rocprim {

#define END_ROCPRIM_NAMESPACE \
    } /* rocprim */

#include <limits>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if __cplusplus < 201402L
    #error "rocPRIM requires at least C++14"
#endif

#if !defined(ROCPRIM_DEVICE) || defined(DOXYGEN_DOCUMENTATION_BUILD)
    #define ROCPRIM_DEVICE __device__
    #define ROCPRIM_HOST __host__
    #define ROCPRIM_HOST_DEVICE __host__ __device__
    #define ROCPRIM_SHARED_MEMORY __shared__
    #ifdef _WIN32
        #define ROCPRIM_KERNEL __global__ static
    #else
        #define ROCPRIM_KERNEL __global__
    #endif
    // TODO: These parameters should be tuned for NAVI in the close future.
    #ifndef ROCPRIM_DEFAULT_MAX_BLOCK_SIZE
        #define ROCPRIM_DEFAULT_MAX_BLOCK_SIZE 256
    #endif
    #ifndef ROCPRIM_DEFAULT_MIN_WARPS_PER_EU
        #define ROCPRIM_DEFAULT_MIN_WARPS_PER_EU 1
    #endif

    #ifndef DOXYGEN_DOCUMENTATION_BUILD
        // Currently HIP on Windows has a bug involving inline device functions generating
        // local memory/register allocation errors during compilation.  Current workaround is to
        // use __attribute__((always_inline)) for the affected functions
        #ifdef _WIN32
            #define ROCPRIM_INLINE inline __attribute__((always_inline))
        #else
            #define ROCPRIM_INLINE inline
        #endif
    #else
        // Prefer simpler signatures to let Sphinx/Breathe parse them
        #define ROCPRIM_FORCE_INLINE inline
        #define ROCPRIM_INLINE inline
    #endif
    #define ROCPRIM_FORCE_INLINE __attribute__((always_inline))
#endif

// DPP is supported only after Volcanic Islands (GFX8+)
// Only defined when support is present, in contrast to ROCPRIM_DETAIL_USE_DPP, which should be
// always defined
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) \
    && (!defined(__GFX6__) && !defined(__GFX7__))
    #define ROCPRIM_DETAIL_HAS_DPP 1
#endif

#if !defined(ROCPRIM_DISABLE_DPP) && defined(ROCPRIM_DETAIL_HAS_DPP)
    #define ROCPRIM_DETAIL_USE_DPP 1
#else
    #define ROCPRIM_DETAIL_USE_DPP 0
#endif

#if defined(ROCPRIM_DETAIL_HAS_DPP) && (defined(__GFX8__) || defined(__GFX9__))
    #define ROCPRIM_DETAIL_HAS_DPP_BROADCAST 1
#endif

#ifndef ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS 1
#endif

#ifndef ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS 1
#endif


// Defines targeted AMD architecture. Supported values:
// * 803 (gfx803)
// * 900 (gfx900)
// * 906 (gfx906)
// * 908 (gfx908)
// * 910 (gfx90a)
#ifndef ROCPRIM_TARGET_ARCH
    #define ROCPRIM_TARGET_ARCH 0
#endif

#ifndef ROCPRIM_NAVI
    #if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX10__) || defined(__GFX11__))
        #define ROCPRIM_NAVI 1
    #else
        #define ROCPRIM_NAVI 0
    #endif
#endif
#define ROCPRIM_ARCH_90a 910

/// Supported warp sizes
#define ROCPRIM_WARP_SIZE_32 32u
#define ROCPRIM_WARP_SIZE_64 64u
#define ROCPRIM_MAX_WARP_SIZE ROCPRIM_WARP_SIZE_64

#if (defined(_MSC_VER) && !defined(__clang__)) || (defined(__GNUC__) && !defined(__clang__))
#define ROCPRIM_UNROLL
#define ROCPRIM_NO_UNROLL
#else
#define ROCPRIM_UNROLL _Pragma("unroll")
#define ROCPRIM_NO_UNROLL _Pragma("nounroll")
#endif

#ifndef ROCPRIM_GRID_SIZE_LIMIT
#define ROCPRIM_GRID_SIZE_LIMIT std::numeric_limits<unsigned int>::max()
#endif

#if __cpp_if_constexpr >= 201606
#define ROCPRIM_IF_CONSTEXPR constexpr
#else
#define ROCPRIM_IF_CONSTEXPR
#endif

//  Copyright 2001 John Maddock.
//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_STRINGIZE(X)
#define ROCPRIM_STRINGIZE(X) ROCPRIM_DO_STRINGIZE(X)
#define ROCPRIM_DO_STRINGIZE(X) #X

//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_PRAGMA_MESSAGE("message")
//
//  Expands to the equivalent of #pragma message("message")
#if defined(__INTEL_COMPILER)
    #define ROCPRIM_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCPRIM_STRINGIZE(__LINE__) "): note: " x))
#elif defined(__GNUC__)
    #define ROCPRIM_PRAGMA_MESSAGE(x) _Pragma(ROCPRIM_STRINGIZE(message(x)))
#elif defined(_MSC_VER)
    #define ROCPRIM_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCPRIM_STRINGIZE(__LINE__) "): note: " x))
#else
    #define ROCPRIM_PRAGMA_MESSAGE(x)
#endif

/// Static asserts that only gets evaluated during device compile.
#if __HIP_DEVICE_COMPILE__
    #define ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(expression, message) \
        static_assert(expression, message)
#else
    #define ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(expression, message)
#endif

#endif // ROCPRIM_CONFIG_HPP_
