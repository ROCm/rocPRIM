// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_UTIL_PTX_HPP_
#define HIPCUB_ROCPRIM_UTIL_PTX_HPP_

#include <cstdint>
#include <type_traits>

#include "../config.hpp"

#define HIPCUB_WARP_THREADS ::rocprim::warp_size()
#define HIPCUB_ARCH 1 // ignored with rocPRIM backend

BEGIN_HIPCUB_NAMESPACE

// Missing compared to CUB:
// * ThreadExit - not supported
// * ThreadTrap - not supported
// * FFMA_RZ, FMUL_RZ - not in CUB public API
// * WARP_SYNC - not supported, not CUB public API
// * CTA_SYNC_AND - not supported, not CUB public API
// * MatchAny - not in CUB public API
//
// Differences:
// * Warp thread masks (when used) are 64-bit unsigned integers
// * member_mask argument is ignored in WARP_[ALL|ANY|BALLOT] funcs
// * Arguments first_lane, last_lane, and member_mask are ignored
// in Shuffle* funcs
// * count in BAR is ignored, BAR works like CTA_SYNC

// ID functions etc.

HIPCUB_DEVICE inline
int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (hipThreadIdx_z * block_dim_x * block_dim_y))
        + ((block_dim_y == 1) ? 0 : (hipThreadIdx_y * block_dim_x))
        + hipThreadIdx_x;
}

HIPCUB_DEVICE inline
unsigned int LaneId()
{
    return ::rocprim::lane_id();
}

HIPCUB_DEVICE inline
unsigned int WarpId()
{
    return ::rocprim::warp_id();
}

// Returns the warp lane mask of all lanes less than the calling thread
HIPCUB_DEVICE inline
uint64_t LaneMaskLt()
{
    return (uint64_t(1) << LaneId()) - 1;
}

// Returns the warp lane mask of all lanes less than or equal to the calling thread
HIPCUB_DEVICE inline
uint64_t LaneMaskLe()
{
    return ((uint64_t(1) << LaneId()) << 1) - 1;
}

// Returns the warp lane mask of all lanes greater than the calling thread
HIPCUB_DEVICE inline
uint64_t LaneMaskGt()
{
    return uint64_t(-1)^LaneMaskLe();
}

// Returns the warp lane mask of all lanes greater than or equal to the calling thread
HIPCUB_DEVICE inline
uint64_t LaneMaskGe()
{
    return uint64_t(-1)^LaneMaskLt();
}

// Shuffle funcs

template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleUp(T input,
            int src_offset,
            int first_thread,
            unsigned int member_mask)
{
    // Not supproted in rocPRIM.
    (void) first_thread;
    // Member mask is not supported in rocPRIM, because it's
    // not supported in ROCm.
    (void) member_mask;
    return ::rocprim::warp_shuffle_up(
        input, src_offset, LOGICAL_WARP_THREADS
    );
}

template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleDown(T input,
              int src_offset,
              int last_thread,
              unsigned int member_mask)
{
    // Not supproted in rocPRIM.
    (void) last_thread;
    // Member mask is not supported in rocPRIM, because it's
    // not supported in ROCm.
    (void) member_mask;
    return ::rocprim::warp_shuffle_down(
        input, src_offset, LOGICAL_WARP_THREADS
    );
}

template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleIndex(T input,
               int src_lane,
               unsigned int member_mask)
{
    // Member mask is not supported in rocPRIM, because it's
    // not supported in ROCm.
    (void) member_mask;
    return ::rocprim::warp_shuffle(
        input, src_lane, LOGICAL_WARP_THREADS
    );
}

// Other

HIPCUB_DEVICE inline
unsigned int SHR_ADD(unsigned int x,
                     unsigned int shift,
                     unsigned int addend)
{
    return (x >> shift) + addend;
}

HIPCUB_DEVICE inline
unsigned int SHL_ADD(unsigned int x,
                     unsigned int shift,
                     unsigned int addend)
{
    return (x << shift) + addend;
}

namespace detail {

template <typename UnsignedBits>
HIPCUB_DEVICE inline
auto unsigned_bit_extract(UnsignedBits source,
                          unsigned int bit_start,
                          unsigned int num_bits)
    -> typename std::enable_if<sizeof(UnsignedBits) == 8, unsigned int>::type
{
    #ifdef __HIP_PLATFORM_HCC__
        return __bitextract_u64(source, bit_start, num_bits);
    #else
        return (source << (64 - bit_start - num_bits)) >> (64 - num_bits);
    #endif // __HIP_PLATFORM_HCC__
}

template <typename UnsignedBits>
HIPCUB_DEVICE inline
auto unsigned_bit_extract(UnsignedBits source,
                          unsigned int bit_start,
                          unsigned int num_bits)
    -> typename std::enable_if<sizeof(UnsignedBits) < 8, unsigned int>::type
{
    #ifdef __HIP_PLATFORM_HCC__
        return __bitextract_u32(source, bit_start, num_bits);
    #else
        return (static_cast<unsigned int>(source) << (32 - bit_start - num_bits)) >> (32 - num_bits);
    #endif // __HIP_PLATFORM_HCC__
}

} // end namespace detail

// Bitfield-extract.
// Extracts \p num_bits from \p source starting at bit-offset \p bit_start.
// The input \p source may be an 8b, 16b, 32b, or 64b unsigned integer type.
template <typename UnsignedBits>
HIPCUB_DEVICE inline
unsigned int BFE(UnsignedBits source,
                 unsigned int bit_start,
                 unsigned int num_bits)
{
    static_assert(std::is_unsigned<UnsignedBits>::value, "UnsignedBits must be unsigned");
    return detail::unsigned_bit_extract(source, bit_start, num_bits);
}

// Bitfield insert.
// Inserts the \p num_bits least significant bits of \p y into \p x at bit-offset \p bit_start.
HIPCUB_DEVICE inline
void BFI(unsigned int &ret,
         unsigned int x,
         unsigned int y,
         unsigned int bit_start,
         unsigned int num_bits)
{
    #ifdef __HIP_PLATFORM_HCC__
        ret = __bitinsert_u32(x, y, bit_start, num_bits);
    #else
        x <<= bit_start;
        unsigned int MASK_X = ((1 << num_bits) - 1) << bit_start;
        unsigned int MASK_Y = ~MASK_X;
        ret = (y & MASK_Y) | (x & MASK_X);
    #endif // __HIP_PLATFORM_HCC__
}

HIPCUB_DEVICE inline
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
    return x + y + z;
}

HIPCUB_DEVICE inline
int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    return ::__byte_perm(a, b, index);
}

HIPCUB_DEVICE inline
void BAR(int count)
{
    (void) count;
    __syncthreads();
}

HIPCUB_DEVICE inline
void CTA_SYNC()
{
    __syncthreads();
}

HIPCUB_DEVICE inline
void WARP_SYNC(unsigned int member_mask)
{
    // Does nothing, on ROCm threads in warp are always in sync
    (void) member_mask;
}

HIPCUB_DEVICE inline
int WARP_ANY(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return ::__any(predicate);
}

HIPCUB_DEVICE inline
int WARP_ALL(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return ::__all(predicate);
}

HIPCUB_DEVICE inline
int64_t WARP_BALLOT(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return __ballot(predicate);
}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_PTX_HPP_
