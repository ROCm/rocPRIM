/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2024, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define ROCPRIM_THREAD_THREAD_LOAD_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief These enum values are used to specify caching behaviour on load
enum cache_load_modifier : int
{
    load_default,   ///< Default (no modifier)
    load_ca,        ///< Cache at all levels
    load_cg,        ///< Cache at global level
    load_cs,        ///< Cache streaming (likely to be accessed once)
    load_cv,        ///< Cache as volatile (including cached system lines)
    load_ldg,       ///< Cache as texture
    load_volatile,  ///< Volatile (any memory space)
};

namespace detail
{

template<cache_load_modifier MODIFIER = load_default, typename T>
ROCPRIM_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval{};
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

#if ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS == 1

// Important for syncing. Check section 9.2.2 or 7.3 in the following document
// http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
#define ROCPRIM_ASM_THREAD_LOAD(cache_modifier,                                        \
                               llvm_cache_modifier,                                    \
                               type,                                                   \
                               interim_type,                                           \
                               asm_operator,                                           \
                               output_modifier,                                        \
                               wait_inst,                                              \
                               wait_cmd)                                               \
    template<>                                                                         \
    ROCPRIM_DEVICE __forceinline__ type AsmThreadLoad<cache_modifier, type>(void* ptr) \
    {                                                                                  \
        interim_type retval;                                                           \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier "\n\t"               \
                                   wait_inst wait_cmd "(%2)"                           \
                     : "=" #output_modifier(retval)                                    \
                     : "v"(ptr), "I"(0x00));                                           \
        return retval;                                                                 \
    }

// TODO Add specialization for custom larger data types
#define ROCPRIM_ASM_THREAD_LOAD_GROUP(cache_modifier, llvm_cache_modifier, wait_inst, wait_cmd)                                  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_load_sbyte, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_load_sshort, v, wait_inst, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_load_ubyte, v, wait_inst, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_load_ushort, v, wait_inst, wait_cmd);  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_load_dword, v, wait_inst, wait_cmd);   \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_load_dword, v, wait_inst, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_load_dwordx2, v, wait_inst, wait_cmd); \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_load_dwordx2, v, wait_inst, wait_cmd);

// [HIP-CPU] MSVC: erronous inline assembly specification (Triggers error C2059: syntax error: 'volatile')
#ifndef __HIP_CPU_RT__
#if defined(__gfx940__) || defined(__gfx941__)
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ca, "sc0", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "sc1", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cv, "sc0 sc1", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_volatile, "sc0 sc1", "s_waitcnt", "vmcnt");
#elif defined(__gfx942__)
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ca, "sc0", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "sc0 nt", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cv, "sc0", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_volatile, "sc0", "s_waitcnt", "vmcnt");
#elif defined(__gfx1200__) ||  defined(__gfx1201__)
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ca, "scope:SCOPE_DEV", "s_wait_loadcnt_dscnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "th:TH_DEFAULT scope:SCOPE_DEV", "s_wait_loadcnt_dscnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cv, "th:TH_DEFAULT scope:SCOPE_DEV", "s_wait_loadcnt_dscnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_volatile, "th:TH_DEFAULT scope:SCOPE_DEV", "s_wait_loadcnt_dscnt", "");
#else
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ca, "glc", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "glc slc", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cv, "glc", "s_waitcnt", "vmcnt");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_volatile, "glc", "s_waitcnt", "vmcnt");
#endif

// TODO find correct modifiers to match these
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ldg, "", "s_waitcnt", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cs, "", "s_waitcnt", "");
#endif // __HIP_CPU_RT__

#endif

}

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER        - Value in enum for determine which type of cache store modifier to be used
/// \tparam InputIteratorT - Type of Output Iterator
/// \param itr [in]         - Iterator to location where data is to be stored
/// \return Data that is loaded from memory
template <
    cache_load_modifier MODIFIER = load_default,
    typename InputIteratorT>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::iterator_traits<InputIteratorT>::value_type
thread_load(InputIteratorT itr)
{
    using T = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = thread_load<MODIFIER>(&(*itr));
    return *itr;
}

/// \brief Load data using the default load instruction. No support for cache modified loads yet
/// \tparam MODIFIER        - Value in enum for determine which type of cache store modifier to be used
/// \tparam T               - Type of Data to be loaded
/// \param ptr [in] - Pointer to data to be loaded
/// \return Data that is loaded from memory
template <
    cache_load_modifier MODIFIER = load_default,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T thread_load(T* ptr)
{
#ifndef __HIP_CPU_RT__
    return detail::AsmThreadLoad<MODIFIER, T>(ptr);
#else
    T retval;
    std::memcpy(&retval, ptr, sizeof(T));
    return retval;
#endif
}

END_ROCPRIM_NAMESPACE

#endif
