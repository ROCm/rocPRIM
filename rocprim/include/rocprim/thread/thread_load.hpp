/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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

enum cache_load_modifier : int
{
    load_default,       ///< Default (no modifier)
    load_ca,            ///< Cache at all levels
    load_cg,            ///< Cache at global level
    load_cs,            ///< Cache streaming (likely to be accessed once)
    load_cv,            ///< Cache as volatile (including cached system lines)
    load_ldg,           ///< Cache as texture
    load_volatile,      ///< Volatile (any memory space)
};

/// \brief Store data using the default load instruction. No support for cache modified stores yet
/// \tparam MODIFIER        - Value in enum for determine which type of cache store modifier to be used
/// \tparam InputIteratorT - Type of Output Iterator
/// \param itr [in]         - Iterator to location where data is to be stored
/// \param val [in]         - Data to be stored
/// \return Data that is loaded from memory
template <
    cache_load_modifier MODIFIER = load_default,
    typename InputIteratorT>
ROCPRIM_DEVICE inline
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
ROCPRIM_DEVICE inline
T thread_load(T* ptr)
{
    T retval;
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

END_ROCPRIM_NAMESPACE

#endif
