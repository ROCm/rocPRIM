/******************************************************************************
* Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2022, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    // Details of the Merge-Path Algorithm can be found in:
    // S. Odeh, O. Green, Z. Mwassi, O. Shmueli, Y. Birk, " Merge Path - Parallel
    // Merging Made Simple", Multithreaded Architectures and Applications (MTAAP)
    // Workshop, IEEE 26th International Parallel & Distributed Processing
    // Symposium (IPDPS), 2012
    template <typename KeyT,
              typename KeyIteratorT,
              typename OffsetT,
              typename BinaryPred>
    ROCPRIM_DEVICE
    OffsetT merge_path(KeyIteratorT keys1,
                       KeyIteratorT keys2,
                       const OffsetT keys1_count,
                       const OffsetT keys2_count,
                       const OffsetT diag,
                       BinaryPred binary_pred)
    {
        OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
        OffsetT keys1_end   = rocprim::min(diag, keys1_count);

        while (keys1_begin < keys1_end)
        {
            const OffsetT mid = midpoint<OffsetT>(keys1_begin, keys1_end);
            const KeyT key1   = keys1[mid];
            const KeyT key2   = keys2[diag - 1 - mid];
            bool pred   = binary_pred(key2, key1);

            if (pred)
            {
                keys1_end = mid;
            }
            else
            {
                keys1_begin = mid + 1;
            }
        }
        return keys1_begin;
    }

    template <typename KeyT, typename KeyIteratorT, typename CompareOp, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE
    void merge_serial(KeyIteratorT keys_shared,
                      unsigned int keys1_beg,
                      unsigned int keys2_beg,
                      const unsigned int keys1_count,
                      const unsigned int keys2_count,
                      KeyT (&output)[ItemsPerThread],
                      unsigned int (&indices)[ItemsPerThread],
                      CompareOp compare_op)
    {
        const unsigned int keys1_end = keys1_beg + keys1_count;
        const unsigned int keys2_end = keys2_beg + keys2_count;

        KeyT key1 = keys_shared[keys1_beg];
        KeyT key2 = keys_shared[keys2_beg];

        ROCPRIM_UNROLL
        for (unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            bool pred = (keys2_beg < keys2_end) &&
                        (keys1_beg >= keys1_end || compare_op(key2, key1));

            output[item]  = pred ? key2 : key1;
            indices[item] = pred ? keys2_beg++ : keys1_beg++;

            if (pred)
            {
                key2 = keys_shared[keys2_beg];
            }
            else
            {
                key1 = keys_shared[keys1_beg];
            }
        }
    }
} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_
