// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_TEMP_STORAGE_HPP_
#define ROCPRIM_DETAIL_TEMP_STORAGE_HPP_

#include <cstddef>

#include "../config.hpp"
#include "various.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{
// A structure describing a request to allocate some temporary global memory.
struct temp_storage_req
{
    // The location of the resulting pointer to the allocated memory.
    void** ptr;
    // The total number of bytes of global memory that should be allocated.
    size_t size;
    // The minimum alignment that the memory should have,
    size_t alignment = 256;

    template<typename T>
    temp_storage_req(T** ptr, size_t size, size_t alignment = 256)
        : ptr(reinterpret_cast<void**>(ptr)), size(size), alignment(alignment)
    {}

    template<typename T>
    static temp_storage_req ptr_aligned_array(T** ptr, size_t items)
    {
        return temp_storage_req(ptr, sizeof(T) * items, alignof(T));
    }
};

// A helper function to compute the total size of temporary storage, and to break it down into
// individual pointers.
// If temporary_storage is nullptr, this function will just compute the total amount of
// temporary memory that is required and write it to `storage_size`.
// Otherwise, this function will assign the pointers in each temp_storage_req to the
// correct offset from temporary_storage. If the passed storage_size is not large enough,
// this function will return an error.
// This function handles allocation sizes of 0. The alignment of these will not influence the
// alignment of other allocations, and the resulting pointer of these will be set to nullptr.
// temporary_storage is assumed to have at least the maximum alignment of all requested alignments.
template<size_t NumberOfAllocations>
hipError_t alias_temp_storage(void* const temporary_storage,
                              size_t&     storage_size,
                              const temp_storage_req (&reqs)[NumberOfAllocations])
{
    // Perform an exclusive scan over the sizes, and adjust each offset so that the alignment is correct.
    size_t offsets[NumberOfAllocations];
    offsets[0] = 0;
    for(size_t i = 1; i < NumberOfAllocations; ++i)
    {
        // If the required size of this request is 0, we don't want it to influence the final pointer.
        size_t alignment = reqs[i].size == 0 ? 1 : reqs[i].alignment;
        offsets[i]       = align_size(offsets[i - 1] + reqs[i - 1].size, alignment);
    }
    size_t required_storage_size
        = offsets[NumberOfAllocations - 1] + reqs[NumberOfAllocations - 1].size;

    if(temporary_storage == nullptr)
    {
        if(required_storage_size == 0)
        {
            // Make sure the user wont try to allocate 0 bytes of memory.
            required_storage_size = 4;
        }
        storage_size = required_storage_size;
        return hipSuccess;
    }
    else if(storage_size < required_storage_size)
    {
        return hipErrorInvalidValue;
    }

    char* const base = static_cast<char*>(temporary_storage);
    for(size_t i = 0; i < NumberOfAllocations; ++i)
    {
        *reqs[i].ptr = reqs[i].size == 0 ? nullptr : static_cast<void*>(base + offsets[i]);
    }

    return hipSuccess;
}
} // namespace detail
END_ROCPRIM_NAMESPACE

#endif
