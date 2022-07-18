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

constexpr const size_t default_alignment = 256;

/// \brief This structure describes some required partition of temporary global memory.
struct temp_storage_partition
{
    /// \brief The location of the resulting pointer to the partitioned memory.
    void** ptr;
    /// \brief The total number of bytes of global memory that should be partitioned.
    size_t size;
    /// \brief The minimum alignment that the memory should have,
    size_t alignment = default_alignment;

    /// \brief Construct a \p temp_storage_partition for some pointer,
    /// which requires some amount of memory and with some required alignment.
    /// size and alignment are in bytes.
    /// \tparam T - The element-type of the array to create a partition for
    /// \param ptr       - Pointer to where to write the pointer to the partitioned storage.
    /// \param size      - The number of elements to partition.
    /// \param alignment - The minimum required alignment of the partition.
    template<typename T>
    temp_storage_partition(T** ptr, size_t size, size_t alignment = default_alignment)
        : ptr(reinterpret_cast<void**>(ptr)), size(size), alignment(alignment)
    {}

    /// \brief Create a \p temp_storage_partition for a naturally-aligned pointer
    /// to an array of some number of element.
    /// items is in elements of T.
    /// \tparam T    - The element-type of the array to create a partition for
    /// \param  ptr  - Pointer to where to write to the pointer to the partitioned storage.
    /// \param  size - The number of elements to partition.
    template<typename T>
    static temp_storage_partition ptr_aligned_array(T** ptr, size_t items)
    {
        return temp_storage_partition(ptr, sizeof(T) * items, alignof(T));
    }
};

/// A helper function to compute the total size of temporary storage, and to break it down into
/// individual pointers. If \p temporary_storage is \p nullptr, this function will only compute
/// the total amount of temporary memory that is required and write it to \p storage_size.
/// Otherwise, this function will assign the pointers in each \p temp_storage_partition to the
/// correct offset from \p temporary_storage. If the passed storage_size is not large enough,
/// this function will return an error.
/// This function handles allocation sizes of 0. The alignment of these will not influence the
/// alignment of other allocations, and the resulting pointer of these will be set to nullptr.
/// temporary_storage is assumed to have at least the maximum alignment of all requested alignments.
/// \tparam NumberOfAllocations  - The number of partitions to allocate
/// \param temporary_storage     - The base pointer to the allocated temporary memory. May be \p nullptr.
/// \param storage_size [in,out] - The size of \p temporary_storage.
/// \param parts        [in,out] - The partitions to partition the temporary memory for.
template<size_t NumberOfAllocations>
hipError_t partition_temp_storage(void* const temporary_storage,
                                  size_t&     storage_size,
                                  const temp_storage_partition (&parts)[NumberOfAllocations])
{
    // Perform an exclusive scan over the sizes, and adjust each offset so that the alignment is correct.
    size_t offsets[NumberOfAllocations];
    offsets[0] = 0;
    for(size_t i = 1; i < NumberOfAllocations; ++i)
    {
        // If the required size of this partitioned is 0, we don't want it to influence the final pointer.
        size_t alignment = parts[i].size == 0 ? 1 : parts[i].alignment;
        offsets[i]       = align_size(offsets[i - 1] + parts[i - 1].size, alignment);
    }
    size_t required_storage_size
        = offsets[NumberOfAllocations - 1] + parts[NumberOfAllocations - 1].size;

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
        *parts[i].ptr = parts[i].size == 0 ? nullptr : static_cast<void*>(base + offsets[i]);
    }

    return hipSuccess;
}
} // namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_TEMP_STORAGE_HPP_
