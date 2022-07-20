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
#include "../types.hpp"
#include "various.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

/// \brief The default alignment to use.
constexpr const size_t default_alignment = 256;

namespace temp_storage
{

/// \brief This value-structure describes the required layout of some piece of
/// temporary memory, which includes the required size and the required alignment.
struct layout
{
    /// \brief The required size of the temporary memory, in bytes.
    size_t size;

    /// \brief The required alignment of the temporary memory, in bytes. Defaults to \p default_alignment.
    size_t alignment = default_alignment;
};

/// \brief This structure describes a single required partition of temporary global memory, as well
/// as where to store the allocated pointer.
/// \tparam T - The base type to allocate temporary memory for.
template<typename T>
struct temp_storage_partition
{
    /// \brief The location to store the pointer to the partitioned memory.
    T** dest;
    /// \brief The required memory layout of the memory that this partition should get.
    layout storage_layout;

    /// Compute the required layout for this type and return it.
    layout get_layout()
    {
        return this->storage_layout;
    }

    /// \brief Assigns the final storage for this partition. `storage` is assumed to have the required
    /// alignment and size as described by the layout returned by `get_layout()`.
    /// \param storage - Base pointer to the storage to be used for this partition.
    void set_storage(void* const storage)
    {
        *this->dest = this->storage_layout.size == 0 ? nullptr : static_cast<T*>(storage);
    }
};

/// \brief Construct a simple `temp_storage_partition` with a particular layout.
/// \tparam T              - The base type to allocate temporary memory for
/// \param  dest           - Pointer to where to store the final allocated pointer
/// \param  storage_layout - The required layout that the memory allocated to `*dest` should have.
template<typename T>
temp_storage_partition<T> temp_storage(T** dest, layout storage_layout)
{
    return temp_storage_partition<T>{dest, storage_layout};
}

/// \brief Construct a simple `temp_storage_partition` from a size and an aligment that forms the layout.
/// \tparam T         - The base type to allocate temporary memory for
/// \param  dest      - Pointer to where to store the final allocated pointer
/// \param  size      - The required size that the memory allocated to `*dest` should have.
/// \param  alignment - The required alignment that the memory allocated to `*dest` should have.
template<typename T>
temp_storage_partition<T> temp_storage(T** dest, size_t size, size_t alignment = default_alignment)
{
    return temp_storage(dest, {size, alignment});
}

/// \brief Construct a `temp_storage_partition` for a type, given a total number of _elements_ that the allocated
/// temporary memory should consist of. The natural alignment for `T` is used.
/// \tparam T        - The base type to allocate temporary memory for
/// \param  dest     - Pointer to where to store the final allocated pointer
/// \param  elements - The number of elements of `T` that the memory allocated to `dest` should consist of.
template<typename T>
temp_storage_partition<T> ptr_aligned_array(T** dest, size_t elements)
{
    return temp_storage(dest, elements * sizeof(T), alignof(T));
}

/// \brief A partition that represents a sequence of sub-partitions. This structure can be used to
/// allocate multiple sub-partitions, each of which are sequentially allocated in order, and packed
/// such that the only padding between memory of different sub-partitions is due to required alignment.
/// \tparam Ts - The sub-partitions to allocate temporary memory for. Each should have the following member functions:
///   `layout get_layout()` - Compute the required storage layout for the partition.
///   `void set_storage(void* const storage)` - Update the internal destination pointer or the destination pointers
///     of sub-partitions with the given pointer. `storage` has at least the required size and alignment as described
///     by the result of `get_layout()`.
template<typename... Ts>
struct sequence_partition
{
    /// \brief The sub-partitions in this `sequence_partition`.
    ::rocprim::tuple<Ts...> sub_partitions;

    /// \brief Constructor.
    sequence_partition(Ts... sub_partitions) : sub_partitions{sub_partitions...} {}

    /// Compute the required layout for this type and return it.
    layout get_layout()
    {
        size_t required_alignment = 1;
        size_t required_size      = 0;

        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition)
                          {
                              const auto sub_layout = sub_partition.get_layout();

                              required_alignment
                                  = std::max(required_alignment, sub_layout.alignment);

                              if(sub_layout.size > 0)
                                  required_size = align_size(required_size, sub_layout.alignment)
                                                  + sub_layout.size;
                          });

        return {required_size, required_alignment};
    }

    /// \brief Assigns the final storage for this partition. `storage` is assumed to have the required
    /// alignment and size as described by the layout returned by `get_layout()`.
    /// \param storage - Base pointer to the storage to be used for this partition.
    void set_storage(void* const storage)
    {
        size_t offset = 0;
        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition)
                          {
                              const auto sub_layout = sub_partition.get_layout();

                              if(sub_layout.size > 0)
                                  offset = align_size(offset, sub_layout.alignment);

                              sub_partition.set_storage(
                                  static_cast<void*>(static_cast<char*>(storage) + offset));
                              offset += sub_layout.size;
                          });
    }
};

/// \brief Construct a `sequence_partition` from sub-partitions.
/// \tparam Ts - The sub-partitions to allocate temporary memory for.
/// \see sequence_partition
template<typename... Ts>
sequence_partition<Ts...> sequence(Ts... ts)
{
    return sequence_partition<Ts...>(ts...);
}

/// \brief A partition that represents a union of sub-partitions of temporary memories which are not used
/// at the same time, and for which the allocated temporary memory can be shared.
/// \tparam Ts - The sub-partitions to allocate temporary memory for. Each should have the following member functions:
///   `layout get_layout()` - Compute the required storage layout for the partition.
///   `void set_storage(void* const storage)` - Update the internal destination pointer or the destination pointers
///     of sub-partitions with the given pointer. `storage` has at least the required size and alignment as described
///     by the result of `get_layout()`.
template<typename... Ts>
struct mutually_exclusive_partition
{
    /// \brief The sub-partitions in this `mutually_exclusive_partition`.
    ::rocprim::tuple<Ts...> sub_partitions;

    /// \brief Constructor.
    mutually_exclusive_partition(Ts... sub_partitions) : sub_partitions{sub_partitions...} {}

    /// Compute the required layout for this type and return it.
    layout get_layout()
    {
        size_t required_alignment = 1;
        size_t required_size      = 0;

        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition)
                          {
                              const auto sub_layout = sub_partition.get_layout();

                              required_alignment
                                  = std::max(required_alignment, sub_layout.alignment);
                              required_size = std::max(required_size, sub_layout.size);
                          });

        return {required_size, required_alignment};
    }

    /// \brief Assigns the final storage for this partition. `storage` is assumed to have the required
    /// alignment and size as described by the layout returned by `get_layout()`.
    /// \param storage - Base pointer to the storage to be used for this partition.
    void set_storage(void* const storage)
    {
        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition) { sub_partition.set_storage(storage); });
    }
};

/// \brief Construct a `mutually_exclusive_partition` from sub-partitions.
/// \tparam Ts - The sub-partitions to allocate temporary memory for.
/// \see mutually_exclusive_partition
template<typename... Ts>
mutually_exclusive_partition<Ts...> mutually_exclusive(Ts... ts)
{
    return mutually_exclusive_partition<Ts...>(ts...);
}

/// \brief This function helps with allocating temporary global memory for device algorithms. It serves
/// both to compute the total required amount of temporary memory, as well as to break it down into individual
/// allocations.
///
/// When `temporary_storage` is `nullptr`, this function computes the total amount of required tempory memory,
/// and writes the result to `storage_size`. Note, this value will always be more than 0, even if there is technically
/// no temporary memory required.
///
/// If `temporary_storage` is not `nullptr`, this function will assign offsets relative to `temporary_storage` to the
/// `partition`, which in turn may assign offsets to any sub-partition. The required amount of temporary memory is
/// also checked against the `storage_size` that the user passed in - if it is insufficient, `hipErrorInvalidValue` is
/// returned.
///
/// This function has a special case for allocations of size 0: When any (sub-)partition in `partition` requires no
/// memory, its alignment is not factored into the total required memory, and its destination pointer will ne set to
/// `nullptr`.
///
/// \tparam TempStoragePartition - The root partition to allocate temporary memory for. It should have the following
///   member functions:
///   `layout get_layout()` - Compute the required storage layout for the partition.
///   `void set_storage(void* const storage)` - Update the internal destination pointer or the destination pointers of
///     sub-partitions with the given pointer. `storage` has at least the required size and alignment as described by
///     the result of `get_layout()`.
/// \param temporary_storage     - The base pointer to the allocated temporary memory. May be `nullptr`.
/// \param storage_size [in,out] - The size of `temporary_storage`.
/// \param partition    [in,out] - The root partition to allocate temporary memory to.
template<typename TempStoragePartition>
hipError_t
    partition(void* const temporary_storage, size_t& storage_size, TempStoragePartition partition)
{
    const auto layout = partition.get_layout();
    // Make sure the user wont try to allocate 0 bytes of memory.
    const size_t required_size = layout.size == 0 ? 4 : layout.size;

    if(temporary_storage == nullptr)
    {
        storage_size = required_size;
        return hipSuccess;
    }
    else if(storage_size < required_size)
    {
        return hipErrorInvalidValue;
    }

    partition.set_storage(temporary_storage);

    return hipSuccess;
}
} // namespace temp_storage

} // namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_TEMP_STORAGE_HPP_
