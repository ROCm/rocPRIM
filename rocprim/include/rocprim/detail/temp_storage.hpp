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

constexpr const size_t default_alignment = 256;

namespace temp_storage
{
struct layout
{
    size_t size;
    size_t alignment = default_alignment;
};

template<typename T>
struct temp_storage_partition
{
    T**    dest;
    layout storage_layout;

    layout get_layout()
    {
        return this->storage_layout;
    }

    void set_storage(void* const storage)
    {
        *this->dest = this->storage_layout.size == 0 ? nullptr : static_cast<T*>(storage);
    }
};

template<typename T>
temp_storage_partition<T> temp_storage(T** dest, layout storage_layout)
{
    return temp_storage_partition<T>{dest, storage_layout};
}

template<typename T>
temp_storage_partition<T> temp_storage(T** dest, size_t size, size_t alignment = default_alignment)
{
    return temp_storage(dest, {size, alignment});
}

template<typename T>
temp_storage_partition<T> ptr_aligned_array(T** dest, size_t elements)
{
    return temp_storage(dest, elements * sizeof(T), alignof(T));
}

template<typename... Ts>
struct sequence_partition
{
    ::rocprim::tuple<Ts...> sub_partitions;

    sequence_partition(Ts... sub_partitions) : sub_partitions{sub_partitions...} {}

    layout get_layout()
    {
        size_t required_alignment = 1;
        size_t required_size      = 0;

        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition)
                          {
                              const auto sub_layout = sub_partition.get_layout();

                              if(sub_layout.alignment > required_alignment)
                                  required_alignment = sub_layout.alignment;

                              if(sub_layout.size > 0)
                                  required_size = align_size(required_size, sub_layout.alignment)
                                                  + sub_layout.size;
                          });

        return {required_size, required_alignment};
    }

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

template<typename... Ts>
sequence_partition<Ts...> sequence(Ts... ts)
{
    return sequence_partition<Ts...>(ts...);
}

template<typename... Ts>
struct mutually_exclusive_partition
{
    ::rocprim::tuple<Ts...> sub_partitions;

    mutually_exclusive_partition(Ts... sub_partitions) : sub_partitions{sub_partitions...} {}

    layout get_layout()
    {
        size_t required_alignment = 1;
        size_t required_size      = 0;

        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition)
                          {
                              const auto sub_layout = sub_partition.get_layout();

                              if(sub_layout.alignment > required_alignment)
                                  required_alignment = sub_layout.alignment;
                              if(sub_layout.size > required_size)
                                  required_size = sub_layout.size;
                          });

        return {required_size, required_alignment};
    }

    void set_storage(void* const storage)
    {
        for_each_in_tuple(this->sub_partitions,
                          [&](auto& sub_partition) { sub_partition.set_storage(storage); });
    }
};

template<typename... Ts>
mutually_exclusive_partition<Ts...> mutually_exclusive(Ts... ts)
{
    return mutually_exclusive_partition<Ts...>(ts...);
}

template<typename TempStoragePartition>
hipError_t
    partition(void* const temporary_storage, size_t& storage_size, TempStoragePartition partition)
{
    const auto layout = partition.get_layout();

    if(temporary_storage == nullptr)
    {
        // Make sure the user wont try to allocate 0 bytes of memory.
        storage_size = layout.size == 0 ? 4 : layout.size;
        return hipSuccess;
    }
    else if(storage_size < layout.size)
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
