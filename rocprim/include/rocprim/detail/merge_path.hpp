// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_MERGE_PATH_HPP_
#define ROCPRIM_DETAIL_MERGE_PATH_HPP_

#include "../intrinsics/thread.hpp"

#include "../config.hpp"

#include <cassert>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

struct range_t
{
    unsigned int begin1;
    unsigned int end1;
    unsigned int begin2;
    unsigned int end2;

    ROCPRIM_DEVICE ROCPRIM_INLINE constexpr unsigned int count1() const
    {
        return end1 - begin1;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE constexpr unsigned int count2() const
    {
        return end2 - begin2;
    }
};

template<class KeysInputIterator1, class KeysInputIterator2, class OffsetT, class BinaryFunction>
ROCPRIM_HOST_DEVICE ROCPRIM_INLINE OffsetT merge_path(KeysInputIterator1 keys_input1,
                                                      KeysInputIterator2 keys_input2,
                                                      const OffsetT      input1_size,
                                                      const OffsetT      input2_size,
                                                      const OffsetT      diag,
                                                      BinaryFunction     compare_function)
{
    using key_type_1 = typename std::iterator_traits<KeysInputIterator1>::value_type;
    using key_type_2 = typename std::iterator_traits<KeysInputIterator2>::value_type;

    OffsetT begin = diag < input2_size ? 0u : diag - input2_size;
    OffsetT end   = min(diag, input1_size);

    while(begin < end)
    {
        OffsetT    a       = (begin + end) / 2;
        OffsetT    b       = diag - 1 - a;
        key_type_1 input_a = keys_input1[a];
        key_type_2 input_b = keys_input2[b];
        if(!compare_function(input_b, input_a))
        {
            begin = a + 1;
        }
        else
        {
            end = a;
        }
    }

    return begin;
}

template<unsigned int ItemsPerThread, class KeyType, class BinaryFunction, class OutputFunction>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void serial_merge(KeyType*       keys_shared,
                  range_t        range,
                  BinaryFunction compare_function,
                  OutputFunction output_function)
{
    // pre-loaded keys so we don't have to re-fetch multiple times from memory
    KeyType key_a = keys_shared[range.begin1];
    KeyType key_b = keys_shared[range.begin2];

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        // if we read outside the latter half of the range, it will loop back into the earlier half
        const bool compare = (range.begin2 >= range.end2)
                             || ((range.begin1 < range.end1) && !compare_function(key_b, key_a));

        // get the index of the item we want to write away
        const unsigned int read_index = compare ? range.begin1 : range.begin2;

        // output results: we don't care how it's done and not every value needs to be used
        output_function(i, compare ? key_a : key_b, read_index);

        // we're done writing, time to load in the next value
        const unsigned int next_index = read_index + 1;

        // we shouldn't read more than we need to.
        assert(next_index < range.end2);

        // update ranges and cached keys
        const KeyType c = keys_shared[next_index];
        if(compare)
        {
            key_a        = c;
            range.begin1 = next_index;
        }
        else
        {
            key_b        = c;
            range.begin2 = next_index;
        }
    }
}

template<class KeyType, unsigned int ItemsPerThread, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  unsigned int (&indices)[ItemsPerThread],
                  range_t        range,
                  BinaryFunction compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](unsigned i, KeyType key, unsigned int index)
                                 {
                                     outputs[i] = key;
                                     indices[i] = index;
                                 });
}

template<class KeyType, unsigned int ItemsPerThread, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  range_t        range,
                  BinaryFunction compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](unsigned i, KeyType key, unsigned int) { outputs[i] = key; });
}

template<class KeyType, class ValueType, unsigned int ItemsPerThread, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  ValueType* values_shared,
                  ValueType (&values)[ItemsPerThread],
                  range_t        range,
                  BinaryFunction compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](unsigned i, KeyType key, unsigned int index)
                                 {
                                     outputs[i] = key;
                                     values[i]  = values_shared[index];
                                 });
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MERGE_PATH_HPP_
