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

#include "../config.hpp"

#include <cassert>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class OffsetT = unsigned int>
struct range_t
{
    OffsetT                begin1;
    OffsetT                end1;
    OffsetT                begin2;
    OffsetT                end2;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    constexpr unsigned int count1() const
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

template<unsigned int ItemsPerThread, class KeyType, class BinaryFunction, class OutputFunction, class OffsetT>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void serial_merge(KeyType*       keys_shared,
                  range_t<OffsetT> range,
                  BinaryFunction compare_function,
                  OutputFunction output_function)
{
    // Pre condition, we're including some edge cases too.
    assert(range.begin1 <= range.end1);
    assert(range.begin2 <= range.end2);

    // Pre-loaded keys so we don't have to re-fetch multiple times from memory.
    // These will be updated every iteration.
    KeyType key_a;
    KeyType key_b;
    auto    num_a = range.end1 - range.begin1;
    auto    num_b = range.end2 - range.begin2;

    // Only load valid keys, otherwise might be out of bounds!
    if(num_a > 0)
    {
        key_a = keys_shared[range.begin1];
    }
    if(num_b > 0)
    {
        key_b = keys_shared[range.begin2];
    }

    ROCPRIM_UNROLL
    for(OffsetT i = 0; i < ItemsPerThread; ++i)
    {
        // If we don't have any in b, we always take from a. Then, if we don't
        // have any in a, we take from b. Otherwise we take the smallest item.
        //
        // We're using the number of items left as a shortcut for comparing
        // items.
        const bool take_a = (num_b == 0) || ((num_a > 0) && !compare_function(key_b, key_a));

        // Retrieve info about the smallest key.
        const auto idx = take_a ? range.begin1 : range.begin2;
        const auto num = take_a ? num_a : num_b;
        const auto key = take_a ? key_a : key_b;

        // Output results.
        output_function(i, key, idx);

        // Get the next key from the array that we consumed from. We need two
        // seperate checks:
        // 1) We need at least two items to read the next item, as we're already
        //    pre-loaded one before we started looping.
        // 2) We need at least one item to read from the input. Otherwise we
        //    don't do anything!
        const auto next_idx = num > 1 ? idx + 1 : idx;
        const auto next_num = num > 0 ? num - 1 : num;

        // We don't access out of range!
        assert(next_idx < range.end2);

        // Load the next item. The compiler *should* be smart enough to optimize
        // away the case where we don't have any items to read.
        const auto next_key = keys_shared[next_idx];

        // Store the info about the next key.
        if(take_a)
        {
            range.begin1 = next_idx;
            key_a        = next_key;
            num_a        = next_num;
        }
        else
        {
            range.begin2 = next_idx;
            key_b        = next_key;
            num_b        = next_num;
        }
    }

    // We don't finish with a block sync since this may be used on thread or
    // warp granularity!
}

template<class KeyType, unsigned int ItemsPerThread, class BinaryFunction, class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  unsigned int (&indices)[ItemsPerThread],
                  range_t<OffsetT> range,
                  BinaryFunction   compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](OffsetT i, KeyType key, OffsetT index)
                                 {
                                     outputs[i] = key;
                                     indices[i] = index;
                                 });
}

template<class KeyType, unsigned int ItemsPerThread, class BinaryFunction, class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  range_t<OffsetT> range,
                  BinaryFunction   compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](OffsetT i, KeyType key, OffsetT) { outputs[i] = key; });
}

template<class KeyType,
         class ValueType,
         unsigned int ItemsPerThread,
         class BinaryFunction,
         class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  ValueType* values_shared,
                  ValueType (&values)[ItemsPerThread],
                  range_t<OffsetT> range,
                  BinaryFunction   compare_function)
{
    serial_merge<ItemsPerThread>(keys_shared,
                                 range,
                                 compare_function,
                                 [&](OffsetT i, KeyType key, OffsetT index)
                                 {
                                     outputs[i] = key;
                                     values[i]  = values_shared[index];
                                 });
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MERGE_PATH_HPP_
