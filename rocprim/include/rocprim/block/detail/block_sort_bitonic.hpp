// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Key,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    unsigned int ItemsPerThread,
    class Value
>
class block_sort_bitonic
{
    static constexpr unsigned int BlockSize     = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int ItemsPerBlock = BlockSize * ItemsPerThread;

    template<class KeyType, class ValueType>
    struct storage_type_
    {
        KeyType   key[BlockSize * ItemsPerThread];
        ValueType value[BlockSize * ItemsPerThread];
    };

    template<class KeyType>
    struct storage_type_<KeyType, empty_type>
    {
        KeyType key[BlockSize * ItemsPerThread];
    };

public:
    using storage_type = detail::raw_storage<storage_type_<Key, Value>>;

    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl<BlockSize, ItemsPerThread>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_key);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl<BlockSize, ItemsPerThread>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key& thread_key,
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_keys, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl<BlockSize, ItemsPerThread>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_key,
            thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        this->sort_impl<BlockSize, ItemsPerThread>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys,
            thread_values);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, thread_value, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_keys, thread_values, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&               thread_key,
                                            storage_type&      storage,
                                            const unsigned int size,
                                            BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_key);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            storage_type&      storage,
                                            const unsigned int size,
                                            BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&               thread_key,
                                            Value&             thread_value,
                                            storage_type&      storage,
                                            const unsigned int size,
                                            BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_key,
                        thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            storage_type&      storage,
                                            const unsigned int size,
                                            BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_keys,
                        thread_values);
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void copy_to_shared(Key& k, const unsigned int flat_tid, storage_type& storage)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        storage_.key[flat_tid] = k;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void copy_to_shared(Key (&k)[ItemsPerThread], const unsigned int flat_tid, storage_type& storage) {
        storage_type_<Key, Value>& storage_ = storage.get();
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            storage_.key[item * BlockSize + flat_tid] = k[item];
        }
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void copy_to_shared(Key& k, Value& v, const unsigned int flat_tid, storage_type& storage)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        storage_.key[flat_tid] = k;
        storage_.value[flat_tid] = v;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void copy_to_shared(Key (&k)[ItemsPerThread],
                        Value (&v)[ItemsPerThread],
                        const unsigned int flat_tid,
                        storage_type&      storage)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            storage_.key[item * BlockSize + flat_tid]   = k[item];
            storage_.value[item * BlockSize + flat_tid] = v[item];
        }
        ::rocprim::syncthreads();
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void swap(Key& key,
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key next_key = storage_.key[next_id];
        bool compare = (next_id < flat_tid) ? compare_function(key, next_key) : compare_function(next_key, key);
        bool swap = compare ^ dir;
        if(swap)
        {
            key = next_key;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void swap(Key (&key)[ItemsPerThread],
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            Key next_key = storage_.key[item * BlockSize + next_id];
            bool compare = (next_id < flat_tid) ? compare_function(key[item], next_key) : compare_function(next_key, key[item]);
            bool swap = compare ^ dir;
            if(swap)
            {
                key[item] = next_key;
            }
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void swap(Key& key,
              Value& value,
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key next_key = storage_.key[next_id];
        bool b = next_id < flat_tid;
        bool compare = compare_function(b ? key : next_key, b ? next_key : key);
        bool swap = compare ^ dir;
        if(swap)
        {
            key = next_key;
            value = storage_.value[next_id];
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void swap(Key (&key)[ItemsPerThread],
              Value (&value)[ItemsPerThread],
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            Key next_key = storage_.key[item * BlockSize + next_id];
            bool b = next_id < flat_tid;
            bool compare = compare_function(b ? key[item] : next_key, b ? next_key : key[item]);
            bool swap = compare ^ dir;
            if(swap)
            {
                key[item]   = next_key;
                value[item] = storage_.value[item * BlockSize + next_id];
            }
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void swap_oddeven(Key&               key,
                                                    const unsigned int next_id,
                                                    const unsigned int /* item */,
                                                    const unsigned int next_item_id,
                                                    bool               dir,
                                                    storage_type&      storage,
                                                    BinaryFunction     compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key                        next_key = storage_.key[next_item_id * BlockSize + next_id];
        // swap items instead of branching for compare function, to achieve superior perf (ROCm 5.3)
        if(dir)
        {
            rocprim::swap(next_key, key);
        }
        bool swap = compare_function(next_key, key);
        if(dir)
        {
            rocprim::swap(next_key, key);
        }
        if(swap)
        {
            key = next_key;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void swap_oddeven(Key (&keys)[ItemsPerThread],
                                                    const unsigned int next_id,
                                                    const unsigned int item,
                                                    const unsigned int next_item_id,
                                                    bool               dir,
                                                    storage_type&      storage,
                                                    BinaryFunction     compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key                        next_key = storage_.key[next_item_id * BlockSize + next_id];
        // swap items instead of branching for compare function, to achieve superior perf (ROCm 5.3)
        if(dir)
        {
            rocprim::swap(next_key, keys[item]);
        }
        bool swap = compare_function(next_key, keys[item]);
        if(dir)
        {
            rocprim::swap(next_key, keys[item]);
        }
        if(swap)
        {
            keys[item] = next_key;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void swap_oddeven(Key&               key,
                                                    Value&             value,
                                                    const unsigned int next_id,
                                                    const unsigned int /* item */,
                                                    const unsigned int next_item_id,
                                                    bool               dir,
                                                    storage_type&      storage,
                                                    BinaryFunction     compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key                        next_key = storage_.key[next_item_id * BlockSize + next_id];
        // swap items instead of branching for compare function, to achieve superior perf (ROCm 5.3)
        if(dir)
        {
            rocprim::swap(next_key, key);
        }
        bool swap = compare_function(next_key, key);
        if(dir)
        {
            rocprim::swap(next_key, key);
        }
        if(swap)
        {
            key   = next_key;
            value = storage_.value[next_item_id * BlockSize + next_id];
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void swap_oddeven(Key (&keys)[ItemsPerThread],
                                                    Value (&values)[ItemsPerThread],
                                                    const unsigned int next_id,
                                                    const unsigned int item,
                                                    const unsigned int next_item_id,
                                                    bool               dir,
                                                    storage_type&      storage,
                                                    BinaryFunction     compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key                        next_key = storage_.key[next_item_id * BlockSize + next_id];
        // swap items instead of branching for compare function, to achieve superior perf (ROCm 5.3)
        if(dir)
        {
            rocprim::swap(next_key, keys[item]);
        }
        bool swap = compare_function(next_key, keys[item]);
        if(dir)
        {
            rocprim::swap(next_key, keys[item]);
        }
        if(swap)
        {
            keys[item]   = next_key;
            values[item] = storage_.value[next_item_id * BlockSize + next_id];
        }
    }

    template<
        unsigned int Size,
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Size <= ::rocprim::device_warp_size())>::type
    sort_power_two(const unsigned int flat_tid,
                   storage_type& storage,
                   BinaryFunction compare_function,
                   KeyValue&... kv)
    {
        (void) flat_tid;
        (void) storage;

        ::rocprim::warp_sort<Key, Size, Value> wsort;
        wsort.sort(kv..., compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_swap(Key& k, Value& v, int mask, bool dir, BinaryFunction compare_function)
    {
        Key k1    = warp_shuffle_xor(k, mask);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if (swap)
        {
            k = k1;
            v = warp_shuffle_xor(v, mask);
        }
    }

    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_swap(Key (&k)[ItemsPerThread],
                   Value (&v)[ItemsPerThread],
                   int            mask,
                   bool           dir,
                   BinaryFunction compare_function)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            Key k1    = warp_shuffle_xor(k[item], mask);
            bool swap = compare_function(dir ? k[item] : k1, dir ? k1 : k[item]);
            if (swap)
            {
                k[item] = k1;
                v[item] = warp_shuffle_xor(v[item], mask);
            }
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_swap(Key& k, int mask, bool dir, BinaryFunction compare_function)
    {
        Key k1    = warp_shuffle_xor(k, mask);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if (swap)
        {
            k = k1;
        }
    }

    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_swap(Key (&k)[ItemsPerThread], int mask, bool dir, BinaryFunction compare_function)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item) {
            Key k1    = warp_shuffle_xor(k[item], mask);
            bool swap = compare_function(dir ? k[item] : k1, dir ? k1 : k[item]);
            if (swap)
            {
                k[item] = k1;
            }
        }
    }

    template <class BinaryFunction, unsigned int Items = ItemsPerThread, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Items < 2)>::type
    thread_merge(bool /*dir*/, BinaryFunction /*compare_function*/, KeyValue&... /*kv*/)
    {
    }

    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void thread_swap(Key (&k)[ItemsPerThread],
                     Value (&v)[ItemsPerThread],
                     bool           dir,
                     unsigned int   i,
                     unsigned int   j,
                     BinaryFunction compare_function)
    {
        if(compare_function(k[i], k[j]) == dir)
        {
            Key k_temp   = k[i];
            k[i]         = k[j];
            k[j]         = k_temp;
            Value v_temp = v[i];
            v[i]         = v[j];
            v[j]         = v_temp;
        }
    }
    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void thread_swap(Key (&k)[ItemsPerThread],
                     bool           dir,
                     unsigned int   i,
                     unsigned int   j,
                     BinaryFunction compare_function)
    {
        if(compare_function(k[i], k[j]) == dir)
        {
            Key k_temp = k[i];
            k[i]       = k[j];
            k[j]       = k_temp;
        }
    }

    template <class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void thread_shuffle(unsigned int offset, bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        ROCPRIM_UNROLL
        for(unsigned base = 0; base < ItemsPerThread; base += 2 * offset)
        {
            ROCPRIM_UNROLL
// Workaround to prevent the compiler thinking this is a 'Parallel Loop' on clang 15
// because it leads to invalid code generation with `T` = `char` and `ItemsPerthread` = 4
#if defined(__clang_major__) && __clang_major__ >= 15
    #pragma clang loop vectorize(disable)
#endif
            for(unsigned i = 0; i < offset; ++i)
            {
                thread_swap(kv..., dir, base + i, base + i + offset, compare_function);
            }
        }
    }

    template <class BinaryFunction, unsigned int Items = ItemsPerThread, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(Items < 2)>::type
    thread_merge(bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        ROCPRIM_UNROLL
        for(unsigned int k = ItemsPerThread / 2; k > 0; k /= 2)
        {
            thread_shuffle(k, dir, compare_function, kv...);
        }    
    }

    /// Bitonic sort.
    /// Requires a power of two block size and a power of two items per thread.
    template<unsigned int BS, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        typename std::enable_if<(BS > ::rocprim::device_warp_size())>::type
        sort_power_two(const unsigned int flat_tid,
                       storage_type&      storage,
                       BinaryFunction     compare_function,
                       KeyValue&... kv)
    {
        const auto warp_id_is_even = ((flat_tid / ::rocprim::device_warp_size()) % 2) == 0;
        ::rocprim::warp_sort<Key, ::rocprim::device_warp_size(), Value> wsort;
        auto compare_function2 =
            [compare_function, warp_id_is_even](const Key& a, const Key& b) mutable -> bool
            {
                auto r = compare_function(a, b);
                if(warp_id_is_even)
                    return r;
                return !r;
            };
        wsort.sort(kv..., compare_function2);

        ROCPRIM_UNROLL
        for(unsigned int length = ::rocprim::device_warp_size(); length < BS; length *= 2)
        {
            const bool dir = (flat_tid & (length * 2)) != 0;
            ROCPRIM_UNROLL
            for(unsigned int k = length; k > ::rocprim::device_warp_size() / 2; k /= 2)
            {
                copy_to_shared(kv..., flat_tid, storage);
                swap(kv..., flat_tid, flat_tid ^ k, dir, storage, compare_function);
                ::rocprim::syncthreads();
            }

            ROCPRIM_UNROLL
            for(unsigned int k = ::rocprim::device_warp_size() / 2; k > 0;  k /= 2)
            {
                const bool length_even = ((detail::logical_lane_id<::rocprim::device_warp_size()>() / k ) % 2 ) == 0;
                const bool local_dir = length_even ? dir : !dir;
                warp_swap(kv..., k, local_dir, compare_function);
            }
            thread_merge(dir, compare_function, kv...);
        }
    }

    template<unsigned int BS, unsigned int IPT, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        typename std::enable_if<is_power_of_two(BS) && is_power_of_two(IPT)>::type
        sort_impl(const unsigned int flat_tid,
                  storage_type&      storage,
                  BinaryFunction     compare_function,
                  KeyValue&... kv)
    {
        static constexpr unsigned int PairSize = sizeof...(KeyValue);
        static_assert(PairSize < 3,
                      "KeyValue parameter pack can be 1 or 2 elements (key, or key and value)");

        sort_power_two<BS>(flat_tid, storage, compare_function, kv...);
    }

    /// Odd-even sort.
    /// No restrictions on block size or items per thread, but slower than bitonic sort.
    template<bool SizeCheck, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void odd_even_sort(const unsigned int flat_tid,
                                                     const unsigned int size,
                                                     storage_type&      storage,
                                                     BinaryFunction     compare_function,
                                                     KeyValue&... kv)
    {
        static constexpr unsigned int PairSize = sizeof...(KeyValue);
        static_assert(PairSize < 3,
                      "KeyValue parameter pack can be 1 or 2 elements (key, or key and value)");

        if(SizeCheck && size > ItemsPerBlock)
        {
            return;
        }

        copy_to_shared(kv..., flat_tid, storage);

        for(unsigned int i = 0; i < size; i++)
        {
            bool is_even_iter = i % 2 == 0;
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                // the element in the original array that key[item] represents
                unsigned int linear_id   = flat_tid * ItemsPerThread + item;
                bool         is_even_lid = linear_id % 2 == 0;

                // one up/down from the linear_id
                unsigned int odd_lid  = is_even_lid ? ::rocprim::max(linear_id, 1u) - 1
                                                    : ::rocprim::min(linear_id + 1, size - 1);
                unsigned int even_lid = is_even_lid ? ::rocprim::min(linear_id + 1, size - 1)
                                                    : ::rocprim::max(linear_id, 1u) - 1;

                // determine if the odd or even index must be used
                unsigned int next_lid = is_even_iter ? even_lid : odd_lid;

                // map the linear_id back to item and thread id for indexing shared memory
                unsigned int next_id      = next_lid / ItemsPerThread;
                unsigned int next_item_id = next_lid % ItemsPerThread;

                // prevent calling the compare function with out-of-bounds items
                if(!SizeCheck || (linear_id < size && next_lid < size))
                {
                    swap_oddeven(kv...,
                                 next_id,
                                 item,
                                 next_item_id,
                                 next_lid < linear_id,
                                 storage,
                                 compare_function);
                }
            }
            ::rocprim::syncthreads();
            copy_to_shared(kv..., flat_tid, storage);
        }
    }

    template<unsigned int BS, unsigned int IPT, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        typename std::enable_if<!is_power_of_two(BS) || !is_power_of_two(IPT)>::type
        sort_impl(const unsigned int flat_tid,
                  storage_type&      storage,
                  BinaryFunction     compare_function,
                  KeyValue&... kv)
    {
        odd_even_sort<false>(flat_tid, ItemsPerBlock, storage, compare_function, kv...);
    }

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_impl(const unsigned int flat_tid,
                                                 const unsigned int size,
                                                 storage_type&      storage,
                                                 BinaryFunction     compare_function,
                                                 KeyValue&... kv)
    {
        odd_even_sort<true>(flat_tid, size, storage, compare_function, kv...);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
