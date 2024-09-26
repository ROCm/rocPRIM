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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_exchange.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_load_func.hpp"
#include "../../block/block_radix_rank.hpp"
#include "../../block/block_radix_sort.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store_func.hpp"
#include "../../thread/radix_key_codec.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Wrapping functions that allow one to call proper methods (with or without values)
// (a variant with values is enabled only when Value is not empty_type)
template<bool Descending = false,
         class SortType,
         class SortKey,
         class SortValue,
         unsigned int ItemsPerThread,
         class Decomposer>
ROCPRIM_DEVICE ROCPRIM_INLINE void sort_block(SortType sorter,
                                              SortKey (&keys)[ItemsPerThread],
                                              SortValue (&values)[ItemsPerThread],
                                              typename SortType::storage_type& storage,
                                              Decomposer                       decomposer,
                                              unsigned int                     begin_bit,
                                              unsigned int                     end_bit)
{
    if ROCPRIM_IF_CONSTEXPR(Descending)
    {
        sorter.sort_desc(keys, values, storage, begin_bit, end_bit, decomposer);
    }
    else
    {
        sorter.sort(keys, values, storage, begin_bit, end_bit, decomposer);
    }
}

template<bool Descending = false,
         class SortType,
         class SortKey,
         unsigned int ItemsPerThread,
         class Decomposer>
ROCPRIM_DEVICE ROCPRIM_INLINE void sort_block(SortType sorter,
                                              SortKey (&keys)[ItemsPerThread],
                                              ::rocprim::empty_type (&values)[ItemsPerThread],
                                              typename SortType::storage_type& storage,
                                              Decomposer                       decomposer,
                                              unsigned int                     begin_bit,
                                              unsigned int                     end_bit)
{
    (void) values;
    if ROCPRIM_IF_CONSTEXPR(Descending)
    {
        sorter.sort_desc(keys, storage, begin_bit, end_bit, decomposer);
    }
    else
    {
        sorter.sort(keys, storage, begin_bit, end_bit, decomposer);
    }
}

template<
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending
>
struct radix_digit_count_helper
{
    static constexpr unsigned int radix_size     = 1 << RadixBits;
    static constexpr unsigned int warp_size = WarpSize;
    static constexpr unsigned int atomic_stripes = 4;
    static constexpr unsigned int counters       = radix_size * atomic_stripes;

    ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(BlockSize % ::rocprim::device_warp_size() == 0,
                                        "BlockSize must be divisible by warp size");
    static_assert(radix_size <= BlockSize, "Radix size must not exceed BlockSize");

    struct storage_type
    {
        unsigned int digit_counters[counters];
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int&
        get_counter(const unsigned stripe, const unsigned int digit, storage_type& storage)
    {
        return storage.digit_counters[digit * atomic_stripes + stripe];
    }

    template<
        bool IsFull = false,
        class KeysInputIterator,
        class Offset
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void count_digits(KeysInputIterator keys_input,
                      Offset begin_offset,
                      Offset end_offset,
                      unsigned int bit,
                      unsigned int current_radix_bits,
                      storage_type& storage,
                      unsigned int& digit_count)  // i-th thread will get i-th digit's value
    {
        constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

        using key_codec    = ::rocprim::radix_key_codec<key_type, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int stripe  = flat_id % atomic_stripes;

        for(unsigned int i = flat_id; i < counters; i += BlockSize)
        {
            storage.digit_counters[i] = 0;
        }

        ::rocprim::syncthreads();

        for(Offset block_offset = begin_offset; block_offset < end_offset; block_offset += items_per_block)
        {
            key_type keys[ItemsPerThread];
            unsigned int valid_count;
            // Use loading into a striped arrangement because an order of items is irrelevant,
            // only totals matter
            if(IsFull || (block_offset + items_per_block <= end_offset))
            {
                valid_count = items_per_block;
                block_load_direct_striped<BlockSize>(flat_id, keys_input + block_offset, keys);
            }
            else
            {
                valid_count = end_offset - block_offset;
                block_load_direct_striped<BlockSize>(flat_id, keys_input + block_offset, keys, valid_count);
            }

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const bit_key_type bit_key = key_codec::encode(keys[i]);
                const unsigned int digit = key_codec::extract_digit(bit_key, bit, current_radix_bits);
                const unsigned int pos = i * BlockSize + flat_id;

                if(IsFull || pos < valid_count)
                {
                    atomic_add(&get_counter(stripe, digit, storage), 1);
                }
            }
        }

        ::rocprim::syncthreads();

        digit_count = 0;
        if(flat_id < radix_size)
        {
            // Sum counters from all stripes
            ROCPRIM_UNROLL
            for(unsigned int stripe = 0; stripe < atomic_stripes; ++stripe)
            {
                digit_count += get_counter(stripe, flat_id, storage);
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending,
    class Key,
    class Value
>
struct radix_sort_single_helper
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using key_type = Key;
    using value_type = Value;

    using sort_type = ::rocprim::block_radix_sort<key_type, BlockSize, ItemsPerThread, value_type>;

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    struct storage_type
    {
        typename sort_type::storage_type sort;
    };

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class Decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_single(KeysInputIterator    keys_input,
                                                   KeysOutputIterator   keys_output,
                                                   ValuesInputIterator  values_input,
                                                   ValuesOutputIterator values_output,
                                                   unsigned int         size,
                                                   Decomposer           decomposer,
                                                   unsigned int         bit,
                                                   unsigned int         current_radix_bits,
                                                   storage_type&        storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
        const unsigned int block_offset        = flat_block_id * items_per_block;
        const bool         is_incomplete_block = flat_block_id == (size / items_per_block);
        const unsigned int valid_in_last_block = size - block_offset;

        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

        using key_codec = radix_key_codec<key_type, Descending>;

        key_type keys[ItemsPerThread];
        value_type values[ItemsPerThread];
        if(!is_incomplete_block)
        {
            block_load_direct_blocked(flat_id, keys_input + block_offset, keys);
            if ROCPRIM_IF_CONSTEXPR(with_values)
            {
                block_load_direct_blocked(flat_id, values_input + block_offset, values);
            }
        }
        else
        {
            const key_type out_of_bounds = key_codec::get_out_of_bounds_key(decomposer);
            block_load_direct_blocked(flat_id,
                                      keys_input + block_offset,
                                      keys,
                                      valid_in_last_block,
                                      out_of_bounds);
            if ROCPRIM_IF_CONSTEXPR(with_values)
            {
                block_load_direct_blocked(flat_id,
                                          values_input + block_offset,
                                          values,
                                          valid_in_last_block);
            }
        }

        sort_block<Descending>(sort_type(),
                               keys,
                               values,
                               storage.sort,
                               decomposer,
                               bit,
                               bit + current_radix_bits);

        // Store keys and values
        if(!is_incomplete_block)
        {
            block_store_direct_blocked(flat_id, keys_output + block_offset, keys);
            if ROCPRIM_IF_CONSTEXPR(with_values)
            {
                block_store_direct_blocked(flat_id, values_output + block_offset, values);
            }
        }
        else
        {
            block_store_direct_blocked(flat_id,
                                       keys_output + block_offset,
                                       keys,
                                       valid_in_last_block);
            if ROCPRIM_IF_CONSTEXPR(with_values)
            {
                block_store_direct_blocked(flat_id,
                                           values_output + block_offset,
                                           values,
                                           valid_in_last_block);
            }
        }
    }
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         bool         Descending,
         class Key,
         class Value,
         class Offset,
         block_radix_rank_algorithm RadixRankAlgorithm = block_radix_rank_algorithm::match>
struct radix_sort_and_scatter_helper
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    static constexpr unsigned int radix_size = 1 << RadixBits;
    static constexpr unsigned int digits_per_thread = 1;
    static constexpr bool         with_values = !std::is_same<Value, ::rocprim::empty_type>::value;

    using key_codec       = radix_key_codec<Key, Descending>;
    using radix_rank_type = ::rocprim::block_radix_rank<BlockSize, RadixBits, RadixRankAlgorithm>;

    static constexpr bool load_warp_striped
        = RadixRankAlgorithm == block_radix_rank_algorithm::match;

    static_assert(radix_size <= BlockSize, "Radix size must not exceed BlockSize");

    struct storage_type_
    {
        Offset digit_offsets[radix_size];
        union
        {
            typename radix_rank_type::storage_type rank;

            Key   ordered_tile_keys[items_per_block];
            Value ordered_tile_values[items_per_block];
        };
    };

    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<bool IsFull = false,
             class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_and_scatter(KeysInputIterator    keys_input,
                          KeysOutputIterator   keys_output,
                          ValuesInputIterator  values_input,
                          ValuesOutputIterator values_output,
                          Offset               begin_offset,
                          Offset               end_offset,
                          unsigned int         bit,
                          unsigned int         current_radix_bits,
                          Offset        digit_start, // i-th thread must pass i-th digit's value
                          storage_type& storage_)
    {
        auto&              storage = storage_.get();
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

        if(flat_id < radix_size)
        {
            storage.digit_offsets[flat_id] = digit_start;
        }

        for(Offset block_offset = begin_offset; block_offset < end_offset; block_offset += items_per_block)
        {
            Key keys[ItemsPerThread];

            unsigned int valid_items;
            if(IsFull || (block_offset + items_per_block <= end_offset))
            {
                valid_items = items_per_block;
                if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                {
                    block_load_direct_warp_striped(flat_id, keys_input + block_offset, keys);
                }
                else
                {
                    block_load_direct_blocked(flat_id, keys_input + block_offset, keys);
                }
            }
            else
            {
                valid_items = end_offset - block_offset;
                // Fill the out-of-bounds elements of the key array with the key value with
                // the largest digit. This will make sure they are sorted (ranked) last, and
                // thus will be omitted when we compare the item offset against `valid_items` later.
                // Note that this will lead to an incorrect digit count. Since this is the very last digit,
                // it does not matter. It does cause the final digit offset to be increased past its end,
                // but again this does not matter since this is the last iteration in which it will be used anyway.
                const Key out_of_bounds = key_codec::get_out_of_bounds_key();
                if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                {
                    block_load_direct_warp_striped(flat_id,
                                                   keys_input + block_offset,
                                                   keys,
                                                   valid_items,
                                                   out_of_bounds);
                }
                else
                {
                    block_load_direct_blocked(flat_id,
                                              keys_input + block_offset,
                                              keys,
                                              valid_items,
                                              out_of_bounds);
                }
            }

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                key_codec::encode_inplace(keys[i]);
            }

            unsigned int ranks[ItemsPerThread];
            unsigned int exclusive_digit_prefix[digits_per_thread];
            unsigned int digit_counts[digits_per_thread];

            radix_rank_type{}.rank_keys(
                keys,
                ranks,
                storage.rank,
                [bit, current_radix_bits](const Key& key)
                { return key_codec::extract_digit(key, bit, current_radix_bits); },
                exclusive_digit_prefix,
                digit_counts);

            ::rocprim::syncthreads();

            // Subtract the exclusive digit prefix from the digit offsets since we're ordering
            // the keys in shared memory already.
            if(flat_id < radix_size)
            {
                storage.digit_offsets[flat_id] -= exclusive_digit_prefix[0];
            }

            // Order keys in shared memory.
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                storage.ordered_tile_keys[ranks[i]] = keys[i];
            }

            ::rocprim::syncthreads();

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                const unsigned int rank = i * BlockSize + flat_id;
                if(IsFull || rank < valid_items)
                {
                    Key                key = storage.ordered_tile_keys[rank];
                    const unsigned int digit
                        = key_codec::extract_digit(key, bit, current_radix_bits);
                    key_codec::decode_inplace(key);
                    const Offset global_offset        = storage.digit_offsets[digit];
                    keys_output[rank + global_offset] = key;
                }
            }

            // Gather and scatter values if necessary
            if ROCPRIM_IF_CONSTEXPR(with_values)
            {
                Value values[ItemsPerThread];
                if ROCPRIM_IF_CONSTEXPR(IsFull)
                {
                    if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                    {
                        block_load_direct_warp_striped(flat_id,
                                                       values_input + block_offset,
                                                       values);
                    }
                    else
                    {
                        block_load_direct_blocked(flat_id, values_input + block_offset, values);
                    }
                }
                else
                {
                    if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                    {
                        block_load_direct_warp_striped(flat_id,
                                                       values_input + block_offset,
                                                       values,
                                                       valid_items);
                    }
                    else
                    {
                        block_load_direct_blocked(flat_id,
                                                  values_input + block_offset,
                                                  values,
                                                  valid_items);
                    }
                }

                // Compute digits up-front so that we can re-use shared memory between ordered_tile_keys and
                // ordered_tile_values.
                unsigned int digits[ItemsPerThread];
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    const unsigned int rank = i * BlockSize + flat_id;
                    if(IsFull || rank < valid_items)
                    {
                        const Key key = storage.ordered_tile_keys[rank];
                        digits[i]     = key_codec::extract_digit(key, bit, current_radix_bits);
                    }
                }

                ::rocprim::syncthreads();

                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    storage.ordered_tile_values[ranks[i]] = values[i];
                }

                ::rocprim::syncthreads();

                // And scatter the values to global memory.
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    const unsigned int rank = i * BlockSize + flat_id;
                    if(IsFull || rank < valid_items)
                    {
                        const Value  value                  = storage.ordered_tile_values[rank];
                        const Offset global_offset          = storage.digit_offsets[digits[i]];
                        values_output[rank + global_offset] = value;
                    }
                }
            }

            ::rocprim::syncthreads();

            // Update the digit offsets
            if(flat_id < radix_size)
            {
                storage.digit_offsets[flat_id]
                    += exclusive_digit_prefix[flat_id] + digit_counts[flat_id];
            }
        }
    }
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Decomposer>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort_single(KeysInputIterator    keys_input,
                                                     KeysOutputIterator   keys_output,
                                                     ValuesInputIterator  values_input,
                                                     ValuesOutputIterator values_output,
                                                     unsigned int         size,
                                                     Decomposer           decomposer,
                                                     unsigned int         bit,
                                                     unsigned int         current_radix_bits)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using sort_single_helper = radix_sort_single_helper<
        BlockSize, ItemsPerThread, Descending,
        key_type, value_type
    >;

    ROCPRIM_SHARED_MEMORY typename sort_single_helper::storage_type storage;

    sort_single_helper().template sort_single(keys_input,
                                              keys_output,
                                              values_input,
                                              values_output,
                                              size,
                                              decomposer,
                                              bit,
                                              current_radix_bits,
                                              storage);
}

template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto compare_nan_sensitive(const T& a, const T& b)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, bool>::type
{
    // Beware: the performance of this function is extremely vulnerable to refactoring.
    // Always check benchmark_device_segmented_radix_sort and benchmark_device_radix_sort
    // when making changes to this function.

    using bit_key_type = typename float_bit_mask<T>::bit_type;
    static constexpr auto sign_bit = float_bit_mask<T>::sign_bit;

    auto a_bits = ::rocprim::detail::bit_cast<bit_key_type>(a);
    auto b_bits = ::rocprim::detail::bit_cast<bit_key_type>(b);

    // convert -0.0 to +0.0
    a_bits = a_bits == sign_bit ? 0 : a_bits;
    b_bits = b_bits == sign_bit ? 0 : b_bits;
    // invert negatives, put 1 into sign bit for positives
    a_bits ^= (sign_bit & a_bits) == 0 ? sign_bit : bit_key_type(-1);
    b_bits ^= (sign_bit & b_bits) == 0 ? sign_bit : bit_key_type(-1);

    // sort numbers and NaNs according to their bit representation
    return a_bits > b_bits;
}

template<class T>
ROCPRIM_DEVICE auto compare_nan_sensitive(const T& a, const T& b) ->
    typename std::enable_if<!rocprim::is_floating_point<T>::value, bool>::type
{
    return a > b;
}

template<bool Descending, bool UseRadixMask, class T, class Decomposer = identity_decomposer>
struct radix_merge_compare;

template<class T>
struct radix_merge_compare<false, false, T, identity_decomposer>
{
    ROCPRIM_DEVICE bool operator()(const T& a, const T& b) const
    {
        return compare_nan_sensitive<T>(b, a);
    }
};

template<class T>
struct radix_merge_compare<true, false, T, identity_decomposer>
{
    ROCPRIM_DEVICE bool operator()(const T& a, const T& b) const
    {
        return compare_nan_sensitive<T>(a, b);
    }
};

template<bool Descending, class T>
struct radix_merge_compare<Descending, true, T, identity_decomposer>
{
    T radix_mask;

    ROCPRIM_HOST_DEVICE radix_merge_compare(const unsigned int start_bit,
                                            const unsigned int current_radix_bits,
                                            identity_decomposer = {})
    {
        T radix_mask_upper  = (T(1) << (current_radix_bits + start_bit)) - 1;
        T radix_mask_bottom = (T(1) << start_bit) - 1;
        radix_mask = radix_mask_upper ^ radix_mask_bottom;
    }

    ROCPRIM_DEVICE bool operator()(const T& a, const T& b) const
    {
        const T masked_key_a  = a & radix_mask;
        const T masked_key_b  = b & radix_mask;
        return Descending ? masked_key_a > masked_key_b : masked_key_b > masked_key_a;
    }
};

template<bool Descending, class T, class Decomposer>
struct radix_merge_compare<Descending, true, T, Decomposer>
{
    Decomposer   decomposer_;
    unsigned int start_bit_;
    unsigned int radix_bits_;

    ROCPRIM_HOST_DEVICE radix_merge_compare(const unsigned int start_bit,
                                            const unsigned int current_radix_bits,
                                            Decomposer         decomposer)
        : decomposer_(decomposer), start_bit_(start_bit), radix_bits_(current_radix_bits)
    {}

    ROCPRIM_HOST_DEVICE bool operator()(T lhs, T rhs) const
    {
        using codec_t = radix_key_codec<T, Descending>;

        // Encoding the values considers the ascending / descending nature of the sort
        codec_t::encode_inplace(lhs, decomposer_);
        codec_t::encode_inplace(rhs, decomposer_);

        // Digits can be extracted in 32 bit batches, but radix_bits_ can be larger than that
        static constexpr int digit_batch_size = 32;

        // Moving from MSB to LSB
        int current_start_bit
            = rocprim::max(0, static_cast<int>(start_bit_ + radix_bits_) - digit_batch_size);
        unsigned int remaining_radix_bits = radix_bits_;
        for(; remaining_radix_bits > 0;)
        {
            const unsigned int current_radix_bits
                = rocprim::min(remaining_radix_bits, static_cast<unsigned int>(digit_batch_size));
            remaining_radix_bits -= current_radix_bits;

            const unsigned int lhs_digits
                = codec_t::extract_digit(lhs,
                                         static_cast<unsigned int>(current_start_bit),
                                         current_radix_bits,
                                         decomposer_);
            const unsigned int rhs_digits
                = codec_t::extract_digit(rhs,
                                         static_cast<unsigned int>(current_start_bit),
                                         current_radix_bits,
                                         decomposer_);

            // Since we are moving from MSB to LSB, the earlier iteration implies the relation (if digits are not equal)
            if(lhs_digits != rhs_digits)
            {
                return rhs_digits > lhs_digits;
            }
            current_start_bit
                = rocprim::max(current_start_bit - static_cast<int>(current_radix_bits),
                               static_cast<int>(start_bit_));
        }
        return false;
    }
};

template<class KeyType,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         bool         Descending,
         class Decomposer>
struct onesweep_histograms_helper
{
    static constexpr unsigned int radix_size = 1u << RadixBits;
    // Upper bound, this value does not take into account the actual size of the number of bits
    // that are to be considered in the radix sort.
    static constexpr unsigned int max_digit_places
        = ::rocprim::detail::ceiling_div(sizeof(KeyType) * 8, RadixBits);
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    static constexpr unsigned int digits_per_thread
        = ::rocprim::detail::ceiling_div(radix_size, BlockSize);
    static constexpr unsigned int atomic_stripes = 4;
    static constexpr unsigned int histogram_counters
        = radix_size * max_digit_places * atomic_stripes;

    using counter_type = uint32_t;
    using key_codec    = radix_key_codec<KeyType, Descending>;

    struct storage_type
    {
        counter_type histogram[histogram_counters];
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE counter_type& get_counter(const unsigned     stripe_index,
                                                            const unsigned int place,
                                                            const unsigned int digit,
                                                            storage_type&      storage)
    {
        return storage.histogram[(place * radix_size + digit) * atomic_stripes + stripe_index];
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void clear_histogram(const unsigned int flat_id,
                                                       storage_type&      storage)
    {
        for(unsigned int i = flat_id; i < histogram_counters; i += BlockSize)
        {
            storage.histogram[i] = 0;
        }
    }

    template<bool IsFull>
    ROCPRIM_DEVICE void count_digits_at_place(const unsigned int flat_id,
                                              const unsigned int stripe,
                                              const KeyType (&keys)[ItemsPerThread],
                                              const unsigned int place,
                                              Decomposer         decomposer,
                                              const unsigned int start_bit,
                                              const unsigned int current_radix_bits,
                                              const unsigned int valid_count,
                                              storage_type&      storage)
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int pos = i * BlockSize + flat_id;
            if(IsFull || pos < valid_count)
            {
                const unsigned int digit
                    = key_codec::extract_digit(keys[i], start_bit, current_radix_bits, decomposer);
                ::rocprim::detail::atomic_add(&get_counter(stripe, place, digit, storage), 1);
            }
        }
    }

    template<bool IsFull, class KeysInputIterator, class Offset>
    ROCPRIM_DEVICE void count_digits(KeysInputIterator  keys_input,
                                     Offset*            global_digit_counts,
                                     const unsigned int valid_count,
                                     Decomposer         decomposer,
                                     const unsigned int begin_bit,
                                     const unsigned int end_bit,
                                     storage_type&      storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int stripe  = flat_id % atomic_stripes;

        KeyType keys[ItemsPerThread];
        // Load using a striped arrangement, the order doesn't matter here.
        if ROCPRIM_IF_CONSTEXPR(IsFull)
        {
            block_load_direct_striped<BlockSize>(flat_id, keys_input, keys);
        }
        else
        {
            block_load_direct_striped<BlockSize>(flat_id, keys_input, keys, valid_count);
        }

        // Initialize shared counters to zero.
        clear_histogram(flat_id, storage);

        ::rocprim::syncthreads();

        // Compute a shared histogram for each digit and each place.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            key_codec::encode_inplace(keys[i], decomposer);
        }

        for(unsigned int bit = begin_bit, place = 0; bit < end_bit; bit += RadixBits, ++place)
        {
            count_digits_at_place<IsFull>(flat_id,
                                          stripe,
                                          keys,
                                          place,
                                          decomposer,
                                          bit,
                                          min(RadixBits, end_bit - bit),
                                          valid_count,
                                          storage);
        }

        ::rocprim::syncthreads();

        // Combine the local histograms into a global histogram.

        unsigned int place = 0;
        for(unsigned int bit = begin_bit; bit < end_bit; bit += RadixBits)
        {
            for(unsigned int digit = flat_id; digit < radix_size; digit += BlockSize)
            {
                counter_type total = 0;

                ROCPRIM_UNROLL
                for(unsigned int stripe = 0; stripe < atomic_stripes; ++stripe)
                {
                    total += get_counter(stripe, place, digit, storage);
                }

                ::rocprim::detail::atomic_add(&global_digit_counts[place * radix_size + digit],
                                              total);
            }
            ++place;
        }
    }
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         bool         Descending,
         class KeysInputIterator,
         class Offset,
         class Decomposer>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void onesweep_histograms(KeysInputIterator  keys_input,
                                                             Offset*            global_digit_counts,
                                                             const Offset       size,
                                                             const Offset       full_blocks,
                                                             Decomposer         decomposer,
                                                             const unsigned int begin_bit,
                                                             const unsigned int end_bit)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using count_helper_type = onesweep_histograms_helper<key_type,
                                                         BlockSize,
                                                         ItemsPerThread,
                                                         RadixBits,
                                                         Descending,
                                                         Decomposer>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const Offset block_id     = ::rocprim::detail::block_id<0>();
    const Offset block_offset = block_id * ItemsPerThread * BlockSize;

    ROCPRIM_SHARED_MEMORY typename count_helper_type::storage_type storage;

    if(block_id < full_blocks)
    {
        count_helper_type{}.template count_digits<true>(keys_input + block_offset,
                                                        global_digit_counts,
                                                        items_per_block,
                                                        decomposer,
                                                        begin_bit,
                                                        end_bit,
                                                        storage);
    }
    else
    {
        const unsigned int valid_in_last_block = size - items_per_block * full_blocks;
        count_helper_type{}.template count_digits<false>(keys_input + block_offset,
                                                         global_digit_counts,
                                                         valid_in_last_block,
                                                         decomposer,
                                                         begin_bit,
                                                         end_bit,
                                                         storage);
    }
}

template<unsigned int BlockSize, unsigned int RadixBits, class Offset>
ROCPRIM_DEVICE void onesweep_scan_histograms(Offset* global_digit_offsets)
{
    using block_scan_type = block_scan<Offset, BlockSize>;

    constexpr unsigned int radix_size       = 1u << RadixBits;
    constexpr unsigned int items_per_thread = ::rocprim::detail::ceiling_div(radix_size, BlockSize);

    const unsigned int flat_id      = ::rocprim::detail::block_thread_id<0>();
    const unsigned int digit_place  = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = digit_place * radix_size;

    Offset offsets[items_per_thread];
    block_load_direct_blocked(flat_id, global_digit_offsets + block_offset, offsets, radix_size);
    block_scan_type{}.exclusive_scan(offsets, offsets, 0);
    block_store_direct_blocked(flat_id, global_digit_offsets + block_offset, offsets, radix_size);
}

struct onesweep_lookback_state
{
    // The two most significant bits are used to indicate the status of the prefix - leaving the other 30 bits for the
    // counter value.
    using underlying_type = uint32_t;

    static constexpr unsigned int state_bits = 8u * sizeof(underlying_type);

    enum prefix_flag : underlying_type
    {
        EMPTY    = 0,
        PARTIAL  = 1u << (state_bits - 2),
        COMPLETE = 2u << (state_bits - 2)
    };

    static constexpr underlying_type status_mask = 3u << (state_bits - 2);
    static constexpr underlying_type value_mask  = ~status_mask;

    underlying_type state;

    ROCPRIM_DEVICE ROCPRIM_INLINE explicit onesweep_lookback_state(underlying_type state)
        : state(state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE onesweep_lookback_state(prefix_flag status, underlying_type value)
        : state(static_cast<underlying_type>(status) | value)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE underlying_type value() const
    {
        return this->state & value_mask;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE prefix_flag status() const
    {
        return static_cast<prefix_flag>(this->state & status_mask);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static onesweep_lookback_state load(onesweep_lookback_state* ptr)
    {
        underlying_type state = ::rocprim::detail::atomic_load(&ptr->state);
        return onesweep_lookback_state(state);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void store(onesweep_lookback_state* ptr) const
    {
        ::rocprim::detail::atomic_store(&ptr->state, this->state);
    }
};

template<class Key,
         class Value,
         class Offset,
         unsigned int               BlockSize,
         unsigned int               ItemsPerThread,
         unsigned int               RadixBits,
         bool                       Descending,
         block_radix_rank_algorithm RadixRankAlgorithm,
         class Decomposer>
struct onesweep_iteration_helper
{
    static constexpr unsigned int radix_size      = 1u << RadixBits;
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    static constexpr bool         with_values = !std::is_same<Value, rocprim::empty_type>::value;

    using key_codec       = radix_key_codec<Key, Descending>;
    using radix_rank_type = ::rocprim::block_radix_rank<BlockSize, RadixBits, RadixRankAlgorithm>;

    static constexpr bool load_warp_striped
        = RadixRankAlgorithm == block_radix_rank_algorithm::match;

    static constexpr unsigned int digits_per_thread = radix_rank_type::digits_per_thread;

    union storage_type_
    {
        typename radix_rank_type::storage_type rank;
        struct
        {
            Offset global_digit_offsets[radix_size];
            union
            {
                Key          ordered_block_keys[items_per_block];
                Value        ordered_block_values[items_per_block];
            };
        };
    };

    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<bool IsFull,
             class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE void onesweep(KeysInputIterator        keys_input,
                                 KeysOutputIterator       keys_output,
                                 ValuesInputIterator      values_input,
                                 ValuesOutputIterator     values_output,
                                 Offset*                  global_digit_offsets_in,
                                 Offset*                  global_digit_offsets_out,
                                 onesweep_lookback_state* lookback_states,
                                 Decomposer               decomposer,
                                 const unsigned int       bit,
                                 const unsigned int       current_radix_bits,
                                 const unsigned int       valid_items,
                                 storage_type_&           storage)
    {
        const unsigned int flat_id      = ::rocprim::detail::block_thread_id<0>();
        const unsigned int block_id     = ::rocprim::detail::block_id<0>();
        const unsigned int block_offset = block_id * items_per_block;

        // Load keys into private memory, and encode them to unsigned integers.
        Key keys[ItemsPerThread];
        if ROCPRIM_IF_CONSTEXPR(IsFull)
        {
            if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
            {
                block_load_direct_warp_striped(flat_id, keys_input + block_offset, keys);
            }
            else
            {
                block_load_direct_blocked(flat_id, keys_input + block_offset, keys);
            }
        }
        else
        {
            // Fill the out-of-bounds elements of the key array with the key value with
            // the largest digit. This will make sure they are sorted (ranked) last, and
            // thus will be omitted when we compare the item offset against `valid_items` later.
            // Note that this will lead to an incorrect digit count. Since this is the very last digit,
            // it does not matter. It does cause the final digit offset to be increased past its end,
            // but again this does not matter since this is the last iteration in which it will be used anyway.
            const Key out_of_bounds = key_codec::get_out_of_bounds_key(decomposer);
            if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
            {
                block_load_direct_warp_striped(flat_id,
                                               keys_input + block_offset,
                                               keys,
                                               valid_items,
                                               out_of_bounds);
            }
            else
            {
                block_load_direct_blocked(flat_id,
                                          keys_input + block_offset,
                                          keys,
                                          valid_items,
                                          out_of_bounds);
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            key_codec::encode_inplace(keys[i], decomposer);
        }

        // Compute the block-based key ranks, the digit counts, and the prefix sum of the digit counts.
        unsigned int ranks[ItemsPerThread];
        // Tile-wide digit offset
        unsigned int exclusive_digit_prefix[digits_per_thread];
        // Tile-wide digit count
        unsigned int digit_counts[digits_per_thread];
        radix_rank_type{}.rank_keys(
            keys,
            ranks,
            storage.rank,
            [bit, current_radix_bits, decomposer](const Key& key)
            { return key_codec::extract_digit(key, bit, current_radix_bits, decomposer); },
            exclusive_digit_prefix,
            digit_counts);

        ::rocprim::syncthreads();

        // Order keys in shared memory.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            storage.ordered_block_keys[ranks[i]] = keys[i];
        }

        ::rocprim::syncthreads();

        // Compute the global prefix for each histogram.
        // At this point `lookback_states` already hold `onesweep_lookback_state::EMPTY`.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < digits_per_thread; ++i)
        {
            const unsigned int digit = flat_id * digits_per_thread + i;
            if(radix_size % BlockSize == 0 || digit < radix_size)
            {
                onesweep_lookback_state* block_state
                    = &lookback_states[block_id * radix_size + digit];
                onesweep_lookback_state(onesweep_lookback_state::PARTIAL, digit_counts[i])
                    .store(block_state);

                unsigned int exclusive_prefix  = 0;
                unsigned int lookback_block_id = block_id;
                // The main back tracking loop.
                while(lookback_block_id > 0)
                {
                    --lookback_block_id;
                    onesweep_lookback_state* lookback_state_ptr
                        = &lookback_states[lookback_block_id * radix_size + digit];
                    onesweep_lookback_state lookback_state
                        = onesweep_lookback_state::load(lookback_state_ptr);
                    while(lookback_state.status() == onesweep_lookback_state::EMPTY)
                    {
                        lookback_state = onesweep_lookback_state::load(lookback_state_ptr);
                    }

                    exclusive_prefix += lookback_state.value();
                    if(lookback_state.status() == onesweep_lookback_state::COMPLETE)
                    {
                        break;
                    }
                }

                // Update the state for the current block.
                const unsigned int inclusive_digit_prefix = exclusive_prefix + digit_counts[i];
                // Note that this should not deadlock, as HSA guarantees that blocks with a lower block ID launch before
                // those with a higher block id.
                onesweep_lookback_state(onesweep_lookback_state::COMPLETE, inclusive_digit_prefix)
                    .store(block_state);

                // Subtract the exclusive digit prefix from the global offset here, since we already ordered the keys in shared
                // memory.
                storage.global_digit_offsets[digit]
                    = global_digit_offsets_in[digit] - exclusive_digit_prefix[i] + exclusive_prefix;
            }
        }

        ::rocprim::syncthreads();

        // Scatter the keys to global memory in a sorted fashion.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int rank = i * BlockSize + flat_id;
            if(IsFull || rank < valid_items)
            {
                Key                key = storage.ordered_block_keys[rank];
                const unsigned int digit
                    = key_codec::extract_digit(key, bit, current_radix_bits, decomposer);
                key_codec::decode_inplace(key, decomposer);
                const Offset global_offset        = storage.global_digit_offsets[digit];
                keys_output[rank + global_offset] = key;
            }
        }

        // Gather and scatter values if necessary.
        if(with_values)
        {
            Value values[ItemsPerThread];
            if ROCPRIM_IF_CONSTEXPR(IsFull)
            {
                if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                {
                    block_load_direct_warp_striped(flat_id, values_input + block_offset, values);
                }
                else
                {
                    block_load_direct_blocked(flat_id, values_input + block_offset, values);
                }
            }
            else
            {
                if ROCPRIM_IF_CONSTEXPR(load_warp_striped)
                {
                    block_load_direct_warp_striped(flat_id,
                                                   values_input + block_offset,
                                                   values,
                                                   valid_items);
                }
                else
                {
                    block_load_direct_blocked(flat_id,
                                              values_input + block_offset,
                                              values,
                                              valid_items);
                }
            }

            // Compute digits up-front so that we can re-use shared memory between ordered_block_keys and
            // ordered_block_values.
            unsigned int digits[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                const unsigned int rank = i * BlockSize + flat_id;
                if(IsFull || rank < valid_items)
                {
                    const Key key = storage.ordered_block_keys[rank];
                    digits[i] = key_codec::extract_digit(key, bit, current_radix_bits, decomposer);
                }
            }

            ::rocprim::syncthreads();

            // Order values in shared memory
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                storage.ordered_block_values[ranks[i]] = values[i];
            }

            ::rocprim::syncthreads();

            // And scatter the values to global memory.
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                const unsigned int rank = i * BlockSize + flat_id;
                if(IsFull || rank < valid_items)
                {
                    const Value  value                  = storage.ordered_block_values[rank];
                    const Offset global_offset          = storage.global_digit_offsets[digits[i]];
                    values_output[rank + global_offset] = value;
                }
            }
        }

        // Update the global digit offset if we are batching
        const bool is_last_block = block_id == rocprim::detail::grid_size<0>() - 1;
        if(is_last_block)
        {
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < digits_per_thread; ++i)
            {
                const unsigned int digit = flat_id * digits_per_thread + i;
                if(radix_size % BlockSize == 0 || digit < radix_size)
                {
                    global_digit_offsets_out[digit] = storage.global_digit_offsets[digit]
                                                      + exclusive_digit_prefix[i] + digit_counts[i];
                }
            }
        }
    }
};

template<unsigned int               BlockSize,
         unsigned int               ItemsPerThread,
         unsigned int               RadixBits,
         bool                       Descending,
         block_radix_rank_algorithm RadixRankAlgorithm,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Offset,
         class Decomposer>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    onesweep_iteration(KeysInputIterator        keys_input,
                       KeysOutputIterator       keys_output,
                       ValuesInputIterator      values_input,
                       ValuesOutputIterator     values_output,
                       const unsigned int       size,
                       Offset*                  global_digit_offsets_in,
                       Offset*                  global_digit_offsets_out,
                       onesweep_lookback_state* lookback_states,
                       Decomposer               decomposer,
                       const unsigned int       bit,
                       const unsigned int       current_radix_bits,
                       const unsigned int       full_blocks)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using onesweep_iteration_helper_type = onesweep_iteration_helper<key_type,
                                                                     value_type,
                                                                     Offset,
                                                                     BlockSize,
                                                                     ItemsPerThread,
                                                                     RadixBits,
                                                                     Descending,
                                                                     RadixRankAlgorithm,
                                                                     Decomposer>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     block_id        = ::rocprim::detail::block_id<0>();

    ROCPRIM_SHARED_MEMORY typename onesweep_iteration_helper_type::storage_type storage;

    if(block_id < full_blocks)
    {
        onesweep_iteration_helper_type{}.template onesweep<true>(keys_input,
                                                                 keys_output,
                                                                 values_input,
                                                                 values_output,
                                                                 global_digit_offsets_in,
                                                                 global_digit_offsets_out,
                                                                 lookback_states,
                                                                 decomposer,
                                                                 bit,
                                                                 current_radix_bits,
                                                                 items_per_block,
                                                                 storage.get());
    }
    else
    {
        const unsigned int valid_in_last_block = size - items_per_block * full_blocks;
        onesweep_iteration_helper_type{}.template onesweep<false>(keys_input,
                                                                  keys_output,
                                                                  values_input,
                                                                  values_output,
                                                                  global_digit_offsets_in,
                                                                  global_digit_offsets_out,
                                                                  lookback_states,
                                                                  decomposer,
                                                                  bit,
                                                                  current_radix_bits,
                                                                  valid_in_last_block,
                                                                  storage.get());
    }
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_
