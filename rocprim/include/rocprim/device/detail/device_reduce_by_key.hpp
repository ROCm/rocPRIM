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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_

#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"
#include "../../detail/match_result_type.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics/thread.hpp"

#include "../../config.hpp"

#include <iterator>
#include <utility>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

namespace reduce_by_key
{
template<typename KeyType,
         typename AccumulatorType,
         unsigned int      BlockSize,
         unsigned int      ItemsPerThread,
         block_load_method load_keys_method,
         block_load_method load_values_method>
struct load_helper
{
    using block_load_keys = block_load<KeyType, BlockSize, ItemsPerThread, load_keys_method>;
    using block_load_values
        = block_load<AccumulatorType, BlockSize, ItemsPerThread, load_values_method>;
    union storage_type
    {
        typename block_load_keys::storage_type   keys;
        typename block_load_values::storage_type values;
    };

    template<typename KeyIterator, typename ValueIterator>
    ROCPRIM_DEVICE void load_keys_values(KeyIterator        block_keys,
                                         ValueIterator      block_values,
                                         const bool         is_last_block,
                                         const unsigned int valid_in_last_block,
                                         KeyType (&keys)[ItemsPerThread],
                                         AccumulatorType (&values)[ItemsPerThread],
                                         storage_type& storage)
    {
        if(!is_last_block)
        {
            block_load_keys{}.load(block_keys, keys, storage.keys);
            ::rocprim::syncthreads();
            block_load_values{}.load(block_values, values, storage.values);
        }
        else
        {
            block_load_keys{}.load(
                block_keys,
                keys,
                valid_in_last_block,
                *block_keys, // Any value is okay, so discontinuity doesn't access undefined items
                storage.keys);
            ::rocprim::syncthreads();

            block_load_values{}.load(block_values, values, valid_in_last_block, storage.values);
        }
    }
};

template<typename KeyType, unsigned int BlockSize>
struct discontinuity_helper
{
    using block_discontinuity_type = block_discontinuity<KeyType, BlockSize>;
    using storage_type             = typename block_discontinuity_type::storage_type;

    template<typename KeyIterator, typename CompareFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void flag_heads_and_tails(KeyIterator block_keys,
                                             const KeyType (&keys)[ItemsPerThread],
                                             CompareFunction compare,
                                             unsigned int (&head_flags)[ItemsPerThread],
                                             bool (&tail_flags)[ItemsPerThread],
                                             const bool         is_first_block,
                                             const bool         is_last_block,
                                             const unsigned int valid_in_last_block,
                                             const unsigned int flat_thread_id,
                                             storage_type&      storage)
    {
        static constexpr auto items_per_block = BlockSize * ItemsPerThread;

        auto not_equal = [compare](const auto& a, const auto& b) mutable { return !compare(a, b); };

        if(!is_first_block)
        {
            const KeyType tile_predecessor = block_keys[-1];
            block_discontinuity_type{}.flag_heads(head_flags,
                                                  tile_predecessor,
                                                  keys,
                                                  not_equal,
                                                  storage);
        }
        else
        {
            block_discontinuity_type{}.flag_heads(head_flags, keys, not_equal, storage);
        }
        ::rocprim::syncthreads();

        if(!is_last_block)
        {
            const KeyType tile_successor = block_keys[items_per_block];
            block_discontinuity_type{}.flag_tails(tail_flags,
                                                  tile_successor,
                                                  keys,
                                                  not_equal,
                                                  storage);
        }
        else
        {
            block_discontinuity_type{}.flag_tails(tail_flags, keys, not_equal, storage);
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                const unsigned int index = flat_thread_id * ItemsPerThread + i;
                if(index == valid_in_last_block - 1)
                {
                    tail_flags[i] = true; // Very last item is the end of a sequence, must be saved
                }
                else if(index >= valid_in_last_block)
                {
                    head_flags[i] = 0;
                    tail_flags[i] = false;
                }
            }
        }
    }
};

template<typename ValueType, unsigned int BlockSize, unsigned int ItemsPerThread>
struct scatter_helper
{
    using storage_type = detail::raw_storage<ValueType[BlockSize * ItemsPerThread]>;

    template<typename ValueIterator, typename Flag, typename ValueFunction, typename IndexFunction>
    ROCPRIM_DEVICE void scatter(ValueIterator   block_values,
                                ValueFunction&& values,
                                const Flag (&is_selected)[ItemsPerThread],
                                IndexFunction&&    block_indices,
                                const unsigned int selected_in_block,
                                const unsigned int flat_thread_id,
                                storage_type&      storage)
    {
        if(selected_in_block > BlockSize)
        {
            auto& scatter_storage = storage.get();
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                if(is_selected[i])
                {
                    scatter_storage[block_indices(i)] = values(i);
                }
            }
            ::rocprim::syncthreads();

            // Coalesced write from shared memory to global memory
            for(unsigned int i = flat_thread_id; i < selected_in_block; i += BlockSize)
            {
                block_values[i] = scatter_storage[i];
            }
        }
        else
        {
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                if(is_selected[i])
                {
                    block_values[block_indices(i)] = values(i);
                }
            }
        }
    }
};

template<typename Iterator>
using value_type = typename std::iterator_traits<Iterator>::value_type;

template<typename Iterator>
using prev_value_type = rocprim::tuple<value_type<Iterator>, std::size_t>;

template<typename Config,
         typename KeyIterator,
         typename ValueIterator,
         typename UniqueIterator,
         typename ReductionIterator,
         typename UniqueCountIterator,
         typename CompareFunction,
         typename BinaryOp,
         typename LookBackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    kernel_impl(const KeyIterator              keys_input,
                const ValueIterator            values_input,
                UniqueIterator                 unique_keys,
                ReductionIterator              reductions,
                UniqueCountIterator            unique_count,
                BinaryOp                       reduce_op,
                const CompareFunction          compare,
                LookBackScanState              scan_state,
                ordered_block_id<unsigned int> ordered_bid,
                const std::size_t              starting_block,
                const std::size_t              number_of_blocks,
                const std::size_t              size //,
                /*const prev_value_type<KeyIterator> previous_last_value*/)
{
    using accumulator_type =
        typename detail::match_result_type<value_type<ValueIterator>, BinaryOp>::type;

    static constexpr unsigned int block_size       = Config::block_size;
    static constexpr unsigned int items_per_thread = Config::items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;

    static constexpr block_load_method load_keys_method   = Config::load_keys_method;
    static constexpr block_load_method load_values_method = Config::load_values_method;

    using key_type     = value_type<KeyIterator>;
    using wrapped_type = rocprim::tuple<accumulator_type, unsigned int>;

    using load_type = reduce_by_key::load_helper<key_type,
                                                 accumulator_type,
                                                 block_size,
                                                 items_per_thread,
                                                 load_keys_method,
                                                 load_values_method>;

    using discontinuity_type = reduce_by_key::discontinuity_helper<key_type, block_size>;
    using block_scan_type    = rocprim::block_scan<wrapped_type, block_size>;

    // Modified binary operation that respects segment boundaries and counts segments
    auto wrapped_op = [&](const wrapped_type& lhs, const wrapped_type& rhs)
    {
        return wrapped_type{rocprim::get<1>(rhs) == 0
                                ? reduce_op(rocprim::get<0>(lhs), rocprim::get<0>(rhs))
                                : rocprim::get<0>(rhs),
                            rocprim::get<1>(lhs) + rocprim::get<1>(rhs)};
    };

    using prefix_op_type = detail::
        offset_lookback_scan_prefix_op<wrapped_type, decltype(scan_state), decltype(wrapped_op)>;

    using scatter_keys_type = reduce_by_key::scatter_helper<key_type, block_size, items_per_thread>;
    using scatter_values_type
        = reduce_by_key::scatter_helper<accumulator_type, block_size, items_per_thread>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename ordered_block_id<unsigned int>::storage_type block_id;
        typename load_type::storage_type                      load;
        typename discontinuity_type::storage_type             flags;
        struct
        {
            typename prefix_op_type::storage_type  prefix;
            typename block_scan_type::storage_type scan;
        } scan;
        struct
        {
            bool first_segment_starts_before_block;
            bool last_segment_ends_after_block;
        } block_boundary;
        typename scatter_keys_type::storage_type   scatter_keys;
        typename scatter_values_type::storage_type scatter_values;
    } storage;

    const unsigned int flat_thread_id = threadIdx.x;
    const unsigned int flat_block_id  = ordered_bid.get(flat_thread_id, storage.block_id);

    const bool         is_first_block = starting_block + flat_block_id == 0;
    const bool         is_last_block  = starting_block + flat_block_id == number_of_blocks - 1;
    const unsigned int valid_in_last_block
        = static_cast<unsigned int>(size - (number_of_blocks - 1) * items_per_block);

    const unsigned int  block_offset = flat_block_id * items_per_block;
    const KeyIterator   block_keys   = keys_input + block_offset;
    const ValueIterator block_values = values_input + block_offset;

    key_type         keys[items_per_thread];
    accumulator_type values[items_per_thread];

    ::rocprim::syncthreads();
    load_type{}.load_keys_values(block_keys,
                                 block_values,
                                 is_last_block,
                                 valid_in_last_block,
                                 keys,
                                 values,
                                 storage.load);
    ::rocprim::syncthreads();

    unsigned int head_flags[items_per_thread];
    bool         tail_flags[items_per_thread];
    discontinuity_type{}.flag_heads_and_tails(block_keys,
                                              keys,
                                              compare,
                                              head_flags,
                                              tail_flags,
                                              is_first_block,
                                              is_last_block,
                                              valid_in_last_block,
                                              flat_thread_id,
                                              storage.flags);

    wrapped_type wrapped_values[items_per_thread];
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        rocprim::get<0>(wrapped_values[i]) = values[i];
        rocprim::get<1>(wrapped_values[i]) = head_flags[i];
    }
    ::rocprim::syncthreads();

    // TODO: Refactor scan to separate function
    unsigned int segment_heads_before   = 0;
    unsigned int segment_heads_in_block = 0;
    if(flat_block_id == 0)
    {
        // TODO: Handle previous grid launch, i.e. large indices
        wrapped_type reduction;
        block_scan_type{}.inclusive_scan(wrapped_values,
                                         wrapped_values,
                                         reduction,
                                         storage.scan.scan,
                                         wrapped_op);

        if(flat_thread_id == 0)
        {
            scan_state.set_complete(flat_block_id, reduction);
        }

        segment_heads_in_block = rocprim::get<1>(reduction);
    }
    else
    {
        auto prefix_op = prefix_op_type{flat_block_id, scan_state, storage.scan.prefix, wrapped_op};

        block_scan_type{}.inclusive_scan(wrapped_values,
                                         wrapped_values,
                                         storage.scan.scan,
                                         prefix_op,
                                         wrapped_op);
        rocprim::syncthreads();

        segment_heads_before   = rocprim::get<1>(prefix_op.get_prefix());
        segment_heads_in_block = rocprim::get<1>(prefix_op.get_reduction());
    }
    rocprim::syncthreads();

    if(flat_thread_id == 0)
    {
        storage.block_boundary.first_segment_starts_before_block = head_flags[0] == 0;
    }

    if(!is_last_block && flat_thread_id == block_size - 1)
    {
        storage.block_boundary.last_segment_ends_after_block = !tail_flags[items_per_thread - 1];
    }
    else if(is_last_block && flat_thread_id == 0)
    {
        storage.block_boundary.last_segment_ends_after_block = false;
    }
    rocprim::syncthreads();

    const unsigned int segment_tails_in_block
        = segment_heads_in_block
          + (storage.block_boundary.first_segment_starts_before_block ? 1 : 0)
          - (storage.block_boundary.last_segment_ends_after_block ? 1 : 0);

    const unsigned int segment_tails_before
        = segment_heads_before - (storage.block_boundary.first_segment_starts_before_block ? 1 : 0);
    rocprim::syncthreads();

    if(segment_heads_in_block > 0)
    {
        const auto get_segment_index_for_head = [&](const unsigned int i)
        { return rocprim::get<1>(wrapped_values[i]) - segment_heads_before - head_flags[i]; };
        scatter_keys_type{}.scatter(
            unique_keys + segment_heads_before,
            [&keys](unsigned int i) { return keys[i]; },
            head_flags,
            get_segment_index_for_head,
            segment_heads_in_block,
            flat_thread_id,
            storage.scatter_keys);
    }

    if(segment_tails_in_block > 0)
    {
        rocprim::syncthreads();

        const auto get_segment_index_for_tail = [&](const unsigned int i)
        { return rocprim::get<1>(wrapped_values[i]) - segment_tails_before - 1; };
        scatter_values_type{}.scatter(
            reductions + segment_tails_before,
            [&wrapped_values](unsigned int i) { return rocprim::get<0>(wrapped_values[i]); },
            tail_flags,
            get_segment_index_for_tail,
            segment_tails_in_block,
            flat_thread_id,
            storage.scatter_values);
    }

    if(is_last_block && flat_thread_id == 0)
    {
        // TODO: Handle large indices
        *unique_count = segment_heads_before + segment_heads_in_block;
    }
}
} // namespace reduce_by_key

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_
