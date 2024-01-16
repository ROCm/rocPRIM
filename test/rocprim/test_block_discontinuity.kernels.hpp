// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_
#define TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_

template<class T>
struct custom_flag_op1
{
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b, unsigned int b_index) const
    {
        return (a == b) || (b_index % 10 == 0);
    }
};

struct custom_flag_op2
{
    template<class T>
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return (a - b > 5);
    }
};

// Host (CPU) implementaions of the wrapping function that allows to pass 3 args
template <class T, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index)
    -> decltype(flag_op(a, b, b_index))
{
    return flag_op(a, b, b_index);
}

template<class T, class FlagOp>
auto apply(FlagOp flag_op, const T& a, const T& b, unsigned int) -> decltype(flag_op(a, b))
{
    return flag_op(a, b);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_kernel(Type* device_input, long long* device_heads)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads(head_flags, tile_predecessor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_heads(head_flags, input, FlagOpType());
    }

    rocprim::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_tails_kernel(Type* device_input, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_tails(tail_flags, tile_successor_item, input, FlagOpType());
    }
    else
    {
        bdiscontinuity.flag_tails(tail_flags, input, FlagOpType());
    }

    rocprim::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_and_tails_kernel(Type* device_input, long long* device_heads, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_discontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 3)
    {
        bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, FlagOpType());
    }

    rocprim::block_store_direct_blocked(lid, device_heads + block_offset, head_flags);
    rocprim::block_store_direct_blocked(lid, device_tails + block_offset, tail_flags);
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 0>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> heads(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = bi % 2 == 1
                        ? apply(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_heads, heads.size() * sizeof(typename decltype(heads)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_heads_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_heads
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_heads));
    }

}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 1>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = bi % 2 == 0
                        ? apply(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_tails;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_tails
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_tails));
    }

}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
auto test_block_discontinuity()
-> typename std::enable_if<Method == 2>::type
{
    using type = Type;
    // std::vector<bool> is a special case that will cause an error in hipMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, FlagType>::value,
                               int,
                               FlagType
                           >::type;
    using flag_type = FlagType;
    using flag_op_type = FlagOpType;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 20;
    static constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<long long> heads(size);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = (bi % 4 == 1 || bi % 4 == 2)
                        ? apply(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = (bi % 4 == 0 || bi % 4 == 1)
                        ? apply(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_heads, tails.size() * sizeof(typename decltype(heads)::value_type)));
        long long* device_tails;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flag_heads_and_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_heads, device_tails
        );
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_heads));
        HIP_CHECK(hipFree(device_tails));
    }

}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize = 256U
>
struct static_for
{
    static void run()
    {
        int device_id = test_common_utils::obtain_device_from_ctest();
        SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
        HIP_CHECK(hipSetDevice(device_id));

        test_block_discontinuity<Type, FlagType, FlagOpType, Method, BlockSize, items[First]>();
        static_for<First + 1, Last, Type, FlagType, FlagOpType, Method, BlockSize>::run();
    }
};

template <
    unsigned int N,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize
>
struct static_for<N, N, Type, FlagType, FlagOpType, Method, BlockSize>
{
    static void run()
    {
    }
};

#endif // TEST_BLOCK_DISCONTINUITY_KERNELS_HPP_
