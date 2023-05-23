// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_utils_sort_comparator.hpp"

block_sort_test_suite_type_def(suite_name, name_suffix)

    typed_test_suite_def(suite_name, name_suffix, block_params);

// using header guards for these test functions because this file is included multiple times:
// once for the integrals test suite and once for the floating point test suite.
#ifndef TEST_ROCPRIM_TEST_BLOCK_SORT_HPP_
    #define TEST_ROCPRIM_TEST_BLOCK_SORT_HPP_

#include <typeinfo>

template<unsigned int block_size,
         unsigned int items_per_thread,
         class key_type,
         class value_type,
         rocprim::block_sort_algorithm algo,
         class binary_op_type>
void TestSortKey(std::vector<size_t> sizes)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    static constexpr const unsigned int items_per_block = block_size * items_per_thread;
    hipStream_t                         stream          = 0; // default

    if(!is_buildable(block_size, items_per_thread, algo))
    {
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            if(size == 0 && test_common_utils::use_hmm())
            {
                // hipMallocManaged() currently doesnt support zero byte allocation
                continue;
            }
            // Generate data
            std::vector<key_type> output
                = test_utils::get_random_data<key_type>(size, -100, 100, seed_value);

            // Calculate expected results on host
            std::vector<key_type> expected(output);
            binary_op_type        binary_op;

            for(size_t i = 0; i < rocprim::detail::ceiling_div(output.size(), items_per_block); i++)
            {
                std::sort(expected.begin() + (i * items_per_block),
                          expected.begin() + std::min(size, ((i + 1) * items_per_block)),
                          binary_op);
            }

            // Preparing device
            key_type* device_key_output;
            key_type asd;
            std::cout << "Datatype:" <<  typeid(asd).name() << " Block Size:" << block_size << " ItemsPerThread:" << items_per_thread << " Data size:" << size << " Seed Value:" << seed_value << std::endl;
            HIP_CHECK(test_common_utils::hipMallocHelper(&device_key_output,
                                                         output.size() * sizeof(key_type)));

            HIP_CHECK(hipMemcpy(device_key_output,
                                output.data(),
                                output.size() * sizeof(key_type),
                                hipMemcpyHostToDevice));

            const unsigned int grid_size = rocprim::detail::ceiling_div(size, items_per_block);
            // Running kernel, ignored if invalid size
            if(size > 0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_keys_kernel<block_size,
                                                                    items_per_thread,
                                                                    key_type*,
                                                                    algo,
                                                                    binary_op_type>),
                                   dim3(grid_size),
                                   dim3(block_size),
                                   0,
                                   stream,
                                   device_key_output,
                                   size);
            }

            // Reading results back
            HIP_CHECK(hipMemcpy(output.data(),
                                device_key_output,
                                output.size() * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            test_utils::assert_eq(output, expected);

            HIP_CHECK(hipFree(device_key_output));
        }
    }
}

struct less_tuple
{
    template<class A, class B>
    ROCPRIM_HOST_DEVICE inline constexpr bool operator()(const rocprim::tuple<A, B>& a,
                                                         const rocprim::tuple<A, B>& b) const
    {
        return rocprim::get<0>(a) < rocprim::get<0>(b);
    }
};
#endif

typed_test_def(suite_name, name_suffix, SortKeyInputSizeNotMultipleOfBlockSize)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 1;
    if(algo == rocprim::block_sort_algorithm::bitonic_sort && items_per_thread != 1u)
    {
        GTEST_SKIP();
    }
    std::vector<size_t> sizes
        = {0, 53, 512, 5000, 34567, (1 << 17) - 1220, 1134 * 256, (1 << 20) - 123};
    TestSortKey<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>(sizes);
}
