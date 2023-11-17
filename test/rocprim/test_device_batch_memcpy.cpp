// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "rocprim/detail/various.hpp"
#include "rocprim/device/device_memcpy.hpp"
#include "rocprim/intrinsics/thread.hpp"

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

#include <cstring>

#include <stdint.h>

template<class ValueType,
         class SizeType,
         bool     Shuffled   = false,
         uint32_t NumBuffers = 1024,
         uint32_t MaxSize    = 4 * 1024>
struct DeviceBatchMemcpyParams
{
    using value_type                      = ValueType;
    using size_type                       = SizeType;
    static constexpr bool     shuffled    = Shuffled;
    static constexpr uint32_t num_buffers = NumBuffers;
    static constexpr uint32_t max_size    = MaxSize;
};

template<class Params>
struct DeviceBatchMemcpyTests : public ::testing::Test
{
    using value_type                      = typename Params::value_type;
    using size_type                       = typename Params::size_type;
    static constexpr bool     shuffled    = Params::shuffled;
    static constexpr uint32_t num_buffers = Params::num_buffers;
    static constexpr uint32_t max_size    = Params::max_size;
};

typedef ::testing::Types<
    // Ignore copy/move
    DeviceBatchMemcpyParams<test_utils::custom_non_copyable_type<uint8_t>, uint32_t, false>,
    DeviceBatchMemcpyParams<test_utils::custom_non_moveable_type<uint8_t>, uint32_t, false>,
    DeviceBatchMemcpyParams<test_utils::custom_non_default_type<uint8_t>, uint32_t, false>,

    // Unshuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false>,
    DeviceBatchMemcpyParams<uint32_t, uint32_t, false>,
    DeviceBatchMemcpyParams<uint64_t, uint32_t, false>,
    // size_type: uint16_t
    DeviceBatchMemcpyParams<uint8_t, uint16_t, false, 1024, 1024>,
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, false, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, false, 1024, 128 * 1024>,

    // weird amount of buffers
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 3 * 1023>,
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 3 * 1025>,
    DeviceBatchMemcpyParams<uint8_t, uint32_t, false, 1024 * 1024, 256>,

    // Shuffled inputs and outputs
    // Variable value_type
    DeviceBatchMemcpyParams<uint8_t, uint32_t, true>,
    DeviceBatchMemcpyParams<uint32_t, uint32_t, true>,
    DeviceBatchMemcpyParams<uint64_t, uint32_t, true>,
    // size_type: uint16_t
    DeviceBatchMemcpyParams<uint8_t, uint16_t, true, 1024, 1024>,
    // size_type: int64_t
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, 1024, 64 * 1024>,
    DeviceBatchMemcpyParams<uint8_t, int64_t, true, 1024, 128 * 1024>>
    DeviceBatchMemcpyTestsParams;

TYPED_TEST_SUITE(DeviceBatchMemcpyTests, DeviceBatchMemcpyTestsParams);

// Used for generating offsets. We generate a permutation map and then derive
// offsets via a sum scan over the sizes in the order of the permutation. This
// allows us to keep the order of buffers we pass to batch_memcpy, but still
// have source and destinations mappings not be the identity function:
//
//  batch_memcpy(
//    [&a0 , &b0 , &c0 , &d0 ], // from (note the order is still just a, b, c, d!)
//    [&a0', &b0', &c0', &d0'], // to   (order is the same as above too!)
//    [3   , 2   , 1   , 2   ]) // size
//
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │b0 │b1 │a0 │a1 │a2 │d0 │d1 │c0 │ buffer x contains buffers a, b, c, d
// └───┴───┴───┴───┴───┴───┴───┴───┘ note that the order of buffers is shuffled!
//  ───┬─── ─────┬───── ───┬─── ───
//     └─────────┼─────────┼───┐
//           ┌───┘     ┌───┘   │ what batch_memcpy does
//           ▼         ▼       ▼
//  ─── ─────────── ─────── ───────
// ┌───┬───┬───┬───┬───┬───┬───┬───┐
// │c0'│a0'│a1'│a2'│d0'│d1'│b0'│b1'│ buffer y contains buffers a', b', c', d'
// └───┴───┴───┴───┴───┴───┴───┴───┘
template<class T, class S, class RandomGenerator>
std::vector<T> shuffled_exclusive_scan(const std::vector<S>& input, RandomGenerator& rng)
{
    const size_t n = input.size();
    assert(n > 0);

    std::vector<T> result(n);
    std::vector<T> permute(n);

    std::iota(permute.begin(), permute.end(), 0);
    std::shuffle(permute.begin(), permute.end(), rng);

    T sum = 0;
    for(size_t i = 0; i < n; ++i)
    {
        result[permute[i]] = sum;
        sum += input[permute[i]];
    }

    return result;
}

TYPED_TEST(DeviceBatchMemcpyTests, SizeAndTypeVariation)
{
    using value_type         = typename TestFixture::value_type;
    using buffer_size_type   = typename TestFixture::size_type;
    using buffer_offset_type = uint32_t;
    using byte_offset_type   = size_t;

    constexpr int32_t num_buffers = TestFixture::num_buffers;
    constexpr int32_t max_size    = TestFixture::max_size;
    constexpr bool    shuffled    = TestFixture::shuffled;

    constexpr int32_t wlev_min_size = rocprim::batch_memcpy_config<>::wlev_size_threshold;
    constexpr int32_t blev_min_size = rocprim::batch_memcpy_config<>::blev_size_threshold;

    constexpr int32_t wlev_min_elems
        = rocprim::detail::ceiling_div(wlev_min_size, sizeof(value_type));
    constexpr int32_t blev_min_elems
        = rocprim::detail::ceiling_div(blev_min_size, sizeof(value_type));
    constexpr int32_t max_elems = max_size / sizeof(value_type);

    constexpr int32_t enabled_size_categories
        = (blev_min_elems <= max_elems) + (wlev_min_elems <= max_elems) + 1;

    constexpr int32_t num_blev
        = blev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int32_t num_wlev
        = wlev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int32_t num_tlev = num_buffers - num_blev - num_wlev;

    // Get random buffer sizes
    uint32_t seed = 0;
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);
    std::mt19937_64 rng{seed};

    std::vector<buffer_size_type> h_buffer_num_elements(num_buffers);

    auto iter = h_buffer_num_elements.begin();

    iter = test_utils::generate_random_data_n(iter, num_tlev, 1, wlev_min_elems - 1, rng);
    iter = test_utils::generate_random_data_n(iter,
                                              num_wlev,
                                              wlev_min_elems,
                                              blev_min_elems - 1,
                                              rng);
    iter = test_utils::generate_random_data_n(iter, num_blev, blev_min_elems, max_elems, rng);

    const byte_offset_type total_num_elements = std::accumulate(h_buffer_num_elements.begin(),
                                                                h_buffer_num_elements.end(),
                                                                byte_offset_type{0});

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // Get the byte size of each buffer
    std::vector<buffer_size_type> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(value_type);
    }

    // And the total byte size
    const byte_offset_type total_num_bytes = total_num_elements * sizeof(value_type);

    // Device pointers
    value_type*       d_input        = nullptr;
    value_type*       d_output       = nullptr;
    value_type**      d_buffer_srcs  = nullptr;
    value_type**      d_buffer_dsts  = nullptr;
    buffer_size_type* d_buffer_sizes = nullptr;

    // Calculate temporary storage

    size_t temp_storage_bytes = 0;

    HIP_CHECK(rocprim::batch_memcpy(nullptr,
                                    temp_storage_bytes,
                                    d_buffer_srcs,
                                    d_buffer_dsts,
                                    d_buffer_sizes,
                                    num_buffers));

    void* d_temp_storage = nullptr;

    // Allocate memory.
    HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
    HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

    HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(*d_buffer_srcs)));
    HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(*d_buffer_dsts)));
    HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(*d_buffer_sizes)));

    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Generate data.
    std::independent_bits_engine<std::mt19937_64, 64, uint64_t> bits_engine{rng};

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    auto         h_input  = std::make_unique<unsigned char[]>(num_ints * sizeof(uint64_t));

    // generate_n for uninitialized memory, pragmatically use placement-new, since there are no
    // uint64_t objects alive yet in the storage.
    std::for_each(reinterpret_cast<uint64_t*>(h_input.get()),
                  reinterpret_cast<uint64_t*>(h_input.get() + num_ints * sizeof(uint64_t)),
                  [&bits_engine](uint64_t& elem) { ::new(&elem) uint64_t{bits_engine()}; });

    // Generate the source and shuffled destination offsets.
    std::vector<buffer_offset_type> src_offsets;
    std::vector<buffer_offset_type> dst_offsets;

    if(shuffled)
    {
        src_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
        dst_offsets = shuffled_exclusive_scan<buffer_offset_type>(h_buffer_num_elements, rng);
    }
    else
    {
        src_offsets = std::vector<buffer_offset_type>(num_buffers);
        dst_offsets = std::vector<buffer_offset_type>(num_buffers);

        // Consecutive offsets (no shuffling).
        // src/dst offsets first element is 0, so skip that!
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         src_offsets.begin() + 1);
        std::partial_sum(h_buffer_num_elements.begin(),
                         h_buffer_num_elements.end() - 1,
                         dst_offsets.begin() + 1);
    }

    // Generate the source and destination pointers.
    std::vector<value_type*> h_buffer_srcs(num_buffers);
    std::vector<value_type*> h_buffer_dsts(num_buffers);

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = d_input + src_offsets[i];
        h_buffer_dsts[i] = d_output + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    HIP_CHECK(hipMemcpy(d_input, h_input.get(), total_num_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(*d_buffer_srcs),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(*d_buffer_dsts),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(*d_buffer_sizes),
                        hipMemcpyHostToDevice));

    // Run batched memcpy.
    HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                    temp_storage_bytes,
                                    d_buffer_srcs,
                                    d_buffer_dsts,
                                    d_buffer_sizes,
                                    num_buffers,
                                    hipStreamDefault));
    // Verify results.
    auto h_output = std::make_unique<unsigned char[]>(total_num_bytes);
    HIP_CHECK(hipMemcpy(h_output.get(), d_output, total_num_bytes, hipMemcpyDeviceToHost));

    for(int32_t i = 0; i < num_buffers; ++i)
    {
        ASSERT_EQ(std::memcmp(h_input.get() + src_offsets[i] * sizeof(value_type),
                              h_output.get() + dst_offsets[i] * sizeof(value_type),
                              h_buffer_num_bytes[i]),
                  0)
            << "with index = " << i;
    }

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_buffer_sizes));
    HIP_CHECK(hipFree(d_buffer_dsts));
    HIP_CHECK(hipFree(d_buffer_srcs));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
}
