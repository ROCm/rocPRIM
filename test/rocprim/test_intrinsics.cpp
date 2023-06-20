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

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/intrinsics/warp_shuffle.hpp>

// required test headers
#include "test_utils_types.hpp"

#include <bitset>
#include <random>

// An integer type large enough to hold a lane_mask_type of any device
using max_lane_mask_type = uint64_t;

constexpr static max_lane_mask_type all_lanes_active
    = std::numeric_limits<max_lane_mask_type>::max();

ROCPRIM_HOST_DEVICE bool is_lane_active(const max_lane_mask_type mask, const unsigned int lane)
{
    return (mask >> lane) & 1;
}

// Custom structure
struct custom_notaligned
{
    short i;
    double d;
    float f;
    unsigned int u;

    ROCPRIM_HOST_DEVICE constexpr custom_notaligned() : i(123), d(1234), f(12345), u(12345){};

    ROCPRIM_HOST_DEVICE constexpr custom_notaligned(short i, double d, float f, unsigned int u)
        : i(i), d(d), f(f), u(u)
    {}
};

ROCPRIM_HOST_DEVICE
inline bool operator==(const custom_notaligned& lhs,
                       const custom_notaligned& rhs)
{
    return lhs.i == rhs.i && lhs.d == rhs.d
        && lhs.f == rhs.f &&lhs.u == rhs.u;
}

// Custom structure aligned to 16 bytes
struct alignas(16) custom_16aligned
{
    int i;
    unsigned int u;
    float        f;

    ROCPRIM_HOST_DEVICE constexpr custom_16aligned() : i(123), u(1234), f(12345){};

    ROCPRIM_HOST_DEVICE constexpr custom_16aligned(int i, unsigned int u, float f)
        : i(i), u(u), f(f)
    {}
};

inline ROCPRIM_HOST_DEVICE
bool operator==(const custom_16aligned& lhs, const custom_16aligned& rhs)
{
    return lhs.i == rhs.i && lhs.f == rhs.f && lhs.u == rhs.u;
}

template<typename T>
struct test_type_helper
{
    /// Returns the 'zero' value for this type: In this representation, all of the type's
    /// bits should be initialized to 0. This function is here because some rocPRIM intrinsic operations
    /// zero-initialize a type like this.
    ROCPRIM_HOST_DEVICE constexpr static T zero()
    {
        return 0;
    }

    /// Returns an 'uninitialized' representation for this type: Some rocPRIM intrinsics are not supposed
    /// to touch inactive lanes. In the relevant test scenarios, an output value is first initialized with
    /// this 'initialized' value, so that these values can later be checked if they were really unaltered after
    /// the intrinsic operation.
    ROCPRIM_HOST_DEVICE constexpr static T uninitialized()
    {
        return 0xAA;
    }

    /// Initialize some random data for this type, of \p n elements and with random seed <tt>seed</tt>.
    static std::vector<T> get_random_data(size_t n, seed_type seed)
    {
        return test_utils::get_random_data<T>(n, static_cast<T>(-100), static_cast<T>(100), seed);
    }
};

template<>
struct test_type_helper<custom_notaligned>
{
    ROCPRIM_HOST_DEVICE constexpr static custom_notaligned zero()
    {
        return {0, 0, 0, 0};
    }

    ROCPRIM_HOST_DEVICE constexpr static custom_notaligned uninitialized()
    {
        return custom_notaligned(0xAA, 0xAA, 0xAA, 0xAA);
    }

    static std::vector<custom_notaligned> get_random_data(size_t n, seed_type seed)
    {
        std::vector<double> random_data
            = test_utils::get_random_data<double>(4 * n, -100, 100, seed);
        std::vector<custom_notaligned> result(n);
        for(size_t i = 0; i < result.size(); ++i)
        {
            result[i].i = static_cast<short>(random_data[i * 4]);
            result[i].d = random_data[i * 4 + 1];
            result[i].f = static_cast<float>(random_data[i * 4 + 2]);
            result[i].u = static_cast<unsigned int>(random_data[i * 4 + 3]);
        }

        return result;
    }
};

template<>
struct test_type_helper<custom_16aligned>
{
    ROCPRIM_HOST_DEVICE constexpr static custom_16aligned zero()
    {
        return {0, 0, 0};
    }

    ROCPRIM_HOST_DEVICE constexpr static custom_16aligned uninitialized()
    {
        return custom_16aligned(0xAA, 0xAA, 0xAA);
    }

    static std::vector<custom_16aligned> get_random_data(size_t n, seed_type seed)
    {
        std::vector<float> random_data = test_utils::get_random_data<float>(3 * n, -100, 100, seed);
        std::vector<custom_16aligned> result(n);
        for(size_t i = 0; i < result.size(); ++i)
        {
            result[i].i = static_cast<short>(random_data[i * 3]);
            result[i].u = static_cast<unsigned int>(random_data[i * 4 + 1]);
            result[i].f = random_data[i * 3 + 2];
        }

        return result;
    }
};

// Params for tests
template<class T>
struct params
{
    using type = T;
};

template<class Params>
class RocprimIntrinsicsTests : public ::testing::Test
{
public:
    using type = typename Params::type;
};

typedef ::testing::Types<params<int>,
                         params<float>,
                         params<double>,
                         params<unsigned char>,
                         params<custom_notaligned>,
                         params<custom_16aligned>>
    IntrinsicsTestParams;

TYPED_TEST_SUITE(RocprimIntrinsicsTests, IntrinsicsTestParams);

/// A safe helper function to extract the least-significant \p bits bits from \p value. Shifting more bits
/// than a type's width is undefined behavior on C++. With this function, it is safe to extract an amount of
/// bits less than or equal to the number of bits of type <tt>T</tt>.
template<typename T>
T bit_extract(const T value, const unsigned int bits)
{
    // Shifting an amount larger or equal to the amount of bits in a type is undefined behavior.
    const unsigned int bit_size = sizeof(T) * CHAR_BIT;
    return bits == bit_size ? value : value & ((T{1} << bits) - T{1});
}

std::vector<max_lane_mask_type> active_lanes_tests()
{
    std::vector<max_lane_mask_type> tests
        = {all_lanes_active, 0x0123'4567'89AB'CDEF, 0xAAAA'AAAA'AAAA'AAAA};

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    for(auto& test : tests)
    {
        test = bit_extract(test, hardware_warp_size);
    }

    return tests;
}

enum class shuffle_test_type
{
    shuffle_up,
    shuffle_down,
    shuffle_xor,
};

template<shuffle_test_type test_type, typename T>
__global__ void shuffle_kernel(T*                       data,
                               const unsigned int       delta,
                               const unsigned int       width,
                               const max_lane_mask_type active_lanes)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    const T value  = data[index];
    T       result = test_type_helper<T>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
    {
        switch(test_type)
        {
            case shuffle_test_type::shuffle_up:
                result = rocprim::warp_shuffle_up(value, delta, width);
                break;
            case shuffle_test_type::shuffle_down:
                result = rocprim::warp_shuffle_down(value, delta, width);
                break;
            case shuffle_test_type::shuffle_xor:
                result = rocprim::warp_shuffle_xor(value, delta, width);
                break;
        }
    }
    data[index] = result;
}

template<shuffle_test_type test_type, typename T>
void test_shuffle()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t size = hardware_warp_size;

    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    const auto shuffle = [&](T*                       out,
                             const T*                 in,
                             const unsigned int       delta,
                             const unsigned int       logical_warp_size,
                             const max_lane_mask_type active_lanes)
    {
        for(size_t warp = 0; warp < size; warp += hardware_warp_size)
        {
            for(unsigned int lane = 0; lane < hardware_warp_size; ++lane)
            {
                const unsigned int logical_warp_base = lane - lane % logical_warp_size;
                unsigned int       logical_src_lane  = lane % logical_warp_size;
                switch(test_type)
                {
                    case shuffle_test_type::shuffle_down:
                        if(logical_src_lane + delta < logical_warp_size)
                            logical_src_lane += delta;
                        break;
                    case shuffle_test_type::shuffle_up:
                        if(logical_src_lane >= delta)
                            logical_src_lane -= delta;
                        break;
                    case shuffle_test_type::shuffle_xor: logical_src_lane ^= delta; break;
                }

                const unsigned int src_lane = logical_warp_base + logical_src_lane;
                out[warp + lane]
                    = !is_lane_active(active_lanes, lane) ? test_type_helper<T>::uninitialized()
                      : !is_lane_active(active_lanes, src_lane) ? test_type_helper<T>::zero()
                                                                : in[warp + src_lane];
            }
        }
    };

    T* d_data;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_data, size * sizeof(T)));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate input
        auto           input = test_type_helper<T>::get_random_data(size, seed_value);
        std::vector<T> output(input.size());

        for(const auto active_lanes : active_lanes_tests())
        {
            SCOPED_TRACE(testing::Message()
                         << "with active_lanes = " << std::bitset<64>(active_lanes));
            for(unsigned int logical_warp_size = hardware_warp_size; logical_warp_size > 1;
                logical_warp_size >>= 1)
            {
                SCOPED_TRACE(testing::Message()
                             << "where logical_warp_size = " << logical_warp_size);

                const auto deltas = test_utils::get_random_data<unsigned int>(
                    std::max<size_t>(1, logical_warp_size / 2),
                    1U,
                    std::max<unsigned int>(1, logical_warp_size - 1),
                    seed_value);

                for(const auto delta : deltas)
                {
                    SCOPED_TRACE(testing::Message() << "where delta = " << delta);
                    // Calculate expected results on host
                    std::vector<T> expected(size, test_type_helper<T>::zero());
                    shuffle(expected.data(), input.data(), delta, logical_warp_size, active_lanes);

                    // Writing to device memory
                    HIP_CHECK(hipMemcpy(d_data,
                                        input.data(),
                                        input.size() * sizeof(T),
                                        hipMemcpyHostToDevice));

                    // Launching kernel
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_kernel<test_type, T>),
                                       dim3(1),
                                       dim3(hardware_warp_size),
                                       0,
                                       0,
                                       d_data,
                                       delta,
                                       logical_warp_size,
                                       active_lanes);
                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());

                    // Read from device memory
                    HIP_CHECK(hipMemcpy(output.data(),
                                        d_data,
                                        output.size() * sizeof(T),
                                        hipMemcpyDeviceToHost));

                    test_utils::assert_eq(output, expected);
                }
            }
        }
    }

    hipFree(d_data);
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleUp)
{
    test_shuffle<shuffle_test_type::shuffle_up, typename TestFixture::type>();
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleDown)
{
    test_shuffle<shuffle_test_type::shuffle_down, typename TestFixture::type>();
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleXor)
{
    test_shuffle<shuffle_test_type::shuffle_xor, typename TestFixture::type>();
}

template<class T>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void shuffle_index_kernel(T* data, int* src_lanes, unsigned int width)
{
    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    T value = data[index];
    value = rocprim::warp_shuffle(
        value, src_lanes[threadIdx.x/width], width
    );
    data[index] = value;
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleIndex)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate input
        auto           input = test_type_helper<T>::get_random_data(size, seed_value);
        std::vector<T> output(input.size());

        T* device_data;
        int * device_src_lanes;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_src_lanes,
                hardware_warp_size * sizeof(int)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto src_lanes
                = test_utils::get_random_data<int>(hardware_warp_size / logical_warp_size,
                                                   0,
                                                   std::max<int>(0, logical_warp_size - 1),
                                                   seed_value);

            // Calculate expected results on host
            std::vector<T> expected(size, test_type_helper<T>::zero());
            for(size_t i = 0; i < input.size()/logical_warp_size; i++)
            {
                int src_lane = src_lanes[i];
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index = j + logical_warp_size * i;
                    if(src_lane >= int(logical_warp_size) || src_lane < 0) src_lane = index;
                    expected[index] = input[src_lane + logical_warp_size * i];
                }
            }

            // Writing to device memory
            HIP_CHECK(
                hipMemcpy(
                    device_data, input.data(),
                    input.size() * sizeof(typename decltype(input)::value_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    device_src_lanes, src_lanes.data(),
                    src_lanes.size() * sizeof(typename decltype(src_lanes)::value_type),
                    hipMemcpyHostToDevice
                )
            );

            // Launching kernel
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(shuffle_index_kernel<T>),
                dim3(1), dim3(hardware_warp_size), 0, 0,
                device_data, device_src_lanes, logical_warp_size
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Read from device memory
            HIP_CHECK(
                hipMemcpy(
                    output.data(), device_data,
                    output.size() * sizeof(T),
                    hipMemcpyDeviceToHost
                )
            );

            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
            }
        }
        hipFree(device_data);
        hipFree(device_src_lanes);
    }
}

__global__ void lane_id_kernel(unsigned int* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    data[index]              = rocprim::lane_id();
}

TEST(RocprimIntrinsicsTests, LaneId)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t warps_per_block    = 4;
    const size_t block_size         = warps_per_block * hardware_warp_size;
    const size_t blocks             = 2;
    const size_t size               = blocks * block_size;
    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    unsigned int* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(unsigned int)));

    hipLaunchKernelGGL(lane_id_kernel,
                       dim3(blocks),
                       dim3(block_size),
                       0,
                       hipStreamDefault,
                       d_output);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> h_output(size);
    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, size * sizeof(unsigned int), hipMemcpyDeviceToHost));

    size_t i = 0;
    for(size_t block = 0; block < blocks; ++block)
    {
        for(size_t warp = 0; warp < warps_per_block; ++warp)
        {
            for(size_t lane = 0; lane < hardware_warp_size; ++lane)
            {
                ASSERT_EQ(lane, h_output[i++]) << "where block = " << block << ", warp = " << warp
                                               << ", lane = " << lane << ", index = " << i;
            }
        }
    }

    hipFree(d_output);
}

__global__ void masked_bit_count_kernel(unsigned int*             out,
                                        const max_lane_mask_type* in,
                                        const unsigned int        add,
                                        const max_lane_mask_type  active_lanes)
{
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int in_index  = out_index / rocprim::warp_size();

    const auto   value  = static_cast<rocprim::lane_mask_type>(in[in_index]);
    unsigned int result = test_type_helper<unsigned int>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
        result = rocprim::masked_bit_count(value, add);
    out[out_index] = result;
}

TEST(RocprimIntrinsicsTests, MaskedBitCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t warps_per_block    = 4;
    const size_t block_size         = warps_per_block * hardware_warp_size;
    const size_t blocks             = 2;
    const size_t in_size            = blocks * warps_per_block;
    const size_t out_size           = blocks * block_size;
    const size_t n_add              = 2;

    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    max_lane_mask_type* d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, in_size * sizeof(max_lane_mask_type)));

    unsigned int* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, out_size * sizeof(unsigned int)));

    std::vector<unsigned int> expected(out_size);
    std::vector<unsigned int> output(out_size);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate input
        const auto input = test_utils::get_random_data<max_lane_mask_type>(in_size,
                                                                           0,
                                                                           all_lanes_active,
                                                                           seed_value);
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            in_size * sizeof(max_lane_mask_type),
                            hipMemcpyHostToDevice));

        auto adds
            = test_utils::get_random_data<unsigned int>(n_add,
                                                        1,
                                                        std::numeric_limits<unsigned int>::max(),
                                                        seed_value);
        adds.push_back(0);
        for(const unsigned int add : adds)
        {
            SCOPED_TRACE(testing::Message() << "with add = " << add);

            for(const auto active_lanes : active_lanes_tests())
            {
                SCOPED_TRACE(testing::Message()
                             << "with active_lanes = " << std::bitset<64>(active_lanes));

                size_t i = 0;
                for(size_t block = 0; block < blocks; ++block)
                {
                    for(size_t warp = 0; warp < warps_per_block; ++warp)
                    {
                        const size_t in_index         = block * warps_per_block + warp;
                        unsigned int masked_bit_count = add;
                        for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                        {
                            expected[i++] = is_lane_active(active_lanes, lane)
                                                ? masked_bit_count
                                                : test_type_helper<unsigned int>::uninitialized();
                            masked_bit_count += (input[in_index] >> lane) & 1;
                        }
                    }
                }

                hipLaunchKernelGGL(masked_bit_count_kernel,
                                   dim3(blocks),
                                   dim3(block_size),
                                   0,
                                   hipStreamDefault,
                                   d_output,
                                   d_input,
                                   add,
                                   active_lanes);
                HIP_CHECK(hipGetLastError());

                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    out_size * sizeof(unsigned int),
                                    hipMemcpyDeviceToHost));

                test_utils::assert_eq(output, expected);
            }
        }
    }

    hipFree(d_input);
    hipFree(d_output);
}

enum class warp_any_all_test_type
{
    any,
    all
};

template<warp_any_all_test_type test_type>
__global__ void warp_any_all_kernel(unsigned int*             out,
                                    const max_lane_mask_type* in,
                                    max_lane_mask_type        active_lanes)
{
    const unsigned int index     = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int predicate = (in[index / rocprim::warp_size()] >> rocprim::lane_id()) & 1;

    unsigned int result = test_type_helper<unsigned int>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
    {
        switch(test_type)
        {
            case warp_any_all_test_type::any: result = rocprim::detail::warp_any(predicate); break;
            case warp_any_all_test_type::all: result = rocprim::detail::warp_all(predicate); break;
        }
    }
    out[index] = result;
}

template<warp_any_all_test_type test_type>
void warp_any_all_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t warps_per_block    = 4;
    const size_t block_size         = warps_per_block * hardware_warp_size;
    const size_t blocks             = 2;
    const size_t in_size            = blocks * warps_per_block;
    const size_t out_size           = blocks * block_size;

    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    max_lane_mask_type* d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, in_size * sizeof(max_lane_mask_type)));

    unsigned int* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, out_size * sizeof(unsigned int)));

    std::vector<unsigned int> output(out_size);
    std::vector<unsigned int> expected(out_size);
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Just straight up generating random bytes here would not get a very high chance of any value returning
        // false, so first generate a list values which ought to be true.
        std::vector<unsigned int> in_expected
            = test_utils::get_random_data01<unsigned int>(in_size, 0.25f, seed_value);

        auto input = test_utils::get_random_data<max_lane_mask_type>(in_size,
                                                                     1,
                                                                     all_lanes_active - 1,
                                                                     seed_value);

        for(const auto active_lanes : active_lanes_tests())
        {
            SCOPED_TRACE(testing::Message()
                         << "with active_lanes = " << std::bitset<64>(active_lanes));

            for(size_t i = 0; i < in_size; ++i)
            {
                // Change the input values based on the expected outcome
                switch(test_type)
                {
                    case warp_any_all_test_type::any:
                        if(in_expected[i] == 0)
                            input[i] = 0;
                        break;
                    case warp_any_all_test_type::all:
                        if(in_expected[i] != 0)
                            input[i] = all_lanes_active;
                        break;
                };

                for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                {
                    expected[i * hardware_warp_size + lane]
                        = is_lane_active(active_lanes, lane)
                              ? in_expected[i]
                              : test_type_helper<unsigned int>::uninitialized();
                }
            }

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                in_size * sizeof(max_lane_mask_type),
                                hipMemcpyHostToDevice));

            hipLaunchKernelGGL(HIP_KERNEL_NAME(warp_any_all_kernel<test_type>),
                               dim3(blocks),
                               dim3(block_size),
                               0,
                               hipStreamDefault,
                               d_output,
                               d_input,
                               active_lanes);
            HIP_CHECK(hipGetLastError());

            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                out_size * sizeof(unsigned int),
                                hipMemcpyDeviceToHost));

            test_utils::assert_eq(output, expected);
        }
    }

    hipFree(d_input);
    hipFree(d_output);
}

TEST(RocprimIntrinsicsTests, WarpAny)
{
    warp_any_all_test<warp_any_all_test_type::any>();
}

TEST(RocprimIntrinsicsTests, WarpAll)
{
    warp_any_all_test<warp_any_all_test_type::all>();
}

template<typename T>
__global__ void warp_permute_kernel(T*                       out,
                                    const T*                 in,
                                    const unsigned int*      indices,
                                    const unsigned int       logical_warp_size,
                                    const max_lane_mask_type active_lanes)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    const auto dst_lane = indices[index];
    const auto value    = in[index];
    T          result   = test_type_helper<T>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
        result = rocprim::warp_permute(value, dst_lane, logical_warp_size);

    out[index] = result;
}

TYPED_TEST(RocprimIntrinsicsTests, WarpPermute)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t warps_per_block    = 4;
    const size_t block_size         = warps_per_block * hardware_warp_size;
    const size_t blocks             = 2;
    const size_t size               = blocks * block_size;

    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    T* d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(T)));

    T* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(T)));

    unsigned int* d_indices;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_indices, size * sizeof(unsigned int)));

    const auto permute = [&](T*                       out,
                             const T*                 in,
                             const unsigned int*      indices,
                             const unsigned           logical_warp_size,
                             const max_lane_mask_type active_lanes)
    {
        size_t i = 0;
        for(unsigned int block = 0; block < blocks; ++block)
        {
            for(unsigned int warp = 0; warp < warps_per_block; ++warp)
            {
                for(unsigned int lane = 0; lane < hardware_warp_size; ++lane)
                {
                    const size_t base     = i & ~(hardware_warp_size - 1);
                    const size_t mask     = logical_warp_size - 1;
                    const size_t dst_lane = (indices[i] & mask) + (lane & ~mask);
                    out[base + dst_lane]  = !is_lane_active(active_lanes, dst_lane)
                                                ? test_type_helper<T>::uninitialized()
                                            : is_lane_active(active_lanes, lane)
                                                ? in[i]
                                                : test_type_helper<T>::zero();
                    ++i;
                }
            }
        }
    };

    std::vector<T>            output(size);
    std::vector<T>            expected(size);
    std::vector<unsigned int> indices(size);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto input = test_type_helper<T>::get_random_data(size, seed_value);

        const auto wrap = test_utils::get_random_data<unsigned int>(size, 0, 4, seed_value);

        for(const auto active_lanes : active_lanes_tests())
        {
            SCOPED_TRACE(testing::Message()
                         << "with active_lanes = " << std::bitset<64>(active_lanes));

            for(unsigned int logical_warp_size = hardware_warp_size; logical_warp_size > 1;
                logical_warp_size >>= 1)
            {
                SCOPED_TRACE(testing::Message()
                             << "with logical_warp_size = " << logical_warp_size);

                std::mt19937 g(seed_value);
                for(size_t i = 0; i < size; i += logical_warp_size)
                {
                    const auto start = indices.begin() + i;
                    const auto end   = start + logical_warp_size;

                    std::iota(start, end, wrap[i] * logical_warp_size);
                    std::shuffle(start, end, g);
                }

                permute(expected.data(),
                        input.data(),
                        indices.data(),
                        logical_warp_size,
                        active_lanes);

                HIP_CHECK(
                    hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_indices,
                                    indices.data(),
                                    size * sizeof(unsigned int),
                                    hipMemcpyHostToDevice));

                hipLaunchKernelGGL(HIP_KERNEL_NAME(warp_permute_kernel<T>),
                                   dim3(blocks),
                                   dim3(block_size),
                                   0,
                                   hipStreamDefault,
                                   d_output,
                                   d_input,
                                   d_indices,
                                   logical_warp_size,
                                   active_lanes);
                HIP_CHECK(hipGetLastError());

                HIP_CHECK(
                    hipMemcpy(output.data(), d_output, size * sizeof(T), hipMemcpyDeviceToHost));

                test_utils::assert_eq(output, expected);
            }
        }
    }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_indices);
}

template<unsigned int LabelBits>
__global__ void match_any_kernel(max_lane_mask_type* output,
                                 unsigned int*       input,
                                 max_lane_mask_type  active_lanes)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    const auto         value  = input[index];
    max_lane_mask_type result = test_type_helper<max_lane_mask_type>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
        result = rocprim::match_any<LabelBits>(value);
    output[index] = result;
}

TEST(RocprimIntrinsicsTests, MatchAny)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t           hardware_warp_size = ::rocprim::host_warp_size();
    const size_t           warps_per_block    = 4;
    const size_t           block_size         = warps_per_block * hardware_warp_size;
    const size_t           blocks             = 2;
    const size_t           size               = blocks * block_size;
    constexpr unsigned int label_bits         = 3;
    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    unsigned int* d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(unsigned int)));

    max_lane_mask_type* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(max_lane_mask_type)));

    std::vector<max_lane_mask_type> expected(size);
    std::vector<max_lane_mask_type> output(size);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto input = test_utils::get_random_data<unsigned int>(size,
                                                                     0,
                                                                     1u << (label_bits + 3),
                                                                     seed_value);

        for(const auto active_lanes : active_lanes_tests())
        {
            SCOPED_TRACE(testing::Message()
                         << "with active_lanes = " << std::bitset<64>(active_lanes));

            for(size_t block = 0; block < blocks; ++block)
            {
                for(size_t warp = 0; warp < warps_per_block; ++warp)
                {
                    const auto base = (block * warps_per_block + warp) * hardware_warp_size;
                    std::vector<max_lane_mask_type> histogram(1u << label_bits, 0);

                    for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                    {
                        if(is_lane_active(active_lanes, lane))
                        {
                            const auto value = bit_extract(input[base + lane], label_bits);
                            histogram[value] |= max_lane_mask_type{1} << lane;
                        }
                    }

                    for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                    {
                        if(is_lane_active(active_lanes, lane))
                        {
                            const auto value      = bit_extract(input[base + lane], label_bits);
                            expected[base + lane] = histogram[value];
                        }
                        else
                        {
                            expected[base + lane] = test_type_helper<unsigned int>::uninitialized();
                        }
                    }
                }
            }

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                size * sizeof(unsigned int),
                                hipMemcpyHostToDevice));

            hipLaunchKernelGGL(HIP_KERNEL_NAME(match_any_kernel<label_bits>),
                               dim3(blocks),
                               dim3(block_size),
                               0,
                               hipStreamDefault,
                               d_output,
                               d_input,
                               active_lanes);
            HIP_CHECK(hipGetLastError());

            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                size * sizeof(max_lane_mask_type),
                                hipMemcpyDeviceToHost));

            test_utils::assert_eq(output, expected);
        }
    }

    hipFree(d_input);
    hipFree(d_output);
}

__global__ void
    ballot_kernel(max_lane_mask_type* output, unsigned int* input, max_lane_mask_type active_lanes)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    const auto         value  = input[index];
    max_lane_mask_type result = test_type_helper<unsigned int>::uninitialized();
    if(is_lane_active(active_lanes, rocprim::lane_id()))
        result = rocprim::ballot(value);
    output[index] = result;
}

TEST(RocprimIntrinsicsTests, Ballot)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const size_t hardware_warp_size = ::rocprim::host_warp_size();
    const size_t warps_per_block    = 4;
    const size_t block_size         = warps_per_block * hardware_warp_size;
    const size_t blocks             = 2;
    const size_t size               = blocks * block_size;
    SCOPED_TRACE(testing::Message() << "with hardware_warp_size = " << hardware_warp_size);

    unsigned int* d_input;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(unsigned int)));

    max_lane_mask_type* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(max_lane_mask_type)));

    std::vector<max_lane_mask_type> expected(size);
    std::vector<max_lane_mask_type> output(size);

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto input = test_utils::get_random_data01<unsigned int>(size, 0.5f, seed_value);

        for(const auto active_lanes : active_lanes_tests())
        {
            SCOPED_TRACE(testing::Message()
                         << "with active_lanes = " << std::bitset<64>(active_lanes));

            for(size_t block = 0; block < blocks; ++block)
            {
                for(size_t warp = 0; warp < warps_per_block; ++warp)
                {
                    const auto         base = (block * warps_per_block + warp) * hardware_warp_size;
                    max_lane_mask_type true_mask = 0;

                    for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                    {
                        if(is_lane_active(active_lanes, lane) && input[base + lane])
                            true_mask |= max_lane_mask_type{1} << lane;
                    }

                    for(size_t lane = 0; lane < hardware_warp_size; ++lane)
                        expected[base + lane]
                            = is_lane_active(active_lanes, lane)
                                  ? true_mask
                                  : test_type_helper<unsigned int>::uninitialized();
                }
            }

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                size * sizeof(unsigned int),
                                hipMemcpyHostToDevice));

            hipLaunchKernelGGL(ballot_kernel,
                               dim3(blocks),
                               dim3(block_size),
                               0,
                               hipStreamDefault,
                               d_output,
                               d_input,
                               active_lanes);
            HIP_CHECK(hipGetLastError());

            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                size * sizeof(max_lane_mask_type),
                                hipMemcpyDeviceToHost));

            test_utils::assert_eq(output, expected);
        }
    }

    hipFree(d_input);
    hipFree(d_output);
}
