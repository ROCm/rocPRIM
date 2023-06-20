// MIT License
//
// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/functional.hpp>
#include <rocprim/device/device_binary_search.hpp>

// required test headers
#include "test_utils_types.hpp"

template<class Haystack,
         class Needle,
         class Output          = size_t,
         class CompareFunction = rocprim::less<>,
         class Config          = rocprim::default_config>
struct params
{
    using haystack_type = Haystack;
    using needle_type = Needle;
    using output_type = Output;
    using compare_op_type = CompareFunction;
    using config          = Config;
};

template<class Params>
class RocprimDeviceBinarySearch : public ::testing::Test {
public:
    using params = Params;
};

using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

using custom_config_0 = rocprim::transform_config<128, 4>;
using custom_config_1 = rocprim::binary_search_config<64, 2>;
struct custom_config_2
{
    static constexpr unsigned int block_size       = 256;
    static constexpr unsigned int items_per_thread = 1;
    static constexpr unsigned int size_limit       = ROCPRIM_GRID_SIZE_LIMIT;
};

typedef ::testing::Types<params<int, int>,
                         params<unsigned long long,
                                unsigned long long,
                                size_t,
                                rocprim::greater<unsigned long long>,
                                custom_config_0>,
                         params<float, double, unsigned int, rocprim::greater<double>>,
                         params<double, int>,
                         params<int8_t, int8_t>,
                         params<uint8_t, uint8_t>,
                         params<rocprim::half, rocprim::half, size_t, rocprim::less<rocprim::half>>,
                         params<rocprim::bfloat16,
                                rocprim::bfloat16,
                                size_t,
                                rocprim::less<rocprim::bfloat16>,
                                custom_config_1>,
                         params<custom_int2, custom_int2>,
                         params<custom_double2,
                                custom_double2,
                                unsigned int,
                                rocprim::greater<custom_double2>,
                                custom_config_2>>
    Params;

TYPED_TEST_SUITE(RocprimDeviceBinarySearch, Params);

TYPED_TEST(RocprimDeviceBinarySearch, LowerBound)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;
    using config          = typename TestFixture::params::config;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const size_t haystack_size = size;
            const size_t needles_size = (size_t)std::sqrt(size); // cast promises no data loss, silences warning
            const size_t d = haystack_size / 100;

            // Generate data
            std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
                haystack_size, 0, haystack_size + 2 * d, seed_value
            );
            std::sort(haystack.begin(), haystack.end(), compare_op);

            // Use a narrower range for needles for checking out-of-haystack cases
            std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
                needles_size, d, haystack_size + d, seed_value
            );

            haystack_type * d_haystack;
            needle_type * d_needles;
            output_type * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_haystack, haystack_size * sizeof(haystack_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_needles, needles_size * sizeof(needle_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, needles_size * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_haystack, haystack.data(),
                    haystack_size * sizeof(haystack_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_needles, needles.data(),
                    needles_size * sizeof(needle_type),
                    hipMemcpyHostToDevice
                )
            );

            // Calculate expected results on host
            std::vector<output_type> expected(needles_size);
            for(size_t i = 0; i < needles_size; i++)
            {
                expected[i] =
                    std::lower_bound(haystack.begin(), haystack.end(), needles[i], compare_op) -
                    haystack.begin();
            }

            void * d_temporary_storage = nullptr;
            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::lower_bound<config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_haystack,
                                                   d_needles,
                                                   d_output,
                                                   haystack_size,
                                                   needles_size,
                                                   compare_op,
                                                   stream,
                                                   debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            HIP_CHECK(rocprim::lower_bound<config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_haystack,
                                                   d_needles,
                                                   d_output,
                                                   haystack_size,
                                                   needles_size,
                                                   compare_op,
                                                   stream,
                                                   debug_synchronous));

            std::vector<output_type> output(needles_size);
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    needles_size * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_haystack));
            HIP_CHECK(hipFree(d_needles));
            HIP_CHECK(hipFree(d_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }


}

TYPED_TEST(RocprimDeviceBinarySearch, UpperBound)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;
    using config          = typename TestFixture::params::config;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        seed_type seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            const size_t haystack_size = size;
            const size_t needles_size = (size_t)std::sqrt(size); // cast promises no data loss, silences warning
            const size_t d = haystack_size / 100;

            // Generate data
            std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
                haystack_size, 0, haystack_size + 2 * d, seed_value
            );
            std::sort(haystack.begin(), haystack.end(), compare_op);

            // Use a narrower range for needles for checking out-of-haystack cases
            std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
                needles_size, d, haystack_size + d, seed_value
            );

            haystack_type * d_haystack;
            needle_type * d_needles;
            output_type * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_haystack, haystack_size * sizeof(haystack_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_needles, needles_size * sizeof(needle_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, needles_size * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_haystack, haystack.data(),
                    haystack_size * sizeof(haystack_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_needles, needles.data(),
                    needles_size * sizeof(needle_type),
                    hipMemcpyHostToDevice
                )
            );

            // Calculate expected results on host
            std::vector<output_type> expected(needles_size);
            for(size_t i = 0; i < needles_size; i++)
            {
                expected[i] =
                    std::upper_bound(haystack.begin(), haystack.end(), needles[i], compare_op) -
                    haystack.begin();
            }

            void * d_temporary_storage = nullptr;
            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::upper_bound<config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_haystack,
                                                   d_needles,
                                                   d_output,
                                                   haystack_size,
                                                   needles_size,
                                                   compare_op,
                                                   stream,
                                                   debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            HIP_CHECK(rocprim::upper_bound<config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_haystack,
                                                   d_needles,
                                                   d_output,
                                                   haystack_size,
                                                   needles_size,
                                                   compare_op,
                                                   stream,
                                                   debug_synchronous));

            std::vector<output_type> output(needles_size);
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    needles_size * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_haystack));
            HIP_CHECK(hipFree(d_needles));
            HIP_CHECK(hipFree(d_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }


}

TYPED_TEST(RocprimDeviceBinarySearch, BinarySearch)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;
    using config          = typename TestFixture::params::config;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const size_t haystack_size = size;
            const size_t needles_size = (size_t)std::sqrt(size); // cast promises no data loss, silences warning
            const size_t d = haystack_size / 100;

            // Generate data
            std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
                haystack_size, 0, haystack_size + 2 * d, seed_value
            );
            std::sort(haystack.begin(), haystack.end(), compare_op);

            // Use a narrower range for needles for checking out-of-haystack cases
            std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
                needles_size, d, haystack_size + d, seed_value
            );

            haystack_type * d_haystack;
            needle_type * d_needles;
            output_type * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_haystack, haystack_size * sizeof(haystack_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_needles, needles_size * sizeof(needle_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, needles_size * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_haystack, haystack.data(),
                    haystack_size * sizeof(haystack_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_needles, needles.data(),
                    needles_size * sizeof(needle_type),
                    hipMemcpyHostToDevice
                )
            );

            // Calculate expected results on host
            std::vector<output_type> expected(needles_size);
            for(size_t i = 0; i < needles_size; i++)
            {
                expected[i] = std::binary_search(haystack.begin(), haystack.end(), needles[i], compare_op);
            }

            void * d_temporary_storage = nullptr;
            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::binary_search<config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_haystack,
                                                     d_needles,
                                                     d_output,
                                                     haystack_size,
                                                     needles_size,
                                                     compare_op,
                                                     stream,
                                                     debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            HIP_CHECK(rocprim::binary_search<config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_haystack,
                                                     d_needles,
                                                     d_output,
                                                     haystack_size,
                                                     needles_size,
                                                     compare_op,
                                                     stream,
                                                     debug_synchronous));

            std::vector<output_type> output(needles_size);
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    needles_size * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_haystack));
            HIP_CHECK(hipFree(d_needles));
            HIP_CHECK(hipFree(d_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }
}
