// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/device/device_segmented_scan.hpp>

// required test headers
#include "test_utils_types.hpp"

#include <numeric>

template<
    class Input,
    class Output,
    class ScanOp = ::rocprim::plus<Input>,
    int Init = 0, // as only integral types supported, int is used here even for floating point inputs
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000,
    // Tests output iterator with void value_type (OutputIterator concept)
    // Segmented scan primitives which use head flags do not support this kind
    // of output iterators.
    bool UseIdentityIterator = false,
    bool UseGraphs = false
>
struct params
{
    using input_type = Input;
    using output_type = Output;
    using scan_op_type = ScanOp;
    static constexpr int init = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class RocprimDeviceSegmentedScan : public ::testing::Test {
public:
    using params = Params;
};

using custom_short2  = test_utils::custom_test_type<short>;
using custom_int2    = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;
using half           = rocprim::half;
using bfloat16       = rocprim::bfloat16;

typedef ::testing::Types<
    params<unsigned char, unsigned int, rocprim::plus<unsigned int>>,
    params<int, int, rocprim::plus<int>, -100, 0, 10000>,
    params<int8_t, int8_t, rocprim::plus<int8_t>, -100, 0, 10000>,
    params<custom_double2, custom_double2, rocprim::minimum<custom_double2>, 1000, 0, 10000>,
    params<custom_int2, custom_short2, rocprim::maximum<custom_int2>, 10, 1000, 10000>,
    params<double, double, rocprim::maximum<double>, 50, 2, 10>,
    params<float, float, rocprim::plus<float>, 123, 100, 200, true>,
    params<bfloat16, float, rocprim::plus<bfloat16>, 0, 3, 50, true>,
    params<bfloat16, bfloat16, rocprim::minimum<bfloat16>, 0, 1000, 30000>,
    params<half, float, rocprim::plus<float>, 0, 10, 200, true>,
    params<half, half, rocprim::minimum<half>, 0, 1000, 30000>,
    params<unsigned char, long long, rocprim::plus<int>, 10, 3000, 4000>,
    params<int, int, ::rocprim::plus<int>, 0, 0, 1000, false, true>>
    Params;

TYPED_TEST_SUITE(RocprimDeviceSegmentedScan, Params);

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using offset_type  = unsigned int;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool debug_synchronous = false;

    scan_op_type scan_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    hipStream_t stream = 0; // default stream
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(size);
            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int segments_count = 0;
            size_t offset = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                input_type aggregate    = values_input[offset];
                values_expected[offset] = aggregate;
                for(size_t i = offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i]);
                    values_expected[i] = aggregate;
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // intermediate results of inclusive scan are stored as input_type,
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<input_type>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type  * d_values_input;
            offset_type * d_offsets;
            output_type * d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::segmented_inclusive_scan(
                nullptr,
                temporary_storage_bytes,
                d_values_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                segments_count,
                d_offsets,
                d_offsets + 1,
                scan_op,
                stream,
                debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);
            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(
                rocprim::segmented_inclusive_scan(
                    d_temporary_storage, temporary_storage_bytes,
                    d_values_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                    segments_count,
                    d_offsets, d_offsets + 1,
                    scan_op,
                    stream, debug_synchronous
                )
            );

            
            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            std::vector<output_type> values_output(size);
            HIP_CHECK(
                hipMemcpy(
                    values_output.data(), d_values_output,
                    values_output.size() * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(values_output, values_expected, precision));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_values_output));

            if(TestFixture::params::use_graphs)
            {
               gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using offset_type  = unsigned int;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool     debug_synchronous     = false;

    const input_type init = input_type{TestFixture::params::init};

    scan_op_type scan_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    hipStream_t stream = 0; // default stream
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(size);
            std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            unsigned int segments_count = 0;
            size_t offset = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const size_t end = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                input_type aggregate    = init;
                values_expected[offset] = aggregate;
                for(size_t i = offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i - 1]);
                    values_expected[i] = output_type(aggregate);
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // intermediate results of exclusive scan are stored as decltype(init),
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<decltype(init)>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type  * d_values_input;
            offset_type * d_offsets;
            output_type * d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_values_input, values_input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            size_t temporary_storage_bytes;
            HIP_CHECK(
                rocprim::segmented_exclusive_scan(
                    nullptr, temporary_storage_bytes,
                    d_values_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                    segments_count,
                    d_offsets, d_offsets + 1,
                    init, scan_op,
                    stream, debug_synchronous
                )
            );

            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_GT(temporary_storage_bytes, 0);
            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(
                rocprim::segmented_exclusive_scan(
                    d_temporary_storage, temporary_storage_bytes,
                    d_values_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                    segments_count,
                    d_offsets, d_offsets + 1,
                    init, scan_op,
                    stream, debug_synchronous
                )
            );

            
            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            std::vector<output_type> values_output(size);
            HIP_CHECK(
                hipMemcpy(
                    values_output.data(), d_values_output,
                    values_output.size() * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(values_output, values_expected, precision));

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_values_output));

            if(TestFixture::params::use_graphs)
            {
               gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScanUsingHeadFlags)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Does not support output iterator with void value_type
    using input_type   = typename TestFixture::params::input_type;
    using flag_type    = unsigned int;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;

    const bool debug_synchronous = false;

    // scan function
    scan_op_type scan_op;

    hipStream_t stream = 0; // default stream
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 10, seed_value);
            std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10, seed_value);

            if(size != 0)
                flags[0] = 1U;

            // generate segments and find their maximum width
            size_t max_segment_length = 1;
            size_t curr_segment_start = 0;
            for(size_t i = 1; i < size; ++i)
            {
                if(flags[i] == 1U)
                {
                    size_t curr_segment = i - curr_segment_start;
                    if(curr_segment > max_segment_length)
                        max_segment_length = curr_segment;
                    curr_segment_start = i;
                }
                else
                    flags[i] = 0U;
            }
            {
                size_t curr_segment = size - curr_segment_start;
                if(curr_segment > max_segment_length)
                    max_segment_length = curr_segment;
            }

            // intermediate results of inclusive scan are stored as input_type,
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<input_type>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type * d_input;
            flag_type * d_flags;
            output_type * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(input_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(flag_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_flags, flags.data(),
                    flags.size() * sizeof(flag_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            std::vector<output_type> expected(input.size());

            test_utils::host_inclusive_segmented_scan_headflags<input_type>(input.begin(),
                                                                            input.end(),
                                                                            flags.begin(),
                                                                            expected.begin(),
                                                                            scan_op);

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::segmented_inclusive_scan(
                    nullptr, temp_storage_size_bytes,
                    d_input, d_output, d_flags,
                    input.size(), scan_op, stream,
                    debug_synchronous
                )
            );

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(
                rocprim::segmented_inclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, d_flags,
                    input.size(), scan_op, stream,
                    debug_synchronous
                )
            );

            
            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            std::vector<output_type> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, precision));

            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_flags));
            HIP_CHECK(hipFree(d_output));

            if(TestFixture::params::use_graphs)
            {
               gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScanUsingHeadFlags)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Does not support output iterator with void value_type
    using input_type   = typename TestFixture::params::input_type;
    using flag_type    = unsigned int;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;

    const bool debug_synchronous = false;

    const input_type init = input_type{TestFixture::params::init};

    // scan function
    scan_op_type scan_op;

    hipStream_t stream = 0; // default stream
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 10, seed_value);
            std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10, seed_value);

            if(size != 0)
                flags[0] = 1U;

            // generate segments and find their maximum width
            size_t max_segment_length = 1;
            size_t curr_segment_start = 0;
            for(size_t i = 1; i < size; ++i)
            {
                if(flags[i] == 1U)
                {
                    size_t curr_segment = i - curr_segment_start;
                    if(curr_segment > max_segment_length)
                        max_segment_length = curr_segment;
                    curr_segment_start = i;
                }
                else
                    flags[i] = 0U;
            }
            {
                size_t curr_segment = size - curr_segment_start;
                if(curr_segment > max_segment_length)
                    max_segment_length = curr_segment;
            }

            // intermediate results of exclusive scan are stored as decltype(init),
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<decltype(init)>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type * d_input;
            flag_type * d_flags;
            output_type * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(input_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(flag_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(output_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_flags, flags.data(),
                    flags.size() * sizeof(flag_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            std::vector<output_type> expected(input.size());

            test_utils::host_exclusive_segmented_scan_headflags(input.begin(),
                                                                input.end(),
                                                                flags.begin(),
                                                                expected.begin(),
                                                                scan_op,
                                                                init);

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::segmented_exclusive_scan(
                    nullptr, temp_storage_size_bytes,
                    d_input, d_output, d_flags, init,
                    input.size(), scan_op, stream, debug_synchronous
                )
            );

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(
                rocprim::segmented_exclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, d_flags, init,
                    input.size(), scan_op, stream, debug_synchronous
                )
            );

            
            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            std::vector<output_type> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(output_type),
                    hipMemcpyDeviceToHost
                )
            );

            if(TestFixture::params::use_graphs)
            {
               gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, precision));

            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_flags));
            HIP_CHECK(hipFree(d_output));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}
