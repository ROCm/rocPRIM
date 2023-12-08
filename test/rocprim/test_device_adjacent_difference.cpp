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

#include "test_utils_types.hpp"

#include <rocprim/device/device_adjacent_difference.hpp>

#include <rocprim/detail/various.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <numeric>
#include <type_traits>

namespace
{

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*in_place*/,
                                  std::false_type /*always_in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference<Config>(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*in_place*/,
                                  std::false_type /*always_in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right<Config>(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*in_place*/,
                                  std::false_type /*always_in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_inplace<Config>(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*in_place*/,
                                  std::false_type /*always_in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right_inplace<Config>(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*in_place*/,
                                  std::true_type /*always_in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_alias<Config>(temporary_storage,
                                                        storage_size,
                                                        input,
                                                        output,
                                                        std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*in_place*/,
                                  std::true_type /*always_in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right_alias<Config>(temporary_storage,
                                                              storage_size,
                                                              input,
                                                              output,
                                                              std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*in_place*/,
                                  std::true_type /*always_in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_alias<Config>(temporary_storage,
                                                        storage_size,
                                                        input,
                                                        input,
                                                        std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*in_place*/,
                                  std::true_type /*always_in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right_alias<Config>(temporary_storage,
                                                              storage_size,
                                                              input,
                                                              input,
                                                              std::forward<Args>(args)...);
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::true_type /*left*/)
{
    std::vector<Output> result(input.size());
    std::adjacent_difference(input.cbegin(), input.cend(), result.begin(), op);
    return result;
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::false_type /*left*/)
{
    std::vector<Output> result(input.size());
    // "right" adjacent difference is just adjacent difference backwards
    std::adjacent_difference(input.crbegin(), input.crend(), result.rbegin(), op);
    return result;
}
} // namespace

// Params for tests
template<class InputType,
         class OutputType         = InputType,
         bool Left                = true,
         bool InPlace             = false,
         bool AlwaysInPlace       = false,
         bool UseIdentityIterator = false,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false>
struct DeviceAdjacentDifferenceParams
{
    using input_type                            = InputType;
    using output_type                           = OutputType;
    static constexpr bool left                  = Left;
    static constexpr bool in_place              = InPlace;
    static constexpr bool always_in_place       = AlwaysInPlace;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    static constexpr bool left                  = Params::left;
    static constexpr bool in_place              = Params::in_place;
    static constexpr bool always_in_place       = Params::always_in_place;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool debug_synchronous     = false;
    using config                                = typename Params::config;
    static constexpr bool use_graphs            = Params::use_graphs;
};

using custom_double2     = test_utils::custom_test_type<double>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 8>;

using custom_config_0 = rocprim::adjacent_difference_config<128, 4>;

template<int SizeLimit>
using custom_size_limit_config
    = rocprim::adjacent_difference_config<1024,
                                          2,
                                          rocprim::block_load_method::block_load_transpose,
                                          rocprim::block_store_method::block_store_transpose,
                                          SizeLimit>;

using RocprimDeviceAdjacentDifferenceTestsParams = ::testing::Types<
    // Tests with default configuration
    DeviceAdjacentDifferenceParams<int>,
    DeviceAdjacentDifferenceParams<float, double, false>,
    DeviceAdjacentDifferenceParams<int8_t, int8_t, true, true>,
    DeviceAdjacentDifferenceParams<custom_double2, custom_double2, false, true>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16, float, true, false, false>,
    DeviceAdjacentDifferenceParams<rocprim::half, rocprim::half, true, true, false>,
    DeviceAdjacentDifferenceParams<custom_int64_array, custom_int64_array, false, true, true>,
    // Tests for supported config structs
    DeviceAdjacentDifferenceParams<rocprim::bfloat16,
                                   float,
                                   true,
                                   false,
                                   false,
                                   false,
                                   custom_config_0>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16, float, true, false, false>,
    // Tests for different size_limits
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   false,
                                   custom_size_limit_config<64>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   false,
                                   custom_size_limit_config<8192>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   false,
                                   custom_size_limit_config<10240>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   false,
                                   rocprim::default_config,
                                   true>,

    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   true,
                                   custom_size_limit_config<64>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   true,
                                   custom_size_limit_config<8192>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   true,
                                   custom_size_limit_config<10240>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   false,
                                   false,
                                   true,
                                   rocprim::default_config,
                                   true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceTests, RocprimDeviceAdjacentDifferenceTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceTests, AdjacentDifference)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using output_type                           = typename TestFixture::output_type;
    static constexpr bool left                  = TestFixture::left;
    static constexpr bool in_place              = TestFixture::in_place;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                = typename TestFixture::config;

    SCOPED_TRACE(testing::Message() << "left = " << left << ", in_place = " << in_place);

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<output_type> output(input.size());

            T*           d_input;
            output_type* d_output = nullptr;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(input[0])));
            HIP_CHECK(hipMemcpy(
                d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

            if(!in_place)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output,
                                                             output.size() * sizeof(output[0])));
            }

            static constexpr auto left_tag     = rocprim::detail::bool_constant<left>{};
            static constexpr auto in_place_tag = rocprim::detail::bool_constant<in_place>{};
            static constexpr auto always_in_place_tag
                = rocprim::detail::bool_constant<always_in_place>{};

            // Calculate expected results on host
            const auto expected
                = get_expected_result<output_type>(input, rocprim::minus<> {}, left_tag);

            const auto output_it
                = test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output);

            hipGraph_t graph;
            hipGraphExec_t graph_instance;
            if (TestFixture::use_graphs)
                graph = test_utils::createGraphHelper(stream);
            
            // Allocate temporary storage
            std::size_t temp_storage_size;
            void*       d_temp_storage = nullptr;
            HIP_CHECK(dispatch_adjacent_difference<Config>(left_tag,
                                                           in_place_tag,
                                                           always_in_place_tag,
                                                           d_temp_storage,
                                                           temp_storage_size,
                                                           d_input,
                                                           output_it,
                                                           size,
                                                           rocprim::minus<>{},
                                                           stream,
                                                           debug_synchronous));

            if (TestFixture::use_graphs)
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            
            ASSERT_GT(temp_storage_size, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            if (TestFixture::use_graphs)
                test_utils::resetGraphHelper(graph, graph_instance, stream);
            
            // Run
            HIP_CHECK(dispatch_adjacent_difference<Config>(left_tag,
                                                           in_place_tag,
                                                           always_in_place_tag,
                                                           d_temp_storage,
                                                           temp_storage_size,
                                                           d_input,
                                                           output_it,
                                                           size,
                                                           rocprim::minus<>{},
                                                           stream,
                                                           debug_synchronous));
            HIP_CHECK(hipGetLastError());

            if (TestFixture::use_graphs)
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(output.data(),
                          in_place ? static_cast<void*>(d_input) : static_cast<void*>(d_output),
                          output.size() * sizeof(output[0]),
                          hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
                output,
                expected,
                std::max(test_utils::precision<T>, test_utils::precision<output_type>)));

            hipFree(d_input);
            if(!in_place)
            {
                hipFree(d_output);
            }
            hipFree(d_temp_storage);

            if (TestFixture::use_graphs)
            {
                test_utils::cleanupGraphHelper(graph, graph_instance);
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

// Params for tests
template<bool Left = true, bool InPlace = false, bool AlwaysInPlace = false, bool UseGraphs = false>
struct DeviceAdjacentDifferenceLargeParams
{
    static constexpr bool left            = Left;
    static constexpr bool in_place        = InPlace;
    static constexpr bool always_in_place = AlwaysInPlace;
    static constexpr bool use_graphs      = UseGraphs;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceLargeTests : public ::testing::Test
{
public:
    static constexpr bool left              = Params::left;
    static constexpr bool in_place          = Params::in_place;
    static constexpr bool always_in_place   = Params::always_in_place;
    static constexpr bool debug_synchronous = false;
    static constexpr bool use_graphs        = Params::use_graphs;
};

template<unsigned int SamplingRate>
class check_output_iterator
{
public:
    using flag_type = unsigned int;

private:
    class check_output
    {
    public:
        __device__ check_output(flag_type* incorrect_flag, size_t current_index, size_t* counter)
            : current_index_(current_index), incorrect_flag_(incorrect_flag), counter_(counter)
        {}

        __device__ check_output& operator=(size_t value)
        {
            if(value != current_index_)
            {
                rocprim::detail::atomic_store(incorrect_flag_, 1);
            }
            if(current_index_ % SamplingRate == 0)
            {
                atomicAdd(counter_, 1);
            }
            return *this;
        }

    private:
        size_t     current_index_;
        flag_type* incorrect_flag_;
        size_t*    counter_;
    };

public:
    using value_type        = size_t;
    using reference         = check_output;
    using pointer           = check_output*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    __host__ __device__ check_output_iterator(flag_type* const incorrect_flag,
                                              size_t* const    counter)
        : current_index_(0), incorrect_flag_(incorrect_flag), counter_(counter)
    {}

    __device__ bool operator==(const check_output_iterator& rhs) const
    {
        return current_index_ == rhs.current_index_;
    }
    __device__ bool operator!=(const check_output_iterator& rhs) const
    {
        return !(*this == rhs);
    }
    __device__ reference operator*()
    {
        return reference(incorrect_flag_, current_index_, counter_);
    }
    __device__ reference operator[](const difference_type distance) const
    {
        return *(*this + distance);
    }
    __host__ __device__ check_output_iterator& operator+=(const difference_type rhs)
    {
        current_index_ += rhs;
        return *this;
    }
    __host__ __device__ check_output_iterator& operator-=(const difference_type rhs)
    {
        current_index_ -= rhs;
        return *this;
    }
    __host__ __device__ difference_type operator-(const check_output_iterator& rhs) const
    {
        return current_index_ - rhs.current_index_;
    }
    __host__ __device__ check_output_iterator operator+(const difference_type rhs) const
    {
        return check_output_iterator(*this) += rhs;
    }
    __host__ __device__ check_output_iterator operator-(const difference_type rhs) const
    {
        return check_output_iterator(*this) -= rhs;
    }
    __host__ __device__ check_output_iterator& operator++()
    {
        ++current_index_;
        return *this;
    }
    __host__ __device__ check_output_iterator& operator--()
    {
        --current_index_;
        return *this;
    }
    __host__ __device__ check_output_iterator operator++(int)
    {
        return ++check_output_iterator{*this};
    }
    __host__ __device__ check_output_iterator operator--(int)
    {
        return --check_output_iterator{*this};
    }

private:
    size_t     current_index_;
    flag_type* incorrect_flag_;
    size_t*    counter_;
};

using RocprimDeviceAdjacentDifferenceLargeTestsParams
    = ::testing::Types<DeviceAdjacentDifferenceLargeParams<true, false>,
                       DeviceAdjacentDifferenceLargeParams<false, false>,
                       DeviceAdjacentDifferenceLargeParams<false, false, true>,
                       DeviceAdjacentDifferenceLargeParams<true, false, false, true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceLargeTests,
                 RocprimDeviceAdjacentDifferenceLargeTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceLargeTests, LargeIndices)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();

#ifdef _WIN32
    if (TestFixture::use_graphs)
    {
        // Skip this test on Navi2x/3x on Windows, since check_output_iterator does not appear to work there.
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        std::string deviceName = std::string(props.gcnArchName);
        if(deviceName.rfind("gfx1030", 0) == 0 ||
           deviceName.rfind("gfx1100", 0) == 0 ||
           deviceName.rfind("gfx1101", 0) == 0 ||
           deviceName.rfind("gfx1102", 0) == 0)
        {
            // This is a Navi2x/3x device, so skip this test
            GTEST_SKIP() << "Temporarily skipping test on Windows for on gfx1030, gfx1100, gfx1101, gfx1102";
        }
    }
#endif

    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                         = size_t;
    static constexpr bool         is_left           = TestFixture::left;
    static constexpr bool         is_in_place       = TestFixture::in_place;
    const bool                    debug_synchronous = TestFixture::debug_synchronous;
    static constexpr unsigned int sampling_rate     = 10000;
    using OutputIterator                            = check_output_iterator<sampling_rate>;
    using flag_type                                 = OutputIterator::flag_type;

    SCOPED_TRACE(testing::Message()
                 << "is_left = " << is_left << ", is_in_place = " << is_in_place);

    hipStream_t stream = 0; // default
    if (TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            flag_type* d_incorrect_flag;
            size_t*    d_counter;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_incorrect_flag, sizeof(*d_incorrect_flag)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_counter, sizeof(*d_counter)));
            HIP_CHECK(hipMemset(d_incorrect_flag, 0, sizeof(*d_incorrect_flag)));
            HIP_CHECK(hipMemset(d_counter, 0, sizeof(*d_counter)));
            OutputIterator output(d_incorrect_flag, d_counter);

            const auto input = rocprim::make_counting_iterator(T{0});

            // Return the position where the adjacent difference is expected to be written out.
            // When called with consecutive values the left value is returned at the left-handed difference, and the right value otherwise.
            // The return value is coherent with the boundary values.
            const auto op = [](const auto& larger_value, const auto& smaller_value)
            { return (smaller_value + larger_value) / 2 + (is_left ? 1 : 0); };

            static constexpr auto left_tag     = rocprim::detail::bool_constant<is_left>{};
            static constexpr auto in_place_tag = rocprim::detail::bool_constant<is_in_place>{};
            static constexpr auto always_in_place_tag
                = rocprim::detail::bool_constant<always_in_place>{};

            hipGraph_t graph;
            hipGraphExec_t graph_instance;
            if (TestFixture::use_graphs)
                graph = test_utils::createGraphHelper(stream);
            
            // Allocate temporary storage
            std::size_t temp_storage_size;
            void*       d_temp_storage = nullptr;
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   in_place_tag,
                                                   always_in_place_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   op,
                                                   stream,
                                                   debug_synchronous));

            if (TestFixture::use_graphs)
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            
            ASSERT_GT(temp_storage_size, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            if (TestFixture::use_graphs)
                test_utils::resetGraphHelper(graph, graph_instance, stream);
            
            // Run
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   in_place_tag,
                                                   always_in_place_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   op,
                                                   stream,
                                                   debug_synchronous));

            if (TestFixture::use_graphs)
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

            // Copy output to host
            flag_type incorrect_flag;
            size_t    counter;
            HIP_CHECK(hipMemcpy(&incorrect_flag,
                                d_incorrect_flag,
                                sizeof(incorrect_flag),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&counter, d_counter, sizeof(counter), hipMemcpyDeviceToHost));

            ASSERT_EQ(incorrect_flag, 0);
            ASSERT_EQ(counter, rocprim::detail::ceiling_div(size, sampling_rate));

            hipFree(d_temp_storage);
            hipFree(d_incorrect_flag);
            hipFree(d_counter);

            if (TestFixture::use_graphs)
                test_utils::cleanupGraphHelper(graph, graph_instance);
        }
    }

    if (TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}
