#include "detail/device_adjacent_difference.hpp"

#include "device_adjacent_difference_config.hpp"

#include "config_types.hpp"
#include "device_transform.hpp"

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/various.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <iterator>

#include <cstddef>

BEGIN_ROCPRIM_NAMESPACE

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

namespace detail
{
template <typename Config,
          bool InPlace,
          bool Right,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction>
void ROCPRIM_KERNEL __launch_bounds__(Config::block_size) adjacent_difference_kernel(
    const InputIt                                             input,
    const OutputIt                                            output,
    const std::size_t                                         size,
    const BinaryFunction                                      op,
    const typename std::iterator_traits<InputIt>::value_type* previous_values,
    const std::size_t                                         starting_block)
{
    adjacent_difference_kernel_impl<Config, InPlace, Right>(
        input, output, size, op, previous_values, starting_block);
}

template <typename Config,
          bool InPlace,
          bool Right,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction>
hipError_t adjacent_difference_impl(void* const          temporary_storage,
                                    std::size_t&         storage_size,
                                    const InputIt        input,
                                    const OutputIt       output,
                                    const std::size_t    size,
                                    const BinaryFunction op,
                                    const hipStream_t    stream,
                                    const bool           debug_synchronous)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_adjacent_difference_config<ROCPRIM_TARGET_ARCH, value_type>>;

    static constexpr unsigned int block_size       = config::block_size;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;

    const std::size_t num_blocks = ceiling_div(size, items_per_block);

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        if(InPlace && num_blocks >= 2) {
            storage_size = align_size((num_blocks - 1) * sizeof(value_type));
        } else {
            storage_size = 4;
        }

        return hipSuccess;
    }

    if(num_blocks == 0)
    {
        return hipSuccess;
    }

    // Copy values before they are overwritten to use as tile predecessors/sucessors
    // this is not dereferenced when the operation is not in place
    auto* const previous_values = static_cast<value_type*>(temporary_storage);
    if ROCPRIM_IF_CONSTEXPR(InPlace)
    {
        // If doing left adjacent diff then the last item of each block is needed for the
        // next block, otherwise the first item is needed for the previous block
        //const auto block_starts_iter
        //    = skip_iterator<InputIt, items_per_block> {input + items_per_block - (Right ? 0 : 1)};

        static constexpr auto offset = items_per_block - (Right ? 0 : 1);

        const auto block_starts_iter = make_transform_iterator(
            rocprim::make_counting_iterator(std::size_t {0}),
            [base = input + offset](std::size_t i) { return base[i * items_per_block]; });

        const hipError_t error = ::rocprim::transform(block_starts_iter,
                                                      previous_values,
                                                      num_blocks - 1,
                                                      rocprim::identity<> {},
                                                      stream,
                                                      debug_synchronous);
        if(error != hipSuccess)
        {
            return error;
        }
    }

    static constexpr unsigned int size_limit     = config::size_limit;
    static constexpr auto number_of_blocks_limit = std::max(size_limit / items_per_block, 1u);
    static constexpr auto aligned_size_limit     = number_of_blocks_limit * items_per_block;

    // Launch number_of_blocks_limit blocks while there is still at least as many blocks
    // left as the limit
    const auto number_of_launch = ceiling_div(size, aligned_size_limit);

    if(debug_synchronous)
    {
        std::cout << "----------------------------------\n";
        std::cout << "size:               " << size << '\n';
        std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
        std::cout << "number_of_launch:   " << number_of_launch << '\n';
        std::cout << "block_size:         " << block_size << '\n';
        std::cout << "items_per_block:    " << items_per_block << '\n';
        std::cout << "----------------------------------\n";
    }

    for(std::size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit)
    {
        const auto current_size
            = static_cast<unsigned int>(std::min<std::size_t>(size - offset, aligned_size_limit));
        const auto current_blocks = ceiling_div(current_size, items_per_block);
        const auto starting_block = i * number_of_blocks_limit;

        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        if(debug_synchronous)
        {
            std::cout << "index:            " << i << '\n';
            std::cout << "current_size:     " << current_size << '\n';
            std::cout << "number of blocks: " << current_blocks << '\n';

            start = std::chrono::high_resolution_clock::now();
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adjacent_difference_kernel<config, InPlace, Right>),
                           dim3(current_blocks),
                           dim3(block_size),
                           0,
                           stream,
                           input + offset,
                           output + offset,
                           size,
                           op,
                           previous_values + starting_block,
                           starting_block);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "adjacent_difference_kernel", current_size, start);
    }
    return hipSuccess;
}
} // namespace detail

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

template <typename Config = default_config,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference(void* const          temporary_storage,
                               std::size_t&         storage_size,
                               const InputIt        input,
                               const OutputIt       output,
                               const std::size_t    size,
                               const BinaryFunction op                = BinaryFunction {},
                               const hipStream_t    stream            = 0,
                               const bool           debug_synchronous = false)
{
    static constexpr bool in_place = false;
    static constexpr bool right    = false;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, input, output, size, op, stream, debug_synchronous);
}

template <typename Config = default_config,
          typename InputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_inplace(void* const          temporary_storage,
                                       std::size_t&         storage_size,
                                       const InputIt        values,
                                       const std::size_t    size,
                                       const BinaryFunction op                = BinaryFunction {},
                                       const hipStream_t    stream            = 0,
                                       const bool           debug_synchronous = false)
{
    static constexpr bool in_place = true;
    static constexpr bool right    = false;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, values, values, size, op, stream, debug_synchronous);
}

template <typename Config = default_config,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_right(void* const          temporary_storage,
                                     std::size_t&         storage_size,
                                     const InputIt        input,
                                     const OutputIt       output,
                                     const std::size_t    size,
                                     const BinaryFunction op                = BinaryFunction {},
                                     const hipStream_t    stream            = 0,
                                     const bool           debug_synchronous = false)
{
    static constexpr bool in_place = false;
    static constexpr bool right    = true;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, input, output, size, op, stream, debug_synchronous);
}

template <typename Config = default_config,
          typename InputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_right_inplace(void* const          temporary_storage,
                                             std::size_t&         storage_size,
                                             const InputIt        values,
                                             const std::size_t    size,
                                             const BinaryFunction op     = BinaryFunction {},
                                             const hipStream_t    stream = 0,
                                             const bool           debug_synchronous = false)
{
    static constexpr bool in_place = true;
    static constexpr bool right    = true;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, values, values, size, op, stream, debug_synchronous);
}

END_ROCPRIM_NAMESPACE