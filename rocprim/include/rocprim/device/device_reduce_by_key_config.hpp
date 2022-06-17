// Copyright (c) 2018-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"

#include "../config.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

// TODO: Add documentation
template<unsigned int      BlockSize,
         unsigned int      ItemsPerThread,
         block_load_method LoadKeysMethod,
         block_load_method LoadValuesMethod>
struct reduce_by_key_config
{
    static constexpr unsigned int      block_size         = BlockSize;
    static constexpr unsigned int      items_per_thread   = ItemsPerThread;
    static constexpr block_load_method load_keys_method   = LoadKeysMethod;
    static constexpr block_load_method load_values_method = LoadValuesMethod;
};

namespace detail
{

template<unsigned int TargetArch, class Key, class Value>
struct default_reduce_by_key_config
    : reduce_by_key_config<256,
                           4,
                           block_load_method::block_load_transpose,
                           block_load_method::block_load_transpose>
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
