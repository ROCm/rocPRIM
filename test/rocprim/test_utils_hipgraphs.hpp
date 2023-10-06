// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TEST_UTILS_HIPGRAPHS_HPP
#define ROCPRIM_TEST_UTILS_HIPGRAPHS_HPP

namespace test_utils
{

// Helper functions for testing with hipGraph stream capture
inline hipGraph_t createGraphHelper(hipStream_t& stream, bool beginCapture=true)
{
    // Create a new graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Optionally begin stream capture
    // Note: this will not work with the default stream
    if (beginCapture)
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    return graph;
}

inline void cleanupGraphHelper(hipGraph_t& graph, hipGraphExec_t& instance)
{
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(instance));
}

inline void resetGraphHelper(hipGraph_t& graph, hipGraphExec_t& instance, hipStream_t& stream, bool beginCapture=true)
{
    // Destroy the old graph and instance
    cleanupGraphHelper(graph, instance);

    // Create a new graph and optionally begin capture
    graph = createGraphHelper(stream, beginCapture);
}

inline hipGraphExec_t execGraphHelper(hipGraph_t& graph, hipStream_t& stream, bool endCapture=true, bool sync_on_complete=true)
{
    // Optionally end stream capture (must be already capturing)
    if (endCapture)
        HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Instantiate the graph
    hipGraphExec_t instance;
    HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    // Launch the graph
    HIP_CHECK(hipGraphLaunch(instance, stream));

    // Optionally synchronize the stream
    if (sync_on_complete)
        HIP_CHECK(hipStreamSynchronize(stream));

    return instance;
}

} // end namespace test_utils

#endif //ROCPRIM_TEST_UTILS_HIPGRAPHS_HPP
