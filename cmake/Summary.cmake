# MIT License
#
# Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_STRIP}")
  get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(GENERATOR_IS_MULTI_CONFIG)
    message(STATUS "  Build types           : ${CMAKE_CONFIGURATION_TYPES}")
  else()
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  endif()
  message(STATUS "  Install prefix        : ${CMAKE_INSTALL_PREFIX}")
  if(NOT USE_HIP_CPU)
    message(STATUS "  Device targets        : ${GPU_TARGETS}")
  endif()
  message(STATUS "")
  message(STATUS "  ONLY_INSTALL              : ${ONLY_INSTALL}")
  message(STATUS "  BUILD_TEST                : ${BUILD_TEST}")
  message(STATUS "  BUILD_BENCHMARK           : ${BUILD_BENCHMARK}")
  message(STATUS "  BUILD_EXAMPLE             : ${BUILD_EXAMPLE}")
  message(STATUS "  BUILD_DOCS                : ${BUILD_DOCS}")
  message(STATUS "  USE_HIP_CPU               : ${USE_HIP_CPU}")
endfunction()
