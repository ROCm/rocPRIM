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

# ###########################
# rocPRIM dependencies
# ###########################

# NOTE1: the reason we don't scope global state meddling using add_subdirectory
#        is because CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL which
#        would promote IMPORTED targets of find_package(CONFIG) to be visible
#        by other parts of the build. So we save and restore global state.
#
# NOTE2: We disable the ROCMChecks.cmake warning noting that we meddle with
#        global state. This is consequence of abusing the CMake CXX language
#        which HIP piggybacks on top of. This kind of HIP support has one chance
#        at observing the global flags, at the find_package(HIP) invocation.
#        The device compiler won't be able to pick up changes after that, hence
#        the warning.
set(USER_CXX_FLAGS ${CMAKE_CXX_FLAGS})
if(DEFINED BUILD_SHARED_LIBS)
  set(USER_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()
set(USER_ROCM_WARN_TOOLCHAIN_VAR ${ROCM_WARN_TOOLCHAIN_VAR})

set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "")
# Turn off warnings and errors for all warnings in dependencies
separate_arguments(CXX_FLAGS_LIST NATIVE_COMMAND ${CMAKE_CXX_FLAGS})
list(REMOVE_ITEM CXX_FLAGS_LIST /WX -Werror -Werror=pendantic -pedantic-errors)
if(MSVC)
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "/[Ww]([0-4]?)(all)?") # Remove MSVC warning flags
  list(APPEND CXX_FLAGS_LIST /w)
else()
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "-W(all|extra|everything)") # Remove GCC/LLVM flags
  list(APPEND CXX_FLAGS_LIST -w)
endif()
list(JOIN CXX_FLAGS_LIST " " CMAKE_CXX_FLAGS)
# Don't build client dependencies as shared
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Global flag to cause add_library() to create shared libraries if on." FORCE)

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included (when not using HIP-CPU).

include(FetchContent)

if(USE_HIP_CPU)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(hip_cpu_rt QUIET)
  endif()
  if(NOT TARGET hip_cpu_rt::hip_cpu_rt)
    message(STATUS "HIP-CPU runtime not found. Fetching...")
    FetchContent_Declare(
      hip-cpu
      GIT_REPOSITORY https://github.com/ROCm-Developer-Tools/HIP-CPU.git
      GIT_TAG        56f559c93be210bb300dad3673c06d2bb0119d13 # master@2022.07.01
    )
    FetchContent_MakeAvailable(hip-cpu)
    if(NOT TARGET hip_cpu_rt::hip_cpu_rt)
      add_library(hip_cpu_rt::hip_cpu_rt ALIAS hip_cpu_rt)
    endif()
  else()
    find_package(hip_cpu_rt REQUIRED)
    # If we found HIP-CPU as binary, search for transitive dependencies
  find_package(Threads REQUIRED)
  set(CMAKE_REQUIRED_FLAGS "-std=c++17")
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(__GLIBCXX__ "cstddef" STL_IS_GLIBCXX)
  set(STL_DEPENDS_ON_TBB ${STL_IS_GLIBCXX})
  if(STL_DEPENDS_ON_TBB)
      find_package(TBB QUIET)
      if(NOT TARGET TBB::tbb AND NOT TARGET tbb)
        message(STATUS "Thread Building Blocks not found. Fetching...")
        FetchContent_Declare(
          thread-building-blocks
          GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
          GIT_TAG        3df08fe234f23e732a122809b40eb129ae22733f # v2021.5.0
      )
        FetchContent_MakeAvailable(thread-building-blocks)
      else()
        find_package(TBB REQUIRED)
    endif()
  endif(STL_DEPENDS_ON_TBB)
  endif()
endif(USE_HIP_CPU)

# Test dependencies
if(BUILD_TEST)
  # NOTE: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
    endif()
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    find_package(GTest CONFIG REQUIRED PATHS ${GTEST_ROOT})
  endif()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Benchmark (https://github.com/google/benchmark.git)
    find_package(benchmark QUIET)
  endif()

  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found or force download Google Benchmark on. Downloading and building Google Benchmark.")
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
    endif()
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/googlebenchmark CACHE PATH "")
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
      # hip-clang cannot compile googlebenchmark for some reason
      set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
    endif()

    download_project(
      PROJ           googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.6.1
      INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
      CMAKE_ARGS     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_STANDARD=14 ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD   TRUE
      LOG_CONFIGURE  TRUE
      LOG_BUILD      TRUE
      LOG_INSTALL    TRUE
      BUILD_PROJECT  TRUE
      UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()

set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

find_package(ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  set(rocm_cmake_url "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip")
  set(rocm_cmake_path "${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}")
  set(rocm_cmake_archive "${rocm_cmake_path}.zip")
  file(DOWNLOAD "${rocm_cmake_url}" "${rocm_cmake_archive}" STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

# Restore user global state
set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS})
if(DEFINED USER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ${USER_BUILD_SHARED_LIBS})
  else()
  unset(BUILD_SHARED_LIBS CACHE )
  endif()
set(ROCM_WARN_TOOLCHAIN_VAR ${USER_ROCM_WARN_TOOLCHAIN_VAR} CACHE BOOL "")

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMHeaderWrapper)
include(ROCMCheckTargetIds)
include(ROCMClients)
