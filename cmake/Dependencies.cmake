# MIT License
#
# Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# Never update automatically from the remote repository
if(CMAKE_VERSION VERSION_LESS 3.2)
  set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
  set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED TRUE")
endif()

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

if(USE_HIP_CPU)
  find_package(Threads REQUIRED)

  include(CheckCXXSymbolExists)
  set(CMAKE_REQUIRED_FLAGS "-std=c++17")
  check_cxx_symbol_exists(__PSTL_PAR_BACKEND_TBB "cstddef" STL_DEPENDS_ON_TBB)
  if(STL_DEPENDS_ON_TBB)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
      # TBB (https://github.com/oneapi-src/oneTBB)
      find_package(TBB QUIET)
    endif()

    if(NOT TBB_FOUND)
      message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
      set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "")
      download_project(
        PROJ                tbb
        GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
        GIT_TAG             v2021.1.1
        INSTALL_DIR         ${TBB_ROOT}
        CMAKE_ARGS          -DTBB_TEST=OFF -DTBB_STRICT=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        LOG_DOWNLOAD        TRUE
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        BUILD_PROJECT       TRUE
        UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
      )
    endif()
    find_package(TBB REQUIRED CONFIG PATHS ${ROCRAND_ROOT} NO_DEFAULT_PATH)
  endif(STL_DEPENDS_ON_TBB)

  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # HIP CPU Runtime (https://github.com/ROCm-Developer-Tools/HIP-CPU)
    find_package(hip_cpu_rt QUIET)
  endif()

  if(NOT hip_cpu_rt_FOUND)
    message(STATUS "Downloading and building HIP CPU Runtime.")
    set(HIP_CPU_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/hip-cpu" CACHE PATH "")
    download_project(
      PROJ                hip-cpu
      GIT_REPOSITORY      https://github.com/ROCm-Developer-Tools/HIP-CPU.git
      GIT_TAG             master
      INSTALL_DIR         "${HIP_CPU_ROOT}"
      CMAKE_ARGS          -Dhip_cpu_rt_BUILD_EXAMPLES=OFF -Dhip_cpu_rt_BUILD_TESTING=OFF -DCMAKE_PREFIX_PATH=${ROCRAND_ROOT} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
  endif()
  find_package(hip_cpu_rt REQUIRED CONFIG PATHS ${HIP_CPU_ROOT})
endif()

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
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.8.1
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
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
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/googlebenchmark CACHE PATH "")
    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
      # hip-clang cannot compile googlebenchmark for some reason
      set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
    endif()

    download_project(
      PROJ           googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.4.0
      INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
      CMAKE_ARGS     -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD   TRUE
      LOG_CONFIGURE  TRUE
      LOG_BUILD      TRUE
      LOG_INSTALL    TRUE
      BUILD_PROJECT  TRUE
      ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
  endif()
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()

# Find or download/install rocm-cmake project
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(
    DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
  )
  list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
  if(rocm_cmake_download_error_code)
    message(FATAL_ERROR "Error: downloading "
      "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
      "error_code: ${rocm_cmake_download_error_code} "
      "log: ${rocm_cmake_download_log} "
    )
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE rocm_cmake_unpack_error_code
  )
  if(rocm_cmake_unpack_error_code)
    message(FATAL_ERROR "Error: unpacking  ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
  endif()
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds OPTIONAL)
