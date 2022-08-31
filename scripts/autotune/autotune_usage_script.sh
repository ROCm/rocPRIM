#!/bin/bash

set -e

gpu_arch=$1;

#git clone -b autotune_preview  https://github.com/ROCmSoftwarePlatform/rocPRIM.git
mkdir -p build && cd build

CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake -D BUILD_BENCHMARK=ON -D BENCHMARK_CONFIG_TUNING=ON ..

# Only build the benchmark_device_reduce since the others don't work on the preview branch
make -j `nproc` benchmark_device_reduce

#Generate json output of the benchmark
./benchmark/benchmark_device_reduce --benchmark_out=./device_reduce_$gpu_arch.json --benchmark_out_format=json

mkdir -p ../configs

#Generate configuration based on the output of the benchmark
python3 ../scripts/autotune/create_optimization.py  --benchmark_files $gpu_arch:./device_reduce_$gpu_arch.json -p  ../configs

#Install the generated configuration
python3 ../scripts/autotune/merge_config.py ../rocprim/include/rocprim/device/device_reduce_config.hpp ../configs/device_reduce $gpu_arch
#cp ../configs/device_reduce ../rocprim/include/rocprim/device/device_reduce_config.hpp

git add ../rocprim/include/rocprim/device/device_reduce_config.hpp
git commit -m "Updating config for ${gpu_arch}"
# git push
#Test if the installed configuration works
#mkdir ../build_noautotune &&  cd ../build_noautotune
#CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake -D BUILD_BENCHMARK=ON -D BENCHMARK_CONFIG_TUNING=OFF  -D CMAKE_CXX_FLAGS="-DROCPRIM_TARGET_ARCH=$gpu_arch" ..

#make -j `nproc` benchmark_device_reduce

#./benchmark/benchmark_device_reduce

