# rocPRIM

## The rocPRIM repository is retired, please use the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) repository

rocPRIM is a header-only library that provides HIP parallel primitives. You can use this library to
develop performant GPU-accelerated code on AMD ROCm platforms.

## Requirements

* Git
* CMake (3.16 or later)
* AMD [ROCm](https://rocm.docs.amd.com/en/latest/) platform (1.8.2 or later)
  * Including
    [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang)
    compiler
* C++17
* Python 3.6 or higher (HIP on Windows only, required only for install script)
* Visual Studio 2019 with Clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

## Documentation

> [!NOTE]
> The published rocPRIM documentation is available [here](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).
