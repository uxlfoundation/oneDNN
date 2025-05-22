Build from Source {#dev_guide_build}
====================================

## Download the Source Code

Download [oneDNN source code](https://github.com/uxlfoundation/oneDNN/archive/main.zip)
or clone [the repository](https://github.com/uxlfoundation/oneDNN.git).

~~~sh
git clone https://github.com/uxlfoundation/oneDNN.git
~~~

## Build the Library

### Set up the environment for the compiler

Ensure that all [software dependencies](https://github.com/uxlfoundation/oneDNN#requirements-for-building-from-source)
are in place and have at least the minimal supported version.

### Generate the build system

The oneDNN build system is based on [CMake](https://cmake.org/cmake/help/latest/manual/cmake.1.html). Use the following command to generate a build system:
~~~sh
cmake -B <path-to-build> [-S <path-to-source>] [<options>]
~~~
In most cases, both `-B` and `-S` options are skipped with the assumption that
`<path-to-build>` is the current folder and `<path-to-source>` is the one
higher in the tree:
~~~sh
cd <path-to-onednn-source>
mkdir -p build ; cd build
cmake .. [<options>]
~~~
The following are a few useful options defined by CMake:

- `G` to specify build system generator (e.g. `"Visual Studio 17 2022"`,
  `Ninja`, `"Unix Makefiles"`).

- `CMAKE_INSTALL_PREFIX` to control the library installation location.

- `CMAKE_BUILD_TYPE` to select between build type (`Release`, `Debug`,
  `RelWithDebInfo`).

- `CMAKE_PREFIX_PATH` to specify directories to be searched for the
  dependencies located at non-standard locations.

See @ref dev_guide_build_options for detailed description of build-time
configuration options defined by oneDNN.

### Build the library

CMake provides a unified method for building a project, independent of the
generator or operating system used:
~~~sh
cmake --build <path-to-build> [<options>]
~~~
Full list of options can be found [here](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project).

### Linux/macOS

#### GCC, Clang, or Intel oneAPI DPC++/C++ Compiler

- Set up the environment for the compiler

~~~sh
# Uncomment the following lines to build with GCC
# export CC=gcc
# export CXX=g++

# Uncomment the following lines to build with Clang
# export CC=clang
# export CXX=clang++

# Uncomment the following lines to build with Intel oneAPI DPC++/C++ Compiler (x64 only)
# export CC=icx
# export CXX=icpx
~~~

- Generate the build system

~~~sh
mkdir -p build ; cd build
cmake ..
~~~

- Build the library

~~~sh
# Some generators, like Unix Makefiles, might default to single-threaded
# compilation. Parallelization can be controlled with:
cmake --build . --parallel $(nproc)
~~~

#### Intel oneAPI DPC++/C++ Compiler with SYCL runtime

- Set up the environment for the compiler

Intel oneAPI DPC++/C++ Compiler uses the `setvars.sh` script to set all
required variables. The command below assumes you installed to the default
folder. If you customized the installation folder, `setvars.sh` (Linux/macOS)
is in your custom folder.
~~~sh
source /opt/intel/oneapi/setvars.sh

# Set Intel oneAPI DPC++/C++ Compiler as default C and C++ compilers
export CC=icx
export CXX=icpx
~~~

- Generate the build system

~~~sh
mkdir -p build ; cd build
cmake .. -DDNNL_CPU_RUNTIME=SYCL \
         -DDNNL_GPU_RUNTIME=SYCL
~~~

@note Open-source version of oneAPI DPC++ Compiler does not have the icx driver,
use clang/clang++ instead. Open-source version of oneAPI DPC++ Compiler may not
contain OpenCL runtime. In this case, you can use `OPENCLROOT` CMake option or
environment variable of the same name to specify path to the OpenCL runtime if
it is installed in a custom location.

- Build the library

~~~sh
cmake --build . --parallel $(nproc)
~~~

#### GCC targeting AArch64 on x64 host

- Set up the environment for the compiler

~~~sh
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
~~~

- Generate the build system

~~~sh
mkdir -p build ; cd build
cmake .. -DCMAKE_SYSTEM_NAME=Linux \
         -DCMAKE_SYSTEM_PROCESSOR=AARCH64 \
         -DCMAKE_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib
~~~

- Build the library

~~~sh
cmake --build . --parallel $(nproc)
~~~

#### GCC with Arm Compute Library (ACL) on AArch64 host

- Set up the environment for the compiler

Download [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary)
or build it from source and set `ACL_ROOT_DIR` to directory where it is
installed.

~~~sh
export ACL_ROOT_DIR=<path/to/ComputeLibrary>
export CC=gcc
export CXX=g++
~~~

- Generate the build system

~~~sh
mkdir -p build ; cd build
cmake .. -DDNNL_AARCH64_USE_ACL=ON
~~~

- Build the library

~~~sh
cmake --build . --parallel $(nproc)
~~~

### Windows

#### Microsoft Visual C++ Compiler

- Set up the environment for the compiler

Microsoft Visual Studio uses the `VsDevCmd.bat` script to set all
required variables. The command below assumes you installed to the default
folder. If you customized the installation folder, `VsDevCmd.bat` is in your
custom folder.
~~~bat
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
~~~
or open `x64 Native Tools Command Prompt` from start menu instead.

- Generate the build system

~~~bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
~~~

- Build the library

~~~bat
cmake --build . --config=Release
~~~

@note Currently, the oneDNN build system has limited support for multi-config
 generators. Build configuration is based on the `CMAKE_BUILD_TYPE` option
 (`Release` by default), and CMake must be rerun from scratch every time
 the build type changes to apply the new build configuration. You can choose
 a specific build type with the `--config` option (the solution file supports
 both `Debug` and `Release` builds), but it must refer to the same build type
 (`Release`, `Debug`, etc.) as selected with the `CMAKE_BUILD_TYPE` option.

@note You can also open `oneDNN.sln` to build the project from the
Microsoft Visual Studio IDE.

#### Intel oneAPI DPC++/C++ Compiler with SYCL Runtime

- Set up the environment for the compiler

Intel oneAPI DPC++/C++ Compiler uses the `setvars.bat` script to set all
required variables. The command below assumes you installed to the default
folder. If you customized the installation folder, `setvars.bat` is in your
custom folder.
~~~bat
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

:: Set Intel oneAPI DPC++/C++ Compiler as default C and C++ compilers
set CC=icx
set CXX=icx
~~~
or open `Intel oneAPI Commmand Prompt` from start menu instead.

- Generate the build system

~~~bat
mkdir build
cd build

cmake .. -G Ninja ^
         -DDNNL_CPU_RUNTIME=SYCL ^
         -DDNNL_GPU_RUNTIME=SYCL
~~~

@warning Intel oneAPI DPC++/C++ Compiler on Windows requires CMake v3.23 or later.

@warning Intel oneAPI DPC++/C++ Compiler does not support CMake's Microsoft Visual
Studio generator.

@note Open-source version of oneAPI DPC++ Compiler does not have the icx driver,
use clang/clang++ instead. Open-source version of oneAPI DPC++ Compiler may not
contain OpenCL runtime. In this case, you can use `OPENCLROOT` CMake option or
environment variable of the same name to specify path to the OpenCL runtime if
it is installed in a custom location.

- Build the library

~~~bat
cmake --build .
~~~

## Validate the Build

After building the library, you can run a predefined test set using:
~~~sh
ctest
~~~
The [`ONEDNN_TEST_SET`](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html#onednn-test-set)
build option set during the build configuration determines determines the scope
and depth of the test set. Useful values are `SMOKE` (smallest set), `CI`
(default), and `NIGHTLY` (most comprehensive). The test set can be reconfigured
after the entire project has been built, and only the missing tests will be
compiled.
~~~sh
cmake .. -DONEDNN_TEST_SET=NIGHTLY
cmake --build .
ctest
~~~
ctest supports filtering the test set by using the `-R` option. For example,
to run only the GPU tests, use:
~~~sh
ctest -R gpu
~~~
Another useful option is `--output-on-failure`, which will print verbose output
in case a test fails. Full set of options can be found [here](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

## Build documentation

- Install the requirements
~~~sh
conda env create -f ../doc/environment.yml
conda activate onednn-doc
~~~

- Build the documentation
~~~sh
cmake --build . --target doc
~~~

## Install library

Install the library, headers, and documentation
~~~sh
cmake --build . --target install
~~~
The install directory is specified by the [CMAKE_INSTALL_PREFIX](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)
cmake variable. When installing in the default directory, the above command
needs to be run with administrative privileges using `sudo` on Linux/Mac or a
command prompt run as administrator on Windows.
