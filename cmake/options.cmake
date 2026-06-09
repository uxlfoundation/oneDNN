#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage different library options
#===============================================================================

if(options_cmake_included)
    return()
endif()
set(options_cmake_included true)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(DNNL_IS_MAIN_PROJECT TRUE)
endif()

# ==============
# Common options
# ==============

# ---------------------
# Library configuration
# ---------------------

set(DNNL_LIBRARY_TYPE "SHARED" CACHE STRING
    "Specifies whether oneDNN library should be SHARED or STATIC")

set(DNNL_LIBRARY_NAME "dnnl" CACHE STRING
    "Specifies name of the library. For example, user can use this variable to
     specify custom library names for CPU and GPU configurations to safely
     include them into their CMake project via add_subdirectory")

message(STATUS "DNNL_LIBRARY_NAME: ${DNNL_LIBRARY_NAME}")

set(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING
    "Specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel C++ Compilers the default option is `-xSSE4.1` which instructs
      the compiler to generate the code for the processors that support SSE4.1
      instructions. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection and Clang, the default option is `-msse4.1` which
      behaves similarly to the description above.

    - For Clang and GCC compilers on RISC-V architecture this option accepts `-march=<ISA-string>` flag
      to control whether or not oneDNN should be compiled with RVV Intrinsics. Use this option with
      `-march=rv64gc` or `-march=rv64gcv` value to compile oneDNN with and without RVV Intrinsics respectively.
      If the option is not provided, CMake will decide based on the active toolchain and compiler flags.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distribution maintainer) one may want to specify DNNL_ARCH_OPT_FLAGS=\"\"
    to not use any host-specific instructions")

set(DNNL_BLAS_VENDOR "NONE" CACHE STRING
    "Use an external BLAS library. Valid values:
      - NONE (default)
        Use internal BLAS implementation. Recommended in most situations.
      - ACCELERATE: Accelerate BLAS
      - ARMPL: Arm Performance Libraries
      - ANY: FindBLAS will search default library paths for a known BLAS
        installation. This vendor is supported for performance analysis
        purposes only.")

set(DNNL_DPCPP_HOST_COMPILER "DEFAULT" CACHE STRING
    "Specifies host compiler for Intel oneAPI DPC++ Compiler")

set(DNNL_INSTALL_MODE "DEFAULT" CACHE STRING
    "Specifies installation mode; supports DEFAULT and BUNDLE.
    When BUNDLE option is set oneDNN will be installed as a bundle
    which contains examples and benchdnn.") # internal
if (NOT "${DNNL_INSTALL_MODE}" MATCHES "^(DEFAULT|BUNDLE)$")
    message(FATAL_ERROR "Unsupported install mode: ${DNNL_INSTALL_MODE}")
endif()

# -------------
# Functionality
# -------------

option(ONEDNN_BUILD_GRAPH "Enables Graph API and related optimizations" ON)

option(DNNL_ENABLE_CONCURRENT_EXEC
    "Disables sharing a common scratchpad between primitives.
    This option must be turned ON if there is a possibility of executing
    distinct primitives concurrently.
    CAUTION: enabling this option increases memory consumption."
    OFF)

option(DNNL_ENABLE_PRIMITIVE_CACHE "Enables primitive cache." ON)

set(DNNL_ENABLE_WORKLOAD "TRAINING" CACHE STRING
    "Specifies a set of functionality to be available at build time. Designed to
    decrease the final memory disk footprint of the shared object or application
    statically linked against the library. Valid values:
    - TRAINING (the default). Includes all functionality to be enabled.
    - INFERENCE. Includes only forward propagation kind functionality and their
      dependencies.")
if(NOT "${DNNL_ENABLE_WORKLOAD}" MATCHES "^(TRAINING|INFERENCE)$")
    message(FATAL_ERROR "Unsupported workload type: ${DNNL_ENABLE_WORKLOAD}")
endif()

set(DNNL_ENABLE_PRIMITIVE "ALL" CACHE STRING
    "Specifies a set of primitives to be available at build time. Valid values:
    - ALL (the default). Includes all primitives to be enabled.
    - <PRIMITIVE_NAME>. Includes only the selected primitive to be enabled.
      Possible values are: BATCH_NORMALIZATION, BINARY, CONCAT, CONVOLUTION,
      DECONVOLUTION, ELTWISE, GATED_MLP, GROUP_NORMALIZATION, INNER_PRODUCT,
      LAYER_NORMALIZATION, LRN, MATMUL, POOLING, PRELU, REDUCTION, REORDER,
      RESAMPLING, RNN, SDPA, SHUFFLE, SOFTMAX, SUM.
    - <PRIMITIVE_NAME>;<PRIMITIVE_NAME>;... Includes only selected primitives to
      be enabled at build time. This is treated as CMake string, thus, semicolon
      is a mandatory delimiter between names. This is the way to specify several
      primitives to be available in the final binary.")

option(DNNL_EXPERIMENTAL
    "Enables experimental features in oneDNN.
    When enabled, each experimental feature has to be individually selected
    using environment variables."
    OFF)

option(DNNL_EXPERIMENTAL_UKERNEL
    "Enables experimental functionality for ukernels. This option works
    independently from DNNL_EXPERIMENTAL."
    OFF)

option(DNNL_EXPERIMENTAL_GROUPED_MEMORY
    "Enables experimental support for grouped memory format and grouped GEMM.
    This option works independently from DNNL_EXPERIMENTAL."
    OFF)

option(DNNL_EXPERIMENTAL_PROFILING
    "Enables experimental profiling capabilities. This option works independently
    from DNNL_EXPERIMENTAL."
    OFF)

option(DNNL_EXPERIMENTAL_LOGGING
    "Enables experimental functionality for logging. This option works
    independently from DNNL_EXPERIMENTAL."
    OFF)

option(DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
    "Enables experimental SYCL OpenCL kernel compiler extension. This option
    works independently from DNNL_EXPERIMENTAL."
    OFF)

# -------------------
# Debug and profiling
# -------------------

option(DNNL_ENABLE_JIT_PROFILING
    "Enables registration of oneDNN kernels that are generated at
    runtime with VTune Profiler (on by default). Without the
    registrations, VTune Profiler would report data collected inside
    the kernels as `outside any known module`."
    ON)

option(DNNL_ENABLE_ITT_TASKS
    "Enables ITT Tasks tagging feature and tag all primitive execution
    (on by default). VTune Profiler can group profiling results based
    on those ITT tasks and show corresponding timeline information."
    ON)

option(ONEDNN_ENABLE_GRAPH_DUMP "Enables saving subgraphs defined using
    Graph API to disk when ONEDNN_GRAPH_DUMP environment variable is set."
    ON)

option(DNNL_VERBOSE "Enables verbose mode output when ONEDNN_VERBOSE
    environment variable set" ON)

option(DNNL_DEV_MODE "Enables internal tracing capabilities" OFF)

# -------------
# Documentation
# -------------

option(DNNL_BUILD_DOC "Enables building documentation" OFF)

set(ONEDNN_DOC_VERSIONS_JSON "" CACHE STRING "Location of JSON file for
    PyData Sphinx Theme version switcher. Must be a stable, persistent,
    fully resolved URL. Enables documentation version switcher when set.")

# ----------
# Validation
# ----------

option(DNNL_BUILD_EXAMPLES "Enables building examples" ${DNNL_IS_MAIN_PROJECT})

option(DNNL_BUILD_TESTS "Enables building tests" ${DNNL_IS_MAIN_PROJECT})

set(DNNL_TEST_SET "CI" CACHE STRING
    "Specifies the testing coverage. The variable consists of two parts:
    the set value defining the number of test cases, and the modifiers for
    testing commands. The input is expected in the CMake list style - a
    semicolon separated string, e.g., DNNL_TEST_SET=CI;NO_CORR.")

set(DNNL_CODE_COVERAGE "OFF" CACHE STRING
    "Enables code coverage instrumentation. Currently only gcov supported")
if(NOT "${DNNL_CODE_COVERAGE}" MATCHES "^(OFF|GCOV)$")
    message(FATAL_ERROR "Unsupported code coverage tool: ${DNNL_CODE_COVERAGE}")
endif()

set(DNNL_USE_CLANG_SANITIZER "" CACHE STRING
    "Instructs build system to use a Clang sanitizer. Possible values:
    Address: enables AddressSanitizer
    Leak: enables LeakSanitizer
    Memory: enables MemorySanitizer
    MemoryWithOrigin: enables MemorySanitizer with origin tracking
    Thread: enables ThreadSanitizer
    Undefined: enables UndefinedBehaviourSanitizer
    This feature is only available on Linux.")

set(DNNL_USE_CLANG_TIDY "NONE" CACHE STRING
    "Instructs build system to use clang-tidy. Valid values:
    - NONE (default)
      Clang-tidy is disabled.
    - CHECK
      Enables checks from .clang-tidy.
    - FIX
      Enables checks from .clang-tidy and fix found issues.
    This feature is only available on Linux.")

option(DNNL_WERROR "Enables treating warnings as errors" OFF)

option(DNNL_BUILD_FOR_CI
    "Specifies whether oneDNN library will use special testing environment for
    internal testing processes"
    OFF) # internal

option(DNNL_ENABLE_MEM_DEBUG "Enables memory-related debug functionality,
    such as buffer overflow (default) and underflow, using gtests and benchdnn.
    Additionally, this option enables testing of out-of-memory handling by the
    library, such as failed memory allocations, using primitive-related gtests.
    This feature is experimental and is only available on Linux." OFF) # internal

option(DNNL_ENABLE_STACK_CHECKER "Enables stack checker that can be used to get
    information about stack consumption for a particular library entry point.
    This feature is only available on Linux (see src/common/stack_checker.hpp
    for more details).
    Note: This option requires enabling concurrent scratchpad
    (DNNL_ENABLE_CONCURRENT_EXEC)." OFF) # internal

option(BENCHDNN_USE_RDPMC
    "Enables rdpmc counter to report precise CPU frequency in benchdnn.
    CAUTION: may not work on all cpus (hence disabled by default)"
    OFF) # internal

# ===========
# CPU options
# ===========

# ------------------
# Common CPU options
# ------------------

set(DNNL_CPU_RUNTIME "OMP" CACHE STRING
    "Specifies the threading runtime for CPU engines;
    supports OMP (default), TBB, SYCL, SEQ, or THREADPOOL.")

if(NOT "${DNNL_CPU_RUNTIME}" MATCHES "^(NONE|OMP|TBB|SEQ|THREADPOOL|DPCPP|SYCL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_CPU_RUNTIME}")
endif()

set(_DNNL_TEST_THREADPOOL_IMPL "STANDALONE" CACHE STRING
    "Specifies which threadpool implementation will be used in tests when
    the librariy is built with DNNL_CPU_RUNTIME=THREADPOOL. Valid values: STANDALONE, EIGEN,
    EIGEN_ASYNC, TBB")
if(NOT "${_DNNL_TEST_THREADPOOL_IMPL}" MATCHES "^(STANDALONE|TBB|EIGEN|EIGEN_ASYNC)$")
    message(FATAL_ERROR
        "Unsupported threadpool implementation: ${_DNNL_TEST_THREADPOOL_IMPL}")
endif()

set(TBBROOT "" CACHE STRING
    "Path to Threading Building Blocks (TBB) package.")

# ---------------
# x64 CPU options
# ---------------

option(DNNL_ENABLE_MAX_CPU_ISA
    "Enables control of CPU ISA detected by oneDNN via DNNL_MAX_CPU_ISA
    environment variable and dnnl_set_max_cpu_isa() function" ON)

option(DNNL_ENABLE_CPU_ISA_HINTS
    "Enables control of CPU ISA specific hints by oneDNN via DNNL_CPU_ISA_HINTS
    environment variable and dnnl_set_cpu_isa_hints() function" ON)

set(DNNL_ENABLE_PRIMITIVE_CPU_ISA "ALL" CACHE STRING
    "Specifies a set of implementations using specific CPU ISA to be available
    at build time. Regardless of value chosen, compiler-based optimized
    implementations will always be available. Valid values:
    - ALL (the default). Includes all ISA to be enabled.
    - <ISA_NAME>. Includes selected and all \"less\" ISA to be enabled.
      Possible values are: SSE41, AVX2, AVX512, AMX. The linear order is
      SSE41 < AVX2 < AVX512 < AMX. It means that if user selects, e.g. AVX2 ISA,
      SSE41 implementations will also be available at build time.")

set(ONEDNN_ENABLE_GEMM_KERNELS_ISA "ALL" CACHE STRING
    "Specifies an ISA set of GeMM kernels residing in x64/gemm folder to be
    available at build time. Valid values:
    - ALL (the default). Includes all ISA kernels to be enabled.
    - NONE. Removes all kernels and interfaces.
    - <ISA_NAME>. Enables all ISA up to ISA_NAME included.
      Possible value are: SSE41, AVX2, AVX512. The linear order is
      SSE41 < AVX2 < AVX512 < AMX (or ALL). It means that if user selects, e.g.
      AVX2 ISA, SSE41 kernels will also present at build time.")

option(DNNL_SAFE_RBP
    "Prohibits RBP register clobbering in JIT kernels. Use this option to enable
    runtime profiling with tools like Flame Graph."
    OFF)

# -------------------
# AArch64 CPU options
# -------------------

option(DNNL_AARCH64_USE_ACL "Enables use of AArch64 optimised functions
    from Arm Compute Library.
    This is only supported on AArch64 builds and assumes there is a
    functioning Compute Library build available at the location specified by the
    environment variable ACL_ROOT_DIR." OFF)

# ===========
# GPU options
# ===========

# ------------------
# Common GPU options
# ------------------

set(DNNL_GPU_RUNTIME "NONE" CACHE STRING
    "Specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines), OCL (OpenCL GPU engines)
    or SYCL (SYCL GPU engines).

    Using OpenCL for GPU requires setting OPENCLROOT if the libraries are
    installed in a non-standard location.")

if(NOT "${DNNL_GPU_RUNTIME}" MATCHES "^(OCL|NONE|DPCPP|SYCL|ZE)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GPU_RUNTIME}")
endif()

set(DNNL_GPU_VENDOR "NONE" CACHE STRING
    "When DNNL_GPU_RUNTIME is not NONE DNNL_GPU_VENDOR specifies target GPU
    vendor for GPU engines. Can be INTEL (default when DNNL_GPU_RUNTIME is
    not NONE), NVIDIA, AMD, or GENERIC.")

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE" AND DNNL_GPU_VENDOR STREQUAL "NONE")
    set(DNNL_GPU_VENDOR "INTEL")
endif()

if(NOT "${DNNL_GPU_VENDOR}" MATCHES "^(NONE|INTEL|NVIDIA|AMD|GENERIC)$")
    message(FATAL_ERROR "Unsupported GPU vendor: ${DNNL_GPU_VENDOR}")
endif()

# -----------------
# Intel GPU options
# -----------------

set(DNNL_ENABLE_PRIMITIVE_GPU_ISA "ALL" CACHE STRING
    "Specifies a set of implementations using specific GPU ISA to be available
    at build time. Regardless of value chosen, reference OpenCL-based
    implementations will always be available. Valid values:
    - ALL (the default). Includes all ISA to be enabled.
    - <ISA_NAME>;<ISA_NAME>;... Includes only selected ISA to be enabled.
      Possible values are: XELP, XEHP, XEHPG, XEHPC, XE2, XE3, XE3P.")

set(OPENCLROOT "" CACHE STRING
    "Path to OpenCL SDK.
    Use this option to specify custom location for OpenCL.")

set(DNNL_OCL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/opencl" CACHE STRING
    "Path to OpenCL headers. Defaults to the headers bundled in
    third_party/opencl.")

set(DNNL_ZE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third_party/level_zero" CACHE STRING
    "Path to Level Zero headers. Defaults to the headers bundled in
    third_party/level_zero.")

option(DNNL_DISABLE_GPU_REF_KERNELS
    "Disables use of reference implementations on Intel GPUs."
    OFF) # internal

# -----------------
# Other GPU options
# -----------------

set(DNNL_AMD_SYCL_KERNELS_TARGET_ARCH "" CACHE STRING
    "Specifies the target architecture (e.g. gfx90a when compiling on AMD MI210)
    to be used for compiling generic SYCL kernels for AMD vendor.
    When this option is set to a valid architecture (see LLVM target column in
    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus
    for supported architectures), the generic SYCL kernels will be enabled for AMD
    vendor. If not set, the SYCL kernels will not be compiled.
    Warning: This option is temporary and will be removed as soon as the compiler
    stops to require specifying the target architecture. After removing the option
    the generic SYCL kernels will always be enabled for AMD vendor.") # internal
