/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_L0_COMPILER_HPP
#define GPU_INTEL_L0_COMPILER_HPP

#include "xpu/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

static inline cl_program ocl_compile(const cl_device_id device,
        const cl_context context, const char *kernel_code,
        const char *options) {
    cl_int err;
    cl_program program = xpu::ocl::clCreateProgramWithSource(
            context, 1, &kernel_code, nullptr, &err);
    if (err != CL_SUCCESS) return nullptr;
    err = xpu::ocl::clBuildProgram(
            program, 1, &device, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        xpu::ocl::clReleaseProgram(program);
        return nullptr;
    }
    return program;
}

inline bool ocl_mayiuse_microkernels(const cl_device_id device,
        const cl_context context, const char *kernel_code) {
    cl_program program = ocl_compile(device, context, kernel_code, "");
    if (program) {
        xpu::ocl::clReleaseProgram(program);
        return true;
    }
    return false;
}

inline status_t ocl_build_kernels(const cl_device_id device,
        const cl_context context, const char *kernel_code, const char *options,
        xpu::binary_t &binary) {
    cl_program program = ocl_compile(device, context, kernel_code, options);
    if (!program) return status::runtime_error;

    size_t binary_size = 0;
    OCL_CHECK(xpu::ocl::clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
            sizeof(binary_size), &binary_size, nullptr));

    binary.resize(binary_size);
    auto binary_data = binary.data();
    OCL_CHECK(xpu::ocl::clGetProgramInfo(program, CL_PROGRAM_BINARIES,
            sizeof(binary_data), &binary_data, nullptr));

    OCL_CHECK(xpu::ocl::clReleaseProgram(program));

    return status::success;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_COMPILER_HPP
