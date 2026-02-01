/*******************************************************************************
* Copyright 2024 Intel Corporation
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

// Include for:
// - CL_PLATFORM_NOT_FOUND_KHR
// - CL_UUID_SIZE_KHR
// - CL_DEVICE_UUID_KHR
#include <CL/cl_ext.h>

#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/utils.hpp"

// XXX: Include this header for VERROR_ENGINE.
// TODO: Move VERROR_ENGINE and other similar macros to a separate file.
#include "common/engine.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

std::string get_kernel_name(cl_kernel kernel) {
    size_t name_size;
    cl_int err = xpu::ocl::clGetKernelInfo(
            kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &name_size);
    // Ignore error.
    UNUSED_OCL_RESULT(err);

    // Include null terminator explicitly - to safely overwrite it in
    // clGetKernelInfo
    std::string name(name_size, 0);
    err = xpu::ocl::clGetKernelInfo(
            kernel, CL_KERNEL_FUNCTION_NAME, name_size, &name[0], nullptr);
    // Ignore error.
    UNUSED_OCL_RESULT(err);

    // Remove the null terminator as std::string already includes it
    name.resize(name_size - 1);
    return name;
}

status_t get_device_index(size_t *index, cl_device_id device) {
    std::vector<cl_device_id> ocl_devices;
    cl_device_type device_type;
    OCL_CHECK(xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_TYPE,
            sizeof(device_type), &device_type, nullptr));
    CHECK(get_devices(&ocl_devices, device_type));

    // Search the top level device unconditionally
    auto parent_device = device;
    auto top_level_device = device;
    while (parent_device) {
        top_level_device = parent_device;
        OCL_CHECK(xpu::ocl::clGetDeviceInfo(top_level_device,
                CL_DEVICE_PARENT_DEVICE, sizeof(cl_device_id), &parent_device,
                nullptr));
    }

    // Find the top level device in the list
    auto it = std::find(
            ocl_devices.begin(), ocl_devices.end(), top_level_device);
    if (it != ocl_devices.end()) {
        *index = it - ocl_devices.begin();
        return status::success;
    } else {
        *index = SIZE_MAX;
        return status::invalid_arguments;
    }
}

cl_platform_id get_platform(cl_device_id device) {
    cl_platform_id platform;
    cl_int err = xpu::ocl::clGetDeviceInfo(
            device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    if (err != CL_SUCCESS) return nullptr;
    return platform;
}

cl_platform_id get_platform(engine_t *engine) {
    return utils::downcast<const xpu::ocl::engine_impl_t *>(engine->impl())
            ->platform();
}

status_t create_program(ocl::wrapper_t<cl_program> &ocl_program,
        cl_device_id dev, cl_context ctx, const xpu::binary_t &binary) {
    cl_int err;
    const unsigned char *binary_buffer = binary.data();
    size_t binary_size = binary.size();
    assert(binary_size > 0);

    ocl_program = xpu::ocl::clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_buffer, nullptr, &err);
    OCL_CHECK(err);
    err = xpu::ocl::clBuildProgram(
            ocl_program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    return status::success;
}

status_t check_device(
        engine_kind_t eng_kind, cl_device_id dev, cl_context ctx) {
    assert(dev && ctx);

    // Check device and context consistency.
    size_t dev_bytes;
    OCL_CHECK(xpu::ocl::clGetContextInfo(
            ctx, CL_CONTEXT_DEVICES, 0, nullptr, &dev_bytes));

    std::vector<cl_device_id> ctx_devices(dev_bytes / sizeof(cl_device_id));
    OCL_CHECK(xpu::ocl::clGetContextInfo(
            ctx, CL_CONTEXT_DEVICES, dev_bytes, &ctx_devices[0], nullptr));

    bool found = false;
    for (size_t i = 0; i < ctx_devices.size(); ++i) {
        if (ctx_devices[i] == dev) {
            found = true;
            break;
        }
    }
    VERROR_ENGINE(
            found, status::invalid_arguments, VERBOSE_DEVICE_CTX_MISMATCH);

    // Check engine kind and device consistency.
    cl_device_type dev_type;
    OCL_CHECK(xpu::ocl::clGetDeviceInfo(
            dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr));
    VERROR_ENGINE(!((eng_kind == engine_kind::cpu)
                          && (dev_type & CL_DEVICE_TYPE_CPU) == 0),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);
    VERROR_ENGINE(!((eng_kind == engine_kind::gpu)
                          && (dev_type & CL_DEVICE_TYPE_GPU) == 0),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    // Check that the platform is an Intel platform.
    cl_platform_id platform;
    OCL_CHECK(xpu::ocl::clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));

    VERROR_ENGINE(is_intel_platform(platform), status::invalid_arguments,
            VERBOSE_INVALID_PLATFORM, "ocl", "intel",
            get_platform_name(platform).c_str());
#endif
    return status::success;
}

status_t clone_kernel(cl_kernel kernel, cl_kernel *cloned_kernel) {
    cl_int err;
#if defined(CL_VERSION_2_1)
    *cloned_kernel = xpu::ocl::clCloneKernel(kernel, &err);
    OCL_CHECK(err);
#else
    // clCloneKernel is not available - recreate from the program.
    auto name = get_kernel_name(kernel);

    cl_program program;
    err = xpu::ocl::clGetKernelInfo(
            kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    OCL_CHECK(err);

    *cloned_kernel = xpu::ocl::clCreateKernel(program, name.c_str(), &err);
    OCL_CHECK(err);
#endif

    return status::success;
}

cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret) {
    return xpu::ocl::clCreateBuffer(
            context, flags, size, host_ptr, errcode_ret);
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
