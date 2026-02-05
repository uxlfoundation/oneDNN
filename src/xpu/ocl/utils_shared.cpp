/*******************************************************************************
* Copyright 2026 Intel Corporation
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

// Note: this file shares function bodies with Level Zero runtime. The reason
// it's a dedicated file is an engine_t dependency in utils.cpp that doesn't
// let it link if compiled for Level Zero backend. Thus, this file contains
// pure CL code bodies only.
// Bodies can be moved back with this file removal once the necessity in CL
// objects ends for Level Zero backend.

// Include for:
// - CL_PLATFORM_NOT_FOUND_KHR
// - CL_UUID_SIZE_KHR
// - CL_DEVICE_UUID_KHR
#include <CL/cl_ext.h>

#include "common/utils.hpp"

#include "xpu/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t convert_to_dnnl(cl_int cl_status) {
    switch (cl_status) {
        case CL_SUCCESS: return status::success;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY: return status::out_of_memory;
        case CL_DEVICE_NOT_FOUND:
        case CL_DEVICE_NOT_AVAILABLE:
        case CL_COMPILER_NOT_AVAILABLE:
        case CL_PROFILING_INFO_NOT_AVAILABLE:
        case CL_MEM_COPY_OVERLAP:
        case CL_IMAGE_FORMAT_MISMATCH:
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        case CL_BUILD_PROGRAM_FAILURE:
        case CL_MAP_FAILURE:
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        case CL_COMPILE_PROGRAM_FAILURE:
        case CL_LINKER_NOT_AVAILABLE:
        case CL_LINK_PROGRAM_FAILURE:
        case CL_DEVICE_PARTITION_FAILED:
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        case CL_INVALID_PLATFORM:
        case CL_INVALID_DEVICE:
        case CL_SYMBOL_NOT_FOUND: return status::runtime_error;
        case CL_INVALID_VALUE:
        case CL_INVALID_DEVICE_TYPE:
        case CL_INVALID_CONTEXT:
        case CL_INVALID_QUEUE_PROPERTIES:
        case CL_INVALID_COMMAND_QUEUE:
        case CL_INVALID_HOST_PTR:
        case CL_INVALID_MEM_OBJECT:
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        case CL_INVALID_IMAGE_SIZE:
        case CL_INVALID_SAMPLER:
        case CL_INVALID_BINARY:
        case CL_INVALID_BUILD_OPTIONS:
        case CL_INVALID_PROGRAM:
        case CL_INVALID_PROGRAM_EXECUTABLE:
        case CL_INVALID_KERNEL_NAME:
        case CL_INVALID_KERNEL_DEFINITION:
        case CL_INVALID_KERNEL:
        case CL_INVALID_ARG_INDEX:
        case CL_INVALID_ARG_VALUE:
        case CL_INVALID_ARG_SIZE:
        case CL_INVALID_KERNEL_ARGS:
        case CL_INVALID_WORK_DIMENSION:
        case CL_INVALID_WORK_GROUP_SIZE:
        case CL_INVALID_WORK_ITEM_SIZE:
        case CL_INVALID_GLOBAL_OFFSET:
        case CL_INVALID_EVENT_WAIT_LIST:
        case CL_INVALID_EVENT:
        case CL_INVALID_OPERATION:
        case CL_INVALID_GL_OBJECT:
        case CL_INVALID_BUFFER_SIZE:
        case CL_INVALID_MIP_LEVEL:
        case CL_INVALID_GLOBAL_WORK_SIZE: return status::invalid_arguments;

        default: return status::runtime_error;
    }
}

// Ordered by value as defined by opencl
const char *convert_cl_int_to_str(cl_int cl_status) {
#define CL_STATUS_CASE(status) \
    case status: return #status
    switch (cl_status) {
        CL_STATUS_CASE(CL_SUCCESS);
        CL_STATUS_CASE(CL_DEVICE_NOT_FOUND);
        CL_STATUS_CASE(CL_DEVICE_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_COMPILER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CL_STATUS_CASE(CL_OUT_OF_RESOURCES);
        CL_STATUS_CASE(CL_OUT_OF_HOST_MEMORY);
        CL_STATUS_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_COPY_OVERLAP);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_MISMATCH);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CL_STATUS_CASE(CL_BUILD_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_MAP_FAILURE);
        CL_STATUS_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CL_STATUS_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        CL_STATUS_CASE(CL_COMPILE_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_LINKER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_LINK_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_DEVICE_PARTITION_FAILED);
        CL_STATUS_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_INVALID_VALUE);
        CL_STATUS_CASE(CL_INVALID_DEVICE_TYPE);
        CL_STATUS_CASE(CL_INVALID_PLATFORM);
        CL_STATUS_CASE(CL_INVALID_DEVICE);
        CL_STATUS_CASE(CL_INVALID_CONTEXT);
        CL_STATUS_CASE(CL_INVALID_QUEUE_PROPERTIES);
        CL_STATUS_CASE(CL_INVALID_COMMAND_QUEUE);
        CL_STATUS_CASE(CL_INVALID_HOST_PTR);
        CL_STATUS_CASE(CL_INVALID_MEM_OBJECT);
        CL_STATUS_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CL_STATUS_CASE(CL_INVALID_IMAGE_SIZE);
        CL_STATUS_CASE(CL_INVALID_SAMPLER);
        CL_STATUS_CASE(CL_INVALID_BINARY);
        CL_STATUS_CASE(CL_INVALID_BUILD_OPTIONS);
        CL_STATUS_CASE(CL_INVALID_PROGRAM);
        CL_STATUS_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_NAME);
        CL_STATUS_CASE(CL_INVALID_KERNEL_DEFINITION);
        CL_STATUS_CASE(CL_INVALID_KERNEL);
        CL_STATUS_CASE(CL_INVALID_ARG_INDEX);
        CL_STATUS_CASE(CL_INVALID_ARG_VALUE);
        CL_STATUS_CASE(CL_INVALID_ARG_SIZE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_ARGS);
        CL_STATUS_CASE(CL_INVALID_WORK_DIMENSION);
        CL_STATUS_CASE(CL_INVALID_WORK_GROUP_SIZE);
        CL_STATUS_CASE(CL_INVALID_WORK_ITEM_SIZE);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_OFFSET);
        CL_STATUS_CASE(CL_INVALID_EVENT_WAIT_LIST);
        CL_STATUS_CASE(CL_INVALID_EVENT);
        CL_STATUS_CASE(CL_INVALID_OPERATION);
        CL_STATUS_CASE(CL_INVALID_GL_OBJECT);
        CL_STATUS_CASE(CL_INVALID_BUFFER_SIZE);
        CL_STATUS_CASE(CL_INVALID_MIP_LEVEL);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
        CL_STATUS_CASE(CL_SYMBOL_NOT_FOUND);
#undef CL_STATUS_CASE
        default: return "unknown macro name";
    }
}

bool is_intel_platform(cl_platform_id platform) {
    auto name = get_platform_name(platform);
    return name.find("Intel") != std::string::npos;
}

std::string get_platform_name(cl_platform_id platform) {
    size_t name_size;
    cl_int err = xpu::ocl::clGetPlatformInfo(
            platform, CL_PLATFORM_NAME, 0, nullptr, &name_size);
    // Ignore error.
    UNUSED_OCL_RESULT(err);

    // Include null terminator explicitly - to safely overwrite it in
    // clGetPlatformInfo
    std::string name(name_size, 0);
    err = xpu::ocl::clGetPlatformInfo(
            platform, CL_PLATFORM_NAME, name_size, &name[0], nullptr);
    // Ignore error.
    UNUSED_OCL_RESULT(err);

    // Remove the null terminator as std::string already includes it
    name.resize(name_size - 1);
    return name;
}

status_t get_devices(std::vector<cl_device_id> *devices,
        cl_device_type device_type, cl_uint vendor_id /* = 0x8086 */) {
    cl_uint num_platforms = 0;

    cl_int err = xpu::ocl::clGetPlatformIDs(0, nullptr, &num_platforms);
    // No platforms - a valid scenario
    if (err == CL_PLATFORM_NOT_FOUND_KHR) return status::success;

    OCL_CHECK(err);

    std::vector<cl_platform_id> platforms(num_platforms);
    OCL_CHECK(
            xpu::ocl::clGetPlatformIDs(num_platforms, &platforms[0], nullptr));

    for (size_t i = 0; i < platforms.size(); ++i) {
        if (!is_intel_platform(platforms[i])) continue;

        cl_uint num_devices = 0;
        cl_int err = xpu::ocl::clGetDeviceIDs(
                platforms[i], device_type, 0, nullptr, &num_devices);

        if (!utils::one_of(err, CL_SUCCESS, CL_DEVICE_NOT_FOUND)) {
            return status::runtime_error;
        }

        if (num_devices != 0) {
            std::vector<cl_device_id> plat_devices;
            plat_devices.resize(num_devices);
            OCL_CHECK(xpu::ocl::clGetDeviceIDs(platforms[i], device_type,
                    num_devices, &plat_devices[0], nullptr));

            // Use the devices for the requested vendor only.
            for (size_t j = 0; j < plat_devices.size(); ++j) {
                cl_uint v_id;
                OCL_CHECK(xpu::ocl::clGetDeviceInfo(plat_devices[j],
                        CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &v_id, nullptr));
                if (v_id == vendor_id) { devices->push_back(plat_devices[j]); }
            }
        }
    }
    // No devices found but still return success
    return status::success;
}

status_t get_devices(std::vector<cl_device_id> *devices,
        std::vector<wrapper_t<cl_device_id>> *sub_devices,
        cl_device_type device_type) {
    std::vector<cl_device_id> devices_tmp;
    std::vector<wrapper_t<cl_device_id>> sub_devices_tmp;

    CHECK(get_devices(&devices_tmp, device_type));

    for (cl_device_id d : devices_tmp) {
        cl_uint max_sub_devices;
        cl_device_partition_property properties[3]
                = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                        CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0};
        cl_int err = xpu::ocl::clCreateSubDevices(
                d, properties, 0, nullptr, &max_sub_devices);
        if (err == CL_DEVICE_PARTITION_FAILED) continue;
        OCL_CHECK(err);
        std::vector<cl_device_id> sds(max_sub_devices);
        OCL_CHECK(xpu::ocl::clCreateSubDevices(
                d, properties, max_sub_devices, sds.data(), nullptr));
        for (cl_device_id sd : sds)
            sub_devices_tmp.emplace_back(sd);
    }
    *devices = devices_tmp;
    *sub_devices = std::move(sub_devices_tmp);
    return status::success;
}

#ifndef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
status_t get_device_uuid(xpu::device_uuid_t &uuid, cl_device_id ocl_dev) {
    // This function is used only with SYCL that works with OpenCL 3.0
    // that supports `cl_khr_device_uuid` extension.
#if defined(cl_khr_device_uuid)
    static_assert(
            CL_UUID_SIZE_KHR == 16, "CL_UUID_SIZE_KHR is expected to be 16");

    cl_uchar ocl_dev_uuid[CL_UUID_SIZE_KHR] = {};
    OCL_CHECK(xpu::ocl::clGetDeviceInfo(ocl_dev, CL_DEVICE_UUID_KHR,
            CL_UUID_SIZE_KHR, ocl_dev_uuid, nullptr));

    uint64_t uuid_packed[CL_UUID_SIZE_KHR / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < CL_UUID_SIZE_KHR; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid_packed[i / sizeof(uint64_t)]
                |= (((uint64_t)ocl_dev_uuid[i]) << shift);
    }
    uuid = xpu::device_uuid_t(uuid_packed[0], uuid_packed[1]);
    return status::success;
#endif
    return status::runtime_error;
}
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
