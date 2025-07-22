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

#ifndef GPU_INTEL_L0_UTILS_UTILS_HPP
#define GPU_INTEL_L0_UTILS_UTILS_HPP

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include "gpu/intel/compute/kernel.hpp"

#include "level_zero/ze_api.h"
#include "level_zero/ze_intel_gpu.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

static inline std::string to_string(ze_result_t r) {
#define ZE_STATUS_CASE(status) \
    case status: return #status
    switch (r) {
        ZE_STATUS_CASE(ZE_RESULT_SUCCESS);
        ZE_STATUS_CASE(ZE_RESULT_NOT_READY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNINITIALIZED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNKNOWN);
        ZE_STATUS_CASE(ZE_RESULT_FORCE_UINT32);
        default: return std::to_string((int)r);
    }
#undef ZE_STATUS_CASE
}

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = to_string(res_); \
            VERROR(common, level_zero, "errcode %s", err_str_.c_str()); \
            return status::runtime_error; \
        } \
    } while (false)

#if defined(_WIN32)
#define L0_LIB_NAME "ze_loader.dll"
#elif defined(__linux__)
#define L0_LIB_NAME "libze_loader.so.1"
#endif

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)xpu::find_symbol(L0_LIB_NAME, symbol);
}
#undef L0_LIB_NAME

#define INDIRECT_L0_CALL(f) \
    template <typename... Args> \
    status_t f(Args &&...args) { \
        const ze_init_flags_t default_ze_flags = 0; \
        static auto init_ = find_ze_symbol<decltype(&::zeInit)>("zeInit"); \
        if (!init_) return status::runtime_error; \
        ZE_CHECK(init_(default_ze_flags)); \
        static auto f_ = find_ze_symbol<decltype(&::f)>(#f); \
        if (!f_) return status::runtime_error; \
        ZE_CHECK(f_(std::forward<Args>(args)...)); \
        return status::success; \
    }
INDIRECT_L0_CALL(zeDriverGet)
INDIRECT_L0_CALL(zeDriverGetProperties)
INDIRECT_L0_CALL(zeDeviceGet)
INDIRECT_L0_CALL(zeDeviceGetProperties)
INDIRECT_L0_CALL(zeDeviceGetComputeProperties)
INDIRECT_L0_CALL(zeDeviceGetModuleProperties)
INDIRECT_L0_CALL(zeDeviceGetMemoryAccessProperties)
INDIRECT_L0_CALL(zeDeviceGetCacheProperties)
INDIRECT_L0_CALL(zeContextCreate)
INDIRECT_L0_CALL(zeContextDestroy)
INDIRECT_L0_CALL(zeCommandListCreateImmediate)
INDIRECT_L0_CALL(zeCommandListDestroy)
INDIRECT_L0_CALL(zeCommandListHostSynchronize)
INDIRECT_L0_CALL(zeCommandListGetContextHandle)
INDIRECT_L0_CALL(zeCommandListAppendBarrier)
INDIRECT_L0_CALL(zeCommandListAppendMemoryCopy)
INDIRECT_L0_CALL(zeCommandListAppendMemoryFill)
INDIRECT_L0_CALL(zeEventPoolCreate)
INDIRECT_L0_CALL(zeEventPoolDestroy)
INDIRECT_L0_CALL(zeEventCreate)
INDIRECT_L0_CALL(zeEventDestroy)
INDIRECT_L0_CALL(zeEventHostSynchronize)
INDIRECT_L0_CALL(zeEventQueryKernelTimestamp)
INDIRECT_L0_CALL(zeMemAllocShared)
INDIRECT_L0_CALL(zeMemAllocDevice)
INDIRECT_L0_CALL(zeMemAllocHost)
INDIRECT_L0_CALL(zeMemFree)
INDIRECT_L0_CALL(zeMemGetAllocProperties)
INDIRECT_L0_CALL(zeModuleCreate)
INDIRECT_L0_CALL(zeModuleDestroy)
INDIRECT_L0_CALL(zeModuleBuildLogGetString)
INDIRECT_L0_CALL(zeModuleBuildLogDestroy)
INDIRECT_L0_CALL(zeModuleGetNativeBinary)
INDIRECT_L0_CALL(zeKernelCreate)
INDIRECT_L0_CALL(zeKernelDestroy)
INDIRECT_L0_CALL(zeKernelSetArgumentValue)
INDIRECT_L0_CALL(zeKernelGetName)
INDIRECT_L0_CALL(zeKernelGetBinaryExp)
INDIRECT_L0_CALL(zeKernelSetGroupSize)
INDIRECT_L0_CALL(zeKernelSuggestGroupSize)
INDIRECT_L0_CALL(zeCommandListAppendLaunchKernel)
#undef INDIRECT_L0_CALL

class event_wrapper_t {
public:
    event_wrapper_t(ze_event_handle_t event);
    ~event_wrapper_t();
    operator ze_event_handle_t() const;

private:
    ze_event_handle_t event_;

    event_wrapper_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(event_wrapper_t);
};

class event_pool_wrapper_t {
public:
    event_pool_wrapper_t(ze_event_pool_handle_t event_pool);
    ~event_pool_wrapper_t();
    operator ze_event_pool_handle_t() const;

private:
    ze_event_pool_handle_t event_pool_;

    event_pool_wrapper_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(event_pool_wrapper_t);
};

class module_wrapper_t {
public:
    module_wrapper_t(ze_module_handle_t module);
    ~module_wrapper_t();
    operator ze_module_handle_t() const;

private:
    ze_module_handle_t module_;

    module_wrapper_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(module_wrapper_t);
};

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, compute::gpu_product_t &product,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels);
xpu::device_uuid_t get_device_uuid(const ze_device_handle_t device);
status_t get_device_index(const ze_device_handle_t device, size_t *index);
std::string get_kernel_name(const ze_kernel_handle_t kernel);
status_t get_module_binary(
        const ze_module_handle_t module, xpu::binary_t &binary);
status_t get_kernel_binary(
        const ze_kernel_handle_t kernel, xpu::binary_t &binary);
bool mayiuse_microkernels(const ze_device_handle_t device,
        const ze_context_handle_t context, const std::string &code);
status_t compile_ocl_module_to_binary(const ze_device_handle_t device,
        const ze_context_handle_t context, const std::string &code,
        const std::string &options, xpu::binary_t &binary);
status_t create_kernels(const ze_device_handle_t device,
        const ze_context_handle_t context,
        const std::vector<const char *> &kernel_names,
        const xpu::binary_t &binary, ze_module_handle_t *module,
        std::vector<ze_kernel_handle_t> &kernels);
ze_memory_type_t get_pointer_type(const ze_context_handle_t, const void *ptr);

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_UTILS_UTILS_HPP
