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

#include "gpu/intel/l0/utils/utils.hpp"
#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "ngen_level_zero.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

std::string ze_result_to_string(ze_result_t r) {
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

#define ZE_CHECK_COMMON(f, retval) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = ze_result_to_string(res_); \
            VERROR(common, level_zero, "errcode %s", err_str_.c_str()); \
            return retval; \
        } \
    } while (false)

#define ZE_CHECK(f) ZE_CHECK_COMMON(f, status::runtime_error)
#define ZE_CHECK_VP(f) ZE_CHECK_COMMON(f, nullptr)

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    // Use LOAD_LIBRARY_SEARCH_SYSTEM32 flag to avoid DLL hijacking issue.
    HMODULE handle = LoadLibraryExA(
            "ze_loader.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#endif
    if (!handle) {
        VERROR(common, level_zero, "cannot find loader library");
        assert(!"not expected");
        return nullptr;
    }

    using zeInit_decl_t = ze_result_t (*)(ze_init_flags_t flags);
    const ze_init_flags_t default_ze_flags = 0;
#if defined(__linux__)
    static const ze_result_t ze_result = reinterpret_cast<zeInit_decl_t>(
            dlsym(handle, "zeInit"))(default_ze_flags);
    void *f = reinterpret_cast<void *>(dlsym(handle, symbol));
#elif defined(_WIN32)
    static const ze_result_t ze_result = reinterpret_cast<zeInit_decl_t>(
            GetProcAddress(handle, "zeInit"))(default_ze_flags);
    void *f = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#endif
    ZE_CHECK_VP(ze_result);

    if (!f) {
        VERROR(common, level_zero, "cannot find symbol: %s", symbol);
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

status_t func_zeDriverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers) {
    static auto f = find_ze_symbol<decltype(&zeDriverGet)>("zeDriverGet");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(pCount, phDrivers));
    return status::success;
}

status_t func_zeDriverGetProperties(
        ze_driver_handle_t hDriver, ze_driver_properties_t *pDriverProperties) {
    static auto f = find_ze_symbol<decltype(&zeDriverGetProperties)>(
            "zeDriverGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDriver, pDriverProperties));
    return status::success;
}

status_t func_zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount,
        ze_device_handle_t *phDevices) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGet)>("zeDeviceGet");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDriver, pCount, phDevices));
    return status::success;
}

status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetProperties)>(
            "zeDeviceGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

status_t func_zeDeviceGetComputeProperties(ze_device_handle_t hDevice,
        ze_device_compute_properties_t *pComputeProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetComputeProperties)>(
            "zeDeviceGetComputeProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pComputeProperties));
    return status::success;
}

status_t func_zeDeviceGetModuleProperties(ze_device_handle_t hDevice,
        ze_device_module_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetModuleProperties)>(
            "zeDeviceGetModuleProperties");

    if (!f) {
        VERROR(common, level_zero,
                "failed to find systolic query extension (maybe update the "
                "driver?)");
        return status::runtime_error;
    }
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

status_t func_zeDeviceGetMemoryAccessProperties(ze_device_handle_t hDevice,
        ze_device_memory_access_properties_t *pMemAccessProperties) {
    static auto f
            = find_ze_symbol<decltype(&zeDeviceGetMemoryAccessProperties)>(
                    "zeDeviceGetMemoryAccessProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pMemAccessProperties));
    return status::success;
}

status_t func_zeDeviceGetCacheProperties(ze_device_handle_t hDevice,
        uint32_t *pCount, ze_device_cache_properties_t *pCacheProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetCacheProperties)>(
            "zeDeviceGetCacheProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pCount, pCacheProperties));
    return status::success;
}

status_t func_zeContextCreate(ze_driver_handle_t hDriver,
        const ze_context_desc_t *desc, ze_context_handle_t *phContext) {
    static auto f
            = find_ze_symbol<decltype(&zeContextCreate)>("zeContextCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDriver, desc, phContext));
    return status::success;
}

status_t func_zeContextDestroy(ze_context_handle_t hContext) {
    static auto f
            = find_ze_symbol<decltype(&zeContextDestroy)>("zeContextDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext));
    return status::success;
}

status_t func_zeCommandListCreateImmediate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_command_queue_desc_t *altdesc,
        ze_command_list_handle_t *phCommandList) {
    static auto f = find_ze_symbol<decltype(&zeCommandListCreateImmediate)>(
            "zeCommandListCreateImmediate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, hDevice, altdesc, phCommandList));
    return status::success;
}

status_t func_zeCommandListDestroy(ze_command_list_handle_t hCommandList) {
    static auto f = find_ze_symbol<decltype(&zeCommandListDestroy)>(
            "zeCommandListDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList));
    return status::success;
}

status_t func_zeCommandListHostSynchronize(
        ze_command_list_handle_t hCommandList, uint64_t timeout) {
    static auto f = find_ze_symbol<decltype(&zeCommandListHostSynchronize)>(
            "zeCommandListHostSynchronize");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, timeout));
    return status::success;
}

status_t func_zeCommandListGetContextHandle(
        ze_command_list_handle_t hCommandList, ze_context_handle_t *phContext) {
    static auto f = find_ze_symbol<decltype(&zeCommandListGetContextHandle)>(
            "zeCommandListGetContextHandle");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, phContext));
    return status::success;
}

status_t func_zeCommandListAppendBarrier(ze_command_list_handle_t hCommandList,
        ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
        ze_event_handle_t *phWaitEvents) {
    static auto f = find_ze_symbol<decltype(&zeCommandListAppendBarrier)>(
            "zeCommandListAppendBarrier");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, hSignalEvent, numWaitEvents, phWaitEvents));
    return status::success;
}

status_t func_zeCommandListAppendMemoryCopy(
        ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr,
        size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
        ze_event_handle_t *phWaitEvents) {
    static auto f = find_ze_symbol<decltype(&zeCommandListAppendMemoryCopy)>(
            "zeCommandListAppendMemoryCopy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, dstptr, srcptr, size, hSignalEvent, numWaitEvents,
            phWaitEvents));
    return status::success;
}

status_t func_zeCommandListAppendMemoryFill(
        ze_command_list_handle_t hCommandList, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, ze_event_handle_t hSignalEvent,
        uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {
    static auto f = find_ze_symbol<decltype(&zeCommandListAppendMemoryFill)>(
            "zeCommandListAppendMemoryFill");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, ptr, pattern, pattern_size, size, hSignalEvent,
            numWaitEvents, phWaitEvents));
    return status::success;
}

status_t func_zeEventPoolCreate(ze_context_handle_t hContext,
        const ze_event_pool_desc_t *desc, uint32_t numDevices,
        ze_device_handle_t *phDevices, ze_event_pool_handle_t *phEventPool) {
    static auto f
            = find_ze_symbol<decltype(&zeEventPoolCreate)>("zeEventPoolCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, desc, numDevices, phDevices, phEventPool));
    return status::success;
}

status_t func_zeEventPoolDestroy(ze_event_pool_handle_t hEventPool) {
    static auto f = find_ze_symbol<decltype(&zeEventPoolDestroy)>(
            "zeEventPoolDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hEventPool));
    return status::success;
}

status_t func_zeEventCreate(ze_event_pool_handle_t hEventPool,
        const ze_event_desc_t *desc, ze_event_handle_t *phEvent) {
    static auto f = find_ze_symbol<decltype(&zeEventCreate)>("zeEventCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hEventPool, desc, phEvent));
    return status::success;
}

status_t func_zeEventDestroy(ze_event_handle_t hEvent) {
    static auto f = find_ze_symbol<decltype(&zeEventDestroy)>("zeEventDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hEvent));
    return status::success;
}

status_t func_zeEventHostSynchronize(
        ze_event_handle_t hEvent, uint64_t timeout) {
    static auto f = find_ze_symbol<decltype(&zeEventHostSynchronize)>(
            "zeEventHostSynchronize");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hEvent, timeout));
    return status::success;
}

status_t func_zeMemAllocShared(ze_context_handle_t hContext,
        const ze_device_mem_alloc_desc_t *device_desc,
        const ze_host_mem_alloc_desc_t *host_desc, size_t size,
        size_t alignment, ze_device_handle_t hDevice, void **pptr) {
    static auto f
            = find_ze_symbol<decltype(&zeMemAllocShared)>("zeMemAllocShared");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(
            hContext, device_desc, host_desc, size, alignment, hDevice, pptr));
    return status::success;
}

status_t func_zeMemAllocDevice(ze_context_handle_t hContext,
        const ze_device_mem_alloc_desc_t *device_desc, size_t size,
        size_t alignment, ze_device_handle_t hDevice, void **pptr) {
    static auto f
            = find_ze_symbol<decltype(&zeMemAllocDevice)>("zeMemAllocDevice");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, device_desc, size, alignment, hDevice, pptr));
    return status::success;
}

status_t func_zeMemAllocHost(ze_context_handle_t hContext,
        const ze_host_mem_alloc_desc_t *host_desc, size_t size,
        size_t alignment, void **pptr) {
    static auto f = find_ze_symbol<decltype(&zeMemAllocHost)>("zeMemAllocHost");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, host_desc, size, alignment, pptr));
    return status::success;
}

status_t func_zeMemFree(ze_context_handle_t hContext, void *ptr) {
    static auto f = find_ze_symbol<decltype(&zeMemFree)>("zeMemFree");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, ptr));
    return status::success;
}

status_t func_zeMemGetAllocProperties(ze_context_handle_t hContext,
        const void *ptr, ze_memory_allocation_properties_t *pMemAllocProperties,
        ze_device_handle_t *phDevice) {
    static auto f = find_ze_symbol<decltype(&zeMemGetAllocProperties)>(
            "zeMemGetAllocProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, ptr, pMemAllocProperties, phDevice));
    return status::success;
}

status_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, hDevice, desc, phModule, phBuildLog));
    return status::success;
}

status_t func_zeModuleDestroy(ze_module_handle_t hModule) {
    static auto f
            = find_ze_symbol<decltype(&zeModuleDestroy)>("zeModuleDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule));
    return status::success;
}

status_t func_zeModuleGetNativeBinary(ze_module_handle_t hModule, size_t *pSize,
        uint8_t *pModuleNativeBinary) {
    static auto f = find_ze_symbol<decltype(&zeModuleGetNativeBinary)>(
            "zeModuleGetNativeBinary");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule, pSize, pModuleNativeBinary));
    return status::success;
}

status_t func_zeKernelCreate(ze_module_handle_t hModule,
        const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel) {
    static auto f = find_ze_symbol<decltype(&zeKernelCreate)>("zeKernelCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule, desc, phKernel));
    return status::success;
}

status_t func_zeKernelDestroy(ze_kernel_handle_t hKernel) {
    static auto f
            = find_ze_symbol<decltype(&zeKernelDestroy)>("zeKernelDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel));
    return status::success;
}

status_t func_zeKernelSetArgumentValue(ze_kernel_handle_t hKernel,
        uint32_t argIndex, size_t argSize, const void *pArgValue) {
    static auto f = find_ze_symbol<decltype(&zeKernelSetArgumentValue)>(
            "zeKernelSetArgumentValue");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel, argIndex, argSize, pArgValue));
    return status::success;
}

status_t func_zeKernelGetName(
        ze_kernel_handle_t hKernel, size_t *pSize, char *pName) {
    static auto f
            = find_ze_symbol<decltype(&zeKernelGetName)>("zeKernelGetName");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel, pSize, pName));
    return status::success;
}

status_t func_zeGetKernelBinary(
        ze_kernel_handle_t hKernel, size_t *pSize, uint8_t *pKernelBinary) {
    static auto f = find_ze_symbol<decltype(&zeKernelGetBinaryExp)>(
            "zeKernelGetBinaryExp");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel, pSize, pKernelBinary));
    return status::success;
}

status_t func_zeKernelSetGroupSize(ze_kernel_handle_t hKernel,
        uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ) {
    static auto f = find_ze_symbol<decltype(&zeKernelSetGroupSize)>(
            "zeKernelSetGroupSize");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel, groupSizeX, groupSizeY, groupSizeZ));
    return status::success;
}

status_t func_zeKernelSuggestGroupSize(ze_kernel_handle_t hKernel,
        uint32_t globalSizeX, uint32_t globalSizeY, uint32_t globalSizeZ,
        uint32_t *groupSizeX, uint32_t *groupSizeY, uint32_t *groupSizeZ) {
    static auto f = find_ze_symbol<decltype(&zeKernelSuggestGroupSize)>(
            "zeKernelSuggestGroupSize");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hKernel, globalSizeX, globalSizeY, globalSizeZ, groupSizeX,
            groupSizeY, groupSizeZ));
    return status::success;
}

status_t func_zeCommandListAppendLaunchKernel(
        ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
        const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
        uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents) {
    static auto f = find_ze_symbol<decltype(&zeCommandListAppendLaunchKernel)>(
            "zeCommandListAppendLaunchKernel");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hCommandList, hKernel, pLaunchFuncArgs, hSignalEvent,
            numWaitEvents, phWaitEvents));
    return status::success;
}

status_t get_device_ip(ze_device_handle_t device, uint32_t &ip_version) {
    ze_device_ip_version_ext_t device_ip_version_ext = {};
    device_ip_version_ext.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;
    device_ip_version_ext.pNext = nullptr;

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &device_ip_version_ext;

    CHECK(func_zeDeviceGetProperties(device, &device_properties));

    ip_version = device_ip_version_ext.ipVersion;

    return status::success;
}

status_t get_l0_device_enabled_systolic_intel(
        ze_device_handle_t device, bool &mayiuse_systolic) {
    ze_intel_device_module_dp_exp_properties_t
            intel_device_module_dp_exp_properties
            = {};
    intel_device_module_dp_exp_properties.stype
            = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES;
    intel_device_module_dp_exp_properties.pNext = nullptr;

    ze_device_module_properties_t device_module_properties = {};
    device_module_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    device_module_properties.pNext = &intel_device_module_dp_exp_properties;

    CHECK(func_zeDeviceGetModuleProperties(device, &device_module_properties));

    mayiuse_systolic = intel_device_module_dp_exp_properties.flags
            & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS;

    return status::success;
}

status_t get_l0_device_enabled_native_float_atomics(
        ze_device_handle_t device, uint64_t &native_extensions) {
    using namespace gpu::intel::compute;

    ze_float_atomic_ext_properties_t float_atomic_ext_properties = {};
    float_atomic_ext_properties.stype
            = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;
    float_atomic_ext_properties.pNext = nullptr;

    ze_device_module_properties_t device_module_properties = {};
    device_module_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    device_module_properties.pNext = &float_atomic_ext_properties;

    CHECK(func_zeDeviceGetModuleProperties(device, &device_module_properties));

    ze_device_fp_atomic_ext_flags_t atomic_load_store
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE;
    ze_device_fp_atomic_ext_flags_t atomic_add
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD;
    ze_device_fp_atomic_ext_flags_t atomic_min_max
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX;

    if ((float_atomic_ext_properties.fp16Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_load_store;
    if ((float_atomic_ext_properties.fp16Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_add;
    if ((float_atomic_ext_properties.fp16Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_min_max;

    if ((float_atomic_ext_properties.fp32Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_load_store;
    if ((float_atomic_ext_properties.fp32Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_add;
    if ((float_atomic_ext_properties.fp32Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_min_max;

    if ((float_atomic_ext_properties.fp64Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_load_store;
    if ((float_atomic_ext_properties.fp64Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_add;
    if ((float_atomic_ext_properties.fp64Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_min_max;

    return status::success;
}

status_t get_l0_device_eu_count(ze_device_handle_t device, int &eu_count) {
    ze_eu_count_ext_t eu_count_ext = {};
    eu_count_ext.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;
    eu_count_ext.pNext = nullptr;

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &eu_count_ext;

    CHECK(func_zeDeviceGetProperties(device, &device_properties));

    eu_count = eu_count_ext.numTotalEUs;

    return status::success;
}

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, compute::gpu_product_t &product_,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels) {
    using namespace ngen;
    ngen::Product product = LevelZeroCodeGenerator<HW::Unknown>::detectHWInfo(
            context, device);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(ngen::getCore(product.family));
    std::memcpy(&product_, &product, sizeof(ngen::Product));

    mayiuse_systolic = false;
    if (get_l0_device_enabled_systolic_intel(device, mayiuse_systolic)
            != status::success)
        mayiuse_systolic = false;

    /* Some old drivers do not report systolic availability. Manually override
       systolic availability based on product family. */
    switch (product.family) {
        case ProductFamily::DG2:
        case ProductFamily::ARL:
        case ProductFamily::PVC: mayiuse_systolic = true;
        default: break;
    }

    CHECK(get_l0_device_enabled_native_float_atomics(
            device, native_extensions));

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels, engine);
    if (status != status::success) mayiuse_ngen_kernels = false;

    ip_version = 0;

    return get_device_ip(device, ip_version);
}

xpu::device_uuid_t get_device_uuid(const ze_device_handle_t device) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = nullptr;

    auto status = func_zeDeviceGetProperties(device, &device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &device_id = device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)device_id[i]) << shift);
    }

    return xpu::device_uuid_t(uuid[0], uuid[1]);
}

status_t get_device_index(const ze_device_handle_t device, size_t *index) {
    uint32_t driver_count = 0;
    CHECK(func_zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(func_zeDriverGet(&driver_count, drivers.data()));

    uint32_t device_count = 0;
    CHECK(func_zeDeviceGet(drivers[0], &device_count, nullptr));

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(func_zeDeviceGet(drivers[0], &device_count, devices.data()));

    for (size_t i = 0; i < device_count; i++) {
        if (device == devices[i]) {
            *index = i;

            return status::success;
        }
    }

    return status::invalid_arguments;
}

status_t get_kernel_name(
        const ze_kernel_handle_t kernel, std::string &kernel_name) {
    size_t kernel_name_size = 0;
    CHECK(func_zeKernelGetName(kernel, &kernel_name_size, nullptr));

    kernel_name.resize(kernel_name_size, 0);
    CHECK(func_zeKernelGetName(kernel, &kernel_name_size, &kernel_name[0]));

    // Remove the null terminator as std::string already includes it
    kernel_name.resize(kernel_name_size - 1);

    return status::success;
}

status_t get_kernel_binary(
        const ze_kernel_handle_t kernel, xpu::binary_t &binary) {
    size_t binary_size = 0;
    CHECK(func_zeGetKernelBinary(kernel, &binary_size, nullptr));

    binary.resize(binary_size);
    CHECK(func_zeGetKernelBinary(kernel, &binary_size, binary.data()));

    return status::success;
}

status_t create_kernels(const ze_device_handle_t device,
        const ze_context_handle_t context,
        const std::vector<const char *> &kernel_names,
        const xpu::binary_t &binary, ze_module_handle_t *module,
        std::vector<ze_kernel_handle_t> &kernels) {
    ze_module_desc_t module_desc;
    module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_desc.pNext = nullptr;
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = binary.size();
    module_desc.pInputModule = binary.data();
    module_desc.pBuildFlags = "";
    module_desc.pConstants = nullptr;

    CHECK(func_zeModuleCreate(context, device, &module_desc, module, nullptr));

    kernels.resize(kernel_names.size(), nullptr);
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (kernel_names[i] == nullptr) {
            kernels[i] = nullptr;
            continue;
        }

        ze_kernel_desc_t kernel_desc = {};
        kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        kernel_desc.pNext = nullptr;
        kernel_desc.flags = 0;
        kernel_desc.pKernelName = kernel_names[i];

        ze_kernel_handle_t kernel;
        CHECK(func_zeKernelCreate(*module, &kernel_desc, &kernel));

        kernels[i] = kernel;
    }

    return status::success;
}

ze_memory_type_t get_pointer_type(
        const ze_context_handle_t context, const void *ptr) {
    ze_memory_allocation_properties_t memory_allocation_properties;
    memory_allocation_properties.stype
            = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    memory_allocation_properties.pNext = nullptr;

    func_zeMemGetAllocProperties(
            context, ptr, &memory_allocation_properties, nullptr);

    return memory_allocation_properties.type;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
