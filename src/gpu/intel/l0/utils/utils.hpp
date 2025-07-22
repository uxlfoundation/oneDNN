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

#ifndef GPU_INTEL_L0_UTILS_HPP
#define GPU_INTEL_L0_UTILS_HPP

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

status_t func_zeDriverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers);
status_t func_zeDriverGetProperties(
        ze_driver_handle_t hDriver, ze_driver_properties_t *pDriverProperties);
status_t func_zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount,
        ze_device_handle_t *phDevices);
status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties);
status_t func_zeDeviceGetComputeProperties(ze_device_handle_t hDevice,
        ze_device_compute_properties_t *pComputeProperties);
status_t func_zeDeviceGetModuleProperties(ze_device_handle_t hDevice,
        ze_device_module_properties_t *pDeviceProperties);
status_t func_zeDeviceGetMemoryAccessProperties(ze_device_handle_t hDevice,
        ze_device_memory_access_properties_t *pMemAccessProperties);
status_t func_zeDeviceGetCacheProperties(ze_device_handle_t hDevice,
        uint32_t *pCount, ze_device_cache_properties_t *pCacheProperties);
status_t func_zeContextCreate(ze_driver_handle_t hDriver,
        const ze_context_desc_t *desc, ze_context_handle_t *phContext);
status_t func_zeContextDestroy(ze_context_handle_t hContext);
status_t func_zeCommandListCreateImmediate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_command_queue_desc_t *altdesc,
        ze_command_list_handle_t *phCommandList);
status_t func_zeCommandListDestroy(ze_command_list_handle_t hCommandList);
status_t func_zeCommandListHostSynchronize(
        ze_command_list_handle_t hCommandList, uint64_t timeout);
status_t func_zeCommandListAppendBarrier(ze_command_list_handle_t hCommandList,
        ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
        ze_event_handle_t *phWaitEvents);
status_t func_zeCommandListAppendMemoryCopy(
        ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr,
        size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
        ze_event_handle_t *phWaitEvents);
status_t func_zeCommandListAppendMemoryFill(
        ze_command_list_handle_t hCommandList, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, ze_event_handle_t hSignalEvent,
        uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
status_t func_zeEventPoolCreate(ze_context_handle_t hContext,
        const ze_event_pool_desc_t *desc, uint32_t numDevices,
        ze_device_handle_t *phDevices, ze_event_pool_handle_t *phEventPool);
status_t func_zeEventPoolDestroy(ze_event_pool_handle_t hEventPool);
status_t func_zeEventCreate(ze_event_pool_handle_t hEventPool,
        const ze_event_desc_t *desc, ze_event_handle_t *phEvent);
status_t func_zeEventDestroy(ze_event_handle_t hEvent);
status_t func_zeMemAllocShared(ze_context_handle_t hContext,
        const ze_device_mem_alloc_desc_t *device_desc,
        const ze_host_mem_alloc_desc_t *host_desc, size_t size,
        size_t alignment, ze_device_handle_t hDevice, void **pptr);
status_t func_zeMemAllocDevice(ze_context_handle_t hContext,
        const ze_device_mem_alloc_desc_t *device_desc, size_t size,
        size_t alignment, ze_device_handle_t hDevice, void **pptr);
status_t func_zeMemAllocHost(ze_context_handle_t hContext,
        const ze_host_mem_alloc_desc_t *host_desc, size_t size,
        size_t alignment, void **pptr);
status_t func_zeMemFree(ze_context_handle_t hContext, void *ptr);
status_t func_zeMemGetAllocProperties(ze_context_handle_t hContext,
        const void *ptr, ze_memory_allocation_properties_t *pMemAllocProperties,
        ze_device_handle_t *phDevice);
status_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule, ze_module_build_log_handle_t *phBuildLog);
status_t func_zeModuleDestroy(ze_module_handle_t hModule);
status_t func_zeModuleGetNativeBinary(ze_module_handle_t hModule, size_t *pSize,
        uint8_t *pModuleNativeBinary);
status_t func_zeKernelCreate(ze_module_handle_t hModule,
        const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel);
status_t func_zeKernelDestroy(ze_kernel_handle_t hKernel);
status_t func_zeKernelSetArgumentValue(ze_kernel_handle_t hKernel,
        uint32_t argIndex, size_t argSize, const void *pArgValue);
status_t func_zeKernelGetName(
        ze_kernel_handle_t hKernel, size_t *pSize, char *pName);
status_t func_zeGetKernelBinary(
        ze_kernel_handle_t hKernel, size_t *pSize, uint8_t *pKernelBinary);
status_t func_zeCommandListAppendLaunchKernel(
        ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
        const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
        uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, compute::gpu_product_t &product,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels);
xpu::device_uuid_t get_device_uuid(const ze_device_handle_t device);
status_t get_device_index(const ze_device_handle_t device, size_t *index);
status_t get_kernel_name(
        const ze_kernel_handle_t kernel, std::string &kernel_name);
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

#endif // GPU_INTEL_L0_UTILS_HPP
