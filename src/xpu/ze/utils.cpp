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

#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

status_t ze_initialize() {
    static std::once_flag flag;
    static status_t status = status::success;
    std::call_once(flag, [&] {
        auto _init_drivers
                = find_ze_symbol<decltype(&::zeInitDrivers)>("zeInitDrivers");
        if (!_init_drivers) {
            status = status::runtime_error;
            return;
        }

        uint32_t driver_count = 0;
        ze_init_driver_type_desc_t desc;
        desc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
        desc.pNext = nullptr;
        desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
        if (_init_drivers(&driver_count, nullptr, &desc) != ZE_RESULT_SUCCESS) {
            status = status::runtime_error;
            return;
        }
    });

    return status;
}

xpu::device_uuid_t get_device_uuid(ze_device_handle_t device) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = nullptr;

    auto status = ze::zeDeviceGetProperties(device, &device_properties);
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

status_t get_device_index(size_t *index, ze_device_handle_t device) {
    uint32_t driver_count = 0;
    CHECK(ze::zeDriverGet(&driver_count, nullptr));
    if (driver_count <= 0) return status::invalid_arguments;

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(ze::zeDriverGet(&driver_count, drivers.data()));

    uint32_t device_count = 0;
    CHECK(ze::zeDeviceGet(drivers[0], &device_count, nullptr));

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(ze::zeDeviceGet(drivers[0], &device_count, devices.data()));

    for (size_t i = 0; i < device_count; i++) {
        if (device == devices[i]) {
            *index = i;

            return status::success;
        }
    }

    return status::invalid_arguments;
}

std::string get_kernel_name(ze_kernel_handle_t kernel) {
    std::string kernel_name;

    size_t kernel_name_size = 0;
    ze::zeKernelGetName(kernel, &kernel_name_size, nullptr);

    kernel_name.resize(kernel_name_size, 0);
    ze::zeKernelGetName(kernel, &kernel_name_size, &kernel_name[0]);

    // Remove the null terminator as std::string already includes it
    kernel_name.resize(kernel_name_size - 1);

    return kernel_name;
}

ze_memory_type_t get_pointer_type(
        ze_context_handle_t context, const void *ptr) {
    ze_memory_allocation_properties_t memory_allocation_properties;
    memory_allocation_properties.stype
            = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    memory_allocation_properties.pNext = nullptr;

    ze::zeMemGetAllocProperties(
            context, ptr, &memory_allocation_properties, nullptr);

    return memory_allocation_properties.type;
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
