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

#include "xpu/ze/engine_factory.hpp"

// #if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
// #include "gpu/intel/ze/engine.hpp"
// #endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

engine_factory_t::engine_factory_t(engine_kind_t engine_kind) {
    MAYBE_UNUSED(engine_kind);
    assert(engine_kind == engine_kind::gpu);
}

size_t engine_factory_t::count() const {
    uint32_t driver_count = 0;
    status_t status = status::success;

    status = ze::zeDriverGet(&driver_count, nullptr);
    if (status != status::success || driver_count == 0) return 0;

    std::vector<ze_driver_handle_t> drivers(driver_count);
    status = ze::zeDriverGet(&driver_count, drivers.data());
    if (status != status::success) return 0;

    uint32_t device_count = 0;
    status = ze::zeDeviceGet(drivers[0], &device_count, nullptr);
    if (status != status::success) return 0;

    return device_count;
}

status_t engine_factory_t::engine_create(
        impl::engine_t **engine, size_t index) const {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;

    uint32_t driver_count = 0;
    CHECK(ze::zeDriverGet(&driver_count, nullptr));
    VERROR_ENGINE(driver_count > 0, status::invalid_arguments,
            "no drivers to query devices were found");

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(ze::zeDriverGet(&driver_count, drivers.data()));
    driver = drivers[0];

    uint32_t device_count = 0;
    CHECK(ze::zeDeviceGet(driver, &device_count, nullptr));
    VERROR_ENGINE(index < device_count, status::invalid_arguments,
            "asked for device %zu but only %u devices are found", index,
            device_count);

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(ze::zeDeviceGet(driver, &device_count, devices.data()));
    device = devices[index];

    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    context_desc.pNext = nullptr;
    context_desc.flags = 0;

    CHECK(ze::zeContextCreate(driver, &context_desc, &context));

    return engine_create(engine, driver, device, context, index);
}

status_t engine_factory_t::engine_create(impl::engine_t **engine,
        ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context, size_t index) const {
    // return gpu::intel::ze::engine_create(
    //         engine, driver, device, context, index);
    return status::success;
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
