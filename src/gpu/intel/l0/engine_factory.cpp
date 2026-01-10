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

#include "gpu/intel/l0/engine_factory.hpp"
#include "gpu/intel/l0/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

engine_factory_t::engine_factory_t(engine_kind_t engine_kind)
    : engine_kind_(engine_kind) {
    assert(utils::one_of(engine_kind_, engine_kind::gpu));
}

size_t engine_factory_t::count() const {
    uint32_t driver_count = 0;
    l0::zeDriverGet(&driver_count, nullptr);

    std::vector<ze_driver_handle_t> drivers(driver_count);
    l0::zeDriverGet(&driver_count, drivers.data());

    uint32_t device_count = 0;
    l0::zeDeviceGet(drivers[0], &device_count, nullptr);

    return device_count;
}

status_t engine_factory_t::engine_create(
        impl::engine_t **engine, size_t index) const {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;

    uint32_t driver_count = 0;
    CHECK(l0::zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(l0::zeDriverGet(&driver_count, drivers.data()));
    driver = drivers[0];

    uint32_t device_count = 0;
    CHECK(l0::zeDeviceGet(driver, &device_count, nullptr));
    VERROR_ENGINE(index < device_count, status::invalid_arguments,
            "asked for device %zu but only %u devices are found", index,
            device_count);

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(l0::zeDeviceGet(driver, &device_count, devices.data()));
    device = devices[index];

    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    context_desc.pNext = nullptr;
    context_desc.flags = 0;

    CHECK(l0::zeContextCreate(driver, &context_desc, &context));

    return engine_create(engine, driver, device, context, index);
}

status_t engine_factory_t::engine_create(impl::engine_t **engine,
        const ze_driver_handle_t driver, const ze_device_handle_t device,
        const ze_context_handle_t context, size_t index) const {
    return gpu::intel::l0::engine_create(
            engine, engine_kind_, driver, device, context, index);
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
