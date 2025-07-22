/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_l0.h"

#include "common/utils.hpp"
#include "gpu/intel/l0/engine.hpp"
#include "gpu/intel/l0/engine_factory.hpp"

dnnl_status_t dnnl_l0_interop_engine_create(dnnl_engine_t *engine,
        const ze_driver_handle_t adriver, const ze_device_handle_t adevice,
        const ze_context_handle_t acontext) {
    bool args_ok
            = !dnnl::impl::utils::any_null(engine, adriver, adevice, acontext);
    if (!args_ok) return dnnl::impl::status::invalid_arguments;

    dnnl::impl::gpu::intel::l0::engine_factory_t f(
            dnnl::impl::engine_kind::gpu);

    size_t index;
    CHECK(dnnl::impl::gpu::intel::l0::get_device_index(adevice, &index));

    return f.engine_create(engine, adriver, adevice, acontext, index);
}

dnnl_status_t dnnl_l0_interop_engine_get_context(
        dnnl_engine_t engine, ze_context_handle_t context) {
    bool args_ok = !dnnl::impl::utils::any_null(engine, context)
            && (engine->runtime_kind() == dnnl::impl::runtime_kind::l0);
    if (!args_ok) return dnnl::impl::status::invalid_arguments;

    auto *l0_engine = dnnl::impl::utils::downcast<
            const dnnl::impl::gpu::intel::l0::engine_t *>(engine);
    context = l0_engine->context();
    return dnnl::impl::status::success;
}

dnnl_status_t dnnl_l0_interop_engine_get_device(
        dnnl_engine_t engine, ze_device_handle_t device) {
    bool args_ok = !dnnl::impl::utils::any_null(engine, device)
            && (engine->runtime_kind() == dnnl::impl::runtime_kind::l0);
    if (!args_ok) return dnnl::impl::status::invalid_arguments;

    auto *l0_engine = dnnl::impl::utils::downcast<
            const dnnl::impl::gpu::intel::l0::engine_t *>(engine);
    device = l0_engine->device();
    return dnnl::impl::status::success;
}

dnnl_status_t dnnl_l0_interop_engine_get_driver(
        dnnl_engine_t engine, ze_driver_handle_t driver) {
    bool args_ok = !dnnl::impl::utils::any_null(engine, driver)
            && (engine->runtime_kind() == dnnl::impl::runtime_kind::l0);
    if (!args_ok) return dnnl::impl::status::invalid_arguments;

    auto *l0_engine = dnnl::impl::utils::downcast<
            const dnnl::impl::gpu::intel::l0::engine_t *>(engine);
    driver = l0_engine->driver();
    return dnnl::impl::status::success;
}
