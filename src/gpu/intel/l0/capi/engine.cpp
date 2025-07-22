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

using namespace dnnl::impl;

dnnl_status_t dnnl_l0_interop_engine_create(dnnl_engine_t *engine,
        const ze_driver_handle_t adriver, const ze_device_handle_t adevice,
        const ze_context_handle_t acontext) {
    bool args_ok = !utils::any_null(engine, adriver, adevice, acontext);
    if (!args_ok) return status::invalid_arguments;

    gpu::intel::l0::engine_factory_t f(engine_kind::gpu);

    size_t index;
    CHECK(gpu::intel::l0::get_device_index(adevice, &index));

    return f.engine_create(engine, adriver, adevice, acontext, index);
}

dnnl_status_t dnnl_l0_interop_engine_get_context(
        dnnl_engine_t engine, ze_context_handle_t context) {
    bool args_ok = !utils::any_null(engine, context)
            && (engine->runtime_kind() == runtime_kind::l0);
    if (!args_ok) return status::invalid_arguments;

    auto *l0_engine = utils::downcast<const gpu::intel::l0::engine_t *>(engine);
    context = l0_engine->context();

    return status::success;
}

dnnl_status_t dnnl_l0_interop_engine_get_device(
        dnnl_engine_t engine, ze_device_handle_t device) {
    bool args_ok = !utils::any_null(engine, device)
            && (engine->runtime_kind() == runtime_kind::l0);
    if (!args_ok) return status::invalid_arguments;

    auto *l0_engine = utils::downcast<const gpu::intel::l0::engine_t *>(engine);
    device = l0_engine->device();

    return status::success;
}

dnnl_status_t dnnl_l0_interop_engine_get_driver(
        dnnl_engine_t engine, ze_driver_handle_t driver) {
    bool args_ok = !utils::any_null(engine, driver)
            && (engine->runtime_kind() == runtime_kind::l0);
    if (!args_ok) return status::invalid_arguments;

    auto *l0_engine = utils::downcast<const gpu::intel::l0::engine_t *>(engine);
    driver = l0_engine->driver();

    return status::success;
}
