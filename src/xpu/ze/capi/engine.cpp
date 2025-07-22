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

#include "oneapi/dnnl/dnnl_ze.h"

#include "common/utils.hpp"

#include "xpu/ze/engine_factory.hpp"
#include "xpu/ze/engine_impl.hpp"
#include "xpu/ze/utils.hpp"

using namespace dnnl::impl;

status_t dnnl_ze_interop_engine_create(engine_t **engine,
        ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context) {
    bool args_ok = !utils::any_null(engine, driver, device, context);
    if (!args_ok) return status::invalid_arguments;

    xpu::ze::engine_factory_t f(engine_kind::gpu);

    size_t index;
    CHECK(xpu::ze::get_device_index(&index, device));

    return f.engine_create(engine, driver, device, context, index);
}

status_t dnnl_ze_interop_engine_get_context(
        engine_t *engine, ze_context_handle_t *context) {
    bool args_ok = !utils::any_null(engine, context)
            && (engine->runtime_kind() == runtime_kind::ze);
    if (!args_ok) return status::invalid_arguments;

    auto *ze_engine_impl
            = utils::downcast<const xpu::ze::engine_impl_t *>(engine->impl());
    *context = ze_engine_impl->context();

    return status::success;
}

status_t dnnl_ze_interop_engine_get_device(
        engine_t *engine, ze_device_handle_t *device) {
    bool args_ok = !utils::any_null(engine, device)
            && (engine->runtime_kind() == runtime_kind::ze);
    if (!args_ok) return status::invalid_arguments;

    auto *ze_engine_impl
            = utils::downcast<const xpu::ze::engine_impl_t *>(engine->impl());
    *device = ze_engine_impl->device();

    return status::success;
}

status_t dnnl_ze_interop_engine_get_driver(
        engine_t *engine, ze_driver_handle_t *driver) {
    bool args_ok = !utils::any_null(engine, driver)
            && (engine->runtime_kind() == runtime_kind::ze);
    if (!args_ok) return status::invalid_arguments;

    auto *ze_engine_impl
            = utils::downcast<const xpu::ze::engine_impl_t *>(engine->impl());
    *driver = ze_engine_impl->driver();

    return status::success;
}
