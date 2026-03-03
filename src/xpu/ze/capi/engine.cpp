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
    VERROR_ENGINE(engine, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(driver, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(device, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(context, status::invalid_arguments, VERBOSE_NULL_ARG);

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

status_t dnnl_ze_interop_engine_create_from_cache_blob(engine_t **engine,
        ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context, size_t size, const uint8_t *cache_blob) {
    VERROR_ENGINE(engine, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(driver, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(device, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(context, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(cache_blob, status::invalid_arguments, VERBOSE_NULL_ARG);
    VERROR_ENGINE(size > 0, status::invalid_arguments, VERBOSE_NULL_ARG);

    xpu::ze::engine_factory_t f(engine_kind::gpu);

    size_t index;
    CHECK(xpu::ze::get_device_index(&index, device));

    const std::vector<uint8_t> cb(cache_blob, cache_blob + size);
    return f.engine_create(engine, driver, device, context, index, cb);
}

status_t dnnl_ze_interop_engine_get_cache_blob(
        engine_t *engine, size_t *size, uint8_t *cache_blob) {
    VERROR_ENGINE(engine->kind() == engine_kind::gpu, status::invalid_arguments,
            VERBOSE_BAD_ENGINE_KIND);
    VERROR_ENGINE(size, status::invalid_arguments, VERBOSE_NULL_ARG);

    if (!cache_blob) {
        size_t sz = 0;
        CHECK(engine->get_cache_blob_size(&sz));
        (*size) = sz;
        return status::success;
    }

    CHECK(engine->get_cache_blob(*size, cache_blob));
    return status::success;
}

status_t dnnl_ze_interop_engine_get_cache_blob_id(ze_driver_handle_t driver,
        ze_device_handle_t device, size_t *size, uint8_t *cache_blob) {
    VERROR_ENGINE(size, status::invalid_arguments, VERBOSE_NULL_ARG);
    size_t &id_size = *size;

    // Get oneDNN version.
    auto version = dnnl_version();

    // Get device name.
    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    CHECK(xpu::ze::zeDeviceGetProperties(device, &device_properties));
    auto device_name = std::string(device_properties.name);

    // Get driver version size.
    // Note: copied from engine_impl. See a comment there.
    ze_result_t (*pfnGetDriverVersionFn)(ze_driver_handle_t, char *, size_t *);
    CHECK(xpu::ze::zeDriverGetExtensionFunctionAddress(driver,
            "zeIntelGetDriverVersionString", (void **)&pfnGetDriverVersionFn));

    size_t driver_version_len = 0;
    ZE_CHECK(pfnGetDriverVersionFn(driver, nullptr, &driver_version_len));
    driver_version_len++; // driver_version_len does not account for '\0'.

    if (!cache_blob) {
        // The last component corresponds to the number of
        // `sstream.append_array` calls which adds the size itself besides
        // the content of the array.
        id_size = device_name.size() + driver_version_len
                + sizeof(version->major) + sizeof(version->minor)
                + sizeof(version->patch) + std::strlen(version->hash)
                + 3 * sizeof(size_t);
        return status::success;
    }

    // Get driver version.
    std::string driver_version(driver_version_len, '\0');
    ZE_CHECK(pfnGetDriverVersionFn(driver,
            const_cast<char *>(driver_version.data()), &driver_version_len));

    serialization_stream_t sstream;

    // Serialize device name.
    sstream.append_array(device_name.size(), device_name.data());

    // Serialize driver version.
    sstream.append_array(driver_version.size(), driver_version.data());

    // Serialize oneDNN version.
    sstream.append(version->major);
    sstream.append(version->minor);
    sstream.append(version->patch);

    // Serialize oneDNN hash.
    sstream.append_array(std::strlen(version->hash), version->hash);

    VERROR_ENGINE(id_size == sstream.get_data().size(),
            status::invalid_arguments,
            "not enough buffer space for copying cache blob");

    std::memcpy(cache_blob, sstream.get_data().data(), id_size);

    return status::success;
}
