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

#include "xpu/ze/engine_impl.hpp"
#include "xpu/ze/engine_id.hpp"
#include "xpu/ze/stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

engine_impl_t::engine_impl_t(engine_kind_t kind, ze_driver_handle_t driver,
        ze_device_handle_t device, ze_context_handle_t context, size_t index)
    : impl::engine_impl_t(kind, runtime_kind::ze, index)
    , driver_(driver)
    , device_(device)
    , context_(context, /* owner = */ !context) {}

status_t engine_impl_t::init() {
    cl_int err;
    std::vector<cl_device_id> ocl_devices;
    xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);

    ocl_device_ = nullptr;
    ocl_context_ = nullptr;
    xpu::device_uuid_t ze_dev_uuid = get_device_uuid(device_);
    for (const cl_device_id &d : ocl_devices) {
        xpu::device_uuid_t ocl_dev_uuid;
        xpu::ocl::get_device_uuid(ocl_dev_uuid, d);
        if (ze_dev_uuid == ocl_dev_uuid) {
            ocl_device_ = xpu::ocl::make_wrapper(d);
            ocl_context_ = xpu::ocl::make_wrapper(xpu::ocl::clCreateContext(
                    nullptr, 1, &ocl_device_.unwrap(), nullptr, nullptr, &err));
            OCL_CHECK(err);
        }
    }

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = nullptr;
    CHECK(ze::zeDeviceGetProperties(device_, &device_properties));
    name_ = device_properties.name;

    // Note: the method below is undocumented. Implementation is sneak peeked
    // from oneAPI DPC++ Compiler (git@github.com:intel/llvm.git) at
    // "devops/scripts/benchmarks/utils/detect_versions.cpp".
    //
    // Note: use this version of the code as it matches SYCL runtime reported
    // values. There's another version of parsing in
    // src/gpu/intel/ze/devince_info.cpp, but it reports different numbers.
    // TODO: figure it out.
    ze_result_t (*pfnGetDriverVersionFn)(ze_driver_handle_t, char *, size_t *);
    CHECK(ze::zeDriverGetExtensionFunctionAddress(driver_,
            "zeIntelGetDriverVersionString", (void **)&pfnGetDriverVersionFn));

    size_t driver_version_len = 0;
    ZE_CHECK(pfnGetDriverVersionFn(driver_, nullptr, &driver_version_len));
    driver_version_len++; // driver_version_len does not account for '\0'.
    std::string driver_version(driver_version_len, '\0');

    ZE_CHECK(pfnGetDriverVersionFn(driver_,
            const_cast<char *>(driver_version.data()), &driver_version_len));
    if (runtime_version_.set_from_string(driver_version.data())
            != status::success) {
        runtime_version_.major = 0;
        runtime_version_.minor = 0;
        runtime_version_.build = 0;
    }

    return status::success;
}

status_t engine_impl_t::create_stream_impl(
        impl::stream_impl_t **stream_impl, unsigned flags) const {
    auto *si = new xpu::ze::stream_impl_t(flags, context_, device_);
    if (!si) return status::out_of_memory;

    *stream_impl = si;

    return status::success;
}

status_t engine_impl_t::create_memory_storage(impl::memory_storage_t **storage,
        impl::engine_t *engine, unsigned flags, size_t size,
        void *handle) const {
    std::unique_ptr<memory_storage_t> _storage;
    _storage.reset(new memory_storage_t(engine, memory_storage_kind_t::device));
    if (!_storage) return status::out_of_memory;

    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) return status;

    *storage = _storage.release();

    return status::success;
}

engine_id_t engine_impl_t::engine_id() const {
    return engine_id_t(new xpu::ze::engine_id_impl_t(
            device(), context(), kind(), runtime_kind(), index()));
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
