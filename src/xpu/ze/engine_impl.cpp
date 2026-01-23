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
    , context_(context, /* owner = */ !context) {
    cl_int err;
    std::vector<cl_device_id> ocl_devices;
    xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);

    ocl_device_ = nullptr;
    ocl_context_ = nullptr;
    xpu::device_uuid_t ze_dev_uuid = get_device_uuid(device);
    for (const cl_device_id &d : ocl_devices) {
        xpu::device_uuid_t ocl_dev_uuid;
        xpu::ocl::get_device_uuid(ocl_dev_uuid, d);
        if (ze_dev_uuid == ocl_dev_uuid) {
            ocl_device_ = xpu::ocl::make_wrapper(d);
            ocl_context_ = xpu::ocl::make_wrapper(xpu::ocl::clCreateContext(
                    nullptr, 1, &ocl_device_.unwrap(), nullptr, nullptr, &err));
        }
    }
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
