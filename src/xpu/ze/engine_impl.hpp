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

#ifndef XPU_ZE_ENGINE_IMPL_HPP
#define XPU_ZE_ENGINE_IMPL_HPP

#include "common/engine_impl.hpp"

#include "xpu/ocl/utils.hpp"
#include "xpu/ze/memory_storage.hpp"
#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t() = delete;
    engine_impl_t(engine_kind_t kind, ze_driver_handle_t driver,
            ze_device_handle_t device, ze_context_handle_t context,
            size_t index);

    ~engine_impl_t() override = default;

    status_t init() override;

    ze_driver_handle_t driver() const { return driver_; }
    ze_device_handle_t device() const { return device_; }
    ze_context_handle_t context() const { return context_; }

    xpu::ocl::wrapper_t<cl_device_id> ocl_device() const { return ocl_device_; }
    xpu::ocl::wrapper_t<cl_context> ocl_context() const { return ocl_context_; }

    engine_id_t engine_id() const override;

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override;

    status_t create_memory_storage(impl::memory_storage_t **storage,
            impl::engine_t *engine, unsigned flags, size_t size,
            void *handle) const override;

    const std::string &name() const { return name_; }
    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }

    int get_buffer_alignment() const override { return 128; }

private:
    ze_driver_handle_t driver_;
    ze_device_handle_t device_;
    // Note: wrapper only prevents from deleting external context, when it was
    // passed by the user. It doesn't help to deal with engine copies.
    xpu::ze::wrapper_t<ze_context_handle_t> context_;

    xpu::ocl::wrapper_t<cl_device_id> ocl_device_;
    xpu::ocl::wrapper_t<cl_context> ocl_context_;

    std::string name_;
    runtime_version_t runtime_version_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_impl_t);
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_ENGINE_IMPL_HPP
