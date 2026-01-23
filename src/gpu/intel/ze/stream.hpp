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

#ifndef GPU_INTEL_ZE_STREAM_HPP
#define GPU_INTEL_ZE_STREAM_HPP

#include "xpu/ze/context.hpp"
#include "xpu/ze/stream_impl.hpp"

#include "gpu/intel/stream.hpp"
#include "gpu/intel/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

class stream_t : public intel::stream_t {
public:
    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl);

    xpu::ze::context_t &ze_ctx() { return impl()->ze_ctx(); }
    const xpu::ze::context_t &ze_ctx() const;
    xpu::context_t &ctx() override { return impl()->ze_ctx(); }
    const xpu::context_t &ctx() const override;

    ze_event_handle_t get_output_event() const;
    std::shared_ptr<xpu::ze::wrapper_t<ze_event_handle_t>> create_event() {
        return impl()->create_event();
    }
    std::shared_ptr<xpu::ze::wrapper_t<ze_event_pool_handle_t>>
    get_event_pool() {
        return impl()->get_event_pool();
    }

    ze_command_list_handle_t list() const;

    status_t wait() override { return impl()->wait(); }
    status_t barrier() override { return impl()->barrier(); }

    void before_exec_hook() override;
    void after_exec_hook() override;
    status_t reset_profiling() override;
    status_t get_profiling_data(profiling_data_kind_t data_kind,
            int *num_entries, uint64_t *data) const override;

    status_t copy(const impl::memory_storage_t &src,
            const impl::memory_storage_t &dst, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep) override {
        return impl()->copy(src, dst, size, deps, out_dep);
    }
    status_t fill(const impl::memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override {
        return impl()->fill(dst, pattern, size, deps, out_dep);
    }

private:
    xpu::ze::stream_impl_t *impl() const {
        return static_cast<xpu::ze::stream_impl_t *>(impl::stream_t::impl());
    }

    stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl);
    stream_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_t);
};

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_ZE_STREAM_HPP
