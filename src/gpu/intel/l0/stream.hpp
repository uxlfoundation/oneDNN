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

#ifndef GPU_INTEL_L0_STREAM_HPP
#define GPU_INTEL_L0_STREAM_HPP

#include "common/thread_local_storage.hpp"
#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/l0/context.hpp"
#include "gpu/intel/l0/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags, ze_command_list_handle_t list);
    stream_impl_t(unsigned flags, ze_context_handle_t context,
            ze_device_handle_t device);
    ~stream_impl_t();

    ze_command_list_handle_t list();
    context_t &ctx();
    const context_t &ctx() const;
    ze_event_handle_t get_output_event() const;

    status_t wait();
    status_t barrier();

    status_t copy(const impl::memory_storage_t &src,
            const impl::memory_storage_t &dst, size_t size,
            const xpu::event_t &dep, xpu::event_t &out_dep);
    status_t fill(const impl::memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps, xpu::event_t &out_dep);

private:
    bool allocated_;
    ze_command_list_handle_t list_;
    mutable utils::thread_local_storage_t<context_t> ctx_;
};

class stream_t : public compute::compute_stream_t {
public:
    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl);

    stream_impl_t *impl() const {
        return static_cast<stream_impl_t *>(impl::stream_t::impl_.get());
    }
    const ze_command_list_handle_t list() const { return impl()->list(); };
    xpu::context_t &ctx() override { return impl()->ctx(); };
    const xpu::context_t &ctx() const override { return impl()->ctx(); };
    ze_event_handle_t get_output_event() const {
        return impl()->get_output_event();
    }
    status_t wait() override { return impl()->wait(); }
    status_t barrier() override { return impl()->barrier(); }

    void before_exec_hook() override;
    void after_exec_hook() override;
    status_t reset_profiling() override;
    status_t get_profiling_data(profiling_data_kind_t data_kind,
            int *num_entries, uint64_t *data) const override;

    status_t copy(const impl::memory_storage_t &src,
            const impl::memory_storage_t &dst, size_t size,
            const xpu::event_t &dep, xpu::event_t &out_dep) override {
        return impl()->copy(src, dst, size, dep, out_dep);
    }
    status_t fill(const impl::memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override {
        return impl()->fill(dst, pattern, size, deps, out_dep);
    }
private:
    stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
        : gpu::intel::compute::compute_stream_t(engine, stream_impl) {}
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_STREAM_HPP
