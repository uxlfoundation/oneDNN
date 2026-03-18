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

#ifndef XPU_ZE_STREAM_IMPL_HPP
#define XPU_ZE_STREAM_IMPL_HPP

#include "common/stream_impl.hpp"
#include "common/thread_local_storage.hpp"

#include "xpu/ze/context.hpp"
#include "xpu/ze/utils.hpp"

#include <list>

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags, ze_command_list_handle_t list = nullptr)
        : impl::stream_impl_t(flags), list_(list, /* owner = */ !list) {}

    ~stream_impl_t() override = default;

    status_t init(ze_context_handle_t context = nullptr,
            ze_device_handle_t device = nullptr);

    status_t wait();

    status_t copy(const impl::memory_storage_t &src,
            const impl::memory_storage_t &dst, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep);

    status_t fill(const impl::memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps, xpu::event_t &out_dep);

    status_t barrier();

    const xpu::ze::context_t &ze_ctx() const;
    xpu::ze::context_t &ze_ctx();
    xpu::context_t &ctx();
    const xpu::context_t &ctx() const;

    ze_event_handle_t get_output_event() const;

    ze_event_handle_t create_event();

    ze_command_list_handle_t list() const { return list_; }

    static status_t init_flags(
            unsigned *flags, ze_command_list_handle_t list, bool profiling);

private:
    xpu::ze::wrapper_t<ze_command_list_handle_t> list_;
    // TODO: `event_pool_` seems to belong to `ctx_` as events can't be created
    // in multithreaded scenario and having a thread_local event pool should
    // address it.
    xpu::ze::wrapper_t<ze_event_pool_handle_t> event_pool_;
    // TODO: additionally, the management of event should be probably done in
    // `event_t` struct as well and not stored in the stream as a list. However,
    // it seems there's a challenge to keep track of event lifetime if it's
    // needed in two different places, e.g. out-of-order queue and profiling
    // (if this mode will ever be supported).
    std::list<xpu::ze::wrapper_t<ze_event_handle_t>> events_;

    mutable utils::thread_local_storage_t<context_t> ctx_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t);
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_STREAM_IMPL_HPP
