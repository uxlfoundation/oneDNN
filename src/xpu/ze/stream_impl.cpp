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

#include "xpu/ze/stream_impl.hpp"
#include "xpu/ze/memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

status_t stream_impl_t::init(
        ze_context_handle_t context, ze_device_handle_t device) {
    if (!list_) {
        assert(context && device);

        ze_command_queue_desc_t command_queue_desc {};
        command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        command_queue_desc.ordinal = 0;
        command_queue_desc.index = 0;
        command_queue_desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
        command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
        command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

        ZE_CHECK(ze::zeCommandListCreateImmediate(
                context, device, &command_queue_desc, &list_.unwrap()));
    } else {
        ZE_CHECK(ze::zeCommandListGetContextHandle(list_, &context));
    }

    // Pre-initializes the verbose profiler ahead of event pool creation,
    // assuming a GPU engine kind. This is required because the event pool
    // must be created with ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP when the
    // verbose profiler is enabled, but the engine kind is not yet available
    // at this stage of stream_impl_t initialization. The engine kind is
    // validated and the profiler is fully enabled during the subsequent
    // stream_t::init() call.
    init_verbose_profiler(engine_kind::gpu);

    if ((flags() & stream_flags::out_of_order) && is_profiling_enabled()) {
        VERROR(common, ze,
                "Level Zero kernel profiling is not supported with "
                "out-of-order queues");
        return status::invalid_arguments;
    } else if ((flags() & stream_flags::out_of_order) || is_profiling_enabled()
            || is_verbose_profiler_enabled()) {
        ze_event_pool_desc_t event_pool_desc {};
        event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        if (is_profiling_enabled() || is_verbose_profiler_enabled())
            event_pool_desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        // Note: 16K number is taken randomly as big enough to fit mode=F perf
        // validation or a single model profiling.
        event_pool_desc.count = 16 * 1024;

        ZE_CHECK(ze::zeEventPoolCreate(
                context, &event_pool_desc, 0, nullptr, &event_pool_.unwrap()));
    }

    return status::success;
}

const xpu::ze::context_t &stream_impl_t::ze_ctx() const {
    return ctx_.get();
}

xpu::ze::context_t &stream_impl_t::ze_ctx() {
    static xpu::ze::context_t empty_ctx {};
    return ctx_.get_or_set(empty_ctx);
}

xpu::context_t &stream_impl_t::ctx() {
    return ze_ctx();
}

const xpu::context_t &stream_impl_t::ctx() const {
    return ze_ctx();
}

ze_event_handle_t stream_impl_t::get_output_event() const {
    const auto &ze_deps = event_t::from(ctx().get_deps());
    if (ze_deps.size() > 0) return ze_deps[0];

    return nullptr;
}

ze_event_handle_t stream_impl_t::create_event() {
    if (!event_pool_) return xpu::ze::wrapper_t<ze_event_handle_t>();

    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.pNext = nullptr;
    event_desc.index = static_cast<uint32_t>(events_.size());
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

    ze_event_handle_t event;
    auto ze_status = ze::zeEventCreate(event_pool_, &event_desc, &event);
    if (ze_status != ZE_RESULT_SUCCESS) return nullptr;

    events_.emplace_back(event);

    return event;
}

status_t stream_impl_t::wait() {
    ZE_CHECK(ze::zeCommandListHostSynchronize(list_, UINT64_MAX));

    return status::success;
}

status_t stream_impl_t::barrier() {
    ZE_CHECK(ze::zeCommandListAppendBarrier(list_, nullptr, 0, nullptr));

    return status::success;
}

status_t stream_impl_t::init_verbose_profiler(engine_kind_t eng) {
    use_verbose_profiler_ = false;
    if (!get_verbose(verbose_t::exec_profile)) return status::success;
    if (eng != engine_kind::gpu) return status::success;
    // verbose profiling support is only for in-order queues
    if (flags() & stream_flags::out_of_order) return status::success;
    use_verbose_profiler_ = true;
    return status::success;
}

status_t stream_impl_t::init_flags(
        unsigned *flags, ze_command_list_handle_t list, bool profiling) {
    *flags = 0;
    // Note: determining if the passed list is in-order/out-of-order is
    // impossible with ze API.
    // TODO: add whenever such API appears.
    *flags |= stream_flags::in_order;

    // Note: ze_command_list_handle_t doesn't have a property of being profiled,
    // it's indicated by the user directly.
    if (profiling) {
#ifdef DNNL_EXPERIMENTAL_PROFILING
        *flags |= stream_flags::profiling;
#endif
    }

    return status::success;
}

status_t stream_impl_t::copy(const impl::memory_storage_t &src,
        const impl::memory_storage_t &dst, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (size == 0) return status::success;

    const auto &ze_deps = event_t::from(deps);
    ze_event_handle_t out_event = create_event();
    CHECK(append_memory_copy(list(), list_mutex(), dst.data_handle(),
            src.data_handle(), size, out_event, ze_deps.size(),
            ze_deps.data()));
    if (out_event) event_t::from(out_dep).append(out_event);

    return status::success;
}

status_t stream_impl_t::fill(const impl::memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (size == 0) return status::success;

    const auto &ze_deps = event_t::from(deps);
    ze_event_handle_t out_event = create_event();
    CHECK(append_memory_fill(list(), list_mutex(), dst.data_handle(), &pattern,
            sizeof(pattern), size, out_event, ze_deps.size(), ze_deps.data()));
    if (out_event) event_t::from(out_dep).append(out_event);

    return status::success;
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
