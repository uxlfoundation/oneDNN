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

        CHECK(ze::zeCommandListCreateImmediate(
                context, device, &command_queue_desc, &list_.unwrap()));
    } else {
        CHECK(ze::zeCommandListGetContextHandle(list_, &context));
    }

    if ((flags() & stream_flags::out_of_order) && is_profiling_enabled()) {
        VERROR(common, ze,
                "Level Zero kernel profiling is not supported with "
                "out-of-order queues");
        return status::invalid_arguments;
    } else if ((flags() & stream_flags::out_of_order)
            || is_profiling_enabled()) {
        ze_event_pool_desc_t event_pool_desc {};
        event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        if (is_profiling_enabled())
            event_pool_desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        // Note: 16K number is taken randomly as big enough to fit mode=F perf
        // validation or a single model profiling.
        event_pool_desc.count = 16 * 1024;

        CHECK(ze::zeEventPoolCreate(
                context, &event_pool_desc, 0, nullptr, &event_pool_.unwrap()));
    }

    return status::success;
}

const xpu::ze::context_t &stream_impl_t::ze_ctx() const {
    static xpu::ze::context_t empty_ctx {};
    return ctx_.get(empty_ctx);
}

xpu::ze::context_t &stream_impl_t::ze_ctx() {
    const xpu::ze::context_t &ctx
            = const_cast<const stream_impl_t *>(this)->ze_ctx();
    return *const_cast<xpu::ze::context_t *>(&ctx);
}

xpu::context_t &stream_impl_t::ctx() {
    return ze_ctx();
}

const xpu::context_t &stream_impl_t::ctx() const {
    return ze_ctx();
}

ze_event_handle_t stream_impl_t::get_output_event() const {
    auto &ze_deps = event_t::from(ctx().get_deps()).ze_events_;
    if (!ze_deps.empty()) return ze_deps[0];

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
    auto st = ze::zeEventCreate(event_pool_, &event_desc, &event);
    if (st != status::success) return nullptr;

    events_.emplace_back(event);

    return event;
}

status_t stream_impl_t::wait() {
    CHECK(ze::zeCommandListHostSynchronize(list_, UINT64_MAX));

    return status::success;
}

status_t stream_impl_t::barrier() {
    CHECK(ze::zeCommandListAppendBarrier(list_, nullptr, 0, nullptr));

    return status::success;
}

status_t stream_impl_t::init_flags(
        unsigned *flags, ze_command_list_handle_t list, bool profiling) {
    *flags = 0;
    // Note: determininig if the passed list is in-order/out-of-order is
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

    std::vector<ze_event_handle_t> ze_deps
            = utils::downcast<const event_t *>(&deps)->ze_events_;

    ze_event_handle_t out_event = create_event();
    CHECK(ze::zeCommandListAppendMemoryCopy(list_, dst.data_handle(),
            src.data_handle(), size, out_event,
            static_cast<uint32_t>(ze_deps.size()), ze_deps.data()));
    if (out_event)
        utils::downcast<event_t *>(&out_dep)->ze_events_.push_back(out_event);

    return status::success;
}

status_t stream_impl_t::fill(const impl::memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (size == 0) return status::success;

    std::vector<ze_event_handle_t> ze_deps
            = utils::downcast<const event_t *>(&deps)->ze_events_;

    ze_event_handle_t out_event = create_event();
    CHECK(ze::zeCommandListAppendMemoryFill(list_, dst.data_handle(), &pattern,
            sizeof(pattern), size, out_event,
            static_cast<uint32_t>(ze_deps.size()), ze_deps.data()));
    if (out_event)
        utils::downcast<event_t *>(&out_dep)->ze_events_.push_back(out_event);

    return status::success;
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
