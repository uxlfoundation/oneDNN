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

#include "gpu/intel/l0/stream.hpp"
#include "gpu/intel/l0/engine.hpp"
#include "gpu/intel/l0/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

status_t stream_t::create_stream(impl::stream_t **stream,
        impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
    std::unique_ptr<intel::l0::stream_t> s(
            new intel::l0::stream_t(engine, stream_impl));
    if (!s) return status::out_of_memory;

    *stream = s.release();

    return status::success;
}

stream_t::stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
    : gpu::intel::stream_t(engine, stream_impl) {
    if (is_profiling_enabled()) {
        ze_device_properties_t device_properties = {};
        device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
        device_properties.pNext = nullptr;

        l0::zeDeviceGetProperties(utils::downcast<engine_t *>(engine)->device(),
                &device_properties);
        profiler_ = utils::make_unique<l0::stream_profiler_t>(this,
                1e9 / device_properties.timerResolution,
                ~(-1L << device_properties.kernelTimestampValidBits));
    }
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void stream_t::after_exec_hook() {
    l0_ctx().set_deps(event_t());

    if (is_profiling_enabled()) profiler_->stop_profiling();
}

status_t stream_t::reset_profiling() {
    if (!is_profiling_enabled()) return status::invalid_arguments;

    profiler_->reset();

    return status::success;
}

status_t stream_t::get_profiling_data(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    if (!is_profiling_enabled()) return status::invalid_arguments;

    return profiler_->get_info(data_kind, num_entries, data);
}

stream_impl_t::stream_impl_t(unsigned flags, ze_command_list_handle_t list)
    : impl::stream_impl_t(flags)
    , allocated_(false)
    , list_(list)
    , event_pool_(nullptr) {
    l0::zeCommandListGetContextHandle(list_, &context_);
    if (flags & stream_flags::out_of_order || is_profiling_enabled())
        create_event_pool();
}

stream_impl_t::stream_impl_t(
        unsigned flags, ze_context_handle_t context, ze_device_handle_t device)
    : impl::stream_impl_t(flags)
    , context_(context)
    , allocated_(true)
    , event_pool_(nullptr) {
    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.ordinal = 0;
    command_queue_desc.index = 0;
    command_queue_desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    l0::zeCommandListCreateImmediate(
            context_, device, &command_queue_desc, &list_);

    if (flags & stream_flags::out_of_order || is_profiling_enabled())
        create_event_pool();
}

void stream_impl_t::create_event_pool() {
    ze_event_pool_desc_t event_pool_desc = {};
    event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    event_pool_desc.pNext = nullptr;
    event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    if (is_profiling_enabled())
        event_pool_desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    event_pool_desc.count = 16384;

    ze_event_pool_handle_t event_pool;
    l0::zeEventPoolCreate(context_, &event_pool_desc, 0, nullptr, &event_pool);
    event_pool_ = std::make_shared<event_pool_wrapper_t>(event_pool);
}

stream_impl_t::~stream_impl_t() {
    wait();
    if (allocated_) l0::zeCommandListDestroy(list_);
}

xpu::context_t &stream_impl_t::ctx() {
    return l0_ctx();
}

const xpu::context_t &stream_impl_t::ctx() const {
    return l0_ctx();
}

context_t &stream_impl_t::l0_ctx() {
    const context_t &ctx = const_cast<const stream_impl_t *>(this)->l0_ctx();
    return *const_cast<context_t *>(&ctx);
}

const context_t &stream_impl_t::l0_ctx() const {
    static context_t empty_ctx;
    return ctx_.get(empty_ctx);
}

ze_event_handle_t stream_impl_t::get_output_event() const {
    auto &deps = event_t::from(ctx().get_deps()).events_;
    if (deps.size()) return deps[0];

    return nullptr;
}

std::shared_ptr<event_wrapper_t> stream_impl_t::create_event() {
    if (!event_pool_.get()) return std::make_shared<event_wrapper_t>(nullptr);

    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.pNext = nullptr;
    event_desc.index = static_cast<uint32_t>(events_.size());
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

    ze_event_handle_t event;
    l0::zeEventCreate(*(event_pool_.get()), &event_desc, &event);

    std::shared_ptr<event_wrapper_t> event_ptr
            = std::make_shared<event_wrapper_t>(event);
    events_.push_back(event_ptr);

    return event_ptr;
}

std::shared_ptr<event_pool_wrapper_t> stream_impl_t::get_event_pool() {
    return event_pool_;
}

ze_command_list_handle_t stream_impl_t::list() {
    return list_;
}

status_t stream_impl_t::wait() {
    CHECK(l0::zeCommandListHostSynchronize(list_, UINT64_MAX));

    return status::success;
}

status_t stream_impl_t::barrier() {
    CHECK(l0::zeCommandListAppendBarrier(list_, nullptr, 0, nullptr));

    return status::success;
}

status_t stream_impl_t::copy(const impl::memory_storage_t &src,
        const impl::memory_storage_t &dst, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (size == 0) return status::success;
    std::vector<ze_event_handle_t> l0_deps
            = utils::downcast<const event_t *>(&deps)->events_;

    ze_event_handle_t out_event = *(create_event().get());
    CHECK(l0::zeCommandListAppendMemoryCopy(list_, dst.data_handle(),
            src.data_handle(), size, out_event,
            static_cast<uint32_t>(l0_deps.size()),
            l0_deps.size() ? l0_deps.data() : nullptr));
    if (out_event)
        utils::downcast<event_t *>(&out_dep)->events_.push_back(out_event);

    return status::success;
}

status_t stream_impl_t::fill(const impl::memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    if (size == 0) return status::success;
    std::vector<ze_event_handle_t> l0_deps
            = utils::downcast<const event_t *>(&deps)->events_;

    ze_event_handle_t out_event = *(create_event().get());
    CHECK(l0::zeCommandListAppendMemoryFill(list_, dst.data_handle(), &pattern,
            sizeof(pattern), size, out_event,
            static_cast<uint32_t>(l0_deps.size()),
            l0_deps.size() ? l0_deps.data() : nullptr));
    if (out_event)
        utils::downcast<event_t *>(&out_dep)->events_.push_back(out_event);

    return status::success;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
