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

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void stream_t::after_exec_hook() {
    impl()->ctx().set_deps(event_t());

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
    : impl::stream_impl_t(flags), allocated_(false), list_(list) {};

stream_impl_t::stream_impl_t(
        unsigned flags, ze_context_handle_t context, ze_device_handle_t device)
    : impl::stream_impl_t(flags), allocated_(true) {
    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.ordinal = 0;
    command_queue_desc.index = 0;
    command_queue_desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    func_zeCommandListCreateImmediate(
            context, device, &command_queue_desc, &list_);
}

stream_impl_t::~stream_impl_t() {
    if (allocated_) {
        func_zeCommandListDestroy(list_);
    }
}

ze_command_list_handle_t stream_impl_t::list() {
    return list_;
}

context_t &stream_impl_t::ctx() {
    const context_t &ctx = const_cast<const stream_impl_t *>(this)->ctx();

    return *const_cast<context_t *>(&ctx);
}

const context_t &stream_impl_t::ctx() const {
    static context_t empty_ctx {};

    return ctx_.get(empty_ctx);
}

event_wrapper_t stream_impl_t::get_output_event() const {
    auto &deps = event_t::from(ctx().get_deps());

    return deps[0];
}

status_t stream_impl_t::wait() {
    CHECK(func_zeCommandListHostSynchronize(list_, UINT64_MAX));

    return status::success;
}

status_t stream_impl_t::barrier() {
    CHECK(func_zeCommandListAppendBarrier(list_, nullptr, 0, nullptr));

    return status::success;
}

status_t stream_impl_t::copy(const impl::memory_storage_t &src,
        const impl::memory_storage_t &dst, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep) {
    std::vector<ze_event_handle_t> l0_deps;
    utils::downcast<const gpu::intel::l0::event_t *>(&deps)->get_l0_events(
            l0_deps);

    std::vector<ze_event_handle_t> l0_out_dep;
    utils::downcast<const gpu::intel::l0::event_t *>(&out_dep)->get_l0_events(
            l0_out_dep);

    CHECK(func_zeCommandListAppendMemoryCopy(list_, dst.data_handle(),
            src.data_handle(), size, l0_out_dep[0],
            static_cast<uint32_t>(l0_deps.size()), l0_deps.data()));

    return status::success;
}

status_t stream_impl_t::fill(const impl::memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    std::vector<ze_event_handle_t> l0_deps;
    utils::downcast<const gpu::intel::l0::event_t *>(&deps)->get_l0_events(
            l0_deps);

    std::vector<ze_event_handle_t> l0_out_dep;
    utils::downcast<const gpu::intel::l0::event_t *>(&out_dep)->get_l0_events(
            l0_out_dep);

    CHECK(func_zeCommandListAppendMemoryFill(list_, dst.data_handle(), &pattern,
            sizeof(pattern), size, l0_out_dep[0],
            static_cast<uint32_t>(l0_deps.size()), l0_deps.data()));

    return status::success;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
