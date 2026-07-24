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

#include "gpu/intel/ze/stream.hpp"
#include "gpu/intel/ze/engine.hpp"

#include "xpu/ze/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t stream_t::create_stream(impl::stream_t **stream,
        impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
    std::unique_ptr<intel::ze::stream_t> s(new stream_t(engine, stream_impl));
    if (!s) return status::out_of_memory;

    status_t status = s->init();
    if (status != status::success) {
        // Stream owns stream_impl only if it's created successfully
        // (including initialization).
        s->impl_.release();
        return status;
    }

    *stream = s.release();

    return status::success;
}

status_t stream_t::init() {

    // Finalizes verbose profiler initialization by validating the engine kind
    // and enabling the profiler for the stream. The verbose profiler is
    // pre-initialized during stream_impl_t::init() with an assumed GPU engine
    // kind to ensure the event pool is created with the correct
    // ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag. This call confirms the engine
    // kind and completes the initialization.
    // Note: The verbose profiler state is fixed at stream initialization and
    // does not respond to runtime changes made via set_dnnl_verbose().
    // TODO: allow runtime control of the asynchronous verbose mode via
    // set_dnnl_verbose()
    CHECK(impl()->init_verbose_profiler(engine()->kind()));

    if (is_profiling_enabled() || is_verbose_profiler_enabled()) {
        std::pair<double, uint64_t> device_props = get_device_properties(
                utils::downcast<engine_t *>(engine())->device());
        double timer_frequency = device_props.first;
        uint64_t max_timestamp_value = device_props.second;

        if (is_profiling_enabled()) {
            profiler_ = utils::make_unique<xpu::ze::stream_profiler_t>(
                    this, timer_frequency, max_timestamp_value);
        }

        if (is_verbose_profiler_enabled()) {
            verbose_profiler_.set(
                    utils::make_unique<xpu::ze::verbose_profiler_t>(
                            this, timer_frequency, max_timestamp_value));
        }
    }

    return status::success;
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
    if (is_verbose_profiler_enabled()) {
        std::pair<double, uint64_t> device_props = get_device_properties(
                utils::downcast<engine_t *>(engine())->device());
        double timer_frequency = device_props.first;
        uint64_t max_timestamp_value = device_props.second;
        auto &verbose_profiler = verbose_profiler_.get_or_set(
                utils::make_unique<xpu::ze::verbose_profiler_t>(
                        this, timer_frequency, max_timestamp_value));
        verbose_profiler->update_event_list();
    }
}

void stream_t::after_exec_hook() {
    ze_ctx().set_deps(xpu::ze::event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
    if (auto *vp = verbose_profiler()) { vp->check_for_completed_primitives(); }
}

status_t stream_t::run_verbose_profiler(
        const std::string &pd_info, double start_ms) {
    if (!is_verbose_profiler_enabled()) {
        VERROR(primitive, exec,
                "running verbose profiler while it is not enabled");
        return status::success;
    }

    auto *vp = verbose_profiler();
    vp->add_to_pending_primitive_list(start_ms, pd_info);
    return status::success;
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

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
