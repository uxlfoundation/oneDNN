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

    // Enables profiling capabilities to allow the verbose mode to print
    // profiling info using device measured times.
    // The verbose profiler state is fixed at stream initialization and does not
    // respond to runtime changes made via set_dnnl_verbose().
    // TODO: allow runtime control of the asynchronous verbose mode via
    // set_dnnl_verbose()
    CHECK(init_verbose_profiler());

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
        auto &profiler = verbose_profiler_.get(
                utils::make_unique<xpu::ze::verbose_profiler_t>(
                        this, timer_frequency, max_timestamp_value));
        profiler->update_primitive_stamp();
    }
}

void stream_t::after_exec_hook() {
    ze_ctx().set_deps(xpu::ze::event_t());

    if (is_profiling_enabled()) profiler_->stop_profiling();
    if (is_verbose_profiler_enabled() && verbose_profiler_.is_set())
        verbose_profiler().check_for_completed_primitives();
}

status_t stream_t::run_verbose_profiler(
        const std::string &pd_info, double start_ms) const {
    if (!is_verbose_profiler_enabled()) return status::invalid_arguments;
    CHECK(verbose_profiler_.get()->add_to_pending_primitive_list(
            start_ms, pd_info));
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
