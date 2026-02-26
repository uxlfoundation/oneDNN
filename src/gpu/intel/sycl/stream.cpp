/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <map>
#include <memory>
#include <CL/cl.h>

#include "common/verbose.hpp"

#include "xpu/sycl/stream_profiler.hpp"

#include "gpu/intel/sycl/stream.hpp"

#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    if (is_profiling_enabled())
        profiler_ = utils::make_unique<xpu::sycl::stream_profiler_t>(this);

    const auto &sycl_engine_impl
            = *utils::downcast<const xpu::sycl::engine_impl_t *>(
                    engine()->impl());
    auto &sycl_ctx = sycl_engine_impl.context();
    auto &sycl_dev = sycl_engine_impl.device();

    // If queue_ is not set then construct it
    if (!impl()->queue()) {
        ::sycl::property_list props;
        if (is_profiling_enabled() && sycl_dev.is_gpu()) {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::property_list {::sycl::property::queue::
                                                     in_order {},
                              ::sycl::property::queue::enable_profiling {}}
                    : ::sycl::property_list {
                              ::sycl::property::queue::enable_profiling {}};
        } else {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::property_list {::sycl::property::queue::
                                      in_order {}}
                    : ::sycl::property_list {};
        }
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, props));
    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue().get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || xpu::sycl::is_host(sycl_dev)));
        if (!args_ok) return status::invalid_arguments;
    }

    if (is_profiling_enabled() && sycl_dev.is_gpu() && !queue().is_in_order()) {
        VERROR(common, dpcpp,
                "DPC++ kernel profiling is not supported with out-of-order "
                "queues");
        return status::invalid_arguments;
    }

    return status::success;
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void stream_t::after_exec_hook() {
    sycl_ctx().set_deps(xpu::sycl::event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
}

status_t stream_t::run_verbose_profiler(
        std::string &pd_info, double start_ms) const {

    // utilize the verbose profiler only for profile_exec verbose levels.
    if (!is_verbose_profiler_enabled()) return status::invalid_arguments;

    // Captured output event acts as the anchor to track primitive execution
    ::sycl::event out_evt = get_output_event();

    // as the stamp count increments each time the profiler is unpaused, it
    // also tracks the primitive exeuctions.
    uint64_t curr_stamp = profiler_->stamp();

    if (!curr_stamp) {
        VWARN(primitive, exec,
                "%s, profiling error: failed to record primitive events in "
                "context",
                pd_info.c_str());
        VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(), 0.f);
        return status::success;
    }

    xpu::stream_profiler_t *prof_ptr = profiler_.get();

    struct payload_t {
        xpu::stream_profiler_t *prof;
        std::string info_str;
        double start;
        uint64_t stamp;
    };

    std::unique_ptr<payload_t> payload(new payload_t());
    payload->prof = prof_ptr;
    payload->info_str = pd_info;
    payload->start = start_ms;
    payload->stamp = curr_stamp;

    // The prompt ensures the verbose headers are printed if they aren't
    // already. Printing the headers asynchronously during the callback can
    // result in access failures when printing engine-specific info.
    verbose_printf(verbose_t::exec_profile, "\r");

    try {
        ::sycl::queue q = queue();
        payload_t *user = payload.get();

        q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(out_evt);
            cgh.host_task([user]() {
                std::unique_ptr<payload_t> hold(user);

                double duration_ms = 0.0;

                if (hold->prof) {
                    // aggregate execution times are calculated from the start and end times
                    // of the first and last queued events for the primitive respectively
                    hold->prof->get_aggregate_exec_timing(
                            hold->stamp, duration_ms);
                } else {
                    VWARN(primitive, exec,
                            "%s, profiling error: profiler absent",
                            hold->info_str.c_str());
                }

                VPROF(hold->start, primitive, exec, VERBOSE_profile,
                        hold->info_str.c_str(), duration_ms);
            });
        });

        (void)payload.release();

    } catch (...) {
        VWARN(primitive, exec,
                "%s, profiling error: failed to submit host_task for async "
                "verbose logging",
                pd_info.c_str());
        VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(), 0.f);
        return status::success;
    }

    return status::success;
}

// The following code needs sycl::queue::ext_oneapi_get_graph(), but it may
//  not be defined. Some SFINAE is needed to avoid compile errors in this case.
namespace syclex = ::sycl::ext::oneapi::experimental;
template <typename Q>
static auto get_graph_internal(
        const Q &q, bool &success, int) -> decltype(q.ext_oneapi_get_graph()) {
    success = true;
    return q.ext_oneapi_get_graph();
}

template <typename Q>
static syclex::command_graph<syclex::graph_state::modifiable>
get_graph_internal(const Q &q, bool &success, long) {
    success = false;
    return syclex::command_graph<syclex::graph_state::modifiable>(
            q.get_context(), q.get_device());
}

static syclex::command_graph<syclex::graph_state::modifiable> get_graph(
        const ::sycl::queue *q, bool &success) {
    return get_graph_internal(*q, success, 0);
}

bool stream_t::recording() const {
    return impl()->queue()->ext_oneapi_get_state()
            == syclex::queue_state::recording;
}

stream_t::weak_graph_t stream_t::get_current_graph_weak() const {
    bool success;
    stream_t::weak_graph_t result = get_graph(impl()->queue(), success);
    if (!success) result.reset();
    return result;
}

status_t stream_t::enter_immediate_mode() {
    std::lock_guard<std::mutex> lock(immediate_mode_mutex_);
    if (!immediate_mode_level_++) pause_recording();
    return status::success;
}

status_t stream_t::exit_immediate_mode() {
    std::lock_guard<std::mutex> lock(immediate_mode_mutex_);
    if (immediate_mode_level_ > 0) {
        if (!--immediate_mode_level_) resume_recording();
    } else {
        assert(!"exit_immediate_mode called without enter");
        return status::runtime_error;
    }
    return status::success;
}

status_t stream_t::pause_recording() {
    using graph_t = syclex::command_graph<syclex::graph_state::modifiable>;
    if (recording()) {
        bool success;
        assert(!paused_graph_);
        paused_graph_.reset(new graph_t(get_graph(impl()->queue(), success)));
        if (!success) return status::runtime_error;
        paused_graph_->end_recording();
        auto &cur_dep = xpu::sycl::event_t::from(ctx().get_deps());
        paused_dep_ = xpu::sycl::event_t {};
        std::swap(paused_dep_, cur_dep);
    }
    return status::success;
}

status_t stream_t::resume_recording() {
    if (paused_graph_) {
        paused_graph_->begin_recording(*impl()->queue());
        paused_graph_.reset();
        auto &cur_dep = xpu::sycl::event_t::from(ctx().get_deps());
        std::swap(paused_dep_, cur_dep);
        paused_dep_ = xpu::sycl::event_t {};
    }
    return status::success;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
