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

#ifndef CPU_CPU_STREAM_HPP
#define CPU_CPU_STREAM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_stream_t : public stream_t {
    cpu_stream_t(engine_t *engine, impl::stream_impl_t *stream_impl)
        : stream_t(engine, stream_impl) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        if (is_verbose_profiler_enabled()) {
            dnnl::threadpool_interop::threadpool_iface *tp;
            if (this->get_threadpool(&tp) == status::success && tp)
                tp->set_verbose_profiling(
                        get_verbose(verbose_t::exec_profile));
        }
#endif
    }

    ~cpu_stream_t() override {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        // Flush any pending verbose profiling entries.
        for (auto &p : pending_primitives_) {
            if (p.event) p.event->wait();
            double duration_ms = p.event ? p.event->exec_time_ms() : 0.0;
            VPROF(p.start_ms, primitive, exec, VERBOSE_profile,
                    p.pd_info.c_str(), duration_ms);
        }
        pending_primitives_.clear();
#endif
    }

    dnnl::impl::status_t wait() override {
        // CPU execution is synchronous so return immediately
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc == status::success && tp) {
            if (tp->get_flags()
                    & threadpool_interop::threadpool_iface::ASYNCHRONOUS)
                tp->wait();
        }
#endif
        return dnnl::impl::status::success;
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    cpu_stream_t(engine_t *engine,
            dnnl::threadpool_interop::threadpool_iface *threadpool)
        : stream_t(engine, new impl::stream_impl_t(threadpool)) {
        if (is_verbose_profiler_enabled())
            threadpool->set_verbose_profiling(get_verbose(verbose_t::exec_profile));
    }

    void before_exec_hook() override {
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc == status::success) threadpool_utils::activate_threadpool(tp);
    }

    void after_exec_hook() override {
        threadpool_utils::deactivate_threadpool();
        if (!is_verbose_profiler_enabled()) return;
        // Poll pending primitives in order; stop at first incomplete.
        size_t completed = 0;
        for (auto &p : pending_primitives_) {
            if (!p.event || !p.event->is_complete()) break;
            double duration_ms = p.event->exec_time_ms();
            VPROF(p.start_ms, primitive, exec, VERBOSE_profile,
                    p.pd_info.c_str(), duration_ms);
            completed++;
        }
        if (completed > 0)
            pending_primitives_.erase(pending_primitives_.begin(),
                    pending_primitives_.begin() + completed);
    }

    status_t run_verbose_profiler(
            const std::string &pd_info, double start_ms) override {
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc != status::success || !tp) return status::success;

        auto event = tp->get_completion_event();
        pending_primitives_.push_back({start_ms, pd_info, std::move(event)});
        return status::success;
    }

private:
    struct pending_primitive_t {
        double start_ms;
        std::string pd_info;
        std::shared_ptr<threadpool_interop::threadpool_event_t> event;
    };
    std::vector<pending_primitive_t> pending_primitives_;
#endif
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
