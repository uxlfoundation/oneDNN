/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef XPU_STREAM_PROFILER_HPP
#define XPU_STREAM_PROFILER_HPP

#include <atomic>
#include <cassert>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "xpu/context.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

struct stream_profiler_t {
    stream_profiler_t(const stream_t *stream, int stamp = 0)
        : stamp_(stamp), stream_(stream) {}
    virtual ~stream_profiler_t() = default;

    struct entry_t {
        uint64_t min_nsec = std::numeric_limits<uint64_t>::max();
        uint64_t max_nsec = 0;
        double freq = 0;
        int kernel_count = 0;

        uint64_t get_nsec() const { return max_nsec - min_nsec; }
    };

    struct registered_event_t {
        registered_event_t(
                std::unique_ptr<xpu::event_t> &&event, uint64_t stamp)
            : event(std::move(event)), stamp(stamp) {}

        std::unique_ptr<xpu::event_t> event;
        uint64_t stamp;
    };

    virtual status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const
            = 0;

    uint64_t stamp() const { return stamp_; }

    void register_event(std::unique_ptr<xpu::event_t> &&event) {
        events_.emplace_back(std::move(event), stamp_);
    }

    void reset() {
        events_.clear();
        m_.lock();
        stamp_ = 0;
        m_.unlock();
    }

    // The contract is profiler interfaces are called only in between
    // `start_profiling` and `stop_profiling`, which provide a secure
    // multi-threaded access because of the lock. It allows to strip the lock
    // from all other calls, e.g., `stamp, or `register_event` (except `reset`)
    // to reduce the overhead for profiling.
    void start_profiling() {
        m_.lock();
        stamp_++;
    }
    void stop_profiling() { m_.unlock(); }

    void set_callback(void (*callback)(uint64_t, uint64_t)) {
        callback_ = callback;
    }

    status_t notify_profiling_complete() const {
        if (callback_) callback_(0, std::numeric_limits<uint64_t>::max());
        return status::success;
    }

protected:
    status_t get_info_impl(const std::map<uint64_t, entry_t> &stamp2entry,
            profiling_data_kind_t data_kind, uint64_t *data) const {
        int idx = 0;
        for (auto &kv : stamp2entry) {
            auto &e = kv.second;
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = e.get_nsec(); break;
                case profiling_data_kind::cycles: {
                    double freq = e.freq / e.kernel_count;
                    data[idx] = static_cast<uint64_t>(
                            freq * static_cast<double>(e.get_nsec()) / 1e9);
                    if (callback_) callback_(kv.first, e.get_nsec());
                    break;
                }
                default: assert(!"unexpected data kind");
            }
            idx++;
        }
        return status::success;
    }

    std::recursive_mutex m_;
    std::vector<registered_event_t> events_;
    uint64_t stamp_;
    const stream_t *stream_;
    void (*callback_)(uint64_t, uint64_t) = nullptr;
};

// The verbose profiler is intended to log primitive profiling info with
// execution times computed from device measured timing data without
// host-to-device synchronization costs and without blocking stream.wait()
// calls. This profiler operates independently from other stream profilers
// during the primitive execution.
struct verbose_profiler_t {
    verbose_profiler_t(const stream_t *stream, int stamp = 0)
        : profiler_paused_(false)
        , current_primitive_stamp_(stamp)
        , stream_(stream) {}

    virtual ~verbose_profiler_t() = default;

    uint64_t stamp() const { return current_primitive_stamp_; }

    struct prim_profile_data_t {
        double start_ms = 0.0;
        std::string pd_info;
        std::vector<std::shared_ptr<xpu::event_t>> prim_evts;
    };

    void reset() {
        current_primitive_stamp_ = 0;
        event_map_.clear();
    }

    // This allows force-pauses the profiler for unsupported scenarios -
    // pausing action is localized to each thread for multi-threaded
    // execution
    void pause_profiling() { profiler_paused_ = true; }
    void unpause_profiling() { profiler_paused_ = false; }
    bool is_profiler_paused() const { return profiler_paused_; }

    // The profiler operates via management of the event_map_ which tracks
    // the profiler events w.r.t primitives. Each primitive is assigned
    // a unique stamp which is updated during before_exec_hook() calls
    // before primitive execution.
    void update_primitive_stamp() {
        if (profiler_paused_) return;

        current_primitive_stamp_++;
        // Creates new empty entry which is later populated with profiling info
        event_map_[current_primitive_stamp_] = prim_profile_data_t {};
    }

    /* maps kernel events to the primitive stamp as they are enqueued */
    void register_primitive_event(const std::shared_ptr<xpu::event_t> &event) {
        if (profiler_paused_) return;

        auto prof_data = event_map_.find(current_primitive_stamp_);
        if (prof_data != event_map_.end()) {
            prof_data->second.prim_evts.push_back(event);
        }
    }

    /* populates profiling data based on the enqueued event map */
    status_t add_to_pending_primitive_list(
            double start_ms, const std::string &pd_info) {
        if (profiler_paused_) return status::success;

        auto prof_data = event_map_.find(current_primitive_stamp_);
        if (prof_data != event_map_.end()) {
            prof_data->second.start_ms = start_ms;
            prof_data->second.pd_info = pd_info;
        }
        return status::success;
    }

    // Completed primitive executions are periodically checked and logged
    // during after_exec_hook() calls and during stream destruction.
    // The profiler does not wait for pending events to complete
    // and instead prints them at the next concurrent after_exec_hook()
    // call.
    void check_for_completed_primitives() {
        if (profiler_paused_) return;

        for (auto it = event_map_.begin(); it != event_map_.end();) {
            uint64_t cstamp = it->first;
            auto &prof_data = it->second;
            auto &evts = prof_data.prim_evts;
            double duration_ms = 0.0;

            // handles primitives with no kernels - here,
            // device time is effectively zero
            if (evts.empty() && !prof_data.pd_info.empty()) {
                VPROF(prof_data.start_ms, primitive, exec, VERBOSE_profile,
                        prof_data.pd_info.c_str(), duration_ms);
                it = event_map_.erase(it);
                continue;
            }

            // avoids logging info for empty descriptors
            if (prof_data.pd_info.empty()) {
                it = event_map_.erase(it);
                continue;
            }

            // the polling check is paused at the first pending primitive
            // and resumed again at the next check. This ensures the
            // primitives are logged in the same order they were enqueued
            if (!is_event_complete(evts.back())) { break; }

            status_t status = get_aggregate_exec_time(cstamp, duration_ms);

            if (status == status::success) {
                VPROF(prof_data.start_ms, primitive, exec, VERBOSE_profile,
                        prof_data.pd_info.c_str(), duration_ms);
            }

            // the primitive info entry is removed after logging
            // to avoid blowing up the sizes of event_map_
            it = event_map_.erase(it);
        }
    }

    // This is invoked during profiler destruction to account
    // for any pending primitives that have not yet been logged.
    void wait_for_pending_primitives() {
        if (profiler_paused_) return;

        for (auto &map_entry : event_map_) {
            auto &prof_data = map_entry.second;
            auto &evts = prof_data.prim_evts;

            // For in-order queues, waiting on the last event is sufficient
            if (!evts.empty() && evts.back()) {
                wait_for_event_completion(evts.back());
            }
        }
        check_for_completed_primitives();

        if (!event_map_.empty())
            VWARN(primitive, exec,
                    "profiling error: failed to log pending primitive info");

        reset();
    }

    virtual status_t get_aggregate_exec_time(
            uint64_t stamp, double &duration_ms) const
            = 0;
    virtual bool is_event_complete(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;
    virtual void wait_for_event_completion(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;

protected:
    bool profiler_paused_;
    uint64_t current_primitive_stamp_;
    const stream_t *stream_;
    std::map<uint64_t, prim_profile_data_t> event_map_;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
