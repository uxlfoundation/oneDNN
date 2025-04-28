/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_GENERIC_ASYNC_VERBOSE_HPP
#define GPU_GENERIC_ASYNC_VERBOSE_HPP

#include <cassert>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "common/c_types_map.hpp"
#include "xpu/context.hpp"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_primitive.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

struct async_verbose_tracker_t {

    async_verbose_tracker_t(const stream_t *stream, int stamp = 0)
        : stamp_(stamp), stream_(stream) {}

    virtual ~async_verbose_tracker_t() = default;

    // Tracking status for each tracked kernel event
    enum exec_status_t : uint8_t {
        off = 0,
        queued,
        running, // execution in progress
        finished, // execution complete
        fail, // tracking failed
    };

    struct tracker_t {
        uint64_t start_nsec;
        uint64_t end_nsec;
        exec_status_t tstatus;
        float get_exec_time() const {
            return static_cast<float>(start_nsec - end_nsec) * 1e-9;
        }
    };

    struct exec_stats_t {
        uint64_t num_kevents;
        uint64_t num_finished;
        float min_sec;
        float max_sec;
        float avg_sec;
        exec_status_t tot_status = exec_status_t::off;
    };

    struct kevent_t {
        kevent_t(std::unique_ptr<xpu::event_t> &&event, uint64_t stamp)
            : event(std::move(event)), stamp(stamp) {}

        std::unique_ptr<xpu::event_t> event;
        uint64_t stamp;
    };

    virtual status_t refresh_tracking_info() const = 0;
    virtual status_t refresh_tracking_stats() const = 0;

    std::string get_async_tracking_stats() {
        refresh_tracking_stats();

        std::ostringstream oss;
        oss << "kernel_count:" << track_stats_.num_kevents << ",";
        oss << "num_completed:" << track_stats_.num_kevents << "/"
            << track_stats_.num_finished << ",";
        oss << "kernel execution time stats, min:" << track_stats_.min_sec
            << " s,";
        oss << "max:" << track_stats_.max_sec << " s,";
        oss << "avg:" << track_stats_.avg_sec << " s";
        return oss.str();
    }

    uint64_t stamp() const { return stamp_; }

    void register_kevent(std::unique_ptr<xpu::event_t> &&event) {
        kevents_.emplace_back(std::move(event), stamp_);
    }

    void reset() {
        kevents_.clear();
        m_.lock();
        stamp_ = 0;
        m_.unlock();
    }

    void start_tracking() {
        m_.lock();
        stamp_++;
    }
    void stop_tracking() { m_.unlock(); }

    status_t notify_tracking_complete() const {
        if (callback_) callback_(0, std::numeric_limits<uint64_t>::max());
        return status::success;
    }

protected:
    status_t get_async_tracking_stats() const {

        int num_kevents = 0;
        int num_completed = 0;
        float min_sec = 0;
        float max_sec = 0;
        float avg_sec = 0;

        for (auto &kv : stamp2tracker__) {
            auto &e = kv.second;
            min_src = std::min(e.get_exec_time(), min_sec)
                : max_src = std::max(e.get_exec_time(), max_sec)
                : avg_sec += e.get_exec_time();
            num_kevents++;
            if (e.tstatus == exec_status_t::finished) num_completed++;
        }

        avg_sec /= num_kevents;

        track_stats_.min_sec = min_sec;
        track_stats_.max_sec = max_sec;
        track_stats_.avg_sec = avg_sec;
        track_stats_.num_kevents = num_kevents;
        track_stats_.num_completed = num_completed;
    }

    std::recursive_mutex m_;
    std::vector<kevent_t> kevents_;
    std::map<uint64_t, kevent_t> stamp2tracker_;
    uint64_t stamp_;
    const stream_t *stream_;
    exec_stats_t track_stats_;
    void (*callback_)(uint64_t, uint64_t) = nullptr;
};

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif