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

#ifndef XPU_ZE_STREAM_PROFILER_HPP
#define XPU_ZE_STREAM_PROFILER_HPP

#include "xpu/stream_profiler.hpp"
#include "xpu/ze/context.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

// Handles timestamp counter wraparound.
inline uint64_t duration_cycles(
        uint64_t start, uint64_t end, uint64_t max_val) {
    return (end >= start) ? (end - start) : ((max_val - start) + end + 1);
}

class stream_profiler_t : public xpu::stream_profiler_t {
public:
    class entry_t {
    public:
        entry_t() = default;

        entry_t(const ze_kernel_timestamp_result_t &kernel_timestamp_result,
                uint64_t max_timestamp_value, double timestamp_freq)
            : start_(kernel_timestamp_result.context.kernelStart)
            , end_(kernel_timestamp_result.context.kernelEnd)
            , max_timestamp_value_(max_timestamp_value)
            , freq_(timestamp_freq) {}

        uint64_t start() const { return start_; }
        uint64_t end() const { return end_; }

        uint64_t get_cycles() const {
            return duration_cycles(start_, end_, max_timestamp_value_);
        }

        uint64_t get_nsec() const {
            return static_cast<uint64_t>(freq_ * get_cycles());
        }

    private:
        uint64_t start_ = 0;
        uint64_t end_ = 0;
        uint64_t max_timestamp_value_ = 0;
        double freq_ = 0;
    };

    stream_profiler_t(const impl::stream_t *stream, double timestamp_freq,
            uint64_t max_timestamp_value)
        : xpu::stream_profiler_t(stream)
        , timestamp_freq_(timestamp_freq)
        , max_timestamp_value_(max_timestamp_value) {}

    // L0 does not reuse get_info_generic because L0 uses raw cycles that wrap
    // at max_timestamp_value_, needing cycles-to-nsec conversion done here.
    status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const override {
        if (!num_entries) return status::invalid_arguments;

        bool is_per_kernel
                = (data_kind == profiling_data_kind::time_per_kernel);
        if (!data) {
            if (is_per_kernel) {
                *num_entries = (int)events_.size();
                return status::success;
            }
            std::unordered_set<uint64_t> seen;
            for (auto &ev : events_)
                seen.insert(ev.stamp);
            *num_entries = (int)seen.size();
            return status::success;
        }

        if (is_per_kernel) {
            int idx = 0;
            for (auto &ev : events_) {
                entry_t entry;
                CHECK(query_entry(*ev.event, entry));
                data[idx++] = entry.get_nsec();
            }
            return status::success;
        }

        // A primitive may run multiple kernels sharing one stamp.
        std::map<uint64_t, span_t> stamp2span;
        for (auto &ev : events_) {
            entry_t entry;
            CHECK(query_entry(*ev.event, entry));
            auto &s = stamp2span[ev.stamp];
            s.start = std::min(s.start, entry.start());
            s.end = std::max(s.end, entry.end());
        }

        int idx = 0;
        for (auto &kv : stamp2span) {
            uint64_t cycles = get_duration(kv.second.start, kv.second.end);
            uint64_t nsec = static_cast<uint64_t>(timestamp_freq_ * cycles);
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = nsec; break;
                case profiling_data_kind::cycles:
                    data[idx] = cycles;
                    if (callback_) callback_(kv.first, nsec);
                    break;
                default: assert(!"unexpected data kind");
            }
            idx++;
        }
        return status::success;
    }

private:
    stream_profiler_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_profiler_t);

    struct span_t {
        uint64_t start = std::numeric_limits<uint64_t>::max();
        uint64_t end = 0;
    };

    status_t query_entry(const xpu::event_t &event, entry_t &entry) const {
        const auto &ze_event = xpu::ze::event_t::from(event);
        assert(ze_event.size() == 1);
        ze_kernel_timestamp_result_t kernel_timestamp_result;
        ZE_CHECK(ze::zeEventQueryKernelTimestamp(
                ze_event[0], &kernel_timestamp_result));
        entry = entry_t(
                kernel_timestamp_result, max_timestamp_value_, timestamp_freq_);
        return status::success;
    }

    uint64_t get_duration(uint64_t start, uint64_t end) const {
        return duration_cycles(start, end, max_timestamp_value_);
    }

    double timestamp_freq_;
    uint64_t max_timestamp_value_;
};

struct verbose_profiler_t : public xpu::verbose_profiler_t {
    verbose_profiler_t(const impl::stream_t *stream, double timestamp_freq,
            uint64_t max_timestamp_value)
        : xpu::verbose_profiler_t(stream)
        , timestamp_freq_(timestamp_freq)
        , max_timestamp_value_(max_timestamp_value) {}

    ~verbose_profiler_t() override { cleanup(); }
    verbose_profiler_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(verbose_profiler_t);

private:
    status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const override;

    bool is_event_complete(
            const std::shared_ptr<xpu::event_t> &event) const override;

    void wait_for_event_completion(
            const std::shared_ptr<xpu::event_t> &event) const override;

    uint64_t get_duration_cycles(
            uint64_t start_cycles, uint64_t end_cycles) const {
        return duration_cycles(start_cycles, end_cycles, max_timestamp_value_);
    }

    double timestamp_freq_;
    uint64_t max_timestamp_value_;
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_STREAM_PROFILER_HPP
