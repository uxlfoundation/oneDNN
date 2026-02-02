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

#include <unordered_set>

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

class stream_profiler_t : public xpu::stream_profiler_t {
public:
    class entry_t {
    public:
        entry_t() = delete;

        entry_t(ze_kernel_timestamp_result_t &kernel_timestamp_result,
                uint64_t max_timestamp_value, double timestamp_freq)
            : context_(get_timestamp(
                      kernel_timestamp_result.context, max_timestamp_value))
            , freq_(timestamp_freq) {}

        uint64_t get_cycles() const { return context_; }

        uint64_t get_nsec() const { return get_cycles() * freq_; }

    private:
        uint64_t get_timestamp(
                ze_kernel_timestamp_data_t &ts, uint64_t max_timestamp_value) {
            return (ts.kernelEnd >= ts.kernelStart)
                    ? (ts.kernelEnd - ts.kernelStart)
                    : ((max_timestamp_value - ts.kernelStart) + ts.kernelEnd
                              + 1);
        }

        uint64_t context_;
        double freq_;
    };

    stream_profiler_t(const impl::stream_t *stream, double timestamp_freq,
            uint64_t max_timestamp_value)
        : xpu::stream_profiler_t(stream)
        , timestamp_freq_(timestamp_freq)
        , max_timestamp_value_(max_timestamp_value) {}

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

        std::map<uint64_t, entry_t> stamp2entry;
        int idx = 0;
        for (auto &ev : events_) {
            const ze::event_t &ze_event
                    = *utils::downcast<ze::event_t *>(ev.event.get());

            ze_kernel_timestamp_result_t kernel_timestamp_result;
            CHECK(ze::zeEventQueryKernelTimestamp(
                    ze_event[0], &kernel_timestamp_result));

            entry_t entry(kernel_timestamp_result, max_timestamp_value_,
                    timestamp_freq_);
            if (is_per_kernel) {
                data[idx++] = entry.get_nsec();
                continue;
            }
            stamp2entry.emplace(ev.stamp, entry);
        }
        if (is_per_kernel) return status::success;

        return get_info_impl(stamp2entry, data_kind, data);
    }

private:
    stream_profiler_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_profiler_t);

    status_t get_info_impl(const std::map<uint64_t, entry_t> &stamp2entry,
            profiling_data_kind_t data_kind, uint64_t *data) const {
        int idx = 0;
        for (auto &kv : stamp2entry) {
            auto &e = kv.second;
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = e.get_nsec(); break;
                case profiling_data_kind::cycles: {
                    data[idx] = e.get_cycles();
                    if (callback_) callback_(kv.first, e.get_nsec());
                    break;
                }
                default: assert(!"unexpected data kind");
            }
            idx++;
        }
        return status::success;
    }

    double timestamp_freq_;
    uint64_t max_timestamp_value_;
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_STREAM_PROFILER_HPP
