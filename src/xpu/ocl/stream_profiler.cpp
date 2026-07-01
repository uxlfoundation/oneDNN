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

#include <CL/cl.h>

#include <limits>
#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/context.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    if (!num_entries) return status::invalid_arguments;
    bool is_per_kernel = (data_kind == profiling_data_kind::time_per_kernel);
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

    std::map<uint64_t, xpu::stream_profiler_t::entry_t> stamp2entry;
    int idx = 0;
    for (auto &ev : events_) {
        const auto &ocl_event = xpu::ocl::event_t::from(*ev.event);
        cl_ulong beg, end;
        assert(ocl_event.size() == 1);
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_START, sizeof(beg), &beg, nullptr));
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        if (is_per_kernel) {
            data[idx++] = static_cast<uint64_t>(end - beg);
            continue;
        }
        auto &entry = stamp2entry[ev.stamp];
        entry.min_nsec = std::min(entry.min_nsec, beg);
        entry.max_nsec = std::max(entry.max_nsec, end);
        const auto *gpu_stream
                = utils::downcast<const gpu::stream_t *>(stream_);
        entry.freq += gpu_stream->get_freq(*ev.event);
        entry.kernel_count++;
    }
    if (is_per_kernel) return status::success;
    return xpu::stream_profiler_t::get_info_impl(stamp2entry, data_kind, data);
}

status_t verbose_profiler_t::get_aggregate_exec_time(
        size_t index, double &duration_ms) const {
    if (index >= profiling_data_.size()) {
        VERROR(primitive, exec,
                "profiling error: invalid index %zu, profiling_data size is "
                "%zu",
                index, profiling_data_.size());
        return status::success;
    }
    const auto &prof_data = profiling_data_[index];

    const auto &evts = prof_data.prim_events_;
    if (evts.empty()) {
        duration_ms = 0.0;
        return status::success;
    }

    cl_ulong agg_start = std::numeric_limits<cl_ulong>::max();
    cl_ulong agg_end = 0;

    // For verbose logging, aggregate execution time for a primitive is
    // determined from the start time of the first queued primitive event
    // and the end time of the last primitive event
    for (const auto &ev : evts) {
        const auto &ocl_event = xpu::ocl::event_t::from(*ev);
        size_t last_idx = ocl_event.size() - 1;
        assert(last_idx >= 0);

        cl_ulong evbeg, evend;
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_START, sizeof(evbeg), &evbeg, nullptr));
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[last_idx].get(),
                CL_PROFILING_COMMAND_END, sizeof(evend), &evend, nullptr));
        agg_start = std::min(agg_start, evbeg);
        agg_end = std::max(agg_end, evend);
    }

    if (agg_end < agg_start) { return status::runtime_error; }

    // TODO: Consolidate timing calculation calls between different
    // profilers to avoid code duplication and ensure consistent time
    // conversion logic
    duration_ms = static_cast<double>(agg_end - agg_start) * 1e-6;
    return status::success;
}

bool verbose_profiler_t::is_event_complete(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!event) return true;
    const auto &ocl_event = xpu::ocl::event_t::from(*event);
    size_t last_idx = ocl_event.size() - 1;
    assert(last_idx >= 0);
    cl_int status;
    cl_int err = xpu::ocl::clGetEventInfo(ocl_event[last_idx],
            CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status,
            nullptr);

    if (err != CL_SUCCESS) {
        VERROR(primitive, exec, "failed to get event status: %d", err);
        return false;
    }

    return status == CL_COMPLETE;
}

void verbose_profiler_t::wait_for_event_completion(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!event) return;
    const auto &ocl_event = xpu::ocl::event_t::from(*event);
    size_t last_idx = ocl_event.size() - 1;
    assert(last_idx >= 0);
    cl_event cl_ev = ocl_event[last_idx];
    cl_int err = xpu::ocl::clWaitForEvents(1, &cl_ev);

    if (err != CL_SUCCESS) {
        // Note: Cannot throw from destructor context, so just
        // logging error
        VERROR(primitive, exec, "failed to wait for event completion: %d", err);
    }
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
