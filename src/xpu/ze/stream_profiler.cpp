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

#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/ze/context.hpp"
#include "xpu/ze/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

status_t verbose_profiler_t::get_aggregate_exec_time(
        uint64_t stamp, double &duration_ms) const {

    auto prof_data = event_map_.find(stamp);
    if (prof_data == event_map_.end()) return status::invalid_arguments;

    const auto &evts = prof_data->second.prim_evts;
    if (evts.empty()) {
        duration_ms = 0.0;
        return status::success;
    }

    uint64_t agg_start = UINT64_MAX;
    uint64_t agg_end = 0;

    // For verbose logging, aggregate execution time for a primitive is
    // determined from the start time of the first queued primitive event
    // and the end time of the last primitive event
    for (const auto &ev : evts) {
        const xpu::ze::event_t &ze_event
                = *utils::downcast<xpu::ze::event_t *>(ev.get());

        if (ze_event.size() == 0 || ze_event[0] == nullptr) { continue; }

        ze_kernel_timestamp_result_t timestamp_result;
        CHECK(ze::zeEventQueryKernelTimestamp(ze_event[0], &timestamp_result));

        uint64_t start_ns = timestamp_result.global.kernelStart;
        uint64_t end_ns = timestamp_result.global.kernelEnd;

        agg_start = std::min(agg_start, start_ns);
        agg_end = std::max(agg_end, end_ns);
    }

    if (agg_start == UINT64_MAX) {
        duration_ms = 0.0;
        return status::success;
    }

    uint64_t duration_cycles = get_duration_cycles(agg_start, agg_end);
    uint64_t duration_ns
            = static_cast<uint64_t>(timestamp_freq_ * duration_cycles);
    duration_ms = static_cast<double>(duration_ns) * 1e-6;
    return status::success;
}

bool verbose_profiler_t::is_event_complete(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!event) return true;

    const xpu::ze::event_t &ze_event
            = *utils::downcast<xpu::ze::event_t *>(event.get());

    status_t result = ze::zeEventQueryStatus(ze_event[0]);

    return (result == status::success);
}

void verbose_profiler_t::wait_for_event_completion(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!event) return;

    const xpu::ze::event_t &ze_event
            = *utils::downcast<xpu::ze::event_t *>(event.get());

    status_t result = ze::zeEventHostSynchronize(ze_event[0], UINT64_MAX);
    if (result != status::success) {
        VWARN(primitive, exec, "ze event synchronization failed");
    }
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
