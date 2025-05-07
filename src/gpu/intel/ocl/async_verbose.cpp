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

#include <CL/cl.h>

#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "gpu/generic/async_verbose.hpp"
#include "xpu/ocl/context.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t async_verbose_tracker_t::refresh_tracking_info() const {

    for (auto &ev : kevents_) {
        const xpu::ocl::event_t &ocl_kevent
                = *utils::downcast<xpu::ocl::event_t *>(ev.event.get());
        cl_ulong start, end;
        OCL_CHECK(clGetEventProfilingInfo(ocl_kevent[0].get(),
                CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(ocl_kevent[0].get(),
                CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));

        cl_int e_status;
        cl_int_err = clGetEventInfo(ocl_kevent[0].get(),
                CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(e_status), &e_status,
                nullptr);

        switch (e_status) {
            case CL_QUEUED: tracker.tstatus = exec_status_t::queued; break;
            case CL_RUNNING: tracker.tstatus = exec_status_t::running; break;
            case CL_COMPLETE: tracker.tstatus = exec_status_t::finished; break;
            default: tracker.tstatus = exec_status_t::off; break;
        }

        auto &tracker = stamp2tracker_[ev.stamp];
        tracker.min_nsec = start;
        tracker.max_nsec = end;
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl