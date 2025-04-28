/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "xpu/sycl/context.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

status_t async_verbose_tracker_t::refresh_tracking_info() const {

    using namespace ::sycl::info;

    for (auto &ev : kevents_) {
        const xpu::sycl::event_t &sycl_kevent
                = *utils::downcast<xpu::sycl::event_t *>(ev.event.get());
        auto start
                = sycl_event[0]
                          .get_profiling_info<event_profiling::command_start>();
        auto end = sycl_event[0]
                           .get_profiling_info<event_profiling::command_end>();
        auto e_status
                = sycl_event[0].get_info<event::command_execution_status>();

        switch (e_status) {
            case event_command_status::queued:
                tracker.tstatus = exec_status_t::queued;
                break;
            case event_command_status::running:
                tracker.tstatus = exec_status_t::running;
                break;
            case event_command_status::complete:
                tracker.tstatus = exec_status_t::finished;
                break;
            default: tracker.tstatus = exec_status_t::off; break;
        }

        auto &entry = stamp2tracker_[ev.stamp];
        tracker.min_nsec = start;
        tracker.max_nsec = end;
    }
    return status::success;
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
