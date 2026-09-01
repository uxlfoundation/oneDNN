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

#ifndef COMMON_VERBOSE_PROFILER_HPP
#define COMMON_VERBOSE_PROFILER_HPP

#include <string>
#include "c_types_map.hpp"

namespace dnnl {
namespace impl {

// The verbose profiler logs primitive profiling information using device-
// measured execution times without host-to-device synchronization overhead or
// blocking stream.wait() calls. It operates asynchronously by polling device
// events to track primitive completion status.
// During primitive execution, the profiler groups and registers kernel events
// for each primitive with the associated profiling metadata. During each
// primitive post-exec hook, it polls previously registered events to identify
// completed primitives and logs their timing info. Pending primitives remain
// in the profiling data containers (different for different runtimes) list
// until detected as complete in subsequent polling cycles.
// During profiler destruction, any remaining primitives are checked and
// waited for to ensure no executions are left unlogged.
// This profiler is intended to be thread-local via thread_local_storage_t,
// ensuring thread-safety for multi-threaded execution environments. Each
// thread maintains its own profiler instance and event tracking state,
// operating independently from other stream profilers during primitive
// execution.
struct verbose_profiler_t {
    verbose_profiler_t(const stream_t *stream)
        : stream_(stream), active_(true) {}

    virtual ~verbose_profiler_t() = default;

    // Pausing capabilities are added to allow skipping event profiling
    // queries when they are temporarily unavailable.
    // These methods check and update profiler status where such force-pausing
    // is required. Pausing action is localized to each thread for multi-
    // threaded execution
    bool is_active() const { return active_; }
    void start_profiling() { active_ = true; }
    void pause_profiling() { active_ = false; }

    // The profiler operates through a multi-step event tracking workflow:
    // 1. stream->before_exec_hook() calls update_event_list()
    //    to add a new entry for the current primitive. Since there can be
    //    multiple event registration calls, this spot is a guaranteed single
    //    call for the coming primitive.
    // 2. During primitive execution, register_event() (defined in subclasses)
    //    adds device events to the latest primitive entry corresponding to the
    //    number of invoked kernels.
    // 3. add_to_pending_primitive_list() stores profiling metadata
    //    for the registered primitive
    // 4. stream->after_exec_hook() calls check_for_completed_primitives()
    //    to poll events and log completed primitives
    // 5. Incomplete primitives are not removed until detected as
    //    complete in future polling cycles
    // This asynchronous workflow allows tracking multiple concurrent
    // primitives without blocking execution.
    virtual void update_event_list() = 0;

    // populates profiling metadata for the last primitive entry
    virtual void add_to_pending_primitive_list(
            double start_ms, const std::string &pd_info, uint64_t component)
            = 0;

    // Completed primitive executions are periodically checked and logged
    // during after_exec_hook() calls and during stream destruction.
    // The profiler does not wait for pending events to complete
    // and instead prints them at the next concurrent after_exec_hook()
    // call.
    virtual void check_for_completed_primitives() = 0;

protected:
    const stream_t *stream_;
    bool active_;

private:
    // This is invoked during profiler destruction to account for any
    // pending primitives that have not yet been logged.
    virtual void wait_for_pending_primitives() = 0;
};

} // namespace impl
} // namespace dnnl

#endif
