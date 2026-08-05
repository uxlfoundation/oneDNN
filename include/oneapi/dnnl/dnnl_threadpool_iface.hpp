/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @file
/// Threadpool Interoperability C++ Types

#ifndef ONEAPI_DNNL_DNNL_THREADPOOL_IFACE_HPP
#define ONEAPI_DNNL_DNNL_THREADPOOL_IFACE_HPP
// NOLINTBEGIN(readability-identifier-naming)

#include <cstdint>
#include <functional>
#include <memory>

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_threadpool_interop
/// @{

namespace threadpool_interop {

/// Completion event for async threadpool work. oneDNN's verbose profiler
/// polls these to determine when deferred execution has finished and to
/// extract timing information.
struct threadpool_event_t {
    virtual ~threadpool_event_t() = default;
    /// Returns true if the associated work has completed.
    virtual bool is_complete() const = 0;
    /// Blocks until completion.
    virtual void wait() const = 0;
    /// Measured execution time in milliseconds. Valid only after completion.
    virtual double exec_time_ms() const = 0;
};

/// Abstract threadpool interface. The users are expected to subclass this
/// interface and pass an object to the library during CPU stream creation or
/// directly in case of BLAS functions.
struct threadpool_iface {
    /// Returns the number of worker threads.
    virtual int get_num_threads() const = 0;

    /// Returns true if the calling thread belongs to this threadpool.
    virtual bool get_in_parallel() const = 0;

    /// Submits n instances of a closure for execution in parallel:
    ///
    /// for (int i = 0; i < n; i++) fn(i, n);
    ///
    virtual void parallel_for(int n, const std::function<void(int, int)> &fn)
            = 0;

    /// Returns threadpool behavior flags bit mask (see below).
    virtual uint64_t get_flags() const = 0;

    // Does nothing if SYNCHRONOUS, waits for all jobs for ASYNCHRONOUS
    virtual void wait() = 0;

    /// Enables or disables verbose profiling event tracking. When enabled,
    /// the threadpool creates completion events that oneDNN can poll.
    /// Called by the stream during initialization.
    virtual void set_verbose_profiling(bool) {}

    /// Returns a completion event for the work submitted since the last call.
    /// oneDNN's verbose profiler polls this to detect completion and extract
    /// timing. Returns nullptr if profiling is disabled or unsupported.
    virtual std::shared_ptr<threadpool_event_t> get_completion_event() {
        return nullptr;
    }

    /// If set, parallel_for() returns immediately and oneDNN needs implement
    /// waiting for the submitted closures to finish execution on its own.
    static constexpr uint64_t ASYNCHRONOUS = 1;

    virtual ~threadpool_iface() = default;
};

} // namespace threadpool_interop

/// @} dnnl_api_threadpool_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

// NOLINTEND(readability-identifier-naming)
#endif /* ONEAPI_DNNL_DNNL_THREADPOOL_IFACE_HPP */
