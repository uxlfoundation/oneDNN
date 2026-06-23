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

#ifndef XPU_SYCL_STREAM_PROFILER_HPP
#define XPU_SYCL_STREAM_PROFILER_HPP

#include "common/c_types_map.hpp"

#include "xpu/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

struct stream_profiler_t : public xpu::stream_profiler_t {
    stream_profiler_t(const stream_t *stream)
        : xpu::stream_profiler_t(stream) {}

    status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const override;
};

struct verbose_profiler_t : public xpu::verbose_profiler_t {
    verbose_profiler_t(const impl::stream_t *stream)
        : xpu::verbose_profiler_t(stream), active_(true) {}

    ~verbose_profiler_t() override { cleanup(); }

    status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const override;

    bool is_event_complete(
            const std::shared_ptr<xpu::event_t> &event) const override;

    void wait_for_event_completion(
            const std::shared_ptr<xpu::event_t> &event) const override;

    // pausing capabilities are added to allow skipping event profiling
    // queires when they are temporarily unvaiable
    // (sycl graph execution) -pausing action is localized to each
    // thread for multi-threaded execution
    bool is_active() const override { return active_; }
    void start_profiling() { active_ = true; }
    void pause_profiling() { active_ = false; }

protected:
    bool active_;
};

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
