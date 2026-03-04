/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <cstring>

#include <CL/cl.h>

#include "common/verbose.hpp"

#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/memory_storage.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/stream.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t stream_t::init() {
    if (is_profiling_enabled()) {
        profiler_ = utils::make_unique<xpu::ocl::stream_profiler_t>(this);
        mdapi_helper_ = utils::make_unique<mdapi_helper_t>();
    }
    // Restore queue on successful exit, otherwise queue may be released
    // without retain
    cl_command_queue queue = impl()->queue();
    CHECK(impl()->set_queue(nullptr));

    assert(engine()->kind() == engine_kind::gpu);

    const auto *ocl_engine_impl
            = utils::downcast<const xpu::ocl::engine_impl_t *>(
                    engine()->impl());

    // Create queue if it is not set
    if (!queue) {
        cl_int err;
        queue = create_queue(
                ocl_engine_impl->context(), ocl_engine_impl->device(), &err);
        OCL_CHECK(err);
    } else {
        // Check that queue is compatible with the engine
        cl_context ocl_ctx;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                sizeof(cl_context), &ocl_ctx, nullptr));

        cl_device_id ocl_dev;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                sizeof(cl_device_id), &ocl_dev, nullptr));

        if (ocl_engine_impl->device() != ocl_dev
                || ocl_engine_impl->context() != ocl_ctx)
            return status::invalid_arguments;

        OCL_CHECK(xpu::ocl::clRetainCommandQueue(queue));
    }
    CHECK(impl()->set_queue(queue));

    if (is_profiling_enabled()) {
        cl_command_queue_properties props;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(impl()->queue(),
                CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr));
        bool is_out_of_order
                = (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        if (is_out_of_order) {
            VERROR(common, ocl,
                    "OpenCL kernel profiling is not "
                    "supported with out-of-order queues");
            return status::invalid_arguments;
        }
    }

    return status::success;
}

cl_command_queue stream_t::create_queue(
        cl_context ctx, cl_device_id dev, cl_int *err) const {
    if (is_profiling_enabled() && mdapi_helper_) {
        auto ret = mdapi_helper_->create_queue(ctx, dev, err);
        if (ret) return ret;
    }

    const bool is_out_of_order = (flags() & stream_flags::out_of_order);

    cl_command_queue_properties queue_props {};
    if (is_profiling_enabled()) queue_props |= CL_QUEUE_PROFILING_ENABLE;
    if (is_out_of_order) queue_props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, queue_props, 0};
    return xpu::ocl::clCreateCommandQueueWithProperties(ctx, dev, props, err);
#else
    return xpu::ocl::clCreateCommandQueue(ctx, dev, queue_props, err);
#endif
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void stream_t::after_exec_hook() {
    ocl_ctx().set_deps(xpu::ocl::event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
}

status_t stream_t::run_verbose_profiler(
        std::string &pd_info, double start_ms) const {

    // utilize the verbose profiler only for profile_exec verbose levels.
    if (!is_verbose_profiler_enabled()) return status::invalid_arguments;

    // Captured output event acts as the anchor to track primitive execution
    cl_event out_evt = get_output_event();

    // as the stamp count increments each time the profiler is unpaused, it
    // also tracks the primitive executions.
    uint64_t curr_stamp = profiler_->stamp();

    if (!curr_stamp) {
        VWARN(primitive, exec,
                "%s, profiling error: failed to record primitive events in "
                "context",
                pd_info.c_str());
        VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(), 0.f);
        return status::success;
    }

    // An OpenCL marker is used for asynchronous printing of profiling info.
    // The callback triggered after primitive execution calculates and prints
    // the execution times.
    cl_command_queue q = queue();
    cl_event marker = nullptr;
    cl_int err = xpu::ocl::clEnqueueMarkerWithWaitList(q, 1, &out_evt, &marker);

    if (err != CL_SUCCESS || !marker) {
        VWARN(primitive, exec,
                "%s, profiling error: failed to attach OpenCL marker to output "
                "event",
                pd_info.c_str());
        VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(), 0.f);
        return status::success;
    }

    xpu::stream_profiler_t *prof_ptr = profiler_.get();

    struct payload_t {
        xpu::stream_profiler_t *prof;
        std::string info_str;
        double start;
        uint64_t stamp;
    };

    std::unique_ptr<payload_t> payload(new payload_t());
    payload->prof = prof_ptr;
    payload->info_str = pd_info;
    payload->start = start_ms;
    payload->stamp = curr_stamp;
    void *pluser = payload.get();

    // The prompt ensures the verbose headers are printed if they aren't
    // already. Printing the headers asynchronously during the callback can
    // result in access failures when printing engine-specific info.
    verbose_printf(verbose_t::exec_profile, "\r");

    err = xpu::ocl::clSetEventCallback(
            marker, CL_COMPLETE, [](cl_event ev, cl_int, void *user) {
        std::unique_ptr<payload_t> hold(static_cast<payload_t *>(user));

        double duration_ms = 0.0;

        if (hold->prof) {
            hold->prof->get_aggregate_exec_timing(hold->stamp, duration_ms);
        } else {
            VWARN(primitive, exec, "%s, profiling error: profiler absent",
                    hold->info_str.c_str());
        }

        VPROF(hold->start, primitive, exec, VERBOSE_profile,
                hold->info_str.c_str(), duration_ms);

        xpu::ocl::clReleaseEvent(ev);
    }, pluser);

    if (err != CL_SUCCESS) {
        xpu::ocl::clReleaseEvent(marker);
        VWARN(primitive, exec,
                "%s, profiling error: failed to set event callback for "
                "printing exec info",
                pd_info.c_str());
        VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(), 0.f);
        return status::success;
    }

    (void)payload.release();

    return status::success;
}

status_t stream_t::copy(const memory_storage_t &src,
        const memory_storage_t &dst, size_t size, const xpu::event_t &deps,
        xpu::event_t &out_dep) {
    return impl()->copy(this, src, dst, size, deps, out_dep, profiler_.get());
}

status_t stream_t::fill(const memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    return impl()->fill(
            this, dst, pattern, size, deps, out_dep, profiler_.get());
}

status_t stream_t::barrier() {
    return impl()->barrier();
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
