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

#include "utils/execution_mode.hpp"

#include "common.hpp"
#include "utils/stream_kind.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

execution_mode_t default_execution_mode {execution_mode_t::direct};
execution_mode_t execution_mode {default_execution_mode};

std::ostream &operator<<(std::ostream &s, execution_mode_t mode) {
    if (mode == execution_mode_t::direct) s << "direct";
    if (mode == execution_mode_t::graph) s << "graph";
    if (mode == execution_mode_t::native_graph) s << "native_graph";

    return s;
}

bool use_sycl_graph_exec(const engine_t &engine) {
    return is_gpu(engine) && is_sycl_engine(engine)
            && (execution_mode == execution_mode_t::graph
                    || execution_mode == execution_mode_t::native_graph);
}

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
namespace {

namespace syclex = ::sycl::ext::oneapi::experimental;
using sycl_exec_graph_t
        = syclex::command_graph<syclex::graph_state::executable>;

// Validate that the queue supports SYCL command graph (Level Zero backend).
// Returns OK on success, FAIL on error (with res->state set to FAILED).
int validate_backend(const ::sycl::queue &queue, res_t *res) {
    if (queue.get_device().get_backend()
            != ::sycl::backend::ext_oneapi_level_zero) {
        BENCHDNN_PRINT(0, "%s %s\n",
                "[ERROR] SYCL graph execution is only available on Level "
                "Zero backend; currently using:",
                ::sycl::detail::get_backend_name_no_vendor(
                        queue.get_device().get_backend())
                        .data());
        if (res) res->state = FAILED;
        return FAIL;
    }
    return OK;
}

// Verify that compiler supports this mode.
// Note: can't be checked in parser.cpp since this macros requires <sycl.hpp>
// header to be defined.
int validate_features(res_t *res) {
    if (execution_mode == execution_mode_t::native_graph) {
#if defined(SYCL_EXT_ONEAPI_GRAPH) && SYCL_EXT_ONEAPI_GRAPH >= 2
        return OK;
#elif defined(SYCL_EXT_ONEAPI_GRAPH) && SYCL_EXT_ONEAPI_GRAPH < 2
        BENCHDNN_PRINTF(0, "%s",
                "Error: this compiler version doesn't support native graph "
                "recording mode");
        if (res) res->state = FAILED;
        return FAIL;
#elif !defined(SYCL_EXT_ONEAPI_GRAPH)
        BENCHDNN_PRINTF(0, "%s",
                "Error: this compiler doesn't support native graph recording");
        if (res) res->state = FAILED;
        return FAIL;
#endif
    }
    return OK;
}

// Record operations into a SYCL command graph and return the finalized
// executable. `record_func` is called while the queue is in recording state.
// Returns nullptr on failure (with res->state set to FAILED).
std::unique_ptr<sycl_exec_graph_t> record_and_finalize(stream_t &stream,
        ::sycl::queue &queue, const std::function<void()> &record_func,
        res_t *res) {
    BENCHDNN_PRINT(
            2, "%s\n", "[INFO] Using experimental SYCL graph execution.");

    // Drain the queue to clear any pending events before recording.
    stream.wait();
    queue.wait_and_throw();

    ::sycl::property_list prop_list {
            syclex::property::graph::assume_buffer_outlives_graph {}};
    if (execution_mode == execution_mode_t::native_graph) {
#if defined(SYCL_EXT_ONEAPI_GRAPH) && SYCL_EXT_ONEAPI_GRAPH >= 2
        prop_list = {syclex::property::graph::assume_buffer_outlives_graph {},
                syclex::property::graph::enable_native_recording {}};
#endif
    }
    syclex::command_graph sycl_graph {
            queue.get_context(), queue.get_device(), prop_list};

    try {
        sycl_graph.begin_recording(queue);
        record_func();
        sycl_graph.end_recording(queue);
        return std::unique_ptr<sycl_exec_graph_t>(
                new sycl_exec_graph_t(sycl_graph.finalize()));
    } catch (const std::exception &e) {
        // Ensure recording is stopped.
        try {
            sycl_graph.end_recording(queue);
        } catch (...) {}
        BENCHDNN_PRINT(
                0, "%s %s\n", "[ERROR] SYCL graph record exception:", e.what());
        if (res) res->state = FAILED;
        return nullptr;
    }
}

// Replay a finalized executable graph. Returns OK on success, FAIL on error.
int replay(::sycl::queue &queue, const sycl_exec_graph_t &exec_graph,
        res_t *res, bool wait = true) {
    try {
        auto event = queue.ext_oneapi_graph(exec_graph);
        if (wait) event.wait();
        return OK;
    } catch (const std::exception &e) {
        BENCHDNN_PRINT(
                0, "%s %s\n", "[ERROR] SYCL graph replay exception:", e.what());
        if (res) res->state = FAILED;
        return FAIL;
    }
}

} // namespace
#endif

int execute_in_graph_mode(stream_t &stream,
        const std::function<void()> &record_func, res_t *res) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    try {
        ::sycl::queue queue = dnnl::sycl_interop::get_queue(stream);
        if (validate_backend(queue, res) != OK) return FAIL;
        if (validate_features(res) != OK) return FAIL;

        auto exec = record_and_finalize(stream, queue, record_func, res);
        if (!exec) return FAIL;

        int replay_status = OK;
        TIME_EXECUTE(replay_status = replay(queue, *exec, res));
        return replay_status;
    } catch (const std::exception &e) {
        BENCHDNN_PRINT(0, "%s %s\n",
                "[ERROR] SYCL graph execution exception:", e.what());
        if (res) res->state = FAILED;
        return FAIL;
    }
#else
    BENCHDNN_PRINT(0, "%s\n",
            "[ERROR] Graph execution is only available on SYCL runtime with "
            "Level Zero backend.");
    if (res) res->state = FAILED;
    return FAIL;
#endif
}
