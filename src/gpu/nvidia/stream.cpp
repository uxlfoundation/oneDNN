/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "common/verbose.hpp"

#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_compat.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

cublasHandle_t &stream_t::get_cublas_handle(CUstream cuda_stream) {
    if (!cuda_stream) cuda_stream = get_underlying_stream();
    auto e = utils::downcast<nvidia::engine_t *>(engine());
    assert(e->context() == queue().get_context());
    e->activate_stream_cublas(cuda_stream);
    return *(e->get_cublas_handle());
}

cudnnHandle_t &stream_t::get_cudnn_handle(CUstream cuda_stream) {
    if (!cuda_stream) cuda_stream = get_underlying_stream();
    auto e = utils::downcast<nvidia::engine_t *>(engine());
    assert(e->context() == queue().get_context());
    e->activate_stream_cudnn(cuda_stream);
    return *(e->get_cudnn_handle());
}

// If interop_handle is being submitted to a SYCL queue that is recording to a
// graph, then put cuda stream into capture mode.
void stream_t::begin_recording_if_graph(
        const ::sycl::interop_handle &ih, CUstream cuda_stream) {
#if SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
    if (!ih.ext_codeplay_has_graph()) { return; }
    // After CUDA 12.3 we can use cuStreamBeginCaptureToGraph to capture
    // the stream directly in the native graph, rather than needing to
    // instantiate the stream capture as a new graph.
#if CUDA_VERSION >= 12030
    CUgraph cuda_graph = ih.ext_codeplay_get_native_graph<
            sycl::backend::ext_oneapi_cuda>();
    CUDA_EXECUTE_FUNC(cuStreamBeginCaptureToGraph, cuda_stream, cuda_graph,
            nullptr, nullptr, 0, CU_STREAM_CAPTURE_MODE_GLOBAL);
#else
    CUDA_EXECUTE_FUNC(
            cuStreamBeginCapture, cuda_stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
#endif
#endif
}

void stream_t::end_recording_if_graph(
        const ::sycl::interop_handle &ih, CUstream cuda_stream) {
#if SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND >= 2
    if (!ih.ext_codeplay_has_graph()) { return; }

    CUgraph cuda_graph = ih.ext_codeplay_get_native_graph<
            sycl::backend::ext_oneapi_cuda>();
#if CUDA_VERSION >= 12030
    CUDA_EXECUTE_FUNC(cuStreamEndCapture, cuda_stream, &cuda_graph);
#else
    // cuStreamEndCapture returns a new graph, if we overwrite
    // "cuda_graph" it won't be picked up by the SYCL runtime, as
    // "ext_codeplay_get_native_graph" returns a passed-by-value pointer.
    CUgraph recorded_graph;
    CUDA_EXECUTE_FUNC(cuStreamEndCapture, cuda_stream, &recorded_graph);

    // Add graph to native graph as a child node
    // Need to return a node object for the node to be created,
    // can't be nullptr.
    CUgraphNode node;
    CUDA_EXECUTE_FUNC(cuGraphAddChildGraphNode, &node, cuda_graph, nullptr, 0,
            recorded_graph);
#endif
#endif
}

// the stream_t will not own this. it is an observer pointer
CUstream stream_t::get_underlying_stream() {
    return compat::get_native<CUstream>(queue());
}

// the stream_t will not own this. it is an observer pointer
CUcontext stream_t::get_underlying_context() {
    return compat::get_native<CUcontext>(queue().get_device());
}

// the stream_t will not own this. it is an observer pointer
CUdevice stream_t::get_underlying_device() {
    return compat::get_native<CUdevice>(queue().get_device());
}

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    VCONDCHECK(primitive, create, check, stream,
            is_profiling_enabled() == false, status::unimplemented,
            VERBOSE_PROFILING_UNSUPPORTED);

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(engine());
    auto status = status::success;

    if (!impl()->queue()) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        ::sycl::property_list prop_list;
        if (flags() & stream_flags::in_order)
            prop_list = {::sycl::property::queue::in_order {}};
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, prop_list));
    } else {
        auto sycl_dev = queue().get_device();
        bool args_ok
                = engine()->kind() == engine_kind::gpu && sycl_dev.is_gpu();
        if (!args_ok) return status::invalid_arguments;

        auto queue_context = get_underlying_context();
        auto queue_device = get_underlying_device();

        auto engine_context = sycl_engine.get_underlying_context();
        auto engine_device = sycl_engine.get_underlying_device();

        status = ((engine_device != queue_device)
                         || (engine_context != queue_context))
                ? status::invalid_arguments
                : status::success;

        // We don't want to keep a reference to engine_context, which is
        // retained in get_underlying_context
        CUDA_EXECUTE_FUNC(cuDevicePrimaryCtxRelease_v2, engine_device);
        CUDA_EXECUTE_FUNC(cuDevicePrimaryCtxRelease_v2, queue_device);
    }

    return status;
}

status_t stream_t::interop_task(
        std::function<void(::sycl::handler &)> sycl_cuda_interop_) {
    try {
        auto event = queue().submit([&](::sycl::handler &cgh) {
            cgh.depends_on(sycl_ctx().get_sycl_deps().events);
            sycl_cuda_interop_(cgh);
        });
        this->sycl_ctx().get_sycl_deps().events = {event};
        return status::success;
    } catch (std::runtime_error &e) { return status::runtime_error; }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
