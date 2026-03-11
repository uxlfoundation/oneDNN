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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_GATED_MLP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_GATED_MLP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/gated_mlp_primitive.hpp"
#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

#define VDISPATCH_GRAPH_GATED_MLP(msg, ...) \
    VINFO(graph, create, dispatch, compile, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <bool quantized = false>
struct gated_mlp_base_t : public kernel_base_t {
private:
    std::shared_ptr<kernel_base_t> kernel;

public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *engine, const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        const engine_kind_t ekind = engine->kind();
        bool enable_ukernel = false;

        if (ekind == engine_kind::gpu) { enable_ukernel = !force_primitive(); }

        status_t ret = status::unimplemented;

        if (enable_ukernel) {
            kernel = std::make_shared<
                    gated_mlp_primitive_kernel_t<quantized>>();
            ret = kernel->compile_impl(part, engine, inputs, outputs);
        }

        if (ret != status::success) {
            kernel = std::make_shared<larger_partition_kernel_t>();
            ret = kernel->compile_impl(part, engine, inputs, outputs);
        }
        if (ret == status::success)
            VDISPATCH_GRAPH_GATED_MLP(
                    "gated_mlp is dispatched to (%s)", kernel->str().c_str());
        else
            VDISPATCH_GRAPH_GATED_MLP("gated_mlp is failed to dispatch");
        return ret;
    }

    // Use large partition kernel as defautl. Turn the env var to 0 to select
    // gated_mlp primitive.
    bool force_primitive() const {
        const int force = graph::utils::getenv_int_internal(
                "GRAPH_GATED_MLP_FORCE_PRIMITIVE", 1);
        return force > 0;
    }

    status_t execute_impl(const stream_t *stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        return kernel->execute_impl(stream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &deps,
            ::sycl::event *event) override {
        return kernel->sycl_execute_impl(stream, inputs, outputs, deps, event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &deps, cl_event *event) override {
        return kernel->ocl_execute_impl(stream, inputs, outputs, deps, event);
    }
#endif

    std::string str() const override { return kernel->str(); }
};
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
