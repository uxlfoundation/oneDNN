/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_MQA_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_MQA_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/mqa_decomp.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The second template param dt is used to indicate the internal data type of
// int8 mqa pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
struct mqa_base_t : public kernel_base_t {
private:
    std::shared_ptr<kernel_base_t> kernel;

public:
    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        const engine_kind_t ekind = eng->kind();
        const bool enable_decomp
                = ekind == engine_kind::cpu && enable_decomp_kernel();
        status_t mqa_decomp_status = status::success;
        if (enable_decomp) {
            kernel = std::make_shared<mqa_decomp_kernel_t<quantized, dt>>();
            mqa_decomp_status
                    = kernel->compile_impl(part, eng, inputs, outputs);
        }

        if (!enable_decomp || mqa_decomp_status != status::success) {
            kernel = std::make_shared<larger_partition_kernel_t>();
            return kernel->compile_impl(part, eng, inputs, outputs);
        }
        return mqa_decomp_status;
    }

    // It is used to check if enable the decomposition kernel based on user's
    // env and params. Decomposition kernel is enabled when:
    // - CPU runtime is OMP or THREADPOOl.
    // - Primitive based implementation is not forced by the internal env var.
    bool enable_decomp_kernel() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP \
        || DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        const int force_prim = graph::utils::getenv_int_internal(
                "GRAPH_SDPA_FORCE_PRIMITIVE", 0);
        return force_prim == 0;
#else
        return false;
#endif
    }

    status_t execute_impl(stream_t *stream, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override {
        return kernel->execute_impl(stream, inputs, outputs, scratchpad_buf);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(stream_t *stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        return kernel->sycl_execute_impl(
                stream, inputs, outputs, scratchpad_buf, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(stream_t *stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf, const std::vector<cl_event> &deps,
            cl_event *event) override {
        return kernel->ocl_execute_impl(
                stream, inputs, outputs, scratchpad_buf, deps, event);
    }
#endif

    std::string str() const override { return kernel->str(); }
    size_t get_scratchpad_size() const override {
        return kernel->get_scratchpad_size();
    }
};
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
