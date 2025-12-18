/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_MEMORY_REPARSER_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_MEMORY_REPARSER_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct memory_reparser_t : public dummy_impl_t {
    DECLARE_ARG_INDICES_GETTER;

    memory_reparser_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const fpmath_t &fpmath,
            bool use_block_layout) {
        UNUSED(op);
        UNUSED(p_engine);
        UNUSED(pd_cache);
        UNUSED(fpmath);
        UNUSED(use_block_layout);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override;

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override;
#endif
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_MEMORY_REPARSER_HPP
