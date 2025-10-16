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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_REDUCTION_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_REDUCTION_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct reduction_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::reduction::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::reduction);

    reduction_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::reduction(desc);

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }

        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }

        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::reduction prim_;
    bool with_sum_ {false};
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_REDUCTION_HPP
