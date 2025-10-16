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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_MATMUL_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_MATMUL_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct matmul_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::matmul::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;
    DECLARE_RESET_ENGINE(dnnl::matmul);

    matmul_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const fpmath_t &fpmath,
            bool use_block_layout) {
        using ltw = logical_tensor_wrapper_t;
        // if with zero dimension, the matmul op will take no effect, we
        // construct a dummy kernel
        if (ltw(op->get_input_value(0)->get_logical_tensor()).has_zero_dim()
                || ltw(op->get_input_value(1)->get_logical_tensor())
                           .has_zero_dim()) {
            is_dummy_ = true;
            return;
        }

        auto desc
                = create_desc(op, p_engine, pd_cache, fpmath, use_block_layout);
        prim_ = dnnl::matmul(desc);

        // The scratchpad size of pd created by using any format tag may be
        // different from the scratchpad size of pd created by using queried
        // optimal format tag
        dnnl::memory::desc stored = make_dnnl_memory_desc(
                op->get_output_value(1)->get_logical_tensor());
        dnnl::memory::desc real = desc.scratchpad_desc();
        if (stored != real) {
            auto scratchpad_val = op->get_output_value(1);
            scratchpad_val->set_layout_type(layout_type::any);
            fill_layout_info(scratchpad_val, real);
        }

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (is_dummy_) {
            dummy_impl_.execute(stream, args);
            return;
        }

        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return;
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        if (is_dummy_) { return dummy_impl_.execute_sycl(stream, args, deps); }

        auto sycl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

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
        if (is_dummy_) { return dummy_impl_.execute_ocl(stream, args, deps); }

        auto ocl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

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
    dnnl::matmul prim_;
    bool with_sum_ {false};
    bool is_dummy_ {false};
    dummy_impl_t dummy_impl_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_MATMUL_HPP
