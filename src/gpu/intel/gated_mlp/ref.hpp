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

#ifndef GPU_INTEL_GATED_MLP_REF_HPP
#define GPU_INTEL_GATED_MLP_REF_HPP

#include "common/c_types_map.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/matmul_pd.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "gpu/gpu_resource.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gated_mlp {

struct ref_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gated_mlp_pd_t {
        using gated_mlp_pd_t::gated_mlp_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_t);

        status_t init(impl::engine_t *engine) {
            VDISPATCH_GATED_MLP(pd_ok(), VERBOSE_INCONSISTENT_PRB);
            VDISPATCH_GATED_MLP(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            memory_desc_t inter_md;
            CHECK(get_inter_md(inter_md));

            primitive_attr_t gate_attr;
            CHECK(move_attr(
                    gate_attr, DNNL_ARG_WEIGHTS_GATE, DNNL_ARG_WEIGHTS));
            CHECK(gate_attr.post_ops_.append_eltwise(
                    1.f, activation(), 1.f, 0.f));
            CHECK(gate_attr.post_ops_.append_binary(
                    alg_kind::binary_mul, &inter_md));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_gate_pd_, engine, gate_attr,
                            arg_md(DNNL_ARG_SRC), arg_md(DNNL_ARG_WEIGHTS_GATE),
                            &inter_md),
                    "internal error in gemm_gate_pd");

            primitive_attr_t up_attr;
            CHECK(move_attr(up_attr, DNNL_ARG_WEIGHTS_UP, DNNL_ARG_WEIGHTS));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_up_pd_, engine, up_attr,
                            arg_md(DNNL_ARG_SRC), arg_md(DNNL_ARG_WEIGHTS_UP),
                            &inter_md),
                    "internal error in gemm_up_pd");

            primitive_attr_t down_attr;
            CHECK(move_attr(
                    down_attr, DNNL_ARG_WEIGHTS_DOWN, DNNL_ARG_WEIGHTS));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_down_pd_, engine, down_attr, &inter_md,
                            arg_md(DNNL_ARG_WEIGHTS_DOWN),
                            arg_md(DNNL_ARG_DST)),
                    "internal error in gemm_down_pd");

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            const memory_desc_wrapper inter_mdw(inter_md);
            scratchpad.book(key_matmul_src_trans, inter_mdw.size(), 1,
                    OCL_BUFFER_ALIGNMENT);
            scratchpad.book(key_matmul_wei_trans, inter_mdw.size(), 1,
                    OCL_BUFFER_ALIGNMENT);
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_GATE,
                    gemm_gate_pd_->scratchpad_registry());
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_UP,
                    gemm_up_pd_->scratchpad_registry());
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_DOWN,
                    gemm_down_pd_->scratchpad_registry());
            return status::success;
        }

        status_t get_inter_md(memory_desc_t &retn) const {
            std::vector<dim_t> dims {MB(), OC()};
            CHECK(memory_desc_init_by_tag(retn, int(dims.size()), dims.data(),
                    data_type::f16, format_tag::ab));
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> gemm_gate_pd_, gemm_up_pd_,
                gemm_down_pd_;

    private:
        status_t move_attr(primitive_attr_t &retn, int from, int to) {
            CHECK(retn.set_fpmath_mode(
                    attr()->fpmath_.mode_, attr()->fpmath_.apply_to_int_));
            CHECK(retn.scales_.set(to, attr()->scales_.get(from)));
            CHECK(retn.zero_points_.set(to, attr()->zero_points_.get(from)));
            return status::success;
        }

        status_t create_matmul(std::shared_ptr<primitive_desc_t> &retn,
                impl::engine_t *e, const primitive_attr_t &attr,
                const memory_desc_t *src_desc, const memory_desc_t *wei_desc,
                const memory_desc_t *dst_desc) {
            auto desc = matmul_desc_t();
            CHECK(impl::matmul_desc_init(
                    &desc, src_desc, wei_desc, nullptr, dst_desc));
            primitive_desc_iterator_t it(e, (op_desc_t *)&desc, &attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            retn = *(++it);
            return (retn) ? status::success : status::unimplemented;
        }
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(create_nested_primitive(gemm_gate_, pd()->gemm_gate_pd_, engine));
        CHECK(create_nested_primitive(gemm_up_, pd()->gemm_up_pd_, engine));
        CHECK(create_nested_primitive(gemm_down_, pd()->gemm_down_pd_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto prep_weights_and_run
                = [&](exec_args_t &args, int idx,
                          const std::shared_ptr<impl::primitive_t> &prim) {
            args[DNNL_ARG_WEIGHTS] = ctx.args().at(idx);
            if (!pd()->attr()->scales_.has_default_values(idx))
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS]
                        = ctx.args().at(DNNL_ARG_ATTR_SCALES | idx);
            if (!pd()->attr()->zero_points_.has_default_values(idx))
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS]
                        = ctx.args().at(DNNL_ARG_ATTR_ZERO_POINTS | idx);
            exec_ctx_t nested_ctx(ctx, std::move(args));
            auto *nested_grantor
                    = create_nested_grantor(ctx.get_scratchpad_grantor(),
                            memory_tracking::names::key_nested_multiple + idx,
                            prim->pd()->scratchpad_registry());
            nested_ctx.set_scratchpad_grantor(nested_grantor);
            CHECK(prim->execute(nested_ctx));
            return status::success;
        };
        memory_desc_t inter_md;
        CHECK(pd()->get_inter_md(inter_md));

        std::unique_ptr<memory_t, memory_deleter_t> inter_src_mem;
        auto inter_src_stor = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_matmul_src_trans);
        CHECK(safe_ptr_assign(inter_src_mem,
                new memory_t(ctx.stream()->engine(), &inter_md,
                        std::move(inter_src_stor))));

        std::unique_ptr<memory_t, memory_deleter_t> inter_wei_mem;
        auto inter_wei_stor = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_matmul_wei_trans);
        CHECK(safe_ptr_assign(inter_wei_mem,
                new memory_t(ctx.stream()->engine(), &inter_md,
                        std::move(inter_wei_stor))));

        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_SRC);
            args[DNNL_ARG_DST] = memory_arg_t {inter_src_mem.get(), false};
            CHECK(prep_weights_and_run(args, DNNL_ARG_WEIGHTS_UP, gemm_up_));
        } while (false);
        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_SRC);
            args[DNNL_ARG_DST] = memory_arg_t {inter_wei_mem.get(), false};
            args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = memory_arg_t {inter_src_mem.get(), true};
            CHECK(prep_weights_and_run(
                    args, DNNL_ARG_WEIGHTS_GATE, gemm_gate_));
        } while (false);
        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = memory_arg_t {inter_wei_mem.get(), true};
            args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
            CHECK(prep_weights_and_run(
                    args, DNNL_ARG_WEIGHTS_DOWN, gemm_down_));
        } while (false);

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> gemm_gate_, gemm_up_, gemm_down_;
};

} // namespace gated_mlp
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
