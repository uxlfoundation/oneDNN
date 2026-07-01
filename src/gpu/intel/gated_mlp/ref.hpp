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
            memory_desc_t gate_dst_md, up_dst_md;
            CHECK(get_gate_dst_md(gate_dst_md));
            CHECK(get_up_dst_md(up_dst_md));
            const auto src_md = arg_md(DNNL_ARG_SRC);

            primitive_attr_t gate_attr;
            CHECK(move_attr(
                    gate_attr, src_md->data_type, DNNL_ARG_WEIGHTS_GATE));
            CHECK(gate_attr.post_ops_.append_eltwise(
                    1.f, activation(), 1.f, 0.f));
            CHECK(gate_attr.post_ops_.append_binary(
                    alg_kind::binary_mul, &up_dst_md));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_gate_pd_, engine, gate_attr, src_md,
                            arg_md(DNNL_ARG_WEIGHTS_GATE), &gate_dst_md),
                    "internal error in gemm_gate_pd");

            primitive_attr_t up_attr;
            CHECK(move_attr(up_attr, src_md->data_type, DNNL_ARG_WEIGHTS_UP));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_up_pd_, engine, up_attr, src_md,
                            arg_md(DNNL_ARG_WEIGHTS_UP), &up_dst_md),
                    "internal error in gemm_up_pd");

            primitive_attr_t down_attr;
            CHECK(move_attr(
                    down_attr, gate_dst_md.data_type, DNNL_ARG_WEIGHTS_DOWN));
            VDISPATCH_GATED_MLP_SC(
                    create_matmul(gemm_down_pd_, engine, down_attr,
                            &gate_dst_md, arg_md(DNNL_ARG_WEIGHTS_DOWN),
                            arg_md(DNNL_ARG_DST)),
                    "internal error in gemm_down_pd");

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_matmul_wei_trans,
                    memory_desc_wrapper(gate_dst_md).size(), 1,
                    OCL_BUFFER_ALIGNMENT);
            scratchpad.book(key_matmul_src_trans,
                    memory_desc_wrapper(up_dst_md).size(), 1,
                    OCL_BUFFER_ALIGNMENT);
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_GATE,
                    gemm_gate_pd_->scratchpad_registry());
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_UP,
                    gemm_up_pd_->scratchpad_registry());
            scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_DOWN,
                    gemm_down_pd_->scratchpad_registry());
            return status::success;
        }

        status_t get_gate_dst_md(memory_desc_t &retn) const {
            data_type_t dt = arg_md(DNNL_ARG_WEIGHTS_DOWN)->data_type;
            switch (dt) {
                case data_type::f32:
                case data_type::bf16:
                case data_type::f16: break;
                case data_type::u4:
                case data_type::s4:
                case data_type::u8:
                case data_type::s8:
                    dt = arg_md(DNNL_ARG_SRC)->data_type;
                    if (!utils::one_of(dt, data_type::bf16, data_type::f16))
                        dt = data_type::f16;
                    break;
                default: return status::unimplemented;
            }
            return get_gate_up_dst_md(retn, dt);
        }

        status_t get_up_dst_md(memory_desc_t &retn) const {
            return get_gate_up_dst_md(retn, data_type::f32);
        }

        std::shared_ptr<primitive_desc_t> gemm_gate_pd_, gemm_up_pd_,
                gemm_down_pd_;

    private:
        status_t get_gate_up_dst_md(memory_desc_t &retn, data_type_t dt) const {
            std::vector<dim_t> dims {MB(), 1};
            format_tag_t tag = format_tag::abc;
            if (arg_md(DNNL_ARG_WEIGHTS_GATE)->ndims == 2) {
                dims[1] = OC();
                tag = format_tag::ab;
            } else {
                gpu_assert(arg_md(DNNL_ARG_WEIGHTS_GATE)->ndims == 3);
                dims.emplace_back(OC());
            }
            CHECK(memory_desc_init_by_tag(
                    retn, int(dims.size()), dims.data(), dt, tag));
            return status::success;
        }

        status_t move_attr(
                primitive_attr_t &retn, data_type_t src_dt, int w_from) const {
            const int w_to = DNNL_ARG_WEIGHTS;
            auto w_dt = arg_md(w_from)->data_type;
            CHECK((utils::one_of(src_dt, dnnl_f32, dnnl_f16, dnnl_bf16)
                          == utils::one_of(w_dt, dnnl_f32, dnnl_f16, dnnl_bf16))
                            ? retn.set_fpmath_mode(fpmath_mode::strict, false)
                            : retn.set_fpmath_mode(fpmath_mode::any, true));
            if (w_from == DNNL_ARG_WEIGHTS_DOWN) { // Down
                // all per-primitive post-ops are for Down, not for Gate/Up
                CHECK(retn.set_post_ops(attr()->post_ops_));
            } else { // Gate or Up
                // Quant on SRC
                CHECK(retn.scales_.set(
                        DNNL_ARG_SRC, attr()->scales_.get(DNNL_ARG_SRC)));
                CHECK(retn.zero_points_.set(
                        DNNL_ARG_SRC, attr()->zero_points_.get(DNNL_ARG_SRC)));
                // Precomp on WEI
                CHECK(retn.precomputed_reductions_.set(
                        w_to, attr()->precomputed_reductions_.get(w_from)));
            }
            // Quant on WEI, same logic for all 3 of the GEMMs
            CHECK(retn.scales_.set(w_to, attr()->scales_.get(w_from)));
            CHECK(retn.zero_points_.set(
                    w_to, attr()->zero_points_.get(w_from)));
            return status::success;
        }

        status_t create_matmul(std::shared_ptr<primitive_desc_t> &retn,
                impl::engine_t *e, const primitive_attr_t &attr,
                const memory_desc_t *src_desc, const memory_desc_t *wei_desc,
                const memory_desc_t *dst_desc) const {
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
        auto prep_quant_and_run
                = [&](exec_args_t &&args, int src, int wei,
                          const std::shared_ptr<impl::primitive_t> &prim) {
            if (!pd()->attr()->scales_.has_default_values(src))
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC]
                        = ctx.args().at(DNNL_ARG_ATTR_SCALES | src);
            if (!pd()->attr()->zero_points_.has_default_values(src))
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC]
                        = ctx.args().at(DNNL_ARG_ATTR_ZERO_POINTS | src);

            args[DNNL_ARG_WEIGHTS] = ctx.args().at(wei);
            if (!pd()->attr()->scales_.has_default_values(wei))
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS]
                        = ctx.args().at(DNNL_ARG_ATTR_SCALES | wei);
            if (!pd()->attr()->zero_points_.has_default_values(wei))
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS]
                        = ctx.args().at(DNNL_ARG_ATTR_ZERO_POINTS | wei);
            if (!pd()->attr()->precomputed_reductions_.has_default_values(wei))
                args[DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_WEIGHTS]
                        = ctx.args().at(
                                DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | wei);

            exec_ctx_t nested_ctx(ctx, std::move(args));
            auto *nested_grantor
                    = create_nested_grantor(ctx.get_scratchpad_grantor(),
                            memory_tracking::names::key_nested_multiple + wei,
                            prim->pd()->scratchpad_registry());
            nested_ctx.set_scratchpad_grantor(nested_grantor);
            CHECK(prim->execute(nested_ctx));
            return status::success;
        };
        memory_desc_t gate_dst_md, up_dst_md;
        CHECK(pd()->get_gate_dst_md(gate_dst_md));
        CHECK(pd()->get_up_dst_md(up_dst_md));

        std::unique_ptr<memory_t, memory_deleter_t> inter_src_mem;
        auto inter_src_stor = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_matmul_src_trans);
        CHECK(safe_ptr_assign(inter_src_mem,
                new memory_t(ctx.stream()->engine(), &up_dst_md,
                        std::move(inter_src_stor))));

        std::unique_ptr<memory_t, memory_deleter_t> inter_wei_mem;
        auto inter_wei_stor = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_matmul_wei_trans);
        CHECK(safe_ptr_assign(inter_wei_mem,
                new memory_t(ctx.stream()->engine(), &gate_dst_md,
                        std::move(inter_wei_stor))));

        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_SRC);
            args[DNNL_ARG_DST] = memory_arg_t {inter_src_mem.get(), false};
            CHECK(prep_quant_and_run(std::move(args), DNNL_ARG_SRC,
                    DNNL_ARG_WEIGHTS_UP, gemm_up_));
        } while (false);
        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_SRC);
            args[DNNL_ARG_DST] = memory_arg_t {inter_wei_mem.get(), false};
            // N.B.: not POST_OP(0), since POST_OP(0) is occupied by activation
            args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = memory_arg_t {inter_src_mem.get(), true};
            CHECK(prep_quant_and_run(std::move(args), DNNL_ARG_SRC,
                    DNNL_ARG_WEIGHTS_GATE, gemm_gate_));
        } while (false);
        do {
            exec_args_t args;
            args[DNNL_ARG_SRC] = memory_arg_t {inter_wei_mem.get(), true};
            args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
            const auto &post_ops = pd()->attr()->post_ops_;
            for (int p = 0, pl = post_ops.len(); p < pl; p++) {
                if (!post_ops.entry_[p].is_like_binary()) continue;
                auto idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(p) | DNNL_ARG_SRC_1;
                args[idx] = ctx.args().at(idx);
            }
            CHECK(prep_quant_and_run(std::move(args), DNNL_ARG_UNDEF,
                    DNNL_ARG_WEIGHTS_DOWN, gemm_down_));
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
