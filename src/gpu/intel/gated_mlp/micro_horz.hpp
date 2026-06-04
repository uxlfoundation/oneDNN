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

#ifndef GPU_INTEL_GATED_MLP_MICRO_HORZ_HPP
#define GPU_INTEL_GATED_MLP_MICRO_HORZ_HPP

#include <vector>
#include "common/c_types_map.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/gemm/utils.hpp"

#include "gemmstone/microkernel/package.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gated_mlp {

struct micro_horz_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gated_mlp_pd_t {
        using gated_mlp_pd_t::gated_mlp_pd_t;

        DECLARE_COMMON_PD_T("ocl:micro_horz:any", micro_horz_t);

        status_t init(impl::engine_t *engine);

        status_t get_gate_dst_md(memory_desc_t &retn) const {
            // inferring the type
            auto down_md = arg_md(DNNL_ARG_WEIGHTS_DOWN);
            data_type_t dt = down_md->data_type;
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
            // now inferring the shape
            std::vector<dim_t> dims {MB(), 1};
            format_tag_t tag = format_tag::abc;
            if (down_md->ndims == 2) {
                dims[1] = OC();
                tag = format_tag::ab;
            } else {
                gpu_assert(down_md->ndims == 3);
                dims.emplace_back(OC());
            }
            CHECK(memory_desc_init_by_tag(
                    retn, int(dims.size()), dims.data(), dt, tag));
            return status::success;
        }

        dnnl_data_type_t get_accum_type() const {
            auto is_float = [](dnnl_data_type_t dt) {
                return utils::one_of(dt, dnnl_f32, dnnl_f16, dnnl_bf16);
            };
            static const std::vector<int> all_idx {
                    DNNL_ARG_SRC, DNNL_ARG_WEIGHTS_UP, DNNL_ARG_WEIGHTS_GATE};
            auto src_dt = arg_md(all_idx[0])->data_type;
            auto wu_dt = arg_md(all_idx[1])->data_type;
            auto wg_dt = arg_md(all_idx[2])->data_type;
            if (is_float(wu_dt) != is_float(wg_dt)) return dnnl_data_type_undef;
            if (is_float(src_dt) || is_float(wu_dt) || is_float(wg_dt))
                return dnnl_f32;
            bool has_scales = false;
            for (auto &i : all_idx)
                has_scales |= !attr()->scales_.has_default_values(i)
                        && (attr()->scales_.get_mask(i) != 0);
            return (has_scales) ? dnnl_f32 : dnnl_s32;
        }

        status_t set_default_formats() {
            CHECK(check_format(arg_md(DNNL_ARG_SRC), dnnl_notrans));
            CHECK(check_format(arg_md(DNNL_ARG_WEIGHTS_GATE), dnnl_trans));
            CHECK(check_format(arg_md(DNNL_ARG_WEIGHTS_UP), dnnl_trans));
            CHECK(check_format(arg_md(DNNL_ARG_WEIGHTS_DOWN), dnnl_trans));
            CHECK(check_format(arg_md(DNNL_ARG_DST), dnnl_notrans));
            return status::success;
        }

        const gemmstone::microkernel::Package &gemm_gate_up_pkg() const {
            return gemm_gate_up_pkg_;
        }

        std::shared_ptr<primitive_desc_t> gemm_down_pd_;

    private:
        status_t move_attr_down(primitive_attr_t &retn, int w_from, int w_to) {
            auto wd_dt = arg_md(w_from)->data_type;
            // Down SRC is always floating-point, but WEI isn't
            CHECK((utils::one_of(wd_dt, dnnl_f32, dnnl_f16, dnnl_bf16))
                            ? retn.set_fpmath_mode(fpmath_mode::strict, false)
                            : retn.set_fpmath_mode(fpmath_mode::any, true));
            // all per-primitive post-ops are for Down, not for Gate/Up
            CHECK(retn.set_post_ops(attr()->post_ops_));
            // quantizations on weights
            CHECK(retn.scales_.set(w_to, attr()->scales_.get(w_from)));
            CHECK(retn.zero_points_.set(
                    w_to, attr()->zero_points_.get(w_from)));
            return status::success;
        }

        status_t check_format(const memory_desc_t *md, transpose_t t) {
            memory_desc_wrapper mdw(*md);
            if (mdw.format_any()) return status::unimplemented;
            if (!is_md_gemm_compatible_plain_format(md))
                return status::unimplemented;
            if (gemm_desc_t::get_trans(*md) != t) return status::unimplemented;
            return status::success;
        }

        status_t init_microkernels(
                impl::engine_t *engine, const memory_desc_t *inter_md);

        gemmstone::microkernel::Package gemm_gate_up_pkg_;
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;

    compute::kernel_t gemm_gate_up_;

    std::shared_ptr<impl::primitive_t> gemm_down_;
};

} // namespace gated_mlp
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
