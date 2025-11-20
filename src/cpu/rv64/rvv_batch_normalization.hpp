/******************************************************************************
* Copyright 2025 ZTE Corporation
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
******************************************************************************/

#ifndef CPU_RV64_RVV_BATCH_NORMALIZATION_HPP
#define CPU_RV64_RVV_BATCH_NORMALIZATION_HPP

#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_batch_normalization_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::
                cpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);

            using namespace data_type;

            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);

            const data_type_t dtsrc = src_md()->data_type;
            const data_type_t dtdst = dst_md()->data_type;
            bool types_ok = (dtsrc == f32 && dtdst == f32)
                    && platform::has_data_type_support(dtsrc)
                    && IMPLICATION(is_training(),
                            platform::has_training_support(dtsrc));
            VDISPATCH_BNORM(types_ok, VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            // Require global stats (G). Flags C/H/R(inference) are optional. Disallow none and A.
            VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "fuse_norm_add_relu not supported");
            VDISPATCH_BNORM(use_global_stats(), VERBOSE_UNSUPPORTED_FEATURE,
                    "stats must already have been computed (use global stats)");
            using smask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_BNORM(!(fuse_norm_relu()
                                    && desc()->prop_kind
                                            == prop_kind::forward_training),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "forward training with fused ReLU is not supported");
            // Only support eltwise ReLU without alpha/beta post-op as current rvv_postops requires.
            VDISPATCH_BNORM(attr()->has_default_values(smask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            {
                const post_ops_t &po = attr()->post_ops_;
                bool relu_no_params_ok = true;
                if (po.len() == 1) {
                    const auto &e = po.entry_[0];
                    relu_no_params_ok = e.is_eltwise()
                            && e.eltwise.alg == alg_kind::eltwise_relu
                            && e.eltwise.alpha == 0.f && e.eltwise.beta == 0.f;
                } else if (po.len() > 1) {
                    relu_no_params_ok = false;
                }
                VDISPATCH_BNORM(relu_no_params_ok, VERBOSE_UNSUPPORTED_ATTR);
            }
            VDISPATCH_BNORM(rv64::rvv_postops_t::post_ops_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_ATTR);

            // Simplest memory layouts only: plain, dense, same layout src/dst, no blocking/padding.
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            VDISPATCH_BNORM(
                    check_layouts(src_d, dst_d), VERBOSE_UNSUPPORTED_TAG);

            fused_relu_in_kernel_ = fuse_norm_relu();
            init_scratchpad();

            return status::success;
        }
        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &dst_d) const {
            // Require plain, dense, no blocking/padding, same plain layout.
            bool ndims_ok = utils::one_of(ndims(), 3, 4, 5);
            bool plain_dense = src_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false);
            bool same_layouts = src_d.similar_to(dst_d, /*with_strides=*/true,
                    /*with_pads=*/false);
            return ndims_ok && plain_dense && same_layouts;
        }

        bool fused_relu_in_kernel() const { return fused_relu_in_kernel_; }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            // Reserve per-channel temporary buffers for axb (channels-dense) path
            scratchpad.template book<float>(key_bnorm_tmp_mean, C());
            scratchpad.template book<float>(key_bnorm_tmp_var, C());
        }
        bool fused_relu_in_kernel_ = false;
    };

    rvv_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_BATCH_NORMALIZATION_HPP
