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

#ifndef CPU_RV64_RVV_GEMM_INNER_PRODUCT_HPP
#define CPU_RV64_RVV_GEMM_INNER_PRODUCT_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_gemm_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV:gemm", rvv_gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;

            VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const bool types_ok = src_type == f32 && wei_type == f32
                    && dst_type == f32
                    && IMPLICATION(with_bias(), bia_type == f32);
            VDISPATCH_INNER_PRODUCT(types_ok, VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(smask_t::none),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_INNER_PRODUCT(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper wei_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            VDISPATCH_INNER_PRODUCT(check_layouts(src_d, wei_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);

            nthr_ = dnnl_get_max_threads();

            return status::success;
        }

        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &wei_d,
                const memory_desc_wrapper &dst_d) const {
            // Only support 2D tensors to ensure simple GEMM mapping
            if (src_d.ndims() != 2 || wei_d.ndims() != 2 || dst_d.ndims() != 2)
                return false;

            const bool plain_dense = src_d.blocking_desc().inner_nblks == 0
                    && wei_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(/*with_padding=*/false)
                    && wei_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false);
            if (!plain_dense) return false;

            // GEMM requires Src and Dst to be Row Major (last stride 1)
            if (src_d.blocking_desc().strides[src_d.ndims() - 1] != 1)
                return false;
            if (dst_d.blocking_desc().strides[dst_d.ndims() - 1] != 1)
                return false;

            // For Weights, we can support both Row Major (ab) and Col Major (ba)
            const auto &w_strides = wei_d.blocking_desc().strides;
            bool is_w_row_major = w_strides[wei_d.ndims() - 1] == 1;
            bool is_w_col_major = w_strides[0] == 1;

            if (!is_w_row_major && !is_w_col_major) return false;

            return true;
        }

        int nthr_;
    };

    rvv_gemm_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_GEMM_INNER_PRODUCT_HPP
