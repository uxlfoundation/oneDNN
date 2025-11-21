/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
#ifndef CPU_RV64_RVV_MATMUL_HPP
#define CPU_RV64_RVV_MATMUL_HPP

#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

struct rvv_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV", rvv_matmul_t)

        static constexpr data_type_t d_type = data_type::f32;

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using smask_t = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_mdw(src_md(0));
            const memory_desc_wrapper weights_mdw(weights_md(0));
            const memory_desc_wrapper dst_mdw(dst_md(0));
            const memory_desc_wrapper bias_mdw = bias_md_;

            VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                            && !weights_mdw.has_runtime_dims_or_strides()
                            && !dst_mdw.has_runtime_dims_or_strides()
                            && !bias_mdw.has_runtime_dims_or_strides(),
                    VERBOSE_UNSUPPORTED_TAG);

            const bool types_ok = src_mdw.data_type() == d_type
                    && weights_mdw.data_type() == d_type
                    && dst_mdw.data_type() == d_type
                    && desc()->accum_data_type == d_type;
            VDISPATCH_MATMUL(types_ok, VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::post_ops, d_type),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_MATMUL(rvv_postops_t::post_ops_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_MATMUL(check_layouts(src_mdw, weights_mdw, dst_mdw),
                    VERBOSE_UNSUPPORTED_TAG);

            {
                const auto wei_ndims = weights_mdw.ndims();
                bool bc_ok = true;
                for (int i = 0; i < wei_ndims - 2; ++i) {
                    if (src_mdw.dims()[i] != weights_mdw.dims()[i]
                            && weights_mdw.dims()[i] != 1) {
                        bc_ok = false;
                        break;
                    }
                }
                VDISPATCH_MATMUL(bc_ok, VERBOSE_UNSUPPORTED_TAG);
            }

            VDISPATCH_MATMUL(check_bias(dst_mdw, bias_mdw),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            init_gemm_conf(src_mdw, weights_mdw);

            return status::success;
        }

        bool is_row_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            if (strides[ndims - 1] != 1) return false;

            dim_t expected_stride = mdw.dims()[ndims - 1];
            for (int d = ndims - 2; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= mdw.dims()[d];
            }
            return true;
        }

        bool is_col_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            const auto &dims = mdw.dims();

            if (strides[ndims - 2] != 1) return false;
            if (strides[ndims - 1] != dims[ndims - 2]) return false;

            dim_t expected_stride = dims[ndims - 2] * dims[ndims - 1];
            for (int d = ndims - 3; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= dims[d];
            }
            return true;
        }

        bool check_layouts(const memory_desc_wrapper &src_mdw,
                const memory_desc_wrapper &wei_mdw,
                const memory_desc_wrapper &dst_mdw) const {
            if (!is_row_major(src_mdw) || !is_row_major(dst_mdw)) return false;
            if (!is_row_major(wei_mdw) && !is_col_major(wei_mdw)) return false;
            return true;
        }

        bool check_bias(const memory_desc_wrapper &dst_mdw,
                const memory_desc_wrapper &bias_mdw) const {
            if (bias_mdw.is_zero()) return true;

            if (bias_mdw.data_type() != d_type) return false;

            const int dst_ndims = dst_mdw.ndims();
            const int bias_ndims = bias_mdw.ndims();
            if (bias_ndims > dst_ndims) return false;

            const auto *dst_dims = dst_mdw.dims();
            const auto *bias_dims = bias_mdw.dims();

            for (int d = 1; d <= bias_ndims; ++d) {
                const dim_t bias_dim = bias_dims[bias_ndims - d];
                const dim_t dst_dim = dst_dims[dst_ndims - d];
                if (bias_dim != 1 && bias_dim != dst_dim) return false;
            }
            return true;
        }

        void init_gemm_conf(const memory_desc_wrapper &src_mdw,
                const memory_desc_wrapper &weights_mdw) {
            const int ndims = src_mdw.ndims();
            const int wei_ndims = weights_mdw.ndims();

            const dim_t *src_dims = src_mdw.dims();
            const dim_t *wei_dims = weights_mdw.dims();

            batch_ = 1;
            for (int i = 0; i < ndims - 2; ++i)
                batch_ *= src_dims[i];

            M_ = src_dims[ndims - 2];
            K_ = src_dims[ndims - 1];
            N_ = wei_dims[wei_ndims - 1];

            dim_t weights_batch_size = 1;
            for (int i = 0; i < wei_ndims - 2; ++i)
                weights_batch_size *= wei_dims[i];
            weights_are_broadcast_ = (weights_batch_size == 1 && batch_ > 1);
        }

        dim_t M_ = 0;
        dim_t N_ = 0;
        dim_t K_ = 0;
        dim_t batch_ = 0;
        bool weights_are_broadcast_ = false;
    };

    rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_MATMUL_HPP
