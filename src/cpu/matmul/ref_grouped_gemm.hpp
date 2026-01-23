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

#ifndef CPU_MATMUL_REF_GROUPED_GEMM_HPP
#define CPU_MATMUL_REF_GROUPED_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_GEMM

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_grouped_gemm_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref_grouped:any", ref_grouped_gemm_t);

        // Weights are 3D: [G, K, N]
        // Override masks to include 0th expert dimension
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }

        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Check for grouped encoding on src and dst
            VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Weights should be dense, abc (K x N) or acb (N x K) format
            VDISPATCH_MATMUL(
                    !wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    wei_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
                    VERBOSE_UNSUPPORTED_TAG);

            // Validate matching number of groups
            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            const auto &dst_grouped = dst_d.sparse_desc().grouped_desc;

            VDISPATCH_MATMUL(src_grouped.ngroups == dst_grouped.ngroups,
                    VERBOSE_INCONSISTENT_DIM, "src_ngroups",
                    (int)src_grouped.ngroups, "dst_ngroups",
                    (int)dst_grouped.ngroups);

            // Supported data types: fp and int8/int4 for src/wei
            const bool is_fp_src = utils::one_of(src_type, f32, bf16, f16);
            const bool is_int_src = utils::one_of(src_type, u8, s8);
            const bool is_int_wei = utils::one_of(wei_type, u8, s8, s4, u4);

            VDISPATCH_MATMUL(is_fp_src || is_int_src, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    utils::one_of(wei_type, f32, bf16, f16, u8, s8, s4, u4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            // No support for weights only quantization as of now, both src/wei should be int
            VDISPATCH_MATMUL(IMPLICATION(is_int_src, is_int_wei),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Check that offsets are int32
            VDISPATCH_MATMUL(src_d.metadata_type(0) == s32
                            && dst_d.metadata_type(0) == s32,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Check for limited Bias support
            if (with_bias()) {
                memory_desc_wrapper bia_d(weights_md(1));
                VDISPATCH_MATMUL(
                        !bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                        VERBOSE_UNSUPPORTED_BIAS_CFG);
                VDISPATCH_MATMUL(
                        bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
                // Bias shape should be [ngroups, N]
                VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.ngroups,
                        VERBOSE_INCONSISTENT_DIM, "bias_dim[0]",
                        (int)bia_d.dims()[0], "ngroups",
                        (int)src_grouped.ngroups);
                VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                        VERBOSE_INCONSISTENT_DIM, "bias_dim[1]",
                        (int)bia_d.dims()[1], "N_dim", (int)wei_d.dims()[2]);
            }

            // Check for supported quantization schemes
            const auto &attr_scales = attr()->scales_;
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
                const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
                const int rowwise_mask = src_qmask_M();
                // Only rowwise f32 scales supported for src
                VDISPATCH_MATMUL(src_mask == rowwise_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(attr_scales.get_data_type(DNNL_ARG_SRC) == f32,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        attr_scales.get(DNNL_ARG_SRC).has_default_groups(),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
                const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
                const int colwise_mask = wei_qmask_N();
                const int blocked_mask = wei_qmask_K() | wei_qmask_N();
                // Allow column-wise or blocked (K grouping) scales for weights
                VDISPATCH_MATMUL(utils::one_of(attr_scales.get_data_type(
                                                       DNNL_ARG_WEIGHTS),
                                         f32, bf16),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        wei_mask == colwise_mask || wei_mask == blocked_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (!attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    VDISPATCH_MATMUL(utils::one_of(wei_type, u8, s8, s4, u4),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gK = attr_scales.get_group(DNNL_ARG_WEIGHTS, 0);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    VDISPATCH_MATMUL(
                            K() % gK == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gN = attr_scales.get_group(DNNL_ARG_WEIGHTS, 1);
                    VDISPATCH_MATMUL(gN == 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
            VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Zero-points are not supported
            VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(),
                    VERBOSE_UNSUPPORTED_ATTR);

            // No post-ops
            VDISPATCH_MATMUL(attr()->post_ops_.has_default_values(),
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }
    };

    ref_grouped_gemm_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_GEMM
#endif // CPU_MATMUL_REF_GROUPED_GEMM_HPP
