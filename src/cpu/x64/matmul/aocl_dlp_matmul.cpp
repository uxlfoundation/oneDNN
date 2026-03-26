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

#include "cpu/x64/matmul/aocl_dlp_matmul.hpp"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/matmul_utils.hpp"

#include <aocl_dlp.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace data_type;

status_t aocl_dlp_matmul_t::pd_t::init(engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;
    const auto src_type = src_md()->data_type;
    const auto wei_type = weights_md()->data_type;
    const auto dst_type = dst_md()->data_type;

    // AOCL-DLP supports: f32f32f32of32, bf16bf16f32of32, bf16bf16f32obf16,
    // f16f16f16of16
    const bool is_f32 = utils::everyone_is(f32, src_type, wei_type, dst_type);
    const bool is_bf16f32 = utils::everyone_is(bf16, src_type, wei_type)
            && dst_type == f32;
    const bool is_bf16bf16 = utils::everyone_is(bf16, src_type, wei_type)
            && dst_type == bf16;
    const bool is_f16 = utils::everyone_is(f16, src_type, wei_type, dst_type)
            && platform::has_data_type_support(f16);

    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(utils::one_of(true, is_f32, is_bf16f32, is_bf16bf16,
                             is_f16),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    // Only support bias of f32 type (or no bias).
    VDISPATCH_MATMUL(
            IMPLICATION(with_bias(), weights_md(1)->data_type == f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_MATMUL(attr()->has_default_values(smask_t::post_ops
                                     | smask_t::scales_data_type
                                     | smask_t::sum_dt,
                             dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(attr()->post_ops_.check_sum_consistency(dst_type,
                             /* is_int8 */ false),
            VERBOSE_UNSUPPORTED_POSTOP);
    // Only 2D matmul supported (no batch dimension).
    VDISPATCH_MATMUL(src_md()->ndims == 2, VERBOSE_BAD_NDIMS, "src",
            src_md()->ndims);
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    return status::success;
}

status_t aocl_dlp_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto wei_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const auto src_type = src_d.data_type();
    const auto dst_type = dst_d.data_type();

    const dim_t M = src_d.dims()[0];
    const dim_t K = src_d.dims()[1];
    const dim_t N = wei_d.dims()[1];

    const dim_t lda = src_d.blocking_desc().strides[0];
    const dim_t ldb = wei_d.blocking_desc().strides[0];
    const dim_t ldc = dst_d.blocking_desc().strides[0];

    const char order = 'R'; // row-major
    const char transa = 'N';
    const char transb = 'N';
    const char mem_fmt_a = 'N'; // unpacked
    const char mem_fmt_b = 'N'; // unpacked

    // Set up post-ops metadata.
    dlp_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));

    // Bias post-op.
    dlp_post_op_bias bias_op;
    memset(&bias_op, 0, sizeof(bias_op));
    DLP_POST_OP_TYPE seq[AOCL_DLP_MAX_POST_OPS];
    md_t seq_len = 0;

    if (pd()->with_bias()) {
        auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
        bias_op.bias = const_cast<float *>(bias);
        bias_op.stor_type = DLP_F32;
        metadata.bias = &bias_op;
        seq[seq_len++] = BIAS;
    }

    // Handle sum/eltwise post-ops.
    dlp_post_op_eltwise eltwise_op;
    memset(&eltwise_op, 0, sizeof(eltwise_op));
    const auto &post_ops = pd()->attr()->post_ops_;
    for (int i = 0; i < post_ops.len(); ++i) {
        const auto &entry = post_ops.entry_[i];
        if (entry.is_eltwise()) {
            DLP_ELT_ALGO_TYPE algo;
            switch (entry.eltwise.alg) {
                case alg_kind::eltwise_relu: algo = RELU; break;
                case alg_kind::eltwise_gelu_tanh: algo = GELU_TANH; break;
                case alg_kind::eltwise_gelu_erf: algo = GELU_ERF; break;
                case alg_kind::eltwise_swish: algo = SWISH; break;
                case alg_kind::eltwise_tanh: algo = TANH; break;
                case alg_kind::eltwise_logistic: algo = SIGMOID; break;
                case alg_kind::eltwise_clip:
                case alg_kind::eltwise_clip_v2: algo = CLIP; break;
                default: return status::unimplemented;
            }
            float alpha_val = entry.eltwise.alpha;
            float beta_val = entry.eltwise.beta;
            eltwise_op.algo.alpha = &alpha_val;
            eltwise_op.algo.beta = &beta_val;
            eltwise_op.algo.algo_type = algo;
            eltwise_op.algo.stor_type = DLP_F32;
            metadata.eltwise = &eltwise_op;
            seq[seq_len++] = ELTWISE;
        }
    }

    if (seq_len > 0) {
        metadata.seq_length = seq_len;
        metadata.seq_vector = seq;
    }

    if (src_type == f32) {
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        aocl_gemm_f32f32f32of32(order, transa, transb,
                static_cast<md_t>(M), static_cast<md_t>(N),
                static_cast<md_t>(K), alpha, src, static_cast<md_t>(lda),
                mem_fmt_a, wei, static_cast<md_t>(ldb), mem_fmt_b, beta, dst,
                static_cast<md_t>(ldc), &metadata);
    } else if (src_type == bf16 && dst_type == f32) {
        auto src = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        aocl_gemm_bf16bf16f32of32(order, transa, transb,
                static_cast<md_t>(M), static_cast<md_t>(N),
                static_cast<md_t>(K), alpha,
                reinterpret_cast<const bfloat16 *>(src),
                static_cast<md_t>(lda), mem_fmt_a,
                reinterpret_cast<const bfloat16 *>(wei),
                static_cast<md_t>(ldb), mem_fmt_b, beta, dst,
                static_cast<md_t>(ldc), &metadata);
    } else if (src_type == bf16 && dst_type == bf16) {
        auto src = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DST);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        aocl_gemm_bf16bf16f32obf16(order, transa, transb,
                static_cast<md_t>(M), static_cast<md_t>(N),
                static_cast<md_t>(K), alpha,
                reinterpret_cast<const bfloat16 *>(src),
                static_cast<md_t>(lda), mem_fmt_a,
                reinterpret_cast<const bfloat16 *>(wei),
                static_cast<md_t>(ldb), mem_fmt_b, beta,
                reinterpret_cast<bfloat16 *>(dst), static_cast<md_t>(ldc),
                &metadata);
    } else if (src_type == f16) {
        auto src = CTX_IN_MEM(const float16_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const float16_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float16_t *, DNNL_ARG_DST);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        aocl_gemm_f16f16f16of16(order, transa, transb,
                static_cast<md_t>(M), static_cast<md_t>(N),
                static_cast<md_t>(K), alpha,
                reinterpret_cast<const float16 *>(src),
                static_cast<md_t>(lda), mem_fmt_a,
                reinterpret_cast<const float16 *>(wei),
                static_cast<md_t>(ldb), mem_fmt_b, beta,
                reinterpret_cast<float16 *>(dst), static_cast<md_t>(ldc),
                &metadata);
    } else {
        return status::unimplemented;
    }

    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
