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
#include "common/utils.hpp"

#include "cpu/matmul/matmul_utils.hpp"

#include <aocl_dlp.h>
#include <vector>

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
    // Batched f16 not supported (no batch GEMM API for f16).
    VDISPATCH_MATMUL(IMPLICATION(is_f16, src_md()->ndims == 2),
            VERBOSE_BAD_NDIMS, "src", src_md()->ndims);
    // No batch broadcasting support: all batch dims must match across
    // src, weights, and dst.
    if (batched()) {
        for (int d = 0; d < ndims() - 2; ++d) {
            VDISPATCH_MATMUL(src_md()->dims[d] == dst_md()->dims[d]
                            && weights_md()->dims[d] == dst_md()->dims[d],
                    "batch broadcasting unsupported");
        }
    }
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    return status::success;
}

status_t aocl_dlp_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto wei_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const auto src_type = src_d.data_type();
    const auto dst_type = dst_d.data_type();

    using namespace dnnl::impl::cpu::matmul;
    matmul_helper_t helper(src_d, wei_d, dst_d);

    const dim_t M = helper.M();
    const dim_t K = helper.K();
    const dim_t N = helper.N();
    const dim_t batch = helper.batch();

    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();

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

    // Non-batched path (2D) or f16.
    if (batch == 1) {
        if (src_type == f32) {
            auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
            auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
            auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            aocl_gemm_f32f32f32of32(order, transa, transb,
                    static_cast<md_t>(M), static_cast<md_t>(N),
                    static_cast<md_t>(K), alpha, src,
                    static_cast<md_t>(lda), mem_fmt_a, wei,
                    static_cast<md_t>(ldb), mem_fmt_b, beta, dst,
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
                    reinterpret_cast<bfloat16 *>(dst),
                    static_cast<md_t>(ldc), &metadata);
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
                    reinterpret_cast<float16 *>(dst),
                    static_cast<md_t>(ldc), &metadata);
        } else {
            return status::unimplemented;
        }
        return status::success;
    }

    // Batched path: use aocl_batch_gemm_* APIs.
    // Since we require dense format and no broadcasting, the offset for
    // flattened batch index b is simply b * stride[ndims-3], where ndims-3
    // is the innermost batch dimension.
    const int ndims = src_d.ndims();
    const int batch_dim = ndims - 3;
    const dim_t src_batch_stride = src_d.blocking_desc().strides[batch_dim];
    const dim_t wei_batch_stride = wei_d.blocking_desc().strides[batch_dim];
    const dim_t dst_batch_stride = dst_d.blocking_desc().strides[batch_dim];

    // Build per-batch parameter arrays for the group.
    // All GEMMs in the batch share the same M, N, K, lda, ldb, ldc.
    std::vector<md_t> m_arr(batch, static_cast<md_t>(M));
    std::vector<md_t> n_arr(batch, static_cast<md_t>(N));
    std::vector<md_t> k_arr(batch, static_cast<md_t>(K));
    std::vector<md_t> lda_arr(batch, static_cast<md_t>(lda));
    std::vector<md_t> ldb_arr(batch, static_cast<md_t>(ldb));
    std::vector<md_t> ldc_arr(batch, static_cast<md_t>(ldc));
    std::vector<float> alpha_arr(batch, 1.0f);
    std::vector<float> beta_arr(batch, 0.0f);
    std::vector<char> order_arr(batch, order);
    std::vector<char> transa_arr(batch, transa);
    std::vector<char> transb_arr(batch, transb);
    std::vector<char> mem_fmt_a_arr(batch, mem_fmt_a);
    std::vector<char> mem_fmt_b_arr(batch, mem_fmt_b);
    std::vector<dlp_metadata_t *> meta_arr(batch, &metadata);

    const md_t group_count = 1;
    const md_t group_size = static_cast<md_t>(batch);

    if (src_type == f32) {
        auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        std::vector<const float *> a_ptrs(batch);
        std::vector<const float *> b_ptrs(batch);
        std::vector<float *> c_ptrs(batch);
        for (dim_t b = 0; b < batch; ++b) {
            a_ptrs[b] = src + b * src_batch_stride;
            b_ptrs[b] = wei + b * wei_batch_stride;
            c_ptrs[b] = dst + b * dst_batch_stride;
        }

        aocl_batch_gemm_f32f32f32of32(order_arr.data(), transa_arr.data(),
                transb_arr.data(), m_arr.data(), n_arr.data(), k_arr.data(),
                alpha_arr.data(), a_ptrs.data(), lda_arr.data(),
                b_ptrs.data(), ldb_arr.data(), beta_arr.data(),
                c_ptrs.data(), ldc_arr.data(), group_count, &group_size,
                mem_fmt_a_arr.data(), mem_fmt_b_arr.data(),
                meta_arr.data());
    } else if (src_type == bf16 && dst_type == f32) {
        auto src = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        std::vector<const bfloat16 *> a_ptrs(batch);
        std::vector<const bfloat16 *> b_ptrs(batch);
        std::vector<float *> c_ptrs(batch);
        for (dim_t b = 0; b < batch; ++b) {
            a_ptrs[b] = reinterpret_cast<const bfloat16 *>(
                    src + b * src_batch_stride);
            b_ptrs[b] = reinterpret_cast<const bfloat16 *>(
                    wei + b * wei_batch_stride);
            c_ptrs[b] = dst + b * dst_batch_stride;
        }

        aocl_batch_gemm_bf16bf16f32of32(order_arr.data(), transa_arr.data(),
                transb_arr.data(), m_arr.data(), n_arr.data(), k_arr.data(),
                alpha_arr.data(), a_ptrs.data(), lda_arr.data(),
                b_ptrs.data(), ldb_arr.data(), beta_arr.data(),
                c_ptrs.data(), ldc_arr.data(), group_count, &group_size,
                mem_fmt_a_arr.data(), mem_fmt_b_arr.data(),
                meta_arr.data());
    } else if (src_type == bf16 && dst_type == bf16) {
        auto src = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DST);

        std::vector<const bfloat16 *> a_ptrs(batch);
        std::vector<const bfloat16 *> b_ptrs(batch);
        std::vector<bfloat16 *> c_ptrs(batch);
        for (dim_t b = 0; b < batch; ++b) {
            a_ptrs[b] = reinterpret_cast<const bfloat16 *>(
                    src + b * src_batch_stride);
            b_ptrs[b] = reinterpret_cast<const bfloat16 *>(
                    wei + b * wei_batch_stride);
            c_ptrs[b] = reinterpret_cast<bfloat16 *>(
                    dst + b * dst_batch_stride);
        }

        aocl_batch_gemm_bf16bf16f32obf16(order_arr.data(), transa_arr.data(),
                transb_arr.data(), m_arr.data(), n_arr.data(), k_arr.data(),
                alpha_arr.data(), a_ptrs.data(), lda_arr.data(),
                b_ptrs.data(), ldb_arr.data(), beta_arr.data(),
                c_ptrs.data(), ldc_arr.data(), group_count, &group_size,
                mem_fmt_a_arr.data(), mem_fmt_b_arr.data(),
                meta_arr.data());
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
