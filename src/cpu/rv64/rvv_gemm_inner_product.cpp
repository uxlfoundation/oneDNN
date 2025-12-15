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

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/gemm/rvv_gemm_f32.hpp"
#include "cpu/rv64/rvv_gemm_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t rvv_gemm_inner_product_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    void *dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t K = pd()->IC_total_padded();
    const float *src_f32 = reinterpret_cast<const float *>(src);
    const float *wei_f32 = reinterpret_cast<const float *>(weights);
    float *dst_f32 = reinterpret_cast<float *>(dst);

    char transa = 'T';
    char transb = 'N';
    dim_t M = OC;
    dim_t N = MB;
    dim_t Kdim = K;
    dim_t lda = K;
    dim_t ldb = K;
    dim_t ldc = OC;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Check if weights are Row Major (last stride == 1) or Col Major
    if (wei_d.blocking_desc().strides[wei_d.ndims() - 1] != 1) {
        // Weights are Column Major (ba)
        // Logical: [OC, K]
        // Physical: [OC, K] (Col Major)
        // GEMM View (Col Major): A is [OC, K]
        // Target: C = Wei * Src^T = A * B
        transa = 'N';
        lda = OC;
    }

    status = rvv_gemm_f32(&transa, &transb, &M, &N, &Kdim, &alpha, wei_f32,
            &lda, src_f32, &ldb, &beta, dst_f32, &ldc, /*bias=*/nullptr);
    if (status != status::success) return status;

    if (bias) {
        const float *bias_f32 = reinterpret_cast<const float *>(bias);
        parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
            const dim_t off = dst_d.off(mb, oc);
            dst_f32[off] += bias_f32[oc];
        });
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
