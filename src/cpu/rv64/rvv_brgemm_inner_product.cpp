/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/rvv_brgemm_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace format_tag;

status_t rvv_brgemm_inner_product_fwd_t::pd_t::init(engine_t *engine) {
    VDISPATCH_INNER_PRODUCT(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

    VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;
    const auto bia_type = weights_md(1)->data_type;
    const bool types_ok = src_type == f32 && wei_type == f32 && dst_type == f32
            && IMPLICATION(with_bias(), bia_type == f32);
    VDISPATCH_INNER_PRODUCT(types_ok, VERBOSE_UNSUPPORTED_DT);

    // No post-ops supported
    VDISPATCH_INNER_PRODUCT(
            attr()->has_default_values(primitive_attr_t::skip_mask_t::none),
            VERBOSE_UNSUPPORTED_ATTR);

    // Only support 2D tensors
    VDISPATCH_INNER_PRODUCT(src_md(0)->ndims == 2, VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_INNER_PRODUCT(weights_md(0)->ndims == 2, VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_INNER_PRODUCT(dst_md(0)->ndims == 2, VERBOSE_UNSUPPORTED_TAG);

    // Set default formats
    // brgemm requires weights in ba format (OC-contiguous)
    if (src_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md_, format_tag::ab));
    if (weights_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_md_, format_tag::ba));
    if (dst_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md_, format_tag::ab));
    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    VDISPATCH_INNER_PRODUCT(
            attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_TAG);

    // Shape guards — fallback to gemm_ip when brgemm is not beneficial
    VDISPATCH_INNER_PRODUCT(
            OC() >= 16, VERBOSE_IMPL_HEURISTIC_FAIL, "OC too small for brgemm");
    VDISPATCH_INNER_PRODUCT(IC_total_padded() >= 64,
            VERBOSE_IMPL_HEURISTIC_FAIL, "IC too small for brgemm");
    // The brgemm kernel accesses weights (ba format) with stride OC * 4 bytes
    // between K rows. Two complementary guards prevent regressions:
    //
    // Guard A: When MB < 4, the JIT kernel's N-loop falls entirely into the
    // single-column tail (no pipelining). Combined with L1D cache set
    // conflicts (stride is a multiple of way_size when OC % 1024 == 0 for
    // typical 32KB 8-way L1D), this causes severe regression vs gemm_ip.
    VDISPATCH_INNER_PRODUCT(MB() >= 4 || OC() % 1024 != 0,
            VERBOSE_IMPL_HEURISTIC_FAIL,
            "small MB with cache-conflicting OC stride");
    // Guard B: When MB > 4, each additional 4-column group re-loads the
    // entire A matrix from L2+. With OC >= 2048 the stride (>= 8KB) spans
    // multiple pages, causing severe TLB thrashing that compounds with the
    // repeated loads. The gemm driver avoids this via copy_A packing.
    VDISPATCH_INNER_PRODUCT(MB() <= 4 || OC() < 2048,
            VERBOSE_IMPL_HEURISTIC_FAIL,
            "large MB with large OC causes TLB thrashing");

    const memory_desc_wrapper src_d(src_md(0));
    const memory_desc_wrapper wei_d(weights_md(0));
    const memory_desc_wrapper dst_d(dst_md(0));

    // All must be plain dense
    VDISPATCH_INNER_PRODUCT(src_d.blocking_desc().inner_nblks == 0
                    && wei_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(false) && wei_d.is_dense(false)
                    && dst_d.is_dense(false),
            VERBOSE_UNSUPPORTED_TAG);

    // Src and Dst must be row-major (last stride == 1)
    VDISPATCH_INNER_PRODUCT(
            src_d.blocking_desc().strides[src_d.ndims() - 1] == 1,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_INNER_PRODUCT(
            dst_d.blocking_desc().strides[dst_d.ndims() - 1] == 1,
            VERBOSE_UNSUPPORTED_TAG);

    // Weights must be ba format
    VDISPATCH_INNER_PRODUCT(
            wei_d.blocking_desc().strides[0] == 1, VERBOSE_UNSUPPORTED_TAG);

    // Create brgemm descriptor and JIT kernel
    const dim_t M = OC();
    const dim_t K = IC_total_padded();
    const dim_t LDA = M; // weights ba: stride[1] = OC
    const dim_t LDB = K; // src row-major: stride[0] = IC
    const dim_t LDC = M; // dst row-major: stride[0] = OC

    brgemm_desc_t brg_desc;
    CHECK(brgemm_desc_init(&brg_desc, v, brgemm_strd, data_type::f32,
            data_type::f32, brgemm_col_major, 1.0f, 0.0f, LDA, LDB, LDC, M,
            MB(), K));

    brgemm_kernel_t *kernel = nullptr;
    CHECK(brgemm_kernel_create(&kernel, brg_desc));
    brg_kernel_.reset(kernel);

    return status::success;
}

status_t rvv_brgemm_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bia = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t K = pd()->IC_total_padded();

    const auto *brg_kernel = pd()->brg_kernel_.get();
    const auto &brg = brg_kernel->get_brg();
    const int bd = brg.bd_block;
    const int total_m_tiles = brg.bdb + (brg.bdb_tail > 0 ? 1 : 0);

    const int nthr = dnnl_get_max_threads();

    if (MB >= nthr) {
        // Sufficient MB rows: parallelize along N (MB) dimension only.
        // K-outer M-inner loop order in brgemm_kernel_execute gives
        // best B data reuse in L1D cache across M tiles.
        parallel(0, [&](int ithr, int nthr_actual) {
            dim_t n_start {0}, n_end {0};
            balance211(MB, nthr_actual, ithr, n_start, n_end);
            const dim_t n_work = n_end - n_start;
            if (n_work <= 0) return;

            brgemm_kernel_execute(brg_kernel, wei, src + n_start * K,
                    dst + n_start * OC, n_work, 0.0f);

            if (bia) {
                for (dim_t n = 0; n < n_work; n++) {
                    float *d = dst + (n_start + n) * OC;
                    for (dim_t oc = 0; oc < OC; oc++)
                        d[oc] += bia[oc];
                }
            }
        });
    } else {
        // MB < nthr: not enough rows for 1D parallelism.
        // Distribute work across M (OC) tiles so all cores are utilized.
        // Each thread processes ALL MB rows for its assigned M tile range.
        const dim_t BK = BRGEMM_BK;

        parallel(0, [&](int ithr, int nthr_actual) {
            dim_t mt_start {0}, mt_end {0};
            balance211(
                    (dim_t)total_m_tiles, nthr_actual, ithr, mt_start, mt_end);

            for (dim_t mt = mt_start; mt < mt_end; mt++) {
                const int m_size = (mt < brg.bdb) ? bd : brg.bdb_tail;
                const dim_t m_offset = mt * bd;

                for (dim_t kb = 0; kb < K; kb += BK) {
                    const dim_t K_inner = nstl::min(BK, K - kb);
                    const float beta_kb = (kb == 0) ? 0.0f : 1.0f;

                    brgemm_kernel_params_t p;
                    p.ptr_A = wei + kb * OC + m_offset;
                    p.ptr_B = src + kb;
                    p.ptr_C = dst + m_offset;
                    p.N = MB;
                    p.M = m_size;
                    p.K = K_inner;
                    p.beta = beta_kb;
                    (*brg_kernel)(&p);
                }

                if (bia) {
                    for (dim_t n = 0; n < MB; n++) {
                        float *d = dst + n * OC + m_offset;
                        for (int oc = 0; oc < m_size; oc++)
                            d[oc] += bia[m_offset + oc];
                    }
                }
            }
        });
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
