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

#include "common/dnnl_thread.hpp"
#include "cpu/x64/matmul/aocl_dlp_reorder.hpp"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include <aocl_dlp.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

status_t aocl_dlp_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    using namespace data_type;

    // Only support reorder to aocl_dlp_packed format.
    VDISPATCH_REORDER_IC(
            dst_md->format_kind == format_kind::aocl_dlp_packed,
            "dst format must be aocl_dlp_packed");

    // Source must be plain blocked format.
    VDISPATCH_REORDER_IC(
            src_md->format_kind == format_kind::blocked,
            "src format must be blocked");

    // Only 2D or batched matrices supported (ndims >= 2).
    VDISPATCH_REORDER_IC(src_md->ndims >= 2, "ndims must be >= 2");

    // Data types must match, or source can be f32 (for benchdnn fill path).
    VDISPATCH_REORDER_IC(
            src_md->data_type == dst_md->data_type
                    || src_md->data_type == f32,
            "src and dst data types must match or src must be f32");

    // Supported destination data types.
    VDISPATCH_REORDER_IC(
            utils::one_of(dst_md->data_type, f32, bf16, f16, s8),
            "unsupported data type");

    // No post-ops beyond optional sum.
    VDISPATCH_REORDER_IC(attr->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    VDISPATCH_REORDER_IC(
            _pd->init(engine, src_engine, dst_engine) == status::success,
            "pd init failed");

    *reorder_pd = _pd.release();
    return status::success;
}

status_t aocl_dlp_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto src_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());

    const auto src_dt = src_d.data_type();
    const auto dst_dt = dst_d.data_type();
    const int ndims = src_d.ndims();
    const dim_t K = src_d.dims()[ndims - 2];
    const dim_t N = src_d.dims()[ndims - 1];

    // Total batch count (product of all dims except last 2).
    dim_t batch = 1;
    for (int d = 0; d < ndims - 2; ++d)
        batch *= src_d.dims()[d];

    // The GEMM source type determines which AOCL-DLP reorder variant to use.
    const auto gemm_src_dt
            = dst_d.aocl_dlp_packed_desc().gemm_src_dt;
    const size_t per_slice_size
            = dst_d.aocl_dlp_packed_desc().per_slice_size;

    // Determine if source K×N slice is transposed.
    const auto &strides = src_d.blocking_desc().strides;
    const bool is_transposed = strides[ndims - 2] < strides[ndims - 1];

    const char order = 'R';
    const char trans = is_transposed ? 'T' : 'N';
    const dim_t ldb = is_transposed ? strides[ndims - 1] : strides[ndims - 2];

    // Source batch stride in bytes (stride of innermost batch dim).
    const size_t src_batch_bytes = (ndims > 2)
            ? static_cast<size_t>(strides[ndims - 3])
                    * types::data_type_size(src_dt)
            : 0;

    const dim_t nelems = K * N;

    auto src_base = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto dst_base = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    using namespace data_type;

    for (dim_t bi = 0; bi < batch; ++bi) {
        const char *src_slice = src_base + bi * src_batch_bytes;
        char *dst_slice = dst_base + bi * per_slice_size;

        dlp_metadata_t metadata;
        memset(&metadata, 0, sizeof(metadata));
        const char mat_type = 'B';

        // Cross-type path: convert f32 source to destination type, pack.
        if (src_dt == f32 && dst_dt != f32) {
            auto src_f32 = reinterpret_cast<const float *>(src_slice);
            if (dst_dt == bf16) {
                std::vector<bfloat16_t> tmp(nelems);
                for (dim_t k = 0; k < K; k++)
                    for (dim_t n = 0; n < N; n++)
                        tmp[k * N + n] = static_cast<bfloat16_t>(
                                src_f32[k * strides[ndims - 2]
                                        + n * strides[ndims - 1]]);
                aocl_reorder_bf16bf16f32of32(order, 'N', mat_type,
                        reinterpret_cast<const bfloat16 *>(tmp.data()),
                        reinterpret_cast<bfloat16 *>(dst_slice),
                        static_cast<md_t>(K), static_cast<md_t>(N),
                        static_cast<md_t>(N), &metadata);
            } else if (dst_dt == f16) {
                std::vector<float16_t> tmp(nelems);
                for (dim_t k = 0; k < K; k++)
                    for (dim_t n = 0; n < N; n++)
                        tmp[k * N + n] = static_cast<float16_t>(
                                src_f32[k * strides[ndims - 2]
                                        + n * strides[ndims - 1]]);
                aocl_reorder_f16f16f16of16(order, 'N', mat_type,
                        reinterpret_cast<const float16 *>(tmp.data()),
                        reinterpret_cast<float16 *>(dst_slice),
                        static_cast<md_t>(K), static_cast<md_t>(N),
                        static_cast<md_t>(N), &metadata);
            } else if (dst_dt == s8) {
                std::vector<int8_t> tmp(nelems);
                for (dim_t k = 0; k < K; k++)
                    for (dim_t n = 0; n < N; n++) {
                        float v = src_f32[k * strides[ndims - 2]
                                + n * strides[ndims - 1]];
                        v = nstl::max(-128.f, nstl::min(127.f, v));
                        tmp[k * N + n]
                                = static_cast<int8_t>(nearbyintf(v));
                    }
                if (gemm_src_dt == u8) {
                    aocl_reorder_u8s8s32os32(order, 'N', mat_type,
                            tmp.data(),
                            reinterpret_cast<int8_t *>(dst_slice),
                            static_cast<md_t>(K), static_cast<md_t>(N),
                            static_cast<md_t>(N), &metadata);
                } else {
                    aocl_reorder_s8s8s32os32(order, 'N', mat_type,
                            tmp.data(),
                            reinterpret_cast<int8_t *>(dst_slice),
                            static_cast<md_t>(K), static_cast<md_t>(N),
                            static_cast<md_t>(N), &metadata);
                }
            } else {
                return status::unimplemented;
            }
            continue;
        }

        // Same-type path.
        if (dst_dt == f32) {
            aocl_reorder_f32f32f32of32(order, trans, mat_type,
                    reinterpret_cast<const float *>(src_slice),
                    reinterpret_cast<float *>(dst_slice),
                    static_cast<md_t>(K), static_cast<md_t>(N),
                    static_cast<md_t>(ldb), &metadata);
        } else if (dst_dt == bf16) {
            aocl_reorder_bf16bf16f32of32(order, trans, mat_type,
                    reinterpret_cast<const bfloat16 *>(src_slice),
                    reinterpret_cast<bfloat16 *>(dst_slice),
                    static_cast<md_t>(K), static_cast<md_t>(N),
                    static_cast<md_t>(ldb), &metadata);
        } else if (dst_dt == f16) {
            aocl_reorder_f16f16f16of16(order, trans, mat_type,
                    reinterpret_cast<const float16 *>(src_slice),
                    reinterpret_cast<float16 *>(dst_slice),
                    static_cast<md_t>(K), static_cast<md_t>(N),
                    static_cast<md_t>(ldb), &metadata);
        } else if (dst_dt == s8) {
            if (gemm_src_dt == u8) {
                aocl_reorder_u8s8s32os32(order, trans, mat_type,
                        reinterpret_cast<const int8_t *>(src_slice),
                        reinterpret_cast<int8_t *>(dst_slice),
                        static_cast<md_t>(K), static_cast<md_t>(N),
                        static_cast<md_t>(ldb), &metadata);
            } else {
                aocl_reorder_s8s8s32os32(order, trans, mat_type,
                        reinterpret_cast<const int8_t *>(src_slice),
                        reinterpret_cast<int8_t *>(dst_slice),
                        static_cast<md_t>(K), static_cast<md_t>(N),
                        static_cast<md_t>(ldb), &metadata);
            }
        } else {
            return status::unimplemented;
        }
    }

    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
