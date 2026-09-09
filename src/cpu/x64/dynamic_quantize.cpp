/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#if !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/dynamic_quantize.hpp"
#include "cpu/x64/dynamic_quantize_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace dq = dynamic_quantize_kernels;

status_t dynamic_quantize_reduction_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace format_tag;
    using namespace utils;
    VDISPATCH_REDUCTION(is_dynamic_quantize(), VERBOSE_BAD_ALGORITHM);

    // The ported kernels require AVX-512 (bf16 uses an AVX-512F shim; f16 uses
    // F16C, present on every avx512_core CPU).
    isa_ = isa_undef;
    if (mayiuse(avx512_core_fp16)) {
        isa_ = avx512_core_fp16;
    } else if (mayiuse(avx512_core_bf16)) {
        isa_ = avx512_core_bf16;
    } else if (mayiuse(avx512_core)) {
        isa_ = avx512_core;
    }
    VDISPATCH_REDUCTION(isa_ != isa_undef, VERBOSE_UNSUPPORTED_ISA);

    VDISPATCH_REDUCTION(one_of(src_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REDUCTION(scale_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);

    // The fused kernels always write the quantized dst; compute-only mode is
    // left to the portable reference implementation.
    VDISPATCH_REDUCTION(
            !is_compute_only(), VERBOSE_UNSUPPORTED_FEATURE, "compute_only");

    // The kernels address memory as dense, row-major 2D matrices.
    VDISPATCH_REDUCTION(
            src_md()->ndims == 2, VERBOSE_BAD_NDIMS, "src", src_md()->ndims);

    VDISPATCH_REDUCTION(
            set_default_params() == status::success, VERBOSE_UNSUPPORTED_TAG);

    VDISPATCH_REDUCTION(memory_desc_matches_tag(*src_md(), ab)
                    && memory_desc_matches_tag(*dst_md(), ab)
                    && memory_desc_matches_tag(*scale_md(), ab),
            VERBOSE_UNSUPPORTED_TAG);

    // Determine the granularity handled by the ported kernels from the scale
    // shape. Everything else (per-tensor, per-col, per-group-row) is left to
    // the reference implementation.
    M_ = src_md()->dims[0];
    N_ = src_md()->dims[1];
    const dim_t sm_dim = scale_md()->dims[0];
    const dim_t sn_dim = scale_md()->dims[1];
    if (sm_dim == M_ && sn_dim == 1) {
        gran_ = granularity_t::per_token;
        G_ = 1;
    } else if (sm_dim == M_ && sn_dim > 1 && N_ % sn_dim == 0) {
        gran_ = granularity_t::per_group;
        G_ = sn_dim;
    } else {
        return status::unimplemented;
    }

    return status::success;
}

status_t dynamic_quantize_reduction_t::execute(const exec_ctx_t &ctx) const {
    using namespace data_type;

    const auto *p = pd();
    const auto sdt = p->src_md()->data_type;
    const dim_t M = p->M_, N = p->N_, G = p->G_;

    const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(int8_t *, DNNL_ARG_DST);
    auto *scale = CTX_OUT_MEM(float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto *src_u16 = static_cast<const uint16_t *>(src);
    const auto *src_f32 = static_cast<const float *>(src);

    if (p->gran_ == pd_t::granularity_t::per_token) {
        switch (sdt) {
            case f32:
                dq::dynamic_per_token_quant_f32_s8_native(
                        src_f32, dst, scale, M, N);
                break;
            case bf16:
                dq::dynamic_per_token_quant_bf16_s8_native(
                        src_u16, dst, scale, M, N);
                break;
            case f16:
                dq::dynamic_per_token_quant_f16_s8_native(
                        src_u16, dst, scale, M, N);
                break;
            default: return status::runtime_error;
        }
    } else { // per_group
        switch (sdt) {
            case f32:
                dq::dynamic_per_group_quant_f32_s8_native(
                        src_f32, dst, scale, M, N, G);
                break;
            case bf16:
                dq::dynamic_per_group_quant_bf16_s8_native(
                        src_u16, dst, scale, M, N, G);
                break;
            case f16:
                dq::dynamic_per_group_quant_f16_s8_native(
                        src_u16, dst, scale, M, N, G);
                break;
            default: return status::runtime_error;
        }
    }

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
