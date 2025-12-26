/*******************************************************************************
* Copyright 2022 IBM Corporation
* Copyright 2025 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#include "cpu/rv64/rvv_gemm_reorder.hpp"
#include "cpu/reorder/simple_reorder.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <unistd.h> // For thread sleep
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::cpu::q10n;

status_t rvv_matrixA_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    using namespace status;

    using namespace format_tag;

    status_t status = cpu_reorder_pd_t::init(engine, src_engine, dst_engine);
    if (status != success) return status;

    const memory_desc_wrapper id(src_md_), od(dst_md_);

    const int ndims = id.ndims();

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();

    const auto in_strides = id.strides();
    const auto out_strides = od.strides();

    const bool is_row_major = ((in_strides[0] == out_strides[0])
                                      && (in_strides[1] == out_strides[1])
                                      && (out_strides[1] == 1))
            ? true
            : false;
    const bool dt_ok = true && utils::one_of(type_i, data_type::f32)
            && utils::one_of(type_o, data_type::u8, data_type::s8);
    const bool args_ok = dt_ok && ndims == 2 && is_row_major;

    if (!args_ok) return invalid_arguments;
    init_scratchpad();
    return status::success;
}

status_t rvv_matrixA_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);

    if (_pd == nullptr) return status::out_of_memory;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());
    return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd.release());
}

template <typename InputType, typename OutputType>
void kernel(InputType *inp, OutputType *out, int N, const float SrcScale,
        const float DstScale, const int SrcZeroPoint, const int DstZeroPoint,
        const float beta) {

    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    while (N > 0) {
        size_t vl = __riscv_vsetvl_e32m1(N);

        vfloat32m1_t FloatVector = __riscv_vle32_v_f32m1(inp, vl);

        FloatVector
                = __riscv_vfsub_vf_f32m1(FloatVector, float(SrcZeroPoint), vl);
        FloatVector = __riscv_vfmul_vf_f32m1(FloatVector, SrcScale, vl);

        if (beta) {
            vuint8mf4_t vec_out_u8 = __riscv_vle8_v_u8mf4((uint8_t *)out, vl);
            vuint16mf2_t vec_out_u16
                    = __riscv_vwcvtu_x_x_v_u16mf2(vec_out_u8, vl);
            vuint32m1_t vec_out_u32
                    = __riscv_vwcvtu_x_x_v_u32m1(vec_out_u16, vl);
            vfloat32m1_t vec_out_f32
                    = __riscv_vfcvt_f_xu_v_f32m1(vec_out_u32, vl);
            vfloat32m1_t BetaOut
                    = __riscv_vfmul_vf_f32m1(vec_out_f32, beta, vl);
            FloatVector = __riscv_vfadd_vv_f32m1(FloatVector, BetaOut, vl);
        }

        FloatVector = __riscv_vfmul_vf_f32m1(FloatVector, DstScale, vl);
        FloatVector = __riscv_vfcvt_f_x_v_f32m1(
                __riscv_vfcvt_x_f_v_i32m1_rm(FloatVector, __RISCV_FRM_RNE, vl),
                vl);
        FloatVector
                = __riscv_vfadd_vf_f32m1(FloatVector, float(DstZeroPoint), vl);

        FloatVector
                = __riscv_vfmax_vf_f32m1(FloatVector, float(MinimumValue), vl);
        FloatVector
                = __riscv_vfmin_vf_f32m1(FloatVector, float(MaximumValue), vl);

        vuint32m1_t UIntegerVector
                = __riscv_vfcvt_xu_f_v_u32m1(FloatVector, vl);
        vuint16mf2_t UShortVector
                = __riscv_vncvt_x_x_w_u16mf2(UIntegerVector, vl);
        vuint8mf4_t UCharVector = __riscv_vncvt_x_x_w_u8mf4(UShortVector, vl);
        __riscv_vse8_v_u8mf4((uint8_t *)out, UCharVector, vl);

        out += vl;
        inp += vl;
        N -= vl;
    }
}

status_t rvv_matrixA_reorder_t::execute_body(const exec_ctx_t &ctx) const {
    using namespace utils;

    const auto input = CTX_IN_MEM(const float *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(unsigned char *, DNNL_ARG_TO);
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    MAYBE_UNUSED(scratchpad);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());

    DEFINE_ARG_SCALES_BUFFER_ATTR(pd()->attr(), src_scales, DNNL_ARG_FROM);
    DEFINE_ARG_SCALES_BUFFER_ATTR(pd()->attr(), dst_scales_, DNNL_ARG_TO);

    int src_scales_mask, dst_scales_mask;
    CHECK(get_scales_mask(pd()->attr(), &src_scales_mask, &dst_scales_mask));

    int scales_mask = std::max(src_scales_mask, dst_scales_mask);
    MAYBE_UNUSED(scales_mask);

    dim_t D_start, D_mask, D_rest;
    pd()->get_D_values(input_d, scales_mask, &D_start, &D_mask, &D_rest);

    const float *dst_scales = pd()->precompute_scales(
            scratchpad, pd()->attr(), D_mask, dst_scales_);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM);
    int src_zp = src_zero_points ? src_zero_points[0] : 0;

    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO);
    int dst_zp = dst_zero_points ? dst_zero_points[0] : 0;

    const float alpha = src_scales[0] * dst_scales[0];
    MAYBE_UNUSED(alpha);
    const float beta = pd()->beta();

    const auto &dims = input_d.dims();
    const auto in_strides = input_d.blocking_desc().strides;
    const auto M = dims[0];
    const auto K = dims[1];

    // Calculate block sizes
    dim_t M_b = 16;
    dim_t K_b = 64;
    K_b = std::min(K_b, K);

    const dim_t num_M_blocks = (M + M_b - 1) / M_b;
    const dim_t num_K_blocks = (K + K_b - 1) / K_b;

    parallel_nd(num_M_blocks, num_K_blocks, [&](dim_t mb, dim_t kb) {
        dim_t M_start = mb * M_b;
        dim_t M_end = nstl::min(M_start + M_b, M);
        dim_t K_start = kb * K_b;
        dim_t K_end = nstl::min(K_start + K_b, K);
        // Iterate over the block
        for (dim_t i = M_start; i < M_end; ++i) {
            kernel<const float, unsigned char>(
                    input + i * in_strides[0] + K_start,
                    output + i * in_strides[0] + K_start, K_end - K_start,
                    src_scales[0], dst_scales[0], src_zp, dst_zp, beta);
        }
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
