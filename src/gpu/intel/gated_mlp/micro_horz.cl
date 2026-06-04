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

#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types.h"

#include "gemm_gateup.h"

#define QUANTIZE_2D 2

#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

typedef ugemm_wgu_c_type s_tile_type;

#define binary_add(x, y) ((x) + (y))
#define binary_mul(x, y) ((x) * (y))

#ifdef ACTIVATION_SWISH

#define unary_activation(x) ((x) / (1.f + exp(-1.f * (x))))

#elif defined ACTIVATION_GELU_ERF

#define sqrt_2_over_2 0.707106769084930419921875f
#define unary_activation(x) (0.5f * (x) * (1.f + erf((x) * sqrt_2_over_2)))

#elif defined ACTIVATION_GELU_TANH

#define sqrt_2_over_pi 0.79788458347320556640625f
#define fitting_const 0.044715f
#define unary_activation(x) \
    (0.5f * (x) \
            * (1.f \
                    + tanh(sqrt_2_over_pi * (x) \
                            * (1 + fitting_const * (x) * (x)))))

#else
#error "Unknown activation function defined"
#endif

#define WGU_slm_size \
    (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n * sizeof(ACCUM_DATA_T))

#define BR ugemm_wgu_c_type_block0
#define BC ugemm_wgu_c_type_block1
#define NBR ugemm_wgu_c_type_nblock0
#define NBC ugemm_wgu_c_type_nblock1

#if defined ACCUM_DT_S32
DECLARE_2D_TILE(s_tile_type_f32, float, SUBGROUP_SIZE, BR, BC, NBR, NBC)
#elif defined ACCUM_DT_F32
#define s_tile_type_f32 s_tile_type
#endif
DECLARE_2D_TILE(s_tile_type_dst, INTER_DATA_T, SUBGROUP_SIZE, BR, BC, NBR, NBC)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type_f32, SUBGROUP_SIZE, BR, BC, NBR, NBC,
        s_tile_type_dst, SUBGROUP_SIZE, BR, BC, NBR, NBC, CONVERT_DATA_T)

#if (SRC_ELEMENTS_PER_BYTE > 1)
#define AS_SRC_PTR(p) ((const global uchar *)(p))
#else
#define AS_SRC_PTR(p) (p)
#endif

#if (SRC_ZP_ELEMENTS_PER_BYTE > 1)
#define AS_SRC_ZP_PTR(p) ((const global uchar *)(p))
#else
#define AS_SRC_ZP_PTR(p) (p)
#endif

#if (WTS_GATE_ELEMENTS_PER_BYTE > 1)
#define AS_WTS_GATE_PTR(p) ((const global uchar *)(p))
#else
#define AS_WTS_GATE_PTR(p) (p)
#endif

#if (WTS_GATE_ZP_ELEMENTS_PER_BYTE > 1)
#define AS_WTS_GATE_ZP_PTR(p) ((const global uchar *)(p))
#else
#define AS_WTS_GATE_ZP_PTR(p) (p)
#endif

#if (WTS_UP_ELEMENTS_PER_BYTE > 1)
#define AS_WTS_UP_PTR(p) ((const global uchar *)(p))
#else
#define AS_WTS_UP_PTR(p) (p)
#endif

#if (WTS_UP_ZP_ELEMENTS_PER_BYTE > 1)
#define AS_WTS_UP_ZP_PTR(p) ((const global uchar *)(p))
#else
#define AS_WTS_UP_ZP_PTR(p) (p)
#endif

#if (NDIMS == 3)
#define W_GATE_LD W_GATE_S2
#define W_UP_LD W_UP_S2
#elif (NDIMS == 2)
#define W_GATE_LD W_GATE_S1
#define W_UP_LD W_UP_S1
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
micro_gated_mlp_horz(const __global SRC_DATA_T *src,
        const __global WTS_GATE_DATA_T *W_gate,
        const __global WTS_UP_DATA_T *W_up,
        const __global WTS_DOWN_DATA_T *W_down, __global DST_DATA_T *dst,
        long MB, long IC, long OC, __global INTER_DATA_T *tmp_reduce_mem,
        const __global SRC_ATTR_SCALES_DATA_T *src_scales,
        const __global SRC_ATTR_ZP_DATA_T *src_zp,
        const __global WTS_GATE_ATTR_SCALES_DATA_T *wts_gate_scales,
        const __global WTS_GATE_ATTR_ZP_DATA_T *wts_gate_zp,
        const __global WTS_UP_ATTR_SCALES_DATA_T *wts_up_scales,
        const __global WTS_UP_ATTR_ZP_DATA_T *wts_up_zp,
        const __global WTS_DOWN_ATTR_SCALES_DATA_T *wts_down_scales,
        const __global WTS_DOWN_ATTR_ZP_DATA_T *wts_down_zp) {

    local char slm[ugemm_wgu_slm_size + WGU_slm_size];
    local char *S_WU_slm = slm + ugemm_wgu_slm_size;
#if WITH_SLM
    local char *GEMM_slm = slm;
#else
    local char *GEMM_slm = NULL;
#endif // WITH_SLM

    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_m; // OC
    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_n; // MB

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m;
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m;

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_m;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_n;

    s_tile_type_f32 S_tile;
    do {
#if defined ACCUM_DT_S32
        s_tile_type S_tile_s32 =
#elif defined ACCUM_DT_F32
        S_tile =
#endif // defined ACCUM_DT_S32
                ugemm_wgu(AS_WTS_UP_PTR(W_up), W_UP_LD, AS_SRC_PTR(src), SRC_S0,
                        OC, MB, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu,
                        GEMM_slm
#if WTS_UP_SCALES == QUANTIZE_2D
                        ,
                        wts_up_scales
#endif // WTS_UP_SCALES == QUANTIZE_2D
#if WTS_UP_ZERO_POINTS
                        ,
                        AS_WTS_UP_ZP_PTR(wts_up_zp)
#endif // WTS_UP_ZERO_POINTS
#if (WTS_UP_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
                                ,
                        WGU_QUANT_S0
#endif // (WTS_UP_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
#if SRC_SCALES == QUANTIZE_2D
                        ,
                        src_scales
#endif // SRC_SCALES == QUANTIZE_2D
#if SRC_ZERO_POINTS
                        ,
                        AS_SRC_ZP_PTR(src_zp)
#endif // SRC_ZERO_POINTS
#if (SRC_SCALES == QUANTIZE_2D) || SRC_ZERO_POINTS
                                ,
                        SRC_QUANT_S0
#endif // (SRC_SCALES == QUANTIZE_2D) || SRC_ZERO_POINTS
                );

#if defined ACCUM_DT_S32
        tile_convert(S_tile_s32, S_tile, convert_float);
#elif defined ACCUM_DT_F32
#endif // defined ACCUM_DT_S32
    } while (false);

#if (SRC_SCALES == QUANTIZE_COMMON) || (WTS_UP_SCALES == QUANTIZE_COMMON)
    do {
#define scale_op(x) ((x) * scale)
#if (SRC_SCALES == QUANTIZE_COMMON) && (WTS_UP_SCALES == QUANTIZE_COMMON)
        float scale
                = convert_float(*wts_up_scales) * convert_float(*src_scales);
#elif SRC_SCALES == QUANTIZE_COMMON
        float scale = convert_float(*src_scales);
#else
        float scale = convert_float(*wts_up_scales);
#endif // (SRC_SCALES == QUANTIZE_COMMON) && (WTS_UP_SCALES == QUANTIZE_COMMON)
        tile_elementwise(S_tile, scale_op);
    } while (false);
#endif // (SRC_SCALES == QUANTIZE_COMMON) || (WTS_UP_SCALES == QUANTIZE_COMMON)

#ifndef UGEMM_UP_ONLY
    do {
        tile_store(S_tile, (local float *)S_WU_slm, OC, MB, ugemm_wgu_wg_tile_m,
                sg_i0_wgu, sg_j0_wgu);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE); // no wg communication here

#if defined ACCUM_DT_S32
        s_tile_type S_tile_s32 =
#elif defined ACCUM_DT_F32
        S_tile =
#endif // defined ACCUM_DT_S32
                ugemm_wgu(AS_WTS_GATE_PTR(W_gate), W_GATE_LD, AS_SRC_PTR(src),
                        SRC_S0, OC, MB, IC, wg_i0, wg_j0, 0, sg_i_wgu, sg_j_wgu,
                        GEMM_slm
#if WTS_GATE_SCALES == QUANTIZE_2D
                        ,
                        wts_gate_scales
#endif // WTS_GATE_SCALES == QUANTIZE_2D
#if WTS_GATE_ZERO_POINTS
                        ,
                        AS_WTS_GATE_ZP_PTR(wts_gate_zp)
#endif // WTS_GATE_ZERO_POINTS
#if (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
                                ,
                        WGU_QUANT_S0
#endif // (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
#if SRC_SCALES == QUANTIZE_2D
                        ,
                        src_scales
#endif // SRC_SCALES == QUANTIZE_2D
#if SRC_ZERO_POINTS
                        ,
                        AS_SRC_ZP_PTR(src_zp)
#endif // SRC_ZERO_POINTS
#if (SRC_SCALES == QUANTIZE_2D) || SRC_ZERO_POINTS
                                ,
                        SRC_QUANT_S0
#endif // (SRC_SCALES == QUANTIZE_2D) || SRC_ZERO_POINTS
                );

#if defined ACCUM_DT_S32
        tile_convert(S_tile_s32, S_tile, convert_float);
#elif defined ACCUM_DT_F32
#endif // defined ACCUM_DT_S32
    } while (false);
    do {
        s_tile_type_f32 S_WU_tile;
        tile_load(&S_WU_tile, (local float *)S_WU_slm, OC, MB,
                ugemm_wgu_wg_tile_m, sg_i0_wgu, sg_j0_wgu);

#if (SRC_SCALES == QUANTIZE_COMMON) || (WTS_GATE_SCALES == QUANTIZE_COMMON)
#define activation(x) unary_activation((x) * scale)
#if (SRC_SCALES == QUANTIZE_COMMON) && (WTS_GATE_SCALES == QUANTIZE_COMMON)
        float scale
                = convert_float(*wts_gate_scales) * convert_float(*src_scales);
#elif SRC_SCALES == QUANTIZE_COMMON
        float scale = convert_float(*src_scales);
#else
        float scale = convert_float(*wts_gate_scales);
#endif // (SRC_SCALES == QUANTIZE_COMMON) && (WTS_GATE_SCALES == QUANTIZE_COMMON)
#else
#define activation(x) unary_activation(x)
#endif // (SRC_SCALES == QUANTIZE_COMMON) || (WTS_GATE_SCALES == QUANTIZE_COMMON)
        tile_elementwise(S_tile, activation);
        //sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        tile_binary(S_tile, S_WU_tile, binary_mul);
    } while (false);
#endif // UGEMM_UP_ONLY

    s_tile_type_dst S_tile_dst;
    tile_copy_reblock(S_tile, &S_tile_dst);
    tile_store(S_tile_dst, tmp_reduce_mem, OC, MB, INTER_S0, wg_i0 + sg_i0_wgu,
            wg_j0 + sg_j0_wgu);
}
