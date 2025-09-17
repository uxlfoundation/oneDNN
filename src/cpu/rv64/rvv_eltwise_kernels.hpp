/******************************************************************************
 * Copyright 2025
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

#ifndef CPU_RV64_RVV_ELTWISE_KERNELS_HPP
#define CPU_RV64_RVV_ELTWISE_KERNELS_HPP

#include <math.h>
#include <vector>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using eval_fwd_f32_fn_t = vfloat32m1_t (*)(vfloat32m1_t, float, float, size_t);
using eval_fwd_f32_fn_m4_t
        = vfloat32m4_t (*)(vfloat32m4_t, float, float, size_t);
using eval_fwd_s32_fn_t = vint32m1_t (*)(vint32m1_t, float, float, size_t);
using eval_fwd_s8_fn_t = vint8m1_t (*)(vint8m1_t, float, float, size_t);
using eval_fwd_u8_fn_t = vuint8m1_t (*)(vuint8m1_t, float, float, size_t);
using eval_bwd_f32_fn_t
        = vfloat32m1_t (*)(vfloat32m1_t, vfloat32m1_t, float, float, size_t);
using eval_bwd_f32_fn_m4_t
        = vfloat32m4_t (*)(vfloat32m4_t, vfloat32m4_t, float, float, size_t);
using eval_bwd_s32_fn_t
        = vint32m1_t (*)(vint32m1_t, vint32m1_t, float, float, size_t);
using eval_bwd_s8_fn_t
        = vint8m1_t (*)(vint8m1_t, vint8m1_t, float, float, size_t);
using eval_bwd_u8_fn_t
        = vuint8m1_t (*)(vuint8m1_t, vuint8m1_t, float, float, size_t);

/*** Kernels for forward pass ***/
inline void rvv_eltwise_fwd_kernel_f32(const float *src, float *dst, dim_t len,
        float alpha, float beta, eval_fwd_f32_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        vfloat32m1_t vin = __riscv_vle32_v_f32m1(src + i, vl);
        vfloat32m1_t vout = eval(vin, alpha, beta, vl);
        __riscv_vse32_v_f32m1(dst + i, vout, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_fwd_kernel_s32(const int32_t *src, int32_t *dst,
        dim_t len, float alpha, float beta, eval_fwd_s32_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        vint32m1_t vin = __riscv_vle32_v_i32m1(src + i, vl);
        vint32m1_t vout = eval(vin, alpha, beta, vl);
        __riscv_vse32_v_i32m1(dst + i, vout, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_fwd_kernel_s8(const int8_t *src, int8_t *dst, dim_t len,
        float alpha, float beta, eval_fwd_s8_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vint8m1_t vin = __riscv_vle8_v_i8m1(src + i, vl);
        vint8m1_t vout = eval(vin, alpha, beta, vl);
        __riscv_vse8_v_i8m1(dst + i, vout, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_fwd_kernel_u8(const uint8_t *src, uint8_t *dst,
        dim_t len, float alpha, float beta, eval_fwd_u8_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vuint8m1_t vin = __riscv_vle8_v_u8m1(src + i, vl);
        vuint8m1_t vout = eval(vin, alpha, beta, vl);
        __riscv_vse8_v_u8m1(dst + i, vout, vl);
        i += static_cast<dim_t>(vl);
    }
}

/*** Convert methods for s32/s8/u8 and apply fwd operations ***/
inline vint32m1_t rvv_convert_fwd_and_apply_f32_to_s32(vint32m1_t vin,
        float alpha, float beta, size_t vl, eval_fwd_f32_fn_t eval) {
    vfloat32m1_t vin_f32 = __riscv_vfcvt_f_x_v_f32m1(vin, vl);
    vfloat32m1_t vout_f32 = eval(vin_f32, alpha, beta, vl);
    vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(-2147483648.0f, vl);
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(2147483647.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m1(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m1(vout_f32, vmax, vl);
    return __riscv_vfcvt_x_f_v_i32m1(vout_f32, vl);
}
inline vint8m1_t rvv_convert_fwd_and_apply_f32_to_s8(vint8m1_t vin, float alpha,
        float beta, size_t vl, eval_fwd_f32_fn_m4_t eval) {
    vint32m4_t vin_s32 = __riscv_vsext_vf4_i32m4(vin, vl);
    vfloat32m4_t vin_f32 = __riscv_vfcvt_f_x_v_f32m4(vin_s32, vl);
    vfloat32m4_t vout_f32 = eval(vin_f32, alpha, beta, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(-128.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(127.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m4(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m4(vout_f32, vmax, vl);
    vint32m4_t vout_s32 = __riscv_vfcvt_x_f_v_i32m4(vout_f32, vl);
    vint16m2_t vout_s16 = __riscv_vncvt_x_x_w_i16m2(vout_s32, vl);
    return __riscv_vncvt_x_x_w_i8m1(vout_s16, vl);
}
inline vuint8m1_t rvv_convert_fwd_and_apply_f32_to_u8(vuint8m1_t vin,
        float alpha, float beta, size_t vl, eval_fwd_f32_fn_m4_t eval) {
    vuint32m4_t vin_u32 = __riscv_vzext_vf4_u32m4(vin, vl);
    vfloat32m4_t vin_f32 = __riscv_vfcvt_f_xu_v_f32m4(vin_u32, vl);
    vfloat32m4_t vout_f32 = eval(vin_f32, alpha, beta, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(255.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m4(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m4(vout_f32, vmax, vl);
    vuint32m4_t vout_u32 = __riscv_vfcvt_xu_f_v_u32m4(vout_f32, vl);
    vuint16m2_t vout_u16 = __riscv_vncvt_x_x_w_u16m2(vout_u32, vl);
    return __riscv_vncvt_x_x_w_u8m1(vout_u16, vl);
}

/*** Operation definitions for forward pass ***/
// ReLU
inline vfloat32m1_t rvv_eltwise_fwd_relu_f32(
        vfloat32m1_t vin, float alpha, float /*beta*/, size_t vl) {
    if (alpha == 0.f) {
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
        return __riscv_vfmax_vv_f32m1(vin, zero, vl);
    } else {
        vbool32_t p = __riscv_vmfgt_vf_f32m1_b32(vin, 0.f, vl);
        vfloat32m1_t vneg = __riscv_vfmul_vf_f32m1(vin, alpha, vl);
        // if (x > 0) pick x, else pick alpha * x
        return __riscv_vmerge_vvm_f32m1(vneg, vin, p, vl);
    }
}
// m4 is used to handle functions converted from s8/u8
inline vfloat32m4_t rvv_eltwise_fwd_relu_f32_m4(
        vfloat32m4_t vin, float alpha, float /*beta*/, size_t vl) {
    if (alpha == 0.f) {
        vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.f, vl);
        return __riscv_vfmax_vv_f32m4(vin, zero, vl);
    } else {
        vbool8_t p = __riscv_vmfgt_vf_f32m4_b8(vin, 0.f, vl);
        vfloat32m4_t vneg = __riscv_vfmul_vf_f32m4(vin, alpha, vl);
        // if (x > 0) pick x, else pick alpha * x
        return __riscv_vmerge_vvm_f32m4(vneg, vin, p, vl);
    }
}
inline vint32m1_t rvv_eltwise_fwd_relu_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
        return __riscv_vmax_vv_i32m1(vin, zero, vl);
    } else {
        return rvv_convert_fwd_and_apply_f32_to_s32(
                vin, alpha, beta, vl, rvv_eltwise_fwd_relu_f32);
    }
}
inline vint8m1_t rvv_eltwise_fwd_relu_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
        return __riscv_vmax_vv_i8m1(vin, zero, vl);
    } else {
        return rvv_convert_fwd_and_apply_f32_to_s8(
                vin, alpha, beta, vl, rvv_eltwise_fwd_relu_f32_m4);
    }
}
inline vuint8m1_t rvv_eltwise_fwd_relu_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
        return __riscv_vmaxu_vv_u8m1(vin, zero, vl);
    } else {
        return rvv_convert_fwd_and_apply_f32_to_u8(
                vin, alpha, beta, vl, rvv_eltwise_fwd_relu_f32_m4);
    }
}

// Square
inline vfloat32m1_t rvv_eltwise_fwd_square_f32(
        vfloat32m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vfmul_vv_f32m1(vin, vin, vl);
}
inline vint32m1_t rvv_eltwise_fwd_square_s32(
        vint32m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vmul_vv_i32m1(vin, vin, vl);
}
inline vint8m1_t rvv_eltwise_fwd_square_s8(
        vint8m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vmul_vv_i8m1(vin, vin, vl);
}
inline vuint8m1_t rvv_eltwise_fwd_square_u8(
        vuint8m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vmul_vv_u8m1(vin, vin, vl);
}

// Abs
inline vfloat32m1_t rvv_eltwise_fwd_abs_f32(
        vfloat32m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat32m1_t vneg = __riscv_vfneg_v_f32m1(vin, vl);
    return __riscv_vfmax_vv_f32m1(vin, vneg, vl);
}
inline vint32m1_t rvv_eltwise_fwd_abs_s32(
        vint32m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    vint32m1_t vneg = __riscv_vneg_v_i32m1(vin, vl);
    return __riscv_vmax_vv_i32m1(vin, vneg, vl);
}
inline vint8m1_t rvv_eltwise_fwd_abs_s8(
        vint8m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    vint8m1_t vneg = __riscv_vneg_v_i8m1(vin, vl);
    return __riscv_vmax_vv_i8m1(vin, vneg, vl);
}
inline vuint8m1_t rvv_eltwise_fwd_abs_u8(
        vuint8m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    // always positive of u8
    return vin;
}

// Sqrt
inline vfloat32m1_t rvv_eltwise_fwd_sqrt_f32(
        vfloat32m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vfsqrt_v_f32m1(vin, vl);
}
inline vfloat32m4_t rvv_eltwise_fwd_sqrt_f32_m4(
        vfloat32m4_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vfsqrt_v_f32m4(vin, vl);
}
inline vint32m1_t rvv_eltwise_fwd_sqrt_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s32(
            vin, alpha, beta, vl, rvv_eltwise_fwd_sqrt_f32);
}
inline vint8m1_t rvv_eltwise_fwd_sqrt_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_sqrt_f32_m4);
}
inline vuint8m1_t rvv_eltwise_fwd_sqrt_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_u8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_sqrt_f32_m4);
}

// Linear: alpha * x + beta
inline vfloat32m1_t rvv_eltwise_fwd_linear_f32(
        vfloat32m1_t vin, float alpha, float beta, size_t vl) {
    vfloat32m1_t v = __riscv_vfmul_vf_f32m1(vin, alpha, vl);
    return __riscv_vfadd_vf_f32m1(v, beta, vl);
}
inline vfloat32m4_t rvv_eltwise_fwd_linear_f32_m4(
        vfloat32m4_t vin, float alpha, float beta, size_t vl) {
    vfloat32m4_t v = __riscv_vfmul_vf_f32m4(vin, alpha, vl);
    return __riscv_vfadd_vf_f32m4(v, beta, vl);
}
inline vint32m1_t rvv_eltwise_fwd_linear_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s32(
            vin, alpha, beta, vl, rvv_eltwise_fwd_linear_f32);
}
inline vint8m1_t rvv_eltwise_fwd_linear_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_linear_f32_m4);
}
inline vuint8m1_t rvv_eltwise_fwd_linear_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_u8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_linear_f32_m4);
}

// clip: clamp(x, alpha, beta)
inline vfloat32m1_t rvv_eltwise_fwd_clip_f32(
        vfloat32m1_t vin, float alpha, float beta, size_t vl) {
    vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(alpha, vl);
    vfloat32m1_t vb = __riscv_vfmv_v_f_f32m1(beta, vl);
    vfloat32m1_t vmax = __riscv_vfmax_vv_f32m1(vin, va, vl);
    return __riscv_vfmin_vv_f32m1(vmax, vb, vl);
}
inline vfloat32m4_t rvv_eltwise_fwd_clip_f32_m4(
        vfloat32m4_t vin, float alpha, float beta, size_t vl) {
    vfloat32m4_t va = __riscv_vfmv_v_f_f32m4(alpha, vl);
    vfloat32m4_t vb = __riscv_vfmv_v_f_f32m4(beta, vl);
    vfloat32m4_t vmax = __riscv_vfmax_vv_f32m4(vin, va, vl);
    return __riscv_vfmin_vv_f32m4(vmax, vb, vl);
}
inline vint32m1_t rvv_eltwise_fwd_clip_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s32(
            vin, alpha, beta, vl, rvv_eltwise_fwd_clip_f32);
}
inline vint8m1_t rvv_eltwise_fwd_clip_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_s8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_clip_f32_m4);
}
inline vuint8m1_t rvv_eltwise_fwd_clip_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    return rvv_convert_fwd_and_apply_f32_to_u8(
            vin, alpha, beta, vl, rvv_eltwise_fwd_clip_f32_m4);
}

// hardsigmoid: clamp(alpha*x + beta, 0, 1)
inline vfloat32m1_t rvv_eltwise_fwd_hardsigmoid_f32(
        vfloat32m1_t vin, float alpha, float beta, size_t vl) {
    vfloat32m1_t v = rvv_eltwise_fwd_linear_f32(vin, alpha, beta, vl);
    return rvv_eltwise_fwd_clip_f32(v, 0.f, 1.f, vl);
}
inline vint32m1_t rvv_eltwise_fwd_hardsigmoid_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    vint32m1_t v = rvv_eltwise_fwd_linear_s32(vin, alpha, beta, vl);
    return rvv_eltwise_fwd_clip_s32(v, 0.f, 1.f, vl);
}
inline vint8m1_t rvv_eltwise_fwd_hardsigmoid_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    vint8m1_t v = rvv_eltwise_fwd_linear_s8(vin, alpha, beta, vl);
    return rvv_eltwise_fwd_clip_s8(v, 0.f, 1.f, vl);
}
inline vuint8m1_t rvv_eltwise_fwd_hardsigmoid_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    vuint8m1_t v = rvv_eltwise_fwd_linear_u8(vin, alpha, beta, vl);
    return rvv_eltwise_fwd_clip_u8(v, 0.f, 1.f, vl);
}

// hardswish: x * hardsigmoid(x, alpha, beta)
inline vfloat32m1_t rvv_eltwise_fwd_hardswish_f32(
        vfloat32m1_t vin, float alpha, float beta, size_t vl) {
    return __riscv_vfmul_vv_f32m1(
            vin, rvv_eltwise_fwd_hardsigmoid_f32(vin, alpha, beta, vl), vl);
}
inline vint32m1_t rvv_eltwise_fwd_hardswish_s32(
        vint32m1_t vin, float alpha, float beta, size_t vl) {
    return __riscv_vmul_vv_i32m1(
            vin, rvv_eltwise_fwd_hardsigmoid_s32(vin, alpha, beta, vl), vl);
}
inline vint8m1_t rvv_eltwise_fwd_hardswish_s8(
        vint8m1_t vin, float alpha, float beta, size_t vl) {
    return __riscv_vmul_vv_i8m1(
            vin, rvv_eltwise_fwd_hardsigmoid_s8(vin, alpha, beta, vl), vl);
}
inline vuint8m1_t rvv_eltwise_fwd_hardswish_u8(
        vuint8m1_t vin, float alpha, float beta, size_t vl) {
    return __riscv_vmul_vv_u8m1(
            vin, rvv_eltwise_fwd_hardsigmoid_u8(vin, alpha, beta, vl), vl);
}

/*** Dispatch getters for forward pass ***/
inline eval_fwd_f32_fn_t get_eval_fwd_f32(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_fwd_relu_f32;
        case alg_kind::eltwise_square: return rvv_eltwise_fwd_square_f32;
        case alg_kind::eltwise_abs: return rvv_eltwise_fwd_abs_f32;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_fwd_sqrt_f32;
        case alg_kind::eltwise_linear: return rvv_eltwise_fwd_linear_f32;
        case alg_kind::eltwise_clip: return rvv_eltwise_fwd_clip_f32;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_fwd_hardsigmoid_f32;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_fwd_hardswish_f32;
        default: return nullptr;
    }
}

inline eval_fwd_s32_fn_t get_eval_fwd_s32(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_fwd_relu_s32;
        case alg_kind::eltwise_square: return rvv_eltwise_fwd_square_s32;
        case alg_kind::eltwise_abs: return rvv_eltwise_fwd_abs_s32;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_fwd_sqrt_s32;
        case alg_kind::eltwise_linear: return rvv_eltwise_fwd_linear_s32;
        case alg_kind::eltwise_clip: return rvv_eltwise_fwd_clip_s32;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_fwd_hardsigmoid_s32;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_fwd_hardswish_s32;
        default: return nullptr;
    }
}
inline eval_fwd_s8_fn_t get_eval_fwd_s8(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_fwd_relu_s8;
        case alg_kind::eltwise_square: return rvv_eltwise_fwd_square_s8;
        case alg_kind::eltwise_abs: return rvv_eltwise_fwd_abs_s8;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_fwd_sqrt_s8;
        case alg_kind::eltwise_linear: return rvv_eltwise_fwd_linear_s8;
        case alg_kind::eltwise_clip: return rvv_eltwise_fwd_clip_s8;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_fwd_hardsigmoid_s8;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_fwd_hardswish_s8;
        default: return nullptr;
    }
}
inline eval_fwd_u8_fn_t get_eval_fwd_u8(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_fwd_relu_u8;
        case alg_kind::eltwise_square: return rvv_eltwise_fwd_square_u8;
        case alg_kind::eltwise_abs: return rvv_eltwise_fwd_abs_u8;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_fwd_sqrt_u8;
        case alg_kind::eltwise_linear: return rvv_eltwise_fwd_linear_u8;
        case alg_kind::eltwise_clip: return rvv_eltwise_fwd_clip_u8;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_fwd_hardsigmoid_u8;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_fwd_hardswish_u8;
        default: return nullptr;
    }
}

/*** Apply methods for forward pass ***/
inline void rvv_eltwise_apply_fwd_f32(alg_kind_t alg, const float *src,
        float *dst, dim_t len, float alpha, float beta) {
    auto eval = get_eval_fwd_f32(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_fwd_f32] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_fwd_kernel_f32(src, dst, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_fwd_s32(alg_kind_t alg, const int32_t *src,
        int32_t *dst, dim_t len, float alpha, float beta) {
    auto eval = get_eval_fwd_s32(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_fwd_s32] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_fwd_kernel_s32(src, dst, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_fwd_s8(alg_kind_t alg, const int8_t *src,
        int8_t *dst, dim_t len, float alpha, float beta) {
    auto eval = get_eval_fwd_s8(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_fwd_s8] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_fwd_kernel_s8(src, dst, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_fwd_u8(alg_kind_t alg, const uint8_t *src,
        uint8_t *dst, dim_t len, float alpha, float beta) {
    auto eval = get_eval_fwd_u8(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_fwd_u8] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_fwd_kernel_u8(src, dst, len, alpha, beta, eval);
}

/* --- Backward pass --- */
// For backward pass, we need to compute the gradient of the loss with respect to the input
// and the parameters. Thus, diff_src is the output, diff_dst and src are the inputs.

/*** Kernels for backward pass ***/
inline void rvv_eltwise_bwd_kernel_f32(float *diff_src, const float *diff_dst,
        const float *src, dim_t len, float alpha, float beta,
        eval_bwd_f32_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        vfloat32m1_t vdiff_dst = __riscv_vle32_v_f32m1(diff_dst + i, vl);
        vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(src + i, vl);
        vfloat32m1_t vdiff_src = eval(vdiff_dst, vsrc, alpha, beta, vl);
        __riscv_vse32_v_f32m1(diff_src + i, vdiff_src, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_bwd_kernel_s32(int32_t *diff_src,
        const int32_t *diff_dst, const int32_t *src, dim_t len, float alpha,
        float beta, eval_bwd_s32_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        vint32m1_t vdiff_dst = __riscv_vle32_v_i32m1(diff_dst + i, vl);
        vint32m1_t vsrc = __riscv_vle32_v_i32m1(src + i, vl);
        vint32m1_t vdiff_src = eval(vdiff_dst, vsrc, alpha, beta, vl);
        __riscv_vse32_v_i32m1(diff_src + i, vdiff_src, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_bwd_kernel_s8(int8_t *diff_src, const int8_t *diff_dst,
        const int8_t *src, dim_t len, float alpha, float beta,
        eval_bwd_s8_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vint8m1_t vdiff_dst = __riscv_vle8_v_i8m1(diff_dst + i, vl);
        vint8m1_t vsrc = __riscv_vle8_v_i8m1(src + i, vl);
        vint8m1_t vdiff_src = eval(vdiff_dst, vsrc, alpha, beta, vl);
        __riscv_vse8_v_i8m1(diff_src + i, vdiff_src, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_eltwise_bwd_kernel_u8(uint8_t *diff_src,
        const uint8_t *diff_dst, const uint8_t *src, dim_t len, float alpha,
        float beta, eval_bwd_u8_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vuint8m1_t vdiff_dst = __riscv_vle8_v_u8m1(diff_dst + i, vl);
        vuint8m1_t vsrc = __riscv_vle8_v_u8m1(src + i, vl);
        vuint8m1_t vdiff_src = eval(vdiff_dst, vsrc, alpha, beta, vl);
        __riscv_vse8_v_u8m1(diff_src + i, vdiff_src, vl);
        i += static_cast<dim_t>(vl);
    }
}

/*** Convert methods for s32/s8/u8 and apply bwd operations ***/
inline vint32m1_t rvv_convert_bwd_and_apply_f32_to_s32(vint32m1_t vin_dd,
        vint32m1_t vin_s, float alpha, float beta, size_t vl,
        eval_bwd_f32_fn_t eval) {
    vfloat32m1_t vin_dd_f32 = __riscv_vfcvt_f_x_v_f32m1(vin_dd, vl);
    vfloat32m1_t vin_s_f32 = __riscv_vfcvt_f_x_v_f32m1(vin_s, vl);
    vfloat32m1_t vout_ds_f32 = eval(vin_dd_f32, vin_s_f32, alpha, beta, vl);
    vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(-2147483648.0f, vl);
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(2147483647.0f, vl);
    vout_ds_f32 = __riscv_vfmax_vv_f32m1(vout_ds_f32, vmin, vl);
    vout_ds_f32 = __riscv_vfmin_vv_f32m1(vout_ds_f32, vmax, vl);
    return __riscv_vfcvt_x_f_v_i32m1(vout_ds_f32, vl);
}
inline vint8m1_t rvv_convert_bwd_and_apply_f32_to_s8(vint8m1_t vin_dd,
        vint8m1_t vin_s, float alpha, float beta, size_t vl,
        eval_bwd_f32_fn_m4_t eval) {
    vint32m4_t vin_dd_s32 = __riscv_vsext_vf4_i32m4(vin_dd, vl);
    vint32m4_t vin_s_s32 = __riscv_vsext_vf4_i32m4(vin_s, vl);
    vfloat32m4_t vin_dd_f32 = __riscv_vfcvt_f_x_v_f32m4(vin_dd_s32, vl);
    vfloat32m4_t vin_s_f32 = __riscv_vfcvt_f_x_v_f32m4(vin_s_s32, vl);
    vfloat32m4_t vout_ds_f32 = eval(vin_dd_f32, vin_s_f32, alpha, beta, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(-128.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(127.0f, vl);
    vout_ds_f32 = __riscv_vfmax_vv_f32m4(vout_ds_f32, vmin, vl);
    vout_ds_f32 = __riscv_vfmin_vv_f32m4(vout_ds_f32, vmax, vl);
    vint32m4_t vout_ds_s32 = __riscv_vfcvt_x_f_v_i32m4(vout_ds_f32, vl);
    vint16m2_t vout_ds_s8 = __riscv_vncvt_x_x_w_i16m2(vout_ds_s32, vl);
    return __riscv_vncvt_x_x_w_i8m1(vout_ds_s8, vl);
}
inline vuint8m1_t rvv_convert_bwd_and_apply_f32_to_u8(vuint8m1_t vin_dd,
        vuint8m1_t vin_s, float alpha, float beta, size_t vl,
        eval_bwd_f32_fn_m4_t eval) {
    vuint32m4_t vin_dd_u32 = __riscv_vzext_vf4_u32m4(vin_dd, vl);
    vuint32m4_t vin_s_u32 = __riscv_vzext_vf4_u32m4(vin_s, vl);
    vfloat32m4_t vin_dd_f32 = __riscv_vfcvt_f_xu_v_f32m4(vin_dd_u32, vl);
    vfloat32m4_t vin_s_f32 = __riscv_vfcvt_f_xu_v_f32m4(vin_s_u32, vl);
    vfloat32m4_t vout_ds_f32 = eval(vin_dd_f32, vin_s_f32, alpha, beta, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(255.0f, vl);
    vout_ds_f32 = __riscv_vfmax_vv_f32m4(vout_ds_f32, vmin, vl);
    vout_ds_f32 = __riscv_vfmin_vv_f32m4(vout_ds_f32, vmax, vl);
    vuint32m4_t vout_ds_u32 = __riscv_vfcvt_xu_f_v_u32m4(vout_ds_f32, vl);
    vuint16m2_t vout_ds_u16 = __riscv_vncvt_x_x_w_u16m2(vout_ds_u32, vl);
    return __riscv_vncvt_x_x_w_u8m1(vout_ds_u16, vl);
}

/*** Operation definitions for backward pass ***/
// ReLU : return s > 0 ? dd : (U)(dd * alpha);
inline vfloat32m1_t rvv_eltwise_bwd_relu_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float alpha, float /*beta*/, size_t vl) {
    vbool32_t vcond = __riscv_vmfgt_vf_f32m1_b32(vin_s, 0.f, vl);
    vfloat32m1_t vt = __riscv_vfmul_vf_f32m1(vin_dd, alpha, vl);
    return __riscv_vmerge_vvm_f32m1(vt, vin_dd, vcond, vl);
}
inline vfloat32m4_t rvv_eltwise_bwd_relu_f32_m4(vfloat32m4_t vin_dd,
        vfloat32m4_t vin_s, float alpha, float /*beta*/, size_t vl) {
    vbool8_t vcond = __riscv_vmfgt_vf_f32m4_b8(vin_s, 0.f, vl);
    vfloat32m4_t vt = __riscv_vfmul_vf_f32m4(vin_dd, alpha, vl);
    return __riscv_vmerge_vvm_f32m4(vt, vin_dd, vcond, vl);
}
inline vint32m1_t rvv_eltwise_bwd_relu_s32(vint32m1_t vin_dd, vint32m1_t vin_s,
        float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vbool32_t vcond = __riscv_vmsgt_vx_i32m1_b32(
                vin_s, static_cast<int32_t>(0), vl);
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
        return __riscv_vmerge_vvm_i32m1(vzero, vin_dd, vcond, vl);
    } else {
        return rvv_convert_bwd_and_apply_f32_to_s32(
                vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_relu_f32);
    }
}
inline vint8m1_t rvv_eltwise_bwd_relu_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vbool8_t vcond
                = __riscv_vmsgt_vx_i8m1_b8(vin_s, static_cast<int8_t>(0), vl);
        vint8m1_t vzero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
        return __riscv_vmerge_vvm_i8m1(vzero, vin_dd, vcond, vl);
    } else {
        return rvv_convert_bwd_and_apply_f32_to_s8(
                vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_relu_f32_m4);
    }
}
inline vuint8m1_t rvv_eltwise_bwd_relu_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float alpha, float beta, size_t vl) {
    if (alpha == 0.f) {
        vbool8_t vcond
                = __riscv_vmsgtu_vx_u8m1_b8(vin_s, static_cast<uint8_t>(0), vl);
        vuint8m1_t vzero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
        return __riscv_vmerge_vvm_u8m1(vzero, vin_dd, vcond, vl);
    } else {
        return rvv_convert_bwd_and_apply_f32_to_u8(
                vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_relu_f32_m4);
    }
}
// Square : return dd * 2 * s;
inline vfloat32m1_t rvv_eltwise_bwd_square_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat32m1_t vt = __riscv_vfmul_vv_f32m1(vin_dd, vin_s, vl);
    vfloat32m1_t v2 = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    return __riscv_vfmul_vv_f32m1(vt, v2, vl);
}
inline vint32m1_t rvv_eltwise_bwd_square_s32(vint32m1_t vin_dd,
        vint32m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vint32m1_t vt = __riscv_vmul_vv_i32m1(vin_dd, vin_s, vl);
    vint32m1_t v2 = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(2), vl);
    return __riscv_vmul_vv_i32m1(vt, v2, vl);
}
inline vint8m1_t rvv_eltwise_bwd_square_s8(vint8m1_t vin_dd, vint8m1_t vin_s,
        float /*alpha*/, float /*beta*/, size_t vl) {
    vint8m1_t vt = __riscv_vmul_vv_i8m1(vin_dd, vin_s, vl);
    vint8m1_t v2 = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(2), vl);
    return __riscv_vmul_vv_i8m1(vt, v2, vl);
}
inline vuint8m1_t rvv_eltwise_bwd_square_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float /*alpha*/, float /*beta*/, size_t vl) {
    vuint8m1_t vt = __riscv_vmul_vv_u8m1(vin_dd, vin_s, vl);
    vuint8m1_t v2 = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(2), vl);
    return __riscv_vmul_vv_u8m1(vt, v2, vl);
}
// Abs:  return s > 0 ? dd : s < 0 ? (U)-dd : (U)0;
inline vfloat32m1_t rvv_eltwise_bwd_abs_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vbool32_t vcond1 = __riscv_vmfgt_vf_f32m1_b32(vin_s, 0.f, vl);
    vbool32_t vcond2 = __riscv_vmflt_vf_f32m1_b32(vin_s, 0.f, vl);
    vfloat32m1_t vdd_neg = __riscv_vfneg_v_f32m1(vin_dd, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t vt = __riscv_vmerge_vvm_f32m1(vzero, vdd_neg, vcond2, vl);
    return __riscv_vmerge_vvm_f32m1(vt, vin_dd, vcond1, vl);
}
inline vint32m1_t rvv_eltwise_bwd_abs_s32(vint32m1_t vin_dd, vint32m1_t vin_s,
        float /*alpha*/, float /*beta*/, size_t vl) {
    vbool32_t vcond1
            = __riscv_vmsgt_vx_i32m1_b32(vin_s, static_cast<int32_t>(0), vl);
    vbool32_t vcond2
            = __riscv_vmslt_vx_i32m1_b32(vin_s, static_cast<int32_t>(0), vl);
    vint32m1_t vdd_neg = __riscv_vneg_v_i32m1(vin_dd, vl);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    vint32m1_t vt = __riscv_vmerge_vvm_i32m1(vzero, vdd_neg, vcond2, vl);
    return __riscv_vmerge_vvm_i32m1(vt, vin_dd, vcond1, vl);
}
inline vint8m1_t rvv_eltwise_bwd_abs_s8(vint8m1_t vin_dd, vint8m1_t vin_s,
        float /*alpha*/, float /*beta*/, size_t vl) {
    vbool8_t vcond1
            = __riscv_vmsgt_vx_i8m1_b8(vin_s, static_cast<int8_t>(0), vl);
    vbool8_t vcond2
            = __riscv_vmslt_vx_i8m1_b8(vin_s, static_cast<int8_t>(0), vl);
    vint8m1_t vdd_neg = __riscv_vneg_v_i8m1(vin_dd, vl);
    vint8m1_t vzero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    vint8m1_t vt = __riscv_vmerge_vvm_i8m1(vzero, vdd_neg, vcond2, vl);
    return __riscv_vmerge_vvm_i8m1(vt, vin_dd, vcond1, vl);
}
inline vuint8m1_t rvv_eltwise_bwd_abs_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float /*alpha*/, float /*beta*/, size_t vl) {
    // s == 0 ? 0 : dd;
    vbool8_t vcond
            = __riscv_vmseq_vx_u8m1_b8(vin_s, static_cast<uint8_t>(0), vl);
    vuint8m1_t vzero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vvm_u8m1(vin_dd, vzero, vcond, vl);
}
// Sqrt:  return (U)(dd / (2 * ::sqrtf((float)(s))));
inline vfloat32m1_t rvv_eltwise_bwd_sqrt_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat32m1_t vsqrt = __riscv_vfsqrt_v_f32m1(vin_s, vl);
    vfloat32m1_t v2 = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    vfloat32m1_t denom = __riscv_vfmul_vv_f32m1(vsqrt, v2, vl);
    return __riscv_vfdiv_vv_f32m1(vin_dd, denom, vl);
}
inline vfloat32m4_t rvv_eltwise_bwd_sqrt_f32_m4(vfloat32m4_t vin_dd,
        vfloat32m4_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat32m4_t vsqrt = __riscv_vfsqrt_v_f32m4(vin_s, vl);
    vfloat32m4_t v2 = __riscv_vfmv_v_f_f32m4(2.0f, vl);
    vfloat32m4_t denom = __riscv_vfmul_vv_f32m4(vsqrt, v2, vl);
    return __riscv_vfdiv_vv_f32m4(vin_dd, denom, vl);
}
inline vint32m1_t rvv_eltwise_bwd_sqrt_s32(vint32m1_t vin_dd, vint32m1_t vin_s,
        float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s32(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_sqrt_f32);
}
inline vint8m1_t rvv_eltwise_bwd_sqrt_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_sqrt_f32_m4);
}
inline vuint8m1_t rvv_eltwise_bwd_sqrt_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_u8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_sqrt_f32_m4);
}
// Linear:  return (U)(dd * alpha);
inline vfloat32m1_t rvv_eltwise_bwd_linear_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float alpha, float beta, size_t vl) {
    return __riscv_vfmul_vf_f32m1(vin_dd, alpha, vl);
}
inline vfloat32m4_t rvv_eltwise_bwd_linear_f32_m4(vfloat32m4_t vin_dd,
        vfloat32m4_t vin_s, float alpha, float beta, size_t vl) {
    return __riscv_vfmul_vf_f32m4(vin_dd, alpha, vl);
}
inline vint32m1_t rvv_eltwise_bwd_linear_s32(vint32m1_t vin_dd,
        vint32m1_t vin_s, float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s32(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_linear_f32);
}
inline vint8m1_t rvv_eltwise_bwd_linear_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_linear_f32_m4);
}
inline vuint8m1_t rvv_eltwise_bwd_linear_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_u8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_linear_f32_m4);
}
// Clip:  return dd * (alpha < s && s <= beta ? 1 : 0);
inline vfloat32m1_t rvv_eltwise_bwd_clip_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float alpha, float beta, size_t vl) {
    vbool32_t vcond = __riscv_vmfgt_vf_f32m1_b32(vin_s, alpha, vl);
    vbool32_t vcond2 = __riscv_vmfle_vf_f32m1_b32(vin_s, beta, vl);
    vcond = __riscv_vmand_mm_b32(vcond, vcond2, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.f, vl);
    vzero = __riscv_vmerge_vvm_f32m1(vzero, vone, vcond, vl);
    return __riscv_vfmul_vv_f32m1(vin_dd, vzero, vl);
}
inline vfloat32m4_t rvv_eltwise_bwd_clip_f32_m4(vfloat32m4_t vin_dd,
        vfloat32m4_t vin_s, float alpha, float beta, size_t vl) {
    vbool8_t vcond = __riscv_vmfgt_vf_f32m4_b8(vin_s, alpha, vl);
    vbool8_t vcond2 = __riscv_vmfle_vf_f32m4_b8(vin_s, beta, vl);
    vcond = __riscv_vmand_mm_b8(vcond, vcond2, vl);
    vfloat32m4_t vzero = __riscv_vfmv_v_f_f32m4(0.f, vl);
    vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.f, vl);
    vzero = __riscv_vmerge_vvm_f32m4(vzero, vone, vcond, vl);
    return __riscv_vfmul_vv_f32m4(vin_dd, vzero, vl);
}
inline vint32m1_t rvv_eltwise_bwd_clip_s32(vint32m1_t vin_dd, vint32m1_t vin_s,
        float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s32(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_clip_f32);
}
inline vint8m1_t rvv_eltwise_bwd_clip_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_s8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_clip_f32_m4);
}
inline vuint8m1_t rvv_eltwise_bwd_clip_u8(vuint8m1_t vin_dd, vuint8m1_t vin_s,
        float alpha, float beta, size_t vl) {
    return rvv_convert_bwd_and_apply_f32_to_u8(
            vin_dd, vin_s, alpha, beta, vl, rvv_eltwise_bwd_clip_f32_m4);
}
// HardSigmoid:  float v = alpha * s + beta; return v <= 0.f ? 0.f : v >= 1.f ? 0.f : dd * alpha;
inline vfloat32m1_t rvv_eltwise_bwd_hardsigmoid_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float alpha, float beta, size_t vl) {
    vfloat32m1_t vt = __riscv_vfmul_vf_f32m1(vin_s, alpha, vl);
    vt = __riscv_vfadd_vf_f32m1(vt, beta, vl);
    vbool32_t vcond = __riscv_vmfge_vf_f32m1_b32(vt, 1.f, vl);
    vfloat32m1_t vout = __riscv_vfmul_vf_f32m1(vin_dd, alpha, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vout = __riscv_vmerge_vvm_f32m1(vout, vzero, vcond, vl);
    vcond = __riscv_vmfle_vf_f32m1_b32(vt, 0.f, vl);
    return __riscv_vmerge_vvm_f32m1(vout, vzero, vcond, vl);
}
inline vint32m1_t rvv_eltwise_bwd_hardsigmoid_s32(vint32m1_t vin_dd,
        vint32m1_t vin_s, float alpha, float beta, size_t vl) {
    vint32m1_t vt
            = __riscv_vmul_vx_i32m1(vin_s, static_cast<int32_t>(alpha), vl);
    vt = __riscv_vadd_vx_i32m1(vt, static_cast<int32_t>(beta), vl);
    vbool32_t vcond
            = __riscv_vmsge_vx_i32m1_b32(vt, static_cast<int32_t>(1), vl);
    vint32m1_t vout
            = __riscv_vmul_vx_i32m1(vin_dd, static_cast<int32_t>(alpha), vl);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    vout = __riscv_vmerge_vvm_i32m1(vout, vzero, vcond, vl);
    vcond = __riscv_vmslt_vx_i32m1_b32(vt, static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vvm_i32m1(vout, vzero, vcond, vl);
}
inline vint8m1_t rvv_eltwise_bwd_hardsigmoid_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    vint8m1_t vt = __riscv_vmul_vx_i8m1(vin_s, static_cast<int8_t>(alpha), vl);
    vt = __riscv_vadd_vx_i8m1(vt, static_cast<int8_t>(beta), vl);
    vbool8_t vcond = __riscv_vmsge_vx_i8m1_b8(vt, static_cast<int8_t>(1), vl);
    vint8m1_t vout = __riscv_vmul_vx_i8m1(vin_dd, alpha, vl);
    vint8m1_t vzero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    vout = __riscv_vmerge_vvm_i8m1(vout, vzero, vcond, vl);
    vcond = __riscv_vmslt_vx_i8m1_b8(vt, static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vvm_i8m1(vout, vzero, vcond, vl);
}
inline vuint8m1_t rvv_eltwise_bwd_hardsigmoid_u8(vuint8m1_t vin_dd,
        vuint8m1_t vin_s, float alpha, float beta, size_t vl) {
    vuint8m1_t vt
            = __riscv_vmul_vx_u8m1(vin_s, static_cast<uint8_t>(alpha), vl);
    vt = __riscv_vadd_vx_u8m1(vt, static_cast<uint8_t>(beta), vl);
    vbool8_t vcond = __riscv_vmsgeu_vx_u8m1_b8(vt, static_cast<uint8_t>(1), vl);
    vuint8m1_t vout
            = __riscv_vmul_vx_u8m1(vin_dd, static_cast<uint8_t>(alpha), vl);
    vuint8m1_t vzero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    vout = __riscv_vmerge_vvm_u8m1(vout, vzero, vcond, vl);
    vcond = __riscv_vmsltu_vx_u8m1_b8(vt, static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vvm_u8m1(vout, vzero, vcond, vl);
}
// HardSwish:  float v = alpha * s + beta; float w = 2.f * alpha * s + beta;
// return v <= 0.f ? 0.f : v >= 1.f ? dd : dd * w;
inline vfloat32m1_t rvv_eltwise_bwd_hardswish_f32(vfloat32m1_t vin_dd,
        vfloat32m1_t vin_s, float alpha, float beta, size_t vl) {
    vfloat32m1_t vt = __riscv_vfmul_vf_f32m1(vin_s, alpha, vl);
    vfloat32m1_t vout = __riscv_vfmul_vf_f32m1(vt, 2.f, vl);
    vt = __riscv_vfadd_vf_f32m1(vt, beta, vl);
    vout = __riscv_vfadd_vf_f32m1(vout, beta, vl);
    vout = __riscv_vfmul_vv_f32m1(vin_dd, vout, vl);
    vbool32_t vcond = __riscv_vmfge_vf_f32m1_b32(vt, 1.f, vl);
    vout = __riscv_vmerge_vvm_f32m1(vout, vin_dd, vcond, vl);
    vcond = __riscv_vmfle_vf_f32m1_b32(vt, 0.f, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vmerge_vvm_f32m1(vout, vzero, vcond, vl);
}
inline vint32m1_t rvv_eltwise_bwd_hardswish_s32(vint32m1_t vin_dd,
        vint32m1_t vin_s, float alpha, float beta, size_t vl) {
    vint32m1_t vt
            = __riscv_vmul_vx_i32m1(vin_s, static_cast<int32_t>(alpha), vl);
    vint32m1_t vout = __riscv_vmul_vx_i32m1(vt, static_cast<int32_t>(2), vl);
    vt = __riscv_vadd_vx_i32m1(vt, static_cast<int32_t>(beta), vl);
    vout = __riscv_vadd_vx_i32m1(vout, static_cast<int32_t>(beta), vl);
    vout = __riscv_vmul_vv_i32m1(vin_dd, vout, vl);
    vbool32_t vcond
            = __riscv_vmsge_vx_i32m1_b32(vt, static_cast<int32_t>(1), vl);
    vout = __riscv_vmerge_vvm_i32m1(vout, vin_dd, vcond, vl);
    vcond = __riscv_vmslt_vx_i32m1_b32(vt, static_cast<int32_t>(0), vl);
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vvm_i32m1(vout, vzero, vcond, vl);
}
inline vint8m1_t rvv_eltwise_bwd_hardswish_s8(
        vint8m1_t vin_dd, vint8m1_t vin_s, float alpha, float beta, size_t vl) {
    vint8m1_t vt = __riscv_vmul_vx_i8m1(vin_s, static_cast<int8_t>(alpha), vl);
    vint8m1_t vout = __riscv_vmul_vx_i8m1(vt, static_cast<int8_t>(2), vl);
    vt = __riscv_vadd_vx_i8m1(vt, static_cast<int8_t>(beta), vl);
    vout = __riscv_vadd_vx_i8m1(vout, static_cast<int8_t>(beta), vl);
    vout = __riscv_vmul_vv_i8m1(vin_dd, vout, vl);
    vbool8_t vcond = __riscv_vmsge_vx_i8m1_b8(vt, static_cast<int8_t>(1), vl);
    vout = __riscv_vmerge_vvm_i8m1(vout, vin_dd, vcond, vl);
    vcond = __riscv_vmslt_vx_i8m1_b8(vt, static_cast<int8_t>(0), vl);
    vint8m1_t vzero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vvm_i8m1(vout, vzero, vcond, vl);
}
inline vuint8m1_t rvv_eltwise_bwd_hardswish_u8(vuint8m1_t vin_dd,
        vuint8m1_t vin_s, float alpha, float beta, size_t vl) {
    vuint8m1_t vt
            = __riscv_vmul_vx_u8m1(vin_s, static_cast<uint8_t>(alpha), vl);
    vuint8m1_t vout = __riscv_vmul_vx_u8m1(vt, static_cast<uint8_t>(2), vl);
    vt = __riscv_vadd_vx_u8m1(vt, static_cast<uint8_t>(beta), vl);
    vout = __riscv_vadd_vx_u8m1(vout, static_cast<uint8_t>(beta), vl);
    vout = __riscv_vmul_vv_u8m1(vin_dd, vout, vl);
    vbool8_t vcond = __riscv_vmsgeu_vx_u8m1_b8(vt, static_cast<uint8_t>(1), vl);
    vout = __riscv_vmerge_vvm_u8m1(vout, vin_dd, vcond, vl);
    vcond = __riscv_vmsltu_vx_u8m1_b8(vt, static_cast<uint8_t>(0), vl);
    vuint8m1_t vzero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vvm_u8m1(vout, vzero, vcond, vl);
}

/*** Dispatch getters for backward pass ***/
inline eval_bwd_f32_fn_t get_eval_bwd_f32(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_bwd_relu_f32;
        case alg_kind::eltwise_square: return rvv_eltwise_bwd_square_f32;
        case alg_kind::eltwise_abs: return rvv_eltwise_bwd_abs_f32;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_bwd_sqrt_f32;
        case alg_kind::eltwise_linear: return rvv_eltwise_bwd_linear_f32;
        case alg_kind::eltwise_clip: return rvv_eltwise_bwd_clip_f32;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_bwd_hardsigmoid_f32;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_bwd_hardswish_f32;
        default: return nullptr;
    }
}
inline eval_bwd_s32_fn_t get_eval_bwd_s32(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_bwd_relu_s32;
        case alg_kind::eltwise_square: return rvv_eltwise_bwd_square_s32;
        case alg_kind::eltwise_abs: return rvv_eltwise_bwd_abs_s32;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_bwd_sqrt_s32;
        case alg_kind::eltwise_linear: return rvv_eltwise_bwd_linear_s32;
        case alg_kind::eltwise_clip: return rvv_eltwise_bwd_clip_s32;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_bwd_hardsigmoid_s32;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_bwd_hardswish_s32;
        default: return nullptr;
    }
}
inline eval_bwd_s8_fn_t get_eval_bwd_s8(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_bwd_relu_s8;
        case alg_kind::eltwise_square: return rvv_eltwise_bwd_square_s8;
        case alg_kind::eltwise_abs: return rvv_eltwise_bwd_abs_s8;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_bwd_sqrt_s8;
        case alg_kind::eltwise_linear: return rvv_eltwise_bwd_linear_s8;
        case alg_kind::eltwise_clip: return rvv_eltwise_bwd_clip_s8;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_bwd_hardsigmoid_s8;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_bwd_hardswish_s8;
        default: return nullptr;
    }
}
inline eval_bwd_u8_fn_t get_eval_bwd_u8(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_bwd_relu_u8;
        case alg_kind::eltwise_square: return rvv_eltwise_bwd_square_u8;
        case alg_kind::eltwise_abs: return rvv_eltwise_bwd_abs_u8;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_bwd_sqrt_u8;
        case alg_kind::eltwise_linear: return rvv_eltwise_bwd_linear_u8;
        case alg_kind::eltwise_clip: return rvv_eltwise_bwd_clip_u8;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_bwd_hardsigmoid_u8;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_bwd_hardswish_u8;
        default: return nullptr;
    }
}

/*** Apply methods for backward pass ***/
inline void rvv_eltwise_apply_bwd_f32(alg_kind_t alg, float *diff_src,
        const float *diff_dst, const float *src, dim_t len, float alpha,
        float beta) {
    auto eval = get_eval_bwd_f32(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_bwd_f32] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_bwd_kernel_f32(diff_src, diff_dst, src, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_bwd_s32(alg_kind_t alg, int32_t *diff_src,
        const int32_t *diff_dst, const int32_t *src, dim_t len, float alpha,
        float beta) {
    auto eval = get_eval_bwd_s32(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_bwd_s32] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_bwd_kernel_s32(diff_src, diff_dst, src, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_bwd_s8(alg_kind_t alg, int8_t *diff_src,
        const int8_t *diff_dst, const int8_t *src, dim_t len, float alpha,
        float beta) {
    auto eval = get_eval_bwd_s8(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_bwd_s8] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_bwd_kernel_s8(diff_src, diff_dst, src, len, alpha, beta, eval);
}
inline void rvv_eltwise_apply_bwd_u8(alg_kind_t alg, uint8_t *diff_src,
        const uint8_t *diff_dst, const uint8_t *src, dim_t len, float alpha,
        float beta) {
    auto eval = get_eval_bwd_u8(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_bwd_u8] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_bwd_kernel_u8(diff_src, diff_dst, src, len, alpha, beta, eval);
}

/* F16 integration with Zvfh extension */
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)

using eval_fwd_f16_fn_t = vfloat16m1_t (*)(vfloat16m1_t, float, float, size_t);
using eval_bwd_f16_fn_t
        = vfloat16m1_t (*)(vfloat16m1_t, vfloat16m1_t, float, float, size_t);

inline void rvv_eltwise_fwd_kernel_f16(const _Float16 *src, _Float16 *dst,
        dim_t len, float alpha, float beta, eval_fwd_f16_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e16m1(static_cast<size_t>(len - i));
        vfloat16m1_t vin = __riscv_vle16_v_f16m1(src + i, vl);
        vfloat16m1_t vout = eval(vin, alpha, beta, vl);
        __riscv_vse16_v_f16m1(dst + i, vout, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline vfloat16m1_t rvv_eltwise_fwd_relu_f16(
        vfloat16m1_t vin, float alpha, float /*beta*/, size_t vl) {
    if (alpha == 0.f) {
        vfloat16m1_t zero
                = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.f), vl);
        return __riscv_vfmax_vv_f16m1(vin, zero, vl);
    } else {
        vbool16_t p = __riscv_vmfgt_vf_f16m1_b16(
                vin, static_cast<_Float16>(0.f), vl);
        vfloat16m1_t vneg
                = __riscv_vfmul_vf_f16m1(vin, static_cast<_Float16>(alpha), vl);
        // if (x > 0) pick x, else pick alpha * x
        return __riscv_vmerge_vvm_f16m1(vneg, vin, p, vl);
    }
}
inline vfloat16m1_t rvv_eltwise_fwd_square_f16(
        vfloat16m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vfmul_vv_f16m1(vin, vin, vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_abs_f16(
        vfloat16m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat16m1_t vneg = __riscv_vfneg_v_f16m1(vin, vl);
    return __riscv_vfmax_vv_f16m1(vin, vneg, vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_sqrt_f16(
        vfloat16m1_t vin, float /*alpha*/, float /*beta*/, size_t vl) {
    return __riscv_vfsqrt_v_f16m1(vin, vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_linear_f16(
        vfloat16m1_t vin, float alpha, float beta, size_t vl) {
    vfloat16m1_t v
            = __riscv_vfmul_vf_f16m1(vin, static_cast<_Float16>(alpha), vl);
    return __riscv_vfadd_vf_f16m1(v, static_cast<_Float16>(beta), vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_clip_f16(
        vfloat16m1_t vin, float alpha, float beta, size_t vl) {
    vfloat16m1_t va = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(alpha), vl);
    vfloat16m1_t vb = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(beta), vl);
    vfloat16m1_t vmax = __riscv_vfmax_vv_f16m1(vin, va, vl);
    return __riscv_vfmin_vv_f16m1(vmax, vb, vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_hardsigmoid_f16(
        vfloat16m1_t vin, float alpha, float beta, size_t vl) {
    vfloat32m2_t vx = __riscv_vfwcvt_f_f_v_f32m2(vin, vl);
    vfloat32m2_t vlin = __riscv_vfmv_v_f_f32m2(beta, vl);
    vlin = __riscv_vfmacc_vf_f32m2(vlin, alpha, vx, vl);
    // clamp to [0,1] in f32
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vfloat32m2_t vone = __riscv_vfmv_v_f_f32m2(1.f, vl);
    vfloat32m2_t vmax = __riscv_vfmax_vv_f32m2(vlin, vzero, vl);
    vfloat32m2_t vclamp = __riscv_vfmin_vv_f32m2(vmax, vone, vl);
    return __riscv_vfncvt_f_f_w_f16m1(vclamp, vl);
}
inline vfloat16m1_t rvv_eltwise_fwd_hardswish_f16(
        vfloat16m1_t vin, float alpha, float beta, size_t vl) {
    vfloat32m2_t vx = __riscv_vfwcvt_f_f_v_f32m2(vin, vl);
    vfloat32m2_t vlin = __riscv_vfmv_v_f_f32m2(beta, vl);
    vlin = __riscv_vfmacc_vf_f32m2(vlin, alpha, vx, vl);
    // clamp to [0,1] in f32
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vfloat32m2_t vone = __riscv_vfmv_v_f_f32m2(1.f, vl);
    vfloat32m2_t vmax = __riscv_vfmax_vv_f32m2(vlin, vzero, vl);
    vfloat32m2_t vclamp = __riscv_vfmin_vv_f32m2(vmax, vone, vl);
    vfloat32m2_t vout = __riscv_vfmul_vv_f32m2(vx, vclamp, vl);
    return __riscv_vfncvt_f_f_w_f16m1(vout, vl);
}
inline void rvv_eltwise_bwd_kernel_f16(_Float16 *diff_src,
        const _Float16 *diff_dst, const _Float16 *src, dim_t len, float alpha,
        float beta, eval_bwd_f16_fn_t eval) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e16m1(static_cast<size_t>(len - i));
        vfloat16m1_t vdiff_dst = __riscv_vle16_v_f16m1(diff_dst + i, vl);
        vfloat16m1_t vsrc = __riscv_vle16_v_f16m1(src + i, vl);
        vfloat16m1_t vdiff_src = eval(vdiff_dst, vsrc, alpha, beta, vl);
        __riscv_vse16_v_f16m1(diff_src + i, vdiff_src, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline vfloat16m1_t rvv_eltwise_bwd_relu_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float alpha, float /*beta*/, size_t vl) {
    vbool16_t vcond
            = __riscv_vmfgt_vf_f16m1_b16(vin_s, static_cast<_Float16>(0.f), vl);
    vfloat16m1_t vt
            = __riscv_vfmul_vf_f16m1(vin_dd, static_cast<_Float16>(alpha), vl);
    return __riscv_vmerge_vvm_f16m1(vt, vin_dd, vcond, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_square_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat32m2_t vt = __riscv_vfwmul_vv_f32m2(vin_dd, vin_s, vl);
    vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
    vfloat32m2_t vo = __riscv_vfmul_vv_f32m2(vt, v2, vl);
    return __riscv_vfncvt_f_f_w_f16m1(vo, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_abs_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vbool16_t vcond1
            = __riscv_vmfgt_vf_f16m1_b16(vin_s, static_cast<_Float16>(0.f), vl);
    vbool16_t vcond2
            = __riscv_vmflt_vf_f16m1_b16(vin_s, static_cast<_Float16>(0.f), vl);
    vfloat16m1_t vdd_neg = __riscv_vfneg_v_f16m1(vin_dd, vl);
    vfloat16m1_t vzero = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.f), vl);
    vfloat16m1_t vt = __riscv_vmerge_vvm_f16m1(vzero, vdd_neg, vcond2, vl);
    return __riscv_vmerge_vvm_f16m1(vt, vin_dd, vcond1, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_sqrt_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float /*alpha*/, float /*beta*/, size_t vl) {
    vfloat16m1_t vsqrt = __riscv_vfsqrt_v_f16m1(vin_s, vl);
    vfloat16m1_t v2 = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(2.0f), vl);
    vfloat16m1_t denom = __riscv_vfmul_vv_f16m1(vsqrt, v2, vl);
    return __riscv_vfdiv_vv_f16m1(vin_dd, denom, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_linear_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float alpha, float beta, size_t vl) {
    return __riscv_vfmul_vf_f16m1(vin_dd, static_cast<_Float16>(alpha), vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_clip_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float alpha, float beta, size_t vl) {
    vbool16_t vcond = __riscv_vmfgt_vf_f16m1_b16(
            vin_s, static_cast<_Float16>(alpha), vl);
    vbool16_t vcond2 = __riscv_vmfle_vf_f16m1_b16(
            vin_s, static_cast<_Float16>(beta), vl);
    vcond = __riscv_vmand_mm_b16(vcond, vcond2, vl);
    vfloat16m1_t vzero = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.f), vl);
    vfloat16m1_t vone = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(1.f), vl);
    vzero = __riscv_vmerge_vvm_f16m1(vzero, vone, vcond, vl);
    return __riscv_vfmul_vv_f16m1(vin_dd, vzero, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_hardsigmoid_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float alpha, float beta, size_t vl) {
    vfloat32m2_t vs = __riscv_vfwcvt_f_f_v_f32m2(vin_s, vl);
    vfloat32m2_t vdd = __riscv_vfwcvt_f_f_v_f32m2(vin_dd, vl);
    vfloat32m2_t vt = __riscv_vfmv_v_f_f32m2(beta, vl);
    vt = __riscv_vfmacc_vf_f32m2(vt, alpha, vs, vl);
    vbool16_t vcond_hi = __riscv_vmfge_vf_f32m2_b16(vt, 1.f, vl);
    vfloat32m2_t vout = __riscv_vfmul_vf_f32m2(vdd, alpha, vl);
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vout = __riscv_vmerge_vvm_f32m2(vout, vzero, vcond_hi, vl);
    vbool16_t vcond_lo = __riscv_vmfle_vf_f32m2_b16(vt, 0.f, vl);
    vout = __riscv_vmerge_vvm_f32m2(vout, vzero, vcond_lo, vl);
    return __riscv_vfncvt_f_f_w_f16m1(vout, vl);
}
inline vfloat16m1_t rvv_eltwise_bwd_hardswish_f16(vfloat16m1_t vin_dd,
        vfloat16m1_t vin_s, float alpha, float beta, size_t vl) {
    vfloat32m2_t vs = __riscv_vfwcvt_f_f_v_f32m2(vin_s, vl);
    vfloat32m2_t vdd = __riscv_vfwcvt_f_f_v_f32m2(vin_dd, vl);
    vfloat32m2_t vt = __riscv_vfmv_v_f_f32m2(beta, vl);
    vt = __riscv_vfmacc_vf_f32m2(vt, alpha, vs, vl);
    vfloat32m2_t w = __riscv_vfmv_v_f_f32m2(beta, vl);
    w = __riscv_vfmacc_vf_f32m2(w, 2.f * alpha, vs, vl);
    vfloat32m2_t vout = __riscv_vfmul_vv_f32m2(vdd, w, vl);
    vbool16_t vcond_hi = __riscv_vmfge_vf_f32m2_b16(vt, 1.f, vl);
    vout = __riscv_vmerge_vvm_f32m2(vout, vdd, vcond_hi, vl);
    vbool16_t vcond_lo = __riscv_vmfle_vf_f32m2_b16(vt, 0.f, vl);
    vfloat32m2_t vzero = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vout = __riscv_vmerge_vvm_f32m2(vout, vzero, vcond_lo, vl);
    return __riscv_vfncvt_f_f_w_f16m1(vout, vl);
}

inline eval_fwd_f16_fn_t get_eval_fwd_f16(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_fwd_relu_f16;
        case alg_kind::eltwise_square: return rvv_eltwise_fwd_square_f16;
        case alg_kind::eltwise_abs: return rvv_eltwise_fwd_abs_f16;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_fwd_sqrt_f16;
        case alg_kind::eltwise_linear: return rvv_eltwise_fwd_linear_f16;
        case alg_kind::eltwise_clip: return rvv_eltwise_fwd_clip_f16;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_fwd_hardsigmoid_f16;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_fwd_hardswish_f16;
        default: return nullptr;
    }
}
inline eval_bwd_f16_fn_t get_eval_bwd_f16(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_relu: return rvv_eltwise_bwd_relu_f16;
        case alg_kind::eltwise_square: return rvv_eltwise_bwd_square_f16;
        case alg_kind::eltwise_abs: return rvv_eltwise_bwd_abs_f16;
        case alg_kind::eltwise_sqrt: return rvv_eltwise_bwd_sqrt_f16;
        case alg_kind::eltwise_linear: return rvv_eltwise_bwd_linear_f16;
        case alg_kind::eltwise_clip: return rvv_eltwise_bwd_clip_f16;
        case alg_kind::eltwise_hardsigmoid:
            return rvv_eltwise_bwd_hardsigmoid_f16;
        case alg_kind::eltwise_hardswish: return rvv_eltwise_bwd_hardswish_f16;
        default: return nullptr;
    }
}

#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)

inline void rvv_eltwise_apply_fwd_f16(alg_kind_t alg, const _Float16 *src,
        _Float16 *dst, dim_t len, float alpha, float beta) {
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    auto eval = get_eval_fwd_f16(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_fwd_f16] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_fwd_kernel_f16(src, dst, len, alpha, beta, eval);
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
}
inline void rvv_eltwise_apply_bwd_f16(alg_kind_t alg, _Float16 *diff_src,
        const _Float16 *diff_dst, const _Float16 *src, dim_t len, float alpha,
        float beta) {
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    auto eval = get_eval_bwd_f16(alg);
    if (!eval) {
        assert(!"[rvv_eltwise_apply_bwd_f16] unknown eltwise alg_kind");
        return;
    }
    rvv_eltwise_bwd_kernel_f16(diff_src, diff_dst, src, len, alpha, beta, eval);
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_ELTWISE_KERNELS_HPP