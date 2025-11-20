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

#ifndef CPU_RV64_RVV_BINARY_KERNELS_HPP
#define CPU_RV64_RVV_BINARY_KERNELS_HPP

#include <math.h>
#include <vector>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using eval_f32m1_t = vfloat32m1_t (*)(vfloat32m1_t, vfloat32m1_t, size_t);
using eval_f32m4_t = vfloat32m4_t (*)(vfloat32m4_t, vfloat32m4_t, size_t);
using eval_s32m1_t = vint32m1_t (*)(vint32m1_t, vint32m1_t, size_t);
using eval_s8m1_t = vint8m1_t (*)(vint8m1_t, vint8m1_t, size_t);
using eval_u8m1_t = vuint8m1_t (*)(vuint8m1_t, vuint8m1_t, size_t);

/*** Kernel methods ***/
static inline void rvv_binary_kernel_f32(const void *x_base, const void *y_base,
        void *dst_base, const int8_t * /*c*/, dim_t len, eval_f32m1_t eval,
        const data_type_t dt) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        const float *x = reinterpret_cast<const float *>(
                static_cast<const char *>(x_base)
                + i * types::data_type_size(dt));
        const float *y = reinterpret_cast<const float *>(
                static_cast<const char *>(y_base)
                + i * types::data_type_size(dt));
        float *dst = reinterpret_cast<float *>(
                static_cast<char *>(dst_base) + i * types::data_type_size(dt));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y, vl);
        vfloat32m1_t vd = eval(vx, vy, vl);
        __riscv_vse32_v_f32m1(dst, vd, vl);
        i += static_cast<dim_t>(vl);
    }
}
static inline void rvv_binary_kernel_s32(const void *x_base, const void *y_base,
        void *dst_base, const int8_t * /*c*/, dim_t len, eval_s32m1_t eval,
        const data_type_t dt) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        const int32_t *x = reinterpret_cast<const int32_t *>(
                static_cast<const char *>(x_base)
                + i * types::data_type_size(dt));
        const int32_t *y = reinterpret_cast<const int32_t *>(
                static_cast<const char *>(y_base)
                + i * types::data_type_size(dt));
        int32_t *dst = reinterpret_cast<int32_t *>(
                static_cast<char *>(dst_base) + i * types::data_type_size(dt));
        vint32m1_t vx = __riscv_vle32_v_i32m1(x, vl);
        vint32m1_t vy = __riscv_vle32_v_i32m1(y, vl);
        vint32m1_t vd = eval(vx, vy, vl);
        __riscv_vse32_v_i32m1(dst, vd, vl);
        i += static_cast<dim_t>(vl);
    }
}
static inline void rvv_binary_kernel_s8(const void *x_base, const void *y_base,
        void *dst_base, const int8_t * /*c*/, dim_t len, eval_s8m1_t eval,
        const data_type_t dt) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        const int8_t *x = reinterpret_cast<const int8_t *>(
                static_cast<const char *>(x_base)
                + i * types::data_type_size(dt));
        const int8_t *y = reinterpret_cast<const int8_t *>(
                static_cast<const char *>(y_base)
                + i * types::data_type_size(dt));
        int8_t *dst = reinterpret_cast<int8_t *>(
                static_cast<char *>(dst_base) + i * types::data_type_size(dt));
        vint8m1_t vx = __riscv_vle8_v_i8m1(x, vl);
        vint8m1_t vy = __riscv_vle8_v_i8m1(y, vl);
        vint8m1_t vd = eval(vx, vy, vl);
        __riscv_vse8_v_i8m1(dst, vd, vl);
        i += static_cast<dim_t>(vl);
    }
}
static inline void rvv_binary_kernel_u8(const void *x_base, const void *y_base,
        void *dst_base, const int8_t * /*c*/, dim_t len, eval_u8m1_t eval,
        const data_type_t dt) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        const uint8_t *x = reinterpret_cast<const uint8_t *>(
                static_cast<const char *>(x_base)
                + i * types::data_type_size(dt));
        const uint8_t *y = reinterpret_cast<const uint8_t *>(
                static_cast<const char *>(y_base)
                + i * types::data_type_size(dt));
        uint8_t *dst = reinterpret_cast<uint8_t *>(
                static_cast<char *>(dst_base) + i * types::data_type_size(dt));
        vuint8m1_t vx = __riscv_vle8_v_u8m1(x, vl);
        vuint8m1_t vy = __riscv_vle8_v_u8m1(y, vl);
        vuint8m1_t vd = eval(vx, vy, vl);
        __riscv_vse8_v_u8m1(dst, vd, vl);
        i += static_cast<dim_t>(vl);
    }
}

/*** Convert methods for f16/s32/s8/u8 and apply in f32 domain ***/
inline vint32m1_t rvv_convert_and_apply_f32_to_s32(
        vint32m1_t x, vint32m1_t y, size_t vl, eval_f32m1_t eval) {
    vfloat32m1_t vx = __riscv_vfcvt_f_x_v_f32m1(x, vl);
    vfloat32m1_t vy = __riscv_vfcvt_f_x_v_f32m1(y, vl);
    vfloat32m1_t vout_f32 = eval(vx, vy, vl);
    vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(-2147483648.0f, vl);
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(2147483647.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m1(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m1(vout_f32, vmax, vl);
    return __riscv_vfcvt_x_f_v_i32m1(vout_f32, vl);
}
inline vint8m1_t rvv_convert_and_apply_f32_to_s8(
        vint8m1_t x, vint8m1_t y, size_t vl, eval_f32m4_t eval) {
    vint32m4_t vx_s32 = __riscv_vsext_vf4_i32m4(x, vl);
    vint32m4_t vy_s32 = __riscv_vsext_vf4_i32m4(y, vl);
    vfloat32m4_t vx_f32 = __riscv_vfcvt_f_x_v_f32m4(vx_s32, vl);
    vfloat32m4_t vy_f32 = __riscv_vfcvt_f_x_v_f32m4(vy_s32, vl);
    vfloat32m4_t vout_f32 = eval(vx_f32, vy_f32, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(-128.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(127.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m4(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m4(vout_f32, vmax, vl);
    vint32m4_t vout_s32 = __riscv_vfcvt_x_f_v_i32m4(vout_f32, vl);
    vint16m2_t vout_s16 = __riscv_vncvt_x_x_w_i16m2(vout_s32, vl);
    return __riscv_vncvt_x_x_w_i8m1(vout_s16, vl);
}

inline vuint8m1_t rvv_convert_and_apply_f32_to_u8(
        vuint8m1_t x, vuint8m1_t y, size_t vl, eval_f32m4_t eval) {
    vuint32m4_t vx_u32 = __riscv_vzext_vf4_u32m4(x, vl);
    vuint32m4_t vy_u32 = __riscv_vzext_vf4_u32m4(y, vl);
    vfloat32m4_t vx_f32 = __riscv_vfcvt_f_xu_v_f32m4(vx_u32, vl);
    vfloat32m4_t vy_f32 = __riscv_vfcvt_f_xu_v_f32m4(vy_u32, vl);
    vfloat32m4_t vout_f32 = eval(vx_f32, vy_f32, vl);
    vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(255.0f, vl);
    vout_f32 = __riscv_vfmax_vv_f32m4(vout_f32, vmin, vl);
    vout_f32 = __riscv_vfmin_vv_f32m4(vout_f32, vmax, vl);
    vuint32m4_t vout_u32 = __riscv_vfcvt_xu_f_v_u32m4(vout_f32, vl);
    vuint16m2_t vout_u16 = __riscv_vncvt_x_x_w_u16m2(vout_u32, vl);
    return __riscv_vncvt_x_x_w_u8m1(vout_u16, vl);
}

/*** Operations (evaluate in f32 domain) ***/
// Add
inline vfloat32m1_t rvv_binary_add_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfadd_vv_f32m1(x, y, vl);
}
inline vint32m1_t rvv_binary_add_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return __riscv_vadd_vv_i32m1(x, y, vl);
}
inline vint8m1_t rvv_binary_add_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return __riscv_vadd_vv_i8m1(x, y, vl);
}
inline vuint8m1_t rvv_binary_add_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    return __riscv_vadd_vv_u8m1(x, y, vl);
}
// Div
inline vfloat32m1_t rvv_binary_div_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfdiv_vv_f32m1(x, y, vl);
}
inline vfloat32m4_t rvv_binary_div_f32_m4(
        vfloat32m4_t x, vfloat32m4_t y, size_t vl) {
    return __riscv_vfdiv_vv_f32m4(x, y, vl);
}
inline vint32m1_t rvv_binary_div_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_s32(x, y, vl, rvv_binary_div_f32);
}
inline vint8m1_t rvv_binary_div_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_s8(x, y, vl, rvv_binary_div_f32_m4);
}
inline vuint8m1_t rvv_binary_div_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_u8(x, y, vl, rvv_binary_div_f32_m4);
}
// Max
inline vfloat32m1_t rvv_binary_max_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfmax_vv_f32m1(x, y, vl);
}
inline vint32m1_t rvv_binary_max_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return __riscv_vmax_vv_i32m1(x, y, vl);
}
inline vint8m1_t rvv_binary_max_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return __riscv_vmax_vv_i8m1(x, y, vl);
}
inline vuint8m1_t rvv_binary_max_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    return __riscv_vmaxu_vv_u8m1(x, y, vl);
}
// Min
inline vfloat32m1_t rvv_binary_min_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfmin_vv_f32m1(x, y, vl);
}
inline vint32m1_t rvv_binary_min_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return __riscv_vmin_vv_i32m1(x, y, vl);
}
inline vint8m1_t rvv_binary_min_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return __riscv_vmin_vv_i8m1(x, y, vl);
}
inline vuint8m1_t rvv_binary_min_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    return __riscv_vminu_vv_u8m1(x, y, vl);
}
// Mul
inline vfloat32m1_t rvv_binary_mul_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfmul_vv_f32m1(x, y, vl);
}
inline vfloat32m4_t rvv_binary_mul_f32_m4(
        vfloat32m4_t x, vfloat32m4_t y, size_t vl) {
    return __riscv_vfmul_vv_f32m4(x, y, vl);
}
inline vint32m1_t rvv_binary_mul_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_s32(x, y, vl, rvv_binary_mul_f32);
}
inline vint8m1_t rvv_binary_mul_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_s8(x, y, vl, rvv_binary_mul_f32_m4);
}
inline vuint8m1_t rvv_binary_mul_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    return rvv_convert_and_apply_f32_to_u8(x, y, vl, rvv_binary_mul_f32_m4);
}
// Sub
inline vfloat32m1_t rvv_binary_sub_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    return __riscv_vfsub_vv_f32m1(x, y, vl);
}
inline vfloat32m4_t rvv_binary_sub_f32_m4(
        vfloat32m4_t x, vfloat32m4_t y, size_t vl) {
    return __riscv_vfsub_vv_f32m4(x, y, vl);
}
inline vint32m1_t rvv_binary_sub_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    return __riscv_vsub_vv_i32m1(x, y, vl);
}
inline vint8m1_t rvv_binary_sub_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    return __riscv_vsub_vv_i8m1(x, y, vl);
}
inline vuint8m1_t rvv_binary_sub_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    // Compute in f32 domain and clamp to [0, 255] to avoid unsigned wrap-around
    return rvv_convert_and_apply_f32_to_u8(x, y, vl, rvv_binary_sub_f32_m4);
}
// Ge
inline vfloat32m1_t rvv_binary_ge_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmfge_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_ge_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmsge_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_ge_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsge_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_ge_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsgeu_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}
// Gt
inline vfloat32m1_t rvv_binary_gt_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmfgt_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_gt_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmsgt_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_gt_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsgt_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_gt_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsgtu_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}
// Le
inline vfloat32m1_t rvv_binary_le_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmfle_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_le_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmsle_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_le_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsle_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_le_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsleu_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}
// Lt
inline vfloat32m1_t rvv_binary_lt_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_lt_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmslt_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_lt_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmslt_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_lt_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsltu_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}
// Eq
inline vfloat32m1_t rvv_binary_eq_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmfeq_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_eq_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmseq_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_eq_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmseq_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_eq_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmseq_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}
// Ne
inline vfloat32m1_t rvv_binary_ne_f32(
        vfloat32m1_t x, vfloat32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmfne_vv_f32m1_b32(x, y, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
    return __riscv_vfmerge_vfm_f32m1(zero, 1.f, mask, vl);
}
inline vint32m1_t rvv_binary_ne_s32(vint32m1_t x, vint32m1_t y, size_t vl) {
    vbool32_t mask = __riscv_vmsne_vv_i32m1_b32(x, y, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(static_cast<int32_t>(0), vl);
    return __riscv_vmerge_vxm_i32m1(zero, static_cast<int32_t>(1), mask, vl);
}
inline vint8m1_t rvv_binary_ne_s8(vint8m1_t x, vint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsne_vv_i8m1_b8(x, y, vl);
    vint8m1_t zero = __riscv_vmv_v_x_i8m1(static_cast<int8_t>(0), vl);
    return __riscv_vmerge_vxm_i8m1(zero, static_cast<int8_t>(1), mask, vl);
}
inline vuint8m1_t rvv_binary_ne_u8(vuint8m1_t x, vuint8m1_t y, size_t vl) {
    vbool8_t mask = __riscv_vmsne_vv_u8m1_b8(x, y, vl);
    vuint8m1_t zero = __riscv_vmv_v_x_u8m1(static_cast<uint8_t>(0), vl);
    return __riscv_vmerge_vxm_u8m1(zero, static_cast<uint8_t>(1), mask, vl);
}

/*** Select op needs three oprands: x, y, c mask. Do it specially without op dispatching ***/
inline void rvv_binary_select_kernel_f32(const float *x, const float *y,
        float *dst, const int8_t *c, dim_t len) {
    for (dim_t i = 0; i < len;) {
        // Set VL for f32m4 ops and compute vector length in elements
        size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(len - i));
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
        // Mask
        vint8m1_t vc8 = __riscv_vle8_v_i8m1(c + i, vl);
        vbool8_t mask
                = __riscv_vmsne_vx_i8m1_b8(vc8, static_cast<int8_t>(0), vl);
        vfloat32m4_t vsel = __riscv_vmerge_vvm_f32m4(vy, vx, mask, vl);
        __riscv_vse32_v_f32m4(dst + i, vsel, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_binary_select_kernel_s32(const int32_t *x, const int32_t *y,
        int32_t *dst, const int8_t *c, dim_t len) {
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(len - i));
        vint32m4_t vx32 = __riscv_vle32_v_i32m4(x + i, vl);
        vint32m4_t vy32 = __riscv_vle32_v_i32m4(y + i, vl);
        vfloat32m4_t vx = __riscv_vfcvt_f_x_v_f32m4(vx32, vl);
        vfloat32m4_t vy = __riscv_vfcvt_f_x_v_f32m4(vy32, vl);
        // Mask
        vint8m1_t vc8 = __riscv_vle8_v_i8m1(c + i, vl);
        vbool8_t mask
                = __riscv_vmsne_vx_i8m1_b8(vc8, static_cast<int8_t>(0), vl);
        // Merge and clamp in f32
        vfloat32m4_t vsel = __riscv_vmerge_vvm_f32m4(vy, vx, mask, vl);
        vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(-2147483648.0f, vl);
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(2147483647.0f, vl);
        vsel = __riscv_vfmax_vv_f32m4(vsel, vmin, vl);
        vsel = __riscv_vfmin_vv_f32m4(vsel, vmax, vl);
        vint32m4_t vds32 = __riscv_vfcvt_x_f_v_i32m4(vsel, vl);
        __riscv_vse32_v_i32m4(dst + i, vds32, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_binary_select_kernel_s8(const int8_t *x, const int8_t *y,
        int8_t *dst, const int8_t *c, dim_t len) {
    for (dim_t i = 0; i < len;) {
        // Load e8m1
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vint8m1_t vxb = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vyb = __riscv_vle8_v_i8m1(y + i, vl);
        // Widen to f32m4
        vint32m4_t vxi = __riscv_vsext_vf4_i32m4(vxb, vl);
        vint32m4_t vyi = __riscv_vsext_vf4_i32m4(vyb, vl);
        vfloat32m4_t vx = __riscv_vfcvt_f_x_v_f32m4(vxi, vl);
        vfloat32m4_t vy = __riscv_vfcvt_f_x_v_f32m4(vyi, vl);
        // Mask
        vint8m1_t vc8 = __riscv_vle8_v_i8m1(c + i, vl);
        vbool8_t mask
                = __riscv_vmsne_vx_i8m1_b8(vc8, static_cast<int8_t>(0), vl);
        // Merge and clamp in f32
        vfloat32m4_t vsel = __riscv_vmerge_vvm_f32m4(vy, vx, mask, vl);
        vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(-128.0f, vl);
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(127.0f, vl);
        vsel = __riscv_vfmax_vv_f32m4(vsel, vmin, vl);
        vsel = __riscv_vfmin_vv_f32m4(vsel, vmax, vl);
        vint32m4_t vds32 = __riscv_vfcvt_x_f_v_i32m4(vsel, vl);
        vint16m2_t vds16 = __riscv_vncvt_x_x_w_i16m2(vds32, vl);
        vint8m1_t vds8 = __riscv_vncvt_x_x_w_i8m1(vds16, vl);
        // Store e8m1
        __riscv_vse8_v_i8m1(dst + i, vds8, vl);
        i += static_cast<dim_t>(vl);
    }
}
inline void rvv_binary_select_kernel_u8(const uint8_t *x, const uint8_t *y,
        uint8_t *dst, const int8_t *c, dim_t len) {
    for (dim_t i = 0; i < len;) {
        // Load e8m1
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vuint8m1_t vxb = __riscv_vle8_v_u8m1(x + i, vl);
        vuint8m1_t vyb = __riscv_vle8_v_u8m1(y + i, vl);
        // Widen to f32m4
        vuint32m4_t vxu = __riscv_vzext_vf4_u32m4(vxb, vl);
        vuint32m4_t vyu = __riscv_vzext_vf4_u32m4(vyb, vl);
        vfloat32m4_t vx = __riscv_vfcvt_f_xu_v_f32m4(vxu, vl);
        vfloat32m4_t vy = __riscv_vfcvt_f_xu_v_f32m4(vyu, vl);
        // Mask
        vint8m1_t vc8 = __riscv_vle8_v_i8m1(c + i, vl);
        vbool8_t mask
                = __riscv_vmsne_vx_i8m1_b8(vc8, static_cast<int8_t>(0), vl);
        // Merge and clamp
        vfloat32m4_t vsel = __riscv_vmerge_vvm_f32m4(vy, vx, mask, vl);
        vfloat32m4_t vmin = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(255.0f, vl);
        vsel = __riscv_vfmax_vv_f32m4(vsel, vmin, vl);
        vsel = __riscv_vfmin_vv_f32m4(vsel, vmax, vl);
        vuint32m4_t vdu32 = __riscv_vfcvt_xu_f_v_u32m4(vsel, vl);
        vuint16m2_t vdu16 = __riscv_vncvt_x_x_w_u16m2(vdu32, vl);
        vuint8m1_t vdu8 = __riscv_vncvt_x_x_w_u8m1(vdu16, vl);
        // Store e8m1
        __riscv_vse8_v_u8m1(dst + i, vdu8, vl);
        i += static_cast<dim_t>(vl);
    }
}

/*** Dispatch getters for different data types ***/
inline eval_f32m1_t get_eval_f32(const alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return rvv_binary_add_f32;
        case alg_kind::binary_div: return rvv_binary_div_f32;
        case alg_kind::binary_max: return rvv_binary_max_f32;
        case alg_kind::binary_min: return rvv_binary_min_f32;
        case alg_kind::binary_mul: return rvv_binary_mul_f32;
        case alg_kind::binary_sub: return rvv_binary_sub_f32;
        case alg_kind::binary_ge: return rvv_binary_ge_f32;
        case alg_kind::binary_gt: return rvv_binary_gt_f32;
        case alg_kind::binary_le: return rvv_binary_le_f32;
        case alg_kind::binary_lt: return rvv_binary_lt_f32;
        case alg_kind::binary_eq: return rvv_binary_eq_f32;
        case alg_kind::binary_ne: return rvv_binary_ne_f32;
        default: return nullptr;
    }
}
inline eval_s32m1_t get_eval_s32(const alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return rvv_binary_add_s32;
        case alg_kind::binary_div: return rvv_binary_div_s32;
        case alg_kind::binary_max: return rvv_binary_max_s32;
        case alg_kind::binary_min: return rvv_binary_min_s32;
        case alg_kind::binary_mul: return rvv_binary_mul_s32;
        case alg_kind::binary_sub: return rvv_binary_sub_s32;
        case alg_kind::binary_ge: return rvv_binary_ge_s32;
        case alg_kind::binary_gt: return rvv_binary_gt_s32;
        case alg_kind::binary_le: return rvv_binary_le_s32;
        case alg_kind::binary_lt: return rvv_binary_lt_s32;
        case alg_kind::binary_eq: return rvv_binary_eq_s32;
        case alg_kind::binary_ne: return rvv_binary_ne_s32;
        default: return nullptr;
    }
}
inline eval_s8m1_t get_eval_s8(const alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return rvv_binary_add_s8;
        case alg_kind::binary_div: return rvv_binary_div_s8;
        case alg_kind::binary_max: return rvv_binary_max_s8;
        case alg_kind::binary_min: return rvv_binary_min_s8;
        case alg_kind::binary_mul: return rvv_binary_mul_s8;
        case alg_kind::binary_sub: return rvv_binary_sub_s8;
        case alg_kind::binary_ge: return rvv_binary_ge_s8;
        case alg_kind::binary_gt: return rvv_binary_gt_s8;
        case alg_kind::binary_le: return rvv_binary_le_s8;
        case alg_kind::binary_lt: return rvv_binary_lt_s8;
        case alg_kind::binary_eq: return rvv_binary_eq_s8;
        case alg_kind::binary_ne: return rvv_binary_ne_s8;
        default: return nullptr;
    }
}
inline eval_u8m1_t get_eval_u8(const alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return rvv_binary_add_u8;
        case alg_kind::binary_div: return rvv_binary_div_u8;
        case alg_kind::binary_max: return rvv_binary_max_u8;
        case alg_kind::binary_min: return rvv_binary_min_u8;
        case alg_kind::binary_mul: return rvv_binary_mul_u8;
        case alg_kind::binary_sub: return rvv_binary_sub_u8;
        case alg_kind::binary_ge: return rvv_binary_ge_u8;
        case alg_kind::binary_gt: return rvv_binary_gt_u8;
        case alg_kind::binary_le: return rvv_binary_le_u8;
        case alg_kind::binary_lt: return rvv_binary_lt_u8;
        case alg_kind::binary_eq: return rvv_binary_eq_u8;
        case alg_kind::binary_ne: return rvv_binary_ne_u8;
        default: return nullptr;
    }
}

/*** Apply methods ***/
static inline void rvv_binary_apply_f32(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    if (alg == alg_kind::binary_select) {
        rvv_binary_select_kernel_f32(reinterpret_cast<const float *>(x),
                reinterpret_cast<const float *>(y),
                reinterpret_cast<float *>(dst), c, len);
        return;
    }
    auto eval = get_eval_f32(alg);
    if (!eval) {
        assert(!"[rvv_binary_apply_f32] unknown binary alg_kind");
        return;
    }
    rvv_binary_kernel_f32(x, y, dst, c, len, eval, dt);
}
static inline void rvv_binary_apply_s32(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    if (alg == alg_kind::binary_select) {
        rvv_binary_select_kernel_s32(reinterpret_cast<const int32_t *>(x),
                reinterpret_cast<const int32_t *>(y),
                reinterpret_cast<int32_t *>(dst), c, len);
        return;
    }
    auto eval = get_eval_s32(alg);
    if (!eval) {
        assert(!"[rvv_binary_apply_s32] unknown binary alg_kind");
        return;
    }
    rvv_binary_kernel_s32(x, y, dst, c, len, eval, dt);
}
static inline void rvv_binary_apply_s8(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    if (alg == alg_kind::binary_select) {
        rvv_binary_select_kernel_s8(reinterpret_cast<const int8_t *>(x),
                reinterpret_cast<const int8_t *>(y),
                reinterpret_cast<int8_t *>(dst), c, len);
        return;
    }
    auto eval = get_eval_s8(alg);
    if (!eval) {
        assert(!"[rvv_binary_apply_s8] unknown binary alg_kind");
        return;
    }
    rvv_binary_kernel_s8(x, y, dst, c, len, eval, dt);
}
static inline void rvv_binary_apply_u8(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    if (alg == alg_kind::binary_select) {
        rvv_binary_select_kernel_u8(reinterpret_cast<const uint8_t *>(x),
                reinterpret_cast<const uint8_t *>(y),
                reinterpret_cast<uint8_t *>(dst), c, len);
        return;
    }
    auto eval = get_eval_u8(alg);
    if (!eval) {
        assert(!"[rvv_binary_apply_u8] unknown binary alg_kind");
        return;
    }
    rvv_binary_kernel_u8(x, y, dst, c, len, eval, dt);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_BINARY_KERNELS_HPP
