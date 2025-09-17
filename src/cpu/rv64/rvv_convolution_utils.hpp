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

#ifndef CPU_RV64_RVV_CONVOLUTION_UTILS_HPP
#define CPU_RV64_RVV_CONVOLUTION_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// NHWC -> NCHW reorder for dst
inline void reorder_dst_nhwc_to_nchw(data_type_t dt, const void *src, void *dst,
        dim_t MB, dim_t OC, dim_t OH, dim_t OW) {
    const size_t esize = types::data_type_size(dt);
    const char *src_bytes = reinterpret_cast<const char *>(src);
    char *dst_bytes = reinterpret_cast<char *>(dst);
    parallel_nd(MB, OC, OH, OW, [&](dim_t n, dim_t c, dim_t h, dim_t w) {
        const size_t src_elem_off
                = (((static_cast<size_t>(n) * OH + h) * OW + w) * OC + c);
        const size_t dst_elem_off
                = (((static_cast<size_t>(n) * OC + c) * OH + h) * OW + w);
        std::memcpy(dst_bytes + dst_elem_off * esize,
                src_bytes + src_elem_off * esize, esize);
    });
}

// NCHW -> NHWC reorder for a single tensor
inline void reorder_src_nchw_to_nhwc(data_type_t dt, const void *src, void *dst,
        dim_t MB, dim_t IC, dim_t IH, dim_t IW) {
    const size_t esize = types::data_type_size(dt);
    const char *src_bytes = reinterpret_cast<const char *>(src);
    char *dst_bytes = reinterpret_cast<char *>(dst);
    const size_t hw_stride = static_cast<size_t>(IH) * static_cast<size_t>(IW);
    parallel_nd(MB, IH, IW, [&](dim_t n, dim_t h, dim_t w) {
        const size_t n_base_src = (static_cast<size_t>(n) * IC * IH * IW);
        const size_t n_hw_dst
                = (((static_cast<size_t>(n) * IH + h) * IW + w) * IC);
        for (dim_t c = 0; c < IC; ++c) {
            const size_t src_elem_off = n_base_src
                    + static_cast<size_t>(c) * hw_stride
                    + static_cast<size_t>(h) * IW + static_cast<size_t>(w);
            const size_t dst_elem_off = n_hw_dst + static_cast<size_t>(c);
            std::memcpy(dst_bytes + dst_elem_off * esize,
                    src_bytes + src_elem_off * esize, esize);
        }
    });
}

// Pack weights from OIHW to [OC][KH][KW][IC] (IC contiguous)
inline void pack_weights_oihw_to_oc_kh_kw_ic(data_type_t dt,
        const void *wei_oihw, void *wei_pack, dim_t OC, dim_t IC, dim_t KH,
        dim_t KW) {
    const size_t esize = types::data_type_size(dt);
    const char *src_bytes = reinterpret_cast<const char *>(wei_oihw);
    char *dst_bytes = reinterpret_cast<char *>(wei_pack);
    auto idx_oihw = [&](dim_t oc, dim_t ic, dim_t kh, dim_t kw) -> size_t {
        return static_cast<size_t>((((oc * IC) + ic) * KH + kh) * KW + kw);
    };
    parallel_nd(OC, KH, KW, [&](dim_t oc, dim_t kh, dim_t kw) {
        const size_t dst_panel_off
                = (((static_cast<size_t>(oc) * KH + kh) * KW + kw) * IC);
        for (dim_t ic = 0; ic < IC; ++ic) {
            const size_t src_elem_off = idx_oihw(oc, ic, kh, kw);
            const size_t dst_elem_off = dst_panel_off + static_cast<size_t>(ic);
            std::memcpy(dst_bytes + dst_elem_off * esize,
                    src_bytes + src_elem_off * esize, esize);
        }
    });
}

// Helpers for dtype branches
static inline const void *ptr_add_elems(
        const void *base, data_type_t dt, size_t elems) {
    using namespace data_type;
    if (dt == f32) return reinterpret_cast<const float *>(base) + elems;
    if (dt == s8) return reinterpret_cast<const int8_t *>(base) + elems;
    if (dt == u8) return reinterpret_cast<const uint8_t *>(base) + elems;
    if (dt == s32) return reinterpret_cast<const int32_t *>(base) + elems;
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    if (dt == f16) return reinterpret_cast<const _Float16 *>(base) + elems;
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    return base;
}

static inline void *ptr_add_elems_mut(
        void *base, data_type_t dt, size_t elems) {
    using namespace data_type;
    if (dt == f32) return reinterpret_cast<float *>(base) + elems;
    if (dt == s8) return reinterpret_cast<int8_t *>(base) + elems;
    if (dt == u8) return reinterpret_cast<uint8_t *>(base) + elems;
    if (dt == s32) return reinterpret_cast<int32_t *>(base) + elems;
#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    if (dt == f16) return reinterpret_cast<_Float16 *>(base) + elems;
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    return base;
}

static inline void pack_weights_dispatch(data_type_t dt, const void *wei_oihw,
        void *wei_pack, dim_t OC, dim_t IC, dim_t KH, dim_t KW) {
    pack_weights_oihw_to_oc_kh_kw_ic(dt, wei_oihw, wei_pack, OC, IC, KH, KW);
}

static inline void reorder_src_to_nhwc_dispatch(data_type_t dt,
        const void *src_ncsp, void *dst_nhwc, dim_t MB, dim_t IC, dim_t IH,
        dim_t IW) {
    reorder_src_nchw_to_nhwc(dt, src_ncsp, dst_nhwc, MB, IC, IH, IW);
}

static inline void reorder_dst_to_nchw_dispatch(data_type_t dt,
        const void *src_nhwc, void *dst_nchw, dim_t MB, dim_t OC, dim_t OH,
        dim_t OW) {
    reorder_dst_nhwc_to_nchw(dt, src_nhwc, dst_nchw, MB, OC, OH, OW);
}

// === Forward helpers ===
// Per-dtype inner-kernel: dot over IC with RVV intrinsics (f32 x f32 -> f32 accumulate)
inline float rvv_dot_ic_fwd_f32_f32(
        const float *sp, const float *wp, dim_t IC) {
    float out_scalar = 0.f;
    for (dim_t ic = 0; ic < IC;) {
        const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(IC - ic));
        vfloat32m1_t vin = __riscv_vle32_v_f32m1(sp + ic, vl);
        vfloat32m1_t wv = __riscv_vle32_v_f32m1(wp + ic, vl);
        vfloat32m1_t vmul = __riscv_vfmul_vv_f32m1(vin, wv, vl);
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vmul, vzero, vl);
        float chunk = __riscv_vfmv_f_s_f32m1_f32(vred);
        out_scalar += chunk;
        ic += static_cast<dim_t>(vl);
    }
    return out_scalar;
}

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
// Per-dtype inner-kernel: dot over IC with RVV intrinsics (f16 x f16 -> f32 accumulate)
inline float rvv_dot_ic_fwd_f16_f16(
        const _Float16 *sp, const _Float16 *wp, dim_t IC) {
    float out_scalar = 0.f;
    for (dim_t ic = 0; ic < IC;) {
        const size_t vl = __riscv_vsetvl_e16m1(static_cast<size_t>(IC - ic));
        vfloat16m1_t vin_h = __riscv_vle16_v_f16m1(sp + ic, vl);
        vfloat16m1_t wv_h = __riscv_vle16_v_f16m1(wp + ic, vl);
        // widen to f32 and multiply in f32
        vfloat32m2_t vin_f = __riscv_vfwcvt_f_f_v_f32m2(vin_h, vl);
        vfloat32m2_t wv_f = __riscv_vfwcvt_f_f_v_f32m2(wv_h, vl);
        vfloat32m2_t vmul_f = __riscv_vfmul_vv_f32m2(vin_f, wv_f, vl);
        // reduce to scalar f32
        vfloat32m1_t vzero_f = __riscv_vfmv_v_f_f32m1(0.f, 1);
        vfloat32m1_t vred_f
                = __riscv_vfredusum_vs_f32m2_f32m1(vmul_f, vzero_f, vl);
        float chunk = __riscv_vfmv_f_s_f32m1_f32(vred_f);
        out_scalar += chunk;
        ic += static_cast<dim_t>(vl);
    }
    return out_scalar;
}
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
// Per-dtype inner-kernel: dot over IC with RVV intrinsics (f32 x f16 -> f32 accumulate)
inline float rvv_dot_ic_fwd_f32_f16(
        const float *sp, const _Float16 *wp, dim_t IC) {
    float out_scalar = 0.f;
    for (dim_t ic = 0; ic < IC;) {
        const size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(IC - ic));
        // load f32 src chunk using m2
        vfloat32m2_t vin_f = __riscv_vle32_v_f32m2(sp + ic, vl);
        // load f16 weights and widen to f32
        vfloat16m1_t wv_h = __riscv_vle16_v_f16m1(wp + ic, vl);
        vfloat32m2_t wv_f = __riscv_vfwcvt_f_f_v_f32m2(wv_h, vl);
        // multiply in f32 and reduce to scalar f32
        vfloat32m2_t vmul_f = __riscv_vfmul_vv_f32m2(vin_f, wv_f, vl);
        vfloat32m1_t vzero_f = __riscv_vfmv_v_f_f32m1(0.f, 1);
        vfloat32m1_t vred_f
                = __riscv_vfredusum_vs_f32m2_f32m1(vmul_f, vzero_f, vl);
        float chunk = __riscv_vfmv_f_s_f32m1_f32(vred_f);
        out_scalar += chunk;
        ic += static_cast<dim_t>(vl);
    }
    return out_scalar;
}
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)

// Per-dtype inner-kernel: dot over IC with RVV intrinsics (s8 x s8 -> f32 accumulate)
inline float rvv_dot_ic_fwd_s8_s8(
        const int8_t *sp, const int8_t *wp, dim_t IC) {
    long long acc64 = 0;
    for (dim_t ic = 0; ic < IC;) {
        const size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(IC - ic));
        vint8m1_t va8 = __riscv_vle8_v_i8m1(sp + ic, vl);
        vint8m1_t vb8 = __riscv_vle8_v_i8m1(wp + ic, vl);
        // widen to i16
        vint16m2_t va16 = __riscv_vsext_vf2_i16m2(va8, vl);
        vint16m2_t vb16 = __riscv_vsext_vf2_i16m2(vb8, vl);
        // widen multiply to i32
        vint32m4_t vprod32 = __riscv_vwmul_vv_i32m4(va16, vb16, vl);
        // reduce i32 to scalar
        vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t vred32
                = __riscv_vredsum_vs_i32m4_i32m1(vprod32, vzero32, vl);
        int32_t chunk = __riscv_vmv_x_s_i32m1_i32(vred32);
        acc64 += static_cast<long long>(chunk);
        ic += static_cast<dim_t>(vl);
    }
    return static_cast<float>(acc64);
}

// Per-dtype inner-kernel: dot over IC with RVV intrinsics (u8 x s8 -> f32 accumulate)
inline float rvv_dot_ic_fwd_u8_s8(
        const uint8_t *sp, const int8_t *wp, dim_t IC) {
    long long acc64 = 0;
    for (dim_t ic = 0; ic < IC;) {
        const size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(IC - ic));
        vuint8m1_t vu8 = __riscv_vle8_v_u8m1(sp + ic, vl);
        vint8m1_t vi8 = __riscv_vle8_v_i8m1(wp + ic, vl);
        // widen to u16 / i16
        vuint16m2_t vu16 = __riscv_vzext_vf2_u16m2(vu8, vl);
        vint16m2_t vi16 = __riscv_vsext_vf2_i16m2(vi8, vl);
        // widen multiply (signed x unsigned) to i32
        vint32m4_t vprod32 = __riscv_vwmulsu_vv_i32m4(vi16, vu16, vl);
        // reduce i32 to scalar
        vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t vred32
                = __riscv_vredsum_vs_i32m4_i32m1(vprod32, vzero32, vl);
        int32_t chunk = __riscv_vmv_x_s_i32m1_i32(vred32);
        acc64 += static_cast<long long>(chunk);
        ic += static_cast<dim_t>(vl);
    }
    return static_cast<float>(acc64);
}

// Data-type dispatcher: choose dot kernel by src/wei data_types, return f32-accumulator
inline float compute_dot_ic_fwd(data_type_t sdt, data_type_t wdt,
        const void *sp, const void *wp, dim_t IC) {
    using namespace data_type;
    if (sdt == f32 && wdt == f32) {
        return rvv_dot_ic_fwd_f32_f32(static_cast<const float *>(sp),
                static_cast<const float *>(wp), IC);
    } else if (sdt == s8 && wdt == s8) {
        return rvv_dot_ic_fwd_s8_s8(static_cast<const int8_t *>(sp),
                static_cast<const int8_t *>(wp), IC);
    } else if (sdt == u8 && wdt == s8) {
        return rvv_dot_ic_fwd_u8_s8(static_cast<const uint8_t *>(sp),
                static_cast<const int8_t *>(wp), IC);
    }

#if defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    if (sdt == f16 && wdt == f16) {
        return rvv_dot_ic_fwd_f16_f16(static_cast<const _Float16 *>(sp),
                static_cast<const _Float16 *>(wp), IC);
    } else if (sdt == f32 && wdt == f16) {
        return rvv_dot_ic_fwd_f32_f16(static_cast<const float *>(sp),
                static_cast<const _Float16 *>(wp), IC);
    }
#endif // defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
    return 0.f;
}

// Helper: finalize convolution accumulator with bias, scales, post-ops (VL-aware), and dst scales
inline float finalize_conv_acc(float acc_dot, float bias_val,
        const float *src_scales, const float *wei_scales,
        const float *dst_scales, int wei_scale_mask, dim_t oc,
        const rvv_postops_t &postops_handler) {
    float scale = 1.0f;
    if (src_scales) scale *= src_scales[0];
    if (wei_scales) {
        const int wei_idx_mult = wei_scale_mask > 0;
        scale *= wei_scales[oc * wei_idx_mult];
    }
    float val = bias_val + acc_dot * scale;
    const size_t vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v = __riscv_vfmv_v_f_f32m1(val, vl);
    v = postops_handler.apply(v, vl);
    val = __riscv_vfmv_f_s_f32m1_f32(v);
    if (dst_scales) val *= dst_scales[0];
    return val;
}

// Saturating cast from float accumulator to destination scalar type
// _Float16 cast uses default narrowing
template <typename TDst>
inline TDst saturate_cast(float x) {
    return static_cast<TDst>(x);
}

template <>
inline int8_t saturate_cast<int8_t>(float x) {
    float r = std::nearbyint(x);
    if (r > 127.f) r = 127.f;
    if (r < -128.f) r = -128.f;
    return static_cast<int8_t>(r);
}

template <>
inline uint8_t saturate_cast<uint8_t>(float x) {
    float r = std::nearbyint(x);
    if (r > 255.f) r = 255.f;
    if (r < 0.f) r = 0.f;
    return static_cast<uint8_t>(r);
}

template <>
inline int32_t saturate_cast<int32_t>(float x) {
    // clamp to int32 range
    float maxv = static_cast<float>(std::numeric_limits<int32_t>::max());
    float minv = static_cast<float>(std::numeric_limits<int32_t>::min());
    float r = std::nearbyint(x);
    if (r > maxv) r = maxv;
    if (r < minv) r = minv;
    return static_cast<int32_t>(r);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif