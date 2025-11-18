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

#include <riscv_vector.h>

#include "cpu/rv64/rvv_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

dim_t get_ip_data_off(const memory_desc_wrapper &mdw, int ndims, dim_t mb,
        dim_t c, dim_t id, dim_t ih, dim_t iw) {
    switch (ndims) {
        case 5: return mdw.off(mb, c, id, ih, iw);
        case 4: return mdw.off(mb, c, ih, iw);
        case 3: return mdw.off(mb, c, iw);
        case 2: return mdw.off(mb, c);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

dim_t get_ip_weights_off(const memory_desc_wrapper &mdw, int ndims, dim_t oc,
        dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
    switch (ndims) {
        case 5: return mdw.off(oc, ic, kd, kh, kw);
        case 4: return mdw.off(oc, ic, kh, kw);
        case 3: return mdw.off(oc, ic, kw);
        case 2: return mdw.off(oc, ic);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

float compute_ip_rvv_fwd_f32_f32(
        const void *src, const void *weights, const dim_t len) {
    const float *x = reinterpret_cast<const float *>(src);
    const float *w = reinterpret_cast<const float *>(weights);

    float acc_scalar = 0.0f;
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(static_cast<size_t>(len - i));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vw = __riscv_vle32_v_f32m1(w + i, vl);
        vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vx, vw, vl);
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vzero, vl);
        float partial = __riscv_vfmv_f_s_f32m1_f32(vred);
        acc_scalar += partial;
        i += static_cast<dim_t>(vl);
    }
    return acc_scalar;
}

float compute_ip_rvv_fwd_s8_s8(
        const void *src, const void *weights, const dim_t len) {
    const int8_t *x = reinterpret_cast<const int8_t *>(src);
    const int8_t *w = reinterpret_cast<const int8_t *>(weights);

    int32_t acc_i32 = 0;
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vint8m1_t vx_b = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vw_b = __riscv_vle8_v_i8m1(w + i, vl);
        vint16m2_t vprod_i16 = __riscv_vwmul_vv_i16m2(vx_b, vw_b, vl);
        vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vsum32
                = __riscv_vwredsum_vs_i16m2_i32m1(vprod_i16, vzero32, vl);
        int32_t partial = __riscv_vmv_x_s_i32m1_i32(vsum32);
        acc_i32 += partial;
        i += static_cast<dim_t>(vl);
    }
    return static_cast<float>(acc_i32);
}

float compute_ip_rvv_fwd_u8_s8(
        const void *src, const void *weights, const dim_t len) {
    const uint8_t *x = reinterpret_cast<const uint8_t *>(src);
    const int8_t *w = reinterpret_cast<const int8_t *>(weights);

    int32_t acc_i32 = 0;
    for (dim_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(len - i));
        vuint8m1_t vx_b = __riscv_vle8_v_u8m1(x + i, vl);
        vint8m1_t vw_b = __riscv_vle8_v_i8m1(w + i, vl);
        // Mixed-sign widen multiply: u8 * s8 -> i16
        vint16m2_t vprod_i16 = __riscv_vwmulsu_vv_i16m2(vw_b, vx_b, vl);
        vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vsum32
                = __riscv_vwredsum_vs_i16m2_i32m1(vprod_i16, vzero32, vl);
        int32_t partial = __riscv_vmv_x_s_i32m1_i32(vsum32);
        acc_i32 += partial;
        i += static_cast<dim_t>(vl);
    }
    return static_cast<float>(acc_i32);
}

float compute_ip_rvv_fwd(const void *src_base, const void *wei_base,
        const dim_t len, const data_type_t src_dt, const data_type_t wei_dt) {
    float acc;
    switch (src_dt) {
        case data_type::f32:
            if (wei_dt == data_type::f32) {
                acc = compute_ip_rvv_fwd_f32_f32(src_base, wei_base, len);
            }
            break;
        case data_type::s8:
            if (wei_dt == data_type::s8) {
                acc = compute_ip_rvv_fwd_s8_s8(src_base, wei_base, len);
            }
            break;
        case data_type::u8:
            if (wei_dt == data_type::s8) {
                acc = compute_ip_rvv_fwd_u8_s8(src_base, wei_base, len);
            }
            break;
        default: assert(!"Unsupported src data type for RVV inner product");
    }
    return acc;
}

void finalize_ip_acc(float acc, const void *bias, void *dst,
        const data_type_t bias_dt, const data_type_t dst_dt) {
    if (bias) {
        switch (bias_dt) {
            case data_type::f32:
                acc += *reinterpret_cast<const float *>(bias);
                break;
            case data_type::s32:
                acc += static_cast<float>(
                        *reinterpret_cast<const int32_t *>(bias));
                break;
            case data_type::s8:
                acc += static_cast<float>(
                        *reinterpret_cast<const int8_t *>(bias));
                break;
            case data_type::u8:
                acc += static_cast<float>(
                        *reinterpret_cast<const uint8_t *>(bias));
                break;
            default:
                assert(!"Unsupported bias data type for RVV inner product");
        }
    }

    switch (dst_dt) {
        case data_type::f32: *reinterpret_cast<float *>(dst) = acc; break;
        case data_type::s32: {
            float clamped = fminf(fmaxf(acc, -2147483648.0f), 2147483647.0f);
            int32_t v = static_cast<int32_t>(lrintf(clamped));
            *reinterpret_cast<int32_t *>(dst) = v;
            break;
        }
        case data_type::s8: {
            float clamped = fminf(fmaxf(acc, -128.0f), 127.0f);
            int8_t v = static_cast<int8_t>(lrintf(clamped));
            *reinterpret_cast<int8_t *>(dst) = v;
            break;
        }
        case data_type::u8: {
            float clamped = fminf(fmaxf(acc, 0.0f), 255.0f);
            uint8_t v = static_cast<uint8_t>(lrintf(clamped));
            *reinterpret_cast<uint8_t *>(dst) = v;
            break;
        }
        default: assert(!"Unsupported dst data type for RVV inner product");
    }
}

} // namespace

status_t rvv_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    void *dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const int nd = pd()->ndims();
    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();

    const data_type_t src_dt = src_d.data_type();
    const data_type_t wei_dt = weights_d.data_type();
    const data_type_t dst_dt = dst_d.data_type();
    const data_type_t bias_dt = bias ? bias_d.data_type() : src_dt;

    const dim_t K = IC * KD * KH * KW;
    const size_t src_elt = types::data_type_size(src_dt);
    const size_t wei_elt = types::data_type_size(wei_dt);
    const size_t dst_elt = types::data_type_size(dst_dt);
    const size_t bia_elt = types::data_type_size(bias_dt);

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const int nthr = pd()->nthr_;
    const size_t src_pack_per_thread = (size_t)K * src_elt;
    const size_t wei_pack_per_thread = (size_t)K * wei_elt;

    parallel(nthr, [&](int ithr, int nthr) {
        void *src_pack_storage = scratchpad.get<void>(
                memory_tracking::names::key_iprod_src_reorder);
        void *wei_pack_storage = scratchpad.get<void>(
                memory_tracking::names::key_iprod_weights_reorder);
        char *src_pack_thread = src_pack_storage
                ? static_cast<char *>(src_pack_storage)
                        + (size_t)ithr * src_pack_per_thread
                : nullptr;
        char *wei_pack_thread = wei_pack_storage
                ? static_cast<char *>(wei_pack_storage)
                        + (size_t)ithr * wei_pack_per_thread
                : nullptr;

        for_nd(ithr, nthr, MB, OC, [&](dim_t mb, dim_t oc) {
            const dim_t src_off
                    = get_ip_data_off(src_d, nd, mb, /*c*/ 0, 0, 0, 0);
            const dim_t wei_off
                    = get_ip_weights_off(weights_d, nd, oc, /*ic*/ 0, 0, 0, 0);
            const dim_t dst_off = dst_d.off(mb, oc);

            const char *src_base
                    = static_cast<const char *>(src) + src_off * src_elt;
            const char *wei_base
                    = static_cast<const char *>(weights) + wei_off * wei_elt;
            void *dst_ptr = static_cast<char *>(dst) + dst_off * dst_elt;

            const void *bia_ptr = nullptr;
            if (bias) {
                const dim_t bia_off = bias_d.off(oc);
                bia_ptr = static_cast<const char *>(bias) + bia_off * bia_elt;
            }

            float acc = 0.0f;
            const dim_t so0 = get_ip_data_off(src_d, nd, mb, 0, 0, 0, 0);
            const dim_t so1 = get_ip_data_off(src_d, nd, mb, 1, 0, 0, 0);
            const dim_t wo0 = get_ip_weights_off(weights_d, nd, oc, 0, 0, 0, 0);
            const dim_t wo1 = get_ip_weights_off(weights_d, nd, oc, 1, 0, 0, 0);
            const ptrdiff_t src_bstride = (so1 == so0)
                    ? (ptrdiff_t)src_elt
                    : (so1 - so0) * (ptrdiff_t)src_elt;
            const ptrdiff_t wei_bstride = (wo1 == wo0)
                    ? (ptrdiff_t)wei_elt
                    : (wo1 - wo0) * (ptrdiff_t)wei_elt;

            const bool src_is_contig = (src_bstride == (ptrdiff_t)src_elt);
            const bool wei_is_contig = (wei_bstride == (ptrdiff_t)wei_elt);

            const void *src_ptr_for_compute = src_base;
            const void *wei_ptr_for_compute = wei_base;

            if (!src_is_contig) {
                assert(src_pack_thread != nullptr);
                char *src_pack = src_pack_thread;
                char *dstp = src_pack;
                const char *p = src_base;
                for (dim_t i = 0; i < K; ++i) {
                    utils::array_copy(reinterpret_cast<unsigned char *>(dstp),
                            reinterpret_cast<const unsigned char *>(p),
                            src_elt);
                    dstp += src_elt;
                    p += src_bstride;
                }
                src_ptr_for_compute = src_pack;
            }
            if (!wei_is_contig) {
                assert(wei_pack_thread != nullptr);
                char *wei_pack = wei_pack_thread;
                char *dstp = wei_pack;
                const char *p = wei_base;
                for (dim_t i = 0; i < K; ++i) {
                    utils::array_copy(reinterpret_cast<unsigned char *>(dstp),
                            reinterpret_cast<const unsigned char *>(p),
                            wei_elt);
                    dstp += wei_elt;
                    p += wei_bstride;
                }
                wei_ptr_for_compute = wei_pack;
            }

            acc = compute_ip_rvv_fwd(src_ptr_for_compute, wei_ptr_for_compute,
                    K, src_dt, wei_dt);

            finalize_ip_acc(acc, bia_ptr, dst_ptr, bias_dt, dst_dt);
        });
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
