/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <math.h>
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "cpu/rv64/rvv_layer_normalization.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace {

inline void stats_loop_unroll4(
        const float *src, dim_t C, double &sum_out, double &sumsq_out) {
    size_t vl_max = __riscv_vsetvlmax_e32m1();
    dim_t c = 0;
    vfloat64m2_t v_sum0 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum1 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum2 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum3 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq0 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq1 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq2 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq3 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);

    for (; c + 4 * vl_max <= (size_t)C; c += 4 * vl_max) {
        vfloat32m1_t v_x0 = __riscv_vle32_v_f32m1(src + c + 0 * vl_max, vl_max);
        vfloat32m1_t v_x1 = __riscv_vle32_v_f32m1(src + c + 1 * vl_max, vl_max);
        vfloat32m1_t v_x2 = __riscv_vle32_v_f32m1(src + c + 2 * vl_max, vl_max);
        vfloat32m1_t v_x3 = __riscv_vle32_v_f32m1(src + c + 3 * vl_max, vl_max);

        v_sum0 = __riscv_vfwadd_wv_f64m2(v_sum0, v_x0, vl_max);
        v_sum1 = __riscv_vfwadd_wv_f64m2(v_sum1, v_x1, vl_max);
        v_sum2 = __riscv_vfwadd_wv_f64m2(v_sum2, v_x2, vl_max);
        v_sum3 = __riscv_vfwadd_wv_f64m2(v_sum3, v_x3, vl_max);

        v_sq0 = __riscv_vfwmacc_vv_f64m2(v_sq0, v_x0, v_x0, vl_max);
        v_sq1 = __riscv_vfwmacc_vv_f64m2(v_sq1, v_x1, v_x1, vl_max);
        v_sq2 = __riscv_vfwmacc_vv_f64m2(v_sq2, v_x2, v_x2, vl_max);
        v_sq3 = __riscv_vfwmacc_vv_f64m2(v_sq3, v_x3, v_x3, vl_max);
    }

    vfloat64m2_t v_sum_all = __riscv_vfadd_vv_f64m2(
            __riscv_vfadd_vv_f64m2(v_sum0, v_sum1, vl_max),
            __riscv_vfadd_vv_f64m2(v_sum2, v_sum3, vl_max), vl_max);
    vfloat64m2_t v_sq_all = __riscv_vfadd_vv_f64m2(
            __riscv_vfadd_vv_f64m2(v_sq0, v_sq1, vl_max),
            __riscv_vfadd_vv_f64m2(v_sq2, v_sq3, vl_max), vl_max);

    while (c < C) {
        size_t vl = __riscv_vsetvl_e32m1(C - c);
        vfloat32m1_t v_x = __riscv_vle32_v_f32m1(src + c, vl);
        v_sum_all = __riscv_vfwadd_wv_f64m2(v_sum_all, v_x, vl);
        v_sq_all = __riscv_vfwmacc_vv_f64m2(v_sq_all, v_x, v_x, vl);
        c += vl;
    }

    vfloat64m1_t v_red_zero = __riscv_vfmv_v_f_f64m1(0.0, vl_max);
    sum_out = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m2_f64m1(v_sum_all, v_red_zero, vl_max));
    sumsq_out = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m2_f64m1(v_sq_all, v_red_zero, vl_max));
}

inline void norm_loop_unroll4(const float *src, float *dst, dim_t C,
        const float *scale, const float *shift, float mean_val,
        float inv_std_val) {
    size_t vl_max = __riscv_vsetvlmax_e32m1();
    dim_t c = 0;
    vfloat32m1_t v_inv_std = __riscv_vfmv_v_f_f32m1(inv_std_val, vl_max);
    vfloat32m1_t v_mean = __riscv_vfmv_v_f_f32m1(mean_val, vl_max);

    for (; c + 4 * vl_max <= (size_t)C; c += 4 * vl_max) {
        vfloat32m1_t v0 = __riscv_vle32_v_f32m1(src + c + 0 * vl_max, vl_max);
        vfloat32m1_t v1 = __riscv_vle32_v_f32m1(src + c + 1 * vl_max, vl_max);
        vfloat32m1_t v2 = __riscv_vle32_v_f32m1(src + c + 2 * vl_max, vl_max);
        vfloat32m1_t v3 = __riscv_vle32_v_f32m1(src + c + 3 * vl_max, vl_max);

        v0 = __riscv_vfsub_vv_f32m1(v0, v_mean, vl_max);
        v1 = __riscv_vfsub_vv_f32m1(v1, v_mean, vl_max);
        v2 = __riscv_vfsub_vv_f32m1(v2, v_mean, vl_max);
        v3 = __riscv_vfsub_vv_f32m1(v3, v_mean, vl_max);

        v0 = __riscv_vfmul_vv_f32m1(v0, v_inv_std, vl_max);
        v1 = __riscv_vfmul_vv_f32m1(v1, v_inv_std, vl_max);
        v2 = __riscv_vfmul_vv_f32m1(v2, v_inv_std, vl_max);
        v3 = __riscv_vfmul_vv_f32m1(v3, v_inv_std, vl_max);

        if (scale) {
            vfloat32m1_t s0
                    = __riscv_vle32_v_f32m1(scale + c + 0 * vl_max, vl_max);
            vfloat32m1_t s1
                    = __riscv_vle32_v_f32m1(scale + c + 1 * vl_max, vl_max);
            vfloat32m1_t s2
                    = __riscv_vle32_v_f32m1(scale + c + 2 * vl_max, vl_max);
            vfloat32m1_t s3
                    = __riscv_vle32_v_f32m1(scale + c + 3 * vl_max, vl_max);
            if (shift) {
                vfloat32m1_t h0
                        = __riscv_vle32_v_f32m1(shift + c + 0 * vl_max, vl_max);
                vfloat32m1_t h1
                        = __riscv_vle32_v_f32m1(shift + c + 1 * vl_max, vl_max);
                vfloat32m1_t h2
                        = __riscv_vle32_v_f32m1(shift + c + 2 * vl_max, vl_max);
                vfloat32m1_t h3
                        = __riscv_vle32_v_f32m1(shift + c + 3 * vl_max, vl_max);
                v0 = __riscv_vfmacc_vv_f32m1(h0, s0, v0, vl_max);
                v1 = __riscv_vfmacc_vv_f32m1(h1, s1, v1, vl_max);
                v2 = __riscv_vfmacc_vv_f32m1(h2, s2, v2, vl_max);
                v3 = __riscv_vfmacc_vv_f32m1(h3, s3, v3, vl_max);
            } else {
                v0 = __riscv_vfmul_vv_f32m1(v0, s0, vl_max);
                v1 = __riscv_vfmul_vv_f32m1(v1, s1, vl_max);
                v2 = __riscv_vfmul_vv_f32m1(v2, s2, vl_max);
                v3 = __riscv_vfmul_vv_f32m1(v3, s3, vl_max);
            }
        } else if (shift) {
            vfloat32m1_t h0
                    = __riscv_vle32_v_f32m1(shift + c + 0 * vl_max, vl_max);
            vfloat32m1_t h1
                    = __riscv_vle32_v_f32m1(shift + c + 1 * vl_max, vl_max);
            vfloat32m1_t h2
                    = __riscv_vle32_v_f32m1(shift + c + 2 * vl_max, vl_max);
            vfloat32m1_t h3
                    = __riscv_vle32_v_f32m1(shift + c + 3 * vl_max, vl_max);
            v0 = __riscv_vfadd_vv_f32m1(v0, h0, vl_max);
            v1 = __riscv_vfadd_vv_f32m1(v1, h1, vl_max);
            v2 = __riscv_vfadd_vv_f32m1(v2, h2, vl_max);
            v3 = __riscv_vfadd_vv_f32m1(v3, h3, vl_max);
        }

        __riscv_vse32_v_f32m1(dst + c + 0 * vl_max, v0, vl_max);
        __riscv_vse32_v_f32m1(dst + c + 1 * vl_max, v1, vl_max);
        __riscv_vse32_v_f32m1(dst + c + 2 * vl_max, v2, vl_max);
        __riscv_vse32_v_f32m1(dst + c + 3 * vl_max, v3, vl_max);
    }

    while (c < C) {
        size_t vl = __riscv_vsetvl_e32m1(C - c);
        vfloat32m1_t v_x = __riscv_vfmul_vf_f32m1(
                __riscv_vfsub_vf_f32m1(
                        __riscv_vle32_v_f32m1(src + c, vl), mean_val, vl),
                inv_std_val, vl);
        if (scale) {
            vfloat32m1_t v_s = __riscv_vle32_v_f32m1(scale + c, vl);
            if (shift)
                v_x = __riscv_vfmacc_vv_f32m1(
                        __riscv_vle32_v_f32m1(shift + c, vl), v_s, v_x, vl);
            else
                v_x = __riscv_vfmul_vv_f32m1(v_x, v_s, vl);
        } else if (shift) {
            v_x = __riscv_vfadd_vv_f32m1(
                    v_x, __riscv_vle32_v_f32m1(shift + c, vl), vl);
        }
        __riscv_vse32_v_f32m1(dst + c, v_x, vl);
        c += vl;
    }
}
} // namespace

status_t rvv_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    float *mean_ptr = nullptr;
    float *variance_ptr = nullptr;
    auto scratchpad = ctx.get_scratchpad_grantor();
    if (pd()->use_tmp_stats()) {
        mean_ptr = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance_ptr = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean_ptr = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance_ptr = pd()->stats_are_src()
                ? const_cast<float *>(
                          CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    parallel_nd(N, [&](dim_t n) {
        const float *src_row = src + n * C;
        float *dst_row = dst + n * C;
        float v_mean, v_var;

        if (!calculate_stats) {
            v_mean = mean_ptr[n];
            v_var = variance_ptr[n];
        } else {
            double sum = 0.0, sumsq = 0.0;
            stats_loop_unroll4(src_row, C, sum, sumsq);
            double mean_d = sum / (double)C;
            double var_d = sumsq / (double)C - mean_d * mean_d;
            if (var_d < 0 && var_d > -1e-12) var_d = 0.0;
            v_mean = (float)mean_d;
            v_var = (float)var_d;
            if (save_stats) {
                mean_ptr[n] = v_mean;
                variance_ptr[n] = v_var;
            }
        }
        norm_loop_unroll4(src_row, dst_row, C, scale, shift, v_mean,
                1.f / sqrtf(v_var + eps));
    });
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
