/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#include <riscv_vector.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/rvv_group_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace {

inline void stats_reduction(
        const float *src, size_t len, double &sum_out, double &sumsq_out) {
    size_t vl_max = __riscv_vsetvlmax_e32m1();

    vfloat64m2_t v_sum0 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum1 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum2 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sum3 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);

    vfloat64m2_t v_sq0 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq1 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq2 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);
    vfloat64m2_t v_sq3 = __riscv_vfmv_v_f_f64m2(0.0, vl_max);

    size_t idx = 0;

    for (; idx + 4 * vl_max <= len; idx += 4 * vl_max) {
        vfloat32m1_t v_x0
                = __riscv_vle32_v_f32m1(src + idx + 0 * vl_max, vl_max);
        vfloat32m1_t v_x1
                = __riscv_vle32_v_f32m1(src + idx + 1 * vl_max, vl_max);
        vfloat32m1_t v_x2
                = __riscv_vle32_v_f32m1(src + idx + 2 * vl_max, vl_max);
        vfloat32m1_t v_x3
                = __riscv_vle32_v_f32m1(src + idx + 3 * vl_max, vl_max);

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

    while (idx < len) {
        size_t vl = __riscv_vsetvl_e32m1(len - idx);
        vfloat32m1_t v_x = __riscv_vle32_v_f32m1(src + idx, vl);
        v_sum_all = __riscv_vfwadd_wv_f64m2(v_sum_all, v_x, vl);
        v_sq_all = __riscv_vfwmacc_vv_f64m2(v_sq_all, v_x, v_x, vl);
        idx += vl;
    }

    vfloat64m1_t v_red_zero = __riscv_vfmv_v_f_f64m1(0.0, vl_max);
    sum_out = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m2_f64m1(v_sum_all, v_red_zero, vl_max));
    sumsq_out = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m2_f64m1(v_sq_all, v_red_zero, vl_max));
}

inline void norm_spatial_loop(const float *src, float *dst, size_t len,
        float mean_val, float inv_std_val, float gamma_val, float beta_val,
        bool use_scale, bool use_shift) {

    size_t vl_max = __riscv_vsetvlmax_e32m1();
    size_t idx = 0;

    vfloat32m1_t v_mean = __riscv_vfmv_v_f_f32m1(mean_val, vl_max);
    vfloat32m1_t v_inv_std = __riscv_vfmv_v_f_f32m1(inv_std_val, vl_max);
    vfloat32m1_t v_gamma = __riscv_vfmv_v_f_f32m1(gamma_val, vl_max);
    vfloat32m1_t v_beta = __riscv_vfmv_v_f_f32m1(beta_val, vl_max);

    for (; idx + 4 * vl_max <= len; idx += 4 * vl_max) {
        vfloat32m1_t v0 = __riscv_vle32_v_f32m1(src + idx + 0 * vl_max, vl_max);
        vfloat32m1_t v1 = __riscv_vle32_v_f32m1(src + idx + 1 * vl_max, vl_max);
        vfloat32m1_t v2 = __riscv_vle32_v_f32m1(src + idx + 2 * vl_max, vl_max);
        vfloat32m1_t v3 = __riscv_vle32_v_f32m1(src + idx + 3 * vl_max, vl_max);

        v0 = __riscv_vfsub_vv_f32m1(v0, v_mean, vl_max);
        v1 = __riscv_vfsub_vv_f32m1(v1, v_mean, vl_max);
        v2 = __riscv_vfsub_vv_f32m1(v2, v_mean, vl_max);
        v3 = __riscv_vfsub_vv_f32m1(v3, v_mean, vl_max);

        v0 = __riscv_vfmul_vv_f32m1(v0, v_inv_std, vl_max);
        v1 = __riscv_vfmul_vv_f32m1(v1, v_inv_std, vl_max);
        v2 = __riscv_vfmul_vv_f32m1(v2, v_inv_std, vl_max);
        v3 = __riscv_vfmul_vv_f32m1(v3, v_inv_std, vl_max);

        if (use_scale) {
            v0 = __riscv_vfmul_vv_f32m1(v0, v_gamma, vl_max);
            v1 = __riscv_vfmul_vv_f32m1(v1, v_gamma, vl_max);
            v2 = __riscv_vfmul_vv_f32m1(v2, v_gamma, vl_max);
            v3 = __riscv_vfmul_vv_f32m1(v3, v_gamma, vl_max);
        }
        if (use_shift) {
            v0 = __riscv_vfadd_vv_f32m1(v0, v_beta, vl_max);
            v1 = __riscv_vfadd_vv_f32m1(v1, v_beta, vl_max);
            v2 = __riscv_vfadd_vv_f32m1(v2, v_beta, vl_max);
            v3 = __riscv_vfadd_vv_f32m1(v3, v_beta, vl_max);
        }

        __riscv_vse32_v_f32m1(dst + idx + 0 * vl_max, v0, vl_max);
        __riscv_vse32_v_f32m1(dst + idx + 1 * vl_max, v1, vl_max);
        __riscv_vse32_v_f32m1(dst + idx + 2 * vl_max, v2, vl_max);
        __riscv_vse32_v_f32m1(dst + idx + 3 * vl_max, v3, vl_max);
    }

    while (idx < len) {
        size_t vl = __riscv_vsetvl_e32m1(len - idx);
        vfloat32m1_t v_x = __riscv_vle32_v_f32m1(src + idx, vl);

        v_x = __riscv_vfsub_vf_f32m1(v_x, mean_val, vl);
        v_x = __riscv_vfmul_vf_f32m1(v_x, inv_std_val, vl);

        if (use_scale) v_x = __riscv_vfmul_vf_f32m1(v_x, gamma_val, vl);
        if (use_shift) v_x = __riscv_vfadd_vf_f32m1(v_x, beta_val, vl);

        __riscv_vse32_v_f32m1(dst + idx, v_x, vl);
        idx += vl;
    }
}

} // namespace

status_t rvv_group_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    float *mean = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
    float *variance = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);

    const auto N = pd()->MB();
    const auto C = pd()->C();
    const auto D = pd()->D();
    const auto H = pd()->H();
    const auto W = pd()->W();
    const size_t SP = D * H * W;

    const auto G = pd()->desc()->groups;
    const auto eps = pd()->desc()->group_norm_epsilon;
    const auto calculate_stats = !pd()->stats_is_src();
    const auto save_stats = pd()->is_training();

    const auto C_PER_G = C / G;

    parallel_nd(N, G, [&](dim_t n, dim_t g) {
        dim_t c_start = g * C_PER_G;
        size_t group_off = n * C * SP + c_start * SP;
        size_t group_len = C_PER_G * SP;

        float v_mean = 0.0f;
        float v_var = 0.0f;

        if (calculate_stats) {
            double sum = 0.0;
            double sumsq = 0.0;
            stats_reduction(src + group_off, group_len, sum, sumsq);

            double mean_d = sum / (double)group_len;
            double var_d = sumsq / (double)group_len - mean_d * mean_d;

            if (var_d < 0) var_d = 0;

            v_mean = (float)mean_d;
            v_var = (float)var_d;

            if (save_stats) {
                size_t stat_off = n * G + g;
                mean[stat_off] = v_mean;
                variance[stat_off] = v_var;
            }
        } else {
            size_t stat_off = n * G + g;
            v_mean = mean[stat_off];
            v_var = variance[stat_off];
        }

        float inv_std = 1.0f / sqrtf(v_var + eps);

        for (dim_t c = 0; c < C_PER_G; ++c) {
            dim_t global_c = c_start + c;
            size_t channel_off = group_off + c * SP;

            float gamma = scale ? scale[global_c] : 1.0f;
            float beta = shift ? shift[global_c] : 0.0f;

            bool use_scale = (scale != nullptr);
            bool use_shift = (shift != nullptr);

            norm_spatial_loop(src + channel_off, dst + channel_off, SP, v_mean,
                    inv_std, gamma, beta, use_scale, use_shift);
        }
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
