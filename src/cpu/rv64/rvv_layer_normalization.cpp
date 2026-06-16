/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2026 SpacemiT Corporation
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

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

rvv_layer_normalization_fwd_t::rvv_layer_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd) {
    if (pd()->src_md()->data_type == data_type::f16) {
        const bool weights_f16
                = pd()->weights_md()->data_type == data_type::f16;
        f16_fused_kernel_.reset(new jit_rvv_layernorm_f16_fused_kernel_t(
                pd()->use_scale(), pd()->use_shift(), weights_f16));
    } else {
        fused_kernel_.reset(new jit_rvv_layernorm_fused_kernel_t(
                pd()->use_scale(), pd()->use_shift()));
        data_kernel_.reset(new jit_rvv_layernorm_data_kernel_t(
                pd()->use_scale(), pd()->use_shift()));
    }
}

status_t rvv_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

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

    if (pd()->src_md()->data_type == data_type::f16) {
        const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const void *scale = pd()->use_scale()
                ? CTX_IN_MEM(const void *, DNNL_ARG_SCALE)
                : nullptr;
        const void *shift = pd()->use_shift()
                ? CTX_IN_MEM(const void *, DNNL_ARG_SHIFT)
                : nullptr;
        const size_t dt_size = types::data_type_size(data_type::f16);

        parallel_nd(N, [&](dim_t n) {
            jit_rvv_layernorm_f16_fused_kernel_t::call_params_t p;
            p.src = reinterpret_cast<const char *>(src) + n * C * dt_size;
            p.dst = reinterpret_cast<char *>(dst) + n * C * dt_size;
            p.scale = scale;
            p.shift = shift;
            p.len = C;
            p.eps = eps;
            p.mean = save_stats ? mean_ptr + n : nullptr;
            p.variance = save_stats ? variance_ptr + n : nullptr;
            (*f16_fused_kernel_)(&p);
        });
        return status::success;
    }

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    if (!calculate_stats) {
        parallel_nd(N, [&](dim_t n) {
            const float *src_row = src + n * C;
            float *dst_row = dst + n * C;
            const float mean_val = mean_ptr[n];
            const float inv_std = 1.f / sqrtf(variance_ptr[n] + eps);

            jit_rvv_layernorm_data_kernel_t::call_params_t data_p;
            data_p.src = src_row;
            data_p.dst = dst_row;
            data_p.scale = scale;
            data_p.shift = shift;
            data_p.len = C;
            data_p.mean = mean_val;
            data_p.inv_std = inv_std;
            (*data_kernel_)(&data_p);
        });
        return status::success;
    }

    parallel_nd(N, [&](dim_t n) {
        const float *src_row = src + n * C;
        float *dst_row = dst + n * C;
        jit_rvv_layernorm_fused_kernel_t::call_params_t fused_p;
        fused_p.src = src_row;
        fused_p.dst = dst_row;
        fused_p.scale = scale;
        fused_p.shift = shift;
        fused_p.len = C;
        fused_p.eps = eps;
        fused_p.mean = save_stats ? mean_ptr + n : nullptr;
        fused_p.variance = save_stats ? variance_ptr + n : nullptr;
        (*fused_kernel_)(&fused_p);
    });
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
