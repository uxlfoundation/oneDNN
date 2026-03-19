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

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/rvv_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

rvv_layer_normalization_fwd_t::rvv_layer_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd) {
    fused_kernel_.reset(new jit_rvv_layernorm_fused_kernel_t(
            pd()->use_scale(), pd()->use_shift()));
    data_kernel_.reset(new jit_rvv_layernorm_data_kernel_t(
            pd()->use_scale(), pd()->use_shift()));
}

status_t rvv_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
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
