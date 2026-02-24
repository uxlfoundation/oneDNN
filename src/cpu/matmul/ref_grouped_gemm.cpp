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

#include "cpu/matmul/ref_grouped_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <atomic>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_grouped_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    // src: [total_tokens, K] grouped
    // wei: [num_experts, K, N] dense with abc or acb
    // dst: [total_tokens, N] grouped
    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t group_count = src_grouped.group_count;
    const dim_t K = wei_d.dims()[1];
    const dim_t N = wei_d.dims()[2];
    const dim_t total_M = src_d.dims()[0];

    const void *src_data = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
    const int32_t *src_offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const void *wei_data = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *dst_data = CTX_OUT_MEM(void *, DNNL_ARG_DST, 0);
    const int32_t *dst_offsets = CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto dst_dt = pd()->dst_md()->data_type;

    const bool with_bias = pd()->with_bias();
    const void *bias_data
            = with_bias ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS) : nullptr;
    const auto bia_dt
            = with_bias ? pd()->weights_md(1)->data_type : data_type::undef;

    // src scales: row-wise
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);

    // wei scales: column-wise or blocked (K grouping)
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_dt = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, 0);
    const auto wei_scale_ngroups_k
            = wei_scale_group_k > 1 ? K / wei_scale_group_k : 1;
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);

    // wei zero_points: column-wise or blocked (K grouping)
    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_wei_zps = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_zp_dt = attr_zps.get_data_type(DNNL_ARG_WEIGHTS);
    const void *wei_zps = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    // Number of groups for zps and scales must be the same
    const dim_t n_k_groups = wei_scale_ngroups_k;
    const dim_t k_group_size = K / n_k_groups;

    // Check if int arithmetic (int4/int8 src and wei)
    const bool use_int_arithmetic
            = utils::one_of(src_dt, data_type::s8, data_type::u8)
            && utils::one_of(wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4);

    // Check if WOQ (fp src + int wei with fpmath apply_to_int)
    const bool use_woq = utils::one_of(src_dt, data_type::f32, data_type::bf16,
                                 data_type::f16)
            && utils::one_of(wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4)
            && pd()->attr()->fpmath_.apply_to_int_;

    // Parallelize over groups (experts in MoE)
    // Expectation is to see 128-256+ groups, with varying M per group
    // and possibly some empty groups (M == 0)
    std::atomic<status_t> st(status::success);
    parallel_nd(group_count, [&](dim_t group_id) {
        const dim_t src_offset_start
                = (group_id == 0) ? 0 : src_offsets[group_id - 1];
        const dim_t src_offset_end = src_offsets[group_id];
        const dim_t dst_offset_start
                = (group_id == 0) ? 0 : dst_offsets[group_id - 1];
        const dim_t dst_offset_end = dst_offsets[group_id];

        // Validate offsets
        if (src_offset_start < 0 || src_offset_end > total_M
                || src_offset_end < src_offset_start || dst_offset_start < 0
                || dst_offset_end > total_M
                || dst_offset_end < dst_offset_start) {
            st = status::invalid_arguments;
            return;
        }

        const dim_t M = src_offset_end - src_offset_start;
        if (M == 0) return; // skip if no rows in this group

        const dim_t src_base_idx = src_offset_start * K;
        const dim_t dst_base_idx = dst_offset_start * N;
        const dim_t wei_group_base = group_id * K * N;
        const dim_t wei_stride_k = wei_d.blocking_desc().strides[1];
        const dim_t wei_stride_n = wei_d.blocking_desc().strides[2];

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t n = 0; n < N; ++n) {
                float result = 0.0f;

                // int wei path (either int src + int wei or fp src + int wei WOQ)
                // int src follows ref_matmul_int8_t
                // fp  src (WOQ) follows dequantize then multiply
                if (use_int_arithmetic || use_woq) {
                    for (dim_t i_group = 0; i_group < n_k_groups; i_group++) {
                        float wei_scale = 1.0f;
                        int wei_zp_val = 0;

                        if (with_wei_scales) {
                            const dim_t idx = group_id * n_k_groups * N
                                    + i_group * N + n;
                            wei_scale = io::load_float_value(
                                    wei_scale_dt, wei_scales, idx);
                        }
                        if (with_wei_zps) {
                            const dim_t idx = group_id * n_k_groups * N
                                    + i_group * N + n;
                            wei_zp_val = io::load_int_value(
                                    wei_zp_dt, wei_zps, idx);
                        }

                        float acc = 0.0f;
                        if (use_int_arithmetic) {
                            int acc_int = 0;
                            for (dim_t k = 0; k < k_group_size; ++k) {
                                const dim_t k_abs = k + i_group * k_group_size;
                                const dim_t src_idx
                                        = src_base_idx + m * K + k_abs;
                                const dim_t wei_idx = wei_group_base
                                        + k_abs * wei_stride_k
                                        + n * wei_stride_n;
                                const int s = io::load_int_value(
                                        src_dt, src_data, src_idx);
                                const int w = io::load_int_value(
                                        wei_dt, wei_data, wei_idx);
                                acc_int += s * (w - wei_zp_val);
                            }
                            acc = static_cast<float>(acc_int);
                        } else {
                            for (dim_t k = 0; k < k_group_size; ++k) {
                                const dim_t k_abs = k + i_group * k_group_size;
                                const dim_t src_idx
                                        = src_base_idx + m * K + k_abs;
                                const dim_t wei_idx = wei_group_base
                                        + k_abs * wei_stride_k
                                        + n * wei_stride_n;
                                const float s = io::load_float_value(
                                        src_dt, src_data, src_idx);
                                const int w_int = io::load_int_value(
                                        wei_dt, wei_data, wei_idx);
                                acc += s
                                        * static_cast<float>(
                                                w_int - wei_zp_val);
                            }
                        }

                        if (with_src_scales) {
                            const dim_t idx = src_offset_start + m;
                            const float src_scale = io::load_float_value(
                                    data_type::f32, src_scales, idx);
                            acc *= src_scale;
                        }

                        result += acc * wei_scale;
                    }
                } else {
                    // fp arithmetic
                    float acc = 0.0f;

                    for (dim_t k = 0; k < K; ++k) {
                        const dim_t src_idx = src_base_idx + m * K + k;
                        const dim_t wei_idx = wei_group_base + k * wei_stride_k
                                + n * wei_stride_n;

                        const float s = io::load_float_value(
                                src_dt, src_data, src_idx);
                        const float w = io::load_float_value(
                                wei_dt, wei_data, wei_idx);
                        acc += s * w;
                    }

                    if (with_src_scales) {
                        const dim_t idx = src_offset_start + m;
                        const float scale = io::load_float_value(
                                data_type::f32, src_scales, idx);
                        acc *= scale;
                    }

                    if (with_wei_scales) {
                        const dim_t idx = group_id * N + n;
                        const float scale = io::load_float_value(
                                wei_scale_dt, wei_scales, idx);
                        acc *= scale;
                    }

                    result = acc;
                }

                // Add bias
                if (with_bias) {
                    const dim_t bias_idx = group_id * N + n;
                    result += io::load_float_value(bia_dt, bias_data, bias_idx);
                }

                const dim_t dst_idx = dst_base_idx + m * N + n;
                io::store_float_value(dst_dt, result, dst_data, dst_idx);
            }
        }
    });

    return st;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
