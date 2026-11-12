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

#include <algorithm>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

// Two grouped matmul patterns are supported, `*` is the variable dim:
// grouped src {G, *, K} x dense wei [G, K, N] -> grouped dst {G, *, N}
// grouped src {G, M, *} x grouped wei {G, *, N} -> dense dst [G, M, N]
status_t ref_grouped_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper bia_d(pd()->weights_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const bool is_2dby2d = pd()->is_2dby2d();

    const dim_t group_count = src_d.dims()[0];
    // M is total_M for 2Dx3D, K is total_K for 2Dx2D
    const dim_t M = pd()->M();
    const dim_t K = pd()->K();
    const dim_t N = pd()->N();

    // Layouts are plain, so K is walked with a stride
    const dim_t src_stride_k = src_d.strides()[2];
    const dim_t wei_stride_k = wei_d.strides()[1];

    const void *src_data = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
    const int32_t *src_offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const void *wei_data = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS, 0);
    const void *bias_data = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    void *dst_data = CTX_OUT_MEM(void *, DNNL_ARG_DST, 0);
    // Query second offsets: wei for 2Dx2D or dst for 2Dx3D
    const int32_t *partner_offsets = is_2dby2d
            ? CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 1)
            : CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto bia_dt = bia_d.data_type();
    const auto dst_dt = dst_d.data_type();

    const bool with_bias = pd()->with_bias();
    // int src follows ref_matmul_int8_t, the pd guarantees int wei for it
    const bool use_int_arithmetic = types::is_integral_dt(src_dt);

    // Scales and zero points are dense row-major tensors over the src
    // {G, M, K} and wei {G, K, N} dims in their masks, a mask covering the
    // variable dim indexes it over all groups
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const auto src_scale_dt = attr_scales.get_data_type(DNNL_ARG_SRC);
    const int src_scale_mask = attr_scales.get_mask(DNNL_ARG_SRC);
    const auto src_scale_group_m = attr_scales.get_group(DNNL_ARG_SRC, -2);
    const auto src_scale_group_k = attr_scales.get_group(DNNL_ARG_SRC, -1);
    memory_desc_t src_scale_md {};
    CHECK(attr_scales.get(DNNL_ARG_SRC).get_md(src_scale_md, *src_d.md_));
    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);

    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_dt = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
    const int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
    const auto wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);
    memory_desc_t wei_scale_md {};
    CHECK(attr_scales.get(DNNL_ARG_WEIGHTS).get_md(wei_scale_md, *wei_d.md_));
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);

    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_src_zps = !attr_zps.has_default_values(DNNL_ARG_SRC);
    const auto src_zp_dt = attr_zps.get_data_type(DNNL_ARG_SRC);
    const int src_zp_mask = attr_zps.get_mask(DNNL_ARG_SRC);
    const auto src_zp_group_m = attr_zps.get_group(DNNL_ARG_SRC, -2);
    const auto src_zp_group_k = attr_zps.get_group(DNNL_ARG_SRC, -1);
    memory_desc_t src_zp_md {};
    CHECK(attr_zps.get(DNNL_ARG_SRC).get_md(src_zp_md, *src_d.md_));
    const void *src_zps = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    const bool with_wei_zps = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_zp_dt = attr_zps.get_data_type(DNNL_ARG_WEIGHTS);
    const int wei_zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
    const auto wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
    memory_desc_t wei_zp_md {};
    CHECK(attr_zps.get(DNNL_ARG_WEIGHTS).get_md(wei_zp_md, *wei_d.md_));
    const void *wei_zps = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    // Finest K-group of scales and zero points, K when there are none
    dim_t k_block = K;
    for (const dim_t group_k : {src_scale_group_k, wei_scale_group_k,
                 src_zp_group_k, wei_zp_group_k})
        if (group_k > 1) k_block = math::gcd(k_block, group_k);

    const auto &po = pd()->attr()->post_ops_;

    // Validate runtime group offsets: the tensor split along with src and
    // grouped binary post-op tensors must share the src offsets, and 2Dx2D
    // groups must align with the coarsest scales K-group
    const dim_t src_total = src_d.grouped_total();
    const dim_t k_align
            = is_2dby2d ? std::max(src_scale_group_k, wei_scale_group_k) : 1;
    for (dim_t group_id = 0; group_id < group_count; ++group_id) {
        const dim_t start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
        const dim_t end = src_offsets[group_id];
        if (end < start || end > src_total || partner_offsets[group_id] != end
                || end % k_align != 0)
            return status::invalid_arguments;

        for (int po_idx = 0; po_idx < po.len(); ++po_idx) {
            const auto &entry = po.entry_[po_idx];
            if (!entry.is_binary()
                    || !memory_desc_wrapper(entry.binary.src1_desc)
                                .is_grouped_desc())
                continue;
            const int po_arg
                    = DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx) | DNNL_ARG_SRC_1;
            const auto bin_offsets = CTX_IN_MEM(const int32_t *, po_arg, 1);
            if (bin_offsets[group_id] != end) return status::invalid_arguments;
        }
    }

    // Parallelize over dst rows flattened across all groups to keep the work
    // balanced when group sizes are highly skewed (e.g., "hot" benchdnn
    // profile where one group holds almost all tokens)
    const dim_t total_rows = is_2dby2d ? group_count * M : M;
    parallel_nd(total_rows, [= COMPAT_THIS_CAPTURE](dim_t row) {
        // Logical indices of the elements: the group, then an index along
        // the variable dim that spans all groups. K and N are set below
        dims_t src_dims_idx {}, wei_dims_idx {}, dst_dims_idx {};
        dim_t group_id, k_start = 0, K_g = K;
        if (is_2dby2d) {
            // src {G, M, *} and wei {G, *, N} are split along K, dst is
            // [G, M, N]
            group_id = row / M;
            k_start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
            K_g = src_offsets[group_id] - k_start;
            src_dims_idx[1] = dst_dims_idx[1] = row % M;
        } else {
            // src {G, *, K} and dst {G, *, N} are split along M, wei is
            // [G, K, N]
            group_id = std::upper_bound(
                               src_offsets, src_offsets + group_count, row)
                    - src_offsets;
            // Rows past the last offset belong to no group and stay untouched
            if (group_id == group_count) return;
            src_dims_idx[1] = dst_dims_idx[1] = row;
        }
        src_dims_idx[0] = wei_dims_idx[0] = dst_dims_idx[0] = group_id;
        auto &src_k_dim = src_dims_idx[2];
        auto &wei_k_dim = wei_dims_idx[1];
        auto &wei_n_dim = wei_dims_idx[2];
        auto &dst_n_dim = dst_dims_idx[2];

        // Note, that K_g == 0 must still write zeros
        const dim_t k_group_size = std::min(k_block, K_g);
        const dim_t n_k_groups = K_g ? K_g / k_group_size : 0;

        for (dim_t n = 0; n < N; ++n) {
            wei_n_dim = dst_n_dim = n;
            float result = 0.0f;

            for (dim_t i_group = 0; i_group < n_k_groups; i_group++) {
                // Scales and zero points are the same across the K-group
                src_k_dim = wei_k_dim = k_start + i_group * k_group_size;

                int src_zp = 0, wei_zp = 0;
                float src_scale = 1.0f, wei_scale = 1.0f;
                if (with_src_zps) {
                    const dim_t src_zp_offset = matmul_helper_t::get_quant_off(
                            src_dims_idx, 3, src_zp_mask, src_zp_group_m,
                            src_zp_group_k, src_zp_md);
                    src_zp = io::load_int_value(
                            src_zp_dt, src_zps, src_zp_offset);
                }
                if (with_wei_zps) {
                    const dim_t wei_zp_offset = matmul_helper_t::get_quant_off(
                            wei_dims_idx, 3, wei_zp_mask, wei_zp_group_k,
                            wei_zp_group_n, wei_zp_md);
                    wei_zp = io::load_int_value(
                            wei_zp_dt, wei_zps, wei_zp_offset);
                }
                if (with_src_scales) {
                    const dim_t src_scale_offset
                            = matmul_helper_t::get_quant_off(src_dims_idx, 3,
                                    src_scale_mask, src_scale_group_m,
                                    src_scale_group_k, src_scale_md);
                    src_scale = io::load_float_value(
                            src_scale_dt, src_scales, src_scale_offset);
                }
                if (with_wei_scales) {
                    const dim_t wei_scale_offset
                            = matmul_helper_t::get_quant_off(wei_dims_idx, 3,
                                    wei_scale_mask, wei_scale_group_k,
                                    wei_scale_group_n, wei_scale_md);
                    wei_scale = io::load_float_value(
                            wei_scale_dt, wei_scales, wei_scale_offset);
                }

                const dim_t src_off = src_d.off_v(src_dims_idx);
                const dim_t wei_off = wei_d.off_v(wei_dims_idx);
                float acc = 0.0f;
                if (use_int_arithmetic) {
                    int acc_int = 0;
                    for (dim_t k = 0; k < k_group_size; ++k) {
                        const int s = io::load_int_value(
                                src_dt, src_data, src_off + k * src_stride_k);
                        const int w = io::load_int_value(
                                wei_dt, wei_data, wei_off + k * wei_stride_k);
                        acc_int += (s - src_zp) * (w - wei_zp);
                    }
                    acc = static_cast<float>(acc_int);
                } else {
                    // fp src with int wei (WOQ) dequantizes them first
                    for (dim_t k = 0; k < k_group_size; ++k) {
                        const float s = io::load_float_value(
                                src_dt, src_data, src_off + k * src_stride_k);
                        const float w = io::load_float_value(
                                wei_dt, wei_data, wei_off + k * wei_stride_k);
                        acc += s * (w - wei_zp);
                    }
                }
                result += acc * src_scale * wei_scale;
            }

            if (with_bias)
                result += io::load_float_value(
                        bia_dt, bias_data, bia_d.off(group_id, 0, n));

            // Post-Ops support: binary mul with grouped or dense
            // (incl. NVFP4 recipe with global scale) tensor, eltwise
            //
            // Note: ref_post_ops_t won't allow using grouped tensor (with their
            // own per-group offsets) in binary po
            int eltwise_idx = 0, binary_idx = 0;
            for (int po_idx = 0; po_idx < po.len(); ++po_idx) {
                const auto &entry = po.entry_[po_idx];
                if (entry.is_eltwise()) {
                    result = eltwise_po_[eltwise_idx++].compute_scalar(result);
                } else if (entry.is_binary()) {
                    const int po_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx)
                            | DNNL_ARG_SRC_1;
                    const auto bin_data = CTX_IN_MEM(const void *, po_arg);
                    const memory_desc_wrapper bin_d(entry.binary.src1_desc);
                    // src1 is indexed as dst and broadcast along its unit dims
                    dims_t bin_dims_idx {};
                    for (int d = 0; d < 3; ++d)
                        bin_dims_idx[d]
                                = bin_d.dims()[d] == 1 ? 0 : dst_dims_idx[d];
                    const float val = io::load_float_value(bin_d.data_type(),
                            bin_data, bin_d.off_v(bin_dims_idx));
                    result = binary_po_[binary_idx++].compute_scalar(
                            result, val, false);
                }
            }

            io::store_float_value(
                    dst_dt, result, dst_data, dst_d.off_v(dst_dims_idx));
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
