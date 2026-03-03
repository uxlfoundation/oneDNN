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

#include <algorithm>
#include <float.h>

#include "utils/parallel.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

int64_t wei_ab_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->k + k) * prb->n + n;
}
int64_t wei_ba_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->n + n) * prb->k + k;
}

// Stores parameters that are invariant across different (mc, nc) chunks
// Precomputed once per problem to reduce overhead
struct chunk_params_t {
    // Pointers to memory objects (no ownership here)
    const dnn_mem_t *src_m = nullptr, *wei_m = nullptr, *bia_m = nullptr,
                    *dst_m = nullptr;
    const dnn_mem_t *src_scales = nullptr, *wei_scales = nullptr,
                    *dst_scales = nullptr;
    const dnn_mem_t *src_zps = nullptr, *wei_zps = nullptr, *dst_zps = nullptr;
    const dnn_mem_t *dropout_mask = nullptr;

    // Problem dims
    int64_t dst_M_group = 1, dst_N_group = 1;

    // Feature flags
    bool has_src_scale = false, has_wei_scale = false, has_dst_scale = false;
    bool has_dst_dynamic = false, has_dst_mx = false,
         has_dst_dynamic_fp = false;
    bool has_src_zp = false, has_wei_zp = false, has_dst_zp = false;
    bool has_src_single_scale = false, has_wei_single_scale = false;
    bool has_src_single_zp = false, has_wei_single_zp = false;

    // Quantization masks
    int src_scale_mask = 0, wei_scale_mask = 0, dst_scale_mask = 0;
    int src_zp_mask = 0, wei_zp_mask = 0, dst_zp_mask = 0;

    // Scale/zp group vectors
    std::vector<int64_t> src_scale_groups, wei_scale_groups;
    std::vector<int64_t> src_zp_groups, wei_zp_groups;
    std::vector<int64_t> dst_scale_groups;

    // K-grouping related
    int64_t smallest_k_group = 0, n_k_groups = 0;

    // Dst scale storage type
    dnnl_data_type_t dst_scale_dt;

    // Post-ops element masks (one pair per post-op entry).
    std::vector<std::pair<int, int>> v_po_masks;

    // Pre-fetched single-value quant params (valid when has_*_single_* is true).
    int src_zp_single = 0, wei_zp_single = 0;
    float src_scale_single = 1.f, wei_scale_single = 1.f;

    // Data types
    dnnl_data_type_t bia_dt, dst_dt;
};

// Precompute parameters for compute_ref_matmul_chunk
// based on the problem definition and the provided arguments
static chunk_params_t make_chunk_params(const prb_t *prb, const args_t &args) {
    chunk_params_t p;

    p.src_m = &args.find(DNNL_ARG_SRC);
    p.wei_m = &args.find(DNNL_ARG_WEIGHTS);
    p.bia_m = &args.find(DNNL_ARG_BIAS);
    p.dst_m = &args.find(DNNL_ARG_DST);
    p.src_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    p.wei_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    p.dst_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    p.src_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    p.wei_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    p.dst_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    p.dropout_mask = &args.find(DNNL_ARG_ATTR_DROPOUT_MASK);

    p.has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    p.has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    p.has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    p.has_dst_dynamic = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic();
    p.has_dst_mx = prb->attr.scales.get(DNNL_ARG_DST).is_mx();
    p.has_dst_dynamic_fp = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic_fp();

    p.src_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, p.src_m->ndims());
    p.wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, p.wei_m->ndims());
    p.dst_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_DST, dnnl_matmul, p.dst_m->ndims());

    p.has_src_single_scale = p.has_src_scale && p.src_scale_mask == 0;
    p.has_wei_single_scale = p.has_wei_scale && p.wei_scale_mask == 0;

    p.has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    p.has_wei_zp = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    p.has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();

    p.src_zp_mask = p.has_src_zp ? prb->attr.zero_points.get_mask(DNNL_ARG_SRC,
                                           dnnl_matmul, p.src_m->ndims())
                                 : 0;
    p.wei_zp_mask = p.has_wei_zp
            ? prb->attr.zero_points.get_mask(
                      DNNL_ARG_WEIGHTS, dnnl_matmul, p.wei_m->ndims())
            : 0;
    p.dst_zp_mask = p.has_dst_zp ? prb->attr.zero_points.get_mask(DNNL_ARG_DST,
                                           dnnl_matmul, p.dst_m->ndims())
                                 : 0;

    p.has_src_single_zp = p.has_src_zp && p.src_zp_mask == 0;
    p.has_wei_single_zp = p.has_wei_zp && p.wei_zp_mask == 0;

    p.src_scale_groups = prb->attr.scales.get(DNNL_ARG_SRC).groups;
    p.wei_scale_groups = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
    p.src_zp_groups = prb->attr.zero_points.get(DNNL_ARG_SRC).groups;
    p.wei_zp_groups = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;

    const int64_t src_scale_group = !p.src_scale_groups.empty()
            ? p.src_scale_groups[1]
            : ((p.src_scale_mask >> (p.src_m->ndims() - 1)) % 2) > 0 ? 1
                                                                     : prb->k;
    const int64_t wei_scale_group = !p.wei_scale_groups.empty()
            ? p.wei_scale_groups[0]
            : ((p.wei_scale_mask >> (p.wei_m->ndims() - 2)) % 2) > 0 ? 1
                                                                     : prb->k;
    const int64_t src_zp_group = !p.src_zp_groups.empty() ? p.src_zp_groups[1]
            : ((p.src_zp_mask >> (p.src_m->ndims() - 1)) % 2) > 0 ? 1
                                                                  : prb->k;
    const int64_t wei_zp_group = !p.wei_zp_groups.empty() ? p.wei_zp_groups[0]
            : ((p.wei_zp_mask >> (p.wei_m->ndims() - 2)) % 2) > 0 ? 1
                                                                  : prb->k;
    p.smallest_k_group = gcd<int64_t>(
            {src_scale_group, wei_scale_group, src_zp_group, wei_zp_group});
    p.n_k_groups = prb->k / p.smallest_k_group;

    p.dst_scale_dt = prb->attr.scales.get(DNNL_ARG_DST).dt;

    const auto &dsg = prb->attr.scales.get(DNNL_ARG_DST).groups;
    p.dst_M_group = !dsg.empty() ? dsg[0] : 1;
    p.dst_N_group = !dsg.empty() ? dsg[1] : 1;
    p.dst_scale_groups = dsg;

    p.v_po_masks = prb->attr.post_ops.get_po_masks(prb->ndims);

    p.src_zp_single = p.has_src_single_zp ? p.src_zps->get_elem(0) : 0;
    p.wei_zp_single = p.has_wei_single_zp ? p.wei_zps->get_elem(0) : 0;
    p.src_scale_single
            = p.has_src_single_scale ? p.src_scales->get_f32_elem(0) : 1.f;
    p.wei_scale_single
            = p.has_wei_single_scale ? p.wei_scales->get_f32_elem(0) : 1.f;

    p.bia_dt = prb->bia_dt;
    p.dst_dt = prb->dst_dt();

    return p;
}

// Computational kernel for a single (mc, nc) chunk of output
//
// _base and _stride are used to compute the actual offsets as follows:
//   src(m, k)    = (src_row_base + m) * K + k
//   wei_ab(k, n) = wei_base + k * N + n (for scales and zps)
//   wei(k, n)    = wei_base + k * wei_k_stride + n * wei_n_stride
//   dst(m, n)    = (dst_row_base + m) * N + n
//   bia(m, n)    = bia_base + m * bia_m_stride + n * bia_n_stride
static void compute_ref_matmul_chunk(const chunk_params_t &p, int64_t M,
        int64_t N, int64_t K, int64_t mc, int64_t nc, int64_t src_row_base,
        int64_t wei_base, int64_t wei_k_stride, int64_t wei_n_stride,
        int64_t dst_row_base, int64_t bia_base, int64_t bia_m_stride,
        int64_t bia_n_stride, const attr_t &attr, const args_t &args) {
    // Mutable per-element quant params; initialised to the single value when
    // applicable and overwritten per K-group otherwise.
    int src_zp = p.src_zp_single;
    int wei_zp = p.wei_zp_single;
    float src_scale = p.src_scale_single;
    float wei_scale = p.wei_scale_single;

    for_(int64_t m = mc * p.dst_M_group; m < MIN2((mc + 1) * p.dst_M_group, M);
            ++m)
    for_(int64_t n = nc * p.dst_N_group; n < MIN2((nc + 1) * p.dst_N_group, N);
            ++n)
    {
        float dst = 0;
        for (int64_t gK = 0; gK < p.n_k_groups; gK++) {
            const auto src_gK_off
                    = (src_row_base + m) * K + gK * p.smallest_k_group;
            // Note: scales/zero-points are still always in `tag::abx` format.
            const auto wei_gK_off = wei_base + gK * p.smallest_k_group * N + n;

            if (p.has_src_zp && !p.has_src_single_zp) {
                const auto src_zp_idx = p.src_m->get_idx(src_gK_off,
                        p.src_zp_mask, p.src_m->ndims(), p.src_zp_groups);
                src_zp = p.src_zps->get_elem(src_zp_idx);
            }
            if (p.has_wei_zp && !p.has_wei_single_zp) {
                const auto wei_zp_idx = p.wei_m->get_idx(wei_gK_off,
                        p.wei_zp_mask, p.wei_m->ndims(), p.wei_zp_groups);
                wei_zp = p.wei_zps->get_elem(wei_zp_idx);
            }

            if (p.has_src_scale && !p.has_src_single_scale) {
                const auto src_scale_idx = p.src_m->get_idx(src_gK_off,
                        p.src_scale_mask, p.src_m->ndims(), p.src_scale_groups);
                src_scale = p.src_scales->get_f32_elem(src_scale_idx);
            }
            if (p.has_wei_scale && !p.has_wei_single_scale) {
                const auto wei_scale_idx = p.wei_m->get_idx(wei_gK_off,
                        p.wei_scale_mask, p.wei_m->ndims(), p.wei_scale_groups);
                wei_scale = p.wei_scales->get_f32_elem(wei_scale_idx);
            }

            for (int64_t k = 0; k < p.smallest_k_group; ++k) {
                const auto kk = gK * p.smallest_k_group + k;
                const auto src_off = (src_row_base + m) * K + kk;
                const auto wei_off
                        = wei_base + kk * wei_k_stride + n * wei_n_stride;

                auto s = src_scale * (p.src_m->get_f32_elem(src_off) - src_zp);
                auto w = wei_scale * (p.wei_m->get_f32_elem(wei_off) - wei_zp);

                dst += s * w;
            }
        }

        const auto dst_off = (dst_row_base + m) * N + n;
        if (p.bia_dt != dnnl_data_type_undef) {
            const auto bia_idx = bia_base + m * bia_m_stride + n * bia_n_stride;
            dst += p.bia_m->get_f32_elem(bia_idx);
        }

        const auto v_po_vals
                = prepare_po_vals(*p.dst_m, args, p.v_po_masks, dst_off);
        maybe_dropout(attr, dst, dst_off, *p.dropout_mask);
        const auto sum_val = p.dst_m->get_f32_elem(dst_off);
        maybe_post_ops(attr, dst, sum_val, v_po_vals);

        // We use dst as temporary storage
        p.dst_m->set_f32_elem(dst_off, dst);
    }

    // Now we can do downconversion and write back to dst.
    // Compute scales if dyn_quant.
    float dst_scale = 1.f;
    if (p.has_dst_dynamic) {
        // Note: Mantissa-less dt would round-up zero to min normal.
        // Note: Mantissa-ed dt needs initial value to be zero to properly
        // handle the final value if the block is full of zero values.
        dst_scale = 0.f;
        for_(int64_t m = mc * p.dst_M_group;
                m < MIN2((mc + 1) * p.dst_M_group, M); ++m)
        for (int64_t n = nc * p.dst_N_group;
                n < MIN2((nc + 1) * p.dst_N_group, N); ++n) {
            const auto dst_off = (dst_row_base + m) * N + n;
            dst_scale = MAX2(fabsf(p.dst_m->get_f32_elem(dst_off)), dst_scale);
        }
        if (p.has_dst_mx) {
            dst_scale
                    = round_to_nearest_representable(p.dst_scale_dt, dst_scale)
                    / round_to_nearest_representable(
                            p.dst_scale_dt, max_dt(p.dst_dt));
            dst_scale
                    = round_to_nearest_representable(p.dst_scale_dt, dst_scale);
        } else if (p.has_dst_dynamic_fp) {
            dst_scale = dst_scale == 0.f
                    ? 1.f
                    : round_to_nearest_representable(
                              p.dst_scale_dt, dst_scale / max_dt(p.dst_dt));
        }
        const auto dst_off
                = (dst_row_base + mc * p.dst_M_group) * N + nc * p.dst_N_group;
        const auto dscale_idx = p.dst_m->get_idx(dst_off, p.dst_scale_mask,
                p.dst_m->ndims(), p.dst_scale_groups);
        p.dst_scales->set_f32_elem(dscale_idx, dst_scale);
        // Pre-invert the scale to apply it as a multiplier for the group.
        // Note, that it can't be done upfront, as it must be written to
        // the memory before. Can't be zero.
        dst_scale = 1.f / dst_scale;
    }

    // Apply scales and downconvert.
    for_(int64_t m = mc * p.dst_M_group; m < MIN2((mc + 1) * p.dst_M_group, M);
            ++m)
    for_(int64_t n = nc * p.dst_N_group; n < MIN2((nc + 1) * p.dst_N_group, N);
            ++n)
    {
        int dst_zp = 0;
        const auto dst_off = (dst_row_base + m) * N + n;

        if (p.has_dst_zp) {
            const auto dst_zp_idx = p.dst_m->get_idx(dst_off, p.dst_zp_mask);
            dst_zp = p.dst_zps->get_elem(dst_zp_idx);
        }
        if (p.has_dst_scale && !p.has_dst_dynamic) {
            dst_scale = 1.f
                    / p.dst_scales->get_f32_elem(p.dst_scale_mask > 0 ? n : 0);
        }
        float dst = p.dst_m->get_f32_elem(dst_off);
        float dst_val = dst_scale * dst + dst_zp;
        maybe_round(attr, DNNL_ARG_DST, dst_val, dst_off, p.dst_dt);
        p.dst_m->set_f32_elem(dst_off, dst_val);
    }
}

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
// Reference implementation for grouped gemm
// Computes per-expert matmuls: for each expert e, computes dst[e] = src[e] * wei[e]
//
// Note, that all tensors are concatenated
// src and dst in grouped format and wei in 3D [group_count, K, N] format
//
// TODO: extract common computation kernel from regular matmul and reuse it here
void compute_ref_grouped_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &wei_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    const int64_t group_count = prb->sparse_options.get_group_count();
    const auto &M_dims = prb->sparse_options.get_group_sizes();
    const int64_t K = prb->k;
    const int64_t N = prb->n;

    const bool has_bias = prb->bia_dt != dnnl_data_type_undef;
    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_wei_zp
            = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();

    const auto &wei_scale_groups
            = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
    const auto &wei_zp_groups
            = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;

    // For grouped GEMM, weights are 3D: [group_count, K, N]
    // wei_scale_groups[0] is the K dimension group size
    const int64_t wei_scale_group_k
            = !wei_scale_groups.empty() ? wei_scale_groups[0] : K;
    const int64_t wei_zp_group_k
            = !wei_zp_groups.empty() ? wei_zp_groups[0] : K;

    // Use finest granularity k-group (GCD) to handle mixed scale/ZP groups
    const int64_t smallest_k_group
            = gcd<int64_t>({wei_scale_group_k, wei_zp_group_k});
    const int64_t n_k_groups = K / smallest_k_group;

    std::vector<int64_t> group_offsets(group_count + 1);
    group_offsets[0] = 0;
    for (int64_t g = 0; g < group_count; g++) {
        group_offsets[g + 1] = group_offsets[g] + M_dims[g];
    }

    benchdnn_parallel_nd(group_count, [&](int64_t g) {
        const int64_t M_g = M_dims[g];
        if (M_g == 0) return;

        const int64_t offset = group_offsets[g];

        for (int64_t m = 0; m < M_g; m++) {
            const int64_t src_offset = offset + m;

            // retrieve row-wise src scale for this row if any
            float src_scale = 1.0f;
            if (has_src_scale) {
                src_scale = src_scales.get_f32_elem(src_offset);
            }

            for (int64_t n = 0; n < N; n++) {
                float acc = 0.0f;

                for (int64_t gK = 0; gK < n_k_groups; gK++) {
                    float acc_group = 0.0f;

                    // Scale and ZP k-group indices for this fine k-group
                    const int64_t scale_kg
                            = gK * smallest_k_group / wei_scale_group_k;
                    const int64_t zp_kg
                            = gK * smallest_k_group / wei_zp_group_k;

                    float wei_scale = 1.0f;
                    if (has_wei_scale) {
                        const int64_t wei_scale_idx
                                = (g * n_k_groups + scale_kg) * N + n;
                        wei_scale = wei_scales.get_f32_elem(wei_scale_idx);
                    }

                    float wei_zp = 0.0f;
                    if (has_wei_zp) {
                        const int64_t wei_zp_idx
                                = (g * n_k_groups + zp_kg) * N + n;
                        wei_zp = wei_zps.get_f32_elem(wei_zp_idx);
                    }

                    for (int64_t k = 0; k < smallest_k_group; k++) {
                        const int64_t k_idx = gK * smallest_k_group + k;
                        const int64_t src_idx = src_offset * K + k_idx;
                        dnnl_dims_t wei_pos = {g, k_idx, n};
                        const int64_t wei_idx = md_off_v(wei_m, wei_pos);

                        const float src_val = src_m.get_f32_elem(src_idx);
                        const float wei_val = wei_m.get_f32_elem(wei_idx);
                        acc_group += src_val * (wei_val - wei_zp);
                    }

                    acc += acc_group * wei_scale;
                }

                // Apply per-token scale
                acc *= src_scale;

                // Add bias if present
                if (has_bias) {
                    const int64_t bias_idx = g * N + n;
                    acc += bia_m.get_f32_elem(bias_idx);
                }

                // dst: plain [total_M, N], indexed as [(offset + m), n]
                const int64_t dst_idx = src_offset * N + n;
                dst_m.set_f32_elem(dst_idx, acc);
            }
        }
    });
}
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

void compute_ref_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const dnn_mem_t &src_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const dnn_mem_t &wei_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    const dnn_mem_t &dropout_mask = args.find(DNNL_ARG_ATTR_DROPOUT_MASK);

    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;
    const int64_t MB = prb->mb;
    const int batch_ndims = dst_m.ndims() - 2;

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    const bool has_dst_dynamic
            = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic();
    const bool has_dst_mx = prb->attr.scales.get(DNNL_ARG_DST).is_mx();
    const bool has_dst_dynamic_fp
            = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic_fp();

    const int src_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, src_m.ndims());
    const int wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims());
    const int dst_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_DST, dnnl_matmul, dst_m.ndims());

    const bool has_src_single_scale = has_src_scale && src_scale_mask == 0;
    const bool has_wei_single_scale = has_wei_scale && wei_scale_mask == 0;

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_zp
            = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();

    const int src_zp_mask = has_src_zp
            ? prb->attr.zero_points.get_mask(
                      DNNL_ARG_SRC, dnnl_matmul, src_m.ndims())
            : 0;
    const int wei_zp_mask = has_wei_zp
            ? prb->attr.zero_points.get_mask(
                      DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims())
            : 0;
    const int dst_zp_mask = has_dst_zp
            ? prb->attr.zero_points.get_mask(
                      DNNL_ARG_DST, dnnl_matmul, dst_m.ndims())
            : 0;

    const bool has_src_single_zp = has_src_zp && src_zp_mask == 0;
    const bool has_wei_single_zp = has_wei_zp && wei_zp_mask == 0;

    const auto &src_scale_groups = prb->attr.scales.get(DNNL_ARG_SRC).groups;
    const auto &wei_scale_groups
            = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
    const auto &src_zp_groups = prb->attr.zero_points.get(DNNL_ARG_SRC).groups;
    const auto &wei_zp_groups
            = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;

    const int64_t src_scale_group = !src_scale_groups.empty()
            ? src_scale_groups[1]
            : ((src_scale_mask >> (src_m.ndims() - 1)) % 2) > 0 ? 1
                                                                : K;
    const int64_t wei_scale_group = !wei_scale_groups.empty()
            ? wei_scale_groups[0]
            : ((wei_scale_mask >> (wei_m.ndims() - 2)) % 2) > 0 ? 1
                                                                : K;
    const int64_t src_zp_group = !src_zp_groups.empty()      ? src_zp_groups[1]
            : ((src_zp_mask >> (src_m.ndims() - 1)) % 2) > 0 ? 1
                                                             : K;
    const int64_t wei_zp_group = !wei_zp_groups.empty()      ? wei_zp_groups[0]
            : ((wei_zp_mask >> (wei_m.ndims() - 2)) % 2) > 0 ? 1
                                                             : K;

    const auto smallest_k_group = gcd<int64_t>(
            {src_scale_group, wei_scale_group, src_zp_group, wei_zp_group});

    const auto n_k_groups = K / smallest_k_group;

    // Fast return if any dim is zero. Common logic doesn't apply because of
    // broadcast semantics.
    for (int d = 0; d < dst_m.ndims(); d++) {
        if (prb->src_dims()[d] == 0 || prb->weights_dims()[d] == 0) return;
    }

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();
    auto v_po_masks = prb->attr.post_ops.get_po_masks(prb->ndims);

    const auto dst_scale_dt = prb->attr.scales.get(DNNL_ARG_DST).dt;
    const auto &dst_scale_groups = prb->attr.scales.get(DNNL_ARG_DST).groups;
    const int64_t dst_M_group
            = !dst_scale_groups.empty() ? dst_scale_groups[0] : 1;
    const int64_t dst_N_group
            = !dst_scale_groups.empty() ? dst_scale_groups[1] : 1;
    const int64_t M_chunks = div_up(M, dst_M_group);
    const int64_t N_chunks = div_up(N, dst_N_group);

    benchdnn_parallel_nd(
            MB, M_chunks, N_chunks, [&](int64_t mb, int64_t mc, int64_t nc) {
        int64_t src_mb = 0;
        int64_t wei_mb = 0;
        if (MB > 1) {
            src_mb = dst_m.get_idx(mb, src_broadcast_mask, batch_ndims);
            wei_mb = dst_m.get_idx(mb, wei_broadcast_mask, batch_ndims);
        }

        int src_zp = has_src_single_zp ? src_zps.get_elem(0) : 0;
        int wei_zp = has_wei_single_zp ? wei_zps.get_elem(0) : 0;
        float src_scale
                = has_src_single_scale ? src_scales.get_f32_elem(0) : 1.f;
        float wei_scale
                = has_wei_single_scale ? wei_scales.get_f32_elem(0) : 1.f;

        for_(int64_t m = mc * dst_M_group; m < MIN2((mc + 1) * dst_M_group, M);
                ++m)
        for_(int64_t n = nc * dst_N_group; n < MIN2((nc + 1) * dst_N_group, N);
                ++n)
        {
            float dst = 0;
            for (int64_t gK = 0; gK < n_k_groups; gK++) {
                const auto src_gK_off
                        = src_off_f(prb, src_mb, m, gK * smallest_k_group);
                // Note: scales/zero-points are still always in `tag::abx` format.
                const auto wei_gK_off
                        = wei_ab_off_f(prb, wei_mb, gK * smallest_k_group, n);

                if (has_src_zp && !has_src_single_zp) {
                    const auto src_zp_idx = src_m.get_idx(src_gK_off,
                            src_zp_mask, src_m.ndims(), src_zp_groups);
                    src_zp = src_zps.get_elem(src_zp_idx);
                }
                if (has_wei_zp && !has_wei_single_zp) {
                    const auto wei_zp_idx = wei_m.get_idx(wei_gK_off,
                            wei_zp_mask, wei_m.ndims(), wei_zp_groups);
                    wei_zp = wei_zps.get_elem(wei_zp_idx);
                }

                if (has_src_scale && !has_src_single_scale) {
                    const auto src_scale_idx = src_m.get_idx(src_gK_off,
                            src_scale_mask, src_m.ndims(), src_scale_groups);
                    src_scale = src_scales.get_f32_elem(src_scale_idx);
                }
                if (has_wei_scale && !has_wei_single_scale) {
                    const auto wei_scale_idx = wei_m.get_idx(wei_gK_off,
                            wei_scale_mask, wei_m.ndims(), wei_scale_groups);
                    wei_scale = wei_scales.get_f32_elem(wei_scale_idx);
                }

                for (int64_t k = 0; k < smallest_k_group; ++k) {
                    const auto src_off = src_off_f(
                            prb, src_mb, m, gK * smallest_k_group + k);
                    const auto wei_off = wei_ba_off_f(
                            prb, wei_mb, gK * smallest_k_group + k, n);

                    auto s = src_scale * (src_m.get_f32_elem(src_off) - src_zp);
                    auto w = wei_scale * (wei_m.get_f32_elem(wei_off) - wei_zp);

                    dst += s * w;
                }
            }

            const auto dst_off = dst_off_f(prb, mb, m, n);
            if (prb->bia_dt != dnnl_data_type_undef) {
                const auto bia_idx
                        = dst_m.get_idx(dst_off, bias_broadcast_mask);
                dst += bia_m.get_f32_elem(bia_idx);
            }

            const auto v_po_vals
                    = prepare_po_vals(dst_m, args, v_po_masks, dst_off);
            maybe_dropout(prb->attr, dst, dst_off, dropout_mask);
            const auto sum_val = dst_m.get_f32_elem(dst_off);
            maybe_post_ops(prb->attr, dst, sum_val, v_po_vals);

            // We use dst as temporary storage
            dst_m.set_f32_elem(dst_off, dst);
        }

        // Now we can do downconversion and write back to dst
        // Compute scales if dyn_quant
        float dst_scale = 1.f;
        if (has_dst_dynamic) {
            // Note: Mantissa-less dt would round-up zero to min normal.
            // Note: Mantissa-ed dt needs initial value to be zero to properly
            // handle the final value if the block is full of zero values.
            dst_scale = 0.f;
            for_(int64_t m = mc * dst_M_group;
                    m < MIN2((mc + 1) * dst_M_group, M); ++m)
            for (int64_t n = nc * dst_N_group;
                    n < MIN2((nc + 1) * dst_N_group, N); ++n) {
                const auto dst_off = dst_off_f(prb, mb, m, n);
                dst_scale = MAX2(fabsf(dst_m.get_f32_elem(dst_off)), dst_scale);
            }
            if (has_dst_mx) {
                dst_scale = round_to_nearest_representable(
                                    dst_scale_dt, dst_scale)
                        / round_to_nearest_representable(
                                dst_scale_dt, max_dt(prb->dst_dt()));
                dst_scale = round_to_nearest_representable(
                        dst_scale_dt, dst_scale);
            } else if (has_dst_dynamic_fp) {
                dst_scale = dst_scale == 0.f
                        ? 1.f
                        : round_to_nearest_representable(dst_scale_dt,
                                  dst_scale / max_dt(prb->dst_dt()));
            }
            const auto dst_off
                    = dst_off_f(prb, mb, mc * dst_M_group, nc * dst_N_group);
            const auto dscale_idx = dst_m.get_idx(
                    dst_off, dst_scale_mask, dst_m.ndims(), dst_scale_groups);
            dst_scales.set_f32_elem(dscale_idx, dst_scale);
            // Pre-invert the scale to apply it as a multiplier for the group.
            // Note, that it can't be done upfront, as it must be written to
            // the memory before. Can't be zero.
            dst_scale = 1.f / dst_scale;
        }

        // Apply scales and downconvert
        for_(int64_t m = mc * dst_M_group; m < MIN2((mc + 1) * dst_M_group, M);
                ++m)
        for_(int64_t n = nc * dst_N_group; n < MIN2((nc + 1) * dst_N_group, N);
                ++n)
        {
            int dst_zp = 0;
            const auto dst_off = dst_off_f(prb, mb, m, n);

            if (has_dst_zp) {
                const auto dst_zp_idx = dst_m.get_idx(dst_off, dst_zp_mask);
                dst_zp = dst_zps.get_elem(dst_zp_idx);
            }
            if (has_dst_scale && !has_dst_dynamic) {
                dst_scale = 1.f
                        / dst_scales.get_f32_elem(dst_scale_mask > 0 ? n : 0);
            }
            float dst = dst_m.get_f32_elem(dst_off);
            float dst_val = dst_scale * dst + dst_zp;
            maybe_round(
                    prb->attr, DNNL_ARG_DST, dst_val, dst_off, prb->dst_dt());
            dst_m.set_f32_elem(dst_off, dst_val);
        }
    });
}

void cvt_coo_indices_to_csr_pointers(const int32_t *indices, int32_t *pointers,
        const int nnz, const int nrows) {
    for (int i = 0; i < nnz; ++i) {
        ++pointers[indices[i] + 1];
    }
    for (int i = 0; i < nrows; ++i) {
        pointers[i + 1] += pointers[i];
    }
}

void compute_ref_sparse_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    const bool is_src_sparse
            = src_encoding == dnnl_csr || src_encoding == dnnl_coo;
    const bool is_wei_sparse
            = wei_encoding == dnnl_csr || wei_encoding == dnnl_coo;
    auto encoding = is_src_sparse ? src_encoding : wei_encoding;

    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // TODO: Depending on the matrix dimensions the pointer buffer may take
    // up a significant amount of memory. This wil require a mechanism to
    // register the memory needed for the current scratchpad during
    // COO-to-CSR format conversion.
    std::vector<int32_t> pointer_buffer(1 + (is_src_sparse ? M : K), 0);

    // Batch is not supported.
    const int64_t mb = 0;
    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        dst_m.set_f32_elem(dst_off_f(prb, mb, m, n), 0.0f);
    });

    if (is_wei_sparse) {
        int32_t *wei_indices = wei_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *wei_pointers = wei_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *wei_row_indices = wei_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(wei_m.md_);

            benchdnn_parallel_nd(
                    K + 1, [&](int64_t i) { pointer_buffer[i] = 0; });
            cvt_coo_indices_to_csr_pointers(
                    wei_row_indices, pointer_buffer.data(), nnz, K);
            wei_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            for (int64_t k = 0; k < K; k++) {
                const int64_t row_start = wei_pointers[k];
                const int64_t row_end = wei_pointers[k + 1];
                for (int64_t n = row_start; n < row_end; n++) {
                    const int64_t src_idx = src_off_f(prb, mb, m, k);
                    const int64_t dst_idx
                            = dst_off_f(prb, mb, m, wei_indices[n]);
                    const float src_val = src_m.get_f32_elem(src_idx);
                    const float wei_val = wei_m.get_elem(n, 0);
                    float dst_val = dst_m.get_f32_elem(dst_idx);
                    dst_val += src_val * wei_val;
                    dst_m.set_f32_elem(dst_idx, dst_val);
                }
            }
        });
    } else if (is_src_sparse) {
        int32_t *src_indices = src_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *src_pointers = src_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *src_row_indices = src_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(src_m.md_);
            cvt_coo_indices_to_csr_pointers(
                    src_row_indices, pointer_buffer.data(), nnz, M);
            src_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            const int64_t row_start = src_pointers[m];
            const int64_t row_end = src_pointers[m + 1];
            for (int64_t n = 0; n < N; n++) {
                const int64_t dst_idx = dst_off_f(prb, mb, m, n);
                float dst_val = dst_m.get_f32_elem(dst_idx);

                for (int64_t k = row_start; k < row_end; k++) {
                    const int64_t wei_idx
                            = wei_ba_off_f(prb, mb, src_indices[k], n);
                    const float src_val = src_m.get_elem(k, 0);
                    const float wei_val = wei_m.get_f32_elem(wei_idx);
                    dst_val += src_val * wei_val;
                }
                dst_m.set_f32_elem(dst_idx, dst_val);
            }
        });
    }
}

void compute_ref(const prb_t *prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    const auto dst_encoding = prb->sparse_options.get_encoding(DNNL_ARG_DST);

    if (src_encoding == dnnl_grouped && dst_encoding == dnnl_grouped) {
        compute_ref_grouped_matmul(prb, args);
    } else
#endif
            if (src_encoding == dnnl_csr || wei_encoding == dnnl_csr
                    || src_encoding == dnnl_coo || wei_encoding == dnnl_coo) {
        compute_ref_sparse_matmul(prb, args);
    } else {
        compute_ref_matmul(prb, args);
    }
}

} // namespace matmul
