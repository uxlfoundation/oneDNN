/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <cstring>
#include <vector>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

// Reference GatedMLP: generates gold data by composing existing oneDNN
// primitives (matmul, eltwise) instead of reimplementing from scratch.
//
// Pipeline:
//   up     = matmul(src, W_up)            -> [MB, OC]
//   gate   = matmul(src, W_gate)          -> [MB, OC]
//   gate   = eltwise(gate, activation)    -> [MB, OC]
//   up     = up * gate                    -> [MB, OC]  (elementwise)
//   dst    = matmul(up, W_down)           -> [MB, IC]
//
// All intermediate computation is done in f32 on the CPU engine.

namespace {

// Execute a matmul primitive on CPU: dst = src x wei.
void exec_matmul(dnnl_engine_t eng, dnnl_stream_t strm, const dnn_mem_t &src,
        const dnn_mem_t &wei, const dnn_mem_t &dst, const args_t &all_args,
        const_dnnl_primitive_attr_t attr = nullptr) {
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_matmul_primitive_desc_create(
            &pd, eng, src.md_, wei.md_, nullptr, dst.md_, attr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    std::vector<dnnl_exec_arg_t> args = {
            {DNNL_ARG_SRC, src.m_},
            {DNNL_ARG_WEIGHTS, wei.m_},
            {DNNL_ARG_DST, dst.m_},
    };
    for (int p = 0, pl = 32 * !!attr; p < pl; p++) {
        auto idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(p) | DNNL_ARG_SRC_1;
        auto &maybe_buf = all_args.find(idx);
        if (maybe_buf) args.emplace_back(dnnl_exec_arg_t {idx, maybe_buf.m_});
    }
    DNN_SAFE_V(dnnl_primitive_execute(
            prim, strm, static_cast<int>(args.size()), args.data()));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Execute in-place eltwise forward on CPU.
void exec_eltwise(dnnl_engine_t eng, dnnl_stream_t strm, dnn_mem_t &mem,
        dnnl_alg_kind_t alg) {
    // Swish uses alpha=1.0 by default. GELU variants ignore alpha/beta.
    float alpha = (alg == dnnl_eltwise_swish) ? 1.0f : 0.0f;
    float beta = 0.0f;

    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_eltwise_forward_primitive_desc_create(&pd, eng,
            dnnl_forward_inference, alg, mem.md_, mem.md_, alpha, beta,
            nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC, mem.m_},
            {DNNL_ARG_DST, mem.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 2, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Execute in-place binary operation on CPU: lhs = lhs <alg> rhs.
void exec_binary(dnnl_engine_t eng, dnnl_stream_t strm, dnn_mem_t &lhs,
        const dnn_mem_t &rhs, dnnl_alg_kind_t alg) {
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_binary_primitive_desc_create(
            &pd, eng, alg, lhs.md_, rhs.md_, lhs.md_, nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC_0, lhs.m_},
            {DNNL_ARG_SRC_1, rhs.m_},
            {DNNL_ARG_DST, lhs.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 3, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Create an f32 plain memory on `eng` with the given dims.
dnn_mem_t make_mem(dnnl_engine_t eng, const dims_t &d) {
    auto md = dnn_mem_t::init_md(
            static_cast<int>(d.size()), d.data(), dnnl_f32, tag::abx);
    return dnn_mem_t(md, eng, /* prefill = */ false);
}

// Dequantize a tensor in-place:
//   t[k][n] = scale[idx] * (t[k][n] - zp[idx]).
// For weights shaped [K, N]:
//   mask=2: scale/zp indexed by n only.
//   mask=3: scale/zp indexed by (k / group_k) and n.
void dequantize_buf(const dnn_mem_t &buf, int64_t K, int64_t N, bool has_scale,
        int scale_mask, const std::vector<dnnl_dim_t> &scale_grp,
        const dnn_mem_t &scales_m, bool has_zp, int zp_mask,
        const std::vector<dnnl_dim_t> &zp_grp, const dnn_mem_t &zps_m, bool T) {
    if (!has_scale && !has_zp) return;

    int kmask = 1 << (int)T;
    int nmask = 1 << (int)!T;
    if ((buf.ndims() == 3) && (buf.dims()[0] >= 2) && (buf.dims()[1] == 1)) {
        kmask <<= (int)T;
        nmask <<= (int)!T;
    }
    if ((buf.ndims() == 3) && (buf.dims()[0] == 1) && (buf.dims()[1] >= 2)) {
        kmask <<= 1;
        nmask <<= 1;
    }

    // Determine K-group size for scales and zero-points;
    // mask bit 1 (1<<0) set means per-K; k*_group subdivides K dimension
    int64_t kz_grp = (zp_mask & kmask) ? (zp_grp.empty()) ? 1 : zp_grp[T] : K;
    int64_t ks_grp
            = (scale_mask & kmask) ? (scale_grp.empty()) ? 1 : scale_grp[T] : K;

    bool nz_inc = zp_mask & nmask;
    bool ns_inc = scale_mask & nmask;

    auto t = (float *)buf;
    if (T) { // if transposed
        int64_t nz_mult = (nz_inc) ? K / kz_grp : 1;
        int64_t ns_mult = (ns_inc) ? K / ks_grp : 1;
        for (int64_t n = 0, nz = 0, ns = 0; n < N;
                ++n, nz += nz_inc, ns += ns_inc)
            for (int64_t k = 0, kz = 0, ks = 0; k < K; ++k,
                         kz = (k >= (kz + 1) * kz_grp) ? kz + 1 : kz,
                         ks = (k >= (ks + 1) * ks_grp) ? ks + 1 : ks) {
                auto z = (has_zp) ? zps_m.get_f32_elem(nz * nz_mult + kz) : 0.f;
                auto s = (has_scale) ? scales_m.get_f32_elem(ns * ns_mult + ks)
                                     : 1.f;
                t[n * K + k] = s * (t[n * K + k] - z);
            }
    } else { // if not transposed
        int64_t kz_mult = (nz_inc) ? N : 1;
        int64_t ks_mult = (ns_inc) ? N : 1;
        for (int64_t k = 0, kz = 0, ks = 0; k < K; ++k,
                     kz = (k >= (kz + 1) * kz_grp) ? kz + 1 : kz,
                     ks = (k >= (ks + 1) * ks_grp) ? ks + 1 : ks)
            for (int64_t n = 0, nz = 0, ns = 0; n < N;
                    ++n, nz += nz_inc, ns += ns_inc) {
                auto z = (has_zp) ? zps_m.get_f32_elem(nz + kz * kz_mult) : 0.f;
                auto s = (has_scale) ? scales_m.get_f32_elem(ns + ks * ks_mult)
                                     : 1.f;
                t[k * N + n] = s * (t[k * N + n] - z);
            }
    }
}

} // anonymous namespace

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t) {
    const prb_t *prb = prb_t::from(base_prb);
    const auto &eng = get_cpu_engine();
    dnnl_stream_t strm {};
    DNN_SAFE_V(dnnl_stream_create(&strm, eng, dnnl_stream_default_flags));

    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &w_gate_m = args.find(DNNL_ARG_WEIGHTS_GATE);
    const dnn_mem_t &w_up_m = args.find(DNNL_ARG_WEIGHTS_UP);
    const dnn_mem_t &w_down_m = args.find(DNNL_ARG_WEIGHTS_DOWN);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const int64_t MB = prb->mb;
    const int64_t IC = prb->ic;
    const int64_t OC = prb->oc;

    // f32 memories for intermediate results.
    auto dims = (src_m.ndims() == 2) ? dims_t {MB, OC} : dims_t {MB, 1, OC};
    auto up_result = make_mem(eng, dims);
    auto gate_result = make_mem(eng, dims);

    // Dequantize if quantization attributes are set.
    const auto &attr = prb->attr;
    auto dequant = [&](const dnn_mem_t &buf, int64_t K, int64_t N, int arg) {
        const bool has_scale = !attr.scales.get(arg).is_def();
        const bool has_zp = !attr.zero_points.get(arg).is_def();
        if (!has_scale && !has_zp) return dnn_mem_t();

        const auto &sc_groups = (has_scale) ? attr.scales.get(arg).groups
                                            : std::vector<dnnl_dim_t> {};
        const auto &zp_groups = (has_zp) ? attr.zero_points.get(arg).groups
                                         : std::vector<dnnl_dim_t> {};

        const auto md_dims = args.find(arg).dims();
        const dims_t dims(md_dims, md_dims + args.find(arg).ndims());

        const auto undef = dnnl_undefined_primitive;
        const int sc_mask = (has_scale) ? attr.scales.get_mask(arg, undef,
                                                  static_cast<int>(dims.size()))
                                        : 0;
        const int zp_mask = (has_zp) ? attr.zero_points.get_mask(arg, undef,
                                               static_cast<int>(dims.size()))
                                     : 0;

        auto retn = make_mem(eng, dims);
        std::memcpy((float *)retn, (float *)buf, K * N * sizeof(float));
        dequantize_buf(retn, K, N, has_scale, sc_mask, sc_groups,
                args.find(DNNL_ARG_ATTR_SCALES | arg), has_zp, zp_mask,
                zp_groups, args.find(DNNL_ARG_ATTR_ZERO_POINTS | arg),
                arg == DNNL_ARG_SRC);
        return retn;
    };

    auto src_ref = dequant(src_m, IC, MB, DNNL_ARG_SRC);
    auto w_gate_ref = dequant(w_gate_m, IC, OC, DNNL_ARG_WEIGHTS_GATE);
    auto w_up_ref = dequant(w_up_m, IC, OC, DNNL_ARG_WEIGHTS_UP);
    auto w_down_ref = dequant(w_down_m, OC, IC, DNNL_ARG_WEIGHTS_DOWN);

    // Step 1: up_result = matmul(src, W_up).
    exec_matmul(eng, strm, (src_ref) ? src_ref : src_m,
            (w_up_ref) ? w_up_ref : w_up_m, up_result, args);

    // Step 2: gate_result = matmul(src, W_gate).
    exec_matmul(eng, strm, (src_ref) ? src_ref : src_m,
            (w_gate_ref) ? w_gate_ref : w_gate_m, gate_result, args);

    // Step 3: gate_result = activation(gate_result).
    exec_eltwise(eng, strm, gate_result, prb->activation);

    // Step 4: up_result = up_result * gate_result (element-wise).
    exec_binary(eng, strm, up_result, gate_result, dnnl_binary_mul);

    // Step 5: dst = matmul(up_result, W_down).
    attr_t down_attr;
    down_attr.post_ops = prb->attr.post_ops;
    for (auto &po : down_attr.post_ops.entry)
        if (po.is_binary_kind()) po.binary.src1_dt = dnnl_f32;
    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            down_attr, dst_m.ndims(), dst_m.dims(), dnnl_matmul);
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(down_attr, attr_args, prb->ndims));
    exec_matmul(eng, strm, up_result, (w_down_ref) ? w_down_ref : w_down_m,
            dst_m, args, dnnl_attr);

    DNN_SAFE_V(dnnl_stream_destroy(strm));
}

} // namespace gated_mlp
