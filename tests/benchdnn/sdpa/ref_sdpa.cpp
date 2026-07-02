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

#include <cmath>
#include <cstring>
#include <limits>

#include "oneapi/dnnl/dnnl.h"

// Internal alg_kind used by the GPU SDPA kernel. Must be removed once
// softmax_accurate_inf_as_zero is promoted to a public value.
#include "src/common/c_types_map.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

// Reference SDPA: generates gold data by composing existing oneDNN primitives
// (matmul, softmax) instead of reimplementing the algorithm from scratch.
//
// Pipeline: score = matmul(Q, K)  ->  scale  ->  [mask/causal]
//           ->  softmax(score)  ->  matmul(prob, V)  ->  DST
//
// All intermediate computation is done in f32 on the CPU engine.

namespace {

// Execute a matmul primitive on CPU: dst = src x wei.
void exec_matmul(dnnl_engine_t eng, dnnl_stream_t strm, const dnn_mem_t &src,
        const dnn_mem_t &wei, dnn_mem_t &dst) {
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_matmul_primitive_desc_create(
            &pd, eng, src.md_, wei.md_, nullptr, dst.md_, nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC, src.m_},
            {DNNL_ARG_WEIGHTS, wei.m_},
            {DNNL_ARG_DST, dst.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 3, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Execute in-place softmax on CPU over the given axis.
void exec_softmax(
        dnnl_engine_t eng, dnnl_stream_t strm, dnn_mem_t &mem, int axis) {
    // Use softmax_accurate_inf_as_zero to match the GPU SDPA kernel.
    const auto alg = static_cast<dnnl_alg_kind_t>(
            dnnl::impl::alg_kind::softmax_accurate_inf_as_zero);
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_softmax_forward_primitive_desc_create(&pd, eng,
            dnnl_forward_inference, alg, mem.md_, mem.md_, axis, nullptr));
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

// Execute softmax backward on CPU: diff_src = softmax_bwd(fwd_dst, diff_dst).
void exec_softmax_bwd(dnnl_engine_t eng, dnnl_stream_t strm,
        const dnn_mem_t &fwd_dst, const dnn_mem_t &diff_dst,
        dnn_mem_t &diff_src, int axis) {
    const auto alg = static_cast<dnnl_alg_kind_t>(
            dnnl::impl::alg_kind::softmax_accurate_inf_as_zero);
    dnnl_primitive_desc_t fwd_pd {};
    DNN_SAFE_V(dnnl_softmax_forward_primitive_desc_create(&fwd_pd, eng,
            dnnl_forward_training, alg, fwd_dst.md_, fwd_dst.md_, axis,
            nullptr));
    auto fwd_pd_w = make_benchdnn_dnnl_wrapper(fwd_pd);

    dnnl_primitive_desc_t bwd_pd {};
    DNN_SAFE_V(dnnl_softmax_backward_primitive_desc_create(&bwd_pd, eng,
            dnnl_softmax_accurate, diff_src.md_, diff_dst.md_, fwd_dst.md_,
            axis, fwd_pd, nullptr));
    auto bwd_pd_w = make_benchdnn_dnnl_wrapper(bwd_pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, bwd_pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_DST, fwd_dst.m_},
            {DNNL_ARG_DIFF_DST, diff_dst.m_},
            {DNNL_ARG_DIFF_SRC, diff_src.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 3, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Create a 3-D f32 plain memory on `eng`.
dnn_mem_t make_3d(dnnl_engine_t eng, int64_t d0, int64_t d1, int64_t d2) {
    dnnl_dims_t dims = {d0, d1, d2};
    auto md = dnn_mem_t::init_md(3, dims, dnnl_f32, tag::abx);
    return dnn_mem_t(md, eng, /* prefill = */ false);
}

// GQA/MQA helper: replicate KV-heads so their count matches Q-heads.
// `src` has [outer_batch * kv_heads, ...] rows, `dst` has
// [outer_batch * q_heads, ...] rows. Each KV-head is copied `groups` times.
void expand_kv_heads(const dnn_mem_t &src, dnn_mem_t &dst, int64_t outer_batch,
        int64_t q_heads, int64_t kv_heads, int64_t head_elems) {
    const float *s = static_cast<float *>(src);
    float *d = static_cast<float *>(dst);
    const int64_t groups = q_heads / kv_heads;
    for (int64_t ob = 0; ob < outer_batch; ob++) {
        for (int64_t kvh = 0; kvh < kv_heads; kvh++) {
            const float *head = s + (ob * kv_heads + kvh) * head_elems;
            for (int64_t g = 0; g < groups; g++) {
                float *out = d + (ob * q_heads + kvh * groups + g) * head_elems;
                std::memcpy(out, head, head_elems * sizeof(float));
            }
        }
    }
}

// GQA/MQA helper: reduce (sum) Q-head groups back into KV-head count.
void reduce_kv_heads(const dnn_mem_t &src, const dnn_mem_t &dst,
        int64_t outer_batch, int64_t q_heads, int64_t kv_heads,
        int64_t head_elems) {
    const float *s = static_cast<float *>(src);
    float *d = static_cast<float *>(dst);
    const int64_t groups = q_heads / kv_heads;
    std::memset(d, 0, outer_batch * kv_heads * head_elems * sizeof(float));
    for (int64_t ob = 0; ob < outer_batch; ob++) {
        for (int64_t kvh = 0; kvh < kv_heads; kvh++) {
            float *out = d + (ob * kv_heads + kvh) * head_elems;
            for (int64_t g = 0; g < groups; g++) {
                const float *in
                        = s + (ob * q_heads + kvh * groups + g) * head_elems;
                for (int64_t e = 0; e < head_elems; e++)
                    out[e] += in[e];
            }
        }
    }
}

// Transpose last two dims of a 3D f32 memory: [d0, d1, d2] → [d0, d2, d1].
dnn_mem_t transpose_2d(dnnl_engine_t eng, const dnn_mem_t &src, int64_t d0,
        int64_t d1, int64_t d2) {
    auto dst = make_3d(eng, d0, d2, d1);
    const float *s = static_cast<float *>(src);
    float *d = static_cast<float *>(dst);
    for (int64_t i = 0; i < d0; i++)
        for (int64_t j = 0; j < d1; j++)
            for (int64_t k = 0; k < d2; k++)
                d[(i * d2 + k) * d1 + j] = s[(i * d1 + j) * d2 + k];
    return dst;
}

} // anonymous namespace

// Scale all elements of `mem` by 1/sqrt(head_size), or by the user-provided
// scale value when --scale=mul or --scale=div is configured.
static void scale_scores(
        const prb_t *prb, const args_t &args, dnn_mem_t &mem, int64_t n) {
    float sv = 1.0f / std::sqrt(static_cast<float>(prb->head_size));
    if (prb->with_scale()) {
        float s = args.find(DNNL_ARG_SCALE).get_f32_elem(0);
        sv = prb->invert_scale() ? 1.0f / s : s;
    }
    float *p = static_cast<float *>(mem);
    for (int64_t i = 0; i < n; i++)
        p[i] *= sv;
}

// Shared forward computation: computes score, applies scale/mask/causal,
// softmax, optional dropout, and optionally the final matmul.
// Returns `score2` = softmax probs (pre-dropout, for backward softmax_bwd)
// and `score2_dp` = post-dropout probs (for BMM2 and backward dV).
// When dropout is not configured, score2_dp == score2.
static void compute_fwd(const prb_t *prb, dnnl_engine_t eng, dnnl_stream_t strm,
        const dnn_mem_t &q_ref, const dnn_mem_t &k_ref, const dnn_mem_t &v_ref,
        const args_t &args, dnn_mem_t &score2, dnn_mem_t &score2_dp,
        dnn_mem_t *out) {
    const int64_t MB = prb->mb;
    const int64_t SQ = prb->n_queries;
    const int64_t SK = prb->n_keys;
    const int64_t V = prb->n_values;

    // Step 1: score = Q x K  (matmul primitive).
    auto score = make_3d(eng, MB, SQ, SK);
    exec_matmul(eng, strm, q_ref, k_ref, score);

    // Step 2: Scale.
    scale_scores(prb, args, score, MB * SQ * SK);

    // Step 3: Apply attention mask (buffer or causal).
    // For causal masks, generate a [1, SQ, SK] buffer with 0/-inf, then
    // apply it through the same broadcast-add path as explicit buffers.
    if (prb->with_mask() || prb->with_causal_mask()) {
        float *sp = static_cast<float *>(score);

        dnn_mem_t causal_buf;
        if (prb->with_causal_mask()) {
            causal_buf = make_3d(eng, 1, SQ, SK);
            float *cp = static_cast<float *>(causal_buf);
            for (int64_t q = 0; q < SQ; q++)
                for (int64_t k = 0; k < SK; k++) {
                    const bool masked = (prb->mask_type == MASK_CAUSAL_TOP_LEFT)
                            ? (k > q)
                            : (k > q + (SK - SQ));
                    cp[q * SK + k] = masked
                            ? -std::numeric_limits<float>::infinity()
                            : 0.0f;
                }
        }

        const float *mp = prb->with_mask()
                ? static_cast<float *>(args.find(DNNL_ARG_ATTN_MASK))
                : static_cast<float *>(causal_buf);

        int64_t msk_mb = 1, msk_sq = SQ;
        if (prb->with_mask()) {
            for (int i = 0; i < prb->ndims - 2; i++)
                msk_mb *= prb->msk_dims[i];
            msk_sq = prb->msk_dims[prb->ndims - 2];
        }
        const int64_t ms_mb = (msk_mb > 1) ? msk_sq * SK : 0;
        const int64_t ms_sq = (msk_sq > 1) ? SK : 0;

        for (int64_t mb = 0; mb < MB; mb++)
            for (int64_t sq = 0; sq < SQ; sq++)
                for (int64_t sk = 0; sk < SK; sk++)
                    sp[(mb * SQ + sq) * SK + sk]
                            += mp[mb * ms_mb + sq * ms_sq + sk];
    }

    // Step 4: Softmax over K dimension (axis = 2 of the 3-D score tensor).
    // Copy to score2 and run softmax in-place there; the pre-softmax score
    // is not used further.
    score2 = make_3d(eng, MB, SQ, SK);
    std::memcpy(static_cast<float *>(score2), static_cast<float *>(score),
            MB * SQ * SK * sizeof(float));
    exec_softmax(eng, strm, score2, /* axis = */ 2);

    // Step 4b: Dropout (optional). score2_dp starts as a copy of score2;
    // when dropout is configured, dropped elements are zeroed and the rest
    // scaled by 1/(1-p).  score2 keeps the clean probs for backward.
    const int64_t score_n = MB * SQ * SK;
    score2_dp = make_3d(eng, MB, SQ, SK);
    std::memcpy(static_cast<float *>(score2_dp), static_cast<float *>(score2),
            score_n * sizeof(float));
    if (!prb->attr.dropout.is_def()) {
        float *sp = static_cast<float *>(score2_dp);
        const dnn_mem_t &dropout_mask = args.find(DNNL_ARG_ATTR_DROPOUT_MASK);
        for (int64_t i = 0; i < score_n; i++)
            maybe_dropout(prb->attr, sp[i], i, dropout_mask);
    }

    // Step 5: output = prob_dp x V  (matmul primitive).
    if (out) {
        *out = make_3d(eng, MB, SQ, V);
        exec_matmul(eng, strm, score2_dp, v_ref, *out);
    }
}

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t) {
    const prb_t *prb = prb_t::from(base_prb);
    const auto &eng = get_cpu_engine();
    stream_t strm(eng);

    const dnn_mem_t &q_m = args.find(DNNL_ARG_QUERIES);
    const dnn_mem_t &k_m = args.find(DNNL_ARG_KEYS);
    const dnn_mem_t &v_m = args.find(DNNL_ARG_VALUES);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const int64_t MB = prb->mb; // product of all batch dims (incl. heads)
    const int64_t SQ = prb->n_queries;
    const int64_t SK = prb->n_keys;
    const int64_t H = prb->head_size;
    const int64_t V = prb->n_values;
    const int nd = prb->ndims;

    // GQA/MQA: K/V may have fewer heads than Q.
    const int64_t q_heads = (nd >= 3) ? prb->q_dims()[nd - 3] : 1;
    const int64_t kv_heads = (nd >= 3) ? prb->k_dims()[nd - 3] : q_heads;
    const int64_t outer_batch = MB / q_heads;
    const bool is_gqa = (kv_heads != q_heads);

    // 3-D f32 memories for matmul: [MB, rows, cols].
    auto q_ref = make_3d(eng, MB, SQ, H);
    auto k_ref = make_3d(eng, MB, H, SK);
    auto v_ref = make_3d(eng, MB, SK, V);

    // Copy Q (always same batch count).
    std::memcpy(static_cast<float *>(q_ref), static_cast<float *>(q_m),
            MB * SQ * H * sizeof(float));

    if (!is_gqa) {
        std::memcpy(static_cast<float *>(k_ref), static_cast<float *>(k_m),
                MB * H * SK * sizeof(float));
        std::memcpy(static_cast<float *>(v_ref), static_cast<float *>(v_m),
                MB * SK * V * sizeof(float));
    } else {
        expand_kv_heads(k_m, k_ref, outer_batch, q_heads, kv_heads, H * SK);
        expand_kv_heads(v_m, v_ref, outer_batch, q_heads, kv_heads, SK * V);
    }

    if (dir & FLAG_FWD) {
        dnn_mem_t score2, score2_dp, out;
        compute_fwd(prb, eng, strm, q_ref, k_ref, v_ref, args, score2,
                score2_dp, &out);

        // Copy result to DST.
        std::memcpy(static_cast<float *>(dst_m), static_cast<float *>(out),
                MB * SQ * V * sizeof(float));

        // Per-element conditioning magnitude  absmag[q,d] = sum_k prob_k*|V_k,d|
        // for the tighter per-element DST threshold (see setup_cmp). prob is the
        // post-dropout softmax. Since prob >= 0, this is matmul(prob, |V|), the
        // same contraction as the output but with |V|, so
        // absmag >= |out| and <= max|V|.
        const dnn_mem_t &absmag_m = args.find(SDPA_REF_ARG_OUT_ABSMAG);
        if (absmag_m.nelems() > 0) {
            auto abs_v = make_3d(eng, MB, SK, V);
            const float *vp = static_cast<float *>(v_ref);
            float *avp = static_cast<float *>(abs_v);
            for (int64_t i = 0; i < MB * SK * V; i++)
                avp[i] = std::fabs(vp[i]);

            auto absmag = make_3d(eng, MB, SQ, V);
            exec_matmul(eng, strm, score2_dp, abs_v, absmag);
            std::memcpy(static_cast<float *>(absmag_m),
                    static_cast<float *>(absmag), MB * SQ * V * sizeof(float));
        }
    }

    if (dir & FLAG_BWD) {
        const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
        const dnn_mem_t &diff_q_m = args.find(DNNL_ARG_DIFF_QUERIES);
        const dnn_mem_t &diff_k_m = args.find(DNNL_ARG_DIFF_KEYS);
        const dnn_mem_t &diff_v_m = args.find(DNNL_ARG_DIFF_VALUES);

        // Recompute forward intermediates to get softmax probabilities.
        // score2 = pre-dropout probs (for softmax_bwd), score2_dp = post-
        // dropout probs (for dV).
        dnn_mem_t score2, score2_dp;
        compute_fwd(prb, eng, strm, q_ref, k_ref, v_ref, args, score2,
                score2_dp, /* out = */ nullptr);

        // dO in 3-D layout [MB, SQ, V].
        auto dO = make_3d(eng, MB, SQ, V);
        std::memcpy(static_cast<float *>(dO), static_cast<float *>(diff_dst_m),
                MB * SQ * V * sizeof(float));

        // B1: dS2 = dO × V^T  →  [MB, SQ, SK]
        auto v_t = transpose_2d(eng, v_ref, MB, SK, V);
        auto dS2 = make_3d(eng, MB, SQ, SK);
        exec_matmul(eng, strm, dO, v_t, dS2);

        // B2: dV = S2_dp^T × dO  →  [MB, SK, V]  (uses post-dropout probs)
        auto s2_t = transpose_2d(eng, score2_dp, MB, SQ, SK);
        auto dV_full = make_3d(eng, MB, SK, V);
        exec_matmul(eng, strm, s2_t, dO, dV_full);

        // B2b: Dropout backward on dS2 (same mask, same scaling as fwd).
        if (!prb->attr.dropout.is_def()) {
            float *dsp = static_cast<float *>(dS2);
            const dnn_mem_t &dropout_mask
                    = args.find(DNNL_ARG_ATTR_DROPOUT_MASK);
            for (int64_t i = 0, n = MB * SQ * SK; i < n; i++)
                maybe_dropout(prb->attr, dsp[i], i, dropout_mask);
        }

        // B3: softmax backward — dS = softmax_bwd(S2, dS2)
        // Uses pre-dropout probs (score2) as the forward DST.
        auto dS = make_3d(eng, MB, SQ, SK);
        exec_softmax_bwd(eng, strm, score2, dS2, dS, /* axis = */ 2);

        // B4: Scale dS.
        scale_scores(prb, args, dS, MB * SQ * SK);

        // B5: dQ = dS × K^T  →  [MB, SQ, H]
        // K is [MB, H, SK], K^T is [MB, SK, H].
        auto k_t = transpose_2d(eng, k_ref, MB, H, SK);
        auto dQ = make_3d(eng, MB, SQ, H);
        exec_matmul(eng, strm, dS, k_t, dQ);

        // B6: dK = Q^T × dS  →  [MB, H, SK]
        auto q_t = transpose_2d(eng, q_ref, MB, SQ, H);
        auto dK_full = make_3d(eng, MB, H, SK);
        exec_matmul(eng, strm, q_t, dS, dK_full);

        // Copy/reduce results into output memories.
        std::memcpy(static_cast<float *>(diff_q_m), static_cast<float *>(dQ),
                MB * SQ * H * sizeof(float));

        if (!is_gqa) {
            std::memcpy(static_cast<float *>(diff_k_m),
                    static_cast<float *>(dK_full), MB * H * SK * sizeof(float));
            std::memcpy(static_cast<float *>(diff_v_m),
                    static_cast<float *>(dV_full), MB * SK * V * sizeof(float));
        } else {
            reduce_kv_heads(
                    dK_full, diff_k_m, outer_batch, q_heads, kv_heads, H * SK);
            reduce_kv_heads(
                    dV_full, diff_v_m, outer_batch, q_heads, kv_heads, SK * V);
        }
    }
}

} // namespace sdpa
