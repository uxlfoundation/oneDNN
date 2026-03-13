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
// Pipeline: score = matmul(Q, K)  ->  scale  ->  [mask]  ->  [causal]
//           ->  softmax(score)  ->  matmul(prob, V)  ->  DST
//
// All intermediate computation is done in f32 on the CPU engine.

namespace {

// Execute a matmul primitive on CPU: dst = src x wei.
void exec_matmul(dnnl_engine_t eng, dnnl_stream_t strm,
        const dnn_mem_t &src, const dnn_mem_t &wei, dnn_mem_t &dst) {
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

// Create a 3-D f32 plain memory on `eng`.
dnn_mem_t make_3d(dnnl_engine_t eng, int64_t d0, int64_t d1, int64_t d2) {
    dnnl_dims_t dims = {d0, d1, d2};
    auto md = dnn_mem_t::init_md(3, dims, dnnl_f32, tag::abx);
    return dnn_mem_t(md, eng, /* prefill = */ false);
}

// GQA/MQA helper: replicate KV-heads so their count matches Q-heads.
// `src` has [outer_batch * kv_heads, ...] rows, `dst` has
// [outer_batch * q_heads, ...] rows. Each KV-head is copied `groups` times.
void expand_kv_heads(const dnn_mem_t &src, dnn_mem_t &dst,
        int64_t outer_batch, int64_t q_heads, int64_t kv_heads,
        int64_t head_elems) {
    const float *s = static_cast<float *>(src);
    float *d = static_cast<float *>(dst);
    const int64_t groups = q_heads / kv_heads;
    for (int64_t ob = 0; ob < outer_batch; ob++) {
        for (int64_t kvh = 0; kvh < kv_heads; kvh++) {
            const float *head = s + (ob * kv_heads + kvh) * head_elems;
            for (int64_t g = 0; g < groups; g++) {
                float *out
                        = d + (ob * q_heads + kvh * groups + g) * head_elems;
                std::memcpy(out, head, head_elems * sizeof(float));
            }
        }
    }
}

} // anonymous namespace

void compute_ref(
        const prb_t *prb, dir_t dir, const args_t &args, dnnl_primitive_t) {
    const auto eng = get_cpu_engine();
    dnnl_stream_t strm {};
    DNN_SAFE_V(dnnl_stream_create(&strm, eng, dnnl_stream_default_flags));

    const dnn_mem_t &q_m = args.find(DNNL_ARG_SRC_0);
    const dnn_mem_t &k_m = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &v_m = args.find(DNNL_ARG_SRC_2);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const int64_t MB = prb->mb; // product of all batch dims (incl. heads)
    const int64_t SQ = prb->n_queries;
    const int64_t SK = prb->n_keys;
    const int64_t H = prb->head_size;
    const int64_t V = prb->n_values;
    const int nd = prb->ndims;

    // GQA/MQA: K/V may have fewer heads than Q.
    const int64_t q_heads = (nd >= 3) ? prb->q_dims()[nd - 3] : 1;
    const int64_t kv_heads = (nd >= 3 && prb->kv_head_number > 0)
            ? prb->kv_head_number
            : q_heads;
    const int64_t outer_batch = MB / q_heads;
    const bool is_gqa = (kv_heads != q_heads);

    // 3-D f32 memories for matmul: [MB, rows, cols].
    auto q_ref = make_3d(eng, MB, SQ, H);
    auto k_ref = make_3d(eng, MB, H, SK);
    auto v_ref = make_3d(eng, MB, SK, V);
    auto score = make_3d(eng, MB, SQ, SK);
    auto out = make_3d(eng, MB, SQ, V);

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

    // Step 1: score = Q x K  (matmul primitive).
    exec_matmul(eng, strm, q_ref, k_ref, score);

    // Step 2: Scale.
    {
        float sv = 1.0f / std::sqrt(static_cast<float>(H));
        if (prb->with_scale()) {
            float s = args.find(DNNL_ARG_SCALE).get_f32_elem(0);
            sv = prb->invert_scale() ? 1.0f / s : s;
        }
        float *sp = static_cast<float *>(score);
        for (int64_t i = 0, n = MB * SQ * SK; i < n; i++)
            sp[i] *= sv;
    }

    // Step 3: Add attention mask buffer.
    if (prb->with_mask()) {
        float *sp = static_cast<float *>(score);
        const float *mp = static_cast<float *>(args.find(DNNL_ARG_SHIFT));
        for (int64_t i = 0, n = MB * SQ * SK; i < n; i++)
            sp[i] += mp[i];
    }

    // Step 4: Apply causal mask (set future positions to -inf).
    if (prb->with_causal_mask()) {
        float *sp = static_cast<float *>(score);
        for (int64_t b = 0; b < MB; b++)
            for (int64_t q = 0; q < SQ; q++)
                for (int64_t k = 0; k < SK; k++) {
                    const bool masked
                            = (prb->mask_type == MASK_CAUSAL_TOP_LEFT)
                            ? (k > q)
                            : (k > q + (SK - SQ));
                    if (masked)
                        sp[(b * SQ + q) * SK + k]
                                = -std::numeric_limits<float>::infinity();
                }
    }

    // Step 5: Softmax over K dimension (axis = 2 of the 3-D score tensor).
    exec_softmax(eng, strm, score, /* axis = */ 2);

    // Step 6: output = prob x V  (matmul primitive).
    exec_matmul(eng, strm, score, v_ref, out);

    // Copy result to DST.
    std::memcpy(static_cast<float *>(dst_m), static_cast<float *>(out),
            MB * SQ * V * sizeof(float));

    DNN_SAFE_V(dnnl_stream_destroy(strm));
}

} // namespace sdpa
