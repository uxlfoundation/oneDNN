/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/*
 * Cell execution LSTM
 */

#include "common/bit_cast.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/rnn/postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <typename src_data_t, typename scratch_data_t, typename T1,
        typename T2, typename T3, typename T4, typename T5>
void gru_fwd_part1_postgemm_template(T1 func1, T2 to_src, T3 acc_to_float,
        T4 src_to_float, T5 reinterpret_as_acc, const float *scales,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, void *ws_gates_,
        void *scratch_gates_, const void *augru_attention_, void *dst_layer_,
        void *dst_iter_, const void *src_iter_, const void *bias_,
        dim_t block_step) {
    const ws_gates_aoc_t<src_data_t> ws_gates(
            rnn, static_cast<src_data_t *>(ws_gates_));
    const scratch_gates_aoc_t<scratch_data_t> scratch_gates(
            rnn, static_cast<scratch_data_t *>(scratch_gates_));
    const auto bias_aoc = rnn_utils::make_raw_aoc(
            bias_, types::data_type_size(rnn.bias_dt), rnn.n_bias, rnn.dhc);
    const auto bias = [&](int gate_id, dim_t dhc_id) {
        return to_float(bias_aoc(gate_id, dhc_id), rnn.bias_dt);
    };

    const auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);

    const ws_states_layer_aoc_t<src_data_t> dst_layer(
            rnn, static_cast<src_data_t *>(dst_layer_), dst_layer_ld);
    const ws_states_iter_aoc_t<src_data_t> dst_iter(
            rnn, static_cast<src_data_t *>(dst_iter_), dst_iter_ld);
    const ws_states_iter_aoc_t<const src_data_t> src_iter(
            rnn, static_cast<const src_data_t *>(src_iter_), src_iter_ld);

    const float *scales_G1 = scales ? scales + 1 : nullptr;

    const auto postgemm_call = [&](dim_t i) {
        const dim_t n_elem = block_step;
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < n_elem; j++) {
            const auto G0 // default func1 is sigmoid
                    = func1(scales,
                            acc_to_float(scratch_gates(i, 0, j), 0, j)
                                    + bias(0, j));
            const auto G1 // default func1 is sigmoid
                    = func1(scales_G1,
                            acc_to_float(scratch_gates(i, 1, j), 1, j)
                                    + bias(1, j));
            /* TODO: Can be optimized for fwd_training by using ws_gates instead of scratch_gates in p2 */
            scratch_gates(i, 0, j) = reinterpret_as_acc(G0);
            const auto t = to_src(src_to_float(src_iter(i, j)) * G1);
            if (dst_layer_) dst_layer(i, j) = t;
            if (dst_iter_) dst_iter(i, j) = t;

            if (rnn.is_training) {
                ws_gates(i, 0, j) = to_src(G0);
                ws_gates(i, 1, j) = to_src(G1);
            }
        }
    };

    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (dim_t i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) { postgemm_call(i); });
    }
}

template <typename src_data_t, typename scratch_data_t, typename T1,
        typename T2, typename T3, typename T4, typename T5>
void gru_fwd_part2_postgemm_template(T1 func1, T2 to_src, T3 acc_to_float,
        T4 src_to_float, T5 reinterpret_as_float, const float *scales,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, void *ws_gates_,
        void *scratch_gates_, const void *augru_attention_, void *dst_layer_,
        void *dst_iter_, const void *src_iter_, const void *bias_,
        dim_t block_step) {
    const ws_gates_aoc_t<src_data_t> ws_gates(
            rnn, static_cast<src_data_t *>(ws_gates_));
    const scratch_gates_aoc_t<scratch_data_t> scratch_gates(
            rnn, static_cast<scratch_data_t *>(scratch_gates_));
    const auto bias_aoc = rnn_utils::make_raw_aoc(
            bias_, types::data_type_size(rnn.bias_dt), rnn.n_bias, rnn.dhc);
    const auto bias = [&](int gate_id, dim_t dhc_id) {
        return to_float(bias_aoc(gate_id, dhc_id), rnn.bias_dt);
    };

    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    const auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);
    const augru_attention_aoc_t<const src_data_t> augru_attention(
            rnn, static_cast<const src_data_t *>(augru_attention_));
    const ws_states_layer_aoc_t<src_data_t> dst_layer(
            rnn, static_cast<src_data_t *>(dst_layer_), dst_layer_ld);
    const ws_states_iter_aoc_t<src_data_t> dst_iter(
            rnn, static_cast<src_data_t *>(dst_iter_), dst_iter_ld);
    const ws_states_iter_aoc_t<const src_data_t> src_iter(
            rnn, static_cast<const src_data_t *>(src_iter_), src_iter_ld);

    const float *scales_G2 = scales ? scales + 2 : nullptr;

    const auto postgemm_call = [&](dim_t i) {
        const dim_t n_elem = block_step;
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < n_elem; j++) {
            auto G0 = reinterpret_as_float(scratch_gates(i, 0, j));
            const auto G2 // default func1 is tanh
                    = func1(scales_G2,
                            acc_to_float(scratch_gates(i, 2, j), 2, j)
                                    + bias(2, j));

            if (rnn.is_augru) {
                const auto a = reinterpret_as_float(augru_attention(i));
                G0 = (1.0f - a) * G0;
            }

            const auto tmp = to_src(
                    src_to_float(src_iter(i, j)) * G0 + (1.0f - G0) * G2);
            if (dst_layer_ != nullptr) dst_layer(i, j) = tmp;
            if (dst_iter_ != nullptr) dst_iter(i, j) = tmp;

            if (rnn.is_training) { ws_gates(i, 2, j) = to_src(G2); }
        }
    };

    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (dim_t i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) { postgemm_call(i); });
    }
}

namespace {
template <typename gates_t>
void gru_fwd_part1_postgemm(const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    const float *scales = pd->attr()->rnn_tparams_.scales_;
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };

    const auto cvt_from_f32 = [](float f) { return static_cast<gates_t>(f); };
    const auto cvt_to_f32 = [](gates_t b) { return float(b); };
    const auto deq_id = [](float f, int i, dim_t j) { return f; };
    const auto id = [](float f) { return f; };

    if (!pd->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part1_postgemm_template<gates_t, float>(logistic_f,
                cvt_from_f32, deq_id, cvt_to_f32, id, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, augru_attention_,
                dst_layer_, dst_iter_, src_iter_, bias_, block_step);
    else
        gru_fwd_part1_postgemm_template<gates_t, float>(linear_f, cvt_from_f32,
                deq_id, cvt_to_f32, id, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
}

template <typename gates_t>
void gru_fwd_part2_postgemm(const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    const float *scales = pd->attr()->rnn_tparams_.scales_;
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    const auto cvt_from_f32 = [](float f) { return static_cast<gates_t>(f); };
    const auto cvt_to_f32 = [](gates_t b) { return float(b); };
    const auto deq_id = [](float f, int i, dim_t j) { return f; };
    const auto id = [](float f) { return f; };

    if (!pd->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part2_postgemm_template<gates_t, float>(tanh_f, cvt_from_f32,
                deq_id, cvt_to_f32, id, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
    else
        gru_fwd_part2_postgemm_template<gates_t, float>(linear_f, cvt_from_f32,
                deq_id, cvt_to_f32, id, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
}

void gru_fwd_part1_postgemm_u8(const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    const float *scales = pd->attr()->rnn_tparams_.scales_;
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        qf = nstl::min(qf, 255.0f);
        qf = nstl::max(qf, 0.0f);
        return (uint8_t)mxcsr_cvt(qf);
    };

    const auto dequantize_s32_f32 = [&](int32_t s, int gate, dim_t j) {
        const float wscale = pd->attr()->rnn_weights_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[gate * rnn.dhc + j];
        return q10n::saturate<float>(s) * (1.f / (wscale * data_scale));
    };

    const auto dequantize_u8_f32 = [&](uint8_t s) {
        return (static_cast<float>(s) - data_shift) * (1.f / data_scale);
    };

    const auto reinterpret_f32_s32
            = [](float a) { return bit_cast<int32_t>(a); };

    if (!pd->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part1_postgemm_template<uint8_t, int32_t>(logistic_f,
                quantize_f32_u8, dequantize_s32_f32, dequantize_u8_f32,
                reinterpret_f32_s32, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
    else
        gru_fwd_part1_postgemm_template<uint8_t, int32_t>(linear_f,
                quantize_f32_u8, dequantize_s32_f32, dequantize_u8_f32,
                reinterpret_f32_s32, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
}

void gru_fwd_part2_postgemm_u8(const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    const float *scales = pd->attr()->rnn_tparams_.scales_;
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        qf = nstl::min(qf, 255.0f);
        qf = nstl::max(qf, 0.0f);
        return (uint8_t)mxcsr_cvt(qf);
    };

    const auto dequantize_s32_f32 = [&](int32_t s, int gate, dim_t j) {
        const float wscale = pd->attr()->rnn_weights_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[gate * rnn.dhc + j];
        return q10n::saturate<float>(s) * (1.f / (wscale * data_scale));
    };

    const auto dequantize_u8_f32 = [&](uint8_t s) {
        return (static_cast<float>(s) - data_shift) * (1.f / data_scale);
    };

    const auto reinterpret_s32_f32
            = [](int32_t a) { return bit_cast<float>(a); };

    if (!pd->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part2_postgemm_template<uint8_t, int32_t>(tanh_f,
                quantize_f32_u8, dequantize_s32_f32, dequantize_u8_f32,
                reinterpret_s32_f32, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
    else
        gru_fwd_part2_postgemm_template<uint8_t, int32_t>(linear_f,
                quantize_f32_u8, dequantize_s32_f32, dequantize_u8_f32,
                reinterpret_s32_f32, scales, rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_,
                src_iter_, bias_, block_step);
}
} // namespace

rnn_postgemm_sig(rnn_postgemm_fwd_t::gru_part1_postgemm) {
    switch (rnn.cell_dt) {
        case data_type::f32:
            gru_fwd_part1_postgemm<float>(pd_, rnn_postgemm_args);
            break;
        case data_type::bf16:
            gru_fwd_part1_postgemm<bfloat16_t>(pd_, rnn_postgemm_args);
            break;
        case data_type::f16:
            gru_fwd_part1_postgemm<float16_t>(pd_, rnn_postgemm_args);
            break;
        case data_type::u8:
            gru_fwd_part1_postgemm_u8(pd_, rnn_postgemm_args);
            break;
        case data_type::s8: assert(!"GRU signed int8 is not supported"); break;
        default: assert(!"unsupported data type");
    }
}

rnn_postgemm_sig(rnn_postgemm_fwd_t::gru_part2_postgemm) {
    switch (rnn.cell_dt) {
        case data_type::f32:
            gru_fwd_part2_postgemm<float>(pd_, rnn_postgemm_args);
            break;
        case data_type::bf16:
            gru_fwd_part2_postgemm<bfloat16_t>(pd_, rnn_postgemm_args);
            break;
        case data_type::f16:
            gru_fwd_part2_postgemm<float16_t>(pd_, rnn_postgemm_args);
            break;
        case data_type::u8:
            gru_fwd_part2_postgemm_u8(pd_, rnn_postgemm_args);
            break;
        case data_type::s8: assert(!"GRU signed int8 is not supported"); break;
        default: assert(!"unsupported data type");
    }
}

template <typename src_data_t, typename acc_data_t, typename scratch_data_t,
        typename T>
void gru_bwd_part1_postgemm_template(T to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, void *ws_gates_, void *scratch_gates_,
        const void *augru_attention_, void *dst_layer_, const void *src_iter_,
        void *diff_src_iter_, void *diff_dst_iter_, void *diff_augru_attention_,
        void *diff_dst_layer_) {
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);

    const augru_attention_aoc_t<const src_data_t> augru_attention(
            rnn, static_cast<const src_data_t *>(augru_attention_));
    const augru_attention_aoc_t<acc_data_t> diff_augru_attention(
            rnn, static_cast<acc_data_t *>(diff_augru_attention_));

    const ws_states_iter_aoc_t<const src_data_t> src_iter(
            rnn, static_cast<const src_data_t *>(src_iter_), src_iter_ld);
    const ws_gates_aoc_t<src_data_t> ws_gates(
            rnn, static_cast<src_data_t *>(ws_gates_));
    const scratch_gates_aoc_t<scratch_data_t> scratch_gates(
            rnn, static_cast<scratch_data_t *>(scratch_gates_));
    const ws_diff_states_iter_aoc_t<acc_data_t> diff_src_iter(
            rnn, static_cast<acc_data_t *>(diff_src_iter_));
    const ws_diff_states_iter_aoc_t<acc_data_t> diff_dst_iter(
            rnn, static_cast<acc_data_t *>(diff_dst_iter_));
    const ws_diff_states_layer_aoc_t<acc_data_t> diff_dst_layer(
            rnn, static_cast<acc_data_t *>(diff_dst_layer_));

    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(rnn.mb, [&](dim_t i) {
        acc_data_t diff_attention = 0.0f;
        PRAGMA_OMP_SIMD(reduction(+ : diff_attention))
        for (dim_t j = 0; j < rnn.dhc; j++) {
            const float h = src_iter(i, j);
            const float dHt = diff_dst_iter(i, j) + diff_dst_layer(i, j);
            const float dG2 = (1.0f - ws_gates(i, 0, j)) * dHt
                    * one_m_square(ws_gates(i, 2, j));
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));

            if (rnn.is_augru) {
                diff_attention -= dG0 * ws_gates(i, 0, j);
                dG0 *= 1.0f - augru_attention(i);
            }

            diff_src_iter(i, j) = dHt * ws_gates(i, 0, j);
            scratch_gates(i, 0, j) = to_src(dG0);
            scratch_gates(i, 2, j) = to_src(dG2);
        }
        if (rnn.is_augru) diff_augru_attention(i) = diff_attention;
    });
}

template <typename src_data_t, typename acc_data_t, typename scratch_data_t,
        typename T>
void gru_bwd_part2_postgemm_template(T to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, void *ws_gates_, void *scratch_gates_,
        void *dst_layer_, const void *src_iter_, void *diff_src_layer_,
        void *diff_src_iter_, void *diff_dst_iter_, void *diff_dst_layer_,
        void *scratch_cell_) {
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);
    // auto dst_ld = rnn.dst_ld(cell_position);
    // ws_states_layer_aoc_t<src_data_t> dst_layer(rnn, dst_layer_, dst_ld);
    const ws_states_iter_aoc_t<const src_data_t> src_iter(
            rnn, static_cast<const src_data_t *>(src_iter_), src_iter_ld);
    const ws_gates_aoc_t<src_data_t> ws_gates(
            rnn, static_cast<src_data_t *>(ws_gates_));
    const scratch_gates_aoc_t<scratch_data_t> scratch_gates(
            rnn, static_cast<scratch_data_t *>(scratch_gates_));
    const ws_diff_states_layer_aoc_t<acc_data_t> diff_dst_layer(
            rnn, static_cast<acc_data_t *>(diff_dst_layer_));
    const ws_diff_states_iter_aoc_t<acc_data_t> diff_dst_iter(
            rnn, static_cast<acc_data_t *>(diff_dst_iter_));

    const ws_diff_states_layer_aoc_t<acc_data_t> dhG1(
            rnn, static_cast<acc_data_t *>(diff_src_layer_));
    const ws_diff_states_iter_aoc_t<acc_data_t> diff_src_iter(
            rnn, static_cast<acc_data_t *>(diff_src_iter_));
    const AOC<scratch_data_t, 2> hG1(
            static_cast<scratch_data_t *>(scratch_cell_),
            rnn.ws_states_layer_nld, rnn.ws_states_layer_ld);

    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(rnn.mb, [&](dim_t i) {
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < rnn.dhc; j++) {
            const float h = src_iter(i, j);
            const float G1 = ws_gates(i, 1, j);
            diff_src_iter(i, j) += dhG1(i, j) * G1;
            scratch_gates(i, 1, j) = to_src(dhG1(i, j) * h * x_m_square(G1));
            hG1(i, j) = to_src(G1 * h);
        }
    });
}

namespace {
template <typename gates_t>
void gru_bwd_part1_postgemm(rnn_postgemm_sig_args) {
    const auto to_src = [](float a) { return gates_t(a); };

    gru_bwd_part1_postgemm_template<gates_t, float, gates_t>(to_src, rnn,
            cell_position, ws_gates_, scratch_gates_, augru_attention_,
            dst_layer_, src_iter_, diff_src_iter_, diff_dst_iter_,
            diff_augru_attention_, diff_dst_layer_);
}

template <typename gates_t>
void gru_bwd_part2_postgemm(rnn_postgemm_sig_args) {
    const auto to_src = [](float a) { return gates_t(a); };

    gru_bwd_part2_postgemm_template<gates_t, float, gates_t>(to_src, rnn,
            cell_position, ws_gates_, scratch_gates_, dst_layer_, src_iter_,
            diff_src_layer_, diff_src_iter_, diff_dst_iter_, diff_dst_layer_,
            scratch_cell_);
}
} // namespace

rnn_postgemm_sig(rnn_postgemm_bwd_t::gru_part1_postgemm) {
    switch (rnn.cell_dt) {
        case data_type::f32:
            gru_bwd_part1_postgemm<float>(rnn_postgemm_args);
            break;
        case data_type::bf16:
            gru_bwd_part1_postgemm<bfloat16_t>(rnn_postgemm_args);
            break;
        case data_type::f16:
            gru_bwd_part1_postgemm<float16_t>(rnn_postgemm_args);
            break;
        default: assert(!"unsupported data type");
    }
}

rnn_postgemm_sig(rnn_postgemm_bwd_t::gru_part2_postgemm) {
    switch (rnn.cell_dt) {
        case data_type::f32:
            gru_bwd_part2_postgemm<float>(rnn_postgemm_args);
            break;
        case data_type::bf16:
            gru_bwd_part2_postgemm<bfloat16_t>(rnn_postgemm_args);
            break;
        case data_type::f16:
            gru_bwd_part2_postgemm<float16_t>(rnn_postgemm_args);
            break;
        default: assert(!"unsupported data type");
    }
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace dnnl
