/*******************************************************************************
* Copyright 2020 Intel Corporation
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
 * Cell execution LSTM projection
 */

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

namespace {
void proj_dst_copy(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, void *dst_iter_,
        const void *dst_layer_, dim_t block_step) {
    assert(rnn.dic == rnn.dlc);
    // If dst_iter is not nullptr, we need to copy the state to dst_iter
    if (dst_iter_ == nullptr) return;

    // dst_layer and dst_iter share `cell_dt`, so a byte copy is enough.
    const size_t dt_size = types::data_type_size(rnn.cell_dt);
    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true) * dt_size;
    const auto dst_iter_ld = rnn.dst_iter_ld(cell_position) * dt_size;
    const auto dst_iter = static_cast<char *>(dst_iter_);
    const auto dst_layer = static_cast<const char *>(dst_layer_);

    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (dim_t i = 0; i < rnn.m_block; i++)
            std::memcpy(dst_iter + i * dst_iter_ld,
                    dst_layer + i * dst_layer_ld, block_step);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) {
            std::memcpy(dst_iter + i * dst_iter_ld,
                    dst_layer + i * dst_layer_ld, block_step);
        });
    }
}

void lstm_projection_fwd_postgemm_lp(rnn_postgemm_sig_args) {
    const size_t dt_size = types::data_type_size(rnn.cell_dt);
    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);

    // Currently, scratch_gates_ contains the output of the projection
    const dim_t n_elem = block_step / dt_size;

    const dim_t m_block
            = (rnn.is_brgemm && !rnn.unfused_post_gemm) ? rnn.m_block : rnn.mb;

    const auto dst_layer = static_cast<char *>(dst_layer_);
    const auto scratch_gates = static_cast<const float *>(scratch_gates_);
    for (int i = 0; i < m_block; i++)
        types::cvt_from_float(rnn.cell_dt,
                dst_layer + i * dst_layer_ld * dt_size,
                scratch_gates + i * rnn.scratch_gates_ld, n_elem);

    // we copy to dst_iter if necessary
    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_, block_step);
}

void lstm_projection_fwd_postgemm_u8(
        const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    // Here, we use
    // - scratch_gates to pass the s32 output of the projection
    // - src_iter_c to pass the projection compensation

    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);
    const auto w_proj_comp = static_cast<const float *>(src_iter_c_);
    const auto scratch_gates = static_cast<const int32_t *>(scratch_gates_);
    const auto dst_layer = static_cast<uint8_t *>(dst_layer_);

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        qf = nstl::min(qf, 255.0f);
        qf = nstl::max(qf, 0.0f);
        return q10n::qz_a1b0_t<float, uint8_t>()(qf);
    };

    const auto dequantize_s32_f32 = [&](int32_t s, dim_t j) {
        const float wscale
                = pd->attr()->rnn_weights_projection_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[j];
        const float wcomp = w_proj_comp[j] * data_shift;

        return (q10n::saturate<float>(s) - wcomp) / (wscale * data_scale);
    };

    auto postgemm_call = [&](dim_t i) {
        const dim_t n_elem = block_step / sizeof(uint8_t);
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < n_elem; j++) {
            const dim_t scratch_off = i * rnn.scratch_gates_ld + j;
            const dim_t dst_off = i * dst_layer_ld + j;
            const float tmp = dequantize_s32_f32(scratch_gates[scratch_off], j);
            dst_layer[dst_off] = quantize_f32_u8(tmp);
        }
    };
    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (dim_t i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) { postgemm_call(i); });
    }
    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_, block_step);
}

void lstm_projection_fwd_postgemm_s8(
        const rnn_pd_t *pd, rnn_postgemm_sig_args) {
    // Here, we use
    // - scratch_gates to pass the s32 output of the projection
    // - no need to pass the projection compensation for s8s8 amx
    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);
    const auto scratch_gates = static_cast<const int32_t *>(scratch_gates_);
    const auto dst_layer = static_cast<int8_t *>(dst_layer_);

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_s8 = [&](float f) {
        const float qf = f * data_scale + data_shift;
        return q10n::qz_a1b0_t<float, int8_t>()(qf);
    };

    const auto dequantize_s32_f32 = [&](int32_t s, dim_t j) {
        const float wscale
                = pd->attr()->rnn_weights_projection_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[j];

        return q10n::saturate<float>(s) / (wscale * data_scale);
    };

    const auto postgemm_call = [&](dim_t i) {
        const dim_t n_elem = block_step / sizeof(int8_t);
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < n_elem; j++) {
            const dim_t scratch_off = i * rnn.scratch_gates_ld + j;
            const dim_t dst_off = i * dst_layer_ld + j;
            const float tmp = dequantize_s32_f32(scratch_gates[scratch_off], j);
            dst_layer[dst_off] = quantize_f32_s8(tmp);
        }
    };
    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (dim_t i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) { postgemm_call(i); });
    }
    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_, block_step);
}
} // namespace

rnn_postgemm_sig(rnn_postgemm_fwd_t::lstm_projection_postgemm) {
    switch (rnn.cell_dt) {
        case data_type::f32:
            // nothing to do for f32, except copy to dst_iter if needed
            proj_dst_copy(
                    rnn, cell_position, dst_iter_, dst_layer_, block_step);
            break;
        case data_type::bf16:
        case data_type::f16:
            lstm_projection_fwd_postgemm_lp(rnn_postgemm_args);
            break;
        case data_type::u8:
            lstm_projection_fwd_postgemm_u8(pd_, rnn_postgemm_args);
            break;
        case data_type::s8:
            lstm_projection_fwd_postgemm_s8(pd_, rnn_postgemm_args);
            break;
        default: assert(!"unsupported data type");
    }
}

rnn_postgemm_sig(rnn_postgemm_bwd_t::lstm_projection_postgemm) {
    assert(!"unsupported");
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
