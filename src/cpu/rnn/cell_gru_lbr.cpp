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
 * Cell execution GRU with linear before reset
 */
#pragma warning(disable : 4503) /* name is too long */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

rnn_cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr) {
    if (rnn.is_int8_conf()) {
        assert(!"GRU LBR int8 is not supported");
        return dnnl_unimplemented;
    }

    const auto src_layer_ld = rnn.src_layer_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);

    if (rnn.need_gemm_layer(cell_position)) {
        if (rnn.use_matmul) {
            CHECK(this->execute_matmul(ctx,
                    this->get_matmul_layer(cell_position), w_layer_[0],
                    src_layer_, scratch_gates_));
        } else {
            CHECK((this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dhc,
                    rnn.mb, rnn.slc, 1.0f, w_layer_[0], rnn.weights_layer_ld,
                    src_layer_, src_layer_ld, 0.0f, scratch_gates_,
                    rnn.scratch_gates_ld));
        }
    }
    if (rnn.use_matmul) {
        CHECK(this->execute_matmul(ctx, this->get_matmul_iter(cell_position),
                w_iter_[0], src_iter_, scratch_cell_));
    } else {
        CHECK((this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
                rnn.sic, 1.0f, w_iter_[0], rnn.weights_iter_ld, src_iter_,
                src_iter_ld, 0.0f, scratch_cell_, rnn.scratch_gates_ld));
    }
    this->rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
            diff_src_layer_, diff_augru_attention_, diff_src_iter_,
            diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_, nullptr, nullptr,
            bias_[0], ws_grid_, scratch_cell_, dst_iter_, nullptr, 0);

    return dnnl_success;
}

namespace {

template <typename scratch_data_t>
void gru_lbr_bwd_bias_r_body(const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, const scratch_data_t *scratch_cell_,
        float *diff_bias_) {
    const scratch_gates_aoc_t<const scratch_data_t> scratch_gates_r(
            rnn, scratch_cell_);
    parallel_nd(rnn.dhc, [&](dim_t j) {
        if (rnn.diff_weights_overwrite
                && (cell_position & rnn_utils::last_iter))
            diff_bias_[3 * rnn.dhc + j] = 0.0f;
        for (int i = 0; i < rnn.mb; i++) {
            diff_bias_[3 * rnn.dhc + j] += scratch_gates_r(i, 2, j);
        }
    });
}

void gru_lbr_bwd_bias_r(const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, const void *scratch_cell_,
        float *diff_bias_) {
    switch (rnn.scratch_dt) {
        case data_type::f32:
            gru_lbr_bwd_bias_r_body(rnn, cell_position,
                    static_cast<const float *>(scratch_cell_), diff_bias_);
            break;
        case data_type::bf16:
            gru_lbr_bwd_bias_r_body(rnn, cell_position,
                    static_cast<const bfloat16_t *>(scratch_cell_), diff_bias_);
            break;
        case data_type::f16:
            gru_lbr_bwd_bias_r_body(rnn, cell_position,
                    static_cast<const float16_t *>(scratch_cell_), diff_bias_);
            break;
        default: assert(!"unsupported data type");
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
dnnl_status_t common_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm,
        const rnn_utils::rnn_conf_t &rnn, cell_position_t cell_position,
        void *dst_layer_, float *diff_src_layer_, float *diff_augru_attention_,
        float *diff_src_iter_, void **w_layer_, void **w_iter_, void **bias_,
        const void *src_layer_, const void *augru_attention_,
        const void *src_iter_, float *diff_dst_layer_, float *diff_dst_iter_,
        float *diff_w_layer_, float *diff_w_iter_, float *diff_bias_,
        void *ws_gates_, void *ws_grid_, void *scratch_gates_,
        void *scratch_cell_, void *dst_iter_) {
    const auto src_layer_ld = rnn.src_layer_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);

    rnn_postgemm->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, nullptr, src_iter_, nullptr, nullptr,
            diff_augru_attention_, diff_src_iter_, nullptr, diff_dst_layer_,
            diff_dst_iter_, nullptr, nullptr, bias_[0], ws_grid_, scratch_cell_,
            dst_iter_, nullptr, 0);

    // dWx +=  dG^t * x
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(
                scratch_gates_, src_layer_, src_layer_ld, diff_w_layer_));

    //  dx = dG * Wx^t
    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    // dh +=  dGr * Wh^t
    CHECK(gemm_iter_f(w_iter_[0], scratch_cell_, diff_src_iter_));

    // dWh += dGr^t * h
    CHECK(gemm_weights_iter_f(
            scratch_cell_, src_iter_, src_iter_ld, diff_w_iter_));

    // db1-3 += e * dG
    // db4 += e * (r * dG2)
    gates_reduction(rnn, cell_position, scratch_gates_, diff_bias_);
    gru_lbr_bwd_bias_r(rnn, cell_position, scratch_cell_, diff_bias_);

    return dnnl_success;
}

} // namespace

rnn_cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr) {
    const auto gemm_layer = [&](const void *A, const void *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    const auto gemm_iter = [&](const void *A, const void *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_iter_ld, B,
                rnn.ws_gates_ld, 1.0f, C, rnn.ws_diff_states_iter_ld);
    };
    const auto gemm_weights_layer
            = [&](const void *A, const void *B, dim_t ldb, float *C) {
        const float beta = rnn.diff_weights_beta(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0f, A,
                rnn.scratch_gates_ld, B, ldb, beta, C,
                rnn.diff_weights_layer_ld);
    };
    const auto gemm_weights_iter
            = [&](const void *A, const void *B, dim_t ldb, float *C) {
        const float beta = rnn.diff_weights_beta(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb, 1.0f, A,
                rnn.ws_gates_ld, B, ldb, beta, C, rnn.diff_weights_iter_ld);
    };

    CHECK(common_bwd_cell_exec_template(gemm_layer, gemm_iter,
            gemm_weights_layer, gemm_weights_iter, this->rnn_postgemm_, rnn,
            cell_position, dst_layer_, diff_src_layer_, diff_augru_attention_,
            diff_src_iter_, w_layer_, w_iter_, bias_, src_layer_,
            augru_attention_, src_iter_, diff_dst_layer_, diff_dst_iter_,
            diff_w_layer_, diff_w_iter_, diff_bias_, ws_gates_, ws_grid_,
            scratch_gates_, scratch_cell_, dst_iter_));
    return dnnl_success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
