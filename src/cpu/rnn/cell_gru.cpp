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
 * Cell execution GRU
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

rnn_cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru) {
    void *scratch_gates_2
            = inc_ptr(scratch_gates_, rnn.scratch_dt, 2 * rnn.dhc);
    const auto weights_scales = this->pd_->attr()->rnn_weights_qparams_.scales_;

    const auto src_layer_ld = rnn.src_layer_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);
    const auto dst_iter_part2_ld = rnn.dst_iter_part2_ld(cell_position);

    // 1. gemm Wx[0-2],x
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

    // 2. gemm Wh[0-1],h
    if (rnn.use_matmul) {
        CHECK(this->execute_matmul(ctx, this->get_matmul_iter(cell_position),
                w_iter_[0], src_iter_, scratch_gates_));
    } else {
        CHECK((this->*gemm_iter_func)('N', 'N', (rnn.n_gates - 1) * rnn.dhc,
                rnn.mb, rnn.sic, 1.0f, w_iter_[0], rnn.weights_iter_ld,
                src_iter_, src_iter_ld, 1.0f, scratch_gates_,
                rnn.scratch_gates_ld));
    }
    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    this->rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, nullptr, src_iter_, nullptr,
            diff_src_layer_, diff_augru_attention_, diff_src_iter_, nullptr,
            diff_dst_layer_, diff_dst_iter_, nullptr, nullptr, bias_[0],
            nullptr, nullptr, dst_iter_, weights_scales, rnn.dhc);

    // 4. gemm Wh[2],h~t
    if (rnn.use_matmul) {
        CHECK(this->execute_matmul(ctx, this->get_matmul_part2(cell_position),
                w_iter_[1], dst_layer_, scratch_gates_2));
    } else {
        CHECK((this->*gemm_iter_func)('N', 'N', rnn.dhc, rnn.mb, rnn.sic, 1.0f,
                w_iter_[1], rnn.weights_iter_ld, dst_layer_, dst_iter_part2_ld,
                1.0f, scratch_gates_2, rnn.scratch_gates_ld));
    }
    // 5. activation h~t + calculate ht
    this->rnn_postgemm_->execute_part2(rnn, cell_position, ws_gates_,
            scratch_gates_, augru_attention_, dst_layer_, dst_iter_c_,
            src_iter_, src_iter_c_, diff_src_layer_, diff_augru_attention_,
            diff_src_iter_, nullptr, diff_dst_layer_, diff_dst_iter_, nullptr,
            nullptr, bias_[0], nullptr, nullptr, dst_iter_, weights_scales,
            rnn.dhc);

    return dnnl_success;
}

namespace {

template <typename T1, typename T2, typename T3, typename T4, typename T5>
dnnl_status_t gru_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm_,
        const rnn_utils::rnn_conf_t &rnn, cell_position_t cell_position,
        void *ws_gates_, void *scratch_gates_, void *dst_layer_,
        const void *src_iter_, const void *src_layer_,
        const void *augru_attention_, void **w_layer_, void **w_iter_,
        float *diff_w_layer_, float *diff_w_iter_, float *diff_src_layer_,
        float *diff_augru_attention_, float *diff_src_iter_,
        float *diff_dst_iter_, float *diff_dst_layer_, float *diff_bias_,
        void *scratch_cell_, void *dst_iter_) {
    void *scratch_gates_2
            = inc_ptr(scratch_gates_, rnn.scratch_dt, 2 * rnn.dhc);

    const auto src_layer_ld = rnn.src_layer_ld(cell_position);
    const auto src_iter_ld = rnn.src_iter_ld(cell_position);
    const ws_diff_w_iter_aoc_t diff_w_iter(rnn, diff_w_iter_);

    // use state memory for intermediate computations
    // TODO: use cell ws for that
    float *dhG1_ = diff_src_layer_;

    // 1. calculate dG2, dG1, and part of dht-1
    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, nullptr, src_iter_, nullptr,
            diff_src_layer_, diff_augru_attention_, diff_src_iter_, nullptr,
            diff_dst_layer_, diff_dst_iter_, nullptr, nullptr, nullptr, nullptr,
            scratch_cell_, dst_iter_, nullptr, 0);

    // 2. calculate intermediate d(hG1)
    // d(hG1) = dG2 * W2h^t
    CHECK(gemm_iter_f(rnn.sic, rnn.mb, rnn.dhc, w_iter_[1], scratch_gates_2,
            0.0f, dhG1_));

    // 3. calculate dG1^ and part of dht-1
    rnn_postgemm_->execute_part2(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, nullptr, src_iter_, nullptr,
            diff_src_layer_, diff_augru_attention_, diff_src_iter_, nullptr,
            diff_dst_layer_, diff_dst_iter_, nullptr, nullptr, nullptr, nullptr,
            scratch_cell_, dst_iter_, nullptr, 0);

    // 4. calculate diff weights
    // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    CHECK(gemm_weights_iter_f((rnn.n_gates - 1) * rnn.dhc, rnn.sic, rnn.mb,
            scratch_gates_, src_iter_, src_iter_ld, diff_w_iter_));
    CHECK(gemm_weights_iter_f(rnn.dhc, rnn.sic, rnn.mb, scratch_gates_2,
            scratch_cell_, rnn.ws_states_layer_ld, &(diff_w_iter(0, 2, 0))));

    // 5. calculate diff states
    // dht-1 += dG1 * W1h + dG0 * W0h
    CHECK(gemm_iter_f(rnn.sic, rnn.mb, (rnn.n_gates - 1) * rnn.dhc, w_iter_[0],
            scratch_gates_, 1.0f, diff_src_iter_));

    // dWx += [dG0 dG1 dG2] * [x]
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(
                scratch_gates_, src_layer_, src_layer_ld, diff_w_layer_));

    // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    // 6. calculate diff bias
    gates_reduction(rnn, cell_position, scratch_gates_, diff_bias_);

    return dnnl_success;
}

} // namespace

rnn_cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru) {
    auto gemm_iter_f = [&](dim_t m, dim_t n, dim_t k, const void *A,
                               const void *B, float beta, float *C) {
        return (this->*gemm_iter_func)('N', 'N', m, n, k, 1.0f, A,
                rnn.weights_iter_ld, B, rnn.scratch_gates_ld, beta, C,
                rnn.ws_diff_states_iter_ld);
    };
    auto gemm_layer_f = [&](const void *A, const void *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_weights_layer_f
            = [&](const void *A, const void *B, dim_t ldb, float *C) {
        const float beta = rnn.diff_weights_beta(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0f, A,
                rnn.scratch_gates_ld, B, ldb, beta, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter_f = [&](dim_t m, dim_t n, dim_t k, const void *A,
                                       const void *B, dim_t ldb, float *C) {
        const float beta = rnn.diff_weights_beta(cell_position);
        return gemm('N', 'T', m, n, k, 1.0f, A, rnn.ws_gates_ld, B, ldb, beta,
                C, rnn.diff_weights_iter_ld);
    };

    return gru_bwd_cell_exec_template(gemm_layer_f, gemm_iter_f,
            gemm_weights_layer_f, gemm_weights_iter_f, this->rnn_postgemm_, rnn,
            cell_position, ws_gates_, scratch_gates_, dst_layer_, src_iter_,
            src_layer_, augru_attention_, w_layer_, w_iter_, diff_w_layer_,
            diff_w_iter_, diff_src_layer_, diff_augru_attention_,
            diff_src_iter_, diff_dst_iter_, diff_dst_layer_, diff_bias_,
            scratch_cell_, dst_iter_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
