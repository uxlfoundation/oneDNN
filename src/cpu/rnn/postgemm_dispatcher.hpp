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

#ifndef CPU_RNN_POSTGEMM_DISPATCHER_HPP
#define CPU_RNN_POSTGEMM_DISPATCHER_HPP

#include <memory>

#include "common/z_magic.hpp"

#include "cpu/platform.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/rnn_utils.hpp"

#if DNNL_X64
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_1_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_1_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_2_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_2_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_lbr_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_lbr_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

float activation(alg_kind_t alg_kind, prop_kind_t prop_kind, float s,
        float alpha, float cliping);

#if DNNL_X64
template <template <x64::cpu_isa_t> class ker_t>
x64::jit_uni_rnn_postgemm_t *create_jit_postgemm(
        const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd,
        data_type_t src_dt, data_type_t scratch_dt) {
    if (x64::mayiuse(x64::avx512_core))
        return new ker_t<x64::avx512_core>(rnn, pd, src_dt, scratch_dt);
    return new ker_t<x64::avx2>(rnn, pd, src_dt, scratch_dt);
}
#endif

struct rnn_postgemm_dispatcher_t {
    using postgemm_f
            = void (rnn_postgemm_dispatcher_t::*)(rnn_postgemm_sig_args) const;

    rnn_postgemm_dispatcher_t(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : pd_(pd) {
        // add check if in testing mode
        if (pd->attr()->rnn_tparams_.test_mode_) {
            const auto ngates = utils::map(pd->cell_kind(), 0,
                    alg_kind::vanilla_rnn, 1, alg_kind::vanilla_lstm, 4,
                    alg_kind::vanilla_gru, 3, alg_kind::lbr_gru, 3,
                    alg_kind::vanilla_augru, 3, alg_kind::lbr_augru, 3);
            assert(pd->attr()->rnn_tparams_.ngates_ == ngates);
            MAYBE_UNUSED(ngates);
        }

        switch (pd->cell_kind()) {
            case alg_kind::vanilla_lstm:
                postgemm_func = &rnn_postgemm_dispatcher_t::lstm_postgemm;
                // used for int8 requantization after projection
                postgemm_part2_func = pd->is_lstm_projection() && pd_->is_fwd()
                        ? &rnn_postgemm_dispatcher_t::lstm_projection_postgemm
                        : nullptr;
                break;
            case alg_kind::vanilla_rnn:
                postgemm_func = &rnn_postgemm_dispatcher_t::rnn_postgemm;
                break;
            case alg_kind::vanilla_gru:
            case alg_kind::vanilla_augru:
                postgemm_func = &rnn_postgemm_dispatcher_t::gru_part1_postgemm;
                postgemm_part2_func
                        = &rnn_postgemm_dispatcher_t::gru_part2_postgemm;
                break;
            case alg_kind::lbr_gru:
            case alg_kind::lbr_augru:
                postgemm_func = &rnn_postgemm_dispatcher_t::gru_lbr_postgemm;
                break;
            default: assert(!"Unsupported algorithm kind"); break;
        }
    }

    virtual ~rnn_postgemm_dispatcher_t() = default;

    status_t init(const rnn_utils::rnn_conf_t &rnn) {
        DNNL_X64_ONLY(CHECK(initialize_jit(rnn)));
        return status::success;
    }

    rnn_postgemm_sig(unpoison) {
        // XXX (rsdubtso): This is a big hammer that unpoisons everything
        // that a postgemm may touch to avoid writing per-cell-kind
        // versions of unpoisoning code. This must be removed alongside with
        // the big unpoison_outputs() hammer in common/primitive.cpp.

        const size_t states_nelems
                = rnn.ws_states_layer_nld * rnn.ws_states_layer_ld;
        const size_t gates_nelems
                = rnn.scratch_gates_nld * rnn.scratch_gates_ld;
        const size_t src_dt_size = types::data_type_size(rnn.cell_dt);
        const size_t scratch_dt_size = types::data_type_size(rnn.scratch_dt);
        const size_t acc_dt_size = types::data_type_size(rnn.acc_dt);

        if (pd_->is_fwd()) {
            msan_unpoison(dst_layer_, src_dt_size * states_nelems);
            msan_unpoison(dst_iter_, src_dt_size * states_nelems);
            if (rnn.is_training)
                msan_unpoison(ws_gates_, src_dt_size * gates_nelems);
        } else {
            msan_unpoison(diff_src_layer_,
                    acc_dt_size * (rnn.n_iter + 1)
                            * rnn.ws_diff_states_layer_nld
                            * rnn.ws_diff_states_layer_ld);
            msan_unpoison(diff_augru_attention_,
                    acc_dt_size * rnn.n_iter * rnn.mb * rnn.dhc);
            msan_unpoison(diff_src_iter_,
                    acc_dt_size * (rnn.n_iter + 1) * rnn.ws_diff_states_iter_nld
                            * rnn.ws_diff_states_iter_ld);
            msan_unpoison(diff_src_iter_c_,
                    acc_dt_size * (rnn.n_iter + 1)
                            * rnn.ws_diff_states_iter_c_nld
                            * rnn.ws_diff_states_iter_c_ld);
            msan_unpoison(scratch_gates_, scratch_dt_size * gates_nelems);
            msan_unpoison(scratch_cell_, scratch_dt_size * states_nelems);
        }
    }

    rnn_postgemm_sig(execute) {
        /* This block has an impact on performance in case it is executed
         * multiple times. Be careful when changing it.
         * XXX: The code is compiler sensitive, jit might help with that.
         */
#if DNNL_X64
        if (rnn_postgemm_) {
            rnn_postgemm_->execute(rnn, cell_position, ws_gates_,
                    scratch_gates_, augru_attention_, dst_layer_, dst_iter_c_,
                    src_iter_, src_iter_c_, diff_src_layer_,
                    diff_augru_attention_, diff_src_iter_, diff_src_iter_c_,
                    diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                    weights_peephole_, bias_, ws_grid_, scratch_cell_,
                    dst_iter_, weights_scales_, block_step);
            unpoison(rnn, cell_position, ws_gates_, scratch_gates_,
                    augru_attention_, dst_layer_, dst_iter_c_, src_iter_,
                    src_iter_c_, diff_src_layer_, diff_augru_attention_,
                    diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                    diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                    ws_grid_, scratch_cell_, dst_iter_, weights_scales_,
                    block_step);
            return;
        }
#endif
        (this->*postgemm_func)(rnn, cell_position, ws_gates_, scratch_gates_,
                augru_attention_, dst_layer_, dst_iter_c_, src_iter_,
                src_iter_c_, diff_src_layer_, diff_augru_attention_,
                diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                ws_grid_, scratch_cell_, dst_iter_, weights_scales_,
                block_step);
    }

    rnn_postgemm_sig(execute_part2) {
        /* This block has an impact on performance in case it is executed
         * multiple times. Be careful when changing it.
         * XXX: The code is compiler sensitive, jit might help with that.
         */
#if DNNL_X64
        if (rnn_postgemm_part2_) {
            rnn_postgemm_part2_->execute(rnn, cell_position, ws_gates_,
                    scratch_gates_, augru_attention_, dst_layer_, dst_iter_c_,
                    src_iter_, src_iter_c_, diff_src_layer_,
                    diff_augru_attention_, diff_src_iter_, diff_src_iter_c_,
                    diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                    weights_peephole_, bias_, ws_grid_, scratch_cell_,
                    dst_iter_, weights_scales_, block_step);
            unpoison(rnn, cell_position, ws_gates_, scratch_gates_,
                    augru_attention_, dst_layer_, dst_iter_c_, src_iter_,
                    src_iter_c_, diff_src_layer_, diff_augru_attention_,
                    diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                    diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                    ws_grid_, scratch_cell_, dst_iter_, weights_scales_,
                    block_step);
            return;
        }
#endif
        (this->*postgemm_part2_func)(rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_layer_, dst_iter_c_,
                src_iter_, src_iter_c_, diff_src_layer_, diff_augru_attention_,
                diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                ws_grid_, scratch_cell_, dst_iter_, weights_scales_,
                block_step);
    }

protected:
    virtual rnn_postgemm_sig(rnn_postgemm) = 0;
    virtual rnn_postgemm_sig(lstm_postgemm) = 0;
    virtual rnn_postgemm_sig(lstm_projection_postgemm) = 0;
    virtual rnn_postgemm_sig(gru_part1_postgemm) = 0;
    virtual rnn_postgemm_sig(gru_part2_postgemm) = 0;
    virtual rnn_postgemm_sig(gru_lbr_postgemm) = 0;

    const rnn_pd_t *pd_ {};

    postgemm_f postgemm_func {};
    postgemm_f postgemm_part2_func {};

    DNNL_DISALLOW_COPY_AND_ASSIGN(rnn_postgemm_dispatcher_t);

#if DNNL_X64
    std::unique_ptr<x64::jit_uni_rnn_postgemm_t> rnn_postgemm_;
    std::unique_ptr<x64::jit_uni_rnn_postgemm_t> rnn_postgemm_part2_;

    status_t initialize_jit(const rnn_utils::rnn_conf_t &rnn) {
        using namespace dnnl::impl::cpu::x64;

        if (pd_->attr()->rnn_tparams_.test_mode_) return status::success;

        const bool jit_fwd = pd_->is_fwd()
                && utils::one_of(rnn.cell_dt, data_type::f32, data_type::u8,
                        data_type::s8, data_type::bf16, data_type::f16);
        const bool jit_bwd = !pd_->is_fwd()
                && utils::one_of(rnn.cell_dt, data_type::f32, data_type::bf16,
                        data_type::f16);

        // Note: Using ref (no-jit) post-gemm for avx2_vnni_2 for now.
        // Implement jit version of post-gemm if perf becomes a concern.
        if (utils::one_of(rnn.cell_dt, data_type::bf16, data_type::f16)
                && !mayiuse(avx512_core))
            return status::success;
        if (!mayiuse(avx2)) return status::success;

//NOLINTBEGIN(bugprone-macro-parentheses)
// Can't put types into `()`:
// error: expected type-specifier before ‘)’ token
#define CREATE_WITH_DIR(k, ker_t) \
    (k).reset(create_jit_postgemm<ker_t>(rnn, pd_, rnn.cell_dt, rnn.scratch_dt))
#define CREATE(k, ker_t) \
    do { \
        if (jit_fwd) CREATE_WITH_DIR((k), CONCAT2(ker_t, _fwd_t)); \
        if (jit_bwd) CREATE_WITH_DIR((k), CONCAT2(ker_t, _bwd_t)); \
    } while (0)
        //NOLINTEND(bugprone-macro-parentheses)

        if (pd_->cell_kind() == alg_kind::vanilla_lstm) {
            CREATE(rnn_postgemm_, jit_uni_lstm_cell_postgemm);
        } else if (pd_->cell_kind() == alg_kind::vanilla_rnn) {
            CREATE(rnn_postgemm_, jit_uni_rnn_cell_postgemm);
        } else if (utils::one_of(pd_->cell_kind(), alg_kind::vanilla_gru,
                           alg_kind::vanilla_augru)) {
            CREATE(rnn_postgemm_, jit_uni_gru_cell_postgemm_part1);
            CREATE(rnn_postgemm_part2_, jit_uni_gru_cell_postgemm_part2);
        } else if (utils::one_of(pd_->cell_kind(), alg_kind::lbr_gru,
                           alg_kind::lbr_augru)) {
            CREATE(rnn_postgemm_, jit_uni_gru_lbr_cell_postgemm);
        }

#undef CREATE
#undef CREATE_WITH_DIR

        if (rnn_postgemm_) CHECK(rnn_postgemm_->init());
        if (rnn_postgemm_part2_) CHECK(rnn_postgemm_part2_->init());
        return status::success;
    }
#endif
};

struct rnn_postgemm_fwd_t : public rnn_postgemm_dispatcher_t {
    using rnn_postgemm_dispatcher_t::rnn_postgemm_dispatcher_t;

    virtual rnn_postgemm_sig(rnn_postgemm) override;
    virtual rnn_postgemm_sig(lstm_postgemm) override;
    virtual rnn_postgemm_sig(lstm_projection_postgemm) override;
    virtual rnn_postgemm_sig(gru_part1_postgemm) override;
    virtual rnn_postgemm_sig(gru_part2_postgemm) override;
    virtual rnn_postgemm_sig(gru_lbr_postgemm) override;
};

struct rnn_postgemm_bwd_t : public rnn_postgemm_dispatcher_t {
    using rnn_postgemm_dispatcher_t::rnn_postgemm_dispatcher_t;

    virtual rnn_postgemm_sig(rnn_postgemm) override;
    virtual rnn_postgemm_sig(lstm_postgemm) override;
    virtual rnn_postgemm_sig(lstm_projection_postgemm) override;
    virtual rnn_postgemm_sig(gru_part1_postgemm) override;
    virtual rnn_postgemm_sig(gru_part2_postgemm) override;
    virtual rnn_postgemm_sig(gru_lbr_postgemm) override;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
