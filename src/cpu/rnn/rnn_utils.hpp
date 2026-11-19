/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2026 FUJITSU LIMITED
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

#ifndef CPU_RNN_RNN_UTILS_HPP
#define CPU_RNN_RNN_UTILS_HPP

#include <memory>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/gemm_pack.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#elif DNNL_AARCH64
#include "cpu/aarch64/cpu_isa_traits.hpp"
#endif

#define rnn_postgemm_sig_args \
    const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, gates_t *ws_gates_, \
            scratch_t *scratch_gates_, const dst_layer_t *augru_attention_, \
            dst_layer_t *dst_layer_, void *dst_iter_c_, \
            const src_iter_t *src_iter_, const void *src_iter_c_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_src_iter_, gemm_acc_t *diff_src_iter_c_, \
            gemm_acc_t *diff_dst_layer_, gemm_acc_t *diff_dst_iter_, \
            gemm_acc_t *diff_dst_iter_c_, const float *weights_peephole_, \
            const void *bias_, gates_t *ws_grid_, scratch_t *scratch_cell_, \
            dst_iter_t *dst_iter_, float *weights_scales_, dim_t block_step

#define rnn_postgemm_sig(f) void f(rnn_postgemm_sig_args) const

// gemm_acc_scratchpad is a GEMM accumulation scratchpad used by tile-based
// brgemm kernels (AMX on x64, MMLA/SME on aarch64). It can be nullptr on
// implementations that don't need it (e.g. current AArch64 SVE kernels).

#if DNNL_X64 || DNNL_AARCH64
#define rnn_merged_layer_execution_sig_args \
    const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, weights_t **w_layer_, \
            const src_layer_t *src_layer_, scratch_t *scratch_gates_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_w_layer_, \
            gemm_acc_t *gemm_acc_scratchpad, \
            brgemm_batch_element_t *addr_batch_global

#define rnn_cell_execution_sig_args \
    const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, dst_layer_t *dst_layer_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_augru_attention_, gemm_acc_t *diff_src_iter_, \
            gemm_acc_t *diff_src_iter_c_, weights_t **w_layer_, \
            weights_t **w_iter_, weights_t **w_projection_, \
            const float *weights_peephole_, const float *w_proj_comp, \
            void **bias_, const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, gemm_acc_t *diff_dst_layer_, \
            gemm_acc_t *diff_dst_iter_, gemm_acc_t *diff_dst_iter_c_, \
            gemm_acc_t *diff_w_layer_, gemm_acc_t *diff_w_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gates_t *ws_gates_, scratch_t *scratch_gates_, \
            ht_t *proj_ht_, gemm_acc_t *scratch_diff_ht_, gates_t *ws_grid_, \
            scratch_t *scratch_cell_, scratch_t *scratch_gates_blocked_, \
            scratch_t *scratch_src_layer_, scratch_t *scratch_src_iter_, \
            dst_iter_t *dst_iter_, gemm_acc_t *gemm_acc_scratchpad, \
            brgemm_batch_element_t *addr_batch_global

#define rnn_grid_execution_sig_args \
    const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            weights_t **weights_layer_, weights_t **weights_iter_, \
            weights_t **weights_projection_, const float *weights_peephole_, \
            const float *w_proj_comp, void **bias_, \
            const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, dst_layer_t *dst_layer_, \
            dst_iter_t *dst_iter_, void *dst_iter_c_, \
            src_layer_t *ws_states_layer_, src_iter_t *ws_states_iter_, \
            void *ws_states_iter_c_, gemm_acc_t *ws_diff_states_layer_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, gates_t *ws_gates_, \
            ht_t *ws_ht_, gates_t *ws_grid_, scratch_t *scratch_gates_, \
            ht_t *scratch_ht_, gemm_acc_t *scratch_diff_ht_, \
            scratch_t *scratch_cell_, scratch_t *scratch_gates_blocked_, \
            scratch_t *scratch_src_layer_, scratch_t *scratch_src_iter_, \
            gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_weights_layer_, gemm_acc_t *diff_weights_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gemm_acc_t *gemm_acc_scratchpad, \
            brgemm_batch_element_t *addr_batch_global

#else

#define rnn_merged_layer_execution_sig_args \
    const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, weights_t **w_layer_, \
            const src_layer_t *src_layer_, scratch_t *scratch_gates_, \
            gemm_acc_t *diff_src_layer_, gemm_acc_t *diff_w_layer_

#define rnn_cell_execution_sig_args \
    const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, dst_layer_t *dst_layer_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_augru_attention_, gemm_acc_t *diff_src_iter_, \
            gemm_acc_t *diff_src_iter_c_, weights_t **w_layer_, \
            weights_t **w_iter_, weights_t **w_projection_, \
            const float *weights_peephole_, const float *w_proj_comp, \
            void **bias_, const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, gemm_acc_t *diff_dst_layer_, \
            gemm_acc_t *diff_dst_iter_, gemm_acc_t *diff_dst_iter_c_, \
            gemm_acc_t *diff_w_layer_, gemm_acc_t *diff_w_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gates_t *ws_gates_, scratch_t *scratch_gates_, \
            ht_t *proj_ht_, gemm_acc_t *scratch_diff_ht_, gates_t *ws_grid_, \
            scratch_t *scratch_cell_, dst_iter_t *dst_iter_, \
            gemm_acc_t *gemm_acc_scratchpad

#define rnn_grid_execution_sig_args \
    const exec_ctx_t &ctx, const rnn_utils::rnn_conf_t &rnn, \
            weights_t **weights_layer_, weights_t **weights_iter_, \
            weights_t **weights_projection_, const float *weights_peephole_, \
            const float *w_proj_comp, void **bias_, \
            const src_layer_t *src_layer_, \
            const src_layer_t *augru_attention_, const src_iter_t *src_iter_, \
            const void *src_iter_c_, dst_layer_t *dst_layer_, \
            dst_iter_t *dst_iter_, void *dst_iter_c_, \
            src_layer_t *ws_states_layer_, src_iter_t *ws_states_iter_, \
            void *ws_states_iter_c_, gemm_acc_t *ws_diff_states_layer_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, gates_t *ws_gates_, \
            ht_t *ws_ht_, gates_t *ws_grid_, scratch_t *scratch_gates_, \
            ht_t *scratch_ht_, gemm_acc_t *scratch_diff_ht_, \
            scratch_t *scratch_cell_, gemm_acc_t *diff_augru_attention_, \
            gemm_acc_t *diff_weights_layer_, gemm_acc_t *diff_weights_iter_, \
            float *diff_weights_projection_, float *diff_weights_peephole_, \
            float *diff_bias_, gemm_acc_t *gemm_acc_scratchpad

#endif

#define rnn_cell_execution_sig(f) \
    dnnl_status_t f(rnn_cell_execution_sig_args) const

#define rnn_grid_execution_sig(f) \
    dnnl_status_t f(rnn_grid_execution_sig_args) const

#define rnn_merged_layer_execution_sig(f) \
    dnnl_status_t f(rnn_merged_layer_execution_sig_args) const

#define rnn_matmul_sig(f) \
    dnnl_status_t f(const exec_ctx_t &ctx, \
            const std::shared_ptr<dnnl::impl::primitive_t> &matmul_prim, \
            const weights_t *a_, const gemm_data_t *b_, gemm_acc_t *c_) const

#define rnn_gemm_sig_args \
    const char transA, const char transB, dim_t m, dim_t n, dim_t k, \
            const float alpha, const weights_t *a_, const dim_t ldA, \
            const gemm_data_t *b_, const dim_t ldB, const float beta, \
            gemm_acc_t *c_, const dim_t ldC

#define rnn_gemm_sig(f) dnnl_status_t f(rnn_gemm_sig_args) const

#define rnn_bias_prepare_sig_args \
    const rnn_utils::rnn_conf_t &rnn, void **bias_, const void *b_, \
            void *scratch_bias_

#define rnn_bias_prepare_sig(f) void f(rnn_bias_prepare_sig_args) const

#define rnn_bias_prepare_sig_templ(f) \
    template <typename T> \
    static void f(const rnn_utils::rnn_conf_t &rnn, T **bias_, const T *b_, \
            T *scratch_bias_)

#define rnn_bias_finalize_sig_args \
    const rnn_utils::rnn_conf_t &rnn, void *scratch_bias_, \
            const float *w_iter_comp, const float *w_layer_comp

#define rnn_bias_finalize_sig(f) void f(rnn_bias_finalize_sig_args) const

#define rnn_weights_assign_sig_args \
    const rnn_utils::rnn_conf_t &rnn, const memory_desc_t *md, int n_parts, \
            const int *gates_per_part, weights_t **weights_, \
            const weights_t *w_

#define rnn_weights_assign_sig(f) void f(rnn_weights_assign_sig_args) const

namespace dnnl {
namespace impl {
namespace cpu {

#if DNNL_X64
using namespace x64;
#elif DNNL_AARCH64
using namespace aarch64;
#endif

namespace rnn_utils {

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum cell_position_t {
    middle_cell = 0x0,
    first_layer = 0x1,
    first_iter = 0x2,
    last_layer = 0x4,
    last_iter = 0x8,
    c_state_first_iter = 0x10,
    c_state_last_iter = 0x20,
    merged_iter = 0x40,
    merged_layer = 0x80
};

enum class weights_type_t {
    layer,
    iter,
    projection,
    peephole,
};

inline cell_position_t &operator|=(cell_position_t &lhs, cell_position_t rhs) {
    lhs = static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline cell_position_t operator|(cell_position_t lhs, cell_position_t rhs) {
    return static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

enum data_type_conf_t {
    all_f32,
    all_bf16,
    all_f16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8,
    s8s8s8f32,
    f32s8f32f32,
    s8s8s8s8,
    f32s8f32s8
};

enum brgemm_rnn_execute_loop_order_t {
    // default for kernels w/o loop order choice
    undefined = 0x0,
    // m_blocking loop is outermost
    mblk_nblk = 0x1,
    // n_blocking loop is outermost
    nblk_mblk = 0x2
};

struct diff_src_brgemm_conf_t {
    dim_t M = 0, N = 0, K = 0;

    dim_t n_block = 0, N_blocks = 0, n_tail = 0;
    dim_t m_block = 0, M_blocks = 0;

    dim_t K_blocks = 0, k_block = 0, k_tail = 0;
    dim_t Kpadded = 0;

    dim_t N_iter = 0, N_layer = 0;
    dim_t N_layer_blocks = 0, n_layer_tail = 0;
    dim_t N_iter_blocks = 0, n_iter_tail = 0;
    dim_t LDA = 0, LDB = 0, LDC = 0;

#if DNNL_X64 || DNNL_AARCH64
    cpu_isa_t isa = isa_undef;
#endif

    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;
    dim_t gates_block;
};

struct diff_wei_brgemm_conf_t {
    dim_t M = 0, M_layer = 0, M_iter = 0, N = 0, K = 0;

    dim_t n_block = 0, N_blocks = 0, n_tail = 0;
    dim_t m_block = 0, M_blocks = 0;
    dim_t K_blocks = 0, k_block = 0, k_tail = 0;
    dim_t Kpadded = 0;

    dim_t LDA_layer = 0, LDA_iter = 0, LDB = 0, LDC_iter = 0, LDC_layer = 0;

    bool global_transpose = false;

#if DNNL_X64 || DNNL_AARCH64
    cpu_isa_t isa = isa_undef;
#endif

    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;
};

struct rnn_conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    data_type_t cell_dt = data_type::undef; // The data type used by cell
    data_type_t weights_dt = data_type::undef;
    data_type_t acc_dt = data_type::undef;
    data_type_t scratch_dt = data_type::undef;
    data_type_t bias_dt = data_type::undef;
    data_type_t src_iter_c_dt = data_type::undef;
    data_type_t dst_iter_c_dt = data_type::undef;

    dim_t n_layer = 0, n_iter = 0, n_dir = 0, n_gates = 0, n_states = 0;
    dim_t mb = 0;
    dim_t slc = 0, sic = 0, dhc = 0, dic = 0, dlc = 0;
    //int gates_ld, gates_nld, gates_ws_ld;

    int n_parts_weights_layer = 0;
    int parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_iter = 0;
    int parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_projection = 0;
    int parts_weights_projection[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_projection_pack_size[DNNL_RNN_MAX_N_PARTS];

    dim_t n_bias = 0;
    int n_parts_bias = 0;
    int parts_bias[DNNL_RNN_MAX_N_PARTS];

    /* Size of packed data in bytes */
    size_t weights_layer_comp_offset = 0, weights_layer_pack_size = 0;
    size_t weights_iter_comp_offset = 0, weights_iter_pack_size = 0;
    size_t weights_projection_comp_offset = 0, weights_projection_pack_size = 0;

    bool copy_bias = false;
    dim_t weights_layer_ld = 0, weights_layer_nld = 0;
    dim_t diff_weights_layer_ld = 0, diff_weights_layer_nld = 0;
    dim_t weights_iter_ld = 0, weights_iter_nld = 0;
    dim_t diff_weights_iter_ld = 0, diff_weights_iter_nld = 0;
    dim_t weights_projection_ld = 0, weights_projection_nld = 0;
    dim_t diff_weights_projection_ld = 0, diff_weights_projection_nld = 0;

    dim_t proj_ht_ld = 0, proj_ht_nld = 0;

    dim_t ws_gates_ld = 0, ws_gates_nld = 0;
    dim_t ws_ht_ld = 0, ws_ht_nld = 0;
    dim_t ws_states_layer_ld = 0, ws_states_layer_nld = 0;
    dim_t ws_states_iter_ld = 0, ws_states_iter_nld = 0;
    dim_t ws_states_iter_c_ld = 0, ws_states_iter_c_nld = 0;
    dim_t ws_diff_states_layer_ld = 0, ws_diff_states_layer_nld = 0;
    dim_t ws_diff_states_iter_ld = 0, ws_diff_states_iter_nld = 0;
    dim_t ws_diff_states_iter_c_ld = 0, ws_diff_states_iter_c_nld = 0;

    dim_t scratch_gates_ld = 0, scratch_gates_nld = 0;
    dim_t scratch_ht_ld = 0, scratch_ht_nld = 0;
    dim_t scratch_diff_ht_ld = 0, scratch_diff_ht_nld = 0;

    dim_t src_layer_ld_ = 0, src_layer_nld_ = 0;
    dim_t src_iter_ld_ = 0, src_iter_nld_ = 0;
    dim_t src_iter_c_ld_ = 0, src_iter_c_nld_ = 0;
    dim_t dst_layer_ld_ = 0, dst_layer_nld_ = 0;
    dim_t dst_iter_ld_ = 0, dst_iter_nld_ = 0;
    dim_t dst_iter_c_ld_ = 0, dst_iter_c_nld_ = 0;

    dim_t weights_iter_compensation_size = 0,
          weights_layer_compensation_size = 0;
    bool is_fwd = false, is_training = false, is_lbr = false,
         is_lstm_peephole = false, is_lstm_projection = false, is_augru = false,
         is_orig_gru = false;
    bool use_workspace = false;

    // Size of workspace for each tensor in bytes
    // Notes:
    // 1. For non-LSTMP ws_states_iter_size == ws_states_layer_size. The corresponding
    //    pointers should point to the same places.
    size_t ws_gates_size = 0;
    size_t ws_ht_size = 0;
    size_t ws_states_layer_size = 0;
    size_t ws_states_iter_size = 0;
    size_t ws_states_iter_c_size = 0;
    size_t ws_diff_states_layer_size = 0;
    size_t ws_diff_states_iter_size = 0;
    size_t ws_diff_states_iter_c_size = 0;
    size_t scratch_gates_size = 0;

    size_t scratch_gates_blocked_size = 0;
    size_t scratch_gates_blocked_nested_reorder_size = 0;
    size_t scratch_src_layer_size = 0;
    size_t scratch_src_layer_nested_reorder_size = 0;
    size_t scratch_src_iter_size = 0;
    size_t scratch_src_iter_nested_reorder_size = 0;

    size_t scratch_ht_size = 0;
    size_t scratch_diff_ht_size = 0;
    size_t scratch_cell_size = 0;
    size_t ws_grid_comp_size = 0;
    size_t ws_per_cell = 0;
    size_t ws_bias_size = 0;

    bool src_layer_is_trivial_stride = false;
    bool dst_layer_is_trivial_stride = false;
    bool merge_gemm_iter = false, merge_gemm_layer = false,
         force_nocopy = false, use_layer_packed_gemm = false,
         use_iter_packed_gemm = false, use_projection_packed_gemm = false;
    dim_t n_iter_scratch_gates = 0;

    bool diff_weights_overwrite = false;
    bool use_matmul = false;

    inline bool is_int8_conf() const {
        return is_signed_int8_conf() || is_unsigned_int8_conf();
    }
    inline bool is_signed_int8_conf() const {
        return utils::one_of(
                dt_conf, s8s8s8f32, f32s8f32f32, s8s8s8s8, f32s8f32s8);
    }
    inline bool is_unsigned_int8_conf() const {
        return utils::one_of(
                dt_conf, u8u8u8f32, f32u8f32f32, u8u8u8u8, f32u8f32u8);
    }

    inline bool is_cell_dt_int8() const {
        return is_cell_dt_signed_int8() || is_cell_dt_unsigned_int8();
    }
    inline bool is_cell_dt_signed_int8() const {
        return cell_dt == data_type::s8;
    }
    inline bool is_cell_dt_unsigned_int8() const {
        return cell_dt == data_type::u8;
    }

    inline bool is_cell_int8_amx() const {
#if DNNL_X64
        return is_cell_dt_int8()
                && is_superset(brgemm_isa, x64::avx512_core_amx);
#else
        return false;
#endif
    }

    inline bool is_bf16_conf() const { return dt_conf == all_bf16; }
    inline bool is_f16_conf() const { return dt_conf == all_f16; }
    inline bool is_xf16_conf() const { return is_bf16_conf() || is_f16_conf(); }
    inline bool is_f32_conf() const { return dt_conf == all_f32; }

    inline bool is_cell_dt_f32() const { return cell_dt == data_type::f32; }
    inline bool is_cell_dt_bf16() const { return cell_dt == data_type::bf16; }
    inline bool is_cell_dt_f16() const { return cell_dt == data_type::f16; }
    inline bool is_cell_dt_xf16() const {
        return is_cell_dt_bf16() || is_cell_dt_f16();
    }
    inline bool is_cell_bf16_amx() const {
#if DNNL_X64
        return brgemm_isa == x64::avx512_core_amx && is_cell_dt_bf16();
#else
        return false;
#endif
    }
    inline bool is_cell_f16_amx() const {
#if DNNL_X64
        return brgemm_isa == x64::avx512_core_amx_fp16 && is_cell_dt_f16();
#else
        return false;
#endif
    }

    inline bool is_cell_xf16_amx() const {
        return is_cell_bf16_amx() || is_cell_f16_amx();
    }

    inline bool is_cell_amx() const {
        return is_cell_bf16_amx() || is_cell_int8_amx() || is_cell_f16_amx();
    }

    inline bool is_bf32() const { return is_cell_bf16_amx() && is_f32_conf(); }

    inline bool skip_src_layer_copy() const {
        return (exec_dir == l2r) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8f32, f32s8f32f32, s8s8s8s8,
                        f32s8f32s8, u8u8u8u8, u8u8u8f32, f32u8f32u8,
                        f32u8f32f32, all_f32, all_bf16, all_f16);
    }
    inline bool skip_src_iter_copy() const {
        return (exec_dir == l2r) && (src_iter_ld_ > 0) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, s8s8s8f32, u8u8u8u8,
                        u8u8u8f32, all_f32, all_bf16, all_f16);
    }
    inline bool skip_dst_layer_copy() const {
        return (exec_dir == l2r) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, f32s8f32s8, u8u8u8u8,
                        f32u8f32u8, all_f32, all_bf16, all_f16);
    }
    inline bool skip_dst_iter_copy() const {
        return (exec_dir == l2r) && (dst_iter_ld_ > 0) && !is_bf32()
                && utils::one_of(dt_conf, s8s8s8s8, s8s8s8f32, u8u8u8u8,
                        u8u8u8f32, all_f32, all_bf16, all_f16);
    }

    inline dim_t src_layer_ld(cell_position_t cell_position) const {
        return (cell_position & first_layer) && skip_src_layer_copy()
                ? src_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_layer_ld;
    }

    inline dim_t src_iter_ld(cell_position_t cell_position) const {
        return (cell_position & first_iter) && skip_src_iter_copy()
                ? src_iter_ld_
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                                          && !(cell_position & first_iter)
                                  ? dst_layer_ld_
                                  : ws_states_iter_ld);
    }

    inline dim_t layer_brgemm_desc(cell_position_t cell_position) const {
        return ((cell_position & first_layer) && skip_src_layer_copy()) ? 0
                : ((cell_position & last_iter) && skip_dst_iter_copy()) ? 1
                                                                        : 2;
    }

    inline dim_t iter_brgemm_desc(cell_position_t cell_position) const {
        return ((cell_position & first_iter) && skip_src_iter_copy()) ? 0
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                          && !(cell_position & first_iter))
                ? 1
                : 2;
    }

    // Returns index of brgemm kernel for 2nd part of iteration gemm in vanilla
    // GRU cell for the current position.
    // Note: this method must be aligned with dst_iter_part2_ld() and LDA2_2[]
    // values initialization order
    inline dim_t iter_part2_brgemm_desc(cell_position_t cell_position) const {
        if (cell_position & last_layer) {
            return (cell_position & last_layer) && skip_dst_layer_copy()  ? 0
                    : (cell_position & last_iter) && skip_dst_iter_copy() ? 1
                                                                          : 2;
        } else {
            return (cell_position & last_iter) && skip_dst_iter_copy() ? 1 : 3;
        }
    }

    inline dim_t src_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_first_iter) ? src_iter_c_ld_
                                                    : ws_states_iter_c_ld;
    }

    inline dim_t dst_layer_ld(
            cell_position_t cell_position, bool after_proj = false) const {
        // We use scratch_ht and not dst_layer for lstmp
        if (is_lstm_projection && !after_proj) return scratch_ht_ld;

        return (cell_position & last_layer) && skip_dst_layer_copy()
                ? dst_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_layer_ld;
    }

    inline dim_t dst_brgemm_desc(
            cell_position_t cell_position, bool after_proj = false) const {
        // We use scratch_ht and not dst_layer for lstmp
        if (is_lstm_projection && !after_proj) return 0;

        return (cell_position & last_layer) && skip_dst_layer_copy()  ? 1
                : (cell_position & last_iter) && skip_dst_iter_copy() ? 2
                                                                      : 3;
    }

    inline dim_t dst_iter_ld(cell_position_t cell_position) const {
        return (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_iter_ld;
    }

    // Returns dst tensor leading dimension for 2nd part of iteration gemm in
    // vanilla GRU cell for the current position
    inline dim_t dst_iter_part2_ld(cell_position_t cell_position) const {
        return (cell_position & last_layer) ? dst_layer_ld(cell_position)
                                            : dst_iter_ld(cell_position);
    }

    inline dim_t dst_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_last_iter) ? dst_iter_c_ld_
                                                   : ws_states_iter_c_ld;
    }

    // // when skipping copy, the output ld can be states_ws_ld,
    // // dst_iter_ld or dst_layer_ld depending on the cell position
    // inline dim_t dst_ld(cell_position_t cell_position) const {
    //     return (cell_position & last_layer) ? dst_layer_ld(cell_position)
    //                                         : dst_iter_ld(cell_position);
    // }
    inline dim_t dst_copy_ld(cell_position_t cell_position) const {
        return dst_iter_ld(cell_position);
    }

    inline bool need_gemm_layer(cell_position_t cell_position) const {
        // In case of merge_gemm_layer we might still need a layer gemm if we store
        // the states of the last iteration in the destination memory. The
        // exception of this rule is the first layer though, in which case all
        // states are kept in user's src_layer, hence making full merged gemm
        // possible.
        return IMPLICATION(merge_gemm_layer,
                skip_dst_iter_copy() && (cell_position & last_iter)
                        && !(cell_position & first_layer));
    }

    // get diff_weights_beta based on cell position
    inline float diff_weights_beta(cell_position_t cell_position) const {
        if (diff_weights_overwrite) {
            // Initialize diff weights if needed
            if (cell_position & merged_iter) return 0.0f;
            if ((cell_position & merged_layer)
                    && !need_gemm_layer(cell_position | last_iter))
                return 0.0f;
            if (cell_position & last_iter) return 0.0f;
        }
        return 1.0f;
    }

    bool is_brgemm;

    diff_src_brgemm_conf_t diff_src_brgemm;
    diff_wei_brgemm_conf_t diff_wei_brgemm;

    dim_t M, N, K1, K2;

    dim_t LDB1, LDB2;
    dim_t LDA1[3];
    dim_t LDA2[3];
    // LDA for iter part2 gemm in vanilla gru cell
    dim_t LDA2_2[4];
    dim_t LDC;

    dim_t m_block, M_blocks;
    dim_t n_block, N_blocks, n_tail;

    dim_t k2_block, k1_block, k1_tail, k2_tail;
    dim_t KB1_blocks, KB2_blocks;
    dim_t K1padded, K2padded;

    dim_t Kproj, Kprojpadded;
    dim_t kproj_block, KBproj_blocks, kproj_tail;

    dim_t Nproj, Nproj_blocks, nproj_tail;
    dim_t LDAproj, LDBproj, LDCproj[4];
    dim_t dhc_block_peephole, dhc_tail_peephole, dhc_blocks_peephole;
    bool brgemm_fwd_iter_layer_fuse_possible = false;

    int nthr;
#if DNNL_X64 || DNNL_AARCH64
    cpu_isa_t brgemm_isa = isa_undef;
#endif
    bool unfused_post_gemm;
    brgemm_rnn_execute_loop_order_t loop_order
            = brgemm_rnn_execute_loop_order_t::undefined;

    // for merged layer computation in brgemm
    dim_t Mlayermerged;
    dim_t mlayermerged_block, Mlayermerged_blocks;

    alg_kind_t cell_kind = alg_kind::undef;
};

bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);
bool is_ldio(const memory_desc_wrapper &md);
bool is_ldoi(const memory_desc_wrapper &md);
bool is_ldigo_blocked(const memory_desc_wrapper &md);
bool is_ldgoi_blocked(const memory_desc_wrapper &md);
bool is_ldio_blocked(const memory_desc_wrapper &md);
bool is_ldoi_blocked(const memory_desc_wrapper &md);

dim_t get_good_ld(dim_t dim, int sizeof_dt);

bool init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const primitive_attr_t &attr, data_type_t src_type,
        data_type_t weights_type, data_type_t acc_type,
        data_type_t scratch_type, const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d,
        const memory_desc_wrapper &bias_d);

void set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_weights_projection_d);

void set_workspace_sizes(rnn_conf_t &rnn, const rnn_desc_t &rd,
        data_type_t acc_type, data_type_t scratch_type);

void set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_ht_offset, size_t &ws_state_layer_offset,
        size_t &ws_states_iter_offset, size_t &ws_states_iter_c_offset,
        size_t &ws_diff_states_layer_offset, size_t &ws_diff_states_iter_offset,
        size_t &ws_diff_states_iter_c_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_ht_offset, size_t &scratch_diff_ht_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size);

void get_scratchpad_and_workspace_sizes(
        const rnn_conf_t &rnn, size_t &scratchpad_size, size_t &workspace_size);
status_t set_expected_desc(rnn_conf_t &rnn, memory_desc_t &weights_md,
        weights_type_t weights_type);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);

using byte = unsigned char;
template <size_t Tdims>
struct raw_array_offset_calculator_t {
    template <typename... Targs>
    raw_array_offset_calculator_t(
            const byte *base, const dim_t dt_size, Targs... Fargs)
        : base_ptr_(base), dt_size_(dt_size), dims_ {Fargs...} {}

    template <typename... Targs>
    raw_array_offset_calculator_t(std::nullptr_t, Targs... Fargs) = delete;

    template <typename... Targs>
    inline const void *operator()(Targs... Fargs) const {
        assert(static_cast<bool>(base_ptr_));
        return base_ptr_ + (offset(1, Fargs...) * dt_size_);
    }

private:
    template <typename... Targs>
    inline size_t offset(size_t const dimension, size_t element) const {
        return element;
    }
    template <typename... Targs>
    inline size_t offset(
            size_t const dimension, size_t theta, size_t element) const {
        return element + (dims_[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs) const {
        const size_t t_prime = element + (dims_[dimension] * theta);
        return offset(dimension + 1, t_prime, Fargs...);
    }

    const byte *const base_ptr_;
    const size_t dt_size_;
    const dim_t dims_[Tdims];
};

template <typename... Targs>
raw_array_offset_calculator_t<sizeof...(Targs)> make_raw_aoc(
        const void *base, const size_t dt_size, Targs... Fargs) {
    return raw_array_offset_calculator_t<sizeof...(Targs)>(
            static_cast<const byte *>(base), dt_size,
            std::forward<Targs>(Fargs)...);
}

template <typename T>
struct ws_gates_aoc_t {
    ws_gates_aoc_t(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.ws_gates_nld, rnn.ws_gates_ld), DHC_(rnn.dhc) {}
    T &operator()(dim_t batch, dim_t gate, dim_t dhc) const {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    const dim_t DHC_;
};

template <typename T>
struct scratch_gates_aoc_t {
    scratch_gates_aoc_t(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.scratch_gates_nld, rnn.scratch_gates_ld)
        , DHC_(rnn.dhc) {}
    T &operator()(dim_t batch, dim_t gate, dim_t dhc) const {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    const dim_t DHC_;
};

template <typename T>
struct weights_peephole_aoc_t {
    weights_peephole_aoc_t(const rnn_conf_t &rnn, T *data)
        : weights_peephole_(data, 3, rnn.dhc) {}
    T &operator()(dim_t g, dim_t dhc) const {
        return weights_peephole_(g, dhc);
    }

private:
    const utils::array_offset_calculator<T, 2> weights_peephole_;
};

float to_float(const void *data, const data_type_t dt);

struct bias_linear_exec_aoc_t {
    bias_linear_exec_aoc_t(const rnn_conf_t &rnn, void **bias)
        : bias_dt_(rnn.bias_dt), bias_present_(static_cast<bool>(bias)) {

        if (bias_dt_ == data_type::f32)
            new (std::addressof(bias_f32_aoc_))
                    utils::array_offset_calculator<float *, 3>(
                            reinterpret_cast<float **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else if (bias_dt_ == data_type::bf16)
            new (std::addressof(bias_bf16_aoc_))
                    utils::array_offset_calculator<bfloat16_t *, 3>(
                            reinterpret_cast<bfloat16_t **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else if (bias_dt_ == data_type::f16)
            new (std::addressof(bias_f16_aoc_))
                    utils::array_offset_calculator<float16_t *, 3>(
                            reinterpret_cast<float16_t **>(bias), rnn.n_layer,
                            rnn.n_dir, rnn.n_parts_bias);
        else
            assert(!"unsupported data type");
    }

    void **operator()(dim_t layer, dim_t dir) const {
        if (bias_present_) {
            if (bias_dt_ == data_type::f32)
                return reinterpret_cast<void **>(
                        &bias_f32_aoc_.operator()(layer, dir, 0));
            else if (bias_dt_ == data_type::bf16)
                return reinterpret_cast<void **>(
                        &bias_bf16_aoc_.operator()(layer, dir, 0));
            else if (bias_dt_ == data_type::f16)
                return reinterpret_cast<void **>(
                        &bias_f16_aoc_.operator()(layer, dir, 0));
            else
                assert(!"unsupported data type");
        }

        return nullptr;
    }

    ~bias_linear_exec_aoc_t() {
        if (bias_dt_ == data_type::f32)
            bias_f32_aoc_.~array_offset_calculator<float *, 3>();
        else if (bias_dt_ == data_type::bf16)
            bias_bf16_aoc_.~array_offset_calculator<bfloat16_t *, 3>();
        else if (bias_dt_ == data_type::f16)
            bias_f16_aoc_.~array_offset_calculator<float16_t *, 3>();
        else
            assert(!"unsupported data type");
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(bias_linear_exec_aoc_t);
    bias_linear_exec_aoc_t(bias_linear_exec_aoc_t &&) = delete;
    bias_linear_exec_aoc_t &operator=(bias_linear_exec_aoc_t &&) = delete;

private:
    data_type_t bias_dt_;
    bool bias_present_;
    union {
        utils::array_offset_calculator<float *, 3> bias_f32_aoc_;
        utils::array_offset_calculator<bfloat16_t *, 3> bias_bf16_aoc_;
        utils::array_offset_calculator<float16_t *, 3> bias_f16_aoc_;
    };
};

template <typename T>
struct ws_states_layer_aoc_t {
    ws_states_layer_aoc_t(const rnn_conf_t &rnn, T *data, dim_t leading_dim)
        : state_(data, rnn.ws_states_layer_nld, leading_dim) {}
    ws_states_layer_aoc_t(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld) {}
    T &operator()(dim_t batch, dim_t dhc) const { return state_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_states_iter_aoc_t {
    ws_states_iter_aoc_t(const rnn_conf_t &rnn, T *data, dim_t leading_dim)
        : state_(data, rnn.ws_states_iter_nld, leading_dim) {}
    ws_states_iter_aoc_t(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_iter_nld, rnn.ws_states_iter_ld) {}
    T &operator()(dim_t batch, dim_t dhc) const { return state_(batch, dhc); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct augru_attention_aoc_t {
    augru_attention_aoc_t(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.mb) {}
    T &operator()(dim_t batch) const { return state_(batch); }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 1> state_;
};

template <typename T>
struct ws_diff_states_layer_aoc_t {
    ws_diff_states_layer_aoc_t(const rnn_conf_t &rnn, T *data)
        : diff_states_layer_(data, rnn.ws_diff_states_layer_nld,
                  rnn.ws_diff_states_layer_ld) {}
    T &operator()(dim_t batch, dim_t dhc) const {
        return diff_states_layer_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_layer_;
};

template <typename T>
struct ws_diff_states_iter_aoc_t {
    ws_diff_states_iter_aoc_t(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_(data, rnn.ws_diff_states_iter_nld,
                  rnn.ws_diff_states_iter_ld) {}
    T &operator()(dim_t batch, dim_t dhc) const {
        return diff_states_iter_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_;
};

template <typename T>
struct ws_diff_states_iter_c_aoc_t {
    ws_diff_states_iter_c_aoc_t(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_c_(data, rnn.ws_diff_states_iter_c_nld,
                  rnn.ws_diff_states_iter_c_ld) {}
    T &operator()(dim_t batch, dim_t dhc) const {
        return diff_states_iter_c_(batch, dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_c_;
};

struct ws_diff_w_iter_aoc_t {
    ws_diff_w_iter_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_weights_iter_(
                  data, rnn.diff_weights_iter_nld, rnn.diff_weights_iter_ld)
        , DHC_(rnn.dhc) {}
    float &operator()(dim_t sic, dim_t gate, dim_t dhc) const {
        return diff_weights_iter_(sic, gate * DHC_ + dhc);
    }

private:
    const dnnl::impl::utils::array_offset_calculator<float, 2>
            diff_weights_iter_;
    const dim_t DHC_;
};

const void *inc_ptr(const void *data, data_type_t data_type, dim_t offset);
void *inc_ptr(void *data, data_type_t data_type, dim_t offset);

} // namespace rnn_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
