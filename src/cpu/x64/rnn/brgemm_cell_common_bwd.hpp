/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_RNN_BRGEMM_CELL_COMMON_BWD_HPP
#define CPU_X64_RNN_BRGEMM_CELL_COMMON_BWD_HPP

#include "common/bfloat16.hpp"
#include "cpu/rnn/rnn_utils.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_reorders.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_utils.hpp"
#include "cpu/x64/rnn/jit_diff_weights_peephole.hpp"
#include "cpu/x64/rnn/jit_gates_reduction.hpp"
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

/*
 * Calculations:
 * scratch * w_iter = dff_src_iter
 * A (mb x rnn.n_gates * rnn.dhc) * B (rnn.n_gates * rnn.dhc, rnn.sic) =
 * C (mb x rnn.sic)
 *
 * scratch * w_layer = dff_src_layer
 * A (mb x rnn.n_gates * rnn.dhc) * B (rnn.n_gates * rnn.dhc, rnn.slc) =
 * C (mb x rnn.slc)
 *
 * Data formats:
 * scratch = igo (mb, n_gates, rnn.dhc)
 * w_iter = gIo32i(f32)/gIO32i2o(bf16) (n_gates, rnn.sic, rnn.dhc)
 * w_layer = gIo32i(f32)/gIO32i2o(bf16) (n_gates, rnn.slc, rnn.dhc)
 * diff_src_layer = io (mb, rnn.slc)
 * diff_src_iter = io (mb, rnn.sic)
 */
class brgemm_diff_src_layer_iter_t {
public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_src_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, void *scratch_gates,
            void *w_iter, void *w_layer, void *diff_src_iter,
            void *diff_src_layer, void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    struct thread_exec_ctx_t {
        x64::brgemm_batch_element_t *addr_batch;
        char *amx_buffer;
        amx_tile_configuration_loader_t tile_configure_if_needed;
    };

    void kernel_amx(const int ithr, const int nthr) const;
    void kernel_amx_compute_iter(const dim_t m_block_id, const dim_t n_block_id,
            const int gates_start, const int gates_end,
            thread_exec_ctx_t &ctx) const;
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t wei_dt_size_;
    const dim_t scratch_dt_size_;
    const dim_t acc_dt_size_;
    const char *const A_;
    const char *const B_wei_iter_;
    const char *const B_wei_layer_;
    char *const C_diff_iter_;
    char *const C_diff_layer_;
    const dim_t k_blocks_n_gates_;
    const dim_t k_blocks_;
    const dim_t k_tail_;
    const dim_t k_block_;
    const dim_t A_k_tail_offset_;
    const dim_t B_k_tail_offset_;
    const dim_t B_nb_offset_;
    const dim_t B_kb_offset_;
    const dim_t B_gb_iter_offset_;
    const dim_t B_gb_layer_offset_;
    const dim_t LDA_;
    const dim_t LDC_;
    const int max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t work_amount_;
    const dim_t max_n_layer_blocks_;
    const dim_t max_n_iter_blocks_;
    const bool gemm_layer_needed_;
    const brgemm_kernel_t *const kernel_iter_full_blocks_b0_;
    const brgemm_kernel_t *const kernel_iter_full_blocks_b1_;
    const brgemm_kernel_t *const kernel_iter_n_tail_b0_;
    const brgemm_kernel_t *const kernel_iter_n_tail_b1_;
    const brgemm_kernel_t *const kernel_iter_k_tail_;
    const brgemm_kernel_t *const kernel_iter_nk_tail_;
    const brgemm_kernel_t *const kernel_layer_full_blocks_b0_;
    const brgemm_kernel_t *const kernel_layer_full_blocks_b1_;
    const brgemm_kernel_t *const kernel_layer_n_tail_b0_;
    const brgemm_kernel_t *const kernel_layer_n_tail_b1_;
    const brgemm_kernel_t *const kernel_layer_k_tail_;
    const brgemm_kernel_t *const kernel_layer_nk_tail_;
    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
};

/*
 * Calculations:
 * src_layer^T * scratch = dff_weights_layer
 * A before transpose (rnn.mb, rnn.slc) - layout in memory
 * A (rnn.slc, rnn.mb) * B (rnn.mb, rnn.n_gates * rnn.dhc) =
 * C (rnn.slc, rnn.n_gates * rnn.dhc)
 * src_iter^T * scratch = dff_weights_iter
 * A (rnn.sic, rnn.mb) * B (rnn.mb, rnn.n_gates * rnn.dhc) =
 * C (rnn.sic, rnn.n_gates * rnn.dhc)
 *
 * Performing gates reduction
 * diff_bias = scratch_blocked reduction over mb
 *
 * Data formats:
 * src_iter  = io (mb,  rnn.sic) -> transposed oi (rnn.sic, mb)
 * src_layer = io (mb, rnn.slc) -> transposed oi (rnn.sic, mb)
 *
 * scratch = igo (mb, n_gates, rnn.dhc)
 * Note:
 * For calculation purposes scratch is transformed locally to blocked
 * (in case of bf16 vnni friendly) format Oi32o(f32)/OI32o2i(bf16)
 *
 * dff_weights_iter = igo (rnn.sic, rnn.n_gates, rnn.dhc)
 * dff_weights_layer = igo (rnn.slc, rnn.n_gates, rnn.dhc)
 * diff_bias = go(n_gates, rnn.dhc)
 */
class brgemm_diff_weights_layer_iter_t {

public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_weights_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_iter,
            void *const A_iter_transposed_scratch, const void *src_layer,
            void *const A_layer_transposed_scratch, const void *scratch,
            void *scratch_gates_blocked, void *diff_weights_iter,
            void *diff_weights_layer, void *diff_bias, void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t scratch_dt_size_;
    const dim_t acc_dt_size_;
    const bool is_amx_;
    const char *const A_iter_;
    char *const A_iter_transposed_scratch_;
    const char *const A_layer_;
    char *const A_layer_transposed_scratch_;
    const char *const B_;
    char *const B_blocked_scratch_;
    char *const C_iter_;
    char *const C_layer_;
    char *const diff_bias_;
    const dim_t LDA_iter_;
    const dim_t LDA_layer_;
    const dim_t LDC_iter_;
    const dim_t LDC_layer_;
    const int max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t k_blocks_;
    const dim_t k_tail_;
    const dim_t k_block_;
    const dim_t m_iter_block_;
    const dim_t m_layer_block_;
    const dim_t A_k_iter_tail_offset_;
    const dim_t A_k_layer_tail_offset_;
    const dim_t B_kb_offset_;
    const dim_t B_k_tail_offset_blocked_;
    const dim_t work_amount_;
    const brgemm_kernel_t *const kernel_iter_full_blocks_;
    const brgemm_kernel_t *const kernel_iter_n_tail_;
    const brgemm_kernel_t *const kernel_iter_k_tail_;
    const brgemm_kernel_t *const kernel_iter_nk_tail_;
    const brgemm_kernel_t *const kernel_layer_full_blocks_;
    const brgemm_kernel_t *const kernel_layer_n_tail_;
    const brgemm_kernel_t *const kernel_layer_k_tail_;
    const brgemm_kernel_t *const kernel_layer_nk_tail_;
    const jit_gates_reduction_t *const kernel_gates_reduction_;
    const jit_gates_reduction_t *const kernel_gates_reduction_tail_;
    const jit_brgemm_transpose_single_row_t *const kernel_transpose_iter_;
    const jit_brgemm_transpose_single_row_t *const kernel_transpose_layer_;
    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;

    void kernel_amx(const int ithr, const int nthr) const;
    void kernel(const int ithr, const int nthr) const;
    void reorder_scratch_gates(
            const void *src, void *dst, const bool do_n_tail) const;
};

class brgemm_diff_wei_peep_t {
public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_wei_peep_t(const ref_rnn_brgemm_t &rnn_brgemm,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *scratch_gates,
            const void *src_iter_c, const void *dst_iter_c,
            float *diff_weights_peephole);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const int n_gates_ = 3;
    const rnn_utils::rnn_conf_t &rnn_;
    const void *scratch_gates_;
    const void *src_iter_c_;
    const void *dst_iter_c_;
    float *diff_weights_peephole_;
    const dim_t work_amount_;
    const dim_t dst_iter_c_ld_;
    const dim_t src_iter_c_ld_;
    const jit_diff_weights_peephole_t *const kernel_;
    const jit_diff_weights_peephole_t *const kernel_tail_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
