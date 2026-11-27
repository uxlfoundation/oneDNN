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

#ifndef CPU_X64_RNN_BRGEMM_CELL_COMMON_FWD_HPP
#define CPU_X64_RNN_BRGEMM_CELL_COMMON_FWD_HPP

#include <functional>
#include "common/bfloat16.hpp"
#include "cpu/rnn/rnn_utils.hpp"
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class brgemm_dst_layer_iter_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const void *, void *, void *, dim_t)>;
    brgemm_dst_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_iter,
            const void *src_layer, void *w_iter, void *w_layer,
            void *scratch_gates, void *scratch_cell_, void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm);
    void execute() const;

private:
    int calculate_nthr() const;
    void kernel(const int ithr, const int nthr) const;
    void kernel_fused_iter_layer(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t scratch_dt_size_;
    const dim_t acc_dt_size_;
    const bool need_gemm_layer_;
    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;
    const char *const Al_;
    const char *const Ai_;
    const char *const Bl_;
    const char *const Bi_;
    char *const C_gates_;
    char *const C_cell_;
    const dim_t LDAl_;
    const dim_t LDAi_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t work_amount_;
    const dim_t Bl_n_offset_;
    const dim_t Bi_n_offset_;
    const dim_t Bl_g_offset_;
    const dim_t Bi_g_offset_;
    const dim_t Al_k_tail_offset_;
    const dim_t Ai_k_tail_offset_;
    const dim_t Bl_kb_offset_;
    const dim_t Bi_kb_offset_;
    const dim_t Bl_k_tail_offset_;
    const dim_t Bi_k_tail_offset_;
    const dim_t n_gates_;
    const brgemm_kernel_t *const brgemm_kernel_iter_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    const char *pallete_buff_iter_main_;
    const char *pallete_buff_iter_n_tail_;
    const char *pallete_buff_iter_k_tail_;
    const char *pallete_buff_iter_nk_tail_;

    const char *pallete_buff_layer_main_;
    const char *pallete_buff_layer_n_tail_;
    const char *pallete_buff_layer_k_tail_;
    const char *pallete_buff_layer_nk_tail_;

    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
    const postgemm_fused_t fused_postgemm_;
    const bool is_fused_layer_iter_brgemm_;
    const int max_nthr_;
};

class brgemm_dst_proj_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(dim_t, dim_t, void *, dim_t)>;
    brgemm_dst_proj_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *proj_ht,
            const void *w_projection, void *output, void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t acc_dt_size_;
    const dim_t proj_desc_idx_;
    const char *const A_;
    const char *const B_;
    char *const C_;
    const dim_t LDC_;
    const int max_nthr_;
    const dim_t work_amount_proj_;
    const dim_t B_n_offset_;
    const dim_t Bp_kb_offset_;
    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;

    const brgemm_kernel_t *const brgemm_kernel_main_;
    const brgemm_kernel_t *const brgemm_kernel_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_nk_tail_;
    const brgemm_kernel_t *const brgemm_kernel_k_tail_;
    const postgemm_fused_t fused_postgemm_;
};

class brgemm_gru_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const void *, void *, void *, dim_t)>;
    brgemm_gru_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_iter,
            const void *src_layer, void *w_iter, void *w_iter1, void *w_layer,
            const void *d_layer, void *scratch_gates, void *scratch_cell,
            void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm_part1,
            const postgemm_fused_t &fused_postgemm_part2);
    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t scratch_dt_size_;
    const dim_t acc_dt_size_;
    const bool need_gemm_layer_;
    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;
    const dim_t iter_part2_desc_idx_;
    const char *const Al_;
    const char *const Ai_;
    const char *const Bl_;
    const char *const Bi_;
    const char *const Bi2_;
    char *const C_gates_;
    char *const C_cell_;
    const char *const Dl_;
    const dim_t LDAl_;
    const dim_t LDAi_p1_;
    const dim_t LDAi_p2_;
    const int max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t work_amount_;
    const dim_t Bl_n_offset_;
    const dim_t Bi_n_offset_;
    const dim_t Bl_g_offset_;
    const dim_t Bi_g_offset_;
    const dim_t Bl_kb_offset_;
    const dim_t Bi_kb_offset_;
    const dim_t n_gates_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_nk_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    const char *pallete_buff_iter_main_;
    const char *pallete_buff_iter_n_tail_;
    const char *pallete_buff_iter_k_tail_;
    const char *pallete_buff_iter_nk_tail_;

    const char *pallete_buff_layer_main_;
    const char *pallete_buff_layer_n_tail_;
    const char *pallete_buff_layer_k_tail_;
    const char *pallete_buff_layer_nk_tail_;

    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
    const postgemm_fused_t fused_postgemm_part1_;
    const postgemm_fused_t fused_postgemm_part2_;
};

class brgemm_merged_layer_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    brgemm_merged_layer_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_layer,
            void *w_layer, void *scratch_gates, void *amx_scratchpad,
            x64::brgemm_batch_element_t *addr_batch_global);
    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t scratch_dt_size_;
    const dim_t acc_dt_size_;
    const dim_t layer_desc_idx_;
    const char *const Al_;
    const char *const Bl_;
    char *const C_;
    const dim_t LDAl_;
    const int max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t work_amount_;
    const dim_t Bl_n_offset_;
    const dim_t Bl_g_offset_;
    const dim_t Al_k_tail_offset_;
    const dim_t Bl_kb_offset_;
    const dim_t Bl_k_tail_offset_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    const char *pallete_buff_layer_main_;
    const char *pallete_buff_layer_n_tail_;
    const char *pallete_buff_layer_k_tail_;
    const char *pallete_buff_layer_nk_tail_;

    char *const amx_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
