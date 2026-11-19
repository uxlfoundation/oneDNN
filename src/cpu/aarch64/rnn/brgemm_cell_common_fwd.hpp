/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP
#define CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP

#include <functional>

#include "common/bfloat16.hpp"
#include "cpu/aarch64/rnn/rnn_brgemm_utils.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

class brgemm_dst_layer_iter_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const void *, void *, void *, int)>;

    brgemm_dst_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_iter,
            const void *src_layer, void *w_iter, void *w_layer,
            void *scratch_gates, void *scratch_cell_, void *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
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

    const bool need_gemm_layer_;

    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;

    const char *const A_layer_;
    const char *const A_iter_;

    const char *const B_layer_;
    const char *const B_iter_;

    char *const C_gates_;
    char *const C_cell_;

    const dim_t LDA_layer_;
    const dim_t LDA_iter_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_iter_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t B_iter_g_offset_;
    const dim_t A_layer_k_tail_offset_;
    const dim_t A_iter_k_tail_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_iter_kb_offset_;
    const dim_t B_layer_k_tail_offset_;
    const dim_t B_iter_k_tail_offset_;

    const dim_t n_gates_;

    const brgemm_kernel_t *const brgemm_kernel_iter_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    brgemm_batch_element_t *const addr_batch_global_;
    const postgemm_fused_t fused_postgemm_;

    const bool is_fused_layer_iter_brgemm_;

    const int max_nthr_;
};

class brgemm_dst_proj_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;

    using postgemm_fused_t = std::function<void(dim_t, dim_t, void *, int)>;

    brgemm_dst_proj_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *proj_ht,
            const void *w_projection, void *output, void *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t acc_dt_size_;

    const int proj_desc_idx_;

    const char *const A_;
    const char *const B_;
    char *const C_;

    const dim_t LDC_;
    const dim_t max_nthr_;
    const int work_amount_proj_;

    const dim_t B_n_offset_;

    brgemm_batch_element_t *const addr_batch_global_;

    const brgemm_kernel_t *const brgemm_kernel_main_;
    const brgemm_kernel_t *const brgemm_kernel_n_tail_;

    const postgemm_fused_t fused_postgemm_;
};

class brgemm_gru_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const void *, void *, void *, int)>;

    brgemm_gru_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const void *src_iter,
            const void *src_layer, void *w_iter, void *w_iter1, void *w_layer,
            void *d_layer, void *scratch_gates, void *scratch_cell,
            void *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
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

    const bool need_gemm_layer_;

    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;
    const dim_t iter_part2_desc_idx_;

    const char *const A_layer_;
    const char *const A_iter_;

    const char *const B_layer_;
    const char *const B_iter_;
    const char *const B_iter_part2_;

    char *const C_gates_;
    char *const C_cell_;
    char *const D_layer_;

    const dim_t LDA_layer_;
    const dim_t LDA_iter_part1_;
    const dim_t LDA_iter_part2_;

    const dim_t max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_iter_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t B_iter_g_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_iter_kb_offset_;

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
            void *w_layer, void *scratch_gates, void *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;
    const dim_t src_dt_size_;
    const dim_t wei_dt_size_;
    const dim_t scratch_dt_size_;

    const dim_t layer_desc_idx_;

    const char *const A_layer_;
    const char *const B_layer_;
    char *const C_;

    const dim_t LDA_layer_;
    const dim_t max_nthr_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t A_layer_k_tail_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_layer_k_tail_offset_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    brgemm_batch_element_t *const addr_batch_global_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP
