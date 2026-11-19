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

#include "brgemm_cell_common_fwd.hpp"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_utils.hpp"

#include <cstring>

using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

brgemm_dst_layer_iter_t::brgemm_dst_layer_iter_t(
        const ref_rnn_brgemm_t &rnn_brgemm, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const void *src_iter,
        const void *src_layer, void *w_iter, void *w_layer, void *scratch_gates,
        void *scratch_cell, void *amx_scratchpad,
        x64::brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , src_dt_size_(types::data_type_size(rnn.cell_dt))
    , wei_dt_size_(types::data_type_size(rnn.weights_dt))
    , scratch_dt_size_(types::data_type_size(rnn.scratch_dt))
    , acc_dt_size_(types::data_type_size(rnn.acc_dt))
    , need_gemm_layer_(rnn_.need_gemm_layer(cell_position))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , iter_desc_idx_(rnn_.iter_brgemm_desc(cell_position))
    , Al_(static_cast<const char *>(src_layer))
    , Ai_(static_cast<const char *>(src_iter))
    , Bl_(static_cast<const char *>(w_layer))
    , Bi_(static_cast<const char *>(w_iter))
    , C_gates_(static_cast<char *>(scratch_gates))
    , C_cell_(static_cast<char *>(scratch_cell))
    , LDAl_(rnn_.src_layer_ld(cell_position))
    , LDAi_(rnn_.src_iter_ld(cell_position))
    , n_blocking_((rnn_.unfused_post_gemm) ? rnn_.N_blocks * rnn_.n_gates
                                           : rnn_.N_blocks)
    , m_blocking_(rnn_.M_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , Bl_n_offset_(rnn_.K1padded * rnn_.n_block)
    , Bi_n_offset_(rnn_.K2padded * rnn_.n_block)
    , Bl_g_offset_(rnn_.N_blocks * Bl_n_offset_)
    , Bi_g_offset_(rnn_.N_blocks * Bi_n_offset_)
    , Al_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , Ai_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block)
    , Bl_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , Bi_kb_offset_(rnn_.k2_block * rnn_.n_block)
    , Bl_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , Bi_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block * rnn_.n_block)
    , n_gates_(rnn.unfused_post_gemm ? 1 : rnn.n_gates)
    , brgemm_kernel_iter_main_(
              rnn_brgemm_.kernel_iter_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_n_tail_(
              rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_k_tail_(
              rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_nk_tail_(
              rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layer_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx_].get())
    , pallete_buff_iter_main_(rnn.k1_block == rnn.k2_block && need_gemm_layer_
                      ? rnn_brgemm_.pallete_buff_layer_
                      : rnn_brgemm_.pallete_buff_iter_)
    , pallete_buff_iter_n_tail_(rnn.k1_block == rnn.k2_block && need_gemm_layer_
                      ? rnn_brgemm_.pallete_buff_layer_n_tail_
                      : rnn_brgemm_.pallete_buff_iter_n_tail_)
    , pallete_buff_iter_k_tail_(rnn.k1_tail == rnn.k2_tail && need_gemm_layer_
                      ? rnn_brgemm_.pallete_buff_k1_tail_
                      : rnn_brgemm_.pallete_buff_k2_tail_)
    , pallete_buff_iter_nk_tail_(rnn.k1_tail == rnn.k2_tail && need_gemm_layer_
                      ? rnn_brgemm_.pallete_buff_nk1_tail_
                      : rnn_brgemm_.pallete_buff_nk2_tail_)
    , pallete_buff_layer_main_(rnn_brgemm_.pallete_buff_layer_)
    , pallete_buff_layer_n_tail_(rnn_brgemm_.pallete_buff_layer_n_tail_)
    , pallete_buff_layer_k_tail_(rnn_brgemm_.pallete_buff_k1_tail_)
    , pallete_buff_layer_nk_tail_(rnn_brgemm_.pallete_buff_nk1_tail_)
    , amx_scratchpad_(static_cast<char *>(amx_scratchpad))
    , addr_batch_global_(addr_batch_global)
    , fused_postgemm_(fused_postgemm)
    , is_fused_layer_iter_brgemm_(!rnn_.is_lbr && rnn_.sic == rnn_.slc
              && LDAi_ == LDAl_ && need_gemm_layer_)
    , max_nthr_(calculate_nthr()) {}

void brgemm_dst_layer_iter_t::execute() const {
    if (is_fused_layer_iter_brgemm_) {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel_fused_iter_layer(ithr, nthr);
        });
    } else {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel(ithr, nthr);
        });
    }
}

// Returns the number of threads to use. Returns 1 for small problems to avoid multithreading overhead.
// Implementation is based on empirical data.
int brgemm_dst_layer_iter_t::calculate_nthr() const {
    const auto max_nthr = nstl::min(dnnl_get_current_num_threads(), rnn_.nthr);
    // TODO: add support for other cases
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    constexpr dim_t k2_i8_avx2 = 192;
    constexpr dim_t k2_f32_avx2 = 112;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    constexpr dim_t k2_i8_avx2 = 176;
    constexpr dim_t k2_f32_avx2 = 96;
#endif
#if (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB) \
        || (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP)
    if (rnn_.brgemm_isa == x64::avx2 && !need_gemm_layer_ && rnn_.M == 1
            && utils::one_of(rnn_.cell_kind, alg_kind::vanilla_lstm,
                    alg_kind::lbr_gru)) {
        if (rnn_.cell_dt == data_type::u8 && rnn_.weights_dt == data_type::s8) {
            return rnn_.K2 <= k2_i8_avx2 ? 1 : max_nthr;
        } else if (rnn_.cell_dt == data_type::f32
                && rnn_.weights_dt == data_type::f32) {
            return rnn_.K2 <= k2_f32_avx2 ? 1 : max_nthr;
        }
    }
#endif
    return max_nthr;
}

void brgemm_dst_layer_iter_t::kernel(const int ithr, const int nthr) const {
    using namespace cpu::rnn_utils;

    dim_t start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = is_superset(rnn_.brgemm_isa, x64::avx512_core_amx);
    char *const amx_buffer = is_amx ? amx_scratchpad_
                    + rnn_.m_block * rnn_.n_block * ithr * acc_dt_size_
                                    : nullptr;
    const dim_t max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff_iter = nullptr;
    const char *pallete_buff_layer = nullptr;
    const char *pallete_buff_iter_k_tail = nullptr;
    const char *pallete_buff_layer_k_tail = nullptr;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    amx_tile_configuration_loader_t load_cfg_if_needed;

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const Al_m = Al_ + m * LDAl_ * src_dt_size_;
        const auto *const Ai_m = Ai_ + m * LDAi_ * src_dt_size_;
        const auto *const Bl_n = Bl_ + nb * Bl_n_offset_ * wei_dt_size_;
        const auto *const Bi_n = Bi_ + nb * Bi_n_offset_ * wei_dt_size_;
        auto *const C_n = C_gates_ + (m * rnn_.LDC + n) * scratch_dt_size_;
        const auto cell_stride = rnn_.LDC;
        auto *const C_cell_i = C_cell_
                ? C_cell_ + (m * cell_stride + n) * scratch_dt_size_
                : C_cell_;
        assert(rnn_.LDC == rnn_.scratch_gates_ld);

        const brgemm_kernel_t *brgemm_kernel_layer_b0
                = brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_iter = brgemm_kernel_iter_main_;
        const brgemm_kernel_t *brgemm_kernel_layer_k_tail
                = brgemm_kernel_layer_k_tail_;
        const brgemm_kernel_t *brgemm_kernel_iter_k_tail
                = brgemm_kernel_iter_k_tail_;

        if (is_amx) {
            pallete_buff_iter = pallete_buff_iter_main_;
            pallete_buff_layer = pallete_buff_layer_main_;
            pallete_buff_iter_k_tail = pallete_buff_iter_k_tail_;
            pallete_buff_layer_k_tail = pallete_buff_layer_k_tail_;
        }

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
        if (do_n_tail) {
            brgemm_kernel_layer_b0 = brgemm_kernel_layer_n_tail_;
            brgemm_kernel_iter = brgemm_kernel_iter_n_tail_;
            brgemm_kernel_layer_k_tail = brgemm_kernel_layer_nk_tail_;
            brgemm_kernel_iter_k_tail = brgemm_kernel_iter_nk_tail_;

            if (is_amx) {
                pallete_buff_iter = pallete_buff_iter_n_tail_;
                pallete_buff_layer = pallete_buff_layer_n_tail_;
                pallete_buff_iter_k_tail = pallete_buff_iter_nk_tail_;
                pallete_buff_layer_k_tail = pallete_buff_layer_nk_tail_;
            }
        }

        if (rnn_.is_lbr) {
            for (dim_t i = 0; i < rnn_.m_block; ++i) {
                std::memset(C_cell_i + i * cell_stride * scratch_dt_size_, 0,
                        nstl::min(rnn_.n_block, cell_stride)
                                * scratch_dt_size_);
            }
        }

        for (int g = 0; g < n_gates_; g++) {
            const dim_t lg = g + g_unfused;
            const auto *const Bl_g = Bl_n + lg * Bl_g_offset_ * wei_dt_size_;
            const auto *const Bi_g = Bi_n + lg * Bi_g_offset_ * wei_dt_size_;
            auto *C_g = C_n + lg * rnn_.N * scratch_dt_size_;

            if (need_gemm_layer_) {
                if (is_amx) load_cfg_if_needed(pallete_buff_layer);
                for (int i = 0; i < rnn_.KB1_blocks; i++) {
                    addr_batch[i].ptr.A
                            = Al_m + i * rnn_.k1_block * src_dt_size_;
                    addr_batch[i].ptr.B
                            = Bl_g + i * Bl_kb_offset_ * wei_dt_size_;
                }
                brgemm_kernel_execute(brgemm_kernel_layer_b0, rnn_.KB1_blocks,
                        addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
            }

            if (rnn_.is_lbr && g == n_gates_ - 1) C_g = C_cell_i;
            for (int i = 0; i < rnn_.KB2_blocks; i++) {
                addr_batch[i].ptr.A = Ai_m + i * rnn_.k2_block * src_dt_size_;
                addr_batch[i].ptr.B = Bi_g + i * Bi_kb_offset_ * wei_dt_size_;
            }
            if (is_amx) load_cfg_if_needed(pallete_buff_iter);
            brgemm_kernel_execute(brgemm_kernel_iter, rnn_.KB2_blocks,
                    addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
        }

        if (rnn_.k1_tail && need_gemm_layer_) {
            if (is_amx) load_cfg_if_needed(pallete_buff_layer_k_tail);

            for (int g = 0; g < n_gates_; g++) {
                const dim_t lg = g + g_unfused;
                const auto *const Bl_g
                        = Bl_n + lg * Bl_g_offset_ * wei_dt_size_;
                auto *const C_g = C_n + lg * rnn_.N * scratch_dt_size_;

                addr_batch[0].ptr.A = Al_m + Al_k_tail_offset_ * src_dt_size_;
                addr_batch[0].ptr.B = Bl_g + Bl_k_tail_offset_ * wei_dt_size_;
                brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (rnn_.k2_tail) {
            if (is_amx) load_cfg_if_needed(pallete_buff_iter_k_tail);

            for (int g = 0; g < n_gates_; g++) {
                const dim_t lg = g + g_unfused;
                const auto *const Bi_g
                        = Bi_n + lg * Bi_g_offset_ * wei_dt_size_;
                auto *C_g = C_n + lg * rnn_.N * scratch_dt_size_;
                if (rnn_.is_lbr && g == n_gates_ - 1) C_g = C_cell_i;
                addr_batch[0].ptr.A = Ai_m + Ai_k_tail_offset_ * src_dt_size_;
                addr_batch[0].ptr.B = Bi_g + Bi_k_tail_offset_ * wei_dt_size_;
                brgemm_kernel_execute(brgemm_kernel_iter_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            auto block_step = (do_n_tail ? rnn_.n_tail : rnn_.n_block);
            if (!rnn_.is_lbr) block_step *= scratch_dt_size_;
            fused_postgemm_(m, n, nb_i, Ai_m + n * src_dt_size_, C_n, C_cell_i,
                    block_step);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

void brgemm_dst_layer_iter_t::kernel_fused_iter_layer(
        const int ithr, const int nthr) const {
    using namespace cpu::rnn_utils;

    dim_t start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = is_superset(rnn_.brgemm_isa, x64::avx512_core_amx);
    char *const amx_buffer = is_amx ? amx_scratchpad_
                    + rnn_.m_block * rnn_.n_block * ithr * acc_dt_size_
                                    : nullptr;
    const dim_t max_K_Block = 2
            * nstl::max(rnn_.KB1_blocks + 1,
                    nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff = nullptr;
    const char *pallete_buff_k_tail = nullptr;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    amx_tile_configuration_loader_t load_cfg_if_needed;
    const auto LDA = LDAl_;
    const auto B_n_offset = Bl_n_offset_;
    const auto B_g_offset = Bl_g_offset_;
    const auto B_kb_offset = Bl_kb_offset_;
    const auto KB_blocks
            = (need_gemm_layer_ ? rnn_.KB1_blocks : 0) + rnn_.KB2_blocks;
    const auto KB_blocks_tail = (need_gemm_layer_ ? 1 : 0) + 1;
    const auto A_k_tail_offset = Al_k_tail_offset_;
    const auto B_k_tail_offset = Bl_k_tail_offset_;

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const Al_m = Al_ + m * LDA * src_dt_size_;
        const auto *const Ai_m = Ai_ + m * LDA * src_dt_size_;
        const auto *const Bl_n = Bl_ + nb * B_n_offset * wei_dt_size_;
        const auto *const Bi_n = Bi_ + nb * B_n_offset * wei_dt_size_;
        auto *const C_n = C_gates_ + (m * rnn_.LDC + n) * scratch_dt_size_;

        const brgemm_kernel_t *brgemm_kernel = brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_k_tail
                = brgemm_kernel_layer_k_tail_;

        if (is_amx) {
            pallete_buff = pallete_buff_layer_main_;
            pallete_buff_k_tail = pallete_buff_layer_k_tail_;
        }

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
        if (do_n_tail) {
            brgemm_kernel = brgemm_kernel_layer_n_tail_;
            brgemm_kernel_k_tail = brgemm_kernel_layer_nk_tail_;

            if (is_amx) {
                pallete_buff = pallete_buff_layer_n_tail_;
                pallete_buff_k_tail = pallete_buff_layer_nk_tail_;
            }
        }

        for (int g = 0; g < n_gates_; g++) {
            const dim_t lg = g + g_unfused;
            const auto *const Bl_g = Bl_n + lg * B_g_offset * wei_dt_size_;
            const auto *const Bi_g = Bi_n + lg * B_g_offset * wei_dt_size_;
            auto *const C_g = C_n + lg * rnn_.N * scratch_dt_size_;
            int batch_idx = 0;

            if (need_gemm_layer_) {
                for (; batch_idx < rnn_.KB1_blocks; batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = Al_m + batch_idx * rnn_.k1_block * src_dt_size_;
                    addr_batch[batch_idx].ptr.B
                            = Bl_g + batch_idx * B_kb_offset * wei_dt_size_;
                }
            }

            int iter_idx = 0;
            for (; batch_idx < KB_blocks; batch_idx++) {
                addr_batch[batch_idx].ptr.A
                        = Ai_m + iter_idx * rnn_.k2_block * src_dt_size_;
                addr_batch[batch_idx].ptr.B
                        = Bi_g + iter_idx * B_kb_offset * wei_dt_size_;
                iter_idx++;
            }

            if (is_amx) load_cfg_if_needed(pallete_buff);
            brgemm_kernel_execute(brgemm_kernel, KB_blocks, addr_batch,
                    reinterpret_cast<void *>(C_g), amx_buffer);
        }

        if (rnn_.k2_tail) {
            for (int g = 0; g < n_gates_; g++) {
                const dim_t lg = g + g_unfused;
                auto *const C_g = C_n + lg * rnn_.N * scratch_dt_size_;

                int batch_idx = 0;
                if (need_gemm_layer_) {
                    const auto *const Bl_g
                            = Bl_n + lg * B_g_offset * wei_dt_size_;
                    addr_batch[batch_idx].ptr.A
                            = Al_m + A_k_tail_offset * src_dt_size_;
                    addr_batch[batch_idx].ptr.B
                            = Bl_g + B_k_tail_offset * wei_dt_size_;
                    batch_idx++;
                }
                const auto *const Bi_g = Bi_n + lg * B_g_offset * wei_dt_size_;
                addr_batch[batch_idx].ptr.A
                        = Ai_m + A_k_tail_offset * src_dt_size_;
                addr_batch[batch_idx].ptr.B
                        = Bi_g + B_k_tail_offset * wei_dt_size_;

                if (is_amx) load_cfg_if_needed(pallete_buff_k_tail);
                brgemm_kernel_execute(brgemm_kernel_k_tail, KB_blocks_tail,
                        addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            const auto block_step = (do_n_tail ? rnn_.n_tail : rnn_.n_block)
                    * scratch_dt_size_;
            fused_postgemm_(m, n, nb_i, Ai_m, C_n, C_cell_, block_step);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

brgemm_dst_proj_t::brgemm_dst_proj_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const void *proj_ht,
        const void *w_projection, void *output, void *amx_scratchpad,
        x64::brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , src_dt_size_(types::data_type_size(rnn.cell_dt))
    , wei_dt_size_(types::data_type_size(rnn.weights_dt))
    , acc_dt_size_(types::data_type_size(rnn.acc_dt))
    , proj_desc_idx_(rnn_.is_cell_dt_f32()
                      ? rnn_.dst_brgemm_desc(cell_position, true)
                      : 0)
    , A_(static_cast<const char *>(proj_ht))
    , B_(static_cast<const char *>(w_projection))
    , C_(static_cast<char *>(output))
    , LDC_(rnn_.is_cell_dt_f32() ? rnn_.dst_layer_ld(cell_position, true)
                                 : rnn_.scratch_gates_ld)
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , work_amount_proj_(rnn_.Nproj_blocks * rnn_.M_blocks)
    , B_n_offset_(rnn_.Kprojpadded * rnn_.n_block)
    , Bp_kb_offset_(rnn_.kproj_block * rnn_.n_block)
    , amx_scratchpad_(static_cast<char *>(amx_scratchpad))
    , addr_batch_global_(addr_batch_global)
    , brgemm_kernel_main_(rnn_brgemm_.kernel_proj_b0_[proj_desc_idx_].get())
    , brgemm_kernel_n_tail_(
              rnn_brgemm_.kernel_proj_N_tail_b0_[proj_desc_idx_].get())
    , brgemm_kernel_nk_tail_(
              rnn_brgemm_.kernel_proj_NK_tail_b1_[proj_desc_idx_].get())
    , brgemm_kernel_k_tail_(
              rnn_brgemm_.kernel_proj_K_tail_b1_[proj_desc_idx_].get())
    , fused_postgemm_(fused_postgemm) {}

void brgemm_dst_proj_t::execute() const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

void brgemm_dst_proj_t::kernel(const int ithr, const int nthr) const {
    using namespace cpu::rnn_utils;

    dim_t start = 0, end = 0;
    balance211(work_amount_proj_, nthr, ithr, start, end);
    const bool is_amx = is_superset(rnn_.brgemm_isa, x64::avx512_core_amx);
    const dim_t max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    auto *const amx_buffer = is_amx ? amx_scratchpad_
                    + rnn_.m_block * rnn_.n_block * ithr * acc_dt_size_
                                    : nullptr;
    auto *const addr_batch = is_amx ? addr_batch_global_ + ithr * max_K_Block
                                    : addr_batch_global_ + ithr;
    amx_tile_configuration_loader_t load_cfg_if_needed;

    if (is_amx) load_cfg_if_needed(rnn_brgemm_.pallete_buff_proj_);

    dim_t nb = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, rnn_.M_blocks, nb, rnn_.Nproj_blocks);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
            break;
        default: assert(!"unsupported loop order");
    }

    while (start < end) {
        const dim_t n = nb * rnn_.n_block;
        const dim_t m = mb * rnn_.m_block;
        const bool do_n_tail = (n + rnn_.n_block) > rnn_.Nproj;
        const dim_t block_step
                = ((do_n_tail) ? rnn_.nproj_tail : rnn_.n_block) * src_dt_size_;

        const auto *const Ap_m = A_ + m * rnn_.LDAproj * src_dt_size_;
        const auto *const Bp_n = B_ + nb * B_n_offset_ * wei_dt_size_;
        auto *const Cp_n = C_ + (m * LDC_ + n) * acc_dt_size_;

        const brgemm_kernel_t *const brgemm_kernel_proj_b0
                = do_n_tail ? brgemm_kernel_n_tail_ : brgemm_kernel_main_;

        if (is_amx) {
            if (do_n_tail)
                load_cfg_if_needed(rnn_brgemm_.pallete_buff_nproj_tail_);
            for (int k = 0; k < rnn_.KBproj_blocks; k++) {
                addr_batch[k].ptr.A
                        = Ap_m + k * rnn_.kproj_block * src_dt_size_;
                addr_batch[k].ptr.B = Bp_n + k * Bp_kb_offset_ * wei_dt_size_;
            }
            brgemm_kernel_execute(brgemm_kernel_proj_b0, rnn_.KBproj_blocks,
                    addr_batch, reinterpret_cast<void *>(Cp_n), amx_buffer);

            if (rnn_.kproj_tail) {
                const brgemm_kernel_t *brgemm_kernel_proj_tail;
                const char *tail_cfg_kproj, *tail_recfg;
                if (do_n_tail) {
                    tail_cfg_kproj = rnn_brgemm_.pallete_buff_nkproj_tail_;
                    tail_recfg = rnn_brgemm_.pallete_buff_nproj_tail_;
                    brgemm_kernel_proj_tail = brgemm_kernel_nk_tail_;
                } else {
                    tail_cfg_kproj = rnn_brgemm_.pallete_buff_kproj_tail_;
                    tail_recfg = rnn_brgemm_.pallete_buff_proj_;
                    brgemm_kernel_proj_tail = brgemm_kernel_k_tail_;
                }
                load_cfg_if_needed(tail_cfg_kproj);
                addr_batch[0].ptr.A = Ap_m
                        + rnn_.KBproj_blocks * rnn_.kproj_block * src_dt_size_;
                addr_batch[0].ptr.B = Bp_n
                        + rnn_.KBproj_blocks * rnn_.kproj_block * rnn_.n_block
                                * wei_dt_size_;
                brgemm_kernel_execute(brgemm_kernel_proj_tail, 1, addr_batch,
                        reinterpret_cast<void *>(Cp_n), amx_buffer);
                load_cfg_if_needed(tail_recfg);
            }
        } else {
            addr_batch[0].ptr.A = Ap_m;
            addr_batch[0].ptr.B = Bp_n;
            brgemm_kernel_execute(brgemm_kernel_proj_b0, 1, addr_batch,
                    reinterpret_cast<void *>(Cp_n), amx_buffer);
        }

        if (!rnn_.unfused_post_gemm) {
            fused_postgemm_(m, n, Cp_n, block_step);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, rnn_.M_blocks, nb, rnn_.Nproj_blocks);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

brgemm_gru_t::brgemm_gru_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const void *src_iter,
        const void *src_layer, void *w_iter0, void *w_iter1, void *w_layer,
        const void *d_layer, void *scratch_gates, void *scratch_cell,
        void *amx_scratchpad, x64::brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm_part1,
        const postgemm_fused_t &fused_postgemm_part2)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , src_dt_size_(types::data_type_size(rnn.cell_dt))
    , wei_dt_size_(types::data_type_size(rnn.weights_dt))
    , scratch_dt_size_(types::data_type_size(rnn.scratch_dt))
    , acc_dt_size_(types::data_type_size(rnn.acc_dt))
    , need_gemm_layer_(rnn_.need_gemm_layer(cell_position))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , iter_desc_idx_(rnn_.iter_brgemm_desc(cell_position))
    , iter_part2_desc_idx_(rnn_.iter_part2_brgemm_desc(cell_position))
    , Al_(static_cast<const char *>(src_layer))
    , Ai_(static_cast<const char *>(src_iter))
    , Bl_(static_cast<const char *>(w_layer))
    , Bi_(static_cast<const char *>(w_iter0))
    , Bi2_(static_cast<const char *>(w_iter1))
    , C_gates_(static_cast<char *>(scratch_gates))
    , C_cell_(static_cast<char *>(scratch_cell))
    , Dl_(static_cast<const char *>(d_layer))
    , LDAl_(rnn_.src_layer_ld(cell_position))
    , LDAi_p1_(rnn_.src_iter_ld(cell_position))
    , LDAi_p2_(rnn_.dst_iter_part2_ld(cell_position))
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , n_blocking_((rnn_.unfused_post_gemm) ? rnn_.N_blocks * rnn_.n_gates
                                           : rnn_.N_blocks)
    , m_blocking_(rnn_.M_blocks)
    , work_amount_(m_blocking_)
    , Bl_n_offset_(rnn_.K1padded * rnn_.n_block)
    , Bi_n_offset_(rnn_.K2padded * rnn_.n_block)
    , Bl_g_offset_(rnn_.N_blocks * Bl_n_offset_)
    , Bi_g_offset_(rnn_.N_blocks * Bi_n_offset_)
    , Bl_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , Bi_kb_offset_(rnn_.k2_block * rnn_.n_block)
    , n_gates_(rnn.unfused_post_gemm ? 1 : rnn.n_gates)
    , brgemm_kernel_iter_p0_main_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_b0_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p0_n_tail_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_N_tail_b0_[iter_desc_idx_]
                                .get())
    , brgemm_kernel_iter_p0_k_tail_(
              rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p0_nk_tail_(
              rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p1_main_(
              rnn_brgemm_.kernel_iter_p2_b1_[iter_part2_desc_idx_].get())
    , brgemm_kernel_iter_p1_n_tail_(
              rnn_brgemm_.kernel_iter_p2_N_tail_b1_[iter_part2_desc_idx_].get())
    , brgemm_kernel_iter_p1_k_tail_(
              rnn_brgemm_.kernel_iter_p2_K2_tail_b1_[iter_part2_desc_idx_]
                      .get())
    , brgemm_kernel_iter_p1_nk_tail_(
              rnn_brgemm_.kernel_iter_p2_NK2_tail_b1_[iter_part2_desc_idx_]
                      .get())
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layer_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx_].get())
    , pallete_buff_iter_main_(rnn.k1_block == rnn.k2_block
                      ? rnn_brgemm_.pallete_buff_layer_
                      : rnn_brgemm_.pallete_buff_iter_)
    , pallete_buff_iter_n_tail_(rnn.k1_block == rnn.k2_block
                      ? rnn_brgemm_.pallete_buff_layer_n_tail_
                      : rnn_brgemm_.pallete_buff_iter_n_tail_)
    , pallete_buff_iter_k_tail_(rnn.k1_tail == rnn.k2_tail
                      ? rnn_brgemm_.pallete_buff_k1_tail_
                      : rnn_brgemm_.pallete_buff_k2_tail_)
    , pallete_buff_iter_nk_tail_(rnn.k1_tail == rnn.k2_tail
                      ? rnn_brgemm_.pallete_buff_nk1_tail_
                      : rnn_brgemm_.pallete_buff_nk2_tail_)
    , pallete_buff_layer_main_(rnn_brgemm_.pallete_buff_layer_)
    , pallete_buff_layer_n_tail_(rnn_brgemm_.pallete_buff_layer_n_tail_)
    , pallete_buff_layer_k_tail_(rnn_brgemm_.pallete_buff_k1_tail_)
    , pallete_buff_layer_nk_tail_(rnn_brgemm_.pallete_buff_nk1_tail_)
    , amx_scratchpad_(static_cast<char *>(amx_scratchpad))
    , addr_batch_global_(addr_batch_global)
    , fused_postgemm_part1_(fused_postgemm_part1)
    , fused_postgemm_part2_(fused_postgemm_part2) {}

void brgemm_gru_t::execute() const {
    // Work is split over M blocks only; skip the parallel region when there
    // is a single block.
    const int nthr = work_amount_ == 1 ? 1 : max_nthr_;
    parallel(nthr, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

void brgemm_gru_t::kernel(const int ithr, const int nthr) const {
    dim_t start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = is_superset(rnn_.brgemm_isa, x64::avx512_core_amx);
    char *const amx_buffer = is_amx ? amx_scratchpad_
                    + rnn_.m_block * rnn_.n_block * ithr * acc_dt_size_
                                    : nullptr;
    const dim_t max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff_layer = nullptr;
    const char *pallete_buff_layer_k_tail = nullptr;
    const char *pallete_buff_iter = nullptr;
    const char *pallete_buff_iter_k_tail = nullptr;

    amx_tile_configuration_loader_t load_cfg_if_needed;
    while (start < end) {
        dim_t mb = start;
        const auto m = mb * rnn_.m_block;
        const auto *const Al_m = Al_ + m * LDAl_ * src_dt_size_;
        const auto *const Ai_m = Ai_ + m * LDAi_p1_ * src_dt_size_;
        const auto *const Ai2_m = Dl_ + m * LDAi_p2_ * src_dt_size_;

        for (dim_t nb_i = 0; nb_i < n_blocking_; nb_i++) {
            const auto nb
                    = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
            const auto n = nb * rnn_.n_block;

            const auto *const Bl_n = Bl_ + nb * Bl_n_offset_ * wei_dt_size_;
            const auto *const Bi_n = Bi_ + nb * Bi_n_offset_ * wei_dt_size_;
            auto *const C_gates_n
                    = C_gates_ + (m * rnn_.LDC + n) * scratch_dt_size_;
            auto *const C_cell_n
                    = C_cell_ + (m * rnn_.LDC + n) * scratch_dt_size_;

            const brgemm_kernel_t *brgemm_kernel_layer
                    = brgemm_kernel_layer_main_;
            const brgemm_kernel_t *brgemm_kernel_layer_k_tail
                    = brgemm_kernel_layer_k_tail_;
            const brgemm_kernel_t *brgemm_kernel_iter_p0
                    = brgemm_kernel_iter_p0_main_;
            const brgemm_kernel_t *brgemm_kernel_iter_p0_k_tail
                    = brgemm_kernel_iter_p0_k_tail_;

            if (is_amx) {
                pallete_buff_layer = pallete_buff_layer_main_;
                pallete_buff_layer_k_tail = pallete_buff_layer_k_tail_;
                pallete_buff_iter = pallete_buff_iter_main_;
                pallete_buff_iter_k_tail = pallete_buff_iter_k_tail_;
            }

            const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
            if (do_n_tail) {
                brgemm_kernel_layer = brgemm_kernel_layer_n_tail_;
                brgemm_kernel_layer_k_tail = brgemm_kernel_layer_nk_tail_;
                brgemm_kernel_iter_p0 = brgemm_kernel_iter_p0_n_tail_;
                brgemm_kernel_iter_p0_k_tail = brgemm_kernel_iter_p0_nk_tail_;

                if (is_amx) {
                    pallete_buff_layer = pallete_buff_layer_n_tail_;
                    pallete_buff_layer_k_tail = pallete_buff_layer_nk_tail_;
                    pallete_buff_iter = pallete_buff_iter_n_tail_;
                    pallete_buff_iter_k_tail = pallete_buff_iter_nk_tail_;
                }
            }

            if (need_gemm_layer_) {
                if (is_amx) load_cfg_if_needed(pallete_buff_layer);
                for (int g = 0; g < n_gates_; g++) {
                    const auto *const Bl_g
                            = Bl_n + g * Bl_g_offset_ * wei_dt_size_;
                    auto *const C_gates_g
                            = C_gates_n + g * rnn_.N * scratch_dt_size_;

                    for (int batch_idx = 0; batch_idx < rnn_.KB1_blocks;
                            batch_idx++) {
                        addr_batch[batch_idx].ptr.A = Al_m
                                + batch_idx * rnn_.k1_block * src_dt_size_;
                        addr_batch[batch_idx].ptr.B = Bl_g
                                + batch_idx * Bl_kb_offset_ * wei_dt_size_;
                    }
                    brgemm_kernel_execute(brgemm_kernel_layer, rnn_.KB1_blocks,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            amx_buffer);
                }
            }

            if (need_gemm_layer_ && rnn_.k1_tail > 0) {
                if (is_amx) load_cfg_if_needed(pallete_buff_layer_k_tail);
                for (int g = 0; g < n_gates_; g++) {
                    const auto *const Bl_g
                            = Bl_n + g * Bl_g_offset_ * wei_dt_size_;
                    auto *const C_gates_g
                            = C_gates_n + g * rnn_.N * scratch_dt_size_;

                    addr_batch[0].ptr.A = Al_m
                            + rnn_.KB1_blocks * rnn_.k1_block * src_dt_size_;
                    addr_batch[0].ptr.B = Bl_g
                            + rnn_.KB1_blocks * Bl_kb_offset_ * wei_dt_size_;
                    brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            amx_buffer);
                }
            }
            if (is_amx) load_cfg_if_needed(pallete_buff_iter);
            for (int g = 0; g < n_gates_ - 1; g++) {
                const auto *const Bi_g = Bi_n + g * Bi_g_offset_ * wei_dt_size_;
                auto *const C_gates_g
                        = C_gates_n + g * rnn_.N * scratch_dt_size_;

                for (int batch_idx = 0; batch_idx < rnn_.KB2_blocks;
                        batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = Ai_m + batch_idx * rnn_.k2_block * src_dt_size_;
                    addr_batch[batch_idx].ptr.B
                            = Bi_g + batch_idx * Bi_kb_offset_ * wei_dt_size_;
                }

                brgemm_kernel_execute(brgemm_kernel_iter_p0, rnn_.KB2_blocks,
                        addr_batch, reinterpret_cast<void *>(C_gates_g),
                        amx_buffer);
            }

            if (rnn_.k2_tail > 0) {
                if (is_amx) load_cfg_if_needed(pallete_buff_iter_k_tail);
                for (int g = 0; g < n_gates_ - 1; g++) {
                    const auto *const Bi_g
                            = Bi_n + g * Bi_g_offset_ * wei_dt_size_;
                    auto *const C_gates_g
                            = C_gates_n + g * rnn_.N * scratch_dt_size_;

                    addr_batch[0].ptr.A = Ai_m
                            + rnn_.KB2_blocks * rnn_.k2_block * src_dt_size_;
                    addr_batch[0].ptr.B = Bi_g
                            + rnn_.KB2_blocks * Bi_kb_offset_ * wei_dt_size_;

                    brgemm_kernel_execute(brgemm_kernel_iter_p0_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            amx_buffer);
                }
            }

            if (!rnn_.unfused_post_gemm) {
                const auto block_step
                        = (do_n_tail ? rnn_.n_tail : rnn_.n_block);
                fused_postgemm_part1_(m, n, nb_i, Ai_m + n * src_dt_size_,
                        C_gates_n, C_cell_n, block_step);
            }
        }

        for (dim_t nb_i = 0; nb_i < n_blocking_; nb_i++) {
            const auto nb
                    = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
            const auto n = nb * rnn_.n_block;

            const auto *const Bi2_n = Bi2_ + nb * Bi_n_offset_ * wei_dt_size_;
            auto *const C_gates_n
                    = C_gates_ + (m * rnn_.LDC + n) * scratch_dt_size_;

            const brgemm_kernel_t *brgemm_kernel_iter_p1
                    = brgemm_kernel_iter_p1_main_;
            const brgemm_kernel_t *brgemm_kernel_iter_p1_k_tail
                    = brgemm_kernel_iter_p1_k_tail_;

            if (is_amx) {
                pallete_buff_iter = pallete_buff_iter_main_;
                pallete_buff_iter_k_tail = pallete_buff_iter_k_tail_;
            }

            const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
            if (do_n_tail) {
                brgemm_kernel_iter_p1 = brgemm_kernel_iter_p1_n_tail_;
                brgemm_kernel_iter_p1_k_tail = brgemm_kernel_iter_p1_nk_tail_;

                if (is_amx) {
                    pallete_buff_iter = pallete_buff_iter_n_tail_;
                    pallete_buff_iter_k_tail = pallete_buff_iter_nk_tail_;
                }
            }

            if (is_amx) load_cfg_if_needed(pallete_buff_iter);
            for (int g = 0; g < 1; g++) {
                const auto *const Bi2_g
                        = Bi2_n + g * Bi_g_offset_ * wei_dt_size_;
                auto *const C_gates_g = C_gates_n
                        + (n_gates_ - 1) * rnn_.N * scratch_dt_size_;

                for (int batch_idx = 0; batch_idx < rnn_.KB2_blocks;
                        batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = Ai2_m + batch_idx * rnn_.k2_block * src_dt_size_;
                    addr_batch[batch_idx].ptr.B
                            = Bi2_g + batch_idx * Bi_kb_offset_ * wei_dt_size_;
                }

                brgemm_kernel_execute(brgemm_kernel_iter_p1, rnn_.KB2_blocks,
                        addr_batch, reinterpret_cast<void *>(C_gates_g),
                        amx_buffer);
            }

            if (rnn_.k2_tail > 0) {
                if (is_amx) load_cfg_if_needed(pallete_buff_iter_k_tail);
                for (int g = 0; g < 1; g++) {
                    const auto *const Bi2_g
                            = Bi2_n + g * Bi_g_offset_ * wei_dt_size_;
                    auto *const C_gates_g = C_gates_n
                            + (n_gates_ - 1) * rnn_.N * scratch_dt_size_;

                    addr_batch[0].ptr.A = Ai2_m
                            + rnn_.KB2_blocks * rnn_.k2_block * src_dt_size_;
                    addr_batch[0].ptr.B = Bi2_g
                            + rnn_.KB2_blocks * Bi_kb_offset_ * wei_dt_size_;

                    brgemm_kernel_execute(brgemm_kernel_iter_p1_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            amx_buffer);
                }
            }
            if (!rnn_.unfused_post_gemm && nb_i == n_blocking_ - 1) {
                fused_postgemm_part2_(m, 0, 0, Ai_m,
                        C_gates_ + m * rnn_.LDC * scratch_dt_size_,
                        C_cell_ + m * rnn_.LDC * scratch_dt_size_, rnn_.N);
            }
        }
        ++start;
    }
}

brgemm_merged_layer_t::brgemm_merged_layer_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const void *src_layer,
        void *w_layer, void *scratch_gates, void *amx_scratchpad,
        x64::brgemm_batch_element_t *addr_batch_global)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , src_dt_size_(types::data_type_size(rnn.cell_dt))
    , wei_dt_size_(types::data_type_size(rnn.weights_dt))
    , scratch_dt_size_(types::data_type_size(rnn.scratch_dt))
    , acc_dt_size_(types::data_type_size(rnn.acc_dt))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , Al_(static_cast<const char *>(src_layer))
    , Bl_(static_cast<const char *>(w_layer))
    , C_(static_cast<char *>(scratch_gates))
    , LDAl_(rnn_.src_layer_ld(cell_position))
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , n_blocking_(rnn_.N_blocks * rnn_.n_gates)
    , m_blocking_(rnn_.Mlayermerged_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , Bl_n_offset_(rnn_.K1padded * rnn_.n_block)
    , Bl_g_offset_(rnn_.N_blocks * Bl_n_offset_)
    , Al_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , Bl_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , Bl_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layermerged_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layermerged_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layermerged_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layermerged_NK1_tail_b1_[layer_desc_idx_]
                      .get())
    , pallete_buff_layer_main_(rnn_brgemm_.pallete_buff_layermerged_)
    , pallete_buff_layer_n_tail_(rnn_brgemm_.pallete_buff_layermerged_n_tail_)
    , pallete_buff_layer_k_tail_(rnn_brgemm_.pallete_buff_layermerged_k1_tail_)
    , pallete_buff_layer_nk_tail_(
              rnn_brgemm_.pallete_buff_layermerged_nk1_tail_)
    , amx_scratchpad_(static_cast<char *>(amx_scratchpad))
    , addr_batch_global_(addr_batch_global) {}

void brgemm_merged_layer_t::execute() const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

void brgemm_merged_layer_t::kernel(const int ithr, const int nthr) const {
    using namespace cpu::rnn_utils;

    dim_t start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = is_superset(rnn_.brgemm_isa, x64::avx512_core_amx);
    const auto m_block = rnn_.mlayermerged_block;
    char *const amx_buffer = is_amx
            ? amx_scratchpad_ + m_block * rnn_.n_block * ithr * acc_dt_size_
            : nullptr;
    const dim_t max_K_Block = rnn_.KB1_blocks + 1;
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff_layer = nullptr;
    const char *pallete_buff_layer_k_tail = nullptr;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    amx_tile_configuration_loader_t load_cfg_if_needed;

    while (start < end) {
        const auto m = mb * m_block;
        const auto nb = nb_i / rnn_.n_gates;
        const auto n = nb * rnn_.n_block;
        const auto g = nb_i % rnn_.n_gates;

        const auto *const Al_m = Al_ + m * LDAl_ * src_dt_size_;
        const auto *const Bl_n = Bl_ + nb * Bl_n_offset_ * wei_dt_size_;
        auto *const C_n = C_ + (m * rnn_.LDC + n) * scratch_dt_size_;

        const brgemm_kernel_t *brgemm_kernel_layer_b0
                = brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_layer_k_tail
                = brgemm_kernel_layer_k_tail_;

        if (is_amx) {
            pallete_buff_layer = pallete_buff_layer_main_;
            pallete_buff_layer_k_tail = pallete_buff_layer_k_tail_;
        }

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
        if (do_n_tail) {
            brgemm_kernel_layer_b0 = brgemm_kernel_layer_n_tail_;
            brgemm_kernel_layer_k_tail = brgemm_kernel_layer_nk_tail_;

            if (is_amx) {
                pallete_buff_layer = pallete_buff_layer_n_tail_;
                pallete_buff_layer_k_tail = pallete_buff_layer_nk_tail_;
            }
        }

        const auto *const Bl_g = Bl_n + g * Bl_g_offset_ * wei_dt_size_;
        auto *const C_g = C_n + g * rnn_.N * scratch_dt_size_;

        if (is_amx) load_cfg_if_needed(pallete_buff_layer);
        for (int i = 0; i < rnn_.KB1_blocks; i++) {
            addr_batch[i].ptr.A = Al_m + i * rnn_.k1_block * src_dt_size_;
            addr_batch[i].ptr.B = Bl_g + i * Bl_kb_offset_ * wei_dt_size_;
        }
        brgemm_kernel_execute(brgemm_kernel_layer_b0, rnn_.KB1_blocks,
                addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);

        if (rnn_.k1_tail) {
            if (is_amx) load_cfg_if_needed(pallete_buff_layer_k_tail);

            const auto *const Bl_g = Bl_n + g * Bl_g_offset_ * wei_dt_size_;
            auto *const C_g = C_n + g * rnn_.N * scratch_dt_size_;

            addr_batch[0].ptr.A = Al_m + Al_k_tail_offset_ * src_dt_size_;
            addr_batch[0].ptr.B = Bl_g + Bl_k_tail_offset_ * wei_dt_size_;
            brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1, addr_batch,
                    reinterpret_cast<void *>(C_g), amx_buffer);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
