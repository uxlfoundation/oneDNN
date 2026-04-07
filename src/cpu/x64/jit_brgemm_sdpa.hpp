/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_SDPA_HPP
#define CPU_X64_JIT_BRGEMM_SDPA_HPP

#include "common/c_types_map.hpp"
// #include "common/dnnl_thread.hpp"
// #include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
// #include "common/utils.hpp"

#include "common/sdpa_pd.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_sdpa_fwd_t : public primitive_t {
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("brgemm:", isa, ""), brgemm_sdpa_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;

            // auto src_dt = invariant_src_md()->data_type;
            // auto dst_dt = invariant_dst_md()->data_type;
            // auto wei_dt = invariant_wei_md()->data_type;
            // const bool is_int8 = one_of(src_dt, u8, s8);

            // using skip_mask_t = primitive_attr_t::skip_mask_t;
            // auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
            //         | skip_mask_t::fpmath_mode;
            // if (is_int8) skip_mask |= skip_mask_t::scales;
            // // disabling verbose dispatch messages for unsupported isa for
            // // better readability
            // if (!mayiuse(isa)) return status::unimplemented;

            // VDISPATCH_sdpa(
            //         get_prop_kind() == prop_kind::forward_training,
            //         VERBOSE_BAD_PROPKIND);
            // VDISPATCH_sdpa(
            //         expect_data_types(src_dt, wei_dt, data_type::undef, dst_dt,
            //                 data_type::undef),
            //         VERBOSE_UNSUPPORTED_DT);
            // VDISPATCH_sdpa(
            //         IMPLICATION(with_bias() && is_int8,
            //                 one_of(bias_md_.data_type, f32, bf16, s32, s8, u8)),
            //         VERBOSE_UNSUPPORTED_DT);
            // VDISPATCH_sdpa(
            //         IMPLICATION(with_bias() && !is_int8,
            //                 one_of(bias_md_.data_type, f32, src_dt)),
            //         VERBOSE_UNSUPPORTED_DT);
            // VDISPATCH_sdpa(
            //         attr()->has_default_values(skip_mask, dst_dt),
            //         VERBOSE_UNSUPPORTED_ATTR);
            // VDISPATCH_sdpa(
            //         attr()->post_ops_.check_sum_consistency(dst_dt, is_int8),
            //         VERBOSE_UNSUPPORTED_POSTOP);
            // VDISPATCH_sdpa(
            //         !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            // VDISPATCH_sdpa(
            //         arg_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

            // CHECK(jbgp_.init_conf(isa, *desc(), src_md_, weights_md_, dst_md_,
            //         bias_md_, attr_, dnnl_get_max_threads()));

            // bool are_post_ops_applicable = one_of(true, jbgp_.with_sum,
            //         jbgp_.with_bias, jbgp_.with_src_scales,
            //         jbgp_.with_wei_scales, jbgp_.with_dst_scales,
            //         jbgp_.with_eltwise, jbgp_.with_binary,
            //         jbgp_.acc_dt != jbgp_.dst_dt, jbgp_.req_s8s8_compensation);

            // const float alpha = 1.0;
            // const float beta = 1.0;
            // const float beta_init = 0.0;

            // for_(int i_bs = 0; i_bs < 2; i_bs++)
            // for_(int i_init = 0; i_init < 2; i_init++)
            // for_(int i_M = 0; i_M < 2; i_M++)
            // for_(int i_N = 0; i_N < 2; i_N++)
            // for (int i_K = 0; i_K < 2; i_K++) {
            //     auto vbeta = (i_init) ? beta_init : beta;
            //     auto vM = (i_M) ? jbgp_.M_tail : jbgp_.M;
            //     auto vN = (i_N) ? jbgp_.N_tail : jbgp_.N;
            //     auto vK = (i_K) ? jbgp_.K_tail : jbgp_.K;
            //     int bs = get_brg_batchsize(i_bs, i_K);
            //     int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
            //     if (idx < 0) continue;
            //     brgemm_desc_t &brg = brg_descs_[idx];
            //     CHECK(brgemm_desc_init(&brg, isa, jbgp_.brg_type, jbgp_.src_dt,
            //             jbgp_.wei_dt, false, false, brgemm_row_major, alpha,
            //             vbeta, jbgp_.LDA, jbgp_.LDB, jbgp_.LDC, vM, vN, vK));

            //     CHECK(brgemm_desc_set_postops(
            //             &brg, attr(), &dst_md_, jbgp_.LDD, jbgp_.bia_dt));

            //     brgemm_attr_t brgattr;
            //     if (jbgp_.is_amx) {
            //         brgattr.max_bs = bs;
            //         brgattr.wary_A_k_tail_read = false;
            //         brgattr.hint_expected_A_size
            //                 = static_cast<dim_t>(jbgp_.mb) * jbgp_.ic;
            //         brgattr.hint_expected_B_size
            //                 = static_cast<dim_t>(jbgp_.oc) * jbgp_.ic;
            //         brgattr.hint_expected_C_size
            //                 = static_cast<dim_t>(jbgp_.mb) * jbgp_.oc;
            //         brgattr.hint_innermost_loop = brgemm_innermost_undef;
            //         brgattr.use_uker = jbgp_.use_uker;
            //         brgattr.use_interleave_stores = jbgp_.use_interleave_stores;
            //         brgattr.hint_prefetching = jbgp_.hint_prefetching;
            //         brgattr.fpmath_mode = attr()->fpmath_.mode_;
            //     }
            //     if (are_post_ops_applicable && jbgp_.nthr_ic_b > 1) {
            //         brgattr.generate_skip_accumulation = true;
            //     }

            //     CHECK(brgemm_desc_set_attr(&brg, brgattr));
            //     CHECK(brgemm_desc_finalize(&brg));

            //     if (jbgp_.is_amx)
            //         jbgp_.amx_buf_size_per_thread
            //                 = nstl::max(brg.get_wsp_buffer_size(),
            //                         jbgp_.amx_buf_size_per_thread);
            // }

            // auto scratchpad = scratchpad_registry().registrar();
            // jbgp_.init_scratchpad(scratchpad);

            return status::success;
        }

        // int get_brg_kernel_idx(bool is_bs_tail, bool do_initialization,
        //         bool is_M_tail, bool is_N_tail, bool is_K_tail, int bs) const {
        //     auto vM = (is_M_tail) ? jbgp_.M_tail : jbgp_.M;
        //     auto vN = (is_N_tail) ? jbgp_.N_tail : jbgp_.N;
        //     auto vK = (is_K_tail) ? jbgp_.K_tail : jbgp_.K;

        //     if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || jbgp_.LDA < vK
        //             || jbgp_.LDB < vN || jbgp_.LDC < vN)
        //         return -1;
        //     return brgemm_sdpa_utils::get_brg_kernel_index(is_bs_tail,
        //             do_initialization, is_M_tail, is_N_tail, is_K_tail);
        // }

        // int get_brg_batchsize(bool is_bs_tail, bool is_K_tail) const {
        //     auto adj_ic = jbgp_.use_buffer_a
        //             ? utils::rnd_up(jbgp_.ic, jbgp_.ic_block)
        //             : jbgp_.ic;
        //     auto bs = (is_K_tail)
        //             ? 1
        //             : ((is_bs_tail) ? (adj_ic / jbgp_.K) % jbgp_.gemm_batch_size
        //                             : jbgp_.gemm_batch_size);
        //     return bs;
        // }
    };

    brgemm_sdpa_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        // for_(int i_bs = 0; i_bs < 2; i_bs++)
        // for_(int i_M = 0; i_M < 2; i_M++)
        // for_(int i_N = 0; i_N < 2; i_N++)
        // for_(int i_K = 0; i_K < 2; i_K++)
        // for (int i_init = 0; i_init < 2; i_init++) {
        //     int bs = pd()->get_brg_batchsize(i_bs, i_K);
        //     int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, bs);
        //     if (idx < 0) continue;

        //     brgemm_kernel_t *ker = nullptr;
        //     CHECK(brgemm_kernel_create(&ker, pd()->brg_descs_[idx]));
        //     CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        //     if (pd()->jbgp_.is_amx)
        //         brgemm_palettes_.insert(idx, pd()->brg_descs_[idx]);
        // }
        // if (pd()->jbgp_.use_buffer_a)
        //     CHECK(create_brgemm_copy_to_coarse(copy_src_kernel_, &pd()->jbgp_));
        // if (pd()->jbgp_.nthr_ic_b > 1) {
        //     CHECK(safe_ptr_assign(
        //             acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
        //     CHECK(acc_ker_->create_kernel());
        // }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // std::unique_ptr<brgemm_kernel_t> brg_kernels_[];
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
