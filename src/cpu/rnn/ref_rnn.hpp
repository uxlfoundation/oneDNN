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

#ifndef CPU_RNN_REF_RNN_HPP
#define CPU_RNN_REF_RNN_HPP

#include <assert.h>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/postgemm_dispatcher.hpp"
#if DNNL_X64
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"
#elif DNNL_AARCH64
#include "cpu/aarch64/rnn/rnn_brgemm_utils.hpp"
#endif
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
template <typename gates_t>
// The loop body needs to be put in a function as some versions of icc have
// an issue with lambdas & macros inside omp simd loops
inline void body_loop(int i, int k, const gates_t *ws_gates, float *diff_bias,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position) {
    if (rnn.diff_weights_overwrite && (cell_position & rnn_utils::last_iter))
        diff_bias[i * rnn.dhc + k] = 0.0f;
    for (int j = 0; j < rnn.mb; j++)
        diff_bias[i * rnn.dhc + k]
                += ws_gates[j * rnn.scratch_gates_ld + i * rnn.dhc + k];
}
} // namespace

template <typename gates_t>
void gates_reduction_body(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const gates_t *ws_gates_,
        float *diff_bias_) {

    // @TODO: block k on simd-width to enable vectorization in
    // parallel_nd path
    // Note: `defined(_OPENMP)` is needed for `icx -fsycl` compilation. The
    // second pass for device code doesn't contain OpenMP macros.
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP && defined(_OPENMP) \
        && _OPENMP >= 201307 \
        && (!defined(__INTEL_COMPILER) || __INTEL_COMPILER < 1910)
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dhc; k++)
            body_loop(i, k, ws_gates_, diff_bias_, rnn, cell_position);
#else
    parallel_nd(rnn.n_gates, rnn.dhc, [&](dim_t i, dim_t k) {
        body_loop(static_cast<int>(i), static_cast<int>(k), ws_gates_,
                diff_bias_, rnn, cell_position);
    });
#endif
}

inline void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const void *ws_gates_,
        float *diff_bias_) {
    switch (rnn.scratch_dt) {
        case data_type::f32:
            gates_reduction_body(rnn, cell_position,
                    static_cast<const float *>(ws_gates_), diff_bias_);
            break;
        case data_type::bf16:
            gates_reduction_body(rnn, cell_position,
                    static_cast<const bfloat16_t *>(ws_gates_), diff_bias_);
            break;
        case data_type::f16:
            gates_reduction_body(rnn, cell_position,
                    static_cast<const float16_t *>(ws_gates_), diff_bias_);
            break;
        default: assert(!"unsupported data type");
    }
}

struct ref_rnn_fwd_t;
struct ref_rnn_bwd_t;

// Data type agnostic part of the RNN primitive descriptor. It is templated
// only on the propagation kind to keep the number of instantiations low.
template <prop_kind_t aprop>
struct ref_rnn_common_pd_t
    : public utils::conditional<aprop == prop_kind::forward, cpu_rnn_fwd_pd_t,
              cpu_rnn_bwd_pd_t>::type {
    using base_pd_t = typename utils::conditional<aprop == prop_kind::forward,
            cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;
    using base_pd_t::base_pd_t;
#if DNNL_X64 || DNNL_AARCH64
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<aprop>;
#endif

    const char *impl_name() const {
#if DNNL_X64 || DNNL_AARCH64
        return rnn_.is_brgemm
                ? JIT_IMPL_NAME_HELPER("brgemm:", rnn_.brgemm_isa, "")
                : rnn_.use_matmul ? "ref+matmul"
                                  : "ref";
#else
        return "ref";
#endif
    }

    rnn_utils::rnn_conf_t rnn_;
    std::shared_ptr<primitive_desc_t> matmul_layer_1_pd_;
    std::shared_ptr<primitive_desc_t> matmul_layer_2_pd_;
    std::shared_ptr<primitive_desc_t> matmul_layer_3_pd_;
    std::shared_ptr<primitive_desc_t> matmul_iter_1_pd_;
    std::shared_ptr<primitive_desc_t> matmul_iter_2_pd_;
    std::shared_ptr<primitive_desc_t> matmul_iter_3_pd_;
    std::shared_ptr<primitive_desc_t> matmul_part2_1_pd_;
    std::shared_ptr<primitive_desc_t> matmul_part2_2_pd_;
    std::shared_ptr<primitive_desc_t> matmul_part2_3_pd_;
    std::shared_ptr<primitive_desc_t> matmul_part2_4_pd_;
#if DNNL_X64
    std::shared_ptr<primitive_desc_t> bf32_wei_layer_reorder_pd_;
    std::shared_ptr<primitive_desc_t> bf32_wei_iter_reorder_pd_;
#endif

protected:
    // Picks the first (src, weights, acc) data type triplet that the
    // descriptor and the hardware support.
    status_t init(const engine_t *engine);

private:
    status_t init(const engine_t *engine, data_type_t src_type,
            data_type_t weights_type, data_type_t acc_type);
    status_t init_ref(const engine_t *engine, data_type_t src_type,
            data_type_t weights_type, data_type_t acc_type);
    status_t init_brgemm(const engine_t *engine, data_type_t src_type,
            data_type_t weights_type, data_type_t acc_type);
    void init_scratchpad(size_t scratchpad_sz, data_type_t acc_type);
};

template <prop_kind_t aprop>
struct ref_rnn_common_t : public primitive_t {
    using impl_t = typename utils::conditional<aprop == prop_kind::forward,
            ref_rnn_fwd_t, ref_rnn_bwd_t>::type;
    using postgemm_t = typename utils::conditional<aprop == prop_kind::forward,
            rnn_postgemm_fwd_t, rnn_postgemm_bwd_t>::type;

    using class_name = ref_rnn_common_t<aprop>;
#if DNNL_X64 || DNNL_AARCH64
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<aprop>;
#endif

    using cell_execution_f
            = dnnl_status_t (class_name::*)(rnn_cell_execution_sig_args) const;
    using grid_execution_f
            = dnnl_status_t (class_name::*)(rnn_grid_execution_sig_args) const;
    using merged_layer_execution_f = dnnl_status_t (class_name::*)(
            rnn_merged_layer_execution_sig_args) const;

    using gemm_t = dnnl_status_t (class_name::*)(rnn_gemm_sig_args) const;
    using bias_prepare_t
            = void (class_name::*)(rnn_bias_prepare_sig_args) const;
    using bias_finalize_t
            = void (class_name::*)(rnn_bias_finalize_sig_args) const;
    using weights_assign_t
            = void (class_name::*)(rnn_weights_assign_sig_args) const;

    struct pd_t : public ref_rnn_common_pd_t<aprop> {
        using ref_rnn_common_pd_t<aprop>::ref_rnn_common_pd_t;

        DECLARE_COMMON_PD_T(this->impl_name(), impl_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine) {
            return ref_rnn_common_pd_t<aprop>::init(engine);
        }
    };

    ref_rnn_common_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    ~ref_rnn_common_t() override { delete rnn_postgemm_; }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
#if DNNL_X64 || DNNL_AARCH64
    ref_rnn_brgemm_t rnn_brgemm_;
#endif
#if DNNL_X64
    std::shared_ptr<primitive_t> bf32_wei_layer_reorder_;
    std::shared_ptr<primitive_t> bf32_wei_iter_reorder_;
#endif

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            void *ws_states_layer_, float *ws_diff_states_layer_,
            const void *xt_, const float *diff_dst_layer_) const;

    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn, void *ws_states_iter_,
            void *ws_states_iter_c_, float *ws_diff_states_iter_,
            float *ws_diff_states_iter_c_, const void *src_iter_,
            const void *src_iter_c_, const float *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn, void *dst_layer_,
            float *diff_src_layer_, const void *dst_iter_,
            const void *ws_states_layer_,
            const float *ws_diff_states_layer_) const;

    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn, void *dst_iter_,
            void *dst_iter_c_, float *diff_src_iter_, float *diff_src_iter_c_,
            const void *dst_layer_, const void *ws_states_iter_,
            const void *ws_states_iter_c_, const float *ws_diff_states_iter_,
            const float *ws_diff_states_iter_c_) const;

    rnn_grid_execution_sig(linear_execution);
    rnn_matmul_sig(execute_matmul);
    virtual rnn_cell_execution_sig(cell_execution_ref) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_execution_ref) = 0;
    virtual rnn_cell_execution_sig(cell_execution_brgemm) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_brgemm) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru_lbr) = 0;
    rnn_gemm_sig(gemm);
    rnn_gemm_sig(packed_gemm);
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    const std::shared_ptr<primitive_t> &get_matmul_layer(
            rnn_utils::cell_position_t cell_position) const;
    const std::shared_ptr<primitive_t> &get_matmul_iter(
            rnn_utils::cell_position_t cell_position) const;
    const std::shared_ptr<primitive_t> &get_matmul_part2(
            rnn_utils::cell_position_t cell_position) const;

    float (*activation_func)(float s, float alpha, float cliping) = nullptr;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t ws_gates_offset_ {};
    size_t ws_ht_offset_ {};
    size_t ws_states_layer_offset_ {};
    size_t ws_states_iter_offset_ {};
    size_t ws_states_iter_c_offset_ {};
    size_t ws_bias_offset_ {};
    size_t ws_diff_states_layer_offset_ {};
    size_t ws_diff_states_iter_offset_ {};
    size_t ws_diff_states_iter_c_offset_ {};
    size_t ws_grid_comp_offset_ {};
    size_t scratch_gates_offset_ {};
    size_t scratch_ht_offset_ {};
    size_t scratch_diff_ht_offset_ {};
    size_t scratch_cell_offset_ {};
    postgemm_t *rnn_postgemm_ {};

    grid_execution_f grid_computation {};
    cell_execution_f cell_func {};
    merged_layer_execution_f merged_layer_func {};

    bias_prepare_t bias_preparation_func {};
    bias_finalize_t bias_finalization_func {};
    weights_assign_t weights_layer_assign_func {};
    weights_assign_t weights_iter_assign_func {};
    weights_assign_t weights_projection_assign_func {};

    // While using Matmul instead of GeMM, we require multiple matmuls, due to
    // differences in M, N, K, LDB and post-ops at different cell_positions.
    // TODO: Maybe replace them with runtime matmul if it becomes unmanageable.
    std::shared_ptr<primitive_t> matmul_layer_1_;
    std::shared_ptr<primitive_t> matmul_layer_2_;
    std::shared_ptr<primitive_t> matmul_layer_3_;
    std::shared_ptr<primitive_t> matmul_iter_1_;
    std::shared_ptr<primitive_t> matmul_iter_2_;
    std::shared_ptr<primitive_t> matmul_iter_3_;
    std::shared_ptr<primitive_t> matmul_part2_1_;
    std::shared_ptr<primitive_t> matmul_part2_2_;
    std::shared_ptr<primitive_t> matmul_part2_3_;
    std::shared_ptr<primitive_t> matmul_part2_4_;

    gemm_t gemm_layer_func {};
    gemm_t gemm_iter_func {};
    gemm_t gemm_projection_func {};
};

// TODO: move to a .cpp file once this implementation gets one.
template <prop_kind_t aprop>
DEFINE_PD_CREATE(ref_rnn_common_t<aprop>::pd_t)

struct ref_rnn_fwd_t : public ref_rnn_common_t<prop_kind::forward> {
    using base_t = ref_rnn_common_t<prop_kind::forward>;

    using base_t::cell_func;
    using base_t::grid_computation;
    using base_t::merged_layer_func;

    using base_t::bias_finalization_func;
    using base_t::bias_preparation_func;
    using base_t::weights_iter_assign_func;
    using base_t::weights_layer_assign_func;
    using base_t::weights_projection_assign_func;

    using base_t::gemm_iter_func;
    using base_t::gemm_layer_func;
    using base_t::gemm_projection_func;

    using base_t::base_t;

private:
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
};

struct ref_rnn_bwd_t : public ref_rnn_common_t<prop_kind::backward> {
    using base_t = ref_rnn_common_t<prop_kind::backward>;

    using base_t::cell_func;
    using base_t::grid_computation;
    using base_t::merged_layer_func;

    using base_t::bias_finalization_func;
    using base_t::bias_preparation_func;
    using base_t::weights_iter_assign_func;
    using base_t::weights_layer_assign_func;
    using base_t::weights_projection_assign_func;

    using base_t::gemm_iter_func;
    using base_t::gemm_layer_func;
    using base_t::gemm_projection_func;

    using base_t::base_t;

private:
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override {
        return dnnl_runtime_error;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
