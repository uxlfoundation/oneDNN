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

#ifndef CPU_RNN_REF_RNN_HPP
#define CPU_RNN_REF_RNN_HPP

#include <assert.h>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/postgemm_dispatcher.hpp"
#if DNNL_X64
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"
#endif
#include "cpu/rnn/rnn_utils.hpp"
namespace dnnl {
namespace impl {
namespace cpu {

namespace {
template <typename gates_t, typename acc_t>
// The loop body needs to be put in a function as some versions of icc have
// an issue with lambdas & macros inside omp simd loops
inline void body_loop(int i, int k, const gates_t *ws_gates, acc_t *diff_bias,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position) {
    if (rnn.diff_weights_overwrite && (cell_position & rnn_utils::last_iter))
        diff_bias[i * rnn.dhc + k] = 0.0f;
    for (int j = 0; j < rnn.mb; j++)
        diff_bias[i * rnn.dhc + k]
                += ws_gates[j * rnn.scratch_gates_ld + i * rnn.dhc + k];
}
} // namespace

template <typename gates_t, typename acc_t>
void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const gates_t *ws_gates_,
        acc_t *diff_bias_) {

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
        body_loop(i, k, ws_gates_, diff_bias_, rnn, cell_position);
    });
#endif
}

// A set of matmul primitive descriptors for unique shapes, leading dimensions, etc.
class rnn_matmul_primitive_descriptors_t {
public:
    using pd_ptr = std::shared_ptr<primitive_desc_t>;

    struct key_t {
        dim_t M, N, K, LDA, LDB, LDC;
        data_type_t A_type, B_type, C_type;
        bool transB, do_sum;

        key_t(dim_t M, dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                data_type_t A_type, data_type_t B_type, data_type_t C_type,
                bool transB, bool sum_po)
            : M(M)
            , N(N)
            , K(K)
            , LDA(LDA)
            , LDB(LDB)
            , LDC(LDC)
            , A_type(A_type)
            , B_type(B_type)
            , C_type(C_type)
            , transB(transB)
            , do_sum(sum_po) {}

        bool operator==(const key_t &rhs) const {
            return M == rhs.M && N == rhs.N && K == rhs.K && LDA == rhs.LDA
                    && LDB == rhs.LDB && LDC == rhs.LDC && A_type == rhs.A_type
                    && B_type == rhs.B_type && C_type == rhs.C_type
                    && transB == rhs.transB && do_sum == rhs.do_sum;
        };
    };

    // Create a new primitive descriptor if the descriptor with the same key data doesn't exist
    status_t create_if_not_exist(const key_t &key, engine_t *engine);

    size_t get_max_scratchpad_size() const;

private:
    using pd_vector = std::vector<std::pair<key_t, pd_ptr>>;
    pd_vector pds_;

    using pd_iter = pd_vector::const_iterator;
    pd_iter begin() const { return pds_.begin(); }
    pd_iter end() const { return pds_.end(); }

    friend class rnn_matmul_primitives_t;
};

// A set of matmul primitives for unique shapes, leading dimensions, etc.
class rnn_matmul_primitives_t {
public:
    using key_t = rnn_matmul_primitive_descriptors_t::key_t;

    status_t create_primitives(
            const rnn_matmul_primitive_descriptors_t &pds, engine_t *engine);

    // Create execution context for each matmul primitive instance using the provided
    // RNN execution context. To be called once per RNN execution.
    void create_all_ctx(exec_ctx_t const &ctx) const;

    // Run execute function for corresponding matmul primitive instance
    status_t apply(const exec_ctx_t &ctx, const key_t &key, const void *A,
            const void *B, void *C) const;

private:
    using mm_ptr = std::shared_ptr<primitive_t>;

    struct matmul_primitive_data_t {
        using mem_ptr = std::shared_ptr<memory_t>;
        mm_ptr mm_prim_;
        mem_ptr A_mem_;
        mem_ptr B_mem_;
        mem_ptr C_mem_;
        std::shared_ptr<exec_ctx_t> ctx_;
    };

    using mm_data_ptr = std::shared_ptr<matmul_primitive_data_t>;
    std::vector<std::pair<key_t, mm_data_ptr>> mms_;

    mm_data_ptr find(const key_t &key) const {
        for (const auto &kv : mms_) {
            if (kv.first == key) return kv.second;
        }
        return {};
    }

    static void create_mem_wrappers(const mm_data_ptr &matmul_data);
    static void create_ctx(
            const mm_data_ptr &matmul_data, const exec_ctx_t &ctx);
};

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct ref_rnn_fwd_t;

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct ref_rnn_bwd_t;

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
struct ref_rnn_common_t : public primitive_t {
    static constexpr impl::data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    using fwd_t = ref_rnn_fwd_t<src_type, weights_type, acc_type>;
    using bwd_t = ref_rnn_bwd_t<src_type, weights_type, acc_type>;
    using impl_t = typename utils::conditional<aprop == prop_kind::forward,
            fwd_t, bwd_t>::type;
    using postgemm_t = typename utils::conditional<aprop == prop_kind::forward,
            rnn_postgemm_fwd_t<src_type, scratch_type, acc_type>,
            rnn_postgemm_bwd_t<src_type, scratch_type, acc_type>>::type;

    /* These types are defined for each element in the cell execution */
    using src_layer_t = typename prec_traits_t<src_type>::type;
    using src_iter_t = typename prec_traits_t<src_type>::type;
    using dst_layer_t = typename prec_traits_t<src_type>::type;
    using dst_iter_t = typename prec_traits_t<src_type>::type;
    using weights_t = typename prec_traits_t<weights_type>::type;
    using gemm_data_t = typename prec_traits_t<src_type>::type;
    using gemm_acc_t = typename prec_traits_t<acc_type>::type;
    using scratch_t = typename prec_traits_t<scratch_type>::type;
    using ht_t = typename prec_traits_t<src_type>::type;
    using gates_t = typename prec_traits_t<src_type>::type;

    using class_name
            = ref_rnn_common_t<aprop, src_type, weights_type, acc_type>;
#if DNNL_X64
    using ref_rnn_brgemm_t = x64::rnn_brgemm_utils::rnn_brgemm_t<aprop>;
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

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        using base_pd_t::base_pd_t;

        const char *impl_name() const {
#if DNNL_X64
            using namespace dnnl::impl::cpu::x64;
            return rnn_.is_brgemm
                    ? JIT_IMPL_NAME_HELPER("brgemm:", rnn_.brgemm_isa, "")
                    : rnn_.use_matmul ? "ref+matmul"
                                      : "ref";
#else
            return "ref";
#endif
        }

        DECLARE_COMMON_PD_T(impl_name(), impl_t, USE_GLOBAL_SCRATCHPAD);

        status_t init_ref(engine_t *engine);
        status_t init_brgemm(engine_t *engine);
        status_t init(engine_t *engine);

        rnn_utils::rnn_conf_t rnn_;
        rnn_matmul_primitive_descriptors_t matmul_pds_;
#if DNNL_X64
        std::shared_ptr<primitive_desc_t> bf32_wei_layer_reorder_pd_;
        std::shared_ptr<primitive_desc_t> bf32_wei_iter_reorder_pd_;
#endif
    protected:
        void init_scratchpad(size_t scratchpad_sz);
    };

    ref_rnn_common_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    ~ref_rnn_common_t() override { delete rnn_postgemm_; }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
#if DNNL_X64
    ref_rnn_brgemm_t rnn_brgemm_;
    std::shared_ptr<primitive_t> bf32_wei_layer_reorder_;
    std::shared_ptr<primitive_t> bf32_wei_iter_reorder_;
#endif

    template <typename input_t>
    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_,
            const input_t *xt_, const gemm_acc_t *diff_dst_layer) const;

    template <typename input_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_iter_t *ws_states_iter_, void *ws_states_iter_c_,
            gemm_acc_t *ws_diff_states_iter_,
            gemm_acc_t *ws_diff_states_iter_c_, const input_t *src_iter_,
            const void *src_iter_c_, const gemm_acc_t *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    template <typename dst_layer_dt, typename dst_iter_dt>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer_,
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_,
            const gemm_acc_t *ws_diff_states_layer_) const;

    template <typename prim_dst_iter_t, typename prim_dst_layer_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            prim_dst_iter_t *dst_iter_, void *dst_iter_c_,
            gemm_acc_t *diff_src_iter_, float *diff_src_iter_c_,
            const prim_dst_layer_t *dst_layer_,
            const src_iter_t *ws_states_iter_, const void *ws_states_iter_c,
            const gemm_acc_t *ws_diff_states_iter_,
            const gemm_acc_t *ws_diff_states_iter_c_) const;

    rnn_grid_execution_sig(linear_execution);
    virtual rnn_cell_execution_sig(cell_execution_ref) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_execution_ref) = 0;
    virtual rnn_cell_execution_sig(cell_execution_brgemm) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_brgemm) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru_lbr) = 0;
    virtual rnn_gemm_sig(gemm) = 0;
    virtual rnn_gemm_sig(packed_gemm) = 0;
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

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
    rnn_matmul_primitives_t mm_primitives_ {};

    gemm_t gemm_layer_func {};
    gemm_t gemm_iter_func {};
    gemm_t gemm_projection_func {};
};

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct ref_rnn_fwd_t : public ref_rnn_common_t<prop_kind::forward, src_type,
                               weights_type, acc_type> {
    using base_t = ref_rnn_common_t<prop_kind::forward, src_type, weights_type,
            acc_type>;
    using src_layer_t = typename base_t::src_layer_t;
    using src_iter_t = typename base_t::src_iter_t;
    using dst_layer_t = typename base_t::dst_layer_t;
    using dst_iter_t = typename base_t::dst_iter_t;
    using weights_t = typename base_t::weights_t;
    using gemm_data_t = typename base_t::gemm_data_t;
    using gemm_acc_t = typename base_t::gemm_acc_t;
    using scratch_t = typename base_t::scratch_t;
    using ht_t = typename base_t::ht_t;
    using gates_t = typename base_t::gates_t;

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

    using base_t::scratch_type;

private:
    rnn_gemm_sig(gemm) override;
    rnn_gemm_sig(packed_gemm) override;
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
};

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct ref_rnn_bwd_t : public ref_rnn_common_t<prop_kind::backward, src_type,
                               weights_type, acc_type> {
    using base_t = ref_rnn_common_t<prop_kind::backward, src_type, weights_type,
            acc_type>;
    using src_layer_t = typename base_t::src_layer_t;
    using src_iter_t = typename base_t::src_iter_t;
    using dst_layer_t = typename base_t::dst_layer_t;
    using dst_iter_t = typename base_t::dst_iter_t;
    using weights_t = typename base_t::weights_t;
    using gemm_data_t = typename base_t::gemm_data_t;
    using gemm_acc_t = typename base_t::gemm_acc_t;
    using scratch_t = typename base_t::scratch_t;
    using ht_t = typename base_t::ht_t;
    using gates_t = typename base_t::gates_t;

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
    rnn_gemm_sig(gemm) override;
    rnn_gemm_sig(packed_gemm) override;
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override {
        return dnnl_runtime_error;
    };
};

using ref_rnn_common_fwd_f32_t = ref_rnn_common_t<prop_kind::forward,
        data_type::f32, data_type::f32, data_type::f32>;
using ref_rnn_common_bwd_f32_t = ref_rnn_common_t<prop_kind::backward,
        data_type::f32, data_type::f32, data_type::f32>;

using ref_rnn_common_fwd_bf16_t = ref_rnn_common_t<prop_kind::forward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_common_bwd_bf16_t = ref_rnn_common_t<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_common_fwd_f16_t = ref_rnn_common_t<prop_kind::forward,
        data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_common_bwd_f16_t = ref_rnn_common_t<prop_kind::backward,
        data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_common_fwd_u8s8_t = ref_rnn_common_t<prop_kind::forward,
        data_type::u8, data_type::s8, data_type::s32>;
using ref_rnn_common_fwd_s8s8_t = ref_rnn_common_t<prop_kind::forward,
        data_type::s8, data_type::s8, data_type::s32>;

using ref_rnn_fwd_f32_t
        = ref_rnn_fwd_t<data_type::f32, data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t
        = ref_rnn_bwd_t<data_type::f32, data_type::f32, data_type::f32>;

using ref_rnn_fwd_bf16_t
        = ref_rnn_fwd_t<data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_bwd_bf16_t
        = ref_rnn_bwd_t<data_type::bf16, data_type::bf16, data_type::f32>;

using ref_rnn_fwd_f16_t
        = ref_rnn_fwd_t<data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_bwd_f16_t
        = ref_rnn_bwd_t<data_type::f16, data_type::f16, data_type::f32>;

using ref_rnn_fwd_u8s8_t
        = ref_rnn_fwd_t<data_type::u8, data_type::s8, data_type::s32>;
using ref_rnn_fwd_s8s8_t
        = ref_rnn_fwd_t<data_type::s8, data_type::s8, data_type::s32>;
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
