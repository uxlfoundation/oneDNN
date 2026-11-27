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

#include <vector>

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <prop_kind_t aprop>
status_t ref_rnn_common_pd_t<aprop>::init_ref(const engine_t *engine,
        data_type_t src_type, data_type_t weights_type, data_type_t acc_type) {
    const data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    using namespace prop_kind;
    using namespace utils;
    using namespace format_tag;
    using namespace rnn_utils;
    const alg_kind_t cell_kind = this->desc()->cell_kind;

    const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    const data_type_t weights_iter_dt
            = this->desc()->weights_iter_desc.data_type;
    const data_type_t weights_layer_dt
            = this->desc()->weights_layer_desc.data_type;

    VDISPATCH_RNN(utils::one_of(cell_kind, alg_kind::vanilla_rnn,
                          alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                          alg_kind::lbr_gru, alg_kind::vanilla_augru,
                          alg_kind::lbr_augru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          utils::one_of(this->desc()->prop_kind,
                                  forward_training, forward_inference)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          utils::one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(src_layer_dt == src_type, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(
            utils::everyone_is(weights_type, weights_iter_dt, weights_layer_dt),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(this->set_default_params() == status::success,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    rnn_ = utils::zero<decltype(rnn_)>();
    rnn_.is_brgemm = false;
    VDISPATCH_RNN(rnn_utils::init_conf(rnn_, *this->desc(), *this->attr(),
                          src_type, weights_type, acc_type, scratch_type,
                          this->src_md(0), this->src_md(1), this->src_md(2),
                          this->weights_md(0), this->weights_md(1),
                          this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                          this->dst_md(0), this->dst_md(1), this->dst_md(2),
                          this->arg_md(DNNL_ARG_BIAS)),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "rnn");

    VDISPATCH_RNN(IMPLICATION(rnn_.is_f16_conf(), !rnn_.is_training),
            VERBOSE_UNSUPPORTED_FEATURE, "f16 training not supported");

    if (rnn_.is_xf16_conf()) {
        VDISPATCH_RNN(
                !(!utils::one_of(rnn_.bias_dt, src_type, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                src_type, data_type::f32)),
                VERBOSE_UNSUPPORTED_DT_CFG);
    } else {
        VDISPATCH_RNN(!(rnn_.bias_dt != data_type::f32
                              || !utils::one_of(rnn_.src_iter_c_dt,
                                      data_type::undef, data_type::f32)
                              || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt),
                VERBOSE_UNSUPPORTED_DT_CFG);
    }

    /* check that no data shift have been passed to s8s8 lstm */
    VDISPATCH_RNN(IMPLICATION(rnn_.is_signed_int8_conf(),
                          this->attr()->rnn_data_qparams_.shift_ == 0.f),
            VERBOSE_UNSUPPORTED_FEATURE,
            "s8s8 lstm does not support data shift");

    /* INT8 cases with non-trivial strides are not supported */
    VDISPATCH_RNN(!(rnn_.is_int8_conf()
                          && !(rnn_.src_layer_is_trivial_stride
                                  && rnn_.dst_layer_is_trivial_stride)),
            VERBOSE_NONTRIVIAL_STRIDE);

    /* check that only supported attr have been passed */
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8)
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_projection_qparams;

    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // Set weights descriptors to desired format
    memory_desc_t new_weights_layer_md = *this->weights_md(0);
    CHECK(rnn_utils::set_expected_desc(
            rnn_, new_weights_layer_md, rnn_utils::weights_type_t::layer));
    if (this->weights_layer_md_.format_kind == format_kind::any) {
        this->weights_layer_md_ = new_weights_layer_md;
    } else if (this->weights_layer_md_.format_kind == format_kind::rnn_packed) {
        VDISPATCH_RNN(this->weights_layer_md_ == new_weights_layer_md,
                VERBOSE_INCONSISTENT_MDS, "weights_layer", "new_weights_layer");
    }

    memory_desc_t new_weights_iter_md = *this->weights_md(1);
    CHECK(rnn_utils::set_expected_desc(
            rnn_, new_weights_iter_md, rnn_utils::weights_type_t::iter));
    if (this->weights_iter_md_.format_kind == format_kind::any) {
        this->weights_iter_md_ = new_weights_iter_md;
    } else if (this->weights_iter_md_.format_kind == format_kind::rnn_packed) {
        VDISPATCH_RNN(this->weights_iter_md_ == new_weights_iter_md,
                VERBOSE_INCONSISTENT_MDS, "weights_iter", "new_weights_iter");
    }

    if (rnn_.is_lstm_projection) {
        memory_desc_t new_weights_projection_md
                = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
        CHECK(rnn_utils::set_expected_desc(rnn_, new_weights_projection_md,
                rnn_utils::weights_type_t::projection));
        if (this->weights_projection_md_.format_kind == format_kind::any) {
            this->weights_projection_md_ = new_weights_projection_md;
        } else if (this->weights_projection_md_.format_kind
                == format_kind::rnn_packed) {
            VDISPATCH_RNN(
                    this->weights_projection_md_ == new_weights_projection_md,
                    VERBOSE_INCONSISTENT_MDS, "weights_projection",
                    "new_weights_projection");
        }
    }

    VDISPATCH_RNN(this->check_layout_consistency(false /*is_brgemm*/)
                    == status::success,
            "layout consistency check failed");

    rnn_utils::set_conf(rnn_, *this->desc(), this->weights_md(0),
            this->weights_md(1), this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
            this->diff_weights_md(0), this->diff_weights_md(1),
            this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
    rnn_utils::set_workspace_sizes(rnn_, *this->desc(), acc_type, scratch_type);

    // INIT MATMULS
    auto init_matmul_pd
            = [&](std::shared_ptr<primitive_desc_t> &mpd, dim_t M, dim_t N,
                      dim_t K, dim_t LDA, dim_t LDB, dim_t LDC, bool sum_po) {
        memory_desc_t src_desc;
        const dims_t src_dims = {M, K};
        const dims_t src_strides = {LDA, 1};
        CHECK(memory_desc_init_by_strides(
                src_desc, 2, src_dims, src_type, src_strides));

        memory_desc_t wei_desc;
        const dims_t wei_dims = {K, N};
        const dims_t wei_strides = {LDB, 1};
        CHECK(memory_desc_init_by_strides(
                wei_desc, 2, wei_dims, weights_type, wei_strides));

        memory_desc_t dst_desc;
        const dims_t dst_dims = {M, N};
        const dims_t dst_strides = {LDC, 1};
        CHECK(memory_desc_init_by_strides(
                dst_desc, 2, dst_dims, scratch_type, dst_strides));

        matmul_desc_t matmul_desc;
        CHECK(matmul_desc_init(
                &matmul_desc, &src_desc, &wei_desc, nullptr, &dst_desc));
        post_ops_t po;
        CHECK(po.append_sum(1.0f));
        primitive_attr_t attr;
        CHECK(attr.set_post_ops(po));
        primitive_desc_iterator_t it(engine, (op_desc_t *)(&matmul_desc),
                sum_po ? &attr : nullptr, nullptr);
        if (!it.is_initialized()) return status::out_of_memory;

        while (++it != it.end()) {
            mpd = *it;
            const bool ok = mpd->weights_md()->extra.flags == 0;
            if (ok) return status::success;
        }
        return status::unimplemented;
    };

    if (rnn_.use_matmul) {
        { // init layer matmuls
            const dim_t M = rnn_.mb;
            const dim_t N = static_cast<dim_t>(rnn_.n_gates) * rnn_.dhc;
            const dim_t K = rnn_.slc;
            const dim_t LDA1 = rnn_.src_layer_ld_;
            const dim_t LDA2 = rnn_.ws_states_layer_ld;
            const dim_t LDA3 = rnn_.dst_iter_ld_;
            const dim_t LDB = rnn_.weights_layer_ld;
            const dim_t LDC = rnn_.scratch_gates_ld;
            const bool do_sum = false;
            if (LDA1 >= K)
                CHECK(init_matmul_pd(
                        matmul_layer_1_pd_, M, N, K, LDA1, LDB, LDC, do_sum));
            if (LDA2 >= K && LDA2 != LDA1)
                CHECK(init_matmul_pd(
                        matmul_layer_2_pd_, M, N, K, LDA2, LDB, LDC, do_sum));
            if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                CHECK(init_matmul_pd(
                        matmul_layer_3_pd_, M, N, K, LDA3, LDB, LDC, do_sum));
        }

        { // init iter matmuls
            const dim_t M = rnn_.mb;
            const dim_t N = static_cast<dim_t>(rnn_.dhc)
                    * (rnn_.n_gates - rnn_.is_orig_gru);
            const dim_t K = rnn_.sic;
            const dim_t LDA1 = rnn_.src_iter_ld_;
            const dim_t LDA2 = rnn_.ws_states_iter_ld;
            const dim_t LDA3 = rnn_.dst_layer_ld_;
            const dim_t LDB = rnn_.weights_iter_ld;
            const dim_t LDC = rnn_.scratch_gates_ld;
            const bool do_sum = !rnn_.is_lbr;
            if (LDA1 >= K)
                CHECK(init_matmul_pd(
                        matmul_iter_1_pd_, M, N, K, LDA1, LDB, LDC, do_sum));
            if (LDA2 >= K && LDA2 != LDA1)
                CHECK(init_matmul_pd(
                        matmul_iter_2_pd_, M, N, K, LDA2, LDB, LDC, do_sum));
            if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                CHECK(init_matmul_pd(
                        matmul_iter_3_pd_, M, N, K, LDA3, LDB, LDC, do_sum));

            if (rnn_.is_orig_gru) {
                const dim_t N_part2 = rnn_.dhc;
                const dim_t LDA1 = rnn_.ws_states_layer_ld;
                const dim_t LDA2 = rnn_.ws_states_iter_ld;
                const dim_t LDA3 = rnn_.dst_layer_ld_;
                const dim_t LDA4 = rnn_.dst_iter_ld_;
                if (LDA1 >= K)
                    CHECK(init_matmul_pd(matmul_part2_1_pd_, M, N_part2, K,
                            LDA1, LDB, LDC, do_sum));
                if (LDA2 >= K && LDA2 != LDA1)
                    CHECK(init_matmul_pd(matmul_part2_2_pd_, M, N_part2, K,
                            LDA2, LDB, LDC, do_sum));
                if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                    CHECK(init_matmul_pd(matmul_part2_3_pd_, M, N_part2, K,
                            LDA3, LDB, LDC, do_sum));
                if (LDA4 >= K && !utils::one_of(LDA4, LDA1, LDA2, LDA3))
                    CHECK(init_matmul_pd(matmul_part2_4_pd_, M, N_part2, K,
                            LDA4, LDB, LDC, do_sum));
            }
        }
    }
    return status::success;
}

template <prop_kind_t aprop>
status_t ref_rnn_common_pd_t<aprop>::init_brgemm(const engine_t *engine,
        data_type_t src_type, data_type_t weights_type, data_type_t acc_type) {
    using namespace prop_kind;
    using namespace utils;
    using namespace format_tag;
    using namespace rnn_utils;
#if DNNL_X64 || DNNL_AARCH64
    const alg_kind_t cell_kind = this->desc()->cell_kind;
    const data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    const data_type_t weights_iter_dt
            = this->desc()->weights_iter_desc.data_type;
    const data_type_t weights_layer_dt
            = this->desc()->weights_layer_desc.data_type;
#if DNNL_X64
    bool is_f32 = utils::everyone_is(
            data_type::f32, src_layer_dt, weights_iter_dt, weights_layer_dt);
    bool is_impl_bf16
            = utils::everyone_is(data_type::bf16, src_type, weights_type);
    bool is_fpmath_bf16 = utils::one_of(
            this->attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
    bool allow_down_conversion_to_bf16
            = is_f32 && is_fpmath_bf16 && is_impl_bf16;
#endif

    // Initialized rnn_ early to get correct verbose output
    rnn_ = utils::zero<decltype(rnn_)>();
    rnn_.is_brgemm = true;
    VDISPATCH_RNN(utils::one_of(cell_kind, alg_kind::vanilla_rnn,
                          alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                          alg_kind::lbr_gru, alg_kind::vanilla_augru,
                          alg_kind::lbr_augru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          utils::one_of(this->desc()->prop_kind,
                                  forward_training, forward_inference)),
            VERBOSE_BAD_PROPKIND);
    // LBR is not supported for training in brgemm
    VDISPATCH_RNN(IMPLICATION(utils::one_of(cell_kind, alg_kind::lbr_gru,
                                      alg_kind::lbr_augru),
                          this->desc()->prop_kind == forward_inference),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          utils::one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    // TODO: Enable diff_weights_overwrite support
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          this->diff_weights_overwrite() == false),
            VERBOSE_BAD_PROPKIND);
    // cell_type (or src_type) and primitive data type should
    // match, except for the bf32 case.
#if DNNL_X64
    VDISPATCH_RNN(IMPLICATION(!allow_down_conversion_to_bf16,
                          src_layer_dt == src_type
                                  && utils::everyone_is(weights_type,
                                          weights_iter_dt, weights_layer_dt)),
            VERBOSE_UNSUPPORTED_DT);
#else
    VDISPATCH_RNN(src_layer_dt == src_type
                    && utils::everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt),
            VERBOSE_UNSUPPORTED_DT);
#endif
    VDISPATCH_RNN(this->set_default_params() == status::success,
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    VDISPATCH_RNN(rnn_utils::init_conf(rnn_, *this->desc(), *this->attr(),
                          src_type, weights_type, acc_type, scratch_type,
                          this->src_md(0), this->src_md(1), this->src_md(2),
                          this->weights_md(0), this->weights_md(1),
                          this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                          this->dst_md(0), this->dst_md(1), this->dst_md(2),
                          this->arg_md(DNNL_ARG_BIAS)),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "rnn");

    VDISPATCH_RNN(IMPLICATION(utils::one_of(this->desc()->prop_kind,
                                      forward_training, backward),
                          (rnn_.is_xf16_conf() || rnn_.is_f32_conf())),
            VERBOSE_PROPKIND_DT_MISMATCH);

    // Support for GRU / AUGRU cell in BRGEMM-based implementation is
    // limited by forward_inference pass for now. all_f32 is enabled only
    // for small shapes on avx2, where it avoids the per-call overhead of
    // the nested matmul in the reference implementation. For larger shapes
    // it is slower, as the cell is parallelized over M blocks only.
    // TODO: Improve GRU / AUGRU coverage in BRGEMM-based implementation
#if DNNL_X64
    const dim_t max_c = nstl::max(rnn_.slc, nstl::max(rnn_.sic, rnn_.dhc));
    const bool is_small_f32_avx2 = !mayiuse(avx512_core)
            && ((rnn_.mb <= 4 && max_c <= 128) || max_c <= 64);
#else
    const bool is_small_f32_avx2 = false;
#endif
    VDISPATCH_RNN(IMPLICATION(rnn_.is_orig_gru,
                          this->desc()->prop_kind == forward_inference
                                  && IMPLICATION(rnn_.is_cell_dt_f32(),
                                          is_small_f32_avx2)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "gru/augru cell in brgemm-based forward inference");

    VDISPATCH_RNN(!(rnn_.is_cell_dt_f32()
                          && utils::one_of(this->desc()->prop_kind, backward,
                                  forward_training)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "f32 datatype in brgemm-based implementation");

    VDISPATCH_RNN((IMPLICATION((cell_kind == alg_kind::vanilla_lstm
                                       && rnn_.is_lstm_projection),
                          this->desc()->prop_kind == forward_inference)),
            "bad algorithm for lstm projection for forward inference");

#if DNNL_AARCH64
    VDISPATCH_RNN(mayiuse(asimd), VERBOSE_ISA_DT_MISMATCH);
#endif

    if (rnn_.is_bf16_conf()) {
        const bool isa_dt_not_ok = (
#if DNNL_X64
                !mayiuse(avx512_core_bf16) ||
#else
                !mayiuse_bf16() ||
#endif
                !utils::one_of(rnn_.bias_dt, data_type::bf16, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                        data_type::bf16, data_type::f32));
        VDISPATCH_RNN(!isa_dt_not_ok, VERBOSE_ISA_DT_MISMATCH);
    } else if (rnn_.is_f16_conf()) {
        const bool isa_dt_not_ok = (
#if DNNL_X64
                !mayiuse(avx512_core_amx_fp16) ||
#endif
                !utils::one_of(rnn_.bias_dt, data_type::f16, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                        data_type::f16, data_type::f32));
        VDISPATCH_RNN(!isa_dt_not_ok, VERBOSE_ISA_DT_MISMATCH);
    } else {
        const bool dt_not_ok = (rnn_.bias_dt != data_type::f32
                || !utils::one_of(
                        rnn_.src_iter_c_dt, data_type::undef, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt);
        VDISPATCH_RNN(!dt_not_ok, VERBOSE_UNSUPPORTED_DT_CFG);
    }
#if DNNL_X64
    const auto isa = get_max_cpu_isa();
    VDISPATCH_RNN(
            !(rnn_.is_signed_int8_conf() && !is_superset(isa, avx512_core_amx)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_RNN(!(rnn_.is_int8_conf() && !is_superset(isa, avx2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_RNN(!(rnn_.is_f32_conf() && !is_superset(isa, avx2)),
            VERBOSE_ISA_DT_MISMATCH);

    /* check that no shift have been passed to s8s8 amx lstm */
    VDISPATCH_RNN(IMPLICATION(rnn_.is_signed_int8_conf(),
                          this->attr()->rnn_data_qparams_.shift_ == 0),
            VERBOSE_UNSUPPORTED_FEATURE,
            "s8s8 amx lstm does not support shift");
#endif

    /* INT8 cases with non-trivial strides are not supported */
    VDISPATCH_RNN(!(rnn_.is_int8_conf()
                          && !(rnn_.src_layer_is_trivial_stride
                                  && rnn_.dst_layer_is_trivial_stride)),
            VERBOSE_NONTRIVIAL_STRIDE);

    /* check that only supported attr have been passed */
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8)
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_projection_qparams
#if DNNL_X64
                | primitive_attr_t::skip_mask_t::fpmath_mode
#endif
                ;
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    rnn_utils::set_conf(rnn_, *this->desc(), this->weights_md(0),
            this->weights_md(1), this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
            this->diff_weights_md(0), this->diff_weights_md(1),
            this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

    CHECK(ref_rnn_brgemm_t::configure_brgemm(rnn_, this->desc()->cell_kind,
            types::data_type_size(src_type),
            types::data_type_size(scratch_type)));

    // must be called after configure_brgemm()
    rnn_utils::set_workspace_sizes(rnn_, *this->desc(), acc_type, scratch_type);

#if DNNL_X64
    // Only AMX LSTM supports s8s8 now
    VDISPATCH_RNN(!(rnn_.is_signed_int8_conf() && !rnn_.is_cell_int8_amx()),
            VERBOSE_UNSUPPORTED_DT);
#endif

    // Set weights descriptors to desired format
    memory_desc_t new_weights_layer_md = *this->weights_md(0);
    CHECK(rnn_utils::set_expected_desc(
            rnn_, new_weights_layer_md, rnn_utils::weights_type_t::layer));
    if (this->weights_layer_md_.format_kind == format_kind::any) {
        this->weights_layer_md_ = new_weights_layer_md;
    } else {
        VDISPATCH_RNN(this->weights_layer_md_ == new_weights_layer_md,
                VERBOSE_INCONSISTENT_MDS, "weights_layer", "new_weights_layer");
    }

    memory_desc_t new_weights_iter_md = *this->weights_md(1);
    CHECK(rnn_utils::set_expected_desc(
            rnn_, new_weights_iter_md, rnn_utils::weights_type_t::iter));
    if (this->weights_iter_md_.format_kind == format_kind::any) {
        this->weights_iter_md_ = new_weights_iter_md;
    } else {
        VDISPATCH_RNN(this->weights_iter_md_ == new_weights_iter_md,
                VERBOSE_INCONSISTENT_MDS, "weights_iter", "new_weights_iter");
    }
    if (rnn_.is_lstm_projection) {
        memory_desc_t new_weights_projection_md
                = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
        CHECK(rnn_utils::set_expected_desc(rnn_, new_weights_projection_md,
                rnn_utils::weights_type_t::projection));
        if (this->weights_projection_md_.format_kind == format_kind::any) {
            this->weights_projection_md_ = new_weights_projection_md;
        } else {
            VDISPATCH_RNN(
                    this->weights_projection_md_ == new_weights_projection_md,
                    VERBOSE_INCONSISTENT_MDS, "weights_projection",
                    "new_weights_projection");
        }
    }
    if (rnn_.is_unsigned_int8_conf()) {
        const memory_desc_wrapper &weights_layer_d(this->weights_layer_md_);
        const memory_desc_wrapper &weights_iter_d(this->weights_iter_md_);
        const auto &pdims_l = weights_layer_d.padded_dims();
        const auto &pdims_i = weights_iter_d.padded_dims();
        rnn_.weights_layer_comp_offset = rnn_.n_layer * rnn_.n_dir
                * rnn_.n_gates * pdims_l[2] * pdims_l[4];
        rnn_.weights_iter_comp_offset = rnn_.n_layer * rnn_.n_dir * rnn_.n_gates
                * pdims_i[2] * pdims_i[4];
        if (rnn_.is_lstm_projection) {
            const memory_desc_wrapper &weights_proj_d(
                    this->weights_projection_md_);
            const auto &pdims_p = weights_proj_d.padded_dims();
            rnn_.weights_projection_comp_offset
                    = rnn_.n_layer * rnn_.n_dir * pdims_p[2] * pdims_p[3];
        } else {
            rnn_.weights_projection_comp_offset = 0;
        }
    }
    VDISPATCH_RNN(this->check_layout_consistency(true /*is_brgemm*/)
                    == status::success,
            "layout consistency check failed");
#if DNNL_X64
    if (rnn_.is_bf32()) {
        const memory_desc_wrapper weights_layer_d(this->weights_layer_md_);
        memory_desc_t weights_layer_md;
        const memory_desc_wrapper weights_iter_d(this->weights_iter_md_);
        memory_desc_t weights_iter_md;

        const auto bf16_tag = rnn_.n_block == 64 ? format_tag::ldgOI64o2i
                                                 : format_tag::ldgOI32o2i;
        CHECK(memory_desc_init_by_tag(weights_layer_md, weights_layer_d.ndims(),
                weights_layer_d.dims(), data_type::bf16, bf16_tag));
        CHECK(reorder_primitive_desc_create(bf32_wei_layer_reorder_pd_, engine,
                weights_layer_d.md_, &weights_layer_md, nullptr));

        CHECK(memory_desc_init_by_tag(weights_iter_md, weights_iter_d.ndims(),
                weights_iter_d.dims(), data_type::bf16, bf16_tag));
        CHECK(reorder_primitive_desc_create(bf32_wei_iter_reorder_pd_, engine,
                weights_iter_d.md_, &weights_iter_md, nullptr));
    }
#else
    VDISPATCH_RNN(!rnn_.is_bf32(), VERBOSE_UNSUPPORTED_FEATURE,
            "bf32 is not supported in aarch64 brgemm rnn as No bf32 reorder on "
            "aarch64.");
#endif

    return status::success;
#else
    return status::unimplemented;
#endif
}

template <prop_kind_t aprop>
status_t ref_rnn_common_pd_t<aprop>::init(const engine_t *engine) {
    using namespace data_type;
    struct dts_t {
        data_type_t src, weights, acc;
    };
    const bool is_fwd = aprop == prop_kind::forward;
    // The order defines the preference. bf16 goes first for f32 forward
    // since it may serve the f32 descriptor via bf32 fpmath mode.
    std::vector<dts_t> candidates;
    switch (this->desc()->src_layer_desc.data_type) {
        case f32:
            if (is_fwd)
                candidates = {{bf16, bf16, f32}, {f32, f32, f32}};
            else
                candidates = {{f32, f32, f32}, {bf16, bf16, f32}};
            break;
        case bf16: candidates = {{bf16, bf16, f32}}; break;
        case f16: candidates = {{f16, f16, f32}}; break;
        case u8:
            if (is_fwd) candidates = {{u8, s8, s32}};
            break;
        case s8:
            if (is_fwd) candidates = {{s8, s8, s32}};
            break;
        default: break;
    }
    VDISPATCH_RNN(!candidates.empty(), VERBOSE_UNSUPPORTED_DT);

    // A failed attempt may leave weights formats and nested descriptors
    // initialized for its data types, hence restore them for the next one.
    const memory_desc_t weights_layer_md = this->weights_layer_md_;
    const memory_desc_t weights_iter_md = this->weights_iter_md_;
    const memory_desc_t weights_projection_md = this->weights_projection_md_;

    status_t st = status::unimplemented;
    for (const auto &c : candidates) {
        this->weights_layer_md_ = weights_layer_md;
        this->weights_iter_md_ = weights_iter_md;
        this->weights_projection_md_ = weights_projection_md;
        for (auto *n_pd : {&matmul_layer_1_pd_, &matmul_layer_2_pd_,
                     &matmul_layer_3_pd_, &matmul_iter_1_pd_,
                     &matmul_iter_2_pd_, &matmul_iter_3_pd_,
                     &matmul_part2_1_pd_, &matmul_part2_2_pd_,
                     &matmul_part2_3_pd_, &matmul_part2_4_pd_})
            n_pd->reset();
#if DNNL_X64
        bf32_wei_layer_reorder_pd_.reset();
        bf32_wei_iter_reorder_pd_.reset();
#endif
        st = init(engine, c.src, c.weights, c.acc);
        if (st == status::success) break;
    }
    return st;
}

template <prop_kind_t aprop>
status_t ref_rnn_common_pd_t<aprop>::init(const engine_t *engine,
        data_type_t src_type, data_type_t weights_type, data_type_t acc_type) {
    status_t st = init_brgemm(engine, src_type, weights_type, acc_type);
    if (st != status::success) {
        rnn_.is_brgemm = false;
        st = init_ref(engine, src_type, weights_type, acc_type);
    }
    if (st == status::success) {
        size_t scratchpad_sz {0}, ws_sz {0};
        rnn_utils::get_scratchpad_and_workspace_sizes(
                rnn_, scratchpad_sz, ws_sz);

        init_scratchpad(scratchpad_sz, acc_type);
        // initialize the workspace if needed
        if (rnn_.is_training) {
            dims_t ws_dims = {(dim_t)ws_sz};
            CHECK(memory_desc_init_by_tag(
                    this->ws_md_, 1, ws_dims, data_type::u8, format_tag::x));
        }
        rnn_.cell_kind = this->desc()->cell_kind;
    }
    return st;
}

template <prop_kind_t aprop>
void ref_rnn_common_pd_t<aprop>::init_scratchpad(
        size_t scratchpad_sz, data_type_t acc_type) {
    using namespace memory_tracking::names;
    auto scratchpad = this->scratchpad_registry().registrar();

    {
        static constexpr size_t data_size
                = 1; // "true" data size already incorporated
        static constexpr size_t data_align
                = alignof(float); // "worst" case scenario
        static constexpr size_t perf_align = 4096;
        scratchpad.book(memory_tracking::names::key_rnn_space, scratchpad_sz,
                data_size, data_align, perf_align);
    }

    const int max_nparts
            = utils::one_of(this->cell_kind(), alg_kind::vanilla_gru,
                      alg_kind::vanilla_augru)
            ? 2
            : 1;
    const dim_t ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
    scratchpad.template book<float *>(
            memory_tracking::names::key_rnn_ptrs_wei_layer, ptr_wei_sz);
    scratchpad.template book<float *>(
            memory_tracking::names::key_rnn_ptrs_wei_iter, ptr_wei_sz);
    scratchpad.template book<float *>(
            memory_tracking::names::key_rnn_ptrs_wei_projection, ptr_wei_sz);

    const auto bias_dt_size
            = types::data_type_size(this->arg_md(DNNL_ARG_BIAS)->data_type);
    scratchpad.template book<void *>(memory_tracking::names::key_rnn_ptrs_bia,
            ptr_wei_sz * bias_dt_size);

#if DNNL_X64 || DNNL_AARCH64
    if (rnn_.is_brgemm)
        ref_rnn_brgemm_t::init_scratchpad(rnn_, scratchpad,
                types::data_type_size(acc_type),
                types::data_type_size(acc_type));
#endif

    // Below primitives may be run as part of execution.Fortunately, none of
    // them run simulataneously. So, we can re-use the same scratchpad across
    // all primitives. Iterate through them to find the largest scratchpad
    // required.
    const auto nested_pds = {matmul_layer_1_pd_, matmul_layer_2_pd_,
            matmul_layer_3_pd_, matmul_iter_1_pd_, matmul_iter_2_pd_,
            matmul_iter_3_pd_, matmul_part2_1_pd_, matmul_part2_2_pd_,
            matmul_part2_3_pd_, matmul_part2_4_pd_,
#if DNNL_X64
            bf32_wei_layer_reorder_pd_, bf32_wei_iter_reorder_pd_
#endif
    };

    size_t max_nested_scratchpad_size = 0;
    for (const auto &n_pd : nested_pds) {
        if (n_pd)
            max_nested_scratchpad_size = nstl::max(max_nested_scratchpad_size,
                    n_pd->scratchpad_registry().size());
    }

    scratchpad.template book<void *>(
            memory_tracking::names::key_nested_multiple + 0,
            max_nested_scratchpad_size);
}

template struct ref_rnn_common_pd_t<prop_kind::forward>;
template struct ref_rnn_common_pd_t<prop_kind::backward>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
