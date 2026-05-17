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

#ifndef GPU_INTEL_GEMM_JIT_JIT_GEMM_PD_HPP
#define GPU_INTEL_GEMM_JIT_JIT_GEMM_PD_HPP

#include <climits>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_attr_quant.hpp"
#include "gemmstone/problem.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

#define VDISPATCH_JIT_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_JIT_GEMM_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, matmul, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

#define GEMM_MAX_PO 36

// Shared base for JIT GEMM impls registered against matmul.
//
// Design rule (doctrine, decided 2026-05-18; tightened 2026-05-20): the pd
// stores matmul-convention state only (un-mutated src_md_/weights_md_/
// dst_md_/bias_md_, matmul-keyed attr_) plus the swap_ab_ flag. Accessor
// naming convention:
//   * `M()`, `N()`, `K()`, `src_md_`, `weights_md_`, `dst_md_`, `c_type()`,
//     `stride_c()`, `with_bias()`, etc. — matmul-natural; never depend on
//     swap_ab_. `with_bias()` is inherited from matmul_pd_t unchanged.
//   * `gemm_m()`, `gemm_n()`, `gemm_k()`, `gemm_lda()`, `gemm_ldb()`,
//     `gemm_transa()`, `gemm_transb()`, `gemm_a_type()`, `gemm_b_type()`,
//     `gemm_stride_a()`, `gemm_stride_b()`, `gemm_bias_mask()`,
//     `gemm_bias_cmask()`, `gemm_sum_ab()`, `gemm_sum_ab_cmask()` —
//     kernel-view (BLAS convention, post-swap); compute on demand from
//     un-mutated mds + swap_ab_. These are the ONLY swap-aware accessors.
// `init_GEMMProblem` reads matmul-natural state and writes directly into a
// GEMMProblem; if swap_ab_ is set, the problem is GEMMProblem::transpose()'d
// once at the end. Post-init, kernel-side code reads runtime shape from the
// pd via the `gemm_*` accessors and compile-time catalog state from the
// stowed problem. NO pd-side shadow of attr/quant-state survives past
// init_GEMMProblem.
struct jit_gemm_pd_t : public gpu::gpu_matmul_pd_t {
    using gpu::gpu_matmul_pd_t::gpu_matmul_pd_t;

    // Kernel-internal A/B/C identifiers. Match matmul arg keys; since attr_ is
    // never re-keyed, attr().X.get(kA) always returns the matmul-SRC entry,
    // attr().X.get(kB) the matmul-WEIGHTS entry, attr().X.get(kC) the DST one.
    static constexpr int kA = DNNL_ARG_SRC;
    static constexpr int kB = DNNL_ARG_WEIGHTS;
    static constexpr int kC = DNNL_ARG_DST;

    // Quant-mask bit layout in M/N/K terms. Bit 0 is the per-tensor scalar
    // bit; bits 1 and 2 are the inner-two-dim bits whose semantic axis depends
    // on the operand: A (kA = SRC) is M×K, B (kB = WEIGHTS) is K×N, C (kC =
    // DST) is M×N. Same underlying constants, operand-specific names.
    static constexpr int mask_scalar = 1 << 0;
    static constexpr int mask_a_m = 1 << 1;
    static constexpr int mask_a_k = 1 << 2;
    static constexpr int mask_b_k = 1 << 1;
    static constexpr int mask_b_n = 1 << 2;
    static constexpr int mask_c_m = 1 << 1;
    static constexpr int mask_c_n = 1 << 2;

    struct binary_src_t {
        enum type_t { none, scales, bias, binary, prelu } type;
        // For type==scales: matmul attr key (DNNL_ARG_SRC/WEIGHTS/DST). The
        // exec path looks up `DNNL_ARG_ATTR_SCALES | index` directly — no
        // swap routing. For type==binary/prelu: matmul post-op index. For
        // other types: unused.
        int index;

        binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
    };

    static constexpr post_op::specializations_t get_post_op_specializations() {
        using mode_t = post_op::specializations_t::inline_mode_t;
        using sum_t = post_op::specializations_t::sum_t;
        // The sum scale is handled as GEMM beta argument.
        return {{}, sum_t(mode_t::impl_managed(), {}), {}};
    }

    static constexpr bool supported_binary_op(alg_kind_t alg) {
        using namespace alg_kind;
        return utils::one_of(alg, binary_add, binary_sub, binary_mul,
                binary_div, binary_min, binary_max);
    }

    // ----- Init helpers (call from derived impl::init in this order). -----
    // gemm_-prefixed to avoid shadowing matmul_pd_t::set_default_formats(); the
    // jit version adds bias_md / reduce_md handling on top of the base behavior.
    bool gemm_set_default_formats();
    status_t canonicalize_post_ops();
    status_t apply_swap_ab(); // Flag flip + canonicalize only — no md/attr mutation.
    void init_quant_mds();

    // Broadcast-batch collapse to 2D (when weights batch is fully 1) or 3D
    // (when ndims > 3). Mutates src_md_/weights_md_/dst_md_/bias_md_ and
    // attr_.scales_/zero_points_/precomputed_reductions_/post_ops_ in place.
    // Snapshots the originals into orig_*_md_ so commit_reshape() can later
    // re-project resolved tags back to the user N-D shape. Idempotent; safe
    // no-op when inapplicable or when GEMM_ALLOW_RESHAPE is disabled.
    status_t maybe_reshape_2d();

    // Finalize: re-project resolved 2D/3D tags onto orig_*_md_ at the user
    // ndims via memory_desc_reshape, then set reshape_applied_=true. After
    // this point the external src_md/weights_md/dst_md overrides return the
    // shadow N-D mds while internal field reads of src_md_/weights_md_/
    // dst_md_/bias_md_ continue to see the active 2D/3D state.
    status_t commit_reshape();

    // Kernel-view accessor for the active dst md. Always returns &dst_md_
    // regardless of reshape_applied_, so xe_hp_systolic's init_compute can
    // build gpu_post_ops_t against the 2D ndim_normalizer reference.
    const memory_desc_t *kernel_dst_md() const { return &dst_md_; }

    status_t init_post_ops(impl::engine_t *engine);
    status_t init_attrs(impl::engine_t *engine);
    status_t scales_ok(impl::engine_t *engine);
    status_t zp_ok(impl::engine_t *engine);
    status_t gs_ok(impl::engine_t *engine);

    status_t init_GEMMProblem(gemmstone::GEMMProblem &problem,
            const intel::engine_t *engine) const;

    // Cache kernel-view LDA/LDB/transa/transb from un-mutated matmul mds.
    // Only gen_t uses these fields (for the skinny-N flip + ld pad in
    // gen_t::pd_t::init). Must NOT be called from impls that resolve WEIGHTS
    // to a packed format before init (e.g. xe_hp_systolic), because get_ld
    // asserts on packed mds — and those impls don't need the cache anyway
    // (xe_hp_systolic has its own lda_packed/ldb_packed/ldc_packed).
    void init_kernel_lds() {
        gemm_lda_ = compute_gemm_lda();
        gemm_ldb_ = compute_gemm_ldb();
        gemm_transa_ = (compute_gemm_transa() == transpose::trans);
        gemm_transb_ = (compute_gemm_transb() == transpose::trans);
    }

    status_t init(impl::engine_t *engine, compute::gpu_arch_t arch) {
        arch_ = arch;

        CHECK(init_attrs(engine));
        CHECK(scales_ok(engine));
        CHECK(zp_ok(engine));
        CHECK(gs_ok(engine));
        CHECK(init_post_ops(engine));
        return status::success;
    }

    bool dy_quant_enabled() const;
    bool wei_decomp() const;
    bool quant_enabled() const;

    bool valid_2d_mask(int mask, int ndims, bool per_tensor_ok = true);

    static int quant_entry_ndims(
            const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx);

    // ----- Accessors. Swap-aware kernel-view accessors carry the gemm_ prefix
    //       (BLAS convention, post-swap). The non-prefixed M/N/K/ldc/stride_c/
    //       c_type/etc. are matmul-natural and never depend on swap_ab_. -----
    bool swap_ab() const { return swap_ab_; }

    dim_t gemm_m() const { return swap_ab_ ? N() : M(); }
    dim_t gemm_n() const { return swap_ab_ ? M() : N(); }
    dim_t gemm_k() const { return K(); }

    int batch_dims() const { return nstl::max(ndims() - 2, 0); }
    bool is_batched() const { return ndims() > 2; }
    dim_t batch() const { return matmul_pd_t::batch(); }

    // gemm_lda()/gemm_ldb() return the kernel-tuned cache when populated
    // (gen_t's init_kernel_lds + skinny-N flip + pad-to-16; xe_hp_systolic's
    // init sets these for unpacked sides too) and fall back to on-demand
    // compute_gemm_lda/b otherwise. The fallback covers any future impl that
    // forgets to populate the cache — review.md #6 trap: the previous
    // unconditional `return lda_` silently returned 0 (default) when not
    // populated. gemm_ldc() computes on demand unconditionally (no impl needs
    // a tuned-ldc cache; matmul-DST is the kernel-C always). Named with the
    // gemm_ prefix to make the kernel-view semantics explicit and avoid
    // shadowing matmul_pd_t::ldc() (which reads strides[ndims-2] directly).
    dim_t gemm_lda() const {
        return gemm_lda_ != 0 ? gemm_lda_ : compute_gemm_lda();
    }
    dim_t gemm_ldb() const {
        return gemm_ldb_ != 0 ? gemm_ldb_ : compute_gemm_ldb();
    }
    dim_t gemm_ldc() const { return compute_ldc(); }
    dim_t ld_bias() const { return with_bias() ? compute_ld_bias() : 0; }

    // Batch-dim strides. Only valid for dim < ndims-2 (M/N are not batch).
    dim_t gemm_stride_a(int dim = 0) const {
        return stride_for(gemm_a_md(), dim);
    }
    dim_t gemm_stride_b(int dim = 0) const {
        return stride_for(gemm_b_md(), dim);
    }
    dim_t stride_c(int dim = 0) const { return stride_for(dst_md_, dim); }

    data_type_t gemm_a_type() const { return gemm_a_md().data_type; }
    data_type_t gemm_b_type() const { return gemm_b_md().data_type; }
    data_type_t c_type() const { return dst_md_.data_type; }
    data_type_t acc_type() const { return desc()->accum_data_type; }

    // DOCTRINE: matmul-driven bias is ALWAYS routed through binary post-op.
    // The legacy cOffset=Pre / co_t bias-as-CO path is gone — init_post_ops
    // prepends bias as a binary_add post-op and the kernel-side CO channel
    // carries c-zero-points / sum_ab only. `with_bias()` resolves to the
    // inherited `matmul_pd_t::with_bias()` (true iff a bias md was supplied).
    data_type_t bias_type() const {
        return with_bias() ? bias_md_.data_type : data_type::undef;
    }

    int gemm_bias_mask() const;
    int gemm_bias_cmask() const {
        unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        const int bm = gemm_bias_mask();
        assert(unsigned(bm) < 8);
        return with_bias() ? to_cmask[bm & 7] : -1;
    }

    transpose_t gemm_transa() const;
    transpose_t gemm_transb() const;
    transpose_t trans_bias() const;

    bool gemm_trans_a() const { return gemm_transa_; }
    bool gemm_trans_b() const { return gemm_transb_; }

    // Reduce slot subsumes gemm's sum_ab.
    sum_ab_t gemm_sum_ab() const;
    bool with_sum_ab() const { return gemm_sum_ab() != sum_ab::sum_none; }
    data_type_t sum_ab_type() const {
        return with_reduce() ? reduce_md(0)->data_type : data_type::undef;
    }
    int gemm_sum_ab_cmask() const {
        switch (gemm_sum_ab()) {
            default:
            case sum_ab::sum_none: return 0;
            case sum_ab::sum_a_row: return 1;
            case sum_ab::sum_b_col: return 2;
        }
    }

    // ----- Quant entry accessors (matmul-keyed; attr_ is never re-keyed). -----
    bool with_a_scales() const { return a_scale_ndims() >= 0; }
    bool with_b_scales() const { return b_scale_ndims() >= 0; }
    bool with_c_scales() const {
        return !attr()->scales_.has_default_values(kC);
    }

    bool with_a_zero_points() const { return a_zp_ndims() >= 0; }
    bool with_b_zero_points() const { return b_zp_ndims() >= 0; }
    bool with_c_zero_points() const {
        return !attr()->zero_points_.has_default_values(kC);
    }

    bool with_a_group_sums() const { return a_gs_ndims() >= 0; }
    bool with_b_group_sums() const { return b_gs_ndims() >= 0; }

    bool with_sround() const {
        return attr()->rounding_mode_.get(DNNL_ARG_DST)
                == rounding_mode::stochastic;
    }
    bool with_mx_scale() const {
        const auto &c_scales = attr()->scales_.get(kC);
        return !c_scales.has_default_groups() && c_scales.is_mx();
    }

    bool a_scales_2d() const { return a_scale_ndims() > 1; }
    bool b_scales_2d() const { return b_scale_ndims() > 1; }
    bool c_scales_2d() const { return c_scale_ndims() > 1; }

    bool a_zp_2d() const { return a_zp_ndims() >= 2; }
    bool b_zp_2d() const { return b_zp_ndims() >= 2; }
    bool a_gs_2d() const { return a_gs_ndims() >= 2; }
    bool b_gs_2d() const { return b_gs_ndims() >= 2; }

    data_type_t a_scale_dt() const {
        return attr()->scales_.get(kA).get_data_type();
    }
    data_type_t b_scale_dt() const {
        return attr()->scales_.get(kB).get_data_type();
    }
    data_type_t c_scale_dt() const {
        return attr()->scales_.get(kC).get_data_type();
    }
    data_type_t a_zp_dt() const {
        return attr()->zero_points_.get(kA).get_data_type();
    }
    data_type_t b_zp_dt() const {
        return attr()->zero_points_.get(kB).get_data_type();
    }
    data_type_t c_zp_dt() const {
        return attr()->zero_points_.get(kC).get_data_type();
    }
    data_type_t a_gs_dt() const {
        return attr()->precomputed_reductions_.get(kA).get_data_type();
    }
    data_type_t b_gs_dt() const {
        return attr()->precomputed_reductions_.get(kB).get_data_type();
    }

    int a_zp_ndims() const {
        return quant_entry_ndims(
                attr()->zero_points_.get(kA), src_zp_md_, ndims() - 1);
    }
    int b_zp_ndims() const {
        return quant_entry_ndims(
                attr()->zero_points_.get(kB), wei_zp_md_, ndims() - 2);
    }
    int c_zp_ndims() const {
        return quant_entry_ndims(
                attr()->zero_points_.get(kC), dst_zp_md_, -1);
    }
    int a_gs_ndims() const {
        return quant_entry_ndims(attr()->precomputed_reductions_.get(kA),
                src_gs_md_, ndims() - 1);
    }
    int b_gs_ndims() const {
        return quant_entry_ndims(attr()->precomputed_reductions_.get(kB),
                wei_gs_md_, ndims() - 2);
    }
    int a_scale_ndims() const {
        if (a_scale_ndims_override_ != INT_MIN) return a_scale_ndims_override_;
        return quant_entry_ndims(
                attr()->scales_.get(kA), src_scale_md_, ndims() - 1);
    }
    int b_scale_ndims() const {
        if (b_scale_ndims_override_ != INT_MIN) return b_scale_ndims_override_;
        return quant_entry_ndims(
                attr()->scales_.get(kB), wei_scale_md_, ndims() - 2);
    }
    int c_scale_ndims() const {
        return quant_entry_ndims(attr()->scales_.get(kC), dst_scale_md_, -1);
    }

    bool a_force_gs() const {
        return !attr()->precomputed_reductions_.has_default_values(kA);
    }
    bool b_force_gs() const {
        return !attr()->precomputed_reductions_.has_default_values(kB);
    }

    bool zp_host_scalar(int arg) const {
        return attr()->zero_points_.get(arg).is_host_scalar();
    }
    bool a_zp_host_scalar() const { return zp_host_scalar(kA); }
    bool b_zp_host_scalar() const { return zp_host_scalar(kB); }
    bool c_zp_host_scalar() const { return zp_host_scalar(kC); }

    // ----- Per-arg uniform accessors. arg in {kA, kB, kC}; kernel-view. -----
    dim_t gemm_ld(int arg) const {
        if (arg == kA) return gemm_lda();
        if (arg == kB) return gemm_ldb();
        if (arg == kC) return gemm_ldc();
        gpu_error_not_expected();
        return 0;
    }
    dim_t gemm_stride(int arg, int dim) const {
        if (arg == kA) return gemm_stride_a(dim);
        if (arg == kB) return gemm_stride_b(dim);
        if (arg == kC) return stride_c(dim);
        gpu_error_not_expected();
        return 0;
    }
    data_type_t gemm_get_type(int arg) const {
        if (arg == kA) return gemm_a_type();
        if (arg == kB) return gemm_b_type();
        if (arg == kC) return c_type();
        gpu_error_not_expected();
        return data_type::undef;
    }
    int gemm_align(int arg) const {
        auto dt = gemm_get_type(arg);
        auto align = utils::max_pow2_div(
                types::elements_to_bytes(dt, gemm_ld(arg)));
        for (int b = 0; b < batch_dims(); b++) {
            auto stride_bytes = utils::max_pow2_div(
                    types::elements_to_bytes(dt, gemm_stride(arg, b)));
            align = (stride_bytes ? nstl::min(align, stride_bytes) : align);
        }
        return int(align);
    }
    // Matmul-natural group probe per matmul-side arg key (kA=SRC, kB=WEIGHTS,
    // kC=DST). The OR-of-(kA,kB) queries downstream are swap-invariant, so
    // there's no swap-aware variant.
    bool grouped(int matmul_arg) const;
    bool a_grouped() const { return grouped(kA); }
    bool b_grouped() const { return grouped(kB); }

    // ----- Alpha/beta. alpha != 1.0 only for host-scalar scales (real value
    //       resolved at execute time; the 9.99 sentinel keeps catalog
    //       lookup honest). -----
    float alpha() const {
        const auto &scales = attr()->scales_;
        bool host_scales_by_alpha = scales.get(kA).is_host_scalar()
                || scales.get(kB).is_host_scalar()
                || (scales.get(kC).is_host_scalar()
                        && attr()->post_ops_.len() == 0
                        && !with_bias());
        if (host_scales_by_alpha) return 9.99f;
        return 1.0f;
    }

    // beta == the sum-post-op scale (0.0f when no sum is present). with_sum()
    // / sum_at_begin() derive from the final canonicalized post_ops_ (note
    // that bias-via-binary prepends a binary_add at index 0; if the user's
    // sum was originally at index 0 it ends up at index 1 post-prepend, so
    // sum_at_begin() correctly reports false — the kernel needs higher
    // accumulator precision when anything precedes the sum). All three are
    // derived rather than stored to satisfy the doctrine "no pd-side
    // shadow members for post_ops_-derivable state".
    // NOTE: is_sum() with default args requires scale==1.0 and zp==0 — too
    // restrictive for our purposes. Match the kind directly so a sum with
    // arbitrary scale (which is exactly what beta() must track) is detected.
    bool with_sum() const {
        for (int i = 0; i < post_ops_.len(); ++i)
            if (post_ops_.entry_[i].kind == primitive_kind::sum) return true;
        return false;
    }
    bool sum_at_begin() const {
        return post_ops_.len() > 0
                && post_ops_.entry_[0].kind == primitive_kind::sum;
    }
    float beta() const {
        for (int i = 0; i < post_ops_.len(); ++i)
            if (post_ops_.entry_[i].kind == primitive_kind::sum)
                return post_ops_.entry_[i].sum.scale;
        return 0.0f;
    }

    // ----- Binary / scale / zp / gs strides (per post-op index or per arg). -----
    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;
    dim_t scale_stride(int idx, int arg) const;
    dim_t zp_stride(int idx, int arg) const;
    dim_t gs_stride(int idx, int arg) const;

    const post_ops_t *post_ops() const { return &post_ops_; }
    const std::vector<binary_src_t> &binary_srcs() const {
        return binary_srcs_;
    }

    // True iff any post-op contributes work beyond a scalar scale: eltwise,
    // user-binary, prelu (canonicalized to binary), or bias (via binary).
    // Scale-derived binaries (binary_srcs_[i].type == scales) do NOT count
    // — they're a representation choice for what is semantically a scale.
    // Sum entries also don't count (they're handled via problem.beta and
    // a separate gate `with_sum() && with_c_scales()`).
    bool non_scale_po() const {
        for (int i = 0; i < post_ops_.len(); ++i) {
            const auto &e = post_ops_.entry_[i];
            if (e.is_eltwise()) return true;
            if (e.is_binary()) {
                auto t = binary_srcs_[i].type;
                if (t == binary_src_t::binary || t == binary_src_t::bias
                        || t == binary_src_t::prelu)
                    return true;
            }
        }
        return false;
    }

    // ----- Quant memory descriptors (matmul-keyed: src=A, wei=B, dst=C). -----
    const memory_desc_t &a_scale_md() const { return src_scale_md_; }
    const memory_desc_t &b_scale_md() const { return wei_scale_md_; }
    const memory_desc_t &c_scale_md() const { return dst_scale_md_; }
    const memory_desc_t &a_zp_md() const { return src_zp_md_; }
    const memory_desc_t &b_zp_md() const { return wei_zp_md_; }
    const memory_desc_t &c_zp_md() const { return dst_zp_md_; }
    const memory_desc_t &a_gs_md() const { return src_gs_md_; }
    const memory_desc_t &b_gs_md() const { return wei_gs_md_; }

    // ----- External md getters. When maybe_reshape_2d() has collapsed the
    //       active mds to 2D/3D, the user-facing query surface (framework
    //       arg binding, with_post_ops adoption, etc.) needs the original
    //       N-D shape. The shadow orig_*_md_ are populated by commit_reshape
    //       with the resolved tags at user ndims. Internal reads of
    //       src_md_/weights_md_/dst_md_/bias_md_ field references continue
    //       to return the kernel-view 2D/3D state. -----
    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (reshape_applied_ && !user_input && index == 0)
            return &orig_src_md_;
        return matmul_pd_t::src_md(index, user_input);
    }
    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (reshape_applied_ && !user_input) {
            if (index == 0) return &orig_weights_md_;
            if (index == 1) return &orig_bias_md_;
        }
        return matmul_pd_t::weights_md(index, user_input);
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (reshape_applied_ && !user_input && index == 0)
            return &orig_dst_md_;
        return matmul_pd_t::dst_md(index, user_input);
    }

    // ----- Pd state (stable across swap or used only at init time). -----
    bool swap_ab_ = false;

    // Shadow mds carrying the user-facing N-D shape with resolved tags. See
    // maybe_reshape_2d / commit_reshape. Empty (ndims==0) when reshape is not
    // applied or when reshape rejected via bcast_ok.
    memory_desc_t orig_src_md_ {};
    memory_desc_t orig_weights_md_ {};
    memory_desc_t orig_dst_md_ {};
    memory_desc_t orig_bias_md_ {};
    bool reshape_applied_ = false;

    // a/b_scale_ndims_override_: INT_MIN sentinel = override unset. The
    // override is set to -1 after a 1D scale is converted to a binary
    // post-op, so that with_a_scales() / with_b_scales() / a_scale_ndims()
    // see "no scale" post-conversion (the scale buffer now flows via
    // binary_srcs_[i] with type == scales, not via attr_.scales_).
    //
    // TODO (review.md FIX #2): doctrine-clean alternatives are (a) derive
    // post-conversion state from binary_srcs_ (scan for entry of type
    // scales && index == kA/kB) instead of an int sentinel, or (b) clear
    // the attr_.scales_ entry on conversion. (a) is cheap but the readers
    // are on the hot init path; (b) risks framework binding side effects
    // since attr_ keys the user's `DNNL_ARG_ATTR_SCALES | arg` lookup at
    // exec time. Leaving in place for this pass.
    int a_scale_ndims_override_ = INT_MIN;
    int b_scale_ndims_override_ = INT_MIN;

    post_ops_t post_ops_;
    std::vector<binary_src_t> binary_srcs_;

    memory_desc_t prelu_wei_md = {};

    // Kernel-view (post-swap) tuned leading dims and trans-A/B; may differ
    // from natural ld/trans (e.g. padded for skinny K). Set by base ::init()
    // from un-mutated matmul mds + swap_ab_; derived ::init() may override.
    dim_t gemm_lda_ = 0, gemm_ldb_ = 0;
    bool gemm_transa_ = false;
    bool gemm_transb_ = false;
    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

    // Natural leading-dim / transpose helpers — exact mirror of
    // gemm_desc_t::get_ld / get_trans on a matmul md.
    static transpose_t get_trans(const memory_desc_t &md);
    static dim_t get_ld(const memory_desc_t &md);
    static dim_t get_stride(const memory_desc_t &md, int idx) {
        if (idx >= md.ndims - 2 || md.dims[idx] == 1) return 0;
        return md.format_desc.blocking.strides[idx];
    }

    // Swap trailing M/N axes of an md in place (matmul→kernel view).
    static void transpose_mn_axes(memory_desc_t &md);

protected:
    // Matmul-keyed quant memory descriptors. Populated by init_quant_mds()
    // from matmul attrs keyed by DNNL_ARG_SRC/WEIGHTS/DST. Never re-keyed
    // under swap_ab_; the swap is applied inside init_GEMMProblem.
    memory_desc_t src_scale_md_ = {}, wei_scale_md_ = {}, dst_scale_md_ = {};
    memory_desc_t src_zp_md_ = {}, wei_zp_md_ = {}, dst_zp_md_ = {};
    memory_desc_t src_gs_md_ = {}, wei_gs_md_ = {};

private:
    dim_t stride_for(const memory_desc_t &md, int dim) const {
        if (dim >= md.ndims - 2 || md.dims[dim] == 1) return 0;
        return md.format_desc.blocking.strides[dim];
    }
    static int mask_to_gemm(int mask, int nd);
    static int swap_mask_bits(int mask, int i, int j);

    // The matmul md that plays the kernel-A role: matmul WEIGHTS under swap,
    // matmul SRC otherwise. Likewise for kernel-B. Used only by the
    // compute_gemm_* helpers below to derive kernel-view ld/trans/type/stride.
    const memory_desc_t &gemm_a_md() const {
        return swap_ab_ ? weights_md_ : src_md_;
    }
    const memory_desc_t &gemm_b_md() const {
        return swap_ab_ ? src_md_ : weights_md_;
    }

    // Kernel-view leading-dim / transpose computations. base's gemm path
    // used these directly on the matmul md (with the matmul-natural axis
    // ordering); the post-init compute_gemm_*() helpers mirror that. The
    // previous implementation mn-transposed the md under swap_ab_ and then
    // NEGATEd get_trans — which is mathematically equivalent for
    // non-degenerate matrices but diverges for the [m, 1] / [1, n] / K=1
    // single-axis cases (mn-transpose is a no-op semantically, but flips
    // which get_ld / get_trans branch fires). For matmul K=1 with compact
    // SRC strides this produced ldb=N/trans=T where base had ldb=1/trans=N,
    // and the resulting pad branch corrupted output. Reading the matmul md
    // un-mutated keeps the behavior identical to base for both cases.
    dim_t compute_gemm_lda() const { return get_ld(gemm_a_md()); }
    dim_t compute_gemm_ldb() const { return get_ld(gemm_b_md()); }
    dim_t compute_ldc() const { return get_ld(dst_md_); }
    // Bias helpers MUST read the active bias_md_ field, not the virtual
    // weights_md(1) — under reshape_applied_ the latter returns the N-D
    // shadow used for framework arg binding, not the 2D kernel view.
    dim_t compute_ld_bias() const {
        return bias_md_.ndims >= 2 ? get_ld(bias_md_) : 1;
    }
    transpose_t compute_gemm_transa() const { return get_trans(gemm_a_md()); }
    transpose_t compute_gemm_transb() const { return get_trans(gemm_b_md()); }
    transpose_t compute_trans_bias() const {
        if (!with_bias()) return transpose::notrans;
        return get_trans(bias_md_);
    }
};

// Build gemmstone post-ops from gpu_post_ops_t (moved out of jit::pd_t in the
// previous gemm-base; kept as a free function here).
status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops);

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
