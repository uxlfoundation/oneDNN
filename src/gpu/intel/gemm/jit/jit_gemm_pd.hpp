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
// Design rule (doctrine, decided 2026-05-18; tightened 2026-05-20; further
// tightened 2026-05-20): there are two namespaces:
//   * USER-facing — `attr()`, `desc()`, and the inherited matmul_pd_t mds
//     `src_md_/weights_md_/dst_md_/bias_md_`. These carry the user N-D
//     shape and remain un-mutated for shape/mask throughout init. At the
//     end of init they receive resolved format tags via commit_resolved_tags.
//     This is the surface the framework queries via `arg_md()`/`src_md()`.
//   * KERNEL-facing — `kernel_input_` (jit_gemm_input_t). Holds the 2D/3D
//     reshape-collapsed mds, reshape-rescaled quant entries (scales/zero
//     points/precomputed_reductions), the canonicalized post_ops_, and the
//     derived quant mds. This is the surface that init_GEMMProblem reads.
// Accessor naming convention:
//   * `M()`, `N()`, `K()`, `ndims()`, `batch()`, `c_type()`,
//     `stride_c()`, `with_bias()` — kernel-view; read from kernel_input_.
//   * `kernel_src_md()`, `kernel_weights_md()`, `kernel_dst_md()`,
//     `kernel_bias_md()`, `kernel_post_ops()` — explicit kernel-view
//     accessors for code that needs the underlying mds.
//   * `gemm_m()`, `gemm_n()`, `gemm_k()`, `gemm_lda()`, `gemm_ldb()`,
//     `gemm_transa()`, `gemm_transb()`, `gemm_a_type()`, `gemm_b_type()`,
//     `gemm_stride_a()`, `gemm_stride_b()`, `gemm_bias_mask()`,
//     `gemm_bias_cmask()`, `gemm_sum_ab()`, `gemm_sum_ab_cmask()` —
//     BLAS-view (post-swap); compute on demand from kernel_input_ mds +
//     swap_ab_. These are the ONLY swap-aware accessors.
// `init_GEMMProblem` reads kernel_input_ and writes directly into a
// GEMMProblem; if swap_ab_ is set, the problem is GEMMProblem::transpose()'d
// once at the end. Post-init, kernel-side code reads runtime shape from the
// pd via the `gemm_*` accessors and compile-time catalog state from the
// stowed problem. NO pd-side shadow of attr/quant-state outside of
// kernel_input_ survives past init.
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

    // Kernel-facing input view, populated by maybe_reshape_2d. Holds the
    // (possibly 2D/3D-collapsed) mds, reshape-rescaled quant entries, the
    // canonicalized post_ops_ (prelu→binary in init_post_ops), and the
    // derived quant mds. Always populated after maybe_reshape_2d, whether
    // or not collapse fires (no-collapse path copies user state as-is).
    struct jit_gemm_input_t {
        memory_desc_t src_md {};
        memory_desc_t weights_md {};
        memory_desc_t dst_md {};
        memory_desc_t bias_md {};

        // Reshape-rescaled attr copies. Framework does NOT leak these via
        // query() (only attr_ keys are looked up for buffer binding), so
        // we are free to keep canonical kernel-view copies here without
        // touching attr_.
        scales_t scales {};
        zero_points_t zero_points {};
        precomputed_reductions_t precomputed_reductions {};

        // Canonicalized (prelu→binary, bias-via-binary prepended) post-ops
        // with src1_desc built against kernel-view dst_md. attr_.post_ops_
        // stays in user/framework form for arg validation.
        post_ops_t post_ops {};
        std::vector<binary_src_t> binary_srcs {};
        memory_desc_t prelu_wei_md {};

        // Cached quant mds derived from the entries + the kernel-view src/
        // weights/dst mds. Built by init_quant_mds.
        memory_desc_t src_scale_md {}, wei_scale_md {}, dst_scale_md {};
        memory_desc_t src_zp_md {}, wei_zp_md {}, dst_zp_md {};
        memory_desc_t src_gs_md {}, wei_gs_md {};

        // True iff maybe_reshape_2d collapsed dims (ndims < user ndims).
        bool reshape_applied = false;
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
    // (when ndims > 3). Populates kernel_input_ with the reshaped state.
    // When collapse doesn't apply, copies the user-view mds/attr entries
    // into kernel_input_ unchanged. attr_ and the user-facing
    // src_md_/weights_md_/dst_md_/bias_md_ are NEVER mutated for shape or
    // mask — only commit_resolved_tags writes resolved format tags back.
    // Idempotent; safe no-op when called twice.
    status_t maybe_reshape_2d();

    // Finalize: project resolved format tags from kernel_input_.{src,
    // weights,dst,bias}_md back onto the user-facing src_md_/weights_md_/
    // dst_md_/bias_md_ at the user ndims via memory_desc_reshape. This is
    // how `format::any` gets surfaced to the framework — same direction as
    // base's set_default_params.
    status_t commit_resolved_tags();

    // Kernel-view accessors for the active mds. These return the
    // (possibly 2D/3D-collapsed) state that init_GEMMProblem consumes,
    // independently of what the external query surface returns. Use these
    // wherever code needs the kernel-view md (post_ops binding via
    // ndim_normalizer, batch-dim strides in execute, etc.).
    const memory_desc_t *kernel_src_md() const {
        return &kernel_input_.src_md;
    }
    const memory_desc_t *kernel_weights_md() const {
        return &kernel_input_.weights_md;
    }
    const memory_desc_t *kernel_dst_md() const {
        return &kernel_input_.dst_md;
    }
    const memory_desc_t *kernel_bias_md() const {
        return &kernel_input_.bias_md;
    }
    const post_ops_t *kernel_post_ops() const {
        return &kernel_input_.post_ops;
    }

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

    // Kernel-view M/N/K/ndims/batch — shadow matmul_pd_t's helpers so that
    // every read returns the (possibly 2D/3D-collapsed) kernel sizes. The
    // user-view sizes are recoverable via `desc()->{src,dst}_desc.dims`
    // or via the un-overridden inherited helpers (none of the JIT impls
    // need them).
    int ndims() const { return kernel_input_.dst_md.ndims; }
    dim_t M() const { return kernel_input_.dst_md.dims[ndims() - 2]; }
    dim_t N() const { return kernel_input_.dst_md.dims[ndims() - 1]; }
    dim_t K() const { return kernel_input_.src_md.dims[ndims() - 1]; }
    dim_t batch() const {
        return utils::array_product(kernel_input_.dst_md.dims, batch_dims());
    }
    bool with_bias() const { return kernel_input_.bias_md.ndims != 0; }
    bool batched() const { return ndims() > 2; }

    dim_t gemm_m() const { return swap_ab_ ? N() : M(); }
    dim_t gemm_n() const { return swap_ab_ ? M() : N(); }
    dim_t gemm_k() const { return K(); }

    int batch_dims() const { return nstl::max(ndims() - 2, 0); }
    bool is_batched() const { return ndims() > 2; }

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
    dim_t stride_c(int dim = 0) const {
        return stride_for(kernel_input_.dst_md, dim);
    }

    data_type_t gemm_a_type() const { return gemm_a_md().data_type; }
    data_type_t gemm_b_type() const { return gemm_b_md().data_type; }
    data_type_t c_type() const { return kernel_input_.dst_md.data_type; }
    data_type_t acc_type() const { return desc()->accum_data_type; }

    // DOCTRINE: matmul-driven bias is ALWAYS routed through binary post-op.
    // The legacy cOffset=Pre / co_t bias-as-CO path is gone — init_post_ops
    // prepends bias as a binary_add post-op and the kernel-side CO channel
    // carries c-zero-points / sum_ab only.
    data_type_t bias_type() const {
        return with_bias() ? kernel_input_.bias_md.data_type
                           : data_type::undef;
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

    // ----- Quant entry accessors (matmul-keyed; kernel_input_ entries are
    //       never re-keyed, mirroring attr_). All reads go via kernel_input_
    //       so reshape-rescaled masks/groups are seen consistently. -----
    bool with_a_scales() const { return a_scale_ndims() >= 0; }
    bool with_b_scales() const { return b_scale_ndims() >= 0; }
    bool with_c_scales() const {
        return !kernel_input_.scales.has_default_values(kC);
    }

    bool with_a_zero_points() const { return a_zp_ndims() >= 0; }
    bool with_b_zero_points() const { return b_zp_ndims() >= 0; }
    bool with_c_zero_points() const {
        return !kernel_input_.zero_points.has_default_values(kC);
    }

    bool with_a_group_sums() const { return a_gs_ndims() >= 0; }
    bool with_b_group_sums() const { return b_gs_ndims() >= 0; }

    bool with_sround() const {
        return attr()->rounding_mode_.get(DNNL_ARG_DST)
                == rounding_mode::stochastic;
    }
    bool with_mx_scale() const {
        const auto &c_scales = kernel_input_.scales.get(kC);
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
        return kernel_input_.scales.get(kA).get_data_type();
    }
    data_type_t b_scale_dt() const {
        return kernel_input_.scales.get(kB).get_data_type();
    }
    data_type_t c_scale_dt() const {
        return kernel_input_.scales.get(kC).get_data_type();
    }
    data_type_t a_zp_dt() const {
        return kernel_input_.zero_points.get(kA).get_data_type();
    }
    data_type_t b_zp_dt() const {
        return kernel_input_.zero_points.get(kB).get_data_type();
    }
    data_type_t c_zp_dt() const {
        return kernel_input_.zero_points.get(kC).get_data_type();
    }
    data_type_t a_gs_dt() const {
        return kernel_input_.precomputed_reductions.get(kA).get_data_type();
    }
    data_type_t b_gs_dt() const {
        return kernel_input_.precomputed_reductions.get(kB).get_data_type();
    }

    int a_zp_ndims() const {
        return quant_entry_ndims(kernel_input_.zero_points.get(kA),
                kernel_input_.src_zp_md, ndims() - 1);
    }
    int b_zp_ndims() const {
        return quant_entry_ndims(kernel_input_.zero_points.get(kB),
                kernel_input_.wei_zp_md, ndims() - 2);
    }
    int c_zp_ndims() const {
        return quant_entry_ndims(kernel_input_.zero_points.get(kC),
                kernel_input_.dst_zp_md, -1);
    }
    int a_gs_ndims() const {
        return quant_entry_ndims(kernel_input_.precomputed_reductions.get(kA),
                kernel_input_.src_gs_md, ndims() - 1);
    }
    int b_gs_ndims() const {
        return quant_entry_ndims(kernel_input_.precomputed_reductions.get(kB),
                kernel_input_.wei_gs_md, ndims() - 2);
    }
    int a_scale_ndims() const {
        if (a_scale_ndims_override_ != INT_MIN) return a_scale_ndims_override_;
        return quant_entry_ndims(kernel_input_.scales.get(kA),
                kernel_input_.src_scale_md, ndims() - 1);
    }
    int b_scale_ndims() const {
        if (b_scale_ndims_override_ != INT_MIN) return b_scale_ndims_override_;
        return quant_entry_ndims(kernel_input_.scales.get(kB),
                kernel_input_.wei_scale_md, ndims() - 2);
    }
    int c_scale_ndims() const {
        return quant_entry_ndims(kernel_input_.scales.get(kC),
                kernel_input_.dst_scale_md, -1);
    }

    bool a_force_gs() const {
        return !kernel_input_.precomputed_reductions.has_default_values(kA);
    }
    bool b_force_gs() const {
        return !kernel_input_.precomputed_reductions.has_default_values(kB);
    }

    bool zp_host_scalar(int arg) const {
        return kernel_input_.zero_points.get(arg).is_host_scalar();
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
    // / sum_at_begin() derive from the final canonicalized
    // kernel_input_.post_ops (note that bias-via-binary prepends a
    // binary_add at index 0; if the user's sum was originally at index 0
    // it ends up at index 1 post-prepend, so sum_at_begin() correctly
    // reports false — the kernel needs higher accumulator precision when
    // anything precedes the sum).
    // NOTE: is_sum() with default args requires scale==1.0 and zp==0 — too
    // restrictive for our purposes. Match the kind directly so a sum with
    // arbitrary scale (which is exactly what beta() must track) is detected.
    bool with_sum() const {
        const auto &po = kernel_input_.post_ops;
        for (int i = 0; i < po.len(); ++i)
            if (po.entry_[i].kind == primitive_kind::sum) return true;
        return false;
    }
    bool sum_at_begin() const {
        const auto &po = kernel_input_.post_ops;
        return po.len() > 0 && po.entry_[0].kind == primitive_kind::sum;
    }
    float beta() const {
        const auto &po = kernel_input_.post_ops;
        for (int i = 0; i < po.len(); ++i)
            if (po.entry_[i].kind == primitive_kind::sum)
                return po.entry_[i].sum.scale;
        return 0.0f;
    }

    // ----- Binary / scale / zp / gs strides (per post-op index or per arg). -----
    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;
    dim_t scale_stride(int idx, int arg) const;
    dim_t zp_stride(int idx, int arg) const;
    dim_t gs_stride(int idx, int arg) const;

    const post_ops_t *post_ops() const { return &kernel_input_.post_ops; }
    const std::vector<binary_src_t> &binary_srcs() const {
        return kernel_input_.binary_srcs;
    }

    // True iff any post-op contributes work beyond a scalar scale: eltwise,
    // user-binary, prelu (canonicalized to binary), or bias (via binary).
    // Scale-derived binaries (binary_srcs[i].type == scales) do NOT count
    // — they're a representation choice for what is semantically a scale.
    // Sum entries also don't count (they're handled via problem.beta and
    // a separate gate `with_sum() && with_c_scales()`).
    bool non_scale_po() const {
        const auto &po = kernel_input_.post_ops;
        const auto &bsrcs = kernel_input_.binary_srcs;
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry_[i];
            if (e.is_eltwise()) return true;
            if (e.is_binary()) {
                auto t = bsrcs[i].type;
                if (t == binary_src_t::binary || t == binary_src_t::bias
                        || t == binary_src_t::prelu)
                    return true;
            }
        }
        return false;
    }

    // ----- Quant memory descriptors (matmul-keyed: src=A, wei=B, dst=C). -----
    const memory_desc_t &a_scale_md() const {
        return kernel_input_.src_scale_md;
    }
    const memory_desc_t &b_scale_md() const {
        return kernel_input_.wei_scale_md;
    }
    const memory_desc_t &c_scale_md() const {
        return kernel_input_.dst_scale_md;
    }
    const memory_desc_t &a_zp_md() const { return kernel_input_.src_zp_md; }
    const memory_desc_t &b_zp_md() const { return kernel_input_.wei_zp_md; }
    const memory_desc_t &c_zp_md() const { return kernel_input_.dst_zp_md; }
    const memory_desc_t &a_gs_md() const { return kernel_input_.src_gs_md; }
    const memory_desc_t &b_gs_md() const { return kernel_input_.wei_gs_md; }

    // External md getters fall through to matmul_pd_t — `src_md(0, false)`
    // returns &src_md_ (user ndims, resolved tags via commit_resolved_tags),
    // `src_md(0, true)` returns desc()->src_desc (raw user input). The
    // kernel-view mds are reached via the explicit `kernel_*_md()` helpers.

    // ----- Pd state (stable across swap or used only at init time). -----
    bool swap_ab_ = false;

    // Kernel-facing input view. Source of truth for shapes/masks consumed
    // by init_GEMMProblem and downstream kernel code.
    jit_gemm_input_t kernel_input_ {};

    // a/b_scale_ndims_override_: INT_MIN sentinel = override unset. The
    // override is set to -1 after a 1D scale is converted to a binary
    // post-op, so that with_a_scales() / with_b_scales() / a_scale_ndims()
    // see "no scale" post-conversion (the scale buffer now flows via
    // binary_srcs[i] with type == scales, not via kernel_input_.scales).
    int a_scale_ndims_override_ = INT_MIN;
    int b_scale_ndims_override_ = INT_MIN;

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

private:
    dim_t stride_for(const memory_desc_t &md, int dim) const {
        if (dim >= md.ndims - 2 || md.dims[dim] == 1) return 0;
        return md.format_desc.blocking.strides[dim];
    }
    static int mask_to_gemm(int mask, int nd);
    static int swap_mask_bits(int mask, int i, int j);

    // The kernel-view md that plays the kernel-A role: matmul WEIGHTS under
    // swap, matmul SRC otherwise. Likewise for kernel-B. Used only by the
    // compute_gemm_* helpers below to derive kernel-view ld/trans/type/stride.
    const memory_desc_t &gemm_a_md() const {
        return swap_ab_ ? kernel_input_.weights_md : kernel_input_.src_md;
    }
    const memory_desc_t &gemm_b_md() const {
        return swap_ab_ ? kernel_input_.src_md : kernel_input_.weights_md;
    }

    // Kernel-view leading-dim / transpose computations. base's gemm path
    // used these directly on the matmul md (with the matmul-natural axis
    // ordering); the post-init compute_gemm_*() helpers mirror that.
    dim_t compute_gemm_lda() const { return get_ld(gemm_a_md()); }
    dim_t compute_gemm_ldb() const { return get_ld(gemm_b_md()); }
    dim_t compute_ldc() const { return get_ld(kernel_input_.dst_md); }
    // Bias helpers read kernel_input_.bias_md (the 2D-reshaped view if
    // reshape fired), so that ld_bias matches the kernel view of dst.
    dim_t compute_ld_bias() const {
        return kernel_input_.bias_md.ndims >= 2
                ? get_ld(kernel_input_.bias_md)
                : 1;
    }
    transpose_t compute_gemm_transa() const { return get_trans(gemm_a_md()); }
    transpose_t compute_gemm_transb() const { return get_trans(gemm_b_md()); }
    transpose_t compute_trans_bias() const {
        if (!with_bias()) return transpose::notrans;
        return get_trans(kernel_input_.bias_md);
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
