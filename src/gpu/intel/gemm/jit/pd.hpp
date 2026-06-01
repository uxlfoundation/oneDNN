/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_JIT_PD_HPP
#define GPU_INTEL_GEMM_JIT_PD_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/memory_storage.hpp"
#include "gemmstone/problem.hpp"
#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/post_ops.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

#define GEMM_MAX_PO 36

// Reverse of convert_dnnl_to_kernel_type (gen_kernel.hpp): map a gemmstone
// kernel Type back to a dnnl data_type_t. The forward map is a bijection over
// the GEMM-relevant types, so this round-trips exactly. Used by kernel_config_t's
// a/b/c_type() accessors, which keep the matrix types only as
// problem.Ta/Tb/Tc_ext.
inline data_type_t convert_kernel_to_dnnl_type(gemmstone::Type type) {
    using gemmstone::Type;
    switch (type) {
        case Type::f64: return data_type::f64;
        case Type::f32: return data_type::f32;
        case Type::f16: return data_type::f16;
        case Type::bf16: return data_type::bf16;
        case Type::bf8: return data_type::f8_e5m2;
        case Type::hf8: return data_type::f8_e4m3;
        case Type::f8_e8m0: return data_type::e8m0;
        case Type::f4_e2m1: return data_type::f4_e2m1;
        case Type::f4_e3m0: return data_type::f4_e3m0;
        case Type::s32: return data_type::s32;
        case Type::u8: return data_type::u8;
        case Type::s8: return data_type::s8;
        case Type::u4: return data_type::u4;
        case Type::s4: return data_type::s4;
        case Type::invalid: return data_type::undef;
        default: gpu_error_not_expected(); return data_type::undef;
    }
}

struct binary_src_t {
    enum type_t { none, scales, bias, binary, prelu } type;
    int index;
    // Source memory descriptor for the post-op input (binary src1 / bias /
    // prelu weights / scales md), in user orientation. Snapshot taken in
    // init_post_ops; swap-invariant (matches the old swap-invariant
    // post_ops/prelu_wei_md). Feeds ld_binary/stride_binary so they no longer
    // index a stored post_ops_t. Empty ({}) for the `none` types (sum/eltwise)
    // and for `scales` in ld_binary (which returns 1, as before).
    memory_desc_t src_md = {};

    binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
};

// Canonical kernel-facing projection of pd_t state. Phase A populators
// (pd_t::init_attrs / init_post_ops) write into this cfg in user
// orientation; Phase B applies swap_fold in place to flip to kernel
// orientation. swap_ab is encoded exactly once inside swap_fold;
// downstream consumers (Phase C validators, init_GEMMProblem,
// transfer_post_ops, gen_t::execute, launch_nocopy) read cfg.* uniformly
// and never branch on pd()->swap_ab() again.
struct kernel_config_t {
    // Embedded kernel projection. Phase-A populators write the migrated
    // scalar subset here; swap_fold keeps it kernel-oriented via
    // GEMMProblem::transpose(); init_GEMMProblem consumes it. The legacy
    // fields below remain as Phase-A populator scratch / validator input.
    gemmstone::GEMMProblem problem;

    // Matrix types are the embedded problem's Ta/Tb/Tc_ext; these accessors
    // project them back to dnnl data_type_t. swap_fold flips A/B via
    // problem.transpose(), so a_type()/b_type() follow the swap automatically
    // (no separate fields to keep in sync).
    data_type_t a_type() const {
        return convert_kernel_to_dnnl_type(problem.Ta_ext);
    }
    data_type_t b_type() const {
        return convert_kernel_to_dnnl_type(problem.Tb_ext);
    }
    data_type_t c_type() const {
        return convert_kernel_to_dnnl_type(problem.Tc_ext);
    }
    // A/B orientation lives in problem.A/B.layout (seeded in Phase A from the
    // user trans flags; folded by problem.transpose() in swap_fold). The C-side
    // (trans_co) is likewise problem.CO.layout — no accessor is needed since
    // only init_GEMMProblem consumes it, straight from the folded problem.
    bool trans_a() const {
        return problem.A.layout == gemmstone::MatrixLayout::T;
    }
    bool trans_b() const {
        return problem.B.layout == gemmstone::MatrixLayout::T;
    }

    // sum_ab lives in the embedded problem as sumA/sumB (populated in Phase A,
    // swapped by problem.transpose() in swap_fold).

    // A/B/C scalars.
    dim_t m = 0, n = 0, k = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;

    // Quant memory descriptors. A/B-side are swap-folded; C-side is
    // passthrough. Plain MDs; strides used for batch-dim lookups.
    memory_desc_t a_scale_md = {}, b_scale_md = {}, c_scale_md = {};
    memory_desc_t a_zp_md = {}, b_zp_md = {}, c_zp_md = {};
    memory_desc_t a_gs_md = {}, b_gs_md = {};

    // Post-op tail (populated by init_post_ops; swap-invariant). The lowered
    // post-op semantics live solely in problem.postOps/binary/Tbinary; the
    // irreducible per-entry runtime residue (source md for ld/stride lookups +
    // arg routing) lives on binary_srcs[*].{src_md,type,index}. There is no
    // stored post_ops_t: with_sum/sum_at_begin derive from the original user
    // post-ops via pd_t::with_sum()/sum_at_begin(); binary/eltwise presence
    // derives from problem.postOps.
    std::vector<binary_src_t> binary_srcs;
    bool non_scale_po = false;

    // Bias cfg (swap-invariant; bias is fused as the C-side
    // offset/binary input).
    bool with_bias = false;
    int bias_cmask = -1;
    dim_t ld_bias = 0;

    // Init-time scalar — extracted from sum post-op in Phase A.
    float beta = 0.0f;

    // Carries the swap signal to transfer_post_ops for per-binary layout
    // selection and to init_GEMMProblem for layout flipping on
    // AO/BO/A_scale/B_scale/Ag/Bg.
    bool swap_ab = false;

    // Leading-dim / stride lookups for post-op binary src (swap-invariant;
    // the bias/binary/prelu memory descriptors are in user orientation).
    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;
};

struct pd_t : public gemm::pd_t {
    using gemm::pd_t::pd_t;

    using binary_src_t = jit::binary_src_t;

    // Assumes desc() was already initialized with default formats. Runs
    // Phase A (populate cfg in user orientation, then validate user attrs).
    // Phase B (decide_swap_ab + pad_lda + swap_fold) and Phase C
    // (kernel-cap checks) run inside derived gen_t::pd_t::init.
    status_t init(impl::engine_t *engine, compute::gpu_arch_t arch);

    static constexpr post_op::specializations_t get_post_op_specializations() {
        using mode_t = post_op::specializations_t::inline_mode_t;
        using sum_t = post_op::specializations_t::sum_t;
        // The sum scale is handled as GEMM beta argument
        return {{}, sum_t(mode_t::impl_managed(), {}), {}};
    }

    static constexpr bool supported_binary_op(alg_kind_t alg) {
        using namespace alg_kind;
        return utils::one_of(alg, binary_add, binary_sub, binary_mul,
                binary_div, binary_min, binary_max);
    }

    // Phase A populators — write into cfg in user orientation. The engine
    // parameter is unused but required so the VDISPATCH_GEMM macro can
    // call this->info(engine) in error reporting.
    status_t init_attrs(kernel_config_t &cfg, impl::engine_t *engine);
    status_t init_post_ops(kernel_config_t &cfg, impl::engine_t *engine);

    // Phase A finalizer — seed the kernel projection from desc(): the A/B/C
    // scalars (m/n/k/ld*), the embedded problem's matrix types and A/B/C/CO
    // orientation. The post-op binary metadata (problem.postOps/binary/
    // Tbinary) is committed by init_post_ops, which must run first. swap_fold
    // then folds the whole problem to kernel orientation. Shared by gen_t and
    // xe_hp_systolic.
    status_t seed_problem(kernel_config_t &cfg);

    // Phase A validators — read cfg + attr; report errors in user terms.
    status_t scales_ok(const kernel_config_t &cfg, impl::engine_t *engine);
    status_t zp_ok(const kernel_config_t &cfg, impl::engine_t *engine);
    status_t gs_ok(const kernel_config_t &cfg, impl::engine_t *engine);

    // Phase B — decide whether to swap A/B. Overridden by gen_t; default
    // returns false (no swap) for impls that don't support the swap path.
    virtual bool decide_swap_ab(const kernel_config_t &cfg) const {
        return false;
    }

    bool valid_2d_mask(int mask, int ndims, bool per_tensor_ok = true);

    status_t init_GEMMProblem(gemmstone::GEMMProblem &problem,
            const intel::engine_t *engine, const kernel_config_t &cfg) const;

    // Canonical kernel-facing projection (Phase A populates in user
    // orientation; Phase B's swap_fold flips it to kernel orientation).
    kernel_config_t cfg_;
    const kernel_config_t &cfg() const { return cfg_; }

    static constexpr int mask_scalar = 1 << 0;
    static constexpr int mask_per_oc = 1 << 1;
    static constexpr int mask_per_ic = 1 << 2;

    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

    int cmask_a() const {
        return attr()->zero_points_.get(DNNL_ARG_A).get_mask();
    }
    int cmask_b() const {
        return attr()->zero_points_.get(DNNL_ARG_B).get_mask();
    }
    int cmask_c() const {
        return attr()->zero_points_.get(DNNL_ARG_C).get_mask();
    }

    float alpha() const {
        auto attr_info = attr_info_t::create(attr());
        bool host_scales_by_alpha = attr_info.with_host_src_scale
                || attr_info.with_host_wei_scale
                || (attr_info.with_host_dst_scale
                        && attr()->post_ops_.len() == 0);
        // Bogus non-one value for host scalar.
        // Actual value will be passed on execution step
        if (host_scales_by_alpha) return 9.99f;
        return 1.0f;
    }

    sum_ab_t sum_ab() const { return desc()->sum_ab; }
    bool with_sum_ab() const { return sum_ab() != sum_ab::sum_none; }

    // Sum post-op presence and whether it is the first post-op. Derived from
    // the original user post-ops: the sum entry survives the bias/scale->binary
    // conversions in init_post_ops, and reading the pre-conversion ordering for
    // sum_at_begin reproduces the value captured before those prepends.
    bool with_sum() const {
        return attr()->post_ops_.find(primitive_kind::sum) != -1;
    }
    bool sum_at_begin() const {
        const auto &po = attr()->post_ops_;
        return po.len() > 0 && po.entry_[0].kind == primitive_kind::sum;
    }

    int sum_ab_cmask() const {
        switch (sum_ab()) {
            default:
            case sum_ab::sum_none: return 0;
            case sum_ab::sum_a_row: return 1;
            case sum_ab::sum_b_col: return 2;
        }
    }
    bool with_c_scales() const {
        return !attr()->scales_.has_default_values(DNNL_ARG_DST);
    }
    bool with_c_zero_points() const {
        return !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    }

    bool with_sround() const {
        return attr()->rounding_mode_.get(DNNL_ARG_DST)
                == rounding_mode::stochastic;
    }
    bool with_mx_scale() const {
        return attr()->scales_.get(DNNL_ARG_C).is_mx();
    }

    bool dy_quant_enabled() const;
    bool wei_decomp() const;

    bool swap_ab() const { return cfg_.swap_ab; }

    int batch_dims() const { return nstl::max(desc()->c_desc.ndims - 2, 0); }
    bool trans_bias() const { return desc()->trans_bias() == dnnl_trans; }

    dim_t stride(int arg, int dim) const {
        if (arg == DNNL_ARG_A) return desc()->stride_a(dim);
        if (arg == DNNL_ARG_B) return desc()->stride_b(dim);
        if (arg == DNNL_ARG_C) return desc()->stride_c(dim);
        gpu_error_not_expected();
        return 0;
    }

    // Per-batch-dim strides of the (already swap-folded) quant memory
    // descriptors. Named by kernel side (A/B) and take no DNNL_ARG, so a
    // caller cannot re-apply the A/B swap that swap_fold already baked into
    // cfg_.{a,b}_*_md. Contrast stride() above, which reads the unswapped
    // desc() and so legitimately takes a user-frame DNNL_ARG.
    dim_t a_scale_stride(int idx) const;
    dim_t b_scale_stride(int idx) const;
    dim_t a_zp_stride(int idx) const;
    dim_t b_zp_stride(int idx) const;
    dim_t a_gs_stride(int idx) const;
    dim_t b_gs_stride(int idx) const;
    bool a_zp_host_scalar() const {
        auto attr_info = attr_info_t::create(attr());
        return attr_info.with_host_wei_zp;
    }
    bool b_zp_host_scalar() const {
        auto attr_info = attr_info_t::create(attr());
        return attr_info.with_host_src_zp;
    }
    bool c_zp_host_scalar() const {
        auto attr_info = attr_info_t::create(attr());
        return attr_info.with_host_dst_zp;
    }
    // Leading-dim / batch-stride alignment in bytes. `ld` and `dt` are the
    // (kernel-oriented) leading dim and type taken from the cfg; `batch_arg`
    // selects the desc() batch strides — a user-frame DNNL_ARG, so the caller
    // applies the swap_ab fold by passing the effective side.
    int align(dim_t ld, data_type_t dt, int batch_arg) const {
        auto align = utils::max_pow2_div(types::elements_to_bytes(dt, ld));
        for (int b = 0; b < batch_dims(); b++) {
            auto stride_bytes = utils::max_pow2_div(
                    types::elements_to_bytes(dt, stride(batch_arg, b)));
            align = (stride_bytes ? nstl::min(align, stride_bytes) : align);
        }
        return int(align);
    }
};

// Phase B — leading-dim padding and in-place swap fold. swap_fold
// flips user→kernel orientation; pad_lda is the orientation-aware
// padding that runs between the swap decision and swap_fold.
void pad_lda(kernel_config_t &cfg, bool swap);
void swap_fold(kernel_config_t &cfg, bool swap);

// Populate problem.postOps and problem.binary from a gpu_post_ops_t, in user
// (un-swapped) orientation. The A/B swap is folded by the caller via
// problem.transpose() (in swap_fold), so this builds no swap_ab dependency.
status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops);

// Runtime-time projection of pd_t + exec_ctx_t state in kernel-cfg
// orientation. Inherits kernel_config_t's swap-folded scalars/flags and adds
// runtime-bound storages, host scalars, base offsets, and stride lookups.
// All A/B-side fields (storages, ao/bo, scales, group sums, cmask) are
// already swap_ab-folded; downstream consumers (gen_t::execute,
// gen_t::launch_nocopy) read cfg.* uniformly and never branch on
// pd()->swap_ab() again. Init-time consumers (init_GEMMProblem,
// transfer_post_ops) take 'const kernel_config_t&' and so cannot reach any of
// the runtime fields below — the type still enforces phase separation.
struct exec_config_t : kernel_config_t {
    // Primary A/B/C storages (A/B routed under swap_ab).
    const memory_storage_t *a = nullptr;
    const memory_storage_t *b = nullptr;
    const memory_storage_t *c = nullptr;

    // Quantization storages (swap-folded).
    const memory_storage_t *ao = nullptr;
    const memory_storage_t *bo = nullptr;
    const memory_storage_t *a_scales = nullptr;
    const memory_storage_t *b_scales = nullptr;
    const memory_storage_t *c_scales = nullptr;
    const memory_storage_t *ag = nullptr;
    const memory_storage_t *bg = nullptr;

    // Merged c-offset / bias / sum_ab storage; the kernel sees one ptr.
    const memory_storage_t *co = nullptr;

    // Stochastic round seed.
    const memory_storage_t *sround_seed = nullptr;

    // Host scalar values; ao/bo are sign-flipped and swap-folded.
    int16_t ao_host_scalar = 0;
    int16_t bo_host_scalar = 0;
    int16_t co_host_scalar = 0;

    // Per-side flags (swap-folded).
    bool with_a_zero_points = false;
    bool with_b_zero_points = false;

    // Resolved at execute time (host-scalar scales fold into alpha).
    float alpha = 1.0f;
    bool with_mx_scale = false;

    // Already swap_ab-folded (A/B mask bits exchanged under swap_ab).
    int cmask = 0;

    // Base offsets, in elements (kernel-cfg: a is the kernel's A storage).
    size_t off_a0 = 0;
    size_t off_b0 = 0;
    size_t off_c0 = 0;
    int64_t off_co0 = 0;

    // Backing pd_t + effective DNNL_ARG routing for batch/scale/zp/gs
    // stride lookups. eff_a_arg/eff_b_arg are swapped if swap_ab.
    const pd_t *pd = nullptr;
    int eff_a_arg = DNNL_ARG_A;
    int eff_b_arg = DNNL_ARG_B;

    dim_t stride_a(int dim) const { return pd->stride(eff_a_arg, dim); }
    dim_t stride_b(int dim) const { return pd->stride(eff_b_arg, dim); }
    dim_t stride_c(int dim) const { return pd->desc()->stride_c(dim); }
    // The quant memory descriptors (a/b_scale_md, a/b_zp_md, a/b_gs_md) live
    // on the cfg and are already swap-folded to kernel orientation by
    // swap_fold. Forward to the kernel-side pd_t accessors, which take no
    // DNNL_ARG — so the A/B swap cannot be re-applied here (re-applying it
    // read the empty opposite-side md: crash on batched + swap_ab + one-sided
    // quant). Contrast stride_a/b above, which reach the unswapped desc() and
    // so must route via eff_arg.
    dim_t scale_stride_a(int idx) const { return pd->a_scale_stride(idx); }
    dim_t scale_stride_b(int idx) const { return pd->b_scale_stride(idx); }
    dim_t zp_stride_a(int idx) const { return pd->a_zp_stride(idx); }
    dim_t zp_stride_b(int idx) const { return pd->b_zp_stride(idx); }
    dim_t gs_stride_a(int idx) const { return pd->a_gs_stride(idx); }
    dim_t gs_stride_b(int idx) const { return pd->b_gs_stride(idx); }
};

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
