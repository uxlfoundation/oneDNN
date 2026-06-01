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

struct quant_params {
    data_type_t scales_type = data_type::undef;
    data_type_t zp_type = data_type::undef;
    data_type_t gs_type = data_type::undef;
    int scale_ndims = -1;
    int zp_ndims = -1;
    int gs_ndims = -1;
    int group_k = 0;
    int group_m = 0;
    int group_n = 0;
    bool force_gs = false;
    bool zp_host_scalar = false;
};

struct binary_src_t {
    enum type_t { none, scales, bias, binary, prelu } type;
    int index;

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
    // A/B/C scalars.
    data_type_t a_type = data_type::undef;
    data_type_t b_type = data_type::undef;
    data_type_t c_type = data_type::undef;
    dim_t m = 0, n = 0, k = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;
    bool trans_a = false, trans_b = false;
    int align_a = 0, align_b = 0, align_c = 0;

    // Sub-configurations in kernel-cfg orientation. group_m/n are
    // swapped under swap_ab.
    quant_params a_quant, b_quant, c_quant;
    sum_ab_t sum_ab = sum_ab::sum_none;
    bool trans_co = false;

    // Quant memory descriptors. A/B-side are swap-folded; C-side is
    // passthrough. Plain MDs; strides used for batch-dim lookups.
    memory_desc_t a_scale_md = {}, b_scale_md = {}, c_scale_md = {};
    memory_desc_t a_zp_md = {}, b_zp_md = {}, c_zp_md = {};
    memory_desc_t a_gs_md = {}, b_gs_md = {};

    // Post-op tail (populated by init_post_ops; swap-invariant).
    post_ops_t post_ops;
    std::vector<binary_src_t> binary_srcs;
    memory_desc_t prelu_wei_md = {};
    bool with_sum = false;
    bool sum_at_begin = false;
    bool non_scale_po = false;

    // Bias cfg (swap-invariant; bias is fused as the C-side
    // offset/binary input).
    bool bias_via_binary = false;
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

    dim_t lda_ = 0, ldb_ = 0;
    bool transa_ = false, transb_ = false;
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
    bool trans_a() const { return transa_; }
    bool trans_b() const { return transb_; }
    bool trans_bias() const { return desc()->trans_bias() == dnnl_trans; }

    dim_t ld(int arg) const {
        if (arg == DNNL_ARG_A) return lda_;
        if (arg == DNNL_ARG_B) return ldb_;
        if (arg == DNNL_ARG_C) return desc()->ldc();
        gpu_error_not_expected();
        return 0;
    }
    dim_t stride(int arg, int dim) const {
        if (arg == DNNL_ARG_A) return desc()->stride_a(dim);
        if (arg == DNNL_ARG_B) return desc()->stride_b(dim);
        if (arg == DNNL_ARG_C) return desc()->stride_c(dim);
        gpu_error_not_expected();
        return 0;
    }
    data_type_t get_type(int arg) const {
        if (arg == DNNL_ARG_A) return desc()->a_type();
        if (arg == DNNL_ARG_B) return desc()->b_type();
        if (arg == DNNL_ARG_C) return desc()->c_type();
        gpu_error_not_expected();
        return data_type::undef;
    }

    dim_t scale_stride(int idx, int arg) const;
    dim_t zp_stride(int idx, int arg) const;
    dim_t gs_stride(int idx, int arg) const;
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
    int align(int arg) const {
        auto dt = get_type(arg);
        auto align = utils::max_pow2_div(types::elements_to_bytes(dt, ld(arg)));
        for (int b = 0; b < batch_dims(); b++) {
            auto stride_bytes = utils::max_pow2_div(
                    types::elements_to_bytes(dt, stride(arg, b)));
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

// Populate problem.postOps and problem.binary from a gpu_post_ops_t,
// folding per-binary layout selection by cfg.swap_ab so that no post-hoc
// problem.postOps.transpose()/b.transpose() pass is required.
status_t transfer_post_ops(gemmstone::GEMMProblem &problem,
        gpu_post_ops_t &&post_ops, const kernel_config_t &cfg);

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
    // swap_fold. So index them by the *kernel* side (A/B) directly — NOT
    // eff_arg, which would re-apply the swap and read the empty opposite-side
    // md (crash on batched + swap_ab + one-sided quant). Contrast stride_a/b
    // above, which reach the unswapped desc() and so must route via eff_arg.
    dim_t scale_stride_a(int idx) const {
        return pd->scale_stride(idx, DNNL_ARG_A);
    }
    dim_t scale_stride_b(int idx) const {
        return pd->scale_stride(idx, DNNL_ARG_B);
    }
    dim_t zp_stride_a(int idx) const { return pd->zp_stride(idx, DNNL_ARG_A); }
    dim_t zp_stride_b(int idx) const { return pd->zp_stride(idx, DNNL_ARG_B); }
    dim_t gs_stride_a(int idx) const { return pd->gs_stride(idx, DNNL_ARG_A); }
    dim_t gs_stride_b(int idx) const { return pd->gs_stride(idx, DNNL_ARG_B); }
};

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
