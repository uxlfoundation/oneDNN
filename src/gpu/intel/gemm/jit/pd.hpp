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
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/gemm/types.hpp"
#include "gpu/intel/post_ops.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

#define GEMM_MAX_PO 36

inline data_type_t convert_kernel_to_dnnl_type(gemmstone::Type type) {
    using gemmstone::Type;
    switch (type) {
        case Type::f64: return data_type::f64;
        case Type::f32: return data_type::f32;
        // tf32 buffers are physically f32.
        case Type::tf32: return data_type::f32;
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
    // Post-op input src md (binary/bias/prelu/scales), user orientation.
    memory_desc_t src_md = {};

    binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
};

// Kernel-facing snapshot of pd_t state. swap_ab is folded once in
// apply_swap_ab; consumers read cfg.* without branching on swap_ab.
struct kernel_config_t {
    gemmstone::GEMMProblem problem;

    data_type_t a_type() const {
        return convert_kernel_to_dnnl_type(problem.Ta_ext);
    }
    data_type_t b_type() const {
        return convert_kernel_to_dnnl_type(problem.Tb_ext);
    }
    data_type_t c_type() const {
        return convert_kernel_to_dnnl_type(problem.Tc_ext);
    }
    bool trans_a() const {
        return problem.A.layout == gemmstone::MatrixLayout::T;
    }
    bool trans_b() const {
        return problem.B.layout == gemmstone::MatrixLayout::T;
    }

    // Folded problem: aOffset==Calc is kernel-A's zp state (covers tensor and
    // host-scalar zps; a host scalar folds aoPtrDims to -1 but keeps Calc).
    bool with_a_zero_points() const {
        return problem.aOffset == gemmstone::ABOffset::Calc;
    }
    bool with_b_zero_points() const {
        return problem.bOffset == gemmstone::ABOffset::Calc;
    }

    // sumA/sumB carry the sum_ab reduction; the OR is swap-invariant.
    bool with_sum_ab() const { return problem.sumA || problem.sumB; }

    // swap_ab routing for per-ctx-arg pd lookups (e.g. per-arg strides):
    // the only place swap_ab re-enters once folded.
    int eff_a_arg() const { return swap_ab ? DNNL_ARG_B : DNNL_ARG_A; }
    int eff_b_arg() const { return swap_ab ? DNNL_ARG_A : DNNL_ARG_B; }

    dim_t m = 0, n = 0, k = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;

    // Quant mds: A/B-side are swap_ab-folded; C-side is passthrough.
    memory_desc_t a_scale_md = {}, b_scale_md = {}, c_scale_md = {};
    memory_desc_t a_zp_md = {}, b_zp_md = {}, c_zp_md = {};
    memory_desc_t a_gs_md = {}, b_gs_md = {};

    std::vector<binary_src_t> binary_srcs;
    bool non_scale_po = false;

    bool with_bias = false;
    int bias_cmask = -1;
    dim_t ld_bias = 0;

    // C-side offset mask (DST zp / bias / sum_ab precedence); seeded in user
    // orientation in init_post_ops, column<->row folded by apply_swap_ab.
    int cmask = 0;

    float beta = 0.0f;

    bool swap_ab = false;

    // Descriptor/attr snapshot finalize_problem needs, so finalize reads only
    // cfg.* (+ engine HW state). Batch strides are user-orientation at seed
    // then swap_ab-folded (C is orientation-invariant); ab_layout_N/T are
    // swap_ab-folded orientation defaults; the rest are orientation-invariant.
    static constexpr int max_batch_dims = DNNL_MAX_NDIMS - 2;
    dim_t a_batch_strides[max_batch_dims] = {};
    dim_t b_batch_strides[max_batch_dims] = {};
    dim_t c_batch_strides[max_batch_dims] = {};
    int batch_dims = 0;

    data_type_t bias_type = data_type::undef;
    data_type_t sum_ab_type = data_type::undef;
    accumulation_mode_t acc_mode = accumulation_mode::strict;
    bool wei_decomp = false;
    bool with_sround = false;
    bool with_c_zp = false;
    bool with_mx_scale = false;
    bool with_sum = false;
    bool sum_at_begin = false;
    float alpha = 1.0f;
    gemmstone::MatrixLayout ab_layout_N = gemmstone::MatrixLayout::N;
    gemmstone::MatrixLayout ab_layout_T = gemmstone::MatrixLayout::T;

    // Byte alignment of a leading dim folded with its swap_ab-folded batch
    // strides.
    int align_bytes(dim_t ld, data_type_t dt, const dim_t *bstr) const {
        auto a = utils::max_pow2_div(types::elements_to_bytes(dt, ld));
        for (int b = 0; b < batch_dims; b++) {
            auto sb = utils::max_pow2_div(
                    types::elements_to_bytes(dt, bstr[b]));
            a = (sb ? nstl::min(a, sb) : a);
        }
        return int(a);
    }

    // Debug tripwire: set by finalize_problem, asserted before finalized-problem
    // reads (select_kernel / systolic problem_ / execute). Not serialized.
    bool finalized_ = false;

    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;
};

struct pd_t : public gemm::pd_t {
    using gemm::pd_t::pd_t;

    using binary_src_t = jit::binary_src_t;

    // Assumes desc() was already initialized with default formats.
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

    // engine is unused but required so VDISPATCH_GEMM can call info(engine).
    status_t init_attrs(kernel_config_t &cfg, impl::engine_t *engine);
    status_t init_post_ops(kernel_config_t &cfg, impl::engine_t *engine);

    // Seeds the kernel cfg (types, orientation, m/n/k, ld*) from desc().
    // Must run after set_default_formats so ld* reflect the final strides.
    status_t seed_kernel_config(kernel_config_t &cfg);

    status_t scales_ok(const kernel_config_t &cfg, impl::engine_t *engine);
    status_t zp_ok(const kernel_config_t &cfg, impl::engine_t *engine);
    status_t gs_ok(const kernel_config_t &cfg, impl::engine_t *engine);

    virtual bool decide_swap_ab(const kernel_config_t &cfg) const {
        return false;
    }

    bool valid_2d_mask(int mask, int ndims, bool per_tensor_ok = true);

    // Completes cfg.problem with HW/register-derived fields. Must run after
    // apply_swap_ab: the derived fields are transpose-blind.
    status_t finalize_problem(
            kernel_config_t &cfg, const intel::engine_t *engine) const;

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

    // Derived from the finalized GEMM post-op chain the kernel executes
    // (problem.postOps), not the user attr: bias/scale lowering prepends
    // binaries, so a sum the user wrote first may not be first here. The
    // accumulator-width gate (need_x32_acc) needs this executed-chain order,
    // since "sum is the first applied op" is what allows a narrow accumulator.
    // Valid only after init_post_ops populates problem.postOps.
    bool with_sum() const {
        for (const auto &e : cfg_.problem.postOps.ops)
            if (e.is_sum()) return true;
        return false;
    }
    bool sum_at_begin() const {
        const auto &ops = cfg_.problem.postOps.ops;
        return ops.len() > 0 && ops[0].is_sum();
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

    // Strides of the already swap_ab-folded quant mds; named by kernel side and
    // take no DNNL_ARG so callers cannot re-apply the swap.
    dim_t a_scale_stride(int idx) const;
    dim_t b_scale_stride(int idx) const;
    dim_t a_zp_stride(int idx) const;
    dim_t b_zp_stride(int idx) const;
    dim_t a_gs_stride(int idx) const;
    dim_t b_gs_stride(int idx) const;
};

void pad_lda(kernel_config_t &cfg, bool swap);
void apply_swap_ab(kernel_config_t &cfg, bool swap);

status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops);

// Runtime-only exec state (storage pointers, host scalars, offsets, resolved
// alpha); the finalized cfg_ is reached by reference via cfg(). A/B storage
// is swap_ab-routed once in build_exec_config; the cfg it reaches is folded.
struct exec_config_t {
    const pd_t *pd = nullptr;
    const kernel_config_t &cfg() const { return pd->cfg(); }

    const memory_storage_t *a = nullptr;
    const memory_storage_t *b = nullptr;
    const memory_storage_t *c = nullptr;

    const memory_storage_t *ao = nullptr;
    const memory_storage_t *bo = nullptr;
    const memory_storage_t *a_scales = nullptr;
    const memory_storage_t *b_scales = nullptr;
    const memory_storage_t *c_scales = nullptr;
    const memory_storage_t *ag = nullptr;
    const memory_storage_t *bg = nullptr;

    const memory_storage_t *co = nullptr;

    const memory_storage_t *sround_seed = nullptr;

    int16_t ao_host_scalar = 0;
    int16_t bo_host_scalar = 0;
    int16_t co_host_scalar = 0;

    // Runtime-resolved alpha (cfg().alpha snapshot x host scalar scales),
    // computed in build_exec_config.
    float alpha = 1.0f;

    size_t off_a0 = 0;
    size_t off_b0 = 0;
    size_t off_c0 = 0;
    int64_t off_co0 = 0;

    dim_t stride_a(int dim) const { return pd->stride(cfg().eff_a_arg(), dim); }
    dim_t stride_b(int dim) const { return pd->stride(cfg().eff_b_arg(), dim); }
    dim_t stride_c(int dim) const { return pd->desc()->stride_c(dim); }
    // Quant mds are already swap_ab-folded, so forward to the side-named accessors;
    // routing via eff_arg would double-swap and read the empty opposite-side md.
    // (stride_a/b above reach the unswapped desc(), so they must use eff_arg.)
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
