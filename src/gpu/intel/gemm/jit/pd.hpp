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
#include "gemmstone/driver_info.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
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
    // Post-op input src md, user (pre-swap) orientation.
    memory_desc_t src_md = {};

    binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
};

// Kernel-facing snapshot of pd_t state; swap_ab is folded once in
// apply_swap_ab.
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
    // Valid pre-swap only; apply_swap_ab/finalize_problem fold C back to N.
    bool trans_c() const {
        return problem.C.layout == gemmstone::MatrixLayout::T;
    }

    // Covers tensor and host-scalar zps (a host scalar keeps Calc,
    // aoPtrDims = -1).
    bool with_a_zero_points() const {
        return problem.aOffset == gemmstone::ABOffset::Calc;
    }
    bool with_b_zero_points() const {
        return problem.bOffset == gemmstone::ABOffset::Calc;
    }
    // Covers tensor and host-scalar c-zp; set in init_attrs.
    bool with_c_zero_points() const {
        return problem.cOffset == gemmstone::COffset::Post;
    }
    // Tc_scale is set for any present dst scale (converted/mx/host-scalar).
    bool with_c_scales() const {
        return problem.Tc_scale != gemmstone::Type::invalid;
    }

    // sumA/sumB carry the sum_ab reduction; the OR is swap-invariant.
    bool with_sum_ab() const { return problem.sumA || problem.sumB; }

    // Valid after init_post_ops (reads the lowered post-op chain).
    bool with_sum() const {
        for (const auto &e : problem.postOps.ops)
            if (e.is_sum()) return true;
        return false;
    }
    bool sum_at_begin() const {
        const auto &ops = problem.postOps.ops;
        return ops.len() > 0 && ops[0].is_sum();
    }

    dim_t m = 0, n = 0, k = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;

    // Quant mds: A/B-side are swap_ab-folded; C-side is passthrough.
    memory_desc_t a_scale_md = {}, b_scale_md = {};
    memory_desc_t a_zp_md = {}, b_zp_md = {};
    memory_desc_t a_gs_md = {}, b_gs_md = {};

    std::vector<binary_src_t> binary_srcs;

    // Any post-op besides sum and the converted A/B/C scale binaries.
    bool non_scale_po() const {
        int n = 0;
        for (const auto &s : binary_srcs)
            if (s.type != binary_src_t::scales) n++;
        return n > (with_sum() ? 1 : 0);
    }

    // Bias present and not lowered to a binary post-op (dedicated C-offset).
    bool with_bias() const {
        if (bias_type == data_type::undef) return false;
        for (const auto &s : binary_srcs)
            if (s.type == binary_src_t::bias) return false;
        return true;
    }
    dim_t ld_bias = 0;

    int cmask = 0;

    float beta = 0.0f;

    bool swap_ab = false;

    static constexpr int max_batch_dims = DNNL_MAX_NDIMS - 2;
    dim_t a_batch_strides[max_batch_dims] = {};
    dim_t b_batch_strides[max_batch_dims] = {};
    dim_t c_batch_strides[max_batch_dims] = {};

    data_type_t bias_type = data_type::undef;
    data_type_t sum_ab_type = data_type::undef;
    accumulation_mode_t acc_mode = accumulation_mode::strict;
    bool wei_decomp = false;

    // Host-scalar scales fold into alpha at exec; until then the 9.99 sentinel
    // makes the kernel emit a multiply. Seeded at init from pd host-scale flags.
    float alpha_ = 1.0f;
    float alpha() const { return alpha_; }

    bool a_host_scale = false;
    bool b_host_scale = false;
    bool c_host_scale_to_alpha = false;

    // fpmath compute-mode bits (gen_nocopy_desc_t::compute_mode).
    int fpmath_modes = 0;
    bool deterministic = false;
    dim_t c_batch_sizes[max_batch_dims] = {};
    dim_t batch() const {
        dim_t b = 1;
        for (int i = 0; i < problem.batchDims; i++)
            b *= c_batch_sizes[i];
        return b;
    }
    // Swap-folded orientation defaults (N/T, exchanged under swap_ab).
    gemmstone::MatrixLayout ab_layout_N() const {
        return swap_ab ? gemmstone::MatrixLayout::T
                       : gemmstone::MatrixLayout::N;
    }
    gemmstone::MatrixLayout ab_layout_T() const {
        return swap_ab ? gemmstone::MatrixLayout::N
                       : gemmstone::MatrixLayout::T;
    }

    // Byte alignment of a leading dim combined with its batch strides.
    int align_bytes(dim_t ld, data_type_t dt, const dim_t *bstr) const {
        auto a = utils::max_pow2_div(types::elements_to_bytes(dt, ld));
        for (int b = 0; b < problem.batchDims; b++) {
            auto sb = utils::max_pow2_div(
                    types::elements_to_bytes(dt, bstr[b]));
            a = (sb ? nstl::min(a, sb) : a);
        }
        return int(a);
    }

    // Set by finalize_problem; guards finalized-problem reads.
    bool finalized_ = false;

    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;

    // Strides of the swap_ab-folded quant mds, named by kernel side.
    dim_t a_scale_stride(int idx) const;
    dim_t b_scale_stride(int idx) const;
    dim_t a_zp_stride(int idx) const;
    dim_t b_zp_stride(int idx) const;
    dim_t a_gs_stride(int idx) const;
    dim_t b_gs_stride(int idx) const;
};

struct pd_t : public gemm::pd_t {
    using gemm::pd_t::pd_t;

    using binary_src_t = jit::binary_src_t;

    // Builds the kernel cfg; assumes default formats are already set.
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

    virtual bool decide_swap_ab(const kernel_config_t &cfg) const {
        return false;
    }

    kernel_config_t cfg_;
    const kernel_config_t &cfg() const { return cfg_; }

    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

    sum_ab_t sum_ab() const { return desc()->sum_ab; }

    bool wei_decomp() const;
    bool dy_quant_enabled() const;

    // Host-scalar scale presence; values are resolved from memory args at exec.
    bool a_host_scale() const {
        return attr()->scales_.get(DNNL_ARG_A).is_host_scalar();
    }
    bool b_host_scale() const {
        return attr()->scales_.get(DNNL_ARG_B).is_host_scalar();
    }
    // C host-scalar scale folds to alpha only without other post-ops.
    bool c_host_scale_to_alpha() const {
        return attr()->scales_.get(DNNL_ARG_C).is_host_scalar()
                && attr()->post_ops_.len() == 0;
    }

    int batch_dims() const { return nstl::max(desc()->c_desc.ndims - 2, 0); }
};

status_t init_kernel_config(
        kernel_config_t &cfg, const pd_t *pd, impl::engine_t *engine);
status_t scales_ok(const pd_t *pd, impl::engine_t *engine);
status_t zp_ok(const pd_t *pd, impl::engine_t *engine);
status_t gs_ok(const pd_t *pd, impl::engine_t *engine);

status_t finalize_problem(kernel_config_t &cfg, const intel::engine_t *engine);

void pad_leading_dims(kernel_config_t &cfg, bool swap);
void apply_swap_ab(kernel_config_t &cfg, bool swap);

status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops);

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
