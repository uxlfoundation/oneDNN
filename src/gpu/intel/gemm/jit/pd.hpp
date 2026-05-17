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

status_t transfer_post_ops(gemmstone::GEMMProblem &problem,
        gpu_post_ops_t &&post_ops_);

struct pd_t : public gemm::pd_t {
    using gemm::pd_t::pd_t;

    // Assumes desc() was already initialized with default formats
    status_t init(impl::engine_t *engine, compute::gpu_arch_t arch) {

        arch_ = arch;
        with_sround_ = attr()->rounding_mode_.get(DNNL_ARG_DST)
                == rounding_mode::stochastic;

        lda_ = desc()->lda();
        ldb_ = desc()->ldb();
        transb_ = desc()->transb() == dnnl_trans;

        CHECK(init_attrs(engine));
        CHECK(scales_ok(engine));
        CHECK(zp_ok(engine));
        CHECK(gs_ok(engine));
        CHECK(init_post_ops(engine));
        return status::success;
    }

    struct binary_src_t {
        enum type_t { none, scales, bias, binary, prelu } type;
        int index;

        binary_src_t(type_t type_, int index_) : type(type_), index(index_) {}
    };

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

    status_t init_post_ops(impl::engine_t *engine);
    status_t init_attrs(impl::engine_t *engine);
    status_t scales_ok(impl::engine_t *engine);
    status_t zp_ok(impl::engine_t *engine);
    status_t gs_ok(impl::engine_t *engine);

    static int quant_entry_ndims(
            const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx);

    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;

    const post_ops_t *post_ops() const { return &post_ops_; }
    const std::vector<binary_src_t> &binary_srcs() const {
        return binary_srcs_;
    }
    bool valid_2d_mask(int mask, int ndims, bool per_tensor_ok = true);

    status_t init_GEMMProblem(gemmstone::GEMMProblem &problem,
            const intel::engine_t *engine) const;

    float beta_ = 0.0f;

    bool with_sum_ = false;
    bool sum_at_begin_ = false;

    bool bias_via_binary_ = false;
    bool wei_decomp_ = false;
    bool dy_quant_enabled_ = false;
    bool quant_enabled_ = false;

    // INT_MIN = override unset.
    int a_scale_ndims_override_ = INT_MIN;
    int b_scale_ndims_override_ = INT_MIN;
    int a_group_k_ = 0, a_group_m_ = 0;
    int b_group_k_ = 0, b_group_n_ = 0;
    int c_group_m_ = 0, c_group_n_ = 0;

    bool non_scale_po_ = false;

    post_ops_t post_ops_;
    std::vector<binary_src_t> binary_srcs_;

    int cmask_a_ = INT_MIN;
    int cmask_b_ = INT_MIN;
    int cmask_c_ = INT_MIN;

    const int mask_scalar = 1 << 0;
    const int mask_per_oc = 1 << 1;
    const int mask_per_ic = 1 << 2;

    const int idx_a = DNNL_ARG_WEIGHTS;
    memory_desc_t prelu_wei_md;
    // Kernel-tuned; may differ from desc()->lda()/ldb()/transb().
    dim_t lda_ = 0, ldb_ = 0;
    bool transb_ = false;
    bool with_sround_ = false;
    bool with_mx_scale_ = false;
    compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

    float alpha() const {
        const auto &scales = attr()->scales_;
        bool host_scales_by_alpha
                = scales.get(gemm_arg::A).is_host_scalar()
                || scales.get(gemm_arg::B).is_host_scalar()
                || (scales.get(gemm_arg::C).is_host_scalar()
                        && attr()->post_ops_.len() == 0
                        && !bias_via_binary_);
        // Bogus non-one value for host scalar.
        // Actual value will be passed on execution step
        if (host_scales_by_alpha) return 9.99f;
        return 1.0f;
    }

    float beta() const { return beta_; }

    bool with_bias() const {
        return desc()->bias_type() != data_type::undef && !bias_via_binary_;
    }

    bool bias_via_binary() const { return bias_via_binary_; }

    int bias_cmask() const {
        unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        assert(unsigned(desc()->bias_mask()) < 8);
        return with_bias() ? to_cmask[desc()->bias_mask() & 7] : -1;
    }

    sum_ab_t sum_ab() const { return desc()->sum_ab(); }

    bool a_zp_2d() const { return a_zp_ndims() >= 2; }
    bool b_zp_2d() const { return b_zp_ndims() >= 2; }

    bool a_gs_2d() const { return a_gs_ndims() >= 2; }
    bool b_gs_2d() const { return b_gs_ndims() >= 2; }

    bool with_sum_ab() const { return sum_ab() != sum_ab::sum_none; }

    int sum_ab_cmask() const {
        switch (sum_ab()) {
            default:
            case sum_ab::sum_none: return 0;
            case sum_ab::sum_a_row: return 1;
            case sum_ab::sum_b_col: return 2;
        }
    }
    bool with_a_scales() const { return a_scale_ndims() >= 0; }
    bool with_b_scales() const { return b_scale_ndims() >= 0; }
    bool with_c_scales() const {
        return !attr()->scales_.has_default_values(gemm_arg::C);
    }

    bool with_a_zero_points() const { return a_zp_ndims() >= 0; }
    bool with_b_zero_points() const { return b_zp_ndims() >= 0; }
    bool with_c_zero_points() const {
        return !attr()->zero_points_.has_default_values(gemm_arg::C);
    }

    bool with_a_group_sums() const { return a_gs_ndims() >= 0; }
    bool with_b_group_sums() const { return b_gs_ndims() >= 0; }

    bool with_sround() const { return with_sround_; }
    bool with_mx_scale() const { return with_mx_scale_; }

    bool a_scales_2d() const { return a_scale_ndims() > 1; }
    bool b_scales_2d() const { return b_scale_ndims() > 1; }
    bool c_scales_2d() const { return c_scale_ndims() > 1; }

    bool dy_quant_enabled();
    bool wei_decomp();
    bool quant_enabled();

    int batch_dims() const { return nstl::max(desc()->c_md().ndims - 2, 0); }
    bool trans_a() const { return desc()->transa() == dnnl_trans; }
    bool trans_b() const { return transb_; }
    bool trans_bias() const { return desc()->trans_bias() == dnnl_trans; }

    dim_t ld(int arg) const {
        if (arg == gemm_arg::A) return lda_;
        if (arg == gemm_arg::B) return ldb_;
        if (arg == gemm_arg::C) return desc()->ldc();
        gpu_error_not_expected();
        return 0;
    }
    dim_t stride(int arg, int dim) const {
        if (arg == gemm_arg::A) return desc()->stride_a(dim);
        if (arg == gemm_arg::B) return desc()->stride_b(dim);
        if (arg == gemm_arg::C) return desc()->stride_c(dim);
        gpu_error_not_expected();
        return 0;
    }
    data_type_t get_type(int arg) const {
        if (arg == gemm_arg::A) return desc()->a_type();
        if (arg == gemm_arg::B) return desc()->b_type();
        if (arg == gemm_arg::C) return desc()->c_type();
        gpu_error_not_expected();
        return data_type::undef;
    }

    dim_t scale_stride(int idx, int arg) const;
    dim_t zp_stride(int idx, int arg) const;
    dim_t gs_stride(int idx, int arg) const;
    bool grouped(int arg) const {
        switch (arg) {
            case gemm_arg::A: {
                bool k = 1 < a_group_k_ && a_group_k_ < desc()->k();
                bool m = 1 < a_group_m_ && a_group_m_ < desc()->m();
                return k || m;
            }
            case gemm_arg::B: {
                bool k = 1 < b_group_k_ && b_group_k_ < desc()->k();
                bool n = 1 < b_group_n_ && b_group_n_ < desc()->n();
                return k || n;
            }
            case gemm_arg::C: {
                bool m = 1 < c_group_m_ && c_group_m_ < desc()->m();
                bool n = 1 < c_group_n_ && c_group_n_ < desc()->n();
                return m || n;
            }
            default: gpu_error_not_expected(); return false;
        }
    }
    bool a_grouped() const { return grouped(gemm_arg::A); }
    bool b_grouped() const { return grouped(gemm_arg::B); }

    bool zp_host_scalar(int arg) const {
        return attr()->zero_points_.get(arg).is_host_scalar();
    }
    bool a_zp_host_scalar() const { return zp_host_scalar(gemm_arg::A); }
    bool b_zp_host_scalar() const { return zp_host_scalar(gemm_arg::B); }
    bool c_zp_host_scalar() const { return zp_host_scalar(gemm_arg::C); }

    data_type_t a_scale_dt() const {
        return attr()->scales_.get(gemm_arg::A).get_data_type();
    }
    data_type_t b_scale_dt() const {
        return attr()->scales_.get(gemm_arg::B).get_data_type();
    }
    data_type_t c_scale_dt() const {
        return attr()->scales_.get(gemm_arg::C).get_data_type();
    }
    data_type_t a_zp_dt() const {
        return attr()->zero_points_.get(gemm_arg::A).get_data_type();
    }
    data_type_t b_zp_dt() const {
        return attr()->zero_points_.get(gemm_arg::B).get_data_type();
    }
    data_type_t c_zp_dt() const {
        return attr()->zero_points_.get(gemm_arg::C).get_data_type();
    }
    data_type_t a_gs_dt() const {
        return attr()->precomputed_reductions_.get(gemm_arg::A).get_data_type();
    }
    data_type_t b_gs_dt() const {
        return attr()->precomputed_reductions_.get(gemm_arg::B).get_data_type();
    }
    int a_zp_ndims() const {
        return quant_entry_ndims(attr()->zero_points_.get(gemm_arg::A),
                a_zp_md_, desc()->c_md().ndims - 1);
    }
    int b_zp_ndims() const {
        return quant_entry_ndims(attr()->zero_points_.get(gemm_arg::B),
                b_zp_md_, desc()->c_md().ndims - 2);
    }
    int c_zp_ndims() const {
        return quant_entry_ndims(
                attr()->zero_points_.get(gemm_arg::C), c_zp_md_, -1);
    }
    int a_gs_ndims() const {
        return quant_entry_ndims(
                attr()->precomputed_reductions_.get(gemm_arg::A), a_gs_md_,
                desc()->c_md().ndims - 1);
    }
    int b_gs_ndims() const {
        return quant_entry_ndims(
                attr()->precomputed_reductions_.get(gemm_arg::B), b_gs_md_,
                desc()->c_md().ndims - 2);
    }
    int a_scale_ndims() const {
        if (a_scale_ndims_override_ != INT_MIN) return a_scale_ndims_override_;
        return quant_entry_ndims(attr()->scales_.get(gemm_arg::A), a_scale_md_,
                desc()->c_md().ndims - 1);
    }
    int b_scale_ndims() const {
        if (b_scale_ndims_override_ != INT_MIN) return b_scale_ndims_override_;
        return quant_entry_ndims(attr()->scales_.get(gemm_arg::B), b_scale_md_,
                desc()->c_md().ndims - 2);
    }
    int c_scale_ndims() const {
        return quant_entry_ndims(
                attr()->scales_.get(gemm_arg::C), c_scale_md_, -1);
    }
    bool a_force_gs() const {
        return !attr()->precomputed_reductions_.has_default_values(gemm_arg::A);
    }
    bool b_force_gs() const {
        return !attr()->precomputed_reductions_.has_default_values(gemm_arg::B);
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

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
