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
#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/post_ops.hpp"
// @@@@@@@@@@@@@@@@@@
#include "gpu/intel/gemm/exec_types.hpp"
// @@@@@@@@@@@@@@@@@@

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

#define GEMM_MAX_PO 36

struct pd_t : public gemm::pd_t {
    using gemm::pd_t::pd_t;

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

    status_t init_post_ops();
    status_t init_attrs();
    bool scales_ok();
    bool zp_ok();
    bool gs_ok();

    dim_t ld_binary(int idx) const;
    dim_t stride_binary(int idx, int stride = 0) const;

    const post_ops_t *post_ops() const { return &post_ops_; }
    const std::vector<binary_src_t> &binary_srcs() const {
        return binary_srcs_;
    }
    bool valid_2d_mask(int mask, int ndims, bool per_tensor_ok = true);

    float beta_ = 0.0f;

    bool with_sum_ = false;
    bool sum_at_begin_ = false;

    bool bias_via_binary_ = false;
    bool wei_decomp_ = false;
    bool dy_quant_enabled_ = false;
    bool quant_enabled_ = false;

    // @@@@@@@
    //bool a_zp_host_scalar_ = false;
    //bool b_zp_host_scalar_ = false;
    //bool c_zp_host_scalar_ = false;
    // @@@@@@@
    int a_scales_group_k_ = 0, a_scales_group_m_ = 0;
    int b_scales_group_k_ = 0, b_scales_group_n_ = 0;
    int c_scales_group_m_ = 0, c_scales_group_n_ = 0;
    int a_zp_group_k_ = 0, a_zp_group_m_ = 0;
    int b_zp_group_k_ = 0, b_zp_group_n_ = 0;
    int a_gs_group_k_ = 0, a_gs_group_m_ = 0;
    int b_gs_group_k_ = 0, b_gs_group_n_ = 0;
    bool non_scale_po_ = false;
    data_type_t a_scales_type_ = data_type::undef;
    data_type_t b_scales_type_ = data_type::undef;
    data_type_t c_scales_type_ = data_type::undef;

    int ao_dims_ = -1, bo_dims_ = -1;
    int ag_dims_ = -1, bg_dims_ = -1;
    int asc_dims_ = -1, bsc_dims_ = -1, csc_dims_ = -1;
    post_ops_t post_ops_;
    std::vector<binary_src_t> binary_srcs_;

    int cmask_a_ = INT_MIN;
    int cmask_b_ = INT_MIN;
    int cmask_c_ = INT_MIN;

    const int mask_scalar = 1 << 0;
    const int mask_per_oc = 1 << 1;
    const int mask_per_ic = 1 << 2;

    const int idx_a = DNNL_ARG_WEIGHTS;
    memory_desc_t prelu_wei_md, a_scale_md_, b_scale_md_, c_scale_md_;
    memory_desc_t a_zp_md_, b_zp_md_;
    memory_desc_t a_gs_md_, b_gs_md_;
    bool swap_ab_ = false;
    dim_t eff_lda_ = 0, eff_ldb_ = 0;
    bool eff_transa_ = false, eff_transb_ = false;
    bool with_sround_ = false;
    bool with_mx_scale_ = false;

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

    float beta() const { return beta_; }

    bool with_bias() const {
        return desc()->bias_type() != data_type::undef && !bias_via_binary_;
    }

    int bias_cmask() const {
        unsigned char to_cmask[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        assert(unsigned(desc()->bias_mask()) < 8);
        return with_bias() ? to_cmask[desc()->bias_mask() & 7] : -1;
    }

    sum_ab_t sum_ab() const { return desc()->sum_ab; }
    sum_ab_t eff_sum_ab() const {
        if (swap_ab() && sum_ab() == sum_ab::sum_a_row)
            return sum_ab::sum_b_col;
        if (swap_ab() && sum_ab() == sum_ab::sum_b_col)
            return sum_ab::sum_a_row;
        return sum_ab();
    }

    bool a_zp_2d() const { return ao_dims_ >= 2; }
    bool b_zp_2d() const { return bo_dims_ >= 2; }

    bool a_gs_2d() const { return ag_dims_ >= 2; }
    bool b_gs_2d() const { return bg_dims_ >= 2; }

    bool with_sum_ab() const { return sum_ab() != sum_ab::sum_none; }

    int sum_ab_cmask() const {
        switch (eff_sum_ab()) {
            default:
            case sum_ab::sum_none: return 0;
            case sum_ab::sum_a_row: return 1;
            case sum_ab::sum_b_col: return 2;
        }
    }
    bool with_a_scales() const { return (asc_dims_ >= 0); }
    bool with_b_scales() const { return (bsc_dims_ >= 0); }
    bool with_c_scales() const {
        return !attr()->scales_.has_default_values(DNNL_ARG_DST);
    }

    bool with_a_zero_points() const { return (ao_dims_ >= 0); }
    bool with_b_zero_points() const { return (bo_dims_ >= 0); }
    bool with_c_zero_points() const {
        return !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    }

    bool with_a_group_sums() const { return (ag_dims_ >= 0); }
    bool with_b_group_sums() const { return (bg_dims_ >= 0); }

    bool with_sround() const { return with_sround_; }
    bool with_mx_scale() const { return with_mx_scale_; }

    bool a_scales_2d() const { return asc_dims_ > 1; }
    bool b_scales_2d() const { return bsc_dims_ > 1; }
    bool c_scales_2d() const { return csc_dims_ > 1; }

    bool dy_quant_enabled();
    bool wei_decomp();
    bool quant_enabled();

    bool swap_ab() const { return swap_ab_; }

    int batch_dims() const { return nstl::max(desc()->c_desc.ndims - 2, 0); }
    bool eff_transa() const { return eff_transa_; }
    bool eff_transb() const { return eff_transb_; }
    bool eff_trans_bias() const {
        return swap_ab() ? (desc()->trans_bias() == dnnl_notrans)
                         : (desc()->trans_bias() == dnnl_trans);
    }
    dim_t eff_m() const { return !swap_ab() ? desc()->m() : desc()->n(); }
    dim_t eff_n() const { return !swap_ab() ? desc()->n() : desc()->m(); }
    dim_t eff_lda() const { return eff_lda_; }
    dim_t eff_ldb() const { return eff_ldb_; }
    dim_t eff_stride_a(int dim) const {
        return !swap_ab() ? desc()->stride_a(dim) : desc()->stride_b(dim);
    }
    dim_t eff_stride_b(int dim) const {
        return !swap_ab() ? desc()->stride_b(dim) : desc()->stride_a(dim);
    }
    data_type_t eff_a_type() const {
        return !swap_ab() ? desc()->a_type() : desc()->b_type();
    }
    data_type_t eff_b_type() const {
        return !swap_ab() ? desc()->b_type() : desc()->a_type();
    }
    dim_t eff_scale_stride(int idx, int arg) const;
    dim_t eff_zp_stride(int idx, int arg) const;
    dim_t eff_gs_stride(int idx, int arg) const;
    bool a_scales_grouped() const {
        bool k_grouped
                = 1 < a_scales_group_k_ && a_scales_group_k_ < desc()->k();
        bool m_grouped
                = 1 < a_scales_group_m_ && a_scales_group_m_ < desc()->m();
        return k_grouped || m_grouped;
    }
    bool b_scales_grouped() const {
        bool k_grouped
                = 1 < b_scales_group_k_ && b_scales_group_k_ < desc()->k();
        bool n_grouped
                = 1 < b_scales_group_n_ && b_scales_group_n_ < desc()->n();
        return k_grouped || n_grouped;
    }
    bool a_zp_grouped() const {
        bool k_grouped = 1 < a_zp_group_k_ && a_zp_group_k_ < desc()->k();
        bool m_grouped = 1 < a_zp_group_m_ && a_zp_group_m_ < desc()->m();
        return k_grouped || m_grouped;
    }
    bool b_zp_grouped() const {
        bool k_grouped = 1 < b_zp_group_k_ && b_zp_group_k_ < desc()->k();
        bool n_grouped = 1 < b_zp_group_n_ && b_zp_group_n_ < desc()->n();
        return k_grouped || n_grouped;
    }
    // @@@@@@@@@@@@@@@@@@@@@@@
    bool a_zp_host_scalar() const {
        auto &attr_zps = attr()->zero_points_;
        auto &zp = attr_zps.get(DNNL_ARG_A);
        return zp.is_host_scalar();
    }
    bool b_zp_host_scalar() const {
        auto &attr_zps = attr()->zero_points_;
        auto &zp = attr_zps.get(DNNL_ARG_B);
        return zp.is_host_scalar();
    }
    // @@@@@@@@@@@@@@@@@@@@@@@
    int a_q2d_group_k() const {
        if (a_zp_2d()) {
            return a_zp_group_k_;
        } else if (a_scales_2d()) {
            return a_scales_group_k_;
        } else if (with_a_group_sums()) {
            return a_gs_group_k_;
        }
        return 0;
    }
    int a_q2d_group_m() const {
        if (a_zp_2d()) {
            return a_zp_group_m_;
        } else if (a_scales_2d()) {
            return a_scales_group_m_;
        } else if (with_a_group_sums()) {
            return a_gs_group_m_;
        }
        return 0;
    }
    int b_q2d_group_k() const {
        if (b_zp_2d()) {
            return b_zp_group_k_;
        } else if (b_scales_2d()) {
            return b_scales_group_k_;
        } else if (with_b_group_sums()) {
            return b_gs_group_k_;
        }
        return 0;
    }
    int b_q2d_group_n() const {
        if (b_zp_2d()) {
            return b_zp_group_n_;
        } else if (b_scales_2d()) {
            return b_scales_group_n_;
        } else if (with_b_group_sums()) {
            return b_gs_group_n_;
        }
        return 0;
    }
    int c_q2d_group_m() const {
        if (c_scales_2d() || with_mx_scale()) { return c_scales_group_m_; }
        return 0;
    }
    int c_q2d_group_n() const {
        if (c_scales_2d() || with_mx_scale()) { return c_scales_group_n_; }
        return 0;
    }
    int eff_align_a() const {
        auto dt = eff_a_type();
        auto align
                = utils::max_pow2_div(types::elements_to_bytes(dt, eff_lda()));
        for (int b = 0; b < batch_dims(); b++) {
            auto stride_bytes = utils::max_pow2_div(
                    types::elements_to_bytes(dt, eff_stride_a(b)));
            align = (stride_bytes ? nstl::min(align, stride_bytes) : align);
        }
        return int(align);
    }
    int eff_align_b() const {
        auto dt = eff_b_type();
        auto align
                = utils::max_pow2_div(types::elements_to_bytes(dt, eff_ldb()));
        for (int b = 0; b < batch_dims(); b++) {
            auto stride_bytes = utils::max_pow2_div(
                    types::elements_to_bytes(dt, eff_stride_b(b)));
            align = (stride_bytes ? nstl::min(align, stride_bytes) : align);
        }
        return int(align);
    }
    int align_c() const {
        auto dt = desc()->c_type();
        auto align = utils::max_pow2_div(
                types::elements_to_bytes(dt, desc()->ldc()));
        for (int b = 0; b < batch_dims(); b++)
            align = nstl::min(align,
                    utils::max_pow2_div(
                            types::elements_to_bytes(dt, desc()->stride_c(b))));
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
