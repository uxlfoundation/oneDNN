/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_JIT_GEMM_PD_HPP
#define GPU_INTEL_JIT_GEMM_JIT_GEMM_PD_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/tag_traits.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/intel/gpu_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

#define GEMM_MAX_PO 36

struct jit_gemm_pd_t : public gpu_gemm_pd_t {
    using gpu_gemm_pd_t::gpu_gemm_pd_t;

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
    void init_attrs();
    bool scales_ok();
    bool zp_ok();

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
    int wei_q2d_group_k_ = 0;
    int src_q2d_group_k_ = 0;
    bool src_po_sc_ = false;
    data_type_t wei_scales_type_ = data_type::undef;
    data_type_t src_scales_type_ = data_type::undef;

    int ao_dims_ = -1, bo_dims_ = -1;
    int asc_dims_ = -1, bsc_dims_ = -1;
    post_ops_t post_ops_;
    std::vector<binary_src_t> binary_srcs_;

    int zp_group_k_a_ = -1;
    int zp_group_k_b_ = -1;

    int cmask_a_ = INT_MIN;
    int cmask_b_ = INT_MIN;
    int cmask_c_ = INT_MIN;

    int src_scales_group_k_ = -1;
    int wei_scales_group_k_ = -1;

    const int mask_scalar = 1 << 0;
    const int mask_per_oc = 1 << 1;
    const int mask_per_ic = 1 << 2;

    const int idx_a = DNNL_ARG_WEIGHTS;
    memory_desc_t wei_scales_md, src_scales_md, c_scales_md, prelu_wei_md;
    bool swap_ab_ = false;
    bool a_zp_ = false, b_zp_ = false;
    dim_t eff_lda_ = 0, eff_ldb_ = 0;
    bool eff_transa_ = false, eff_transb_ = false;
    bool with_sround_ = false;

    float alpha() const { return 1.0f; }

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

    bool wei_zp_2d() const { return ao_dims_ == 2; }
    bool src_zp_2d() const { return bo_dims_ == 2; }

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
    bool with_sround() const { return with_sround_; }

    bool wei_scales_2d() const { return asc_dims_ > 1; }
    bool src_scales_2d() const { return bsc_dims_ > 1; }

    bool quant_entry_2d(int arg, const quant_entries_t &entry) const;
    int quant_entry_ndims(
            const quant_entry_t &entry, const memory_desc_t &md) const;
    int quant_entry_group_prod(const quant_entry_t &attr) const;

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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
