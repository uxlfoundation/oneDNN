/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef COMMON_GEMM_PD_HPP
#define COMMON_GEMM_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/gemm_arg.hpp"
#include "common/gemm_types.hpp"
#include "common/memory_desc.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_attr_quant.hpp"
#include "common/primitive_desc.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VDISPATCH_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, gemm, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_GEMM_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, gemm, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

#define VDISPATCH_GEMM_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, gemm, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

// NOLINTBEGIN(google-default-arguments)
struct gemm_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::gemm;

    using base_class = gemm_pd_t;
    using hint_class = gemm_pd_t;

    const gemm_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC_0: return src_md(0);
            case DNNL_ARG_SRC_1: return src_md(1);
            case DNNL_ARG_BIAS: return src_md(2);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.a_md();
            case 1: return &desc_.b_md();
            case 2: return &desc_.bias_md();
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.c_md() : &glob_zero_md;
    }
    bool with_bias() const { return desc_.bias_md().ndims != 0; }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }
    int ndims() const { return desc_.c_md().ndims; }

    int full_tensor_mask() const { return (1 << ndims()) - 1; }

    const memory_desc_t &a_scale_md() const { return a_scale_md_; }
    const memory_desc_t &b_scale_md() const { return b_scale_md_; }
    const memory_desc_t &c_scale_md() const { return c_scale_md_; }
    const memory_desc_t &a_zp_md() const { return a_zp_md_; }
    const memory_desc_t &b_zp_md() const { return b_zp_md_; }
    const memory_desc_t &c_zp_md() const { return c_zp_md_; }
    const memory_desc_t &a_gs_md() const { return a_gs_md_; }
    const memory_desc_t &b_gs_md() const { return b_gs_md_; }

    // Promote prelu to binary; tagged alg=eltwise_relu. Idempotent.
    void canonicalize_post_ops() {
        const int nd = desc_.c_md().ndims;
        for (int i = 0; i < attr_.post_ops_.len(); ++i) {
            auto &e = attr_.post_ops_.entry_[i];
            if (!e.is_prelu()) continue;
            const int mask = e.prelu.mask;
            dims_t weight_dims {};
            for (int d = 0; d < nd; ++d)
                weight_dims[d] = ((mask >> d) & 0x1) ? desc_.c_md().dims[d] : 1;
            format_tag_t tag = format_tag::undef;
            switch (nd) {
                case 1: tag = format_tag::a; break;
                case 2: tag = format_tag::ab; break;
                case 3: tag = format_tag::acb; break;
                case 4: tag = format_tag::acdb; break;
                case 5: tag = format_tag::acdeb; break;
                default: break;
            }
            // For nd outside {1..5} (e.g. nd==0 or nd>5 reachable via matmul
            // with DNNL_MAX_NDIMS), tag remains undef and
            // memory_desc_init_by_tag would silently return zero_md. Skip the
            // entry instead of corrupting it.
            if (tag == format_tag::undef) continue;
            memory_desc_t src1 {};
            if (memory_desc_init_by_tag(
                        src1, nd, weight_dims, data_type::f32, tag)
                    != status::success)
                continue;
            e.kind = primitive_kind::binary;
            e.binary.alg = alg_kind::eltwise_relu;
            e.binary.src1_desc = src1;
            e.binary.user_src1_desc = src1;
            e.binary.src2_desc = memory_desc_t {};
            e.binary.user_src2_desc = memory_desc_t {};
        }
    }

    void apply_swap_ab() {
        canonicalize_post_ops();

        const int nd = desc_.c_md().ndims;

        desc_.apply_swap_ab();

        attr_.scales_.swap_entries(gemm_arg::A, gemm_arg::B);
        attr_.zero_points_.swap_entries(gemm_arg::A, gemm_arg::B);
        attr_.precomputed_reductions_.swap_entries(gemm_arg::A, gemm_arg::B);

        for (int arg : {gemm_arg::A, gemm_arg::B, gemm_arg::C}) {
            swap_entry_mn_axes(attr_.scales_, arg, nd);
            swap_entry_mn_axes(attr_.zero_points_, arg, nd);
            swap_entry_mn_axes(attr_.precomputed_reductions_, arg, nd);
        }

        for (int i = 0; i < attr_.post_ops_.len(); ++i) {
            auto &e = attr_.post_ops_.entry_[i];
            if (e.is_binary()) {
                gemm_desc_t::transpose_mn_axes(e.binary.src1_desc);
                gemm_desc_t::transpose_mn_axes(e.binary.user_src1_desc);
                if (e.is_binary_with_ternary_op()) {
                    gemm_desc_t::transpose_mn_axes(e.binary.src2_desc);
                    gemm_desc_t::transpose_mn_axes(e.binary.user_src2_desc);
                }
            }
        }

        init_quant_mds();
    }

protected:
    // Note: desc is not copied locally to avoid overheads; user mds are
    // lost when 'any' tags are resolved.
    gemm_desc_t desc_;

    gemm_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<gemm_desc_t>(adesc)) {}

    // By default, we just resolve 'any' with blocked layout and trivial strides
    bool set_default_format(memory_desc_t *md) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) {
            if (mdw.has_runtime_dims_or_strides()) return false;
            status_t status = memory_desc_init_by_strides(*md, nullptr);
            if (status != status::success) return false;
        }

        return true;
    }

    bool set_default_formats() {
        bool ok = true;

        for (auto md : {&desc_.a_md(), &desc_.b_md(), &desc_.bias_md(),
                     &desc_.c_md()}) {
            ok = ok && set_default_format(md);
        }

        auto status = attr_.post_ops_.set_default_formats(&desc_.c_md());
        ok = ok && (status == status::success);

        if (ok) init_quant_mds();
        return ok;
    }

    void init_quant_mds() {
        const auto &scales = attr_.scales_;
        const auto &zps = attr_.zero_points_;
        const auto &gs = attr_.precomputed_reductions_;
        (void)scales.get(gemm_arg::A).get_md(a_scale_md_, desc_.a_md());
        (void)scales.get(gemm_arg::B).get_md(b_scale_md_, desc_.b_md());
        (void)scales.get(gemm_arg::C).get_md(c_scale_md_, desc_.c_md());
        (void)zps.get(gemm_arg::A).get_md(a_zp_md_, desc_.a_md());
        (void)zps.get(gemm_arg::B).get_md(b_zp_md_, desc_.b_md());
        (void)zps.get(gemm_arg::C).get_md(c_zp_md_, desc_.c_md());
        (void)gs.get(gemm_arg::A).get_md(a_gs_md_, desc_.a_md());
        (void)gs.get(gemm_arg::B).get_md(b_gs_md_, desc_.b_md());
    }

    static int swap_mask_bits(int mask, int i, int j) {
        if (i < 0 || j < 0 || i == j) return mask;
        const int lo = (mask >> i) & 1;
        const int hi = (mask >> j) & 1;
        if (lo != hi) {
            mask ^= (1 << i);
            mask ^= (1 << j);
        }
        return mask;
    }

    static void swap_entry_mn_axes(quant_entries_t &q, int arg, int ndims) {
        if (q.has_default_values(arg)) return;
        q.set_mask(arg, swap_mask_bits(q.get_mask(arg), ndims - 2, ndims - 1));
        q.swap_group_dims(arg, 0, 1);
    }

    memory_desc_t a_scale_md_ {}, b_scale_md_ {}, c_scale_md_ {};
    memory_desc_t a_zp_md_ {}, b_zp_md_ {}, c_zp_md_ {};
    memory_desc_t a_gs_md_ {}, b_gs_md_ {};
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif
