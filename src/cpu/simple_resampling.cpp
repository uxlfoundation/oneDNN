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

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace resampling_utils;

simple_resampling_kernel_t::simple_resampling_kernel_t(
        const resampling_pd_t *pd)
    : pd_(pd), are_postops_set_(!(pd_->attr()->post_ops_.entry_.empty())) {
    if (pd_->is_fwd()) {
        const memory_desc_wrapper src_d(pd_->src_md());
        inner_stride_ = src_d.blocking_desc().strides[pd_->ndims() - 1];
        nsp_outer_ = src_d.nelems(true)
                / (pd_->ID() * pd_->IH() * pd_->IW() * inner_stride_);
        stride_d_ = pd_->IH() * pd_->IW() * inner_stride_;
        stride_h_ = pd_->IW() * inner_stride_;
        stride_w_ = inner_stride_;
        tail_size_ = pd_->C() % inner_stride_;
        src_dt_ = pd_->src_md()->data_type;
        dst_dt_ = pd_->dst_md()->data_type;
    } else {
        const memory_desc_wrapper diff_src_d(pd_->diff_src_md());
        inner_stride_ = diff_src_d.blocking_desc().strides[pd_->ndims() - 1];
        nsp_outer_ = diff_src_d.nelems(true)
                / (pd_->ID() * pd_->IH() * pd_->IW() * inner_stride_);
        stride_d_ = pd_->OH() * pd_->OW() * inner_stride_;
        stride_h_ = pd_->OW() * inner_stride_;
        stride_w_ = inner_stride_;
        tail_size_ = pd_->C() % inner_stride_;
        src_dt_ = pd_->diff_src_md()->data_type;
        dst_dt_ = pd_->diff_dst_md()->data_type;
    }
}

status_t simple_resampling_kernel_t::init() {
    if (pd_->desc()->alg_kind == alg_kind::resampling_nearest)
        interpolate_fn_ = create_nearest();
    else {
        if (pd_->ndims() == 5)
            interpolate_fn_ = create_trilinear();
        else if (pd_->ndims() == 4)
            interpolate_fn_ = create_bilinear();
        else
            interpolate_fn_ = create_linear();

        fill_coeffs();
        if (!pd_->is_fwd()) fill_weights();
    }
    ref_post_ops_ = utils::make_unique<ref_post_ops_t>(pd_->attr()->post_ops_);
    if (!ref_post_ops_) return status::out_of_memory;
    CHECK(ref_post_ops_->init(pd_->dst_md()));

    return status::success;
}

status_t simple_resampling_kernel_t::execute(const exec_ctx_t &ctx) const {
    const int OD = pd_->OD();
    const int OH = pd_->OH();
    const int OW = pd_->OW();
    const int ID = pd_->ID();
    const int IH = pd_->IH();
    const int IW = pd_->IW();
    const int NB_CH = utils::div_up(pd_->C(), inner_stride_);

    if (pd_->is_fwd()) {
        const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
        auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

        parallel_nd(nsp_outer_, OD, OH,
                [= COMPAT_THIS_CAPTURE](dim_t nsp0, dim_t od, dim_t oh) {
            const bool preserve_zero_padding
                    = (nsp0 + 1) % NB_CH == 0 && tail_size_ != 0;

            for (dim_t ow = 0; ow < OW; ow++) {
                const dim_t src_off = nsp0 * ID * IH * IW * inner_stride_;
                const dim_t dst_off
                        = (nsp0 * OD * OH * OW + od * OH * OW + oh * OW + ow)
                        * inner_stride_;
                interpolate_fn_(src + src_off * types::data_type_size(src_dt_),
                        dst + dst_off * types::data_type_size(dst_dt_), ctx,
                        dst_off, od, oh, ow, preserve_zero_padding);
            }
        });
    } else {
        const auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
        auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

        parallel_nd(nsp_outer_, ID, IH, IW,
                [= COMPAT_THIS_CAPTURE](
                        dim_t nsp, dim_t id, dim_t ih, dim_t iw) {
            const dim_t diff_dst_off = nsp * OD * OH * OW * inner_stride_;
            const dim_t diff_src_off
                    = (nsp * ID * IH * IW + id * IH * IW + ih * IW + iw)
                    * inner_stride_;
            interpolate_fn_(
                    diff_dst + diff_dst_off * types::data_type_size(dst_dt_),
                    diff_src + diff_src_off * types::data_type_size(src_dt_),
                    /* unused */ ctx, /* unused */ 0, id, ih, iw, false);
        });
    }

    return status::success;
}

void simple_resampling_kernel_t::fill_coeffs() {
    if (pd_->is_fwd()) {
        linear_coeffs_.reserve(pd_->OD() + pd_->OH() + pd_->OW());
        for (dim_t od = 0; od < pd_->OD(); od++)
            linear_coeffs_.emplace_back(od, pd_->OD(), pd_->ID());
        for (dim_t oh = 0; oh < pd_->OH(); oh++)
            linear_coeffs_.emplace_back(oh, pd_->OH(), pd_->IH());
        for (dim_t ow = 0; ow < pd_->OW(); ow++)
            linear_coeffs_.emplace_back(ow, pd_->OW(), pd_->IW());
    } else {
        bwd_linear_coeffs_.reserve(pd_->ID() + pd_->IH() + pd_->IW());
        for (dim_t id = 0; id < pd_->ID(); id++)
            bwd_linear_coeffs_.emplace_back(id, pd_->OD(), pd_->ID());
        for (dim_t ih = 0; ih < pd_->IH(); ih++)
            bwd_linear_coeffs_.emplace_back(ih, pd_->OH(), pd_->IH());
        for (dim_t iw = 0; iw < pd_->IW(); iw++)
            bwd_linear_coeffs_.emplace_back(iw, pd_->OW(), pd_->IW());
    }
}

void simple_resampling_kernel_t::fill_weights() {
    assert(!pd_->is_fwd() && "The function is used in bwd path only.");

    using namespace resampling_utils;
    bwd_linear_weights_.reserve(2 * (pd_->OD() + pd_->OH() + pd_->OW()));
    for (dim_t od = 0; od < pd_->OD(); od++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, od, pd_->OD(), pd_->ID()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, od, pd_->OD(), pd_->ID()));
    }
    for (dim_t oh = 0; oh < pd_->OH(); oh++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, oh, pd_->OH(), pd_->IH()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, oh, pd_->OH(), pd_->IH()));
    }
    for (dim_t ow = 0; ow < pd_->OW(); ow++) {
        bwd_linear_weights_.emplace_back(
                linear_weight(0, ow, pd_->OW(), pd_->IW()));
        bwd_linear_weights_.emplace_back(
                linear_weight(1, ow, pd_->OW(), pd_->IW()));
    }
}

simple_resampling_kernel_t::interpolate_fn_t
simple_resampling_kernel_t::create_nearest() const {
    if (pd_->is_fwd()) {
        return [&](const char *src, char *dst, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t od, dim_t oh, dim_t ow,
                       const bool preserve_zero_padding) {
            const dim_t id = nearest_idx(od, pd_->OD(), pd_->ID());
            const dim_t ih = nearest_idx(oh, pd_->OH(), pd_->IH());
            const dim_t iw = nearest_idx(ow, pd_->OW(), pd_->IW());
            const dim_t offset
                    = id * stride_d_ + ih * stride_h_ + iw * stride_w_;
            const char *src_ptr = src + offset * types::data_type_size(src_dt_);

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res
                        = io::load_float_value(src_dt_, src_ptr, innermost_el);

                if (are_postops_set_
                        && IMPLICATION(preserve_zero_padding,
                                innermost_el < tail_size_)) {
                    ref_post_ops_t::args_t args;
                    args.dst_val
                            = io::load_float_value(dst_dt_, dst, innermost_el);
                    args.ctx = &ctx;
                    args.l_offset = dst_off + innermost_el;
                    args.dst_md = pd_->dst_md();
                    ref_post_ops_->execute(res, args);
                }

                io::store_float_value(dst_dt_, res, dst, innermost_el);
            }
        };
    } else {
        return [&](const char *diff_dst, char *diff_src, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t id, dim_t ih, dim_t iw,
                       const bool preserve_zero_padding) {
            auto ow_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OW() / pd_->IW()) - 0.5f);
            };
            auto oh_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OH() / pd_->IH()) - 0.5f);
            };
            auto od_idx = [&](const float in_idx) -> dim_t {
                return ceil_idx((in_idx * pd_->OD() / pd_->ID()) - 0.5f);
            };
            MAYBE_UNUSED(preserve_zero_padding);

            const dim_t ow_start = ow_idx(iw) * stride_w_;
            const dim_t oh_start = oh_idx(ih) * stride_h_;
            const dim_t od_start = od_idx(id) * stride_d_;
            const dim_t ow_end = ow_idx(iw + 1.f) * stride_w_;
            const dim_t oh_end = oh_idx(ih + 1.f) * stride_h_;
            const dim_t od_end = od_idx(id + 1.f) * stride_d_;

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(dim_t od = od_start; od < od_end; od += stride_d_)
                for_(dim_t oh = oh_start; oh < oh_end; oh += stride_h_)
                for (dim_t ow = ow_start; ow < ow_end; ow += stride_w_) {
                    sum += io::load_float_value(
                            dst_dt_, diff_dst, od + oh + ow + innermost_el);
                }
                io::store_float_value(src_dt_, sum, diff_src, innermost_el);
            }
        };
    }
}

simple_resampling_kernel_t::interpolate_fn_t
simple_resampling_kernel_t::create_linear() const {
    if (pd_->is_fwd()) {
        return [&](const char *src, char *dst, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t od, dim_t oh, dim_t ow,
                       const bool preserve_zero_padding) {
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for (int k = 0; k < 2; k++) {
                    const dim_t src_off = iw.idx[k] * stride_w_ + innermost_el;
                    const float s = io::load_float_value(src_dt_, src, src_off);
                    res += s * iw.wei[k];
                }

                if (are_postops_set_
                        && IMPLICATION(preserve_zero_padding,
                                innermost_el < tail_size_)) {
                    ref_post_ops_t::args_t args;
                    args.dst_val
                            = io::load_float_value(dst_dt_, dst, innermost_el);
                    args.ctx = &ctx;
                    args.l_offset = dst_off + innermost_el;
                    args.dst_md = pd_->dst_md();
                    ref_post_ops_->execute(res, args);
                }

                io::store_float_value(dst_dt_, res, dst, innermost_el);
            }
        };
    } else {
        return [&](const char *diff_dst, char *diff_src, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t id, dim_t ih, dim_t iw,
                       const bool preserve_zero_padding) {
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];
            MAYBE_UNUSED(preserve_zero_padding);

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int k = 0; k < 2; k++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    const dim_t diff_dst_off = ow * stride_w_ + innermost_el;
                    const float dd = io::load_float_value(
                            dst_dt_, diff_dst, diff_dst_off);
                    sum += dd
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                io::store_float_value(src_dt_, sum, diff_src, innermost_el);
            }
        };
    }
}

simple_resampling_kernel_t::interpolate_fn_t
simple_resampling_kernel_t::create_bilinear() const {
    if (pd_->is_fwd()) {
        return [&](const char *src, char *dst, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t od, dim_t oh, dim_t ow,
                       const bool preserve_zero_padding) {
            const linear_coeffs_t &ih = linear_coeffs_[pd_->OD() + oh];
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    const dim_t src_off = ih.idx[j] * stride_h_
                            + iw.idx[k] * stride_w_ + innermost_el;
                    const float s = io::load_float_value(src_dt_, src, src_off);
                    res += s * ih.wei[j] * iw.wei[k];
                }

                if (are_postops_set_
                        && IMPLICATION(preserve_zero_padding,
                                innermost_el < tail_size_)) {
                    ref_post_ops_t::args_t args;
                    args.dst_val
                            = io::load_float_value(dst_dt_, dst, innermost_el);
                    args.ctx = &ctx;
                    args.l_offset = dst_off + innermost_el;
                    args.dst_md = pd_->dst_md();
                    ref_post_ops_->execute(res, args);
                }

                io::store_float_value(dst_dt_, res, dst, innermost_el);
            }
        };
    } else {
        return [&](const char *diff_dst, char *diff_src, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t id, dim_t ih, dim_t iw,
                       const bool preserve_zero_padding) {
            const bwd_linear_coeffs_t &h = bwd_linear_coeffs_[pd_->ID() + ih];
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];
            MAYBE_UNUSED(preserve_zero_padding);

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int j = 0; j < 2; j++)
                for_(int k = 0; k < 2; k++)
                for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    const dim_t diff_dst_off
                            = oh * stride_h_ + ow * stride_w_ + innermost_el;
                    const float dd = io::load_float_value(
                            dst_dt_, diff_dst, diff_dst_off);

                    sum += dd * bwd_linear_weights_[2 * (pd_->OD() + oh) + j]
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                io::store_float_value(src_dt_, sum, diff_src, innermost_el);
            }
        };
    }
}

simple_resampling_kernel_t::interpolate_fn_t
simple_resampling_kernel_t::create_trilinear() const {
    if (pd_->is_fwd()) {
        return [&](const char *src, char *dst, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t od, dim_t oh, dim_t ow,
                       const bool preserve_zero_padding) {
            const linear_coeffs_t &id = linear_coeffs_[od];
            const linear_coeffs_t &ih = linear_coeffs_[pd_->OD() + oh];
            const linear_coeffs_t &iw
                    = linear_coeffs_[pd_->OD() + pd_->OH() + ow];

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float res = 0;
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    const dim_t src_off = id.idx[i] * stride_d_
                            + ih.idx[j] * stride_h_ + iw.idx[k] * stride_w_
                            + innermost_el;
                    const float s = io::load_float_value(src_dt_, src, src_off);
                    res += s * id.wei[i] * ih.wei[j] * iw.wei[k];
                }

                if (are_postops_set_
                        && IMPLICATION(preserve_zero_padding,
                                innermost_el < tail_size_)) {
                    ref_post_ops_t::args_t args;
                    args.dst_val
                            = io::load_float_value(dst_dt_, dst, innermost_el);
                    args.ctx = &ctx;
                    args.l_offset = dst_off + innermost_el;
                    args.dst_md = pd_->dst_md();
                    ref_post_ops_->execute(res, args);
                }

                io::store_float_value(dst_dt_, res, dst, innermost_el);
            }
        };
    } else {
        return [&](const char *diff_dst, char *diff_src, const exec_ctx_t &ctx,
                       dim_t dst_off, dim_t id, dim_t ih, dim_t iw,
                       const bool preserve_zero_padding) {
            const bwd_linear_coeffs_t &d = bwd_linear_coeffs_[id];
            const bwd_linear_coeffs_t &h = bwd_linear_coeffs_[pd_->ID() + ih];
            const bwd_linear_coeffs_t &w
                    = bwd_linear_coeffs_[pd_->ID() + pd_->IH() + iw];
            MAYBE_UNUSED(preserve_zero_padding);

            PRAGMA_OMP_SIMD()
            for (dim_t innermost_el = 0; innermost_el < inner_stride_;
                    innermost_el++) {
                float sum = 0;
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for_(int k = 0; k < 2; k++)
                for_(dim_t od = d.start[i]; od < d.end[i]; od++)
                for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                    const dim_t diff_dst_off = od * stride_d_ + oh * stride_h_
                            + ow * stride_w_ + innermost_el;
                    const float dd = io::load_float_value(
                            dst_dt_, diff_dst, diff_dst_off);

                    sum += dd * bwd_linear_weights_[2 * od + i]
                            * bwd_linear_weights_[2 * (pd_->OD() + oh) + j]
                            * bwd_linear_weights_[2
                                            * (pd_->OD() + pd_->OH() + ow)
                                    + k];
                }
                io::store_float_value(src_dt_, sum, diff_src, innermost_el);
            }
        };
    }
}

simple_resampling_fwd_t::simple_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr) {}

status_t simple_resampling_fwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new simple_resampling_kernel_t(pd())));
    return kernel_->init();
}

status_t simple_resampling_fwd_t::execute(const exec_ctx_t &ctx) const {
    return kernel_->execute(ctx);
}

simple_resampling_bwd_t::simple_resampling_bwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr) {}

status_t simple_resampling_bwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new simple_resampling_kernel_t(pd())));
    return kernel_->init();
}

status_t simple_resampling_bwd_t::execute(const exec_ctx_t &ctx) const {
    return kernel_->execute(ctx);
}
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
