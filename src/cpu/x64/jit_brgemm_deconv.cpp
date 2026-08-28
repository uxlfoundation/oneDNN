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

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_deconv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename implementation_pd>
status_t check_embedded_impl_init(const primitive_desc_t *pd) {
    if (dynamic_cast<const implementation_pd *>(pd) != nullptr)
        return status::success; // implementation found
    return status::unimplemented;
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace utils;
    using namespace format_tag;
    using smask_t = primitive_attr_t::skip_mask_t;
    const deconvolution_desc_t *fwd_deconv_d = desc();
    const auto src_type = fwd_deconv_d->src_desc.data_type;
    const auto dst_type = fwd_deconv_d->dst_desc.data_type;
    const bool is_int8 = utils::one_of(src_type, s8, u8);

    auto skip_mask = smask_t::post_ops | smask_t::sum_dt;
    if (is_int8) skip_mask |= smask_t::scales | smask_t::zero_points;

    VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_DECONVOLUTION((desc()->alg_kind & alg_kind::deconvolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_DECONVOLUTION(attr()->has_default_values(skip_mask, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_DECONVOLUTION(
            attr()->post_ops_.check_sum_consistency(dst_type, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_DECONVOLUTION(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_DECONVOLUTION(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_DECONVOLUTION(zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_DECONVOLUTION(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_DECONVOLUTION(
            impl::is_dense_format_kind({src_md(0), diff_weights_md(0),
                    diff_weights_md(1), diff_dst_md(0), dst_md(0)}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    assert(src_type != data_type::undef);

    const int ndims_spatial = fwd_deconv_d->dst_desc.ndims - 2;
    for (int i = 0; i < ndims_spatial; i++) {
        if (fwd_deconv_d->strides[i] != 1) {
            has_strides_ = true;
            break;
        }
    }

    if (has_strides_) {
        // The strided deconvolution is implemented via a backward-by-data
        // convolution with spatially-inverted weights. The nested convolution
        // applies the deconvolution attributes (post-ops, scales) itself, thus
        // the attributes are forwarded and weights compensation is allowed. The
        // filter fetches the brgemm strided implementation while iterating over
        // the convolution implementations list.
        const auto filter = [](const primitive_desc_t *conv_pd) {
            using strided_pd_t =
                    typename brgemm_convolution_bwd_strided_t<isa>::pd_t;
            return check_embedded_impl_init<strided_pd_t>(conv_pd)
                    == status::success;
        };
        const status_t create_status = create_conv_pd(conv_pd_, engine, this,
                dst_md()->data_type, /* force_empty_bias = */ false,
                /* allow_wei_compensation = */ true, /* copy_attr = */ true,
                filter);
        VDISPATCH_DECONVOLUTION_IC(create_status == status::success,
                "brgemm implementation not found for strided convolution");
    } else {
        // The non-strided deconvolution is implemented via a forward
        // convolution with spatially-inverted weights. The nested convolution
        // applies the deconvolution attributes (post-ops, scales) itself, thus
        // the attributes are forwarded. The filter fetches a brgemm forward
        // implementation that honors the weights spatial inversion while
        // iterating over the convolution implementations list. Add more
        // inversion-aware forward implementations to the filter as needed.
        const auto filter = [](const primitive_desc_t *conv_pd) {
            return check_embedded_impl_init<
                           typename brgemm_1x1_convolution_fwd_t<isa>::pd_t>(
                           conv_pd)
                    == status::success
                    || check_embedded_impl_init<
                               typename brgemm_convolution_fwd_t<isa>::pd_t>(
                               conv_pd)
                    == status::success;
        };
        const status_t create_status = create_conv_pd(conv_pd_, engine, this,
                dst_md()->data_type, /* force_empty_bias = */ false,
                /* allow_wei_compensation = */ false, /* copy_attr = */ true,
                filter);
        VDISPATCH_DECONVOLUTION_IC(create_status == status::success,
                "brgemm implementation not found for forward convolution");
    }

    if (weights_md_.format_kind == format_kind::any) {
        if (has_strides_) {
            weights_md_ = utils::downcast<convolution_pd_t *>(conv_pd_.get())
                                  ->weights_md_with_permute_channels();
            VDISPATCH_DECONVOLUTION_IC(!types::is_zero_md(&weights_md_),
                    VERBOSE_DESC_CREATION_FAIL, "weights");
            const bool is_signed_input = src_type == s8;
            const bool scale_adjust_required = is_signed_input
                    && !isa_has_s8s8(isa) && !isa_has_int8_vnni(isa);
            // Set flags after the channels permutation,
            // because it expects flags to be zero
            if (scale_adjust_required)
                weights_md_.extra.flags = 0 | memory_extra_flags::scale_adjust;
        } else
            weights_md_ = *conv_pd_->weights_md();
    }
    if (src_md_.format_kind == format_kind::any) {
        if (has_strides_)
            src_md_ = *conv_pd_->diff_dst_md();
        else
            src_md_ = *conv_pd_->src_md();
    }
    if (dst_md_.format_kind == format_kind::any) {
        if (has_strides_)
            dst_md_ = *conv_pd_->diff_src_md();
        else
            dst_md_ = *conv_pd_->dst_md();
    }
    attr_.set_default_formats(&dst_md_);
    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    init_name();
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested,
            conv_pd_->scratchpad_registry());

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::init(engine_t *engine) {
    return pd()->conv_pd_->create_primitive(conv_p_, engine);
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto &args = ctx.args();
    exec_args_t conv_args(args);
    if (pd()->has_strides_) {
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args.erase(DNNL_ARG_DST);
        conv_args.erase(DNNL_ARG_SRC);
    }

    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested,
            conv_p_->pd()->scratchpad_registry());
    conv_ctx.set_scratchpad_grantor(nested_grantor);
    return conv_p_->execute(conv_ctx);
}

template struct brgemm_deconvolution_fwd_t<avx2>;
template struct brgemm_deconvolution_fwd_t<avx2_vnni>;
template struct brgemm_deconvolution_fwd_t<avx2_vnni_2>;
template struct brgemm_deconvolution_fwd_t<avx512_core>;
template struct brgemm_deconvolution_fwd_t<avx512_core_vnni>;
template struct brgemm_deconvolution_fwd_t<avx512_core_bf16>;
template struct brgemm_deconvolution_fwd_t<avx512_core_fp16>;
template struct brgemm_deconvolution_fwd_t<avx10_2>;
template struct brgemm_deconvolution_fwd_t<avx512_core_amx>;
template struct brgemm_deconvolution_fwd_t<avx512_core_amx_fp16>;
template struct brgemm_deconvolution_fwd_t<avx10_2_amx_2>;
template struct brgemm_deconvolution_fwd_t<avx10_2_ace>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
