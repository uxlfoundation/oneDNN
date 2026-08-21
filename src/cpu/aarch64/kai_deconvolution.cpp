/*******************************************************************************
* Copyright 2018, 2022 Intel Corporation
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_deconvolution.hpp"

#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/kai_direct_1x1_convolution.hpp"
#include "cpu/aarch64/kai_indirect_convolution.hpp"
#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

// Express a unit-stride deconvolution as an inverted forward convolution.
status_t create_fwd_conv_desc(const deconvolution_desc_t *deconv_d,
        convolution_desc_t *conv_d, const memory_desc_t *bias_md,
        data_type_t dst_dt) {
    constexpr int spatial_ndims = 2;
    // Padding used by the nested forward convolution for height and width.
    dims_t padding_l {};
    dims_t padding_r {};
    // Product of the spatial kernel sizes, used to identify non-1x1 cases.
    dim_t kernel_size = 1;

    for (int d = 0; d < spatial_ndims; ++d) {
        if (deconv_d->strides[d] != 1) return status::unimplemented;

        // weights_d is the height or width axis in the weights descriptor.
        const int weights_d = deconv_d->weights_desc.ndims - spatial_ndims + d;
        const dim_t kernel = deconv_d->weights_desc.dims[weights_d];
        // oneDNN stores dilation as the number of skipped points.
        const dim_t dilation = deconv_d->dilates[d] + 1;
        // Distance from the first kernel point to the last one.
        const dim_t effective_kernel = (kernel - 1) * dilation;
        padding_l[d] = effective_kernel - deconv_d->padding[0][d];
        padding_r[d] = effective_kernel - deconv_d->padding[1][d];
        if (padding_l[d] < 0 || padding_r[d] < 0) return status::unimplemented;
        kernel_size *= kernel;
    }

    // Copy the public destination layout with the datatype needed by KAI.
    memory_desc_t dst_md;
    CHECK(memory_desc_init_by_md_and_dt(dst_md, deconv_d->dst_desc, dst_dt));
    CHECK(conv_desc_init(conv_d, prop_kind::forward_training,
            alg_kind::convolution_direct, &deconv_d->src_desc,
            &deconv_d->weights_desc, bias_md, &dst_md, deconv_d->strides,
            deconv_d->dilates, padding_l, padding_r));

    // Distinguish the converted descriptor in the primitive cache.
    if (kernel_size > 1) {
        conv_d->diff_src_desc = conv_d->src_desc;
        conv_d->diff_dst_desc = conv_d->dst_desc;
    }
    conv_d->use_inversion = true;
    return status::success;
}

bool is_kai_convolution(const primitive_desc_t *pd) {
    return dynamic_cast<const kai_direct_1x1_convolution_fwd_t::pd_t *>(pd)
            != nullptr
            || dynamic_cast<const kai_indirect_convolution_fwd_t::pd_t *>(pd)
            != nullptr;
}

} // namespace

status_t kai_deconvolution_fwd_t::pd_t::try_kai_convolution(
        const engine_t *engine, const primitive_attr_t &conv_attr,
        const memory_desc_t *bias_md, data_type_t dst_dt) {
    // conv_d is the equivalent forward-convolution descriptor searched below.
    convolution_desc_t conv_d {};
    const status_t desc_status
            = create_fwd_conv_desc(desc(), &conv_d, bias_md, dst_dt);
    if (desc_status != status::success) return desc_status;

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<const op_desc_t *>(&conv_d), &conv_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    while (++it != it.end()) {
        if (!is_kai_convolution((*it).get())) continue;
        conv_pd_ = *it;
        return status::success;
    }
    return status::unimplemented;
}

status_t kai_deconvolution_fwd_t::pd_t::init(const engine_t *engine) {
    using mask_t = primitive_attr_t::skip_mask_t;
    // Keep the public destination type even if the nested fallback uses f32.
    const data_type_t dst_dt = dst_md()->data_type;
    // These attributes are handled by KAI or by this wrapper's epilogue.
    const auto attr_mask = mask_t::fpmath_mode | mask_t::accumulation_mode
            | mask_t::post_ops;

    VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_DECONVOLUTION(desc()->alg_kind == alg_kind::deconvolution_direct,
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_DECONVOLUTION(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_DECONVOLUTION(
            !with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_DECONVOLUTION(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_DECONVOLUTION(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_DECONVOLUTION(attr()->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_DECONVOLUTION(
            attr()->post_ops_.check_sum_consistency(dst_dt, false),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_DECONVOLUTION(impl::is_dense_format_kind({src_md(0),
                                    weights_md(0), weights_md(1), dst_md(0)}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // A null descriptor tells the nested convolution that there is no bias.
    const memory_desc_t *bias_md = with_bias() ? weights_md(1) : nullptr;
    // First try to let KAI own the complete bias and post-op operation.
    status_t status = try_kai_convolution(engine, *attr(), bias_md, dst_dt);
    if (status == status::out_of_memory) return status;

    if (status != status::success) {
        VDISPATCH_DECONVOLUTION(with_bias() || attr()->post_ops_.len() > 0,
                "KAI convolution implementation not found");
        VDISPATCH_DECONVOLUTION(ref_post_ops_t::post_ops_ok(attr()->post_ops_),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Retry KAI without bias/post-ops and finish them from an f32 result.
        primitive_attr_t conv_attr(*attr());
        if (!conv_attr.is_initialized()) return status::out_of_memory;
        CHECK(conv_attr.set_post_ops(post_ops_t {}));
        CHECK(try_kai_convolution(engine, conv_attr, nullptr, data_type::f32));
        use_outer_epilogue_ = true;
    }

    if (weights_md_.format_kind == format_kind::any)
        weights_md_ = *conv_pd_->weights_md();
    if (src_md_.format_kind == format_kind::any) src_md_ = *conv_pd_->src_md();
    if (dst_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_md_and_dt(
                dst_md_, *conv_pd_->dst_md(), dst_dt));
    }
    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    CHECK(attr_.set_default_formats(dst_md(0)));

    init_name();

    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    // key_nested owns any temporary storage requested by the selected KAI PD.
    scratchpad.book(key_nested, conv_pd_->scratchpad_registry());
    if (use_outer_epilogue_) {
        const memory_desc_wrapper conv_dst_d(conv_pd_->dst_md());
        assert(conv_dst_d.data_type() == data_type::f32);
        // Keep KAI output separate so sum can read the original destination.
        scratchpad.book(
                key_deconv_bias, conv_dst_d.nelems(true), sizeof(float));
    }

    return status::success;
}

void kai_deconvolution_fwd_t::pd_t::init_name() {
    name_ = "kai_deconv+";
    name_.append(conv_pd_->name());
    if (use_outer_epilogue_) name_.append("+outer_post_ops");
}

status_t kai_deconvolution_fwd_t::init(engine_t *engine) {
    CHECK(pd()->conv_pd_->create_primitive(conv_p_, engine));
    if (pd()->use_outer_epilogue_) {
        ref_post_ops_
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops_) return status::out_of_memory;
        CHECK(ref_post_ops_->init(pd()->dst_md()));
    }
    return status::success;
}

status_t kai_deconvolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto &args = ctx.args();
    // Arguments passed to KAI; the fallback starts with source and weights.
    exec_args_t conv_args;
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_SRC);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);

    // View of the f32 output scratchpad; it does not own the storage.
    std::unique_ptr<memory_t, memory_deleter_t> tmp_dst;
    if (pd()->use_outer_epilogue_) {
        const auto &dst = args.at(DNNL_ARG_DST);
        CHECK(safe_ptr_assign(tmp_dst,
                new memory_t(dst.mem()->engine(), pd()->conv_pd_->dst_md(),
                        ctx.get_scratchpad_grantor().get_memory_storage(
                                key_deconv_bias))));
        conv_args[DNNL_ARG_DST] = {tmp_dst.get(), false};
    } else {
        conv_args = args;
    }

    exec_ctx_t conv_ctx(ctx, std::move(conv_args));
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, conv_p_->pd()->scratchpad_registry());
    conv_ctx.set_scratchpad_grantor(nested_grantor);
    CHECK(conv_p_->execute(conv_ctx));

    if (!pd()->use_outer_epilogue_) return status::success;

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    // conv_output is the f32 result that still needs public bias and post-ops.
    const auto *conv_output = scratchpad.get<const float>(key_deconv_bias);
    auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const auto *bias = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;
    // The nested and public destinations can have different datatypes/layouts.
    const memory_desc_wrapper conv_dst_d(pd()->conv_pd_->dst_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    // Sum needs the original public destination value and its declared type.
    const bool has_sum
            = pd()->attr()->post_ops_.find(primitive_kind::sum) != -1;
    const auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    const auto *post_ops = ref_post_ops_.get();
    // Logical output sizes: minibatch, channels, height, and width.
    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();

    parallel_nd(
            MB, OC, OH, OW, [=, &ctx](dim_t mb, dim_t oc, dim_t oh, dim_t ow) {
        // Nested scratch and public output may use different physical offsets.
        const dim_t conv_off = conv_dst_d.off(mb, oc, oh, ow);
        const dim_t dst_off = dst_d.off(mb, oc, oh, ow);
        msan_unpoison((void *)&conv_output[conv_off], sizeof(float));
        float value = conv_output[conv_off];
        if (bias) {
            value += io::load_float_value(bias_d.data_type(), bias, oc);
        }

        ref_post_ops_t::args_t post_ops_args;
        if (has_sum) {
            post_ops_args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
        }
        post_ops_args.ctx = &ctx;
        // Logical NCHW offset used to broadcast binary and PReLU arguments.
        post_ops_args.l_offset = ((mb * OC + oc) * OH + oh) * OW + ow;
        post_ops_args.dst_md = pd()->dst_md();
        post_ops->execute(value, post_ops_args);
        io::store_float_value(dst_d.data_type(), value, dst, dst_off);
    });

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
