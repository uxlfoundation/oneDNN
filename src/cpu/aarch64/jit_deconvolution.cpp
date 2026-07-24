/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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
#include "cpu/aarch64/jit_deconvolution.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace utils;

namespace {

/*
 * Creates the equivalent forward-convolution descriptor used by the nested
 * convolution primitive.
 *
 * This conversion is valid only for unit-stride deconvolution. For each
 * spatial dimension, the padding required by the nested convolution is:
 *
 *     conv_padding = (kernel_size - 1) * (dilation + 1) - deconv_padding
 *
 * The implementation is intentionally limited to 2D H/W geometry.
 */
status_t create_fwd_convolution_desc(
        const jit_deconvolution_fwd_t::pd_t &deconv_pd,
        convolution_desc_t *conv_desc) {

    if (conv_desc == nullptr) return status::invalid_arguments;

    const deconvolution_desc_t *deconv_desc = deconv_pd.desc();

    VDISPATCH_DECONVOLUTION_IC(
            deconv_desc->strides[0] == 1
                    && deconv_desc->strides[1] == 1,
            VERBOSE_UNSUPPORTED_FEATURE,
            "only unit strides are allowed for deconv-to-conv conversion");

    dims_t padding_l {};
    dims_t padding_r {};

    /*
     * Spatial dimension 0 is height:
     *
     *     padding_l[0] = top
     *     padding_r[0] = bottom
     */
    padding_l[0] = (deconv_pd.KH() - 1) * (deconv_pd.KDH() + 1)
            - deconv_pd.padT();

    padding_r[0] = (deconv_pd.KH() - 1) * (deconv_pd.KDH() + 1)
            - deconv_pd.padB();

    /*
     * Spatial dimension 1 is width:
     *
     *     padding_l[1] = left
     *     padding_r[1] = right
     */
    padding_l[1] = (deconv_pd.KW() - 1) * (deconv_pd.KDW() + 1)
            - deconv_pd.padL();

    padding_r[1] = (deconv_pd.KW() - 1) * (deconv_pd.KDW() + 1)
            - deconv_pd.padR();

    CHECK(conv_desc_init(conv_desc,
            deconv_desc->prop_kind,
            alg_kind::convolution_direct,
            deconv_pd.src_md(),
            deconv_pd.weights_md(),
            &deconv_desc->bias_desc,
            deconv_pd.dst_md(),
            deconv_desc->strides,
            deconv_desc->dilates,
            padding_l,
            padding_r));

    /*
     * The selected nested convolution is not expected to interpret
     * use_inversion. Non-1x1 weights are spatially inverted by this wrapper
     * before nested convolution execution.
     */
    conv_desc->use_inversion = false;

    return status::success;
}

/*
 * Returns whether a nested convolution primitive descriptor can be executed
 * by directly forwarding the outer deconvolution arguments.
 *
 * This wrapper does not currently create source, weight, bias, or destination
 * reorders. The selected convolution implementation must therefore use
 * exactly the same memory descriptors as the outer deconvolution primitive.
 * This deliberately limits the wrapper to convolution implementations that
 * accept the deconvolution weight representation unchanged.
 */
bool is_compatible_convolution(
        const jit_deconvolution_fwd_t::pd_t &deconv_pd,
        const std::shared_ptr<primitive_desc_t> &candidate) {

    if (!candidate) return false;

    const memory_desc_t *candidate_src_md = candidate->src_md();
    const memory_desc_t *candidate_weights_md = candidate->weights_md();
    const memory_desc_t *candidate_dst_md = candidate->dst_md();

    if (candidate_src_md == nullptr || candidate_weights_md == nullptr || candidate_dst_md == nullptr)
        return false;

    if (memory_desc_wrapper(*deconv_pd.src_md()) != memory_desc_wrapper(*candidate_src_md))
        return false;

    if (memory_desc_wrapper(*deconv_pd.weights_md()) != memory_desc_wrapper(*candidate_weights_md))
        return false;

    if (memory_desc_wrapper(*deconv_pd.dst_md()) != memory_desc_wrapper(*candidate_dst_md))
        return false;

    if (deconv_pd.with_bias()) {
        const memory_desc_t *candidate_bias_md = candidate->weights_md(1);

        if (candidate_bias_md == nullptr)
            return false;

        if (memory_desc_wrapper(*deconv_pd.invariant_bia_md()) != memory_desc_wrapper(*candidate_bias_md))
            return false;
    }

    return true;
}

/*
 * Returns true when a 2D kernel must be spatially inverted.
 *
 * A 1x1 kernel is unchanged by inversion. Every other kernel, including
 * 1xN and Nx1 kernels, must be inverted along the non-unit spatial dimensions.
 */
bool needs_spatial_inversion(const memory_desc_t *weights_md) {
    if (weights_md == nullptr || weights_md->ndims != 4) return false;

    constexpr int kh_dim = 2;
    constexpr int kw_dim = 3;

    return weights_md->dims[kh_dim] > 1 || weights_md->dims[kw_dim] > 1;
}

/*
 * Spatially inverts ungrouped 2D deconvolution weights.
 *
 * Deconvolution weights are logically indexed as:
 *
 *     [IC, OC, KH, KW]
 *
 * Channel coordinates are preserved while the two spatial coordinates are
 * reversed:
 *
 *     dst[ic, oc, kh, kw] = src[ic, oc, KH - 1 - kh, KW - 1 - kw]
 *
 * This helper requires a dense plain HWIO memory descriptor. In HWIO format,
 * all IC x OC elements belonging to one spatial kernel position are
 * contiguous, allowing a complete channel plane to be copied at once.
 *
 * memory_desc_wrapper::off_v() is used so that logical indexing remains
 * correct for the required HWIO physical layout.
 */
status_t invert_weights(
        const memory_desc_t *weights_md,
        const void *src,
        void *dst) {

    if (weights_md == nullptr || src == nullptr || dst == nullptr)
        return status::invalid_arguments;

    if (weights_md->ndims != 4)
        return status::invalid_arguments;

    if (memory_desc_matches_one_of_tag(
                *weights_md, format_tag::hwio)
            != format_tag::hwio)
        return status::invalid_arguments;

    const size_t element_size = types::data_type_size(weights_md->data_type);

    if (element_size == 0)
        return status::invalid_arguments;

    const auto *src_bytes = static_cast<const uint8_t *>(src);

    auto *dst_bytes = static_cast<uint8_t *>(dst);

    const memory_desc_wrapper weights_d(weights_md);

    const dim_t IC = weights_md->dims[0];
    const dim_t OC = weights_md->dims[1];
    const dim_t KH = weights_md->dims[2];
    const dim_t KW = weights_md->dims[3];

    /*
     * In HWIO format, all IC x OC elements belonging to one spatial kernel
     * position are contiguous. Copy a complete channel plane at once instead
     * of calling off_v() and memcpy() for every individual element.
     */
    const size_t spatial_plane_size
            = static_cast<size_t>(IC)
            * static_cast<size_t>(OC)
            * element_size;

    dims_t dst_pos {};
    dims_t src_pos {};

    for (dim_t kh = 0; kh < KH; ++kh) {
        for (dim_t kw = 0; kw < KW; ++kw) {
            dst_pos[0] = 0;
            dst_pos[1] = 0;
            dst_pos[2] = kh;
            dst_pos[3] = kw;

            src_pos[0] = 0;
            src_pos[1] = 0;
            src_pos[2] = KH - 1 - kh;
            src_pos[3] = KW - 1 - kw;

            const dim_t dst_offset = weights_d.off_v(dst_pos);

            const dim_t src_offset = weights_d.off_v(src_pos);

            std::memcpy(
                    dst_bytes
                            + static_cast<size_t>(dst_offset)
                                    * element_size,
                    src_bytes
                            + static_cast<size_t>(src_offset)
                                    * element_size,
                    spatial_plane_size);
        }
    }

    return status::success;
}

// copying the datatype handling helpers from convolution base

bool regular_swd_ok(
        const jit_deconvolution_fwd_t::pd_t &pd) {

    const auto src_dt = pd.src_md()->data_type;
    const auto wei_dt = pd.weights_md()->data_type;
    const auto dst_dt = pd.dst_md()->data_type;

    return (src_dt == f32 && wei_dt == f32 && dst_dt == f32)
            || (src_dt == bf16 && wei_dt == bf16
                    && one_of(dst_dt, bf16, f32))
            || (src_dt == f16 && wei_dt == f16
                    && one_of(dst_dt, f16, f32));
}

bool bias_ok(
        const jit_deconvolution_fwd_t::pd_t &pd) {

    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type;
}

} // namespace

/*
 * Builds the outer deconvolution implementation name from the selected nested
 * convolution implementation.
 *
 * Examples:
 *
 *     jit_deconv+direct_1x1:kai
 *     jit_deconv+indirect_gemm:kai
 *     jit_deconv+brgconv:sve_256
 *     jit_deconv+gemm:any
 */
void jit_deconvolution_fwd_t::pd_t::init_name() {
    name_ = "jit_deconv+";

    if (conv_pd_)
        name_.append(conv_pd_->name());
    else
        name_.append("unknown");
}

/*
 * Assigns the explicit 2D channel-last memory formats required by the current
 * direct argument-forwarding and weight-inversion paths.
 *
 * Source and destination use NHWC. Deconvolution weights retain their logical
 * [IC, OC, KH, KW] dimensions while HWIO determines their physical order.
 *
 * Convolution implementations using other layouts are rejected by
 * is_compatible_convolution() because this wrapper does not currently create
 * reorders.
 */
status_t jit_deconvolution_fwd_t::pd_t::set_default_formats() {
    using namespace format_tag;

    if (src_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md_, nhwc));

    if (dst_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md_, nhwc));

    if (weights_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_md_, hwio));

    if (with_bias() && bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    return status::success;
}

/*
 * Validates the supported 2D deconvolution configuration, creates the
 * equivalent convolution descriptor, selects a compatible nested convolution
 * implementation from the normal CPU convolution implementation list, and
 * books the required scratchpad.
 *
 * Supported:
 *
 *     - 2D tensors only;
 *     - arbitrary KH and KW;
 *     - arbitrary supported top/bottom/left/right padding;
 *     - arbitrary supported H/W dilation;
 *     - unit H/W stride;
 *     - regular 2D and degenerate H-only or W-only shapes;
 *     - supported f32, bf16, and f16 datatype combinations;
 *     - optional bias with the same datatype as destination.
 *
 * Attributes remain restricted to ungrouped, unit-stride operations and are
 * passed to the nested convolution primitive.
 *
 * The wrapper selects only nested convolution implementations that accept the
 * outer source, weights, bias, and destination descriptors unchanged.
 */
status_t jit_deconvolution_fwd_t::pd_t::init(
        engine_t *engine) {

    using namespace format_tag;

    VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_DECONVOLUTION(
            desc()->alg_kind == alg_kind::deconvolution_direct,
            VERBOSE_BAD_ALGORITHM);

    /*
     * The current deconvolution-to-convolution conversion and weight
     * inversion paths operate on 2D H/W geometry. Degenerate shapes such as
     * IH=KH=OH=1 remain valid 2D operations.
     */
    VDISPATCH_DECONVOLUTION(
            ndims() == 4,
            VERBOSE_BAD_NDIMS,
            "src",
            ndims());

    VDISPATCH_DECONVOLUTION(
            !with_groups(),
            VERBOSE_UNSUPPORTED_FEATURE,
            "groups");

    // data type handling for src, wei, dst

    VDISPATCH_DECONVOLUTION(
            regular_swd_ok(*this),
            VERBOSE_UNSUPPORTED_DT_CFG);

    VDISPATCH_DECONVOLUTION(
            !has_zero_dim_memory(),
            VERBOSE_EMPTY_TENSOR,
            "");

    VDISPATCH_DECONVOLUTION(
            !has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VDISPATCH_DECONVOLUTION(
            attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::fpmath_mode
                            | primitive_attr_t::skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    /*
     * The deconvolution-to-forward-convolution conversion used here is valid
     * only for unit spatial strides.
     */
    VDISPATCH_DECONVOLUTION(
            KSH() == 1 && KSW() == 1,
            VERBOSE_UNSUPPORTED_FEATURE,
            "non-unit stride");

    /*
     * Padding and dilation are intentionally not rejected here. They are
     * converted into the nested convolution descriptor, whose primitive
     * descriptor determines whether the resulting operation is supported.
     */

    CHECK(set_default_formats());

    VDISPATCH_DECONVOLUTION(
            memory_desc_matches_one_of_tag(src_md_, nhwc) == nhwc,
            VERBOSE_UNSUPPORTED_TAG_S,
            "src");

    VDISPATCH_DECONVOLUTION(
            memory_desc_matches_one_of_tag(dst_md_, nhwc) == nhwc,
            VERBOSE_UNSUPPORTED_TAG_S,
            "dst");

    VDISPATCH_DECONVOLUTION(
            memory_desc_matches_one_of_tag(weights_md_, hwio) == hwio,
            VERBOSE_UNSUPPORTED_TAG_S,
            "weights");

    VDISPATCH_DECONVOLUTION(
            impl::is_dense_format_kind(
                    {src_md(), weights_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    /*
     * Bias is optional. When present, check datatype and density.
     */
    VDISPATCH_DECONVOLUTION(
            bias_ok(*this),
            VERBOSE_UNSUPPORTED_DT_CFG);

    if (with_bias()) {
        VDISPATCH_DECONVOLUTION(
                impl::is_dense_format_kind({&bias_md_}),
                VERBOSE_UNSUPPORTED_SPARSE_CFG);
    }

    convolution_desc_t conv_desc {};

    CHECK(create_fwd_convolution_desc(*this, &conv_desc));

    /*
     * primitive_desc_iterator_t obtains convolution candidates from
     * get_convolution_impl_list(), which uses the registration order in
     * cpu_convolution_list.cpp.
     */
    primitive_desc_iterator_t iterator(engine,
            reinterpret_cast<const op_desc_t *>(&conv_desc),
            attr(),
            nullptr);

    if (!iterator.is_initialized()) return status::out_of_memory;

    /*
     * Select the first candidate in normal convolution-list priority order
     * whose descriptors can be used by this wrapper without reorders.
     */
    while (++iterator != iterator.end()) {
        const auto candidate = *iterator;

        if (is_compatible_convolution(*this, candidate)) {
            conv_pd_ = candidate;
            break;
        }
    }

    VDISPATCH_DECONVOLUTION(
            conv_pd_ != nullptr,
            VERBOSE_PRIMITIVE_CREATION_FAIL,
            "compatible convolution implementation not found");

    init_name();

    /*
     * The original source and destination arguments are forwarded directly.
     * The original weights are also forwarded for 1x1 kernels; larger kernels
     * are replaced at execution time by an inverted copy with this same
     * descriptor.
     */
    VDISPATCH_DECONVOLUTION(
            memory_desc_wrapper(src_md_)
                    == memory_desc_wrapper(*conv_pd_->src_md()),
            VERBOSE_UNSUPPORTED_TAG_S,
            "nested src");

    VDISPATCH_DECONVOLUTION(
            memory_desc_wrapper(weights_md_)
                    == memory_desc_wrapper(*conv_pd_->weights_md()),
            VERBOSE_UNSUPPORTED_TAG_S,
            "nested weights");

    if (with_bias()) {
        const memory_desc_t *nested_bias_md = conv_pd_->weights_md(1);

        VDISPATCH_DECONVOLUTION(
                nested_bias_md != nullptr
                        && memory_desc_wrapper(bias_md_)
                                == memory_desc_wrapper(*nested_bias_md),
                VERBOSE_UNSUPPORTED_TAG_S,
                "nested bias");
    }

    VDISPATCH_DECONVOLUTION(
            memory_desc_wrapper(dst_md_)
                    == memory_desc_wrapper(*conv_pd_->dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S,
            "nested dst");

    auto scratchpad = scratchpad_registry().registrar();

    /*
     * Reserve all temporary memory required by the selected nested
     * convolution implementation.
     */
    scratchpad.book(
            memory_tracking::names::key_nested,
            conv_pd_->scratchpad_registry());

    /*
     * Non-1x1 kernels require a temporary spatially inverted weight buffer.
     * A 1x1 kernel is unchanged and uses the original weight memory directly.
     */
    if (needs_spatial_inversion(weights_md())) {
        const memory_desc_wrapper weights_d(weights_md());

        scratchpad.book(
                memory_tracking::names::key_conv_permuted_weights,
                weights_d.size(),
                1,
                64,
                64);
    }

    return status::success;
}

/*
 * Creates the nested convolution primitive selected during primitive
 * descriptor initialization.
 */
status_t jit_deconvolution_fwd_t::init(engine_t *engine) {
    if (!pd()->conv_pd_)
        return status::invalid_arguments;

    return pd()->conv_pd_->create_primitive(conv_p_, engine);
}

/*
 * Executes the selected convolution using the deconvolution arguments.
 *
 * Padding and dilation have already been represented in the converted
 * convolution descriptor. For non-1x1 kernels, the wrapper creates a
 * spatially inverted weight copy and substitutes it for DNNL_ARG_WEIGHTS.
 * Source, destination, and optional bias are forwarded unchanged. The nested
 * convolution receives its own portion of the outer scratchpad.
 */
status_t jit_deconvolution_fwd_t::execute(
        const exec_ctx_t &ctx) const {

    if (!conv_p_)
        return status::runtime_error;

    const auto *deconv_pd = pd();

    const bool invert = needs_spatial_inversion(
            deconv_pd->weights_md());

    exec_args_t conv_args(ctx.args());

    auto execute_nested = [&](exec_args_t &&args) -> status_t {
        exec_ctx_t conv_ctx(ctx, std::move(args));

        /*
         * Give the nested convolution its reserved portion of the outer
         * deconvolution scratchpad.
         */
        auto *nested_grantor = create_nested_grantor(
                ctx.get_scratchpad_grantor(),
                memory_tracking::names::key_nested,
                conv_p_->pd()->scratchpad_registry());

        conv_ctx.set_scratchpad_grantor(nested_grantor);

        return conv_p_->execute(conv_ctx);
    };

    /*
     * A 1x1 kernel does not require inversion.
     */
    if (!invert)
        return execute_nested(std::move(conv_args));

    const auto scratchpad = ctx.get_scratchpad_grantor();

    /*
     * Weight inversion is datatype-independent.
     */
    const void *original_weights
            = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);

    uint8_t *inverted_weights = scratchpad.get<uint8_t>(
            memory_tracking::names::key_conv_permuted_weights);

    if (inverted_weights == nullptr)
        return status::runtime_error;

    CHECK(invert_weights(
            deconv_pd->weights_md(),
            original_weights,
            inverted_weights));

    auto *engine = ctx.stream()->engine();

    /*
     * memory_t has a non-public destructor and must be managed through
     * memory_deleter_t.
     */
    std::unique_ptr<memory_t, memory_deleter_t> inverted_weights_mem;

    inverted_weights_mem.reset(new memory_t(
            engine,
            deconv_pd->conv_pd_->weights_md(),
            use_runtime_ptr,
            inverted_weights));

    /*
     * Replace the original weights argument.
     */
    conv_args[DNNL_ARG_WEIGHTS]
            = {inverted_weights_mem.get(), true};

    return execute_nested(std::move(conv_args));
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl