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

#include "cpu/aarch64/kai_convolution.hpp"

#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_direct_1x1_convolution.hpp"
#include "cpu/aarch64/kai_im2row_convolution.hpp"
#include "cpu/aarch64/kai_indirect_convolution.hpp"
#include "cpu/aarch64/kai_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

kai_convolution_t::pd_t::impl_kind_t
kai_convolution_t::pd_t::choose_impl() const {
    using base_pd_t = kai_convolution_fwd_base_t::pd_t;
    if (base_pd_t::is_1x1(*this)) {
        // Native bf16 NHWC uses indirect as in ACL. The direct and im2row
        // paths require every 1x1 patch to lie within the source image.
        const bool all_bf16
                = utils::everyone_is(data_type::bf16, src_md()->data_type,
                        weights_md()->data_type, dst_md()->data_type);
        if (all_bf16 || !base_pd_t::has_zero_top_left_padding(*this)
                || !base_pd_t::source_pixels_in_bounds(*this))
            return impl_kind_t::indirect;

        const bool unit_stride = KSH() == 1 && KSW() == 1;
        return unit_stride ? impl_kind_t::direct_1x1 : impl_kind_t::im2row;
    }

    return impl_kind_t::indirect;
}

status_t kai_convolution_t::pd_t::init(const engine_t *engine) {
    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");

    // brgconv is faster for small OC * IC (except where brg is not implemented:
    // asimd or f16) We replicate the ACL heuristic to avoid edge case
    // regressions blocking removing ACL, longer term we should re-assess this
    // heuristic specifically for KleidiAI conv
    const bool any_f16 = utils::one_of(data_type::f16, src_md()->data_type,
            weights_md()->data_type, dst_md()->data_type);
    const bool use_brgconv = OC() * IC() <= 2048 && mayiuse(sve) && !any_f16;
    VDISPATCH_CONV(!use_brgconv, "brgconv:sve is faster for small OC * IC");

    impl_kind_ = choose_impl();
    primitive_desc_t *impl_pd = nullptr;
    // Create directly, without traversing the public convolution list. Each
    // child gets the original descriptor, including unresolved formats.
    switch (impl_kind_) {
        case impl_kind_t::direct_1x1:
            CHECK(primitive_desc_t::create<
                    kai_direct_1x1_convolution_fwd_t::pd_t>(&impl_pd, op_desc(),
                    attr(), engine, nullptr, nullptr, nullptr));
            break;
        case impl_kind_t::indirect:
            CHECK(primitive_desc_t::create<
                    kai_indirect_convolution_fwd_t::pd_t>(&impl_pd, op_desc(),
                    attr(), engine, nullptr, nullptr, nullptr));
            break;
        case impl_kind_t::im2row:
            CHECK(primitive_desc_t::create<kai_im2row_convolution_fwd_t::pd_t>(
                    &impl_pd, op_desc(), attr(), engine, nullptr, nullptr,
                    nullptr));
            break;
    }
    impl_pd_.reset(impl_pd);

    src_md_ = *impl_pd_->src_md();
    weights_md_ = *impl_pd_->weights_md();
    bias_md_ = *impl_pd_->weights_md(1);
    dst_md_ = *impl_pd_->dst_md();
    CHECK(attr_.copy_from(*impl_pd_->attr()));
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested,
            impl_pd_->scratchpad_registry());
    return status::success;
}

status_t kai_convolution_t::init(engine_t *engine) {
    // The wrapper and child describe the same operation. Creating the child
    // through the primitive cache would collide with the wrapper's pending
    // entry. Cache the wrapper as a whole and initialize its owned child here.
    using impl_kind_t = pd_t::impl_kind_t;
    switch (pd()->impl_kind_) {
        case impl_kind_t::direct_1x1:
            impl_ = std::make_shared<kai_direct_1x1_convolution_fwd_t>(
                    static_cast<const kai_direct_1x1_convolution_fwd_t::pd_t *>(
                            pd()->impl_pd_.get()));
            break;
        case impl_kind_t::indirect:
            impl_ = std::make_shared<kai_indirect_convolution_fwd_t>(
                    static_cast<const kai_indirect_convolution_fwd_t::pd_t *>(
                            pd()->impl_pd_.get()));
            break;
        case impl_kind_t::im2row:
            impl_ = std::make_shared<kai_im2row_convolution_fwd_t>(
                    static_cast<const kai_im2row_convolution_fwd_t::pd_t *>(
                            pd()->impl_pd_.get()));
            break;
    }
    return impl_->init(engine);
}

status_t kai_convolution_t::execute(const exec_ctx_t &ctx) const {
    exec_args_t args = ctx.args();
    exec_ctx_t impl_ctx(ctx, std::move(args));
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested,
            impl_->pd()->scratchpad_registry());
    impl_ctx.set_scratchpad_grantor(nested_grantor);
    return impl_->execute(impl_ctx);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
