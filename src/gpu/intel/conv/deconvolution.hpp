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

#ifndef GPU_INTEL_CONV_DECONVOLUTION_HPP
#define GPU_INTEL_CONV_DECONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/conv/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deconv {

using namespace conv;

struct conv_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public bwd_weights_pd_t {
        using bwd_weights_pd_t::bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(name_.c_str(), conv_bwd_weights_t);

        status_t init(const impl::engine_t *engine) {
            using namespace format_tag;
            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    (utils::everyone_is(data_type::f32,
                             desc()->src_desc.data_type,
                             desc()->diff_weights_desc.data_type,
                             desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f64,
                                    desc()->src_desc.data_type,
                                    desc()->diff_weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(utils::one_of(desc()->alg_kind,
                                            alg_kind::deconvolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    utils::one_of(desc()->diff_weights_desc.data_type,
                            data_type::bf16, data_type::f16, data_type::f32,
                            data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_DECONVOLUTION_SC(
                    create_conv_pd(conv_pd_, engine, this, data_type::undef,
                            /* force_empty_bias = */ true),
                    "create_conv_pd()");
            if (diff_weights_md_.format_kind == format_kind::any) {
                diff_weights_md_
                        = utils::downcast<dnnl::impl::convolution_pd_t *>(
                                conv_pd_.get())
                                  ->weights_md_with_permute_channels();
                VDISPATCH_DECONVOLUTION(!types::is_zero_md(&diff_weights_md_),
                        "weights_md_with_permute_channels()");
            }
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (diff_dst_md_.format_kind == format_kind::any)
                diff_dst_md_ = *conv_pd_->src_md();
            if (diff_bias_md_.format_kind == format_kind::any) {
                VDISPATCH_DECONVOLUTION_SC(
                        memory_desc_init_by_tag(diff_bias_md_, x),
                        VERBOSE_UNSUPPORTED_TAG);
            }

            init_name();
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = "conv:any";

        void init_name() {
            name_.append("+");
            name_.append(conv_pd_->name());
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        // Creating convolution primitve
        CHECK(create_nested_primitive(nested_p_, pd()->conv_pd_, engine));

        if (!pd()->with_bias()) return status::success;
        // Initializing values for the deconv bias kernel
        compute::kernel_ctx_t kernel_ctx;

        memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        kernel_ctx.set_data_type(pd()->diff_dst_md()->data_type);
        kernel_ctx.require_stateless_addressing(pd()->has_large_buffers());
        kernel_ctx.register_buffer_size(diff_dst_mdw);
        kernel_ctx.register_buffer_size(*pd()->diff_weights_md(1));
        offsets_t off;
        set_offsets(diff_dst_mdw, off.dst_off);
        def_offsets(off.dst_off, kernel_ctx, "DST",
                pd()->desc()->diff_dst_desc.ndims);

        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OC", pd()->OC() / pd()->G());
        kernel_ctx.define_int("NDIMS", pd()->desc()->src_desc.ndims);

        gws[0] = pd()->OC();

        dst_data_type = pd()->diff_dst_md()->data_type;
        bias_data_type = pd()->diff_weights_md(1)->data_type;
        accum_data_type = pd()->desc()->accum_data_type;

        def_data_type(kernel_ctx, dst_data_type, "DST");
        def_data_type(kernel_ctx, bias_data_type, "BIA");
        def_data_type(kernel_ctx, accum_data_type, "ACC");

        CHECK(create_kernel(
                engine, &bias_kernel_, "deconv_backward_bias", kernel_ctx));
        if (!bias_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;

        const auto &args = ctx.args();
        exec_args_t nested_args;
        nested_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        nested_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        nested_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            nested_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t nested_ctx(ctx, std::move(nested_args));

        auto *nested_grantor
                = create_nested_grantor(ctx.get_scratchpad_grantor(),
                        key_nested, nested_p_->pd()->scratchpad_registry());
        nested_ctx.set_scratchpad_grantor(nested_grantor);

        status_t status = nested_p_->execute(nested_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            // Calling the bias kernel if bias=1
            auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
            auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, diff_dst);
            arg_list.set(1, diff_bias);

            // Setting up global work-space to {OC*G, 1, 1}
            auto nd_range = compute::nd_range_t(gws);
            status = parallel_for(ctx, nd_range, bias_kernel_, arg_list);
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> nested_p_;
    compute::kernel_t bias_kernel_;
    compute::range_t gws = compute::range_t::empty(1);
    data_type_t dst_data_type = data_type::undef;
    data_type_t bias_data_type = data_type::undef;
    data_type_t accum_data_type = data_type::undef;
};

} // namespace deconv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
