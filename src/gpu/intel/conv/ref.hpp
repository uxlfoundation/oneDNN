/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_CONV_REF_HPP
#define GPU_INTEL_CONV_REF_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {

struct ref_convolution_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_fwd_pd_t {
        using gpu_convolution_fwd_pd_t::gpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_convolution_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const bool is_int8 = utils::one_of(src_md_.data_type, s8, u8);
            const bool is_fp8
                    = utils::one_of(src_md_.data_type, f8_e5m2, f8_e4m3);

            using sm = primitive_attr_t::skip_mask_t;
            auto attr_skip_mask = sm::post_ops | sm::sum_dt | sm::rounding_mode;
            if (is_int8) {
                attr_skip_mask
                        |= sm::zero_points_data_type | sm::scales_data_type;
            } else if (is_fp8) {
                attr_skip_mask |= sm::scales_data_type;
            }

            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(desc()->alg_kind == alg_kind::convolution_direct,
                    VERBOSE_BAD_ALGORITHM);

            VDISPATCH_CONV(utils::one_of(desc()->prop_kind,
                                   prop_kind::forward_training,
                                   prop_kind::forward_inference),
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_CONV(IMPLICATION(utils::one_of(f16, src_md_.data_type,
                                               weights_md_.data_type,
                                               dst_md_.data_type),
                                   compute_engine->mayiuse(
                                           compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(
                    IMPLICATION(
                            utils::one_of(f64, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)
                                    && attr()->post_ops_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_CONV(
                    !memory_desc_ndims_ok(src_md(), weights_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src, wei", "dst");
            VDISPATCH_CONV(
                    this->set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(attr()->has_default_values(
                                   attr_skip_mask, dst_md_.data_type),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(
                                   dst_md_.data_type, is_int8, true),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(post_ops_with_binary_ok(attr(), *dst_md(), 5),
                    VERBOSE_UNSUPPORTED_POSTOP);

            CHECK(attr_scales_ok({{DNNL_ARG_SRC, {0}},
                    {DNNL_ARG_WEIGHTS, {0, 1}}, {DNNL_ARG_DST, {0, 2}}}));
            CHECK(attr_zero_points_ok({{DNNL_ARG_SRC, {0, 2}},
                    {DNNL_ARG_WEIGHTS, {0}}, {DNNL_ARG_DST, {0, 2}}}));

            subbyte_pack_ = utils::one_of(
                    dst_md_.data_type, data_type::f4_e2m1, data_type::f4_e3m0);
            if (subbyte_pack_) {
                using namespace dnnl::impl::memory_tracking::names;
                const memory_desc_wrapper dst_mdw(dst_md(0));
                const auto &padded_dims = dst_mdw.padded_dims();
                const dim_t ndims = dst_mdw.ndims();
                const dim_t nelems = utils::array_product(padded_dims, ndims);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_conv_pack_space,
                        nelems, sizeof(char), OCL_BUFFER_ALIGNMENT);
            }

            return init_conf(engine);
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        bool subbyte_pack_ = false;

        conv_conf_t conf;

    private:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;
        kernels_.resize(2);

        CHECK(create_kernel(
                engine, &kernels_[0], "ref_convolution_fwd", kernel_ctx));
        if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[1], "subbyte_pack", kernel_ctx));
        if (!kernels_[0]) return status::runtime_error;
        if (pd()->subbyte_pack_ && !kernels_[1]) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
};

struct ref_convolution_bwd_data_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        using gpu_convolution_bwd_data_pd_t::gpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_convolution_bwd_data_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;
            auto attr_skip_mask = sm::post_ops | sm::scales;
            if (utils::one_of(invariant_dst_md()->data_type, s8, u8)) {
                attr_skip_mask |= sm::zero_points_data_type;
            }
            using namespace data_type;
            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(desc()->alg_kind == alg_kind::convolution_direct,
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_CONV(
                    IMPLICATION(utils::one_of(f64, diff_src_md()->data_type,
                                        dst_md()->data_type),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)
                                    && attr()->post_ops_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_CONV(!memory_desc_ndims_ok(diff_src_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "diff_dst");

            VDISPATCH_CONV(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            CHECK(attr_scales_ok({{DNNL_ARG_SRC, {0}},
                    {DNNL_ARG_WEIGHTS, {0, 1}}, {DNNL_ARG_DST, {0, 2}}}));
            CHECK(attr_zero_points_ok(
                    {{DNNL_ARG_SRC, {0, 2}}, {DNNL_ARG_DST, {0, 2}}}));

            VDISPATCH_CONV(
                    this->set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_CONV(post_ops_with_binary_ok(attr(), *dst_md(), ndims()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV_SC(attr_.set_default_formats(diff_src_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            subbyte_pack_
                    = utils::one_of(dst_md()->data_type, f4_e2m1, f4_e3m0);
            if (subbyte_pack_) {
                using namespace dnnl::impl::memory_tracking::names;
                const memory_desc_wrapper dst_mdw(dst_md(0));
                const auto &padded_dims = dst_mdw.padded_dims();
                const dim_t ndims = dst_mdw.ndims();
                const dim_t nelems = utils::array_product(padded_dims, ndims);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_conv_pack_space,
                        nelems, sizeof(char), OCL_BUFFER_ALIGNMENT);
            }

            return init_conf(engine);
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        bool subbyte_pack_ = false;

        conv_conf_t conf;

    private:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        kernels_.resize(2);
        CHECK(create_kernel(
                engine, &kernels_[0], "ref_convolution_bwd_data", kernel_ctx));
        if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[1], "subbyte_pack", kernel_ctx));
        if (!kernels_[0]) return status::runtime_error;
        if (pd()->subbyte_pack_ && !kernels_[1]) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
};

struct ref_convolution_bwd_weights_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        using gpu_convolution_bwd_weights_pd_t::
                gpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_convolution_bwd_weights_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(desc()->alg_kind == alg_kind::convolution_direct,
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_CONV(
                    IMPLICATION(utils::one_of(f16, desc()->src_desc.data_type,
                                        desc()->diff_weights_desc.data_type,
                                        desc()->diff_dst_desc.data_type),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(
                    IMPLICATION(utils::one_of(f64, desc()->src_desc.data_type,
                                        desc()->diff_dst_desc.data_type),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)
                                    && attr()->post_ops_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_CONV(!memory_desc_ndims_ok(src_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "diff_dst");

            VDISPATCH_CONV(utils::one_of(desc()->diff_weights_desc.data_type,
                                   f32, bf16, f16, f64, f8_e5m2, f8_e4m3),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(
                    utils::one_of(desc()->src_desc.data_type, f32, bf16, f16,
                            f64, f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(
                    utils::one_of(desc()->diff_dst_desc.data_type, f32, bf16,
                            f16, f64, f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_CONV(
                    this->set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            subbyte_pack_ = utils::one_of(dst_md()->data_type,
                    data_type::f4_e2m1, data_type::f4_e3m0);
            if (subbyte_pack_) {
                using namespace dnnl::impl::memory_tracking::names;
                const memory_desc_wrapper dst_mdw(dst_md(0));
                const auto &padded_dims = dst_mdw.padded_dims();
                const dim_t ndims = dst_mdw.ndims();
                const dim_t nelems = utils::array_product(padded_dims, ndims);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_conv_pack_space,
                        nelems, sizeof(char), OCL_BUFFER_ALIGNMENT);
            }

            return init_conf(engine);
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        bool subbyte_pack_ = false;

        conv_conf_t conf;

    private:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        kernels_.resize(2);
        CHECK(create_kernel(engine, &kernels_[0], "ref_convolution_bwd_weights",
                kernel_ctx));
        if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[1], "subbyte_pack", kernel_ctx));
        if (!kernels_[0]) return status::runtime_error;
        if (pd()->subbyte_pack_ && !kernels_[1]) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
};

} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
