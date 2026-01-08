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

#ifndef GPU_INTEL_ELTWISE_REF_HPP
#define GPU_INTEL_ELTWISE_REF_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/serialization.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/eltwise/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace eltwise {

struct ref_jit_params_t : trivially_serializable_t<ref_jit_params_t> {
    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names_fwd {"ref_eltwise_fwd"};
        static const std::vector<const char *> names_bwd {"ref_eltwise_bwd"};
        return core.is_fwd ? names_fwd : names_bwd;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;

    serialization_stream_t serialize() const {
        return serialization_stream_t(core, post_ops);
    }

    struct core_t {
        compute::dispatch_compile_params_t params;
        alg_kind_t alg_kind = {};
        int ndims = {};
        data_type_t src_dt = {};
        data_type_t dst_dt = {};
        data_type_t diff_src_dt = {};
        data_type_t diff_dst_dt = {};
        bool is_fwd = {};
        bool requrie_stateless_addressing = {};
        int8_t pad[6] = {};
    };
    DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(core_t);

    core_t core;
    gpu_post_ops_t post_ops;
};

struct ref_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gpu_eltwise_fwd_pd_t {
        using gpu_eltwise_fwd_pd_t::gpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            using namespace alg_kind;
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");

            VDISPATCH_ELTWISE(memory_desc_ndims_ok(dst_md()), VERBOSE_BAD_NDIMS,
                    "dst_md", dst_md()->ndims);
            VDISPATCH_ELTWISE(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    post_ops_with_binary_ok(attr(), *dst_md(), MAX_NDIMS),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f64,
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f16,
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        ref_jit_params_t conf;
        compute::dispatch_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        return create_kernel(engine, kernel_, "ref_eltwise_fwd", pd()->conf);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_dense(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gpu_eltwise_bwd_pd_t {
        using gpu_eltwise_bwd_pd_t::gpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            assert(engine->kind() == engine_kind::gpu);
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(memory_desc_ndims_ok(data_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "data_md",
                    "diff_dst_md", data_md()->ndims, diff_dst_md()->ndims);
            VDISPATCH_ELTWISE(
                    utils::one_of(data_md()->data_type, data_type::f32,
                            data_type::f16, data_type::bf16, data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f64,
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f16,
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        ref_jit_params_t conf;
        compute::dispatch_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        return create_kernel(engine, kernel_, "ref_eltwise_bwd", pd()->conf);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        CHECK(execute_backward_dense(ctx));
        return status::success;
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace eltwise
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
