/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
#define GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct ref_grouped_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref_grouped:any", ref_grouped_t);

        // Weights are 3D: [G, K, N]
        // Override masks to include 0th expert dimension
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }

        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, dense 3D weights
            VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(wei_d.is_blocking_desc() && wei_d.ndims() == 3,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            ngroups_ = src_grouped.group_count;

            // GPU ref currently only supports matching data types
            VDISPATCH_MATMUL(src_dt_ == wei_dt_ && src_dt_ == dst_dt_
                            && utils::one_of(src_dt_, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Check for supported quantization schemes
            const auto &attr_scales = attr()->scales_;
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
                const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
                const int rowwise_mask = src_qmask_M();
                // Only row-wise f32 scales supported for src
                VDISPATCH_MATMUL(src_mask == rowwise_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(attr_scales.get_data_type(DNNL_ARG_SRC) == f32,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                // No groups for src scales
                VDISPATCH_MATMUL(
                        attr_scales.get(DNNL_ARG_SRC).has_default_groups(),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
                const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
                const int colwise_mask = wei_qmask_N();
                // Only column-wise f32 scales supported for weights
                VDISPATCH_MATMUL(wei_mask == colwise_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        attr_scales.get_data_type(DNNL_ARG_WEIGHTS) == f32,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                // No groups for weight scales
                VDISPATCH_MATMUL(
                        attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups(),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Zero-points are not supported
            VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(),
                    VERBOSE_UNSUPPORTED_ATTR);

            // No post-ops for now
            VDISPATCH_MATMUL(attr()->post_ops_.has_default_values(),
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }

        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;
        dim_t ngroups_ = 0;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_dt_);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        kernel_ctx.define_int("K", pd()->src_md()->dims[1]);
        kernel_ctx.define_int("N", pd()->weights_md(0)->dims[2]);
        kernel_ctx.define_int("NGROUPS", pd()->ngroups_);

        // Check if weights are transposed (acb format)
        memory_desc_wrapper wei_d(pd()->weights_md(0));
        const bool wei_transposed = wei_d.matches_tag(format_tag::acb);
        kernel_ctx.define_int("WEI_TRANSPOSED", wei_transposed ? 1 : 0);

        const bool with_bias = pd()->with_bias();
        kernel_ctx.define_int("WITH_BIAS", with_bias ? 1 : 0);
        if (with_bias) {
            def_data_type(kernel_ctx, pd()->weights_md(1)->data_type, "BIA");
        }

        const auto &attr_scales = pd()->attr()->scales_;
        const bool with_src_scales
                = !attr_scales.has_default_values(DNNL_ARG_SRC);
        kernel_ctx.define_int("WITH_SRC_SCALES", with_src_scales ? 1 : 0);

        const bool with_wei_scales
                = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
        kernel_ctx.define_int("WITH_WEI_SCALES", with_wei_scales ? 1 : 0);

        return create_kernel(
                engine, &kernel_, "ref_grouped_gemm_matmul", kernel_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
