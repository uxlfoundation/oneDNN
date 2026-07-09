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

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
struct ref_grouped_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref_grouped:any", ref_grouped_t);

        // For 3D weights [G, K, N], override masks to include 0th expert dim
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }
        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        bool is_2dby2d() const { return is_2dby2d_; }

        status_t init(impl::engine_t *engine) {
            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            VDISPATCH_MATMUL(
                    !with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "reduce");

            // Detect pattern (2Dx3D vs 2Dx2D) and initialize
            VDISPATCH_MATMUL(
                    src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            is_2dby2d_ = wei_d.is_grouped_desc();

            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            group_count_ = src_grouped.group_count;

            return is_2dby2d_ ? init_2dby2d(engine) : init_2dby3d(engine);
        }

        dim_t group_count_ = 0;
        // Re-interpretted binary src1 as 3D view, while attr_.post_ops_
        // keeps the original grouped md
        post_ops_t generic_po_;
        // Re-interpretted grouped dst md as 3D view
        memory_desc_t group_po_dst_md_ = types::zero_md();

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(impl::engine_t *engine) {
            using namespace data_type;

            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, dense 3D weights
            VDISPATCH_MATMUL(
                    dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(wei_d.is_blocking_desc() && wei_d.ndims() == 3,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // GPU ref currently only supports matching data types
            const auto src_dt = src_md()->data_type;
            VDISPATCH_MATMUL(src_dt == weights_md(0)->data_type
                            && src_dt == dst_md()->data_type
                            && utils::one_of(src_dt, f32, bf16, f16),
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
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (attr_.post_ops_.len() > 0) CHECK(setup_post_ops(engine));

            return status::success;
        }

        // Re-interpret src1 as a 3D view so the generic post-op code applies:
        //   per-group  [G, 1]       -> [G, 1, 1]
        //   per-token  [total_M, 1] -> [1, total_M, 1]
        //   per-token  [total_M, N] -> [1, total_M, N]
        status_t setup_post_ops(impl::engine_t *engine) {
            auto &attr_po = attr_.post_ops_;
            generic_po_ = attr_po;

            const dim_t total_tokens = src_md()->dims[0];
            const dim_t N = dst_md()->dims[1];

            // 3D view of grouped dst [G, DNNL_RUNTIME_DIM_VAL, N]
            const dims_t po_dst_dims = {group_count_, DNNL_RUNTIME_DIM_VAL, N};
            CHECK(memory_desc_init_by_strides(group_po_dst_md_, 3, po_dst_dims,
                    dst_md()->data_type, nullptr));

            for (int i = 0; i < attr_po.len(); ++i) {
                auto &e = attr_po.entry_[i];
                if (!e.is_binary()) continue;

                auto &attr_src1 = e.binary.src1_desc;
                // resolve format_any (e.g. NVFP4 per-group scale)
                if (memory_desc_wrapper(attr_src1).format_any())
                    CHECK(memory_desc_init_by_strides(attr_src1, nullptr));

                const memory_desc_wrapper src1_mdw(attr_src1);
                const bool per_group = src1_mdw.ndims() == 2
                        && src1_mdw.dims()[0] == group_count_
                        && src1_mdw.dims()[1] == 1;

                const dims_t dims_3d = {per_group ? group_count_ : 1,
                        per_group ? 1 : total_tokens, src1_mdw.dims()[1]};
                CHECK(memory_desc_init_by_strides(
                        generic_po_.entry_[i].binary.src1_desc, 3, dims_3d,
                        src1_mdw.data_type(), nullptr));
            }
            return status::success;
        }

        status_t init_2dby2d(impl::engine_t *engine) {
            using namespace data_type;

            memory_desc_wrapper dst_d(dst_md());

            // Resolve format_any to plain dense
            if (dst_d.format_any())
                CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

            // Only plain 3D dst is supported
            VDISPATCH_MATMUL(dst_d.is_plain(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(dst_d.ndims() == 3, VERBOSE_BAD_NDIMS, "dst",
                    dst_d.ndims());

            VDISPATCH_MATMUL(src_md()->data_type == weights_md(0)->data_type
                            && src_md()->data_type == dst_md()->data_type
                            && utils::one_of(
                                    src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_md()->data_type);
        def_data_type(kernel_ctx, pd()->src_md()->data_type, "SRC");
        def_data_type(kernel_ctx, pd()->weights_md(0)->data_type, "WEI");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        auto attr_info = attr_info_t::create(pd()->attr());
        CHECK(def_attr_info(kernel_ctx, attr_info, pd()->generic_po_,
                pd()->group_po_dst_md_));

        const bool with_bias = pd()->with_bias();
        kernel_ctx.define_int("WITH_BIAS", with_bias ? 1 : 0);
        if (with_bias)
            def_data_type(kernel_ctx, pd()->weights_md(1)->data_type, "BIA");

        return create_kernel(
                engine, &kernel_, "ref_grouped_gemm_matmul", kernel_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
