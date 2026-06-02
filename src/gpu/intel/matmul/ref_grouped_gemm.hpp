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
#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"
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
        bool with_post_op_ = false;
        po_kind_t po_chain_[3]
                = {po_kind_t::none, po_kind_t::none, po_kind_t::none};
        data_type_t binary_scale_dts_[2] = {data_type::undef, data_type::undef};

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(impl::engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, dense 3D weights
            VDISPATCH_MATMUL(
                    dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(wei_d.is_blocking_desc() && wei_d.ndims() == 3,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // GPU ref currently only supports matching data types
            VDISPATCH_MATMUL(src_type == wei_type && src_type == dst_type
                            && utils::one_of(src_type, f32, bf16, f16),
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

            with_post_op_ = !attr()->post_ops_.has_default_values();
            if (with_post_op_) {
                CHECK(check_post_op_chain(*attr(), dst_d, group_count_,
                        po_chain_, binary_scale_dts_));
            }

            return status::success;
        }

        status_t init_2dby2d(impl::engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper dst_d(dst_md());

            // Resolve format_any to plain dense
            if (dst_d.format_any())
                CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

            // Only plain 3D dst is supported
            VDISPATCH_MATMUL(dst_d.is_plain(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(dst_d.ndims() == 3, VERBOSE_BAD_NDIMS, "dst",
                    dst_d.ndims());

            VDISPATCH_MATMUL(src_type == wei_type && src_type == dst_type
                            && utils::one_of(src_type, f32, bf16, f16),
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
        const auto &po_chain = pd()->po_chain_;
        bool with_binary_grouped_scale
                = (find_po_in_chain(po_chain, po_kind_t::binary_grouped_scale)
                        != -1);
        bool with_binary_dense_scale
                = (find_po_in_chain(po_chain, po_kind_t::binary_dense_scale)
                        != -1);
        bool with_binary_nvfp4_scale
                = (find_po_in_chain(po_chain, po_kind_t::binary_nvfp4_scale)
                        != -1);
        def_data_type(kernel_ctx, pd()->src_md()->data_type, "SRC");
        def_data_type(kernel_ctx, pd()->weights_md(0)->data_type, "WEI");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        kernel_ctx.define_int("WITH_POST_OP", pd()->with_post_op_);
        kernel_ctx.define_int(
                "WITH_BINARY_GROUPED_SCALE", with_binary_grouped_scale);
        kernel_ctx.define_int(
                "WITH_BINARY_DENSE_SCALE", with_binary_dense_scale);
        kernel_ctx.define_int(
                "WITH_BINARY_NVFP4_SCALE", with_binary_nvfp4_scale);

        auto define_binary_scale_dt = [](compute::kernel_ctx_t &ctx,
                                              data_type_t dt, const char *pfx) {
            if (dt == data_type::f16)
                ctx.define_int(std::string(pfx) + "_DT_F16", 1);
            else if (dt == data_type::bf16)
                ctx.define_int(std::string(pfx) + "_DT_BF16", 1);
            else
                ctx.define_int(std::string(pfx) + "_DT_F32", 1);
        };

        if (with_binary_grouped_scale) {
            define_binary_scale_dt(kernel_ctx, pd()->binary_scale_dts_[0],
                    "BINARY_SCALE_GROUPED");
        }

        if (with_binary_dense_scale) {
            define_binary_scale_dt(kernel_ctx, pd()->binary_scale_dts_[1],
                    "BINARY_SCALE_DENSE");
        }

        kernel_ctx.add_custom_header("grouped_post_ops.h",
                generate_post_ops_refgemm_header(*pd()->attr(), po_chain));

        const bool with_bias = pd()->with_bias();
        const auto &attr_scales = pd()->attr()->scales_;
        const bool with_src_scales
                = !attr_scales.has_default_values(DNNL_ARG_SRC);
        const bool with_wei_scales
                = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);

        kernel_ctx.define_int("WITH_BIAS", with_bias ? 1 : 0);
        kernel_ctx.define_int("WITH_SRC_SCALES", with_src_scales ? 1 : 0);
        kernel_ctx.define_int("WITH_WEI_SCALES", with_wei_scales ? 1 : 0);
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
