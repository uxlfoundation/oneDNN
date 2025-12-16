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

struct ref_grouped_gemm_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref_grouped:any", ref_grouped_gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // src, dst - grouped, weights - dense
            VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    !wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            const auto &dst_grouped = dst_d.sparse_desc().grouped_desc;

            VDISPATCH_MATMUL(src_grouped.ngroups == dst_grouped.ngroups,
                    VERBOSE_INCONSISTENT_DIM, "src_ngroups",
                    (int)src_grouped.ngroups, "dst_ngroups",
                    (int)dst_grouped.ngroups);

            ngroups_ = src_grouped.ngroups;

            // only supported dt for now
            VDISPATCH_MATMUL(utils::one_of(src_dt_, f32, f16)
                            && src_dt_ == wei_dt_ && src_dt_ == dst_dt_,
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // only supported offsets type for now
            VDISPATCH_MATMUL(src_d.metadata_type(0) == s32
                            && dst_d.metadata_type(0) == s32,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // no bias/scales/post-ops for now
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

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

        // bias not supported yet
        kernel_ctx.define_int("WITH_BIAS", 0);

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

#endif // GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
