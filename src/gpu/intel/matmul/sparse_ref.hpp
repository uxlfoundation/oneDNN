/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_SPARSE_REF_HPP
#define GPU_INTEL_MATMUL_SPARSE_REF_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/matmul/types.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct ref_sparse_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_sparse_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            bool is_f16_dt = utils::everyone_is(f16, src_dt_, wei_dt_, dst_dt_);
            bool is_f32_dt = utils::everyone_is(f32, src_dt_, wei_dt_, dst_dt_);
            VDISPATCH_MATMUL(
                    is_f32_dt || is_f16_dt, VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_MATMUL(dst_md()->ndims == 2, VERBOSE_BAD_NDIMS, "dst",
                    dst_md()->ndims);

            // One of tensors must be sparse, not none, not both.
            VDISPATCH_MATMUL(src_d.is_sparse_desc() ^ wei_d.is_sparse_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            const auto &sparse_d = src_d.is_sparse_desc() ? src_d : wei_d;
            const auto &dense_d = src_d.is_sparse_desc() ? wei_d : src_d;
            VDISPATCH_MATMUL(
                    utils::one_of(sparse_d.encoding(), sparse_encoding::coo,
                            sparse_encoding::csr),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            bool is_meta_data_valid = sparse_d.metadata_type(0) == s32;
            if (sparse_d.encoding() == sparse_encoding::csr)
                is_meta_data_valid = is_meta_data_valid
                        && sparse_d.metadata_type(1) == s32;
            VDISPATCH_MATMUL(
                    is_meta_data_valid, VERBOSE_UNSUPPORTED_SPARSE_CFG);

            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            bool dense_tag_check
                    = dense_d.matches_one_of_tag(format_tag::ab, format_tag::ba)
                    != format_tag::undef;
            VDISPATCH_MATMUL(dense_tag_check, VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_MATMUL(
                    !with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "reduce");

            return status::success;
        }

        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        int ndims = pd()->dst_md()->ndims;

        kernel_ctx.set_data_type(pd()->dst_dt_);
        kernel_ctx.require_stateless_addressing(pd()->has_large_buffers());

        const memory_desc_wrapper src_d(pd()->src_md(0));
        const memory_desc_wrapper wei_d(pd()->weights_md(0));
        const memory_desc_wrapper dst_d(pd()->dst_md(0));

        bool is_src_sparse = src_d.is_sparse_desc();
        const auto &sparse_d = is_src_sparse ? src_d : wei_d;
        bool is_csr = sparse_d.encoding() == sparse_encoding::csr;
        kernel_ctx.define_int("IS_CSR", is_csr ? 1 : 0);
        kernel_ctx.define_int("SPARSE_WEI", is_src_sparse ? 0 : 1);

        offsets_t off;
        if (!src_d.is_sparse_desc()) {
            set_offsets(src_d, off.src_off);
            def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        }
        if (!wei_d.is_sparse_desc()) {
            set_offsets(wei_d, off.wei_off);
            def_offsets(off.wei_off, kernel_ctx, "WEI", ndims);
        }
        set_offsets(dst_d, off.dst_off);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);
        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("K", pd()->src_md()->dims[1]);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        CHECK(create_kernel(engine, &kernel_, "ref_sparse_matmul", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
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

#endif
