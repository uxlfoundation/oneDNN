/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_WITH_POST_OPS_HPP
#define GPU_INTEL_GEMM_WITH_POST_OPS_HPP

#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/gemm/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

struct with_post_ops_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gemm::pd_t {
        using gemm::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:with_po:any", with_post_ops_t);

        status_t init(impl::engine_t *engine);

        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }
        status_t query(query_t what, int idx, void *result) const override {
            if (!pd_) return gemm::pd_t::query(what, idx, result);
            return pd_->query(what, idx, result);
        }

        std::shared_ptr<primitive_desc_t> pd_;
        bool use_scratchpad_with_post_op_worker = false;
        bool use_reorder = false;
        compute::dispatch_t dispatch_;
        attr_info_t attr_info_;
        bool subbyte_pack_ = false;
        bool mx_scales_ = false;
        data_type_t dst_type_ = data_type::undef;
        data_type_t acc_type_ = data_type::undef;
    };

    status_t init(impl::engine_t *engine) override {
        auto ret_status = create_nested_primitive(prim_, pd()->pd_, engine);
        CHECK(ret_status);
        primitive_attr_t attr;
        int threads_per_eu = 0;
        if (status::success
                == pd()->pd_->query(query::preferred_gpu_threads_per_eu, 0,
                        &threads_per_eu)) {
            CHECK(attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
        }
        compute::kernel_ctx_t kernel_ctx(&attr);
        ret_status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(ret_status);
        kernels_.resize(3);
        int kidx = 0;
        ret_status = create_kernel(
                engine, &kernels_[kidx++], "gemm_post_ops", kernel_ctx);
        const bool mx_scales = pd()->mx_scales_;
        if (mx_scales) {
            compute::kernel_ctx_t alt_ctx(pd()->attr());
            const auto src_info = memory_desc_info_t::create(pd_->dst_md(0));
            dnnl_memory_desc dst_md(*(pd()->dst_md(0)));
            dst_md.data_type = pd()->dst_type_;
            memory_desc_wrapper dst_d(dst_md);
            def_memory_desc_info(alt_ctx, src_info, "SRC", false);
            def_memory_desc_info(
                    alt_ctx, memory_desc_info_t::create(dst_d), "DST", false);
            const int ndims = dst_d.ndims();
            bool runtime_dims
                    = pd()->has_runtime_dims_or_strides() || ndims > 5;
            if (!runtime_dims) {
                offsets_t off;
                set_offsets(dst_d, off.dst_off);
                def_offsets(off.dst_off, alt_ctx, "DST", ndims);
                alt_ctx.define_int("NDIMS", ndims);
            }
            CHECK(create_kernel(
                    engine, &kernels_[kidx++], "mx_scale_dst", alt_ctx));
            if (pd()->subbyte_pack_)
                CHECK(create_kernel(
                        engine, &kernels_[kidx++], "subbyte_pack", alt_ctx));
        } else if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[kidx++], "subbyte_pack", kernel_ctx));
        return ret_status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> prim_;
    std::vector<compute::kernel_t> kernels_;
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
