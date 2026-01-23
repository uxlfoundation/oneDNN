/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_INTEL_ZEROPAD_SIMPLE_HPP
#define GPU_INTEL_ZEROPAD_SIMPLE_HPP

#include "gpu/intel/primitive.hpp"
#include "gpu/intel/zeropad/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace zeropad {

#define VDISPATCH_ZERO_PAD(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, zero_pad, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

struct simple_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public zeropad::pd_t {
        using zeropad::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:simple:any", simple_t);
        status_t init(impl::engine_t *engine) { return status::success; }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        kernel_ctx.require_stateless_addressing(pd()->has_large_buffers());

        std::vector<compute::kernel_t> kernels {};
        CHECK(create_kernels(engine, &kernels,
                {"simple_zero_pad", "simple_zero_pad_subg_16",
                        "simple_zero_pad_subg_16_mask_and_clear_dt_1b"},
                kernel_ctx));

        kernel_ = kernels[0];
        kernel_subg16_ = kernels[1];
        kernel_subg16_mask_and_clear_dt_1b_ = kernels[2];
        auto *compute_engine = utils::downcast<intel::engine_t *>(engine);

        can_use_subg16_ = compute_engine->mayiuse_sub_group(16);

        if (!kernel_) return status::runtime_error;
        if (can_use_subg16_
                && (!kernel_subg16_ || !kernel_subg16_mask_and_clear_dt_1b_))
            return status::runtime_error;

        return status::success;
    }
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_simple(const exec_ctx_t &ctx) const;
    status_t execute_subg_16(const exec_ctx_t &ctx,
            const memory_desc_wrapper &mdw,
            const blocking_desc_t &blocking_desc) const;
    status_t execute_subg_16_mask_and_clear_dt_1B(const exec_ctx_t &ctx,
            const memory_desc_wrapper &mdw,
            const blocking_desc_t &blocking_desc) const;
    bool can_use_subg16_ = false;
    compute::kernel_t kernel_;
    compute::kernel_t kernel_subg16_;
    compute::kernel_t kernel_subg16_mask_and_clear_dt_1b_;
};

} // namespace zeropad
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
