/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
#define GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct grouped_micro_params_t
    : trivially_serializable_t<grouped_micro_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"grouped_micro_gemm"};
        return kernel_names;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;
};

struct grouped_micro_gemm_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("grouped_gemm:micro", grouped_micro_gemm_t);

        status_t init(impl::engine_t *engine);

        int sg_size_ = 0;
        bool use_systolic_ukernel_ = true;
        bool use_256_grf_ = false;
        dim_t sg_per_wg_m_ = 0;
        dim_t sg_per_wg_n_ = 0;
        dim_t sg_tile_m_ = 0;
        dim_t sg_tile_n_ = 0;
        dim_t ngroups_ = 0;
        std::array<dim_t, 2> src_group_sizes_ = {0, 0};
        std::array<dim_t, 3> wei_group_sizes_ = {0, 0, 0};

    private:
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;
    };
    status_t init(impl::engine_t *engine) override;
    status_t init_microkernels(impl::engine_t *engine);

    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_ctx_t kernel_ctx_;
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
