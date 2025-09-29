/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_MATMUL_ACL_LOWP_MATMUL_SQ_HPP
#define CPU_AARCH64_MATMUL_ACL_LOWP_MATMUL_SQ_HPP

#include <memory>

#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/runtime/experimental/operators/CpuGEMMLowp.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_lowp_matmul_sq_conf_t {
    bool with_bias;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::GEMMInfo gemm_info;
};

struct acl_lowp_matmul_sq_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("lowp_gemm_sq:acl", acl_lowp_matmul_sq_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        status_t init_scratchpad(engine_t *engine,
                memory_tracking::registrar_t &scratchpad,
                acl_post_ops_t &post_ops, dnnl::impl::post_ops_t &attr_post_ops,
                arm_compute::ActivationLayerInfo &act_info,
                const dnnl::impl::memory_desc_t &dst_md,
                const arm_compute::experimental::MemoryRequirements
                        &aux_mem_req);

        acl_lowp_matmul_sq_conf_t almc_;
        acl_post_ops_t acl_post_ops;
    };

    // constructor
    acl_lowp_matmul_sq_t(const pd_t *apd)
        : primitive_t(apd)
        , gemm_(std::make_unique<
                  arm_compute::experimental::op::CpuGEMMLowp>()) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx_;
    inline const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
    std::unique_ptr<arm_compute::experimental::op::CpuGEMMLowp> gemm_;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_MATMUL_ACL_LOWP_MATMUL_SQ_HPP
