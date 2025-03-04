/*******************************************************************************
* Copyright 2022-2023, 2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_POOLING_HPP
#define CPU_AARCH64_ACL_POOLING_HPP

#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/experimental/operators/CpuPool2d.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_pooling_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PoolingLayerInfo pool_info;
    arm_compute::TensorInfo ws_info;
    bool use_ws;
};

struct acl_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;
        DECLARE_COMMON_PD_T("acl", acl_pooling_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        bool use_acl_avg_pool_heuristic(int problem_size, int thread_count,
                bool is_nhwc, bool use_square_acl_kernel);
        bool use_acl_max_pool_heuristic(int problem_size, int thread_count,
                bool is_nhwc, bool use_square_acl_kernel, bool is_training);

        status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
                const arm_compute::experimental::MemoryRequirements
                        &aux_mem_req);

        acl_pooling_conf_t asp_;
        arm_compute::experimental::MemoryRequirements aux_mem_req_;
    }; //pd_t

    // constructor
    acl_pooling_fwd_t(const pd_t *apd);

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;
    status_t init(engine_t *engine) override;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    inline const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
    std::unique_ptr<arm_compute::experimental::op::CpuPool2d> pooling_op_;
}; // acl_pooling_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_POOLING_HPP
