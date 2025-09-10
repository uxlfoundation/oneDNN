/*******************************************************************************
* Copyright 2021-2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_INNER_PRODUCT_HPP
#define CPU_AARCH64_ACL_INNER_PRODUCT_HPP

#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_inner_product_pd.hpp"

#include "arm_compute/runtime/experimental/operators/CpuFullyConnected.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_ip_conf_t {
    bool with_bias;
    // If this is true, the result of the inner product goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc_for_sum;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::FullyConnectedLayerInfo fc_info;
    // Additional information about the weights not included in wei_tensor_info
    arm_compute::WeightsInfo weights_info;
};

struct acl_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_inner_product_fwd_t);

        status_t init(engine_t *engine);

        acl_ip_conf_t aip_ = utils::zero<decltype(aip_)>();

        acl_post_ops_t post_ops;

        status_t init_conf_ip(
                engine_t *engine, format_kind_t weights_format_kind_received);
    }; // pd_t

    // constructor
    acl_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t init(engine_t *engine) override;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    inline const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::unique_ptr<arm_compute::experimental::op::CpuFullyConnected>
            inner_product_op_;
}; // acl_inner_product_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INNER_PRODUCT_HPP
