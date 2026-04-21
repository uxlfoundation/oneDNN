/*******************************************************************************
* Copyright 2023-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP
#define CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/experimental/operators/CpuMeanStdDevNormalization.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_layer_norm_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
};

struct acl_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_layer_normalization_fwd_t);

        status_t init(engine_t *engine);
        format_tag_t get_channels_last_format(size_t ndim) const;

        acl_layer_norm_conf_t acl_lnorm_conf;
    }; // pd_t

    acl_layer_normalization_fwd_t(const pd_t *apd);

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }
    status_t init(engine_t *engine) override;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<arm_compute::experimental::op::CpuMeanStdDevNormalization>
            acl_obj_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP
