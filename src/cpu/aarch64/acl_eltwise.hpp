/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_ELTWISE_HPP
#define CPU_AARCH64_ACL_ELTWISE_HPP

#include <memory>
#include "cpu/cpu_eltwise_pd.hpp"

#include "acl_utils.hpp"

#include "arm_compute/runtime/experimental/operators/CpuActivation.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_eltwise_conf_t {
    arm_compute::ActivationLayerInfo act_info;
    // src and dst have the same info
    arm_compute::TensorInfo data_info;
};

struct acl_eltwise_fwd_t : public primitive_t {
    using Op = arm_compute::experimental::op::CpuActivation;

    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;
            const memory_desc_wrapper src_d(src_md());

            bool ok = is_fwd() && one_of(src_d.data_type(), f32, f16, s32, s8)
                    && !has_zero_dim_memory() && attr()->has_default_values()
                    && set_default_formats_common() && src_d.is_dense()
                    && src_d == memory_desc_wrapper(dst_md());
            if (!ok) return status::unimplemented;

            // Workaround for the nan-value caused by tanh of ACL for fp16
            // ARM-software/ComputeLibrary#998. Workaround for the inaccuracies
            // caused by logistic/soft_relu/elu of ACL for fp16.
            // TODO: Relax the error bounds in eltwise checks, or rework these
            // fp16 operations in ACL for better accuracy.
            using namespace dnnl::impl::alg_kind;
            if (src_d.data_type() == f16
                    && utils::one_of(desc_.alg_kind, eltwise_tanh,
                            eltwise_logistic, eltwise_soft_relu, eltwise_elu,
                            eltwise_gelu_erf)) {
                return status::unimplemented;
            }

            auto acl_data_t = acl_utils::get_acl_data_t(src_d.data_type());

            // Operator acts elementwise, so we only require that the product of
            // all the dimensions equals the total number of elements. We are
            // free to swap/combine dimensions. ACL performs SIMD parallelism
            // over the first dimension and thread parallelism over the second.
            // We pick a single dimension to thread over (taking the max of 2 to
            // reduce the chance of it being 1), with the remaining dimensions
            // to SIMD over.
            dim_t thread_dim = std::max(W(), ndims() >= 2 ? C() : 1);
            auto shape = arm_compute::TensorShape(
                    src_d.nelems() / thread_dim, thread_dim);
            aep.data_info = arm_compute::TensorInfo(shape, 1, acl_data_t);

            CHECK(acl_utils::convert_to_acl_act(desc_, aep.act_info));

            ACL_CHECK_VALID(arm_compute::NEActivationLayer::validate(
                    &aep.data_info, &aep.data_info, aep.act_info));

            return status::success;
        }

        acl_eltwise_conf_t aep;

        friend struct acl_post_ops_t;
    };

    acl_eltwise_fwd_t(const pd_t *apd)
        : primitive_t(apd), act_(std::make_unique<Op>()) {}

    status_t execute(const exec_ctx_t &ctx) const override;

    status_t init(engine_t *engine) override;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;

    // Execute forward with arbitrary src and dst, used by acl_post_ops_t
    status_t execute_forward(
            const exec_ctx_t &ctx, const void *src, void *dst) const;

    const pd_t *pd() const;

    std::unique_ptr<Op> act_;

    friend struct acl_post_ops_t;
}; // acl_eltwise_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_ELTWISE_HPP
