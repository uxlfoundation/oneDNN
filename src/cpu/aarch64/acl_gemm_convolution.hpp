/*******************************************************************************
* Copyright 2020-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP

#include "common/memory_tracking.hpp"
#include "cpu/cpu_convolution_pd.hpp"

#include "acl_convolution_utils.hpp"
#include "acl_post_ops.hpp"
#include "arm_compute/runtime/experimental/operators/CpuGemmConv2d.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <data_type_t src_type, data_type_t wei_type = src_type,
        data_type_t dst_type = src_type, data_type_t bia_type = dst_type>
struct acl_gemm_convolution_fwd_t : public primitive_t {

    using Op = arm_compute::experimental::op::CpuGemmConv2d;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), acp_() {}

        DECLARE_COMMON_PD_T(
                "gemm:acl", acl_gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(
                            src_type, wei_type, bia_type, dst_type, undef)
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::fpmath_mode, dst_type)
                    && output_scales_mask_ok() && zero_points_ok();
            if (!ok) return status::unimplemented;

            CHECK(acl_convolution_utils::init_conf_gemm(acp_, src_md_,
                    weights_md_, dst_md_, bias_md_, *desc(), *attr()));

            CHECK(post_ops.init(
                    engine, attr_.post_ops_, dst_md_, acp_.act_info));
            acp_.use_dst_acc_for_sum = post_ops.has_sum();

            if (acp_.use_dst_acc_for_sum) {
                const memory_desc_wrapper dst_d(&dst_md_);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_none,
                        dst_d.nelems(), dst_d.data_type_size());
            }

            return status::success;
        }

        acl_conv_conf_t acp_;
        acl_post_ops_t post_ops;
    };

    // hot fix solution for stateless API which should be replaced soon.
    // acl_gemm_convolution_fwd_t(const pd_t *apd)
    //     : primitive_t(apd), acl_obj_(std::make_unique<acl_obj_t<Op>>()) {}
    acl_gemm_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    using src_data_t = typename prec_traits<src_type>::type;
    using wei_data_t = typename prec_traits<wei_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;
    using bia_data_t = typename prec_traits<bia_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;

    // hot fix solution for stateless API which should be replaced soon.
    std::unique_ptr<acl_obj_t<Op>> reinitialize_acl_obj() const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // commented due to hot fix solution for stateless API which should be replaced soon.
    // std::unique_ptr<acl_obj_t<Op>> acl_obj_;

}; // acl_gemm_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
