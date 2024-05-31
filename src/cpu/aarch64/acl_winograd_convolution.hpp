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

#ifndef CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP

#include "cpu/cpu_convolution_pd.hpp"

#include "acl_convolution_utils.hpp"
#include "arm_compute/runtime/experimental/operators/CpuWinogradConv2d.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_wino_convolution_fwd_t : public primitive_t {
    using Op = arm_compute::experimental::op::CpuWinogradConv2d;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                "wino:acl", acl_wino_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f16);
            const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32);
            bool ok = is_fwd()
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_winograd)
                    && utils::one_of(true, is_fp16_ok, is_fp32_ok)
                    && !has_zero_dim_memory();

            ok = ok && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL;
            if (!ok) return status::unimplemented;

            CHECK(acl_convolution_utils::init_conf_wino(acp_, src_md_,
                    weights_md_, dst_md_, bias_md_, *desc(), *attr()));

            set_default_alg_kind(alg_kind::convolution_winograd);

            CHECK(post_ops.init(
                    engine, attr_.post_ops_, dst_md_, acp_.act_info));
            acp_.use_dst_acc_for_sum = post_ops.has_sum();

            if (acp_.use_dst_acc_for_sum) {
                const memory_desc_wrapper dst_d(&dst_md_);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_generic_acc,
                        dst_d.nelems(), dst_d.data_type_size());
            }

            return status::success;
        }

        acl_conv_conf_t acp_;
        acl_post_ops_t post_ops;

    private:
        status_t init_conf();
    };

    // hot fix solution for stateless API which should be replaced soon.
    // acl_wino_convolution_fwd_t(const pd_t *apd)
    //     : primitive_t(apd), acl_obj_(std::make_unique<acl_obj_t<Op>>()) {}
    acl_wino_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

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
}; // acl_wino_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP
