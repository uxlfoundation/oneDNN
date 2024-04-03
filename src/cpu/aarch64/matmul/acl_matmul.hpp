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

#ifndef ACL_MATMUL_HPP
#define ACL_MATMUL_HPP

#include "common/utils.hpp"
#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const cpu_matmul_pd_t *hint_fwd_pd)
            : cpu_matmul_pd_t(adesc, attr, hint_fwd_pd), amp_() {}

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:acl", acl_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            const bool is_fp32_ok
                    = utils::everyone_is(data_type::f32, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type,
                              desc()->accum_data_type)
                    && platform::has_data_type_support(data_type::f32);
            const bool is_fp16_ok
                    = utils::everyone_is(data_type::f16, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type::f16);
            const bool is_bf16_ok
                    = utils::everyone_is(data_type::bf16, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type::bf16);

            // we need to save this state as it can change inside set_default_formats()
            weights_format_kind_ = weights_md_.format_kind;

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    utils::one_of(true, is_fp32_ok, is_fp16_ok, is_bf16_ok),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::oscale
                            | smask_t::post_ops | smask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(attr_oscale_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(!has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            if (weights_format_kind_ == format_kind::any) {
                CHECK(acl_matmul_utils::init_conf_matmul_fixed_format(
                        amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
            } else {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
                // to avoid seg. fault in case threadpool is enabled and its pointer is null
                if (threadpool_utils::get_active_threadpool() == nullptr)
                    return status::unimplemented;
#endif
                CHECK(acl_matmul_utils::init_conf_matmul_non_fixed_format(
                        amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
            }

            arm_compute::ActivationLayerInfo act_info;
            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_, act_info));
            amp_.gemm_info.set_activation_info(act_info);
            amp_.use_dst_acc = post_ops.has_sum();

            // Validate ACL GEMM
            ACL_CHECK_VALID(arm_compute::NEGEMM::validate(&amp_.src_tensor_info,
                    &amp_.wei_tensor_info, nullptr, &amp_.dst_tensor_info,
                    amp_.alpha, 0.0f, amp_.gemm_info));

            return status::success;
        }

        acl_matmul_conf_t amp_;
        acl_post_ops_t acl_post_ops;
        dnnl::impl::format_kind_t weights_format_kind_;
    };

    acl_matmul_t(const pd_t *apd)
        : primitive_t(apd), acl_obj_(std::make_unique<acl_matmul_obj_t>()) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->weights_format_kind_ == format_kind::any) {
            return execute_forward<true>(ctx);
        } else {
            return execute_forward<false>(ctx);
        }
    }

private:
    template <bool IsFixedFormat>
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<acl_matmul_obj_t> acl_obj_;
}; // acl_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
