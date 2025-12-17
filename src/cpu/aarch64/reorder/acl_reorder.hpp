/*******************************************************************************
* Copyright 2023-2025 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_REORDER_ACL_REORDER_HPP
#define CPU_AARCH64_REORDER_ACL_REORDER_HPP

#include "common/utils.hpp"
#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_reorder_obj_t {
    arm_compute::NEReorderLayer reorder;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::WeightFormat src_wf;
    arm_compute::WeightFormat dst_wf;
};

struct acl_reorder_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::WeightFormat src_wf = arm_compute::WeightFormat::OHWI;
    arm_compute::WeightFormat dst_wf = arm_compute::WeightFormat::OHWI;
    bool transpose;
};

struct acl_reorder_resource_t : public resource_t {
    acl_reorder_resource_t()
        : acl_obj_(utils::make_unique<acl_reorder_obj_t>()) {}

    status_t configure(const acl_reorder_conf_t &app);

    acl_reorder_obj_t &get_acl_obj() const { return *acl_obj_; }
    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_reorder_resource_t);

private:
    std::unique_ptr<acl_reorder_obj_t> acl_obj_;
}; // acl_reorder_resource_t

struct acl_reorder_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {

        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_reorder_fwd_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
        acl_reorder_conf_t app_;

    }; // pd_t

    acl_reorder_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    inline const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }

}; // acl_reorder_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_REORDER_ACL_REORDER_HPP
