/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_RVV_BRGEMM_CONV_HPP
#define CPU_RV64_RVV_BRGEMM_CONV_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/verbose.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/rvv_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_brgemm_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("brgemm:rvv", rvv_brgemm_convolution_fwd_t);

        status_t init(engine_t *engine);

        brgemm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
        std::shared_ptr<brgemm_kernel_t> brg_kernel_;
    };

    rvv_brgemm_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~rvv_brgemm_convolution_fwd_t() override = default;

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init(engine_t *engine) override { return status::success; }

private:
    inline const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
