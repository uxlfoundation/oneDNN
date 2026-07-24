/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_PRELU_JIT_UNI_PRELU_FORWARD_HPP
#define CPU_AARCH64_PRELU_JIT_UNI_PRELU_FORWARD_HPP

#include <memory>

#include "common/broadcast_strategy.hpp"
#include "common/primitive.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_prelu_pd.hpp"

namespace dnnl {
namespace impl {

struct memory_desc_wrapper;

namespace cpu {
namespace aarch64 {

namespace prelu {

broadcasting_strategy_t get_bcast_type(
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d);
bool bcast_supported(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, size_t simd_w);

} // namespace prelu

class jit_prelu_forward_kernel_t;

template <cpu_isa_t isa>
struct jit_uni_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_prelu_fwd_t);

        status_t init(const engine_t *engine);

        // Broadcast kind selected during primitive descriptor initialization.
        // execute() and the JIT kernel both use this to choose pointer movement.
        broadcasting_strategy_t bcast_ = broadcasting_strategy_t::unsupported;
        bool per_oc_blocked_ = false;
        data_type_t data_type_ = data_type::undef;
    };

    explicit jit_uni_prelu_fwd_t(const pd_t *apd);
    ~jit_uni_prelu_fwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::unique_ptr<jit_prelu_forward_kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
