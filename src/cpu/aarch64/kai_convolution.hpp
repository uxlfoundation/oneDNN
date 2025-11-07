/*******************************************************************************
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

#ifndef CPU_AARCH64_KAI_CONVOLUTION_HPP
#define CPU_AARCH64_KAI_CONVOLUTION_HPP

#include <memory>

#include "common/primitive.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// kai_convolution_t wraps other KleidiAI convolution primitives, and chooses
// between them based on a performance heuristic. Each underlying convolution is
// also a valid implementation which can be placed in the cpu conv
// implementation list for testing
struct kai_convolution_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        // Keep the chosen implementation visible to verbose and benchdnn.
        DECLARE_COMMON_PD_T((impl_pd_ ? impl_pd_->name() : "convolution:kai"),
                kai_convolution_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine);

        enum class impl_kind_t { direct_1x1, indirect, im2row };
        impl_kind_t impl_kind_ = impl_kind_t::indirect;
        std::shared_ptr<primitive_desc_t> impl_pd_;

    private:
        impl_kind_t choose_impl() const;
    };

    kai_convolution_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> impl_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
