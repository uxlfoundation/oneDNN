/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_DECONVOLUTION_HPP
#define CPU_AARCH64_KAI_DECONVOLUTION_HPP

#include <memory>
#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kai_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , use_outer_epilogue_(other.use_outer_epilogue_)
            , name_(other.name_) {}

        DECLARE_COMMON_PD_T(name_.c_str(), kai_deconvolution_fwd_t);

        status_t init(const engine_t *engine);

        // Descriptor of the direct 1x1 or indirect KAI convolution we reuse.
        std::shared_ptr<primitive_desc_t> conv_pd_;
        // True when this wrapper applies bias and post-ops after KAI finishes.
        bool use_outer_epilogue_ = false;

    private:
        // Finds an exact KAI convolution for the converted descriptor.
        status_t try_kai_convolution(const engine_t *engine,
                const primitive_attr_t &conv_attr, const memory_desc_t *bias_md,
                data_type_t dst_dt);
        void init_name();

        // Composite name shown by oneDNN verbose output.
        std::string name_;
    };

    kai_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    // Executable nested KAI convolution created from conv_pd_.
    std::shared_ptr<primitive_t> conv_p_;
    // Applies the public post-ops only on the outer-epilogue route.
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
