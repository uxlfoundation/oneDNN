/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_INTEL_POOL_JIT_HPP
#define GPU_INTEL_POOL_JIT_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/intel/pool/config.hpp"
#include "gpu/intel/pool/jit/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace pool {

class gen_fwd_t : public primitive_t {
public:
    struct pd_t : public fwd_pd_t {
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_fwd_t);

        status_t init(impl::engine_t *);

        std::shared_ptr<conf_t> conf;
        std::shared_ptr<jit::dsl::kernel::options_t> options;
        std::shared_ptr<jit::layout_t> src;
        std::shared_ptr<jit::layout_t> dst;
    };

    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    jit::config_t cfg_;
    jit::kernel_info_t kernel_info_;
    compute::kernel_t kernel_;
};

} // namespace pool
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
