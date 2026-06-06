/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_RESHAPE_HPP
#define GPU_INTEL_MATMUL_RESHAPE_HPP

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/memory_desc.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_attr_quant.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/intel/matmul/types.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Front-impl that folds an nd-batch matmul into a 2D/3D one before any
// backend is reached. It squashes the problem, re-dispatches a same-kind
// nested matmul pd (skipping itself), and unsquashes the resolved formats
// back to the user ndims. Reshapeable problems are caught here, so the
// backends only ever see already-reshaped (<=3D) or non-foldable descs.
struct reshape_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T(pd_ ? pd_->name() : "matmul_reshape", reshape_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_desc_t> pd_;

    private:
        // Builds the squashed desc/attr; sets `reshaped` only when a fold
        // actually applies (else the iterator falls through to the backends).
        status_t maybe_reshape(matmul_desc_t &reshaped_desc,
                primitive_attr_t &reshaped_attr, bool &reshaped) const;
        // Unsquash the nested pd's resolved formats back to the user ndims.
        status_t set_default_params();
        void init_scratchpad();
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(prim_, pd()->pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> prim_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
