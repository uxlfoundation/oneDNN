/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_PRIMITIVE_HPP
#define GPU_INTEL_GEMM_PRIMITIVE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

struct primitive_t : public intel::primitive_t {
    using intel::primitive_t::primitive_t;
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;
    status_t execute(const impl::exec_ctx_t &ctx) const override {
        exec_args_t args;
        // TODO: we have to swap a and b because
        // - gemm primitive is created with row major desc,
        // - parameters to gemm are passed as row major
        // - but gemm implementation assumes column major
        args.a = &CTX_IN_STORAGE(DNNL_ARG_A);
        args.b = &CTX_IN_STORAGE(DNNL_ARG_B);
        args.c = &CTX_OUT_STORAGE(DNNL_ARG_C);
        args.a_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_A);
        args.b_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_B);
        args.c_zero_point
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_C);
        args.a_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_A);
        args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_B);
        args.c_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_C);
        VDEBUGINFO(4, primitive, gemm_primitive,"MY !!!!! execute !!!!!! : calling execute({ctx, args}");
        return execute({ctx, args});
    }
};

inline const primitive_t *gemm(const std::shared_ptr<impl::primitive_t> &p) {
    return utils::downcast<primitive_t *>(p.get());
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
