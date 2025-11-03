/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#ifndef GPU_INTEL_JIT_CODEGEN_CODEGEN_EXTENSIONS_HPP
#define GPU_INTEL_JIT_CODEGEN_CODEGEN_EXTENSIONS_HPP

#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/codegen/operand.hpp"
#include "gpu/intel/jit/ir/include/kernel.hpp"
#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct codegen_extension_interface_t {
public:
    struct host_t {
        void *ptr;
        const std::type_info &info;
    };
    virtual host_t root_code_generator() const = 0;
    virtual const kernel::options_t &options() const = 0;
    virtual reg_allocator_t &allocator() = 0;
    virtual std::vector<ngen_operand_t> evaluate(
            const std::vector<expr_t> &exprs, ngen_register_scope_t &scope)
            = 0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
