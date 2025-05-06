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

#ifndef GPU_INTEL_JIT_GEMM_IR_KERNEL_DESC_HPP
#define GPU_INTEL_JIT_GEMM_IR_KERNEL_DESC_HPP

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gpu/intel/jit/gemm/ir/ir_interop.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"

namespace gemmstone {

struct ir_desc_t {
    ir_desc_t(const GEMMProblem &problem, const GEMMStrategy &strategy)
        : problem(problem), strategy(strategy) {}

    std::string kernel_name() const { return "gemm_ir_kernel"; }

    void init_kernel_iface(ir::kernel_iface_t &kernel_iface) const {
        kernel_iface.register_arg("A_ptr", ir::type_t::byte_ptr());
        kernel_iface.register_arg("B_ptr", ir::type_t::byte_ptr());
        kernel_iface.register_arg("C_ptr", ir::type_t::byte_ptr());
        kernel_iface.register_arg("offset_A", ir::type_t::s(64));
        kernel_iface.register_arg("offset_B", ir::type_t::s(64));
        kernel_iface.register_arg("offset_C", ir::type_t::s(64));
        kernel_iface.register_arg("lda", ir::type_t::u(32));
        kernel_iface.register_arg("ldb", ir::type_t::u(32));
        kernel_iface.register_arg("ldc", ir::type_t::u(32));
        kernel_iface.register_arg("m", ir::type_t::u(32));
        kernel_iface.register_arg("n", ir::type_t::u(32));
        kernel_iface.register_arg("k", ir::type_t::u(32));
        kernel_iface.register_arg("alpha", into_ir(problem.Ts));
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
};

} // namespace gemmstone

#endif
