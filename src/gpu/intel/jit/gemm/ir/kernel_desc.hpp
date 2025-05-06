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
#include "gpu/intel/jit/codegen/codegen.hpp"
#include "gpu/intel/jit/gemm/ir/ir_interop.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"

namespace gemmstone {

struct gemm_ir_desc_t {
    gemm_ir_desc_t(const GEMMProblem &problem, const GEMMStrategy &strategy,
            const ngen::InterfaceHandler &ngen_iface, const ir::hw_t &hw)
        : problem(problem), strategy(strategy), iface(ngen_iface), hw(hw) {}

    const char *kernel_name() const { return "gemm_kernel"; }
    const ir::kernel_iface_t &kernel_iface() const { return iface; }

    ir::compile_ctx_t compile_ctx() const {
        auto &wg = strategy.wg;
        return {kernel_name(), kernel_iface(),
                {hw, strategy.GRFs, strategy.subgroupSize},
                strategy.fixedWG({}) ? wg[0] * wg[1] * wg[2] : 0};
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
    ir::kernel_iface_t iface;
    ir::hw_t hw;
};

} // namespace gemmstone

#endif
