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

#ifndef GEMMSTONE_GUARD_KERNEL_DESC_HPP
#define GEMMSTONE_GUARD_KERNEL_DESC_HPP

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"

namespace gemmstone {

struct gemm_ir_desc_t {
    gemm_ir_desc_t(const GEMMProblem &problem, const GEMMStrategy &strategy,
            const ngen::InterfaceHandler &ngen_iface, const ir::hw_t &hw)
        : problem(problem)
        , strategy(strategy)
        , iface(ngen_iface)
        , exec_cfg(hw, strategy.GRFs, strategy.subgroupSize) {}

    const std::string &kernel_name() const { return iface.kernel_name(); }
    const ir::kernel_iface_t &kernel_iface() const { return iface; }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
    ir::kernel_iface_t iface;
    ir::exec_config_t exec_cfg;
};

} // namespace gemmstone

#endif
