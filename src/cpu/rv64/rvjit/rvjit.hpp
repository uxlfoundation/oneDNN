/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVJIT_RVJIT_HPP
#define CPU_RV64_RVJIT_RVJIT_HPP

#include <memory>

#include "cpu/rv64/rvjit/rvjit_arithmetic.hpp"
#include "cpu/rv64/rvjit/rvjit_const_folding.hpp"
#include "cpu/rv64/rvjit/rvjit_control_flow.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"
#include "cpu/rv64/rvjit/rvjit_memory_move.hpp"
#include "cpu/rv64/rvjit/rvjit_register_pool.hpp"
#include "cpu/rv64/rvjit/rvjit_rvv.hpp"
#include "cpu/rv64/rvjit/rvjit_types.hpp"
#include "cpu/rv64/rvjit/rvjit_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Component-based system for Risc-V Just-In-Time code generation
class rvjit_t {
public:
    explicit rvjit_t(codegen_t &cg) : emitter_(cg) {}

    const emitter_t &emitter() const { return emitter_; }

    const_folding_t &const_folding() {
        if (!const_folding_)
            const_folding_ = make_unique<const_folding_t>(emitter_);
        return *const_folding_;
    }

    control_flow_t &control_flow() {
        if (!control_flow_)
            control_flow_
                    = make_unique<control_flow_t>(emitter_, const_folding());
        return *control_flow_;
    }

    register_pool_t &register_pool() {
        if (!register_pool_)
            register_pool_ = make_unique<register_pool_t>(emitter_);
        return *register_pool_;
    }

    memory_move_t &memory_move() {
        if (!mem_move_) mem_move_ = make_unique<memory_move_t>(emitter_);
        return *mem_move_;
    }

    rvv_arithmetic_t &macc() {
        if (!macc_) macc_ = make_unique<rvv_arithmetic_t>(emitter_);
        return *macc_;
    }

    rvv_t &model() {
        if (!model_) model_ = make_unique<rvv_t>();
        return *model_;
    }

    /// Injects a hardware-derived model in place of the lazy zero-default
    ///
    /// @pre Must be called, if at all, before the first `matmul_engine()`
    ///     access: the engine copies `model()` at construction, so a
    ///     `set_model()` call after `matmul_engine()` has already been built
    ///     does not retroactively update it
    void set_model(const rvv_t &model) { model_ = make_unique<rvv_t>(model); }

#if XBYAK_RISCV_V
    rvv_matmul_engine_t &matmul_engine() {
        if (!matmul_engine_)
            matmul_engine_ = make_unique<rvv_matmul_engine_t>(emitter_,
                    register_pool(), memory_move(), macc(), const_folding(),
                    control_flow(), model());
        return *matmul_engine_;
    }
#endif

private:
    emitter_t emitter_;
    std::unique_ptr<const_folding_t> const_folding_;
    std::unique_ptr<control_flow_t> control_flow_;
    std::unique_ptr<register_pool_t> register_pool_;
    std::unique_ptr<memory_move_t> mem_move_;
    std::unique_ptr<rvv_arithmetic_t> macc_;
    std::unique_ptr<rvv_t> model_;
#if XBYAK_RISCV_V
    std::unique_ptr<rvv_matmul_engine_t> matmul_engine_;
#endif
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
