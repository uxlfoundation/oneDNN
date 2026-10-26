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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_LOWERING_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_LOWERING_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

/// Whether a value of type `reg` holds every value of type `mem`
///
/// @param mem The type in memory
/// @param reg The type in registers
///
/// @details An integer is held by a wider integer of the same signedness, or
///     an unsigned one by a wider signed integer; a float by a wider float
///     (f16 and bf16 by f32, f32 by f64). Any type holds itself
inline bool preserves_values(type_t mem, type_t reg) {
    if (mem == reg) return true;
    if (mem == type_t::none || reg == type_t::none) return false;
    if (sewb(reg) <= sewb(mem)) return false;
    if (is_int(mem) != is_int(reg)) return false;
    if (!is_int(mem)) return true;
    return is_signed_int(mem) == is_signed_int(reg) || !is_signed_int(mem);
}

/// Why `lowered` is not a lowering of `given`, nullptr when it is
///
/// @param given The plan a client asked for
/// @param lowered The plan an implementation runs instead
///   (`impl_t::backend_plan`)
///
/// @details A lowering changes how the operands are held, not what is
///     computed: only A's and B's register types may differ, each holding the
///     values of its memory type (`preserves_values`). The accumulators' type,
///     every memory type, the epilogue and bias types, the transposes, the
///     features, the pinned blocks and the post-ops stay as given
///
/// @note The sizes, pointers, strides and scalars are not compared: the
///     engine takes them from the given plan
inline const char *lowering_error(const plan_t &given, const plan_t &lowered) {
    if (given.dtc != lowered.dtc) return "the accumulators' type changed";
    if (given.dta.mem != lowered.dta.mem || given.dtb.mem != lowered.dtb.mem)
        return "an operand's memory type changed";
    if (given.dtepi != lowered.dtepi || given.dtbias != lowered.dtbias
            || given.bias_kind != lowered.bias_kind)
        return "the epilogue's or the bias' type changed";
    if (!preserves_values(lowered.dta.mem, lowered.dta.reg)
            || !preserves_values(lowered.dtb.mem, lowered.dtb.reg))
        return "an operand register type does not hold its memory type";
    if (given.is_transa != lowered.is_transa
            || given.is_transb != lowered.is_transb
            || given.flags != lowered.flags)
        return "the transposes or the features changed";
    if (given.blocks.mb != lowered.blocks.mb
            || given.blocks.nb != lowered.blocks.nb
            || given.blocks.kb != lowered.blocks.kb)
        return "the pinned blocks changed";
    if (given.n_postops != lowered.n_postops) return "the post-ops changed";
    for (int i = 0; i < given.n_postops; ++i)
        if (given.postops[i] != lowered.postops[i])
            return "the post-ops changed";
    return nullptr;
}

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#endif
