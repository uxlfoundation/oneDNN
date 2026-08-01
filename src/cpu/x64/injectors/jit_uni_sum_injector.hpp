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

#ifndef CPU_X64_INJECTORS_JIT_UNI_SUM_INJECTOR_HPP
#define CPU_X64_INJECTORS_JIT_UNI_SUM_INJECTOR_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Injects the sum post-op: acc += scale * (prev - zp), where `prev` is the
// accumulator's previous in-memory value read as f32.
//
// The typed, tail-aware read is delegated to the binary injector
// (`load_acc_as_f32`), so this injector only owns the sum arithmetic, the
// scale/zero-point constants and the single scratch vector register.
template <cpu_isa_t isa, typename Vmm = typename cpu_isa_traits_t<isa>::Vmm>
class jit_uni_sum_injector_t {
public:
    // `sum` supplies the scale, zero-point and read data type. When
    // `sum.dt == undef` the read type falls back to `dst_dt`.
    // `binary_injector` provides the load path and must outlive this injector.
    // `scratch_vmm_idx` is the binary injector's helper vmm, reused as the only
    // scratch since the two never use that vmm at once in one chain. It is
    // preserved when `preserve_scratch_vmm` is set.
    jit_uni_sum_injector_t(jit_generator_t *host,
            const post_ops_t::entry_t::sum_t &sum, data_type_t dst_dt,
            binary_injector::jit_uni_binary_injector_t<Vmm> *binary_injector,
            size_t scratch_vmm_idx, bool preserve_scratch_vmm);

    // Computes `acc += scale * (prev - zp)` for each accumulator in `vmm_idxs`.
    // The per-accumulator destination address (register + element offset) and
    // tail flag come from `rhs_arg_params`, exactly as for binary post-ops.
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);

    // Creates a table with the scale and zero-point constants used by
    // `compute_vector_range()`. Default values (scale == 1, zero-point == 0)
    // are omitted.
    void prepare_table(bool gen_table);

private:
    jit_generator_t *host_;
    binary_injector::jit_uni_binary_injector_t<Vmm> *binary_injector_;

    // Read type for the previous value (`sum.dt` or `dst_dt` when undefined).
    const data_type_t sum_dt_;
    const float scale_val_;
    // The zero-point is an integer attribute but applied in f32, so it is
    // stored already converted to f32.
    const float zp_val_;

    const bool has_scale_;
    const bool has_zp_;

    const size_t scratch_vmm_idx_;
    const bool preserve_scratch_vmm_;

    Xbyak::Label scale_lbl_;
    Xbyak::Label zp_lbl_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
