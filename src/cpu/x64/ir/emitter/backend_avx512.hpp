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

#ifndef CPU_X64_IR_EMITTER_BACKEND_AVX512_HPP
#define CPU_X64_IR_EMITTER_BACKEND_AVX512_HPP

// AVX-512 backend for the emitter (see `emit()` in `emitter.cpp`).
//
// This backend contains every vector and mask instruction for one ISA family
// and covers all AVX-512 extensions (avx512_core, avx512_core_bf16,
// avx512_core_fp16, and so on). The generic emitter iterates over the IR and
// resolves each virtual register to a physical register index. The backend is
// the only code on this path that constructs Xbyak `Zmm` and `Opmask`
// registers, so each operation builds its own registers from the indices it is
// given.
//
// A mask is a k-register, and a masked access is an EVEX write mask on the
// access instruction itself rather than a separate instruction. A mask
// therefore allocates from a register file of its own (see
// `make_reg_config()`).

#include <cassert>

#include "common/c_types_map.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

struct avx512_backend_t {
    avx512_backend_t(jit_generator_t &gen, cpu_isa_t isa)
        : gen_(gen), isa(isa) {
        // `isa` is stored for the dtype dispatch that tells the AVX-512
        // extensions apart (e.g. avx512_core_bf16) but is not read yet.
        MAYBE_UNUSED(this->isa);
    }

    jit_generator_t &gen() { return gen_; }

    // Plain vector ops.
    void vzero(int d) { // dst = 0
        gen().vpxord(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(d));
    }

    void vload(int d, int base, dim_t disp) { // dst = [base + disp]
        gen().vmovups(Xbyak::Zmm(d), gen().ptr[Xbyak::Reg64(base) + (int)disp]);
    }

    void vstore(int base, dim_t disp, int s) { // [base + disp] = src
        gen().vmovups(gen().ptr[Xbyak::Reg64(base) + (int)disp], Xbyak::Zmm(s));
    }

    // Load one element and zero the rest of `dst`.
    void vload_scalar(int d, int base, dim_t disp, data_type_t dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (dt == data_type::f32)
            gen().vmovss(Xbyak::Xmm(d), addr);
        else { JIT_ASSERT(!"vload_scalar: dtype not implemented"); }
    }

    // Store element 0 of `src`.
    void vstore_scalar(int base, dim_t disp, int s, data_type_t dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (dt == data_type::f32)
            gen().vmovss(addr, Xbyak::Xmm(s));
        else { JIT_ASSERT(!"vstore_scalar: dtype not implemented"); }
    }

    void vadd(int d, int s, data_type_t dt) { // dst += s0
        if (dt == data_type::f32)
            gen().vaddps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vadd: dtype not implemented"); }
    }

    void vmul(int d, int s, data_type_t dt) { // dst *= s0
        if (dt == data_type::f32)
            gen().vmulps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vmul: dtype not implemented"); }
    }

    // dst += a * b. The multiplicand dtype `src_dt` selects the instruction.
    // f32 inputs use `vfmadd231ps`. The accumulator (dst) is always f32.
    void vdot(int d, int a, int b, data_type_t src_dt) {
        if (src_dt == data_type::f32)
            gen().vfmadd231ps(Xbyak::Zmm(d), Xbyak::Zmm(a), Xbyak::Zmm(b));
        else {
            // Only f32 is supported on AVX-512 today.
            JIT_ASSERT(!"vdot: dtype not implemented");
        }
    }

    void vhreduce(int d, int ws, data_type_t dt) {
        if (dt == data_type::f32)
            regops::horizontal_add_ps(&gen(), Xbyak::Zmm(d), Xbyak::Zmm(ws));
        else { JIT_ASSERT(!"vhreduce: dtype not implemented"); }
    }

    // Masked vector ops.
    //
    // Create a mask with `n_elems` active elements. The mask is built inside
    // the k-register file: `kxnorq` sets all 64 bits and `kshiftrq` keeps the
    // low `n_elems` of them. The alternative, `kmov` from an immediate in a
    // gpr, would put mask setup on the scratch registers, and those are meant
    // to go away. `data` is unused for the same reason.
    //
    // The number of active bits is `n_elems` for every element size. A widening
    // load preserves the element count, so a narrower element type changes what
    // one bit masks, not how many bits are set (see `vload_masked`).
    void set_mask_imm(int d, int n_elems, data_section_t & /*data*/) {
        assert(n_elems > 0 && n_elems < 64);
        const Xbyak::Opmask k(d);
        gen().kxnorq(k, k, k);
        gen().kshiftrq(k, k, (uint8_t)(64 - n_elems));
    }

    // Load the elements selected by `mask` and zero the rest of `dst` (`T_z`).
    // An inactive element has to read as zero, or a tail iteration would
    // accumulate whatever the register happened to hold into the dot product.
    void vload_masked(int d, int base, dim_t disp, int mask, data_type_t dt,
            data_section_t & /*data*/) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (dt == data_type::f32) {
            gen().vmovups(
                    Xbyak::Zmm(d) | Xbyak::Opmask(mask) | gen().T_z, addr);
        } else {
            JIT_ASSERT(!"vload_masked: dtype not implemented");
        }
    }

    // Store the elements of `src` selected by `mask`. There is no `T_z` here,
    // since masking a store is already a merge into memory.
    void vstore_masked(int base, dim_t disp, int s, int mask, data_type_t dt,
            data_section_t & /*data*/) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (dt == data_type::f32) {
            gen().vmovups(addr | Xbyak::Opmask(mask), Xbyak::Zmm(s));
        } else {
            JIT_ASSERT(!"vstore_masked: dtype not implemented");
        }
    }

private:
    // The backend is used during the emitter step, which is called from an
    // IR-based kernel during kernel generation. The kernel owns `gen_`.
    jit_generator_t &gen_;

    // ISA is used to dispatch ISA-specific instructions (e.g. on
    // avx512_core_bf16).
    cpu_isa_t isa;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
