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

#ifndef CPU_X64_IR_EMITTER_BACKEND_AVX2_HPP
#define CPU_X64_IR_EMITTER_BACKEND_AVX2_HPP

// AVX2-family backend for the emitter (see `emit()` in `emitter.cpp`).
//
// This backend contains every vector and mask instruction for one ISA family
// and covers all AVX2 extensions (avx2, avx2_vnni, avx2_vnni_2, and so on). The
// generic emitter iterates over the IR and resolves each virtual register to a
// physical register index. The backend is the only code on this path that
// constructs Xbyak `Ymm` registers, so each operation builds its own registers
// from the indices it is given.

#include <cassert>
#include <utility>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

struct avx2_backend_t {
    avx2_backend_t(jit_generator_t &gen, cpu_isa_t isa) : gen_(gen), isa(isa) {
        // `isa` is stored for future ISA-specific dispatch (e.g. avx2_vnni_2)
        // but is not read yet.
        MAYBE_UNUSED(this->isa);
    }

    jit_generator_t &gen() { return gen_; }

    // Plain vector ops.
    void vzero(int d) { // dst = 0
        gen().vxorps(Xbyak::Ymm(d), Xbyak::Ymm(d), Xbyak::Ymm(d));
    }

    void vload(int d, int base, dim_t disp) { // dst = [base + disp]
        gen().vmovups(Xbyak::Ymm(d), gen().ptr[Xbyak::Reg64(base) + (int)disp]);
    }

    void vstore(int base, dim_t disp, int s) { // [base + disp] = src
        gen().vmovups(gen().ptr[Xbyak::Reg64(base) + (int)disp], Xbyak::Ymm(s));
    }

    void vadd(int d, int s, data_type_t dt) { // dst += s0
        if (dt == data_type::f32)
            gen().vaddps(Xbyak::Ymm(d), Xbyak::Ymm(d), Xbyak::Ymm(s));
        else
            assert(!"vadd: dtype not implemented");
    }

    // dst += a * b. The multiplicand dtype `src_dt` selects the instruction.
    // f32 inputs use `vfmadd231ps`. The accumulator (dst) is always f32.
    void vdot(int d, int a, int b, data_type_t src_dt) {
        if (src_dt == data_type::f32)
            gen().vfmadd231ps(Xbyak::Ymm(d), Xbyak::Ymm(a), Xbyak::Ymm(b));
        else
            // Only f32 is supported on AVX2 today.
            assert(!"vdot: dtype not implemented");
    }

    void vhreduce(int d, int ws, data_type_t dt) {
        if (dt == data_type::f32)
            regops::horizontal_add_ps(&gen(), Xbyak::Ymm(d), Xbyak::Ymm(ws));
        else
            assert(!"vhreduce: dtype not implemented");
    }

    // Masked vector ops. On AVX2 a mask is a vector.
    //
    // Create a mask for `n_elems` active elements. The mask bytes are written
    // to the data section and loaded once. A vector mask is a constant, so the
    // builder emits `set_mask_imm` a single time and the mask is reused across
    // all masked ops.
    void set_mask_imm(int d, int n_elems, data_section_t &data) {
        const int elem_bytes = (int)sizeof(float);
        const unsigned char active_byte = 0xff;

        std::vector<unsigned char> bytes((size_t)vlen, 0);
        const int active_bytes = n_elems * elem_bytes;
        for (int i = 0; i < active_bytes; i++)
            bytes[i] = active_byte;

        // This label is the memory location of the mask in the data section.
        data.constants.emplace_back(std::move(bytes), Xbyak::Label());
        Xbyak::Label &lbl = data.constants.back().second;
        gen().vmovups(Xbyak::Ymm(d), gen().ptr[gen().rip + lbl]);
    }

    // Load `n_elems` f32 elements. The count selects the simplest form:
    //   n_elems == 1:       vmovss (no mask register needed)
    //   n_elems == simd_w:  vmovups (no mask register needed)
    //   otherwise:          vmaskmovps with the mask in `mask`
    void vload_masked(int d, int base, dim_t disp, int mask, int n_elems,
            data_type_t dt, data_section_t & /*data*/) {
        const int simd_w = vlen / (int)types::data_type_size(dt);
        assert(n_elems <= simd_w);
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (dt == data_type::f32) {
            if (n_elems == 1)
                gen().vmovss(Xbyak::Xmm(d), addr);
            else if (n_elems == simd_w)
                gen().vmovups(Xbyak::Ymm(d), addr);
            else
                gen().vmaskmovps(Xbyak::Ymm(d), Xbyak::Ymm(mask), addr);
        } else
            // vmaskmovps applies only to f32. Other precisions need a different
            // mechanism here.
            assert(!"vload_masked: dtype not implemented");
    }

    // Store `n_elems` f32 elements. Same case split as `vload_masked()`.
    void vstore_masked(int base, dim_t disp, int s, int mask, int n_elems,
            data_type_t dt, data_section_t & /*data*/) {
        const int simd_w = vlen / (int)types::data_type_size(dt);
        assert(n_elems <= simd_w);
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (dt == data_type::f32) {
            if (n_elems == 1)
                gen().vmovss(addr, Xbyak::Xmm(s));
            else if (n_elems == simd_w)
                gen().vmovups(addr, Xbyak::Ymm(s));
            else
                gen().vmaskmovps(addr, Xbyak::Ymm(mask), Xbyak::Ymm(s));
        } else
            assert(!"vstore_masked: dtype not implemented");
    }

private:
    // The backend is used during the emitter step, which is called from an
    // IR-based kernel during kernel generation. The kernel owns `gen_`.
    jit_generator_t &gen_;

    // ISA is used to dispatch ISA-specific instructions (e.g. on avx2_vnni_2).
    cpu_isa_t isa;

    // Vector register width in bytes.
    const int vlen = cpu_isa_traits_t<avx2>::vlen;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
