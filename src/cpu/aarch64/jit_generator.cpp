/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2020-2025 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
template <typename T,
        typename std::enable_if<std::is_unsigned<T>::value,
                std::nullptr_t>::type
        = nullptr>
bool is_imm12(T imm) {
    uint64_t bit_ptn = static_cast<uint64_t>(imm);

    const uint64_t IMM12_MASK = ~uint64_t(0xfff);
    // <= 4095
    return ((bit_ptn & IMM12_MASK) == 0);
}

template <typename T,
        typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type
        = nullptr>
bool is_imm12(T imm) {
    /* Sign bit must be extended. */
    int64_t bit_ptn = static_cast<int64_t>(imm);
    const uint64_t IMM12_MASK = ~uint64_t(0xfff);

    // <= 4095
    return (imm >= 0) && ((bit_ptn & IMM12_MASK) == 0);
}

} // namespace

void jit_generator_t::preamble() {
    using namespace Xbyak_aarch64::util;
    uint64_t sveLen = get_sve_length();

    stp(x29, x30, pre_ptr(sp, -16));
    /* x29 is a frame pointer. */
    mov(x29, sp);
    sub(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);

    /* x9 can be used as a temporal register. */
    mov(x9, sp);

    if (vreg_to_preserve) {
        st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
        stp(abi_save_gpr_regs[i], abi_save_gpr_regs[i + 1],
                post_ptr(x9, xreg_len * 2));
    }

    if (sveLen) { /* SVE is available. */
        ptrue(P_ALL_ONE.b);
        pfalse(P_ALL_ZERO.b);
    }
    if (sveLen >= SVE_256) {
        ptrue(P_NOT_128.b, Xbyak_aarch64::VL16);
        not_(P_NOT_128.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_128.b);
    }
    if (sveLen >= SVE_512) {
        ptrue(P_NOT_256.b, Xbyak_aarch64::VL32);
        not_(P_NOT_256.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_256.b);
    }
}

void jit_generator_t::postamble() {
    using namespace Xbyak_aarch64::util;

    mov(x9, sp);

    if (vreg_to_preserve) {
        ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
    }

    for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
        ldp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
                Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
                post_ptr(x9, xreg_len * 2));
    }

    add(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);
    ldp(x29, x30, post_ptr(sp, 16));
    ret();
}

// Disallow char-based labels completely
void L(const char *label) = delete;
void jit_generator_t::L(Xbyak_aarch64::Label &label) {
    Xbyak_aarch64::CodeGenerator::L(label);
}

void jit_generator_t::L_aligned(Xbyak_aarch64::Label &label, int alignment) {
    align(alignment);
    L(label);
}

void jit_generator_t::ldr_imm(
        Xbyak_aarch64::XReg reg, Xbyak_aarch64::XReg &addr, uint32_t offset) {
    if (is_imm12(offset)) {
        ldr(reg, ptr(addr, offset));
    } else {
        add_imm(X_DEFAULT_ADDR, addr, offset, X_DEFAULT_ADDR);
        ldr(reg, ptr(X_DEFAULT_ADDR));
    }
}

// Return XReg which points to base + stride_bytes * n in as few instructions as possible
// We return a XReg so that we can use the dst or base reg
Xbyak_aarch64::XReg jit_generator_t::strided_addr(
        const Xbyak_aarch64::XReg &dst, const Xbyak_aarch64::XReg &base,
        const Xbyak_aarch64::XReg &reg_stride_bytes, const int32_t stride_bytes,
        const int32_t n, const Xbyak_aarch64::XReg &tmp) {
    int32_t offset_bytes = stride_bytes * n;
    if (n == 0) {
        // (0 instructions) Use the base
        return base;
    } else if (is_imm12(offset_bytes)) {
        // (1 instruction) The byte offset can be expressed as an int12 immediate in the add instruction
        add(dst, base, static_cast<uint32_t>(offset_bytes & 0xfff));
        return dst;
    } else {
        const auto LSL = Xbyak_aarch64::ShMod::LSL;
        switch (n) {
            case 1:
                // 1 instruction
                add(dst, base, reg_stride_bytes);
                return dst;
            case 2: add(dst, base, reg_stride_bytes, LSL, 1); return dst;
            case 4: add(dst, base, reg_stride_bytes, LSL, 2); return dst;
            case 8: add(dst, base, reg_stride_bytes, LSL, 3); return dst;
            case 16: add(dst, base, reg_stride_bytes, LSL, 4); return dst;
            default: add_imm(dst, base, offset_bytes, tmp); return dst;
        }
    }
}

void jit_generator_t::uni_add(const Xbyak_aarch64::VReg4S &x1,
        const Xbyak_aarch64::VReg4S &x2, const Xbyak_aarch64::VReg4S &op) {
    add(x1, x2, op);
}

void jit_generator_t::uni_add(const Xbyak_aarch64::ZReg &x1,
        const Xbyak_aarch64::ZReg &x2, const Xbyak_aarch64::ZReg &op) {
    add(Xbyak_aarch64::ZRegS(x1.getIdx()), Xbyak_aarch64::ZRegS(x2.getIdx()),
            Xbyak_aarch64::ZRegS(op.getIdx()));
}

void jit_generator_t::uni_clear(const Xbyak_aarch64::VReg &dst) {
    // movi 0 on any part of the SIMD&FP reg will also zero out the rest of the
    // SIMD vector (see the Arm ARM C7.1.1).
    // We use this because movi d is zero latency on Neoverse V2
    movi(Xbyak_aarch64::DReg(dst.getIdx()), 0);
}
void jit_generator_t::uni_clear(const Xbyak_aarch64::ZReg &dst) {
    // movi 0 on any part of the SIMD&FP reg will also zero out the rest of the SVE vector
    // (see the Arm ARM C7.1.1 and B1.4.3 R_WKYLB)
    // We use this because movi d is zero latency on Neoverse V2
    movi(Xbyak_aarch64::DReg(dst.getIdx()), 0);
}

// Float addition across all vector elements
void jit_generator_t::uni_fadd_reduce(
        const Xbyak_aarch64::SReg &dst, const Xbyak_aarch64::ZRegS &src) {
    faddv(dst, P_ALL_ONE / Xbyak_aarch64::T_z, src);
}

void jit_generator_t::uni_fadd_reduce(
        const Xbyak_aarch64::SReg &dst, const Xbyak_aarch64::VReg4S &src) {
    const auto &vreg_dst = Xbyak_aarch64::VReg(dst.getIdx());

    faddp(vreg_dst.s, src, src);
    faddp(vreg_dst.s2, vreg_dst.s2, vreg_dst.s2);
}

void jit_generator_t::uni_fcvtzs(
        const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s) {
    fcvtzs(d, s);
}

void jit_generator_t::uni_fcvtzs(
        const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s) {
    fcvtzs(d, P_ALL_ONE / Xbyak_aarch64::T_z, s);
}

void jit_generator_t::uni_fdiv(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2,
        const Xbyak_aarch64::VReg4S &tmp, const Xbyak_aarch64::PReg &pred) {
    UNUSED(tmp);
    UNUSED(pred);
    fdiv(dst, src, src2);
}

void jit_generator_t::uni_fmad(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2) {
    fmul(dst, dst, src);
    fadd(dst, dst, src2);
}

void jit_generator_t::float_point_fused_multiply_add(
        const Xbyak_aarch64::ZReg &dst_vmm, const Xbyak_aarch64::ZReg &src1_vmm,
        const Xbyak_aarch64::ZReg &src2_vmm) {
    fmla(dst_vmm.s, P_ALL_ONE / Xbyak_aarch64::T_m, src1_vmm.s, src2_vmm.s);
}

void jit_generator_t::float_point_fused_multiply_add(
        const Xbyak_aarch64::VReg &dst_vmm, const Xbyak_aarch64::VReg &src1_vmm,
        const Xbyak_aarch64::VReg &src2_vmm) {
    fmla(dst_vmm.s, src1_vmm.s, src2_vmm.s);
}

void jit_generator_t::uni_fmax(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2) {
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx)
        mov(Xbyak_aarch64::VReg16B(dstIdx), Xbyak_aarch64::VReg16B(srcIdx));
    fmax(dst, src, src2);
}

void jit_generator_t::uni_fmaxnm(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2) {
    fmaxnm(dst, src, src2);
}

void jit_generator_t::uni_fmin(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2) {
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx)
        mov(Xbyak_aarch64::VReg16B(dstIdx), Xbyak_aarch64::VReg16B(srcIdx));
    fmin(dst, src, src2);
}

void jit_generator_t::contiguous_load_unsigned_words(
        const Xbyak_aarch64::ZReg &src_vmm, const Xbyak_aarch64::XReg &addr) {
    ld1w(src_vmm.s, P_ALL_ONE / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(addr));
}

void jit_generator_t::contiguous_load_unsigned_words(
        const Xbyak_aarch64::VReg &src_vmm, const Xbyak_aarch64::XReg &addr) {
    ld1(src_vmm.s, Xbyak_aarch64::ptr(addr));
}

void jit_generator_t::uni_frinti(
        const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s) {
    frinti(d, s);
}

void jit_generator_t::uni_frinti(
        const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s) {
    frinti(d, P_ALL_ONE / Xbyak_aarch64::T_m, s);
}

void jit_generator_t::uni_fsqrt(
        const Xbyak_aarch64::VReg4S &dst, const Xbyak_aarch64::VReg4S &src) {
    fsqrt(dst, src);
}

void jit_generator_t::uni_fsub(const Xbyak_aarch64::VReg4S &v1,
        const Xbyak_aarch64::VReg4S &v2, const Xbyak_aarch64::VReg4S &v3) {
    fsub(v1, v2, v3);
}

void jit_generator_t::uni_fsub(const Xbyak_aarch64::ZRegS &z1,
        const Xbyak_aarch64::ZRegS &z2, const Xbyak_aarch64::ZRegS &z3) {
    fsub(z1, z2, z3);
}

void jit_generator_t::uni_eor(const Xbyak_aarch64::VReg &v1,
        const Xbyak_aarch64::VReg &v2, const Xbyak_aarch64::VReg &v3) {
    eor(Xbyak_aarch64::VReg16B(v1.getIdx()),
            Xbyak_aarch64::VReg16B(v2.getIdx()),
            Xbyak_aarch64::VReg16B(v3.getIdx()));
}

void jit_generator_t::uni_eor(const Xbyak_aarch64::ZReg &z1,
        const Xbyak_aarch64::ZReg &z2, const Xbyak_aarch64::ZReg &z3) {
    eor(Xbyak_aarch64::ZRegD(z1.getIdx()), Xbyak_aarch64::ZRegD(z2.getIdx()),
            Xbyak_aarch64::ZRegD(z3.getIdx()));
}

void jit_generator_t::uni_ld1rw(const Xbyak_aarch64::VReg4S &dst,
        const Xbyak_aarch64::XReg &base, const int64_t off) {
    if (off == 0) {
        ld1r(dst, ptr(base));
    } else {
        add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
        ld1r(dst, ptr(X_DEFAULT_ADDR));
    }
}

void jit_generator_t::uni_ld1rw(const Xbyak_aarch64::ZRegS &dst,
        const Xbyak_aarch64::XReg &base, const int64_t off) {
    if (-32 <= off && off < 32) {
        ld1rw(dst, P_ALL_ONE / Xbyak_aarch64::T_z, ptr(base, (int)off));
    } else {
        add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
        ld1rw(dst, P_ALL_ONE / Xbyak_aarch64::T_z, ptr(X_DEFAULT_ADDR));
    }
}

void jit_generator_t::uni_ldr(
        const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::XReg &addr) {
    ldr(Xbyak_aarch64::QReg(dst.getIdx()), ptr(addr));
}

void jit_generator_t::uni_ldr(
        const Xbyak_aarch64::ZReg &dst, const Xbyak_aarch64::XReg &addr) {
    ldr(dst, ptr(addr));
}

void jit_generator_t::uni_mov(
        const Xbyak_aarch64::VReg &d, const Xbyak_aarch64::VReg &s) {
    if (d.getIdx() != s.getIdx()) { mov(d.b16, s.b16); }
}

void jit_generator_t::uni_mov(
        const Xbyak_aarch64::ZReg &d, const Xbyak_aarch64::ZReg &s) {
    if (d.getIdx() != s.getIdx()) { mov(d.d, s.d); }
}

void jit_generator_t::uni_dup(const Xbyak_aarch64::VReg4S &dst, int32_t value) {
    assert(value >= -128 && value <= 127);

    if (value >= 0)
        movi(dst, static_cast<uint32_t>(value));
    else
        mvni(dst, static_cast<uint32_t>(~value));
}

void jit_generator_t::uni_dup(const Xbyak_aarch64::ZRegS &dst, int32_t value) {
    assert(value >= -128 && value <= 127);
    dup(dst, value);
}

void jit_generator_t::uni_orr(const Xbyak_aarch64::VReg &d,
        const Xbyak_aarch64::VReg &s0, const Xbyak_aarch64::VReg &s1) {
    orr(d.b16, s0.b16, s1.b16);
}

void jit_generator_t::uni_orr(const Xbyak_aarch64::ZReg &d,
        const Xbyak_aarch64::ZReg &s0, const Xbyak_aarch64::ZReg &s1) {
    orr(d.d, s0.d, s1.d);
}

void jit_generator_t::uni_scvtf(
        const Xbyak_aarch64::VReg4S &t0, const Xbyak_aarch64::VReg4S &t1) {
    scvtf(t0, t1);
}

void jit_generator_t::uni_scvtf(
        const Xbyak_aarch64::ZRegS &t0, const Xbyak_aarch64::ZRegS &t1) {
    scvtf(t0, P_ALL_ONE / Xbyak_aarch64::T_m, t1);
}

void jit_generator_t::uni_str(
        const Xbyak_aarch64::VReg &src, const Xbyak_aarch64::XReg &addr) {
    str(Xbyak_aarch64::QReg(src.getIdx()), ptr(addr));
}

void jit_generator_t::uni_str(
        const Xbyak_aarch64::ZReg &src, const Xbyak_aarch64::XReg &addr) {
    str(src, ptr(addr));
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
