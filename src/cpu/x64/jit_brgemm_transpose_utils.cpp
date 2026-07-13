/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_brgemm_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_trans_m_k_f32_t : public jit_brgemm_trans_src_t,
                                    public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_f32_t)

    jit_brgemm_trans_m_k_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf)
        , jit_generator_t(jit_name())
        , transpose_size(isa_max_vlen(conf_->isa) / typesize) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float) };
    const int transpose_size;
    dim_t src_stride = 0, tr_src_stride = 0;

    Xbyak::Label mask_label_;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_K = r10;
    reg64_t reg_loop_M = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg64_t reg_tmp = r14;
    reg32_t regw_tmp = r14d;
    reg64_t reg_row_loop = r15;

    Ymm ymm_tail_mask = ymm15;
    Xmm xmm_lower_tail_mask = xmm15;
    Xmm xmm_upper_tail_mask = xmm14;
    Xmm xmm_zero = xmm13;
    void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    }
    void transpose_16x16(int nrows, int ncolumns);
    void transpose_16x16_avx2(int nrows, int ncolumns);
    void transpose_ker(int nrows, int ncolumns);
    void transpose(int nrows, int ncolumns);
    void init_masks(int tail_length);
    void generate() override;
};

void jit_brgemm_trans_m_k_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    assert(transpose_size == 16 && "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [&](int i) {
        auto src_load = src_zmm(i);
        if (i >= nrows) {
            vpxord(src_load, src_load, src_load);
            return;
        }

        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load,
                EVEX_compress_addr_safe(reg_src, i * src_stride, reg_tmp));
    };

    auto store = [&](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        const bool partial_store = nrows < transpose_size;
        const auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        const auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i * 2;
            const int src_idx1 = src_idx0 + 1;

            const int next_src_idx0 = src_idx0 + 2;
            const int next_src_idx1 = src_idx1 + 2;
            const bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx1);
            const auto src0 = src_zmm(src_idx0);
            const auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            const int select_half = (i < 2) ? 0 : 2;
            const int src_idx0 = base_idx + i + select_half + 0;
            const int src_idx2 = src_idx0 + 2;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx2);
            const auto src0 = src_zmm(src_idx0);
            const auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i;
            const int src_idx4 = src_idx0 + 4;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto src0 = src_zmm(src_idx0);
            const auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        // swap 8
        const auto max_iters_phase_1 = std::min(ncolumns, 8);
        for (int i = 0; i < max_iters_phase_1; i++) {
            const auto tmp = tmp_zmm(i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        const auto max_iters_phase_2 = std::min(ncolumns - 8, 8);
        for (int i = 0; i < max_iters_phase_2; i++) {
            const auto tmp = tmp_zmm(8 + i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_trans_m_k_f32_t::transpose_16x16_avx2(int nrows, int ncolumns) {
    assert(transpose_size == 8 && "Unsupported transpose size");
    auto xmm_tmp = xmm13;

    // Note: For stores we assume, the memory is padded, hence avoiding use of
    // mask stores.
    assert(conf_->os_block % transpose_size == 0);
    auto load_src = [&](Xmm vmm, int r, int c) {
        const int simd_w = vmm.getBit() / (sizeof(float) * 8);
        const auto addr = ptr[reg_src + r * src_stride + c * sizeof(float)];
        if (r >= nrows) {
            uni_vxorps(vmm, vmm, vmm);
        } else if (c + simd_w <= ncolumns) {
            vmovups(vmm, addr);
        } else if (simd_w == 8) {
            vmaskmovps(vmm, ymm_tail_mask, addr);
        } else if (c == 0) {
            vmaskmovps(vmm, xmm_lower_tail_mask, addr);
        } else {
            vmaskmovps(vmm, xmm_upper_tail_mask, addr);
        }
    };

    auto vinsert = [&](Ymm ymm, int r, int c) {
        const int xmm_simd_w = 4;
        const auto addr = ptr[reg_src + r * src_stride + c * sizeof(float)];
        if (r >= nrows) {
            vinsertf128(ymm, ymm, xmm_zero, 1);
        } else if (c + xmm_simd_w <= ncolumns) {
            vinsertf128(ymm, ymm, addr, 1);
        } else {
            vmaskmovps(xmm_tmp,
                    c == 0 ? xmm_lower_tail_mask : xmm_upper_tail_mask, addr);
            vinsertf128(ymm, ymm, xmm_tmp, 1);
        }
    };

    mov(reg_tr_src_tmp, reg_tr_src);
    load_src(xmm0, 0, 0);
    vinsert(ymm0, 4, 0);
    load_src(xmm1, 1, 0);
    vinsert(ymm1, 5, 0);

    vunpcklpd(ymm8, ymm0, ymm1);
    vunpckhpd(ymm9, ymm0, ymm1);
    load_src(xmm2, 2, 0);
    vinsert(ymm2, 6, 0);
    load_src(xmm3, 3, 0);
    vinsert(ymm3, 7, 0);
    vunpcklpd(ymm10, ymm2, ymm3);
    vunpckhpd(ymm11, ymm2, ymm3);
    vshufps(ymm4, ymm8, ymm10, 0x88);
    vmovups(ptr[reg_tr_src_tmp], ymm4);

    vshufps(ymm5, ymm8, ymm10, 0xDD);
    vmovups(ptr[reg_tr_src_tmp + tr_src_stride], ymm5);
    vshufps(ymm6, ymm9, ymm11, 0x88);
    vmovups(ptr[reg_tr_src_tmp + 2 * tr_src_stride], ymm6);
    vshufps(ymm7, ymm9, ymm11, 0xDD);
    vmovups(ptr[reg_tr_src_tmp + 3 * tr_src_stride], ymm7);

    load_src(xmm0, 0, 4);
    vinsert(ymm0, 4, 4);
    load_src(xmm1, 1, 4);
    vinsert(ymm1, 5, 4);
    vunpcklpd(ymm8, ymm0, ymm1);
    vunpckhpd(ymm9, ymm0, ymm1);
    load_src(xmm2, 2, 4);
    vinsert(ymm2, 6, 4);
    load_src(xmm3, 3, 4);
    vinsert(ymm3, 7, 4);
    vunpcklpd(ymm10, ymm2, ymm3);
    vunpckhpd(ymm11, ymm2, ymm3);
    vshufps(ymm4, ymm8, ymm10, 0x88);
    vmovups(ptr[reg_tr_src_tmp + 4 * tr_src_stride], ymm4);
    vshufps(ymm5, ymm8, ymm10, 0xDD);
    vmovups(ptr[reg_tr_src_tmp + 5 * tr_src_stride], ymm5);
    vshufps(ymm6, ymm9, ymm11, 0x88);
    vmovups(ptr[reg_tr_src_tmp + 6 * tr_src_stride], ymm6);
    vshufps(ymm7, ymm9, ymm11, 0xDD);
    vmovups(ptr[reg_tr_src_tmp + 7 * tr_src_stride], ymm7);
}

void jit_brgemm_trans_m_k_f32_t::transpose_ker(int nrows, int ncolumns) {
    if (is_superset(conf_->isa, avx512_core)) {
        transpose_16x16(nrows, ncolumns);
    } else {
        transpose_16x16_avx2(nrows, ncolumns);
    }
}

void jit_brgemm_trans_m_k_f32_t::transpose(int nrows, int ncolumns) {

    Label K_loop, K_tail_or_done, K_done;
    const int num_nrows_loop = nrows / transpose_size;
    const int nrows_tail = nrows % transpose_size;
    const dim_t src_shift = static_cast<dim_t>(transpose_size) * conf_->ic
            * conf_->ks() * typesize;
    const dim_t tr_src_shift = static_cast<dim_t>(transpose_size) * typesize;

    if (num_nrows_loop > 1) mov(reg_row_loop, num_nrows_loop);
    L(K_loop);
    if (num_nrows_loop > 0) transpose_ker(transpose_size, ncolumns);
    if (num_nrows_loop > 1 || (num_nrows_loop > 0 && nrows_tail > 0)) {
        add(reg_src, src_shift);
        add(reg_tr_src, tr_src_shift);
    }
    if (num_nrows_loop > 1) {
        dec(reg_row_loop);
        jg(K_loop);
    }

    if (nrows_tail > 0) { transpose_ker(nrows_tail, ncolumns); }

    if (num_nrows_loop > 1 || nrows_tail > 0) {
        // reset pointers
        sub(reg_src, src_shift * num_nrows_loop);
        sub(reg_tr_src, tr_src_shift * num_nrows_loop);
    }
}

void jit_brgemm_trans_m_k_f32_t::init_masks(int tail_length) {
    if (isa_has_masks(conf_->isa)) {
        kmovw(k3333, 0x3333); // 0011001100110011
        kmovw(k5555, 0x5555); // 0101010101010101
        kmovw(kAAAA, 0xaaaa); // 1010101010101010
        kmovw(kCCCC, 0xcccc); // 1100110011001100
        kmovw(k0F0F, 0x0f0f); // 0000111100001111
        kmovw(kF0F0, 0xf0f0); // 1111000011110000
    } else if (tail_length) {
        lea(reg_tmp, ptr[rip + mask_label_]);
        vmovups(ymm_tail_mask, ptr[reg_tmp]);
        vmovups(xmm_upper_tail_mask, ptr[reg_tmp + vreg_traits_t<Xmm>::vlen]);
    }
}

void jit_brgemm_trans_m_k_f32_t::generate() {
    preamble();
    // [sp][ic] ->  [SP][ic][B][sp], where B = os_block
    // rows -> sp, cols -> ic
    // M -> ic, K -> sp
    assert(conf_->ic_block % transpose_size == 0);
    const int os_block = conf_->os_block;
    const int last_os_block_tail = conf_->K_tail % os_block;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = static_cast<dim_t>(conf_->ic) * conf_->ks() * typesize;
    tr_src_stride = static_cast<dim_t>(conf_->LDA) * typesize;
    const dim_t m_src_shift = static_cast<dim_t>(transpose_size) * typesize;
    const dim_t m_tr_src_shift = tr_src_stride * transpose_size;

    const dim_t batch_src_shift = src_stride * os_block;
    const dim_t batch_tr_src_shift = tr_src_stride * conf_->M;

    const int simd_tail = ic_tail; // last_os_block_tail % transpose_size;
    init_masks(simd_tail);
    if (last_os_block_tail && !isa_has_masks(conf_->isa))
        uni_vxorps(xmm_zero, xmm_zero, xmm_zero);

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    auto compute_M = [&](bool is_os_tail) {
        const auto nrows = is_os_tail ? last_os_block_tail : os_block;
        mov(reg_loop_M, ptr[param1 + GET_OFF(current_M)]);
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label M_loop, M_tail_or_done, M_done;
        if (ic_tail > 0) {
            cmp(reg_loop_M, transpose_size);
            jl(M_tail_or_done, T_NEAR);
        }

        L(M_loop);
        transpose(nrows, transpose_size);
        if (conf_->ic_block > transpose_size) {
            add(reg_src, m_src_shift);
            add(reg_tr_src, m_tr_src_shift);
            sub(reg_loop_M, transpose_size);
            cmp(reg_loop_M, transpose_size);
            jge(M_loop, T_NEAR);
        } else {
            jmp(M_done, T_NEAR);
        }

        L(M_tail_or_done);
        if (ic_tail > 0) {
            cmp(reg_loop_M, 0);
            jle(M_done, T_NEAR);

            transpose(nrows, ic_tail);
        }
        L(M_done);
    };

    auto compute_batch = [&](bool is_os_tail) {
        Label batch_loop;
        L(batch_loop);

        compute_M(is_os_tail);
        add(reg_src_base, batch_src_shift);
        add(reg_tr_src_base, batch_tr_src_shift);

        sub(reg_loop_batch, 1);
        jnz(batch_loop, T_NEAR);
    };

    Label K_tail;
    if (last_os_block_tail > 0) {
        cmp(reg_loop_K, os_block);
        jl(K_tail, T_NEAR);
    }

    compute_batch(false);

    if (last_os_block_tail > 0) {
        Label K_done;
        jmp(K_done, T_NEAR);

        L(K_tail);
        compute_batch(true);
        L(K_done);
    }

    postamble();
    if (simd_tail > 0 && !isa_has_masks(conf_->isa)) {
        align(32);
        L(mask_label_);
        for (int i = 0; i < simd_tail; ++i)
            dd(~uint32_t(0));
        for (int i = simd_tail; i < transpose_size; ++i)
            dd(0);
    }
}

struct jit_brgemm_trans_m_k_bf16_t : public jit_brgemm_trans_src_t,
                                     public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_bf16_t)
    jit_brgemm_trans_m_k_bf16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf), jit_generator_t(jit_name()) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
    };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kFFFF = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t k55 = k5;
    opmask_t kCC = k6;
    opmask_t k33 = k7;
    opmask_t kTail = k1;

    reg32_t regw_tmp = r15d;

    reg64_t reg_k_src = r14;
    reg64_t reg_k_tr_src = r13;

    reg64_t reg_m_src = r12;
    reg64_t reg_m_tr_src = r11;

    reg64_t reg_batch_src = r10;
    reg64_t reg_batch_tr_src = r9;

    reg64_t reg_loop_batch = r8;
    reg64_t reg_loop_K = rax;
    reg64_t reg_loop_M = rbx;

    reg64_t reg_tr_src_tmp = abi_not_param1; // lnx -> rcx
    reg64_t imm_addr64 = rdx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    void transpose(
            reg64_t dst, reg64_t src, int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_bf16_t::transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [](int i) { return Zmm(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    auto kmovd = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovd(k, regw_tmp);
    };

    auto store = [&](Zmm r, int i) {
        mov(reg_tr_src_tmp, dst);

        auto k = kTail;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    const int ic_block = ncolumns;
    kmovd(kFFFF, ic_block < transpose_size ? (1 << ic_block) - 1 : 0xffff);

    for (int i = 0; i < nrows / 2; i++) {
        auto zmm_src0 = src_zmm(2 * i);
        auto zmm_src1 = src_zmm(2 * i + 1);
        auto src1 = src_ymm(2 * i + 1);
        vmovdqu16(zmm_src0 | kFFFF | T_z,
                EVEX_compress_addr(src, 2 * i * src_stride));
        vmovdqu16(zmm_src1 | kFFFF | T_z,
                EVEX_compress_addr(src, (2 * i + 1) * src_stride));
        vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = src_zmm(2 * i);
        vmovdqu16(zmm_src0 | kFFFF | T_z,
                EVEX_compress_addr(src, 2 * i * src_stride));
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < 16; i += 2) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [](int ic_idx) {
        assert(ic_idx < 16 && ic_idx >= 0);
        switch (ic_idx) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
            case 4: return 9;
            case 5: return 8;
            case 6: return 11;
            case 7: return 10;
            case 8: return 5;
            case 9: return 4;
            case 10: return 7;
            case 11: return 6;
            case 12: return 13;
            case 13: return 12;
            case 14: return 15;
            default: return 14;
        }
    };

    int store_tail = rnd_up(nrows, 2);
    kmovw(kTail, (1 << store_tail / 2) - 1);

    for (int ic = 0; ic < ic_block; ic++)
        store(src_zmm(get_vec_idx(ic)), ic);
}

void jit_brgemm_trans_m_k_bf16_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

    constexpr int amx_xf16_granularity = 2;
    const bool last_row_padded = is_superset(conf_->isa, avx512_core_amx)
            && conf_->os % amx_xf16_granularity != 0;
    const int eff_K_tail = conf_->K_tail - (last_row_padded ? 1 : 0);

    const int os_block = conf_->os_block;
    const int last_os_block_tail = eff_K_tail % transpose_size;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = static_cast<dim_t>(conf_->ic) * conf_->ks() * typesize;
    tr_src_stride = conf_->LDA * typesize;

    const dim_t batch_src_shift = static_cast<dim_t>(src_stride) * os_block;
    const dim_t batch_tr_src_shift
            = static_cast<dim_t>(tr_src_stride) * conf_->M;

    const dim_t M_src_shift = static_cast<dim_t>(transpose_size) * typesize;
    const dim_t M_tr_src_shift
            = static_cast<dim_t>(transpose_size) * conf_->LDA * typesize;

    const dim_t K_src_shift = static_cast<dim_t>(transpose_size) * conf_->ic
            * conf_->ks() * typesize;
    const dim_t K_tr_src_shift = static_cast<dim_t>(transpose_size) * typesize;

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff);
    kmovw(k5555, 0x5555);
    kmovw(kAAAA, 0xaaaa);
    kmovw(kAA, 0xaa);
    kmovw(k55, 0x55);
    kmovw(kCC, 0xcc);
    kmovw(k33, 0x33);

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [this](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa32(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, idx2);
    vmovdqa32(vidx3, idx3);
    vmovdqa32(vidx4, idx4);
    vmovdqa32(vidx5, (const int32_t *)idx5);

    auto compute_m_loop
            = [&](reg64_t &reg_base, reg64_t &reg_tr_base, bool is_os_tail) {
        mov(reg_loop_M, ptr[param1 + GET_OFF(current_M)]);
        mov(reg_m_src, reg_base);
        mov(reg_m_tr_src, reg_tr_base);

        Label M_loop_tail, M_loop;
        if (ic_tail > 0) {
            cmp(reg_loop_M, transpose_size);
            jl(M_loop_tail, T_NEAR);
        }
        L(M_loop);
        {
            transpose(reg_m_tr_src, reg_m_src,
                    is_os_tail ? last_os_block_tail : transpose_size,
                    transpose_size);
            add(reg_m_src, M_src_shift);
            add(reg_m_tr_src, M_tr_src_shift);
        }
        sub(reg_loop_M, transpose_size);
        cmp(reg_loop_M, transpose_size);
        jge(M_loop, T_NEAR);

        if (ic_tail > 0) {
            Label M_loop_done;
            L(M_loop_tail);
            cmp(reg_loop_M, 0);
            jle(M_loop_done, T_NEAR);

            transpose(reg_m_tr_src, reg_m_src,
                    is_os_tail ? last_os_block_tail : transpose_size, ic_tail);
            L(M_loop_done);
        }
    };

    auto compute_k_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base) {
        mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);
        mov(reg_k_src, reg_base);
        mov(reg_k_tr_src, reg_tr_base);

        Label K_tail, K_loop, K_done;
        if (last_os_block_tail > 0) {
            cmp(reg_loop_K, transpose_size);
            jl(K_tail, T_NEAR);
        }
        L(K_loop);
        {
            compute_m_loop(reg_k_src, reg_k_tr_src, false);
            add(reg_k_src, K_src_shift);
            add(reg_k_tr_src, K_tr_src_shift);
        }
        sub(reg_loop_K, transpose_size);
        cmp(reg_loop_K, transpose_size);
        jge(K_loop, T_NEAR);

        cmp(reg_loop_K, 0);
        je(K_done, T_NEAR);

        if (last_os_block_tail > 0) {
            L(K_tail);
            compute_m_loop(reg_k_src, reg_k_tr_src, true);
        }
        L(K_done);
    };

    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_batch_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_batch_tr_src, ptr[param1 + GET_OFF(tr_src)]);

    Label batch_loop;
    L(batch_loop);
    {
        compute_k_loop(reg_batch_src, reg_batch_tr_src);

        add(reg_batch_src, batch_src_shift);
        add(reg_batch_tr_src, batch_tr_src_shift);
    }
    sub(reg_loop_batch, 1);
    jnz(batch_loop, T_NEAR);

    postamble();
}

struct jit_brgemm_trans_m_k_f16_t : public jit_brgemm_trans_src_t,
                                    public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_f16_t)

    jit_brgemm_trans_m_k_f16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf), jit_generator_t(jit_name()) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum {
        typesize_in = sizeof(float16_t),
        typesize_out = sizeof(float),
        transpose_size = 16
    };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_K = r10;
    reg64_t reg_loop_M = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t reg_tmp = r14;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_f16_t::transpose_16x16(int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    auto load = [&](int i) {
        auto src_load = src_zmm(i);
        if (i >= nrows) {
            vpxord(src_load, src_load, src_load);
            return;
        }

        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vcvtph2psx(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [&](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        const bool partial_store = nrows < transpose_size;
        const auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        const auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i * 2;
            const int src_idx1 = src_idx0 + 1;

            const int next_src_idx0 = src_idx0 + 2;
            const int next_src_idx1 = src_idx1 + 2;
            const bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx1);
            const auto src0 = src_zmm(src_idx0);
            const auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            const int select_half = (i < 2) ? 0 : 2;
            const int src_idx0 = base_idx + i + select_half + 0;
            const int src_idx2 = src_idx0 + 2;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx2);
            const auto src0 = src_zmm(src_idx0);
            const auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i;
            const int src_idx4 = src_idx0 + 4;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto src0 = src_zmm(src_idx0);
            const auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        // swap 8
        const auto max_iters_phase_1 = std::min(ncolumns, 8);
        for (int i = 0; i < max_iters_phase_1; i++) {
            const auto tmp = tmp_zmm(i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        const auto max_iters_phase_2 = std::min(ncolumns - 8, 8);
        for (int i = 0; i < max_iters_phase_2; i++) {
            const auto tmp = tmp_zmm(8 + i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_trans_m_k_f16_t::generate() {
    preamble();
    assert(conf_->ic_block % transpose_size == 0);
    const int os_block = conf_->os_block;
    const int last_os_block_tail = conf_->K_tail % transpose_size;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = static_cast<dim_t>(conf_->ic) * conf_->ks() * typesize_in;
    tr_src_stride = static_cast<dim_t>(conf_->LDA) * typesize_out;
    const dim_t m_src_shift = transpose_size * typesize_in;
    const dim_t m_tr_src_shift = tr_src_stride * transpose_size;

    const dim_t batch_src_shift = src_stride * os_block;
    const dim_t batch_tr_src_shift = tr_src_stride * conf_->M;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_M = [&](bool is_os_tail) {
        const auto nrows = is_os_tail ? last_os_block_tail : transpose_size;
        mov(reg_loop_M, ptr[param1 + GET_OFF(current_M)]);
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label M_loop, M_tail_or_done, M_done;
        if (ic_tail > 0) {
            cmp(reg_loop_M, transpose_size);
            jl(M_tail_or_done, T_NEAR);
        }

        L(M_loop);
        transpose_16x16(nrows, transpose_size);
        if (conf_->ic_block > transpose_size) {
            add(reg_src, m_src_shift);
            add(reg_tr_src, m_tr_src_shift);
            sub(reg_loop_M, transpose_size);
            cmp(reg_loop_M, transpose_size);
            jge(M_loop, T_NEAR);
        } else {
            jmp(M_done, T_NEAR);
        }

        L(M_tail_or_done);
        if (ic_tail > 0) {
            cmp(reg_loop_M, 0);
            jle(M_done, T_NEAR);

            transpose_16x16(nrows, ic_tail);
        }
        L(M_done);
    };

    auto compute_batch = [&](bool is_os_tail) {
        Label batch_loop;
        L(batch_loop);

        compute_M(is_os_tail);
        add(reg_src_base, batch_src_shift);
        add(reg_tr_src_base, batch_tr_src_shift);

        sub(reg_loop_batch, 1);
        jnz(batch_loop, T_NEAR);
    };

    Label K_tail;
    if (last_os_block_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    compute_batch(false);

    if (last_os_block_tail > 0) {
        Label K_done;
        jmp(K_done, T_NEAR);

        L(K_tail);
        compute_batch(true);
        L(K_done);
    }

    postamble();
}

void jit_brgemm_relo_copy_to_wbuffer_t::generate() {

    const bool is_xf16 = one_of(wjcp.wei_dt, data_type::bf16, data_type::f16);
    const bool is_f32 = wjcp.wei_dt == data_type::f32;

    // required for use of VPERMB instruction
    assert(IMPLICATION(
            !(is_xf16 || is_f32), cpu().has(Xbyak::util::Cpu::tAVX512_VBMI)));
    assert(wjcp.inp_oc_block == 16);

    preamble();

    const int vnni_width = data_type_vnni_granularity(wjcp.wei_dt);
    const auto wei_dsz = types::data_type_size(wjcp.wei_dt);
    const auto inp_ocb_size = wjcp.inp_oc_block * vnni_width * wei_dsz;
    const auto out_ocb_size = wjcp.out_oc_block * vnni_width * wei_dsz;
    const auto oc_chunks = wjcp.out_oc_block / wjcp.inp_oc_block;
    const auto has_ocb_tail = (oc_chunks != wjcp.last_occ_to_copy);
    auto nb_rd = div_up(wjcp.rd, vnni_width);
    if (wjcp.is_rd_padded_to_block) nb_rd = rnd_up(nb_rd, 16);
    const auto rtail = (wjcp.rd % vnni_width) * wjcp.inp_oc_block;
    const auto has_rdb_tail = (nb_rd - 1) * vnni_width >= wjcp.rd;

    auto copy_zmm = [&](bool rd_tail) {
        auto zmm_src_tmp = (rd_tail) ? zmm_src | kmask_load | T_z : zmm_src;
        if (is_xf16) {
            vmovdqu16(zmm_src_tmp, ptr[aux_reg_src]);
            vpermw(zmm_dst, zmm_idx, zmm_src);
            vmovdqu16(ptr[aux_reg_dst], zmm_dst);
        } else if (is_f32) {
            vmovdqu32(zmm_src_tmp, ptr[aux_reg_src]);
            vmovdqu32(ptr[aux_reg_dst], zmm_src_tmp);
        } else {
            vmovdqu8(zmm_src_tmp, ptr[aux_reg_src]);
            vpermb(zmm_dst, zmm_idx, zmm_src);
            vmovdqu8(ptr[aux_reg_dst], zmm_dst);
        }
    };

    auto rdb_loop = [&](bool last_ocb) {
        for (int rdb = 0; rdb < nb_rd; rdb++) {
            mov(aux_reg_src, reg_src);
            mov(aux_reg_dst, reg_dst);

            for (int occ = 0; occ < oc_chunks; occ++) {
                if ((rdb * vnni_width >= wjcp.rd)
                        || (last_ocb && occ >= wjcp.last_occ_to_copy)) {
                    if (is_xf16)
                        vmovdqu16(ptr[aux_reg_dst], zmm_zero);
                    else
                        vmovdqu8(ptr[aux_reg_dst], zmm_zero);
                } else if ((rdb + 1) * vnni_width > wjcp.rd)
                    copy_zmm(true);
                else
                    copy_zmm(false);

                add(aux_reg_src, wjcp.inp_ocb_offs);
                add(aux_reg_dst, inp_ocb_size);
            }
            add(reg_src, inp_ocb_size);
            add(reg_dst, out_ocb_size);
        }
    };

    if (rtail > 0) {
        uint64_t mask = (UINT64_C(1) << rtail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask_load, reg_tmp);
    }

    if (has_rdb_tail || has_ocb_tail) vpxord(zmm_zero, zmm_zero, zmm_zero);

    // load permute indices from data section
    Label full_ocb_label, finish_label, permute_index_table;
    if (!is_f32) {
        if (is_xf16)
            vmovdqu16(zmm_idx, ptr[rip + permute_index_table]);
        else
            vmovdqu8(zmm_idx, ptr[rip + permute_index_table]);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    if (has_ocb_tail) {
        mov(reg_tmp, ptr[param1 + GET_OFF(last_ocb)]);
        cmp(reg_tmp, 0);
        je(full_ocb_label, T_NEAR);
        rdb_loop(true);
        jmp(finish_label, T_NEAR);
    }

    L(full_ocb_label);
    rdb_loop(false);

    L(finish_label);
    postamble();

    align(64);
    L(permute_index_table);
    if (!is_f32) {
        const uint8_t no = 16; // 16o
        for (uint8_t o = 0; o < no; ++o) {
            for (uint8_t r = 0; r < static_cast<uint8_t>(vnni_width); r++) {
                const uint8_t index = o + r * no;
                if (is_xf16)
                    dw(index);
                else
                    db(index);
            }
        }
    }
}

#undef GET_OFF

status_t create_brgemm_trans_src(
        std::unique_ptr<jit_brgemm_trans_src_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf) {

    if (conf->prop_kind != dnnl_backward_weights)
        return status::invalid_arguments;

    if (conf->src_dt == data_type::f32) {
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_m_k_f32_t(conf)));
    } else if (utils::one_of(conf->src_dt, data_type::bf16, data_type::f16)
            && conf->isa != avx512_core_fp16) {
        CHECK(safe_ptr_assign(
                trans_ker, new jit_brgemm_trans_m_k_bf16_t(conf)));
    } else if (conf->src_dt == data_type::f16) {
        assert(conf->isa == avx512_core_fp16);
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_m_k_f16_t(conf)));
    } else {
        return status::invalid_arguments;
    }

    return trans_ker->create_kernel();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
