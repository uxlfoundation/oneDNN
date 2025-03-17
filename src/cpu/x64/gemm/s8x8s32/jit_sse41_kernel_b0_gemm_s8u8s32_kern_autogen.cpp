/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_sse41_kernel_b0_gemm_s8u8s32_kern_t::
        jit_sse41_kernel_b0_gemm_s8u8s32_kern_t()
    : jit_generator_t(jit_name()) {}

void jit_sse41_kernel_b0_gemm_s8u8s32_kern_t::generate() {

#ifndef _WIN32

#define M rdi
#define N rsi
#define K rdx
#define A r8
#define B r9
#define C r10
#define LDC r11

#define AA rcx
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#else

#define M rcx
#define N rdx
#define K r8
#define A rsi
#define B r9
#define C r10
#define LDC r11

#define AA rdi
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#endif

#ifdef _WIN32
#define ARG_A (args_offset - 16) + rsp
#define ARG_B (args_offset - 8) + rsp
#endif
#define ARG_C ((args_offset + 0) + rsp)
#define ARG_LDC ((args_offset + 8) + rsp)

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(91);

        auto stack_alloc_size = 32;
        auto args_offset = stack_alloc_size + get_size_of_abi_save_regs() + 8;
#ifdef _WIN32
        args_offset += 48;
#endif
        preamble();
        sub(rsp, stack_alloc_size);
#ifdef _WIN32
        mov(A, ptr[ARG_A]);
        mov(B, ptr[ARG_B]);
#endif

        mov(C, qword[ARG_C]);
        mov(LDC, qword[ARG_LDC]);
        sub(A, -128);
        sub(B, -128);
        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(K, qword[K]);
        lea(LDC, ptr[LDC * 4 + 0x0]);
        xorps(xmm8, xmm8);
        xorps(xmm9, xmm9);
        xorps(xmm10, xmm10);
        xorps(xmm11, xmm11);
        xorps(xmm12, xmm12);
        xorps(xmm13, xmm13);
        xorps(xmm14, xmm14);
        xorps(xmm15, xmm15);
        mov(H, 0x10001);
        movq(xmm7, H);
        pshufd(xmm7, xmm7, 0x0);
        mov(J, M);
        cmp(J, 0x10);
        jl(labels[75], T_NEAR);
        align(4);

        L(labels[69]);
        mov(CO1, C);
        add(C, 0x40);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x20);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[65], T_NEAR);
        align(4);

        L(labels[78]);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm1, xword[AO - 0x70]);
        movdqu(xmm2, xword[AO - 0x60]);
        movdqu(xmm3, xword[AO - 0x50]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[61], T_NEAR);
        sub(H, 0x8);
        jle(labels[59], T_NEAR);
        align(4);

        L(labels[86]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        movdqu(xmm2, xword[AO - 0x20]);
        movdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO]);
        movdqu(xmm1, xword[AO + 0x10]);
        movdqu(xmm2, xword[AO + 0x20]);
        movdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[86], T_NEAR);
        align(4);

        L(labels[59]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[61], T_NEAR);
        align(4);

        L(labels[60]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        movdqu(xmm2, xword[AO - 0x20]);
        movdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO]);
        movdqu(xmm1, xword[AO + 0x10]);
        movdqu(xmm2, xword[AO + 0x20]);
        movdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[60], T_NEAR);
        align(4);

        L(labels[61]);
        mov(H, K);
        test(H, 0x4);
        je(labels[62], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        add(AO, 0x40);
        add(BO, 0x8);
        align(4);

        L(labels[62]);
        mov(H, K);
        test(H, 0x2);
        je(labels[63], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        punpckhwd(xmm1, xmm6);
        movdqu(xmm2, xword[AO - 0x70]);
        movaps(xmm3, xmm2);
        punpcklwd(xmm2, xmm6);
        punpckhwd(xmm3, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(labels[63]);
        mov(H, K);
        test(H, 0x1);
        je(labels[64], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        pshufd(xmm1, xmm3, 0x55);
        punpcklbw(xmm1, xmm6);
        punpcklwd(xmm1, xmm6);
        pshufd(xmm2, xmm3, 0xaa);
        punpcklbw(xmm2, xmm6);
        punpcklwd(xmm2, xmm6);
        pshufd(xmm3, xmm3, 0xff);
        punpcklbw(xmm3, xmm6);
        punpcklwd(xmm3, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm11, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm13, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm15, xmm4);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(labels[64]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movdqu(xword[CO1 + 0x10], xmm10);
        xorps(xmm10, xmm10);
        movdqu(xword[CO1 + 0x20], xmm12);
        xorps(xmm12, xmm12);
        movdqu(xword[CO1 + 0x30], xmm14);
        xorps(xmm14, xmm14);
        movdqu(xword[CO1 + LDC * 1], xmm9);
        xorps(xmm9, xmm9);
        movdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        xorps(xmm11, xmm11);
        movdqu(xword[CO1 + LDC * 1 + 0x20], xmm13);
        xorps(xmm13, xmm13);
        movdqu(xword[CO1 + LDC * 1 + 0x30], xmm15);
        xorps(xmm15, xmm15);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[78], T_NEAR);
        align(4);

        L(labels[65]);
        test(I, 0x1);
        jle(labels[74], T_NEAR);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm1, xword[AO - 0x70]);
        movdqu(xmm2, xword[AO - 0x60]);
        movdqu(xmm3, xword[AO - 0x50]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[70], T_NEAR);
        sub(H, 0x8);
        jle(labels[67], T_NEAR);
        align(4);

        L(labels[66]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        movdqu(xmm2, xword[AO - 0x20]);
        movdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO]);
        movdqu(xmm1, xword[AO + 0x10]);
        movdqu(xmm2, xword[AO + 0x20]);
        movdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[66], T_NEAR);
        align(4);

        L(labels[67]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[70], T_NEAR);
        align(4);

        L(labels[68]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        movdqu(xmm2, xword[AO - 0x20]);
        movdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO]);
        movdqu(xmm1, xword[AO + 0x10]);
        movdqu(xmm2, xword[AO + 0x20]);
        movdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[68], T_NEAR);
        align(4);

        L(labels[70]);
        mov(H, K);
        test(H, 0x4);
        je(labels[71], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        add(AO, 0x40);
        add(BO, 0x4);
        align(4);

        L(labels[71]);
        mov(H, K);
        test(H, 0x2);
        je(labels[72], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        punpckhwd(xmm1, xmm6);
        movdqu(xmm2, xword[AO - 0x70]);
        movaps(xmm3, xmm2);
        punpcklwd(xmm2, xmm6);
        punpckhwd(xmm3, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        add(AO, 0x20);
        add(BO, 0x2);
        align(4);

        L(labels[72]);
        mov(H, K);
        test(H, 0x1);
        je(labels[73], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        pshufd(xmm1, xmm3, 0x55);
        punpcklbw(xmm1, xmm6);
        punpcklwd(xmm1, xmm6);
        pshufd(xmm2, xmm3, 0xaa);
        punpcklbw(xmm2, xmm6);
        punpcklwd(xmm2, xmm6);
        pshufd(xmm3, xmm3, 0xff);
        punpcklbw(xmm3, xmm6);
        punpcklwd(xmm3, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm1);
        pmaddwd(xmm6, xmm7);
        paddd(xmm10, xmm6);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm2);
        pmaddwd(xmm6, xmm7);
        paddd(xmm12, xmm6);
        pmaddubsw(xmm4, xmm3);
        pmaddwd(xmm4, xmm7);
        paddd(xmm14, xmm4);
        add(AO, 0x10);
        add(BO, 0x1);
        align(4);

        L(labels[73]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movdqu(xword[CO1 + 0x10], xmm10);
        xorps(xmm10, xmm10);
        movdqu(xword[CO1 + 0x20], xmm12);
        xorps(xmm12, xmm12);
        movdqu(xword[CO1 + 0x30], xmm14);
        xorps(xmm14, xmm14);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[74]);
        mov(A, AO);
        sub(J, 0x10);
        cmp(J, 0x10);
        jge(labels[69], T_NEAR);
        align(4);

        L(labels[75]);
        test(J, 0x8);
        jle(labels[4], T_NEAR);
        mov(CO1, C);
        add(C, 0x20);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x10);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[85], T_NEAR);
        align(4);

        L(labels[76]);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm1, xword[AO - 0x70]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[81], T_NEAR);
        sub(H, 0x8);
        jle(labels[79], T_NEAR);
        align(4);

        L(labels[77]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        movdqu(xmm0, xword[AO - 0x60]);
        movdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[77], T_NEAR);
        align(4);

        L(labels[79]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[81], T_NEAR);
        align(4);

        L(labels[80]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        movdqu(xmm0, xword[AO - 0x60]);
        movdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[80], T_NEAR);
        align(4);

        L(labels[81]);
        mov(H, K);
        test(H, 0x4);
        je(labels[82], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        add(AO, 0x20);
        add(BO, 0x8);
        align(4);

        L(labels[82]);
        mov(H, K);
        test(H, 0x2);
        je(labels[83], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        punpckhwd(xmm1, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(labels[83]);
        mov(H, K);
        test(H, 0x1);
        je(labels[84], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        pshufd(xmm1, xmm3, 0x55);
        punpcklbw(xmm1, xmm6);
        punpcklwd(xmm1, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm9, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm11, xmm4);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(labels[84]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movdqu(xword[CO1 + 0x10], xmm10);
        xorps(xmm10, xmm10);
        movdqu(xword[CO1 + LDC * 1], xmm9);
        xorps(xmm9, xmm9);
        movdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        xorps(xmm11, xmm11);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[76], T_NEAR);
        align(4);

        L(labels[85]);
        test(I, 0x1);
        jle(labels[3], T_NEAR);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm1, xword[AO - 0x70]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[90], T_NEAR);
        sub(H, 0x8);
        jle(labels[88], T_NEAR);
        align(4);

        L(labels[87]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x60]);
        movdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[87], T_NEAR);
        align(4);

        L(labels[88]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[90], T_NEAR);
        align(4);

        L(labels[89]);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x60]);
        movdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x40]);
        movdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[89], T_NEAR);
        align(4);

        L(labels[90]);
        mov(H, K);
        test(H, 0x4);
        je(labels[0], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(labels[0]);
        mov(H, K);
        test(H, 0x2);
        je(labels[1], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        punpckhwd(xmm1, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(labels[1]);
        mov(H, K);
        test(H, 0x1);
        je(labels[2], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        pshufd(xmm1, xmm3, 0x55);
        punpcklbw(xmm1, xmm6);
        punpcklwd(xmm1, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        movaps(xmm6, xmm4);
        pmaddubsw(xmm6, xmm0);
        pmaddwd(xmm6, xmm7);
        paddd(xmm8, xmm6);
        pmaddubsw(xmm4, xmm1);
        pmaddwd(xmm4, xmm7);
        paddd(xmm10, xmm4);
        add(AO, 0x8);
        add(BO, 0x1);
        align(4);

        L(labels[2]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movdqu(xword[CO1 + 0x10], xmm10);
        xorps(xmm10, xmm10);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[3]);
        mov(A, AO);
        align(4);

        L(labels[4]);
        test(J, 0x4);
        jle(labels[22], T_NEAR);
        mov(CO1, C);
        add(C, 0x10);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x8);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[13], T_NEAR);
        align(4);

        L(labels[5]);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[9], T_NEAR);
        sub(H, 0x8);
        jle(labels[7], T_NEAR);
        align(4);

        L(labels[6]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[6], T_NEAR);
        align(4);

        L(labels[7]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[9], T_NEAR);
        align(4);

        L(labels[8]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[8], T_NEAR);
        align(4);

        L(labels[9]);
        mov(H, K);
        test(H, 0x4);
        je(labels[10], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x10);
        add(BO, 0x8);
        align(4);

        L(labels[10]);
        mov(H, K);
        test(H, 0x2);
        je(labels[11], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(labels[11]);
        mov(H, K);
        test(H, 0x1);
        je(labels[12], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(labels[12]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movdqu(xword[CO1 + LDC * 1], xmm9);
        xorps(xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[5], T_NEAR);
        align(4);

        L(labels[13]);
        test(I, 0x1);
        jle(labels[21], T_NEAR);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[17], T_NEAR);
        sub(H, 0x8);
        jle(labels[15], T_NEAR);
        align(4);

        L(labels[14]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[14], T_NEAR);
        align(4);

        L(labels[15]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[17], T_NEAR);
        align(4);

        L(labels[16]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[16], T_NEAR);
        align(4);

        L(labels[17]);
        mov(H, K);
        test(H, 0x4);
        je(labels[18], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(labels[18]);
        mov(H, K);
        test(H, 0x2);
        je(labels[19], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(labels[19]);
        mov(H, K);
        test(H, 0x1);
        je(labels[20], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x4);
        add(BO, 0x1);
        align(4);

        L(labels[20]);
        movdqu(xword[CO1], xmm8);
        xorps(xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[21]);
        mov(A, AO);
        align(4);

        L(labels[22]);
        test(J, 0x2);
        jle(labels[40], T_NEAR);
        mov(CO1, C);
        add(C, 0x8);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x4);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[31], T_NEAR);
        align(4);

        L(labels[23]);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[27], T_NEAR);
        sub(H, 0x8);
        jle(labels[25], T_NEAR);
        align(4);

        L(labels[24]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[24], T_NEAR);
        align(4);

        L(labels[25]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[27], T_NEAR);
        align(4);

        L(labels[26]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[26], T_NEAR);
        align(4);

        L(labels[27]);
        mov(H, K);
        test(H, 0x4);
        je(labels[28], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x8);
        add(BO, 0x8);
        align(4);

        L(labels[28]);
        mov(H, K);
        test(H, 0x2);
        je(labels[29], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(labels[29]);
        mov(H, K);
        test(H, 0x1);
        je(labels[30], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(labels[30]);
        movlps(qword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movlps(qword[CO1 + LDC * 1], xmm9);
        xorps(xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[23], T_NEAR);
        align(4);

        L(labels[31]);
        test(I, 0x1);
        jle(labels[39], T_NEAR);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[35], T_NEAR);
        sub(H, 0x8);
        jle(labels[33], T_NEAR);
        align(4);

        L(labels[32]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[32], T_NEAR);
        align(4);

        L(labels[33]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[35], T_NEAR);
        align(4);

        L(labels[34]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[34], T_NEAR);
        align(4);

        L(labels[35]);
        mov(H, K);
        test(H, 0x4);
        je(labels[36], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(labels[36]);
        mov(H, K);
        test(H, 0x2);
        je(labels[37], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(labels[37]);
        mov(H, K);
        test(H, 0x1);
        je(labels[38], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x2);
        add(BO, 0x1);
        align(4);

        L(labels[38]);
        movlps(qword[CO1], xmm8);
        xorps(xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[39]);
        mov(A, AO);
        align(4);

        L(labels[40]);
        test(J, 0x1);
        jle(labels[58], T_NEAR);
        mov(CO1, C);
        add(C, 0x4);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x2);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[49], T_NEAR);
        align(4);

        L(labels[41]);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[45], T_NEAR);
        sub(H, 0x8);
        jle(labels[43], T_NEAR);
        align(4);

        L(labels[42]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[42], T_NEAR);
        align(4);

        L(labels[43]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[45], T_NEAR);
        align(4);

        L(labels[44]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0xaa);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0xff);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        movdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[44], T_NEAR);
        align(4);

        L(labels[45]);
        mov(H, K);
        test(H, 0x4);
        je(labels[46], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x4);
        add(BO, 0x8);
        align(4);

        L(labels[46]);
        mov(H, K);
        test(H, 0x2);
        je(labels[47], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x2);
        add(BO, 0x4);
        align(4);

        L(labels[47]);
        mov(H, K);
        test(H, 0x1);
        je(labels[48], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm9, xmm4);
        add(AO, 0x1);
        add(BO, 0x2);
        align(4);

        L(labels[48]);
        movss(dword[CO1], xmm8);
        xorps(xmm8, xmm8);
        movss(dword[CO1 + LDC * 1], xmm9);
        xorps(xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[41], T_NEAR);
        align(4);

        L(labels[49]);
        test(I, 0x1);
        jle(labels[57], T_NEAR);
        mov(AO, A);
        movdqu(xmm0, xword[AO - 0x80]);
        movdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[53], T_NEAR);
        sub(H, 0x8);
        jle(labels[51], T_NEAR);
        align(4);

        L(labels[50]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[50], T_NEAR);
        align(4);

        L(labels[51]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[53], T_NEAR);
        align(4);

        L(labels[52]);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        movdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        pshufd(xmm4, xmm5, 0x55);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        movdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        movdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[52], T_NEAR);
        align(4);

        L(labels[53]);
        mov(H, K);
        test(H, 0x4);
        je(labels[54], T_NEAR);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(labels[54]);
        mov(H, K);
        test(H, 0x2);
        je(labels[55], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm0, xword[AO - 0x80]);
        movaps(xmm1, xmm0);
        punpcklwd(xmm0, xmm6);
        movss(xmm5, dword[BO - 0x80]);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(labels[55]);
        mov(H, K);
        test(H, 0x1);
        je(labels[56], T_NEAR);
        xorps(xmm6, xmm6);
        movdqu(xmm3, xword[AO - 0x80]);
        pshufd(xmm0, xmm3, 0x0);
        punpcklbw(xmm0, xmm6);
        punpcklwd(xmm0, xmm6);
        movd(xmm5, dword[BO - 0x80]);
        punpcklbw(xmm5, xmm5);
        punpcklwd(xmm5, xmm5);
        pshufd(xmm4, xmm5, 0x0);
        pmaddubsw(xmm4, xmm0);
        pmaddwd(xmm4, xmm7);
        paddd(xmm8, xmm4);
        add(AO, 0x1);
        add(BO, 0x1);
        align(4);

        L(labels[56]);
        movss(dword[CO1], xmm8);
        xorps(xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[57]);
        mov(A, AO);
        align(4);

        L(labels[58]);
        add(rsp, stack_alloc_size);
        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef K
#undef A
#undef B
#undef C
#undef LDC
#undef AA
#undef I
#undef J
#undef H
#undef AO
#undef BO
#undef CO1
#undef CO2
#ifdef _WIN32
#undef ARG_A
#undef ARG_B
#endif
#undef ARG_C
#undef ARG_LDC
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
