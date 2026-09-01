//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include "kai/ops/gemm/kai_ops.hpp"


#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

void sme2_interleaved_nomerge_fp32_mopa_2VLx2VL(const float *const A, const float *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const float *const A,
      const float *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(K * sizeof(float)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
        bias(bias),
        accumulator_buffer(accumulator_buffer),
        flags(0x0)
    {
      if (accumulate)
      {
        flags |= 1 << 0;  // FILL_ACCUMULATORS_FROM_BUFFER
      }
      if (C == nullptr)
      {
        flags |= 1 << 1;  // STORE_ACCUMULATORS_TO_BUFFER
      }
      if (act.type == Activation::Type::None)
      {
        flags |= 1 << 2;  // SKIP_ACTIVATION
      }

      // Initialise the activation values
      switch (act.type)
      {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const float *const A;
    const float *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const float *const bias;


    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x8, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x17, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x8, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c634  // ld1w { z20.s-z23.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c63c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x15, [%x[args], %[offsetof_K]]\n"
      "mov x14, #0\n"
      "mov x13, #0\n"
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "ldr w10, [%x[args], %[offsetof_N]]\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "3:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x27, x9\n"
      ".inst 0x25aa45b0  // whilelt pn8.s, x13, x10, VLx2\n"
      "tbnz x8, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z15.s, #1.0\n"
      ".inst 0xa10d4291  // ld1w { z17.s, z25.s }, p8/Z, [x20, x13, LSL #2]\n"
      ".inst 0x809101e0  // fmopa za0.s, p0/M, p0/M, z15.s, z17.s\n"
      ".inst 0x809901e1  // fmopa za1.s, p0/M, p0/M, z15.s, z25.s\n"
      ".inst 0x809101e2  // fmopa za2.s, p0/M, p0/M, z15.s, z17.s\n"
      ".inst 0x809901e3  // fmopa za3.s, p0/M, p0/M, z15.s, z25.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x13\n"
      "mov x21, x14\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
      "cmp x20, x10\n"
      "mov x20, x8\n"
      "csel x21, x14, x21, LT\n"
      "bfm x8, XZR, #0, #0  // bfc x8, #0, #0x1\n"
      "cmp x21, x11\n"
      "csel x8, x20, x8, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x15, #0x2\n"
      "and x20, x15, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa140c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27]\n"
      ".inst 0xa141c770  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa140c780  // ld1w { z0.s, z4.s, z8.s, z12.s }, pn9.b/Z, [x28]\n"
      ".inst 0xa141c781  // ld1w { z1.s, z5.s, z9.s, z13.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x80800240  // fmopa za0.s, p0/M, p0/M, z18.s, z0.s\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x80840241  // fmopa za1.s, p0/M, p0/M, z18.s, z4.s\n"
      ".inst 0x808002c2  // fmopa za2.s, p0/M, p0/M, z22.s, z0.s\n"
      ".inst 0x808402c3  // fmopa za3.s, p0/M, p0/M, z22.s, z4.s\n"
      ".inst 0x80880340  // fmopa za0.s, p0/M, p0/M, z26.s, z8.s\n"
      ".inst 0x808c0341  // fmopa za1.s, p0/M, p0/M, z26.s, z12.s\n"
      ".inst 0x808803c2  // fmopa za2.s, p0/M, p0/M, z30.s, z8.s\n"
      ".inst 0x808c03c3  // fmopa za3.s, p0/M, p0/M, z30.s, z12.s\n"
      ".inst 0xa140c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27]\n"
      ".inst 0x80810200  // fmopa za0.s, p0/M, p0/M, z16.s, z1.s\n"
      ".inst 0xa140c780  // ld1w { z0.s, z4.s, z8.s, z12.s }, pn9.b/Z, [x28]\n"
      ".inst 0x80850201  // fmopa za1.s, p0/M, p0/M, z16.s, z5.s\n"
      ".inst 0x80810282  // fmopa za2.s, p0/M, p0/M, z20.s, z1.s\n"
      ".inst 0x80850283  // fmopa za3.s, p0/M, p0/M, z20.s, z5.s\n"
      ".inst 0x80890300  // fmopa za0.s, p0/M, p0/M, z24.s, z9.s\n"
      ".inst 0x808d0301  // fmopa za1.s, p0/M, p0/M, z24.s, z13.s\n"
      ".inst 0x80890382  // fmopa za2.s, p0/M, p0/M, z28.s, z9.s\n"
      ".inst 0x808d0383  // fmopa za3.s, p0/M, p0/M, z28.s, z13.s\n"
      ".inst 0xa141c770  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa141c781  // ld1w { z1.s, z5.s, z9.s, z13.s }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x80800240  // fmopa za0.s, p0/M, p0/M, z18.s, z0.s\n"
      ".inst 0x80840241  // fmopa za1.s, p0/M, p0/M, z18.s, z4.s\n"
      ".inst 0x808002c2  // fmopa za2.s, p0/M, p0/M, z22.s, z0.s\n"
      ".inst 0x808402c3  // fmopa za3.s, p0/M, p0/M, z22.s, z4.s\n"
      ".inst 0x80880340  // fmopa za0.s, p0/M, p0/M, z26.s, z8.s\n"
      ".inst 0x808c0341  // fmopa za1.s, p0/M, p0/M, z26.s, z12.s\n"
      ".inst 0x808803c2  // fmopa za2.s, p0/M, p0/M, z30.s, z8.s\n"
      ".inst 0x808c03c3  // fmopa za3.s, p0/M, p0/M, z30.s, z12.s\n"
      ".inst 0x80810200  // fmopa za0.s, p0/M, p0/M, z16.s, z1.s\n"
      ".inst 0x80850201  // fmopa za1.s, p0/M, p0/M, z16.s, z5.s\n"
      ".inst 0x80810282  // fmopa za2.s, p0/M, p0/M, z20.s, z1.s\n"
      ".inst 0x80850283  // fmopa za3.s, p0/M, p0/M, z20.s, z5.s\n"
      ".inst 0x80890300  // fmopa za0.s, p0/M, p0/M, z24.s, z9.s\n"
      ".inst 0x808d0301  // fmopa za1.s, p0/M, p0/M, z24.s, z13.s\n"
      ".inst 0x80890382  // fmopa za2.s, p0/M, p0/M, z28.s, z9.s\n"
      ".inst 0x808d0383  // fmopa za3.s, p0/M, p0/M, z28.s, z13.s\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa040476c  // ld1w { z12.s-z13.s }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      ".inst 0xa1404783  // ld1w { z3.s, z11.s }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #2\n"
      ".inst 0x80830180  // fmopa za0.s, p0/M, p0/M, z12.s, z3.s\n"
      ".inst 0x808b0181  // fmopa za1.s, p0/M, p0/M, z12.s, z11.s\n"
      ".inst 0x808301a2  // fmopa za2.s, p0/M, p0/M, z13.s, z3.s\n"
      ".inst 0x808b01a3  // fmopa za3.s, p0/M, p0/M, z13.s, z11.s\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17]\n"
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xa041c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa042c630  // ld1w { z16.s-z19.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c638  // ld1w { z24.s-z27.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa060c608  // st1w { z8.s-z11.s }, pn9.b, [x16]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa061c614  // st1w { z20.s-z23.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0840703  // mova za3h.s[x12], { z24.s-z27.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c61c  // st1w { z28.s-z31.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c60c  // st1w { z12.s-z15.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 12b\n"
      "b 31f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
      ".inst 0xa060c600  // st1w { z0.s-z3.s }, pn9.b, [x16]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c610  // st1w { z16.s-z19.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c608  // st1w { z8.s-z11.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      ".inst 0xa063c604  // st1w { z4.s-z7.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 14b\n"
      "b 31f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "add x26, x26, x13, LSL #2\n"  // C += n
      "madd x26, x14, x24, x26\n"  // C += m * ldc
      "tbz x8, #2, 22f\n"
      "cntw x23\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xa1604340  // st1w { z0.s, z8.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604341  // st1w { z1.s, z9.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      ".inst 0xa1604342  // st1w { z2.s, z10.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 22f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Skip activation: Accumulator row 1 loop
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604347  // st1w { z7.s, z15.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 19b\n"
      "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "21:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 22f\n"
      "b 29f\n"
      "22:"  // Store to output array: Skip activation: End
      "cntw x23\n"
      "ld1rw { z21.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "ld1rw { z20.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 24f\n"
      "23:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604353  // st1w { z19.s, z27.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 23b\n"
      "24:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 25f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "25:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 29f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 27f\n"
      "26:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860440  // mova { z0.s-z3.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xc1b4caa0  // fclamp { z0.s-z3.s }, z21.s, z20.s\n"
      ".inst 0xc1b4caa8  // fclamp { z8.s-z11.s }, z21.s, z20.s\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604340  // st1w { z0.s, z8.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604341  // st1w { z1.s, z9.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604342  // st1w { z2.s, z10.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      ".inst 0xa1604343  // st1w { z3.s, z11.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "blt 26b\n"
      "27:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 28f\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "28:"  // Store to output array: Accumulator row 1 oddments: End
      "29:"  // Store to output array: End
      "tbz x8, #0, 31f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "30:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c628  // ld1w { z8.s-z11.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 30b\n"
      "31:"  // End block
      "incw x13, ALL, MUL #2\n"
      "cmp x13, x10\n"
      "blt 4b\n"
      "incw x14, ALL, MUL #2\n"
      "mov x13, #0\n"
      "cmp x14, x11\n"
      "mov x9, x27\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
