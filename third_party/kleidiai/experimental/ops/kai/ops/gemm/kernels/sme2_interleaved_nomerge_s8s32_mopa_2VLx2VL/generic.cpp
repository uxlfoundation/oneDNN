//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include "kai/ops/gemm/kai_ops.hpp"

#include <cstdint>
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

void sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      int32_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,

      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(int32_t)),
        M(M), N(N), K(K),

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
      }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    int32_t *const C;
    const long ldcb;
    const long M, N, K;

    const int32_t *const bias;


    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, accumulate, accumulator_buffer);

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
      ".inst 0xa041c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x15, [%x[args], %[offsetof_K]]\n"
      "mov x14, #0\n"
      "mov x13, #0\n"
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "ldr w10, [%x[args], %[offsetof_N]]\n"
      "add x15, x15, #0x3\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "lsr x15, x15, #0x2\n"
      "3:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x27, x9\n"
      ".inst 0x25aa45b0  // whilelt pn8.s, x13, x10, VLx2\n"
      "tbnz x8, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      ".inst 0xa10d4296  // ld1w { z22.s, z30.s }, p8/Z, [x20, x13, LSL #2]\n"
      ".inst 0xc09002c0  // addha za0.s, p0/M, p0/M, z22.s\n"
      ".inst 0xc09003c1  // addha za1.s, p0/M, p0/M, z30.s\n"
      ".inst 0xc09002c2  // addha za2.s, p0/M, p0/M, z22.s\n"
      ".inst 0xc09003c3  // addha za3.s, p0/M, p0/M, z30.s\n"
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
      ".inst 0xa0408778  // ld1b { z24.b-z27.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa0418774  // ld1b { z20.b-z23.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa0408788  // ld1b { z8.b-z11.b }, pn9.b/Z, [x28]\n"
      ".inst 0xa0418790  // ld1b { z16.b-z19.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0880300  // smopa za0.s, p0/M, p0/M, z24.b, z8.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0890301  // smopa za1.s, p0/M, p0/M, z24.b, z9.b\n"
      ".inst 0xa0880322  // smopa za2.s, p0/M, p0/M, z25.b, z8.b\n"
      ".inst 0xa0890323  // smopa za3.s, p0/M, p0/M, z25.b, z9.b\n"
      ".inst 0xa08a0340  // smopa za0.s, p0/M, p0/M, z26.b, z10.b\n"
      ".inst 0xa08b0341  // smopa za1.s, p0/M, p0/M, z26.b, z11.b\n"
      ".inst 0xa08a0362  // smopa za2.s, p0/M, p0/M, z27.b, z10.b\n"
      ".inst 0xa08b0363  // smopa za3.s, p0/M, p0/M, z27.b, z11.b\n"
      ".inst 0xa0408778  // ld1b { z24.b-z27.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa0900280  // smopa za0.s, p0/M, p0/M, z20.b, z16.b\n"
      ".inst 0xa0408788  // ld1b { z8.b-z11.b }, pn9.b/Z, [x28]\n"
      ".inst 0xa0910281  // smopa za1.s, p0/M, p0/M, z20.b, z17.b\n"
      ".inst 0xa09002a2  // smopa za2.s, p0/M, p0/M, z21.b, z16.b\n"
      ".inst 0xa09102a3  // smopa za3.s, p0/M, p0/M, z21.b, z17.b\n"
      ".inst 0xa09202c0  // smopa za0.s, p0/M, p0/M, z22.b, z18.b\n"
      ".inst 0xa09302c1  // smopa za1.s, p0/M, p0/M, z22.b, z19.b\n"
      ".inst 0xa09202e2  // smopa za2.s, p0/M, p0/M, z23.b, z18.b\n"
      ".inst 0xa09302e3  // smopa za3.s, p0/M, p0/M, z23.b, z19.b\n"
      ".inst 0xa0418774  // ld1b { z20.b-z23.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa0418790  // ld1b { z16.b-z19.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0880300  // smopa za0.s, p0/M, p0/M, z24.b, z8.b\n"
      ".inst 0xa0890301  // smopa za1.s, p0/M, p0/M, z24.b, z9.b\n"
      ".inst 0xa0880322  // smopa za2.s, p0/M, p0/M, z25.b, z8.b\n"
      ".inst 0xa0890323  // smopa za3.s, p0/M, p0/M, z25.b, z9.b\n"
      ".inst 0xa08a0340  // smopa za0.s, p0/M, p0/M, z26.b, z10.b\n"
      ".inst 0xa08b0341  // smopa za1.s, p0/M, p0/M, z26.b, z11.b\n"
      ".inst 0xa08a0362  // smopa za2.s, p0/M, p0/M, z27.b, z10.b\n"
      ".inst 0xa08b0363  // smopa za3.s, p0/M, p0/M, z27.b, z11.b\n"
      ".inst 0xa0900280  // smopa za0.s, p0/M, p0/M, z20.b, z16.b\n"
      ".inst 0xa0910281  // smopa za1.s, p0/M, p0/M, z20.b, z17.b\n"
      ".inst 0xa09002a2  // smopa za2.s, p0/M, p0/M, z21.b, z16.b\n"
      ".inst 0xa09102a3  // smopa za3.s, p0/M, p0/M, z21.b, z17.b\n"
      ".inst 0xa09202c0  // smopa za0.s, p0/M, p0/M, z22.b, z18.b\n"
      ".inst 0xa09302c1  // smopa za1.s, p0/M, p0/M, z22.b, z19.b\n"
      ".inst 0xa09202e2  // smopa za2.s, p0/M, p0/M, z23.b, z18.b\n"
      ".inst 0xa09302e3  // smopa za3.s, p0/M, p0/M, z23.b, z19.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa1400777  // ld1b { z23.b, z31.b }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      ".inst 0xa1400787  // ld1b { z7.b, z15.b }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #2\n"
      ".inst 0xa08702e0  // smopa za0.s, p0/M, p0/M, z23.b, z7.b\n"
      ".inst 0xa08f02e1  // smopa za1.s, p0/M, p0/M, z23.b, z15.b\n"
      ".inst 0xa08703e2  // smopa za2.s, p0/M, p0/M, z31.b, z7.b\n"
      ".inst 0xa08f03e3  // smopa za3.s, p0/M, p0/M, z31.b, z15.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c638  // ld1w { z24.s-z27.s }, pn9.b/Z, [x17]\n"
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xa041c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa042c634  // ld1w { z20.s-z23.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c630  // ld1w { z16.s-z19.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa060c61c  // st1w { z28.s-z31.s }, pn9.b, [x16]\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xa061c600  // st1w { z0.s-z3.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c608  // st1w { z8.s-z11.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c60c  // st1w { z12.s-z15.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 12b\n"
      "b 24f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xc0860440  // mova { z0.s-z3.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa060c604  // st1w { z4.s-z7.s }, pn9.b, [x16]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c608  // st1w { z8.s-z11.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c600  // st1w { z0.s-z3.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      ".inst 0xa063c618  // st1w { z24.s-z27.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 14b\n"
      "b 24f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "cntw x24\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x22, x25, x24, LT\n"
      "add x26, x26, x13, LSL #2\n"  // C += n
      "lsr x21, x22, #0x2\n"
      "madd x26, x14, x23, x26\n"  // C += m * ldc
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      ".inst 0xa1604353  // st1w { z19.s, z27.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa1604344  // st1w { z4.s, z12.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604345  // st1w { z5.s, z13.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "beq 18f\n"
      ".inst 0xa1604346  // st1w { z6.s, z14.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 22f\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x20, x25, x24, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      ".inst 0xa1604353  // st1w { z19.s, z27.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa1604350  // st1w { z16.s, z24.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1604351  // st1w { z17.s, z25.s }, p8, [x26]\n"
      "add x26, x26, x23\n"
      "beq 21f\n"
      ".inst 0xa1604352  // st1w { z18.s, z26.s }, p8, [x26]\n"
      "21:"  // Store to output array: Accumulator row 1 oddments: End
      "22:"  // Store to output array: End
      "tbz x8, #0, 24f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "23:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c628  // ld1w { z8.s-z11.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 23b\n"
      "24:"  // End block
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
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
