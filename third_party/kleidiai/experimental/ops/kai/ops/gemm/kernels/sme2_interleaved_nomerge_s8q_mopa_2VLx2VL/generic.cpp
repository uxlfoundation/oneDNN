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

void sme2_interleaved_nomerge_s8q_mopa_2VLx2VL(const int8_t *const A, const int8_t *const B, int8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      int8_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const Requantize32 &rq,
      const int n_0,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(int8_t)),
        M(M), N(N), K(K),
        min(0), max(0),
        bias(bias), n_0(n_0),
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
      if (rq.per_channel_requant)
      {
        flags |= 1 << 2;  // PER_CHANNEL_QUANTISATION
      }
      }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    int8_t *const C;
    const long ldcb;
    const long M, N, K;
    int32_t min;
    int32_t max;

    const int32_t *const bias;
    const int n_0;


    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, rq, n_0, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x8, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x17, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x8, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c628  // ld1w { z8.s-z11.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c634  // ld1w { z20.s-z23.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840681  // mova za1h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
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
      ".inst 0xa00d4286  // ld1w { z6.s-z7.s }, p8/Z, [x20, x13, LSL #2]\n"
      ".inst 0xc09024c0  // addha za0.s, p1/M, p1/M, z6.s\n"
      ".inst 0xc09024e1  // addha za1.s, p1/M, p1/M, z7.s\n"
      ".inst 0xc09024c2  // addha za2.s, p1/M, p1/M, z6.s\n"
      ".inst 0xc09024e3  // addha za3.s, p1/M, p1/M, z7.s\n"
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
      ".inst 0xa1408760  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa0418774  // ld1b { z20.b-z23.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa0408790  // ld1b { z16.b-z19.b }, pn9.b/Z, [x28]\n"
      ".inst 0xa1418783  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0902400  // smopa za0.s, p1/M, p1/M, z0.b, z16.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0912401  // smopa za1.s, p1/M, p1/M, z0.b, z17.b\n"
      ".inst 0xa0902482  // smopa za2.s, p1/M, p1/M, z4.b, z16.b\n"
      ".inst 0xa0912483  // smopa za3.s, p1/M, p1/M, z4.b, z17.b\n"
      ".inst 0xa0922500  // smopa za0.s, p1/M, p1/M, z8.b, z18.b\n"
      ".inst 0xa0932501  // smopa za1.s, p1/M, p1/M, z8.b, z19.b\n"
      ".inst 0xa0922582  // smopa za2.s, p1/M, p1/M, z12.b, z18.b\n"
      ".inst 0xa0932583  // smopa za3.s, p1/M, p1/M, z12.b, z19.b\n"
      ".inst 0xa1408760  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa0832680  // smopa za0.s, p1/M, p1/M, z20.b, z3.b\n"
      ".inst 0xa0408790  // ld1b { z16.b-z19.b }, pn9.b/Z, [x28]\n"
      ".inst 0xa0872681  // smopa za1.s, p1/M, p1/M, z20.b, z7.b\n"
      ".inst 0xa08326a2  // smopa za2.s, p1/M, p1/M, z21.b, z3.b\n"
      ".inst 0xa08726a3  // smopa za3.s, p1/M, p1/M, z21.b, z7.b\n"
      ".inst 0xa08b26c0  // smopa za0.s, p1/M, p1/M, z22.b, z11.b\n"
      ".inst 0xa08f26c1  // smopa za1.s, p1/M, p1/M, z22.b, z15.b\n"
      ".inst 0xa08b26e2  // smopa za2.s, p1/M, p1/M, z23.b, z11.b\n"
      ".inst 0xa08f26e3  // smopa za3.s, p1/M, p1/M, z23.b, z15.b\n"
      ".inst 0xa0418774  // ld1b { z20.b-z23.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa1418783  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x28, #0x4, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0902400  // smopa za0.s, p1/M, p1/M, z0.b, z16.b\n"
      ".inst 0xa0912401  // smopa za1.s, p1/M, p1/M, z0.b, z17.b\n"
      ".inst 0xa0902482  // smopa za2.s, p1/M, p1/M, z4.b, z16.b\n"
      ".inst 0xa0912483  // smopa za3.s, p1/M, p1/M, z4.b, z17.b\n"
      ".inst 0xa0922500  // smopa za0.s, p1/M, p1/M, z8.b, z18.b\n"
      ".inst 0xa0932501  // smopa za1.s, p1/M, p1/M, z8.b, z19.b\n"
      ".inst 0xa0922582  // smopa za2.s, p1/M, p1/M, z12.b, z18.b\n"
      ".inst 0xa0932583  // smopa za3.s, p1/M, p1/M, z12.b, z19.b\n"
      ".inst 0xa0832680  // smopa za0.s, p1/M, p1/M, z20.b, z3.b\n"
      ".inst 0xa0872681  // smopa za1.s, p1/M, p1/M, z20.b, z7.b\n"
      ".inst 0xa08326a2  // smopa za2.s, p1/M, p1/M, z21.b, z3.b\n"
      ".inst 0xa08726a3  // smopa za3.s, p1/M, p1/M, z21.b, z7.b\n"
      ".inst 0xa08b26c0  // smopa za0.s, p1/M, p1/M, z22.b, z11.b\n"
      ".inst 0xa08f26c1  // smopa za1.s, p1/M, p1/M, z22.b, z15.b\n"
      ".inst 0xa08b26e2  // smopa za2.s, p1/M, p1/M, z23.b, z11.b\n"
      ".inst 0xa08f26e3  // smopa za3.s, p1/M, p1/M, z23.b, z15.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa040077c  // ld1b { z28.b-z29.b }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      ".inst 0xa0400794  // ld1b { z20.b-z21.b }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #2\n"
      ".inst 0xa0942780  // smopa za0.s, p1/M, p1/M, z28.b, z20.b\n"
      ".inst 0xa0952781  // smopa za1.s, p1/M, p1/M, z28.b, z21.b\n"
      ".inst 0xa09427a2  // smopa za2.s, p1/M, p1/M, z29.b, z20.b\n"
      ".inst 0xa09527a3  // smopa za3.s, p1/M, p1/M, z29.b, z21.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      ".inst 0xa040477a  // ld1w { z26.s-z27.s }, pn9.b/Z, [x27]\n"
      "addvl x27, x27, #2\n"
      ".inst 0xc0912740  // addva za0.s, p1/M, p1/M, z26.s\n"
      ".inst 0xc0912741  // addva za1.s, p1/M, p1/M, z26.s\n"
      ".inst 0xc0912762  // addva za2.s, p1/M, p1/M, z27.s\n"
      ".inst 0xc0912763  // addva za3.s, p1/M, p1/M, z27.s\n"
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17]\n"
      ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xa041c63c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c638  // ld1w { z24.s-z27.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa060c614  // st1w { z20.s-z23.s }, pn9.b, [x16]\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa061c610  // st1w { z16.s-z19.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0840703  // mova za3h.s[x12], { z24.s-z27.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c60c  // st1w { z12.s-z15.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c608  // st1w { z8.s-z11.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 12b\n"
      "b 25f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860460  // mova { z0.s-z3.s }, za3h.s[x12]\n"
      ".inst 0xa060c604  // st1w { z4.s-z7.s }, pn9.b, [x16]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c614  // st1w { z20.s-z23.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c610  // st1w { z16.s-z19.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      ".inst 0xa063c600  // st1w { z0.s-z3.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 14b\n"
      "b 25f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "ld1rw { z2.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "ld1rw { z10.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z3.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "add x26, x26, x13\n"  // C += n
      "ld1rw { z11.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "madd x26, x14, x24, x26\n"  // C += m * ldc
      "ld1rw { z0.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "ld1rw { z21.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "ld1rw { z20.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x8, #2, 16f\n"
      "ldr w22, [%x[args], %[offsetof_n_0]]\n"
      "ldr x21, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x22, x22, x13\n"
      "add x21, x21, x22, LSL #2\n"
      "add x20, x20, x22, LSL #2\n"
      ".inst 0xa14042a2  // ld1w { z2.s, z10.s }, p8/Z, [x21]\n"
      ".inst 0xa1404283  // ld1w { z3.s, z11.s }, p8/Z, [x20]\n"
      "16:"  // Store to output array: Load per-channel parameters: End
      "cntw x23\n"
      "whilelt p0.h, x13, x10\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860418  // mova { z24.s-z27.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xc1a2ac18  // sqdmulh { z24.s-z27.s }, { z24.s-z27.s }, z2.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1aaac1c  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a3aa38  // srshl { z24.s-z27.s }, { z24.s-z27.s }, z3.s\n"
      ".inst 0xc1abaa3c  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s\n"
      ".inst 0xc1a0ab18  // add { z24.s-z27.s }, { z24.s-z27.s }, z0.s\n"
      ".inst 0xc1a0ab1c  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s\n"
      ".inst 0xc1b4ceb8  // sclamp { z24.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cebc  // sclamp { z28.s-z31.s }, z21.s, z20.s\n"
      "uzp1 z19.h, z24.h, z28.h\n"
      "uzp1 z18.h, z25.h, z29.h\n"
      "uzp1 z17.h, z26.h, z30.h\n"
      "uzp1 z16.h, z27.h, z31.h\n"
      "st1b { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 17b\n"
      "18:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xc1a2ac04  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1aaac1c  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s\n"
      ".inst 0xc1a3aa24  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z3.s\n"
      ".inst 0xc1abaa3c  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s\n"
      ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
      ".inst 0xc1a0ab1c  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s\n"
      ".inst 0xc1b4cea4  // sclamp { z4.s-z7.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cebc  // sclamp { z28.s-z31.s }, z21.s, z20.s\n"
      "uzp1 z16.h, z4.h, z28.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"
      "subs x20, x20, #0x1\n"
      "uzp1 z16.h, z5.h, z29.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"
      "uzp1 z16.h, z6.h, z30.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "19:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 23f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xc1a2ac0c  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1aaac1c  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a3aa2c  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z3.s\n"
      ".inst 0xc1abaa3c  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s\n"
      ".inst 0xc1a0ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s\n"
      ".inst 0xc1a0ab1c  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s\n"
      ".inst 0xc1b4ceac  // sclamp { z12.s-z15.s }, z21.s, z20.s\n"
      ".inst 0xc1b4cebc  // sclamp { z28.s-z31.s }, z21.s, z20.s\n"
      "uzp1 z19.h, z12.h, z28.h\n"
      "uzp1 z18.h, z13.h, z29.h\n"
      "uzp1 z17.h, z14.h, z30.h\n"
      "uzp1 z16.h, z15.h, z31.h\n"
      "st1b { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 20b\n"
      "21:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 22f\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xc1a2ac04  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1aaac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z10.s\n"
      ".inst 0xc1a3aa24  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z3.s\n"
      ".inst 0xc1abaa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z11.s\n"
      ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
      ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
      ".inst 0xc1b4cea4  // sclamp { z4.s-z7.s }, z21.s, z20.s\n"
      ".inst 0xc1b4ceb0  // sclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "uzp1 z16.h, z4.h, z16.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"
      "subs x20, x20, #0x1\n"
      "uzp1 z16.h, z5.h, z17.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"
      "uzp1 z16.h, z6.h, z18.h\n"
      "st1b { z16.h }, p0, [x26]\n"
      "22:"  // Store to output array: Accumulator row 1 oddments: End
      "23:"  // Store to output array: End
      "tbz x8, #0, 25f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "24:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c63c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 24b\n"
      "25:"  // End block
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
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
