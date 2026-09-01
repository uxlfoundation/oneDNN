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

void sme2_interleaved_nomerge_s8q_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer)
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
      "ldr x17, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x17, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c618  // ld1w { z24.s-z27.s }, pn9.b/Z, [x16]\n"
      ".inst 0xa041c61c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c614  // ld1w { z20.s-z23.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x14, [%x[args], %[offsetof_K]]\n"
      "mov x13, #0\n"
      "mov x11, #0\n"
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "ldr w9, [%x[args], %[offsetof_N]]\n"
      "add x14, x14, #0x3\n"
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "lsr x14, x14, #0x2\n"
      "3:"  // M loop
      "ldr x27, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x26, x28\n"
      ".inst 0x25a96570  // whilelt pn8.s, x11, x9, VLx4\n"
      "tbnz x17, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      ".inst 0xa00bc290  // ld1w { z16.s-z19.s }, p8/Z, [x20, x11, LSL #2]\n"
      ".inst 0xc0902600  // addha za0.s, p1/M, p1/M, z16.s\n"
      ".inst 0xc0902621  // addha za1.s, p1/M, p1/M, z17.s\n"
      ".inst 0xc0902642  // addha za2.s, p1/M, p1/M, z18.s\n"
      ".inst 0xc0902663  // addha za3.s, p1/M, p1/M, z19.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x11\n"
      "mov x21, x13\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x9\n"
      "mov x20, x17\n"
      "csel x21, x13, x21, LT\n"
      "bfm x17, XZR, #0, #0  // bfc x17, #0, #0x1\n"
      "cmp x21, x10\n"
      "csel x17, x20, x17, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x14, #0x2\n"
      "and x20, x14, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa1408741  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa1408770  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa1418773  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa1428762  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa1438772  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0902420  // smopa za0.s, p1/M, p1/M, z1.b, z16.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0942421  // smopa za1.s, p1/M, p1/M, z1.b, z20.b\n"
      ".inst 0xa0982422  // smopa za2.s, p1/M, p1/M, z1.b, z24.b\n"
      ".inst 0xa09c2423  // smopa za3.s, p1/M, p1/M, z1.b, z28.b\n"
      ".inst 0xa1408770  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x27]\n"
      ".inst 0xa09324a0  // smopa za0.s, p1/M, p1/M, z5.b, z19.b\n"
      ".inst 0xa09724a1  // smopa za1.s, p1/M, p1/M, z5.b, z23.b\n"
      ".inst 0xa09b24a2  // smopa za2.s, p1/M, p1/M, z5.b, z27.b\n"
      ".inst 0xa09f24a3  // smopa za3.s, p1/M, p1/M, z5.b, z31.b\n"
      ".inst 0xa1418773  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa0822520  // smopa za0.s, p1/M, p1/M, z9.b, z2.b\n"
      ".inst 0xa0862521  // smopa za1.s, p1/M, p1/M, z9.b, z6.b\n"
      ".inst 0xa08a2522  // smopa za2.s, p1/M, p1/M, z9.b, z10.b\n"
      ".inst 0xa08e2523  // smopa za3.s, p1/M, p1/M, z9.b, z14.b\n"
      ".inst 0xa1428762  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa09225a0  // smopa za0.s, p1/M, p1/M, z13.b, z18.b\n"
      ".inst 0xa09625a1  // smopa za1.s, p1/M, p1/M, z13.b, z22.b\n"
      ".inst 0xa09a25a2  // smopa za2.s, p1/M, p1/M, z13.b, z26.b\n"
      ".inst 0xa09e25a3  // smopa za3.s, p1/M, p1/M, z13.b, z30.b\n"
      ".inst 0xa1408741  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa1438772  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0902420  // smopa za0.s, p1/M, p1/M, z1.b, z16.b\n"
      ".inst 0xa0942421  // smopa za1.s, p1/M, p1/M, z1.b, z20.b\n"
      ".inst 0xa0982422  // smopa za2.s, p1/M, p1/M, z1.b, z24.b\n"
      ".inst 0xa09c2423  // smopa za3.s, p1/M, p1/M, z1.b, z28.b\n"
      ".inst 0xa09324a0  // smopa za0.s, p1/M, p1/M, z5.b, z19.b\n"
      ".inst 0xa09724a1  // smopa za1.s, p1/M, p1/M, z5.b, z23.b\n"
      ".inst 0xa09b24a2  // smopa za2.s, p1/M, p1/M, z5.b, z27.b\n"
      ".inst 0xa09f24a3  // smopa za3.s, p1/M, p1/M, z5.b, z31.b\n"
      ".inst 0xa0822520  // smopa za0.s, p1/M, p1/M, z9.b, z2.b\n"
      ".inst 0xa0862521  // smopa za1.s, p1/M, p1/M, z9.b, z6.b\n"
      ".inst 0xa08a2522  // smopa za2.s, p1/M, p1/M, z9.b, z10.b\n"
      ".inst 0xa08e2523  // smopa za3.s, p1/M, p1/M, z9.b, z14.b\n"
      ".inst 0xa09225a0  // smopa za0.s, p1/M, p1/M, z13.b, z18.b\n"
      ".inst 0xa09625a1  // smopa za1.s, p1/M, p1/M, z13.b, z22.b\n"
      ".inst 0xa09a25a2  // smopa za2.s, p1/M, p1/M, z13.b, z26.b\n"
      ".inst 0xa09e25a3  // smopa za3.s, p1/M, p1/M, z13.b, z30.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z16.b }, p1/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #1\n"
      ".inst 0xa0408764  // ld1b { z4.b-z7.b }, pn9.b/Z, [x27]\n"
      "addvl x27, x27, #4\n"
      ".inst 0xa0842600  // smopa za0.s, p1/M, p1/M, z16.b, z4.b\n"
      ".inst 0xa0852601  // smopa za1.s, p1/M, p1/M, z16.b, z5.b\n"
      ".inst 0xa0862602  // smopa za2.s, p1/M, p1/M, z16.b, z6.b\n"
      ".inst 0xa0872603  // smopa za3.s, p1/M, p1/M, z16.b, z7.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "ld1w { z15.s }, p1/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      ".inst 0xc09125e0  // addva za0.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e1  // addva za1.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e2  // addva za2.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e3  // addva za3.s, p1/M, p1/M, z15.s\n"
      "tbz x17, #1, 15f\n"
      "tbz x17, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16]\n"
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      ".inst 0xa041c610  // ld1w { z16.s-z19.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa060c5fc  // st1w { z28.s-z31.s }, pn9.b, [x15]\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa061c5f8  // st1w { z24.s-z27.s }, pn9.b, [x15, #0x4, MUL VL]\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5f4  // st1w { z20.s-z23.s }, pn9.b, [x15, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5e8  // st1w { z8.s-z11.s }, pn9.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 12b\n"
      "b 22f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c5e8  // st1w { z8.s-z11.s }, pn9.b, [x15]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5e4  // st1w { z4.s-z7.s }, pn9.b, [x15, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5ec  // st1w { z12.s-z15.s }, pn9.b, [x15, #0x8, MUL VL]\n"
      ".inst 0xa063c5f0  // st1w { z16.s-z19.s }, pn9.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 14b\n"
      "b 22f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x10, x13\n"
      "ld1rw { z4.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "ld1rw { z5.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z6.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "add x25, x25, x11\n"  // C += n
      "ld1rw { z7.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "madd x25, x13, x23, x25\n"  // C += m * ldc
      "ld1rw { z0.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z1.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z2.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z3.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z8.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "ld1rw { z21.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "ld1rw { z20.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x17, #2, 16f\n"
      "ldr w22, [%x[args], %[offsetof_n_0]]\n"
      "ldr x21, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x22, x22, x11\n"
      "add x21, x21, x22, LSL #2\n"
      "add x20, x20, x22, LSL #2\n"
      ".inst 0xa040c2a4  // ld1w { z4.s-z7.s }, p8/Z, [x21]\n"
      ".inst 0xa040c280  // ld1w { z0.s-z3.s }, p8/Z, [x20]\n"
      "16:"  // Store to output array: Load per-channel parameters: End
      "cntw x20\n"
      "whilelt p0.b, x11, x9\n"
      "cmp x24, x20\n"
      "mov x12, #0\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x1\n"
      "and x20, x20, #0x1\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860010  // mova { z16.s-z17.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc086005e  // mova { z30.s-z31.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc086009a  // mova { z26.s-z27.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600cc  // mova { z12.s-z13.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1a4a410  // sqdmulh { z16.s-z17.s }, { z16.s-z17.s }, z4.s\n"
      ".inst 0xc1a5a41e  // sqdmulh { z30.s-z31.s }, { z30.s-z31.s }, z5.s\n"
      "add x12, x12, #0x2\n"
      ".inst 0xc1a6a41a  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z6.s\n"
      "cmp x12, x21, LSL #1\n"
      ".inst 0xc1a7a40c  // sqdmulh { z12.s-z13.s }, { z12.s-z13.s }, z7.s\n"
      ".inst 0xc1a0a230  // srshl { z16.s-z17.s }, { z16.s-z17.s }, z0.s\n"
      ".inst 0xc1a1a23e  // srshl { z30.s-z31.s }, { z30.s-z31.s }, z1.s\n"
      ".inst 0xc1a2a23a  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z2.s\n"
      ".inst 0xc1a3a22c  // srshl { z12.s-z13.s }, { z12.s-z13.s }, z3.s\n"
      ".inst 0xc1a8a310  // add { z16.s-z17.s }, { z16.s-z17.s }, z8.s\n"
      ".inst 0xc1a8a31e  // add { z30.s-z31.s }, { z30.s-z31.s }, z8.s\n"
      ".inst 0xc1a8a31a  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s\n"
      ".inst 0xc1a8a30c  // add { z12.s-z13.s }, { z12.s-z13.s }, z8.s\n"
      ".inst 0xc1b4c6b0  // sclamp { z16.s-z17.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6be  // sclamp { z30.s-z31.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ba  // sclamp { z26.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ac  // sclamp { z12.s-z13.s }, z21.s, z20.s\n"
      "uzp1 z19.b, z16.b, z30.b\n"
      "uzp1 z18.b, z17.b, z31.b\n"
      "uzp1 z17.b, z26.b, z12.b\n"
      "uzp1 z16.b, z27.b, z13.b\n"
      "uzp1 z17.b, z19.b, z17.b\n"
      "uzp1 z16.b, z18.b, z16.b\n"
      "st1b { z17.b }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z16.b }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 17b\n"
      "18:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc086000a  // mova { z10.s-z11.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc086005a  // mova { z26.s-z27.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc086008e  // mova { z14.s-z15.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600d6  // mova { z22.s-z23.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1a4a40a  // sqdmulh { z10.s-z11.s }, { z10.s-z11.s }, z4.s\n"
      ".inst 0xc1a5a41a  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z5.s\n"
      ".inst 0xc1a6a40e  // sqdmulh { z14.s-z15.s }, { z14.s-z15.s }, z6.s\n"
      ".inst 0xc1a7a416  // sqdmulh { z22.s-z23.s }, { z22.s-z23.s }, z7.s\n"
      ".inst 0xc1a0a22a  // srshl { z10.s-z11.s }, { z10.s-z11.s }, z0.s\n"
      ".inst 0xc1a1a23a  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z1.s\n"
      ".inst 0xc1a2a22e  // srshl { z14.s-z15.s }, { z14.s-z15.s }, z2.s\n"
      ".inst 0xc1a3a236  // srshl { z22.s-z23.s }, { z22.s-z23.s }, z3.s\n"
      ".inst 0xc1a8a30a  // add { z10.s-z11.s }, { z10.s-z11.s }, z8.s\n"
      ".inst 0xc1a8a31a  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s\n"
      ".inst 0xc1a8a30e  // add { z14.s-z15.s }, { z14.s-z15.s }, z8.s\n"
      ".inst 0xc1a8a316  // add { z22.s-z23.s }, { z22.s-z23.s }, z8.s\n"
      ".inst 0xc1b4c6aa  // sclamp { z10.s-z11.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ba  // sclamp { z26.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ae  // sclamp { z14.s-z15.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6b6  // sclamp { z22.s-z23.s }, z21.s, z20.s\n"
      "uzp1 z17.b, z10.b, z26.b\n"
      "uzp1 z16.b, z14.b, z22.b\n"
      "uzp1 z16.b, z17.b, z16.b\n"
      "st1b { z16.b }, p0, [x25]\n"
      "19:"  // Store to output array: Accumulator row 0 oddments: End
      "tbz x17, #0, 22f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "21:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c618  // ld1w { z24.s-z27.s }, pn9.b/Z, [x16]\n"
      ".inst 0xa041c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 21b\n"
      "22:"  // End block
      "incw x11, ALL, MUL #4\n"
      "cmp x11, x9\n"
      "blt 4b\n"
      "incw x13\n"
      "mov x11, #0\n"
      "cmp x13, x10\n"
      "mov x28, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
