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

void sme_interleaved_nomerge_s8s32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
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
      "ldr x5, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p4.b\n"
      "ldr x6, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x7, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x5, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "1:"  // Initial accumulator load from buffer: Loop
      "addvl x22, x6, #4\n"
      "addvl x21, x6, #8\n"
      ".inst 0xe09f10c0  // ld1w { za0h.s[x12] }, p4/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f12c4  // ld1w { za1h.s[x12] }, p4/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f12a8  // ld1w { za2h.s[x12] }, p4/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f128c  // ld1w { za3h.s[x12] }, p4/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09910c1  // ld1w { za0h.s[x12, #1] }, p4/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe09912c5  // ld1w { za1h.s[x12, #1] }, p4/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe09912a9  // ld1w { za2h.s[x12, #1] }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe099128d  // ld1w { za3h.s[x12, #1] }, p4/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09810c2  // ld1w { za0h.s[x12, #2] }, p4/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe09812c6  // ld1w { za1h.s[x12, #2] }, p4/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe09812aa  // ld1w { za2h.s[x12, #2] }, p4/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe098128e  // ld1w { za3h.s[x12, #2] }, p4/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09710c3  // ld1w { za0h.s[x12, #3] }, p4/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe09712c7  // ld1w { za1h.s[x12, #3] }, p4/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe09712ab  // ld1w { za2h.s[x12, #3] }, p4/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe097128f  // ld1w { za3h.s[x12, #3] }, p4/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x8, [%x[args], %[offsetof_K]]\n"
      "mov x17, #0\n"
      "mov x16, #0\n"
      "ldr w15, [%x[args], %[offsetof_M]]\n"
      "ldr w14, [%x[args], %[offsetof_N]]\n"
      "add x8, x8, #0x3\n"
      "ldr x13, [%x[args], %[offsetof_A]]\n"
      "lsr x8, x8, #0x2\n"
      "3:"  // M loop
      "ldr x11, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x20, x16\n"
      "mov x10, x13\n"
      "whilelt p3.s, x20, x14\n"
      "incw x20\n"
      "whilelt p2.s, x20, x14\n"
      "incw x20\n"
      "whilelt p1.s, x20, x14\n"
      "incw x20\n"
      "whilelt p0.s, x20, x14\n"
      "tbnz x5, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "add x20, x20, x16, LSL #2\n"
      "ld1w { z19.s }, p3/Z, [x20]\n"
      "ld1w { z18.s }, p2/Z, [x20, #1, MUL VL]\n"
      "ld1w { z17.s }, p1/Z, [x20, #2, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x20, #3, MUL VL]\n"
      ".inst 0xc0909260  // addha za0.s, p4/M, p4/M, z19.s\n"
      ".inst 0xc0909241  // addha za1.s, p4/M, p4/M, z18.s\n"
      ".inst 0xc0909222  // addha za2.s, p4/M, p4/M, z17.s\n"
      ".inst 0xc0909203  // addha za3.s, p4/M, p4/M, z16.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x16\n"
      "mov x21, x17\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x14\n"
      "mov x20, x5\n"
      "csel x21, x17, x21, LT\n"
      "bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1\n"
      "cmp x21, x15\n"
      "csel x5, x20, x5, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x23, x8, #0x2\n"
      "and x22, x8, #0x3\n"
      "cbz x23, 9f\n"
      "addvl x21, x11, #8\n"
      "addvl x20, x11, #12\n"
      "ld1b { z3.b }, p4/Z, [x10]\n"
      "subs x23, x23, #0x1\n"
      "ld1b { z2.b }, p4/Z, [x10, #1, MUL VL]\n"
      "ld1b { z1.b }, p4/Z, [x10, #2, MUL VL]\n"
      "ld1b { z0.b }, p4/Z, [x10, #3, MUL VL]\n"
      "addvl x10, x10, #4\n"
      "ld1b { z31.b }, p4/Z, [x11]\n"
      "ld1b { z30.b }, p4/Z, [x11, #1, MUL VL]\n"
      "ld1b { z29.b }, p4/Z, [x11, #2, MUL VL]\n"
      "ld1b { z28.b }, p4/Z, [x11, #3, MUL VL]\n"
      "ld1b { z27.b }, p4/Z, [x11, #4, MUL VL]\n"
      "ld1b { z26.b }, p4/Z, [x11, #5, MUL VL]\n"
      "ld1b { z25.b }, p4/Z, [x11, #6, MUL VL]\n"
      "ld1b { z24.b }, p4/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "ld1b { z23.b }, p4/Z, [x21]\n"
      "ld1b { z22.b }, p4/Z, [x21, #1, MUL VL]\n"
      "ld1b { z21.b }, p4/Z, [x21, #2, MUL VL]\n"
      "ld1b { z20.b }, p4/Z, [x21, #3, MUL VL]\n"
      "ld1b { z19.b }, p4/Z, [x20]\n"
      "ld1b { z18.b }, p4/Z, [x20, #1, MUL VL]\n"
      "ld1b { z17.b }, p4/Z, [x20, #2, MUL VL]\n"
      "ld1b { z16.b }, p4/Z, [x20, #3, MUL VL]\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa09f9060  // smopa za0.s, p4/M, p4/M, z3.b, z31.b\n"
      "addvl x21, x11, #8\n"
      "addvl x20, x11, #12\n"
      "ld1b { z31.b }, p4/Z, [x11]\n"
      ".inst 0xa09e9061  // smopa za1.s, p4/M, p4/M, z3.b, z30.b\n"
      "subs x23, x23, #0x1\n"
      "ld1b { z30.b }, p4/Z, [x11, #1, MUL VL]\n"
      ".inst 0xa09d9062  // smopa za2.s, p4/M, p4/M, z3.b, z29.b\n"
      "ld1b { z29.b }, p4/Z, [x11, #2, MUL VL]\n"
      ".inst 0xa09c9063  // smopa za3.s, p4/M, p4/M, z3.b, z28.b\n"
      "ld1b { z3.b }, p4/Z, [x10]\n"
      ".inst 0xa09b9040  // smopa za0.s, p4/M, p4/M, z2.b, z27.b\n"
      "ld1b { z28.b }, p4/Z, [x11, #3, MUL VL]\n"
      ".inst 0xa09a9041  // smopa za1.s, p4/M, p4/M, z2.b, z26.b\n"
      "ld1b { z27.b }, p4/Z, [x11, #4, MUL VL]\n"
      ".inst 0xa0999042  // smopa za2.s, p4/M, p4/M, z2.b, z25.b\n"
      "ld1b { z26.b }, p4/Z, [x11, #5, MUL VL]\n"
      ".inst 0xa0989043  // smopa za3.s, p4/M, p4/M, z2.b, z24.b\n"
      "ld1b { z2.b }, p4/Z, [x10, #1, MUL VL]\n"
      ".inst 0xa0979020  // smopa za0.s, p4/M, p4/M, z1.b, z23.b\n"
      "ld1b { z25.b }, p4/Z, [x11, #6, MUL VL]\n"
      ".inst 0xa0969021  // smopa za1.s, p4/M, p4/M, z1.b, z22.b\n"
      "ld1b { z24.b }, p4/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #16\n"
      ".inst 0xa0959022  // smopa za2.s, p4/M, p4/M, z1.b, z21.b\n"
      "ld1b { z23.b }, p4/Z, [x21]\n"
      ".inst 0xa0949023  // smopa za3.s, p4/M, p4/M, z1.b, z20.b\n"
      "ld1b { z1.b }, p4/Z, [x10, #2, MUL VL]\n"
      ".inst 0xa0939000  // smopa za0.s, p4/M, p4/M, z0.b, z19.b\n"
      "ld1b { z22.b }, p4/Z, [x21, #1, MUL VL]\n"
      ".inst 0xa0929001  // smopa za1.s, p4/M, p4/M, z0.b, z18.b\n"
      "ld1b { z21.b }, p4/Z, [x21, #2, MUL VL]\n"
      ".inst 0xa0919002  // smopa za2.s, p4/M, p4/M, z0.b, z17.b\n"
      "ld1b { z20.b }, p4/Z, [x21, #3, MUL VL]\n"
      ".inst 0xa0909003  // smopa za3.s, p4/M, p4/M, z0.b, z16.b\n"
      "ld1b { z0.b }, p4/Z, [x10, #3, MUL VL]\n"
      "addvl x10, x10, #4\n"
      "ld1b { z19.b }, p4/Z, [x20]\n"
      "ld1b { z18.b }, p4/Z, [x20, #1, MUL VL]\n"
      "ld1b { z17.b }, p4/Z, [x20, #2, MUL VL]\n"
      "ld1b { z16.b }, p4/Z, [x20, #3, MUL VL]\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa09f9060  // smopa za0.s, p4/M, p4/M, z3.b, z31.b\n"
      ".inst 0xa09e9061  // smopa za1.s, p4/M, p4/M, z3.b, z30.b\n"
      ".inst 0xa09d9062  // smopa za2.s, p4/M, p4/M, z3.b, z29.b\n"
      ".inst 0xa09c9063  // smopa za3.s, p4/M, p4/M, z3.b, z28.b\n"
      ".inst 0xa09b9040  // smopa za0.s, p4/M, p4/M, z2.b, z27.b\n"
      ".inst 0xa09a9041  // smopa za1.s, p4/M, p4/M, z2.b, z26.b\n"
      ".inst 0xa0999042  // smopa za2.s, p4/M, p4/M, z2.b, z25.b\n"
      ".inst 0xa0989043  // smopa za3.s, p4/M, p4/M, z2.b, z24.b\n"
      ".inst 0xa0979020  // smopa za0.s, p4/M, p4/M, z1.b, z23.b\n"
      ".inst 0xa0969021  // smopa za1.s, p4/M, p4/M, z1.b, z22.b\n"
      ".inst 0xa0959022  // smopa za2.s, p4/M, p4/M, z1.b, z21.b\n"
      ".inst 0xa0949023  // smopa za3.s, p4/M, p4/M, z1.b, z20.b\n"
      ".inst 0xa0939000  // smopa za0.s, p4/M, p4/M, z0.b, z19.b\n"
      ".inst 0xa0929001  // smopa za1.s, p4/M, p4/M, z0.b, z18.b\n"
      ".inst 0xa0919002  // smopa za2.s, p4/M, p4/M, z0.b, z17.b\n"
      ".inst 0xa0909003  // smopa za3.s, p4/M, p4/M, z0.b, z16.b\n"
      "9:"  // K oddments
      "cbz x22, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z20.b }, p4/Z, [x10]\n"
      "subs x22, x22, #0x1\n"
      "addvl x10, x10, #1\n"
      "ld1b { z19.b }, p4/Z, [x11]\n"
      "ld1b { z18.b }, p4/Z, [x11, #1, MUL VL]\n"
      "ld1b { z17.b }, p4/Z, [x11, #2, MUL VL]\n"
      "ld1b { z16.b }, p4/Z, [x11, #3, MUL VL]\n"
      "addvl x11, x11, #4\n"
      ".inst 0xa0939280  // smopa za0.s, p4/M, p4/M, z20.b, z19.b\n"
      ".inst 0xa0929281  // smopa za1.s, p4/M, p4/M, z20.b, z18.b\n"
      ".inst 0xa0919282  // smopa za2.s, p4/M, p4/M, z20.b, z17.b\n"
      ".inst 0xa0909283  // smopa za3.s, p4/M, p4/M, z20.b, z16.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x5, #1, 15f\n"
      "tbz x5, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x9\n"
      "cntw x28\n"
      "cntw x27, ALL, MUL #2\n"
      "cntw x26, ALL, MUL #3\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xe0bf10e0  // st1w { za0h.s[x12] }, p4/Z, [x7, XZR, LSL #2]\n"
      ".inst 0xe09f10c0  // ld1w { za0h.s[x12] }, p4/Z, [x6, XZR, LSL #2]\n"
      "addvl x25, x7, #4\n"
      "addvl x24, x6, #4\n"
      ".inst 0xe0bc10e1  // st1w { za0h.s[x12, #1] }, p4/Z, [x7, x28, LSL #2]\n"
      ".inst 0xe09c10c1  // ld1w { za0h.s[x12, #1] }, p4/Z, [x6, x28, LSL #2]\n"
      "addvl x23, x7, #8\n"
      "addvl x22, x6, #8\n"
      ".inst 0xe0bb10e2  // st1w { za0h.s[x12, #2] }, p4/Z, [x7, x27, LSL #2]\n"
      ".inst 0xe09b10c2  // ld1w { za0h.s[x12, #2] }, p4/Z, [x6, x27, LSL #2]\n"
      "addvl x21, x7, #12\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe0ba10e3  // st1w { za0h.s[x12, #3] }, p4/Z, [x7, x26, LSL #2]\n"
      ".inst 0xe09a10c3  // ld1w { za0h.s[x12, #3] }, p4/Z, [x6, x26, LSL #2]\n"
      "addvl x7, x7, #16\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe0bf1324  // st1w { za1h.s[x12] }, p4/Z, [x25, XZR, LSL #2]\n"
      ".inst 0xe09f1304  // ld1w { za1h.s[x12] }, p4/Z, [x24, XZR, LSL #2]\n"
      ".inst 0xe0bc1325  // st1w { za1h.s[x12, #1] }, p4/Z, [x25, x28, LSL #2]\n"
      ".inst 0xe09c1305  // ld1w { za1h.s[x12, #1] }, p4/Z, [x24, x28, LSL #2]\n"
      ".inst 0xe0bb1326  // st1w { za1h.s[x12, #2] }, p4/Z, [x25, x27, LSL #2]\n"
      ".inst 0xe09b1306  // ld1w { za1h.s[x12, #2] }, p4/Z, [x24, x27, LSL #2]\n"
      ".inst 0xe0ba1327  // st1w { za1h.s[x12, #3] }, p4/Z, [x25, x26, LSL #2]\n"
      ".inst 0xe09a1307  // ld1w { za1h.s[x12, #3] }, p4/Z, [x24, x26, LSL #2]\n"
      ".inst 0xe0bf12e8  // st1w { za2h.s[x12] }, p4/Z, [x23, XZR, LSL #2]\n"
      ".inst 0xe09f12c8  // ld1w { za2h.s[x12] }, p4/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0bc12e9  // st1w { za2h.s[x12, #1] }, p4/Z, [x23, x28, LSL #2]\n"
      ".inst 0xe09c12c9  // ld1w { za2h.s[x12, #1] }, p4/Z, [x22, x28, LSL #2]\n"
      ".inst 0xe0bb12ea  // st1w { za2h.s[x12, #2] }, p4/Z, [x23, x27, LSL #2]\n"
      ".inst 0xe09b12ca  // ld1w { za2h.s[x12, #2] }, p4/Z, [x22, x27, LSL #2]\n"
      ".inst 0xe0ba12eb  // st1w { za2h.s[x12, #3] }, p4/Z, [x23, x26, LSL #2]\n"
      ".inst 0xe09a12cb  // ld1w { za2h.s[x12, #3] }, p4/Z, [x22, x26, LSL #2]\n"
      ".inst 0xe0bf12ac  // st1w { za3h.s[x12] }, p4/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f128c  // ld1w { za3h.s[x12] }, p4/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0bc12ad  // st1w { za3h.s[x12, #1] }, p4/Z, [x21, x28, LSL #2]\n"
      ".inst 0xe09c128d  // ld1w { za3h.s[x12, #1] }, p4/Z, [x20, x28, LSL #2]\n"
      ".inst 0xe0bb12ae  // st1w { za3h.s[x12, #2] }, p4/Z, [x21, x27, LSL #2]\n"
      ".inst 0xe09b128e  // ld1w { za3h.s[x12, #2] }, p4/Z, [x20, x27, LSL #2]\n"
      ".inst 0xe0ba12af  // st1w { za3h.s[x12, #3] }, p4/Z, [x21, x26, LSL #2]\n"
      ".inst 0xe09a128f  // ld1w { za3h.s[x12, #3] }, p4/Z, [x20, x26, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x9\n"
      "blt 12b\n"
      "b 21f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xe0bf10e0  // st1w { za0h.s[x12] }, p4/Z, [x7, XZR, LSL #2]\n"
      "addvl x22, x7, #4\n"
      "addvl x21, x7, #8\n"
      ".inst 0xe0b910e1  // st1w { za0h.s[x12, #1] }, p4/Z, [x7, x25, LSL #2]\n"
      "addvl x20, x7, #12\n"
      ".inst 0xe0b810e2  // st1w { za0h.s[x12, #2] }, p4/Z, [x7, x24, LSL #2]\n"
      ".inst 0xe0b710e3  // st1w { za0h.s[x12, #3] }, p4/Z, [x7, x23, LSL #2]\n"
      "addvl x7, x7, #16\n"
      ".inst 0xe0bf12c4  // st1w { za1h.s[x12] }, p4/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0b912c5  // st1w { za1h.s[x12, #1] }, p4/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe0b812c6  // st1w { za1h.s[x12, #2] }, p4/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe0b712c7  // st1w { za1h.s[x12, #3] }, p4/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe0bf12a8  // st1w { za2h.s[x12] }, p4/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe0b912a9  // st1w { za2h.s[x12, #1] }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe0b812aa  // st1w { za2h.s[x12, #2] }, p4/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe0b712ab  // st1w { za2h.s[x12, #3] }, p4/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe0bf128c  // st1w { za3h.s[x12] }, p4/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0b9128d  // st1w { za3h.s[x12, #1] }, p4/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe0b8128e  // st1w { za3h.s[x12, #2] }, p4/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe0b7128f  // st1w { za3h.s[x12, #3] }, p4/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 14b\n"
      "b 21f\n"
      "15:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x15, x17\n"
      "cntw x20\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x21, x20\n"
      "mov x12, #0\n"
      "csel x20, x21, x20, LT\n"
      "add x23, x23, x16, LSL #2\n"  // C += n
      "lsr x21, x20, #0x2\n"
      "madd x23, x17, x22, x23\n"  // C += m * ldc
      "and x20, x20, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0821013  // mova z19.s, p4/M, za0h.s[x12]\n"
      ".inst 0xc0821092  // mova z18.s, p4/M, za1h.s[x12]\n"
      "st1w { z19.s }, p3, [x23]\n"
      ".inst 0xc0821111  // mova z17.s, p4/M, za2h.s[x12]\n"
      "st1w { z18.s }, p2, [x23, #1, MUL VL]\n"
      ".inst 0xc0821190  // mova z16.s, p4/M, za3h.s[x12]\n"
      "st1w { z17.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z16.s }, p0, [x23, #3, MUL VL]\n"
      ".inst 0xc082103b  // mova z27.s, p4/M, za0h.s[x12, #1]\n"
      ".inst 0xc082105a  // mova z26.s, p4/M, za0h.s[x12, #2]\n"
      ".inst 0xc0821079  // mova z25.s, p4/M, za0h.s[x12, #3]\n"
      "add x23, x23, x22\n"
      ".inst 0xc08210b8  // mova z24.s, p4/M, za1h.s[x12, #1]\n"
      ".inst 0xc08210d7  // mova z23.s, p4/M, za1h.s[x12, #2]\n"
      "st1w { z27.s }, p3, [x23]\n"
      ".inst 0xc08210f6  // mova z22.s, p4/M, za1h.s[x12, #3]\n"
      "st1w { z24.s }, p2, [x23, #1, MUL VL]\n"
      ".inst 0xc0821135  // mova z21.s, p4/M, za2h.s[x12, #1]\n"
      ".inst 0xc0821154  // mova z20.s, p4/M, za2h.s[x12, #2]\n"
      "st1w { z21.s }, p1, [x23, #2, MUL VL]\n"
      ".inst 0xc0821173  // mova z19.s, p4/M, za2h.s[x12, #3]\n"
      ".inst 0xc08211b2  // mova z18.s, p4/M, za3h.s[x12, #1]\n"
      ".inst 0xc08211d1  // mova z17.s, p4/M, za3h.s[x12, #2]\n"
      "st1w { z18.s }, p0, [x23, #3, MUL VL]\n"
      "add x23, x23, x22\n"
      ".inst 0xc08211f0  // mova z16.s, p4/M, za3h.s[x12, #3]\n"
      "add x12, x12, #0x4\n"
      "st1w { z26.s }, p3, [x23]\n"
      "st1w { z23.s }, p2, [x23, #1, MUL VL]\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z20.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z17.s }, p0, [x23, #3, MUL VL]\n"
      "add x23, x23, x22\n"
      "st1w { z25.s }, p3, [x23]\n"
      "st1w { z22.s }, p2, [x23, #1, MUL VL]\n"
      "st1w { z19.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z16.s }, p0, [x23, #3, MUL VL]\n"
      "add x23, x23, x22\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0821010  // mova z16.s, p4/M, za0h.s[x12]\n"
      ".inst 0xc082103a  // mova z26.s, p4/M, za0h.s[x12, #1]\n"
      "st1w { z16.s }, p3, [x23]\n"
      ".inst 0xc0821059  // mova z25.s, p4/M, za0h.s[x12, #2]\n"
      ".inst 0xc0821098  // mova z24.s, p4/M, za1h.s[x12]\n"
      ".inst 0xc08210b7  // mova z23.s, p4/M, za1h.s[x12, #1]\n"
      "st1w { z24.s }, p2, [x23, #1, MUL VL]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc08210d6  // mova z22.s, p4/M, za1h.s[x12, #2]\n"
      ".inst 0xc0821115  // mova z21.s, p4/M, za2h.s[x12]\n"
      ".inst 0xc0821134  // mova z20.s, p4/M, za2h.s[x12, #1]\n"
      "st1w { z21.s }, p1, [x23, #2, MUL VL]\n"
      ".inst 0xc0821153  // mova z19.s, p4/M, za2h.s[x12, #2]\n"
      ".inst 0xc0821192  // mova z18.s, p4/M, za3h.s[x12]\n"
      ".inst 0xc08211b1  // mova z17.s, p4/M, za3h.s[x12, #1]\n"
      "st1w { z18.s }, p0, [x23, #3, MUL VL]\n"
      "add x23, x23, x22\n"
      ".inst 0xc08211d0  // mova z16.s, p4/M, za3h.s[x12, #2]\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z26.s }, p3, [x23]\n"
      "st1w { z23.s }, p2, [x23, #1, MUL VL]\n"
      "st1w { z20.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z17.s }, p0, [x23, #3, MUL VL]\n"
      "add x23, x23, x22\n"
      "beq 18f\n"
      "st1w { z25.s }, p3, [x23]\n"
      "st1w { z22.s }, p2, [x23, #1, MUL VL]\n"
      "st1w { z19.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z16.s }, p0, [x23, #3, MUL VL]\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "tbz x5, #0, 21f\n"
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "20:"  // Store to output array: Refill accumulators: Loop
      "addvl x22, x6, #4\n"
      "addvl x21, x6, #8\n"
      ".inst 0xe09f10c0  // ld1w { za0h.s[x12] }, p4/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f12c4  // ld1w { za1h.s[x12] }, p4/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f12a8  // ld1w { za2h.s[x12] }, p4/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f128c  // ld1w { za3h.s[x12] }, p4/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09910c1  // ld1w { za0h.s[x12, #1] }, p4/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe09912c5  // ld1w { za1h.s[x12, #1] }, p4/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe09912a9  // ld1w { za2h.s[x12, #1] }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe099128d  // ld1w { za3h.s[x12, #1] }, p4/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09810c2  // ld1w { za0h.s[x12, #2] }, p4/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe09812c6  // ld1w { za1h.s[x12, #2] }, p4/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe09812aa  // ld1w { za2h.s[x12, #2] }, p4/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe098128e  // ld1w { za3h.s[x12, #2] }, p4/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09710c3  // ld1w { za0h.s[x12, #3] }, p4/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe09712c7  // ld1w { za1h.s[x12, #3] }, p4/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe09712ab  // ld1w { za2h.s[x12, #3] }, p4/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe097128f  // ld1w { za3h.s[x12, #3] }, p4/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 20b\n"
      "21:"  // End block
      "incw x16, ALL, MUL #4\n"
      "cmp x16, x14\n"
      "blt 4b\n"
      "incw x17\n"
      "mov x16, #0\n"
      "cmp x17, x15\n"
      "mov x13, x10\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
