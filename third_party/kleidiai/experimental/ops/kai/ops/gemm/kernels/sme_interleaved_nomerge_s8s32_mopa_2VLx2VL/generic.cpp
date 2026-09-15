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

void sme_interleaved_nomerge_s8s32_mopa_2VLx2VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
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
      "ptrue p2.b\n"
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
      ".inst 0xe09f08c0  // ld1w { za0h.s[x12] }, p2/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f0ac4  // ld1w { za1h.s[x12] }, p2/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f0aa8  // ld1w { za2h.s[x12] }, p2/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f0a8c  // ld1w { za3h.s[x12] }, p2/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09908c1  // ld1w { za0h.s[x12, #1] }, p2/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe0990ac5  // ld1w { za1h.s[x12, #1] }, p2/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe0990aa9  // ld1w { za2h.s[x12, #1] }, p2/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe0990a8d  // ld1w { za3h.s[x12, #1] }, p2/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09808c2  // ld1w { za0h.s[x12, #2] }, p2/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe0980ac6  // ld1w { za1h.s[x12, #2] }, p2/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe0980aaa  // ld1w { za2h.s[x12, #2] }, p2/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe0980a8e  // ld1w { za3h.s[x12, #2] }, p2/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09708c3  // ld1w { za0h.s[x12, #3] }, p2/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe0970ac7  // ld1w { za1h.s[x12, #3] }, p2/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe0970aab  // ld1w { za2h.s[x12, #3] }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe0970a8f  // ld1w { za3h.s[x12, #3] }, p2/Z, [x20, x23, LSL #2]\n"
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
      "whilelt p1.s, x20, x14\n"
      "incw x20\n"
      "whilelt p0.s, x20, x14\n"
      "tbnz x5, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "add x20, x20, x16, LSL #2\n"
      "ld1w { z17.s }, p1/Z, [x20]\n"
      "ld1w { z16.s }, p0/Z, [x20, #1, MUL VL]\n"
      ".inst 0xc0904a20  // addha za0.s, p2/M, p2/M, z17.s\n"
      ".inst 0xc0904a01  // addha za1.s, p2/M, p2/M, z16.s\n"
      ".inst 0xc0904a22  // addha za2.s, p2/M, p2/M, z17.s\n"
      ".inst 0xc0904a03  // addha za3.s, p2/M, p2/M, z16.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x16\n"
      "mov x21, x17\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
      "cmp x20, x14\n"
      "mov x20, x5\n"
      "csel x21, x17, x21, LT\n"
      "bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1\n"
      "cmp x21, x15\n"
      "csel x5, x20, x5, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x8, #0x2\n"
      "and x20, x8, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      "ld1b { z31.b }, p2/Z, [x10]\n"
      "ld1b { z30.b }, p2/Z, [x10, #1, MUL VL]\n"
      "ld1b { z29.b }, p2/Z, [x10, #2, MUL VL]\n"
      "ld1b { z28.b }, p2/Z, [x10, #3, MUL VL]\n"
      "ld1b { z27.b }, p2/Z, [x10, #4, MUL VL]\n"
      "ld1b { z26.b }, p2/Z, [x10, #5, MUL VL]\n"
      "ld1b { z25.b }, p2/Z, [x10, #6, MUL VL]\n"
      "ld1b { z24.b }, p2/Z, [x10, #7, MUL VL]\n"
      "addvl x10, x10, #8\n"
      "ld1b { z23.b }, p2/Z, [x11]\n"
      "ld1b { z22.b }, p2/Z, [x11, #1, MUL VL]\n"
      "ld1b { z21.b }, p2/Z, [x11, #2, MUL VL]\n"
      "ld1b { z20.b }, p2/Z, [x11, #3, MUL VL]\n"
      "ld1b { z19.b }, p2/Z, [x11, #4, MUL VL]\n"
      "ld1b { z18.b }, p2/Z, [x11, #5, MUL VL]\n"
      "ld1b { z17.b }, p2/Z, [x11, #6, MUL VL]\n"
      "ld1b { z16.b }, p2/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0974be0  // smopa za0.s, p2/M, p2/M, z31.b, z23.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0964be1  // smopa za1.s, p2/M, p2/M, z31.b, z22.b\n"
      "ld1b { z31.b }, p2/Z, [x10]\n"
      ".inst 0xa0974bc2  // smopa za2.s, p2/M, p2/M, z30.b, z23.b\n"
      "ld1b { z23.b }, p2/Z, [x11]\n"
      ".inst 0xa0964bc3  // smopa za3.s, p2/M, p2/M, z30.b, z22.b\n"
      "ld1b { z30.b }, p2/Z, [x10, #1, MUL VL]\n"
      ".inst 0xa0954ba0  // smopa za0.s, p2/M, p2/M, z29.b, z21.b\n"
      "ld1b { z22.b }, p2/Z, [x11, #1, MUL VL]\n"
      ".inst 0xa0944ba1  // smopa za1.s, p2/M, p2/M, z29.b, z20.b\n"
      "ld1b { z29.b }, p2/Z, [x10, #2, MUL VL]\n"
      ".inst 0xa0954b82  // smopa za2.s, p2/M, p2/M, z28.b, z21.b\n"
      "ld1b { z21.b }, p2/Z, [x11, #2, MUL VL]\n"
      ".inst 0xa0944b83  // smopa za3.s, p2/M, p2/M, z28.b, z20.b\n"
      "ld1b { z28.b }, p2/Z, [x10, #3, MUL VL]\n"
      ".inst 0xa0934b60  // smopa za0.s, p2/M, p2/M, z27.b, z19.b\n"
      "ld1b { z20.b }, p2/Z, [x11, #3, MUL VL]\n"
      ".inst 0xa0924b61  // smopa za1.s, p2/M, p2/M, z27.b, z18.b\n"
      "ld1b { z27.b }, p2/Z, [x10, #4, MUL VL]\n"
      ".inst 0xa0934b42  // smopa za2.s, p2/M, p2/M, z26.b, z19.b\n"
      "ld1b { z19.b }, p2/Z, [x11, #4, MUL VL]\n"
      ".inst 0xa0924b43  // smopa za3.s, p2/M, p2/M, z26.b, z18.b\n"
      "ld1b { z26.b }, p2/Z, [x10, #5, MUL VL]\n"
      ".inst 0xa0914b20  // smopa za0.s, p2/M, p2/M, z25.b, z17.b\n"
      "ld1b { z18.b }, p2/Z, [x11, #5, MUL VL]\n"
      ".inst 0xa0904b21  // smopa za1.s, p2/M, p2/M, z25.b, z16.b\n"
      "ld1b { z25.b }, p2/Z, [x10, #6, MUL VL]\n"
      ".inst 0xa0914b02  // smopa za2.s, p2/M, p2/M, z24.b, z17.b\n"
      "ld1b { z17.b }, p2/Z, [x11, #6, MUL VL]\n"
      ".inst 0xa0904b03  // smopa za3.s, p2/M, p2/M, z24.b, z16.b\n"
      "ld1b { z24.b }, p2/Z, [x10, #7, MUL VL]\n"
      "addvl x10, x10, #8\n"
      "ld1b { z16.b }, p2/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0974be0  // smopa za0.s, p2/M, p2/M, z31.b, z23.b\n"
      ".inst 0xa0964be1  // smopa za1.s, p2/M, p2/M, z31.b, z22.b\n"
      ".inst 0xa0974bc2  // smopa za2.s, p2/M, p2/M, z30.b, z23.b\n"
      ".inst 0xa0964bc3  // smopa za3.s, p2/M, p2/M, z30.b, z22.b\n"
      ".inst 0xa0954ba0  // smopa za0.s, p2/M, p2/M, z29.b, z21.b\n"
      ".inst 0xa0944ba1  // smopa za1.s, p2/M, p2/M, z29.b, z20.b\n"
      ".inst 0xa0954b82  // smopa za2.s, p2/M, p2/M, z28.b, z21.b\n"
      ".inst 0xa0944b83  // smopa za3.s, p2/M, p2/M, z28.b, z20.b\n"
      ".inst 0xa0934b60  // smopa za0.s, p2/M, p2/M, z27.b, z19.b\n"
      ".inst 0xa0924b61  // smopa za1.s, p2/M, p2/M, z27.b, z18.b\n"
      ".inst 0xa0934b42  // smopa za2.s, p2/M, p2/M, z26.b, z19.b\n"
      ".inst 0xa0924b43  // smopa za3.s, p2/M, p2/M, z26.b, z18.b\n"
      ".inst 0xa0914b20  // smopa za0.s, p2/M, p2/M, z25.b, z17.b\n"
      ".inst 0xa0904b21  // smopa za1.s, p2/M, p2/M, z25.b, z16.b\n"
      ".inst 0xa0914b02  // smopa za2.s, p2/M, p2/M, z24.b, z17.b\n"
      ".inst 0xa0904b03  // smopa za3.s, p2/M, p2/M, z24.b, z16.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z19.b }, p2/Z, [x10]\n"
      "subs x20, x20, #0x1\n"
      "ld1b { z18.b }, p2/Z, [x10, #1, MUL VL]\n"
      "addvl x10, x10, #2\n"
      "ld1b { z17.b }, p2/Z, [x11]\n"
      "ld1b { z16.b }, p2/Z, [x11, #1, MUL VL]\n"
      "addvl x11, x11, #2\n"
      ".inst 0xa0914a60  // smopa za0.s, p2/M, p2/M, z19.b, z17.b\n"
      ".inst 0xa0904a61  // smopa za1.s, p2/M, p2/M, z19.b, z16.b\n"
      ".inst 0xa0914a42  // smopa za2.s, p2/M, p2/M, z18.b, z17.b\n"
      ".inst 0xa0904a43  // smopa za3.s, p2/M, p2/M, z18.b, z16.b\n"
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
      ".inst 0xe0bf08e0  // st1w { za0h.s[x12] }, p2/Z, [x7, XZR, LSL #2]\n"
      ".inst 0xe09f08c0  // ld1w { za0h.s[x12] }, p2/Z, [x6, XZR, LSL #2]\n"
      "addvl x25, x7, #4\n"
      "addvl x24, x6, #4\n"
      ".inst 0xe0bc08e1  // st1w { za0h.s[x12, #1] }, p2/Z, [x7, x28, LSL #2]\n"
      ".inst 0xe09c08c1  // ld1w { za0h.s[x12, #1] }, p2/Z, [x6, x28, LSL #2]\n"
      "addvl x23, x7, #8\n"
      "addvl x22, x6, #8\n"
      ".inst 0xe0bb08e2  // st1w { za0h.s[x12, #2] }, p2/Z, [x7, x27, LSL #2]\n"
      ".inst 0xe09b08c2  // ld1w { za0h.s[x12, #2] }, p2/Z, [x6, x27, LSL #2]\n"
      "addvl x21, x7, #12\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe0ba08e3  // st1w { za0h.s[x12, #3] }, p2/Z, [x7, x26, LSL #2]\n"
      ".inst 0xe09a08c3  // ld1w { za0h.s[x12, #3] }, p2/Z, [x6, x26, LSL #2]\n"
      "addvl x7, x7, #16\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe0bf0b24  // st1w { za1h.s[x12] }, p2/Z, [x25, XZR, LSL #2]\n"
      ".inst 0xe09f0b04  // ld1w { za1h.s[x12] }, p2/Z, [x24, XZR, LSL #2]\n"
      ".inst 0xe0bc0b25  // st1w { za1h.s[x12, #1] }, p2/Z, [x25, x28, LSL #2]\n"
      ".inst 0xe09c0b05  // ld1w { za1h.s[x12, #1] }, p2/Z, [x24, x28, LSL #2]\n"
      ".inst 0xe0bb0b26  // st1w { za1h.s[x12, #2] }, p2/Z, [x25, x27, LSL #2]\n"
      ".inst 0xe09b0b06  // ld1w { za1h.s[x12, #2] }, p2/Z, [x24, x27, LSL #2]\n"
      ".inst 0xe0ba0b27  // st1w { za1h.s[x12, #3] }, p2/Z, [x25, x26, LSL #2]\n"
      ".inst 0xe09a0b07  // ld1w { za1h.s[x12, #3] }, p2/Z, [x24, x26, LSL #2]\n"
      ".inst 0xe0bf0ae8  // st1w { za2h.s[x12] }, p2/Z, [x23, XZR, LSL #2]\n"
      ".inst 0xe09f0ac8  // ld1w { za2h.s[x12] }, p2/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0bc0ae9  // st1w { za2h.s[x12, #1] }, p2/Z, [x23, x28, LSL #2]\n"
      ".inst 0xe09c0ac9  // ld1w { za2h.s[x12, #1] }, p2/Z, [x22, x28, LSL #2]\n"
      ".inst 0xe0bb0aea  // st1w { za2h.s[x12, #2] }, p2/Z, [x23, x27, LSL #2]\n"
      ".inst 0xe09b0aca  // ld1w { za2h.s[x12, #2] }, p2/Z, [x22, x27, LSL #2]\n"
      ".inst 0xe0ba0aeb  // st1w { za2h.s[x12, #3] }, p2/Z, [x23, x26, LSL #2]\n"
      ".inst 0xe09a0acb  // ld1w { za2h.s[x12, #3] }, p2/Z, [x22, x26, LSL #2]\n"
      ".inst 0xe0bf0aac  // st1w { za3h.s[x12] }, p2/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f0a8c  // ld1w { za3h.s[x12] }, p2/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0bc0aad  // st1w { za3h.s[x12, #1] }, p2/Z, [x21, x28, LSL #2]\n"
      ".inst 0xe09c0a8d  // ld1w { za3h.s[x12, #1] }, p2/Z, [x20, x28, LSL #2]\n"
      ".inst 0xe0bb0aae  // st1w { za3h.s[x12, #2] }, p2/Z, [x21, x27, LSL #2]\n"
      ".inst 0xe09b0a8e  // ld1w { za3h.s[x12, #2] }, p2/Z, [x20, x27, LSL #2]\n"
      ".inst 0xe0ba0aaf  // st1w { za3h.s[x12, #3] }, p2/Z, [x21, x26, LSL #2]\n"
      ".inst 0xe09a0a8f  // ld1w { za3h.s[x12, #3] }, p2/Z, [x20, x26, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x9\n"
      "blt 12b\n"
      "b 24f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xe0bf08e0  // st1w { za0h.s[x12] }, p2/Z, [x7, XZR, LSL #2]\n"
      "addvl x22, x7, #4\n"
      "addvl x21, x7, #8\n"
      ".inst 0xe0b908e1  // st1w { za0h.s[x12, #1] }, p2/Z, [x7, x25, LSL #2]\n"
      "addvl x20, x7, #12\n"
      ".inst 0xe0b808e2  // st1w { za0h.s[x12, #2] }, p2/Z, [x7, x24, LSL #2]\n"
      ".inst 0xe0b708e3  // st1w { za0h.s[x12, #3] }, p2/Z, [x7, x23, LSL #2]\n"
      "addvl x7, x7, #16\n"
      ".inst 0xe0bf0ac4  // st1w { za1h.s[x12] }, p2/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0b90ac5  // st1w { za1h.s[x12, #1] }, p2/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe0b80ac6  // st1w { za1h.s[x12, #2] }, p2/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe0b70ac7  // st1w { za1h.s[x12, #3] }, p2/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe0bf0aa8  // st1w { za2h.s[x12] }, p2/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe0b90aa9  // st1w { za2h.s[x12, #1] }, p2/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe0b80aaa  // st1w { za2h.s[x12, #2] }, p2/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe0b70aab  // st1w { za2h.s[x12, #3] }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe0bf0a8c  // st1w { za3h.s[x12] }, p2/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0b90a8d  // st1w { za3h.s[x12, #1] }, p2/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe0b80a8e  // st1w { za3h.s[x12, #2] }, p2/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe0b70a8f  // st1w { za3h.s[x12, #3] }, p2/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 14b\n"
      "b 24f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x15, x17\n"
      "cntw x24\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x22, x25, x24, LT\n"
      "add x26, x26, x16, LSL #2\n"  // C += n
      "lsr x21, x22, #0x2\n"
      "madd x26, x17, x23, x26\n"  // C += m * ldc
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0820810  // mova z16.s, p2/M, za0h.s[x12]\n"
      ".inst 0xc0820896  // mova z22.s, p2/M, za1h.s[x12]\n"
      "st1w { z16.s }, p1, [x26]\n"
      "st1w { z22.s }, p0, [x26, #1, MUL VL]\n"
      ".inst 0xc0820835  // mova z21.s, p2/M, za0h.s[x12, #1]\n"
      "add x26, x26, x23\n"
      ".inst 0xc0820854  // mova z20.s, p2/M, za0h.s[x12, #2]\n"
      ".inst 0xc0820873  // mova z19.s, p2/M, za0h.s[x12, #3]\n"
      "st1w { z21.s }, p1, [x26]\n"
      ".inst 0xc08208b2  // mova z18.s, p2/M, za1h.s[x12, #1]\n"
      ".inst 0xc08208d1  // mova z17.s, p2/M, za1h.s[x12, #2]\n"
      "st1w { z18.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      ".inst 0xc08208f0  // mova z16.s, p2/M, za1h.s[x12, #3]\n"
      "add x12, x12, #0x4\n"
      "st1w { z20.s }, p1, [x26]\n"
      "st1w { z17.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z19.s }, p1, [x26]\n"
      "st1w { z16.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0820815  // mova z21.s, p2/M, za0h.s[x12]\n"
      ".inst 0xc0820834  // mova z20.s, p2/M, za0h.s[x12, #1]\n"
      "st1w { z21.s }, p1, [x26]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0820853  // mova z19.s, p2/M, za0h.s[x12, #2]\n"
      ".inst 0xc0820892  // mova z18.s, p2/M, za1h.s[x12]\n"
      ".inst 0xc08208b1  // mova z17.s, p2/M, za1h.s[x12, #1]\n"
      "st1w { z18.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      ".inst 0xc08208d0  // mova z16.s, p2/M, za1h.s[x12, #2]\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z20.s }, p1, [x26]\n"
      "st1w { z17.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "beq 18f\n"
      "st1w { z19.s }, p1, [x26]\n"
      "st1w { z16.s }, p0, [x26, #1, MUL VL]\n"
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
      ".inst 0xc0820910  // mova z16.s, p2/M, za2h.s[x12]\n"
      ".inst 0xc0820996  // mova z22.s, p2/M, za3h.s[x12]\n"
      "st1w { z16.s }, p1, [x26]\n"
      "st1w { z22.s }, p0, [x26, #1, MUL VL]\n"
      ".inst 0xc0820935  // mova z21.s, p2/M, za2h.s[x12, #1]\n"
      "add x26, x26, x23\n"
      ".inst 0xc0820954  // mova z20.s, p2/M, za2h.s[x12, #2]\n"
      ".inst 0xc0820973  // mova z19.s, p2/M, za2h.s[x12, #3]\n"
      "st1w { z21.s }, p1, [x26]\n"
      ".inst 0xc08209b2  // mova z18.s, p2/M, za3h.s[x12, #1]\n"
      ".inst 0xc08209d1  // mova z17.s, p2/M, za3h.s[x12, #2]\n"
      "st1w { z18.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      ".inst 0xc08209f0  // mova z16.s, p2/M, za3h.s[x12, #3]\n"
      "add x12, x12, #0x4\n"
      "st1w { z20.s }, p1, [x26]\n"
      "st1w { z17.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z19.s }, p1, [x26]\n"
      "st1w { z16.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      ".inst 0xc0820915  // mova z21.s, p2/M, za2h.s[x12]\n"
      ".inst 0xc0820934  // mova z20.s, p2/M, za2h.s[x12, #1]\n"
      "st1w { z21.s }, p1, [x26]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0820953  // mova z19.s, p2/M, za2h.s[x12, #2]\n"
      ".inst 0xc0820992  // mova z18.s, p2/M, za3h.s[x12]\n"
      ".inst 0xc08209b1  // mova z17.s, p2/M, za3h.s[x12, #1]\n"
      "st1w { z18.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      ".inst 0xc08209d0  // mova z16.s, p2/M, za3h.s[x12, #2]\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z20.s }, p1, [x26]\n"
      "st1w { z17.s }, p0, [x26, #1, MUL VL]\n"
      "add x26, x26, x23\n"
      "beq 21f\n"
      "st1w { z19.s }, p1, [x26]\n"
      "st1w { z16.s }, p0, [x26, #1, MUL VL]\n"
      "21:"  // Store to output array: Accumulator row 1 oddments: End
      "22:"  // Store to output array: End
      "tbz x5, #0, 24f\n"
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "23:"  // Store to output array: Refill accumulators: Loop
      "addvl x22, x6, #4\n"
      "addvl x21, x6, #8\n"
      ".inst 0xe09f08c0  // ld1w { za0h.s[x12] }, p2/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f0ac4  // ld1w { za1h.s[x12] }, p2/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f0aa8  // ld1w { za2h.s[x12] }, p2/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f0a8c  // ld1w { za3h.s[x12] }, p2/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09908c1  // ld1w { za0h.s[x12, #1] }, p2/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe0990ac5  // ld1w { za1h.s[x12, #1] }, p2/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe0990aa9  // ld1w { za2h.s[x12, #1] }, p2/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe0990a8d  // ld1w { za3h.s[x12, #1] }, p2/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09808c2  // ld1w { za0h.s[x12, #2] }, p2/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe0980ac6  // ld1w { za1h.s[x12, #2] }, p2/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe0980aaa  // ld1w { za2h.s[x12, #2] }, p2/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe0980a8e  // ld1w { za3h.s[x12, #2] }, p2/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09708c3  // ld1w { za0h.s[x12, #3] }, p2/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe0970ac7  // ld1w { za1h.s[x12, #3] }, p2/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe0970aab  // ld1w { za2h.s[x12, #3] }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe0970a8f  // ld1w { za3h.s[x12, #3] }, p2/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 23b\n"
      "24:"  // End block
      "incw x16, ALL, MUL #2\n"
      "cmp x16, x14\n"
      "blt 4b\n"
      "incw x17, ALL, MUL #2\n"
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
