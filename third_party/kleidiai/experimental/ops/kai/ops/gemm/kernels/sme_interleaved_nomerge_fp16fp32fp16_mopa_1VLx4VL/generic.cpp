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

void sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const __fp16 *const A,
      const __fp16 *const B,
      __fp16 *const C, const int ldc,
      const int M, const int N, const int K,
      const __fp16 *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 2) * sizeof(__fp16)),
        C(C), ldcb(ldc * sizeof(__fp16)),
        M(M), N(N), K(K),
        min(-static_cast<__fp16>(std::numeric_limits<float>::infinity())),
        max(static_cast<__fp16>(std::numeric_limits<float>::infinity())),
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

      // Initialise the activation values
      switch (act.type)
      {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            this->max = static_cast<__fp16>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<__fp16>(0);
            break;
      }
    }

    const __fp16 *const A;
    const __fp16 *const B;
    const long kstride_bytes;
    __fp16 *const C;
    const long ldcb;
    const long M, N, K;
    __fp16 min = -static_cast<__fp16>(std::numeric_limits<float>::infinity());
    __fp16 max = static_cast<__fp16>(std::numeric_limits<float>::infinity());

    const __fp16 *const bias;


    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

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
      "add x8, x8, #0x1\n"
      "ldr x13, [%x[args], %[offsetof_A]]\n"
      "lsr x8, x8, #0x1\n"
      "3:"  // M loop
      "ldr x11, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x20, x16\n"
      "mov x10, x13\n"
      "whilelt p8.s, x20, x14\n"
      "incw x20\n"
      "whilelt p8.s, x20, x14\n"
      "incw x20\n"
      "whilelt p8.s, x20, x14\n"
      "incw x20\n"
      "whilelt p8.s, x20, x14\n"
      "tbnz x5, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "mov x21, x16\n"
      "fmov z21.h, #0.0\n"
      "add x20, x20, x16, LSL #1\n"
      "fmov z20.h, #1.0\n"
      "whilelt p1.h, x21, x14\n"
      "inch x21\n"
      "whilelt p0.h, x21, x14\n"
      "ldnt1h { z17.h }, p1/Z, [x20]\n"
      "ldnt1h { z16.h }, p0/Z, [x20, #1, MUL VL]\n"
      "zip1 z19.h, z17.h, z21.h\n"
      "zip2 z18.h, z17.h, z21.h\n"
      "zip1 z17.h, z16.h, z21.h\n"
      "zip2 z16.h, z16.h, z21.h\n"
      ".inst 0x81b34a80  // fmopa za0.s, p2/M, p2/M, z20.h, z19.h\n"
      ".inst 0x81b24a81  // fmopa za1.s, p2/M, p2/M, z20.h, z18.h\n"
      ".inst 0x81b14a82  // fmopa za2.s, p2/M, p2/M, z20.h, z17.h\n"
      ".inst 0x81b04a83  // fmopa za3.s, p2/M, p2/M, z20.h, z16.h\n"
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
      "ld1h { z3.h }, p2/Z, [x10]\n"
      "subs x23, x23, #0x1\n"
      "ld1h { z2.h }, p2/Z, [x10, #1, MUL VL]\n"
      "ld1h { z1.h }, p2/Z, [x10, #2, MUL VL]\n"
      "ld1h { z0.h }, p2/Z, [x10, #3, MUL VL]\n"
      "addvl x10, x10, #4\n"
      "ld1h { z31.h }, p2/Z, [x11]\n"
      "ld1h { z30.h }, p2/Z, [x11, #1, MUL VL]\n"
      "ld1h { z29.h }, p2/Z, [x11, #2, MUL VL]\n"
      "ld1h { z28.h }, p2/Z, [x11, #3, MUL VL]\n"
      "ld1h { z27.h }, p2/Z, [x11, #4, MUL VL]\n"
      "ld1h { z26.h }, p2/Z, [x11, #5, MUL VL]\n"
      "ld1h { z25.h }, p2/Z, [x11, #6, MUL VL]\n"
      "ld1h { z24.h }, p2/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "ld1h { z23.h }, p2/Z, [x21]\n"
      "ld1h { z22.h }, p2/Z, [x21, #1, MUL VL]\n"
      "ld1h { z21.h }, p2/Z, [x21, #2, MUL VL]\n"
      "ld1h { z20.h }, p2/Z, [x21, #3, MUL VL]\n"
      "ld1h { z19.h }, p2/Z, [x20]\n"
      "ld1h { z18.h }, p2/Z, [x20, #1, MUL VL]\n"
      "ld1h { z17.h }, p2/Z, [x20, #2, MUL VL]\n"
      "ld1h { z16.h }, p2/Z, [x20, #3, MUL VL]\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81bf4860  // fmopa za0.s, p2/M, p2/M, z3.h, z31.h\n"
      "addvl x21, x11, #8\n"
      "addvl x20, x11, #12\n"
      "ld1h { z31.h }, p2/Z, [x11]\n"
      ".inst 0x81be4861  // fmopa za1.s, p2/M, p2/M, z3.h, z30.h\n"
      "subs x23, x23, #0x1\n"
      "ld1h { z30.h }, p2/Z, [x11, #1, MUL VL]\n"
      ".inst 0x81bd4862  // fmopa za2.s, p2/M, p2/M, z3.h, z29.h\n"
      "ld1h { z29.h }, p2/Z, [x11, #2, MUL VL]\n"
      ".inst 0x81bc4863  // fmopa za3.s, p2/M, p2/M, z3.h, z28.h\n"
      "ld1h { z3.h }, p2/Z, [x10]\n"
      ".inst 0x81bb4840  // fmopa za0.s, p2/M, p2/M, z2.h, z27.h\n"
      "ld1h { z28.h }, p2/Z, [x11, #3, MUL VL]\n"
      ".inst 0x81ba4841  // fmopa za1.s, p2/M, p2/M, z2.h, z26.h\n"
      "ld1h { z27.h }, p2/Z, [x11, #4, MUL VL]\n"
      ".inst 0x81b94842  // fmopa za2.s, p2/M, p2/M, z2.h, z25.h\n"
      "ld1h { z26.h }, p2/Z, [x11, #5, MUL VL]\n"
      ".inst 0x81b84843  // fmopa za3.s, p2/M, p2/M, z2.h, z24.h\n"
      "ld1h { z2.h }, p2/Z, [x10, #1, MUL VL]\n"
      ".inst 0x81b74820  // fmopa za0.s, p2/M, p2/M, z1.h, z23.h\n"
      "ld1h { z25.h }, p2/Z, [x11, #6, MUL VL]\n"
      ".inst 0x81b64821  // fmopa za1.s, p2/M, p2/M, z1.h, z22.h\n"
      "ld1h { z24.h }, p2/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #16\n"
      ".inst 0x81b54822  // fmopa za2.s, p2/M, p2/M, z1.h, z21.h\n"
      "ld1h { z23.h }, p2/Z, [x21]\n"
      ".inst 0x81b44823  // fmopa za3.s, p2/M, p2/M, z1.h, z20.h\n"
      "ld1h { z1.h }, p2/Z, [x10, #2, MUL VL]\n"
      ".inst 0x81b34800  // fmopa za0.s, p2/M, p2/M, z0.h, z19.h\n"
      "ld1h { z22.h }, p2/Z, [x21, #1, MUL VL]\n"
      ".inst 0x81b24801  // fmopa za1.s, p2/M, p2/M, z0.h, z18.h\n"
      "ld1h { z21.h }, p2/Z, [x21, #2, MUL VL]\n"
      ".inst 0x81b14802  // fmopa za2.s, p2/M, p2/M, z0.h, z17.h\n"
      "ld1h { z20.h }, p2/Z, [x21, #3, MUL VL]\n"
      ".inst 0x81b04803  // fmopa za3.s, p2/M, p2/M, z0.h, z16.h\n"
      "ld1h { z0.h }, p2/Z, [x10, #3, MUL VL]\n"
      "addvl x10, x10, #4\n"
      "ld1h { z19.h }, p2/Z, [x20]\n"
      "ld1h { z18.h }, p2/Z, [x20, #1, MUL VL]\n"
      "ld1h { z17.h }, p2/Z, [x20, #2, MUL VL]\n"
      "ld1h { z16.h }, p2/Z, [x20, #3, MUL VL]\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81bf4860  // fmopa za0.s, p2/M, p2/M, z3.h, z31.h\n"
      ".inst 0x81be4861  // fmopa za1.s, p2/M, p2/M, z3.h, z30.h\n"
      ".inst 0x81bd4862  // fmopa za2.s, p2/M, p2/M, z3.h, z29.h\n"
      ".inst 0x81bc4863  // fmopa za3.s, p2/M, p2/M, z3.h, z28.h\n"
      ".inst 0x81bb4840  // fmopa za0.s, p2/M, p2/M, z2.h, z27.h\n"
      ".inst 0x81ba4841  // fmopa za1.s, p2/M, p2/M, z2.h, z26.h\n"
      ".inst 0x81b94842  // fmopa za2.s, p2/M, p2/M, z2.h, z25.h\n"
      ".inst 0x81b84843  // fmopa za3.s, p2/M, p2/M, z2.h, z24.h\n"
      ".inst 0x81b74820  // fmopa za0.s, p2/M, p2/M, z1.h, z23.h\n"
      ".inst 0x81b64821  // fmopa za1.s, p2/M, p2/M, z1.h, z22.h\n"
      ".inst 0x81b54822  // fmopa za2.s, p2/M, p2/M, z1.h, z21.h\n"
      ".inst 0x81b44823  // fmopa za3.s, p2/M, p2/M, z1.h, z20.h\n"
      ".inst 0x81b34800  // fmopa za0.s, p2/M, p2/M, z0.h, z19.h\n"
      ".inst 0x81b24801  // fmopa za1.s, p2/M, p2/M, z0.h, z18.h\n"
      ".inst 0x81b14802  // fmopa za2.s, p2/M, p2/M, z0.h, z17.h\n"
      ".inst 0x81b04803  // fmopa za3.s, p2/M, p2/M, z0.h, z16.h\n"
      "9:"  // K oddments
      "cbz x22, 11f\n"
      "10:"  // K oddments: Loop
      "ld1h { z20.h }, p2/Z, [x10]\n"
      "subs x22, x22, #0x1\n"
      "addvl x10, x10, #1\n"
      "ld1h { z19.h }, p2/Z, [x11]\n"
      "ld1h { z18.h }, p2/Z, [x11, #1, MUL VL]\n"
      "ld1h { z17.h }, p2/Z, [x11, #2, MUL VL]\n"
      "ld1h { z16.h }, p2/Z, [x11, #3, MUL VL]\n"
      "addvl x11, x11, #4\n"
      ".inst 0x81b34a80  // fmopa za0.s, p2/M, p2/M, z20.h, z19.h\n"
      ".inst 0x81b24a81  // fmopa za1.s, p2/M, p2/M, z20.h, z18.h\n"
      ".inst 0x81b14a82  // fmopa za2.s, p2/M, p2/M, z20.h, z17.h\n"
      ".inst 0x81b04a83  // fmopa za3.s, p2/M, p2/M, z20.h, z16.h\n"
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
      "b 19f\n"
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
      "b 19f\n"
      "15:"  // Store to output array
      "ldr x24, [%x[args], %[offsetof_C]]\n"
      "mov x23, x16\n"
      "sub x22, x15, x17\n"
      "ld1rh { z21.h }, p2/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x21, [%x[args], %[offsetof_ldcb]]\n"
      "cntw x20\n"
      "whilelt p1.h, x23, x14\n"
      "ld1rh { z20.h }, p2/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "inch x23\n"
      "mov x12, #0\n"
      "add x24, x24, x16, LSL #1\n"  // C += n
      "whilelt p0.h, x23, x14\n"
      "cmp x22, x20\n"
      "madd x24, x17, x21, x24\n"  // C += m * ldc
      "csel x22, x22, x20, LT\n"
      "16:"  // Store to output array: Accumulator loop
      ".inst 0xc0020813  // mova z19.b, p2/M, za0h.b[x12]\n"
      ".inst 0xc0020831  // mova z17.b, p2/M, za0h.b[x12, #1]\n"
      "fcvt z19.h, p2/m, z19.s\n"
      ".inst 0xc0020852  // mova z18.b, p2/M, za0h.b[x12, #2]\n"
      "fcvt z17.h, p2/m, z17.s\n"
      ".inst 0xc0020870  // mova z16.b, p2/M, za0h.b[x12, #3]\n"
      "fcvt z18.h, p2/m, z18.s\n"
      "add x12, x12, #0x4\n"
      "fcvt z16.h, p2/m, z16.s\n"
      "cmp x12, x22, LSL #2\n"
      "uzp1 z17.h, z19.h, z17.h\n"
      "uzp1 z16.h, z18.h, z16.h\n"
      "fmin z17.h, p2/M, z17.h, z20.h\n"
      "fmin z16.h, p2/M, z16.h, z20.h\n"
      "fmax z17.h, p2/M, z17.h, z21.h\n"
      "fmax z16.h, p2/M, z16.h, z21.h\n"
      "st1h { z17.h }, p1, [x24]\n"
      "st1h { z16.h }, p0, [x24, #1, MUL VL]\n"
      "add x24, x24, x21\n"
      "blt 16b\n"
      "tbz x5, #0, 19f\n"
      "mov x12, #0\n"
      "cntw x26\n"
      "cntw x25\n"
      "cntw x24, ALL, MUL #2\n"
      "cntw x23, ALL, MUL #3\n"
      "18:"  // Store to output array: Refill accumulators: Loop
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
      "blt 18b\n"
      "19:"  // End block
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
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
