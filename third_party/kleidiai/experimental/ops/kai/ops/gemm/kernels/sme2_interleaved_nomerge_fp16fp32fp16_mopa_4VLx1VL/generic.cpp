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

void sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
      ".inst 0xa040c634  // ld1w { z20.s-z23.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c630  // ld1w { z16.s-z19.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
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
      "add x15, x15, #0x1\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "lsr x15, x15, #0x1\n"
      "3:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x27, x9\n"
      "whilelt p8.s, x13, x10\n"
      "tbnz x8, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z17.h, #0.0\n"
      "whilelt p0.h, x13, x10\n"
      "fmov z30.h, #1.0\n"
      "ld1h { z28.h }, p0/Z, [x20, x13, LSL #1]\n"
      "zip1 z15.h, z28.h, z17.h\n"
      ".inst 0x81af27c0  // fmopa za0.s, p1/M, p1/M, z30.h, z15.h\n"
      ".inst 0x81af27c1  // fmopa za1.s, p1/M, p1/M, z30.h, z15.h\n"
      ".inst 0x81af27c2  // fmopa za2.s, p1/M, p1/M, z30.h, z15.h\n"
      ".inst 0x81af27c3  // fmopa za3.s, p1/M, p1/M, z30.h, z15.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x13\n"
      "mov x21, x14\n"
      "incw x20\n"
      "incw x21, ALL, MUL #4\n"
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
      ".inst 0xa040a768  // ld1h { z8.h-z11.h }, pn9.b/Z, [x27]\n"
      ".inst 0xa041a76c  // ld1h { z12.h-z15.h }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa042a778  // ld1h { z24.h-z27.h }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa043a774  // ld1h { z20.h-z23.h }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      ".inst 0xa040a79c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81bc2500  // fmopa za0.s, p1/M, p1/M, z8.h, z28.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81bc2521  // fmopa za1.s, p1/M, p1/M, z9.h, z28.h\n"
      ".inst 0x81bc2542  // fmopa za2.s, p1/M, p1/M, z10.h, z28.h\n"
      ".inst 0x81bc2563  // fmopa za3.s, p1/M, p1/M, z11.h, z28.h\n"
      ".inst 0xa040a768  // ld1h { z8.h-z11.h }, pn9.b/Z, [x27]\n"
      ".inst 0x81bd2580  // fmopa za0.s, p1/M, p1/M, z12.h, z29.h\n"
      ".inst 0x81bd25a1  // fmopa za1.s, p1/M, p1/M, z13.h, z29.h\n"
      ".inst 0x81bd25c2  // fmopa za2.s, p1/M, p1/M, z14.h, z29.h\n"
      ".inst 0x81bd25e3  // fmopa za3.s, p1/M, p1/M, z15.h, z29.h\n"
      ".inst 0xa041a76c  // ld1h { z12.h-z15.h }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0x81be2700  // fmopa za0.s, p1/M, p1/M, z24.h, z30.h\n"
      ".inst 0x81be2721  // fmopa za1.s, p1/M, p1/M, z25.h, z30.h\n"
      ".inst 0x81be2742  // fmopa za2.s, p1/M, p1/M, z26.h, z30.h\n"
      ".inst 0x81be2763  // fmopa za3.s, p1/M, p1/M, z27.h, z30.h\n"
      ".inst 0xa042a778  // ld1h { z24.h-z27.h }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0x81bf2680  // fmopa za0.s, p1/M, p1/M, z20.h, z31.h\n"
      ".inst 0x81bf26a1  // fmopa za1.s, p1/M, p1/M, z21.h, z31.h\n"
      ".inst 0x81bf26c2  // fmopa za2.s, p1/M, p1/M, z22.h, z31.h\n"
      ".inst 0x81bf26e3  // fmopa za3.s, p1/M, p1/M, z23.h, z31.h\n"
      ".inst 0xa043a774  // ld1h { z20.h-z23.h }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      ".inst 0xa040a79c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81bc2500  // fmopa za0.s, p1/M, p1/M, z8.h, z28.h\n"
      ".inst 0x81bc2521  // fmopa za1.s, p1/M, p1/M, z9.h, z28.h\n"
      ".inst 0x81bc2542  // fmopa za2.s, p1/M, p1/M, z10.h, z28.h\n"
      ".inst 0x81bc2563  // fmopa za3.s, p1/M, p1/M, z11.h, z28.h\n"
      ".inst 0x81bd2580  // fmopa za0.s, p1/M, p1/M, z12.h, z29.h\n"
      ".inst 0x81bd25a1  // fmopa za1.s, p1/M, p1/M, z13.h, z29.h\n"
      ".inst 0x81bd25c2  // fmopa za2.s, p1/M, p1/M, z14.h, z29.h\n"
      ".inst 0x81bd25e3  // fmopa za3.s, p1/M, p1/M, z15.h, z29.h\n"
      ".inst 0x81be2700  // fmopa za0.s, p1/M, p1/M, z24.h, z30.h\n"
      ".inst 0x81be2721  // fmopa za1.s, p1/M, p1/M, z25.h, z30.h\n"
      ".inst 0x81be2742  // fmopa za2.s, p1/M, p1/M, z26.h, z30.h\n"
      ".inst 0x81be2763  // fmopa za3.s, p1/M, p1/M, z27.h, z30.h\n"
      ".inst 0x81bf2680  // fmopa za0.s, p1/M, p1/M, z20.h, z31.h\n"
      ".inst 0x81bf26a1  // fmopa za1.s, p1/M, p1/M, z21.h, z31.h\n"
      ".inst 0x81bf26c2  // fmopa za2.s, p1/M, p1/M, z22.h, z31.h\n"
      ".inst 0x81bf26e3  // fmopa za3.s, p1/M, p1/M, z23.h, z31.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa040a768  // ld1h { z8.h-z11.h }, pn9.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #4\n"
      "ld1h { z6.h }, p1/Z, [x28]\n"
      "addvl x28, x28, #1\n"
      ".inst 0x81a62500  // fmopa za0.s, p1/M, p1/M, z8.h, z6.h\n"
      ".inst 0x81a62521  // fmopa za1.s, p1/M, p1/M, z9.h, z6.h\n"
      ".inst 0x81a62542  // fmopa za2.s, p1/M, p1/M, z10.h, z6.h\n"
      ".inst 0x81a62563  // fmopa za3.s, p1/M, p1/M, z11.h, z6.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c630  // ld1w { z16.s-z19.s }, pn9.b/Z, [x17]\n"
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa041c634  // ld1w { z20.s-z23.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c624  // ld1w { z4.s-z7.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840681  // mova za1h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xa060c61c  // st1w { z28.s-z31.s }, pn9.b, [x16]\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa061c60c  // st1w { z12.s-z15.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c618  // st1w { z24.s-z27.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c608  // st1w { z8.s-z11.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 12b\n"
      "b 30f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa060c610  // st1w { z16.s-z19.s }, pn9.b, [x16]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c600  // st1w { z0.s-z3.s }, pn9.b, [x16, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c618  // st1w { z24.s-z27.s }, pn9.b, [x16, #0x8, MUL VL]\n"
      ".inst 0xa063c60c  // st1w { z12.s-z15.s }, pn9.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 14b\n"
      "b 30f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "cntw x24\n"
      "ld1rh { z21.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "whilelt p0.s, x13, x10\n"
      "cmp x25, x24\n"
      "ld1rh { z20.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x24, LT\n"
      "mov x12, #0\n"
      "add x26, x26, x13, LSL #1\n"  // C += n
      "lsr x21, x22, #0x2\n"
      "madd x26, x14, x23, x26\n"  // C += m * ldc
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "fcvt z28.h, p1/m, z28.s\n"
      "fcvt z29.h, p1/m, z29.s\n"
      "cmp x12, x21, LSL #2\n"
      "fcvt z30.h, p1/m, z30.s\n"
      "fcvt z31.h, p1/m, z31.s\n"
      ".inst 0xc174cabc  // fclamp { z28.h-z31.h }, z21.h, z20.h\n"
      "st1h { z28.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z29.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z30.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z31.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fcvt z16.h, p1/m, z16.s\n"
      "fcvt z17.h, p1/m, z17.s\n"
      "fcvt z18.h, p1/m, z18.s\n"
      "fcvt z19.h, p1/m, z19.s\n"
      ".inst 0xc174cab0  // fclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "st1h { z16.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z17.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 18f\n"
      "st1h { z18.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x22, x25, x24, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "fcvt z0.h, p1/m, z0.s\n"
      "fcvt z1.h, p1/m, z1.s\n"
      "cmp x12, x21, LSL #2\n"
      "fcvt z2.h, p1/m, z2.s\n"
      "fcvt z3.h, p1/m, z3.s\n"
      ".inst 0xc174caa0  // fclamp { z0.h-z3.h }, z21.h, z20.h\n"
      "st1h { z0.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z1.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z2.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z3.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fcvt z24.h, p1/m, z24.s\n"
      "fcvt z25.h, p1/m, z25.s\n"
      "fcvt z26.h, p1/m, z26.s\n"
      "fcvt z27.h, p1/m, z27.s\n"
      ".inst 0xc174cab8  // fclamp { z24.h-z27.h }, z21.h, z20.h\n"
      "st1h { z24.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z25.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 21f\n"
      "st1h { z26.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "21:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x22, x25, x24, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 23f\n"
      "22:"  // Store to output array: Accumulator row 2 loop
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "fcvt z16.h, p1/m, z16.s\n"
      "fcvt z17.h, p1/m, z17.s\n"
      "cmp x12, x21, LSL #2\n"
      "fcvt z18.h, p1/m, z18.s\n"
      "fcvt z19.h, p1/m, z19.s\n"
      ".inst 0xc174cab0  // fclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "st1h { z16.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z17.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z18.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z19.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "blt 22b\n"
      "23:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 24f\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fcvt z28.h, p1/m, z28.s\n"
      "fcvt z29.h, p1/m, z29.s\n"
      "fcvt z30.h, p1/m, z30.s\n"
      "fcvt z31.h, p1/m, z31.s\n"
      ".inst 0xc174cabc  // fclamp { z28.h-z31.h }, z21.h, z20.h\n"
      "st1h { z28.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 24f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z29.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 24f\n"
      "st1h { z30.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "24:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x24\n"
      "mov x12, #0\n"
      "csel x20, x25, x24, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 26f\n"
      "25:"  // Store to output array: Accumulator row 3 loop
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "fcvt z28.h, p1/m, z28.s\n"
      "fcvt z29.h, p1/m, z29.s\n"
      "cmp x12, x21, LSL #2\n"
      "fcvt z30.h, p1/m, z30.s\n"
      "fcvt z31.h, p1/m, z31.s\n"
      ".inst 0xc174cabc  // fclamp { z28.h-z31.h }, z21.h, z20.h\n"
      "st1h { z28.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z29.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z30.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "st1h { z31.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "blt 25b\n"
      "26:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 27f\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      "fcvt z28.h, p1/m, z28.s\n"
      "fcvt z29.h, p1/m, z29.s\n"
      "fcvt z30.h, p1/m, z30.s\n"
      "fcvt z31.h, p1/m, z31.s\n"
      ".inst 0xc174cabc  // fclamp { z28.h-z31.h }, z21.h, z20.h\n"
      "st1h { z28.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 27f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z29.s }, p0, [x26]\n"
      "add x26, x26, x23\n"
      "beq 27f\n"
      "st1h { z30.s }, p0, [x26]\n"
      "27:"  // Store to output array: Accumulator row 3 oddments: End
      "28:"  // Store to output array: End
      "tbz x8, #0, 30f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "29:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c638  // ld1w { z24.s-z27.s }, pn9.b/Z, [x17]\n"
      ".inst 0xa041c62c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c620  // ld1w { z0.s-z3.s }, pn9.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c628  // ld1w { z8.s-z11.s }, pn9.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 29b\n"
      "30:"  // End block
      "incw x13\n"
      "cmp x13, x10\n"
      "blt 4b\n"
      "incw x14, ALL, MUL #4\n"
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
