//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include "kai/ops/gemm/kai_ops.hpp"

#include "kai/ops/bfloat.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

void sme2_interleaved_nomerge_bf16_mopa_2VLx1VL(const bfloat16 *const A, const bfloat16 *const B, bfloat16 *const C, int ldc, const int M, const int N, const int K, const bfloat16 *const bias, const Activation act, bool accumulate, bfloat16 *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const bfloat16 *const A,
      const bfloat16 *const B,
      bfloat16 *const C, const int ldc,
      const int M, const int N, const int K,
      const bfloat16 *const bias,
      const Activation act,
      bool accumulate,
      bfloat16 *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(K * sizeof(bfloat16)),
        C(C), ldcb(ldc * sizeof(bfloat16)),
        M(M), N(N), K(K),
        min(-static_cast<bfloat16>(std::numeric_limits<float>::infinity())),
        max(static_cast<bfloat16>(std::numeric_limits<float>::infinity())),
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
            this->max = static_cast<bfloat16>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<bfloat16>(0);
            break;
      }
    }

    const bfloat16 *const A;
    const bfloat16 *const B;
    const long kstride_bytes;
    bfloat16 *const C;
    const long ldcb;
    const long M, N, K;
    bfloat16 min = -static_cast<bfloat16>(std::numeric_limits<float>::infinity());
    bfloat16 max = static_cast<bfloat16>(std::numeric_limits<float>::infinity());

    const bfloat16 *const bias;


    bfloat16 *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x8, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x17, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x8, #0, 2f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040a22c  // ld1h { z12.h-z15.h }, pn8.b/Z, [x17]\n"
      ".inst 0xa041a234  // ld1h { z20.h-z23.h }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      "addvl x17, x17, #8\n"
      ".inst 0xc0440580  // mova za0h.h[x12, 0:3], { z12.h-z15.h }\n"
      ".inst 0xc0440682  // mova za1h.h[x12, 0:3], { z20.h-z23.h }\n"
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
      "whilelt p0.h, x13, x10\n"
      "tbnz x8, #0, 5f\n"
      "ldr x21, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x21, 6f\n"
      "mov x20, #0x3f80\n"
      "ld1h { z12.h }, p0/Z, [x21, x13, LSL #1]\n"
      "dup z13.h, w20\n"
      ".inst 0x81ac25a8  // bfmopa za0.h, p1/M, p1/M, z13.h, z12.h\n"
      ".inst 0x81ac25a9  // bfmopa za1.h, p1/M, p1/M, z13.h, z12.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x13\n"
      "mov x21, x14\n"
      "inch x20\n"
      "inch x21, ALL, MUL #2\n"
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
      ".inst 0xa040a36c  // ld1h { z12.h-z15.h }, pn8.b/Z, [x27]\n"
      ".inst 0xa141a372  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa040a380  // ld1h { z0.h-z3.h }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81a02588  // bfmopa za0.h, p1/M, p1/M, z12.h, z0.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81a025a9  // bfmopa za1.h, p1/M, p1/M, z13.h, z0.h\n"
      ".inst 0x81a125c8  // bfmopa za0.h, p1/M, p1/M, z14.h, z1.h\n"
      ".inst 0x81a125e9  // bfmopa za1.h, p1/M, p1/M, z15.h, z1.h\n"
      ".inst 0xa040a36c  // ld1h { z12.h-z15.h }, pn8.b/Z, [x27]\n"
      ".inst 0x81a22648  // bfmopa za0.h, p1/M, p1/M, z18.h, z2.h\n"
      ".inst 0x81a226c9  // bfmopa za1.h, p1/M, p1/M, z22.h, z2.h\n"
      ".inst 0x81a32748  // bfmopa za0.h, p1/M, p1/M, z26.h, z3.h\n"
      ".inst 0x81a327c9  // bfmopa za1.h, p1/M, p1/M, z30.h, z3.h\n"
      ".inst 0xa141a372  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa040a380  // ld1h { z0.h-z3.h }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81a02588  // bfmopa za0.h, p1/M, p1/M, z12.h, z0.h\n"
      ".inst 0x81a025a9  // bfmopa za1.h, p1/M, p1/M, z13.h, z0.h\n"
      ".inst 0x81a125c8  // bfmopa za0.h, p1/M, p1/M, z14.h, z1.h\n"
      ".inst 0x81a125e9  // bfmopa za1.h, p1/M, p1/M, z15.h, z1.h\n"
      ".inst 0x81a22648  // bfmopa za0.h, p1/M, p1/M, z18.h, z2.h\n"
      ".inst 0x81a226c9  // bfmopa za1.h, p1/M, p1/M, z22.h, z2.h\n"
      ".inst 0x81a32748  // bfmopa za0.h, p1/M, p1/M, z26.h, z3.h\n"
      ".inst 0x81a327c9  // bfmopa za1.h, p1/M, p1/M, z30.h, z3.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa040237a  // ld1h { z26.h-z27.h }, pn8.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #2\n"
      "ld1h { z1.h }, p1/Z, [x28]\n"
      "addvl x28, x28, #1\n"
      ".inst 0x81a12748  // bfmopa za0.h, p1/M, p1/M, z26.h, z1.h\n"
      ".inst 0x81a12769  // bfmopa za1.h, p1/M, p1/M, z27.h, z1.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040a228  // ld1h { z8.h-z11.h }, pn8.b/Z, [x17]\n"
      ".inst 0xc0460418  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc046045c  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]\n"
      ".inst 0xa041a22c  // ld1h { z12.h-z15.h }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      "addvl x17, x17, #8\n"
      ".inst 0xa060a218  // st1h { z24.h-z27.h }, pn8.b, [x16]\n"
      ".inst 0xc0440500  // mova za0h.h[x12, 0:3], { z8.h-z11.h }\n"
      ".inst 0xa061a21c  // st1h { z28.h-z31.h }, pn8.b, [x16, #0x4, MUL VL]\n"
      "addvl x16, x16, #8\n"
      ".inst 0xc0440582  // mova za1h.h[x12, 0:3], { z12.h-z15.h }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 12b\n"
      "b 31f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cnth x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0460418  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc046044c  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa060a218  // st1h { z24.h-z27.h }, pn8.b, [x16]\n"
      "cmp x12, x20\n"
      ".inst 0xa061a20c  // st1h { z12.h-z15.h }, pn8.b, [x16, #0x4, MUL VL]\n"
      "addvl x16, x16, #8\n"
      "blt 14b\n"
      "b 31f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "add x26, x26, x13, LSL #1\n"  // C += n
      "madd x26, x14, x24, x26\n"  // C += m * ldc
      "tbz x8, #2, 22f\n"
      "cnth x23\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0460410  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0460400  // mova { z0.h-z3.h }, za0h.h[x12, 0:3]\n"
      "st1h { z0.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z1.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "st1h { z2.h }, p0, [x26]\n"
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
      ".inst 0xc0460450  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 19b\n"
      "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc046044c  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]\n"
      "st1h { z12.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z13.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "st1h { z14.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "21:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 22f\n"
      "b 29f\n"
      "22:"  // Store to output array: Skip activation: End
      "cnth x23\n"
      "ld1rh { z21.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "ld1rh { z20.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 24f\n"
      "23:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0460410  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc134cab0  // bfclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "cmp x12, x21, LSL #2\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 23b\n"
      "24:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 25f\n"
      ".inst 0xc0460410  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc134cab0  // bfclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      "st1h { z18.h }, p0, [x26]\n"
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
      ".inst 0xc0460450  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc134cab0  // bfclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "cmp x12, x21, LSL #2\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z18.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1h { z19.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 26b\n"
      "27:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 28f\n"
      ".inst 0xc0460450  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc134cab0  // bfclamp { z16.h-z19.h }, z21.h, z20.h\n"
      "st1h { z16.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      "subs x20, x20, #0x1\n"
      "st1h { z17.h }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      "st1h { z18.h }, p0, [x26]\n"
      "28:"  // Store to output array: Accumulator row 1 oddments: End
      "29:"  // Store to output array: End
      "tbz x8, #0, 31f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "30:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040a224  // ld1h { z4.h-z7.h }, pn8.b/Z, [x17]\n"
      ".inst 0xa041a228  // ld1h { z8.h-z11.h }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      "addvl x17, x17, #8\n"
      ".inst 0xc0440480  // mova za0h.h[x12, 0:3], { z4.h-z7.h }\n"
      ".inst 0xc0440502  // mova za1h.h[x12, 0:3], { z8.h-z11.h }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 30b\n"
      "31:"  // End block
      "inch x13\n"
      "cmp x13, x10\n"
      "blt 4b\n"
      "inch x14, ALL, MUL #2\n"
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
