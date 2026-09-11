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

void sme2_interleaved_nomerge_fp16_mopa_1VLx2VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, __fp16 *const accumulator_buffer)
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
      __fp16 *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(K * sizeof(__fp16)),
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


    __fp16 *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x17, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x17, #0, 2f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040a608  // ld1h { z8.h-z11.h }, pn9.b/Z, [x16]\n"
      ".inst 0xa041a61c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      "addvl x16, x16, #8\n"
      ".inst 0xc0440500  // mova za0h.h[x12, 0:3], { z8.h-z11.h }\n"
      ".inst 0xc0440782  // mova za1h.h[x12, 0:3], { z28.h-z31.h }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x14, [%x[args], %[offsetof_K]]\n"
      "mov x13, #0\n"
      "mov x11, #0\n"
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "ldr w9, [%x[args], %[offsetof_N]]\n"
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "3:"  // M loop
      "ldr x27, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x26, x28\n"
      ".inst 0x25694570  // whilelt pn8.h, x11, x9, VLx2\n"
      "tbnz x17, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z8.h, #1.0\n"
      ".inst 0xa00b2280  // ld1h { z0.h-z1.h }, p8/Z, [x20, x11, LSL #1]\n"
      ".inst 0x81800108  // fmopa za0.h, p0/M, p0/M, z8.h, z0.h\n"
      ".inst 0x81810109  // fmopa za1.h, p0/M, p0/M, z8.h, z1.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x11\n"
      "mov x21, x13\n"
      "inch x20, ALL, MUL #2\n"
      "inch x21\n"
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
      ".inst 0xa040a754  // ld1h { z20.h-z23.h }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa140a763  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x27]\n"
      ".inst 0xa141a761  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81830288  // fmopa za0.h, p0/M, p0/M, z20.h, z3.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81870289  // fmopa za1.h, p0/M, p0/M, z20.h, z7.h\n"
      ".inst 0x818b02a8  // fmopa za0.h, p0/M, p0/M, z21.h, z11.h\n"
      ".inst 0x818f02a9  // fmopa za1.h, p0/M, p0/M, z21.h, z15.h\n"
      ".inst 0xa140a763  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x27]\n"
      ".inst 0x818102c8  // fmopa za0.h, p0/M, p0/M, z22.h, z1.h\n"
      ".inst 0x818502c9  // fmopa za1.h, p0/M, p0/M, z22.h, z5.h\n"
      ".inst 0x818902e8  // fmopa za0.h, p0/M, p0/M, z23.h, z9.h\n"
      ".inst 0x818d02e9  // fmopa za1.h, p0/M, p0/M, z23.h, z13.h\n"
      ".inst 0xa040a754  // ld1h { z20.h-z23.h }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa141a761  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81830288  // fmopa za0.h, p0/M, p0/M, z20.h, z3.h\n"
      ".inst 0x81870289  // fmopa za1.h, p0/M, p0/M, z20.h, z7.h\n"
      ".inst 0x818b02a8  // fmopa za0.h, p0/M, p0/M, z21.h, z11.h\n"
      ".inst 0x818f02a9  // fmopa za1.h, p0/M, p0/M, z21.h, z15.h\n"
      ".inst 0x818102c8  // fmopa za0.h, p0/M, p0/M, z22.h, z1.h\n"
      ".inst 0x818502c9  // fmopa za1.h, p0/M, p0/M, z22.h, z5.h\n"
      ".inst 0x818902e8  // fmopa za0.h, p0/M, p0/M, z23.h, z9.h\n"
      ".inst 0x818d02e9  // fmopa za1.h, p0/M, p0/M, z23.h, z13.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1h { z16.h }, p0/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #1\n"
      ".inst 0xa1402767  // ld1h { z7.h, z15.h }, pn9.b/Z, [x27]\n"
      "addvl x27, x27, #2\n"
      ".inst 0x81870208  // fmopa za0.h, p0/M, p0/M, z16.h, z7.h\n"
      ".inst 0x818f0209  // fmopa za1.h, p0/M, p0/M, z16.h, z15.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x17, #1, 15f\n"
      "tbz x17, #0, 13f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040a608  // ld1h { z8.h-z11.h }, pn9.b/Z, [x16]\n"
      ".inst 0xc046041c  // mova { z28.h-z31.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc0460450  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]\n"
      ".inst 0xa041a60c  // ld1h { z12.h-z15.h }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      "addvl x16, x16, #8\n"
      ".inst 0xa060a5fc  // st1h { z28.h-z31.h }, pn9.b, [x15]\n"
      ".inst 0xc0440500  // mova za0h.h[x12, 0:3], { z8.h-z11.h }\n"
      ".inst 0xa061a5f0  // st1h { z16.h-z19.h }, pn9.b, [x15, #0x4, MUL VL]\n"
      "addvl x15, x15, #8\n"
      ".inst 0xc0440582  // mova za1h.h[x12, 0:3], { z12.h-z15.h }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 12b\n"
      "b 25f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cnth x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc046040c  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc0460444  // mova { z4.h-z7.h }, za1h.h[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa060a5ec  // st1h { z12.h-z15.h }, pn9.b, [x15]\n"
      "cmp x12, x20\n"
      ".inst 0xa061a5e4  // st1h { z4.h-z7.h }, pn9.b, [x15, #0x4, MUL VL]\n"
      "addvl x15, x15, #8\n"
      "blt 14b\n"
      "b 25f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x10, x13\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "add x25, x25, x11, LSL #1\n"  // C += n
      "madd x25, x13, x23, x25\n"  // C += m * ldc
      "tbz x17, #2, 19f\n"
      "cnth x20\n"
      "mov x12, #0\n"
      "cmp x24, x20\n"
      "csel x22, x24, x20, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0460410  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc0460458  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]\n"
      ".inst 0xa1602330  // st1h { z16.h, z24.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa1602331  // st1h { z17.h, z25.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1602332  // st1h { z18.h, z26.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa1602333  // st1h { z19.h, z27.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0460404  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc046044c  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]\n"
      ".inst 0xa1602324  // st1h { z4.h, z12.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1602325  // st1h { z5.h, z13.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      ".inst 0xa1602326  // st1h { z6.h, z14.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x24, x24, x22\n"
      "beq 19f\n"
      "b 23f\n"
      "19:"  // Store to output array: Skip activation: End
      "cnth x20\n"
      "ld1rh { z21.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x24, x20\n"
      "ld1rh { z20.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0460410  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc0460458  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]\n"
      ".inst 0xc174cab0  // fclamp { z16.h-z19.h }, z21.h, z20.h\n"
      ".inst 0xc174cab8  // fclamp { z24.h-z27.h }, z21.h, z20.h\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa1602330  // st1h { z16.h, z24.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa1602331  // st1h { z17.h, z25.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa1602332  // st1h { z18.h, z26.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa1602333  // st1h { z19.h, z27.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 20b\n"
      "21:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 22f\n"
      ".inst 0xc0460404  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]\n"
      ".inst 0xc046044c  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc174caa4  // fclamp { z4.h-z7.h }, z21.h, z20.h\n"
      ".inst 0xc174caac  // fclamp { z12.h-z15.h }, z21.h, z20.h\n"
      ".inst 0xa1602324  // st1h { z4.h, z12.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa1602325  // st1h { z5.h, z13.h }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      ".inst 0xa1602326  // st1h { z6.h, z14.h }, p8, [x25]\n"
      "22:"  // Store to output array: Accumulator row 0 oddments: End
      "23:"  // Store to output array: End
      "tbz x17, #0, 25f\n"
      "mov x12, #0\n"
      "cnth x20\n"
      "24:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040a610  // ld1h { z16.h-z19.h }, pn9.b/Z, [x16]\n"
      ".inst 0xa041a61c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      "addvl x16, x16, #8\n"
      ".inst 0xc0440600  // mova za0h.h[x12, 0:3], { z16.h-z19.h }\n"
      ".inst 0xc0440782  // mova za1h.h[x12, 0:3], { z28.h-z31.h }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 24b\n"
      "25:"  // End block
      "inch x11, ALL, MUL #2\n"
      "cmp x11, x9\n"
      "blt 4b\n"
      "inch x13\n"
      "mov x11, #0\n"
      "cmp x13, x10\n"
      "mov x28, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
