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

void sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
      "ldr x17, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x17, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c21c  // ld1w { z28.s-z31.s }, pn8.b/Z, [x16]\n"
      ".inst 0xa041c208  // ld1w { z8.s-z11.s }, pn8.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c204  // ld1w { z4.s-z7.s }, pn8.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c214  // ld1w { z20.s-z23.s }, pn8.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840501  // mova za1h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x14, [%x[args], %[offsetof_K]]\n"
      "mov x13, #0\n"
      "mov x11, #0\n"
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "ldr w9, [%x[args], %[offsetof_N]]\n"
      "add x14, x14, #0x1\n"
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "lsr x14, x14, #0x1\n"
      "3:"  // M loop
      "ldr x27, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x26, x28\n"
      "tbnz x17, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z4.h, #0.0\n"
      "whilelt p0.h, x11, x9\n"
      "fmov z24.h, #1.0\n"
      "ld1h { z22.h }, p0/Z, [x20, x11, LSL #1]\n"
      "zip1 z0.h, z22.h, z4.h\n"
      "zip2 z9.h, z22.h, z4.h\n"
      ".inst 0x81a02700  // fmopa za0.s, p1/M, p1/M, z24.h, z0.h\n"
      ".inst 0x81a92701  // fmopa za1.s, p1/M, p1/M, z24.h, z9.h\n"
      ".inst 0x81a02702  // fmopa za2.s, p1/M, p1/M, z24.h, z0.h\n"
      ".inst 0x81a92703  // fmopa za3.s, p1/M, p1/M, z24.h, z9.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x11\n"
      "mov x21, x13\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
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
      ".inst 0xa140a350  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x26]\n"
      ".inst 0xa141a340  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      "addvl x26, x26, #8\n"
      ".inst 0xa140a373  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x27]\n"
      ".inst 0xa141a372  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81b32600  // fmopa za0.s, p1/M, p1/M, z16.h, z19.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81b72601  // fmopa za1.s, p1/M, p1/M, z16.h, z23.h\n"
      ".inst 0x81b32682  // fmopa za2.s, p1/M, p1/M, z20.h, z19.h\n"
      ".inst 0x81b72683  // fmopa za3.s, p1/M, p1/M, z20.h, z23.h\n"
      ".inst 0x81bb2700  // fmopa za0.s, p1/M, p1/M, z24.h, z27.h\n"
      ".inst 0x81bf2701  // fmopa za1.s, p1/M, p1/M, z24.h, z31.h\n"
      ".inst 0x81bb2782  // fmopa za2.s, p1/M, p1/M, z28.h, z27.h\n"
      ".inst 0x81bf2783  // fmopa za3.s, p1/M, p1/M, z28.h, z31.h\n"
      ".inst 0xa140a350  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x26]\n"
      ".inst 0x81b22400  // fmopa za0.s, p1/M, p1/M, z0.h, z18.h\n"
      ".inst 0xa140a373  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x27]\n"
      ".inst 0x81b62401  // fmopa za1.s, p1/M, p1/M, z0.h, z22.h\n"
      ".inst 0x81b22482  // fmopa za2.s, p1/M, p1/M, z4.h, z18.h\n"
      ".inst 0x81b62483  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h\n"
      ".inst 0x81ba2500  // fmopa za0.s, p1/M, p1/M, z8.h, z26.h\n"
      ".inst 0x81be2501  // fmopa za1.s, p1/M, p1/M, z8.h, z30.h\n"
      ".inst 0x81ba2582  // fmopa za2.s, p1/M, p1/M, z12.h, z26.h\n"
      ".inst 0x81be2583  // fmopa za3.s, p1/M, p1/M, z12.h, z30.h\n"
      ".inst 0xa141a340  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      "addvl x26, x26, #8\n"
      ".inst 0xa141a372  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81b32600  // fmopa za0.s, p1/M, p1/M, z16.h, z19.h\n"
      ".inst 0x81b72601  // fmopa za1.s, p1/M, p1/M, z16.h, z23.h\n"
      ".inst 0x81b32682  // fmopa za2.s, p1/M, p1/M, z20.h, z19.h\n"
      ".inst 0x81b72683  // fmopa za3.s, p1/M, p1/M, z20.h, z23.h\n"
      ".inst 0x81bb2700  // fmopa za0.s, p1/M, p1/M, z24.h, z27.h\n"
      ".inst 0x81bf2701  // fmopa za1.s, p1/M, p1/M, z24.h, z31.h\n"
      ".inst 0x81bb2782  // fmopa za2.s, p1/M, p1/M, z28.h, z27.h\n"
      ".inst 0x81bf2783  // fmopa za3.s, p1/M, p1/M, z28.h, z31.h\n"
      ".inst 0x81b22400  // fmopa za0.s, p1/M, p1/M, z0.h, z18.h\n"
      ".inst 0x81b62401  // fmopa za1.s, p1/M, p1/M, z0.h, z22.h\n"
      ".inst 0x81b22482  // fmopa za2.s, p1/M, p1/M, z4.h, z18.h\n"
      ".inst 0x81b62483  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h\n"
      ".inst 0x81ba2500  // fmopa za0.s, p1/M, p1/M, z8.h, z26.h\n"
      ".inst 0x81be2501  // fmopa za1.s, p1/M, p1/M, z8.h, z30.h\n"
      ".inst 0x81ba2582  // fmopa za2.s, p1/M, p1/M, z12.h, z26.h\n"
      ".inst 0x81be2583  // fmopa za3.s, p1/M, p1/M, z12.h, z30.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa0402350  // ld1h { z16.h-z17.h }, pn8.b/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #2\n"
      ".inst 0xa0402364  // ld1h { z4.h-z5.h }, pn8.b/Z, [x27]\n"
      "addvl x27, x27, #2\n"
      ".inst 0x81a42600  // fmopa za0.s, p1/M, p1/M, z16.h, z4.h\n"
      ".inst 0x81a52601  // fmopa za1.s, p1/M, p1/M, z16.h, z5.h\n"
      ".inst 0x81a42622  // fmopa za2.s, p1/M, p1/M, z17.h, z4.h\n"
      ".inst 0x81a52623  // fmopa za3.s, p1/M, p1/M, z17.h, z5.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x17, #1, 15f\n"
      "tbz x17, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c214  // ld1w { z20.s-z23.s }, pn8.b/Z, [x16]\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xa041c204  // ld1w { z4.s-z7.s }, pn8.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c20c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c200  // ld1w { z0.s-z3.s }, pn8.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa060c1f0  // st1w { z16.s-z19.s }, pn8.b, [x15]\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa061c1fc  // st1w { z28.s-z31.s }, pn8.b, [x15, #0x4, MUL VL]\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c1f8  // st1w { z24.s-z27.s }, pn8.b, [x15, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c1e8  // st1w { z8.s-z11.s }, pn8.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 12b\n"
      "b 19f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c1e8  // st1w { z8.s-z11.s }, pn8.b, [x15]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c1ec  // st1w { z12.s-z15.s }, pn8.b, [x15, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c1fc  // st1w { z28.s-z31.s }, pn8.b, [x15, #0x8, MUL VL]\n"
      ".inst 0xa063c1f0  // st1w { z16.s-z19.s }, pn8.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 14b\n"
      "b 19f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x10, x13\n"
      "cntw x23, ALL, MUL #2\n"
      "ld1rh { z18.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "whilelt p0.h, x11, x9\n"
      "cmp x24, x23\n"
      "ld1rh { z17.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0\n"
      "mov x21, #0\n"
      "add x25, x25, x11, LSL #1\n"  // C += n
      "mov x20, #0x2\n"
      "madd x25, x13, x22, x25\n"  // C += m * ldc
      "csel x24, x24, x23, LT\n"
      "16:"  // Store to output array: Accumulator loop
      ".inst 0xc006000e  // mova { z14.b-z15.b }, za0h.b[x12, 0:1]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x23, LSL #1\n"
      "add x21, x21, #0x1\n"
      ".inst 0xc120e1d0  // fcvt z16.h, { z14.s-z15.s }\n"
      "csel x12, x12, x20, LT\n"
      "cmp x21, x24\n"
      ".inst 0x64712650  // fclamp z16.h, z18.h, z17.h\n"
      "st1h { z16.h }, p0, [x25]\n"
      "add x25, x25, x22\n"
      "blt 16b\n"
      "tbz x17, #0, 19f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "18:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c20c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x16]\n"
      ".inst 0xa041c204  // ld1w { z4.s-z7.s }, pn8.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c208  // ld1w { z8.s-z11.s }, pn8.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c21c  // ld1w { z28.s-z31.s }, pn8.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840502  // mova za2h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xc0840783  // mova za3h.s[x12], { z28.s-z31.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 18b\n"
      "19:"  // End block
      "incw x11, ALL, MUL #2\n"
      "cmp x11, x9\n"
      "blt 4b\n"
      "incw x13, ALL, MUL #2\n"
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
