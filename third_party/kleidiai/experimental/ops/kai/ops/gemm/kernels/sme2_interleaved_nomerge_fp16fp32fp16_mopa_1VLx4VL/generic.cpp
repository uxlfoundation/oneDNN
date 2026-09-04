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

void sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c5d8  // ld1w { z24.s-z27.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x11, [%x[args], %[offsetof_K]]\n"
      "mov x10, #0\n"
      "mov x9, #0\n"
      "ldr w28, [%x[args], %[offsetof_M]]\n"
      "ldr w27, [%x[args], %[offsetof_N]]\n"
      "add x11, x11, #0x1\n"
      "ldr x26, [%x[args], %[offsetof_A]]\n"
      "lsr x11, x11, #0x1\n"
      "3:"  // M loop
      "ldr x25, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x24, x26\n"
      "tbnz x15, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z5.h, #0.0\n"
      ".inst 0x257b4530  // whilelt pn8.h, x9, x27, VLx2\n"
      "fmov z24.h, #1.0\n"
      ".inst 0xa109229c  // ldnt1h { z20.h, z28.h }, p8/Z, [x20, x9, LSL #1]\n"
      "zip1 z27.h, z20.h, z5.h\n"
      "zip2 z10.h, z20.h, z5.h\n"
      "zip1 z20.h, z28.h, z5.h\n"
      "zip2 z18.h, z28.h, z5.h\n"
      ".inst 0x81bb0300  // fmopa za0.s, p0/M, p0/M, z24.h, z27.h\n"
      ".inst 0x81aa0301  // fmopa za1.s, p0/M, p0/M, z24.h, z10.h\n"
      ".inst 0x81b40302  // fmopa za2.s, p0/M, p0/M, z24.h, z20.h\n"
      ".inst 0x81b20303  // fmopa za3.s, p0/M, p0/M, z24.h, z18.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x9\n"
      "mov x21, x10\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x27\n"
      "mov x20, x15\n"
      "csel x21, x10, x21, LT\n"
      "bfm x15, XZR, #0, #0  // bfc x15, #0, #0x1\n"
      "cmp x21, x28\n"
      "csel x15, x20, x15, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x11, #0x2\n"
      "and x20, x11, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa140a701  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa140a722  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x25]\n"
      ".inst 0xa041a73c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0xa042a730  // ld1h { z16.h-z19.h }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0xa043a734  // ld1h { z20.h-z23.h }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81a20020  // fmopa za0.s, p0/M, p0/M, z1.h, z2.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81a60021  // fmopa za1.s, p0/M, p0/M, z1.h, z6.h\n"
      ".inst 0x81aa0022  // fmopa za2.s, p0/M, p0/M, z1.h, z10.h\n"
      ".inst 0x81ae0023  // fmopa za3.s, p0/M, p0/M, z1.h, z14.h\n"
      ".inst 0xa140a722  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x25]\n"
      ".inst 0x81bc00a0  // fmopa za0.s, p0/M, p0/M, z5.h, z28.h\n"
      ".inst 0x81bd00a1  // fmopa za1.s, p0/M, p0/M, z5.h, z29.h\n"
      ".inst 0x81be00a2  // fmopa za2.s, p0/M, p0/M, z5.h, z30.h\n"
      ".inst 0x81bf00a3  // fmopa za3.s, p0/M, p0/M, z5.h, z31.h\n"
      ".inst 0xa041a73c  // ld1h { z28.h-z31.h }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0x81b00120  // fmopa za0.s, p0/M, p0/M, z9.h, z16.h\n"
      ".inst 0x81b10121  // fmopa za1.s, p0/M, p0/M, z9.h, z17.h\n"
      ".inst 0x81b20122  // fmopa za2.s, p0/M, p0/M, z9.h, z18.h\n"
      ".inst 0x81b30123  // fmopa za3.s, p0/M, p0/M, z9.h, z19.h\n"
      ".inst 0xa042a730  // ld1h { z16.h-z19.h }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0x81b401a0  // fmopa za0.s, p0/M, p0/M, z13.h, z20.h\n"
      ".inst 0x81b501a1  // fmopa za1.s, p0/M, p0/M, z13.h, z21.h\n"
      ".inst 0x81b601a2  // fmopa za2.s, p0/M, p0/M, z13.h, z22.h\n"
      ".inst 0x81b701a3  // fmopa za3.s, p0/M, p0/M, z13.h, z23.h\n"
      ".inst 0xa140a701  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa043a734  // ld1h { z20.h-z23.h }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81a20020  // fmopa za0.s, p0/M, p0/M, z1.h, z2.h\n"
      ".inst 0x81a60021  // fmopa za1.s, p0/M, p0/M, z1.h, z6.h\n"
      ".inst 0x81aa0022  // fmopa za2.s, p0/M, p0/M, z1.h, z10.h\n"
      ".inst 0x81ae0023  // fmopa za3.s, p0/M, p0/M, z1.h, z14.h\n"
      ".inst 0x81bc00a0  // fmopa za0.s, p0/M, p0/M, z5.h, z28.h\n"
      ".inst 0x81bd00a1  // fmopa za1.s, p0/M, p0/M, z5.h, z29.h\n"
      ".inst 0x81be00a2  // fmopa za2.s, p0/M, p0/M, z5.h, z30.h\n"
      ".inst 0x81bf00a3  // fmopa za3.s, p0/M, p0/M, z5.h, z31.h\n"
      ".inst 0x81b00120  // fmopa za0.s, p0/M, p0/M, z9.h, z16.h\n"
      ".inst 0x81b10121  // fmopa za1.s, p0/M, p0/M, z9.h, z17.h\n"
      ".inst 0x81b20122  // fmopa za2.s, p0/M, p0/M, z9.h, z18.h\n"
      ".inst 0x81b30123  // fmopa za3.s, p0/M, p0/M, z9.h, z19.h\n"
      ".inst 0x81b401a0  // fmopa za0.s, p0/M, p0/M, z13.h, z20.h\n"
      ".inst 0x81b501a1  // fmopa za1.s, p0/M, p0/M, z13.h, z21.h\n"
      ".inst 0x81b601a2  // fmopa za2.s, p0/M, p0/M, z13.h, z22.h\n"
      ".inst 0x81b701a3  // fmopa za3.s, p0/M, p0/M, z13.h, z23.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1h { z23.h }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa040a72c  // ld1h { z12.h-z15.h }, pn9.b/Z, [x25]\n"
      "addvl x25, x25, #4\n"
      ".inst 0x81ac02e0  // fmopa za0.s, p0/M, p0/M, z23.h, z12.h\n"
      ".inst 0x81ad02e1  // fmopa za1.s, p0/M, p0/M, z23.h, z13.h\n"
      ".inst 0x81ae02e2  // fmopa za2.s, p0/M, p0/M, z23.h, z14.h\n"
      ".inst 0x81af02e3  // fmopa za3.s, p0/M, p0/M, z23.h, z15.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x15, #1, 15f\n"
      "tbz x15, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      ".inst 0xa041c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa042c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa060c5b4  // st1w { z20.s-z23.s }, pn9.b, [x13]\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa061c5b8  // st1w { z24.s-z27.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5bc  // st1w { z28.s-z31.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 12b\n"
      "b 19f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 14b\n"
      "b 19f\n"
      "15:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x22, x28, x10\n"
      "cntw x21\n"
      "ld1rh { z21.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x20, [%x[args], %[offsetof_ldcb]]\n"
      ".inst 0x257b4530  // whilelt pn8.h, x9, x27, VLx2\n"
      "cmp x22, x21\n"
      "ld1rh { z20.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0\n"
      "csel x22, x22, x21, LT\n"
      "add x23, x23, x9, LSL #1\n"  // C += n
      "madd x23, x10, x20, x23\n"  // C += m * ldc
      "16:"  // Store to output array: Accumulator loop
      ".inst 0xc0060410  // mova { z16.b-z19.b }, za0h.b[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc120e20e  // fcvt z14.h, { z16.s-z17.s }\n"
      ".inst 0xc120e24f  // fcvt z15.h, { z18.s-z19.s }\n"
      "cmp x12, x22, LSL #2\n"
      ".inst 0xc174c2ae  // fclamp { z14.h-z15.h }, z21.h, z20.h\n"
      ".inst 0xa06022ee  // st1h { z14.h-z15.h }, p8, [x23]\n"
      "add x23, x23, x20\n"
      "blt 16b\n"
      "tbz x15, #0, 19f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "18:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5d4  // ld1w { z20.s-z23.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 18b\n"
      "19:"  // End block
      "incw x9, ALL, MUL #4\n"
      "cmp x9, x27\n"
      "blt 4b\n"
      "incw x10\n"
      "mov x9, #0\n"
      "cmp x10, x28\n"
      "mov x26, x24\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
