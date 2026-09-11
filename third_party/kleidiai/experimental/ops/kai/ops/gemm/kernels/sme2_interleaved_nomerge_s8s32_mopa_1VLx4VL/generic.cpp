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

void sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
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
      ".inst 0xa040c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5d4  // ld1w { z20.s-z23.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x11, [%x[args], %[offsetof_K]]\n"
      "mov x10, #0\n"
      "mov x9, #0\n"
      "ldr w28, [%x[args], %[offsetof_M]]\n"
      "ldr w27, [%x[args], %[offsetof_N]]\n"
      "add x11, x11, #0x3\n"
      "ldr x26, [%x[args], %[offsetof_A]]\n"
      "lsr x11, x11, #0x2\n"
      "3:"  // M loop
      "ldr x25, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x24, x26\n"
      ".inst 0x25bb6530  // whilelt pn8.s, x9, x27, VLx4\n"
      "tbnz x15, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      ".inst 0xa109c290  // ld1w { z16.s, z20.s, z24.s, z28.s }, p8/Z, [x20, x9, LSL #2]\n"
      ".inst 0xc0900200  // addha za0.s, p0/M, p0/M, z16.s\n"
      ".inst 0xc0900281  // addha za1.s, p0/M, p0/M, z20.s\n"
      ".inst 0xc0900302  // addha za2.s, p0/M, p0/M, z24.s\n"
      ".inst 0xc0900383  // addha za3.s, p0/M, p0/M, z28.s\n"
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
      ".inst 0xa040870c  // ld1b { z12.b-z15.b }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa0408730  // ld1b { z16.b-z19.b }, pn9.b/Z, [x25]\n"
      ".inst 0xa0418738  // ld1b { z24.b-z27.b }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0xa042873c  // ld1b { z28.b-z31.b }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0xa0438724  // ld1b { z4.b-z7.b }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0900180  // smopa za0.s, p0/M, p0/M, z12.b, z16.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0910181  // smopa za1.s, p0/M, p0/M, z12.b, z17.b\n"
      ".inst 0xa0920182  // smopa za2.s, p0/M, p0/M, z12.b, z18.b\n"
      ".inst 0xa0930183  // smopa za3.s, p0/M, p0/M, z12.b, z19.b\n"
      ".inst 0xa0408730  // ld1b { z16.b-z19.b }, pn9.b/Z, [x25]\n"
      ".inst 0xa09801a0  // smopa za0.s, p0/M, p0/M, z13.b, z24.b\n"
      ".inst 0xa09901a1  // smopa za1.s, p0/M, p0/M, z13.b, z25.b\n"
      ".inst 0xa09a01a2  // smopa za2.s, p0/M, p0/M, z13.b, z26.b\n"
      ".inst 0xa09b01a3  // smopa za3.s, p0/M, p0/M, z13.b, z27.b\n"
      ".inst 0xa0418738  // ld1b { z24.b-z27.b }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0xa09c01c0  // smopa za0.s, p0/M, p0/M, z14.b, z28.b\n"
      ".inst 0xa09d01c1  // smopa za1.s, p0/M, p0/M, z14.b, z29.b\n"
      ".inst 0xa09e01c2  // smopa za2.s, p0/M, p0/M, z14.b, z30.b\n"
      ".inst 0xa09f01c3  // smopa za3.s, p0/M, p0/M, z14.b, z31.b\n"
      ".inst 0xa042873c  // ld1b { z28.b-z31.b }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0xa08401e0  // smopa za0.s, p0/M, p0/M, z15.b, z4.b\n"
      ".inst 0xa08501e1  // smopa za1.s, p0/M, p0/M, z15.b, z5.b\n"
      ".inst 0xa08601e2  // smopa za2.s, p0/M, p0/M, z15.b, z6.b\n"
      ".inst 0xa08701e3  // smopa za3.s, p0/M, p0/M, z15.b, z7.b\n"
      ".inst 0xa040870c  // ld1b { z12.b-z15.b }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa0438724  // ld1b { z4.b-z7.b }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0900180  // smopa za0.s, p0/M, p0/M, z12.b, z16.b\n"
      ".inst 0xa0910181  // smopa za1.s, p0/M, p0/M, z12.b, z17.b\n"
      ".inst 0xa0920182  // smopa za2.s, p0/M, p0/M, z12.b, z18.b\n"
      ".inst 0xa0930183  // smopa za3.s, p0/M, p0/M, z12.b, z19.b\n"
      ".inst 0xa09801a0  // smopa za0.s, p0/M, p0/M, z13.b, z24.b\n"
      ".inst 0xa09901a1  // smopa za1.s, p0/M, p0/M, z13.b, z25.b\n"
      ".inst 0xa09a01a2  // smopa za2.s, p0/M, p0/M, z13.b, z26.b\n"
      ".inst 0xa09b01a3  // smopa za3.s, p0/M, p0/M, z13.b, z27.b\n"
      ".inst 0xa09c01c0  // smopa za0.s, p0/M, p0/M, z14.b, z28.b\n"
      ".inst 0xa09d01c1  // smopa za1.s, p0/M, p0/M, z14.b, z29.b\n"
      ".inst 0xa09e01c2  // smopa za2.s, p0/M, p0/M, z14.b, z30.b\n"
      ".inst 0xa09f01c3  // smopa za3.s, p0/M, p0/M, z14.b, z31.b\n"
      ".inst 0xa08401e0  // smopa za0.s, p0/M, p0/M, z15.b, z4.b\n"
      ".inst 0xa08501e1  // smopa za1.s, p0/M, p0/M, z15.b, z5.b\n"
      ".inst 0xa08601e2  // smopa za2.s, p0/M, p0/M, z15.b, z6.b\n"
      ".inst 0xa08701e3  // smopa za3.s, p0/M, p0/M, z15.b, z7.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z4.b }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa0408730  // ld1b { z16.b-z19.b }, pn9.b/Z, [x25]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa0900080  // smopa za0.s, p0/M, p0/M, z4.b, z16.b\n"
      ".inst 0xa0910081  // smopa za1.s, p0/M, p0/M, z4.b, z17.b\n"
      ".inst 0xa0920082  // smopa za2.s, p0/M, p0/M, z4.b, z18.b\n"
      ".inst 0xa0930083  // smopa za3.s, p0/M, p0/M, z4.b, z19.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x15, #1, 15f\n"
      "tbz x15, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5d4  // ld1w { z20.s-z23.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0860418  // mova { z24.s-z27.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xa041c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa060c5b8  // st1w { z24.s-z27.s }, pn9.b, [x13]\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa061c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5bc  // st1w { z28.s-z31.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 12b\n"
      "b 21f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c5a0  // st1w { z0.s-z3.s }, pn9.b, [x13]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5b8  // st1w { z24.s-z27.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 14b\n"
      "b 21f\n"
      "15:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x28, x10\n"
      "cntw x20\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x21, x20\n"
      "mov x12, #0\n"
      "csel x20, x21, x20, LT\n"
      "add x23, x23, x9, LSL #2\n"  // C += n
      "lsr x21, x20, #0x2\n"
      "madd x23, x10, x22, x23\n"  // C += m * ldc
      "and x20, x20, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e3  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 18f\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "tbz x15, #0, 21f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "20:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 20b\n"
      "21:"  // End block
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
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
