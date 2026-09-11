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

void sme2_interleaved_nomerge_fp32_mopa_1VLx4VL(const float *const A, const float *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const float *const A,
      const float *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(K * sizeof(float)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
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
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const float *const A;
    const float *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const float *const bias;


    float *const accumulator_buffer;
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
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16]\n"
      ".inst 0xa041c618  // ld1w { z24.s-z27.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840701  // mova za1h.s[x12], { z24.s-z27.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
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
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "3:"  // M loop
      "ldr x27, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x26, x28\n"
      ".inst 0x25a96570  // whilelt pn8.s, x11, x9, VLx4\n"
      "tbnz x17, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z18.s, #1.0\n"
      ".inst 0xa10bc280  // ld1w { z0.s, z4.s, z8.s, z12.s }, p8/Z, [x20, x11, LSL #2]\n"
      ".inst 0x80800240  // fmopa za0.s, p0/M, p0/M, z18.s, z0.s\n"
      ".inst 0x80840241  // fmopa za1.s, p0/M, p0/M, z18.s, z4.s\n"
      ".inst 0x80880242  // fmopa za2.s, p0/M, p0/M, z18.s, z8.s\n"
      ".inst 0x808c0243  // fmopa za3.s, p0/M, p0/M, z18.s, z12.s\n"
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
      ".inst 0xa140c753  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa140c770  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x27]\n"
      ".inst 0xa041c768  // ld1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa042c76c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa143c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x80900260  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x80940261  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s\n"
      ".inst 0x80980262  // fmopa za2.s, p0/M, p0/M, z19.s, z24.s\n"
      ".inst 0x809c0263  // fmopa za3.s, p0/M, p0/M, z19.s, z28.s\n"
      ".inst 0xa140c770  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x27]\n"
      ".inst 0x808802e0  // fmopa za0.s, p0/M, p0/M, z23.s, z8.s\n"
      ".inst 0x808902e1  // fmopa za1.s, p0/M, p0/M, z23.s, z9.s\n"
      ".inst 0x808a02e2  // fmopa za2.s, p0/M, p0/M, z23.s, z10.s\n"
      ".inst 0x808b02e3  // fmopa za3.s, p0/M, p0/M, z23.s, z11.s\n"
      ".inst 0xa041c768  // ld1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0x808c0360  // fmopa za0.s, p0/M, p0/M, z27.s, z12.s\n"
      ".inst 0x808d0361  // fmopa za1.s, p0/M, p0/M, z27.s, z13.s\n"
      ".inst 0x808e0362  // fmopa za2.s, p0/M, p0/M, z27.s, z14.s\n"
      ".inst 0x808f0363  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s\n"
      ".inst 0xa042c76c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0x809203e0  // fmopa za0.s, p0/M, p0/M, z31.s, z18.s\n"
      ".inst 0x809603e1  // fmopa za1.s, p0/M, p0/M, z31.s, z22.s\n"
      ".inst 0x809a03e2  // fmopa za2.s, p0/M, p0/M, z31.s, z26.s\n"
      ".inst 0x809e03e3  // fmopa za3.s, p0/M, p0/M, z31.s, z30.s\n"
      ".inst 0xa140c753  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa143c772  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x80900260  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s\n"
      ".inst 0x80940261  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s\n"
      ".inst 0x80980262  // fmopa za2.s, p0/M, p0/M, z19.s, z24.s\n"
      ".inst 0x809c0263  // fmopa za3.s, p0/M, p0/M, z19.s, z28.s\n"
      ".inst 0x808802e0  // fmopa za0.s, p0/M, p0/M, z23.s, z8.s\n"
      ".inst 0x808902e1  // fmopa za1.s, p0/M, p0/M, z23.s, z9.s\n"
      ".inst 0x808a02e2  // fmopa za2.s, p0/M, p0/M, z23.s, z10.s\n"
      ".inst 0x808b02e3  // fmopa za3.s, p0/M, p0/M, z23.s, z11.s\n"
      ".inst 0x808c0360  // fmopa za0.s, p0/M, p0/M, z27.s, z12.s\n"
      ".inst 0x808d0361  // fmopa za1.s, p0/M, p0/M, z27.s, z13.s\n"
      ".inst 0x808e0362  // fmopa za2.s, p0/M, p0/M, z27.s, z14.s\n"
      ".inst 0x808f0363  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s\n"
      ".inst 0x809203e0  // fmopa za0.s, p0/M, p0/M, z31.s, z18.s\n"
      ".inst 0x809603e1  // fmopa za1.s, p0/M, p0/M, z31.s, z22.s\n"
      ".inst 0x809a03e2  // fmopa za2.s, p0/M, p0/M, z31.s, z26.s\n"
      ".inst 0x809e03e3  // fmopa za3.s, p0/M, p0/M, z31.s, z30.s\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1w { z27.s }, p0/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #1\n"
      ".inst 0xa140c763  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn9.b/Z, [x27]\n"
      "addvl x27, x27, #4\n"
      ".inst 0x80830360  // fmopa za0.s, p0/M, p0/M, z27.s, z3.s\n"
      ".inst 0x80870361  // fmopa za1.s, p0/M, p0/M, z27.s, z7.s\n"
      ".inst 0x808b0362  // fmopa za2.s, p0/M, p0/M, z27.s, z11.s\n"
      ".inst 0x808f0363  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x17, #1, 15f\n"
      "tbz x17, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c61c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x16]\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xa041c608  // ld1w { z8.s-z11.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0860440  // mova { z0.s-z3.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa042c614  // ld1w { z20.s-z23.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840501  // mova za1h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xa060c5e4  // st1w { z4.s-z7.s }, pn9.b, [x15]\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xa061c5f0  // st1w { z16.s-z19.s }, pn9.b, [x15, #0x4, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5e0  // st1w { z0.s-z3.s }, pn9.b, [x15, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5f8  // st1w { z24.s-z27.s }, pn9.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 12b\n"
      "b 25f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc0860460  // mova { z0.s-z3.s }, za3h.s[x12]\n"
      ".inst 0xa060c5ec  // st1w { z12.s-z15.s }, pn9.b, [x15]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5e4  // st1w { z4.s-z7.s }, pn9.b, [x15, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5e8  // st1w { z8.s-z11.s }, pn9.b, [x15, #0x8, MUL VL]\n"
      ".inst 0xa063c5e0  // st1w { z0.s-z3.s }, pn9.b, [x15, #0xc, MUL VL]\n"
      "addvl x15, x15, #16\n"
      "blt 14b\n"
      "b 25f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x10, x13\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "add x25, x25, x11, LSL #2\n"  // C += n
      "madd x25, x13, x23, x25\n"  // C += m * ldc
      "tbz x17, #2, 19f\n"
      "cntw x20\n"
      "mov x12, #0\n"
      "cmp x24, x20\n"
      "csel x22, x24, x20, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa160c330  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c331  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c332  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c333  // st1w { z19.s, z23.s, z27.s, z31.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c320  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c321  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      ".inst 0xa160c322  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x24, x24, x22\n"
      "beq 19f\n"
      "b 23f\n"
      "19:"  // Store to output array: Skip activation: End
      "cntw x20\n"
      "ld1rw { z1.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x24, x20\n"
      "ld1rw { z0.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xc1a0c830  // fclamp { z16.s-z19.s }, z1.s, z0.s\n"
      ".inst 0xc1a0c834  // fclamp { z20.s-z23.s }, z1.s, z0.s\n"
      ".inst 0xc1a0c838  // fclamp { z24.s-z27.s }, z1.s, z0.s\n"
      ".inst 0xc1a0c83c  // fclamp { z28.s-z31.s }, z1.s, z0.s\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c330  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c331  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c332  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c333  // st1w { z19.s, z23.s, z27.s, z31.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 20b\n"
      "21:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 22f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xc1a0c830  // fclamp { z16.s-z19.s }, z1.s, z0.s\n"
      ".inst 0xc1a0c834  // fclamp { z20.s-z23.s }, z1.s, z0.s\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1a0c838  // fclamp { z24.s-z27.s }, z1.s, z0.s\n"
      ".inst 0xc1a0c83c  // fclamp { z28.s-z31.s }, z1.s, z0.s\n"
      ".inst 0xa160c330  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c331  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      ".inst 0xa160c332  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x25]\n"
      "22:"  // Store to output array: Accumulator row 0 oddments: End
      "23:"  // Store to output array: End
      "tbz x17, #0, 25f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "24:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c600  // ld1w { z0.s-z3.s }, pn9.b/Z, [x16]\n"
      ".inst 0xa041c60c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x16, #0x4, MUL VL]\n"
      ".inst 0xa042c61c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x16, #0x8, MUL VL]\n"
      ".inst 0xa043c604  // ld1w { z4.s-z7.s }, pn9.b/Z, [x16, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x16, x16, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 24b\n"
      "25:"  // End block
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
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
