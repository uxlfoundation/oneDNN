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

void sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const DequantizeFloat &, const float *const late_bias, const Activation act,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
        bias(bias), late_bias(late_bias),
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
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const int32_t *const bias;
    const float *const late_bias;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, dq, late_bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x8, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x17, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x8, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c22c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x17]\n"
      ".inst 0xa041c224  // ld1w { z4.s-z7.s }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c234  // ld1w { z20.s-z23.s }, pn8.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c238  // ld1w { z24.s-z27.s }, pn8.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840703  // mova za3h.s[x12], { z24.s-z27.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x15, [%x[args], %[offsetof_K]]\n"
      "mov x14, #0\n"
      "mov x13, #0\n"
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "ldr w10, [%x[args], %[offsetof_N]]\n"
      "add x15, x15, #0x3\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "lsr x15, x15, #0x2\n"
      "3:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x27, x9\n"
      "whilelt p0.s, x13, x10\n"
      "tbnz x8, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "ld1w { z20.s }, p0/Z, [x20, x13, LSL #2]\n"
      ".inst 0xc0902680  // addha za0.s, p1/M, p1/M, z20.s\n"
      ".inst 0xc0902681  // addha za1.s, p1/M, p1/M, z20.s\n"
      ".inst 0xc0902682  // addha za2.s, p1/M, p1/M, z20.s\n"
      ".inst 0xc0902683  // addha za3.s, p1/M, p1/M, z20.s\n"
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
      ".inst 0xa0408378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27]\n"
      ".inst 0xa041837c  // ld1b { z28.b-z31.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa1428361  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa0438370  // ld1b { z16.b-z19.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      ".inst 0xa1408382  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0822700  // smopa za0.s, p1/M, p1/M, z24.b, z2.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0822721  // smopa za1.s, p1/M, p1/M, z25.b, z2.b\n"
      ".inst 0xa0822742  // smopa za2.s, p1/M, p1/M, z26.b, z2.b\n"
      ".inst 0xa0822763  // smopa za3.s, p1/M, p1/M, z27.b, z2.b\n"
      ".inst 0xa0408378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27]\n"
      ".inst 0xa0862780  // smopa za0.s, p1/M, p1/M, z28.b, z6.b\n"
      ".inst 0xa08627a1  // smopa za1.s, p1/M, p1/M, z29.b, z6.b\n"
      ".inst 0xa08627c2  // smopa za2.s, p1/M, p1/M, z30.b, z6.b\n"
      ".inst 0xa08627e3  // smopa za3.s, p1/M, p1/M, z31.b, z6.b\n"
      ".inst 0xa041837c  // ld1b { z28.b-z31.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa08a2420  // smopa za0.s, p1/M, p1/M, z1.b, z10.b\n"
      ".inst 0xa08a24a1  // smopa za1.s, p1/M, p1/M, z5.b, z10.b\n"
      ".inst 0xa08a2522  // smopa za2.s, p1/M, p1/M, z9.b, z10.b\n"
      ".inst 0xa08a25a3  // smopa za3.s, p1/M, p1/M, z13.b, z10.b\n"
      ".inst 0xa1428361  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      ".inst 0xa08e2600  // smopa za0.s, p1/M, p1/M, z16.b, z14.b\n"
      ".inst 0xa08e2621  // smopa za1.s, p1/M, p1/M, z17.b, z14.b\n"
      ".inst 0xa08e2642  // smopa za2.s, p1/M, p1/M, z18.b, z14.b\n"
      ".inst 0xa08e2663  // smopa za3.s, p1/M, p1/M, z19.b, z14.b\n"
      ".inst 0xa0438370  // ld1b { z16.b-z19.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      ".inst 0xa1408382  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0822700  // smopa za0.s, p1/M, p1/M, z24.b, z2.b\n"
      ".inst 0xa0822721  // smopa za1.s, p1/M, p1/M, z25.b, z2.b\n"
      ".inst 0xa0822742  // smopa za2.s, p1/M, p1/M, z26.b, z2.b\n"
      ".inst 0xa0822763  // smopa za3.s, p1/M, p1/M, z27.b, z2.b\n"
      ".inst 0xa0862780  // smopa za0.s, p1/M, p1/M, z28.b, z6.b\n"
      ".inst 0xa08627a1  // smopa za1.s, p1/M, p1/M, z29.b, z6.b\n"
      ".inst 0xa08627c2  // smopa za2.s, p1/M, p1/M, z30.b, z6.b\n"
      ".inst 0xa08627e3  // smopa za3.s, p1/M, p1/M, z31.b, z6.b\n"
      ".inst 0xa08a2420  // smopa za0.s, p1/M, p1/M, z1.b, z10.b\n"
      ".inst 0xa08a24a1  // smopa za1.s, p1/M, p1/M, z5.b, z10.b\n"
      ".inst 0xa08a2522  // smopa za2.s, p1/M, p1/M, z9.b, z10.b\n"
      ".inst 0xa08a25a3  // smopa za3.s, p1/M, p1/M, z13.b, z10.b\n"
      ".inst 0xa08e2600  // smopa za0.s, p1/M, p1/M, z16.b, z14.b\n"
      ".inst 0xa08e2621  // smopa za1.s, p1/M, p1/M, z17.b, z14.b\n"
      ".inst 0xa08e2642  // smopa za2.s, p1/M, p1/M, z18.b, z14.b\n"
      ".inst 0xa08e2663  // smopa za3.s, p1/M, p1/M, z19.b, z14.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa1408373  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #4\n"
      "ld1b { z7.b }, p1/Z, [x28]\n"
      "addvl x28, x28, #1\n"
      ".inst 0xa0872660  // smopa za0.s, p1/M, p1/M, z19.b, z7.b\n"
      ".inst 0xa08726e1  // smopa za1.s, p1/M, p1/M, z23.b, z7.b\n"
      ".inst 0xa0872762  // smopa za2.s, p1/M, p1/M, z27.b, z7.b\n"
      ".inst 0xa08727e3  // smopa za3.s, p1/M, p1/M, z31.b, z7.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x8, #1, 15f\n"
      "tbz x8, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c228  // ld1w { z8.s-z11.s }, pn8.b/Z, [x17]\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xa041c22c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c23c  // ld1w { z28.s-z31.s }, pn8.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c234  // ld1w { z20.s-z23.s }, pn8.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa060c200  // st1w { z0.s-z3.s }, pn8.b, [x16]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa061c204  // st1w { z4.s-z7.s }, pn8.b, [x16, #0x4, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c218  // st1w { z24.s-z27.s }, pn8.b, [x16, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c210  // st1w { z16.s-z19.s }, pn8.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 12b\n"
      "b 31f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c208  // st1w { z8.s-z11.s }, pn8.b, [x16]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c20c  // st1w { z12.s-z15.s }, pn8.b, [x16, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c214  // st1w { z20.s-z23.s }, pn8.b, [x16, #0x8, MUL VL]\n"
      ".inst 0xa063c210  // st1w { z16.s-z19.s }, pn8.b, [x16, #0xc, MUL VL]\n"
      "addvl x16, x16, #16\n"
      "blt 14b\n"
      "b 31f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x11, x14\n"
      "ld1rw { z23.s }, p1/Z, [%x[dq], %[offset_DequantizeFloat_scale]]\n"
      "mov z22.s, #0\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "ldr x20, [%x[args], %[offsetof_late_bias]]\n"
      "add x26, x26, x13, LSL #2\n"  // C += n
      "madd x26, x14, x24, x26\n"  // C += m * ldc
      "cbz x20, 16f\n"
      "add x20, x20, x13, LSL #2\n"
      "ld1w { z22.s }, p0/Z, [x20]\n"
      "16:"  // Store to output array: no late bias
      "cntw x23\n"
      "ld1rw { z21.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "ld1rw { z20.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z0.s, p1/M, z23.s, z22.s\n"
      "fmad z1.s, p1/M, z23.s, z22.s\n"
      "fmad z2.s, p1/M, z23.s, z22.s\n"
      "fmad z3.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4caa0  // fclamp { z0.s-z3.s }, z21.s, z20.s\n"
      "st1w { z0.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z1.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z2.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z3.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 17b\n"
      "18:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 19f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "19:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 29f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 20b\n"
      "21:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 22f\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e39c  // scvtf { z28.s-z31.s }, { z28.s-z31.s }\n"
      "fmad z28.s, p1/M, z23.s, z22.s\n"
      "fmad z29.s, p1/M, z23.s, z22.s\n"
      "fmad z30.s, p1/M, z23.s, z22.s\n"
      "fmad z31.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cabc  // fclamp { z28.s-z31.s }, z21.s, z20.s\n"
      "st1w { z28.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z29.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 22f\n"
      "st1w { z30.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "22:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 29f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 24f\n"
      "23:"  // Store to output array: Accumulator row 2 loop
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z12.s, p1/M, z23.s, z22.s\n"
      "fmad z13.s, p1/M, z23.s, z22.s\n"
      "fmad z14.s, p1/M, z23.s, z22.s\n"
      "fmad z15.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4caac  // fclamp { z12.s-z15.s }, z21.s, z20.s\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 23b\n"
      "24:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 25f\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 25f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "25:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 29f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 27f\n"
      "26:"  // Store to output array: Accumulator row 3 loop
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e318  // scvtf { z24.s-z27.s }, { z24.s-z27.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z24.s, p1/M, z23.s, z22.s\n"
      "fmad z25.s, p1/M, z23.s, z22.s\n"
      "fmad z26.s, p1/M, z23.s, z22.s\n"
      "fmad z27.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
      "st1w { z24.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z25.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z26.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z27.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 26b\n"
      "27:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 28f\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 28f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "28:"  // Store to output array: Accumulator row 3 oddments: End
      "29:"  // Store to output array: End
      "tbz x8, #0, 31f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "30:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c23c  // ld1w { z28.s-z31.s }, pn8.b/Z, [x17]\n"
      ".inst 0xa041c220  // ld1w { z0.s-z3.s }, pn8.b/Z, [x17, #0x4, MUL VL]\n"
      ".inst 0xa042c22c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x17, #0x8, MUL VL]\n"
      ".inst 0xa043c224  // ld1w { z4.s-z7.s }, pn8.b/Z, [x17, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x17, x17, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 30b\n"
      "31:"  // End block
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
      : [args] "r" (&args), [dq] "r" (&dq), [offset_DequantizeFloat_scale] "I" (offsetof(DequantizeFloat, scale)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_late_bias] "I" (offsetof(KernelArgs, late_bias)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
