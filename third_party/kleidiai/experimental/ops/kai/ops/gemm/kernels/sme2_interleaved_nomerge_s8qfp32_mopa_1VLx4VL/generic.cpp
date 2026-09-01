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

void sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
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
      ".inst 0xa040c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5c8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc0840502  // mova za2h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
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
      ".inst 0xa109c292  // ld1w { z18.s, z22.s, z26.s, z30.s }, p8/Z, [x20, x9, LSL #2]\n"
      ".inst 0xc0900240  // addha za0.s, p0/M, p0/M, z18.s\n"
      ".inst 0xc09002c1  // addha za1.s, p0/M, p0/M, z22.s\n"
      ".inst 0xc0900342  // addha za2.s, p0/M, p0/M, z26.s\n"
      ".inst 0xc09003c3  // addha za3.s, p0/M, p0/M, z30.s\n"
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
      ".inst 0xa1408711  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa1408730  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x25]\n"
      ".inst 0xa1418733  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0xa1428732  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0xa1438720  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa0900220  // smopa za0.s, p0/M, p0/M, z17.b, z16.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0940221  // smopa za1.s, p0/M, p0/M, z17.b, z20.b\n"
      ".inst 0xa0980222  // smopa za2.s, p0/M, p0/M, z17.b, z24.b\n"
      ".inst 0xa09c0223  // smopa za3.s, p0/M, p0/M, z17.b, z28.b\n"
      ".inst 0xa1408730  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x25]\n"
      ".inst 0xa09302a0  // smopa za0.s, p0/M, p0/M, z21.b, z19.b\n"
      ".inst 0xa09702a1  // smopa za1.s, p0/M, p0/M, z21.b, z23.b\n"
      ".inst 0xa09b02a2  // smopa za2.s, p0/M, p0/M, z21.b, z27.b\n"
      ".inst 0xa09f02a3  // smopa za3.s, p0/M, p0/M, z21.b, z31.b\n"
      ".inst 0xa1418733  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x25, #0x4, MUL VL]\n"
      ".inst 0xa0920320  // smopa za0.s, p0/M, p0/M, z25.b, z18.b\n"
      ".inst 0xa0960321  // smopa za1.s, p0/M, p0/M, z25.b, z22.b\n"
      ".inst 0xa09a0322  // smopa za2.s, p0/M, p0/M, z25.b, z26.b\n"
      ".inst 0xa09e0323  // smopa za3.s, p0/M, p0/M, z25.b, z30.b\n"
      ".inst 0xa1428732  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x25, #0x8, MUL VL]\n"
      ".inst 0xa08003a0  // smopa za0.s, p0/M, p0/M, z29.b, z0.b\n"
      ".inst 0xa08403a1  // smopa za1.s, p0/M, p0/M, z29.b, z4.b\n"
      ".inst 0xa08803a2  // smopa za2.s, p0/M, p0/M, z29.b, z8.b\n"
      ".inst 0xa08c03a3  // smopa za3.s, p0/M, p0/M, z29.b, z12.b\n"
      ".inst 0xa1408711  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn9.b/Z, [x24]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa1438720  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x25, #0xc, MUL VL]\n"
      "addvl x25, x25, #16\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa0900220  // smopa za0.s, p0/M, p0/M, z17.b, z16.b\n"
      ".inst 0xa0940221  // smopa za1.s, p0/M, p0/M, z17.b, z20.b\n"
      ".inst 0xa0980222  // smopa za2.s, p0/M, p0/M, z17.b, z24.b\n"
      ".inst 0xa09c0223  // smopa za3.s, p0/M, p0/M, z17.b, z28.b\n"
      ".inst 0xa09302a0  // smopa za0.s, p0/M, p0/M, z21.b, z19.b\n"
      ".inst 0xa09702a1  // smopa za1.s, p0/M, p0/M, z21.b, z23.b\n"
      ".inst 0xa09b02a2  // smopa za2.s, p0/M, p0/M, z21.b, z27.b\n"
      ".inst 0xa09f02a3  // smopa za3.s, p0/M, p0/M, z21.b, z31.b\n"
      ".inst 0xa0920320  // smopa za0.s, p0/M, p0/M, z25.b, z18.b\n"
      ".inst 0xa0960321  // smopa za1.s, p0/M, p0/M, z25.b, z22.b\n"
      ".inst 0xa09a0322  // smopa za2.s, p0/M, p0/M, z25.b, z26.b\n"
      ".inst 0xa09e0323  // smopa za3.s, p0/M, p0/M, z25.b, z30.b\n"
      ".inst 0xa08003a0  // smopa za0.s, p0/M, p0/M, z29.b, z0.b\n"
      ".inst 0xa08403a1  // smopa za1.s, p0/M, p0/M, z29.b, z4.b\n"
      ".inst 0xa08803a2  // smopa za2.s, p0/M, p0/M, z29.b, z8.b\n"
      ".inst 0xa08c03a3  // smopa za3.s, p0/M, p0/M, z29.b, z12.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z15.b }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa040873c  // ld1b { z28.b-z31.b }, pn9.b/Z, [x25]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa09c01e0  // smopa za0.s, p0/M, p0/M, z15.b, z28.b\n"
      ".inst 0xa09d01e1  // smopa za1.s, p0/M, p0/M, z15.b, z29.b\n"
      ".inst 0xa09e01e2  // smopa za2.s, p0/M, p0/M, z15.b, z30.b\n"
      ".inst 0xa09f01e3  // smopa za3.s, p0/M, p0/M, z15.b, z31.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x15, #1, 15f\n"
      "tbz x15, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa041c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c5d8  // ld1w { z24.s-z27.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5d4  // ld1w { z20.s-z23.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa060c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13]\n"
      ".inst 0xc0840702  // mova za2h.s[x12], { z24.s-z27.s }\n"
      ".inst 0xa061c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 12b\n"
      "b 22f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa060c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 14b\n"
      "b 22f\n"
      "15:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x28, x10\n"
      "ld1rw { z18.s }, p0/Z, [%x[dq], %[offset_DequantizeFloat_scale]]\n"
      "mov z20.s, #0\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "mov z21.s, #0\n"
      "mov z22.s, #0\n"
      "ldr x20, [%x[args], %[offsetof_late_bias]]\n"
      "mov z23.s, #0\n"
      "add x23, x23, x9, LSL #2\n"  // C += n
      "madd x23, x10, x22, x23\n"  // C += m * ldc
      "cbz x20, 16f\n"
      "add x20, x20, x9, LSL #2\n"
      ".inst 0xa040c294  // ld1w { z20.s-z23.s }, p8/Z, [x20]\n"
      "16:"  // Store to output array: no late bias
      "cntw x20\n"
      "ld1rw { z17.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x21, x20\n"
      "ld1rw { z16.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x21, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
      ".inst 0xc132e108  // scvtf { z8.s-z11.s }, { z8.s-z11.s }\n"
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "add x12, x12, #0x4\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      ".inst 0xc1b0ca20  // fclamp { z0.s-z3.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca24  // fclamp { z4.s-z7.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca28  // fclamp { z8.s-z11.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca2c  // fclamp { z12.s-z15.s }, z17.s, z16.s\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e3  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "blt 17b\n"
      "18:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
      ".inst 0xc132e108  // scvtf { z8.s-z11.s }, { z8.s-z11.s }\n"
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "subs x20, x20, #0x1\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      ".inst 0xc1b0ca20  // fclamp { z0.s-z3.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca24  // fclamp { z4.s-z7.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca28  // fclamp { z8.s-z11.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca2c  // fclamp { z12.s-z15.s }, z17.s, z16.s\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 19f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 19f\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "19:"  // Store to output array: Accumulator row 0 oddments: End
      "tbz x15, #0, 22f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "21:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5d4  // ld1w { z20.s-z23.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 21b\n"
      "22:"  // End block
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
      : [args] "r" (&args), [dq] "r" (&dq), [offset_DequantizeFloat_scale] "I" (offsetof(DequantizeFloat, scale)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_late_bias] "I" (offsetof(KernelArgs, late_bias)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
