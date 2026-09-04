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

void sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
      "ptrue p1.b\n"
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
      ".inst 0xe09f04c0  // ld1w { za0h.s[x12] }, p1/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f06c4  // ld1w { za1h.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f06a8  // ld1w { za2h.s[x12] }, p1/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f068c  // ld1w { za3h.s[x12] }, p1/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09904c1  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe09906c5  // ld1w { za1h.s[x12, #1] }, p1/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe09906a9  // ld1w { za2h.s[x12, #1] }, p1/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe099068d  // ld1w { za3h.s[x12, #1] }, p1/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09804c2  // ld1w { za0h.s[x12, #2] }, p1/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe09806c6  // ld1w { za1h.s[x12, #2] }, p1/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe09806aa  // ld1w { za2h.s[x12, #2] }, p1/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe098068e  // ld1w { za3h.s[x12, #2] }, p1/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09704c3  // ld1w { za0h.s[x12, #3] }, p1/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe09706c7  // ld1w { za1h.s[x12, #3] }, p1/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe09706ab  // ld1w { za2h.s[x12, #3] }, p1/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe097068f  // ld1w { za3h.s[x12, #3] }, p1/Z, [x20, x23, LSL #2]\n"
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
      "tbnz x5, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z19.h, #0.0\n"
      "whilelt p0.h, x16, x14\n"
      "fmov z18.h, #1.0\n"
      "ld1h { z16.h }, p0/Z, [x20, x16, LSL #1]\n"
      "zip1 z17.h, z16.h, z19.h\n"
      "zip2 z16.h, z16.h, z19.h\n"
      ".inst 0x81b12640  // fmopa za0.s, p1/M, p1/M, z18.h, z17.h\n"
      ".inst 0x81b02641  // fmopa za1.s, p1/M, p1/M, z18.h, z16.h\n"
      ".inst 0x81b12642  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h\n"
      ".inst 0x81b02643  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x16\n"
      "mov x21, x17\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
      "cmp x20, x14\n"
      "mov x20, x5\n"
      "csel x21, x17, x21, LT\n"
      "bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1\n"
      "cmp x21, x15\n"
      "csel x5, x20, x5, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x8, #0x2\n"
      "and x20, x8, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      "ld1h { z31.h }, p1/Z, [x10]\n"
      "ld1h { z30.h }, p1/Z, [x10, #1, MUL VL]\n"
      "ld1h { z29.h }, p1/Z, [x10, #2, MUL VL]\n"
      "ld1h { z28.h }, p1/Z, [x10, #3, MUL VL]\n"
      "ld1h { z27.h }, p1/Z, [x10, #4, MUL VL]\n"
      "ld1h { z26.h }, p1/Z, [x10, #5, MUL VL]\n"
      "ld1h { z25.h }, p1/Z, [x10, #6, MUL VL]\n"
      "ld1h { z24.h }, p1/Z, [x10, #7, MUL VL]\n"
      "addvl x10, x10, #8\n"
      "ld1h { z23.h }, p1/Z, [x11]\n"
      "ld1h { z22.h }, p1/Z, [x11, #1, MUL VL]\n"
      "ld1h { z21.h }, p1/Z, [x11, #2, MUL VL]\n"
      "ld1h { z20.h }, p1/Z, [x11, #3, MUL VL]\n"
      "ld1h { z19.h }, p1/Z, [x11, #4, MUL VL]\n"
      "ld1h { z18.h }, p1/Z, [x11, #5, MUL VL]\n"
      "ld1h { z17.h }, p1/Z, [x11, #6, MUL VL]\n"
      "ld1h { z16.h }, p1/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #8\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81b727e0  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81b627e1  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h\n"
      "ld1h { z31.h }, p1/Z, [x10]\n"
      ".inst 0x81b727c2  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h\n"
      "ld1h { z23.h }, p1/Z, [x11]\n"
      ".inst 0x81b627c3  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h\n"
      "ld1h { z30.h }, p1/Z, [x10, #1, MUL VL]\n"
      ".inst 0x81b527a0  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h\n"
      "ld1h { z22.h }, p1/Z, [x11, #1, MUL VL]\n"
      ".inst 0x81b427a1  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h\n"
      "ld1h { z29.h }, p1/Z, [x10, #2, MUL VL]\n"
      ".inst 0x81b52782  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h\n"
      "ld1h { z21.h }, p1/Z, [x11, #2, MUL VL]\n"
      ".inst 0x81b42783  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h\n"
      "ld1h { z28.h }, p1/Z, [x10, #3, MUL VL]\n"
      ".inst 0x81b32760  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h\n"
      "ld1h { z20.h }, p1/Z, [x11, #3, MUL VL]\n"
      ".inst 0x81b22761  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h\n"
      "ld1h { z27.h }, p1/Z, [x10, #4, MUL VL]\n"
      ".inst 0x81b32742  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h\n"
      "ld1h { z19.h }, p1/Z, [x11, #4, MUL VL]\n"
      ".inst 0x81b22743  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h\n"
      "ld1h { z26.h }, p1/Z, [x10, #5, MUL VL]\n"
      ".inst 0x81b12720  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h\n"
      "ld1h { z18.h }, p1/Z, [x11, #5, MUL VL]\n"
      ".inst 0x81b02721  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h\n"
      "ld1h { z25.h }, p1/Z, [x10, #6, MUL VL]\n"
      ".inst 0x81b12702  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h\n"
      "ld1h { z17.h }, p1/Z, [x11, #6, MUL VL]\n"
      ".inst 0x81b02703  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h\n"
      "ld1h { z24.h }, p1/Z, [x10, #7, MUL VL]\n"
      "addvl x10, x10, #8\n"
      "ld1h { z16.h }, p1/Z, [x11, #7, MUL VL]\n"
      "addvl x11, x11, #8\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81b727e0  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h\n"
      ".inst 0x81b627e1  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h\n"
      ".inst 0x81b727c2  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h\n"
      ".inst 0x81b627c3  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h\n"
      ".inst 0x81b527a0  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h\n"
      ".inst 0x81b427a1  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h\n"
      ".inst 0x81b52782  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h\n"
      ".inst 0x81b42783  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h\n"
      ".inst 0x81b32760  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h\n"
      ".inst 0x81b22761  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h\n"
      ".inst 0x81b32742  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h\n"
      ".inst 0x81b22743  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h\n"
      ".inst 0x81b12720  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h\n"
      ".inst 0x81b02721  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h\n"
      ".inst 0x81b12702  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h\n"
      ".inst 0x81b02703  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1h { z19.h }, p1/Z, [x10]\n"
      "subs x20, x20, #0x1\n"
      "ld1h { z18.h }, p1/Z, [x10, #1, MUL VL]\n"
      "addvl x10, x10, #2\n"
      "ld1h { z17.h }, p1/Z, [x11]\n"
      "ld1h { z16.h }, p1/Z, [x11, #1, MUL VL]\n"
      "addvl x11, x11, #2\n"
      ".inst 0x81b12660  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h\n"
      ".inst 0x81b02661  // fmopa za1.s, p1/M, p1/M, z19.h, z16.h\n"
      ".inst 0x81b12642  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h\n"
      ".inst 0x81b02643  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h\n"
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
      ".inst 0xe0bf04e0  // st1w { za0h.s[x12] }, p1/Z, [x7, XZR, LSL #2]\n"
      ".inst 0xe09f04c0  // ld1w { za0h.s[x12] }, p1/Z, [x6, XZR, LSL #2]\n"
      "addvl x25, x7, #4\n"
      "addvl x24, x6, #4\n"
      ".inst 0xe0bc04e1  // st1w { za0h.s[x12, #1] }, p1/Z, [x7, x28, LSL #2]\n"
      ".inst 0xe09c04c1  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x28, LSL #2]\n"
      "addvl x23, x7, #8\n"
      "addvl x22, x6, #8\n"
      ".inst 0xe0bb04e2  // st1w { za0h.s[x12, #2] }, p1/Z, [x7, x27, LSL #2]\n"
      ".inst 0xe09b04c2  // ld1w { za0h.s[x12, #2] }, p1/Z, [x6, x27, LSL #2]\n"
      "addvl x21, x7, #12\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe0ba04e3  // st1w { za0h.s[x12, #3] }, p1/Z, [x7, x26, LSL #2]\n"
      ".inst 0xe09a04c3  // ld1w { za0h.s[x12, #3] }, p1/Z, [x6, x26, LSL #2]\n"
      "addvl x7, x7, #16\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe0bf0724  // st1w { za1h.s[x12] }, p1/Z, [x25, XZR, LSL #2]\n"
      ".inst 0xe09f0704  // ld1w { za1h.s[x12] }, p1/Z, [x24, XZR, LSL #2]\n"
      ".inst 0xe0bc0725  // st1w { za1h.s[x12, #1] }, p1/Z, [x25, x28, LSL #2]\n"
      ".inst 0xe09c0705  // ld1w { za1h.s[x12, #1] }, p1/Z, [x24, x28, LSL #2]\n"
      ".inst 0xe0bb0726  // st1w { za1h.s[x12, #2] }, p1/Z, [x25, x27, LSL #2]\n"
      ".inst 0xe09b0706  // ld1w { za1h.s[x12, #2] }, p1/Z, [x24, x27, LSL #2]\n"
      ".inst 0xe0ba0727  // st1w { za1h.s[x12, #3] }, p1/Z, [x25, x26, LSL #2]\n"
      ".inst 0xe09a0707  // ld1w { za1h.s[x12, #3] }, p1/Z, [x24, x26, LSL #2]\n"
      ".inst 0xe0bf06e8  // st1w { za2h.s[x12] }, p1/Z, [x23, XZR, LSL #2]\n"
      ".inst 0xe09f06c8  // ld1w { za2h.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0bc06e9  // st1w { za2h.s[x12, #1] }, p1/Z, [x23, x28, LSL #2]\n"
      ".inst 0xe09c06c9  // ld1w { za2h.s[x12, #1] }, p1/Z, [x22, x28, LSL #2]\n"
      ".inst 0xe0bb06ea  // st1w { za2h.s[x12, #2] }, p1/Z, [x23, x27, LSL #2]\n"
      ".inst 0xe09b06ca  // ld1w { za2h.s[x12, #2] }, p1/Z, [x22, x27, LSL #2]\n"
      ".inst 0xe0ba06eb  // st1w { za2h.s[x12, #3] }, p1/Z, [x23, x26, LSL #2]\n"
      ".inst 0xe09a06cb  // ld1w { za2h.s[x12, #3] }, p1/Z, [x22, x26, LSL #2]\n"
      ".inst 0xe0bf06ac  // st1w { za3h.s[x12] }, p1/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f068c  // ld1w { za3h.s[x12] }, p1/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0bc06ad  // st1w { za3h.s[x12, #1] }, p1/Z, [x21, x28, LSL #2]\n"
      ".inst 0xe09c068d  // ld1w { za3h.s[x12, #1] }, p1/Z, [x20, x28, LSL #2]\n"
      ".inst 0xe0bb06ae  // st1w { za3h.s[x12, #2] }, p1/Z, [x21, x27, LSL #2]\n"
      ".inst 0xe09b068e  // ld1w { za3h.s[x12, #2] }, p1/Z, [x20, x27, LSL #2]\n"
      ".inst 0xe0ba06af  // st1w { za3h.s[x12, #3] }, p1/Z, [x21, x26, LSL #2]\n"
      ".inst 0xe09a068f  // ld1w { za3h.s[x12, #3] }, p1/Z, [x20, x26, LSL #2]\n"
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
      ".inst 0xe0bf04e0  // st1w { za0h.s[x12] }, p1/Z, [x7, XZR, LSL #2]\n"
      "addvl x22, x7, #4\n"
      "addvl x21, x7, #8\n"
      ".inst 0xe0b904e1  // st1w { za0h.s[x12, #1] }, p1/Z, [x7, x25, LSL #2]\n"
      "addvl x20, x7, #12\n"
      ".inst 0xe0b804e2  // st1w { za0h.s[x12, #2] }, p1/Z, [x7, x24, LSL #2]\n"
      ".inst 0xe0b704e3  // st1w { za0h.s[x12, #3] }, p1/Z, [x7, x23, LSL #2]\n"
      "addvl x7, x7, #16\n"
      ".inst 0xe0bf06c4  // st1w { za1h.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe0b906c5  // st1w { za1h.s[x12, #1] }, p1/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe0b806c6  // st1w { za1h.s[x12, #2] }, p1/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe0b706c7  // st1w { za1h.s[x12, #3] }, p1/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe0bf06a8  // st1w { za2h.s[x12] }, p1/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe0b906a9  // st1w { za2h.s[x12, #1] }, p1/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe0b806aa  // st1w { za2h.s[x12, #2] }, p1/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe0b706ab  // st1w { za2h.s[x12, #3] }, p1/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe0bf068c  // st1w { za3h.s[x12] }, p1/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe0b9068d  // st1w { za3h.s[x12, #1] }, p1/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe0b8068e  // st1w { za3h.s[x12, #2] }, p1/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe0b7068f  // st1w { za3h.s[x12, #3] }, p1/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 14b\n"
      "b 19f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x15, x17\n"
      "cntw x23, ALL, MUL #2\n"
      "ld1rh { z19.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "whilelt p0.h, x16, x14\n"
      "cmp x24, x23\n"
      "ld1rh { z18.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0\n"
      "mov x21, #0\n"
      "add x25, x25, x16, LSL #1\n"  // C += n
      "mov x20, #0x2\n"
      "madd x25, x17, x22, x25\n"  // C += m * ldc
      "csel x24, x24, x23, LT\n"
      "16:"  // Store to output array: Accumulator loop
      ".inst 0xc0020411  // mova z17.b, p1/M, za0h.b[x12]\n"
      "add x21, x21, #0x1\n"
      ".inst 0xc0020430  // mova z16.b, p1/M, za0h.b[x12, #1]\n"
      "fcvt z17.h, p1/m, z17.s\n"
      "add x12, x12, #0x4\n"
      "fcvt z16.h, p1/m, z16.s\n"
      "cmp x12, x23, LSL #1\n"
      "csel x12, x12, x20, LT\n"
      "cmp x21, x24\n"
      "uzp1 z16.h, z17.h, z16.h\n"
      "fmin z16.h, p1/M, z16.h, z18.h\n"
      "fmax z16.h, p1/M, z16.h, z19.h\n"
      "st1h { z16.h }, p0, [x25]\n"
      "add x25, x25, x22\n"
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
      ".inst 0xe09f04c0  // ld1w { za0h.s[x12] }, p1/Z, [x6, XZR, LSL #2]\n"
      "addvl x20, x6, #12\n"
      ".inst 0xe09f06c4  // ld1w { za1h.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
      ".inst 0xe09f06a8  // ld1w { za2h.s[x12] }, p1/Z, [x21, XZR, LSL #2]\n"
      ".inst 0xe09f068c  // ld1w { za3h.s[x12] }, p1/Z, [x20, XZR, LSL #2]\n"
      ".inst 0xe09904c1  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x25, LSL #2]\n"
      ".inst 0xe09906c5  // ld1w { za1h.s[x12, #1] }, p1/Z, [x22, x25, LSL #2]\n"
      ".inst 0xe09906a9  // ld1w { za2h.s[x12, #1] }, p1/Z, [x21, x25, LSL #2]\n"
      ".inst 0xe099068d  // ld1w { za3h.s[x12, #1] }, p1/Z, [x20, x25, LSL #2]\n"
      ".inst 0xe09804c2  // ld1w { za0h.s[x12, #2] }, p1/Z, [x6, x24, LSL #2]\n"
      ".inst 0xe09806c6  // ld1w { za1h.s[x12, #2] }, p1/Z, [x22, x24, LSL #2]\n"
      ".inst 0xe09806aa  // ld1w { za2h.s[x12, #2] }, p1/Z, [x21, x24, LSL #2]\n"
      ".inst 0xe098068e  // ld1w { za3h.s[x12, #2] }, p1/Z, [x20, x24, LSL #2]\n"
      ".inst 0xe09704c3  // ld1w { za0h.s[x12, #3] }, p1/Z, [x6, x23, LSL #2]\n"
      "addvl x6, x6, #16\n"
      ".inst 0xe09706c7  // ld1w { za1h.s[x12, #3] }, p1/Z, [x22, x23, LSL #2]\n"
      ".inst 0xe09706ab  // ld1w { za2h.s[x12, #3] }, p1/Z, [x21, x23, LSL #2]\n"
      ".inst 0xe097068f  // ld1w { za3h.s[x12, #3] }, p1/Z, [x20, x23, LSL #2]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x26\n"
      "blt 18b\n"
      "19:"  // End block
      "incw x16, ALL, MUL #2\n"
      "cmp x16, x14\n"
      "blt 4b\n"
      "incw x17, ALL, MUL #2\n"
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
