//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x9, #0
    ldr x1, [x0, #0x60]
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ptrue p1.b
    ldr x3, [x0, #0x50]
    ldr x28, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x9, x9, x1
    cset x1, EQ
    orr x9, x9, x1, LSL #1
    mov x27, x28
    add x26, x28, x2
    tbz x9, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c39c)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xa041c388)  // ld1w { z8.s-z11.s }, pn8.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c384)  // ld1w { z4.s-z7.s }, pn8.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c394)  // ld1w { z20.s-z23.s }, pn8.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840780)  // mova za0h.s[x12], { z28.s-z31.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840501)  // mova za1h.s[x12], { z8.s-z11.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840683)  // mova za3h.s[x12], { z20.s-z23.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x25, [x0, #0x10]
    mov x24, #0
    mov x23, #0
    ldr x22, [x0, #0]
    ldr x21, [x0, #0x8]
    add x25, x25, #0x1
    ldr x20, [x0, #0x20]
    lsr x25, x25, #0x1
KAI_ASM_LABEL(label_3)  // M loop
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x7, x20
    tbnz x9, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z4.h, #0.0
    whilelt p0.h, x23, x21
    fmov z24.h, #1.0
    ld1h { z22.h }, p0/Z, [x1, x23, LSL #1]
    zip1 z0.h, z22.h, z4.h
    zip2 z9.h, z22.h, z4.h
    KAI_ASM_INST(0x81a02700)  // fmopa za0.s, p1/M, p1/M, z24.h, z0.h
    KAI_ASM_INST(0x81a92701)  // fmopa za1.s, p1/M, p1/M, z24.h, z9.h
    KAI_ASM_INST(0x81a02702)  // fmopa za2.s, p1/M, p1/M, z24.h, z0.h
    KAI_ASM_INST(0x81a92703)  // fmopa za3.s, p1/M, p1/M, z24.h, z9.h
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x23
    mov x2, x24
    incw x1, ALL, MUL #2
    incw x2, ALL, MUL #2
    cmp x1, x21
    mov x1, x9
    csel x2, x24, x2, LT
    bfm x9, XZR, #0, #0  // bfc x9, #0, #0x1
    csel x28, x28, x26, LT
    cmp x2, x22
    csel x9, x1, x9, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x25, #0x2
    and x1, x25, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa140a0f0)  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x7]
    KAI_ASM_INST(0xa141a0e0)  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0xa140a113)  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81b32600)  // fmopa za0.s, p1/M, p1/M, z16.h, z19.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81b72601)  // fmopa za1.s, p1/M, p1/M, z16.h, z23.h
    KAI_ASM_INST(0x81b32682)  // fmopa za2.s, p1/M, p1/M, z20.h, z19.h
    KAI_ASM_INST(0x81b72683)  // fmopa za3.s, p1/M, p1/M, z20.h, z23.h
    KAI_ASM_INST(0x81bb2700)  // fmopa za0.s, p1/M, p1/M, z24.h, z27.h
    KAI_ASM_INST(0x81bf2701)  // fmopa za1.s, p1/M, p1/M, z24.h, z31.h
    KAI_ASM_INST(0x81bb2782)  // fmopa za2.s, p1/M, p1/M, z28.h, z27.h
    KAI_ASM_INST(0x81bf2783)  // fmopa za3.s, p1/M, p1/M, z28.h, z31.h
    KAI_ASM_INST(0xa140a0f0)  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x7]
    KAI_ASM_INST(0x81b22400)  // fmopa za0.s, p1/M, p1/M, z0.h, z18.h
    KAI_ASM_INST(0xa140a113)  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0x81b62401)  // fmopa za1.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81b22482)  // fmopa za2.s, p1/M, p1/M, z4.h, z18.h
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
    KAI_ASM_INST(0x81ba2500)  // fmopa za0.s, p1/M, p1/M, z8.h, z26.h
    KAI_ASM_INST(0x81be2501)  // fmopa za1.s, p1/M, p1/M, z8.h, z30.h
    KAI_ASM_INST(0x81ba2582)  // fmopa za2.s, p1/M, p1/M, z12.h, z26.h
    KAI_ASM_INST(0x81be2583)  // fmopa za3.s, p1/M, p1/M, z12.h, z30.h
    KAI_ASM_INST(0xa141a0e0)  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81b32600)  // fmopa za0.s, p1/M, p1/M, z16.h, z19.h
    KAI_ASM_INST(0x81b72601)  // fmopa za1.s, p1/M, p1/M, z16.h, z23.h
    KAI_ASM_INST(0x81b32682)  // fmopa za2.s, p1/M, p1/M, z20.h, z19.h
    KAI_ASM_INST(0x81b72683)  // fmopa za3.s, p1/M, p1/M, z20.h, z23.h
    KAI_ASM_INST(0x81bb2700)  // fmopa za0.s, p1/M, p1/M, z24.h, z27.h
    KAI_ASM_INST(0x81bf2701)  // fmopa za1.s, p1/M, p1/M, z24.h, z31.h
    KAI_ASM_INST(0x81bb2782)  // fmopa za2.s, p1/M, p1/M, z28.h, z27.h
    KAI_ASM_INST(0x81bf2783)  // fmopa za3.s, p1/M, p1/M, z28.h, z31.h
    KAI_ASM_INST(0x81b22400)  // fmopa za0.s, p1/M, p1/M, z0.h, z18.h
    KAI_ASM_INST(0x81b62401)  // fmopa za1.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81b22482)  // fmopa za2.s, p1/M, p1/M, z4.h, z18.h
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
    KAI_ASM_INST(0x81ba2500)  // fmopa za0.s, p1/M, p1/M, z8.h, z26.h
    KAI_ASM_INST(0x81be2501)  // fmopa za1.s, p1/M, p1/M, z8.h, z30.h
    KAI_ASM_INST(0x81ba2582)  // fmopa za2.s, p1/M, p1/M, z12.h, z26.h
    KAI_ASM_INST(0x81be2583)  // fmopa za3.s, p1/M, p1/M, z12.h, z30.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa04020f0)  // ld1h { z16.h-z17.h }, pn8.b/Z, [x7]
    subs x1, x1, #0x1
    addvl x7, x7, #2
    KAI_ASM_INST(0xa0402104)  // ld1h { z4.h-z5.h }, pn8.b/Z, [x8]
    addvl x8, x8, #2
    KAI_ASM_INST(0x81a42600)  // fmopa za0.s, p1/M, p1/M, z16.h, z4.h
    KAI_ASM_INST(0x81a52601)  // fmopa za1.s, p1/M, p1/M, z16.h, z5.h
    KAI_ASM_INST(0x81a42622)  // fmopa za2.s, p1/M, p1/M, z17.h, z4.h
    KAI_ASM_INST(0x81a52623)  // fmopa za3.s, p1/M, p1/M, z17.h, z5.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x9, #1, label_15
    tbz x9, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c394)  // ld1w { z20.s-z23.s }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c384)  // ld1w { z4.s-z7.s }, pn8.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c38c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c380)  // ld1w { z0.s-z3.s }, pn8.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840481)  // mova za1h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xa060c370)  // st1w { z16.s-z19.s }, pn8.b, [x27]
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xa061c37c)  // st1w { z28.s-z31.s }, pn8.b, [x27, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c378)  // st1w { z24.s-z27.s }, pn8.b, [x27, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c368)  // st1w { z8.s-z11.s }, pn8.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_12
    b label_19
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc086045c)  // mova { z28.s-z31.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c368)  // st1w { z8.s-z11.s }, pn8.b, [x27]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c36c)  // st1w { z12.s-z15.s }, pn8.b, [x27, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c37c)  // st1w { z28.s-z31.s }, pn8.b, [x27, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c370)  // st1w { z16.s-z19.s }, pn8.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_14
    b label_19
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x22, x24
    cntw x4, ALL, MUL #2
    ldr x3, [x0, #0x58]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    whilelt p0.h, x23, x21
    cmp x5, x4
    KAI_ASM_INST(0x84c0a452)  // ld1rh { z18.h }, p1/Z, [x2]
    add x6, x6, x23, LSL #1  // C += n
    KAI_ASM_INST(0x84c0a431)  // ld1rh { z17.h }, p1/Z, [x1]
    mov x12, #0
    madd x6, x24, x3, x6  // C += m * ldc
    mov x2, #0
    mov x1, #0x2
    csel x5, x5, x4, LT
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc006000e)  // mova { z14.b-z15.b }, za0h.b[x12, 0:1]
    add x12, x12, #0x4
    cmp x12, x4, LSL #1
    add x2, x2, #0x1
    KAI_ASM_INST(0xc120e1d0)  // fcvt z16.h, { z14.s-z15.s }
    csel x12, x12, x1, LT
    cmp x2, x5
    KAI_ASM_INST(0x64712650)  // fclamp z16.h, z18.h, z17.h
    st1h { z16.h }, p0, [x6]
    add x6, x6, x3
    blt label_16
    tbz x9, #0, label_19
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_18)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c38c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xa041c384)  // ld1w { z4.s-z7.s }, pn8.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c388)  // ld1w { z8.s-z11.s }, pn8.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c39c)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840580)  // mova za0h.s[x12], { z12.s-z15.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840481)  // mova za1h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840502)  // mova za2h.s[x12], { z8.s-z11.s }
    KAI_ASM_INST(0xc0840783)  // mova za3h.s[x12], { z28.s-z31.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_18
KAI_ASM_LABEL(label_19)  // End block
    incw x23, ALL, MUL #2
    cmp x23, x21
    blt label_4
    ldr x1, [x0, #0x70]
    incw x24, ALL, MUL #2
    mov x27, x26
    cmp x24, x22
    mov x23, #0
    mov x20, x7
    add x26, x27, x1
    blt label_3
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)

    KAI_ASM_END
