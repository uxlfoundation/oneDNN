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

    KAI_ASM_CODE(sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x14, #0
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x1, [x0, #0x60]
    ptrue p1.b
    ldr x3, [x0, #0x50]
    ldr x13, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x14, x14, x1
    cset x1, EQ
    orr x14, x14, x1, LSL #1
    mov x11, x13
    add x10, x13, x2
    tbz x14, #0, label_2
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    addvl x3, x13, #4
    addvl x2, x13, #8
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0464)  // ld1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0448)  // ld1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08605a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860465)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860449)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08505a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850466)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085044a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08405a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840467)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084044b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x9, [x0, #0x10]
    mov x28, #0
    mov x27, #0
    ldr x26, [x0, #0]
    ldr x25, [x0, #0x8]
    add x9, x9, #0x3
    ldr x24, [x0, #0x20]
    lsr x9, x9, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x23, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x22, x24
    whilelt p0.s, x27, x25
    tbnz x14, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    ld1w { z16.s }, p0/Z, [x1, x27, LSL #2]
    KAI_ASM_INST(0xc0902600)  // addha za0.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902601)  // addha za1.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902602)  // addha za2.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902603)  // addha za3.s, p1/M, p1/M, z16.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x27
    mov x2, x28
    incw x1
    incw x2, ALL, MUL #4
    cmp x1, x25
    mov x1, x14
    csel x2, x28, x2, LT
    bfm x14, XZR, #0, #0  // bfc x14, #0, #0x1
    csel x13, x13, x10, LT
    cmp x2, x26
    csel x14, x1, x14, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x4, x9, #0x2
    and x3, x9, #0x3
    cbz x4, label_9
    addvl x2, x22, #8
    addvl x1, x22, #12
    ld1b { z3.b }, p1/Z, [x22]
    subs x4, x4, #0x1
    ld1b { z2.b }, p1/Z, [x22, #1, MUL VL]
    ld1b { z1.b }, p1/Z, [x22, #2, MUL VL]
    ld1b { z0.b }, p1/Z, [x22, #3, MUL VL]
    ld1b { z31.b }, p1/Z, [x22, #4, MUL VL]
    ld1b { z30.b }, p1/Z, [x22, #5, MUL VL]
    ld1b { z29.b }, p1/Z, [x22, #6, MUL VL]
    ld1b { z28.b }, p1/Z, [x22, #7, MUL VL]
    addvl x22, x22, #16
    ld1b { z27.b }, p1/Z, [x2]
    ld1b { z26.b }, p1/Z, [x2, #1, MUL VL]
    ld1b { z25.b }, p1/Z, [x2, #2, MUL VL]
    ld1b { z24.b }, p1/Z, [x2, #3, MUL VL]
    ld1b { z23.b }, p1/Z, [x1]
    ld1b { z22.b }, p1/Z, [x1, #1, MUL VL]
    ld1b { z21.b }, p1/Z, [x1, #2, MUL VL]
    ld1b { z20.b }, p1/Z, [x1, #3, MUL VL]
    ld1b { z19.b }, p1/Z, [x23]
    ld1b { z18.b }, p1/Z, [x23, #1, MUL VL]
    ld1b { z17.b }, p1/Z, [x23, #2, MUL VL]
    ld1b { z16.b }, p1/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa0932460)  // smopa za0.s, p1/M, p1/M, z3.b, z19.b
    addvl x2, x22, #8
    addvl x1, x22, #12
    ld1b { z3.b }, p1/Z, [x22]
    KAI_ASM_INST(0xa0932441)  // smopa za1.s, p1/M, p1/M, z2.b, z19.b
    subs x4, x4, #0x1
    ld1b { z2.b }, p1/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa0932422)  // smopa za2.s, p1/M, p1/M, z1.b, z19.b
    ld1b { z1.b }, p1/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa0932403)  // smopa za3.s, p1/M, p1/M, z0.b, z19.b
    ld1b { z0.b }, p1/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0xa09227e0)  // smopa za0.s, p1/M, p1/M, z31.b, z18.b
    ld1b { z31.b }, p1/Z, [x22, #4, MUL VL]
    KAI_ASM_INST(0xa09227c1)  // smopa za1.s, p1/M, p1/M, z30.b, z18.b
    ld1b { z30.b }, p1/Z, [x22, #5, MUL VL]
    KAI_ASM_INST(0xa09227a2)  // smopa za2.s, p1/M, p1/M, z29.b, z18.b
    ld1b { z29.b }, p1/Z, [x22, #6, MUL VL]
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
    ld1b { z28.b }, p1/Z, [x22, #7, MUL VL]
    addvl x22, x22, #16
    KAI_ASM_INST(0xa0912760)  // smopa za0.s, p1/M, p1/M, z27.b, z17.b
    ld1b { z27.b }, p1/Z, [x2]
    KAI_ASM_INST(0xa0912741)  // smopa za1.s, p1/M, p1/M, z26.b, z17.b
    ld1b { z26.b }, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa0912722)  // smopa za2.s, p1/M, p1/M, z25.b, z17.b
    ld1b { z25.b }, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa0912703)  // smopa za3.s, p1/M, p1/M, z24.b, z17.b
    ld1b { z24.b }, p1/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0xa09026e0)  // smopa za0.s, p1/M, p1/M, z23.b, z16.b
    ld1b { z23.b }, p1/Z, [x1]
    KAI_ASM_INST(0xa09026c1)  // smopa za1.s, p1/M, p1/M, z22.b, z16.b
    ld1b { z22.b }, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa09026a2)  // smopa za2.s, p1/M, p1/M, z21.b, z16.b
    ld1b { z21.b }, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa0902683)  // smopa za3.s, p1/M, p1/M, z20.b, z16.b
    ld1b { z20.b }, p1/Z, [x1, #3, MUL VL]
    ld1b { z19.b }, p1/Z, [x23]
    ld1b { z18.b }, p1/Z, [x23, #1, MUL VL]
    ld1b { z17.b }, p1/Z, [x23, #2, MUL VL]
    ld1b { z16.b }, p1/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa0932460)  // smopa za0.s, p1/M, p1/M, z3.b, z19.b
    KAI_ASM_INST(0xa0932441)  // smopa za1.s, p1/M, p1/M, z2.b, z19.b
    KAI_ASM_INST(0xa0932422)  // smopa za2.s, p1/M, p1/M, z1.b, z19.b
    KAI_ASM_INST(0xa0932403)  // smopa za3.s, p1/M, p1/M, z0.b, z19.b
    KAI_ASM_INST(0xa09227e0)  // smopa za0.s, p1/M, p1/M, z31.b, z18.b
    KAI_ASM_INST(0xa09227c1)  // smopa za1.s, p1/M, p1/M, z30.b, z18.b
    KAI_ASM_INST(0xa09227a2)  // smopa za2.s, p1/M, p1/M, z29.b, z18.b
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
    KAI_ASM_INST(0xa0912760)  // smopa za0.s, p1/M, p1/M, z27.b, z17.b
    KAI_ASM_INST(0xa0912741)  // smopa za1.s, p1/M, p1/M, z26.b, z17.b
    KAI_ASM_INST(0xa0912722)  // smopa za2.s, p1/M, p1/M, z25.b, z17.b
    KAI_ASM_INST(0xa0912703)  // smopa za3.s, p1/M, p1/M, z24.b, z17.b
    KAI_ASM_INST(0xa09026e0)  // smopa za0.s, p1/M, p1/M, z23.b, z16.b
    KAI_ASM_INST(0xa09026c1)  // smopa za1.s, p1/M, p1/M, z22.b, z16.b
    KAI_ASM_INST(0xa09026a2)  // smopa za2.s, p1/M, p1/M, z21.b, z16.b
    KAI_ASM_INST(0xa0902683)  // smopa za3.s, p1/M, p1/M, z20.b, z16.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x3, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z20.b }, p1/Z, [x22]
    subs x3, x3, #0x1
    ld1b { z19.b }, p1/Z, [x22, #1, MUL VL]
    ld1b { z18.b }, p1/Z, [x22, #2, MUL VL]
    ld1b { z17.b }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    ld1b { z16.b }, p1/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xa0902680)  // smopa za0.s, p1/M, p1/M, z20.b, z16.b
    KAI_ASM_INST(0xa0902661)  // smopa za1.s, p1/M, p1/M, z19.b, z16.b
    KAI_ASM_INST(0xa0902642)  // smopa za2.s, p1/M, p1/M, z18.b, z16.b
    KAI_ASM_INST(0xa0902623)  // smopa za3.s, p1/M, p1/M, z17.b, z16.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x14, #1, label_15
    tbz x14, #0, label_13
    mov x12, #0
    cntw x21
    cntw x20
    cntw x8, ALL, MUL #2
    cntw x7, ALL, MUL #3
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xe0bf0560)  // st1w { za0h.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x6, x11, #4
    addvl x5, x13, #4
    KAI_ASM_INST(0xe0b40561)  // st1w { za0h.s[x12, #1] }, p1/Z, [x11, x20, LSL #2]
    KAI_ASM_INST(0xe09405a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x20, LSL #2]
    addvl x4, x11, #8
    addvl x3, x13, #8
    KAI_ASM_INST(0xe0a80562)  // st1w { za0h.s[x12, #2] }, p1/Z, [x11, x8, LSL #2]
    KAI_ASM_INST(0xe08805a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x8, LSL #2]
    addvl x2, x11, #12
    addvl x1, x13, #12
    KAI_ASM_INST(0xe0a70563)  // st1w { za0h.s[x12, #3] }, p1/Z, [x11, x7, LSL #2]
    KAI_ASM_INST(0xe08705a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x7, LSL #2]
    addvl x11, x11, #16
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0bf04c4)  // st1w { za1h.s[x12] }, p1/Z, [x6, XZR, LSL #2]
    KAI_ASM_INST(0xe09f04a4)  // ld1w { za1h.s[x12] }, p1/Z, [x5, XZR, LSL #2]
    KAI_ASM_INST(0xe0b404c5)  // st1w { za1h.s[x12, #1] }, p1/Z, [x6, x20, LSL #2]
    KAI_ASM_INST(0xe09404a5)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0xe0a804c6)  // st1w { za1h.s[x12, #2] }, p1/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0xe08804a6)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x5, x8, LSL #2]
    KAI_ASM_INST(0xe0a704c7)  // st1w { za1h.s[x12, #3] }, p1/Z, [x6, x7, LSL #2]
    KAI_ASM_INST(0xe08704a7)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x5, x7, LSL #2]
    KAI_ASM_INST(0xe0bf0488)  // st1w { za2h.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0468)  // ld1w { za2h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b40489)  // st1w { za2h.s[x12, #1] }, p1/Z, [x4, x20, LSL #2]
    KAI_ASM_INST(0xe0940469)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x3, x20, LSL #2]
    KAI_ASM_INST(0xe0a8048a)  // st1w { za2h.s[x12, #2] }, p1/Z, [x4, x8, LSL #2]
    KAI_ASM_INST(0xe088046a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x3, x8, LSL #2]
    KAI_ASM_INST(0xe0a7048b)  // st1w { za2h.s[x12, #3] }, p1/Z, [x4, x7, LSL #2]
    KAI_ASM_INST(0xe087046b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bf044c)  // st1w { za3h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4044d)  // st1w { za3h.s[x12, #1] }, p1/Z, [x2, x20, LSL #2]
    KAI_ASM_INST(0xe094042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x20, LSL #2]
    KAI_ASM_INST(0xe0a8044e)  // st1w { za3h.s[x12, #2] }, p1/Z, [x2, x8, LSL #2]
    KAI_ASM_INST(0xe088042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x8, LSL #2]
    KAI_ASM_INST(0xe0a7044f)  // st1w { za3h.s[x12, #3] }, p1/Z, [x2, x7, LSL #2]
    KAI_ASM_INST(0xe087042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x7, LSL #2]
    add x12, x12, #0x4
    cmp x12, x21
    blt label_12
    b label_31
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xe0bf0560)  // st1w { za0h.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    addvl x3, x11, #4
    addvl x2, x11, #8
    KAI_ASM_INST(0xe0a60561)  // st1w { za0h.s[x12, #1] }, p1/Z, [x11, x6, LSL #2]
    addvl x1, x11, #12
    KAI_ASM_INST(0xe0a50562)  // st1w { za0h.s[x12, #2] }, p1/Z, [x11, x5, LSL #2]
    KAI_ASM_INST(0xe0a40563)  // st1w { za0h.s[x12, #3] }, p1/Z, [x11, x4, LSL #2]
    addvl x11, x11, #16
    KAI_ASM_INST(0xe0bf0464)  // st1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60465)  // st1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0a50466)  // st1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe0a40467)  // st1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe0bf0448)  // st1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60449)  // st1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe0a5044a)  // st1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe0a4044b)  // st1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe0bf042c)  // st1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0a6042d)  // st1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe0a5042e)  // st1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe0a4042f)  // st1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_14
    b label_31
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x26, x28
    KAI_ASM_INST(0x8566c417)  // ld1rw { z23.s }, p1/Z, [x0, #152]
    mov z22.s, #0
    ldr x5, [x0, #0x58]
    ldr x1, [x0, #0x48]
    add x7, x7, x27, LSL #2  // C += n
    madd x7, x28, x5, x7  // C += m * ldc
    cbz x1, label_16
    add x1, x1, x27, LSL #2
    ld1w { z22.s }, p0/Z, [x1]
KAI_ASM_LABEL(label_16)  // Store to output array: no late bias
    cntw x4
    add x2, x0, #0xa0
    cmp x6, x4
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c455)  // ld1rw { z21.s }, p1/Z, [x2]
    csel x3, x6, x4, LT
    KAI_ASM_INST(0x8540c434)  // ld1rw { z20.s }, p1/Z, [x1]
    mov x12, #0
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820413)  // mova z19.s, p1/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820432)  // mova z18.s, p1/M, za0h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc0820451)  // mova z17.s, p1/M, za0h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc0820470)  // mova z16.s, p1/M, za0h.s[x12, #3]
    scvtf z17.s, p1/m, z17.s
    add x12, x12, #0x4
    scvtf z16.s, p1/m, z16.s
    cmp x12, x2, LSL #2
    fmad z19.s, p1/M, z23.s, z22.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z19.s, p1/M, z19.s, z20.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z19.s, p1/M, z19.s, z21.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc0820412)  // mova z18.s, p1/M, za0h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820431)  // mova z17.s, p1/M, za0h.s[x12, #1]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc0820450)  // mova z16.s, p1/M, za0h.s[x12, #2]
    scvtf z17.s, p1/m, z17.s
    scvtf z16.s, p1/m, z16.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    beq label_19
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_19
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_29
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_21
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0820493)  // mova z19.s, p1/M, za1h.s[x12]
    KAI_ASM_INST(0xc08204b2)  // mova z18.s, p1/M, za1h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc08204d1)  // mova z17.s, p1/M, za1h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc08204f0)  // mova z16.s, p1/M, za1h.s[x12, #3]
    scvtf z17.s, p1/m, z17.s
    add x12, x12, #0x4
    scvtf z16.s, p1/m, z16.s
    cmp x12, x2, LSL #2
    fmad z19.s, p1/M, z23.s, z22.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z19.s, p1/M, z19.s, z20.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z19.s, p1/M, z19.s, z21.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0820492)  // mova z18.s, p1/M, za1h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc08204b1)  // mova z17.s, p1/M, za1h.s[x12, #1]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc08204d0)  // mova z16.s, p1/M, za1h.s[x12, #2]
    scvtf z17.s, p1/m, z17.s
    scvtf z16.s, p1/m, z16.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    beq label_22
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_22
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_29
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_24
KAI_ASM_LABEL(label_23)  // Store to output array: Accumulator row 2 loop
    KAI_ASM_INST(0xc0820513)  // mova z19.s, p1/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820532)  // mova z18.s, p1/M, za2h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc0820551)  // mova z17.s, p1/M, za2h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc0820570)  // mova z16.s, p1/M, za2h.s[x12, #3]
    scvtf z17.s, p1/m, z17.s
    add x12, x12, #0x4
    scvtf z16.s, p1/m, z16.s
    cmp x12, x2, LSL #2
    fmad z19.s, p1/M, z23.s, z22.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z19.s, p1/M, z19.s, z20.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z19.s, p1/M, z19.s, z21.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    blt label_23
KAI_ASM_LABEL(label_24)  // Store to output array: Accumulator row 2 oddments
    cbz x1, label_25
    KAI_ASM_INST(0xc0820512)  // mova z18.s, p1/M, za2h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820531)  // mova z17.s, p1/M, za2h.s[x12, #1]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc0820550)  // mova z16.s, p1/M, za2h.s[x12, #2]
    scvtf z17.s, p1/m, z17.s
    scvtf z16.s, p1/m, z16.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    beq label_25
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_25
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_25)  // Store to output array: Accumulator row 2 oddments: End
    subs x6, x6, x3
    beq label_29
    cmp x6, x4
    mov x12, #0
    csel x1, x6, x4, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_27
KAI_ASM_LABEL(label_26)  // Store to output array: Accumulator row 3 loop
    KAI_ASM_INST(0xc0820593)  // mova z19.s, p1/M, za3h.s[x12]
    KAI_ASM_INST(0xc08205b2)  // mova z18.s, p1/M, za3h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc08205d1)  // mova z17.s, p1/M, za3h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc08205f0)  // mova z16.s, p1/M, za3h.s[x12, #3]
    scvtf z17.s, p1/m, z17.s
    add x12, x12, #0x4
    scvtf z16.s, p1/m, z16.s
    cmp x12, x2, LSL #2
    fmad z19.s, p1/M, z23.s, z22.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z19.s, p1/M, z19.s, z20.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z19.s, p1/M, z19.s, z21.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    blt label_26
KAI_ASM_LABEL(label_27)  // Store to output array: Accumulator row 3 oddments
    cbz x1, label_28
    KAI_ASM_INST(0xc0820592)  // mova z18.s, p1/M, za3h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc08205b1)  // mova z17.s, p1/M, za3h.s[x12, #1]
    scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0xc08205d0)  // mova z16.s, p1/M, za3h.s[x12, #2]
    scvtf z17.s, p1/m, z17.s
    scvtf z16.s, p1/m, z16.s
    fmad z18.s, p1/M, z23.s, z22.s
    fmad z17.s, p1/M, z23.s, z22.s
    fmad z16.s, p1/M, z23.s, z22.s
    fmin z18.s, p1/M, z18.s, z20.s
    fmin z17.s, p1/M, z17.s, z20.s
    fmin z16.s, p1/M, z16.s, z20.s
    fmax z18.s, p1/M, z18.s, z21.s
    fmax z17.s, p1/M, z17.s, z21.s
    fmax z16.s, p1/M, z16.s, z21.s
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    beq label_28
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_28
    st1w { z16.s }, p0, [x7]
KAI_ASM_LABEL(label_28)  // Store to output array: Accumulator row 3 oddments: End
KAI_ASM_LABEL(label_29)  // Store to output array: End
    tbz x14, #0, label_31
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_30)  // Store to output array: Refill accumulators: Loop
    addvl x3, x13, #4
    addvl x2, x13, #8
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0464)  // ld1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0448)  // ld1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08605a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860465)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860449)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08505a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850466)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085044a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08405a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840467)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084044b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_30
KAI_ASM_LABEL(label_31)  // End block
    incw x27
    cmp x27, x25
    blt label_4
    ldr x1, [x0, #0x70]
    incw x28, ALL, MUL #4
    mov x11, x10
    cmp x28, x26
    mov x27, #0
    mov x24, x22
    add x10, x11, x1
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL)

    KAI_ASM_END
