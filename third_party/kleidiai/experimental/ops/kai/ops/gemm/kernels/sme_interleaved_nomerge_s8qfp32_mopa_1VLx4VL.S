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

    KAI_ASM_CODE(sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
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
    ptrue p4.b
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
    KAI_ASM_INST(0xe09f11a0)  // ld1w { za0h.s[x12] }, p4/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f1064)  // ld1w { za1h.s[x12] }, p4/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f1048)  // ld1w { za2h.s[x12] }, p4/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f102c)  // ld1w { za3h.s[x12] }, p4/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08611a1)  // ld1w { za0h.s[x12, #1] }, p4/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0861065)  // ld1w { za1h.s[x12, #1] }, p4/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0861049)  // ld1w { za2h.s[x12, #1] }, p4/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086102d)  // ld1w { za3h.s[x12, #1] }, p4/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08511a2)  // ld1w { za0h.s[x12, #2] }, p4/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0851066)  // ld1w { za1h.s[x12, #2] }, p4/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085104a)  // ld1w { za2h.s[x12, #2] }, p4/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085102e)  // ld1w { za3h.s[x12, #2] }, p4/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08411a3)  // ld1w { za0h.s[x12, #3] }, p4/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0841067)  // ld1w { za1h.s[x12, #3] }, p4/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084104b)  // ld1w { za2h.s[x12, #3] }, p4/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084102f)  // ld1w { za3h.s[x12, #3] }, p4/Z, [x1, x4, LSL #2]
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
    mov x1, x27
    mov x22, x24
    whilelt p3.s, x1, x25
    incw x1
    whilelt p2.s, x1, x25
    incw x1
    whilelt p1.s, x1, x25
    incw x1
    whilelt p0.s, x1, x25
    tbnz x14, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    add x1, x1, x27, LSL #2
    ld1w { z19.s }, p3/Z, [x1]
    ld1w { z18.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z17.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x1, #3, MUL VL]
    KAI_ASM_INST(0xc0909260)  // addha za0.s, p4/M, p4/M, z19.s
    KAI_ASM_INST(0xc0909241)  // addha za1.s, p4/M, p4/M, z18.s
    KAI_ASM_INST(0xc0909222)  // addha za2.s, p4/M, p4/M, z17.s
    KAI_ASM_INST(0xc0909203)  // addha za3.s, p4/M, p4/M, z16.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x27
    mov x2, x28
    incw x1, ALL, MUL #4
    incw x2
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
    addvl x2, x23, #8
    addvl x1, x23, #12
    ld1b { z3.b }, p4/Z, [x22]
    subs x4, x4, #0x1
    ld1b { z2.b }, p4/Z, [x22, #1, MUL VL]
    ld1b { z1.b }, p4/Z, [x22, #2, MUL VL]
    ld1b { z0.b }, p4/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    ld1b { z31.b }, p4/Z, [x23]
    ld1b { z30.b }, p4/Z, [x23, #1, MUL VL]
    ld1b { z29.b }, p4/Z, [x23, #2, MUL VL]
    ld1b { z28.b }, p4/Z, [x23, #3, MUL VL]
    ld1b { z27.b }, p4/Z, [x23, #4, MUL VL]
    ld1b { z26.b }, p4/Z, [x23, #5, MUL VL]
    ld1b { z25.b }, p4/Z, [x23, #6, MUL VL]
    ld1b { z24.b }, p4/Z, [x23, #7, MUL VL]
    addvl x23, x23, #16
    ld1b { z23.b }, p4/Z, [x2]
    ld1b { z22.b }, p4/Z, [x2, #1, MUL VL]
    ld1b { z21.b }, p4/Z, [x2, #2, MUL VL]
    ld1b { z20.b }, p4/Z, [x2, #3, MUL VL]
    ld1b { z19.b }, p4/Z, [x1]
    ld1b { z18.b }, p4/Z, [x1, #1, MUL VL]
    ld1b { z17.b }, p4/Z, [x1, #2, MUL VL]
    ld1b { z16.b }, p4/Z, [x1, #3, MUL VL]
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa09f9060)  // smopa za0.s, p4/M, p4/M, z3.b, z31.b
    addvl x2, x23, #8
    addvl x1, x23, #12
    ld1b { z31.b }, p4/Z, [x23]
    KAI_ASM_INST(0xa09e9061)  // smopa za1.s, p4/M, p4/M, z3.b, z30.b
    subs x4, x4, #0x1
    ld1b { z30.b }, p4/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa09d9062)  // smopa za2.s, p4/M, p4/M, z3.b, z29.b
    ld1b { z29.b }, p4/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa09c9063)  // smopa za3.s, p4/M, p4/M, z3.b, z28.b
    ld1b { z3.b }, p4/Z, [x22]
    KAI_ASM_INST(0xa09b9040)  // smopa za0.s, p4/M, p4/M, z2.b, z27.b
    ld1b { z28.b }, p4/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0xa09a9041)  // smopa za1.s, p4/M, p4/M, z2.b, z26.b
    ld1b { z27.b }, p4/Z, [x23, #4, MUL VL]
    KAI_ASM_INST(0xa0999042)  // smopa za2.s, p4/M, p4/M, z2.b, z25.b
    ld1b { z26.b }, p4/Z, [x23, #5, MUL VL]
    KAI_ASM_INST(0xa0989043)  // smopa za3.s, p4/M, p4/M, z2.b, z24.b
    ld1b { z2.b }, p4/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa0979020)  // smopa za0.s, p4/M, p4/M, z1.b, z23.b
    ld1b { z25.b }, p4/Z, [x23, #6, MUL VL]
    KAI_ASM_INST(0xa0969021)  // smopa za1.s, p4/M, p4/M, z1.b, z22.b
    ld1b { z24.b }, p4/Z, [x23, #7, MUL VL]
    addvl x23, x23, #16
    KAI_ASM_INST(0xa0959022)  // smopa za2.s, p4/M, p4/M, z1.b, z21.b
    ld1b { z23.b }, p4/Z, [x2]
    KAI_ASM_INST(0xa0949023)  // smopa za3.s, p4/M, p4/M, z1.b, z20.b
    ld1b { z1.b }, p4/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa0939000)  // smopa za0.s, p4/M, p4/M, z0.b, z19.b
    ld1b { z22.b }, p4/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa0929001)  // smopa za1.s, p4/M, p4/M, z0.b, z18.b
    ld1b { z21.b }, p4/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa0919002)  // smopa za2.s, p4/M, p4/M, z0.b, z17.b
    ld1b { z20.b }, p4/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0xa0909003)  // smopa za3.s, p4/M, p4/M, z0.b, z16.b
    ld1b { z0.b }, p4/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    ld1b { z19.b }, p4/Z, [x1]
    ld1b { z18.b }, p4/Z, [x1, #1, MUL VL]
    ld1b { z17.b }, p4/Z, [x1, #2, MUL VL]
    ld1b { z16.b }, p4/Z, [x1, #3, MUL VL]
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa09f9060)  // smopa za0.s, p4/M, p4/M, z3.b, z31.b
    KAI_ASM_INST(0xa09e9061)  // smopa za1.s, p4/M, p4/M, z3.b, z30.b
    KAI_ASM_INST(0xa09d9062)  // smopa za2.s, p4/M, p4/M, z3.b, z29.b
    KAI_ASM_INST(0xa09c9063)  // smopa za3.s, p4/M, p4/M, z3.b, z28.b
    KAI_ASM_INST(0xa09b9040)  // smopa za0.s, p4/M, p4/M, z2.b, z27.b
    KAI_ASM_INST(0xa09a9041)  // smopa za1.s, p4/M, p4/M, z2.b, z26.b
    KAI_ASM_INST(0xa0999042)  // smopa za2.s, p4/M, p4/M, z2.b, z25.b
    KAI_ASM_INST(0xa0989043)  // smopa za3.s, p4/M, p4/M, z2.b, z24.b
    KAI_ASM_INST(0xa0979020)  // smopa za0.s, p4/M, p4/M, z1.b, z23.b
    KAI_ASM_INST(0xa0969021)  // smopa za1.s, p4/M, p4/M, z1.b, z22.b
    KAI_ASM_INST(0xa0959022)  // smopa za2.s, p4/M, p4/M, z1.b, z21.b
    KAI_ASM_INST(0xa0949023)  // smopa za3.s, p4/M, p4/M, z1.b, z20.b
    KAI_ASM_INST(0xa0939000)  // smopa za0.s, p4/M, p4/M, z0.b, z19.b
    KAI_ASM_INST(0xa0929001)  // smopa za1.s, p4/M, p4/M, z0.b, z18.b
    KAI_ASM_INST(0xa0919002)  // smopa za2.s, p4/M, p4/M, z0.b, z17.b
    KAI_ASM_INST(0xa0909003)  // smopa za3.s, p4/M, p4/M, z0.b, z16.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x3, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z20.b }, p4/Z, [x22]
    subs x3, x3, #0x1
    addvl x22, x22, #1
    ld1b { z19.b }, p4/Z, [x23]
    ld1b { z18.b }, p4/Z, [x23, #1, MUL VL]
    ld1b { z17.b }, p4/Z, [x23, #2, MUL VL]
    ld1b { z16.b }, p4/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa0939280)  // smopa za0.s, p4/M, p4/M, z20.b, z19.b
    KAI_ASM_INST(0xa0929281)  // smopa za1.s, p4/M, p4/M, z20.b, z18.b
    KAI_ASM_INST(0xa0919282)  // smopa za2.s, p4/M, p4/M, z20.b, z17.b
    KAI_ASM_INST(0xa0909283)  // smopa za3.s, p4/M, p4/M, z20.b, z16.b
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
    KAI_ASM_INST(0xe0bf1160)  // st1w { za0h.s[x12] }, p4/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe09f11a0)  // ld1w { za0h.s[x12] }, p4/Z, [x13, XZR, LSL #2]
    addvl x6, x11, #4
    addvl x5, x13, #4
    KAI_ASM_INST(0xe0b41161)  // st1w { za0h.s[x12, #1] }, p4/Z, [x11, x20, LSL #2]
    KAI_ASM_INST(0xe09411a1)  // ld1w { za0h.s[x12, #1] }, p4/Z, [x13, x20, LSL #2]
    addvl x4, x11, #8
    addvl x3, x13, #8
    KAI_ASM_INST(0xe0a81162)  // st1w { za0h.s[x12, #2] }, p4/Z, [x11, x8, LSL #2]
    KAI_ASM_INST(0xe08811a2)  // ld1w { za0h.s[x12, #2] }, p4/Z, [x13, x8, LSL #2]
    addvl x2, x11, #12
    addvl x1, x13, #12
    KAI_ASM_INST(0xe0a71163)  // st1w { za0h.s[x12, #3] }, p4/Z, [x11, x7, LSL #2]
    KAI_ASM_INST(0xe08711a3)  // ld1w { za0h.s[x12, #3] }, p4/Z, [x13, x7, LSL #2]
    addvl x11, x11, #16
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0bf10c4)  // st1w { za1h.s[x12] }, p4/Z, [x6, XZR, LSL #2]
    KAI_ASM_INST(0xe09f10a4)  // ld1w { za1h.s[x12] }, p4/Z, [x5, XZR, LSL #2]
    KAI_ASM_INST(0xe0b410c5)  // st1w { za1h.s[x12, #1] }, p4/Z, [x6, x20, LSL #2]
    KAI_ASM_INST(0xe09410a5)  // ld1w { za1h.s[x12, #1] }, p4/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0xe0a810c6)  // st1w { za1h.s[x12, #2] }, p4/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0xe08810a6)  // ld1w { za1h.s[x12, #2] }, p4/Z, [x5, x8, LSL #2]
    KAI_ASM_INST(0xe0a710c7)  // st1w { za1h.s[x12, #3] }, p4/Z, [x6, x7, LSL #2]
    KAI_ASM_INST(0xe08710a7)  // ld1w { za1h.s[x12, #3] }, p4/Z, [x5, x7, LSL #2]
    KAI_ASM_INST(0xe0bf1088)  // st1w { za2h.s[x12] }, p4/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe09f1068)  // ld1w { za2h.s[x12] }, p4/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b41089)  // st1w { za2h.s[x12, #1] }, p4/Z, [x4, x20, LSL #2]
    KAI_ASM_INST(0xe0941069)  // ld1w { za2h.s[x12, #1] }, p4/Z, [x3, x20, LSL #2]
    KAI_ASM_INST(0xe0a8108a)  // st1w { za2h.s[x12, #2] }, p4/Z, [x4, x8, LSL #2]
    KAI_ASM_INST(0xe088106a)  // ld1w { za2h.s[x12, #2] }, p4/Z, [x3, x8, LSL #2]
    KAI_ASM_INST(0xe0a7108b)  // st1w { za2h.s[x12, #3] }, p4/Z, [x4, x7, LSL #2]
    KAI_ASM_INST(0xe087106b)  // ld1w { za2h.s[x12, #3] }, p4/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bf104c)  // st1w { za3h.s[x12] }, p4/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f102c)  // ld1w { za3h.s[x12] }, p4/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4104d)  // st1w { za3h.s[x12, #1] }, p4/Z, [x2, x20, LSL #2]
    KAI_ASM_INST(0xe094102d)  // ld1w { za3h.s[x12, #1] }, p4/Z, [x1, x20, LSL #2]
    KAI_ASM_INST(0xe0a8104e)  // st1w { za3h.s[x12, #2] }, p4/Z, [x2, x8, LSL #2]
    KAI_ASM_INST(0xe088102e)  // ld1w { za3h.s[x12, #2] }, p4/Z, [x1, x8, LSL #2]
    KAI_ASM_INST(0xe0a7104f)  // st1w { za3h.s[x12, #3] }, p4/Z, [x2, x7, LSL #2]
    KAI_ASM_INST(0xe087102f)  // ld1w { za3h.s[x12, #3] }, p4/Z, [x1, x7, LSL #2]
    add x12, x12, #0x4
    cmp x12, x21
    blt label_12
    b label_22
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xe0bf1160)  // st1w { za0h.s[x12] }, p4/Z, [x11, XZR, LSL #2]
    addvl x3, x11, #4
    addvl x2, x11, #8
    KAI_ASM_INST(0xe0a61161)  // st1w { za0h.s[x12, #1] }, p4/Z, [x11, x6, LSL #2]
    addvl x1, x11, #12
    KAI_ASM_INST(0xe0a51162)  // st1w { za0h.s[x12, #2] }, p4/Z, [x11, x5, LSL #2]
    KAI_ASM_INST(0xe0a41163)  // st1w { za0h.s[x12, #3] }, p4/Z, [x11, x4, LSL #2]
    addvl x11, x11, #16
    KAI_ASM_INST(0xe0bf1064)  // st1w { za1h.s[x12] }, p4/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a61065)  // st1w { za1h.s[x12, #1] }, p4/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0a51066)  // st1w { za1h.s[x12, #2] }, p4/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe0a41067)  // st1w { za1h.s[x12, #3] }, p4/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe0bf1048)  // st1w { za2h.s[x12] }, p4/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe0a61049)  // st1w { za2h.s[x12, #1] }, p4/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe0a5104a)  // st1w { za2h.s[x12, #2] }, p4/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe0a4104b)  // st1w { za2h.s[x12, #3] }, p4/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe0bf102c)  // st1w { za3h.s[x12] }, p4/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0a6102d)  // st1w { za3h.s[x12, #1] }, p4/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe0a5102e)  // st1w { za3h.s[x12, #2] }, p4/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe0a4102f)  // st1w { za3h.s[x12, #3] }, p4/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_14
    b label_22
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x26, x28
    KAI_ASM_INST(0x8566d006)  // ld1rw { z6.s }, p4/Z, [x0, #152]
    mov z5.s, #0
    ldr x4, [x0, #0x58]
    mov z4.s, #0
    mov z3.s, #0
    ldr x1, [x0, #0x48]
    mov z2.s, #0
    add x6, x6, x27, LSL #2  // C += n
    madd x6, x28, x4, x6  // C += m * ldc
    cbz x1, label_16
    add x1, x1, x27, LSL #2
    ld1w { z5.s }, p3/Z, [x1]
    ld1w { z4.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z3.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z2.s }, p0/Z, [x1, #3, MUL VL]
KAI_ASM_LABEL(label_16)  // Store to output array: no late bias
    cntw x3
    add x1, x0, #0xa0
    cmp x5, x3
    add x2, x0, #0xa4
    KAI_ASM_INST(0x8540d021)  // ld1rw { z1.s }, p4/Z, [x1]
    csel x1, x5, x3, LT
    KAI_ASM_INST(0x8540d040)  // ld1rw { z0.s }, p4/Z, [x2]
    mov x12, #0
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc082101f)  // mova z31.s, p4/M, za0h.s[x12]
    KAI_ASM_INST(0xc082109e)  // mova z30.s, p4/M, za1h.s[x12]
    scvtf z31.s, p4/m, z31.s
    KAI_ASM_INST(0xc082111d)  // mova z29.s, p4/M, za2h.s[x12]
    scvtf z30.s, p4/m, z30.s
    KAI_ASM_INST(0xc082119c)  // mova z28.s, p4/M, za3h.s[x12]
    scvtf z29.s, p4/m, z29.s
    KAI_ASM_INST(0xc082103b)  // mova z27.s, p4/M, za0h.s[x12, #1]
    scvtf z28.s, p4/m, z28.s
    KAI_ASM_INST(0xc08210ba)  // mova z26.s, p4/M, za1h.s[x12, #1]
    scvtf z27.s, p4/m, z27.s
    KAI_ASM_INST(0xc0821139)  // mova z25.s, p4/M, za2h.s[x12, #1]
    scvtf z26.s, p4/m, z26.s
    KAI_ASM_INST(0xc08211b8)  // mova z24.s, p4/M, za3h.s[x12, #1]
    scvtf z25.s, p4/m, z25.s
    fmad z31.s, p4/M, z6.s, z5.s
    KAI_ASM_INST(0xc0821057)  // mova z23.s, p4/M, za0h.s[x12, #2]
    scvtf z24.s, p4/m, z24.s
    fmad z30.s, p4/M, z6.s, z4.s
    KAI_ASM_INST(0xc08210d6)  // mova z22.s, p4/M, za1h.s[x12, #2]
    scvtf z23.s, p4/m, z23.s
    fmad z29.s, p4/M, z6.s, z3.s
    KAI_ASM_INST(0xc0821155)  // mova z21.s, p4/M, za2h.s[x12, #2]
    scvtf z22.s, p4/m, z22.s
    fmad z28.s, p4/M, z6.s, z2.s
    KAI_ASM_INST(0xc08211d4)  // mova z20.s, p4/M, za3h.s[x12, #2]
    scvtf z21.s, p4/m, z21.s
    fmad z27.s, p4/M, z6.s, z5.s
    KAI_ASM_INST(0xc0821073)  // mova z19.s, p4/M, za0h.s[x12, #3]
    scvtf z20.s, p4/m, z20.s
    fmad z26.s, p4/M, z6.s, z4.s
    KAI_ASM_INST(0xc08210f2)  // mova z18.s, p4/M, za1h.s[x12, #3]
    scvtf z19.s, p4/m, z19.s
    fmad z25.s, p4/M, z6.s, z3.s
    KAI_ASM_INST(0xc0821171)  // mova z17.s, p4/M, za2h.s[x12, #3]
    scvtf z18.s, p4/m, z18.s
    fmad z24.s, p4/M, z6.s, z2.s
    KAI_ASM_INST(0xc08211f0)  // mova z16.s, p4/M, za3h.s[x12, #3]
    scvtf z17.s, p4/m, z17.s
    fmad z23.s, p4/M, z6.s, z5.s
    add x12, x12, #0x4
    scvtf z16.s, p4/m, z16.s
    fmad z22.s, p4/M, z6.s, z4.s
    cmp x12, x2, LSL #2
    fmad z21.s, p4/M, z6.s, z3.s
    fmad z20.s, p4/M, z6.s, z2.s
    fmad z19.s, p4/M, z6.s, z5.s
    fmad z18.s, p4/M, z6.s, z4.s
    fmad z17.s, p4/M, z6.s, z3.s
    fmin z31.s, p4/M, z31.s, z0.s
    fmad z16.s, p4/M, z6.s, z2.s
    fmin z30.s, p4/M, z30.s, z0.s
    fmin z29.s, p4/M, z29.s, z0.s
    fmin z28.s, p4/M, z28.s, z0.s
    fmin z27.s, p4/M, z27.s, z0.s
    fmin z26.s, p4/M, z26.s, z0.s
    fmin z25.s, p4/M, z25.s, z0.s
    fmin z24.s, p4/M, z24.s, z0.s
    fmin z23.s, p4/M, z23.s, z0.s
    fmin z22.s, p4/M, z22.s, z0.s
    fmin z21.s, p4/M, z21.s, z0.s
    fmin z20.s, p4/M, z20.s, z0.s
    fmin z19.s, p4/M, z19.s, z0.s
    fmin z18.s, p4/M, z18.s, z0.s
    fmin z17.s, p4/M, z17.s, z0.s
    fmin z16.s, p4/M, z16.s, z0.s
    fmax z31.s, p4/M, z31.s, z1.s
    fmax z30.s, p4/M, z30.s, z1.s
    fmax z29.s, p4/M, z29.s, z1.s
    fmax z28.s, p4/M, z28.s, z1.s
    fmax z27.s, p4/M, z27.s, z1.s
    fmax z26.s, p4/M, z26.s, z1.s
    fmax z25.s, p4/M, z25.s, z1.s
    fmax z24.s, p4/M, z24.s, z1.s
    st1w { z31.s }, p3, [x6]
    fmax z23.s, p4/M, z23.s, z1.s
    fmax z22.s, p4/M, z22.s, z1.s
    st1w { z30.s }, p2, [x6, #1, MUL VL]
    fmax z21.s, p4/M, z21.s, z1.s
    fmax z20.s, p4/M, z20.s, z1.s
    st1w { z29.s }, p1, [x6, #2, MUL VL]
    fmax z19.s, p4/M, z19.s, z1.s
    fmax z18.s, p4/M, z18.s, z1.s
    st1w { z28.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    fmax z17.s, p4/M, z17.s, z1.s
    fmax z16.s, p4/M, z16.s, z1.s
    st1w { z27.s }, p3, [x6]
    st1w { z26.s }, p2, [x6, #1, MUL VL]
    st1w { z25.s }, p1, [x6, #2, MUL VL]
    st1w { z24.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    st1w { z23.s }, p3, [x6]
    st1w { z22.s }, p2, [x6, #1, MUL VL]
    st1w { z21.s }, p1, [x6, #2, MUL VL]
    st1w { z20.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    st1w { z19.s }, p3, [x6]
    st1w { z18.s }, p2, [x6, #1, MUL VL]
    st1w { z17.s }, p1, [x6, #2, MUL VL]
    st1w { z16.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc082101b)  // mova z27.s, p4/M, za0h.s[x12]
    KAI_ASM_INST(0xc082103a)  // mova z26.s, p4/M, za0h.s[x12, #1]
    scvtf z27.s, p4/m, z27.s
    KAI_ASM_INST(0xc0821059)  // mova z25.s, p4/M, za0h.s[x12, #2]
    scvtf z26.s, p4/m, z26.s
    KAI_ASM_INST(0xc0821098)  // mova z24.s, p4/M, za1h.s[x12]
    scvtf z25.s, p4/m, z25.s
    KAI_ASM_INST(0xc08210b7)  // mova z23.s, p4/M, za1h.s[x12, #1]
    scvtf z24.s, p4/m, z24.s
    KAI_ASM_INST(0xc08210d6)  // mova z22.s, p4/M, za1h.s[x12, #2]
    scvtf z23.s, p4/m, z23.s
    fmad z27.s, p4/M, z6.s, z5.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0821115)  // mova z21.s, p4/M, za2h.s[x12]
    scvtf z22.s, p4/m, z22.s
    fmad z26.s, p4/M, z6.s, z5.s
    KAI_ASM_INST(0xc0821134)  // mova z20.s, p4/M, za2h.s[x12, #1]
    scvtf z21.s, p4/m, z21.s
    fmad z25.s, p4/M, z6.s, z5.s
    KAI_ASM_INST(0xc0821153)  // mova z19.s, p4/M, za2h.s[x12, #2]
    scvtf z20.s, p4/m, z20.s
    fmad z24.s, p4/M, z6.s, z4.s
    KAI_ASM_INST(0xc0821192)  // mova z18.s, p4/M, za3h.s[x12]
    scvtf z19.s, p4/m, z19.s
    fmad z23.s, p4/M, z6.s, z4.s
    KAI_ASM_INST(0xc08211b1)  // mova z17.s, p4/M, za3h.s[x12, #1]
    scvtf z18.s, p4/m, z18.s
    fmad z22.s, p4/M, z6.s, z4.s
    KAI_ASM_INST(0xc08211d0)  // mova z16.s, p4/M, za3h.s[x12, #2]
    scvtf z17.s, p4/m, z17.s
    fmad z21.s, p4/M, z6.s, z3.s
    scvtf z16.s, p4/m, z16.s
    fmad z20.s, p4/M, z6.s, z3.s
    fmad z19.s, p4/M, z6.s, z3.s
    fmin z27.s, p4/M, z27.s, z0.s
    fmad z18.s, p4/M, z6.s, z2.s
    fmin z26.s, p4/M, z26.s, z0.s
    fmad z17.s, p4/M, z6.s, z2.s
    fmin z25.s, p4/M, z25.s, z0.s
    fmad z16.s, p4/M, z6.s, z2.s
    fmin z24.s, p4/M, z24.s, z0.s
    fmin z23.s, p4/M, z23.s, z0.s
    fmin z22.s, p4/M, z22.s, z0.s
    fmin z21.s, p4/M, z21.s, z0.s
    fmin z20.s, p4/M, z20.s, z0.s
    fmin z19.s, p4/M, z19.s, z0.s
    fmin z18.s, p4/M, z18.s, z0.s
    fmin z17.s, p4/M, z17.s, z0.s
    fmin z16.s, p4/M, z16.s, z0.s
    fmax z27.s, p4/M, z27.s, z1.s
    fmax z26.s, p4/M, z26.s, z1.s
    fmax z25.s, p4/M, z25.s, z1.s
    fmax z24.s, p4/M, z24.s, z1.s
    fmax z23.s, p4/M, z23.s, z1.s
    fmax z22.s, p4/M, z22.s, z1.s
    fmax z21.s, p4/M, z21.s, z1.s
    fmax z20.s, p4/M, z20.s, z1.s
    fmax z19.s, p4/M, z19.s, z1.s
    fmax z18.s, p4/M, z18.s, z1.s
    st1w { z27.s }, p3, [x6]
    fmax z17.s, p4/M, z17.s, z1.s
    fmax z16.s, p4/M, z16.s, z1.s
    st1w { z24.s }, p2, [x6, #1, MUL VL]
    st1w { z21.s }, p1, [x6, #2, MUL VL]
    st1w { z18.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    beq label_19
    subs x1, x1, #0x1
    st1w { z26.s }, p3, [x6]
    st1w { z23.s }, p2, [x6, #1, MUL VL]
    st1w { z20.s }, p1, [x6, #2, MUL VL]
    st1w { z17.s }, p0, [x6, #3, MUL VL]
    add x6, x6, x4
    beq label_19
    st1w { z25.s }, p3, [x6]
    st1w { z22.s }, p2, [x6, #1, MUL VL]
    st1w { z19.s }, p1, [x6, #2, MUL VL]
    st1w { z16.s }, p0, [x6, #3, MUL VL]
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    tbz x14, #0, label_22
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_21)  // Store to output array: Refill accumulators: Loop
    addvl x3, x13, #4
    addvl x2, x13, #8
    KAI_ASM_INST(0xe09f11a0)  // ld1w { za0h.s[x12] }, p4/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f1064)  // ld1w { za1h.s[x12] }, p4/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f1048)  // ld1w { za2h.s[x12] }, p4/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f102c)  // ld1w { za3h.s[x12] }, p4/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08611a1)  // ld1w { za0h.s[x12, #1] }, p4/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0861065)  // ld1w { za1h.s[x12, #1] }, p4/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0861049)  // ld1w { za2h.s[x12, #1] }, p4/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086102d)  // ld1w { za3h.s[x12, #1] }, p4/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08511a2)  // ld1w { za0h.s[x12, #2] }, p4/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0851066)  // ld1w { za1h.s[x12, #2] }, p4/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085104a)  // ld1w { za2h.s[x12, #2] }, p4/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085102e)  // ld1w { za3h.s[x12, #2] }, p4/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08411a3)  // ld1w { za0h.s[x12, #3] }, p4/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0841067)  // ld1w { za1h.s[x12, #3] }, p4/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084104b)  // ld1w { za2h.s[x12, #3] }, p4/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084102f)  // ld1w { za3h.s[x12, #3] }, p4/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_21
KAI_ASM_LABEL(label_22)  // End block
    incw x27, ALL, MUL #4
    cmp x27, x25
    blt label_4
    ldr x1, [x0, #0x70]
    incw x28
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)

    KAI_ASM_END
