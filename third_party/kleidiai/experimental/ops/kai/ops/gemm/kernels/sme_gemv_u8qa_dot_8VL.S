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

    KAI_ASM_CODE(sme_gemv_u8qa_dot_8VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_gemv_u8qa_dot_8VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_gemv_u8qa_dot_8VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_gemv_u8qa_dot_8VL)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x24, #0
    ldr x1, [x0, #0x78]
    cntw x23
    ptrue p2.b
    mov z22.s, #0
    ldr x22, [x0, #0x8]
    mov z23.b, #0x1
    ldr x21, [x0, #0x10]
    cmp x1, #0
    ldr x20, [x0, #0x30]
    cset x1, NE
    add x8, x22, x23
    ldr x7, [x0, #0x50]
    orr x24, x24, x1, LSL #4
    sub x8, x8, #0x1
    ldr x6, [x0, #0x40]
    udiv x8, x8, x23
    bic x24, x24, #0x80000000
KAI_ASM_LABEL(label_1)  // Column loop
    cmp x8, #0x8
    bge label_57
    cmp x8, #0x6
    bgt label_49
    beq label_41
    cmp x8, #0x4
    bgt label_33
    beq label_25
    cmp x8, #0x2
    bgt label_17
    beq label_9
    mov x3, x21
    ldr x2, [x0, #0x20]
    whilelt p1.b, XZR, x22
    mov z24.s, #0
    cmp x3, #0x10
    ble label_5
KAI_ASM_LABEL(label_3)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    addvl x20, x20, #8
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20]
    addvl x20, x20, #8
    ldnt1b { z3.b }, p2/Z, [x20]
    addvl x20, x20, #8
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z4.b }, p2/Z, [x20]
    addvl x20, x20, #8
    udot z24.s, z2.b, z0.b[1]
    udot z24.s, z3.b, z0.b[2]
    udot z24.s, z4.b, z0.b[3]
    tbnz x24, #31, label_4
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_4)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_3
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z5.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    addvl x20, x20, #8
    udot z24.s, z5.b, z0.b[0]
    ble label_6
    ldnt1b { z6.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    addvl x20, x20, #8
    udot z24.s, z6.b, z0.b[1]
    ble label_6
    ldnt1b { z7.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    addvl x20, x20, #8
    udot z24.s, z7.b, z0.b[2]
    ble label_6
    ldnt1b { z8.b }, p2/Z, [x20]
    udot z24.s, z8.b, z0.b[3]
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: multiply skip
    tbnz x24, #31, label_7
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: unique 2: skip row sum
    tbnz x24, #31, label_8
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c841)  // ld1rw { z1.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z1.s, p2/M, z1.s
    mul z22.s, p2/M, z22.s, z1.s
KAI_ASM_LABEL(label_8)  // Width 1: skip row sum fixup
    add z24.s, z24.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x1, x0, #0x98
    add x4, x0, #0x9c
    KAI_ASM_INST(0x8540c828)  // ld1rw { z8.s }, p2/Z, [x1]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    add z24.s, z24.s, z0.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    add z24.s, z24.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z24.b, z24.b, z25.b
    st1b { z24.b }, p1, [x7]
    b label_65
KAI_ASM_LABEL(label_9)  // Width 2
    mov x3, x21
    ldr x2, [x0, #0x20]
    whilelt p1.b, XZR, x22
    mov z24.s, #0
    mov z25.s, #0
    cmp x3, #0x10
    ble label_13
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    ldnt1b { z3.b }, p2/Z, [x20]
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z4.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20]
    ldnt1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    ldnt1b { z7.b }, p2/Z, [x20]
    udot z24.s, z3.b, z0.b[1]
    ldnt1b { z8.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    udot z25.s, z4.b, z0.b[1]
    udot z24.s, z5.b, z0.b[2]
    udot z25.s, z6.b, z0.b[2]
    udot z24.s, z7.b, z0.b[3]
    udot z25.s, z8.b, z0.b[3]
    tbnz x24, #31, label_12
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: unique 3: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_11
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z9.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z10.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z9.b, z0.b[0]
    udot z25.s, z10.b, z0.b[0]
    ble label_14
    ldnt1b { z11.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z12.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z11.b, z0.b[1]
    udot z25.s, z12.b, z0.b[1]
    ble label_14
    ldnt1b { z13.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z14.b }, p2/Z, [x20, #1, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z13.b, z0.b[2]
    udot z25.s, z14.b, z0.b[2]
    ble label_14
    ldnt1b { z15.b }, p2/Z, [x20]
    ldnt1b { z16.b }, p2/Z, [x20, #1, MUL VL]
    udot z24.s, z15.b, z0.b[3]
    udot z25.s, z16.b, z0.b[3]
KAI_ASM_LABEL(label_14)  // Width 2: Multiply loop: multiply skip
    tbnz x24, #31, label_15
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: unique 4: skip row sum
    tbnz x24, #31, label_16
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c842)  // ld1rw { z2.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z2.s, p2/M, z2.s
    mul z22.s, p2/M, z22.s, z2.s
KAI_ASM_LABEL(label_16)  // Width 2: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x1, x0, #0x98
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add x3, x0, #0x94
    KAI_ASM_INST(0x8540c828)  // ld1rw { z8.s }, p2/Z, [x1]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z24.b, z24.b, z25.b
    st1b { z24.b }, p1, [x7]
    b label_65
KAI_ASM_LABEL(label_17)  // Width 3
    mov x3, x21
    ldr x2, [x0, #0x20]
    whilelt p1.b, XZR, x22
    mov z24.s, #0
    mov z25.s, #0
    mov z26.s, #0
    cmp x3, #0x10
    ble label_21
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z4.b }, p2/Z, [x20]
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20, #1, MUL VL]
    udot z26.s, z3.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    ldnt1b { z7.b }, p2/Z, [x20]
    udot z24.s, z4.b, z0.b[1]
    ldnt1b { z8.b }, p2/Z, [x20, #1, MUL VL]
    udot z25.s, z5.b, z0.b[1]
    ldnt1b { z9.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z26.s, z6.b, z0.b[1]
    ldnt1b { z10.b }, p2/Z, [x20]
    ldnt1b { z11.b }, p2/Z, [x20, #1, MUL VL]
    udot z24.s, z7.b, z0.b[2]
    ldnt1b { z12.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z25.s, z8.b, z0.b[2]
    udot z26.s, z9.b, z0.b[2]
    udot z24.s, z10.b, z0.b[3]
    udot z25.s, z11.b, z0.b[3]
    udot z26.s, z12.b, z0.b[3]
    tbnz x24, #31, label_20
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: unique 5: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_19
KAI_ASM_LABEL(label_21)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z13.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z14.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z15.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z13.b, z0.b[0]
    udot z25.s, z14.b, z0.b[0]
    udot z26.s, z15.b, z0.b[0]
    ble label_22
    ldnt1b { z16.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z17.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z16.b, z0.b[1]
    udot z25.s, z17.b, z0.b[1]
    udot z26.s, z18.b, z0.b[1]
    ble label_22
    ldnt1b { z19.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z20.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z21.b }, p2/Z, [x20, #2, MUL VL]
    addvl x20, x20, #8
    udot z24.s, z19.b, z0.b[2]
    udot z25.s, z20.b, z0.b[2]
    udot z26.s, z21.b, z0.b[2]
    ble label_22
    ldnt1b { z1.b }, p2/Z, [x20]
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    udot z24.s, z1.b, z0.b[3]
    udot z25.s, z2.b, z0.b[3]
    udot z26.s, z3.b, z0.b[3]
KAI_ASM_LABEL(label_22)  // Width 3: Multiply loop: multiply skip
    tbnz x24, #31, label_23
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: unique 6: skip row sum
    tbnz x24, #31, label_24
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c843)  // ld1rw { z3.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z3.s, p2/M, z3.s
    mul z22.s, p2/M, z22.s, z3.s
KAI_ASM_LABEL(label_24)  // Width 3: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add x3, x0, #0x94
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z26.s, z26.s, z2.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z24.b, z24.b, z25.b
    st1b { z24.b }, p1, [x7]
    b label_65
KAI_ASM_LABEL(label_25)  // Width 4
    mov x3, x21
    ldr x2, [x0, #0x20]
    whilelt p1.b, XZR, x22
    mov z24.s, #0
    mov z25.s, #0
    mov z26.s, #0
    mov z27.s, #0
    cmp x3, #0x10
    ble label_29
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[0]
    addvl x20, x20, #8
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20]
    udot z26.s, z3.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    udot z27.s, z4.b, z0.b[0]
    ldnt1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z5.b, z0.b[1]
    addvl x20, x20, #8
    udot z25.s, z6.b, z0.b[1]
    ldnt1b { z9.b }, p2/Z, [x20]
    udot z26.s, z7.b, z0.b[1]
    ldnt1b { z10.b }, p2/Z, [x20, #1, MUL VL]
    udot z27.s, z8.b, z0.b[1]
    ldnt1b { z11.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z12.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z9.b, z0.b[2]
    addvl x20, x20, #8
    udot z25.s, z10.b, z0.b[2]
    ldnt1b { z13.b }, p2/Z, [x20]
    udot z26.s, z11.b, z0.b[2]
    ldnt1b { z14.b }, p2/Z, [x20, #1, MUL VL]
    udot z27.s, z12.b, z0.b[2]
    ldnt1b { z15.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z16.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z13.b, z0.b[3]
    addvl x20, x20, #8
    udot z25.s, z14.b, z0.b[3]
    udot z26.s, z15.b, z0.b[3]
    udot z27.s, z16.b, z0.b[3]
    tbnz x24, #31, label_28
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_28)  // Width 4: Multiply loop: unique 7: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_27
KAI_ASM_LABEL(label_29)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z17.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z18.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z19.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z20.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z17.b, z0.b[0]
    addvl x20, x20, #8
    udot z25.s, z18.b, z0.b[0]
    udot z26.s, z19.b, z0.b[0]
    udot z27.s, z20.b, z0.b[0]
    ble label_30
    ldnt1b { z21.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z1.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z2.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z21.b, z0.b[1]
    addvl x20, x20, #8
    udot z25.s, z1.b, z0.b[1]
    udot z26.s, z2.b, z0.b[1]
    udot z27.s, z3.b, z0.b[1]
    ble label_30
    ldnt1b { z4.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z5.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z6.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z7.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z4.b, z0.b[2]
    addvl x20, x20, #8
    udot z25.s, z5.b, z0.b[2]
    udot z26.s, z6.b, z0.b[2]
    udot z27.s, z7.b, z0.b[2]
    ble label_30
    ldnt1b { z8.b }, p2/Z, [x20]
    ldnt1b { z9.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z10.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z11.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z8.b, z0.b[3]
    udot z25.s, z9.b, z0.b[3]
    udot z26.s, z10.b, z0.b[3]
    udot z27.s, z11.b, z0.b[3]
KAI_ASM_LABEL(label_30)  // Width 4: Multiply loop: multiply skip
    tbnz x24, #31, label_31
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_31)  // Width 4: Multiply loop: unique 8: skip row sum
    tbnz x24, #31, label_32
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c844)  // ld1rw { z4.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z4.s, p2/M, z4.s
    mul z22.s, p2/M, z22.s, z4.s
KAI_ASM_LABEL(label_32)  // Width 4: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    add z27.s, z27.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    ld1w { z3.s }, p2/Z, [x6, #3, MUL VL]
    add x1, x0, #0xa4
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z26.s, z26.s, z2.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z27.s, z27.s, z3.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x04a8737b)  // sqdmulh z27.s, z27.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    add z27.s, z27.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13b)  // sclamp z27.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z24.b, z24.b, z25.b
    st1b { z24.b }, p1, [x7]
    b label_65
KAI_ASM_LABEL(label_33)  // Width 5
    lsl x1, x23, #0x2
    mov x3, x21
    ldr x2, [x0, #0x20]
    mov z24.s, #0
    sub x1, x22, x1
    mov z25.s, #0
    mov z26.s, #0
    whilelt p1.b, XZR, x1
    mov z27.s, #0
    mov z28.s, #0
    cmp x3, #0x10
    ble label_37
KAI_ASM_LABEL(label_35)  // Width 5: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[0]
    addvl x20, x20, #8
    udot z26.s, z3.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20]
    udot z27.s, z4.b, z0.b[0]
    ldnt1b { z7.b }, p2/Z, [x20, #1, MUL VL]
    udot z28.s, z5.b, z0.b[0]
    ldnt1b { z8.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z9.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z6.b, z0.b[1]
    ldnt1b { z10.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z7.b, z0.b[1]
    addvl x20, x20, #8
    udot z26.s, z8.b, z0.b[1]
    ldnt1b { z11.b }, p2/Z, [x20]
    udot z27.s, z9.b, z0.b[1]
    ldnt1b { z12.b }, p2/Z, [x20, #1, MUL VL]
    udot z28.s, z10.b, z0.b[1]
    ldnt1b { z13.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z14.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z11.b, z0.b[2]
    ldnt1b { z15.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z12.b, z0.b[2]
    addvl x20, x20, #8
    udot z26.s, z13.b, z0.b[2]
    ldnt1b { z16.b }, p2/Z, [x20]
    udot z27.s, z14.b, z0.b[2]
    ldnt1b { z17.b }, p2/Z, [x20, #1, MUL VL]
    udot z28.s, z15.b, z0.b[2]
    ldnt1b { z18.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z19.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z16.b, z0.b[3]
    ldnt1b { z20.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z17.b, z0.b[3]
    addvl x20, x20, #8
    udot z26.s, z18.b, z0.b[3]
    udot z27.s, z19.b, z0.b[3]
    udot z28.s, z20.b, z0.b[3]
    tbnz x24, #31, label_36
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_36)  // Width 5: Multiply loop: unique 9: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_35
KAI_ASM_LABEL(label_37)  // Width 5: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z21.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z1.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z2.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z21.b, z0.b[0]
    ldnt1b { z4.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z1.b, z0.b[0]
    addvl x20, x20, #8
    udot z26.s, z2.b, z0.b[0]
    udot z27.s, z3.b, z0.b[0]
    udot z28.s, z4.b, z0.b[0]
    ble label_38
    ldnt1b { z5.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z5.b, z0.b[1]
    ldnt1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z6.b, z0.b[1]
    addvl x20, x20, #8
    udot z26.s, z7.b, z0.b[1]
    udot z27.s, z8.b, z0.b[1]
    udot z28.s, z9.b, z0.b[1]
    ble label_38
    ldnt1b { z10.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z11.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z12.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z13.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z10.b, z0.b[2]
    ldnt1b { z14.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z11.b, z0.b[2]
    addvl x20, x20, #8
    udot z26.s, z12.b, z0.b[2]
    udot z27.s, z13.b, z0.b[2]
    udot z28.s, z14.b, z0.b[2]
    ble label_38
    ldnt1b { z15.b }, p2/Z, [x20]
    ldnt1b { z16.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z17.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z15.b, z0.b[3]
    ldnt1b { z19.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z16.b, z0.b[3]
    udot z26.s, z17.b, z0.b[3]
    udot z27.s, z18.b, z0.b[3]
    udot z28.s, z19.b, z0.b[3]
KAI_ASM_LABEL(label_38)  // Width 5: Multiply loop: multiply skip
    tbnz x24, #31, label_39
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_39)  // Width 5: Multiply loop: unique 10: skip row sum
    tbnz x24, #31, label_40
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c845)  // ld1rw { z5.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z5.s, p2/M, z5.s
    mul z22.s, p2/M, z22.s, z5.s
KAI_ASM_LABEL(label_40)  // Width 5: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    add z27.s, z27.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add z28.s, z28.s, z22.s
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    ld1w { z3.s }, p2/Z, [x6, #3, MUL VL]
    add x1, x0, #0xa4
    ld1w { z4.s }, p2/Z, [x6, #4, MUL VL]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    add z26.s, z26.s, z2.s
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z27.s, z27.s, z3.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z28.s, z28.s, z4.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x04a8737b)  // sqdmulh z27.s, z27.s, z8.s
    KAI_ASM_INST(0x04a8739c)  // sqdmulh z28.s, z28.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    add z27.s, z27.s, z8.s
    add z28.s, z28.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13b)  // sclamp z27.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13c)  // sclamp z28.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z26.h, z28.h, z29.h
    uzp1 z24.b, z24.b, z25.b
    uzp1 z25.b, z26.b, z27.b
    st1b { z24.b }, p2, [x7]
    st1b { z25.b }, p1, [x7, #1, MUL VL]
    b label_65
KAI_ASM_LABEL(label_41)  // Width 6
    lsl x1, x23, #0x2
    mov x3, x21
    ldr x2, [x0, #0x20]
    mov z24.s, #0
    sub x1, x22, x1
    mov z25.s, #0
    mov z26.s, #0
    whilelt p1.b, XZR, x1
    mov z27.s, #0
    mov z28.s, #0
    mov z29.s, #0
    cmp x3, #0x10
    ble label_45
KAI_ASM_LABEL(label_43)  // Width 6: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[0]
    addvl x20, x20, #8
    udot z27.s, z4.b, z0.b[0]
    ldnt1b { z7.b }, p2/Z, [x20]
    udot z28.s, z5.b, z0.b[0]
    ldnt1b { z8.b }, p2/Z, [x20, #1, MUL VL]
    udot z29.s, z6.b, z0.b[0]
    ldnt1b { z9.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z10.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z7.b, z0.b[1]
    ldnt1b { z11.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z8.b, z0.b[1]
    ldnt1b { z12.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z9.b, z0.b[1]
    addvl x20, x20, #8
    udot z27.s, z10.b, z0.b[1]
    ldnt1b { z13.b }, p2/Z, [x20]
    udot z28.s, z11.b, z0.b[1]
    ldnt1b { z14.b }, p2/Z, [x20, #1, MUL VL]
    udot z29.s, z12.b, z0.b[1]
    ldnt1b { z15.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z16.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z13.b, z0.b[2]
    ldnt1b { z17.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z14.b, z0.b[2]
    ldnt1b { z18.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z15.b, z0.b[2]
    addvl x20, x20, #8
    udot z27.s, z16.b, z0.b[2]
    ldnt1b { z19.b }, p2/Z, [x20]
    udot z28.s, z17.b, z0.b[2]
    ldnt1b { z20.b }, p2/Z, [x20, #1, MUL VL]
    udot z29.s, z18.b, z0.b[2]
    ldnt1b { z21.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z1.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z19.b, z0.b[3]
    ldnt1b { z2.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z20.b, z0.b[3]
    ldnt1b { z3.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z21.b, z0.b[3]
    addvl x20, x20, #8
    udot z27.s, z1.b, z0.b[3]
    udot z28.s, z2.b, z0.b[3]
    udot z29.s, z3.b, z0.b[3]
    tbnz x24, #31, label_44
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_44)  // Width 6: Multiply loop: unique 11: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_43
KAI_ASM_LABEL(label_45)  // Width 6: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z4.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z5.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z6.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z7.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z4.b, z0.b[0]
    ldnt1b { z8.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z5.b, z0.b[0]
    ldnt1b { z9.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z6.b, z0.b[0]
    addvl x20, x20, #8
    udot z27.s, z7.b, z0.b[0]
    udot z28.s, z8.b, z0.b[0]
    udot z29.s, z9.b, z0.b[0]
    ble label_46
    ldnt1b { z10.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z11.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z12.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z13.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z10.b, z0.b[1]
    ldnt1b { z14.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z11.b, z0.b[1]
    ldnt1b { z15.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z12.b, z0.b[1]
    addvl x20, x20, #8
    udot z27.s, z13.b, z0.b[1]
    udot z28.s, z14.b, z0.b[1]
    udot z29.s, z15.b, z0.b[1]
    ble label_46
    ldnt1b { z16.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z17.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z19.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z16.b, z0.b[2]
    ldnt1b { z20.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z17.b, z0.b[2]
    ldnt1b { z21.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z18.b, z0.b[2]
    addvl x20, x20, #8
    udot z27.s, z19.b, z0.b[2]
    udot z28.s, z20.b, z0.b[2]
    udot z29.s, z21.b, z0.b[2]
    ble label_46
    ldnt1b { z1.b }, p2/Z, [x20]
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[3]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[3]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[3]
    udot z27.s, z4.b, z0.b[3]
    udot z28.s, z5.b, z0.b[3]
    udot z29.s, z6.b, z0.b[3]
KAI_ASM_LABEL(label_46)  // Width 6: Multiply loop: multiply skip
    tbnz x24, #31, label_47
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_47)  // Width 6: Multiply loop: unique 12: skip row sum
    tbnz x24, #31, label_48
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z6.s, p2/M, z6.s
    mul z22.s, p2/M, z22.s, z6.s
KAI_ASM_LABEL(label_48)  // Width 6: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    add z27.s, z27.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add z28.s, z28.s, z22.s
    add z29.s, z29.s, z22.s
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x3, x0, #0x94
    ld1w { z3.s }, p2/Z, [x6, #3, MUL VL]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    ld1w { z4.s }, p2/Z, [x6, #4, MUL VL]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    ld1w { z5.s }, p2/Z, [x6, #5, MUL VL]
    add z26.s, z26.s, z2.s
    add z27.s, z27.s, z3.s
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z28.s, z28.s, z4.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z29.s, z29.s, z5.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x04a8737b)  // sqdmulh z27.s, z27.s, z8.s
    KAI_ASM_INST(0x04a8739c)  // sqdmulh z28.s, z28.s, z8.s
    KAI_ASM_INST(0x04a873bd)  // sqdmulh z29.s, z29.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    KAI_ASM_INST(0x4482881d)  // srshl z29.s, p2/M, z29.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    add z27.s, z27.s, z8.s
    add z28.s, z28.s, z8.s
    add z29.s, z29.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13b)  // sclamp z27.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13c)  // sclamp z28.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13d)  // sclamp z29.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z26.h, z28.h, z29.h
    uzp1 z24.b, z24.b, z25.b
    uzp1 z25.b, z26.b, z27.b
    st1b { z24.b }, p2, [x7]
    st1b { z25.b }, p1, [x7, #1, MUL VL]
    b label_65
KAI_ASM_LABEL(label_49)  // Width 7
    lsl x1, x23, #0x2
    mov x3, x21
    ldr x2, [x0, #0x20]
    mov z24.s, #0
    sub x1, x22, x1
    mov z25.s, #0
    mov z26.s, #0
    whilelt p1.b, XZR, x1
    mov z27.s, #0
    mov z28.s, #0
    mov z29.s, #0
    mov z30.s, #0
    cmp x3, #0x10
    ble label_53
KAI_ASM_LABEL(label_51)  // Width 7: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[0]
    ldnt1b { z7.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z4.b, z0.b[0]
    addvl x20, x20, #8
    udot z28.s, z5.b, z0.b[0]
    ldnt1b { z8.b }, p2/Z, [x20]
    udot z29.s, z6.b, z0.b[0]
    ldnt1b { z9.b }, p2/Z, [x20, #1, MUL VL]
    udot z30.s, z7.b, z0.b[0]
    ldnt1b { z10.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z11.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z8.b, z0.b[1]
    ldnt1b { z12.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z9.b, z0.b[1]
    ldnt1b { z13.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z10.b, z0.b[1]
    ldnt1b { z14.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z11.b, z0.b[1]
    addvl x20, x20, #8
    udot z28.s, z12.b, z0.b[1]
    ldnt1b { z15.b }, p2/Z, [x20]
    udot z29.s, z13.b, z0.b[1]
    ldnt1b { z16.b }, p2/Z, [x20, #1, MUL VL]
    udot z30.s, z14.b, z0.b[1]
    ldnt1b { z17.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z15.b, z0.b[2]
    ldnt1b { z19.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z16.b, z0.b[2]
    ldnt1b { z20.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z17.b, z0.b[2]
    ldnt1b { z21.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z18.b, z0.b[2]
    addvl x20, x20, #8
    udot z28.s, z19.b, z0.b[2]
    ldnt1b { z1.b }, p2/Z, [x20]
    udot z29.s, z20.b, z0.b[2]
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    udot z30.s, z21.b, z0.b[2]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[3]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[3]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[3]
    ldnt1b { z7.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z4.b, z0.b[3]
    addvl x20, x20, #8
    udot z28.s, z5.b, z0.b[3]
    udot z29.s, z6.b, z0.b[3]
    udot z30.s, z7.b, z0.b[3]
    tbnz x24, #31, label_52
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_52)  // Width 7: Multiply loop: unique 13: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_51
KAI_ASM_LABEL(label_53)  // Width 7: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z8.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z9.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z10.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z11.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z8.b, z0.b[0]
    ldnt1b { z12.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z9.b, z0.b[0]
    ldnt1b { z13.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z10.b, z0.b[0]
    ldnt1b { z14.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z11.b, z0.b[0]
    addvl x20, x20, #8
    udot z28.s, z12.b, z0.b[0]
    udot z29.s, z13.b, z0.b[0]
    udot z30.s, z14.b, z0.b[0]
    ble label_54
    ldnt1b { z15.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z16.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z17.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z15.b, z0.b[1]
    ldnt1b { z19.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z16.b, z0.b[1]
    ldnt1b { z20.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z17.b, z0.b[1]
    ldnt1b { z21.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z18.b, z0.b[1]
    addvl x20, x20, #8
    udot z28.s, z19.b, z0.b[1]
    udot z29.s, z20.b, z0.b[1]
    udot z30.s, z21.b, z0.b[1]
    ble label_54
    ldnt1b { z1.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[2]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[2]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[2]
    ldnt1b { z7.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z4.b, z0.b[2]
    addvl x20, x20, #8
    udot z28.s, z5.b, z0.b[2]
    udot z29.s, z6.b, z0.b[2]
    udot z30.s, z7.b, z0.b[2]
    ble label_54
    ldnt1b { z8.b }, p2/Z, [x20]
    ldnt1b { z9.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z10.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z11.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z8.b, z0.b[3]
    ldnt1b { z12.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z9.b, z0.b[3]
    ldnt1b { z13.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z10.b, z0.b[3]
    ldnt1b { z14.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z11.b, z0.b[3]
    udot z28.s, z12.b, z0.b[3]
    udot z29.s, z13.b, z0.b[3]
    udot z30.s, z14.b, z0.b[3]
KAI_ASM_LABEL(label_54)  // Width 7: Multiply loop: multiply skip
    tbnz x24, #31, label_55
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_55)  // Width 7: Multiply loop: unique 14: skip row sum
    tbnz x24, #31, label_56
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c847)  // ld1rw { z7.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z7.s, p2/M, z7.s
    mul z22.s, p2/M, z22.s, z7.s
KAI_ASM_LABEL(label_56)  // Width 7: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    add z27.s, z27.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add z28.s, z28.s, z22.s
    add z29.s, z29.s, z22.s
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x3, x0, #0x94
    add z30.s, z30.s, z22.s
    ld1w { z3.s }, p2/Z, [x6, #3, MUL VL]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    ld1w { z4.s }, p2/Z, [x6, #4, MUL VL]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    ld1w { z5.s }, p2/Z, [x6, #5, MUL VL]
    add z26.s, z26.s, z2.s
    ld1w { z6.s }, p2/Z, [x6, #6, MUL VL]
    add z27.s, z27.s, z3.s
    add z28.s, z28.s, z4.s
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z29.s, z29.s, z5.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z30.s, z30.s, z6.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x04a8737b)  // sqdmulh z27.s, z27.s, z8.s
    KAI_ASM_INST(0x04a8739c)  // sqdmulh z28.s, z28.s, z8.s
    KAI_ASM_INST(0x04a873bd)  // sqdmulh z29.s, z29.s, z8.s
    KAI_ASM_INST(0x04a873de)  // sqdmulh z30.s, z30.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    KAI_ASM_INST(0x4482881d)  // srshl z29.s, p2/M, z29.s, z0.s
    KAI_ASM_INST(0x4482881e)  // srshl z30.s, p2/M, z30.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    add z27.s, z27.s, z8.s
    add z28.s, z28.s, z8.s
    add z29.s, z29.s, z8.s
    add z30.s, z30.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13b)  // sclamp z27.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13c)  // sclamp z28.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13d)  // sclamp z29.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13e)  // sclamp z30.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z26.h, z28.h, z29.h
    uzp1 z27.h, z30.h, z31.h
    uzp1 z24.b, z24.b, z25.b
    uzp1 z25.b, z26.b, z27.b
    st1b { z24.b }, p2, [x7]
    st1b { z25.b }, p1, [x7, #1, MUL VL]
    b label_65
KAI_ASM_LABEL(label_57)  // Width 8
    lsl x1, x23, #0x2
    mov x3, x21
    ldr x2, [x0, #0x20]
    mov z24.s, #0
    sub x1, x22, x1
    mov z25.s, #0
    mov z26.s, #0
    whilelt p1.b, XZR, x1
    mov z27.s, #0
    mov z28.s, #0
    mov z29.s, #0
    mov z30.s, #0
    mov z31.s, #0
    cmp x3, #0x10
    ble label_61
KAI_ASM_LABEL(label_59)  // Width 8: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    ldnt1b { z1.b }, p2/Z, [x20]
    ld1rqb { z0.b }, p0/Z, [x2]
    add x2, x2, #0x10
    ldnt1b { z2.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z3.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z4.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z1.b, z0.b[0]
    ldnt1b { z5.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z2.b, z0.b[0]
    ldnt1b { z6.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z3.b, z0.b[0]
    ldnt1b { z7.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z4.b, z0.b[0]
    ldnt1b { z8.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z5.b, z0.b[0]
    addvl x20, x20, #8
    udot z29.s, z6.b, z0.b[0]
    ldnt1b { z9.b }, p2/Z, [x20]
    udot z30.s, z7.b, z0.b[0]
    ldnt1b { z10.b }, p2/Z, [x20, #1, MUL VL]
    udot z31.s, z8.b, z0.b[0]
    ldnt1b { z11.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z12.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z9.b, z0.b[1]
    ldnt1b { z13.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z10.b, z0.b[1]
    ldnt1b { z14.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z11.b, z0.b[1]
    ldnt1b { z15.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z12.b, z0.b[1]
    ldnt1b { z16.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z13.b, z0.b[1]
    addvl x20, x20, #8
    udot z29.s, z14.b, z0.b[1]
    ldnt1b { z17.b }, p2/Z, [x20]
    udot z30.s, z15.b, z0.b[1]
    ldnt1b { z18.b }, p2/Z, [x20, #1, MUL VL]
    udot z31.s, z16.b, z0.b[1]
    ldnt1b { z19.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z20.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z17.b, z0.b[2]
    ldnt1b { z21.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z18.b, z0.b[2]
    ldnt1b { z1.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z19.b, z0.b[2]
    ldnt1b { z2.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z20.b, z0.b[2]
    ldnt1b { z3.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z21.b, z0.b[2]
    addvl x20, x20, #8
    udot z29.s, z1.b, z0.b[2]
    ldnt1b { z4.b }, p2/Z, [x20]
    udot z30.s, z2.b, z0.b[2]
    ldnt1b { z5.b }, p2/Z, [x20, #1, MUL VL]
    udot z31.s, z3.b, z0.b[2]
    ldnt1b { z6.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z7.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z4.b, z0.b[3]
    ldnt1b { z8.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z5.b, z0.b[3]
    ldnt1b { z9.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z6.b, z0.b[3]
    ldnt1b { z10.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z7.b, z0.b[3]
    ldnt1b { z11.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z8.b, z0.b[3]
    addvl x20, x20, #8
    udot z29.s, z9.b, z0.b[3]
    udot z30.s, z10.b, z0.b[3]
    udot z31.s, z11.b, z0.b[3]
    tbnz x24, #31, label_60
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_60)  // Width 8: Multiply loop: unique 15: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_59
KAI_ASM_LABEL(label_61)  // Width 8: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    ldnt1b { z12.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ld1rqb { z0.b }, p0/Z, [x2]
    ldnt1b { z13.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z14.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z15.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z12.b, z0.b[0]
    ldnt1b { z16.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z13.b, z0.b[0]
    ldnt1b { z17.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z14.b, z0.b[0]
    ldnt1b { z18.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z15.b, z0.b[0]
    ldnt1b { z19.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z16.b, z0.b[0]
    addvl x20, x20, #8
    udot z29.s, z17.b, z0.b[0]
    udot z30.s, z18.b, z0.b[0]
    udot z31.s, z19.b, z0.b[0]
    ble label_62
    ldnt1b { z20.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z21.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z1.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z2.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z20.b, z0.b[1]
    ldnt1b { z3.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z21.b, z0.b[1]
    ldnt1b { z4.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z1.b, z0.b[1]
    ldnt1b { z5.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z2.b, z0.b[1]
    ldnt1b { z6.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z3.b, z0.b[1]
    addvl x20, x20, #8
    udot z29.s, z4.b, z0.b[1]
    udot z30.s, z5.b, z0.b[1]
    udot z31.s, z6.b, z0.b[1]
    ble label_62
    ldnt1b { z7.b }, p2/Z, [x20]
    subs x3, x3, #0x4
    ldnt1b { z8.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z9.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z10.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z7.b, z0.b[2]
    ldnt1b { z11.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z8.b, z0.b[2]
    ldnt1b { z12.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z9.b, z0.b[2]
    ldnt1b { z13.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z10.b, z0.b[2]
    ldnt1b { z14.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z11.b, z0.b[2]
    addvl x20, x20, #8
    udot z29.s, z12.b, z0.b[2]
    udot z30.s, z13.b, z0.b[2]
    udot z31.s, z14.b, z0.b[2]
    ble label_62
    ldnt1b { z15.b }, p2/Z, [x20]
    ldnt1b { z16.b }, p2/Z, [x20, #1, MUL VL]
    ldnt1b { z17.b }, p2/Z, [x20, #2, MUL VL]
    ldnt1b { z18.b }, p2/Z, [x20, #3, MUL VL]
    udot z24.s, z15.b, z0.b[3]
    ldnt1b { z19.b }, p2/Z, [x20, #4, MUL VL]
    udot z25.s, z16.b, z0.b[3]
    ldnt1b { z20.b }, p2/Z, [x20, #5, MUL VL]
    udot z26.s, z17.b, z0.b[3]
    ldnt1b { z21.b }, p2/Z, [x20, #6, MUL VL]
    udot z27.s, z18.b, z0.b[3]
    ldnt1b { z1.b }, p2/Z, [x20, #7, MUL VL]
    udot z28.s, z19.b, z0.b[3]
    addvl x20, x20, #8
    udot z29.s, z20.b, z0.b[3]
    udot z30.s, z21.b, z0.b[3]
    udot z31.s, z1.b, z0.b[3]
KAI_ASM_LABEL(label_62)  // Width 8: Multiply loop: multiply skip
    tbnz x24, #31, label_63
    udot z22.s, z0.b, z23.b
KAI_ASM_LABEL(label_63)  // Width 8: Multiply loop: unique 16: skip row sum
    tbnz x24, #31, label_64
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c848)  // ld1rw { z8.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d22, p0, z22.s
    mov z22.s, z22.s[0]
    neg z8.s, p2/M, z8.s
    mul z22.s, p2/M, z22.s, z8.s
KAI_ASM_LABEL(label_64)  // Width 8: skip row sum fixup
    add z24.s, z24.s, z22.s
    add z25.s, z25.s, z22.s
    ld1w { z0.s }, p2/Z, [x6]
    add x5, x0, #0x98
    add z26.s, z26.s, z22.s
    add z27.s, z27.s, z22.s
    ld1w { z1.s }, p2/Z, [x6, #1, MUL VL]
    add x4, x0, #0x9c
    add z28.s, z28.s, z22.s
    add z29.s, z29.s, z22.s
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    add x3, x0, #0x94
    add z30.s, z30.s, z22.s
    add z31.s, z31.s, z22.s
    ld1w { z3.s }, p2/Z, [x6, #3, MUL VL]
    add x2, x0, #0xa0
    ld1w { z4.s }, p2/Z, [x6, #4, MUL VL]
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    add x1, x0, #0xa4
    ld1w { z5.s }, p2/Z, [x6, #5, MUL VL]
    add z26.s, z26.s, z2.s
    subs x8, x8, #0x8
    orr x24, x24, #0x80000000
    ld1w { z6.s }, p2/Z, [x6, #6, MUL VL]
    add z27.s, z27.s, z3.s
    sub x22, x22, x23, LSL #3
    ld1w { z7.s }, p2/Z, [x6, #7, MUL VL]
    add z28.s, z28.s, z4.s
    addvl x6, x6, #8
    add z29.s, z29.s, z5.s
    KAI_ASM_INST(0x8540c8a8)  // ld1rw { z8.s }, p2/Z, [x5]
    add z30.s, z30.s, z6.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z31.s, z31.s, z7.s
    KAI_ASM_INST(0x8540c849)  // ld1rw { z9.s }, p2/Z, [x2]
    KAI_ASM_INST(0x04a87318)  // sqdmulh z24.s, z24.s, z8.s
    KAI_ASM_INST(0x04a87339)  // sqdmulh z25.s, z25.s, z8.s
    KAI_ASM_INST(0x8540c82a)  // ld1rw { z10.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a8735a)  // sqdmulh z26.s, z26.s, z8.s
    KAI_ASM_INST(0x04a8737b)  // sqdmulh z27.s, z27.s, z8.s
    KAI_ASM_INST(0x04a8739c)  // sqdmulh z28.s, z28.s, z8.s
    KAI_ASM_INST(0x04a873bd)  // sqdmulh z29.s, z29.s, z8.s
    KAI_ASM_INST(0x04a873de)  // sqdmulh z30.s, z30.s, z8.s
    KAI_ASM_INST(0x04a873ff)  // sqdmulh z31.s, z31.s, z8.s
    KAI_ASM_INST(0x8540c868)  // ld1rw { z8.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    KAI_ASM_INST(0x4482881d)  // srshl z29.s, p2/M, z29.s, z0.s
    KAI_ASM_INST(0x4482881e)  // srshl z30.s, p2/M, z30.s, z0.s
    KAI_ASM_INST(0x4482881f)  // srshl z31.s, p2/M, z31.s, z0.s
    add z24.s, z24.s, z8.s
    add z25.s, z25.s, z8.s
    add z26.s, z26.s, z8.s
    add z27.s, z27.s, z8.s
    add z28.s, z28.s, z8.s
    add z29.s, z29.s, z8.s
    add z30.s, z30.s, z8.s
    add z31.s, z31.s, z8.s
    KAI_ASM_INST(0x448ac138)  // sclamp z24.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac139)  // sclamp z25.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13a)  // sclamp z26.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13b)  // sclamp z27.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13c)  // sclamp z28.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13d)  // sclamp z29.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13e)  // sclamp z30.s, z9.s, z10.s
    KAI_ASM_INST(0x448ac13f)  // sclamp z31.s, z9.s, z10.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z25.h, z26.h, z27.h
    uzp1 z26.h, z28.h, z29.h
    uzp1 z27.h, z30.h, z31.h
    uzp1 z24.b, z24.b, z25.b
    uzp1 z25.b, z26.b, z27.b
    st1b { z24.b }, p2, [x7]
    st1b { z25.b }, p1, [x7, #1, MUL VL]
    addvl x7, x7, #2
    bgt label_1
KAI_ASM_LABEL(label_65)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_gemv_u8qa_dot_8VL)

    KAI_ASM_END
