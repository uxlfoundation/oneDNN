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

    KAI_ASM_CODE(sve_hybrid_s8qs_dot_6x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_s8qs_dot_6x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_s8qs_dot_6x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_s8qs_dot_6x4VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ptrue p2.b
    mov x9, #0
    ldr x3, [x0, #0x60]
    ldr x2, [x0, #0x18]
    ldr x1, [x0, #0x78]
    ldr x28, [x0, #0]
    cmp x3, #0
    cset x27, NE
    cmp x2, #0
    cset x2, NE
    cmp x1, #0
    cset x1, NE
    orr x27, x27, x2, LSL #3
    orr x27, x27, x1, LSL #4
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x28, x9
    cmp x1, #0x6
    bge label_66
    cmp x1, #0x4
    bgt label_53
    beq label_40
    cmp x1, #0x2
    bgt label_27
    beq label_14
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    whilelt p1.b, x1, x22
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_4
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_5)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    cbnz x26, label_6
    add x8, x8, x3
KAI_ASM_LABEL(label_6)  // Height 1: input setup done
    cmp x7, #0x10
    ble label_8
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    sdot z8.s, z6.b, z0.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z11.s, z7.b, z0.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z11.s, z7.b, z0.b[3]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    sdot z8.s, z6.b, z0.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z11.s, z7.b, z0.b[0]
    ble label_9
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[1]
    sdot z11.s, z7.b, z0.b[1]
    ble label_9
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[2]
    sdot z11.s, z7.b, z0.b[2]
    ble label_9
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z11.s, z7.b, z0.b[3]
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: multiply skip
    tbz x27, #3, label_10
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_5
KAI_ASM_LABEL(label_10)  // Height 1: no string loop
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add z11.s, z11.s, z3.s
    tbz x27, #4, label_11
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_12
KAI_ASM_LABEL(label_11)  // Height 1: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_12)  // Height 1: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x2, x0, #0x94
    add x1, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    KAI_ASM_INST(0x8540c844)  // ld1rw { z4.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c826)  // ld1rw { z6.s }, p2/Z, [x1]
    add x1, x0, #0xa0
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z9.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z9.b
    st1b { z8.b }, p1, [x25]
    addvl x25, x25, #1
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_2
    b label_79
KAI_ASM_LABEL(label_14)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_15)  // Height 2: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    whilelt p1.b, x1, x22
    mov z14.s, #0
    mov z15.s, #0
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_17
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    b label_19
KAI_ASM_LABEL(label_17)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_18)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x26, label_19
    add x8, x8, x3
    add x6, x6, x3
KAI_ASM_LABEL(label_19)  // Height 2: input setup done
    cmp x7, #0x10
    ble label_21
KAI_ASM_LABEL(label_20)  // Height 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z1.b }, p0/Z, [x6]
    add x6, x6, #0x10
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    bgt label_20
KAI_ASM_LABEL(label_21)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    ld1rqb { z1.b }, p0/Z, [x6]
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    ble label_22
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    ble label_22
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    ble label_22
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: multiply skip
    tbz x27, #3, label_23
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_18
KAI_ASM_LABEL(label_23)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add x8, x25, x1
    add z8.s, z8.s, z0.s
    add z12.s, z12.s, z0.s
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add z13.s, z13.s, z1.s
    add z11.s, z11.s, z3.s
    add z14.s, z14.s, z2.s
    add z15.s, z15.s, z3.s
    tbz x27, #4, label_24
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_25
KAI_ASM_LABEL(label_24)  // Height 2: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_25)  // Height 2: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    add x1, x0, #0xa0
    KAI_ASM_INST(0x04a4718c)  // sqdmulh z12.s, z12.s, z4.s
    KAI_ASM_INST(0x04a571ad)  // sqdmulh z13.s, z13.s, z5.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x04a671ce)  // sqdmulh z14.s, z14.s, z6.s
    KAI_ASM_INST(0x04a771ef)  // sqdmulh z15.s, z15.s, z7.s
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x4482880c)  // srshl z12.s, p2/M, z12.s, z0.s
    KAI_ASM_INST(0x4482882d)  // srshl z13.s, p2/M, z13.s, z1.s
    KAI_ASM_INST(0x4482884e)  // srshl z14.s, p2/M, z14.s, z2.s
    KAI_ASM_INST(0x4482886f)  // srshl z15.s, p2/M, z15.s, z3.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    add z12.s, z12.s, z4.s
    add z13.s, z13.s, z4.s
    add z14.s, z14.s, z4.s
    add z15.s, z15.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    smin z12.s, p2/M, z12.s, z6.s
    smin z13.s, p2/M, z13.s, z6.s
    smin z14.s, p2/M, z14.s, z6.s
    smin z15.s, p2/M, z15.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    smax z12.s, p2/M, z12.s, z5.s
    smax z13.s, p2/M, z13.s, z5.s
    smax z14.s, p2/M, z14.s, z5.s
    smax z15.s, p2/M, z15.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z9.h, z10.h, z11.h
    uzp1 z12.h, z12.h, z13.h
    uzp1 z13.h, z14.h, z15.h
    uzp1 z8.b, z8.b, z9.b
    uzp1 z12.b, z12.b, z13.b
    st1b { z8.b }, p1, [x25]
    addvl x25, x25, #1
    st1b { z12.b }, p1, [x8]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_15
    b label_79
KAI_ASM_LABEL(label_27)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_28)  // Height 3: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    whilelt p1.b, x1, x22
    mov z14.s, #0
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_30
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_32
KAI_ASM_LABEL(label_30)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_31)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x26, label_32
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
KAI_ASM_LABEL(label_32)  // Height 3: input setup done
    cmp x7, #0x10
    ble label_34
KAI_ASM_LABEL(label_33)  // Height 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z1.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z2.b }, p0/Z, [x5]
    add x5, x5, #0x10
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    bgt label_33
KAI_ASM_LABEL(label_34)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x5]
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    ble label_35
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    ble label_35
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    ble label_35
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: multiply skip
    tbz x27, #3, label_36
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_31
KAI_ASM_LABEL(label_36)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add x8, x25, x1
    add x7, x8, x1
    add z8.s, z8.s, z0.s
    add z12.s, z12.s, z0.s
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add z11.s, z11.s, z3.s
    add z13.s, z13.s, z1.s
    add z14.s, z14.s, z2.s
    add z15.s, z15.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    tbz x27, #4, label_37
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_38
KAI_ASM_LABEL(label_37)  // Height 3: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_38)  // Height 3: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    add x1, x0, #0xa0
    KAI_ASM_INST(0x04a4718c)  // sqdmulh z12.s, z12.s, z4.s
    KAI_ASM_INST(0x04a571ad)  // sqdmulh z13.s, z13.s, z5.s
    KAI_ASM_INST(0x04a671ce)  // sqdmulh z14.s, z14.s, z6.s
    KAI_ASM_INST(0x04a771ef)  // sqdmulh z15.s, z15.s, z7.s
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a57231)  // sqdmulh z17.s, z17.s, z5.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x04a67252)  // sqdmulh z18.s, z18.s, z6.s
    KAI_ASM_INST(0x04a77273)  // sqdmulh z19.s, z19.s, z7.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    KAI_ASM_INST(0x4482880c)  // srshl z12.s, p2/M, z12.s, z0.s
    KAI_ASM_INST(0x4482882d)  // srshl z13.s, p2/M, z13.s, z1.s
    KAI_ASM_INST(0x4482884e)  // srshl z14.s, p2/M, z14.s, z2.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x4482886f)  // srshl z15.s, p2/M, z15.s, z3.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    KAI_ASM_INST(0x44828831)  // srshl z17.s, p2/M, z17.s, z1.s
    KAI_ASM_INST(0x44828852)  // srshl z18.s, p2/M, z18.s, z2.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    KAI_ASM_INST(0x44828873)  // srshl z19.s, p2/M, z19.s, z3.s
    add z12.s, z12.s, z4.s
    add z13.s, z13.s, z4.s
    add z14.s, z14.s, z4.s
    add z15.s, z15.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z12.s, p2/M, z12.s, z6.s
    smin z13.s, p2/M, z13.s, z6.s
    smin z14.s, p2/M, z14.s, z6.s
    smin z15.s, p2/M, z15.s, z6.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    smax z12.s, p2/M, z12.s, z5.s
    smax z13.s, p2/M, z13.s, z5.s
    smax z14.s, p2/M, z14.s, z5.s
    smax z15.s, p2/M, z15.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    uzp1 z9.h, z10.h, z11.h
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z12.h, z12.h, z13.h
    uzp1 z13.h, z14.h, z15.h
    uzp1 z16.h, z16.h, z17.h
    uzp1 z8.b, z8.b, z9.b
    uzp1 z17.h, z18.h, z19.h
    uzp1 z12.b, z12.b, z13.b
    st1b { z8.b }, p1, [x25]
    addvl x25, x25, #1
    uzp1 z16.b, z16.b, z17.b
    st1b { z12.b }, p1, [x8]
    st1b { z16.b }, p1, [x7]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_28
    b label_79
KAI_ASM_LABEL(label_40)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_41)  // Height 4: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    whilelt p1.b, x1, x22
    mov z14.s, #0
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    mov z22.s, #0
    mov z23.s, #0
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_43
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_45
KAI_ASM_LABEL(label_43)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_44)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x26, label_45
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_45)  // Height 4: input setup done
    cmp x7, #0x10
    ble label_47
KAI_ASM_LABEL(label_46)  // Height 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z1.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z2.b }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqb { z3.b }, p0/Z, [x4]
    add x4, x4, #0x10
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    sdot z21.s, z7.b, z3.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
    bgt label_46
KAI_ASM_LABEL(label_47)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x5]
    ld1rqb { z3.b }, p0/Z, [x4]
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    sdot z21.s, z7.b, z3.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    ble label_48
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    ble label_48
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    ble label_48
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
KAI_ASM_LABEL(label_48)  // Height 4: Multiply loop: multiply skip
    tbz x27, #3, label_49
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_44
KAI_ASM_LABEL(label_49)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add x8, x25, x1
    add x7, x8, x1
    add z8.s, z8.s, z0.s
    add z12.s, z12.s, z0.s
    add x6, x7, x1
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add z11.s, z11.s, z3.s
    add z13.s, z13.s, z1.s
    add z14.s, z14.s, z2.s
    add z15.s, z15.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z1.s
    add z22.s, z22.s, z2.s
    add z23.s, z23.s, z3.s
    tbz x27, #4, label_50
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_51
KAI_ASM_LABEL(label_50)  // Height 4: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_51)  // Height 4: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    add x1, x0, #0xa0
    KAI_ASM_INST(0x04a4718c)  // sqdmulh z12.s, z12.s, z4.s
    KAI_ASM_INST(0x04a571ad)  // sqdmulh z13.s, z13.s, z5.s
    KAI_ASM_INST(0x04a671ce)  // sqdmulh z14.s, z14.s, z6.s
    KAI_ASM_INST(0x04a771ef)  // sqdmulh z15.s, z15.s, z7.s
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a57231)  // sqdmulh z17.s, z17.s, z5.s
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    KAI_ASM_INST(0x04a67252)  // sqdmulh z18.s, z18.s, z6.s
    KAI_ASM_INST(0x04a77273)  // sqdmulh z19.s, z19.s, z7.s
    KAI_ASM_INST(0x4482880c)  // srshl z12.s, p2/M, z12.s, z0.s
    KAI_ASM_INST(0x4482882d)  // srshl z13.s, p2/M, z13.s, z1.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a572b5)  // sqdmulh z21.s, z21.s, z5.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x4482884e)  // srshl z14.s, p2/M, z14.s, z2.s
    KAI_ASM_INST(0x04a672d6)  // sqdmulh z22.s, z22.s, z6.s
    KAI_ASM_INST(0x04a772f7)  // sqdmulh z23.s, z23.s, z7.s
    KAI_ASM_INST(0x4482886f)  // srshl z15.s, p2/M, z15.s, z3.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828831)  // srshl z17.s, p2/M, z17.s, z1.s
    KAI_ASM_INST(0x44828852)  // srshl z18.s, p2/M, z18.s, z2.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x44828873)  // srshl z19.s, p2/M, z19.s, z3.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    KAI_ASM_INST(0x44828835)  // srshl z21.s, p2/M, z21.s, z1.s
    KAI_ASM_INST(0x44828856)  // srshl z22.s, p2/M, z22.s, z2.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    KAI_ASM_INST(0x44828877)  // srshl z23.s, p2/M, z23.s, z3.s
    add z12.s, z12.s, z4.s
    add z13.s, z13.s, z4.s
    add z14.s, z14.s, z4.s
    add z15.s, z15.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z12.s, p2/M, z12.s, z6.s
    smin z13.s, p2/M, z13.s, z6.s
    add z20.s, z20.s, z4.s
    add z21.s, z21.s, z4.s
    smin z14.s, p2/M, z14.s, z6.s
    smin z15.s, p2/M, z15.s, z6.s
    add z22.s, z22.s, z4.s
    add z23.s, z23.s, z4.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    smin z20.s, p2/M, z20.s, z6.s
    smin z21.s, p2/M, z21.s, z6.s
    smin z22.s, p2/M, z22.s, z6.s
    smin z23.s, p2/M, z23.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    smax z12.s, p2/M, z12.s, z5.s
    smax z13.s, p2/M, z13.s, z5.s
    smax z14.s, p2/M, z14.s, z5.s
    smax z15.s, p2/M, z15.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    uzp1 z9.h, z10.h, z11.h
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z12.h, z12.h, z13.h
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    uzp1 z13.h, z14.h, z15.h
    smax z22.s, p2/M, z22.s, z5.s
    smax z23.s, p2/M, z23.s, z5.s
    uzp1 z16.h, z16.h, z17.h
    uzp1 z8.b, z8.b, z9.b
    uzp1 z17.h, z18.h, z19.h
    uzp1 z20.h, z20.h, z21.h
    uzp1 z12.b, z12.b, z13.b
    uzp1 z21.h, z22.h, z23.h
    st1b { z8.b }, p1, [x25]
    addvl x25, x25, #1
    uzp1 z16.b, z16.b, z17.b
    uzp1 z20.b, z20.b, z21.b
    st1b { z12.b }, p1, [x8]
    st1b { z16.b }, p1, [x7]
    st1b { z20.b }, p1, [x6]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_41
    b label_79
KAI_ASM_LABEL(label_53)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_54)  // Height 5: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    whilelt p1.b, x1, x22
    mov z14.s, #0
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    mov z22.s, #0
    mov z23.s, #0
    mov z24.s, #0
    mov z25.s, #0
    mov z26.s, #0
    mov z27.s, #0
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_56
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_58
KAI_ASM_LABEL(label_56)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_57)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    cbnz x26, label_58
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_58)  // Height 5: input setup done
    cmp x7, #0x10
    ble label_60
KAI_ASM_LABEL(label_59)  // Height 5: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z1.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z2.b }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqb { z3.b }, p0/Z, [x4]
    ld1rqb { z4.b }, p0/Z, [x2]
    add x4, x4, #0x10
    add x2, x2, #0x10
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    sdot z24.s, z6.b, z4.b[0]
    sdot z13.s, z7.b, z1.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    sdot z21.s, z7.b, z3.b[0]
    sdot z25.s, z7.b, z4.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    sdot z26.s, z6.b, z4.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    sdot z27.s, z7.b, z4.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    sdot z24.s, z6.b, z4.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    sdot z25.s, z7.b, z4.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z26.s, z6.b, z4.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    sdot z27.s, z7.b, z4.b[1]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    sdot z24.s, z6.b, z4.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    sdot z25.s, z7.b, z4.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    sdot z26.s, z6.b, z4.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    sdot z27.s, z7.b, z4.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    sdot z24.s, z6.b, z4.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    sdot z25.s, z7.b, z4.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z26.s, z6.b, z4.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
    sdot z27.s, z7.b, z4.b[3]
    bgt label_59
KAI_ASM_LABEL(label_60)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x5]
    ld1rqb { z3.b }, p0/Z, [x4]
    ld1rqb { z4.b }, p0/Z, [x2]
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    sdot z24.s, z6.b, z4.b[0]
    sdot z17.s, z7.b, z2.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z21.s, z7.b, z3.b[0]
    sdot z25.s, z7.b, z4.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    sdot z26.s, z6.b, z4.b[0]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    sdot z27.s, z7.b, z4.b[0]
    ble label_61
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    sdot z24.s, z6.b, z4.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    sdot z25.s, z7.b, z4.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[1]
    addvl x21, x21, #4
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z26.s, z6.b, z4.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    sdot z27.s, z7.b, z4.b[1]
    ble label_61
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    sdot z24.s, z6.b, z4.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    sdot z25.s, z7.b, z4.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    addvl x21, x21, #4
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    sdot z26.s, z6.b, z4.b[2]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    sdot z27.s, z7.b, z4.b[2]
    ble label_61
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    sdot z24.s, z6.b, z4.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    sdot z25.s, z7.b, z4.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z26.s, z6.b, z4.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
    sdot z27.s, z7.b, z4.b[3]
KAI_ASM_LABEL(label_61)  // Height 5: Multiply loop: multiply skip
    tbz x27, #3, label_62
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_57
KAI_ASM_LABEL(label_62)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add x8, x25, x1
    add x7, x8, x1
    add z8.s, z8.s, z0.s
    add z12.s, z12.s, z0.s
    add x6, x7, x1
    add x5, x6, x1
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add z11.s, z11.s, z3.s
    add z13.s, z13.s, z1.s
    add z14.s, z14.s, z2.s
    add z15.s, z15.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z1.s
    add z22.s, z22.s, z2.s
    add z23.s, z23.s, z3.s
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    add z26.s, z26.s, z2.s
    add z27.s, z27.s, z3.s
    tbz x27, #4, label_63
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_64
KAI_ASM_LABEL(label_63)  // Height 5: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_64)  // Height 5: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    add x1, x0, #0xa0
    KAI_ASM_INST(0x04a4718c)  // sqdmulh z12.s, z12.s, z4.s
    KAI_ASM_INST(0x04a571ad)  // sqdmulh z13.s, z13.s, z5.s
    KAI_ASM_INST(0x04a671ce)  // sqdmulh z14.s, z14.s, z6.s
    KAI_ASM_INST(0x04a771ef)  // sqdmulh z15.s, z15.s, z7.s
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a57231)  // sqdmulh z17.s, z17.s, z5.s
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    KAI_ASM_INST(0x04a67252)  // sqdmulh z18.s, z18.s, z6.s
    KAI_ASM_INST(0x04a77273)  // sqdmulh z19.s, z19.s, z7.s
    KAI_ASM_INST(0x4482880c)  // srshl z12.s, p2/M, z12.s, z0.s
    KAI_ASM_INST(0x4482882d)  // srshl z13.s, p2/M, z13.s, z1.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a572b5)  // sqdmulh z21.s, z21.s, z5.s
    KAI_ASM_INST(0x4482884e)  // srshl z14.s, p2/M, z14.s, z2.s
    KAI_ASM_INST(0x4482886f)  // srshl z15.s, p2/M, z15.s, z3.s
    KAI_ASM_INST(0x04a672d6)  // sqdmulh z22.s, z22.s, z6.s
    KAI_ASM_INST(0x04a772f7)  // sqdmulh z23.s, z23.s, z7.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828831)  // srshl z17.s, p2/M, z17.s, z1.s
    KAI_ASM_INST(0x04a47318)  // sqdmulh z24.s, z24.s, z4.s
    KAI_ASM_INST(0x04a57339)  // sqdmulh z25.s, z25.s, z5.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828852)  // srshl z18.s, p2/M, z18.s, z2.s
    KAI_ASM_INST(0x04a6735a)  // sqdmulh z26.s, z26.s, z6.s
    KAI_ASM_INST(0x04a7737b)  // sqdmulh z27.s, z27.s, z7.s
    KAI_ASM_INST(0x44828873)  // srshl z19.s, p2/M, z19.s, z3.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    KAI_ASM_INST(0x44828835)  // srshl z21.s, p2/M, z21.s, z1.s
    KAI_ASM_INST(0x44828856)  // srshl z22.s, p2/M, z22.s, z2.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x44828877)  // srshl z23.s, p2/M, z23.s, z3.s
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    KAI_ASM_INST(0x44828839)  // srshl z25.s, p2/M, z25.s, z1.s
    KAI_ASM_INST(0x4482885a)  // srshl z26.s, p2/M, z26.s, z2.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    KAI_ASM_INST(0x4482887b)  // srshl z27.s, p2/M, z27.s, z3.s
    add z12.s, z12.s, z4.s
    add z13.s, z13.s, z4.s
    add z14.s, z14.s, z4.s
    add z15.s, z15.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z12.s, p2/M, z12.s, z6.s
    smin z13.s, p2/M, z13.s, z6.s
    add z20.s, z20.s, z4.s
    add z21.s, z21.s, z4.s
    smin z14.s, p2/M, z14.s, z6.s
    smin z15.s, p2/M, z15.s, z6.s
    add z22.s, z22.s, z4.s
    add z23.s, z23.s, z4.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    add z24.s, z24.s, z4.s
    add z25.s, z25.s, z4.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    add z26.s, z26.s, z4.s
    add z27.s, z27.s, z4.s
    smin z20.s, p2/M, z20.s, z6.s
    smin z21.s, p2/M, z21.s, z6.s
    smin z22.s, p2/M, z22.s, z6.s
    smin z23.s, p2/M, z23.s, z6.s
    smin z24.s, p2/M, z24.s, z6.s
    smin z25.s, p2/M, z25.s, z6.s
    smin z26.s, p2/M, z26.s, z6.s
    smin z27.s, p2/M, z27.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    smax z12.s, p2/M, z12.s, z5.s
    smax z13.s, p2/M, z13.s, z5.s
    smax z14.s, p2/M, z14.s, z5.s
    smax z15.s, p2/M, z15.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    uzp1 z9.h, z10.h, z11.h
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z12.h, z12.h, z13.h
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    uzp1 z13.h, z14.h, z15.h
    smax z22.s, p2/M, z22.s, z5.s
    smax z23.s, p2/M, z23.s, z5.s
    uzp1 z16.h, z16.h, z17.h
    uzp1 z8.b, z8.b, z9.b
    smax z24.s, p2/M, z24.s, z5.s
    smax z25.s, p2/M, z25.s, z5.s
    uzp1 z17.h, z18.h, z19.h
    smax z26.s, p2/M, z26.s, z5.s
    smax z27.s, p2/M, z27.s, z5.s
    uzp1 z20.h, z20.h, z21.h
    uzp1 z12.b, z12.b, z13.b
    uzp1 z21.h, z22.h, z23.h
    st1b { z8.b }, p1, [x25]
    addvl x25, x25, #1
    uzp1 z24.h, z24.h, z25.h
    uzp1 z16.b, z16.b, z17.b
    uzp1 z25.h, z26.h, z27.h
    st1b { z12.b }, p1, [x8]
    uzp1 z20.b, z20.b, z21.b
    uzp1 z24.b, z24.b, z25.b
    st1b { z16.b }, p1, [x7]
    st1b { z20.b }, p1, [x6]
    st1b { z24.b }, p1, [x5]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_54
    b label_79
KAI_ASM_LABEL(label_66)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_67)  // Height 6: Column loop
    mov x1, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    whilelt p1.b, x1, x22
    mov z14.s, #0
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    mov z22.s, #0
    mov z23.s, #0
    mov z24.s, #0
    mov z25.s, #0
    mov z26.s, #0
    mov z27.s, #0
    mov z28.s, #0
    mov z29.s, #0
    mov z30.s, #0
    mov z31.s, #0
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_69
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_71
KAI_ASM_LABEL(label_69)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_70)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    ldr x1, [x1, #0x28]
    cbnz x26, label_71
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_71)  // Height 6: input setup done
    cmp x7, #0x10
    ble label_73
KAI_ASM_LABEL(label_72)  // Height 6: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z0.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z1.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z2.b }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqb { z3.b }, p0/Z, [x4]
    ld1rqb { z4.b }, p0/Z, [x2]
    ld1rqb { z5.b }, p0/Z, [x1]
    add x4, x4, #0x10
    add x2, x2, #0x10
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    add x1, x1, #0x10
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    sdot z24.s, z6.b, z4.b[0]
    sdot z28.s, z6.b, z5.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z17.s, z7.b, z2.b[0]
    sdot z21.s, z7.b, z3.b[0]
    sdot z25.s, z7.b, z4.b[0]
    sdot z29.s, z7.b, z5.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    sdot z26.s, z6.b, z4.b[0]
    sdot z30.s, z6.b, z5.b[0]
    ld1b { z6.b }, p2/Z, [x21, #4, MUL VL]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    sdot z27.s, z7.b, z4.b[0]
    sdot z31.s, z7.b, z5.b[0]
    ld1b { z7.b }, p2/Z, [x21, #5, MUL VL]
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    sdot z24.s, z6.b, z4.b[1]
    sdot z28.s, z6.b, z5.b[1]
    ld1b { z6.b }, p2/Z, [x21, #6, MUL VL]
    sdot z9.s, z7.b, z0.b[1]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    sdot z25.s, z7.b, z4.b[1]
    sdot z29.s, z7.b, z5.b[1]
    ld1b { z7.b }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    sdot z10.s, z6.b, z0.b[1]
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z26.s, z6.b, z4.b[1]
    sdot z30.s, z6.b, z5.b[1]
    ld1b { z6.b }, p2/Z, [x21, #-8, MUL VL]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    sdot z27.s, z7.b, z4.b[1]
    sdot z31.s, z7.b, z5.b[1]
    ld1b { z7.b }, p2/Z, [x21, #-7, MUL VL]
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    sdot z24.s, z6.b, z4.b[2]
    sdot z28.s, z6.b, z5.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-6, MUL VL]
    sdot z9.s, z7.b, z0.b[2]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    sdot z25.s, z7.b, z4.b[2]
    sdot z29.s, z7.b, z5.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-5, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    sdot z26.s, z6.b, z4.b[2]
    sdot z30.s, z6.b, z5.b[2]
    ld1b { z6.b }, p2/Z, [x21, #-4, MUL VL]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    sdot z27.s, z7.b, z4.b[2]
    sdot z31.s, z7.b, z5.b[2]
    ld1b { z7.b }, p2/Z, [x21, #-3, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    sdot z24.s, z6.b, z4.b[3]
    sdot z28.s, z6.b, z5.b[3]
    ld1b { z6.b }, p2/Z, [x21, #-2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    sdot z25.s, z7.b, z4.b[3]
    sdot z29.s, z7.b, z5.b[3]
    ld1b { z7.b }, p2/Z, [x21, #-1, MUL VL]
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z26.s, z6.b, z4.b[3]
    sdot z30.s, z6.b, z5.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
    sdot z27.s, z7.b, z4.b[3]
    sdot z31.s, z7.b, z5.b[3]
    bgt label_72
KAI_ASM_LABEL(label_73)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    ld1rqb { z0.b }, p0/Z, [x8]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x5]
    ld1rqb { z3.b }, p0/Z, [x4]
    ld1rqb { z4.b }, p0/Z, [x2]
    ld1rqb { z5.b }, p0/Z, [x1]
    sdot z8.s, z6.b, z0.b[0]
    sdot z12.s, z6.b, z1.b[0]
    sdot z9.s, z7.b, z0.b[0]
    sdot z13.s, z7.b, z1.b[0]
    sdot z16.s, z6.b, z2.b[0]
    sdot z20.s, z6.b, z3.b[0]
    sdot z24.s, z6.b, z4.b[0]
    sdot z28.s, z6.b, z5.b[0]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z17.s, z7.b, z2.b[0]
    sdot z21.s, z7.b, z3.b[0]
    sdot z25.s, z7.b, z4.b[0]
    sdot z29.s, z7.b, z5.b[0]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[0]
    sdot z14.s, z6.b, z1.b[0]
    sdot z18.s, z6.b, z2.b[0]
    sdot z22.s, z6.b, z3.b[0]
    sdot z26.s, z6.b, z4.b[0]
    sdot z30.s, z6.b, z5.b[0]
    sdot z11.s, z7.b, z0.b[0]
    sdot z15.s, z7.b, z1.b[0]
    sdot z19.s, z7.b, z2.b[0]
    sdot z23.s, z7.b, z3.b[0]
    sdot z27.s, z7.b, z4.b[0]
    sdot z31.s, z7.b, z5.b[0]
    ble label_74
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[1]
    sdot z12.s, z6.b, z1.b[1]
    sdot z16.s, z6.b, z2.b[1]
    sdot z20.s, z6.b, z3.b[1]
    sdot z24.s, z6.b, z4.b[1]
    sdot z28.s, z6.b, z5.b[1]
    sdot z9.s, z7.b, z0.b[1]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z13.s, z7.b, z1.b[1]
    sdot z17.s, z7.b, z2.b[1]
    sdot z21.s, z7.b, z3.b[1]
    sdot z25.s, z7.b, z4.b[1]
    sdot z29.s, z7.b, z5.b[1]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[1]
    addvl x21, x21, #4
    sdot z14.s, z6.b, z1.b[1]
    sdot z18.s, z6.b, z2.b[1]
    sdot z22.s, z6.b, z3.b[1]
    sdot z26.s, z6.b, z4.b[1]
    sdot z30.s, z6.b, z5.b[1]
    sdot z11.s, z7.b, z0.b[1]
    sdot z15.s, z7.b, z1.b[1]
    sdot z19.s, z7.b, z2.b[1]
    sdot z23.s, z7.b, z3.b[1]
    sdot z27.s, z7.b, z4.b[1]
    sdot z31.s, z7.b, z5.b[1]
    ble label_74
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x4
    sdot z8.s, z6.b, z0.b[2]
    sdot z12.s, z6.b, z1.b[2]
    sdot z16.s, z6.b, z2.b[2]
    sdot z20.s, z6.b, z3.b[2]
    sdot z24.s, z6.b, z4.b[2]
    sdot z28.s, z6.b, z5.b[2]
    sdot z9.s, z7.b, z0.b[2]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z13.s, z7.b, z1.b[2]
    sdot z17.s, z7.b, z2.b[2]
    sdot z21.s, z7.b, z3.b[2]
    sdot z25.s, z7.b, z4.b[2]
    sdot z29.s, z7.b, z5.b[2]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    sdot z10.s, z6.b, z0.b[2]
    addvl x21, x21, #4
    sdot z14.s, z6.b, z1.b[2]
    sdot z18.s, z6.b, z2.b[2]
    sdot z22.s, z6.b, z3.b[2]
    sdot z26.s, z6.b, z4.b[2]
    sdot z30.s, z6.b, z5.b[2]
    sdot z11.s, z7.b, z0.b[2]
    sdot z15.s, z7.b, z1.b[2]
    sdot z19.s, z7.b, z2.b[2]
    sdot z23.s, z7.b, z3.b[2]
    sdot z27.s, z7.b, z4.b[2]
    sdot z31.s, z7.b, z5.b[2]
    ble label_74
    ld1b { z6.b }, p2/Z, [x21]
    ld1b { z7.b }, p2/Z, [x21, #1, MUL VL]
    sdot z8.s, z6.b, z0.b[3]
    sdot z12.s, z6.b, z1.b[3]
    sdot z16.s, z6.b, z2.b[3]
    sdot z20.s, z6.b, z3.b[3]
    sdot z24.s, z6.b, z4.b[3]
    sdot z28.s, z6.b, z5.b[3]
    ld1b { z6.b }, p2/Z, [x21, #2, MUL VL]
    sdot z9.s, z7.b, z0.b[3]
    sdot z13.s, z7.b, z1.b[3]
    sdot z17.s, z7.b, z2.b[3]
    sdot z21.s, z7.b, z3.b[3]
    sdot z25.s, z7.b, z4.b[3]
    sdot z29.s, z7.b, z5.b[3]
    ld1b { z7.b }, p2/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    sdot z10.s, z6.b, z0.b[3]
    sdot z14.s, z6.b, z1.b[3]
    sdot z18.s, z6.b, z2.b[3]
    sdot z22.s, z6.b, z3.b[3]
    sdot z26.s, z6.b, z4.b[3]
    sdot z30.s, z6.b, z5.b[3]
    sdot z11.s, z7.b, z0.b[3]
    sdot z15.s, z7.b, z1.b[3]
    sdot z19.s, z7.b, z2.b[3]
    sdot z23.s, z7.b, z3.b[3]
    sdot z27.s, z7.b, z4.b[3]
    sdot z31.s, z7.b, z5.b[3]
KAI_ASM_LABEL(label_74)  // Height 6: Multiply loop: multiply skip
    tbz x27, #3, label_75
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_70
KAI_ASM_LABEL(label_75)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    add x8, x25, x1
    add x7, x8, x1
    add z8.s, z8.s, z0.s
    add z12.s, z12.s, z0.s
    add x6, x7, x1
    add x5, x6, x1
    add z9.s, z9.s, z1.s
    add z10.s, z10.s, z2.s
    add x4, x5, x1
    add z11.s, z11.s, z3.s
    add z13.s, z13.s, z1.s
    add z14.s, z14.s, z2.s
    add z15.s, z15.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z1.s
    add z22.s, z22.s, z2.s
    add z23.s, z23.s, z3.s
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    add z26.s, z26.s, z2.s
    add z27.s, z27.s, z3.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z1.s
    add z30.s, z30.s, z2.s
    add z31.s, z31.s, z3.s
    tbz x27, #4, label_76
    ld1w { z0.s }, p2/Z, [x23]
    ld1w { z4.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x23, #1, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z2.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z7.s }, p2/Z, [x24, #3, MUL VL]
    addvl x23, x23, #4
    addvl x24, x24, #4
    b label_77
KAI_ASM_LABEL(label_76)  // Height 6: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    KAI_ASM_INST(0x8540c840)  // ld1rw { z0.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    mov z1.d, z0.d
    mov z5.d, z4.d
    mov z2.d, z0.d
    mov z6.d, z4.d
    mov z3.d, z0.d
    mov z7.d, z4.d
KAI_ASM_LABEL(label_77)  // Height 6: parameters loaded
    KAI_ASM_INST(0x04a47108)  // sqdmulh z8.s, z8.s, z4.s
    KAI_ASM_INST(0x04a57129)  // sqdmulh z9.s, z9.s, z5.s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    KAI_ASM_INST(0x04a6714a)  // sqdmulh z10.s, z10.s, z6.s
    KAI_ASM_INST(0x04a7716b)  // sqdmulh z11.s, z11.s, z7.s
    add x1, x0, #0xa0
    KAI_ASM_INST(0x04a4718c)  // sqdmulh z12.s, z12.s, z4.s
    KAI_ASM_INST(0x04a571ad)  // sqdmulh z13.s, z13.s, z5.s
    KAI_ASM_INST(0x04a671ce)  // sqdmulh z14.s, z14.s, z6.s
    KAI_ASM_INST(0x04a771ef)  // sqdmulh z15.s, z15.s, z7.s
    KAI_ASM_INST(0x44828808)  // srshl z8.s, p2/M, z8.s, z0.s
    KAI_ASM_INST(0x44828829)  // srshl z9.s, p2/M, z9.s, z1.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a57231)  // sqdmulh z17.s, z17.s, z5.s
    KAI_ASM_INST(0x4482884a)  // srshl z10.s, p2/M, z10.s, z2.s
    KAI_ASM_INST(0x4482886b)  // srshl z11.s, p2/M, z11.s, z3.s
    KAI_ASM_INST(0x04a67252)  // sqdmulh z18.s, z18.s, z6.s
    KAI_ASM_INST(0x04a77273)  // sqdmulh z19.s, z19.s, z7.s
    KAI_ASM_INST(0x4482880c)  // srshl z12.s, p2/M, z12.s, z0.s
    KAI_ASM_INST(0x4482882d)  // srshl z13.s, p2/M, z13.s, z1.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a572b5)  // sqdmulh z21.s, z21.s, z5.s
    KAI_ASM_INST(0x4482884e)  // srshl z14.s, p2/M, z14.s, z2.s
    KAI_ASM_INST(0x4482886f)  // srshl z15.s, p2/M, z15.s, z3.s
    KAI_ASM_INST(0x04a672d6)  // sqdmulh z22.s, z22.s, z6.s
    KAI_ASM_INST(0x04a772f7)  // sqdmulh z23.s, z23.s, z7.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828831)  // srshl z17.s, p2/M, z17.s, z1.s
    KAI_ASM_INST(0x04a47318)  // sqdmulh z24.s, z24.s, z4.s
    KAI_ASM_INST(0x04a57339)  // sqdmulh z25.s, z25.s, z5.s
    KAI_ASM_INST(0x44828852)  // srshl z18.s, p2/M, z18.s, z2.s
    KAI_ASM_INST(0x44828873)  // srshl z19.s, p2/M, z19.s, z3.s
    KAI_ASM_INST(0x04a6735a)  // sqdmulh z26.s, z26.s, z6.s
    KAI_ASM_INST(0x04a7737b)  // sqdmulh z27.s, z27.s, z7.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    KAI_ASM_INST(0x44828835)  // srshl z21.s, p2/M, z21.s, z1.s
    KAI_ASM_INST(0x04a4739c)  // sqdmulh z28.s, z28.s, z4.s
    KAI_ASM_INST(0x04a573bd)  // sqdmulh z29.s, z29.s, z5.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828856)  // srshl z22.s, p2/M, z22.s, z2.s
    KAI_ASM_INST(0x04a673de)  // sqdmulh z30.s, z30.s, z6.s
    KAI_ASM_INST(0x04a773ff)  // sqdmulh z31.s, z31.s, z7.s
    KAI_ASM_INST(0x44828877)  // srshl z23.s, p2/M, z23.s, z3.s
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828839)  // srshl z25.s, p2/M, z25.s, z1.s
    KAI_ASM_INST(0x4482885a)  // srshl z26.s, p2/M, z26.s, z2.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    KAI_ASM_INST(0x4482887b)  // srshl z27.s, p2/M, z27.s, z3.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    add z8.s, z8.s, z4.s
    add z9.s, z9.s, z4.s
    KAI_ASM_INST(0x4482883d)  // srshl z29.s, p2/M, z29.s, z1.s
    KAI_ASM_INST(0x4482885e)  // srshl z30.s, p2/M, z30.s, z2.s
    add z10.s, z10.s, z4.s
    add z11.s, z11.s, z4.s
    KAI_ASM_INST(0x4482887f)  // srshl z31.s, p2/M, z31.s, z3.s
    add z12.s, z12.s, z4.s
    add z13.s, z13.s, z4.s
    add z14.s, z14.s, z4.s
    add z15.s, z15.s, z4.s
    smin z8.s, p2/M, z8.s, z6.s
    smin z9.s, p2/M, z9.s, z6.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    smin z10.s, p2/M, z10.s, z6.s
    smin z11.s, p2/M, z11.s, z6.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z12.s, p2/M, z12.s, z6.s
    smin z13.s, p2/M, z13.s, z6.s
    add z20.s, z20.s, z4.s
    add z21.s, z21.s, z4.s
    smin z14.s, p2/M, z14.s, z6.s
    smin z15.s, p2/M, z15.s, z6.s
    add z22.s, z22.s, z4.s
    add z23.s, z23.s, z4.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    add z24.s, z24.s, z4.s
    add z25.s, z25.s, z4.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    add z26.s, z26.s, z4.s
    add z27.s, z27.s, z4.s
    smin z20.s, p2/M, z20.s, z6.s
    smin z21.s, p2/M, z21.s, z6.s
    add z28.s, z28.s, z4.s
    add z29.s, z29.s, z4.s
    smin z22.s, p2/M, z22.s, z6.s
    smin z23.s, p2/M, z23.s, z6.s
    add z30.s, z30.s, z4.s
    add z31.s, z31.s, z4.s
    smin z24.s, p2/M, z24.s, z6.s
    smin z25.s, p2/M, z25.s, z6.s
    smin z26.s, p2/M, z26.s, z6.s
    smin z27.s, p2/M, z27.s, z6.s
    smin z28.s, p2/M, z28.s, z6.s
    smin z29.s, p2/M, z29.s, z6.s
    smin z30.s, p2/M, z30.s, z6.s
    smin z31.s, p2/M, z31.s, z6.s
    smax z8.s, p2/M, z8.s, z5.s
    smax z9.s, p2/M, z9.s, z5.s
    smax z10.s, p2/M, z10.s, z5.s
    smax z11.s, p2/M, z11.s, z5.s
    smax z12.s, p2/M, z12.s, z5.s
    smax z13.s, p2/M, z13.s, z5.s
    smax z14.s, p2/M, z14.s, z5.s
    smax z15.s, p2/M, z15.s, z5.s
    uzp1 z8.h, z8.h, z9.h
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    uzp1 z9.h, z10.h, z11.h
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z12.h, z12.h, z13.h
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    uzp1 z13.h, z14.h, z15.h
    smax z22.s, p2/M, z22.s, z5.s
    smax z23.s, p2/M, z23.s, z5.s
    uzp1 z16.h, z16.h, z17.h
    uzp1 z8.b, z8.b, z9.b
    smax z24.s, p2/M, z24.s, z5.s
    smax z25.s, p2/M, z25.s, z5.s
    uzp1 z17.h, z18.h, z19.h
    smax z26.s, p2/M, z26.s, z5.s
    smax z27.s, p2/M, z27.s, z5.s
    uzp1 z20.h, z20.h, z21.h
    uzp1 z12.b, z12.b, z13.b
    smax z28.s, p2/M, z28.s, z5.s
    smax z29.s, p2/M, z29.s, z5.s
    uzp1 z21.h, z22.h, z23.h
    st1b { z8.b }, p1, [x25]
    smax z30.s, p2/M, z30.s, z5.s
    smax z31.s, p2/M, z31.s, z5.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z16.b, z16.b, z17.b
    uzp1 z25.h, z26.h, z27.h
    st1b { z12.b }, p1, [x8]
    addvl x25, x25, #1
    uzp1 z28.h, z28.h, z29.h
    uzp1 z20.b, z20.b, z21.b
    uzp1 z29.h, z30.h, z31.h
    st1b { z16.b }, p1, [x7]
    uzp1 z24.b, z24.b, z25.b
    uzp1 z28.b, z28.b, z29.b
    st1b { z20.b }, p1, [x6]
    st1b { z24.b }, p1, [x5]
    st1b { z28.b }, p1, [x4]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_67
    add x9, x9, #0x6
    cmp x9, x28
    blt label_1
KAI_ASM_LABEL(label_79)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_s8qs_dot_6x4VL)

    KAI_ASM_END
