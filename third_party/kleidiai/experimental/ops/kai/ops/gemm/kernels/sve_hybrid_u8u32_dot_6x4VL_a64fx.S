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

    KAI_ASM_CODE(sve_hybrid_u8u32_dot_6x4VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_u8u32_dot_6x4VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_u8u32_dot_6x4VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_u8u32_dot_6x4VL_a64fx)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    ptrue p5.b
    mov x27, #0
    ldr x2, [x0, #0x60]
    ldr x1, [x0, #0x18]
    ldr x26, [x0, #0]
    cmp x2, #0
    cset x25, NE
    cmp x1, #0
    cset x1, NE
    orr x25, x25, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x26, x27
    cmp x1, #0x6
    bge label_61
    cmp x1, #0x4
    bgt label_49
    beq label_37
    cmp x1, #0x2
    bgt label_25
    beq label_13
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_3
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x25, #0, label_4
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
KAI_ASM_LABEL(label_5)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_6
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    b label_8
KAI_ASM_LABEL(label_6)  // Height 1: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_7)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    cbnz x24, label_8
    add x8, x8, x3
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x7, #0x4
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    ld1b { z6.b }, p5/Z, [x21]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    cmp x7, #0x4
    udot z8.s, z6.b, z0.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z11.s, z7.b, z0.b
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    mov z0.s, z0.s[0]
    udot z8.s, z6.b, z0.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z11.s, z7.b, z0.b
    tbz x25, #3, label_11
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_7
KAI_ASM_LABEL(label_11)  // Height 1: no string loop
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_2
    b label_73
KAI_ASM_LABEL(label_13)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_14)  // Height 2: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_15
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_17
KAI_ASM_LABEL(label_15)  // Height 2: no bias
    tbz x25, #0, label_16
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    add x1, x23, x1
    ld1w { z12.s }, p4/Z, [x1]
    ld1w { z13.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x1, #3, MUL VL]
    b label_17
KAI_ASM_LABEL(label_16)  // Height 2: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
KAI_ASM_LABEL(label_17)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_18
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_20
KAI_ASM_LABEL(label_18)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_19)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_20
    add x8, x8, x3
    add x6, x6, x3
KAI_ASM_LABEL(label_20)  // Height 2: input setup done
    cmp x7, #0x4
    ble label_22
KAI_ASM_LABEL(label_21)  // Height 2: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    KAI_ASM_INST(0x8540d4c1)  // ld1rw { z1.s }, p5/Z, [x6]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    cmp x7, #0x4
    add x6, x6, #0x4
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    bgt label_21
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    ld1b { z1.b }, p0/Z, [x6]
    mov z0.s, z0.s[0]
    udot z8.s, z6.b, z0.b
    udot z9.s, z7.b, z0.b
    mov z1.s, z1.s[0]
    udot z12.s, z6.b, z1.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z13.s, z7.b, z1.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    tbz x25, #3, label_23
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_19
KAI_ASM_LABEL(label_23)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    add x1, x23, x1
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x1]
    st1w { z13.s }, p3, [x1, #1, MUL VL]
    st1w { z14.s }, p2, [x1, #2, MUL VL]
    st1w { z15.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_14
    b label_73
KAI_ASM_LABEL(label_25)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_26)  // Height 3: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_27
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    b label_29
KAI_ASM_LABEL(label_27)  // Height 3: no bias
    tbz x25, #0, label_28
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    add x2, x23, x1
    add x1, x2, x1
    ld1w { z12.s }, p4/Z, [x2]
    ld1w { z13.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x2, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x1]
    ld1w { z17.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x1, #3, MUL VL]
    b label_29
KAI_ASM_LABEL(label_28)  // Height 3: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
KAI_ASM_LABEL(label_29)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_30
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_32
KAI_ASM_LABEL(label_30)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_31)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_32
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
KAI_ASM_LABEL(label_32)  // Height 3: input setup done
    cmp x7, #0x4
    ble label_34
KAI_ASM_LABEL(label_33)  // Height 3: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    KAI_ASM_INST(0x8540d4c1)  // ld1rw { z1.s }, p5/Z, [x6]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    KAI_ASM_INST(0x8540d4a2)  // ld1rw { z2.s }, p5/Z, [x5]
    ld1b { z6.b }, p5/Z, [x21]
    cmp x7, #0x4
    add x6, x6, #0x4
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    add x5, x5, #0x4
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z16.s, z6.b, z2.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    udot z17.s, z7.b, z2.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    bgt label_33
KAI_ASM_LABEL(label_34)  // Height 3: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    ld1b { z1.b }, p0/Z, [x6]
    ld1b { z2.b }, p0/Z, [x5]
    mov z0.s, z0.s[0]
    mov z1.s, z1.s[0]
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    mov z2.s, z2.s[0]
    udot z16.s, z6.b, z2.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z17.s, z7.b, z2.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    tbz x25, #3, label_35
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_31
KAI_ASM_LABEL(label_35)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    add x2, x23, x1
    add x1, x2, x1
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x2]
    st1w { z13.s }, p3, [x2, #1, MUL VL]
    st1w { z14.s }, p2, [x2, #2, MUL VL]
    st1w { z15.s }, p1, [x2, #3, MUL VL]
    st1w { z16.s }, p4, [x1]
    st1w { z17.s }, p3, [x1, #1, MUL VL]
    st1w { z18.s }, p2, [x1, #2, MUL VL]
    st1w { z19.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_26
    b label_73
KAI_ASM_LABEL(label_37)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_38)  // Height 4: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_39
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    b label_41
KAI_ASM_LABEL(label_39)  // Height 4: no bias
    tbz x25, #0, label_40
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    add x3, x23, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z12.s }, p4/Z, [x3]
    ld1w { z13.s }, p3/Z, [x3, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x3, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x2]
    ld1w { z17.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x2, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x1]
    ld1w { z21.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #3, MUL VL]
    b label_41
KAI_ASM_LABEL(label_40)  // Height 4: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
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
KAI_ASM_LABEL(label_41)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_42
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_44
KAI_ASM_LABEL(label_42)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_43)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_44
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_44)  // Height 4: input setup done
    cmp x7, #0x4
    ble label_46
KAI_ASM_LABEL(label_45)  // Height 4: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    KAI_ASM_INST(0x8540d4c1)  // ld1rw { z1.s }, p5/Z, [x6]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    KAI_ASM_INST(0x8540d4a2)  // ld1rw { z2.s }, p5/Z, [x5]
    KAI_ASM_INST(0x8540d483)  // ld1rw { z3.s }, p5/Z, [x4]
    cmp x7, #0x4
    add x6, x6, #0x4
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    add x5, x5, #0x4
    add x4, x4, #0x4
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z16.s, z6.b, z2.b
    udot z20.s, z6.b, z3.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    udot z17.s, z7.b, z2.b
    udot z21.s, z7.b, z3.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    bgt label_45
KAI_ASM_LABEL(label_46)  // Height 4: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    ld1b { z1.b }, p0/Z, [x6]
    ld1b { z2.b }, p0/Z, [x5]
    ld1b { z3.b }, p0/Z, [x4]
    mov z0.s, z0.s[0]
    mov z1.s, z1.s[0]
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    mov z2.s, z2.s[0]
    mov z3.s, z3.s[0]
    udot z16.s, z6.b, z2.b
    udot z17.s, z7.b, z2.b
    udot z20.s, z6.b, z3.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z21.s, z7.b, z3.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    tbz x25, #3, label_47
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_43
KAI_ASM_LABEL(label_47)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    add x3, x23, x1
    add x2, x3, x1
    add x1, x2, x1
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x3]
    st1w { z13.s }, p3, [x3, #1, MUL VL]
    st1w { z14.s }, p2, [x3, #2, MUL VL]
    st1w { z15.s }, p1, [x3, #3, MUL VL]
    st1w { z16.s }, p4, [x2]
    st1w { z17.s }, p3, [x2, #1, MUL VL]
    st1w { z18.s }, p2, [x2, #2, MUL VL]
    st1w { z19.s }, p1, [x2, #3, MUL VL]
    st1w { z20.s }, p4, [x1]
    st1w { z21.s }, p3, [x1, #1, MUL VL]
    st1w { z22.s }, p2, [x1, #2, MUL VL]
    st1w { z23.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_38
    b label_73
KAI_ASM_LABEL(label_49)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_50)  // Height 5: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_51
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    b label_53
KAI_ASM_LABEL(label_51)  // Height 5: no bias
    tbz x25, #0, label_52
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    add x4, x23, x1
    add x3, x4, x1
    add x2, x3, x1
    ld1w { z12.s }, p4/Z, [x4]
    ld1w { z13.s }, p3/Z, [x4, #1, MUL VL]
    add x1, x2, x1
    ld1w { z14.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x3]
    ld1w { z17.s }, p3/Z, [x3, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x3, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x2]
    ld1w { z21.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x2, #3, MUL VL]
    ld1w { z24.s }, p4/Z, [x1]
    ld1w { z25.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z27.s }, p1/Z, [x1, #3, MUL VL]
    b label_53
KAI_ASM_LABEL(label_52)  // Height 5: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
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
KAI_ASM_LABEL(label_53)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_54
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_56
KAI_ASM_LABEL(label_54)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_55)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    cbnz x24, label_56
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_56)  // Height 5: input setup done
    cmp x7, #0x4
    ble label_58
KAI_ASM_LABEL(label_57)  // Height 5: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    KAI_ASM_INST(0x8540d4c1)  // ld1rw { z1.s }, p5/Z, [x6]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    KAI_ASM_INST(0x8540d4a2)  // ld1rw { z2.s }, p5/Z, [x5]
    KAI_ASM_INST(0x8540d483)  // ld1rw { z3.s }, p5/Z, [x4]
    cmp x7, #0x4
    add x6, x6, #0x4
    KAI_ASM_INST(0x8540d444)  // ld1rw { z4.s }, p5/Z, [x2]
    ld1b { z6.b }, p5/Z, [x21]
    add x5, x5, #0x4
    add x4, x4, #0x4
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    add x2, x2, #0x4
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z16.s, z6.b, z2.b
    udot z20.s, z6.b, z3.b
    udot z24.s, z6.b, z4.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    udot z17.s, z7.b, z2.b
    udot z21.s, z7.b, z3.b
    udot z25.s, z7.b, z4.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z26.s, z6.b, z4.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    udot z27.s, z7.b, z4.b
    bgt label_57
KAI_ASM_LABEL(label_58)  // Height 5: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    ld1b { z1.b }, p0/Z, [x6]
    ld1b { z2.b }, p0/Z, [x5]
    ld1b { z3.b }, p0/Z, [x4]
    ld1b { z4.b }, p0/Z, [x2]
    mov z0.s, z0.s[0]
    mov z1.s, z1.s[0]
    mov z2.s, z2.s[0]
    mov z3.s, z3.s[0]
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z9.s, z7.b, z0.b
    mov z4.s, z4.s[0]
    udot z16.s, z6.b, z2.b
    udot z20.s, z6.b, z3.b
    udot z24.s, z6.b, z4.b
    udot z13.s, z7.b, z1.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z17.s, z7.b, z2.b
    udot z21.s, z7.b, z3.b
    udot z25.s, z7.b, z4.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z26.s, z6.b, z4.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    udot z27.s, z7.b, z4.b
    tbz x25, #3, label_59
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_55
KAI_ASM_LABEL(label_59)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    add x4, x23, x1
    add x3, x4, x1
    add x2, x3, x1
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    add x1, x2, x1
    st1w { z12.s }, p4, [x4]
    st1w { z13.s }, p3, [x4, #1, MUL VL]
    st1w { z14.s }, p2, [x4, #2, MUL VL]
    st1w { z15.s }, p1, [x4, #3, MUL VL]
    st1w { z16.s }, p4, [x3]
    st1w { z17.s }, p3, [x3, #1, MUL VL]
    st1w { z18.s }, p2, [x3, #2, MUL VL]
    st1w { z19.s }, p1, [x3, #3, MUL VL]
    st1w { z20.s }, p4, [x2]
    st1w { z21.s }, p3, [x2, #1, MUL VL]
    st1w { z22.s }, p2, [x2, #2, MUL VL]
    st1w { z23.s }, p1, [x2, #3, MUL VL]
    st1w { z24.s }, p4, [x1]
    st1w { z25.s }, p3, [x1, #1, MUL VL]
    st1w { z26.s }, p2, [x1, #2, MUL VL]
    st1w { z27.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_50
    b label_73
KAI_ASM_LABEL(label_61)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_62)  // Height 6: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_63
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    b label_65
KAI_ASM_LABEL(label_63)  // Height 6: no bias
    tbz x25, #0, label_64
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p4/Z, [x23]
    ld1w { z9.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x23, #3, MUL VL]
    add x5, x23, x1
    add x4, x5, x1
    add x3, x4, x1
    ld1w { z12.s }, p4/Z, [x5]
    ld1w { z13.s }, p3/Z, [x5, #1, MUL VL]
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z14.s }, p2/Z, [x5, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x5, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x4]
    ld1w { z17.s }, p3/Z, [x4, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x4, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x3]
    ld1w { z21.s }, p3/Z, [x3, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x3, #3, MUL VL]
    ld1w { z24.s }, p4/Z, [x2]
    ld1w { z25.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z27.s }, p1/Z, [x2, #3, MUL VL]
    ld1w { z28.s }, p4/Z, [x1]
    ld1w { z29.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z30.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z31.s }, p1/Z, [x1, #3, MUL VL]
    b label_65
KAI_ASM_LABEL(label_64)  // Height 6: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
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
KAI_ASM_LABEL(label_65)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_66
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_68
KAI_ASM_LABEL(label_66)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_67)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    ldr x1, [x1, #0x28]
    cbnz x24, label_68
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_68)  // Height 6: input setup done
    cmp x7, #0x4
    ble label_70
KAI_ASM_LABEL(label_69)  // Height 6: Multiply loop: Main loop
    KAI_ASM_INST(0x8540d500)  // ld1rw { z0.s }, p5/Z, [x8]
    KAI_ASM_INST(0x8540d4c1)  // ld1rw { z1.s }, p5/Z, [x6]
    sub x7, x7, #0x4
    add x8, x8, #0x4
    KAI_ASM_INST(0x8540d4a2)  // ld1rw { z2.s }, p5/Z, [x5]
    KAI_ASM_INST(0x8540d483)  // ld1rw { z3.s }, p5/Z, [x4]
    cmp x7, #0x4
    add x6, x6, #0x4
    KAI_ASM_INST(0x8540d444)  // ld1rw { z4.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d425)  // ld1rw { z5.s }, p5/Z, [x1]
    add x5, x5, #0x4
    add x4, x4, #0x4
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    add x2, x2, #0x4
    add x1, x1, #0x4
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z16.s, z6.b, z2.b
    udot z20.s, z6.b, z3.b
    udot z24.s, z6.b, z4.b
    udot z28.s, z6.b, z5.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z9.s, z7.b, z0.b
    udot z13.s, z7.b, z1.b
    udot z17.s, z7.b, z2.b
    udot z21.s, z7.b, z3.b
    udot z25.s, z7.b, z4.b
    udot z29.s, z7.b, z5.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z26.s, z6.b, z4.b
    udot z30.s, z6.b, z5.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    udot z27.s, z7.b, z4.b
    udot z31.s, z7.b, z5.b
    bgt label_69
KAI_ASM_LABEL(label_70)  // Height 6: Multiply loop: Main loop skip
    whilelt p0.b, XZR, x7
    ld1b { z6.b }, p5/Z, [x21]
    ld1b { z7.b }, p5/Z, [x21, #1, MUL VL]
    ld1b { z0.b }, p0/Z, [x8]
    ld1b { z1.b }, p0/Z, [x6]
    ld1b { z2.b }, p0/Z, [x5]
    ld1b { z3.b }, p0/Z, [x4]
    ld1b { z4.b }, p0/Z, [x2]
    ld1b { z5.b }, p0/Z, [x1]
    mov z0.s, z0.s[0]
    mov z1.s, z1.s[0]
    mov z2.s, z2.s[0]
    mov z3.s, z3.s[0]
    udot z8.s, z6.b, z0.b
    udot z12.s, z6.b, z1.b
    udot z9.s, z7.b, z0.b
    mov z4.s, z4.s[0]
    mov z5.s, z5.s[0]
    udot z16.s, z6.b, z2.b
    udot z20.s, z6.b, z3.b
    udot z24.s, z6.b, z4.b
    udot z13.s, z7.b, z1.b
    udot z28.s, z6.b, z5.b
    udot z17.s, z7.b, z2.b
    ld1b { z6.b }, p5/Z, [x21, #2, MUL VL]
    udot z21.s, z7.b, z3.b
    udot z25.s, z7.b, z4.b
    udot z29.s, z7.b, z5.b
    ld1b { z7.b }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    udot z10.s, z6.b, z0.b
    udot z14.s, z6.b, z1.b
    udot z18.s, z6.b, z2.b
    udot z22.s, z6.b, z3.b
    udot z26.s, z6.b, z4.b
    udot z30.s, z6.b, z5.b
    udot z11.s, z7.b, z0.b
    udot z15.s, z7.b, z1.b
    udot z19.s, z7.b, z2.b
    udot z23.s, z7.b, z3.b
    udot z27.s, z7.b, z4.b
    udot z31.s, z7.b, z5.b
    tbz x25, #3, label_71
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_67
KAI_ASM_LABEL(label_71)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    add x5, x23, x1
    add x4, x5, x1
    add x3, x4, x1
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    add x2, x3, x1
    st1w { z12.s }, p4, [x5]
    add x1, x2, x1
    st1w { z13.s }, p3, [x5, #1, MUL VL]
    st1w { z14.s }, p2, [x5, #2, MUL VL]
    st1w { z15.s }, p1, [x5, #3, MUL VL]
    st1w { z16.s }, p4, [x4]
    st1w { z17.s }, p3, [x4, #1, MUL VL]
    st1w { z18.s }, p2, [x4, #2, MUL VL]
    st1w { z19.s }, p1, [x4, #3, MUL VL]
    st1w { z20.s }, p4, [x3]
    st1w { z21.s }, p3, [x3, #1, MUL VL]
    st1w { z22.s }, p2, [x3, #2, MUL VL]
    st1w { z23.s }, p1, [x3, #3, MUL VL]
    st1w { z24.s }, p4, [x2]
    st1w { z25.s }, p3, [x2, #1, MUL VL]
    st1w { z26.s }, p2, [x2, #2, MUL VL]
    st1w { z27.s }, p1, [x2, #3, MUL VL]
    st1w { z28.s }, p4, [x1]
    st1w { z29.s }, p3, [x1, #1, MUL VL]
    st1w { z30.s }, p2, [x1, #2, MUL VL]
    st1w { z31.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_62
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_73)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_u8u32_dot_6x4VL_a64fx)

    KAI_ASM_END
