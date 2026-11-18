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

    KAI_ASM_CODE(sve_hybrid_u8s8qa_mmla_4x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_u8s8qa_mmla_4x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_u8s8qa_mmla_4x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_u8s8qa_mmla_4x4VL)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    stp d8, d9, [sp, 56]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    ptrue p2.b
    mov x26, #0
    ldr x3, [x0, #0x60]
    ldr x2, [x0, #0x18]
    ldr x1, [x0, #0x78]
    ldr x25, [x0, #0]
    cmp x3, #0
    cset x24, NE
    cmp x2, #0
    cset x2, NE
    cmp x1, #0
    cset x1, NE
    orr x24, x24, x2, LSL #3
    orr x24, x24, x1, LSL #4
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x25, x26
    cmp x1, #0x4
    bge label_43
    cmp x1, #0x2
    bgt label_29
    beq label_15
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    mov z11.s, #0
    mov z15.b, #0x1
    bic x24, x24, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    whilelt p1.b, x1, x21
    mov z22.s, #0
    mov z23.s, #0
    ldr x3, [x0, #0x28]
    tbnz x24, #3, label_4
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_5)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    cbnz x23, label_6
    add x6, x6, x3
KAI_ASM_LABEL(label_6)  // Height 1: input setup done
    cmp x5, #0x10
    ble label_9
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    add x6, x6, #0x10
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #16
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    ld1b { z6.b }, p2/Z, [x20, #-8, MUL VL]
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    ld1b { z7.b }, p2/Z, [x20, #-7, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #-6, MUL VL]
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ld1b { z9.b }, p2/Z, [x20, #-5, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #-4, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z4.b }, p2/Z, [x20, #-3, MUL VL]
    ld1b { z5.b }, p2/Z, [x20, #-2, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #-1, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    tbnz x24, #31, label_8
    udot z11.s, z0.b, z15.b
    udot z11.s, z1.b, z15.b
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: unique 1: skip row sum
    sub x5, x5, #0x10
    cmp x5, #0x10
    bgt label_7
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    subs x5, x5, #0x8
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #8
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ble label_10
    ld1b { z6.b }, p2/Z, [x20]
    ld1b { z7.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z4.b }, p2/Z, [x20, #5, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z5.b }, p2/Z, [x20, #6, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: multiply skip
    tbnz x24, #31, label_11
    udot z11.s, z0.b, z15.b
    udot z11.s, z1.b, z15.b
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: unique 2: skip row sum
    tbz x24, #3, label_12
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_5
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    uzp1 z16.d, z16.d, z20.d
    uzp1 z17.d, z17.d, z21.d
    uzp1 z18.d, z18.d, z22.d
    uzp1 z19.d, z19.d, z23.d
    mov z23.d, z16.d
    tbnz x24, #31, label_13
    add x1, x0, #0x90
    KAI_ASM_INST(0x4491a96b)  // addp z11.s, p2/m, z11.s, z11.s
    KAI_ASM_INST(0x8540c821)  // ld1rw { z1.s }, p2/Z, [x1]
    neg z1.s, p2/M, z1.s
    mov z11.s, z11.s[0]
    mul z11.s, p2/M, z11.s, z1.s
KAI_ASM_LABEL(label_13)  // Height 1: skip row sum fixup
    add z23.s, z23.s, z11.s
    add z17.s, z17.s, z11.s
    ld1w { z0.s }, p2/Z, [x8]
    ld1w { z1.s }, p2/Z, [x8, #1, MUL VL]
    add z18.s, z18.s, z11.s
    add z19.s, z19.s, z11.s
    ld1w { z2.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x8, #3, MUL VL]
    add x2, x0, #0x98
    add x1, x0, #0x9c
    add z23.s, z23.s, z0.s
    add z17.s, z17.s, z1.s
    KAI_ASM_INST(0x8540c844)  // ld1rw { z4.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c820)  // ld1rw { z0.s }, p2/Z, [x1]
    add x2, x0, #0x94
    add x1, x0, #0xa4
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    KAI_ASM_INST(0x8540c826)  // ld1rw { z6.s }, p2/Z, [x1]
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    orr x24, x24, #0x80000000
    addvl x8, x8, #4
    KAI_ASM_INST(0x04a472f7)  // sqdmulh z23.s, z23.s, z4.s
    KAI_ASM_INST(0x04a47231)  // sqdmulh z17.s, z17.s, z4.s
    KAI_ASM_INST(0x04a47252)  // sqdmulh z18.s, z18.s, z4.s
    KAI_ASM_INST(0x04a47273)  // sqdmulh z19.s, z19.s, z4.s
    KAI_ASM_INST(0x8540c844)  // ld1rw { z4.s }, p2/Z, [x2]
    KAI_ASM_INST(0x44828817)  // srshl z23.s, p2/M, z23.s, z0.s
    KAI_ASM_INST(0x44828811)  // srshl z17.s, p2/M, z17.s, z0.s
    KAI_ASM_INST(0x44828812)  // srshl z18.s, p2/M, z18.s, z0.s
    KAI_ASM_INST(0x44828813)  // srshl z19.s, p2/M, z19.s, z0.s
    add z23.s, z23.s, z4.s
    add z17.s, z17.s, z4.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z23.s, p2/M, z23.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    smax z23.s, p2/M, z23.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z23.h, z23.h, z17.h
    uzp1 z17.h, z18.h, z19.h
    uzp1 z23.b, z23.b, z17.b
    st1b { z23.b }, p1, [x22]
    addvl x22, x22, #1
    decw x21, ALL, MUL #4
    cmp x21, XZR
    bgt label_2
    b label_57
KAI_ASM_LABEL(label_15)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    mov z11.s, #0
    mov z12.s, #0
    mov z15.b, #0x1
    bic x24, x24, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_16)  // Height 2: Column loop
    mov x1, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    whilelt p1.b, x1, x21
    mov z22.s, #0
    mov z23.s, #0
    ldr x3, [x0, #0x28]
    tbnz x24, #3, label_18
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    b label_20
KAI_ASM_LABEL(label_18)  // Height 2: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_19)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    cbnz x23, label_20
    add x6, x6, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_20)  // Height 2: input setup done
    cmp x5, #0x10
    ble label_23
KAI_ASM_LABEL(label_21)  // Height 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    add x6, x6, #0x10
    add x4, x4, #0x10
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #16
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    ld1b { z6.b }, p2/Z, [x20, #-8, MUL VL]
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ld1b { z7.b }, p2/Z, [x20, #-7, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #-6, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #-5, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #-4, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z4.b }, p2/Z, [x20, #-3, MUL VL]
    ld1b { z5.b }, p2/Z, [x20, #-2, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #-1, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    tbnz x24, #31, label_22
    udot z11.s, z0.b, z15.b
    udot z11.s, z1.b, z15.b
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: unique 3: skip row sum
    sub x5, x5, #0x10
    cmp x5, #0x10
    bgt label_21
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    subs x5, x5, #0x8
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #8
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ble label_24
    ld1b { z6.b }, p2/Z, [x20]
    ld1b { z7.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z4.b }, p2/Z, [x20, #5, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z5.b }, p2/Z, [x20, #6, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: multiply skip
    tbnz x24, #31, label_25
    udot z11.s, z0.b, z15.b
    udot z11.s, z1.b, z15.b
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: unique 4: skip row sum
    tbz x24, #3, label_26
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_19
KAI_ASM_LABEL(label_26)  // Height 2: no string loop
    uzp1 z7.d, z16.d, z20.d
    ldr x1, [x0, #0x58]
    uzp2 z16.d, z16.d, z20.d
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    add x7, x22, x1
    mov z23.d, z7.d
    tbnz x24, #31, label_27
    add x1, x0, #0x90
    KAI_ASM_INST(0x4491a96b)  // addp z11.s, p2/m, z11.s, z11.s
    KAI_ASM_INST(0x8540c822)  // ld1rw { z2.s }, p2/Z, [x1]
    neg z2.s, p2/M, z2.s
    mov z12.s, z11.s[3]
    mov z11.s, z11.s[0]
    mul z11.s, p2/M, z11.s, z2.s
    mul z12.s, p2/M, z12.s, z2.s
KAI_ASM_LABEL(label_27)  // Height 2: skip row sum fixup
    add z23.s, z23.s, z11.s
    add z20.s, z20.s, z11.s
    ld1w { z0.s }, p2/Z, [x8]
    ld1w { z1.s }, p2/Z, [x8, #1, MUL VL]
    add z21.s, z21.s, z11.s
    add z22.s, z22.s, z11.s
    ld1w { z2.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x8, #3, MUL VL]
    add z16.s, z16.s, z12.s
    add z17.s, z17.s, z12.s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    add z18.s, z18.s, z12.s
    add z19.s, z19.s, z12.s
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    add x3, x0, #0x94
    add z23.s, z23.s, z0.s
    add z20.s, z20.s, z1.s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    add z21.s, z21.s, z2.s
    add z22.s, z22.s, z3.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    orr x24, x24, #0x80000000
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    addvl x8, x8, #4
    KAI_ASM_INST(0x04a472f7)  // sqdmulh z23.s, z23.s, z4.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a472b5)  // sqdmulh z21.s, z21.s, z4.s
    KAI_ASM_INST(0x04a472d6)  // sqdmulh z22.s, z22.s, z4.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a47231)  // sqdmulh z17.s, z17.s, z4.s
    KAI_ASM_INST(0x04a47252)  // sqdmulh z18.s, z18.s, z4.s
    KAI_ASM_INST(0x04a47273)  // sqdmulh z19.s, z19.s, z4.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828817)  // srshl z23.s, p2/M, z23.s, z0.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    KAI_ASM_INST(0x44828815)  // srshl z21.s, p2/M, z21.s, z0.s
    KAI_ASM_INST(0x44828816)  // srshl z22.s, p2/M, z22.s, z0.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828811)  // srshl z17.s, p2/M, z17.s, z0.s
    KAI_ASM_INST(0x44828812)  // srshl z18.s, p2/M, z18.s, z0.s
    add z23.s, z23.s, z4.s
    KAI_ASM_INST(0x44828813)  // srshl z19.s, p2/M, z19.s, z0.s
    add z20.s, z20.s, z4.s
    add z21.s, z21.s, z4.s
    add z22.s, z22.s, z4.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    add z18.s, z18.s, z4.s
    smin z23.s, p2/M, z23.s, z6.s
    add z19.s, z19.s, z4.s
    smin z20.s, p2/M, z20.s, z6.s
    smin z21.s, p2/M, z21.s, z6.s
    smin z22.s, p2/M, z22.s, z6.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    smax z23.s, p2/M, z23.s, z5.s
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    smax z22.s, p2/M, z22.s, z5.s
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z23.h, z23.h, z20.h
    uzp1 z20.h, z21.h, z22.h
    uzp1 z16.h, z16.h, z17.h
    uzp1 z17.h, z18.h, z19.h
    uzp1 z23.b, z23.b, z20.b
    uzp1 z16.b, z16.b, z17.b
    st1b { z23.b }, p1, [x22]
    addvl x22, x22, #1
    st1b { z16.b }, p1, [x7]
    decw x21, ALL, MUL #4
    cmp x21, XZR
    bgt label_16
    b label_57
KAI_ASM_LABEL(label_29)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z15.b, #0x1
    bic x24, x24, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_30)  // Height 3: Column loop
    mov x1, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    whilelt p1.b, x1, x21
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
    tbnz x24, #3, label_32
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    b label_34
KAI_ASM_LABEL(label_32)  // Height 3: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_33)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    cbnz x23, label_34
    add x6, x6, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_34)  // Height 3: input setup done
    cmp x5, #0x10
    ble label_37
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    add x6, x6, #0x10
    add x4, x4, #0x10
    ld1rqb { z3.b }, p0/Z, [x2]
    add x2, x2, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #16
    KAI_ASM_INST(0x4586985c)  // usmmla z28.s, z2.b, z6.b
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4588985d)  // usmmla z29.s, z2.b, z8.b
    KAI_ASM_INST(0x4589985a)  // usmmla z26.s, z2.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    ld1b { z6.b }, p2/Z, [x20, #-8, MUL VL]
    KAI_ASM_INST(0x458a985e)  // usmmla z30.s, z2.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    ld1b { z7.b }, p2/Z, [x20, #-7, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #-6, MUL VL]
    KAI_ASM_INST(0x4584985b)  // usmmla z27.s, z2.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ld1b { z9.b }, p2/Z, [x20, #-5, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #-4, MUL VL]
    KAI_ASM_INST(0x4585985f)  // usmmla z31.s, z2.b, z5.b
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z4.b }, p2/Z, [x20, #-3, MUL VL]
    ld1b { z5.b }, p2/Z, [x20, #-2, MUL VL]
    KAI_ASM_INST(0x45869878)  // usmmla z24.s, z3.b, z6.b
    ld1b { z6.b }, p2/Z, [x20, #-1, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x4587987c)  // usmmla z28.s, z3.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45889879)  // usmmla z25.s, z3.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x4589987d)  // usmmla z29.s, z3.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x458a987a)  // usmmla z26.s, z3.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x4584987e)  // usmmla z30.s, z3.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x4585987b)  // usmmla z27.s, z3.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    KAI_ASM_INST(0x4586987f)  // usmmla z31.s, z3.b, z6.b
    tbnz x24, #31, label_36
    udot z11.s, z0.b, z15.b
    udot z13.s, z2.b, z15.b
    udot z11.s, z1.b, z15.b
    udot z13.s, z3.b, z15.b
KAI_ASM_LABEL(label_36)  // Height 3: Multiply loop: unique 5: skip row sum
    sub x5, x5, #0x10
    cmp x5, #0x10
    bgt label_35
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    subs x5, x5, #0x8
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    ld1rqb { z3.b }, p0/Z, [x2]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x4586985c)  // usmmla z28.s, z2.b, z6.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4588985d)  // usmmla z29.s, z2.b, z8.b
    KAI_ASM_INST(0x4589985a)  // usmmla z26.s, z2.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x458a985e)  // usmmla z30.s, z2.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    KAI_ASM_INST(0x4584985b)  // usmmla z27.s, z2.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    KAI_ASM_INST(0x4585985f)  // usmmla z31.s, z2.b, z5.b
    ble label_38
    ld1b { z6.b }, p2/Z, [x20]
    ld1b { z7.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z4.b }, p2/Z, [x20, #5, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    KAI_ASM_INST(0x45869878)  // usmmla z24.s, z3.b, z6.b
    ld1b { z5.b }, p2/Z, [x20, #6, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x4587987c)  // usmmla z28.s, z3.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45889879)  // usmmla z25.s, z3.b, z8.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x4589987d)  // usmmla z29.s, z3.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x458a987a)  // usmmla z26.s, z3.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x4584987e)  // usmmla z30.s, z3.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x4585987b)  // usmmla z27.s, z3.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    KAI_ASM_INST(0x4586987f)  // usmmla z31.s, z3.b, z6.b
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: multiply skip
    tbnz x24, #31, label_39
    udot z11.s, z0.b, z15.b
    udot z13.s, z2.b, z15.b
    udot z11.s, z1.b, z15.b
    udot z13.s, z3.b, z15.b
KAI_ASM_LABEL(label_39)  // Height 3: Multiply loop: unique 6: skip row sum
    tbz x24, #3, label_40
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_33
KAI_ASM_LABEL(label_40)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    add x7, x22, x1
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    add x6, x7, x1
    uzp1 z24.d, z24.d, z28.d
    uzp1 z25.d, z25.d, z29.d
    uzp1 z26.d, z26.d, z30.d
    uzp1 z27.d, z27.d, z31.d
    mov z31.d, z7.d
    tbnz x24, #31, label_41
    add x1, x0, #0x90
    KAI_ASM_INST(0x4491a96b)  // addp z11.s, p2/m, z11.s, z11.s
    KAI_ASM_INST(0x4491a9ad)  // addp z13.s, p2/m, z13.s, z13.s
    KAI_ASM_INST(0x8540c823)  // ld1rw { z3.s }, p2/Z, [x1]
    neg z3.s, p2/M, z3.s
    mov z12.s, z11.s[3]
    mov z11.s, z11.s[0]
    mov z13.s, z13.s[0]
    mul z11.s, p2/M, z11.s, z3.s
    mul z12.s, p2/M, z12.s, z3.s
    mul z13.s, p2/M, z13.s, z3.s
KAI_ASM_LABEL(label_41)  // Height 3: skip row sum fixup
    add z31.s, z31.s, z11.s
    add z20.s, z20.s, z11.s
    ld1w { z0.s }, p2/Z, [x8]
    ld1w { z1.s }, p2/Z, [x8, #1, MUL VL]
    add z21.s, z21.s, z11.s
    add z22.s, z22.s, z11.s
    ld1w { z2.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x8, #3, MUL VL]
    add z16.s, z16.s, z12.s
    add z17.s, z17.s, z12.s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    add z18.s, z18.s, z12.s
    add z19.s, z19.s, z12.s
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    add x3, x0, #0x94
    add z24.s, z24.s, z13.s
    add z25.s, z25.s, z13.s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    add z26.s, z26.s, z13.s
    add z27.s, z27.s, z13.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    add z31.s, z31.s, z0.s
    add z20.s, z20.s, z1.s
    orr x24, x24, #0x80000000
    addvl x8, x8, #4
    add z21.s, z21.s, z2.s
    add z22.s, z22.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z26.s, z26.s, z2.s
    add z27.s, z27.s, z3.s
    KAI_ASM_INST(0x04a473ff)  // sqdmulh z31.s, z31.s, z4.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a472b5)  // sqdmulh z21.s, z21.s, z4.s
    KAI_ASM_INST(0x04a472d6)  // sqdmulh z22.s, z22.s, z4.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a47231)  // sqdmulh z17.s, z17.s, z4.s
    KAI_ASM_INST(0x04a47252)  // sqdmulh z18.s, z18.s, z4.s
    KAI_ASM_INST(0x04a47273)  // sqdmulh z19.s, z19.s, z4.s
    KAI_ASM_INST(0x4482881f)  // srshl z31.s, p2/M, z31.s, z0.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    KAI_ASM_INST(0x04a47318)  // sqdmulh z24.s, z24.s, z4.s
    KAI_ASM_INST(0x04a47339)  // sqdmulh z25.s, z25.s, z4.s
    KAI_ASM_INST(0x44828815)  // srshl z21.s, p2/M, z21.s, z0.s
    KAI_ASM_INST(0x44828816)  // srshl z22.s, p2/M, z22.s, z0.s
    KAI_ASM_INST(0x04a4735a)  // sqdmulh z26.s, z26.s, z4.s
    KAI_ASM_INST(0x04a4737b)  // sqdmulh z27.s, z27.s, z4.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828811)  // srshl z17.s, p2/M, z17.s, z0.s
    KAI_ASM_INST(0x44828812)  // srshl z18.s, p2/M, z18.s, z0.s
    KAI_ASM_INST(0x44828813)  // srshl z19.s, p2/M, z19.s, z0.s
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    add z31.s, z31.s, z4.s
    add z20.s, z20.s, z4.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    add z21.s, z21.s, z4.s
    add z22.s, z22.s, z4.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z31.s, p2/M, z31.s, z6.s
    smin z20.s, p2/M, z20.s, z6.s
    add z24.s, z24.s, z4.s
    add z25.s, z25.s, z4.s
    smin z21.s, p2/M, z21.s, z6.s
    smin z22.s, p2/M, z22.s, z6.s
    add z26.s, z26.s, z4.s
    add z27.s, z27.s, z4.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    smin z24.s, p2/M, z24.s, z6.s
    smin z25.s, p2/M, z25.s, z6.s
    smin z26.s, p2/M, z26.s, z6.s
    smin z27.s, p2/M, z27.s, z6.s
    smax z31.s, p2/M, z31.s, z5.s
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    smax z22.s, p2/M, z22.s, z5.s
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z31.h, z31.h, z20.h
    smax z24.s, p2/M, z24.s, z5.s
    smax z25.s, p2/M, z25.s, z5.s
    uzp1 z20.h, z21.h, z22.h
    smax z26.s, p2/M, z26.s, z5.s
    smax z27.s, p2/M, z27.s, z5.s
    uzp1 z16.h, z16.h, z17.h
    uzp1 z17.h, z18.h, z19.h
    uzp1 z24.h, z24.h, z25.h
    uzp1 z31.b, z31.b, z20.b
    uzp1 z25.h, z26.h, z27.h
    uzp1 z16.b, z16.b, z17.b
    st1b { z31.b }, p1, [x22]
    addvl x22, x22, #1
    uzp1 z24.b, z24.b, z25.b
    st1b { z16.b }, p1, [x7]
    st1b { z24.b }, p1, [x6]
    decw x21, ALL, MUL #4
    cmp x21, XZR
    bgt label_30
    b label_57
KAI_ASM_LABEL(label_43)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    bic x24, x24, #0x80000000
    ldr x21, [x0, #0x8]
    mov z15.b, #0x1
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_44)  // Height 4: Column loop
    mov x1, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    mov z19.s, #0
    mov z20.s, #0
    mov z21.s, #0
    whilelt p1.b, x1, x21
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
    tbnz x24, #3, label_46
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_48
KAI_ASM_LABEL(label_46)  // Height 4: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_47)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    ldr x1, [x1, #0x18]
    cbnz x23, label_48
    add x6, x6, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_48)  // Height 4: input setup done
    cmp x5, #0x10
    ble label_51
KAI_ASM_LABEL(label_49)  // Height 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    add x6, x6, #0x10
    add x4, x4, #0x10
    ld1rqb { z3.b }, p0/Z, [x2]
    ld1rqb { z4.b }, p0/Z, [x1]
    add x2, x2, #0x10
    add x1, x1, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    addvl x20, x20, #16
    KAI_ASM_INST(0x4586985c)  // usmmla z28.s, z2.b, z6.b
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4588985d)  // usmmla z29.s, z2.b, z8.b
    KAI_ASM_INST(0x4589985a)  // usmmla z26.s, z2.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    ld1b { z6.b }, p2/Z, [x20, #-8, MUL VL]
    KAI_ASM_INST(0x458a985e)  // usmmla z30.s, z2.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    ld1b { z7.b }, p2/Z, [x20, #-7, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #-6, MUL VL]
    KAI_ASM_INST(0x4584985b)  // usmmla z27.s, z2.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    ld1b { z9.b }, p2/Z, [x20, #-5, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #-4, MUL VL]
    KAI_ASM_INST(0x4585985f)  // usmmla z31.s, z2.b, z5.b
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1b { z4.b }, p2/Z, [x20, #-3, MUL VL]
    ld1b { z5.b }, p2/Z, [x20, #-2, MUL VL]
    KAI_ASM_INST(0x45869878)  // usmmla z24.s, z3.b, z6.b
    ld1b { z6.b }, p2/Z, [x20, #-1, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x4587987c)  // usmmla z28.s, z3.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45889879)  // usmmla z25.s, z3.b, z8.b
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x4589987d)  // usmmla z29.s, z3.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x458a987a)  // usmmla z26.s, z3.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x4584987e)  // usmmla z30.s, z3.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x4585987b)  // usmmla z27.s, z3.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    KAI_ASM_INST(0x4586987f)  // usmmla z31.s, z3.b, z6.b
    tbnz x24, #31, label_50
    udot z11.s, z0.b, z15.b
    udot z13.s, z2.b, z15.b
    udot z11.s, z1.b, z15.b
    udot z13.s, z3.b, z15.b
KAI_ASM_LABEL(label_50)  // Height 4: Multiply loop: unique 7: skip row sum
    sub x5, x5, #0x10
    cmp x5, #0x10
    bgt label_49
KAI_ASM_LABEL(label_51)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x5
    ld1b { z5.b }, p2/Z, [x20]
    ld1b { z6.b }, p2/Z, [x20, #1, MUL VL]
    subs x5, x5, #0x8
    ld1b { z7.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #5, MUL VL]
    ld1rqb { z1.b }, p0/Z, [x6]
    ld1rqb { z2.b }, p0/Z, [x4]
    ld1rqb { z3.b }, p0/Z, [x2]
    ld1rqb { z4.b }, p0/Z, [x1]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    ld1b { z4.b }, p2/Z, [x20, #6, MUL VL]
    KAI_ASM_INST(0x45859810)  // usmmla z16.s, z0.b, z5.b
    KAI_ASM_INST(0x45869814)  // usmmla z20.s, z0.b, z6.b
    KAI_ASM_INST(0x45879811)  // usmmla z17.s, z0.b, z7.b
    KAI_ASM_INST(0x45889815)  // usmmla z21.s, z0.b, z8.b
    KAI_ASM_INST(0x45899812)  // usmmla z18.s, z0.b, z9.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    ld1b { z5.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x4586985c)  // usmmla z28.s, z2.b, z6.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4588985d)  // usmmla z29.s, z2.b, z8.b
    KAI_ASM_INST(0x4589985a)  // usmmla z26.s, z2.b, z9.b
    KAI_ASM_INST(0x458a9816)  // usmmla z22.s, z0.b, z10.b
    KAI_ASM_INST(0x458a985e)  // usmmla z30.s, z2.b, z10.b
    KAI_ASM_INST(0x45849813)  // usmmla z19.s, z0.b, z4.b
    KAI_ASM_INST(0x4584985b)  // usmmla z27.s, z2.b, z4.b
    KAI_ASM_INST(0x45859817)  // usmmla z23.s, z0.b, z5.b
    KAI_ASM_INST(0x4585985f)  // usmmla z31.s, z2.b, z5.b
    ble label_52
    ld1b { z6.b }, p2/Z, [x20]
    ld1b { z7.b }, p2/Z, [x20, #1, MUL VL]
    ld1b { z8.b }, p2/Z, [x20, #2, MUL VL]
    ld1b { z9.b }, p2/Z, [x20, #3, MUL VL]
    ld1b { z10.b }, p2/Z, [x20, #4, MUL VL]
    ld1b { z4.b }, p2/Z, [x20, #5, MUL VL]
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    KAI_ASM_INST(0x45869878)  // usmmla z24.s, z3.b, z6.b
    ld1b { z5.b }, p2/Z, [x20, #6, MUL VL]
    ld1b { z6.b }, p2/Z, [x20, #7, MUL VL]
    KAI_ASM_INST(0x45879834)  // usmmla z20.s, z1.b, z7.b
    KAI_ASM_INST(0x4587987c)  // usmmla z28.s, z3.b, z7.b
    KAI_ASM_INST(0x45889831)  // usmmla z17.s, z1.b, z8.b
    KAI_ASM_INST(0x45889879)  // usmmla z25.s, z3.b, z8.b
    addvl x20, x20, #8
    KAI_ASM_INST(0x45899835)  // usmmla z21.s, z1.b, z9.b
    KAI_ASM_INST(0x4589987d)  // usmmla z29.s, z3.b, z9.b
    KAI_ASM_INST(0x458a9832)  // usmmla z18.s, z1.b, z10.b
    KAI_ASM_INST(0x458a987a)  // usmmla z26.s, z3.b, z10.b
    KAI_ASM_INST(0x45849836)  // usmmla z22.s, z1.b, z4.b
    KAI_ASM_INST(0x4584987e)  // usmmla z30.s, z3.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x4585987b)  // usmmla z27.s, z3.b, z5.b
    KAI_ASM_INST(0x45869837)  // usmmla z23.s, z1.b, z6.b
    KAI_ASM_INST(0x4586987f)  // usmmla z31.s, z3.b, z6.b
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: multiply skip
    tbnz x24, #31, label_53
    udot z11.s, z0.b, z15.b
    udot z13.s, z2.b, z15.b
    udot z11.s, z1.b, z15.b
    udot z13.s, z3.b, z15.b
KAI_ASM_LABEL(label_53)  // Height 4: Multiply loop: unique 8: skip row sum
    tbz x24, #3, label_54
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_47
KAI_ASM_LABEL(label_54)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    add x7, x22, x1
    add x6, x7, x1
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    add x5, x6, x1
    uzp1 z23.d, z24.d, z28.d
    uzp2 z24.d, z24.d, z28.d
    uzp1 z28.d, z25.d, z29.d
    uzp2 z25.d, z25.d, z29.d
    uzp1 z29.d, z26.d, z30.d
    uzp2 z26.d, z26.d, z30.d
    uzp1 z30.d, z27.d, z31.d
    uzp2 z27.d, z27.d, z31.d
    mov z31.d, z7.d
    tbnz x24, #31, label_55
    add x1, x0, #0x90
    KAI_ASM_INST(0x4491a96b)  // addp z11.s, p2/m, z11.s, z11.s
    KAI_ASM_INST(0x4491a9ad)  // addp z13.s, p2/m, z13.s, z13.s
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    neg z4.s, p2/M, z4.s
    mov z12.s, z11.s[3]
    mov z11.s, z11.s[0]
    mov z14.s, z13.s[3]
    mov z13.s, z13.s[0]
    mul z11.s, p2/M, z11.s, z4.s
    mul z12.s, p2/M, z12.s, z4.s
    mul z13.s, p2/M, z13.s, z4.s
    mul z14.s, p2/M, z14.s, z4.s
KAI_ASM_LABEL(label_55)  // Height 4: skip row sum fixup
    add z31.s, z31.s, z11.s
    add z20.s, z20.s, z11.s
    ld1w { z0.s }, p2/Z, [x8]
    ld1w { z1.s }, p2/Z, [x8, #1, MUL VL]
    add z21.s, z21.s, z11.s
    add z22.s, z22.s, z11.s
    ld1w { z2.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z3.s }, p2/Z, [x8, #3, MUL VL]
    add z16.s, z16.s, z12.s
    add z17.s, z17.s, z12.s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    add z18.s, z18.s, z12.s
    add z19.s, z19.s, z12.s
    KAI_ASM_INST(0x8540c824)  // ld1rw { z4.s }, p2/Z, [x1]
    add x3, x0, #0x94
    add z23.s, z23.s, z13.s
    add z28.s, z28.s, z13.s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    add z29.s, z29.s, z13.s
    add z30.s, z30.s, z13.s
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c825)  // ld1rw { z5.s }, p2/Z, [x1]
    add z24.s, z24.s, z14.s
    add z25.s, z25.s, z14.s
    orr x24, x24, #0x80000000
    addvl x8, x8, #4
    add z26.s, z26.s, z14.s
    add z27.s, z27.s, z14.s
    add z31.s, z31.s, z0.s
    add z20.s, z20.s, z1.s
    add z21.s, z21.s, z2.s
    add z22.s, z22.s, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z1.s
    add z18.s, z18.s, z2.s
    add z19.s, z19.s, z3.s
    add z23.s, z23.s, z0.s
    add z28.s, z28.s, z1.s
    add z29.s, z29.s, z2.s
    add z30.s, z30.s, z3.s
    add z24.s, z24.s, z0.s
    add z25.s, z25.s, z1.s
    KAI_ASM_INST(0x8540c880)  // ld1rw { z0.s }, p2/Z, [x4]
    add z26.s, z26.s, z2.s
    add z27.s, z27.s, z3.s
    KAI_ASM_INST(0x04a473ff)  // sqdmulh z31.s, z31.s, z4.s
    KAI_ASM_INST(0x04a47294)  // sqdmulh z20.s, z20.s, z4.s
    KAI_ASM_INST(0x04a472b5)  // sqdmulh z21.s, z21.s, z4.s
    KAI_ASM_INST(0x04a472d6)  // sqdmulh z22.s, z22.s, z4.s
    KAI_ASM_INST(0x04a47210)  // sqdmulh z16.s, z16.s, z4.s
    KAI_ASM_INST(0x04a47231)  // sqdmulh z17.s, z17.s, z4.s
    KAI_ASM_INST(0x04a47252)  // sqdmulh z18.s, z18.s, z4.s
    KAI_ASM_INST(0x04a47273)  // sqdmulh z19.s, z19.s, z4.s
    KAI_ASM_INST(0x4482881f)  // srshl z31.s, p2/M, z31.s, z0.s
    KAI_ASM_INST(0x44828814)  // srshl z20.s, p2/M, z20.s, z0.s
    KAI_ASM_INST(0x04a472f7)  // sqdmulh z23.s, z23.s, z4.s
    KAI_ASM_INST(0x04a4739c)  // sqdmulh z28.s, z28.s, z4.s
    KAI_ASM_INST(0x44828815)  // srshl z21.s, p2/M, z21.s, z0.s
    KAI_ASM_INST(0x44828816)  // srshl z22.s, p2/M, z22.s, z0.s
    KAI_ASM_INST(0x04a473bd)  // sqdmulh z29.s, z29.s, z4.s
    KAI_ASM_INST(0x04a473de)  // sqdmulh z30.s, z30.s, z4.s
    KAI_ASM_INST(0x44828810)  // srshl z16.s, p2/M, z16.s, z0.s
    KAI_ASM_INST(0x44828811)  // srshl z17.s, p2/M, z17.s, z0.s
    KAI_ASM_INST(0x04a47318)  // sqdmulh z24.s, z24.s, z4.s
    KAI_ASM_INST(0x04a47339)  // sqdmulh z25.s, z25.s, z4.s
    KAI_ASM_INST(0x44828812)  // srshl z18.s, p2/M, z18.s, z0.s
    KAI_ASM_INST(0x44828813)  // srshl z19.s, p2/M, z19.s, z0.s
    KAI_ASM_INST(0x04a4735a)  // sqdmulh z26.s, z26.s, z4.s
    KAI_ASM_INST(0x04a4737b)  // sqdmulh z27.s, z27.s, z4.s
    KAI_ASM_INST(0x8540c864)  // ld1rw { z4.s }, p2/Z, [x3]
    KAI_ASM_INST(0x44828817)  // srshl z23.s, p2/M, z23.s, z0.s
    KAI_ASM_INST(0x4482881c)  // srshl z28.s, p2/M, z28.s, z0.s
    KAI_ASM_INST(0x4482881d)  // srshl z29.s, p2/M, z29.s, z0.s
    KAI_ASM_INST(0x4482881e)  // srshl z30.s, p2/M, z30.s, z0.s
    KAI_ASM_INST(0x44828818)  // srshl z24.s, p2/M, z24.s, z0.s
    KAI_ASM_INST(0x44828819)  // srshl z25.s, p2/M, z25.s, z0.s
    KAI_ASM_INST(0x4482881a)  // srshl z26.s, p2/M, z26.s, z0.s
    add z31.s, z31.s, z4.s
    add z20.s, z20.s, z4.s
    KAI_ASM_INST(0x4482881b)  // srshl z27.s, p2/M, z27.s, z0.s
    add z21.s, z21.s, z4.s
    add z22.s, z22.s, z4.s
    add z16.s, z16.s, z4.s
    add z17.s, z17.s, z4.s
    add z18.s, z18.s, z4.s
    add z19.s, z19.s, z4.s
    smin z31.s, p2/M, z31.s, z6.s
    smin z20.s, p2/M, z20.s, z6.s
    add z23.s, z23.s, z4.s
    add z28.s, z28.s, z4.s
    smin z21.s, p2/M, z21.s, z6.s
    smin z22.s, p2/M, z22.s, z6.s
    add z29.s, z29.s, z4.s
    add z30.s, z30.s, z4.s
    smin z16.s, p2/M, z16.s, z6.s
    smin z17.s, p2/M, z17.s, z6.s
    add z24.s, z24.s, z4.s
    add z25.s, z25.s, z4.s
    smin z18.s, p2/M, z18.s, z6.s
    smin z19.s, p2/M, z19.s, z6.s
    add z26.s, z26.s, z4.s
    add z27.s, z27.s, z4.s
    smin z23.s, p2/M, z23.s, z6.s
    smin z28.s, p2/M, z28.s, z6.s
    smin z29.s, p2/M, z29.s, z6.s
    smin z30.s, p2/M, z30.s, z6.s
    smin z24.s, p2/M, z24.s, z6.s
    smin z25.s, p2/M, z25.s, z6.s
    smin z26.s, p2/M, z26.s, z6.s
    smin z27.s, p2/M, z27.s, z6.s
    smax z31.s, p2/M, z31.s, z5.s
    smax z20.s, p2/M, z20.s, z5.s
    smax z21.s, p2/M, z21.s, z5.s
    smax z22.s, p2/M, z22.s, z5.s
    smax z16.s, p2/M, z16.s, z5.s
    smax z17.s, p2/M, z17.s, z5.s
    smax z18.s, p2/M, z18.s, z5.s
    smax z19.s, p2/M, z19.s, z5.s
    uzp1 z31.h, z31.h, z20.h
    smax z23.s, p2/M, z23.s, z5.s
    smax z28.s, p2/M, z28.s, z5.s
    uzp1 z20.h, z21.h, z22.h
    smax z29.s, p2/M, z29.s, z5.s
    smax z30.s, p2/M, z30.s, z5.s
    uzp1 z16.h, z16.h, z17.h
    smax z24.s, p2/M, z24.s, z5.s
    smax z25.s, p2/M, z25.s, z5.s
    uzp1 z17.h, z18.h, z19.h
    smax z26.s, p2/M, z26.s, z5.s
    smax z27.s, p2/M, z27.s, z5.s
    uzp1 z23.h, z23.h, z28.h
    uzp1 z31.b, z31.b, z20.b
    uzp1 z28.h, z29.h, z30.h
    uzp1 z24.h, z24.h, z25.h
    uzp1 z16.b, z16.b, z17.b
    uzp1 z25.h, z26.h, z27.h
    st1b { z31.b }, p1, [x22]
    addvl x22, x22, #1
    uzp1 z23.b, z23.b, z28.b
    uzp1 z24.b, z24.b, z25.b
    st1b { z16.b }, p1, [x7]
    st1b { z23.b }, p1, [x6]
    st1b { z24.b }, p1, [x5]
    decw x21, ALL, MUL #4
    cmp x21, XZR
    bgt label_44
    add x26, x26, #0x4
    cmp x26, x25
    blt label_1
KAI_ASM_LABEL(label_57)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp d8, d9, [sp, 56]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_u8s8qa_mmla_4x4VL)

    KAI_ASM_END
