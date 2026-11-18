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

    KAI_ASM_CODE(sve_hybrid_u8u32_mmla_6x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_u8u32_mmla_6x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_u8u32_mmla_6x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_u8u32_mmla_6x4VL)
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
    bge label_66
    cmp x1, #0x4
    bgt label_53
    beq label_40
    cmp x1, #0x2
    bgt label_27
    beq label_14
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
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x25, #0, label_4
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
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
    cmp x7, #0x10
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    ble label_11
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: multiply skip
    tbz x25, #3, label_12
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_7
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    uzp1 z8.d, z8.d, z12.d
    uzp1 z9.d, z9.d, z13.d
    uzp1 z10.d, z10.d, z14.d
    uzp1 z11.d, z11.d, z15.d
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_2
    b label_79
KAI_ASM_LABEL(label_14)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_15)  // Height 2: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_16
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    b label_18
KAI_ASM_LABEL(label_16)  // Height 2: no bias
    tbz x25, #0, label_17
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    add x1, x23, x1
    ld1w { z12.s }, p4/Z, [x1]
    ld1w { z13.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x1, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    b label_18
KAI_ASM_LABEL(label_17)  // Height 2: no accumulate
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
KAI_ASM_LABEL(label_18)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_19
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_21
KAI_ASM_LABEL(label_19)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_20)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_21
    add x8, x8, x3
    add x6, x6, x3
KAI_ASM_LABEL(label_21)  // Height 2: input setup done
    cmp x7, #0x10
    ble label_23
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z2.b }, p0/Z, [x6]
    add x6, x6, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    bgt label_22
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    ld1rqb { z2.b }, p0/Z, [x6]
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    ble label_24
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: multiply skip
    tbz x25, #3, label_25
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_20
KAI_ASM_LABEL(label_25)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x1, x23, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    st1w { z7.s }, p4, [x23]
    st1w { z12.s }, p3, [x23, #1, MUL VL]
    st1w { z13.s }, p2, [x23, #2, MUL VL]
    st1w { z14.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z8.s }, p4, [x1]
    st1w { z9.s }, p3, [x1, #1, MUL VL]
    st1w { z10.s }, p2, [x1, #2, MUL VL]
    st1w { z11.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_15
    b label_79
KAI_ASM_LABEL(label_27)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_28)  // Height 3: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_29
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    mov z16.d, z8.d
    mov z20.d, z12.d
    mov z17.d, z9.d
    mov z21.d, z13.d
    mov z18.d, z10.d
    mov z22.d, z14.d
    mov z19.d, z11.d
    mov z23.d, z15.d
    b label_31
KAI_ASM_LABEL(label_29)  // Height 3: no bias
    tbz x25, #0, label_30
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    add x2, x23, x1
    add x1, x2, x1
    ld1w { z12.s }, p4/Z, [x2]
    ld1w { z13.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x2, #3, MUL VL]
    ld1w { z17.s }, p4/Z, [x1]
    ld1w { z18.s }, p3/Z, [x1, #1, MUL VL]
    ld1w { z19.s }, p2/Z, [x1, #2, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    ld1w { z24.s }, p1/Z, [x1, #3, MUL VL]
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    zip1 z16.d, z17.d, z20.d
    zip2 z20.d, z17.d, z20.d
    zip1 z17.d, z18.d, z21.d
    zip2 z21.d, z18.d, z21.d
    zip1 z18.d, z19.d, z22.d
    zip2 z22.d, z19.d, z22.d
    zip1 z19.d, z24.d, z23.d
    zip2 z23.d, z24.d, z23.d
    b label_31
KAI_ASM_LABEL(label_30)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_31)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_32
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_34
KAI_ASM_LABEL(label_32)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_33)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_34
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
KAI_ASM_LABEL(label_34)  // Height 3: input setup done
    cmp x7, #0x10
    ble label_36
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z2.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z3.b }, p0/Z, [x5]
    add x5, x5, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    bgt label_35
KAI_ASM_LABEL(label_36)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    ld1rqb { z2.b }, p0/Z, [x6]
    ld1rqb { z3.b }, p0/Z, [x5]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    ble label_37
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: multiply skip
    tbz x25, #3, label_38
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_33
KAI_ASM_LABEL(label_38)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x2, x23, x1
    add x1, x2, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    st1w { z7.s }, p4, [x23]
    uzp1 z16.d, z16.d, z20.d
    uzp1 z17.d, z17.d, z21.d
    st1w { z12.s }, p3, [x23, #1, MUL VL]
    uzp1 z18.d, z18.d, z22.d
    uzp1 z19.d, z19.d, z23.d
    st1w { z13.s }, p2, [x23, #2, MUL VL]
    st1w { z14.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z8.s }, p4, [x2]
    st1w { z9.s }, p3, [x2, #1, MUL VL]
    st1w { z10.s }, p2, [x2, #2, MUL VL]
    st1w { z11.s }, p1, [x2, #3, MUL VL]
    st1w { z16.s }, p4, [x1]
    st1w { z17.s }, p3, [x1, #1, MUL VL]
    st1w { z18.s }, p2, [x1, #2, MUL VL]
    st1w { z19.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_28
    b label_79
KAI_ASM_LABEL(label_40)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_41)  // Height 4: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_42
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    mov z16.d, z8.d
    mov z20.d, z12.d
    mov z17.d, z9.d
    mov z21.d, z13.d
    mov z18.d, z10.d
    mov z22.d, z14.d
    mov z19.d, z11.d
    mov z23.d, z15.d
    b label_44
KAI_ASM_LABEL(label_42)  // Height 4: no bias
    tbz x25, #0, label_43
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    add x3, x23, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z12.s }, p4/Z, [x3]
    ld1w { z13.s }, p3/Z, [x3, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x3, #3, MUL VL]
    ld1w { z17.s }, p4/Z, [x2]
    ld1w { z18.s }, p3/Z, [x2, #1, MUL VL]
    ld1w { z19.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z24.s }, p1/Z, [x2, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    ld1w { z20.s }, p4/Z, [x1]
    ld1w { z21.s }, p3/Z, [x1, #1, MUL VL]
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    ld1w { z22.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #3, MUL VL]
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    zip1 z16.d, z17.d, z20.d
    zip2 z20.d, z17.d, z20.d
    zip1 z17.d, z18.d, z21.d
    zip2 z21.d, z18.d, z21.d
    zip1 z18.d, z19.d, z22.d
    zip2 z22.d, z19.d, z22.d
    zip1 z19.d, z24.d, z23.d
    zip2 z23.d, z24.d, z23.d
    b label_44
KAI_ASM_LABEL(label_43)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_44)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_45
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_47
KAI_ASM_LABEL(label_45)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_46)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_47
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_47)  // Height 4: input setup done
    cmp x7, #0x10
    ble label_49
KAI_ASM_LABEL(label_48)  // Height 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z2.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z3.b }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqb { z4.b }, p0/Z, [x4]
    add x4, x4, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    bgt label_48
KAI_ASM_LABEL(label_49)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    ld1rqb { z2.b }, p0/Z, [x6]
    ld1rqb { z3.b }, p0/Z, [x5]
    ld1rqb { z4.b }, p0/Z, [x4]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    ble label_50
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
KAI_ASM_LABEL(label_50)  // Height 4: Multiply loop: multiply skip
    tbz x25, #3, label_51
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_46
KAI_ASM_LABEL(label_51)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x3, x23, x1
    add x2, x3, x1
    add x1, x2, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    st1w { z7.s }, p4, [x23]
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    st1w { z12.s }, p3, [x23, #1, MUL VL]
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    st1w { z13.s }, p2, [x23, #2, MUL VL]
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    st1w { z14.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    st1w { z8.s }, p4, [x3]
    st1w { z9.s }, p3, [x3, #1, MUL VL]
    st1w { z10.s }, p2, [x3, #2, MUL VL]
    st1w { z11.s }, p1, [x3, #3, MUL VL]
    st1w { z15.s }, p4, [x2]
    st1w { z20.s }, p3, [x2, #1, MUL VL]
    st1w { z21.s }, p2, [x2, #2, MUL VL]
    st1w { z22.s }, p1, [x2, #3, MUL VL]
    st1w { z16.s }, p4, [x1]
    st1w { z17.s }, p3, [x1, #1, MUL VL]
    st1w { z18.s }, p2, [x1, #2, MUL VL]
    st1w { z19.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_41
    b label_79
KAI_ASM_LABEL(label_53)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_54)  // Height 5: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_55
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    mov z16.d, z8.d
    mov z20.d, z12.d
    mov z17.d, z9.d
    mov z21.d, z13.d
    mov z18.d, z10.d
    mov z22.d, z14.d
    mov z19.d, z11.d
    mov z23.d, z15.d
    mov z24.d, z8.d
    mov z28.d, z12.d
    mov z25.d, z9.d
    mov z29.d, z13.d
    mov z26.d, z10.d
    mov z30.d, z14.d
    mov z27.d, z11.d
    mov z31.d, z15.d
    b label_57
KAI_ASM_LABEL(label_55)  // Height 5: no bias
    tbz x25, #0, label_56
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    add x4, x23, x1
    add x3, x4, x1
    add x2, x3, x1
    ld1w { z12.s }, p4/Z, [x4]
    ld1w { z13.s }, p3/Z, [x4, #1, MUL VL]
    add x1, x2, x1
    ld1w { z14.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x4, #3, MUL VL]
    ld1w { z17.s }, p4/Z, [x3]
    ld1w { z18.s }, p3/Z, [x3, #1, MUL VL]
    ld1w { z19.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z24.s }, p1/Z, [x3, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    ld1w { z20.s }, p4/Z, [x2]
    ld1w { z21.s }, p3/Z, [x2, #1, MUL VL]
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    ld1w { z22.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x2, #3, MUL VL]
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    ld1w { z25.s }, p4/Z, [x1]
    ld1w { z26.s }, p3/Z, [x1, #1, MUL VL]
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    ld1w { z27.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z6.s }, p1/Z, [x1, #3, MUL VL]
    zip1 z16.d, z17.d, z20.d
    zip2 z20.d, z17.d, z20.d
    zip1 z17.d, z18.d, z21.d
    zip2 z21.d, z18.d, z21.d
    zip1 z18.d, z19.d, z22.d
    zip2 z22.d, z19.d, z22.d
    zip1 z19.d, z24.d, z23.d
    zip2 z23.d, z24.d, z23.d
    zip1 z24.d, z25.d, z28.d
    zip2 z28.d, z25.d, z28.d
    zip1 z25.d, z26.d, z29.d
    zip2 z29.d, z26.d, z29.d
    zip1 z26.d, z27.d, z30.d
    zip2 z30.d, z27.d, z30.d
    zip1 z27.d, z6.d, z31.d
    zip2 z31.d, z6.d, z31.d
    b label_57
KAI_ASM_LABEL(label_56)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_57)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_58
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_60
KAI_ASM_LABEL(label_58)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_59)  // Height 5: String loop
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
    cbnz x24, label_60
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_60)  // Height 5: input setup done
    cmp x7, #0x10
    ble label_62
KAI_ASM_LABEL(label_61)  // Height 5: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z2.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z3.b }, p0/Z, [x5]
    ld1rqb { z4.b }, p0/Z, [x4]
    ld1rqb { z5.b }, p0/Z, [x2]
    add x5, x5, #0x10
    add x4, x4, #0x10
    add x2, x2, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79898)  // ummla z24.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989c)  // ummla z28.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79899)  // ummla z25.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989d)  // ummla z29.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989a)  // ummla z26.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989e)  // ummla z30.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989b)  // ummla z27.s, z4.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989f)  // ummla z31.s, z4.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b8)  // ummla z24.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bc)  // ummla z28.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b9)  // ummla z25.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bd)  // ummla z29.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798ba)  // ummla z26.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698be)  // ummla z30.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798bb)  // ummla z27.s, z5.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bf)  // ummla z31.s, z5.b, z6.b
    bgt label_61
KAI_ASM_LABEL(label_62)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    ld1rqb { z2.b }, p0/Z, [x6]
    ld1rqb { z3.b }, p0/Z, [x5]
    ld1rqb { z4.b }, p0/Z, [x4]
    ld1rqb { z5.b }, p0/Z, [x2]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79898)  // ummla z24.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989c)  // ummla z28.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79899)  // ummla z25.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989d)  // ummla z29.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989a)  // ummla z26.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989e)  // ummla z30.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989b)  // ummla z27.s, z4.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989f)  // ummla z31.s, z4.b, z6.b
    ble label_63
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b8)  // ummla z24.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bc)  // ummla z28.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b9)  // ummla z25.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bd)  // ummla z29.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798ba)  // ummla z26.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698be)  // ummla z30.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798bb)  // ummla z27.s, z5.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bf)  // ummla z31.s, z5.b, z6.b
KAI_ASM_LABEL(label_63)  // Height 5: Multiply loop: multiply skip
    tbz x25, #3, label_64
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_59
KAI_ASM_LABEL(label_64)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x4, x23, x1
    add x3, x4, x1
    add x2, x3, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    st1w { z7.s }, p4, [x23]
    add x1, x2, x1
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    st1w { z12.s }, p3, [x23, #1, MUL VL]
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    st1w { z13.s }, p2, [x23, #2, MUL VL]
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    st1w { z14.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    st1w { z8.s }, p4, [x4]
    uzp1 z24.d, z24.d, z28.d
    uzp1 z25.d, z25.d, z29.d
    st1w { z9.s }, p3, [x4, #1, MUL VL]
    uzp1 z26.d, z26.d, z30.d
    uzp1 z27.d, z27.d, z31.d
    st1w { z10.s }, p2, [x4, #2, MUL VL]
    st1w { z11.s }, p1, [x4, #3, MUL VL]
    st1w { z15.s }, p4, [x3]
    st1w { z20.s }, p3, [x3, #1, MUL VL]
    st1w { z21.s }, p2, [x3, #2, MUL VL]
    st1w { z22.s }, p1, [x3, #3, MUL VL]
    st1w { z16.s }, p4, [x2]
    st1w { z17.s }, p3, [x2, #1, MUL VL]
    st1w { z18.s }, p2, [x2, #2, MUL VL]
    st1w { z19.s }, p1, [x2, #3, MUL VL]
    st1w { z24.s }, p4, [x1]
    st1w { z25.s }, p3, [x1, #1, MUL VL]
    st1w { z26.s }, p2, [x1, #2, MUL VL]
    st1w { z27.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_54
    b label_79
KAI_ASM_LABEL(label_66)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_67)  // Height 6: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_68
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    mov z16.d, z8.d
    mov z20.d, z12.d
    mov z17.d, z9.d
    mov z21.d, z13.d
    mov z18.d, z10.d
    mov z22.d, z14.d
    mov z19.d, z11.d
    mov z23.d, z15.d
    mov z24.d, z8.d
    mov z28.d, z12.d
    mov z25.d, z9.d
    mov z29.d, z13.d
    mov z26.d, z10.d
    mov z30.d, z14.d
    mov z27.d, z11.d
    mov z31.d, z15.d
    b label_70
KAI_ASM_LABEL(label_68)  // Height 6: no bias
    tbz x25, #0, label_69
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x23]
    ld1w { z10.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #3, MUL VL]
    add x5, x23, x1
    add x4, x5, x1
    add x3, x4, x1
    ld1w { z12.s }, p4/Z, [x5]
    ld1w { z13.s }, p3/Z, [x5, #1, MUL VL]
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z14.s }, p2/Z, [x5, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x5, #3, MUL VL]
    ld1w { z17.s }, p4/Z, [x4]
    ld1w { z18.s }, p3/Z, [x4, #1, MUL VL]
    ld1w { z19.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z24.s }, p1/Z, [x4, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    ld1w { z20.s }, p4/Z, [x3]
    ld1w { z21.s }, p3/Z, [x3, #1, MUL VL]
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    ld1w { z22.s }, p2/Z, [x3, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x3, #3, MUL VL]
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    ld1w { z25.s }, p4/Z, [x2]
    ld1w { z26.s }, p3/Z, [x2, #1, MUL VL]
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    ld1w { z27.s }, p2/Z, [x2, #2, MUL VL]
    ld1w { z6.s }, p1/Z, [x2, #3, MUL VL]
    zip1 z16.d, z17.d, z20.d
    zip2 z20.d, z17.d, z20.d
    ld1w { z28.s }, p4/Z, [x1]
    ld1w { z29.s }, p3/Z, [x1, #1, MUL VL]
    zip1 z17.d, z18.d, z21.d
    zip2 z21.d, z18.d, z21.d
    ld1w { z30.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z31.s }, p1/Z, [x1, #3, MUL VL]
    zip1 z18.d, z19.d, z22.d
    zip2 z22.d, z19.d, z22.d
    zip1 z19.d, z24.d, z23.d
    zip2 z23.d, z24.d, z23.d
    zip1 z24.d, z25.d, z28.d
    zip2 z28.d, z25.d, z28.d
    zip1 z25.d, z26.d, z29.d
    zip2 z29.d, z26.d, z29.d
    zip1 z26.d, z27.d, z30.d
    zip2 z30.d, z27.d, z30.d
    zip1 z27.d, z6.d, z31.d
    zip2 z31.d, z6.d, z31.d
    b label_70
KAI_ASM_LABEL(label_69)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_70)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_71
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_73
KAI_ASM_LABEL(label_71)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_72)  // Height 6: String loop
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
    cbnz x24, label_73
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_73)  // Height 6: input setup done
    cmp x7, #0x10
    ble label_75
KAI_ASM_LABEL(label_74)  // Height 6: Multiply loop: Main loop head
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    sub x7, x7, #0x10
    cmp x7, #0x10
    ld1rqb { z1.b }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqb { z2.b }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqb { z3.b }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqb { z4.b }, p0/Z, [x4]
    ld1rqb { z5.b }, p0/Z, [x2]
    ld1rqb { z6.b }, p0/Z, [x1]
    add x4, x4, #0x10
    add x2, x2, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    add x1, x1, #0x10
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79898)  // ummla z24.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989c)  // ummla z28.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79899)  // ummla z25.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989d)  // ummla z29.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989a)  // ummla z26.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989e)  // ummla z30.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989b)  // ummla z27.s, z4.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989f)  // ummla z31.s, z4.b, z6.b
    ld1b { z7.b }, p5/Z, [x21, #-8, MUL VL]
    ld1b { z6.b }, p5/Z, [x21, #-7, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b8)  // ummla z24.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-6, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bc)  // ummla z28.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-5, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b9)  // ummla z25.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bd)  // ummla z29.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-3, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798ba)  // ummla z26.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #-2, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698be)  // ummla z30.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #-1, MUL VL]
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798bb)  // ummla z27.s, z5.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bf)  // ummla z31.s, z5.b, z6.b
    bgt label_74
KAI_ASM_LABEL(label_75)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x7
    ld1b { z7.b }, p5/Z, [x21]
    subs x7, x7, #0x8
    ld1rqb { z1.b }, p0/Z, [x8]
    ld1rqb { z2.b }, p0/Z, [x6]
    ld1rqb { z3.b }, p0/Z, [x5]
    ld1rqb { z4.b }, p0/Z, [x4]
    ld1rqb { z5.b }, p0/Z, [x2]
    ld1rqb { z6.b }, p0/Z, [x1]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79808)  // ummla z8.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79850)  // ummla z16.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79898)  // ummla z24.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6980c)  // ummla z12.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69854)  // ummla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989c)  // ummla z28.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79809)  // ummla z9.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79851)  // ummla z17.s, z2.b, z7.b
    KAI_ASM_INST(0x45c79899)  // ummla z25.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6980d)  // ummla z13.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69855)  // ummla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989d)  // ummla z29.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7980a)  // ummla z10.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79852)  // ummla z18.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989a)  // ummla z26.s, z4.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6980e)  // ummla z14.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69856)  // ummla z22.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989e)  // ummla z30.s, z4.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7980b)  // ummla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x45c79853)  // ummla z19.s, z2.b, z7.b
    KAI_ASM_INST(0x45c7989b)  // ummla z27.s, z4.b, z7.b
    KAI_ASM_INST(0x45c6980f)  // ummla z15.s, z0.b, z6.b
    KAI_ASM_INST(0x45c69857)  // ummla z23.s, z2.b, z6.b
    KAI_ASM_INST(0x45c6989f)  // ummla z31.s, z4.b, z6.b
    ble label_76
    ld1b { z7.b }, p5/Z, [x21]
    ld1b { z6.b }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x45c79828)  // ummla z8.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79870)  // ummla z16.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b8)  // ummla z24.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x45c6982c)  // ummla z12.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69874)  // ummla z20.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bc)  // ummla z28.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x45c79829)  // ummla z9.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79871)  // ummla z17.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798b9)  // ummla z25.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x45c6982d)  // ummla z13.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69875)  // ummla z21.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bd)  // ummla z29.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #5, MUL VL]
    KAI_ASM_INST(0x45c7982a)  // ummla z10.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79872)  // ummla z18.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798ba)  // ummla z26.s, z5.b, z7.b
    ld1b { z7.b }, p5/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x45c6982e)  // ummla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69876)  // ummla z22.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698be)  // ummla z30.s, z5.b, z6.b
    ld1b { z6.b }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    KAI_ASM_INST(0x45c7982b)  // ummla z11.s, z1.b, z7.b
    KAI_ASM_INST(0x45c79873)  // ummla z19.s, z3.b, z7.b
    KAI_ASM_INST(0x45c798bb)  // ummla z27.s, z5.b, z7.b
    KAI_ASM_INST(0x45c6982f)  // ummla z15.s, z1.b, z6.b
    KAI_ASM_INST(0x45c69877)  // ummla z23.s, z3.b, z6.b
    KAI_ASM_INST(0x45c698bf)  // ummla z31.s, z5.b, z6.b
KAI_ASM_LABEL(label_76)  // Height 6: Multiply loop: multiply skip
    tbz x25, #3, label_77
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_72
KAI_ASM_LABEL(label_77)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x5, x23, x1
    add x4, x5, x1
    add x3, x4, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    st1w { z7.s }, p4, [x23]
    add x2, x3, x1
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    st1w { z12.s }, p3, [x23, #1, MUL VL]
    add x1, x2, x1
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    st1w { z13.s }, p2, [x23, #2, MUL VL]
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    st1w { z14.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    st1w { z8.s }, p4, [x5]
    uzp1 z23.d, z24.d, z28.d
    uzp2 z24.d, z24.d, z28.d
    st1w { z9.s }, p3, [x5, #1, MUL VL]
    uzp1 z28.d, z25.d, z29.d
    uzp2 z25.d, z25.d, z29.d
    st1w { z10.s }, p2, [x5, #2, MUL VL]
    uzp1 z29.d, z26.d, z30.d
    uzp2 z26.d, z26.d, z30.d
    st1w { z11.s }, p1, [x5, #3, MUL VL]
    uzp1 z30.d, z27.d, z31.d
    uzp2 z27.d, z27.d, z31.d
    st1w { z15.s }, p4, [x4]
    st1w { z20.s }, p3, [x4, #1, MUL VL]
    st1w { z21.s }, p2, [x4, #2, MUL VL]
    st1w { z22.s }, p1, [x4, #3, MUL VL]
    st1w { z16.s }, p4, [x3]
    st1w { z17.s }, p3, [x3, #1, MUL VL]
    st1w { z18.s }, p2, [x3, #2, MUL VL]
    st1w { z19.s }, p1, [x3, #3, MUL VL]
    st1w { z23.s }, p4, [x2]
    st1w { z28.s }, p3, [x2, #1, MUL VL]
    st1w { z29.s }, p2, [x2, #2, MUL VL]
    st1w { z30.s }, p1, [x2, #3, MUL VL]
    st1w { z24.s }, p4, [x1]
    st1w { z25.s }, p3, [x1, #1, MUL VL]
    st1w { z26.s }, p2, [x1, #2, MUL VL]
    st1w { z27.s }, p1, [x1, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_67
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_79)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_u8u32_mmla_6x4VL)

    KAI_ASM_END
