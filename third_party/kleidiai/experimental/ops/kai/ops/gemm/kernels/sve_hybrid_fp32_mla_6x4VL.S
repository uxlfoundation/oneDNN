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

    KAI_ASM_CODE(sve_hybrid_fp32_mla_6x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_fp32_mla_6x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_fp32_mla_6x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_fp32_mla_6x4VL)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ptrue p5.b
    mov x27, #0
    ldr x4, [x0, #0x60]
    ldr w1, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x26, [x0, #0]
    cmp x4, #0
    cset x25, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x25, x25, x1, LSL #1
    cset x1, NE
    orr x25, x25, x2, LSL #1
    orr x25, x25, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x26, x27
    cmp x1, #0x6
    bge label_71
    cmp x1, #0x4
    bgt label_57
    beq label_43
    cmp x1, #0x2
    bgt label_29
    beq label_15
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
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
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
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x7, #0x4
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    fmla z8.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z11.s, z7.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z11.s, z7.s, z0.s[3]
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    fmla z8.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z11.s, z7.s, z0.s[0]
    ble label_11
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z11.s, z7.s, z0.s[1]
    ble label_11
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z11.s, z7.s, z0.s[2]
    ble label_11
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z11.s, z7.s, z0.s[3]
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: multiply skip
    tbz x25, #3, label_12
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_7
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    tbz x25, #1, label_13
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d451)  // ld1rw { z17.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d430)  // ld1rw { z16.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
KAI_ASM_LABEL(label_13)  // Height 1: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_2
    b label_85
KAI_ASM_LABEL(label_15)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_16)  // Height 2: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_17
    ld1w { z8.s }, p4/Z, [x20]
    ld1w { z9.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_19
KAI_ASM_LABEL(label_17)  // Height 2: no bias
    tbz x25, #0, label_18
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
    b label_19
KAI_ASM_LABEL(label_18)  // Height 2: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_19)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_20
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_22
KAI_ASM_LABEL(label_20)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_21)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_22
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_22)  // Height 2: input setup done
    cmp x7, #0x4
    ble label_24
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z1.s }, p0/Z, [x6]
    add x6, x6, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    ld1rqw { z1.s }, p0/Z, [x6]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    ble label_25
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ble label_25
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    ble label_25
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: multiply skip
    tbz x25, #3, label_26
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_21
KAI_ASM_LABEL(label_26)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x7, x23, x1
    tbz x25, #1, label_27
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d451)  // ld1rw { z17.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d430)  // ld1rw { z16.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmin z12.s, p5/M, z12.s, z17.s
    fmin z13.s, p5/M, z13.s, z17.s
    fmin z14.s, p5/M, z14.s, z17.s
    fmin z15.s, p5/M, z15.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
    fmax z12.s, p5/M, z12.s, z16.s
    fmax z13.s, p5/M, z13.s, z16.s
    fmax z14.s, p5/M, z14.s, z16.s
    fmax z15.s, p5/M, z15.s, z16.s
KAI_ASM_LABEL(label_27)  // Height 2: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x7]
    st1w { z13.s }, p3, [x7, #1, MUL VL]
    st1w { z14.s }, p2, [x7, #2, MUL VL]
    st1w { z15.s }, p1, [x7, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_16
    b label_85
KAI_ASM_LABEL(label_29)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_30)  // Height 3: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_31
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
    b label_33
KAI_ASM_LABEL(label_31)  // Height 3: no bias
    tbz x25, #0, label_32
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
    b label_33
KAI_ASM_LABEL(label_32)  // Height 3: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
KAI_ASM_LABEL(label_33)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_34
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_36
KAI_ASM_LABEL(label_34)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_35)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_36
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_36)  // Height 3: input setup done
    cmp x7, #0x4
    ble label_38
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z1.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z2.s }, p0/Z, [x5]
    add x5, x5, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    bgt label_37
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    ld1rqw { z1.s }, p0/Z, [x6]
    ld1rqw { z2.s }, p0/Z, [x5]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    ble label_39
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    ble label_39
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    ble label_39
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
KAI_ASM_LABEL(label_39)  // Height 3: Multiply loop: multiply skip
    tbz x25, #3, label_40
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_35
KAI_ASM_LABEL(label_40)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x7, x23, x1
    add x6, x7, x1
    tbz x25, #1, label_41
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d455)  // ld1rw { z21.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d434)  // ld1rw { z20.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z21.s
    fmin z9.s, p5/M, z9.s, z21.s
    fmin z10.s, p5/M, z10.s, z21.s
    fmin z11.s, p5/M, z11.s, z21.s
    fmin z12.s, p5/M, z12.s, z21.s
    fmin z13.s, p5/M, z13.s, z21.s
    fmin z14.s, p5/M, z14.s, z21.s
    fmin z15.s, p5/M, z15.s, z21.s
    fmin z16.s, p5/M, z16.s, z21.s
    fmin z17.s, p5/M, z17.s, z21.s
    fmin z18.s, p5/M, z18.s, z21.s
    fmin z19.s, p5/M, z19.s, z21.s
    fmax z8.s, p5/M, z8.s, z20.s
    fmax z9.s, p5/M, z9.s, z20.s
    fmax z10.s, p5/M, z10.s, z20.s
    fmax z11.s, p5/M, z11.s, z20.s
    fmax z12.s, p5/M, z12.s, z20.s
    fmax z13.s, p5/M, z13.s, z20.s
    fmax z14.s, p5/M, z14.s, z20.s
    fmax z15.s, p5/M, z15.s, z20.s
    fmax z16.s, p5/M, z16.s, z20.s
    fmax z17.s, p5/M, z17.s, z20.s
    fmax z18.s, p5/M, z18.s, z20.s
    fmax z19.s, p5/M, z19.s, z20.s
KAI_ASM_LABEL(label_41)  // Height 3: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x7]
    st1w { z13.s }, p3, [x7, #1, MUL VL]
    st1w { z14.s }, p2, [x7, #2, MUL VL]
    st1w { z15.s }, p1, [x7, #3, MUL VL]
    st1w { z16.s }, p4, [x6]
    st1w { z17.s }, p3, [x6, #1, MUL VL]
    st1w { z18.s }, p2, [x6, #2, MUL VL]
    st1w { z19.s }, p1, [x6, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_30
    b label_85
KAI_ASM_LABEL(label_43)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_44)  // Height 4: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_45
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
    b label_47
KAI_ASM_LABEL(label_45)  // Height 4: no bias
    tbz x25, #0, label_46
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
    b label_47
KAI_ASM_LABEL(label_46)  // Height 4: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
KAI_ASM_LABEL(label_47)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_48
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_50
KAI_ASM_LABEL(label_48)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_49)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_50
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_50)  // Height 4: input setup done
    cmp x7, #0x4
    ble label_52
KAI_ASM_LABEL(label_51)  // Height 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z1.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z2.s }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqw { z3.s }, p0/Z, [x4]
    add x4, x4, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    bgt label_51
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    ld1rqw { z1.s }, p0/Z, [x6]
    ld1rqw { z2.s }, p0/Z, [x5]
    ld1rqw { z3.s }, p0/Z, [x4]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    ble label_53
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    ble label_53
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    ble label_53
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
KAI_ASM_LABEL(label_53)  // Height 4: Multiply loop: multiply skip
    tbz x25, #3, label_54
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_49
KAI_ASM_LABEL(label_54)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x25, #1, label_55
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d459)  // ld1rw { z25.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d438)  // ld1rw { z24.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z25.s
    fmin z9.s, p5/M, z9.s, z25.s
    fmin z10.s, p5/M, z10.s, z25.s
    fmin z11.s, p5/M, z11.s, z25.s
    fmin z12.s, p5/M, z12.s, z25.s
    fmin z13.s, p5/M, z13.s, z25.s
    fmin z14.s, p5/M, z14.s, z25.s
    fmin z15.s, p5/M, z15.s, z25.s
    fmin z16.s, p5/M, z16.s, z25.s
    fmin z17.s, p5/M, z17.s, z25.s
    fmin z18.s, p5/M, z18.s, z25.s
    fmin z19.s, p5/M, z19.s, z25.s
    fmin z20.s, p5/M, z20.s, z25.s
    fmin z21.s, p5/M, z21.s, z25.s
    fmin z22.s, p5/M, z22.s, z25.s
    fmin z23.s, p5/M, z23.s, z25.s
    fmax z8.s, p5/M, z8.s, z24.s
    fmax z9.s, p5/M, z9.s, z24.s
    fmax z10.s, p5/M, z10.s, z24.s
    fmax z11.s, p5/M, z11.s, z24.s
    fmax z12.s, p5/M, z12.s, z24.s
    fmax z13.s, p5/M, z13.s, z24.s
    fmax z14.s, p5/M, z14.s, z24.s
    fmax z15.s, p5/M, z15.s, z24.s
    fmax z16.s, p5/M, z16.s, z24.s
    fmax z17.s, p5/M, z17.s, z24.s
    fmax z18.s, p5/M, z18.s, z24.s
    fmax z19.s, p5/M, z19.s, z24.s
    fmax z20.s, p5/M, z20.s, z24.s
    fmax z21.s, p5/M, z21.s, z24.s
    fmax z22.s, p5/M, z22.s, z24.s
    fmax z23.s, p5/M, z23.s, z24.s
KAI_ASM_LABEL(label_55)  // Height 4: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x7]
    st1w { z13.s }, p3, [x7, #1, MUL VL]
    st1w { z14.s }, p2, [x7, #2, MUL VL]
    st1w { z15.s }, p1, [x7, #3, MUL VL]
    st1w { z16.s }, p4, [x6]
    st1w { z17.s }, p3, [x6, #1, MUL VL]
    st1w { z18.s }, p2, [x6, #2, MUL VL]
    st1w { z19.s }, p1, [x6, #3, MUL VL]
    st1w { z20.s }, p4, [x5]
    st1w { z21.s }, p3, [x5, #1, MUL VL]
    st1w { z22.s }, p2, [x5, #2, MUL VL]
    st1w { z23.s }, p1, [x5, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_44
    b label_85
KAI_ASM_LABEL(label_57)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_58)  // Height 5: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_59
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
    b label_61
KAI_ASM_LABEL(label_59)  // Height 5: no bias
    tbz x25, #0, label_60
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
    b label_61
KAI_ASM_LABEL(label_60)  // Height 5: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
KAI_ASM_LABEL(label_61)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_62
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_64
KAI_ASM_LABEL(label_62)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_63)  // Height 5: String loop
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
    cbnz x24, label_64
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_64)  // Height 5: input setup done
    cmp x7, #0x4
    ble label_66
KAI_ASM_LABEL(label_65)  // Height 5: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z1.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z2.s }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqw { z3.s }, p0/Z, [x4]
    ld1rqw { z4.s }, p0/Z, [x2]
    add x4, x4, #0x10
    add x2, x2, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    bgt label_65
KAI_ASM_LABEL(label_66)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    ld1rqw { z1.s }, p0/Z, [x6]
    ld1rqw { z2.s }, p0/Z, [x5]
    ld1rqw { z3.s }, p0/Z, [x4]
    ld1rqw { z4.s }, p0/Z, [x2]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    ble label_67
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[1]
    addvl x21, x21, #4
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    ble label_67
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    addvl x21, x21, #4
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    ble label_67
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
KAI_ASM_LABEL(label_67)  // Height 5: Multiply loop: multiply skip
    tbz x25, #3, label_68
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_63
KAI_ASM_LABEL(label_68)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x25, #1, label_69
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d45d)  // ld1rw { z29.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d43c)  // ld1rw { z28.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z29.s
    fmin z9.s, p5/M, z9.s, z29.s
    fmin z10.s, p5/M, z10.s, z29.s
    fmin z11.s, p5/M, z11.s, z29.s
    fmin z12.s, p5/M, z12.s, z29.s
    fmin z13.s, p5/M, z13.s, z29.s
    fmin z14.s, p5/M, z14.s, z29.s
    fmin z15.s, p5/M, z15.s, z29.s
    fmin z16.s, p5/M, z16.s, z29.s
    fmin z17.s, p5/M, z17.s, z29.s
    fmin z18.s, p5/M, z18.s, z29.s
    fmin z19.s, p5/M, z19.s, z29.s
    fmin z20.s, p5/M, z20.s, z29.s
    fmin z21.s, p5/M, z21.s, z29.s
    fmin z22.s, p5/M, z22.s, z29.s
    fmin z23.s, p5/M, z23.s, z29.s
    fmin z24.s, p5/M, z24.s, z29.s
    fmin z25.s, p5/M, z25.s, z29.s
    fmin z26.s, p5/M, z26.s, z29.s
    fmin z27.s, p5/M, z27.s, z29.s
    fmax z8.s, p5/M, z8.s, z28.s
    fmax z9.s, p5/M, z9.s, z28.s
    fmax z10.s, p5/M, z10.s, z28.s
    fmax z11.s, p5/M, z11.s, z28.s
    fmax z12.s, p5/M, z12.s, z28.s
    fmax z13.s, p5/M, z13.s, z28.s
    fmax z14.s, p5/M, z14.s, z28.s
    fmax z15.s, p5/M, z15.s, z28.s
    fmax z16.s, p5/M, z16.s, z28.s
    fmax z17.s, p5/M, z17.s, z28.s
    fmax z18.s, p5/M, z18.s, z28.s
    fmax z19.s, p5/M, z19.s, z28.s
    fmax z20.s, p5/M, z20.s, z28.s
    fmax z21.s, p5/M, z21.s, z28.s
    fmax z22.s, p5/M, z22.s, z28.s
    fmax z23.s, p5/M, z23.s, z28.s
    fmax z24.s, p5/M, z24.s, z28.s
    fmax z25.s, p5/M, z25.s, z28.s
    fmax z26.s, p5/M, z26.s, z28.s
    fmax z27.s, p5/M, z27.s, z28.s
KAI_ASM_LABEL(label_69)  // Height 5: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x7]
    st1w { z13.s }, p3, [x7, #1, MUL VL]
    st1w { z14.s }, p2, [x7, #2, MUL VL]
    st1w { z15.s }, p1, [x7, #3, MUL VL]
    st1w { z16.s }, p4, [x6]
    st1w { z17.s }, p3, [x6, #1, MUL VL]
    st1w { z18.s }, p2, [x6, #2, MUL VL]
    st1w { z19.s }, p1, [x6, #3, MUL VL]
    st1w { z20.s }, p4, [x5]
    st1w { z21.s }, p3, [x5, #1, MUL VL]
    st1w { z22.s }, p2, [x5, #2, MUL VL]
    st1w { z23.s }, p1, [x5, #3, MUL VL]
    st1w { z24.s }, p4, [x4]
    st1w { z25.s }, p3, [x4, #1, MUL VL]
    st1w { z26.s }, p2, [x4, #2, MUL VL]
    st1w { z27.s }, p1, [x4, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_58
    b label_85
KAI_ASM_LABEL(label_71)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_72)  // Height 6: Column loop
    mov x1, #0
    whilelt p4.s, x1, x22
    incw x1
    whilelt p3.s, x1, x22
    incw x1
    whilelt p2.s, x1, x22
    incw x1
    whilelt p1.s, x1, x22
    cbz x20, label_73
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
    b label_75
KAI_ASM_LABEL(label_73)  // Height 6: no bias
    tbz x25, #0, label_74
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
    b label_75
KAI_ASM_LABEL(label_74)  // Height 6: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_75)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_76
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_78
KAI_ASM_LABEL(label_76)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_77)  // Height 6: String loop
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
    cbnz x24, label_78
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_78)  // Height 6: input setup done
    cmp x7, #0x4
    ble label_80
KAI_ASM_LABEL(label_79)  // Height 6: Multiply loop: Main loop head
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    sub x7, x7, #0x4
    cmp x7, #0x4
    ld1rqw { z0.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z1.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z2.s }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqw { z3.s }, p0/Z, [x4]
    ld1rqw { z4.s }, p0/Z, [x2]
    ld1rqw { z5.s }, p0/Z, [x1]
    add x4, x4, #0x10
    add x2, x2, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    add x1, x1, #0x10
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z28.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    fmla z29.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z30.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x21, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    fmla z31.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x21, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    fmla z28.s, z6.s, z5.s[1]
    ld1w { z6.s }, p5/Z, [x21, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    fmla z29.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x21, #7, MUL VL]
    addvl x21, x21, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z30.s, z6.s, z5.s[1]
    ld1w { z6.s }, p5/Z, [x21, #-8, MUL VL]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    fmla z31.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x21, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    fmla z28.s, z6.s, z5.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    fmla z29.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z30.s, z6.s, z5.s[2]
    ld1w { z6.s }, p5/Z, [x21, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    fmla z31.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x21, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    fmla z28.s, z6.s, z5.s[3]
    ld1w { z6.s }, p5/Z, [x21, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    fmla z29.s, z7.s, z5.s[3]
    ld1w { z7.s }, p5/Z, [x21, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z30.s, z6.s, z5.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    fmla z31.s, z7.s, z5.s[3]
    bgt label_79
KAI_ASM_LABEL(label_80)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x7
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    ld1rqw { z0.s }, p0/Z, [x8]
    ld1rqw { z1.s }, p0/Z, [x6]
    ld1rqw { z2.s }, p0/Z, [x5]
    ld1rqw { z3.s }, p0/Z, [x4]
    ld1rqw { z4.s }, p0/Z, [x2]
    ld1rqw { z5.s }, p0/Z, [x1]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z28.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    fmla z29.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z30.s, z6.s, z5.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    fmla z31.s, z7.s, z5.s[0]
    ble label_81
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    fmla z28.s, z6.s, z5.s[1]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    fmla z29.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[1]
    addvl x21, x21, #4
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z30.s, z6.s, z5.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    fmla z31.s, z7.s, z5.s[1]
    ble label_81
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    fmla z28.s, z6.s, z5.s[2]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    fmla z29.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    addvl x21, x21, #4
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z30.s, z6.s, z5.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    fmla z31.s, z7.s, z5.s[2]
    ble label_81
    ld1w { z6.s }, p5/Z, [x21]
    ld1w { z7.s }, p5/Z, [x21, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    fmla z28.s, z6.s, z5.s[3]
    ld1w { z6.s }, p5/Z, [x21, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    fmla z29.s, z7.s, z5.s[3]
    ld1w { z7.s }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z30.s, z6.s, z5.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    fmla z31.s, z7.s, z5.s[3]
KAI_ASM_LABEL(label_81)  // Height 6: Multiply loop: multiply skip
    tbz x25, #3, label_82
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_77
KAI_ASM_LABEL(label_82)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x25, #1, label_83
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d441)  // ld1rw { z1.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d420)  // ld1rw { z0.s }, p5/Z, [x1]
    fmin z8.s, p5/M, z8.s, z1.s
    fmin z9.s, p5/M, z9.s, z1.s
    fmin z10.s, p5/M, z10.s, z1.s
    fmin z11.s, p5/M, z11.s, z1.s
    fmin z12.s, p5/M, z12.s, z1.s
    fmin z13.s, p5/M, z13.s, z1.s
    fmin z14.s, p5/M, z14.s, z1.s
    fmin z15.s, p5/M, z15.s, z1.s
    fmin z16.s, p5/M, z16.s, z1.s
    fmin z17.s, p5/M, z17.s, z1.s
    fmin z18.s, p5/M, z18.s, z1.s
    fmin z19.s, p5/M, z19.s, z1.s
    fmin z20.s, p5/M, z20.s, z1.s
    fmin z21.s, p5/M, z21.s, z1.s
    fmin z22.s, p5/M, z22.s, z1.s
    fmin z23.s, p5/M, z23.s, z1.s
    fmin z24.s, p5/M, z24.s, z1.s
    fmin z25.s, p5/M, z25.s, z1.s
    fmin z26.s, p5/M, z26.s, z1.s
    fmin z27.s, p5/M, z27.s, z1.s
    fmin z28.s, p5/M, z28.s, z1.s
    fmin z29.s, p5/M, z29.s, z1.s
    fmin z30.s, p5/M, z30.s, z1.s
    fmin z31.s, p5/M, z31.s, z1.s
    fmax z8.s, p5/M, z8.s, z0.s
    fmax z9.s, p5/M, z9.s, z0.s
    fmax z10.s, p5/M, z10.s, z0.s
    fmax z11.s, p5/M, z11.s, z0.s
    fmax z12.s, p5/M, z12.s, z0.s
    fmax z13.s, p5/M, z13.s, z0.s
    fmax z14.s, p5/M, z14.s, z0.s
    fmax z15.s, p5/M, z15.s, z0.s
    fmax z16.s, p5/M, z16.s, z0.s
    fmax z17.s, p5/M, z17.s, z0.s
    fmax z18.s, p5/M, z18.s, z0.s
    fmax z19.s, p5/M, z19.s, z0.s
    fmax z20.s, p5/M, z20.s, z0.s
    fmax z21.s, p5/M, z21.s, z0.s
    fmax z22.s, p5/M, z22.s, z0.s
    fmax z23.s, p5/M, z23.s, z0.s
    fmax z24.s, p5/M, z24.s, z0.s
    fmax z25.s, p5/M, z25.s, z0.s
    fmax z26.s, p5/M, z26.s, z0.s
    fmax z27.s, p5/M, z27.s, z0.s
    fmax z28.s, p5/M, z28.s, z0.s
    fmax z29.s, p5/M, z29.s, z0.s
    fmax z30.s, p5/M, z30.s, z0.s
    fmax z31.s, p5/M, z31.s, z0.s
KAI_ASM_LABEL(label_83)  // Height 6: No activation
    st1w { z8.s }, p4, [x23]
    st1w { z9.s }, p3, [x23, #1, MUL VL]
    st1w { z10.s }, p2, [x23, #2, MUL VL]
    st1w { z11.s }, p1, [x23, #3, MUL VL]
    addvl x23, x23, #4
    st1w { z12.s }, p4, [x7]
    st1w { z13.s }, p3, [x7, #1, MUL VL]
    st1w { z14.s }, p2, [x7, #2, MUL VL]
    st1w { z15.s }, p1, [x7, #3, MUL VL]
    st1w { z16.s }, p4, [x6]
    st1w { z17.s }, p3, [x6, #1, MUL VL]
    st1w { z18.s }, p2, [x6, #2, MUL VL]
    st1w { z19.s }, p1, [x6, #3, MUL VL]
    st1w { z20.s }, p4, [x5]
    st1w { z21.s }, p3, [x5, #1, MUL VL]
    st1w { z22.s }, p2, [x5, #2, MUL VL]
    st1w { z23.s }, p1, [x5, #3, MUL VL]
    st1w { z24.s }, p4, [x4]
    st1w { z25.s }, p3, [x4, #1, MUL VL]
    st1w { z26.s }, p2, [x4, #2, MUL VL]
    st1w { z27.s }, p1, [x4, #3, MUL VL]
    st1w { z28.s }, p4, [x3]
    st1w { z29.s }, p3, [x3, #1, MUL VL]
    st1w { z30.s }, p2, [x3, #2, MUL VL]
    st1w { z31.s }, p1, [x3, #3, MUL VL]
    decw x22, ALL, MUL #4
    cmp x22, XZR
    bgt label_72
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_85)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_fp32_mla_6x4VL)

    KAI_ASM_END
