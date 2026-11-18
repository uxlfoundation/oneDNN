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

    KAI_ASM_CODE(sve_hybrid_fp32bf16fp32_mmla_4x6VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_fp32bf16fp32_mmla_4x6VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_fp32bf16fp32_mmla_4x6VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_fp32bf16fp32_mmla_4x6VL)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ptrue p7.b
    mov x25, #0
    ldr x4, [x0, #0x60]
    ldr w1, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x24, [x0, #0]
    cmp x4, #0
    cset x23, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x23, x23, x1, LSL #1
    cset x1, NE
    orr x23, x23, x2, LSL #1
    orr x23, x23, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x24, x25
    cmp x1, #0x4
    bge label_43
    cmp x1, #0x2
    bgt label_29
    beq label_15
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    whilelt p6.s, x1, x20
    incw x1
    whilelt p5.s, x1, x20
    incw x1
    whilelt p4.s, x1, x20
    incw x1
    whilelt p3.s, x1, x20
    incw x1
    whilelt p2.s, x1, x20
    incw x1
    whilelt p1.s, x1, x20
    cbz x7, label_3
    ld1w { z8.s }, p6/Z, [x7]
    ld1w { z9.s }, p5/Z, [x7, #1, MUL VL]
    ld1w { z10.s }, p4/Z, [x7, #2, MUL VL]
    ld1w { z11.s }, p3/Z, [x7, #3, MUL VL]
    ld1w { z12.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z13.s }, p1/Z, [x7, #5, MUL VL]
    addvl x7, x7, #6
    zip2 z14.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z15.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z16.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z17.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    zip2 z18.d, z12.d, z12.d
    zip1 z12.d, z12.d, z12.d
    zip2 z19.d, z13.d, z13.d
    zip1 z13.d, z13.d, z13.d
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x23, #0, label_4
    ld1w { z9.s }, p6/Z, [x21]
    ld1w { z10.s }, p5/Z, [x21, #1, MUL VL]
    ld1w { z11.s }, p4/Z, [x21, #2, MUL VL]
    ld1w { z12.s }, p3/Z, [x21, #3, MUL VL]
    ld1w { z13.s }, p2/Z, [x21, #4, MUL VL]
    ld1w { z20.s }, p1/Z, [x21, #5, MUL VL]
    zip1 z8.d, z9.d, z14.d
    zip2 z14.d, z9.d, z14.d
    zip1 z9.d, z10.d, z15.d
    zip2 z15.d, z10.d, z15.d
    zip1 z10.d, z11.d, z16.d
    zip2 z16.d, z11.d, z16.d
    zip1 z11.d, z12.d, z17.d
    zip2 z17.d, z12.d, z17.d
    zip1 z12.d, z13.d, z18.d
    zip2 z18.d, z13.d, z18.d
    zip1 z13.d, z20.d, z19.d
    zip2 z19.d, z20.d, z19.d
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
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
KAI_ASM_LABEL(label_5)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_6
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    b label_8
KAI_ASM_LABEL(label_6)  // Height 1: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_7)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    cbnz x22, label_8
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x5, #0x4
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    sub x5, x5, #0x4
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    cmp x5, #0x4
    ld1rqw { z0.s }, p0/Z, [x6]
    add x6, x6, #0x10
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    uzp1 z0.h, z0.h, z0.h
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    ld1rqw { z0.s }, p0/Z, [x6]
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    uzp1 z0.h, z0.h, z0.h
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    tbz x23, #3, label_12
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_7
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    uzp1 z8.d, z8.d, z14.d
    uzp1 z9.d, z9.d, z15.d
    uzp1 z10.d, z10.d, z16.d
    uzp1 z11.d, z11.d, z17.d
    uzp1 z12.d, z12.d, z18.d
    uzp1 z13.d, z13.d, z19.d
    tbz x23, #1, label_13
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540dc51)  // ld1rw { z17.s }, p7/Z, [x2]
    KAI_ASM_INST(0x8540dc30)  // ld1rw { z16.s }, p7/Z, [x1]
    fmin z8.s, p7/M, z8.s, z17.s
    fmin z9.s, p7/M, z9.s, z17.s
    fmin z10.s, p7/M, z10.s, z17.s
    fmin z11.s, p7/M, z11.s, z17.s
    fmin z12.s, p7/M, z12.s, z17.s
    fmin z13.s, p7/M, z13.s, z17.s
    fmax z8.s, p7/M, z8.s, z16.s
    fmax z9.s, p7/M, z9.s, z16.s
    fmax z10.s, p7/M, z10.s, z16.s
    fmax z11.s, p7/M, z11.s, z16.s
    fmax z12.s, p7/M, z12.s, z16.s
    fmax z13.s, p7/M, z13.s, z16.s
KAI_ASM_LABEL(label_13)  // Height 1: No activation
    st1w { z8.s }, p6, [x21]
    st1w { z9.s }, p5, [x21, #1, MUL VL]
    st1w { z10.s }, p4, [x21, #2, MUL VL]
    st1w { z11.s }, p3, [x21, #3, MUL VL]
    st1w { z12.s }, p2, [x21, #4, MUL VL]
    st1w { z13.s }, p1, [x21, #5, MUL VL]
    addvl x21, x21, #6
    decw x20, ALL, MUL #6
    cmp x20, XZR
    bgt label_2
    b label_57
KAI_ASM_LABEL(label_15)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_16)  // Height 2: Column loop
    mov x1, #0
    whilelt p6.s, x1, x20
    incw x1
    whilelt p5.s, x1, x20
    incw x1
    whilelt p4.s, x1, x20
    incw x1
    whilelt p3.s, x1, x20
    incw x1
    whilelt p2.s, x1, x20
    incw x1
    whilelt p1.s, x1, x20
    cbz x7, label_17
    ld1w { z8.s }, p6/Z, [x7]
    ld1w { z9.s }, p5/Z, [x7, #1, MUL VL]
    ld1w { z10.s }, p4/Z, [x7, #2, MUL VL]
    ld1w { z11.s }, p3/Z, [x7, #3, MUL VL]
    ld1w { z12.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z13.s }, p1/Z, [x7, #5, MUL VL]
    addvl x7, x7, #6
    zip2 z14.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z15.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z16.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z17.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    zip2 z18.d, z12.d, z12.d
    zip1 z12.d, z12.d, z12.d
    zip2 z19.d, z13.d, z13.d
    zip1 z13.d, z13.d, z13.d
    b label_19
KAI_ASM_LABEL(label_17)  // Height 2: no bias
    tbz x23, #0, label_18
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p6/Z, [x21]
    ld1w { z10.s }, p5/Z, [x21, #1, MUL VL]
    ld1w { z11.s }, p4/Z, [x21, #2, MUL VL]
    ld1w { z12.s }, p3/Z, [x21, #3, MUL VL]
    ld1w { z13.s }, p2/Z, [x21, #4, MUL VL]
    add x1, x21, x1
    ld1w { z20.s }, p1/Z, [x21, #5, MUL VL]
    ld1w { z14.s }, p6/Z, [x1]
    ld1w { z15.s }, p5/Z, [x1, #1, MUL VL]
    ld1w { z16.s }, p4/Z, [x1, #2, MUL VL]
    ld1w { z17.s }, p3/Z, [x1, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #4, MUL VL]
    ld1w { z19.s }, p1/Z, [x1, #5, MUL VL]
    zip1 z8.d, z9.d, z14.d
    zip2 z14.d, z9.d, z14.d
    zip1 z9.d, z10.d, z15.d
    zip2 z15.d, z10.d, z15.d
    zip1 z10.d, z11.d, z16.d
    zip2 z16.d, z11.d, z16.d
    zip1 z11.d, z12.d, z17.d
    zip2 z17.d, z12.d, z17.d
    zip1 z12.d, z13.d, z18.d
    zip2 z18.d, z13.d, z18.d
    zip1 z13.d, z20.d, z19.d
    zip2 z19.d, z20.d, z19.d
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
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
KAI_ASM_LABEL(label_19)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_20
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    b label_22
KAI_ASM_LABEL(label_20)  // Height 2: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_21)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    cbnz x22, label_22
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_22)  // Height 2: input setup done
    cmp x5, #0x4
    ble label_24
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    sub x5, x5, #0x4
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    cmp x5, #0x4
    ld1rqw { z0.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z1.s }, p0/Z, [x4]
    add x4, x4, #0x10
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    trn1 z0.d, z0.d, z1.d
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    bgt label_23
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    ld1rqw { z0.s }, p0/Z, [x6]
    ld1rqw { z1.s }, p0/Z, [x4]
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    trn1 z0.d, z0.d, z1.d
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    tbz x23, #3, label_26
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_21
KAI_ASM_LABEL(label_26)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z4.d, z8.d, z14.d
    uzp2 z8.d, z8.d, z14.d
    uzp1 z14.d, z9.d, z15.d
    uzp2 z9.d, z9.d, z15.d
    uzp1 z15.d, z10.d, z16.d
    uzp2 z10.d, z10.d, z16.d
    uzp1 z16.d, z11.d, z17.d
    uzp2 z11.d, z11.d, z17.d
    add x5, x21, x1
    uzp1 z17.d, z12.d, z18.d
    uzp2 z12.d, z12.d, z18.d
    uzp1 z18.d, z13.d, z19.d
    uzp2 z13.d, z13.d, z19.d
    tbz x23, #1, label_27
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540dc54)  // ld1rw { z20.s }, p7/Z, [x2]
    KAI_ASM_INST(0x8540dc33)  // ld1rw { z19.s }, p7/Z, [x1]
    fmin z4.s, p7/M, z4.s, z20.s
    fmin z14.s, p7/M, z14.s, z20.s
    fmin z15.s, p7/M, z15.s, z20.s
    fmin z16.s, p7/M, z16.s, z20.s
    fmin z17.s, p7/M, z17.s, z20.s
    fmin z18.s, p7/M, z18.s, z20.s
    fmin z8.s, p7/M, z8.s, z20.s
    fmin z9.s, p7/M, z9.s, z20.s
    fmin z10.s, p7/M, z10.s, z20.s
    fmin z11.s, p7/M, z11.s, z20.s
    fmin z12.s, p7/M, z12.s, z20.s
    fmin z13.s, p7/M, z13.s, z20.s
    fmax z4.s, p7/M, z4.s, z19.s
    fmax z14.s, p7/M, z14.s, z19.s
    fmax z15.s, p7/M, z15.s, z19.s
    fmax z16.s, p7/M, z16.s, z19.s
    fmax z17.s, p7/M, z17.s, z19.s
    fmax z18.s, p7/M, z18.s, z19.s
    fmax z8.s, p7/M, z8.s, z19.s
    fmax z9.s, p7/M, z9.s, z19.s
    fmax z10.s, p7/M, z10.s, z19.s
    fmax z11.s, p7/M, z11.s, z19.s
    fmax z12.s, p7/M, z12.s, z19.s
    fmax z13.s, p7/M, z13.s, z19.s
KAI_ASM_LABEL(label_27)  // Height 2: No activation
    st1w { z4.s }, p6, [x21]
    st1w { z14.s }, p5, [x21, #1, MUL VL]
    st1w { z15.s }, p4, [x21, #2, MUL VL]
    st1w { z16.s }, p3, [x21, #3, MUL VL]
    st1w { z17.s }, p2, [x21, #4, MUL VL]
    st1w { z18.s }, p1, [x21, #5, MUL VL]
    addvl x21, x21, #6
    st1w { z8.s }, p6, [x5]
    st1w { z9.s }, p5, [x5, #1, MUL VL]
    st1w { z10.s }, p4, [x5, #2, MUL VL]
    st1w { z11.s }, p3, [x5, #3, MUL VL]
    st1w { z12.s }, p2, [x5, #4, MUL VL]
    st1w { z13.s }, p1, [x5, #5, MUL VL]
    decw x20, ALL, MUL #6
    cmp x20, XZR
    bgt label_16
    b label_57
KAI_ASM_LABEL(label_29)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_30)  // Height 3: Column loop
    mov x1, #0
    whilelt p6.s, x1, x20
    incw x1
    whilelt p5.s, x1, x20
    incw x1
    whilelt p4.s, x1, x20
    incw x1
    whilelt p3.s, x1, x20
    incw x1
    whilelt p2.s, x1, x20
    incw x1
    whilelt p1.s, x1, x20
    cbz x7, label_31
    ld1w { z8.s }, p6/Z, [x7]
    ld1w { z9.s }, p5/Z, [x7, #1, MUL VL]
    ld1w { z10.s }, p4/Z, [x7, #2, MUL VL]
    ld1w { z11.s }, p3/Z, [x7, #3, MUL VL]
    ld1w { z12.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z13.s }, p1/Z, [x7, #5, MUL VL]
    addvl x7, x7, #6
    zip2 z14.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z15.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z16.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z17.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    zip2 z18.d, z12.d, z12.d
    zip1 z12.d, z12.d, z12.d
    zip2 z19.d, z13.d, z13.d
    zip1 z13.d, z13.d, z13.d
    mov z20.d, z8.d
    mov z26.d, z14.d
    mov z21.d, z9.d
    mov z27.d, z15.d
    mov z22.d, z10.d
    mov z28.d, z16.d
    mov z23.d, z11.d
    mov z29.d, z17.d
    mov z24.d, z12.d
    mov z30.d, z18.d
    mov z25.d, z13.d
    mov z31.d, z19.d
    b label_33
KAI_ASM_LABEL(label_31)  // Height 3: no bias
    tbz x23, #0, label_32
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p6/Z, [x21]
    ld1w { z10.s }, p5/Z, [x21, #1, MUL VL]
    ld1w { z11.s }, p4/Z, [x21, #2, MUL VL]
    ld1w { z12.s }, p3/Z, [x21, #3, MUL VL]
    ld1w { z13.s }, p2/Z, [x21, #4, MUL VL]
    add x2, x21, x1
    ld1w { z20.s }, p1/Z, [x21, #5, MUL VL]
    add x1, x2, x1
    ld1w { z14.s }, p6/Z, [x2]
    ld1w { z15.s }, p5/Z, [x2, #1, MUL VL]
    ld1w { z16.s }, p4/Z, [x2, #2, MUL VL]
    ld1w { z17.s }, p3/Z, [x2, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x2, #4, MUL VL]
    ld1w { z19.s }, p1/Z, [x2, #5, MUL VL]
    ld1w { z21.s }, p6/Z, [x1]
    zip1 z8.d, z9.d, z14.d
    zip2 z14.d, z9.d, z14.d
    ld1w { z22.s }, p5/Z, [x1, #1, MUL VL]
    ld1w { z23.s }, p4/Z, [x1, #2, MUL VL]
    zip1 z9.d, z10.d, z15.d
    zip2 z15.d, z10.d, z15.d
    ld1w { z24.s }, p3/Z, [x1, #3, MUL VL]
    ld1w { z25.s }, p2/Z, [x1, #4, MUL VL]
    zip1 z10.d, z11.d, z16.d
    zip2 z16.d, z11.d, z16.d
    ld1w { z4.s }, p1/Z, [x1, #5, MUL VL]
    zip1 z11.d, z12.d, z17.d
    zip2 z17.d, z12.d, z17.d
    zip1 z12.d, z13.d, z18.d
    zip2 z18.d, z13.d, z18.d
    zip1 z13.d, z20.d, z19.d
    zip2 z19.d, z20.d, z19.d
    zip1 z20.d, z21.d, z26.d
    zip2 z26.d, z21.d, z26.d
    zip1 z21.d, z22.d, z27.d
    zip2 z27.d, z22.d, z27.d
    zip1 z22.d, z23.d, z28.d
    zip2 z28.d, z23.d, z28.d
    zip1 z23.d, z24.d, z29.d
    zip2 z29.d, z24.d, z29.d
    zip1 z24.d, z25.d, z30.d
    zip2 z30.d, z25.d, z30.d
    zip1 z25.d, z4.d, z31.d
    zip2 z31.d, z4.d, z31.d
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
KAI_ASM_LABEL(label_33)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_34
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    b label_36
KAI_ASM_LABEL(label_34)  // Height 3: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_35)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    cbnz x22, label_36
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_36)  // Height 3: input setup done
    cmp x5, #0x4
    ble label_38
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    sub x5, x5, #0x4
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    cmp x5, #0x4
    ld1rqw { z0.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z1.s }, p0/Z, [x4]
    add x4, x4, #0x10
    ld1rqw { z2.s }, p0/Z, [x2]
    add x2, x2, #0x10
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    KAI_ASM_INST(0x658abc42)  // bfcvt z2.h, p7/M, z2.s
    trn1 z0.d, z0.d, z1.d
    uzp1 z2.h, z2.h, z2.h
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e45a)  // bfmmla z26.s, z2.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e45b)  // bfmmla z27.s, z2.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45c)  // bfmmla z28.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45d)  // bfmmla z29.s, z2.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e458)  // bfmmla z24.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45e)  // bfmmla z30.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e459)  // bfmmla z25.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45f)  // bfmmla z31.s, z2.h, z7.h
    bgt label_37
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    ld1rqw { z0.s }, p0/Z, [x6]
    ld1rqw { z1.s }, p0/Z, [x4]
    ld1rqw { z2.s }, p0/Z, [x2]
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    KAI_ASM_INST(0x658abc42)  // bfcvt z2.h, p7/M, z2.s
    trn1 z0.d, z0.d, z1.d
    uzp1 z2.h, z2.h, z2.h
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e45a)  // bfmmla z26.s, z2.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e45b)  // bfmmla z27.s, z2.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45c)  // bfmmla z28.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45d)  // bfmmla z29.s, z2.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e458)  // bfmmla z24.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45e)  // bfmmla z30.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e459)  // bfmmla z25.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45f)  // bfmmla z31.s, z2.h, z7.h
    tbz x23, #3, label_40
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_35
KAI_ASM_LABEL(label_40)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z4.d, z8.d, z14.d
    uzp2 z8.d, z8.d, z14.d
    uzp1 z14.d, z9.d, z15.d
    uzp2 z9.d, z9.d, z15.d
    uzp1 z15.d, z10.d, z16.d
    uzp2 z10.d, z10.d, z16.d
    add x5, x21, x1
    uzp1 z16.d, z11.d, z17.d
    uzp2 z11.d, z11.d, z17.d
    add x4, x5, x1
    uzp1 z17.d, z12.d, z18.d
    uzp2 z12.d, z12.d, z18.d
    uzp1 z18.d, z13.d, z19.d
    uzp2 z13.d, z13.d, z19.d
    uzp1 z20.d, z20.d, z26.d
    uzp1 z21.d, z21.d, z27.d
    uzp1 z22.d, z22.d, z28.d
    uzp1 z23.d, z23.d, z29.d
    uzp1 z24.d, z24.d, z30.d
    uzp1 z25.d, z25.d, z31.d
    tbz x23, #1, label_41
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540dc5a)  // ld1rw { z26.s }, p7/Z, [x2]
    KAI_ASM_INST(0x8540dc33)  // ld1rw { z19.s }, p7/Z, [x1]
    fmin z4.s, p7/M, z4.s, z26.s
    fmin z14.s, p7/M, z14.s, z26.s
    fmin z15.s, p7/M, z15.s, z26.s
    fmin z16.s, p7/M, z16.s, z26.s
    fmin z17.s, p7/M, z17.s, z26.s
    fmin z18.s, p7/M, z18.s, z26.s
    fmin z8.s, p7/M, z8.s, z26.s
    fmin z9.s, p7/M, z9.s, z26.s
    fmin z10.s, p7/M, z10.s, z26.s
    fmin z11.s, p7/M, z11.s, z26.s
    fmin z12.s, p7/M, z12.s, z26.s
    fmin z13.s, p7/M, z13.s, z26.s
    fmin z20.s, p7/M, z20.s, z26.s
    fmin z21.s, p7/M, z21.s, z26.s
    fmin z22.s, p7/M, z22.s, z26.s
    fmin z23.s, p7/M, z23.s, z26.s
    fmin z24.s, p7/M, z24.s, z26.s
    fmin z25.s, p7/M, z25.s, z26.s
    fmax z4.s, p7/M, z4.s, z19.s
    fmax z14.s, p7/M, z14.s, z19.s
    fmax z15.s, p7/M, z15.s, z19.s
    fmax z16.s, p7/M, z16.s, z19.s
    fmax z17.s, p7/M, z17.s, z19.s
    fmax z18.s, p7/M, z18.s, z19.s
    fmax z8.s, p7/M, z8.s, z19.s
    fmax z9.s, p7/M, z9.s, z19.s
    fmax z10.s, p7/M, z10.s, z19.s
    fmax z11.s, p7/M, z11.s, z19.s
    fmax z12.s, p7/M, z12.s, z19.s
    fmax z13.s, p7/M, z13.s, z19.s
    fmax z20.s, p7/M, z20.s, z19.s
    fmax z21.s, p7/M, z21.s, z19.s
    fmax z22.s, p7/M, z22.s, z19.s
    fmax z23.s, p7/M, z23.s, z19.s
    fmax z24.s, p7/M, z24.s, z19.s
    fmax z25.s, p7/M, z25.s, z19.s
KAI_ASM_LABEL(label_41)  // Height 3: No activation
    st1w { z4.s }, p6, [x21]
    st1w { z14.s }, p5, [x21, #1, MUL VL]
    st1w { z15.s }, p4, [x21, #2, MUL VL]
    st1w { z16.s }, p3, [x21, #3, MUL VL]
    st1w { z17.s }, p2, [x21, #4, MUL VL]
    st1w { z18.s }, p1, [x21, #5, MUL VL]
    addvl x21, x21, #6
    st1w { z8.s }, p6, [x5]
    st1w { z9.s }, p5, [x5, #1, MUL VL]
    st1w { z10.s }, p4, [x5, #2, MUL VL]
    st1w { z11.s }, p3, [x5, #3, MUL VL]
    st1w { z12.s }, p2, [x5, #4, MUL VL]
    st1w { z13.s }, p1, [x5, #5, MUL VL]
    st1w { z20.s }, p6, [x4]
    st1w { z21.s }, p5, [x4, #1, MUL VL]
    st1w { z22.s }, p4, [x4, #2, MUL VL]
    st1w { z23.s }, p3, [x4, #3, MUL VL]
    st1w { z24.s }, p2, [x4, #4, MUL VL]
    st1w { z25.s }, p1, [x4, #5, MUL VL]
    decw x20, ALL, MUL #6
    cmp x20, XZR
    bgt label_30
    b label_57
KAI_ASM_LABEL(label_43)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_44)  // Height 4: Column loop
    mov x1, #0
    whilelt p6.s, x1, x20
    incw x1
    whilelt p5.s, x1, x20
    incw x1
    whilelt p4.s, x1, x20
    incw x1
    whilelt p3.s, x1, x20
    incw x1
    whilelt p2.s, x1, x20
    incw x1
    whilelt p1.s, x1, x20
    cbz x7, label_45
    ld1w { z8.s }, p6/Z, [x7]
    ld1w { z9.s }, p5/Z, [x7, #1, MUL VL]
    ld1w { z10.s }, p4/Z, [x7, #2, MUL VL]
    ld1w { z11.s }, p3/Z, [x7, #3, MUL VL]
    ld1w { z12.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z13.s }, p1/Z, [x7, #5, MUL VL]
    addvl x7, x7, #6
    zip2 z14.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z15.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z16.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z17.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    zip2 z18.d, z12.d, z12.d
    zip1 z12.d, z12.d, z12.d
    zip2 z19.d, z13.d, z13.d
    zip1 z13.d, z13.d, z13.d
    mov z20.d, z8.d
    mov z26.d, z14.d
    mov z21.d, z9.d
    mov z27.d, z15.d
    mov z22.d, z10.d
    mov z28.d, z16.d
    mov z23.d, z11.d
    mov z29.d, z17.d
    mov z24.d, z12.d
    mov z30.d, z18.d
    mov z25.d, z13.d
    mov z31.d, z19.d
    b label_47
KAI_ASM_LABEL(label_45)  // Height 4: no bias
    tbz x23, #0, label_46
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p6/Z, [x21]
    ld1w { z10.s }, p5/Z, [x21, #1, MUL VL]
    ld1w { z11.s }, p4/Z, [x21, #2, MUL VL]
    ld1w { z12.s }, p3/Z, [x21, #3, MUL VL]
    ld1w { z13.s }, p2/Z, [x21, #4, MUL VL]
    add x3, x21, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z20.s }, p1/Z, [x21, #5, MUL VL]
    ld1w { z14.s }, p6/Z, [x3]
    ld1w { z15.s }, p5/Z, [x3, #1, MUL VL]
    ld1w { z16.s }, p4/Z, [x3, #2, MUL VL]
    ld1w { z17.s }, p3/Z, [x3, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x3, #4, MUL VL]
    ld1w { z19.s }, p1/Z, [x3, #5, MUL VL]
    ld1w { z21.s }, p6/Z, [x2]
    zip1 z8.d, z9.d, z14.d
    zip2 z14.d, z9.d, z14.d
    ld1w { z22.s }, p5/Z, [x2, #1, MUL VL]
    ld1w { z23.s }, p4/Z, [x2, #2, MUL VL]
    zip1 z9.d, z10.d, z15.d
    zip2 z15.d, z10.d, z15.d
    ld1w { z24.s }, p3/Z, [x2, #3, MUL VL]
    ld1w { z25.s }, p2/Z, [x2, #4, MUL VL]
    zip1 z10.d, z11.d, z16.d
    zip2 z16.d, z11.d, z16.d
    ld1w { z4.s }, p1/Z, [x2, #5, MUL VL]
    ld1w { z26.s }, p6/Z, [x1]
    zip1 z11.d, z12.d, z17.d
    zip2 z17.d, z12.d, z17.d
    ld1w { z27.s }, p5/Z, [x1, #1, MUL VL]
    ld1w { z28.s }, p4/Z, [x1, #2, MUL VL]
    zip1 z12.d, z13.d, z18.d
    zip2 z18.d, z13.d, z18.d
    ld1w { z29.s }, p3/Z, [x1, #3, MUL VL]
    ld1w { z30.s }, p2/Z, [x1, #4, MUL VL]
    zip1 z13.d, z20.d, z19.d
    zip2 z19.d, z20.d, z19.d
    ld1w { z31.s }, p1/Z, [x1, #5, MUL VL]
    zip1 z20.d, z21.d, z26.d
    zip2 z26.d, z21.d, z26.d
    zip1 z21.d, z22.d, z27.d
    zip2 z27.d, z22.d, z27.d
    zip1 z22.d, z23.d, z28.d
    zip2 z28.d, z23.d, z28.d
    zip1 z23.d, z24.d, z29.d
    zip2 z29.d, z24.d, z29.d
    zip1 z24.d, z25.d, z30.d
    zip2 z30.d, z25.d, z30.d
    zip1 z25.d, z4.d, z31.d
    zip2 z31.d, z4.d, z31.d
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
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_47)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_48
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_50
KAI_ASM_LABEL(label_48)  // Height 4: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_49)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    ldr x1, [x1, #0x18]
    cbnz x22, label_50
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_50)  // Height 4: input setup done
    cmp x5, #0x4
    ble label_52
KAI_ASM_LABEL(label_51)  // Height 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    sub x5, x5, #0x4
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    cmp x5, #0x4
    ld1rqw { z0.s }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqw { z1.s }, p0/Z, [x4]
    add x4, x4, #0x10
    ld1rqw { z2.s }, p0/Z, [x2]
    ld1rqw { z3.s }, p0/Z, [x1]
    add x2, x2, #0x10
    add x1, x1, #0x10
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    KAI_ASM_INST(0x658abc42)  // bfcvt z2.h, p7/M, z2.s
    KAI_ASM_INST(0x658abc63)  // bfcvt z3.h, p7/M, z3.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    uzp1 z2.h, z2.h, z2.h
    uzp1 z3.h, z3.h, z3.h
    trn1 z0.d, z0.d, z1.d
    trn1 z2.d, z2.d, z3.d
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45a)  // bfmmla z26.s, z2.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e45b)  // bfmmla z27.s, z2.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45c)  // bfmmla z28.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    KAI_ASM_INST(0x6467e45d)  // bfmmla z29.s, z2.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e458)  // bfmmla z24.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45e)  // bfmmla z30.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e459)  // bfmmla z25.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45f)  // bfmmla z31.s, z2.h, z7.h
    bgt label_51
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x5
    ld1h { z4.h }, p7/Z, [x8]
    ld1h { z5.h }, p7/Z, [x8, #1, MUL VL]
    ld1h { z6.h }, p7/Z, [x8, #2, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #3, MUL VL]
    ld1rqw { z0.s }, p0/Z, [x6]
    ld1rqw { z1.s }, p0/Z, [x4]
    ld1rqw { z2.s }, p0/Z, [x2]
    ld1rqw { z3.s }, p0/Z, [x1]
    KAI_ASM_INST(0x658abc00)  // bfcvt z0.h, p7/M, z0.s
    KAI_ASM_INST(0x658abc21)  // bfcvt z1.h, p7/M, z1.s
    KAI_ASM_INST(0x658abc42)  // bfcvt z2.h, p7/M, z2.s
    KAI_ASM_INST(0x658abc63)  // bfcvt z3.h, p7/M, z3.s
    uzp1 z0.h, z0.h, z0.h
    uzp1 z1.h, z1.h, z1.h
    uzp1 z2.h, z2.h, z2.h
    uzp1 z3.h, z3.h, z3.h
    trn1 z0.d, z0.d, z1.d
    trn1 z2.d, z2.d, z3.d
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    ld1h { z4.h }, p7/Z, [x8, #4, MUL VL]
    KAI_ASM_INST(0x6465e40e)  // bfmmla z14.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45a)  // bfmmla z26.s, z2.h, z5.h
    ld1h { z5.h }, p7/Z, [x8, #5, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e40f)  // bfmmla z15.s, z0.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #6, MUL VL]
    KAI_ASM_INST(0x6467e45b)  // bfmmla z27.s, z2.h, z7.h
    ld1h { z7.h }, p7/Z, [x8, #7, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e410)  // bfmmla z16.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45c)  // bfmmla z28.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40b)  // bfmmla z11.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e411)  // bfmmla z17.s, z0.h, z7.h
    ld1h { z4.h }, p7/Z, [x8, #-8, MUL VL]
    ld1h { z5.h }, p7/Z, [x8, #-7, MUL VL]
    KAI_ASM_INST(0x6467e45d)  // bfmmla z29.s, z2.h, z7.h
    ld1h { z6.h }, p7/Z, [x8, #-6, MUL VL]
    ld1h { z7.h }, p7/Z, [x8, #-5, MUL VL]
    addvl x8, x8, #-4
    KAI_ASM_INST(0x6464e40c)  // bfmmla z12.s, z0.h, z4.h
    KAI_ASM_INST(0x6464e458)  // bfmmla z24.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e412)  // bfmmla z18.s, z0.h, z5.h
    KAI_ASM_INST(0x6465e45e)  // bfmmla z30.s, z2.h, z5.h
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e459)  // bfmmla z25.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e413)  // bfmmla z19.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e45f)  // bfmmla z31.s, z2.h, z7.h
    tbz x23, #3, label_54
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_49
KAI_ASM_LABEL(label_54)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z4.d, z8.d, z14.d
    uzp2 z8.d, z8.d, z14.d
    uzp1 z14.d, z9.d, z15.d
    uzp2 z9.d, z9.d, z15.d
    uzp1 z15.d, z10.d, z16.d
    uzp2 z10.d, z10.d, z16.d
    add x5, x21, x1
    add x4, x5, x1
    uzp1 z16.d, z11.d, z17.d
    uzp2 z11.d, z11.d, z17.d
    add x3, x4, x1
    uzp1 z17.d, z12.d, z18.d
    uzp2 z12.d, z12.d, z18.d
    uzp1 z18.d, z13.d, z19.d
    uzp2 z13.d, z13.d, z19.d
    uzp1 z19.d, z20.d, z26.d
    uzp2 z20.d, z20.d, z26.d
    uzp1 z26.d, z21.d, z27.d
    uzp2 z21.d, z21.d, z27.d
    uzp1 z27.d, z22.d, z28.d
    uzp2 z22.d, z22.d, z28.d
    uzp1 z28.d, z23.d, z29.d
    uzp2 z23.d, z23.d, z29.d
    uzp1 z29.d, z24.d, z30.d
    uzp2 z24.d, z24.d, z30.d
    uzp1 z30.d, z25.d, z31.d
    uzp2 z25.d, z25.d, z31.d
    tbz x23, #1, label_55
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540dc40)  // ld1rw { z0.s }, p7/Z, [x2]
    KAI_ASM_INST(0x8540dc3f)  // ld1rw { z31.s }, p7/Z, [x1]
    fmin z4.s, p7/M, z4.s, z0.s
    fmin z14.s, p7/M, z14.s, z0.s
    fmin z15.s, p7/M, z15.s, z0.s
    fmin z16.s, p7/M, z16.s, z0.s
    fmin z17.s, p7/M, z17.s, z0.s
    fmin z18.s, p7/M, z18.s, z0.s
    fmin z8.s, p7/M, z8.s, z0.s
    fmin z9.s, p7/M, z9.s, z0.s
    fmin z10.s, p7/M, z10.s, z0.s
    fmin z11.s, p7/M, z11.s, z0.s
    fmin z12.s, p7/M, z12.s, z0.s
    fmin z13.s, p7/M, z13.s, z0.s
    fmin z19.s, p7/M, z19.s, z0.s
    fmin z26.s, p7/M, z26.s, z0.s
    fmin z27.s, p7/M, z27.s, z0.s
    fmin z28.s, p7/M, z28.s, z0.s
    fmin z29.s, p7/M, z29.s, z0.s
    fmin z30.s, p7/M, z30.s, z0.s
    fmin z20.s, p7/M, z20.s, z0.s
    fmin z21.s, p7/M, z21.s, z0.s
    fmin z22.s, p7/M, z22.s, z0.s
    fmin z23.s, p7/M, z23.s, z0.s
    fmin z24.s, p7/M, z24.s, z0.s
    fmin z25.s, p7/M, z25.s, z0.s
    fmax z4.s, p7/M, z4.s, z31.s
    fmax z14.s, p7/M, z14.s, z31.s
    fmax z15.s, p7/M, z15.s, z31.s
    fmax z16.s, p7/M, z16.s, z31.s
    fmax z17.s, p7/M, z17.s, z31.s
    fmax z18.s, p7/M, z18.s, z31.s
    fmax z8.s, p7/M, z8.s, z31.s
    fmax z9.s, p7/M, z9.s, z31.s
    fmax z10.s, p7/M, z10.s, z31.s
    fmax z11.s, p7/M, z11.s, z31.s
    fmax z12.s, p7/M, z12.s, z31.s
    fmax z13.s, p7/M, z13.s, z31.s
    fmax z19.s, p7/M, z19.s, z31.s
    fmax z26.s, p7/M, z26.s, z31.s
    fmax z27.s, p7/M, z27.s, z31.s
    fmax z28.s, p7/M, z28.s, z31.s
    fmax z29.s, p7/M, z29.s, z31.s
    fmax z30.s, p7/M, z30.s, z31.s
    fmax z20.s, p7/M, z20.s, z31.s
    fmax z21.s, p7/M, z21.s, z31.s
    fmax z22.s, p7/M, z22.s, z31.s
    fmax z23.s, p7/M, z23.s, z31.s
    fmax z24.s, p7/M, z24.s, z31.s
    fmax z25.s, p7/M, z25.s, z31.s
KAI_ASM_LABEL(label_55)  // Height 4: No activation
    st1w { z4.s }, p6, [x21]
    st1w { z14.s }, p5, [x21, #1, MUL VL]
    st1w { z15.s }, p4, [x21, #2, MUL VL]
    st1w { z16.s }, p3, [x21, #3, MUL VL]
    st1w { z17.s }, p2, [x21, #4, MUL VL]
    st1w { z18.s }, p1, [x21, #5, MUL VL]
    addvl x21, x21, #6
    st1w { z8.s }, p6, [x5]
    st1w { z9.s }, p5, [x5, #1, MUL VL]
    st1w { z10.s }, p4, [x5, #2, MUL VL]
    st1w { z11.s }, p3, [x5, #3, MUL VL]
    st1w { z12.s }, p2, [x5, #4, MUL VL]
    st1w { z13.s }, p1, [x5, #5, MUL VL]
    st1w { z19.s }, p6, [x4]
    st1w { z26.s }, p5, [x4, #1, MUL VL]
    st1w { z27.s }, p4, [x4, #2, MUL VL]
    st1w { z28.s }, p3, [x4, #3, MUL VL]
    st1w { z29.s }, p2, [x4, #4, MUL VL]
    st1w { z30.s }, p1, [x4, #5, MUL VL]
    st1w { z20.s }, p6, [x3]
    st1w { z21.s }, p5, [x3, #1, MUL VL]
    st1w { z22.s }, p4, [x3, #2, MUL VL]
    st1w { z23.s }, p3, [x3, #3, MUL VL]
    st1w { z24.s }, p2, [x3, #4, MUL VL]
    st1w { z25.s }, p1, [x3, #5, MUL VL]
    decw x20, ALL, MUL #6
    cmp x20, XZR
    bgt label_44
    add x25, x25, #0x4
    cmp x25, x24
    blt label_1
KAI_ASM_LABEL(label_57)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_fp32bf16fp32_mmla_4x6VL)

    KAI_ASM_END
