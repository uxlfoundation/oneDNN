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

    KAI_ASM_CODE(sve_ffhybrid_bf16fp32_mmla_6x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffhybrid_bf16fp32_mmla_6x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffhybrid_bf16fp32_mmla_6x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffhybrid_bf16fp32_mmla_6x4VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ptrue p5.b
    mov x11, #0
    ldr x4, [x0, #0x60]
    ldr w1, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x10, [x0, #0]
    cmp x4, #0
    cset x9, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x9, x9, x1, LSL #1
    cset x1, NE
    orr x9, x9, x2, LSL #1
    orr x9, x9, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x10, x11
    cmp x1, #0x6
    bge label_76
    cmp x1, #0x4
    bgt label_61
    beq label_46
    cmp x1, #0x2
    bgt label_31
    beq label_16
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_3
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_3
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_3
    mov x22, x23
KAI_ASM_LABEL(label_3)  // Height 1: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_4
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: no bias
    tbz x9, #0, label_5
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    zip1 z8.d, z9.d, z12.d
    zip2 z12.d, z9.d, z12.d
    zip1 z9.d, z10.d, z13.d
    zip2 z13.d, z10.d, z13.d
    zip1 z10.d, z11.d, z14.d
    zip2 z14.d, z11.d, z14.d
    zip1 z11.d, z16.d, z15.d
    zip2 z15.d, z16.d, z15.d
    b label_6
KAI_ASM_LABEL(label_5)  // Height 1: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_6)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_7
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    b label_9
KAI_ASM_LABEL(label_7)  // Height 1: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_8)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    cbnz x28, label_9
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_9)  // Height 1: input setup done
    cmp x7, #0x8
    ble label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    subs x7, x7, #0x4
    addvl x23, x23, #2
    ld1rqh { z1.h }, p0/Z, [x8]
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    ble label_12
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: multiply skip
    tbz x9, #3, label_13
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_8
KAI_ASM_LABEL(label_13)  // Height 1: no string loop
    uzp1 z8.d, z8.d, z12.d
    uzp1 z9.d, z9.d, z13.d
    uzp1 z10.d, z10.d, z14.d
    uzp1 z11.d, z11.d, z15.d
    tbz x9, #1, label_14
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
KAI_ASM_LABEL(label_14)  // Height 1: No activation
    st1w { z8.s }, p4, [x27]
    st1w { z9.s }, p3, [x27, #1, MUL VL]
    st1w { z10.s }, p2, [x27, #2, MUL VL]
    st1w { z11.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_2
    b label_91
KAI_ASM_LABEL(label_16)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_17)  // Height 2: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_18
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_18
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_18
    mov x22, x23
KAI_ASM_LABEL(label_18)  // Height 2: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_19
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    zip2 z12.d, z8.d, z8.d
    zip1 z8.d, z8.d, z8.d
    zip2 z13.d, z9.d, z9.d
    zip1 z9.d, z9.d, z9.d
    zip2 z14.d, z10.d, z10.d
    zip1 z10.d, z10.d, z10.d
    zip2 z15.d, z11.d, z11.d
    zip1 z11.d, z11.d, z11.d
    b label_21
KAI_ASM_LABEL(label_19)  // Height 2: no bias
    tbz x9, #0, label_20
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    add x1, x27, x1
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
    b label_21
KAI_ASM_LABEL(label_20)  // Height 2: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_21)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_22
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    b label_24
KAI_ASM_LABEL(label_22)  // Height 2: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_23)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x28, label_24
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_24)  // Height 2: input setup done
    cmp x7, #0x8
    ble label_26
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z2.h }, p0/Z, [x6]
    add x6, x6, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    bgt label_25
KAI_ASM_LABEL(label_26)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    subs x7, x7, #0x4
    addvl x23, x23, #2
    ld1rqh { z1.h }, p0/Z, [x8]
    ld1rqh { z2.h }, p0/Z, [x6]
    trn1 z0.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    trn2 z1.d, z1.d, z2.d
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    ble label_27
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
KAI_ASM_LABEL(label_27)  // Height 2: Multiply loop: multiply skip
    tbz x9, #3, label_28
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_23
KAI_ASM_LABEL(label_28)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    add x7, x27, x1
    tbz x9, #1, label_29
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d451)  // ld1rw { z17.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d430)  // ld1rw { z16.s }, p5/Z, [x1]
    fmin z7.s, p5/M, z7.s, z17.s
    fmin z12.s, p5/M, z12.s, z17.s
    fmin z13.s, p5/M, z13.s, z17.s
    fmin z14.s, p5/M, z14.s, z17.s
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmax z7.s, p5/M, z7.s, z16.s
    fmax z12.s, p5/M, z12.s, z16.s
    fmax z13.s, p5/M, z13.s, z16.s
    fmax z14.s, p5/M, z14.s, z16.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
KAI_ASM_LABEL(label_29)  // Height 2: No activation
    st1w { z7.s }, p4, [x27]
    st1w { z12.s }, p3, [x27, #1, MUL VL]
    st1w { z13.s }, p2, [x27, #2, MUL VL]
    st1w { z14.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z8.s }, p4, [x7]
    st1w { z9.s }, p3, [x7, #1, MUL VL]
    st1w { z10.s }, p2, [x7, #2, MUL VL]
    st1w { z11.s }, p1, [x7, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_17
    b label_91
KAI_ASM_LABEL(label_31)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_32)  // Height 3: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_33
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_33
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_33
    mov x22, x23
KAI_ASM_LABEL(label_33)  // Height 3: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_34
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_36
KAI_ASM_LABEL(label_34)  // Height 3: no bias
    tbz x9, #0, label_35
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    add x2, x27, x1
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
    b label_36
KAI_ASM_LABEL(label_35)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_36)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_37
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_39
KAI_ASM_LABEL(label_37)  // Height 3: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_38)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x28, label_39
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_39)  // Height 3: input setup done
    cmp x7, #0x8
    ble label_41
KAI_ASM_LABEL(label_40)  // Height 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z2.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z3.h }, p0/Z, [x5]
    add x5, x5, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    bgt label_40
KAI_ASM_LABEL(label_41)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    subs x7, x7, #0x4
    addvl x23, x23, #2
    ld1rqh { z1.h }, p0/Z, [x8]
    ld1rqh { z2.h }, p0/Z, [x6]
    ld1rqh { z3.h }, p0/Z, [x5]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    ble label_42
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
KAI_ASM_LABEL(label_42)  // Height 3: Multiply loop: multiply skip
    tbz x9, #3, label_43
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_38
KAI_ASM_LABEL(label_43)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x7, x27, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    add x6, x7, x1
    uzp1 z16.d, z16.d, z20.d
    uzp1 z17.d, z17.d, z21.d
    uzp1 z18.d, z18.d, z22.d
    uzp1 z19.d, z19.d, z23.d
    tbz x9, #1, label_44
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d455)  // ld1rw { z21.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d434)  // ld1rw { z20.s }, p5/Z, [x1]
    fmin z7.s, p5/M, z7.s, z21.s
    fmin z12.s, p5/M, z12.s, z21.s
    fmin z13.s, p5/M, z13.s, z21.s
    fmin z14.s, p5/M, z14.s, z21.s
    fmin z8.s, p5/M, z8.s, z21.s
    fmin z9.s, p5/M, z9.s, z21.s
    fmin z10.s, p5/M, z10.s, z21.s
    fmin z11.s, p5/M, z11.s, z21.s
    fmin z16.s, p5/M, z16.s, z21.s
    fmin z17.s, p5/M, z17.s, z21.s
    fmin z18.s, p5/M, z18.s, z21.s
    fmin z19.s, p5/M, z19.s, z21.s
    fmax z7.s, p5/M, z7.s, z20.s
    fmax z12.s, p5/M, z12.s, z20.s
    fmax z13.s, p5/M, z13.s, z20.s
    fmax z14.s, p5/M, z14.s, z20.s
    fmax z8.s, p5/M, z8.s, z20.s
    fmax z9.s, p5/M, z9.s, z20.s
    fmax z10.s, p5/M, z10.s, z20.s
    fmax z11.s, p5/M, z11.s, z20.s
    fmax z16.s, p5/M, z16.s, z20.s
    fmax z17.s, p5/M, z17.s, z20.s
    fmax z18.s, p5/M, z18.s, z20.s
    fmax z19.s, p5/M, z19.s, z20.s
KAI_ASM_LABEL(label_44)  // Height 3: No activation
    st1w { z7.s }, p4, [x27]
    st1w { z12.s }, p3, [x27, #1, MUL VL]
    st1w { z13.s }, p2, [x27, #2, MUL VL]
    st1w { z14.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z8.s }, p4, [x7]
    st1w { z9.s }, p3, [x7, #1, MUL VL]
    st1w { z10.s }, p2, [x7, #2, MUL VL]
    st1w { z11.s }, p1, [x7, #3, MUL VL]
    st1w { z16.s }, p4, [x6]
    st1w { z17.s }, p3, [x6, #1, MUL VL]
    st1w { z18.s }, p2, [x6, #2, MUL VL]
    st1w { z19.s }, p1, [x6, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_32
    b label_91
KAI_ASM_LABEL(label_46)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_47)  // Height 4: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_48
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_48
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_48
    mov x22, x23
KAI_ASM_LABEL(label_48)  // Height 4: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_49
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_51
KAI_ASM_LABEL(label_49)  // Height 4: no bias
    tbz x9, #0, label_50
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    add x3, x27, x1
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
    b label_51
KAI_ASM_LABEL(label_50)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_51)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_52
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_54
KAI_ASM_LABEL(label_52)  // Height 4: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_53)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x28, label_54
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_54)  // Height 4: input setup done
    cmp x7, #0x8
    ble label_56
KAI_ASM_LABEL(label_55)  // Height 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z2.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z3.h }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqh { z4.h }, p0/Z, [x4]
    add x4, x4, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    bgt label_55
KAI_ASM_LABEL(label_56)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    subs x7, x7, #0x4
    addvl x23, x23, #2
    ld1rqh { z1.h }, p0/Z, [x8]
    ld1rqh { z2.h }, p0/Z, [x6]
    ld1rqh { z3.h }, p0/Z, [x5]
    ld1rqh { z4.h }, p0/Z, [x4]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    trn2 z3.d, z3.d, z4.d
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    ble label_57
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
KAI_ASM_LABEL(label_57)  // Height 4: Multiply loop: multiply skip
    tbz x9, #3, label_58
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_53
KAI_ASM_LABEL(label_58)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x7, x27, x1
    add x6, x7, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    add x5, x6, x1
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    tbz x9, #1, label_59
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d458)  // ld1rw { z24.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d437)  // ld1rw { z23.s }, p5/Z, [x1]
    fmin z7.s, p5/M, z7.s, z24.s
    fmin z12.s, p5/M, z12.s, z24.s
    fmin z13.s, p5/M, z13.s, z24.s
    fmin z14.s, p5/M, z14.s, z24.s
    fmin z8.s, p5/M, z8.s, z24.s
    fmin z9.s, p5/M, z9.s, z24.s
    fmin z10.s, p5/M, z10.s, z24.s
    fmin z11.s, p5/M, z11.s, z24.s
    fmin z15.s, p5/M, z15.s, z24.s
    fmin z20.s, p5/M, z20.s, z24.s
    fmin z21.s, p5/M, z21.s, z24.s
    fmin z22.s, p5/M, z22.s, z24.s
    fmin z16.s, p5/M, z16.s, z24.s
    fmin z17.s, p5/M, z17.s, z24.s
    fmin z18.s, p5/M, z18.s, z24.s
    fmin z19.s, p5/M, z19.s, z24.s
    fmax z7.s, p5/M, z7.s, z23.s
    fmax z12.s, p5/M, z12.s, z23.s
    fmax z13.s, p5/M, z13.s, z23.s
    fmax z14.s, p5/M, z14.s, z23.s
    fmax z8.s, p5/M, z8.s, z23.s
    fmax z9.s, p5/M, z9.s, z23.s
    fmax z10.s, p5/M, z10.s, z23.s
    fmax z11.s, p5/M, z11.s, z23.s
    fmax z15.s, p5/M, z15.s, z23.s
    fmax z20.s, p5/M, z20.s, z23.s
    fmax z21.s, p5/M, z21.s, z23.s
    fmax z22.s, p5/M, z22.s, z23.s
    fmax z16.s, p5/M, z16.s, z23.s
    fmax z17.s, p5/M, z17.s, z23.s
    fmax z18.s, p5/M, z18.s, z23.s
    fmax z19.s, p5/M, z19.s, z23.s
KAI_ASM_LABEL(label_59)  // Height 4: No activation
    st1w { z7.s }, p4, [x27]
    st1w { z12.s }, p3, [x27, #1, MUL VL]
    st1w { z13.s }, p2, [x27, #2, MUL VL]
    st1w { z14.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z8.s }, p4, [x7]
    st1w { z9.s }, p3, [x7, #1, MUL VL]
    st1w { z10.s }, p2, [x7, #2, MUL VL]
    st1w { z11.s }, p1, [x7, #3, MUL VL]
    st1w { z15.s }, p4, [x6]
    st1w { z20.s }, p3, [x6, #1, MUL VL]
    st1w { z21.s }, p2, [x6, #2, MUL VL]
    st1w { z22.s }, p1, [x6, #3, MUL VL]
    st1w { z16.s }, p4, [x5]
    st1w { z17.s }, p3, [x5, #1, MUL VL]
    st1w { z18.s }, p2, [x5, #2, MUL VL]
    st1w { z19.s }, p1, [x5, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_47
    b label_91
KAI_ASM_LABEL(label_61)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_62)  // Height 5: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_63
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_63
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_63
    mov x22, x23
KAI_ASM_LABEL(label_63)  // Height 5: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_64
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_66
KAI_ASM_LABEL(label_64)  // Height 5: no bias
    tbz x9, #0, label_65
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    add x4, x27, x1
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
    b label_66
KAI_ASM_LABEL(label_65)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_66)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_67
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_69
KAI_ASM_LABEL(label_67)  // Height 5: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_68)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    cbnz x28, label_69
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_69)  // Height 5: input setup done
    cmp x7, #0x8
    ble label_71
KAI_ASM_LABEL(label_70)  // Height 5: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z2.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z3.h }, p0/Z, [x5]
    ld1rqh { z4.h }, p0/Z, [x4]
    ld1rqh { z5.h }, p0/Z, [x2]
    add x5, x5, #0x10
    add x4, x4, #0x10
    add x2, x2, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e498)  // bfmmla z24.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49c)  // bfmmla z28.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e499)  // bfmmla z25.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49d)  // bfmmla z29.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49a)  // bfmmla z26.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49e)  // bfmmla z30.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49b)  // bfmmla z27.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49f)  // bfmmla z31.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b8)  // bfmmla z24.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bc)  // bfmmla z28.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b9)  // bfmmla z25.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bd)  // bfmmla z29.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4ba)  // bfmmla z26.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4be)  // bfmmla z30.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4bb)  // bfmmla z27.s, z5.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bf)  // bfmmla z31.s, z5.h, z6.h
    bgt label_70
KAI_ASM_LABEL(label_71)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    subs x7, x7, #0x4
    ld1rqh { z1.h }, p0/Z, [x8]
    ld1rqh { z2.h }, p0/Z, [x6]
    ld1rqh { z3.h }, p0/Z, [x5]
    ld1rqh { z4.h }, p0/Z, [x4]
    ld1rqh { z5.h }, p0/Z, [x2]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e498)  // bfmmla z24.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49c)  // bfmmla z28.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e499)  // bfmmla z25.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49d)  // bfmmla z29.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49a)  // bfmmla z26.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49e)  // bfmmla z30.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49b)  // bfmmla z27.s, z4.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49f)  // bfmmla z31.s, z4.h, z6.h
    ble label_72
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b8)  // bfmmla z24.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bc)  // bfmmla z28.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b9)  // bfmmla z25.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bd)  // bfmmla z29.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4ba)  // bfmmla z26.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4be)  // bfmmla z30.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4bb)  // bfmmla z27.s, z5.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bf)  // bfmmla z31.s, z5.h, z6.h
KAI_ASM_LABEL(label_72)  // Height 5: Multiply loop: multiply skip
    tbz x9, #3, label_73
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_68
KAI_ASM_LABEL(label_73)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    add x4, x5, x1
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    uzp1 z24.d, z24.d, z28.d
    uzp1 z25.d, z25.d, z29.d
    uzp1 z26.d, z26.d, z30.d
    uzp1 z27.d, z27.d, z31.d
    tbz x9, #1, label_74
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d45c)  // ld1rw { z28.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d437)  // ld1rw { z23.s }, p5/Z, [x1]
    fmin z7.s, p5/M, z7.s, z28.s
    fmin z12.s, p5/M, z12.s, z28.s
    fmin z13.s, p5/M, z13.s, z28.s
    fmin z14.s, p5/M, z14.s, z28.s
    fmin z8.s, p5/M, z8.s, z28.s
    fmin z9.s, p5/M, z9.s, z28.s
    fmin z10.s, p5/M, z10.s, z28.s
    fmin z11.s, p5/M, z11.s, z28.s
    fmin z15.s, p5/M, z15.s, z28.s
    fmin z20.s, p5/M, z20.s, z28.s
    fmin z21.s, p5/M, z21.s, z28.s
    fmin z22.s, p5/M, z22.s, z28.s
    fmin z16.s, p5/M, z16.s, z28.s
    fmin z17.s, p5/M, z17.s, z28.s
    fmin z18.s, p5/M, z18.s, z28.s
    fmin z19.s, p5/M, z19.s, z28.s
    fmin z24.s, p5/M, z24.s, z28.s
    fmin z25.s, p5/M, z25.s, z28.s
    fmin z26.s, p5/M, z26.s, z28.s
    fmin z27.s, p5/M, z27.s, z28.s
    fmax z7.s, p5/M, z7.s, z23.s
    fmax z12.s, p5/M, z12.s, z23.s
    fmax z13.s, p5/M, z13.s, z23.s
    fmax z14.s, p5/M, z14.s, z23.s
    fmax z8.s, p5/M, z8.s, z23.s
    fmax z9.s, p5/M, z9.s, z23.s
    fmax z10.s, p5/M, z10.s, z23.s
    fmax z11.s, p5/M, z11.s, z23.s
    fmax z15.s, p5/M, z15.s, z23.s
    fmax z20.s, p5/M, z20.s, z23.s
    fmax z21.s, p5/M, z21.s, z23.s
    fmax z22.s, p5/M, z22.s, z23.s
    fmax z16.s, p5/M, z16.s, z23.s
    fmax z17.s, p5/M, z17.s, z23.s
    fmax z18.s, p5/M, z18.s, z23.s
    fmax z19.s, p5/M, z19.s, z23.s
    fmax z24.s, p5/M, z24.s, z23.s
    fmax z25.s, p5/M, z25.s, z23.s
    fmax z26.s, p5/M, z26.s, z23.s
    fmax z27.s, p5/M, z27.s, z23.s
KAI_ASM_LABEL(label_74)  // Height 5: No activation
    st1w { z7.s }, p4, [x27]
    st1w { z12.s }, p3, [x27, #1, MUL VL]
    st1w { z13.s }, p2, [x27, #2, MUL VL]
    st1w { z14.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z8.s }, p4, [x7]
    st1w { z9.s }, p3, [x7, #1, MUL VL]
    st1w { z10.s }, p2, [x7, #2, MUL VL]
    st1w { z11.s }, p1, [x7, #3, MUL VL]
    st1w { z15.s }, p4, [x6]
    st1w { z20.s }, p3, [x6, #1, MUL VL]
    st1w { z21.s }, p2, [x6, #2, MUL VL]
    st1w { z22.s }, p1, [x6, #3, MUL VL]
    st1w { z16.s }, p4, [x5]
    st1w { z17.s }, p3, [x5, #1, MUL VL]
    st1w { z18.s }, p2, [x5, #2, MUL VL]
    st1w { z19.s }, p1, [x5, #3, MUL VL]
    st1w { z24.s }, p4, [x4]
    st1w { z25.s }, p3, [x4, #1, MUL VL]
    st1w { z26.s }, p2, [x4, #2, MUL VL]
    st1w { z27.s }, p1, [x4, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_62
    b label_91
KAI_ASM_LABEL(label_76)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_77)  // Height 6: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_78
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_78
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_78
    mov x22, x23
KAI_ASM_LABEL(label_78)  // Height 6: B setup done
    mov x1, #0
    whilelt p4.s, x1, x26
    incw x1
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    cbz x24, label_79
    ld1w { z8.s }, p4/Z, [x24]
    ld1w { z9.s }, p3/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_81
KAI_ASM_LABEL(label_79)  // Height 6: no bias
    tbz x9, #0, label_80
    ldr x1, [x0, #0x58]
    ld1w { z9.s }, p4/Z, [x27]
    ld1w { z10.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z11.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x27, #3, MUL VL]
    add x5, x27, x1
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
    b label_81
KAI_ASM_LABEL(label_80)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_81)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_82
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_84
KAI_ASM_LABEL(label_82)  // Height 6: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_83)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    ldr x1, [x1, #0x28]
    cbnz x28, label_84
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_84)  // Height 6: input setup done
    cmp x7, #0x8
    ble label_86
KAI_ASM_LABEL(label_85)  // Height 6: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z1.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z2.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z3.h }, p0/Z, [x5]
    add x5, x5, #0x10
    ld1rqh { z4.h }, p0/Z, [x4]
    ld1rqh { z5.h }, p0/Z, [x2]
    ld1rqh { z6.h }, p0/Z, [x1]
    add x4, x4, #0x10
    add x2, x2, #0x10
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    add x1, x1, #0x10
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e498)  // bfmmla z24.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49c)  // bfmmla z28.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e499)  // bfmmla z25.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49d)  // bfmmla z29.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49a)  // bfmmla z26.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49e)  // bfmmla z30.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49b)  // bfmmla z27.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49f)  // bfmmla z31.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b8)  // bfmmla z24.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bc)  // bfmmla z28.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b9)  // bfmmla z25.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bd)  // bfmmla z29.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4ba)  // bfmmla z26.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x20, #2, MUL VL]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4be)  // bfmmla z30.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4bb)  // bfmmla z27.s, z5.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bf)  // bfmmla z31.s, z5.h, z6.h
    bgt label_85
KAI_ASM_LABEL(label_86)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z7.h }, p5/Z, [x23]
    subs x7, x7, #0x4
    ld1rqh { z1.h }, p0/Z, [x8]
    ld1rqh { z2.h }, p0/Z, [x6]
    ld1rqh { z3.h }, p0/Z, [x5]
    ld1rqh { z4.h }, p0/Z, [x4]
    ld1rqh { z5.h }, p0/Z, [x2]
    ld1rqh { z6.h }, p0/Z, [x1]
    trn1 z0.d, z1.d, z2.d
    trn2 z1.d, z1.d, z2.d
    trn1 z2.d, z3.d, z4.d
    trn2 z3.d, z3.d, z4.d
    trn1 z4.d, z5.d, z6.d
    trn2 z5.d, z5.d, z6.d
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e408)  // bfmmla z8.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e450)  // bfmmla z16.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e498)  // bfmmla z24.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e40c)  // bfmmla z12.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49c)  // bfmmla z28.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e409)  // bfmmla z9.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e451)  // bfmmla z17.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e499)  // bfmmla z25.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e40d)  // bfmmla z13.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49d)  // bfmmla z29.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e40a)  // bfmmla z10.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e452)  // bfmmla z18.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49a)  // bfmmla z26.s, z4.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e40e)  // bfmmla z14.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49e)  // bfmmla z30.s, z4.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6467e453)  // bfmmla z19.s, z2.h, z7.h
    KAI_ASM_INST(0x6467e49b)  // bfmmla z27.s, z4.h, z7.h
    KAI_ASM_INST(0x6466e40f)  // bfmmla z15.s, z0.h, z6.h
    KAI_ASM_INST(0x6466e457)  // bfmmla z23.s, z2.h, z6.h
    KAI_ASM_INST(0x6466e49f)  // bfmmla z31.s, z4.h, z6.h
    ble label_87
    ld1h { z7.h }, p5/Z, [x23]
    ld1h { z6.h }, p5/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x6467e428)  // bfmmla z8.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e470)  // bfmmla z16.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b8)  // bfmmla z24.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x22]
    KAI_ASM_INST(0x6466e42c)  // bfmmla z12.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e474)  // bfmmla z20.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bc)  // bfmmla z28.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x6467e429)  // bfmmla z9.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e471)  // bfmmla z17.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4b9)  // bfmmla z25.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x21]
    KAI_ASM_INST(0x6466e42d)  // bfmmla z13.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e475)  // bfmmla z21.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bd)  // bfmmla z29.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x6467e42a)  // bfmmla z10.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e472)  // bfmmla z18.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4ba)  // bfmmla z26.s, z5.h, z7.h
    ld1h { z7.h }, p5/Z, [x20]
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e476)  // bfmmla z22.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4be)  // bfmmla z30.s, z5.h, z6.h
    ld1h { z6.h }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #2
    KAI_ASM_INST(0x6467e42b)  // bfmmla z11.s, z1.h, z7.h
    KAI_ASM_INST(0x6467e473)  // bfmmla z19.s, z3.h, z7.h
    KAI_ASM_INST(0x6467e4bb)  // bfmmla z27.s, z5.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    KAI_ASM_INST(0x6466e477)  // bfmmla z23.s, z3.h, z6.h
    KAI_ASM_INST(0x6466e4bf)  // bfmmla z31.s, z5.h, z6.h
KAI_ASM_LABEL(label_87)  // Height 6: Multiply loop: multiply skip
    tbz x9, #3, label_88
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_83
KAI_ASM_LABEL(label_88)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    uzp1 z7.d, z8.d, z12.d
    uzp2 z8.d, z8.d, z12.d
    uzp1 z12.d, z9.d, z13.d
    uzp2 z9.d, z9.d, z13.d
    uzp1 z13.d, z10.d, z14.d
    uzp2 z10.d, z10.d, z14.d
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    uzp1 z14.d, z11.d, z15.d
    uzp2 z11.d, z11.d, z15.d
    add x4, x5, x1
    uzp1 z15.d, z16.d, z20.d
    uzp2 z16.d, z16.d, z20.d
    add x3, x4, x1
    uzp1 z20.d, z17.d, z21.d
    uzp2 z17.d, z17.d, z21.d
    uzp1 z21.d, z18.d, z22.d
    uzp2 z18.d, z18.d, z22.d
    uzp1 z22.d, z19.d, z23.d
    uzp2 z19.d, z19.d, z23.d
    uzp1 z23.d, z24.d, z28.d
    uzp2 z24.d, z24.d, z28.d
    uzp1 z28.d, z25.d, z29.d
    uzp2 z25.d, z25.d, z29.d
    uzp1 z29.d, z26.d, z30.d
    uzp2 z26.d, z26.d, z30.d
    uzp1 z30.d, z27.d, z31.d
    uzp2 z27.d, z27.d, z31.d
    tbz x9, #1, label_89
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d440)  // ld1rw { z0.s }, p5/Z, [x2]
    KAI_ASM_INST(0x8540d43f)  // ld1rw { z31.s }, p5/Z, [x1]
    fmin z7.s, p5/M, z7.s, z0.s
    fmin z12.s, p5/M, z12.s, z0.s
    fmin z13.s, p5/M, z13.s, z0.s
    fmin z14.s, p5/M, z14.s, z0.s
    fmin z8.s, p5/M, z8.s, z0.s
    fmin z9.s, p5/M, z9.s, z0.s
    fmin z10.s, p5/M, z10.s, z0.s
    fmin z11.s, p5/M, z11.s, z0.s
    fmin z15.s, p5/M, z15.s, z0.s
    fmin z20.s, p5/M, z20.s, z0.s
    fmin z21.s, p5/M, z21.s, z0.s
    fmin z22.s, p5/M, z22.s, z0.s
    fmin z16.s, p5/M, z16.s, z0.s
    fmin z17.s, p5/M, z17.s, z0.s
    fmin z18.s, p5/M, z18.s, z0.s
    fmin z19.s, p5/M, z19.s, z0.s
    fmin z23.s, p5/M, z23.s, z0.s
    fmin z28.s, p5/M, z28.s, z0.s
    fmin z29.s, p5/M, z29.s, z0.s
    fmin z30.s, p5/M, z30.s, z0.s
    fmin z24.s, p5/M, z24.s, z0.s
    fmin z25.s, p5/M, z25.s, z0.s
    fmin z26.s, p5/M, z26.s, z0.s
    fmin z27.s, p5/M, z27.s, z0.s
    fmax z7.s, p5/M, z7.s, z31.s
    fmax z12.s, p5/M, z12.s, z31.s
    fmax z13.s, p5/M, z13.s, z31.s
    fmax z14.s, p5/M, z14.s, z31.s
    fmax z8.s, p5/M, z8.s, z31.s
    fmax z9.s, p5/M, z9.s, z31.s
    fmax z10.s, p5/M, z10.s, z31.s
    fmax z11.s, p5/M, z11.s, z31.s
    fmax z15.s, p5/M, z15.s, z31.s
    fmax z20.s, p5/M, z20.s, z31.s
    fmax z21.s, p5/M, z21.s, z31.s
    fmax z22.s, p5/M, z22.s, z31.s
    fmax z16.s, p5/M, z16.s, z31.s
    fmax z17.s, p5/M, z17.s, z31.s
    fmax z18.s, p5/M, z18.s, z31.s
    fmax z19.s, p5/M, z19.s, z31.s
    fmax z23.s, p5/M, z23.s, z31.s
    fmax z28.s, p5/M, z28.s, z31.s
    fmax z29.s, p5/M, z29.s, z31.s
    fmax z30.s, p5/M, z30.s, z31.s
    fmax z24.s, p5/M, z24.s, z31.s
    fmax z25.s, p5/M, z25.s, z31.s
    fmax z26.s, p5/M, z26.s, z31.s
    fmax z27.s, p5/M, z27.s, z31.s
KAI_ASM_LABEL(label_89)  // Height 6: No activation
    st1w { z7.s }, p4, [x27]
    st1w { z12.s }, p3, [x27, #1, MUL VL]
    st1w { z13.s }, p2, [x27, #2, MUL VL]
    st1w { z14.s }, p1, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z8.s }, p4, [x7]
    st1w { z9.s }, p3, [x7, #1, MUL VL]
    st1w { z10.s }, p2, [x7, #2, MUL VL]
    st1w { z11.s }, p1, [x7, #3, MUL VL]
    st1w { z15.s }, p4, [x6]
    st1w { z20.s }, p3, [x6, #1, MUL VL]
    st1w { z21.s }, p2, [x6, #2, MUL VL]
    st1w { z22.s }, p1, [x6, #3, MUL VL]
    st1w { z16.s }, p4, [x5]
    st1w { z17.s }, p3, [x5, #1, MUL VL]
    st1w { z18.s }, p2, [x5, #2, MUL VL]
    st1w { z19.s }, p1, [x5, #3, MUL VL]
    st1w { z23.s }, p4, [x4]
    st1w { z28.s }, p3, [x4, #1, MUL VL]
    st1w { z29.s }, p2, [x4, #2, MUL VL]
    st1w { z30.s }, p1, [x4, #3, MUL VL]
    st1w { z24.s }, p4, [x3]
    st1w { z25.s }, p3, [x3, #1, MUL VL]
    st1w { z26.s }, p2, [x3, #2, MUL VL]
    st1w { z27.s }, p1, [x3, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_77
    add x11, x11, #0x6
    cmp x11, x10
    blt label_1
KAI_ASM_LABEL(label_91)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffhybrid_bf16fp32_mmla_6x4VL)

    KAI_ASM_END
