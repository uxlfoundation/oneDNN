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

    KAI_ASM_CODE(sve_ffhybrid_fp32_mla_6x4VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffhybrid_fp32_mla_6x4VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffhybrid_fp32_mla_6x4VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffhybrid_fp32_mla_6x4VL_a64fx)
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
    ptrue p4.b
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
    bge label_71
    cmp x1, #0x4
    bgt label_57
    beq label_43
    cmp x1, #0x2
    bgt label_29
    beq label_15
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
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_4
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: no bias
    tbz x9, #0, label_5
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    b label_6
KAI_ASM_LABEL(label_5)  // Height 1: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
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
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_9)  // Height 1: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.s, p4/M, z6.s, z0.s
    ld1w { z6.s }, p4/Z, [x21]
    addvl x22, x22, #1
    fmla z9.s, p4/M, z7.s, z0.s
    ld1w { z7.s }, p4/Z, [x20]
    sub x7, x7, #0x1
    add x8, x8, #0x4
    cmp x7, #0x1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    ld1w { z6.s }, p4/Z, [x23]
    fmla z11.s, p4/M, z7.s, z0.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x23, x23, #1
    addvl x22, x22, #1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z11.s, p4/M, z7.s, z0.s
    tbz x9, #3, label_12
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_8
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    tbz x9, #1, label_13
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d051)  // ld1rw { z17.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d030)  // ld1rw { z16.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z17.s
    fmin z9.s, p4/M, z9.s, z17.s
    fmin z10.s, p4/M, z10.s, z17.s
    fmin z11.s, p4/M, z11.s, z17.s
    fmax z8.s, p4/M, z8.s, z16.s
    fmax z9.s, p4/M, z9.s, z16.s
    fmax z10.s, p4/M, z10.s, z16.s
    fmax z11.s, p4/M, z11.s, z16.s
KAI_ASM_LABEL(label_13)  // Height 1: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_2
    b label_85
KAI_ASM_LABEL(label_15)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_16)  // Height 2: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_17
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_17
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_17
    mov x22, x23
KAI_ASM_LABEL(label_17)  // Height 2: B setup done
    mov x1, #0
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_18
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_20
KAI_ASM_LABEL(label_18)  // Height 2: no bias
    tbz x9, #0, label_19
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    add x1, x27, x1
    ld1w { z12.s }, p3/Z, [x1]
    ld1w { z13.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z14.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z15.s }, p0/Z, [x1, #3, MUL VL]
    b label_20
KAI_ASM_LABEL(label_19)  // Height 2: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_20)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_21
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    b label_23
KAI_ASM_LABEL(label_21)  // Height 2: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_22)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x28, label_23
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_23)  // Height 2: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_25
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x22, x22, #1
    sub x7, x7, #0x1
    add x8, x8, #0x4
    cmp x7, #0x1
    add x6, x6, #0x4
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    addvl x21, x21, #1
    addvl x20, x20, #1
    ld1w { z6.s }, p4/Z, [x23]
    fmla z11.s, p4/M, z7.s, z0.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    fmla z15.s, p4/M, z7.s, z1.s
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_24
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    ld1w { z6.s }, p4/Z, [x21]
    addvl x23, x23, #1
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x22, x22, #1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    tbz x9, #3, label_26
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_22
KAI_ASM_LABEL(label_26)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    tbz x9, #1, label_27
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d051)  // ld1rw { z17.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d030)  // ld1rw { z16.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z17.s
    fmin z9.s, p4/M, z9.s, z17.s
    fmin z10.s, p4/M, z10.s, z17.s
    fmin z11.s, p4/M, z11.s, z17.s
    fmin z12.s, p4/M, z12.s, z17.s
    fmin z13.s, p4/M, z13.s, z17.s
    fmin z14.s, p4/M, z14.s, z17.s
    fmin z15.s, p4/M, z15.s, z17.s
    fmax z8.s, p4/M, z8.s, z16.s
    fmax z9.s, p4/M, z9.s, z16.s
    fmax z10.s, p4/M, z10.s, z16.s
    fmax z11.s, p4/M, z11.s, z16.s
    fmax z12.s, p4/M, z12.s, z16.s
    fmax z13.s, p4/M, z13.s, z16.s
    fmax z14.s, p4/M, z14.s, z16.s
    fmax z15.s, p4/M, z15.s, z16.s
KAI_ASM_LABEL(label_27)  // Height 2: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z12.s }, p3, [x7]
    st1w { z13.s }, p2, [x7, #1, MUL VL]
    st1w { z14.s }, p1, [x7, #2, MUL VL]
    st1w { z15.s }, p0, [x7, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_16
    b label_85
KAI_ASM_LABEL(label_29)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_30)  // Height 3: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_31
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_31
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_31
    mov x22, x23
KAI_ASM_LABEL(label_31)  // Height 3: B setup done
    mov x1, #0
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_32
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    b label_34
KAI_ASM_LABEL(label_32)  // Height 3: no bias
    tbz x9, #0, label_33
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    add x2, x27, x1
    add x1, x2, x1
    ld1w { z12.s }, p3/Z, [x2]
    ld1w { z13.s }, p2/Z, [x2, #1, MUL VL]
    ld1w { z14.s }, p1/Z, [x2, #2, MUL VL]
    ld1w { z15.s }, p0/Z, [x2, #3, MUL VL]
    ld1w { z16.s }, p3/Z, [x1]
    ld1w { z17.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #3, MUL VL]
    b label_34
KAI_ASM_LABEL(label_33)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_34)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_35
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_37
KAI_ASM_LABEL(label_35)  // Height 3: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_36)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x28, label_37
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_37)  // Height 3: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_39
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    sub x7, x7, #0x1
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    ld1w { z7.s }, p4/Z, [x20]
    add x8, x8, #0x4
    cmp x7, #0x1
    add x6, x6, #0x4
    add x5, x5, #0x4
    addvl x21, x21, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    addvl x20, x20, #1
    fmla z18.s, p4/M, z6.s, z2.s
    ld1w { z6.s }, p4/Z, [x23]
    fmla z11.s, p4/M, z7.s, z0.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    fmla z15.s, p4/M, z7.s, z1.s
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    fmla z19.s, p4/M, z7.s, z2.s
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_38
KAI_ASM_LABEL(label_39)  // Height 3: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    addvl x21, x21, #1
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    fmla z19.s, p4/M, z7.s, z2.s
    tbz x9, #3, label_40
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_36
KAI_ASM_LABEL(label_40)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    tbz x9, #1, label_41
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d055)  // ld1rw { z21.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d034)  // ld1rw { z20.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z21.s
    fmin z9.s, p4/M, z9.s, z21.s
    fmin z10.s, p4/M, z10.s, z21.s
    fmin z11.s, p4/M, z11.s, z21.s
    fmin z12.s, p4/M, z12.s, z21.s
    fmin z13.s, p4/M, z13.s, z21.s
    fmin z14.s, p4/M, z14.s, z21.s
    fmin z15.s, p4/M, z15.s, z21.s
    fmin z16.s, p4/M, z16.s, z21.s
    fmin z17.s, p4/M, z17.s, z21.s
    fmin z18.s, p4/M, z18.s, z21.s
    fmin z19.s, p4/M, z19.s, z21.s
    fmax z8.s, p4/M, z8.s, z20.s
    fmax z9.s, p4/M, z9.s, z20.s
    fmax z10.s, p4/M, z10.s, z20.s
    fmax z11.s, p4/M, z11.s, z20.s
    fmax z12.s, p4/M, z12.s, z20.s
    fmax z13.s, p4/M, z13.s, z20.s
    fmax z14.s, p4/M, z14.s, z20.s
    fmax z15.s, p4/M, z15.s, z20.s
    fmax z16.s, p4/M, z16.s, z20.s
    fmax z17.s, p4/M, z17.s, z20.s
    fmax z18.s, p4/M, z18.s, z20.s
    fmax z19.s, p4/M, z19.s, z20.s
KAI_ASM_LABEL(label_41)  // Height 3: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z12.s }, p3, [x7]
    st1w { z13.s }, p2, [x7, #1, MUL VL]
    st1w { z14.s }, p1, [x7, #2, MUL VL]
    st1w { z15.s }, p0, [x7, #3, MUL VL]
    st1w { z16.s }, p3, [x6]
    st1w { z17.s }, p2, [x6, #1, MUL VL]
    st1w { z18.s }, p1, [x6, #2, MUL VL]
    st1w { z19.s }, p0, [x6, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_30
    b label_85
KAI_ASM_LABEL(label_43)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_44)  // Height 4: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_45
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_45
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_45
    mov x22, x23
KAI_ASM_LABEL(label_45)  // Height 4: B setup done
    mov x1, #0
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_46
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_48
KAI_ASM_LABEL(label_46)  // Height 4: no bias
    tbz x9, #0, label_47
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    add x3, x27, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z12.s }, p3/Z, [x3]
    ld1w { z13.s }, p2/Z, [x3, #1, MUL VL]
    ld1w { z14.s }, p1/Z, [x3, #2, MUL VL]
    ld1w { z15.s }, p0/Z, [x3, #3, MUL VL]
    ld1w { z16.s }, p3/Z, [x2]
    ld1w { z17.s }, p2/Z, [x2, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x2, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x2, #3, MUL VL]
    ld1w { z20.s }, p3/Z, [x1]
    ld1w { z21.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z22.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z23.s }, p0/Z, [x1, #3, MUL VL]
    b label_48
KAI_ASM_LABEL(label_47)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_48)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_49
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_51
KAI_ASM_LABEL(label_49)  // Height 4: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_50)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x28, label_51
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_51)  // Height 4: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_53
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: Main loop
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    ld1w { z6.s }, p4/Z, [x21]
    sub x7, x7, #0x1
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    add x8, x8, #0x4
    cmp x7, #0x1
    fmla z17.s, p4/M, z7.s, z2.s
    fmla z21.s, p4/M, z7.s, z3.s
    ld1w { z7.s }, p4/Z, [x20]
    add x6, x6, #0x4
    add x5, x5, #0x4
    add x4, x4, #0x4
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    addvl x21, x21, #1
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    addvl x20, x20, #1
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    fmla z19.s, p4/M, z7.s, z2.s
    fmla z23.s, p4/M, z7.s, z3.s
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_52
KAI_ASM_LABEL(label_53)  // Height 4: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    ld1w { z6.s }, p4/Z, [x21]
    addvl x21, x21, #1
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    fmla z21.s, p4/M, z7.s, z3.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    fmla z19.s, p4/M, z7.s, z2.s
    fmla z23.s, p4/M, z7.s, z3.s
    tbz x9, #3, label_54
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_50
KAI_ASM_LABEL(label_54)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x9, #1, label_55
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d059)  // ld1rw { z25.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d038)  // ld1rw { z24.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z25.s
    fmin z9.s, p4/M, z9.s, z25.s
    fmin z10.s, p4/M, z10.s, z25.s
    fmin z11.s, p4/M, z11.s, z25.s
    fmin z12.s, p4/M, z12.s, z25.s
    fmin z13.s, p4/M, z13.s, z25.s
    fmin z14.s, p4/M, z14.s, z25.s
    fmin z15.s, p4/M, z15.s, z25.s
    fmin z16.s, p4/M, z16.s, z25.s
    fmin z17.s, p4/M, z17.s, z25.s
    fmin z18.s, p4/M, z18.s, z25.s
    fmin z19.s, p4/M, z19.s, z25.s
    fmin z20.s, p4/M, z20.s, z25.s
    fmin z21.s, p4/M, z21.s, z25.s
    fmin z22.s, p4/M, z22.s, z25.s
    fmin z23.s, p4/M, z23.s, z25.s
    fmax z8.s, p4/M, z8.s, z24.s
    fmax z9.s, p4/M, z9.s, z24.s
    fmax z10.s, p4/M, z10.s, z24.s
    fmax z11.s, p4/M, z11.s, z24.s
    fmax z12.s, p4/M, z12.s, z24.s
    fmax z13.s, p4/M, z13.s, z24.s
    fmax z14.s, p4/M, z14.s, z24.s
    fmax z15.s, p4/M, z15.s, z24.s
    fmax z16.s, p4/M, z16.s, z24.s
    fmax z17.s, p4/M, z17.s, z24.s
    fmax z18.s, p4/M, z18.s, z24.s
    fmax z19.s, p4/M, z19.s, z24.s
    fmax z20.s, p4/M, z20.s, z24.s
    fmax z21.s, p4/M, z21.s, z24.s
    fmax z22.s, p4/M, z22.s, z24.s
    fmax z23.s, p4/M, z23.s, z24.s
KAI_ASM_LABEL(label_55)  // Height 4: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z12.s }, p3, [x7]
    st1w { z13.s }, p2, [x7, #1, MUL VL]
    st1w { z14.s }, p1, [x7, #2, MUL VL]
    st1w { z15.s }, p0, [x7, #3, MUL VL]
    st1w { z16.s }, p3, [x6]
    st1w { z17.s }, p2, [x6, #1, MUL VL]
    st1w { z18.s }, p1, [x6, #2, MUL VL]
    st1w { z19.s }, p0, [x6, #3, MUL VL]
    st1w { z20.s }, p3, [x5]
    st1w { z21.s }, p2, [x5, #1, MUL VL]
    st1w { z22.s }, p1, [x5, #2, MUL VL]
    st1w { z23.s }, p0, [x5, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_44
    b label_85
KAI_ASM_LABEL(label_57)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_58)  // Height 5: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_59
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_59
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_59
    mov x22, x23
KAI_ASM_LABEL(label_59)  // Height 5: B setup done
    mov x1, #0
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_60
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_62
KAI_ASM_LABEL(label_60)  // Height 5: no bias
    tbz x9, #0, label_61
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    add x4, x27, x1
    add x3, x4, x1
    add x2, x3, x1
    ld1w { z12.s }, p3/Z, [x4]
    ld1w { z13.s }, p2/Z, [x4, #1, MUL VL]
    add x1, x2, x1
    ld1w { z14.s }, p1/Z, [x4, #2, MUL VL]
    ld1w { z15.s }, p0/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p3/Z, [x3]
    ld1w { z17.s }, p2/Z, [x3, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x3, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x3, #3, MUL VL]
    ld1w { z20.s }, p3/Z, [x2]
    ld1w { z21.s }, p2/Z, [x2, #1, MUL VL]
    ld1w { z22.s }, p1/Z, [x2, #2, MUL VL]
    ld1w { z23.s }, p0/Z, [x2, #3, MUL VL]
    ld1w { z24.s }, p3/Z, [x1]
    ld1w { z25.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z26.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z27.s }, p0/Z, [x1, #3, MUL VL]
    b label_62
KAI_ASM_LABEL(label_61)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_62)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_63
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_65
KAI_ASM_LABEL(label_63)  // Height 5: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_64)  // Height 5: String loop
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
    cbnz x28, label_65
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_65)  // Height 5: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    KAI_ASM_INST(0x8540d044)  // ld1rw { z4.s }, p4/Z, [x2]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_67
KAI_ASM_LABEL(label_66)  // Height 5: Multiply loop: Main loop
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    sub x7, x7, #0x1
    add x8, x8, #0x4
    fmla z24.s, p4/M, z6.s, z4.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    cmp x7, #0x1
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    add x6, x6, #0x4
    add x5, x5, #0x4
    fmla z21.s, p4/M, z7.s, z3.s
    fmla z25.s, p4/M, z7.s, z4.s
    ld1w { z7.s }, p4/Z, [x20]
    add x4, x4, #0x4
    add x2, x2, #0x4
    addvl x21, x21, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    addvl x20, x20, #1
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    fmla z26.s, p4/M, z6.s, z4.s
    fmla z11.s, p4/M, z7.s, z0.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    ld1w { z6.s }, p4/Z, [x23]
    fmla z15.s, p4/M, z7.s, z1.s
    fmla z19.s, p4/M, z7.s, z2.s
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    fmla z23.s, p4/M, z7.s, z3.s
    fmla z27.s, p4/M, z7.s, z4.s
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    KAI_ASM_INST(0x8540d044)  // ld1rw { z4.s }, p4/Z, [x2]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_66
KAI_ASM_LABEL(label_67)  // Height 5: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    fmla z24.s, p4/M, z6.s, z4.s
    ld1w { z6.s }, p4/Z, [x21]
    fmla z9.s, p4/M, z7.s, z0.s
    addvl x21, x21, #1
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    fmla z21.s, p4/M, z7.s, z3.s
    fmla z25.s, p4/M, z7.s, z4.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    fmla z26.s, p4/M, z6.s, z4.s
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    fmla z19.s, p4/M, z7.s, z2.s
    fmla z23.s, p4/M, z7.s, z3.s
    fmla z27.s, p4/M, z7.s, z4.s
    tbz x9, #3, label_68
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_64
KAI_ASM_LABEL(label_68)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x9, #1, label_69
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d05d)  // ld1rw { z29.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d03c)  // ld1rw { z28.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z29.s
    fmin z9.s, p4/M, z9.s, z29.s
    fmin z10.s, p4/M, z10.s, z29.s
    fmin z11.s, p4/M, z11.s, z29.s
    fmin z12.s, p4/M, z12.s, z29.s
    fmin z13.s, p4/M, z13.s, z29.s
    fmin z14.s, p4/M, z14.s, z29.s
    fmin z15.s, p4/M, z15.s, z29.s
    fmin z16.s, p4/M, z16.s, z29.s
    fmin z17.s, p4/M, z17.s, z29.s
    fmin z18.s, p4/M, z18.s, z29.s
    fmin z19.s, p4/M, z19.s, z29.s
    fmin z20.s, p4/M, z20.s, z29.s
    fmin z21.s, p4/M, z21.s, z29.s
    fmin z22.s, p4/M, z22.s, z29.s
    fmin z23.s, p4/M, z23.s, z29.s
    fmin z24.s, p4/M, z24.s, z29.s
    fmin z25.s, p4/M, z25.s, z29.s
    fmin z26.s, p4/M, z26.s, z29.s
    fmin z27.s, p4/M, z27.s, z29.s
    fmax z8.s, p4/M, z8.s, z28.s
    fmax z9.s, p4/M, z9.s, z28.s
    fmax z10.s, p4/M, z10.s, z28.s
    fmax z11.s, p4/M, z11.s, z28.s
    fmax z12.s, p4/M, z12.s, z28.s
    fmax z13.s, p4/M, z13.s, z28.s
    fmax z14.s, p4/M, z14.s, z28.s
    fmax z15.s, p4/M, z15.s, z28.s
    fmax z16.s, p4/M, z16.s, z28.s
    fmax z17.s, p4/M, z17.s, z28.s
    fmax z18.s, p4/M, z18.s, z28.s
    fmax z19.s, p4/M, z19.s, z28.s
    fmax z20.s, p4/M, z20.s, z28.s
    fmax z21.s, p4/M, z21.s, z28.s
    fmax z22.s, p4/M, z22.s, z28.s
    fmax z23.s, p4/M, z23.s, z28.s
    fmax z24.s, p4/M, z24.s, z28.s
    fmax z25.s, p4/M, z25.s, z28.s
    fmax z26.s, p4/M, z26.s, z28.s
    fmax z27.s, p4/M, z27.s, z28.s
KAI_ASM_LABEL(label_69)  // Height 5: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z12.s }, p3, [x7]
    st1w { z13.s }, p2, [x7, #1, MUL VL]
    st1w { z14.s }, p1, [x7, #2, MUL VL]
    st1w { z15.s }, p0, [x7, #3, MUL VL]
    st1w { z16.s }, p3, [x6]
    st1w { z17.s }, p2, [x6, #1, MUL VL]
    st1w { z18.s }, p1, [x6, #2, MUL VL]
    st1w { z19.s }, p0, [x6, #3, MUL VL]
    st1w { z20.s }, p3, [x5]
    st1w { z21.s }, p2, [x5, #1, MUL VL]
    st1w { z22.s }, p1, [x5, #2, MUL VL]
    st1w { z23.s }, p0, [x5, #3, MUL VL]
    st1w { z24.s }, p3, [x4]
    st1w { z25.s }, p2, [x4, #1, MUL VL]
    st1w { z26.s }, p1, [x4, #2, MUL VL]
    st1w { z27.s }, p0, [x4, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_58
    b label_85
KAI_ASM_LABEL(label_71)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_72)  // Height 6: Column loop
    mov x23, x25
    ldr x2, [x0, #0x38]
    cntw x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_73
    decw x1
    mov x20, x23
    cmp x26, x1
    bgt label_73
    decw x1
    mov x21, x23
    cmp x26, x1
    bgt label_73
    mov x22, x23
KAI_ASM_LABEL(label_73)  // Height 6: B setup done
    mov x1, #0
    whilelt p3.s, x1, x26
    incw x1
    whilelt p2.s, x1, x26
    incw x1
    whilelt p1.s, x1, x26
    incw x1
    whilelt p0.s, x1, x26
    cbz x24, label_74
    ld1w { z8.s }, p3/Z, [x24]
    ld1w { z9.s }, p2/Z, [x24, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
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
    b label_76
KAI_ASM_LABEL(label_74)  // Height 6: no bias
    tbz x9, #0, label_75
    ldr x1, [x0, #0x58]
    ld1w { z8.s }, p3/Z, [x27]
    ld1w { z9.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z10.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z11.s }, p0/Z, [x27, #3, MUL VL]
    add x5, x27, x1
    add x4, x5, x1
    add x3, x4, x1
    ld1w { z12.s }, p3/Z, [x5]
    ld1w { z13.s }, p2/Z, [x5, #1, MUL VL]
    add x2, x3, x1
    add x1, x2, x1
    ld1w { z14.s }, p1/Z, [x5, #2, MUL VL]
    ld1w { z15.s }, p0/Z, [x5, #3, MUL VL]
    ld1w { z16.s }, p3/Z, [x4]
    ld1w { z17.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x4, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x4, #3, MUL VL]
    ld1w { z20.s }, p3/Z, [x3]
    ld1w { z21.s }, p2/Z, [x3, #1, MUL VL]
    ld1w { z22.s }, p1/Z, [x3, #2, MUL VL]
    ld1w { z23.s }, p0/Z, [x3, #3, MUL VL]
    ld1w { z24.s }, p3/Z, [x2]
    ld1w { z25.s }, p2/Z, [x2, #1, MUL VL]
    ld1w { z26.s }, p1/Z, [x2, #2, MUL VL]
    ld1w { z27.s }, p0/Z, [x2, #3, MUL VL]
    ld1w { z28.s }, p3/Z, [x1]
    ld1w { z29.s }, p2/Z, [x1, #1, MUL VL]
    ld1w { z30.s }, p1/Z, [x1, #2, MUL VL]
    ld1w { z31.s }, p0/Z, [x1, #3, MUL VL]
    b label_76
KAI_ASM_LABEL(label_75)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_76)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_77
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_79
KAI_ASM_LABEL(label_77)  // Height 6: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_78)  // Height 6: String loop
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
    cbnz x28, label_79
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_79)  // Height 6: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    KAI_ASM_INST(0x8540d044)  // ld1rw { z4.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d025)  // ld1rw { z5.s }, p4/Z, [x1]
    ld1w { z6.s }, p4/Z, [x23]
    ld1w { z7.s }, p4/Z, [x22]
    ble label_81
KAI_ASM_LABEL(label_80)  // Height 6: Multiply loop: Main loop
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    sub x7, x7, #0x1
    add x8, x8, #0x4
    fmla z24.s, p4/M, z6.s, z4.s
    fmla z28.s, p4/M, z6.s, z5.s
    ld1w { z6.s }, p4/Z, [x21]
    cmp x7, #0x1
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    add x6, x6, #0x4
    add x5, x5, #0x4
    fmla z17.s, p4/M, z7.s, z2.s
    fmla z21.s, p4/M, z7.s, z3.s
    add x4, x4, #0x4
    add x2, x2, #0x4
    fmla z25.s, p4/M, z7.s, z4.s
    fmla z29.s, p4/M, z7.s, z5.s
    ld1w { z7.s }, p4/Z, [x20]
    add x1, x1, #0x4
    addvl x21, x21, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    addvl x20, x20, #1
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    fmla z26.s, p4/M, z6.s, z4.s
    fmla z30.s, p4/M, z6.s, z5.s
    ld1w { z6.s }, p4/Z, [x23]
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    KAI_ASM_INST(0x8540d100)  // ld1rw { z0.s }, p4/Z, [x8]
    KAI_ASM_INST(0x8540d0c1)  // ld1rw { z1.s }, p4/Z, [x6]
    fmla z19.s, p4/M, z7.s, z2.s
    fmla z23.s, p4/M, z7.s, z3.s
    KAI_ASM_INST(0x8540d0a2)  // ld1rw { z2.s }, p4/Z, [x5]
    KAI_ASM_INST(0x8540d083)  // ld1rw { z3.s }, p4/Z, [x4]
    fmla z27.s, p4/M, z7.s, z4.s
    fmla z31.s, p4/M, z7.s, z5.s
    KAI_ASM_INST(0x8540d044)  // ld1rw { z4.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d025)  // ld1rw { z5.s }, p4/Z, [x1]
    ld1w { z7.s }, p4/Z, [x22]
    bgt label_80
KAI_ASM_LABEL(label_81)  // Height 6: Multiply loop: Main loop skip
    fmla z8.s, p4/M, z6.s, z0.s
    fmla z12.s, p4/M, z6.s, z1.s
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.s, p4/M, z6.s, z2.s
    fmla z20.s, p4/M, z6.s, z3.s
    fmla z24.s, p4/M, z6.s, z4.s
    fmla z28.s, p4/M, z6.s, z5.s
    ld1w { z6.s }, p4/Z, [x21]
    addvl x21, x21, #1
    fmla z9.s, p4/M, z7.s, z0.s
    fmla z13.s, p4/M, z7.s, z1.s
    fmla z17.s, p4/M, z7.s, z2.s
    fmla z21.s, p4/M, z7.s, z3.s
    fmla z25.s, p4/M, z7.s, z4.s
    fmla z29.s, p4/M, z7.s, z5.s
    ld1w { z7.s }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.s, p4/M, z6.s, z0.s
    fmla z14.s, p4/M, z6.s, z1.s
    fmla z18.s, p4/M, z6.s, z2.s
    fmla z22.s, p4/M, z6.s, z3.s
    fmla z26.s, p4/M, z6.s, z4.s
    fmla z30.s, p4/M, z6.s, z5.s
    fmla z11.s, p4/M, z7.s, z0.s
    fmla z15.s, p4/M, z7.s, z1.s
    fmla z19.s, p4/M, z7.s, z2.s
    fmla z23.s, p4/M, z7.s, z3.s
    fmla z27.s, p4/M, z7.s, z4.s
    fmla z31.s, p4/M, z7.s, z5.s
    tbz x9, #3, label_82
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_78
KAI_ASM_LABEL(label_82)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x9, #1, label_83
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540d041)  // ld1rw { z1.s }, p4/Z, [x2]
    KAI_ASM_INST(0x8540d020)  // ld1rw { z0.s }, p4/Z, [x1]
    fmin z8.s, p4/M, z8.s, z1.s
    fmin z9.s, p4/M, z9.s, z1.s
    fmin z10.s, p4/M, z10.s, z1.s
    fmin z11.s, p4/M, z11.s, z1.s
    fmin z12.s, p4/M, z12.s, z1.s
    fmin z13.s, p4/M, z13.s, z1.s
    fmin z14.s, p4/M, z14.s, z1.s
    fmin z15.s, p4/M, z15.s, z1.s
    fmin z16.s, p4/M, z16.s, z1.s
    fmin z17.s, p4/M, z17.s, z1.s
    fmin z18.s, p4/M, z18.s, z1.s
    fmin z19.s, p4/M, z19.s, z1.s
    fmin z20.s, p4/M, z20.s, z1.s
    fmin z21.s, p4/M, z21.s, z1.s
    fmin z22.s, p4/M, z22.s, z1.s
    fmin z23.s, p4/M, z23.s, z1.s
    fmin z24.s, p4/M, z24.s, z1.s
    fmin z25.s, p4/M, z25.s, z1.s
    fmin z26.s, p4/M, z26.s, z1.s
    fmin z27.s, p4/M, z27.s, z1.s
    fmin z28.s, p4/M, z28.s, z1.s
    fmin z29.s, p4/M, z29.s, z1.s
    fmin z30.s, p4/M, z30.s, z1.s
    fmin z31.s, p4/M, z31.s, z1.s
    fmax z8.s, p4/M, z8.s, z0.s
    fmax z9.s, p4/M, z9.s, z0.s
    fmax z10.s, p4/M, z10.s, z0.s
    fmax z11.s, p4/M, z11.s, z0.s
    fmax z12.s, p4/M, z12.s, z0.s
    fmax z13.s, p4/M, z13.s, z0.s
    fmax z14.s, p4/M, z14.s, z0.s
    fmax z15.s, p4/M, z15.s, z0.s
    fmax z16.s, p4/M, z16.s, z0.s
    fmax z17.s, p4/M, z17.s, z0.s
    fmax z18.s, p4/M, z18.s, z0.s
    fmax z19.s, p4/M, z19.s, z0.s
    fmax z20.s, p4/M, z20.s, z0.s
    fmax z21.s, p4/M, z21.s, z0.s
    fmax z22.s, p4/M, z22.s, z0.s
    fmax z23.s, p4/M, z23.s, z0.s
    fmax z24.s, p4/M, z24.s, z0.s
    fmax z25.s, p4/M, z25.s, z0.s
    fmax z26.s, p4/M, z26.s, z0.s
    fmax z27.s, p4/M, z27.s, z0.s
    fmax z28.s, p4/M, z28.s, z0.s
    fmax z29.s, p4/M, z29.s, z0.s
    fmax z30.s, p4/M, z30.s, z0.s
    fmax z31.s, p4/M, z31.s, z0.s
KAI_ASM_LABEL(label_83)  // Height 6: No activation
    st1w { z8.s }, p3, [x27]
    st1w { z9.s }, p2, [x27, #1, MUL VL]
    st1w { z10.s }, p1, [x27, #2, MUL VL]
    st1w { z11.s }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z12.s }, p3, [x7]
    st1w { z13.s }, p2, [x7, #1, MUL VL]
    st1w { z14.s }, p1, [x7, #2, MUL VL]
    st1w { z15.s }, p0, [x7, #3, MUL VL]
    st1w { z16.s }, p3, [x6]
    st1w { z17.s }, p2, [x6, #1, MUL VL]
    st1w { z18.s }, p1, [x6, #2, MUL VL]
    st1w { z19.s }, p0, [x6, #3, MUL VL]
    st1w { z20.s }, p3, [x5]
    st1w { z21.s }, p2, [x5, #1, MUL VL]
    st1w { z22.s }, p1, [x5, #2, MUL VL]
    st1w { z23.s }, p0, [x5, #3, MUL VL]
    st1w { z24.s }, p3, [x4]
    st1w { z25.s }, p2, [x4, #1, MUL VL]
    st1w { z26.s }, p1, [x4, #2, MUL VL]
    st1w { z27.s }, p0, [x4, #3, MUL VL]
    st1w { z28.s }, p3, [x3]
    st1w { z29.s }, p2, [x3, #1, MUL VL]
    st1w { z30.s }, p1, [x3, #2, MUL VL]
    st1w { z31.s }, p0, [x3, #3, MUL VL]
    decw x26, ALL, MUL #4
    cmp x26, XZR
    bgt label_72
    add x11, x11, #0x6
    cmp x11, x10
    blt label_1
KAI_ASM_LABEL(label_85)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffhybrid_fp32_mla_6x4VL_a64fx)

    KAI_ASM_END
