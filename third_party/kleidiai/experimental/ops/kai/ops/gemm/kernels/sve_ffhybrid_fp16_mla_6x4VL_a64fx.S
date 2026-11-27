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

    KAI_ASM_CODE(sve_ffhybrid_fp16_mla_6x4VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffhybrid_fp16_mla_6x4VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffhybrid_fp16_mla_6x4VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffhybrid_fp16_mla_6x4VL_a64fx)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x6, #0xfc00
    mov x5, #0x7c00
    ptrue p4.b
    mov x11, #0
    ldr x4, [x0, #0x60]
    ldrh w1, [x0, #0xa0]
    ldrh w2, [x0, #0xa4]
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_3
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_3
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_3
    mov x22, x23
KAI_ASM_LABEL(label_3)  // Height 1: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_4
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: no bias
    tbz x9, #0, label_5
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_9)  // Height 1: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.h, p4/M, z6.h, z0.h
    ld1h { z6.h }, p4/Z, [x21]
    addvl x22, x22, #1
    fmla z9.h, p4/M, z7.h, z0.h
    ld1h { z7.h }, p4/Z, [x20]
    sub x7, x7, #0x1
    add x8, x8, #0x2
    cmp x7, #0x1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    ld1h { z6.h }, p4/Z, [x23]
    fmla z11.h, p4/M, z7.h, z0.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x23, x23, #1
    addvl x22, x22, #1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z11.h, p4/M, z7.h, z0.h
    tbz x9, #3, label_12
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_8
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    tbz x9, #1, label_13
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b051)  // ld1rh { z17.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b030)  // ld1rh { z16.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z17.h
    fmin z9.h, p4/M, z9.h, z17.h
    fmin z10.h, p4/M, z10.h, z17.h
    fmin z11.h, p4/M, z11.h, z17.h
    fmax z8.h, p4/M, z8.h, z16.h
    fmax z9.h, p4/M, z9.h, z16.h
    fmax z10.h, p4/M, z10.h, z16.h
    fmax z11.h, p4/M, z11.h, z16.h
KAI_ASM_LABEL(label_13)  // Height 1: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    dech x26, ALL, MUL #4
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_17
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_17
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_17
    mov x22, x23
KAI_ASM_LABEL(label_17)  // Height 2: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_18
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_20
KAI_ASM_LABEL(label_18)  // Height 2: no bias
    tbz x9, #0, label_19
    ldr x1, [x0, #0x58]
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
    add x1, x27, x1
    ld1h { z12.h }, p3/Z, [x1]
    ld1h { z13.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z14.h }, p1/Z, [x1, #2, MUL VL]
    ld1h { z15.h }, p0/Z, [x1, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_23)  // Height 2: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_25
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x22, x22, #1
    sub x7, x7, #0x1
    add x8, x8, #0x2
    cmp x7, #0x1
    add x6, x6, #0x2
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    addvl x21, x21, #1
    addvl x20, x20, #1
    ld1h { z6.h }, p4/Z, [x23]
    fmla z11.h, p4/M, z7.h, z0.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    fmla z15.h, p4/M, z7.h, z1.h
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_24
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    ld1h { z6.h }, p4/Z, [x21]
    addvl x23, x23, #1
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x22, x22, #1
    addvl x21, x21, #1
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
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
    KAI_ASM_INST(0x84c0b051)  // ld1rh { z17.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b030)  // ld1rh { z16.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z17.h
    fmin z9.h, p4/M, z9.h, z17.h
    fmin z10.h, p4/M, z10.h, z17.h
    fmin z11.h, p4/M, z11.h, z17.h
    fmin z12.h, p4/M, z12.h, z17.h
    fmin z13.h, p4/M, z13.h, z17.h
    fmin z14.h, p4/M, z14.h, z17.h
    fmin z15.h, p4/M, z15.h, z17.h
    fmax z8.h, p4/M, z8.h, z16.h
    fmax z9.h, p4/M, z9.h, z16.h
    fmax z10.h, p4/M, z10.h, z16.h
    fmax z11.h, p4/M, z11.h, z16.h
    fmax z12.h, p4/M, z12.h, z16.h
    fmax z13.h, p4/M, z13.h, z16.h
    fmax z14.h, p4/M, z14.h, z16.h
    fmax z15.h, p4/M, z15.h, z16.h
KAI_ASM_LABEL(label_27)  // Height 2: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1h { z12.h }, p3, [x7]
    st1h { z13.h }, p2, [x7, #1, MUL VL]
    st1h { z14.h }, p1, [x7, #2, MUL VL]
    st1h { z15.h }, p0, [x7, #3, MUL VL]
    dech x26, ALL, MUL #4
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_31
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_31
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_31
    mov x22, x23
KAI_ASM_LABEL(label_31)  // Height 3: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_32
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
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
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
    add x2, x27, x1
    add x1, x2, x1
    ld1h { z12.h }, p3/Z, [x2]
    ld1h { z13.h }, p2/Z, [x2, #1, MUL VL]
    ld1h { z14.h }, p1/Z, [x2, #2, MUL VL]
    ld1h { z15.h }, p0/Z, [x2, #3, MUL VL]
    ld1h { z16.h }, p3/Z, [x1]
    ld1h { z17.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x1, #2, MUL VL]
    ld1h { z19.h }, p0/Z, [x1, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_37)  // Height 3: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_39
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: Main loop
    addvl x23, x23, #1
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    sub x7, x7, #0x1
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    ld1h { z7.h }, p4/Z, [x20]
    add x8, x8, #0x2
    cmp x7, #0x1
    add x6, x6, #0x2
    add x5, x5, #0x2
    addvl x21, x21, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    addvl x20, x20, #1
    fmla z18.h, p4/M, z6.h, z2.h
    ld1h { z6.h }, p4/Z, [x23]
    fmla z11.h, p4/M, z7.h, z0.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    fmla z15.h, p4/M, z7.h, z1.h
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    fmla z19.h, p4/M, z7.h, z2.h
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_38
KAI_ASM_LABEL(label_39)  // Height 3: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    addvl x21, x21, #1
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    fmla z19.h, p4/M, z7.h, z2.h
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
    KAI_ASM_INST(0x84c0b055)  // ld1rh { z21.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b034)  // ld1rh { z20.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z21.h
    fmin z9.h, p4/M, z9.h, z21.h
    fmin z10.h, p4/M, z10.h, z21.h
    fmin z11.h, p4/M, z11.h, z21.h
    fmin z12.h, p4/M, z12.h, z21.h
    fmin z13.h, p4/M, z13.h, z21.h
    fmin z14.h, p4/M, z14.h, z21.h
    fmin z15.h, p4/M, z15.h, z21.h
    fmin z16.h, p4/M, z16.h, z21.h
    fmin z17.h, p4/M, z17.h, z21.h
    fmin z18.h, p4/M, z18.h, z21.h
    fmin z19.h, p4/M, z19.h, z21.h
    fmax z8.h, p4/M, z8.h, z20.h
    fmax z9.h, p4/M, z9.h, z20.h
    fmax z10.h, p4/M, z10.h, z20.h
    fmax z11.h, p4/M, z11.h, z20.h
    fmax z12.h, p4/M, z12.h, z20.h
    fmax z13.h, p4/M, z13.h, z20.h
    fmax z14.h, p4/M, z14.h, z20.h
    fmax z15.h, p4/M, z15.h, z20.h
    fmax z16.h, p4/M, z16.h, z20.h
    fmax z17.h, p4/M, z17.h, z20.h
    fmax z18.h, p4/M, z18.h, z20.h
    fmax z19.h, p4/M, z19.h, z20.h
KAI_ASM_LABEL(label_41)  // Height 3: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1h { z12.h }, p3, [x7]
    st1h { z13.h }, p2, [x7, #1, MUL VL]
    st1h { z14.h }, p1, [x7, #2, MUL VL]
    st1h { z15.h }, p0, [x7, #3, MUL VL]
    st1h { z16.h }, p3, [x6]
    st1h { z17.h }, p2, [x6, #1, MUL VL]
    st1h { z18.h }, p1, [x6, #2, MUL VL]
    st1h { z19.h }, p0, [x6, #3, MUL VL]
    dech x26, ALL, MUL #4
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_45
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_45
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_45
    mov x22, x23
KAI_ASM_LABEL(label_45)  // Height 4: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_46
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
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
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
    add x3, x27, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1h { z12.h }, p3/Z, [x3]
    ld1h { z13.h }, p2/Z, [x3, #1, MUL VL]
    ld1h { z14.h }, p1/Z, [x3, #2, MUL VL]
    ld1h { z15.h }, p0/Z, [x3, #3, MUL VL]
    ld1h { z16.h }, p3/Z, [x2]
    ld1h { z17.h }, p2/Z, [x2, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x2, #2, MUL VL]
    ld1h { z19.h }, p0/Z, [x2, #3, MUL VL]
    ld1h { z20.h }, p3/Z, [x1]
    ld1h { z21.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z22.h }, p1/Z, [x1, #2, MUL VL]
    ld1h { z23.h }, p0/Z, [x1, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_51)  // Height 4: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_53
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: Main loop
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    ld1h { z6.h }, p4/Z, [x21]
    sub x7, x7, #0x1
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    add x8, x8, #0x2
    cmp x7, #0x1
    fmla z17.h, p4/M, z7.h, z2.h
    fmla z21.h, p4/M, z7.h, z3.h
    ld1h { z7.h }, p4/Z, [x20]
    add x6, x6, #0x2
    add x5, x5, #0x2
    add x4, x4, #0x2
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    addvl x21, x21, #1
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    addvl x20, x20, #1
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    fmla z19.h, p4/M, z7.h, z2.h
    fmla z23.h, p4/M, z7.h, z3.h
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_52
KAI_ASM_LABEL(label_53)  // Height 4: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    ld1h { z6.h }, p4/Z, [x21]
    addvl x21, x21, #1
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    fmla z21.h, p4/M, z7.h, z3.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    fmla z19.h, p4/M, z7.h, z2.h
    fmla z23.h, p4/M, z7.h, z3.h
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
    KAI_ASM_INST(0x84c0b059)  // ld1rh { z25.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b038)  // ld1rh { z24.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z25.h
    fmin z9.h, p4/M, z9.h, z25.h
    fmin z10.h, p4/M, z10.h, z25.h
    fmin z11.h, p4/M, z11.h, z25.h
    fmin z12.h, p4/M, z12.h, z25.h
    fmin z13.h, p4/M, z13.h, z25.h
    fmin z14.h, p4/M, z14.h, z25.h
    fmin z15.h, p4/M, z15.h, z25.h
    fmin z16.h, p4/M, z16.h, z25.h
    fmin z17.h, p4/M, z17.h, z25.h
    fmin z18.h, p4/M, z18.h, z25.h
    fmin z19.h, p4/M, z19.h, z25.h
    fmin z20.h, p4/M, z20.h, z25.h
    fmin z21.h, p4/M, z21.h, z25.h
    fmin z22.h, p4/M, z22.h, z25.h
    fmin z23.h, p4/M, z23.h, z25.h
    fmax z8.h, p4/M, z8.h, z24.h
    fmax z9.h, p4/M, z9.h, z24.h
    fmax z10.h, p4/M, z10.h, z24.h
    fmax z11.h, p4/M, z11.h, z24.h
    fmax z12.h, p4/M, z12.h, z24.h
    fmax z13.h, p4/M, z13.h, z24.h
    fmax z14.h, p4/M, z14.h, z24.h
    fmax z15.h, p4/M, z15.h, z24.h
    fmax z16.h, p4/M, z16.h, z24.h
    fmax z17.h, p4/M, z17.h, z24.h
    fmax z18.h, p4/M, z18.h, z24.h
    fmax z19.h, p4/M, z19.h, z24.h
    fmax z20.h, p4/M, z20.h, z24.h
    fmax z21.h, p4/M, z21.h, z24.h
    fmax z22.h, p4/M, z22.h, z24.h
    fmax z23.h, p4/M, z23.h, z24.h
KAI_ASM_LABEL(label_55)  // Height 4: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1h { z12.h }, p3, [x7]
    st1h { z13.h }, p2, [x7, #1, MUL VL]
    st1h { z14.h }, p1, [x7, #2, MUL VL]
    st1h { z15.h }, p0, [x7, #3, MUL VL]
    st1h { z16.h }, p3, [x6]
    st1h { z17.h }, p2, [x6, #1, MUL VL]
    st1h { z18.h }, p1, [x6, #2, MUL VL]
    st1h { z19.h }, p0, [x6, #3, MUL VL]
    st1h { z20.h }, p3, [x5]
    st1h { z21.h }, p2, [x5, #1, MUL VL]
    st1h { z22.h }, p1, [x5, #2, MUL VL]
    st1h { z23.h }, p0, [x5, #3, MUL VL]
    dech x26, ALL, MUL #4
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_59
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_59
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_59
    mov x22, x23
KAI_ASM_LABEL(label_59)  // Height 5: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_60
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
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
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
    add x4, x27, x1
    add x3, x4, x1
    add x2, x3, x1
    ld1h { z12.h }, p3/Z, [x4]
    ld1h { z13.h }, p2/Z, [x4, #1, MUL VL]
    add x1, x2, x1
    ld1h { z14.h }, p1/Z, [x4, #2, MUL VL]
    ld1h { z15.h }, p0/Z, [x4, #3, MUL VL]
    ld1h { z16.h }, p3/Z, [x3]
    ld1h { z17.h }, p2/Z, [x3, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x3, #2, MUL VL]
    ld1h { z19.h }, p0/Z, [x3, #3, MUL VL]
    ld1h { z20.h }, p3/Z, [x2]
    ld1h { z21.h }, p2/Z, [x2, #1, MUL VL]
    ld1h { z22.h }, p1/Z, [x2, #2, MUL VL]
    ld1h { z23.h }, p0/Z, [x2, #3, MUL VL]
    ld1h { z24.h }, p3/Z, [x1]
    ld1h { z25.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z26.h }, p1/Z, [x1, #2, MUL VL]
    ld1h { z27.h }, p0/Z, [x1, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_65)  // Height 5: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    KAI_ASM_INST(0x84c0b044)  // ld1rh { z4.h }, p4/Z, [x2]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_67
KAI_ASM_LABEL(label_66)  // Height 5: Multiply loop: Main loop
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    sub x7, x7, #0x1
    add x8, x8, #0x2
    fmla z24.h, p4/M, z6.h, z4.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    cmp x7, #0x1
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    add x6, x6, #0x2
    add x5, x5, #0x2
    fmla z21.h, p4/M, z7.h, z3.h
    fmla z25.h, p4/M, z7.h, z4.h
    ld1h { z7.h }, p4/Z, [x20]
    add x4, x4, #0x2
    add x2, x2, #0x2
    addvl x21, x21, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    addvl x20, x20, #1
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    fmla z26.h, p4/M, z6.h, z4.h
    fmla z11.h, p4/M, z7.h, z0.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    ld1h { z6.h }, p4/Z, [x23]
    fmla z15.h, p4/M, z7.h, z1.h
    fmla z19.h, p4/M, z7.h, z2.h
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    fmla z23.h, p4/M, z7.h, z3.h
    fmla z27.h, p4/M, z7.h, z4.h
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    KAI_ASM_INST(0x84c0b044)  // ld1rh { z4.h }, p4/Z, [x2]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_66
KAI_ASM_LABEL(label_67)  // Height 5: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    fmla z24.h, p4/M, z6.h, z4.h
    ld1h { z6.h }, p4/Z, [x21]
    fmla z9.h, p4/M, z7.h, z0.h
    addvl x21, x21, #1
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    fmla z21.h, p4/M, z7.h, z3.h
    fmla z25.h, p4/M, z7.h, z4.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    fmla z26.h, p4/M, z6.h, z4.h
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    fmla z19.h, p4/M, z7.h, z2.h
    fmla z23.h, p4/M, z7.h, z3.h
    fmla z27.h, p4/M, z7.h, z4.h
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
    KAI_ASM_INST(0x84c0b05d)  // ld1rh { z29.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b03c)  // ld1rh { z28.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z29.h
    fmin z9.h, p4/M, z9.h, z29.h
    fmin z10.h, p4/M, z10.h, z29.h
    fmin z11.h, p4/M, z11.h, z29.h
    fmin z12.h, p4/M, z12.h, z29.h
    fmin z13.h, p4/M, z13.h, z29.h
    fmin z14.h, p4/M, z14.h, z29.h
    fmin z15.h, p4/M, z15.h, z29.h
    fmin z16.h, p4/M, z16.h, z29.h
    fmin z17.h, p4/M, z17.h, z29.h
    fmin z18.h, p4/M, z18.h, z29.h
    fmin z19.h, p4/M, z19.h, z29.h
    fmin z20.h, p4/M, z20.h, z29.h
    fmin z21.h, p4/M, z21.h, z29.h
    fmin z22.h, p4/M, z22.h, z29.h
    fmin z23.h, p4/M, z23.h, z29.h
    fmin z24.h, p4/M, z24.h, z29.h
    fmin z25.h, p4/M, z25.h, z29.h
    fmin z26.h, p4/M, z26.h, z29.h
    fmin z27.h, p4/M, z27.h, z29.h
    fmax z8.h, p4/M, z8.h, z28.h
    fmax z9.h, p4/M, z9.h, z28.h
    fmax z10.h, p4/M, z10.h, z28.h
    fmax z11.h, p4/M, z11.h, z28.h
    fmax z12.h, p4/M, z12.h, z28.h
    fmax z13.h, p4/M, z13.h, z28.h
    fmax z14.h, p4/M, z14.h, z28.h
    fmax z15.h, p4/M, z15.h, z28.h
    fmax z16.h, p4/M, z16.h, z28.h
    fmax z17.h, p4/M, z17.h, z28.h
    fmax z18.h, p4/M, z18.h, z28.h
    fmax z19.h, p4/M, z19.h, z28.h
    fmax z20.h, p4/M, z20.h, z28.h
    fmax z21.h, p4/M, z21.h, z28.h
    fmax z22.h, p4/M, z22.h, z28.h
    fmax z23.h, p4/M, z23.h, z28.h
    fmax z24.h, p4/M, z24.h, z28.h
    fmax z25.h, p4/M, z25.h, z28.h
    fmax z26.h, p4/M, z26.h, z28.h
    fmax z27.h, p4/M, z27.h, z28.h
KAI_ASM_LABEL(label_69)  // Height 5: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1h { z12.h }, p3, [x7]
    st1h { z13.h }, p2, [x7, #1, MUL VL]
    st1h { z14.h }, p1, [x7, #2, MUL VL]
    st1h { z15.h }, p0, [x7, #3, MUL VL]
    st1h { z16.h }, p3, [x6]
    st1h { z17.h }, p2, [x6, #1, MUL VL]
    st1h { z18.h }, p1, [x6, #2, MUL VL]
    st1h { z19.h }, p0, [x6, #3, MUL VL]
    st1h { z20.h }, p3, [x5]
    st1h { z21.h }, p2, [x5, #1, MUL VL]
    st1h { z22.h }, p1, [x5, #2, MUL VL]
    st1h { z23.h }, p0, [x5, #3, MUL VL]
    st1h { z24.h }, p3, [x4]
    st1h { z25.h }, p2, [x4, #1, MUL VL]
    st1h { z26.h }, p1, [x4, #2, MUL VL]
    st1h { z27.h }, p0, [x4, #3, MUL VL]
    dech x26, ALL, MUL #4
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
    cnth x1, ALL, MUL #3
    add x22, x23, x2
    add x21, x22, x2
    add x20, x21, x2
    cmp x26, x1
    add x25, x20, x2
    bgt label_73
    dech x1
    mov x20, x23
    cmp x26, x1
    bgt label_73
    dech x1
    mov x21, x23
    cmp x26, x1
    bgt label_73
    mov x22, x23
KAI_ASM_LABEL(label_73)  // Height 6: B setup done
    mov x1, #0
    whilelt p3.h, x1, x26
    inch x1
    whilelt p2.h, x1, x26
    inch x1
    whilelt p1.h, x1, x26
    inch x1
    whilelt p0.h, x1, x26
    cbz x24, label_74
    ld1h { z8.h }, p3/Z, [x24]
    ld1h { z9.h }, p2/Z, [x24, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x24, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x24, #3, MUL VL]
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
    ld1h { z8.h }, p3/Z, [x27]
    ld1h { z9.h }, p2/Z, [x27, #1, MUL VL]
    ld1h { z10.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z11.h }, p0/Z, [x27, #3, MUL VL]
    add x5, x27, x1
    add x4, x5, x1
    add x3, x4, x1
    ld1h { z12.h }, p3/Z, [x5]
    ld1h { z13.h }, p2/Z, [x5, #1, MUL VL]
    add x2, x3, x1
    add x1, x2, x1
    ld1h { z14.h }, p1/Z, [x5, #2, MUL VL]
    ld1h { z15.h }, p0/Z, [x5, #3, MUL VL]
    ld1h { z16.h }, p3/Z, [x4]
    ld1h { z17.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x4, #2, MUL VL]
    ld1h { z19.h }, p0/Z, [x4, #3, MUL VL]
    ld1h { z20.h }, p3/Z, [x3]
    ld1h { z21.h }, p2/Z, [x3, #1, MUL VL]
    ld1h { z22.h }, p1/Z, [x3, #2, MUL VL]
    ld1h { z23.h }, p0/Z, [x3, #3, MUL VL]
    ld1h { z24.h }, p3/Z, [x2]
    ld1h { z25.h }, p2/Z, [x2, #1, MUL VL]
    ld1h { z26.h }, p1/Z, [x2, #2, MUL VL]
    ld1h { z27.h }, p0/Z, [x2, #3, MUL VL]
    ld1h { z28.h }, p3/Z, [x1]
    ld1h { z29.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z30.h }, p1/Z, [x1, #2, MUL VL]
    ld1h { z31.h }, p0/Z, [x1, #3, MUL VL]
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
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_79)  // Height 6: input setup done
    cmp x7, #0x1
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    KAI_ASM_INST(0x84c0b044)  // ld1rh { z4.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b025)  // ld1rh { z5.h }, p4/Z, [x1]
    ld1h { z6.h }, p4/Z, [x23]
    ld1h { z7.h }, p4/Z, [x22]
    ble label_81
KAI_ASM_LABEL(label_80)  // Height 6: Multiply loop: Main loop
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    sub x7, x7, #0x1
    add x8, x8, #0x2
    fmla z24.h, p4/M, z6.h, z4.h
    fmla z28.h, p4/M, z6.h, z5.h
    ld1h { z6.h }, p4/Z, [x21]
    cmp x7, #0x1
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    add x6, x6, #0x2
    add x5, x5, #0x2
    fmla z17.h, p4/M, z7.h, z2.h
    fmla z21.h, p4/M, z7.h, z3.h
    add x4, x4, #0x2
    add x2, x2, #0x2
    fmla z25.h, p4/M, z7.h, z4.h
    fmla z29.h, p4/M, z7.h, z5.h
    ld1h { z7.h }, p4/Z, [x20]
    add x1, x1, #0x2
    addvl x21, x21, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    addvl x20, x20, #1
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    fmla z26.h, p4/M, z6.h, z4.h
    fmla z30.h, p4/M, z6.h, z5.h
    ld1h { z6.h }, p4/Z, [x23]
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    KAI_ASM_INST(0x84c0b100)  // ld1rh { z0.h }, p4/Z, [x8]
    KAI_ASM_INST(0x84c0b0c1)  // ld1rh { z1.h }, p4/Z, [x6]
    fmla z19.h, p4/M, z7.h, z2.h
    fmla z23.h, p4/M, z7.h, z3.h
    KAI_ASM_INST(0x84c0b0a2)  // ld1rh { z2.h }, p4/Z, [x5]
    KAI_ASM_INST(0x84c0b083)  // ld1rh { z3.h }, p4/Z, [x4]
    fmla z27.h, p4/M, z7.h, z4.h
    fmla z31.h, p4/M, z7.h, z5.h
    KAI_ASM_INST(0x84c0b044)  // ld1rh { z4.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b025)  // ld1rh { z5.h }, p4/Z, [x1]
    ld1h { z7.h }, p4/Z, [x22]
    bgt label_80
KAI_ASM_LABEL(label_81)  // Height 6: Multiply loop: Main loop skip
    fmla z8.h, p4/M, z6.h, z0.h
    fmla z12.h, p4/M, z6.h, z1.h
    addvl x23, x23, #1
    addvl x22, x22, #1
    fmla z16.h, p4/M, z6.h, z2.h
    fmla z20.h, p4/M, z6.h, z3.h
    fmla z24.h, p4/M, z6.h, z4.h
    fmla z28.h, p4/M, z6.h, z5.h
    ld1h { z6.h }, p4/Z, [x21]
    addvl x21, x21, #1
    fmla z9.h, p4/M, z7.h, z0.h
    fmla z13.h, p4/M, z7.h, z1.h
    fmla z17.h, p4/M, z7.h, z2.h
    fmla z21.h, p4/M, z7.h, z3.h
    fmla z25.h, p4/M, z7.h, z4.h
    fmla z29.h, p4/M, z7.h, z5.h
    ld1h { z7.h }, p4/Z, [x20]
    addvl x20, x20, #1
    fmla z10.h, p4/M, z6.h, z0.h
    fmla z14.h, p4/M, z6.h, z1.h
    fmla z18.h, p4/M, z6.h, z2.h
    fmla z22.h, p4/M, z6.h, z3.h
    fmla z26.h, p4/M, z6.h, z4.h
    fmla z30.h, p4/M, z6.h, z5.h
    fmla z11.h, p4/M, z7.h, z0.h
    fmla z15.h, p4/M, z7.h, z1.h
    fmla z19.h, p4/M, z7.h, z2.h
    fmla z23.h, p4/M, z7.h, z3.h
    fmla z27.h, p4/M, z7.h, z4.h
    fmla z31.h, p4/M, z7.h, z5.h
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
    KAI_ASM_INST(0x84c0b041)  // ld1rh { z1.h }, p4/Z, [x2]
    KAI_ASM_INST(0x84c0b020)  // ld1rh { z0.h }, p4/Z, [x1]
    fmin z8.h, p4/M, z8.h, z1.h
    fmin z9.h, p4/M, z9.h, z1.h
    fmin z10.h, p4/M, z10.h, z1.h
    fmin z11.h, p4/M, z11.h, z1.h
    fmin z12.h, p4/M, z12.h, z1.h
    fmin z13.h, p4/M, z13.h, z1.h
    fmin z14.h, p4/M, z14.h, z1.h
    fmin z15.h, p4/M, z15.h, z1.h
    fmin z16.h, p4/M, z16.h, z1.h
    fmin z17.h, p4/M, z17.h, z1.h
    fmin z18.h, p4/M, z18.h, z1.h
    fmin z19.h, p4/M, z19.h, z1.h
    fmin z20.h, p4/M, z20.h, z1.h
    fmin z21.h, p4/M, z21.h, z1.h
    fmin z22.h, p4/M, z22.h, z1.h
    fmin z23.h, p4/M, z23.h, z1.h
    fmin z24.h, p4/M, z24.h, z1.h
    fmin z25.h, p4/M, z25.h, z1.h
    fmin z26.h, p4/M, z26.h, z1.h
    fmin z27.h, p4/M, z27.h, z1.h
    fmin z28.h, p4/M, z28.h, z1.h
    fmin z29.h, p4/M, z29.h, z1.h
    fmin z30.h, p4/M, z30.h, z1.h
    fmin z31.h, p4/M, z31.h, z1.h
    fmax z8.h, p4/M, z8.h, z0.h
    fmax z9.h, p4/M, z9.h, z0.h
    fmax z10.h, p4/M, z10.h, z0.h
    fmax z11.h, p4/M, z11.h, z0.h
    fmax z12.h, p4/M, z12.h, z0.h
    fmax z13.h, p4/M, z13.h, z0.h
    fmax z14.h, p4/M, z14.h, z0.h
    fmax z15.h, p4/M, z15.h, z0.h
    fmax z16.h, p4/M, z16.h, z0.h
    fmax z17.h, p4/M, z17.h, z0.h
    fmax z18.h, p4/M, z18.h, z0.h
    fmax z19.h, p4/M, z19.h, z0.h
    fmax z20.h, p4/M, z20.h, z0.h
    fmax z21.h, p4/M, z21.h, z0.h
    fmax z22.h, p4/M, z22.h, z0.h
    fmax z23.h, p4/M, z23.h, z0.h
    fmax z24.h, p4/M, z24.h, z0.h
    fmax z25.h, p4/M, z25.h, z0.h
    fmax z26.h, p4/M, z26.h, z0.h
    fmax z27.h, p4/M, z27.h, z0.h
    fmax z28.h, p4/M, z28.h, z0.h
    fmax z29.h, p4/M, z29.h, z0.h
    fmax z30.h, p4/M, z30.h, z0.h
    fmax z31.h, p4/M, z31.h, z0.h
KAI_ASM_LABEL(label_83)  // Height 6: No activation
    st1h { z8.h }, p3, [x27]
    st1h { z9.h }, p2, [x27, #1, MUL VL]
    st1h { z10.h }, p1, [x27, #2, MUL VL]
    st1h { z11.h }, p0, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1h { z12.h }, p3, [x7]
    st1h { z13.h }, p2, [x7, #1, MUL VL]
    st1h { z14.h }, p1, [x7, #2, MUL VL]
    st1h { z15.h }, p0, [x7, #3, MUL VL]
    st1h { z16.h }, p3, [x6]
    st1h { z17.h }, p2, [x6, #1, MUL VL]
    st1h { z18.h }, p1, [x6, #2, MUL VL]
    st1h { z19.h }, p0, [x6, #3, MUL VL]
    st1h { z20.h }, p3, [x5]
    st1h { z21.h }, p2, [x5, #1, MUL VL]
    st1h { z22.h }, p1, [x5, #2, MUL VL]
    st1h { z23.h }, p0, [x5, #3, MUL VL]
    st1h { z24.h }, p3, [x4]
    st1h { z25.h }, p2, [x4, #1, MUL VL]
    st1h { z26.h }, p1, [x4, #2, MUL VL]
    st1h { z27.h }, p0, [x4, #3, MUL VL]
    st1h { z28.h }, p3, [x3]
    st1h { z29.h }, p2, [x3, #1, MUL VL]
    st1h { z30.h }, p1, [x3, #2, MUL VL]
    st1h { z31.h }, p0, [x3, #3, MUL VL]
    dech x26, ALL, MUL #4
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffhybrid_fp16_mla_6x4VL_a64fx)

    KAI_ASM_END
