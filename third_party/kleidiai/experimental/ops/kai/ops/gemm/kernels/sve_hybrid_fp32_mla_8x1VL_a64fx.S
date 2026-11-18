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

    KAI_ASM_CODE(sve_hybrid_fp32_mla_8x1VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_fp32_mla_8x1VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_fp32_mla_8x1VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_fp32_mla_8x1VL_a64fx)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ptrue p1.b
    mov x9, #0
    ldr x4, [x0, #0x60]
    ldr w1, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x28, [x0, #0]
    cmp x4, #0
    cset x27, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x27, x27, x1, LSL #1
    cset x1, NE
    orr x27, x27, x2, LSL #1
    orr x27, x27, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x28, x9
    cmp x1, #0x8
    bge label_92
    cmp x1, #0x6
    bgt label_79
    beq label_66
    cmp x1, #0x4
    bgt label_53
    beq label_40
    cmp x1, #0x2
    bgt label_27
    beq label_14
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_3
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x27, #0, label_4
    ld1w { z24.s }, p0/Z, [x25]
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
    mov z24.b, #0
KAI_ASM_LABEL(label_5)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_6
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    b label_8
KAI_ASM_LABEL(label_6)  // Height 1: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_7)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    cbnz x26, label_8
    add x21, x21, x3, LSL #2
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    tbz x27, #3, label_11
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_7
KAI_ASM_LABEL(label_11)  // Height 1: no string loop
    tbz x27, #1, label_12
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
KAI_ASM_LABEL(label_12)  // Height 1: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    decw x24
    cmp x24, XZR
    bgt label_2
    b label_105
KAI_ASM_LABEL(label_14)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_15)  // Height 2: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_16
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    b label_18
KAI_ASM_LABEL(label_16)  // Height 2: no bias
    tbz x27, #0, label_17
    ldr x1, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x1
    ld1w { z25.s }, p0/Z, [x1]
    b label_18
KAI_ASM_LABEL(label_17)  // Height 2: no accumulate
    mov z24.b, #0
    mov z25.b, #0
KAI_ASM_LABEL(label_18)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_19
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    b label_21
KAI_ASM_LABEL(label_19)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_20)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    cbnz x26, label_21
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_21)  // Height 2: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    ble label_23
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    addvl x23, x23, #1
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    fmla z25.s, p1/M, z8.s, z1.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    bgt label_22
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    tbz x27, #3, label_24
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_20
KAI_ASM_LABEL(label_24)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    tbz x27, #1, label_25
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
KAI_ASM_LABEL(label_25)  // Height 2: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    decw x24
    cmp x24, XZR
    bgt label_15
    b label_105
KAI_ASM_LABEL(label_27)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_28)  // Height 3: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_29
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    b label_31
KAI_ASM_LABEL(label_29)  // Height 3: no bias
    tbz x27, #0, label_30
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p0/Z, [x1]
    b label_31
KAI_ASM_LABEL(label_30)  // Height 3: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
KAI_ASM_LABEL(label_31)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_32
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    b label_34
KAI_ASM_LABEL(label_32)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_33)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    cbnz x26, label_34
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
KAI_ASM_LABEL(label_34)  // Height 3: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    ble label_36
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    addvl x23, x23, #1
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    bgt label_35
KAI_ASM_LABEL(label_36)  // Height 3: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    tbz x27, #3, label_37
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_33
KAI_ASM_LABEL(label_37)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    tbz x27, #1, label_38
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
KAI_ASM_LABEL(label_38)  // Height 3: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    decw x24
    cmp x24, XZR
    bgt label_28
    b label_105
KAI_ASM_LABEL(label_40)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_41)  // Height 4: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_42
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    b label_44
KAI_ASM_LABEL(label_42)  // Height 4: no bias
    tbz x27, #0, label_43
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z27.s }, p0/Z, [x1]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 4: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
KAI_ASM_LABEL(label_44)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_45
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    b label_47
KAI_ASM_LABEL(label_45)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_46)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    cbnz x26, label_47
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_47)  // Height 4: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    ble label_49
KAI_ASM_LABEL(label_48)  // Height 4: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    add x6, x6, #0x4
    addvl x23, x23, #1
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    fmla z27.s, p1/M, z8.s, z3.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    bgt label_48
KAI_ASM_LABEL(label_49)  // Height 4: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    fmla z27.s, p1/M, z9.s, z3.s
    tbz x27, #3, label_50
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_46
KAI_ASM_LABEL(label_50)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    tbz x27, #1, label_51
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmin z27.s, p1/M, z27.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
    fmax z27.s, p1/M, z27.s, z16.s
KAI_ASM_LABEL(label_51)  // Height 4: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    st1w { z27.s }, p0, [x7]
    decw x24
    cmp x24, XZR
    bgt label_41
    b label_105
KAI_ASM_LABEL(label_53)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_54)  // Height 5: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_55
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    b label_57
KAI_ASM_LABEL(label_55)  // Height 5: no bias
    tbz x27, #0, label_56
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z27.s }, p0/Z, [x1]
    add x1, x1, x2
    ld1w { z28.s }, p0/Z, [x1]
    b label_57
KAI_ASM_LABEL(label_56)  // Height 5: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
KAI_ASM_LABEL(label_57)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_58
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    b label_60
KAI_ASM_LABEL(label_58)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_59)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    ldr x5, [x1, #0x20]
    cbnz x26, label_60
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_60)  // Height 5: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    ble label_62
KAI_ASM_LABEL(label_61)  // Height 5: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    add x6, x6, #0x4
    add x5, x5, #0x4
    addvl x23, x23, #1
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    fmla z27.s, p1/M, z8.s, z3.s
    fmla z28.s, p1/M, z8.s, z4.s
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    bgt label_61
KAI_ASM_LABEL(label_62)  // Height 5: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    fmla z27.s, p1/M, z9.s, z3.s
    fmla z28.s, p1/M, z9.s, z4.s
    tbz x27, #3, label_63
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_59
KAI_ASM_LABEL(label_63)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    tbz x27, #1, label_64
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmin z27.s, p1/M, z27.s, z17.s
    fmin z28.s, p1/M, z28.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
    fmax z27.s, p1/M, z27.s, z16.s
    fmax z28.s, p1/M, z28.s, z16.s
KAI_ASM_LABEL(label_64)  // Height 5: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    st1w { z27.s }, p0, [x7]
    st1w { z28.s }, p0, [x6]
    decw x24
    cmp x24, XZR
    bgt label_54
    b label_105
KAI_ASM_LABEL(label_66)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_67)  // Height 6: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_68
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    b label_70
KAI_ASM_LABEL(label_68)  // Height 6: no bias
    tbz x27, #0, label_69
    ldr x3, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x3
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x3
    ld1w { z26.s }, p0/Z, [x1]
    add x1, x1, x3
    ld1w { z27.s }, p0/Z, [x1]
    add x2, x1, x3
    add x1, x2, x3
    ld1w { z28.s }, p0/Z, [x2]
    ld1w { z29.s }, p0/Z, [x1]
    b label_70
KAI_ASM_LABEL(label_69)  // Height 6: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
KAI_ASM_LABEL(label_70)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_71
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_73
KAI_ASM_LABEL(label_71)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_72)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    ldr x5, [x1, #0x20]
    ldr x4, [x1, #0x28]
    cbnz x26, label_73
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_73)  // Height 6: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    ble label_75
KAI_ASM_LABEL(label_74)  // Height 6: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    add x6, x6, #0x4
    add x5, x5, #0x4
    add x4, x4, #0x4
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    addvl x23, x23, #1
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    fmla z27.s, p1/M, z8.s, z3.s
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    fmla z28.s, p1/M, z8.s, z4.s
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    fmla z29.s, p1/M, z8.s, z5.s
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    bgt label_74
KAI_ASM_LABEL(label_75)  // Height 6: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    fmla z27.s, p1/M, z9.s, z3.s
    fmla z28.s, p1/M, z9.s, z4.s
    fmla z29.s, p1/M, z9.s, z5.s
    tbz x27, #3, label_76
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_72
KAI_ASM_LABEL(label_76)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x27, #1, label_77
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmin z27.s, p1/M, z27.s, z17.s
    fmin z28.s, p1/M, z28.s, z17.s
    fmin z29.s, p1/M, z29.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
    fmax z27.s, p1/M, z27.s, z16.s
    fmax z28.s, p1/M, z28.s, z16.s
    fmax z29.s, p1/M, z29.s, z16.s
KAI_ASM_LABEL(label_77)  // Height 6: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    st1w { z27.s }, p0, [x7]
    st1w { z28.s }, p0, [x6]
    st1w { z29.s }, p0, [x5]
    decw x24
    cmp x24, XZR
    bgt label_67
    b label_105
KAI_ASM_LABEL(label_79)  // Height 7
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_80)  // Height 7: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_81
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    mov z30.d, z24.d
    b label_83
KAI_ASM_LABEL(label_81)  // Height 7: no bias
    tbz x27, #0, label_82
    ldr x4, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x4
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x4
    ld1w { z26.s }, p0/Z, [x1]
    add x1, x1, x4
    ld1w { z27.s }, p0/Z, [x1]
    add x3, x1, x4
    add x2, x3, x4
    add x1, x2, x4
    ld1w { z28.s }, p0/Z, [x3]
    ld1w { z29.s }, p0/Z, [x2]
    ld1w { z30.s }, p0/Z, [x1]
    b label_83
KAI_ASM_LABEL(label_82)  // Height 7: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
KAI_ASM_LABEL(label_83)  // Height 7: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_84
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_86
KAI_ASM_LABEL(label_84)  // Height 7: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_85)  // Height 7: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    ldr x5, [x1, #0x20]
    ldr x4, [x1, #0x28]
    ldr x2, [x1, #0x30]
    cbnz x26, label_86
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_86)  // Height 7: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    KAI_ASM_INST(0x8540c446)  // ld1rw { z6.s }, p1/Z, [x2]
    ble label_88
KAI_ASM_LABEL(label_87)  // Height 7: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    add x6, x6, #0x4
    add x5, x5, #0x4
    add x4, x4, #0x4
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    add x2, x2, #0x4
    addvl x23, x23, #1
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    fmla z27.s, p1/M, z8.s, z3.s
    fmla z28.s, p1/M, z8.s, z4.s
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    fmla z29.s, p1/M, z8.s, z5.s
    fmla z30.s, p1/M, z8.s, z6.s
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    KAI_ASM_INST(0x8540c446)  // ld1rw { z6.s }, p1/Z, [x2]
    bgt label_87
KAI_ASM_LABEL(label_88)  // Height 7: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    fmla z27.s, p1/M, z9.s, z3.s
    fmla z28.s, p1/M, z9.s, z4.s
    fmla z29.s, p1/M, z9.s, z5.s
    fmla z30.s, p1/M, z9.s, z6.s
    tbz x27, #3, label_89
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_85
KAI_ASM_LABEL(label_89)  // Height 7: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x27, #1, label_90
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmin z27.s, p1/M, z27.s, z17.s
    fmin z28.s, p1/M, z28.s, z17.s
    fmin z29.s, p1/M, z29.s, z17.s
    fmin z30.s, p1/M, z30.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
    fmax z27.s, p1/M, z27.s, z16.s
    fmax z28.s, p1/M, z28.s, z16.s
    fmax z29.s, p1/M, z29.s, z16.s
    fmax z30.s, p1/M, z30.s, z16.s
KAI_ASM_LABEL(label_90)  // Height 7: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    st1w { z27.s }, p0, [x7]
    st1w { z28.s }, p0, [x6]
    st1w { z29.s }, p0, [x5]
    st1w { z30.s }, p0, [x4]
    decw x24
    cmp x24, XZR
    bgt label_80
    b label_105
KAI_ASM_LABEL(label_92)  // Height 8
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_93)  // Height 8: Column loop
    mov x1, #0
    whilelt p0.s, x1, x24
    cbz x22, label_94
    ld1w { z24.s }, p0/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    mov z30.d, z24.d
    mov z31.d, z24.d
    b label_96
KAI_ASM_LABEL(label_94)  // Height 8: no bias
    tbz x27, #0, label_95
    ldr x4, [x0, #0x58]
    ld1w { z24.s }, p0/Z, [x25]
    add x1, x25, x4
    ld1w { z25.s }, p0/Z, [x1]
    add x1, x1, x4
    ld1w { z26.s }, p0/Z, [x1]
    add x1, x1, x4
    ld1w { z27.s }, p0/Z, [x1]
    add x3, x1, x4
    add x1, x3, x4
    add x2, x1, x4
    ld1w { z28.s }, p0/Z, [x3]
    ld1w { z29.s }, p0/Z, [x1]
    add x1, x2, x4
    ld1w { z30.s }, p0/Z, [x2]
    ld1w { z31.s }, p0/Z, [x1]
    b label_96
KAI_ASM_LABEL(label_95)  // Height 8: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_96)  // Height 8: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_97
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_99
KAI_ASM_LABEL(label_97)  // Height 8: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_98)  // Height 8: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    ldr x5, [x1, #0x20]
    ldr x4, [x1, #0x28]
    ldr x2, [x1, #0x30]
    ldr x1, [x1, #0x38]
    cbnz x26, label_99
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_99)  // Height 8: input setup done
    cmp x20, #0x1
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    KAI_ASM_INST(0x8540c446)  // ld1rw { z6.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c427)  // ld1rw { z7.s }, p1/Z, [x1]
    ble label_101
KAI_ASM_LABEL(label_100)  // Height 8: Multiply loop: Main loop
    ld1w { z8.s }, p1/Z, [x23]
    sub x20, x20, #0x1
    add x21, x21, #0x4
    cmp x20, #0x1
    add x8, x8, #0x4
    add x7, x7, #0x4
    add x6, x6, #0x4
    add x5, x5, #0x4
    add x4, x4, #0x4
    fmla z24.s, p1/M, z8.s, z0.s
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    add x2, x2, #0x4
    add x1, x1, #0x4
    fmla z25.s, p1/M, z8.s, z1.s
    fmla z26.s, p1/M, z8.s, z2.s
    addvl x23, x23, #1
    KAI_ASM_INST(0x8540c501)  // ld1rw { z1.s }, p1/Z, [x8]
    fmla z27.s, p1/M, z8.s, z3.s
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    fmla z28.s, p1/M, z8.s, z4.s
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    fmla z29.s, p1/M, z8.s, z5.s
    KAI_ASM_INST(0x8540c4a4)  // ld1rw { z4.s }, p1/Z, [x5]
    fmla z30.s, p1/M, z8.s, z6.s
    KAI_ASM_INST(0x8540c485)  // ld1rw { z5.s }, p1/Z, [x4]
    fmla z31.s, p1/M, z8.s, z7.s
    KAI_ASM_INST(0x8540c446)  // ld1rw { z6.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c427)  // ld1rw { z7.s }, p1/Z, [x1]
    bgt label_100
KAI_ASM_LABEL(label_101)  // Height 8: Multiply loop: Main loop skip
    ld1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, p1/M, z9.s, z0.s
    fmla z25.s, p1/M, z9.s, z1.s
    fmla z26.s, p1/M, z9.s, z2.s
    fmla z27.s, p1/M, z9.s, z3.s
    fmla z28.s, p1/M, z9.s, z4.s
    fmla z29.s, p1/M, z9.s, z5.s
    fmla z30.s, p1/M, z9.s, z6.s
    fmla z31.s, p1/M, z9.s, z7.s
    tbz x27, #3, label_102
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_98
KAI_ASM_LABEL(label_102)  // Height 8: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x27, #1, label_103
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    fmin z24.s, p1/M, z24.s, z17.s
    fmin z25.s, p1/M, z25.s, z17.s
    fmin z26.s, p1/M, z26.s, z17.s
    fmin z27.s, p1/M, z27.s, z17.s
    fmin z28.s, p1/M, z28.s, z17.s
    fmin z29.s, p1/M, z29.s, z17.s
    fmin z30.s, p1/M, z30.s, z17.s
    fmin z31.s, p1/M, z31.s, z17.s
    fmax z24.s, p1/M, z24.s, z16.s
    fmax z25.s, p1/M, z25.s, z16.s
    fmax z26.s, p1/M, z26.s, z16.s
    fmax z27.s, p1/M, z27.s, z16.s
    fmax z28.s, p1/M, z28.s, z16.s
    fmax z29.s, p1/M, z29.s, z16.s
    fmax z30.s, p1/M, z30.s, z16.s
    fmax z31.s, p1/M, z31.s, z16.s
KAI_ASM_LABEL(label_103)  // Height 8: No activation
    st1w { z24.s }, p0, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p0, [x20]
    st1w { z26.s }, p0, [x8]
    st1w { z27.s }, p0, [x7]
    st1w { z28.s }, p0, [x6]
    st1w { z29.s }, p0, [x5]
    st1w { z30.s }, p0, [x4]
    st1w { z31.s }, p0, [x3]
    decw x24
    cmp x24, XZR
    bgt label_93
    add x9, x9, #0x8
    cmp x9, x28
    blt label_1
KAI_ASM_LABEL(label_105)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_fp32_mla_8x1VL_a64fx)

    KAI_ASM_END
