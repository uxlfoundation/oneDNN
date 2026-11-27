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

    KAI_ASM_CODE(sve_hybrid_fp32_mla_8x1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_hybrid_fp32_mla_8x1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_hybrid_fp32_mla_8x1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_hybrid_fp32_mla_8x1VL)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ptrue p2.b
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
    bge label_99
    cmp x1, #0x6
    bgt label_85
    beq label_71
    cmp x1, #0x4
    bgt label_57
    beq label_43
    cmp x1, #0x2
    bgt label_29
    beq label_15
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_3
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x27, #0, label_4
    ld1w { z24.s }, p1/Z, [x25]
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
    cmp x20, #0x4
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z24.s, z11.s, z0.s[3]
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    fmla z24.s, z8.s, z0.s[0]
    ble label_11
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    ble label_11
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    ble label_11
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: multiply skip
    tbz x27, #3, label_12
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_7
KAI_ASM_LABEL(label_12)  // Height 1: no string loop
    tbz x27, #1, label_13
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
KAI_ASM_LABEL(label_13)  // Height 1: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    decw x24
    cmp x24, XZR
    bgt label_2
    b label_113
KAI_ASM_LABEL(label_15)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_16)  // Height 2: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_17
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    b label_19
KAI_ASM_LABEL(label_17)  // Height 2: no bias
    tbz x27, #0, label_18
    ldr x1, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x1
    ld1w { z25.s }, p1/Z, [x1]
    b label_19
KAI_ASM_LABEL(label_18)  // Height 2: no accumulate
    mov z24.b, #0
    mov z25.b, #0
KAI_ASM_LABEL(label_19)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_20
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    b label_22
KAI_ASM_LABEL(label_20)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_21)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    cbnz x26, label_22
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_22)  // Height 2: input setup done
    cmp x20, #0x4
    ble label_24
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    ble label_25
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    ble label_25
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    ble label_25
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: multiply skip
    tbz x27, #3, label_26
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_21
KAI_ASM_LABEL(label_26)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    tbz x27, #1, label_27
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
KAI_ASM_LABEL(label_27)  // Height 2: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    decw x24
    cmp x24, XZR
    bgt label_16
    b label_113
KAI_ASM_LABEL(label_29)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_30)  // Height 3: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_31
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    b label_33
KAI_ASM_LABEL(label_31)  // Height 3: no bias
    tbz x27, #0, label_32
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p1/Z, [x1]
    b label_33
KAI_ASM_LABEL(label_32)  // Height 3: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
KAI_ASM_LABEL(label_33)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_34
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    b label_36
KAI_ASM_LABEL(label_34)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_35)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    cbnz x26, label_36
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
KAI_ASM_LABEL(label_36)  // Height 3: input setup done
    cmp x20, #0x4
    ble label_38
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    add x7, x7, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    bgt label_37
KAI_ASM_LABEL(label_38)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    ble label_39
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    ble label_39
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    ble label_39
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
KAI_ASM_LABEL(label_39)  // Height 3: Multiply loop: multiply skip
    tbz x27, #3, label_40
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_35
KAI_ASM_LABEL(label_40)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    tbz x27, #1, label_41
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
KAI_ASM_LABEL(label_41)  // Height 3: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    decw x24
    cmp x24, XZR
    bgt label_30
    b label_113
KAI_ASM_LABEL(label_43)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_44)  // Height 4: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_45
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    b label_47
KAI_ASM_LABEL(label_45)  // Height 4: no bias
    tbz x27, #0, label_46
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z27.s }, p1/Z, [x1]
    b label_47
KAI_ASM_LABEL(label_46)  // Height 4: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
KAI_ASM_LABEL(label_47)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_48
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    b label_50
KAI_ASM_LABEL(label_48)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_49)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    cbnz x26, label_50
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_50)  // Height 4: input setup done
    cmp x20, #0x4
    ble label_52
KAI_ASM_LABEL(label_51)  // Height 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    add x7, x7, #0x10
    add x6, x6, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    bgt label_51
KAI_ASM_LABEL(label_52)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    ble label_53
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    ble label_53
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    ble label_53
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
KAI_ASM_LABEL(label_53)  // Height 4: Multiply loop: multiply skip
    tbz x27, #3, label_54
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_49
KAI_ASM_LABEL(label_54)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    tbz x27, #1, label_55
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
KAI_ASM_LABEL(label_55)  // Height 4: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    st1w { z27.s }, p1, [x7]
    decw x24
    cmp x24, XZR
    bgt label_44
    b label_113
KAI_ASM_LABEL(label_57)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_58)  // Height 5: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_59
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    b label_61
KAI_ASM_LABEL(label_59)  // Height 5: no bias
    tbz x27, #0, label_60
    ldr x2, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x2
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z26.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z27.s }, p1/Z, [x1]
    add x1, x1, x2
    ld1w { z28.s }, p1/Z, [x1]
    b label_61
KAI_ASM_LABEL(label_60)  // Height 5: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
KAI_ASM_LABEL(label_61)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_62
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    b label_64
KAI_ASM_LABEL(label_62)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_63)  // Height 5: String loop
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
    cbnz x26, label_64
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_64)  // Height 5: input setup done
    cmp x20, #0x4
    ble label_66
KAI_ASM_LABEL(label_65)  // Height 5: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    add x7, x7, #0x10
    add x6, x6, #0x10
    ld1rqw { z4.s }, p0/Z, [x5]
    add x5, x5, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    bgt label_65
KAI_ASM_LABEL(label_66)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    ld1rqw { z4.s }, p0/Z, [x5]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    ble label_67
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    ble label_67
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    ble label_67
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
KAI_ASM_LABEL(label_67)  // Height 5: Multiply loop: multiply skip
    tbz x27, #3, label_68
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_63
KAI_ASM_LABEL(label_68)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    tbz x27, #1, label_69
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
KAI_ASM_LABEL(label_69)  // Height 5: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    st1w { z27.s }, p1, [x7]
    st1w { z28.s }, p1, [x6]
    decw x24
    cmp x24, XZR
    bgt label_58
    b label_113
KAI_ASM_LABEL(label_71)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_72)  // Height 6: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_73
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    b label_75
KAI_ASM_LABEL(label_73)  // Height 6: no bias
    tbz x27, #0, label_74
    ldr x3, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x3
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x3
    ld1w { z26.s }, p1/Z, [x1]
    add x1, x1, x3
    ld1w { z27.s }, p1/Z, [x1]
    add x2, x1, x3
    add x1, x2, x3
    ld1w { z28.s }, p1/Z, [x2]
    ld1w { z29.s }, p1/Z, [x1]
    b label_75
KAI_ASM_LABEL(label_74)  // Height 6: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
KAI_ASM_LABEL(label_75)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_76
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_78
KAI_ASM_LABEL(label_76)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_77)  // Height 6: String loop
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
    cbnz x26, label_78
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_78)  // Height 6: input setup done
    cmp x20, #0x4
    ble label_80
KAI_ASM_LABEL(label_79)  // Height 6: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    add x7, x7, #0x10
    add x6, x6, #0x10
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
    bgt label_79
KAI_ASM_LABEL(label_80)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    ble label_81
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    ble label_81
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    ble label_81
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
KAI_ASM_LABEL(label_81)  // Height 6: Multiply loop: multiply skip
    tbz x27, #3, label_82
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_77
KAI_ASM_LABEL(label_82)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x27, #1, label_83
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
KAI_ASM_LABEL(label_83)  // Height 6: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    st1w { z27.s }, p1, [x7]
    st1w { z28.s }, p1, [x6]
    st1w { z29.s }, p1, [x5]
    decw x24
    cmp x24, XZR
    bgt label_72
    b label_113
KAI_ASM_LABEL(label_85)  // Height 7
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_86)  // Height 7: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_87
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    mov z30.d, z24.d
    b label_89
KAI_ASM_LABEL(label_87)  // Height 7: no bias
    tbz x27, #0, label_88
    ldr x4, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x4
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x4
    ld1w { z26.s }, p1/Z, [x1]
    add x1, x1, x4
    ld1w { z27.s }, p1/Z, [x1]
    add x3, x1, x4
    add x2, x3, x4
    add x1, x2, x4
    ld1w { z28.s }, p1/Z, [x3]
    ld1w { z29.s }, p1/Z, [x2]
    ld1w { z30.s }, p1/Z, [x1]
    b label_89
KAI_ASM_LABEL(label_88)  // Height 7: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
KAI_ASM_LABEL(label_89)  // Height 7: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_90
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_92
KAI_ASM_LABEL(label_90)  // Height 7: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_91)  // Height 7: String loop
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
    cbnz x26, label_92
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_92)  // Height 7: input setup done
    cmp x20, #0x4
    ble label_94
KAI_ASM_LABEL(label_93)  // Height 7: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    add x7, x7, #0x10
    add x6, x6, #0x10
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    add x5, x5, #0x10
    add x4, x4, #0x10
    ld1rqw { z6.s }, p0/Z, [x2]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    add x2, x2, #0x10
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    fmla z30.s, z8.s, z6.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    fmla z30.s, z9.s, z6.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    fmla z30.s, z10.s, z6.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
    fmla z30.s, z11.s, z6.s[3]
    bgt label_93
KAI_ASM_LABEL(label_94)  // Height 7: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    ld1rqw { z6.s }, p0/Z, [x2]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    fmla z30.s, z8.s, z6.s[0]
    ble label_95
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    fmla z30.s, z9.s, z6.s[1]
    ble label_95
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    fmla z30.s, z10.s, z6.s[2]
    ble label_95
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
    fmla z30.s, z11.s, z6.s[3]
KAI_ASM_LABEL(label_95)  // Height 7: Multiply loop: multiply skip
    tbz x27, #3, label_96
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_91
KAI_ASM_LABEL(label_96)  // Height 7: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x27, #1, label_97
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
KAI_ASM_LABEL(label_97)  // Height 7: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    st1w { z27.s }, p1, [x7]
    st1w { z28.s }, p1, [x6]
    st1w { z29.s }, p1, [x5]
    st1w { z30.s }, p1, [x4]
    decw x24
    cmp x24, XZR
    bgt label_86
    b label_113
KAI_ASM_LABEL(label_99)  // Height 8
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_100)  // Height 8: Column loop
    mov x1, #0
    whilelt p1.s, x1, x24
    cbz x22, label_101
    ld1w { z24.s }, p1/Z, [x22]
    addvl x22, x22, #1
    mov z25.d, z24.d
    mov z26.d, z24.d
    mov z27.d, z24.d
    mov z28.d, z24.d
    mov z29.d, z24.d
    mov z30.d, z24.d
    mov z31.d, z24.d
    b label_103
KAI_ASM_LABEL(label_101)  // Height 8: no bias
    tbz x27, #0, label_102
    ldr x4, [x0, #0x58]
    ld1w { z24.s }, p1/Z, [x25]
    add x1, x25, x4
    ld1w { z25.s }, p1/Z, [x1]
    add x1, x1, x4
    ld1w { z26.s }, p1/Z, [x1]
    add x1, x1, x4
    ld1w { z27.s }, p1/Z, [x1]
    add x3, x1, x4
    add x1, x3, x4
    add x2, x1, x4
    ld1w { z28.s }, p1/Z, [x3]
    ld1w { z29.s }, p1/Z, [x1]
    add x1, x2, x4
    ld1w { z30.s }, p1/Z, [x2]
    ld1w { z31.s }, p1/Z, [x1]
    b label_103
KAI_ASM_LABEL(label_102)  // Height 8: no accumulate
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_103)  // Height 8: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_104
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
    b label_106
KAI_ASM_LABEL(label_104)  // Height 8: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_105)  // Height 8: String loop
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
    cbnz x26, label_106
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_106)  // Height 8: input setup done
    cmp x20, #0x4
    ble label_108
KAI_ASM_LABEL(label_107)  // Height 8: Multiply loop: Main loop head
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    ld1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    sub x20, x20, #0x4
    ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z11.s }, p2/Z, [x23, #3, MUL VL]
    cmp x20, #0x4
    addvl x23, x23, #4
    ld1rqw { z0.s }, p0/Z, [x21]
    add x21, x21, #0x10
    ld1rqw { z1.s }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    add x7, x7, #0x10
    add x6, x6, #0x10
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    add x5, x5, #0x10
    add x4, x4, #0x10
    ld1rqw { z6.s }, p0/Z, [x2]
    ld1rqw { z7.s }, p0/Z, [x1]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    fmla z30.s, z8.s, z6.s[0]
    fmla z31.s, z8.s, z7.s[0]
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    fmla z30.s, z9.s, z6.s[1]
    fmla z31.s, z9.s, z7.s[1]
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    fmla z30.s, z10.s, z6.s[2]
    fmla z31.s, z10.s, z7.s[2]
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
    fmla z30.s, z11.s, z6.s[3]
    fmla z31.s, z11.s, z7.s[3]
    bgt label_107
KAI_ASM_LABEL(label_108)  // Height 8: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x20
    ld1w { z8.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    ld1rqw { z0.s }, p0/Z, [x21]
    ld1rqw { z1.s }, p0/Z, [x8]
    ld1rqw { z2.s }, p0/Z, [x7]
    ld1rqw { z3.s }, p0/Z, [x6]
    ld1rqw { z4.s }, p0/Z, [x5]
    ld1rqw { z5.s }, p0/Z, [x4]
    ld1rqw { z6.s }, p0/Z, [x2]
    ld1rqw { z7.s }, p0/Z, [x1]
    fmla z24.s, z8.s, z0.s[0]
    fmla z25.s, z8.s, z1.s[0]
    fmla z26.s, z8.s, z2.s[0]
    fmla z27.s, z8.s, z3.s[0]
    fmla z28.s, z8.s, z4.s[0]
    fmla z29.s, z8.s, z5.s[0]
    fmla z30.s, z8.s, z6.s[0]
    fmla z31.s, z8.s, z7.s[0]
    ble label_109
    ld1w { z9.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z9.s, z0.s[1]
    fmla z25.s, z9.s, z1.s[1]
    fmla z26.s, z9.s, z2.s[1]
    fmla z27.s, z9.s, z3.s[1]
    fmla z28.s, z9.s, z4.s[1]
    fmla z29.s, z9.s, z5.s[1]
    fmla z30.s, z9.s, z6.s[1]
    fmla z31.s, z9.s, z7.s[1]
    ble label_109
    ld1w { z10.s }, p2/Z, [x23]
    subs x20, x20, #0x1
    addvl x23, x23, #1
    fmla z24.s, z10.s, z0.s[2]
    fmla z25.s, z10.s, z1.s[2]
    fmla z26.s, z10.s, z2.s[2]
    fmla z27.s, z10.s, z3.s[2]
    fmla z28.s, z10.s, z4.s[2]
    fmla z29.s, z10.s, z5.s[2]
    fmla z30.s, z10.s, z6.s[2]
    fmla z31.s, z10.s, z7.s[2]
    ble label_109
    ld1w { z11.s }, p2/Z, [x23]
    addvl x23, x23, #1
    fmla z24.s, z11.s, z0.s[3]
    fmla z25.s, z11.s, z1.s[3]
    fmla z26.s, z11.s, z2.s[3]
    fmla z27.s, z11.s, z3.s[3]
    fmla z28.s, z11.s, z4.s[3]
    fmla z29.s, z11.s, z5.s[3]
    fmla z30.s, z11.s, z6.s[3]
    fmla z31.s, z11.s, z7.s[3]
KAI_ASM_LABEL(label_109)  // Height 8: Multiply loop: multiply skip
    tbz x27, #3, label_110
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_105
KAI_ASM_LABEL(label_110)  // Height 8: no string loop
    ldr x1, [x0, #0x58]
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x27, #1, label_111
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmin z31.s, p2/M, z31.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
    fmax z31.s, p2/M, z31.s, z16.s
KAI_ASM_LABEL(label_111)  // Height 8: No activation
    st1w { z24.s }, p1, [x25]
    addvl x25, x25, #1
    st1w { z25.s }, p1, [x20]
    st1w { z26.s }, p1, [x8]
    st1w { z27.s }, p1, [x7]
    st1w { z28.s }, p1, [x6]
    st1w { z29.s }, p1, [x5]
    st1w { z30.s }, p1, [x4]
    st1w { z31.s }, p1, [x3]
    decw x24
    cmp x24, XZR
    bgt label_100
    add x9, x9, #0x8
    cmp x9, x28
    blt label_1
KAI_ASM_LABEL(label_113)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_hybrid_fp32_mla_8x1VL)

    KAI_ASM_END
