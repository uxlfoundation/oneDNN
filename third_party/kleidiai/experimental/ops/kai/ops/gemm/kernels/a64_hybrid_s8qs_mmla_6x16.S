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

    KAI_ASM_CODE(a64_hybrid_s8qs_mmla_6x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_s8qs_mmla_6x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_s8qs_mmla_6x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_s8qs_mmla_6x16)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x9, #0
    ldr x2, [x0, #0x60]
    ldr x1, [x0, #0x18]
    cmp x2, #0
    cset x28, NE
    cmp x1, #0
    ldr x1, [x0, #0x78]
    cset x2, NE
    ldr x27, [x0, #0]
    cmp x1, #0
    cset x1, NE
    orr x28, x28, x2, LSL #3
    orr x28, x28, x1, LSL #4
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x27, x9
    cmp x1, #0x6
    bge label_146
    cmp x1, #0x4
    bgt label_117
    beq label_88
    cmp x1, #0x2
    bgt label_59
    beq label_30
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_4
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_5)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    cbnz x26, label_6
    add x8, x8, x3
KAI_ASM_LABEL(label_6)  // Height 1: input setup done
    cmp x7, #0x10
    blt label_9
    ldr q1, [x8, #0]
    ldr q7, [x21, #0]
    cmp x7, #0x20
    ldr q6, [x21, #0x10]
    blt label_8
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Main loop head
    sub x7, x7, #0x10
    add x8, x8, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    ldr q6, [x21, #0x10]
    bge label_7
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: Single iteration only
    add x8, x8, #0x10
    sub x7, x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_16
    cmp x7, #0x8
    blt label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr q6, [x21, #0]
    sub x7, x7, #0x8
    ldr q7, [x21, #0x10]
    cmp x7, #0x8
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    bge label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Skip odd blocks
    cbz x7, label_16
    tbz x7, #2, label_13
    ldr s1, [x8], #0x4
    tbz x7, #1, label_12
    ld1 { v1.h }[2], [x8], #0x2
    tbz x7, #0, label_15
    ld1 { v1.b }[6], [x8]
    b label_15
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_15
    ld1 { v1.b }[4], [x8]
    b label_15
KAI_ASM_LABEL(label_13)  // Height 1: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_14
    ldr h1, [x8], #0x2
    tbz x7, #0, label_15
    ld1 { v1.b }[2], [x8]
    b label_15
KAI_ASM_LABEL(label_14)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
KAI_ASM_LABEL(label_15)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
KAI_ASM_LABEL(label_16)  // Height 1: Multiply loop: No odd multiplies
    tbz x28, #3, label_17
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_5
KAI_ASM_LABEL(label_17)  // Height 1: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v8.2d, v8.2d, v12.2d
    uzp1 v9.2d, v9.2d, v13.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v10.2d, v10.2d, v14.2d
    uzp1 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x25, #0]
    add x20, x20, #0x40
    mov v15.16b, v8.16b
    add v9.4s, v9.4s, v1.4s
    add v15.4s, v15.4s, v0.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    tbz x28, #4, label_18
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_19
KAI_ASM_LABEL(label_18)  // Height 1: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_19)  // Height 1: parameters loaded
    sqdmulh v15.4s, v15.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    add x2, x0, #0x94
    add x1, x0, #0xa4
    ld1r { v4.4s }, [x2]
    sqdmulh v10.4s, v10.4s, v6.4s
    ld1r { v6.4s }, [x1]
    sqdmulh v11.4s, v11.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    ld1r { v5.4s }, [x1]
    srshl v15.4s, v15.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    add v15.4s, v15.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    smin v15.4s, v15.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smax v15.4s, v15.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    uzp1 v15.8h, v15.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v15.16b, v15.16b, v9.16b
    bge label_28
    tbz x22, #3, label_23
    str d15, [x25], #0x8
    tbz x22, #2, label_21
    st1 { v15.s }[2], [x25], #0x4
    tbz x22, #1, label_20
    st1 { v15.h }[6], [x25], #0x2
    tbz x22, #0, label_27
    st1 { v15.b }[14], [x25]
    b label_27
KAI_ASM_LABEL(label_20)  // Height 1: Partial direct writeback: partial_0_12
    tbz x22, #0, label_27
    st1 { v15.b }[12], [x25]
    b label_27
KAI_ASM_LABEL(label_21)  // Height 1: Partial direct writeback: partial_1_8
    tbz x22, #1, label_22
    st1 { v15.h }[4], [x25], #0x2
    tbz x22, #0, label_27
    st1 { v15.b }[10], [x25]
    b label_27
KAI_ASM_LABEL(label_22)  // Height 1: Partial direct writeback: partial_0_8
    tbz x22, #0, label_27
    st1 { v15.b }[8], [x25]
    b label_27
KAI_ASM_LABEL(label_23)  // Height 1: Partial direct writeback: partial_2_0
    tbz x22, #2, label_25
    str s15, [x25], #0x4
    tbz x22, #1, label_24
    st1 { v15.h }[2], [x25], #0x2
    tbz x22, #0, label_27
    st1 { v15.b }[6], [x25]
    b label_27
KAI_ASM_LABEL(label_24)  // Height 1: Partial direct writeback: partial_0_4
    tbz x22, #0, label_27
    st1 { v15.b }[4], [x25]
    b label_27
KAI_ASM_LABEL(label_25)  // Height 1: Partial direct writeback: partial_1_0
    tbz x22, #1, label_26
    str h15, [x25], #0x2
    tbz x22, #0, label_27
    st1 { v15.b }[2], [x25]
    b label_27
KAI_ASM_LABEL(label_26)  // Height 1: Partial direct writeback: partial_0_0
    str b15, [x25, #0]
KAI_ASM_LABEL(label_27)  // Height 1: Partial direct writeback: Done
    b label_29
KAI_ASM_LABEL(label_28)  // Height 1: Full writeback
    str q15, [x25, #0]
    add x25, x25, #0x10
KAI_ASM_LABEL(label_29)  // Height 1: Writeback done
    subs x22, x22, #0x10
    bgt label_2
    b label_175
KAI_ASM_LABEL(label_30)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_31)  // Height 2: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_33
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    b label_35
KAI_ASM_LABEL(label_33)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_34)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x26, label_35
    add x8, x8, x3
    add x6, x6, x3
KAI_ASM_LABEL(label_35)  // Height 2: input setup done
    cmp x7, #0x10
    blt label_38
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_37
KAI_ASM_LABEL(label_36)  // Height 2: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    add x8, x8, #0x10
    add x6, x6, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    ldr q2, [x6, #0]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    ldr q6, [x21, #0x10]
    bge label_36
KAI_ASM_LABEL(label_37)  // Height 2: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    sub x7, x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
KAI_ASM_LABEL(label_38)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_45
    cmp x7, #0x8
    blt label_40
KAI_ASM_LABEL(label_39)  // Height 2: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    cmp x7, #0x8
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    bge label_39
KAI_ASM_LABEL(label_40)  // Height 2: Multiply loop: Skip odd blocks
    cbz x7, label_45
    tbz x7, #2, label_42
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    tbz x7, #1, label_41
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    tbz x7, #0, label_44
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    b label_44
KAI_ASM_LABEL(label_41)  // Height 2: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_44
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    b label_44
KAI_ASM_LABEL(label_42)  // Height 2: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_43
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    tbz x7, #0, label_44
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
KAI_ASM_LABEL(label_44)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
KAI_ASM_LABEL(label_45)  // Height 2: Multiply loop: No odd multiplies
    tbz x28, #3, label_46
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_34
KAI_ASM_LABEL(label_46)  // Height 2: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    ldr x1, [x0, #0x58]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    prfm pstl1keep, [x25, #0]
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x20, x20, #0x40
    mov v15.16b, v7.16b
    add v12.4s, v12.4s, v1.4s
    add x8, x25, x1
    prfm pstl1keep, [x8, #0]
    add v13.4s, v13.4s, v2.4s
    add v8.4s, v8.4s, v0.4s
    add v15.4s, v15.4s, v0.4s
    add v14.4s, v14.4s, v3.4s
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    tbz x28, #4, label_47
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_48
KAI_ASM_LABEL(label_47)  // Height 2: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_48)  // Height 2: parameters loaded
    sqdmulh v15.4s, v15.4s, v4.4s
    sqdmulh v12.4s, v12.4s, v5.4s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    sqdmulh v13.4s, v13.4s, v6.4s
    sqdmulh v14.4s, v14.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    sqdmulh v8.4s, v8.4s, v4.4s
    ld1r { v4.4s }, [x3]
    sqdmulh v9.4s, v9.4s, v5.4s
    ld1r { v5.4s }, [x1]
    sqdmulh v10.4s, v10.4s, v6.4s
    ld1r { v6.4s }, [x2]
    sqdmulh v11.4s, v11.4s, v7.4s
    srshl v15.4s, v15.4s, v0.4s
    srshl v12.4s, v12.4s, v1.4s
    srshl v13.4s, v13.4s, v2.4s
    srshl v14.4s, v14.4s, v3.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    add v15.4s, v15.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    smin v15.4s, v15.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smax v15.4s, v15.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    uzp1 v15.8h, v15.8h, v12.8h
    uzp1 v12.8h, v13.8h, v14.8h
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v15.16b, v15.16b, v12.16b
    uzp1 v8.16b, v8.16b, v9.16b
    bge label_57
    tbz x22, #3, label_52
    str d15, [x25], #0x8
    str d8, [x8], #0x8
    tbz x22, #2, label_50
    st1 { v15.s }[2], [x25], #0x4
    st1 { v8.s }[2], [x8], #0x4
    tbz x22, #1, label_49
    st1 { v15.h }[6], [x25], #0x2
    st1 { v8.h }[6], [x8], #0x2
    tbz x22, #0, label_56
    st1 { v15.b }[14], [x25]
    st1 { v8.b }[14], [x8]
    b label_56
KAI_ASM_LABEL(label_49)  // Height 2: Partial direct writeback: partial_0_12
    tbz x22, #0, label_56
    st1 { v15.b }[12], [x25]
    st1 { v8.b }[12], [x8]
    b label_56
KAI_ASM_LABEL(label_50)  // Height 2: Partial direct writeback: partial_1_8
    tbz x22, #1, label_51
    st1 { v15.h }[4], [x25], #0x2
    st1 { v8.h }[4], [x8], #0x2
    tbz x22, #0, label_56
    st1 { v15.b }[10], [x25]
    st1 { v8.b }[10], [x8]
    b label_56
KAI_ASM_LABEL(label_51)  // Height 2: Partial direct writeback: partial_0_8
    tbz x22, #0, label_56
    st1 { v15.b }[8], [x25]
    st1 { v8.b }[8], [x8]
    b label_56
KAI_ASM_LABEL(label_52)  // Height 2: Partial direct writeback: partial_2_0
    tbz x22, #2, label_54
    str s15, [x25], #0x4
    str s8, [x8], #0x4
    tbz x22, #1, label_53
    st1 { v15.h }[2], [x25], #0x2
    st1 { v8.h }[2], [x8], #0x2
    tbz x22, #0, label_56
    st1 { v15.b }[6], [x25]
    st1 { v8.b }[6], [x8]
    b label_56
KAI_ASM_LABEL(label_53)  // Height 2: Partial direct writeback: partial_0_4
    tbz x22, #0, label_56
    st1 { v15.b }[4], [x25]
    st1 { v8.b }[4], [x8]
    b label_56
KAI_ASM_LABEL(label_54)  // Height 2: Partial direct writeback: partial_1_0
    tbz x22, #1, label_55
    str h15, [x25], #0x2
    str h8, [x8], #0x2
    tbz x22, #0, label_56
    st1 { v15.b }[2], [x25]
    st1 { v8.b }[2], [x8]
    b label_56
KAI_ASM_LABEL(label_55)  // Height 2: Partial direct writeback: partial_0_0
    str b15, [x25, #0]
    str b8, [x8, #0]
KAI_ASM_LABEL(label_56)  // Height 2: Partial direct writeback: Done
    b label_58
KAI_ASM_LABEL(label_57)  // Height 2: Full writeback
    str q15, [x25, #0]
    add x25, x25, #0x10
    str q8, [x8, #0]
KAI_ASM_LABEL(label_58)  // Height 2: Writeback done
    subs x22, x22, #0x10
    bgt label_31
    b label_175
KAI_ASM_LABEL(label_59)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_60)  // Height 3: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_62
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_64
KAI_ASM_LABEL(label_62)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_63)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x26, label_64
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
KAI_ASM_LABEL(label_64)  // Height 3: input setup done
    cmp x7, #0x10
    blt label_67
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_66
KAI_ASM_LABEL(label_65)  // Height 3: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    add x8, x8, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    ldr q6, [x21, #0x10]
    bge label_65
KAI_ASM_LABEL(label_66)  // Height 3: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    prfm pldl1keep, [x8, #0x80]
    sub x7, x7, #0x10
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
KAI_ASM_LABEL(label_67)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_74
    cmp x7, #0x8
    blt label_69
KAI_ASM_LABEL(label_68)  // Height 3: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr d3, [x5], #0x8
    ldr q6, [x21, #0]
    cmp x7, #0x8
    ldr q7, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a450)  // smmla v16.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a454)  // smmla v20.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a451)  // smmla v17.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a455)  // smmla v21.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a452)  // smmla v18.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a456)  // smmla v22.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a453)  // smmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    bge label_68
KAI_ASM_LABEL(label_69)  // Height 3: Multiply loop: Skip odd blocks
    cbz x7, label_74
    tbz x7, #2, label_71
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    tbz x7, #1, label_70
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    tbz x7, #0, label_73
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    b label_73
KAI_ASM_LABEL(label_70)  // Height 3: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_73
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    b label_73
KAI_ASM_LABEL(label_71)  // Height 3: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_72
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    tbz x7, #0, label_73
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_72)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
KAI_ASM_LABEL(label_73)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
KAI_ASM_LABEL(label_74)  // Height 3: Multiply loop: No odd multiplies
    tbz x28, #3, label_75
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_63
KAI_ASM_LABEL(label_75)  // Height 3: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    ldr x1, [x0, #0x58]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    prfm pstl1keep, [x25, #0]
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x20, x20, #0x40
    uzp1 v16.2d, v16.2d, v20.2d
    uzp1 v17.2d, v17.2d, v21.2d
    uzp1 v18.2d, v18.2d, v22.2d
    uzp1 v19.2d, v19.2d, v23.2d
    add x8, x25, x1
    mov v23.16b, v7.16b
    add x7, x8, x1
    prfm pstl1keep, [x8, #0]
    add v12.4s, v12.4s, v1.4s
    prfm pstl1keep, [x7, #0]
    add v13.4s, v13.4s, v2.4s
    add v14.4s, v14.4s, v3.4s
    add v23.4s, v23.4s, v0.4s
    add v8.4s, v8.4s, v0.4s
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    tbz x28, #4, label_76
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_77
KAI_ASM_LABEL(label_76)  // Height 3: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_77)  // Height 3: parameters loaded
    sqdmulh v23.4s, v23.4s, v4.4s
    sqdmulh v12.4s, v12.4s, v5.4s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    sqdmulh v13.4s, v13.4s, v6.4s
    sqdmulh v14.4s, v14.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    ld1r { v4.4s }, [x3]
    sqdmulh v17.4s, v17.4s, v5.4s
    ld1r { v5.4s }, [x1]
    sqdmulh v18.4s, v18.4s, v6.4s
    ld1r { v6.4s }, [x2]
    sqdmulh v19.4s, v19.4s, v7.4s
    srshl v23.4s, v23.4s, v0.4s
    srshl v12.4s, v12.4s, v1.4s
    srshl v13.4s, v13.4s, v2.4s
    srshl v14.4s, v14.4s, v3.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    add v23.4s, v23.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    smin v23.4s, v23.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smax v23.4s, v23.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    uzp1 v23.8h, v23.8h, v12.8h
    uzp1 v12.8h, v13.8h, v14.8h
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v23.16b, v23.16b, v12.16b
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v16.16b, v16.16b, v17.16b
    bge label_86
    tbz x22, #3, label_81
    str d23, [x25], #0x8
    str d8, [x8], #0x8
    str d16, [x7], #0x8
    tbz x22, #2, label_79
    st1 { v23.s }[2], [x25], #0x4
    st1 { v8.s }[2], [x8], #0x4
    st1 { v16.s }[2], [x7], #0x4
    tbz x22, #1, label_78
    st1 { v23.h }[6], [x25], #0x2
    st1 { v8.h }[6], [x8], #0x2
    st1 { v16.h }[6], [x7], #0x2
    tbz x22, #0, label_85
    st1 { v23.b }[14], [x25]
    st1 { v8.b }[14], [x8]
    st1 { v16.b }[14], [x7]
    b label_85
KAI_ASM_LABEL(label_78)  // Height 3: Partial direct writeback: partial_0_12
    tbz x22, #0, label_85
    st1 { v23.b }[12], [x25]
    st1 { v8.b }[12], [x8]
    st1 { v16.b }[12], [x7]
    b label_85
KAI_ASM_LABEL(label_79)  // Height 3: Partial direct writeback: partial_1_8
    tbz x22, #1, label_80
    st1 { v23.h }[4], [x25], #0x2
    st1 { v8.h }[4], [x8], #0x2
    st1 { v16.h }[4], [x7], #0x2
    tbz x22, #0, label_85
    st1 { v23.b }[10], [x25]
    st1 { v8.b }[10], [x8]
    st1 { v16.b }[10], [x7]
    b label_85
KAI_ASM_LABEL(label_80)  // Height 3: Partial direct writeback: partial_0_8
    tbz x22, #0, label_85
    st1 { v23.b }[8], [x25]
    st1 { v8.b }[8], [x8]
    st1 { v16.b }[8], [x7]
    b label_85
KAI_ASM_LABEL(label_81)  // Height 3: Partial direct writeback: partial_2_0
    tbz x22, #2, label_83
    str s23, [x25], #0x4
    str s8, [x8], #0x4
    str s16, [x7], #0x4
    tbz x22, #1, label_82
    st1 { v23.h }[2], [x25], #0x2
    st1 { v8.h }[2], [x8], #0x2
    st1 { v16.h }[2], [x7], #0x2
    tbz x22, #0, label_85
    st1 { v23.b }[6], [x25]
    st1 { v8.b }[6], [x8]
    st1 { v16.b }[6], [x7]
    b label_85
KAI_ASM_LABEL(label_82)  // Height 3: Partial direct writeback: partial_0_4
    tbz x22, #0, label_85
    st1 { v23.b }[4], [x25]
    st1 { v8.b }[4], [x8]
    st1 { v16.b }[4], [x7]
    b label_85
KAI_ASM_LABEL(label_83)  // Height 3: Partial direct writeback: partial_1_0
    tbz x22, #1, label_84
    str h23, [x25], #0x2
    str h8, [x8], #0x2
    str h16, [x7], #0x2
    tbz x22, #0, label_85
    st1 { v23.b }[2], [x25]
    st1 { v8.b }[2], [x8]
    st1 { v16.b }[2], [x7]
    b label_85
KAI_ASM_LABEL(label_84)  // Height 3: Partial direct writeback: partial_0_0
    str b23, [x25, #0]
    str b8, [x8, #0]
    str b16, [x7, #0]
KAI_ASM_LABEL(label_85)  // Height 3: Partial direct writeback: Done
    b label_87
KAI_ASM_LABEL(label_86)  // Height 3: Full writeback
    str q23, [x25, #0]
    add x25, x25, #0x10
    str q8, [x8, #0]
    str q16, [x7, #0]
KAI_ASM_LABEL(label_87)  // Height 3: Writeback done
    subs x22, x22, #0x10
    bgt label_60
    b label_175
KAI_ASM_LABEL(label_88)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_89)  // Height 4: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_91
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_93
KAI_ASM_LABEL(label_91)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_92)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x26, label_93
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_93)  // Height 4: input setup done
    cmp x7, #0x10
    blt label_96
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_95
KAI_ASM_LABEL(label_94)  // Height 4: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    ldr q6, [x21, #0x10]
    bge label_94
KAI_ASM_LABEL(label_95)  // Height 4: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x5, x5, #0x10
    add x4, x4, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    sub x7, x7, #0x10
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
KAI_ASM_LABEL(label_96)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_103
    cmp x7, #0x8
    blt label_98
KAI_ASM_LABEL(label_97)  // Height 4: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x8
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a450)  // smmla v16.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a454)  // smmla v20.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a451)  // smmla v17.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a455)  // smmla v21.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a452)  // smmla v18.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a456)  // smmla v22.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a453)  // smmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    bge label_97
KAI_ASM_LABEL(label_98)  // Height 4: Multiply loop: Skip odd blocks
    cbz x7, label_103
    tbz x7, #2, label_100
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    tbz x7, #1, label_99
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    tbz x7, #0, label_102
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    b label_102
KAI_ASM_LABEL(label_99)  // Height 4: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_102
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    b label_102
KAI_ASM_LABEL(label_100)  // Height 4: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_101
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    tbz x7, #0, label_102
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    b label_102
KAI_ASM_LABEL(label_101)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
KAI_ASM_LABEL(label_102)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
KAI_ASM_LABEL(label_103)  // Height 4: Multiply loop: No odd multiplies
    tbz x28, #3, label_104
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_92
KAI_ASM_LABEL(label_104)  // Height 4: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    ldr x1, [x0, #0x58]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    prfm pstl1keep, [x25, #0]
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x20, x20, #0x40
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    add x8, x25, x1
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    add x7, x8, x1
    prfm pstl1keep, [x8, #0]
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    mov v23.16b, v7.16b
    prfm pstl1keep, [x6, #0]
    add v12.4s, v12.4s, v1.4s
    add v13.4s, v13.4s, v2.4s
    add v14.4s, v14.4s, v3.4s
    add v23.4s, v23.4s, v0.4s
    add v8.4s, v8.4s, v0.4s
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v15.4s, v15.4s, v0.4s
    add v20.4s, v20.4s, v1.4s
    add v21.4s, v21.4s, v2.4s
    add v22.4s, v22.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    tbz x28, #4, label_105
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_106
KAI_ASM_LABEL(label_105)  // Height 4: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_106)  // Height 4: parameters loaded
    sqdmulh v23.4s, v23.4s, v4.4s
    sqdmulh v12.4s, v12.4s, v5.4s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    sqdmulh v13.4s, v13.4s, v6.4s
    sqdmulh v14.4s, v14.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v15.4s, v15.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v5.4s
    sqdmulh v21.4s, v21.4s, v6.4s
    sqdmulh v22.4s, v22.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    ld1r { v4.4s }, [x3]
    sqdmulh v17.4s, v17.4s, v5.4s
    ld1r { v5.4s }, [x1]
    sqdmulh v18.4s, v18.4s, v6.4s
    ld1r { v6.4s }, [x2]
    sqdmulh v19.4s, v19.4s, v7.4s
    srshl v23.4s, v23.4s, v0.4s
    srshl v12.4s, v12.4s, v1.4s
    srshl v13.4s, v13.4s, v2.4s
    srshl v14.4s, v14.4s, v3.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v15.4s, v15.4s, v0.4s
    srshl v20.4s, v20.4s, v1.4s
    srshl v21.4s, v21.4s, v2.4s
    srshl v22.4s, v22.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    add v23.4s, v23.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    smin v23.4s, v23.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smax v23.4s, v23.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    uzp1 v23.8h, v23.8h, v12.8h
    uzp1 v12.8h, v13.8h, v14.8h
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v15.8h, v15.8h, v20.8h
    uzp1 v20.8h, v21.8h, v22.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v23.16b, v23.16b, v12.16b
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v15.16b, v15.16b, v20.16b
    uzp1 v16.16b, v16.16b, v17.16b
    bge label_115
    tbz x22, #3, label_110
    str d23, [x25], #0x8
    str d8, [x8], #0x8
    str d15, [x7], #0x8
    str d16, [x6], #0x8
    tbz x22, #2, label_108
    st1 { v23.s }[2], [x25], #0x4
    st1 { v8.s }[2], [x8], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    tbz x22, #1, label_107
    st1 { v23.h }[6], [x25], #0x2
    st1 { v8.h }[6], [x8], #0x2
    st1 { v15.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    tbz x22, #0, label_114
    st1 { v23.b }[14], [x25]
    st1 { v8.b }[14], [x8]
    st1 { v15.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    b label_114
KAI_ASM_LABEL(label_107)  // Height 4: Partial direct writeback: partial_0_12
    tbz x22, #0, label_114
    st1 { v23.b }[12], [x25]
    st1 { v8.b }[12], [x8]
    st1 { v15.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    b label_114
KAI_ASM_LABEL(label_108)  // Height 4: Partial direct writeback: partial_1_8
    tbz x22, #1, label_109
    st1 { v23.h }[4], [x25], #0x2
    st1 { v8.h }[4], [x8], #0x2
    st1 { v15.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    tbz x22, #0, label_114
    st1 { v23.b }[10], [x25]
    st1 { v8.b }[10], [x8]
    st1 { v15.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    b label_114
KAI_ASM_LABEL(label_109)  // Height 4: Partial direct writeback: partial_0_8
    tbz x22, #0, label_114
    st1 { v23.b }[8], [x25]
    st1 { v8.b }[8], [x8]
    st1 { v15.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    b label_114
KAI_ASM_LABEL(label_110)  // Height 4: Partial direct writeback: partial_2_0
    tbz x22, #2, label_112
    str s23, [x25], #0x4
    str s8, [x8], #0x4
    str s15, [x7], #0x4
    str s16, [x6], #0x4
    tbz x22, #1, label_111
    st1 { v23.h }[2], [x25], #0x2
    st1 { v8.h }[2], [x8], #0x2
    st1 { v15.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    tbz x22, #0, label_114
    st1 { v23.b }[6], [x25]
    st1 { v8.b }[6], [x8]
    st1 { v15.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    b label_114
KAI_ASM_LABEL(label_111)  // Height 4: Partial direct writeback: partial_0_4
    tbz x22, #0, label_114
    st1 { v23.b }[4], [x25]
    st1 { v8.b }[4], [x8]
    st1 { v15.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    b label_114
KAI_ASM_LABEL(label_112)  // Height 4: Partial direct writeback: partial_1_0
    tbz x22, #1, label_113
    str h23, [x25], #0x2
    str h8, [x8], #0x2
    str h15, [x7], #0x2
    str h16, [x6], #0x2
    tbz x22, #0, label_114
    st1 { v23.b }[2], [x25]
    st1 { v8.b }[2], [x8]
    st1 { v15.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    b label_114
KAI_ASM_LABEL(label_113)  // Height 4: Partial direct writeback: partial_0_0
    str b23, [x25, #0]
    str b8, [x8, #0]
    str b15, [x7, #0]
    str b16, [x6, #0]
KAI_ASM_LABEL(label_114)  // Height 4: Partial direct writeback: Done
    b label_116
KAI_ASM_LABEL(label_115)  // Height 4: Full writeback
    str q23, [x25, #0]
    add x25, x25, #0x10
    str q8, [x8, #0]
    str q15, [x7, #0]
    str q16, [x6, #0]
KAI_ASM_LABEL(label_116)  // Height 4: Writeback done
    subs x22, x22, #0x10
    bgt label_89
    b label_175
KAI_ASM_LABEL(label_117)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_118)  // Height 5: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_120
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_122
KAI_ASM_LABEL(label_120)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_121)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    cbnz x26, label_122
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_122)  // Height 5: input setup done
    cmp x7, #0x10
    blt label_125
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q7, [x21, #0]
    blt label_124
KAI_ASM_LABEL(label_123)  // Height 5: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    add x2, x2, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b8)  // smmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bc)  // smmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b9)  // smmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bd)  // smmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4ba)  // smmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4be)  // smmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4bb)  // smmla v27.4s, v5.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x4e86a4bf)  // smmla v31.4s, v5.16b, v6.16b
    ldr q5, [x2, #0]
    bge label_123
KAI_ASM_LABEL(label_124)  // Height 5: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    add x2, x2, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    sub x7, x7, #0x10
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b8)  // smmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bc)  // smmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b9)  // smmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bd)  // smmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4ba)  // smmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4be)  // smmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4bb)  // smmla v27.4s, v5.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bf)  // smmla v31.4s, v5.16b, v6.16b
KAI_ASM_LABEL(label_125)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_132
    cmp x7, #0x8
    blt label_127
KAI_ASM_LABEL(label_126)  // Height 5: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x8
    ldr d5, [x2], #0x8
    ldr q6, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v7.2d
    ldr q7, [x21, #0x10]
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a450)  // smmla v16.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a498)  // smmla v24.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a454)  // smmla v20.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49c)  // smmla v28.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a451)  // smmla v17.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a499)  // smmla v25.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a455)  // smmla v21.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49d)  // smmla v29.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a452)  // smmla v18.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49a)  // smmla v26.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a456)  // smmla v22.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49e)  // smmla v30.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a453)  // smmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49b)  // smmla v27.4s, v4.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49f)  // smmla v31.4s, v4.16b, v7.16b
    bge label_126
KAI_ASM_LABEL(label_127)  // Height 5: Multiply loop: Skip odd blocks
    cbz x7, label_132
    tbz x7, #2, label_129
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    tbz x7, #1, label_128
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    ld1 { v5.h }[2], [x2], #0x2
    tbz x7, #0, label_131
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    ld1 { v5.b }[6], [x2]
    b label_131
KAI_ASM_LABEL(label_128)  // Height 5: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_131
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    ld1 { v5.b }[4], [x2]
    b label_131
KAI_ASM_LABEL(label_129)  // Height 5: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_130
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    ldr h5, [x2], #0x2
    tbz x7, #0, label_131
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    ld1 { v5.b }[2], [x2]
    b label_131
KAI_ASM_LABEL(label_130)  // Height 5: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
    ldr b5, [x2, #0]
KAI_ASM_LABEL(label_131)  // Height 5: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
KAI_ASM_LABEL(label_132)  // Height 5: Multiply loop: No odd multiplies
    tbz x28, #3, label_133
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_121
KAI_ASM_LABEL(label_133)  // Height 5: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    ldr x1, [x0, #0x58]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    prfm pstl1keep, [x25, #0]
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x20, x20, #0x40
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    add x8, x25, x1
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    add x7, x8, x1
    prfm pstl1keep, [x8, #0]
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    uzp1 v24.2d, v24.2d, v28.2d
    uzp1 v25.2d, v25.2d, v29.2d
    prfm pstl1keep, [x5, #0]
    uzp1 v26.2d, v26.2d, v30.2d
    uzp1 v27.2d, v27.2d, v31.2d
    mov v31.16b, v7.16b
    add v12.4s, v12.4s, v1.4s
    add v13.4s, v13.4s, v2.4s
    add v14.4s, v14.4s, v3.4s
    add v31.4s, v31.4s, v0.4s
    add v8.4s, v8.4s, v0.4s
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v15.4s, v15.4s, v0.4s
    add v20.4s, v20.4s, v1.4s
    add v21.4s, v21.4s, v2.4s
    add v22.4s, v22.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add v24.4s, v24.4s, v0.4s
    add v25.4s, v25.4s, v1.4s
    add v26.4s, v26.4s, v2.4s
    add v27.4s, v27.4s, v3.4s
    tbz x28, #4, label_134
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_135
KAI_ASM_LABEL(label_134)  // Height 5: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_135)  // Height 5: parameters loaded
    sqdmulh v31.4s, v31.4s, v4.4s
    sqdmulh v12.4s, v12.4s, v5.4s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    sqdmulh v13.4s, v13.4s, v6.4s
    sqdmulh v14.4s, v14.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v15.4s, v15.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v5.4s
    sqdmulh v21.4s, v21.4s, v6.4s
    sqdmulh v22.4s, v22.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    ld1r { v4.4s }, [x3]
    sqdmulh v25.4s, v25.4s, v5.4s
    ld1r { v5.4s }, [x1]
    sqdmulh v26.4s, v26.4s, v6.4s
    ld1r { v6.4s }, [x2]
    sqdmulh v27.4s, v27.4s, v7.4s
    srshl v31.4s, v31.4s, v0.4s
    srshl v12.4s, v12.4s, v1.4s
    srshl v13.4s, v13.4s, v2.4s
    srshl v14.4s, v14.4s, v3.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v15.4s, v15.4s, v0.4s
    srshl v20.4s, v20.4s, v1.4s
    srshl v21.4s, v21.4s, v2.4s
    srshl v22.4s, v22.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v1.4s
    srshl v26.4s, v26.4s, v2.4s
    srshl v27.4s, v27.4s, v3.4s
    add v31.4s, v31.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v24.4s, v24.4s, v4.4s
    add v25.4s, v25.4s, v4.4s
    add v26.4s, v26.4s, v4.4s
    add v27.4s, v27.4s, v4.4s
    smin v31.4s, v31.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v24.4s, v24.4s, v6.4s
    smin v25.4s, v25.4s, v6.4s
    smin v26.4s, v26.4s, v6.4s
    smin v27.4s, v27.4s, v6.4s
    smax v31.4s, v31.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v24.4s, v24.4s, v5.4s
    smax v25.4s, v25.4s, v5.4s
    smax v26.4s, v26.4s, v5.4s
    smax v27.4s, v27.4s, v5.4s
    uzp1 v31.8h, v31.8h, v12.8h
    uzp1 v12.8h, v13.8h, v14.8h
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v15.8h, v15.8h, v20.8h
    uzp1 v20.8h, v21.8h, v22.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v31.16b, v31.16b, v12.16b
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v15.16b, v15.16b, v20.16b
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v24.16b, v24.16b, v25.16b
    bge label_144
    tbz x22, #3, label_139
    str d31, [x25], #0x8
    str d8, [x8], #0x8
    str d15, [x7], #0x8
    str d16, [x6], #0x8
    str d24, [x5], #0x8
    tbz x22, #2, label_137
    st1 { v31.s }[2], [x25], #0x4
    st1 { v8.s }[2], [x8], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v24.s }[2], [x5], #0x4
    tbz x22, #1, label_136
    st1 { v31.h }[6], [x25], #0x2
    st1 { v8.h }[6], [x8], #0x2
    st1 { v15.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    st1 { v24.h }[6], [x5], #0x2
    tbz x22, #0, label_143
    st1 { v31.b }[14], [x25]
    st1 { v8.b }[14], [x8]
    st1 { v15.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    st1 { v24.b }[14], [x5]
    b label_143
KAI_ASM_LABEL(label_136)  // Height 5: Partial direct writeback: partial_0_12
    tbz x22, #0, label_143
    st1 { v31.b }[12], [x25]
    st1 { v8.b }[12], [x8]
    st1 { v15.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    st1 { v24.b }[12], [x5]
    b label_143
KAI_ASM_LABEL(label_137)  // Height 5: Partial direct writeback: partial_1_8
    tbz x22, #1, label_138
    st1 { v31.h }[4], [x25], #0x2
    st1 { v8.h }[4], [x8], #0x2
    st1 { v15.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    st1 { v24.h }[4], [x5], #0x2
    tbz x22, #0, label_143
    st1 { v31.b }[10], [x25]
    st1 { v8.b }[10], [x8]
    st1 { v15.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    st1 { v24.b }[10], [x5]
    b label_143
KAI_ASM_LABEL(label_138)  // Height 5: Partial direct writeback: partial_0_8
    tbz x22, #0, label_143
    st1 { v31.b }[8], [x25]
    st1 { v8.b }[8], [x8]
    st1 { v15.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    st1 { v24.b }[8], [x5]
    b label_143
KAI_ASM_LABEL(label_139)  // Height 5: Partial direct writeback: partial_2_0
    tbz x22, #2, label_141
    str s31, [x25], #0x4
    str s8, [x8], #0x4
    str s15, [x7], #0x4
    str s16, [x6], #0x4
    str s24, [x5], #0x4
    tbz x22, #1, label_140
    st1 { v31.h }[2], [x25], #0x2
    st1 { v8.h }[2], [x8], #0x2
    st1 { v15.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    st1 { v24.h }[2], [x5], #0x2
    tbz x22, #0, label_143
    st1 { v31.b }[6], [x25]
    st1 { v8.b }[6], [x8]
    st1 { v15.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    st1 { v24.b }[6], [x5]
    b label_143
KAI_ASM_LABEL(label_140)  // Height 5: Partial direct writeback: partial_0_4
    tbz x22, #0, label_143
    st1 { v31.b }[4], [x25]
    st1 { v8.b }[4], [x8]
    st1 { v15.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    st1 { v24.b }[4], [x5]
    b label_143
KAI_ASM_LABEL(label_141)  // Height 5: Partial direct writeback: partial_1_0
    tbz x22, #1, label_142
    str h31, [x25], #0x2
    str h8, [x8], #0x2
    str h15, [x7], #0x2
    str h16, [x6], #0x2
    str h24, [x5], #0x2
    tbz x22, #0, label_143
    st1 { v31.b }[2], [x25]
    st1 { v8.b }[2], [x8]
    st1 { v15.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    st1 { v24.b }[2], [x5]
    b label_143
KAI_ASM_LABEL(label_142)  // Height 5: Partial direct writeback: partial_0_0
    str b31, [x25, #0]
    str b8, [x8, #0]
    str b15, [x7, #0]
    str b16, [x6, #0]
    str b24, [x5, #0]
KAI_ASM_LABEL(label_143)  // Height 5: Partial direct writeback: Done
    b label_145
KAI_ASM_LABEL(label_144)  // Height 5: Full writeback
    str q31, [x25, #0]
    add x25, x25, #0x10
    str q8, [x8, #0]
    str q15, [x7, #0]
    str q16, [x6, #0]
    str q24, [x5, #0]
KAI_ASM_LABEL(label_145)  // Height 5: Writeback done
    subs x22, x22, #0x10
    bgt label_118
    b label_175
KAI_ASM_LABEL(label_146)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x78]
    ldr x23, [x0, #0x88]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_147)  // Height 6: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x28, #3, label_149
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_151
KAI_ASM_LABEL(label_149)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_150)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    ldr x1, [x1, #0x28]
    cbnz x26, label_151
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_151)  // Height 6: input setup done
    cmp x7, #0x10
    blt label_154
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    ldr q7, [x21, #0]
    blt label_153
KAI_ASM_LABEL(label_152)  // Height 6: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x6, x6, #0x10
    add x5, x5, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    add x4, x4, #0x10
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    add x2, x2, #0x10
    add x1, x1, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    cmp x7, #0x20
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b8)  // smmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bc)  // smmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b9)  // smmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bd)  // smmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4ba)  // smmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4be)  // smmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4bb)  // smmla v27.4s, v5.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x4e86a4bf)  // smmla v31.4s, v5.16b, v6.16b
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    bge label_152
KAI_ASM_LABEL(label_153)  // Height 6: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x5, x5, #0x10
    add x4, x4, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    add x2, x2, #0x10
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    add x1, x1, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    prfm pldl1keep, [x5, #0x80]
    sub x7, x7, #0x10
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87a428)  // smmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a470)  // smmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b8)  // smmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86a42c)  // smmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a474)  // smmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bc)  // smmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87a429)  // smmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a471)  // smmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4b9)  // smmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86a42d)  // smmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a475)  // smmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bd)  // smmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87a42a)  // smmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a472)  // smmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4ba)  // smmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a476)  // smmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4be)  // smmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87a42b)  // smmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87a473)  // smmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87a4bb)  // smmla v27.4s, v5.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86a477)  // smmla v23.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86a4bf)  // smmla v31.4s, v5.16b, v6.16b
KAI_ASM_LABEL(label_154)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_161
    cmp x7, #0x8
    blt label_156
KAI_ASM_LABEL(label_155)  // Height 6: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x8
    ldr d5, [x2], #0x8
    ldr d7, [x1], #0x8
    ldr q6, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v7.2d
    ldr q7, [x21, #0x10]
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a450)  // smmla v16.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a498)  // smmla v24.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a454)  // smmla v20.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49c)  // smmla v28.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a451)  // smmla v17.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a499)  // smmla v25.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a455)  // smmla v21.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49d)  // smmla v29.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a452)  // smmla v18.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49a)  // smmla v26.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87a40e)  // smmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a456)  // smmla v22.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49e)  // smmla v30.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86a40b)  // smmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a453)  // smmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49b)  // smmla v27.4s, v4.16b, v6.16b
    KAI_ASM_INST(0x4e87a40f)  // smmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49f)  // smmla v31.4s, v4.16b, v7.16b
    bge label_155
KAI_ASM_LABEL(label_156)  // Height 6: Multiply loop: Skip odd blocks
    cbz x7, label_161
    tbz x7, #2, label_158
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    ldr s6, [x1], #0x4
    tbz x7, #1, label_157
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    ld1 { v5.h }[2], [x2], #0x2
    ld1 { v6.h }[2], [x1], #0x2
    tbz x7, #0, label_160
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    ld1 { v5.b }[6], [x2]
    ld1 { v6.b }[6], [x1]
    b label_160
KAI_ASM_LABEL(label_157)  // Height 6: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_160
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    ld1 { v5.b }[4], [x2]
    ld1 { v6.b }[4], [x1]
    b label_160
KAI_ASM_LABEL(label_158)  // Height 6: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_159
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    ldr h5, [x2], #0x2
    ldr h6, [x1], #0x2
    tbz x7, #0, label_160
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    ld1 { v5.b }[2], [x2]
    ld1 { v6.b }[2], [x1]
    b label_160
KAI_ASM_LABEL(label_159)  // Height 6: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
    ldr b5, [x2, #0]
    ldr b6, [x1, #0]
KAI_ASM_LABEL(label_160)  // Height 6: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87a408)  // smmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a450)  // smmla v16.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a498)  // smmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86a40c)  // smmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49c)  // smmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87a409)  // smmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a451)  // smmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a499)  // smmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86a40d)  // smmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49d)  // smmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87a40a)  // smmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a452)  // smmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49a)  // smmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86a40e)  // smmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49e)  // smmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87a453)  // smmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87a49b)  // smmla v27.4s, v4.16b, v7.16b
    KAI_ASM_INST(0x4e86a40f)  // smmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86a457)  // smmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86a49f)  // smmla v31.4s, v4.16b, v6.16b
KAI_ASM_LABEL(label_161)  // Height 6: Multiply loop: No odd multiplies
    tbz x28, #3, label_162
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_150
KAI_ASM_LABEL(label_162)  // Height 6: no string loop
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    ldr x1, [x0, #0x58]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    prfm pstl1keep, [x25, #0]
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x20, x20, #0x40
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    add x8, x25, x1
    add x7, x8, x1
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    add x6, x7, x1
    prfm pstl1keep, [x8, #0]
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    add x5, x6, x1
    prfm pstl1keep, [x7, #0]
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    add x4, x5, x1
    prfm pstl1keep, [x6, #0]
    uzp1 v23.2d, v24.2d, v28.2d
    uzp2 v24.2d, v24.2d, v28.2d
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    uzp1 v28.2d, v25.2d, v29.2d
    uzp2 v25.2d, v25.2d, v29.2d
    uzp1 v29.2d, v26.2d, v30.2d
    uzp2 v26.2d, v26.2d, v30.2d
    uzp1 v30.2d, v27.2d, v31.2d
    uzp2 v27.2d, v27.2d, v31.2d
    mov v31.16b, v7.16b
    add v12.4s, v12.4s, v1.4s
    add v13.4s, v13.4s, v2.4s
    add v14.4s, v14.4s, v3.4s
    add v31.4s, v31.4s, v0.4s
    add v8.4s, v8.4s, v0.4s
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v15.4s, v15.4s, v0.4s
    add v20.4s, v20.4s, v1.4s
    add v21.4s, v21.4s, v2.4s
    add v22.4s, v22.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add v23.4s, v23.4s, v0.4s
    add v28.4s, v28.4s, v1.4s
    add v29.4s, v29.4s, v2.4s
    add v30.4s, v30.4s, v3.4s
    add v24.4s, v24.4s, v0.4s
    add v25.4s, v25.4s, v1.4s
    add v26.4s, v26.4s, v2.4s
    add v27.4s, v27.4s, v3.4s
    tbz x28, #4, label_163
    ldr q0, [x23, #0]
    ldr q4, [x24, #0]
    ldr q1, [x23, #0x10]
    ldr q5, [x24, #0x10]
    ldr q2, [x23, #0x20]
    ldr q6, [x24, #0x20]
    ldr q3, [x23, #0x30]
    ldr q7, [x24, #0x30]
    add x23, x23, #0x40
    add x24, x24, #0x40
    b label_164
KAI_ASM_LABEL(label_163)  // Height 6: per layer parameters
    add x2, x0, #0x9c
    add x1, x0, #0x98
    ld1r { v0.4s }, [x2]
    ld1r { v4.4s }, [x1]
    mov v1.16b, v0.16b
    mov v5.16b, v4.16b
    mov v2.16b, v0.16b
    mov v6.16b, v4.16b
    mov v3.16b, v0.16b
    mov v7.16b, v4.16b
KAI_ASM_LABEL(label_164)  // Height 6: parameters loaded
    sqdmulh v31.4s, v31.4s, v4.4s
    sqdmulh v12.4s, v12.4s, v5.4s
    add x3, x0, #0x94
    add x2, x0, #0xa4
    sqdmulh v13.4s, v13.4s, v6.4s
    sqdmulh v14.4s, v14.4s, v7.4s
    add x1, x0, #0xa0
    cmp x22, #0x10
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v15.4s, v15.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v5.4s
    sqdmulh v21.4s, v21.4s, v6.4s
    sqdmulh v22.4s, v22.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    sqdmulh v23.4s, v23.4s, v4.4s
    sqdmulh v28.4s, v28.4s, v5.4s
    sqdmulh v29.4s, v29.4s, v6.4s
    sqdmulh v30.4s, v30.4s, v7.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    ld1r { v4.4s }, [x3]
    sqdmulh v25.4s, v25.4s, v5.4s
    ld1r { v5.4s }, [x1]
    sqdmulh v26.4s, v26.4s, v6.4s
    ld1r { v6.4s }, [x2]
    sqdmulh v27.4s, v27.4s, v7.4s
    srshl v31.4s, v31.4s, v0.4s
    srshl v12.4s, v12.4s, v1.4s
    srshl v13.4s, v13.4s, v2.4s
    srshl v14.4s, v14.4s, v3.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v15.4s, v15.4s, v0.4s
    srshl v20.4s, v20.4s, v1.4s
    srshl v21.4s, v21.4s, v2.4s
    srshl v22.4s, v22.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    srshl v23.4s, v23.4s, v0.4s
    srshl v28.4s, v28.4s, v1.4s
    srshl v29.4s, v29.4s, v2.4s
    srshl v30.4s, v30.4s, v3.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v1.4s
    srshl v26.4s, v26.4s, v2.4s
    srshl v27.4s, v27.4s, v3.4s
    add v31.4s, v31.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v23.4s, v23.4s, v4.4s
    add v28.4s, v28.4s, v4.4s
    add v29.4s, v29.4s, v4.4s
    add v30.4s, v30.4s, v4.4s
    add v24.4s, v24.4s, v4.4s
    add v25.4s, v25.4s, v4.4s
    add v26.4s, v26.4s, v4.4s
    add v27.4s, v27.4s, v4.4s
    smin v31.4s, v31.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v23.4s, v23.4s, v6.4s
    smin v28.4s, v28.4s, v6.4s
    smin v29.4s, v29.4s, v6.4s
    smin v30.4s, v30.4s, v6.4s
    smin v24.4s, v24.4s, v6.4s
    smin v25.4s, v25.4s, v6.4s
    smin v26.4s, v26.4s, v6.4s
    smin v27.4s, v27.4s, v6.4s
    smax v31.4s, v31.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v23.4s, v23.4s, v5.4s
    smax v28.4s, v28.4s, v5.4s
    smax v29.4s, v29.4s, v5.4s
    smax v30.4s, v30.4s, v5.4s
    smax v24.4s, v24.4s, v5.4s
    smax v25.4s, v25.4s, v5.4s
    smax v26.4s, v26.4s, v5.4s
    smax v27.4s, v27.4s, v5.4s
    uzp1 v31.8h, v31.8h, v12.8h
    uzp1 v12.8h, v13.8h, v14.8h
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v15.8h, v15.8h, v20.8h
    uzp1 v20.8h, v21.8h, v22.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v23.8h, v23.8h, v28.8h
    uzp1 v28.8h, v29.8h, v30.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v31.16b, v31.16b, v12.16b
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v15.16b, v15.16b, v20.16b
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v23.16b, v23.16b, v28.16b
    uzp1 v24.16b, v24.16b, v25.16b
    bge label_173
    tbz x22, #3, label_168
    str d31, [x25], #0x8
    str d8, [x8], #0x8
    str d15, [x7], #0x8
    str d16, [x6], #0x8
    str d23, [x5], #0x8
    str d24, [x4], #0x8
    tbz x22, #2, label_166
    st1 { v31.s }[2], [x25], #0x4
    st1 { v8.s }[2], [x8], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v23.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    tbz x22, #1, label_165
    st1 { v31.h }[6], [x25], #0x2
    st1 { v8.h }[6], [x8], #0x2
    st1 { v15.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    st1 { v23.h }[6], [x5], #0x2
    st1 { v24.h }[6], [x4], #0x2
    tbz x22, #0, label_172
    st1 { v31.b }[14], [x25]
    st1 { v8.b }[14], [x8]
    st1 { v15.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    st1 { v23.b }[14], [x5]
    st1 { v24.b }[14], [x4]
    b label_172
KAI_ASM_LABEL(label_165)  // Height 6: Partial direct writeback: partial_0_12
    tbz x22, #0, label_172
    st1 { v31.b }[12], [x25]
    st1 { v8.b }[12], [x8]
    st1 { v15.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    st1 { v23.b }[12], [x5]
    st1 { v24.b }[12], [x4]
    b label_172
KAI_ASM_LABEL(label_166)  // Height 6: Partial direct writeback: partial_1_8
    tbz x22, #1, label_167
    st1 { v31.h }[4], [x25], #0x2
    st1 { v8.h }[4], [x8], #0x2
    st1 { v15.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    st1 { v23.h }[4], [x5], #0x2
    st1 { v24.h }[4], [x4], #0x2
    tbz x22, #0, label_172
    st1 { v31.b }[10], [x25]
    st1 { v8.b }[10], [x8]
    st1 { v15.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    st1 { v23.b }[10], [x5]
    st1 { v24.b }[10], [x4]
    b label_172
KAI_ASM_LABEL(label_167)  // Height 6: Partial direct writeback: partial_0_8
    tbz x22, #0, label_172
    st1 { v31.b }[8], [x25]
    st1 { v8.b }[8], [x8]
    st1 { v15.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    st1 { v23.b }[8], [x5]
    st1 { v24.b }[8], [x4]
    b label_172
KAI_ASM_LABEL(label_168)  // Height 6: Partial direct writeback: partial_2_0
    tbz x22, #2, label_170
    str s31, [x25], #0x4
    str s8, [x8], #0x4
    str s15, [x7], #0x4
    str s16, [x6], #0x4
    str s23, [x5], #0x4
    str s24, [x4], #0x4
    tbz x22, #1, label_169
    st1 { v31.h }[2], [x25], #0x2
    st1 { v8.h }[2], [x8], #0x2
    st1 { v15.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    st1 { v23.h }[2], [x5], #0x2
    st1 { v24.h }[2], [x4], #0x2
    tbz x22, #0, label_172
    st1 { v31.b }[6], [x25]
    st1 { v8.b }[6], [x8]
    st1 { v15.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    st1 { v23.b }[6], [x5]
    st1 { v24.b }[6], [x4]
    b label_172
KAI_ASM_LABEL(label_169)  // Height 6: Partial direct writeback: partial_0_4
    tbz x22, #0, label_172
    st1 { v31.b }[4], [x25]
    st1 { v8.b }[4], [x8]
    st1 { v15.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    st1 { v23.b }[4], [x5]
    st1 { v24.b }[4], [x4]
    b label_172
KAI_ASM_LABEL(label_170)  // Height 6: Partial direct writeback: partial_1_0
    tbz x22, #1, label_171
    str h31, [x25], #0x2
    str h8, [x8], #0x2
    str h15, [x7], #0x2
    str h16, [x6], #0x2
    str h23, [x5], #0x2
    str h24, [x4], #0x2
    tbz x22, #0, label_172
    st1 { v31.b }[2], [x25]
    st1 { v8.b }[2], [x8]
    st1 { v15.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    st1 { v23.b }[2], [x5]
    st1 { v24.b }[2], [x4]
    b label_172
KAI_ASM_LABEL(label_171)  // Height 6: Partial direct writeback: partial_0_0
    str b31, [x25, #0]
    str b8, [x8, #0]
    str b15, [x7, #0]
    str b16, [x6, #0]
    str b23, [x5, #0]
    str b24, [x4, #0]
KAI_ASM_LABEL(label_172)  // Height 6: Partial direct writeback: Done
    b label_174
KAI_ASM_LABEL(label_173)  // Height 6: Full writeback
    str q31, [x25, #0]
    add x25, x25, #0x10
    str q8, [x8, #0]
    str q15, [x7, #0]
    str q16, [x6, #0]
    str q23, [x5, #0]
    str q24, [x4, #0]
KAI_ASM_LABEL(label_174)  // Height 6: Writeback done
    subs x22, x22, #0x10
    bgt label_147
    add x9, x9, #0x6
    cmp x9, x27
    blt label_1
KAI_ASM_LABEL(label_175)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_s8qs_mmla_6x16)

    KAI_ASM_END
