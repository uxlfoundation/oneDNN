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

    KAI_ASM_CODE(a64_hybrid_u8s8s32_mmla_6x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_u8s8s32_mmla_6x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_u8s8s32_mmla_6x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_u8s8s32_mmla_6x16)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    sub SP, SP, #0x40
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
    bge label_231
    cmp x1, #0x4
    bgt label_185
    beq label_139
    cmp x1, #0x2
    bgt label_93
    beq label_47
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x20, label_10
    cmp x22, #0x10
    bge label_8
    cbnz x27, label_7
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_4
KAI_ASM_LABEL(label_3)  // Height 1: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_3
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_6
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_5
KAI_ASM_LABEL(label_6)  // Height 1: Partial bias load: element loop skip
KAI_ASM_LABEL(label_7)  // Height 1: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_9
KAI_ASM_LABEL(label_8)  // Height 1: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_9)  // Height 1: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    b label_22
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x25, #0, label_21
    cmp x22, #0x10
    bge label_19
    tbz x22, #3, label_14
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v10.4s }, [x23], #0x10
    tbz x22, #2, label_12
    ld1 { v11.4s }, [x23], #0x10
    tbz x22, #1, label_11
    ldr d16, [x23], #0x8
    mov x1, #0x38
    tbz x22, #0, label_18
    ld1 { v16.s }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_18
    ldr s16, [x23, #0]
    b label_18
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_8
    tbz x22, #1, label_13
    ldr d11, [x23], #0x8
    mov x1, #0x28
    tbz x22, #0, label_18
    ld1 { v11.s }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_18
    ldr s11, [x23, #0]
    b label_18
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_2_0
    tbz x22, #2, label_16
    ld1 { v9.4s }, [x23], #0x10
    tbz x22, #1, label_15
    ldr d10, [x23], #0x8
    mov x1, #0x18
    tbz x22, #0, label_18
    ld1 { v10.s }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_18
    ldr s10, [x23, #0]
    b label_18
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_0
    tbz x22, #1, label_17
    ldr d9, [x23], #0x8
    mov x1, #0x8
    tbz x22, #0, label_18
    ld1 { v9.s }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    mov x1, #0
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: Done
    sub x23, x23, x1
    b label_20
KAI_ASM_LABEL(label_19)  // Height 1: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
KAI_ASM_LABEL(label_20)  // Height 1: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    b label_22
KAI_ASM_LABEL(label_21)  // Height 1: no accumulate
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
KAI_ASM_LABEL(label_22)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_23
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    b label_25
KAI_ASM_LABEL(label_23)  // Height 1: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_24)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    cbnz x24, label_25
    add x8, x8, x3
KAI_ASM_LABEL(label_25)  // Height 1: input setup done
    cmp x7, #0x10
    blt label_28
    ldr q1, [x8, #0]
    ldr q7, [x21, #0]
    cmp x7, #0x20
    ldr q6, [x21, #0x10]
    blt label_27
KAI_ASM_LABEL(label_26)  // Height 1: Multiply loop: Main loop head
    sub x7, x7, #0x10
    add x8, x8, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    ldr q6, [x21, #0x10]
    bge label_26
KAI_ASM_LABEL(label_27)  // Height 1: Multiply loop: Single iteration only
    add x8, x8, #0x10
    sub x7, x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
KAI_ASM_LABEL(label_28)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_35
    cmp x7, #0x8
    blt label_30
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr q6, [x21, #0]
    sub x7, x7, #0x8
    ldr q7, [x21, #0x10]
    cmp x7, #0x8
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    bge label_29
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: Skip odd blocks
    cbz x7, label_35
    tbz x7, #2, label_32
    ldr s1, [x8], #0x4
    tbz x7, #1, label_31
    ld1 { v1.h }[2], [x8], #0x2
    tbz x7, #0, label_34
    ld1 { v1.b }[6], [x8]
    b label_34
KAI_ASM_LABEL(label_31)  // Height 1: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_34
    ld1 { v1.b }[4], [x8]
    b label_34
KAI_ASM_LABEL(label_32)  // Height 1: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_33
    ldr h1, [x8], #0x2
    tbz x7, #0, label_34
    ld1 { v1.b }[2], [x8]
    b label_34
KAI_ASM_LABEL(label_33)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
KAI_ASM_LABEL(label_34)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
KAI_ASM_LABEL(label_35)  // Height 1: Multiply loop: No odd multiplies
    tbz x25, #3, label_36
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_24
KAI_ASM_LABEL(label_36)  // Height 1: no string loop
    cmp x22, #0x10
    uzp1 v8.2d, v8.2d, v12.2d
    uzp1 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v10.2d, v10.2d, v14.2d
    uzp1 v11.2d, v11.2d, v15.2d
    bge label_45
    tbz x22, #3, label_40
    st1 { v8.4s }, [x23], #0x10
    st1 { v9.4s }, [x23], #0x10
    tbz x22, #2, label_38
    st1 { v10.4s }, [x23], #0x10
    tbz x22, #1, label_37
    str d11, [x23], #0x8
    tbz x22, #0, label_44
    st1 { v11.s }[2], [x23]
    b label_44
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_0_12
    tbz x22, #0, label_44
    str s11, [x23, #0]
    b label_44
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_1_8
    tbz x22, #1, label_39
    str d10, [x23], #0x8
    tbz x22, #0, label_44
    st1 { v10.s }[2], [x23]
    b label_44
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_0_8
    tbz x22, #0, label_44
    str s10, [x23, #0]
    b label_44
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_2_0
    tbz x22, #2, label_42
    st1 { v8.4s }, [x23], #0x10
    tbz x22, #1, label_41
    str d9, [x23], #0x8
    tbz x22, #0, label_44
    st1 { v9.s }[2], [x23]
    b label_44
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_0_4
    tbz x22, #0, label_44
    str s9, [x23, #0]
    b label_44
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_1_0
    tbz x22, #1, label_43
    str d8, [x23], #0x8
    tbz x22, #0, label_44
    st1 { v8.s }[2], [x23]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_0_0
    str s8, [x23, #0]
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: Done
    b label_46
KAI_ASM_LABEL(label_45)  // Height 1: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
KAI_ASM_LABEL(label_46)  // Height 1: Writeback done
    subs x22, x22, #0x10
    bgt label_2
    b label_277
KAI_ASM_LABEL(label_47)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_48)  // Height 2: Column loop
    cbz x20, label_56
    cmp x22, #0x10
    bge label_54
    cbnz x27, label_53
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_50
KAI_ASM_LABEL(label_49)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_49
KAI_ASM_LABEL(label_50)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_52
KAI_ASM_LABEL(label_51)  // Height 2: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_51
KAI_ASM_LABEL(label_52)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_53)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_55
KAI_ASM_LABEL(label_54)  // Height 2: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_55)  // Height 2: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    b label_68
KAI_ASM_LABEL(label_56)  // Height 2: no bias
    tbz x25, #0, label_67
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x6, x23, x1
    bge label_65
    tbz x22, #3, label_60
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v10.4s }, [x23], #0x10
    ld1 { v13.4s }, [x6], #0x10
    tbz x22, #2, label_58
    ld1 { v11.4s }, [x23], #0x10
    ld1 { v14.4s }, [x6], #0x10
    tbz x22, #1, label_57
    ldr d16, [x23], #0x8
    ldr d15, [x6], #0x8
    mov x1, #0x38
    tbz x22, #0, label_64
    ld1 { v16.s }[2], [x23]
    ld1 { v15.s }[2], [x6]
    b label_64
KAI_ASM_LABEL(label_57)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_64
    ldr s16, [x23, #0]
    ldr s15, [x6, #0]
    b label_64
KAI_ASM_LABEL(label_58)  // Height 2: Partial accumulate: partial_1_8
    tbz x22, #1, label_59
    ldr d11, [x23], #0x8
    ldr d14, [x6], #0x8
    mov x1, #0x28
    tbz x22, #0, label_64
    ld1 { v11.s }[2], [x23]
    ld1 { v14.s }[2], [x6]
    b label_64
KAI_ASM_LABEL(label_59)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_64
    ldr s11, [x23, #0]
    ldr s14, [x6, #0]
    b label_64
KAI_ASM_LABEL(label_60)  // Height 2: Partial accumulate: partial_2_0
    tbz x22, #2, label_62
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    tbz x22, #1, label_61
    ldr d10, [x23], #0x8
    ldr d13, [x6], #0x8
    mov x1, #0x18
    tbz x22, #0, label_64
    ld1 { v10.s }[2], [x23]
    ld1 { v13.s }[2], [x6]
    b label_64
KAI_ASM_LABEL(label_61)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_64
    ldr s10, [x23, #0]
    ldr s13, [x6, #0]
    b label_64
KAI_ASM_LABEL(label_62)  // Height 2: Partial accumulate: partial_1_0
    tbz x22, #1, label_63
    ldr d9, [x23], #0x8
    ldr d12, [x6], #0x8
    mov x1, #0x8
    tbz x22, #0, label_64
    ld1 { v9.s }[2], [x23]
    ld1 { v12.s }[2], [x6]
    b label_64
KAI_ASM_LABEL(label_63)  // Height 2: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    ldr s12, [x6, #0]
    mov x1, #0
KAI_ASM_LABEL(label_64)  // Height 2: Partial accumulate: Done
    sub x23, x23, x1
    b label_66
KAI_ASM_LABEL(label_65)  // Height 2: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
    ldr q12, [x6, #0]
    ldr q13, [x6, #0x10]
    ldr q14, [x6, #0x20]
    ldr q15, [x6, #0x30]
KAI_ASM_LABEL(label_66)  // Height 2: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    b label_68
KAI_ASM_LABEL(label_67)  // Height 2: no accumulate
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
KAI_ASM_LABEL(label_68)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_69
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_71
KAI_ASM_LABEL(label_69)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_70)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_71
    add x8, x8, x3
    add x6, x6, x3
KAI_ASM_LABEL(label_71)  // Height 2: input setup done
    cmp x7, #0x10
    blt label_74
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_73
KAI_ASM_LABEL(label_72)  // Height 2: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x10
    add x8, x8, #0x10
    add x6, x6, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x8, #0x80]
    ldr q2, [x6, #0]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    ldr q6, [x21, #0x10]
    bge label_72
KAI_ASM_LABEL(label_73)  // Height 2: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    sub x7, x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
KAI_ASM_LABEL(label_74)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_81
    cmp x7, #0x8
    blt label_76
KAI_ASM_LABEL(label_75)  // Height 2: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    cmp x7, #0x8
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    bge label_75
KAI_ASM_LABEL(label_76)  // Height 2: Multiply loop: Skip odd blocks
    cbz x7, label_81
    tbz x7, #2, label_78
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    tbz x7, #1, label_77
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    tbz x7, #0, label_80
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    b label_80
KAI_ASM_LABEL(label_77)  // Height 2: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_80
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    b label_80
KAI_ASM_LABEL(label_78)  // Height 2: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_79
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    tbz x7, #0, label_80
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    b label_80
KAI_ASM_LABEL(label_79)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
KAI_ASM_LABEL(label_80)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
KAI_ASM_LABEL(label_81)  // Height 2: Multiply loop: No odd multiplies
    tbz x25, #3, label_82
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_70
KAI_ASM_LABEL(label_82)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x6, x23, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x6, #0]
    bge label_91
    tbz x22, #3, label_86
    st1 { v7.4s }, [x23], #0x10
    st1 { v12.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v9.4s }, [x6], #0x10
    tbz x22, #2, label_84
    st1 { v13.4s }, [x23], #0x10
    st1 { v10.4s }, [x6], #0x10
    tbz x22, #1, label_83
    str d14, [x23], #0x8
    str d11, [x6], #0x8
    tbz x22, #0, label_90
    st1 { v14.s }[2], [x23]
    st1 { v11.s }[2], [x6]
    b label_90
KAI_ASM_LABEL(label_83)  // Height 2: Partial direct writeback: partial_0_12
    tbz x22, #0, label_90
    str s14, [x23, #0]
    str s11, [x6, #0]
    b label_90
KAI_ASM_LABEL(label_84)  // Height 2: Partial direct writeback: partial_1_8
    tbz x22, #1, label_85
    str d13, [x23], #0x8
    str d10, [x6], #0x8
    tbz x22, #0, label_90
    st1 { v13.s }[2], [x23]
    st1 { v10.s }[2], [x6]
    b label_90
KAI_ASM_LABEL(label_85)  // Height 2: Partial direct writeback: partial_0_8
    tbz x22, #0, label_90
    str s13, [x23, #0]
    str s10, [x6, #0]
    b label_90
KAI_ASM_LABEL(label_86)  // Height 2: Partial direct writeback: partial_2_0
    tbz x22, #2, label_88
    st1 { v7.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    tbz x22, #1, label_87
    str d12, [x23], #0x8
    str d9, [x6], #0x8
    tbz x22, #0, label_90
    st1 { v12.s }[2], [x23]
    st1 { v9.s }[2], [x6]
    b label_90
KAI_ASM_LABEL(label_87)  // Height 2: Partial direct writeback: partial_0_4
    tbz x22, #0, label_90
    str s12, [x23, #0]
    str s9, [x6, #0]
    b label_90
KAI_ASM_LABEL(label_88)  // Height 2: Partial direct writeback: partial_1_0
    tbz x22, #1, label_89
    str d7, [x23], #0x8
    str d8, [x6], #0x8
    tbz x22, #0, label_90
    st1 { v7.s }[2], [x23]
    st1 { v8.s }[2], [x6]
    b label_90
KAI_ASM_LABEL(label_89)  // Height 2: Partial direct writeback: partial_0_0
    str s7, [x23, #0]
    str s8, [x6, #0]
KAI_ASM_LABEL(label_90)  // Height 2: Partial direct writeback: Done
    b label_92
KAI_ASM_LABEL(label_91)  // Height 2: Full writeback
    str q7, [x23, #0]
    str q12, [x23, #0x10]
    str q13, [x23, #0x20]
    str q14, [x23, #0x30]
    add x23, x23, #0x40
    str q8, [x6, #0]
    str q9, [x6, #0x10]
    str q10, [x6, #0x20]
    str q11, [x6, #0x30]
KAI_ASM_LABEL(label_92)  // Height 2: Writeback done
    subs x22, x22, #0x10
    bgt label_48
    b label_277
KAI_ASM_LABEL(label_93)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_94)  // Height 3: Column loop
    cbz x20, label_102
    cmp x22, #0x10
    bge label_100
    cbnz x27, label_99
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_96
KAI_ASM_LABEL(label_95)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_95
KAI_ASM_LABEL(label_96)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_98
KAI_ASM_LABEL(label_97)  // Height 3: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_97
KAI_ASM_LABEL(label_98)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_99)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_101
KAI_ASM_LABEL(label_100)  // Height 3: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_101)  // Height 3: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    mov v16.16b, v8.16b
    mov v20.16b, v12.16b
    mov v17.16b, v9.16b
    mov v21.16b, v13.16b
    mov v18.16b, v10.16b
    mov v22.16b, v14.16b
    mov v19.16b, v11.16b
    mov v23.16b, v15.16b
    b label_114
KAI_ASM_LABEL(label_102)  // Height 3: no bias
    tbz x25, #0, label_113
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x6, x23, x1
    add x5, x6, x1
    bge label_111
    tbz x22, #3, label_106
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v10.4s }, [x23], #0x10
    ld1 { v13.4s }, [x6], #0x10
    ld1 { v18.4s }, [x5], #0x10
    tbz x22, #2, label_104
    ld1 { v11.4s }, [x23], #0x10
    ld1 { v14.4s }, [x6], #0x10
    ld1 { v19.4s }, [x5], #0x10
    tbz x22, #1, label_103
    ldr d16, [x23], #0x8
    ldr d15, [x6], #0x8
    mov x1, #0x38
    ldr d24, [x5], #0x8
    tbz x22, #0, label_110
    ld1 { v16.s }[2], [x23]
    ld1 { v15.s }[2], [x6]
    ld1 { v24.s }[2], [x5]
    b label_110
KAI_ASM_LABEL(label_103)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_110
    ldr s16, [x23, #0]
    ldr s15, [x6, #0]
    ldr s24, [x5, #0]
    b label_110
KAI_ASM_LABEL(label_104)  // Height 3: Partial accumulate: partial_1_8
    tbz x22, #1, label_105
    ldr d11, [x23], #0x8
    ldr d14, [x6], #0x8
    mov x1, #0x28
    ldr d19, [x5], #0x8
    tbz x22, #0, label_110
    ld1 { v11.s }[2], [x23]
    ld1 { v14.s }[2], [x6]
    ld1 { v19.s }[2], [x5]
    b label_110
KAI_ASM_LABEL(label_105)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_110
    ldr s11, [x23, #0]
    ldr s14, [x6, #0]
    ldr s19, [x5, #0]
    b label_110
KAI_ASM_LABEL(label_106)  // Height 3: Partial accumulate: partial_2_0
    tbz x22, #2, label_108
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    tbz x22, #1, label_107
    ldr d10, [x23], #0x8
    ldr d13, [x6], #0x8
    mov x1, #0x18
    ldr d18, [x5], #0x8
    tbz x22, #0, label_110
    ld1 { v10.s }[2], [x23]
    ld1 { v13.s }[2], [x6]
    ld1 { v18.s }[2], [x5]
    b label_110
KAI_ASM_LABEL(label_107)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_110
    ldr s10, [x23, #0]
    ldr s13, [x6, #0]
    ldr s18, [x5, #0]
    b label_110
KAI_ASM_LABEL(label_108)  // Height 3: Partial accumulate: partial_1_0
    tbz x22, #1, label_109
    ldr d9, [x23], #0x8
    ldr d12, [x6], #0x8
    mov x1, #0x8
    ldr d17, [x5], #0x8
    tbz x22, #0, label_110
    ld1 { v9.s }[2], [x23]
    ld1 { v12.s }[2], [x6]
    ld1 { v17.s }[2], [x5]
    b label_110
KAI_ASM_LABEL(label_109)  // Height 3: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    ldr s12, [x6, #0]
    mov x1, #0
    ldr s17, [x5, #0]
KAI_ASM_LABEL(label_110)  // Height 3: Partial accumulate: Done
    sub x23, x23, x1
    b label_112
KAI_ASM_LABEL(label_111)  // Height 3: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
    ldr q12, [x6, #0]
    ldr q13, [x6, #0x10]
    ldr q14, [x6, #0x20]
    ldr q15, [x6, #0x30]
    ldr q17, [x5, #0]
    ldr q18, [x5, #0x10]
    ldr q19, [x5, #0x20]
    ldr q24, [x5, #0x30]
KAI_ASM_LABEL(label_112)  // Height 3: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    zip1 v16.2d, v17.2d, v20.2d
    zip2 v20.2d, v17.2d, v20.2d
    zip1 v17.2d, v18.2d, v21.2d
    zip2 v21.2d, v18.2d, v21.2d
    zip1 v18.2d, v19.2d, v22.2d
    zip2 v22.2d, v19.2d, v22.2d
    zip1 v19.2d, v24.2d, v23.2d
    zip2 v23.2d, v24.2d, v23.2d
    b label_114
KAI_ASM_LABEL(label_113)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_114)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_115
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_117
KAI_ASM_LABEL(label_115)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_116)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_117
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
KAI_ASM_LABEL(label_117)  // Height 3: input setup done
    cmp x7, #0x10
    blt label_120
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_119
KAI_ASM_LABEL(label_118)  // Height 3: Multiply loop: Main loop head
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    ldr q6, [x21, #0x10]
    bge label_118
KAI_ASM_LABEL(label_119)  // Height 3: Multiply loop: Single iteration only
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
KAI_ASM_LABEL(label_120)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_127
    cmp x7, #0x8
    blt label_122
KAI_ASM_LABEL(label_121)  // Height 3: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x8
    ldr d3, [x5], #0x8
    ldr q6, [x21, #0]
    cmp x7, #0x8
    ldr q7, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac50)  // usmmla v16.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac54)  // usmmla v20.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac51)  // usmmla v17.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac55)  // usmmla v21.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac52)  // usmmla v18.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac56)  // usmmla v22.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac53)  // usmmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac57)  // usmmla v23.4s, v2.16b, v7.16b
    bge label_121
KAI_ASM_LABEL(label_122)  // Height 3: Multiply loop: Skip odd blocks
    cbz x7, label_127
    tbz x7, #2, label_124
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    tbz x7, #1, label_123
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    tbz x7, #0, label_126
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    b label_126
KAI_ASM_LABEL(label_123)  // Height 3: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_126
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    b label_126
KAI_ASM_LABEL(label_124)  // Height 3: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_125
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    tbz x7, #0, label_126
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    b label_126
KAI_ASM_LABEL(label_125)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
KAI_ASM_LABEL(label_126)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
KAI_ASM_LABEL(label_127)  // Height 3: Multiply loop: No odd multiplies
    tbz x25, #3, label_128
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_116
KAI_ASM_LABEL(label_128)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x6, x23, x1
    add x5, x6, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x6, #0]
    prfm pstl1keep, [x5, #0]
    uzp1 v16.2d, v16.2d, v20.2d
    uzp1 v17.2d, v17.2d, v21.2d
    uzp1 v18.2d, v18.2d, v22.2d
    uzp1 v19.2d, v19.2d, v23.2d
    bge label_137
    tbz x22, #3, label_132
    st1 { v7.4s }, [x23], #0x10
    st1 { v12.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v9.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    tbz x22, #2, label_130
    st1 { v13.4s }, [x23], #0x10
    st1 { v10.4s }, [x6], #0x10
    st1 { v18.4s }, [x5], #0x10
    tbz x22, #1, label_129
    str d14, [x23], #0x8
    str d11, [x6], #0x8
    str d19, [x5], #0x8
    tbz x22, #0, label_136
    st1 { v14.s }[2], [x23]
    st1 { v11.s }[2], [x6]
    st1 { v19.s }[2], [x5]
    b label_136
KAI_ASM_LABEL(label_129)  // Height 3: Partial direct writeback: partial_0_12
    tbz x22, #0, label_136
    str s14, [x23, #0]
    str s11, [x6, #0]
    str s19, [x5, #0]
    b label_136
KAI_ASM_LABEL(label_130)  // Height 3: Partial direct writeback: partial_1_8
    tbz x22, #1, label_131
    str d13, [x23], #0x8
    str d10, [x6], #0x8
    str d18, [x5], #0x8
    tbz x22, #0, label_136
    st1 { v13.s }[2], [x23]
    st1 { v10.s }[2], [x6]
    st1 { v18.s }[2], [x5]
    b label_136
KAI_ASM_LABEL(label_131)  // Height 3: Partial direct writeback: partial_0_8
    tbz x22, #0, label_136
    str s13, [x23, #0]
    str s10, [x6, #0]
    str s18, [x5, #0]
    b label_136
KAI_ASM_LABEL(label_132)  // Height 3: Partial direct writeback: partial_2_0
    tbz x22, #2, label_134
    st1 { v7.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    tbz x22, #1, label_133
    str d12, [x23], #0x8
    str d9, [x6], #0x8
    str d17, [x5], #0x8
    tbz x22, #0, label_136
    st1 { v12.s }[2], [x23]
    st1 { v9.s }[2], [x6]
    st1 { v17.s }[2], [x5]
    b label_136
KAI_ASM_LABEL(label_133)  // Height 3: Partial direct writeback: partial_0_4
    tbz x22, #0, label_136
    str s12, [x23, #0]
    str s9, [x6, #0]
    str s17, [x5, #0]
    b label_136
KAI_ASM_LABEL(label_134)  // Height 3: Partial direct writeback: partial_1_0
    tbz x22, #1, label_135
    str d7, [x23], #0x8
    str d8, [x6], #0x8
    str d16, [x5], #0x8
    tbz x22, #0, label_136
    st1 { v7.s }[2], [x23]
    st1 { v8.s }[2], [x6]
    st1 { v16.s }[2], [x5]
    b label_136
KAI_ASM_LABEL(label_135)  // Height 3: Partial direct writeback: partial_0_0
    str s7, [x23, #0]
    str s8, [x6, #0]
    str s16, [x5, #0]
KAI_ASM_LABEL(label_136)  // Height 3: Partial direct writeback: Done
    b label_138
KAI_ASM_LABEL(label_137)  // Height 3: Full writeback
    str q7, [x23, #0]
    str q12, [x23, #0x10]
    str q13, [x23, #0x20]
    str q14, [x23, #0x30]
    add x23, x23, #0x40
    str q8, [x6, #0]
    str q9, [x6, #0x10]
    str q10, [x6, #0x20]
    str q11, [x6, #0x30]
    str q16, [x5, #0]
    str q17, [x5, #0x10]
    str q18, [x5, #0x20]
    str q19, [x5, #0x30]
KAI_ASM_LABEL(label_138)  // Height 3: Writeback done
    subs x22, x22, #0x10
    bgt label_94
    b label_277
KAI_ASM_LABEL(label_139)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_140)  // Height 4: Column loop
    cbz x20, label_148
    cmp x22, #0x10
    bge label_146
    cbnz x27, label_145
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_142
KAI_ASM_LABEL(label_141)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_141
KAI_ASM_LABEL(label_142)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_144
KAI_ASM_LABEL(label_143)  // Height 4: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_143
KAI_ASM_LABEL(label_144)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_145)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_147
KAI_ASM_LABEL(label_146)  // Height 4: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_147)  // Height 4: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    mov v16.16b, v8.16b
    mov v20.16b, v12.16b
    mov v17.16b, v9.16b
    mov v21.16b, v13.16b
    mov v18.16b, v10.16b
    mov v22.16b, v14.16b
    mov v19.16b, v11.16b
    mov v23.16b, v15.16b
    b label_160
KAI_ASM_LABEL(label_148)  // Height 4: no bias
    tbz x25, #0, label_159
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_157
    tbz x22, #3, label_152
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v10.4s }, [x23], #0x10
    ld1 { v13.4s }, [x6], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    tbz x22, #2, label_150
    ld1 { v11.4s }, [x23], #0x10
    ld1 { v14.4s }, [x6], #0x10
    ld1 { v19.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    tbz x22, #1, label_149
    ldr d16, [x23], #0x8
    ldr d15, [x6], #0x8
    mov x1, #0x38
    ldr d24, [x5], #0x8
    ldr d23, [x4], #0x8
    tbz x22, #0, label_156
    ld1 { v16.s }[2], [x23]
    ld1 { v15.s }[2], [x6]
    ld1 { v24.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    b label_156
KAI_ASM_LABEL(label_149)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_156
    ldr s16, [x23, #0]
    ldr s15, [x6, #0]
    ldr s24, [x5, #0]
    ldr s23, [x4, #0]
    b label_156
KAI_ASM_LABEL(label_150)  // Height 4: Partial accumulate: partial_1_8
    tbz x22, #1, label_151
    ldr d11, [x23], #0x8
    ldr d14, [x6], #0x8
    mov x1, #0x28
    ldr d19, [x5], #0x8
    ldr d22, [x4], #0x8
    tbz x22, #0, label_156
    ld1 { v11.s }[2], [x23]
    ld1 { v14.s }[2], [x6]
    ld1 { v19.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    b label_156
KAI_ASM_LABEL(label_151)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_156
    ldr s11, [x23, #0]
    ldr s14, [x6, #0]
    ldr s19, [x5, #0]
    ldr s22, [x4, #0]
    b label_156
KAI_ASM_LABEL(label_152)  // Height 4: Partial accumulate: partial_2_0
    tbz x22, #2, label_154
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    tbz x22, #1, label_153
    ldr d10, [x23], #0x8
    ldr d13, [x6], #0x8
    mov x1, #0x18
    ldr d18, [x5], #0x8
    ldr d21, [x4], #0x8
    tbz x22, #0, label_156
    ld1 { v10.s }[2], [x23]
    ld1 { v13.s }[2], [x6]
    ld1 { v18.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    b label_156
KAI_ASM_LABEL(label_153)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_156
    ldr s10, [x23, #0]
    ldr s13, [x6, #0]
    ldr s18, [x5, #0]
    ldr s21, [x4, #0]
    b label_156
KAI_ASM_LABEL(label_154)  // Height 4: Partial accumulate: partial_1_0
    tbz x22, #1, label_155
    ldr d9, [x23], #0x8
    ldr d12, [x6], #0x8
    mov x1, #0x8
    ldr d17, [x5], #0x8
    ldr d20, [x4], #0x8
    tbz x22, #0, label_156
    ld1 { v9.s }[2], [x23]
    ld1 { v12.s }[2], [x6]
    ld1 { v17.s }[2], [x5]
    ld1 { v20.s }[2], [x4]
    b label_156
KAI_ASM_LABEL(label_155)  // Height 4: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    ldr s12, [x6, #0]
    mov x1, #0
    ldr s17, [x5, #0]
    ldr s20, [x4, #0]
KAI_ASM_LABEL(label_156)  // Height 4: Partial accumulate: Done
    sub x23, x23, x1
    b label_158
KAI_ASM_LABEL(label_157)  // Height 4: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
    ldr q12, [x6, #0]
    ldr q13, [x6, #0x10]
    ldr q14, [x6, #0x20]
    ldr q15, [x6, #0x30]
    ldr q17, [x5, #0]
    ldr q18, [x5, #0x10]
    ldr q19, [x5, #0x20]
    ldr q24, [x5, #0x30]
    ldr q20, [x4, #0]
    ldr q21, [x4, #0x10]
    ldr q22, [x4, #0x20]
    ldr q23, [x4, #0x30]
KAI_ASM_LABEL(label_158)  // Height 4: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    zip1 v16.2d, v17.2d, v20.2d
    zip2 v20.2d, v17.2d, v20.2d
    zip1 v17.2d, v18.2d, v21.2d
    zip2 v21.2d, v18.2d, v21.2d
    zip1 v18.2d, v19.2d, v22.2d
    zip2 v22.2d, v19.2d, v22.2d
    zip1 v19.2d, v24.2d, v23.2d
    zip2 v23.2d, v24.2d, v23.2d
    b label_160
KAI_ASM_LABEL(label_159)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_160)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_161
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_163
KAI_ASM_LABEL(label_161)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_162)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_163
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_163)  // Height 4: input setup done
    cmp x7, #0x10
    blt label_166
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    blt label_165
KAI_ASM_LABEL(label_164)  // Height 4: Multiply loop: Main loop head
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    ldr q6, [x21, #0x10]
    bge label_164
KAI_ASM_LABEL(label_165)  // Height 4: Multiply loop: Single iteration only
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
KAI_ASM_LABEL(label_166)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_173
    cmp x7, #0x8
    blt label_168
KAI_ASM_LABEL(label_167)  // Height 4: Multiply loop: Odd block loop
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
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac50)  // usmmla v16.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac54)  // usmmla v20.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac51)  // usmmla v17.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac55)  // usmmla v21.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac52)  // usmmla v18.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac56)  // usmmla v22.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac53)  // usmmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac57)  // usmmla v23.4s, v2.16b, v7.16b
    bge label_167
KAI_ASM_LABEL(label_168)  // Height 4: Multiply loop: Skip odd blocks
    cbz x7, label_173
    tbz x7, #2, label_170
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    tbz x7, #1, label_169
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    tbz x7, #0, label_172
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    b label_172
KAI_ASM_LABEL(label_169)  // Height 4: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_172
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    b label_172
KAI_ASM_LABEL(label_170)  // Height 4: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_171
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    tbz x7, #0, label_172
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    b label_172
KAI_ASM_LABEL(label_171)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
KAI_ASM_LABEL(label_172)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    ldr q6, [x21, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
KAI_ASM_LABEL(label_173)  // Height 4: Multiply loop: No odd multiplies
    tbz x25, #3, label_174
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_162
KAI_ASM_LABEL(label_174)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x6, #0]
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    bge label_183
    tbz x22, #3, label_178
    st1 { v7.4s }, [x23], #0x10
    st1 { v12.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v9.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    st1 { v17.4s }, [x4], #0x10
    tbz x22, #2, label_176
    st1 { v13.4s }, [x23], #0x10
    st1 { v10.4s }, [x6], #0x10
    st1 { v21.4s }, [x5], #0x10
    st1 { v18.4s }, [x4], #0x10
    tbz x22, #1, label_175
    str d14, [x23], #0x8
    str d11, [x6], #0x8
    str d22, [x5], #0x8
    str d19, [x4], #0x8
    tbz x22, #0, label_182
    st1 { v14.s }[2], [x23]
    st1 { v11.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    st1 { v19.s }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_175)  // Height 4: Partial direct writeback: partial_0_12
    tbz x22, #0, label_182
    str s14, [x23, #0]
    str s11, [x6, #0]
    str s22, [x5, #0]
    str s19, [x4, #0]
    b label_182
KAI_ASM_LABEL(label_176)  // Height 4: Partial direct writeback: partial_1_8
    tbz x22, #1, label_177
    str d13, [x23], #0x8
    str d10, [x6], #0x8
    str d21, [x5], #0x8
    str d18, [x4], #0x8
    tbz x22, #0, label_182
    st1 { v13.s }[2], [x23]
    st1 { v10.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    st1 { v18.s }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_177)  // Height 4: Partial direct writeback: partial_0_8
    tbz x22, #0, label_182
    str s13, [x23, #0]
    str s10, [x6, #0]
    str s21, [x5, #0]
    str s18, [x4, #0]
    b label_182
KAI_ASM_LABEL(label_178)  // Height 4: Partial direct writeback: partial_2_0
    tbz x22, #2, label_180
    st1 { v7.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    tbz x22, #1, label_179
    str d12, [x23], #0x8
    str d9, [x6], #0x8
    str d20, [x5], #0x8
    str d17, [x4], #0x8
    tbz x22, #0, label_182
    st1 { v12.s }[2], [x23]
    st1 { v9.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    st1 { v17.s }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_179)  // Height 4: Partial direct writeback: partial_0_4
    tbz x22, #0, label_182
    str s12, [x23, #0]
    str s9, [x6, #0]
    str s20, [x5, #0]
    str s17, [x4, #0]
    b label_182
KAI_ASM_LABEL(label_180)  // Height 4: Partial direct writeback: partial_1_0
    tbz x22, #1, label_181
    str d7, [x23], #0x8
    str d8, [x6], #0x8
    str d15, [x5], #0x8
    str d16, [x4], #0x8
    tbz x22, #0, label_182
    st1 { v7.s }[2], [x23]
    st1 { v8.s }[2], [x6]
    st1 { v15.s }[2], [x5]
    st1 { v16.s }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_181)  // Height 4: Partial direct writeback: partial_0_0
    str s7, [x23, #0]
    str s8, [x6, #0]
    str s15, [x5, #0]
    str s16, [x4, #0]
KAI_ASM_LABEL(label_182)  // Height 4: Partial direct writeback: Done
    b label_184
KAI_ASM_LABEL(label_183)  // Height 4: Full writeback
    str q7, [x23, #0]
    str q12, [x23, #0x10]
    str q13, [x23, #0x20]
    str q14, [x23, #0x30]
    add x23, x23, #0x40
    str q8, [x6, #0]
    str q9, [x6, #0x10]
    str q10, [x6, #0x20]
    str q11, [x6, #0x30]
    str q15, [x5, #0]
    str q20, [x5, #0x10]
    str q21, [x5, #0x20]
    str q22, [x5, #0x30]
    str q16, [x4, #0]
    str q17, [x4, #0x10]
    str q18, [x4, #0x20]
    str q19, [x4, #0x30]
KAI_ASM_LABEL(label_184)  // Height 4: Writeback done
    subs x22, x22, #0x10
    bgt label_140
    b label_277
KAI_ASM_LABEL(label_185)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_186)  // Height 5: Column loop
    cbz x20, label_194
    cmp x22, #0x10
    bge label_192
    cbnz x27, label_191
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_188
KAI_ASM_LABEL(label_187)  // Height 5: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_187
KAI_ASM_LABEL(label_188)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_190
KAI_ASM_LABEL(label_189)  // Height 5: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_189
KAI_ASM_LABEL(label_190)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_191)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_193
KAI_ASM_LABEL(label_192)  // Height 5: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_193)  // Height 5: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    mov v16.16b, v8.16b
    mov v20.16b, v12.16b
    mov v17.16b, v9.16b
    mov v21.16b, v13.16b
    mov v18.16b, v10.16b
    mov v22.16b, v14.16b
    mov v19.16b, v11.16b
    mov v23.16b, v15.16b
    mov v24.16b, v8.16b
    mov v28.16b, v12.16b
    mov v25.16b, v9.16b
    mov v29.16b, v13.16b
    mov v26.16b, v10.16b
    mov v30.16b, v14.16b
    mov v27.16b, v11.16b
    mov v31.16b, v15.16b
    b label_206
KAI_ASM_LABEL(label_194)  // Height 5: no bias
    tbz x25, #0, label_205
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_203
    tbz x22, #3, label_198
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v25.4s }, [x3], #0x10
    ld1 { v10.4s }, [x23], #0x10
    ld1 { v13.4s }, [x6], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    tbz x22, #2, label_196
    ld1 { v11.4s }, [x23], #0x10
    ld1 { v14.4s }, [x6], #0x10
    ld1 { v19.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    tbz x22, #1, label_195
    ldr d16, [x23], #0x8
    ldr d15, [x6], #0x8
    mov x1, #0x38
    ldr d24, [x5], #0x8
    ldr d23, [x4], #0x8
    ldr d6, [x3], #0x8
    tbz x22, #0, label_202
    ld1 { v16.s }[2], [x23]
    ld1 { v15.s }[2], [x6]
    ld1 { v24.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    ld1 { v6.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_195)  // Height 5: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_202
    ldr s16, [x23, #0]
    ldr s15, [x6, #0]
    ldr s24, [x5, #0]
    ldr s23, [x4, #0]
    ldr s6, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_196)  // Height 5: Partial accumulate: partial_1_8
    tbz x22, #1, label_197
    ldr d11, [x23], #0x8
    ldr d14, [x6], #0x8
    mov x1, #0x28
    ldr d19, [x5], #0x8
    ldr d22, [x4], #0x8
    ldr d27, [x3], #0x8
    tbz x22, #0, label_202
    ld1 { v11.s }[2], [x23]
    ld1 { v14.s }[2], [x6]
    ld1 { v19.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    ld1 { v27.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_197)  // Height 5: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_202
    ldr s11, [x23, #0]
    ldr s14, [x6, #0]
    ldr s19, [x5, #0]
    ldr s22, [x4, #0]
    ldr s27, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_198)  // Height 5: Partial accumulate: partial_2_0
    tbz x22, #2, label_200
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v25.4s }, [x3], #0x10
    tbz x22, #1, label_199
    ldr d10, [x23], #0x8
    ldr d13, [x6], #0x8
    mov x1, #0x18
    ldr d18, [x5], #0x8
    ldr d21, [x4], #0x8
    ldr d26, [x3], #0x8
    tbz x22, #0, label_202
    ld1 { v10.s }[2], [x23]
    ld1 { v13.s }[2], [x6]
    ld1 { v18.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    ld1 { v26.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_199)  // Height 5: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_202
    ldr s10, [x23, #0]
    ldr s13, [x6, #0]
    ldr s18, [x5, #0]
    ldr s21, [x4, #0]
    ldr s26, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_200)  // Height 5: Partial accumulate: partial_1_0
    tbz x22, #1, label_201
    ldr d9, [x23], #0x8
    ldr d12, [x6], #0x8
    mov x1, #0x8
    ldr d17, [x5], #0x8
    ldr d20, [x4], #0x8
    ldr d25, [x3], #0x8
    tbz x22, #0, label_202
    ld1 { v9.s }[2], [x23]
    ld1 { v12.s }[2], [x6]
    ld1 { v17.s }[2], [x5]
    ld1 { v20.s }[2], [x4]
    ld1 { v25.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_201)  // Height 5: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    ldr s12, [x6, #0]
    mov x1, #0
    ldr s17, [x5, #0]
    ldr s20, [x4, #0]
    ldr s25, [x3, #0]
KAI_ASM_LABEL(label_202)  // Height 5: Partial accumulate: Done
    sub x23, x23, x1
    b label_204
KAI_ASM_LABEL(label_203)  // Height 5: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
    ldr q12, [x6, #0]
    ldr q13, [x6, #0x10]
    ldr q14, [x6, #0x20]
    ldr q15, [x6, #0x30]
    ldr q17, [x5, #0]
    ldr q18, [x5, #0x10]
    ldr q19, [x5, #0x20]
    ldr q24, [x5, #0x30]
    ldr q20, [x4, #0]
    ldr q21, [x4, #0x10]
    ldr q22, [x4, #0x20]
    ldr q23, [x4, #0x30]
    ldr q25, [x3, #0]
    ldr q26, [x3, #0x10]
    ldr q27, [x3, #0x20]
    ldr q6, [x3, #0x30]
KAI_ASM_LABEL(label_204)  // Height 5: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    zip1 v16.2d, v17.2d, v20.2d
    zip2 v20.2d, v17.2d, v20.2d
    zip1 v17.2d, v18.2d, v21.2d
    zip2 v21.2d, v18.2d, v21.2d
    zip1 v18.2d, v19.2d, v22.2d
    zip2 v22.2d, v19.2d, v22.2d
    zip1 v19.2d, v24.2d, v23.2d
    zip2 v23.2d, v24.2d, v23.2d
    zip1 v24.2d, v25.2d, v28.2d
    zip2 v28.2d, v25.2d, v28.2d
    zip1 v25.2d, v26.2d, v29.2d
    zip2 v29.2d, v26.2d, v29.2d
    zip1 v26.2d, v27.2d, v30.2d
    zip2 v30.2d, v27.2d, v30.2d
    zip1 v27.2d, v6.2d, v31.2d
    zip2 v31.2d, v6.2d, v31.2d
    b label_206
KAI_ASM_LABEL(label_205)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_206)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_207
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_209
KAI_ASM_LABEL(label_207)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_208)  // Height 5: String loop
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
    cbnz x24, label_209
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_209)  // Height 5: input setup done
    cmp x7, #0x10
    blt label_212
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q7, [x21, #0]
    blt label_211
KAI_ASM_LABEL(label_210)  // Height 5: Multiply loop: Main loop head
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    add x2, x2, #0x10
    cmp x7, #0x20
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb8)  // usmmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbc)  // usmmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb9)  // usmmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbd)  // usmmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acba)  // usmmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbe)  // usmmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acbb)  // usmmla v27.4s, v5.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x4e86acbf)  // usmmla v31.4s, v5.16b, v6.16b
    ldr q5, [x2, #0]
    bge label_210
KAI_ASM_LABEL(label_211)  // Height 5: Multiply loop: Single iteration only
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    sub x7, x7, #0x10
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb8)  // usmmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbc)  // usmmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb9)  // usmmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbd)  // usmmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acba)  // usmmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbe)  // usmmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acbb)  // usmmla v27.4s, v5.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbf)  // usmmla v31.4s, v5.16b, v6.16b
KAI_ASM_LABEL(label_212)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_219
    cmp x7, #0x8
    blt label_214
KAI_ASM_LABEL(label_213)  // Height 5: Multiply loop: Odd block loop
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
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac50)  // usmmla v16.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac98)  // usmmla v24.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac54)  // usmmla v20.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9c)  // usmmla v28.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac51)  // usmmla v17.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac99)  // usmmla v25.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac55)  // usmmla v21.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9d)  // usmmla v29.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac52)  // usmmla v18.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9a)  // usmmla v26.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac56)  // usmmla v22.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9e)  // usmmla v30.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac53)  // usmmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9b)  // usmmla v27.4s, v4.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac57)  // usmmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9f)  // usmmla v31.4s, v4.16b, v7.16b
    bge label_213
KAI_ASM_LABEL(label_214)  // Height 5: Multiply loop: Skip odd blocks
    cbz x7, label_219
    tbz x7, #2, label_216
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    tbz x7, #1, label_215
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    ld1 { v5.h }[2], [x2], #0x2
    tbz x7, #0, label_218
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    ld1 { v5.b }[6], [x2]
    b label_218
KAI_ASM_LABEL(label_215)  // Height 5: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_218
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    ld1 { v5.b }[4], [x2]
    b label_218
KAI_ASM_LABEL(label_216)  // Height 5: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_217
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    ldr h5, [x2], #0x2
    tbz x7, #0, label_218
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    ld1 { v5.b }[2], [x2]
    b label_218
KAI_ASM_LABEL(label_217)  // Height 5: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
    ldr b5, [x2, #0]
KAI_ASM_LABEL(label_218)  // Height 5: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
KAI_ASM_LABEL(label_219)  // Height 5: Multiply loop: No odd multiplies
    tbz x25, #3, label_220
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_208
KAI_ASM_LABEL(label_220)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x6, #0]
    add x3, x4, x1
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    prfm pstl1keep, [x5, #0]
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    uzp1 v24.2d, v24.2d, v28.2d
    uzp1 v25.2d, v25.2d, v29.2d
    uzp1 v26.2d, v26.2d, v30.2d
    uzp1 v27.2d, v27.2d, v31.2d
    bge label_229
    tbz x22, #3, label_224
    st1 { v7.4s }, [x23], #0x10
    st1 { v12.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v9.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    st1 { v17.4s }, [x4], #0x10
    st1 { v24.4s }, [x3], #0x10
    st1 { v25.4s }, [x3], #0x10
    tbz x22, #2, label_222
    st1 { v13.4s }, [x23], #0x10
    st1 { v10.4s }, [x6], #0x10
    st1 { v21.4s }, [x5], #0x10
    st1 { v18.4s }, [x4], #0x10
    st1 { v26.4s }, [x3], #0x10
    tbz x22, #1, label_221
    str d14, [x23], #0x8
    str d11, [x6], #0x8
    str d22, [x5], #0x8
    str d19, [x4], #0x8
    str d27, [x3], #0x8
    tbz x22, #0, label_228
    st1 { v14.s }[2], [x23]
    st1 { v11.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    st1 { v19.s }[2], [x4]
    st1 { v27.s }[2], [x3]
    b label_228
KAI_ASM_LABEL(label_221)  // Height 5: Partial direct writeback: partial_0_12
    tbz x22, #0, label_228
    str s14, [x23, #0]
    str s11, [x6, #0]
    str s22, [x5, #0]
    str s19, [x4, #0]
    str s27, [x3, #0]
    b label_228
KAI_ASM_LABEL(label_222)  // Height 5: Partial direct writeback: partial_1_8
    tbz x22, #1, label_223
    str d13, [x23], #0x8
    str d10, [x6], #0x8
    str d21, [x5], #0x8
    str d18, [x4], #0x8
    str d26, [x3], #0x8
    tbz x22, #0, label_228
    st1 { v13.s }[2], [x23]
    st1 { v10.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    st1 { v18.s }[2], [x4]
    st1 { v26.s }[2], [x3]
    b label_228
KAI_ASM_LABEL(label_223)  // Height 5: Partial direct writeback: partial_0_8
    tbz x22, #0, label_228
    str s13, [x23, #0]
    str s10, [x6, #0]
    str s21, [x5, #0]
    str s18, [x4, #0]
    str s26, [x3, #0]
    b label_228
KAI_ASM_LABEL(label_224)  // Height 5: Partial direct writeback: partial_2_0
    tbz x22, #2, label_226
    st1 { v7.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    st1 { v24.4s }, [x3], #0x10
    tbz x22, #1, label_225
    str d12, [x23], #0x8
    str d9, [x6], #0x8
    str d20, [x5], #0x8
    str d17, [x4], #0x8
    str d25, [x3], #0x8
    tbz x22, #0, label_228
    st1 { v12.s }[2], [x23]
    st1 { v9.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    st1 { v17.s }[2], [x4]
    st1 { v25.s }[2], [x3]
    b label_228
KAI_ASM_LABEL(label_225)  // Height 5: Partial direct writeback: partial_0_4
    tbz x22, #0, label_228
    str s12, [x23, #0]
    str s9, [x6, #0]
    str s20, [x5, #0]
    str s17, [x4, #0]
    str s25, [x3, #0]
    b label_228
KAI_ASM_LABEL(label_226)  // Height 5: Partial direct writeback: partial_1_0
    tbz x22, #1, label_227
    str d7, [x23], #0x8
    str d8, [x6], #0x8
    str d15, [x5], #0x8
    str d16, [x4], #0x8
    str d24, [x3], #0x8
    tbz x22, #0, label_228
    st1 { v7.s }[2], [x23]
    st1 { v8.s }[2], [x6]
    st1 { v15.s }[2], [x5]
    st1 { v16.s }[2], [x4]
    st1 { v24.s }[2], [x3]
    b label_228
KAI_ASM_LABEL(label_227)  // Height 5: Partial direct writeback: partial_0_0
    str s7, [x23, #0]
    str s8, [x6, #0]
    str s15, [x5, #0]
    str s16, [x4, #0]
    str s24, [x3, #0]
KAI_ASM_LABEL(label_228)  // Height 5: Partial direct writeback: Done
    b label_230
KAI_ASM_LABEL(label_229)  // Height 5: Full writeback
    str q7, [x23, #0]
    str q12, [x23, #0x10]
    str q13, [x23, #0x20]
    str q14, [x23, #0x30]
    add x23, x23, #0x40
    str q8, [x6, #0]
    str q9, [x6, #0x10]
    str q10, [x6, #0x20]
    str q11, [x6, #0x30]
    str q15, [x5, #0]
    str q20, [x5, #0x10]
    str q21, [x5, #0x20]
    str q22, [x5, #0x30]
    str q16, [x4, #0]
    str q17, [x4, #0x10]
    str q18, [x4, #0x20]
    str q19, [x4, #0x30]
    str q24, [x3, #0]
    str q25, [x3, #0x10]
    str q26, [x3, #0x20]
    str q27, [x3, #0x30]
KAI_ASM_LABEL(label_230)  // Height 5: Writeback done
    subs x22, x22, #0x10
    bgt label_186
    b label_277
KAI_ASM_LABEL(label_231)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_232)  // Height 6: Column loop
    cbz x20, label_240
    cmp x22, #0x10
    bge label_238
    cbnz x27, label_237
    mov x2, x22
    mov x1, SP
    cmp x2, #0x4
    blt label_234
KAI_ASM_LABEL(label_233)  // Height 6: Partial bias load: vector loop
    ld1 { v16.4s }, [x20], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_233
KAI_ASM_LABEL(label_234)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_236
KAI_ASM_LABEL(label_235)  // Height 6: Partial bias load: element loop
    ldr s16, [x20], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_235
KAI_ASM_LABEL(label_236)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_237)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_239
KAI_ASM_LABEL(label_238)  // Height 6: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_239)  // Height 6: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    mov v16.16b, v8.16b
    mov v20.16b, v12.16b
    mov v17.16b, v9.16b
    mov v21.16b, v13.16b
    mov v18.16b, v10.16b
    mov v22.16b, v14.16b
    mov v19.16b, v11.16b
    mov v23.16b, v15.16b
    mov v24.16b, v8.16b
    mov v28.16b, v12.16b
    mov v25.16b, v9.16b
    mov v29.16b, v13.16b
    mov v26.16b, v10.16b
    mov v30.16b, v14.16b
    mov v27.16b, v11.16b
    mov v31.16b, v15.16b
    b label_252
KAI_ASM_LABEL(label_240)  // Height 6: no bias
    tbz x25, #0, label_251
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    add x2, x3, x1
    bge label_249
    tbz x22, #3, label_244
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v25.4s }, [x3], #0x10
    ld1 { v28.4s }, [x2], #0x10
    ld1 { v10.4s }, [x23], #0x10
    ld1 { v13.4s }, [x6], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    ld1 { v29.4s }, [x2], #0x10
    tbz x22, #2, label_242
    ld1 { v11.4s }, [x23], #0x10
    ld1 { v14.4s }, [x6], #0x10
    ld1 { v19.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    ld1 { v30.4s }, [x2], #0x10
    tbz x22, #1, label_241
    ldr d16, [x23], #0x8
    ldr d15, [x6], #0x8
    mov x1, #0x38
    ldr d24, [x5], #0x8
    ldr d23, [x4], #0x8
    ldr d6, [x3], #0x8
    ldr d31, [x2], #0x8
    tbz x22, #0, label_248
    ld1 { v16.s }[2], [x23]
    ld1 { v15.s }[2], [x6]
    ld1 { v24.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    ld1 { v6.s }[2], [x3]
    ld1 { v31.s }[2], [x2]
    b label_248
KAI_ASM_LABEL(label_241)  // Height 6: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x22, #0, label_248
    ldr s16, [x23, #0]
    ldr s15, [x6, #0]
    ldr s24, [x5, #0]
    ldr s23, [x4, #0]
    ldr s6, [x3, #0]
    ldr s31, [x2, #0]
    b label_248
KAI_ASM_LABEL(label_242)  // Height 6: Partial accumulate: partial_1_8
    tbz x22, #1, label_243
    ldr d11, [x23], #0x8
    ldr d14, [x6], #0x8
    mov x1, #0x28
    ldr d19, [x5], #0x8
    ldr d22, [x4], #0x8
    ldr d27, [x3], #0x8
    ldr d30, [x2], #0x8
    tbz x22, #0, label_248
    ld1 { v11.s }[2], [x23]
    ld1 { v14.s }[2], [x6]
    ld1 { v19.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    ld1 { v27.s }[2], [x3]
    ld1 { v30.s }[2], [x2]
    b label_248
KAI_ASM_LABEL(label_243)  // Height 6: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x22, #0, label_248
    ldr s11, [x23, #0]
    ldr s14, [x6, #0]
    ldr s19, [x5, #0]
    ldr s22, [x4, #0]
    ldr s27, [x3, #0]
    ldr s30, [x2, #0]
    b label_248
KAI_ASM_LABEL(label_244)  // Height 6: Partial accumulate: partial_2_0
    tbz x22, #2, label_246
    ld1 { v9.4s }, [x23], #0x10
    ld1 { v12.4s }, [x6], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v25.4s }, [x3], #0x10
    ld1 { v28.4s }, [x2], #0x10
    tbz x22, #1, label_245
    ldr d10, [x23], #0x8
    ldr d13, [x6], #0x8
    mov x1, #0x18
    ldr d18, [x5], #0x8
    ldr d21, [x4], #0x8
    ldr d26, [x3], #0x8
    ldr d29, [x2], #0x8
    tbz x22, #0, label_248
    ld1 { v10.s }[2], [x23]
    ld1 { v13.s }[2], [x6]
    ld1 { v18.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    ld1 { v26.s }[2], [x3]
    ld1 { v29.s }[2], [x2]
    b label_248
KAI_ASM_LABEL(label_245)  // Height 6: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x22, #0, label_248
    ldr s10, [x23, #0]
    ldr s13, [x6, #0]
    ldr s18, [x5, #0]
    ldr s21, [x4, #0]
    ldr s26, [x3, #0]
    ldr s29, [x2, #0]
    b label_248
KAI_ASM_LABEL(label_246)  // Height 6: Partial accumulate: partial_1_0
    tbz x22, #1, label_247
    ldr d9, [x23], #0x8
    ldr d12, [x6], #0x8
    mov x1, #0x8
    ldr d17, [x5], #0x8
    ldr d20, [x4], #0x8
    ldr d25, [x3], #0x8
    ldr d28, [x2], #0x8
    tbz x22, #0, label_248
    ld1 { v9.s }[2], [x23]
    ld1 { v12.s }[2], [x6]
    ld1 { v17.s }[2], [x5]
    ld1 { v20.s }[2], [x4]
    ld1 { v25.s }[2], [x3]
    ld1 { v28.s }[2], [x2]
    b label_248
KAI_ASM_LABEL(label_247)  // Height 6: Partial accumulate: partial_0_0
    ldr s9, [x23, #0]
    ldr s12, [x6, #0]
    mov x1, #0
    ldr s17, [x5, #0]
    ldr s20, [x4, #0]
    ldr s25, [x3, #0]
    ldr s28, [x2, #0]
KAI_ASM_LABEL(label_248)  // Height 6: Partial accumulate: Done
    sub x23, x23, x1
    b label_250
KAI_ASM_LABEL(label_249)  // Height 6: full accumulate
    ldr q9, [x23, #0]
    ldr q10, [x23, #0x10]
    ldr q11, [x23, #0x20]
    ldr q16, [x23, #0x30]
    ldr q12, [x6, #0]
    ldr q13, [x6, #0x10]
    ldr q14, [x6, #0x20]
    ldr q15, [x6, #0x30]
    ldr q17, [x5, #0]
    ldr q18, [x5, #0x10]
    ldr q19, [x5, #0x20]
    ldr q24, [x5, #0x30]
    ldr q20, [x4, #0]
    ldr q21, [x4, #0x10]
    ldr q22, [x4, #0x20]
    ldr q23, [x4, #0x30]
    ldr q25, [x3, #0]
    ldr q26, [x3, #0x10]
    ldr q27, [x3, #0x20]
    ldr q6, [x3, #0x30]
    ldr q28, [x2, #0]
    ldr q29, [x2, #0x10]
    ldr q30, [x2, #0x20]
    ldr q31, [x2, #0x30]
KAI_ASM_LABEL(label_250)  // Height 6: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    zip1 v16.2d, v17.2d, v20.2d
    zip2 v20.2d, v17.2d, v20.2d
    zip1 v17.2d, v18.2d, v21.2d
    zip2 v21.2d, v18.2d, v21.2d
    zip1 v18.2d, v19.2d, v22.2d
    zip2 v22.2d, v19.2d, v22.2d
    zip1 v19.2d, v24.2d, v23.2d
    zip2 v23.2d, v24.2d, v23.2d
    zip1 v24.2d, v25.2d, v28.2d
    zip2 v28.2d, v25.2d, v28.2d
    zip1 v25.2d, v26.2d, v29.2d
    zip2 v29.2d, v26.2d, v29.2d
    zip1 v26.2d, v27.2d, v30.2d
    zip2 v30.2d, v27.2d, v30.2d
    zip1 v27.2d, v6.2d, v31.2d
    zip2 v31.2d, v6.2d, v31.2d
    b label_252
KAI_ASM_LABEL(label_251)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_252)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_253
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_255
KAI_ASM_LABEL(label_253)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_254)  // Height 6: String loop
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
    cbnz x24, label_255
    add x8, x8, x3
    add x6, x6, x3
    add x5, x5, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_255)  // Height 6: input setup done
    cmp x7, #0x10
    blt label_258
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x20
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    ldr q7, [x21, #0]
    blt label_257
KAI_ASM_LABEL(label_256)  // Height 6: Multiply loop: Main loop head
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    add x2, x2, #0x10
    add x1, x1, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    cmp x7, #0x20
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb8)  // usmmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbc)  // usmmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb9)  // usmmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbd)  // usmmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acba)  // usmmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbe)  // usmmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acbb)  // usmmla v27.4s, v5.16b, v7.16b
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x4e86acbf)  // usmmla v31.4s, v5.16b, v6.16b
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    bge label_256
KAI_ASM_LABEL(label_257)  // Height 6: Multiply loop: Single iteration only
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
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    add x1, x1, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    prfm pldl1keep, [x5, #0x80]
    sub x7, x7, #0x10
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x80]
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x90]
    KAI_ASM_INST(0x4e87ac28)  // usmmla v8.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac70)  // usmmla v16.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb8)  // usmmla v24.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xa0]
    KAI_ASM_INST(0x4e86ac2c)  // usmmla v12.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac74)  // usmmla v20.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbc)  // usmmla v28.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xb0]
    KAI_ASM_INST(0x4e87ac29)  // usmmla v9.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac71)  // usmmla v17.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acb9)  // usmmla v25.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xc0]
    KAI_ASM_INST(0x4e86ac2d)  // usmmla v13.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac75)  // usmmla v21.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbd)  // usmmla v29.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xd0]
    KAI_ASM_INST(0x4e87ac2a)  // usmmla v10.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac72)  // usmmla v18.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acba)  // usmmla v26.4s, v5.16b, v7.16b
    ldr q7, [x21, #0xe0]
    KAI_ASM_INST(0x4e86ac2e)  // usmmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac76)  // usmmla v22.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbe)  // usmmla v30.4s, v5.16b, v6.16b
    ldr q6, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4e87ac2b)  // usmmla v11.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e87ac73)  // usmmla v19.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e87acbb)  // usmmla v27.4s, v5.16b, v7.16b
    KAI_ASM_INST(0x4e86ac2f)  // usmmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e86ac77)  // usmmla v23.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e86acbf)  // usmmla v31.4s, v5.16b, v6.16b
KAI_ASM_LABEL(label_258)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_265
    cmp x7, #0x8
    blt label_260
KAI_ASM_LABEL(label_259)  // Height 6: Multiply loop: Odd block loop
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
    KAI_ASM_INST(0x4e86ac08)  // usmmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac50)  // usmmla v16.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac98)  // usmmla v24.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x20]
    KAI_ASM_INST(0x4e87ac0c)  // usmmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac54)  // usmmla v20.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9c)  // usmmla v28.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e86ac09)  // usmmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac51)  // usmmla v17.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac99)  // usmmla v25.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4e87ac0d)  // usmmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac55)  // usmmla v21.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9d)  // usmmla v29.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4e86ac0a)  // usmmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac52)  // usmmla v18.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9a)  // usmmla v26.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4e87ac0e)  // usmmla v14.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac56)  // usmmla v22.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9e)  // usmmla v30.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e86ac0b)  // usmmla v11.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac53)  // usmmla v19.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9b)  // usmmla v27.4s, v4.16b, v6.16b
    KAI_ASM_INST(0x4e87ac0f)  // usmmla v15.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac57)  // usmmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9f)  // usmmla v31.4s, v4.16b, v7.16b
    bge label_259
KAI_ASM_LABEL(label_260)  // Height 6: Multiply loop: Skip odd blocks
    cbz x7, label_265
    tbz x7, #2, label_262
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    ldr s6, [x1], #0x4
    tbz x7, #1, label_261
    ld1 { v1.h }[2], [x8], #0x2
    ld1 { v2.h }[2], [x6], #0x2
    ld1 { v3.h }[2], [x5], #0x2
    ld1 { v4.h }[2], [x4], #0x2
    ld1 { v5.h }[2], [x2], #0x2
    ld1 { v6.h }[2], [x1], #0x2
    tbz x7, #0, label_264
    ld1 { v1.b }[6], [x8]
    ld1 { v2.b }[6], [x6]
    ld1 { v3.b }[6], [x5]
    ld1 { v4.b }[6], [x4]
    ld1 { v5.b }[6], [x2]
    ld1 { v6.b }[6], [x1]
    b label_264
KAI_ASM_LABEL(label_261)  // Height 6: Multiply loop: Ragged operand read: partial_0_4
    tbz x7, #0, label_264
    ld1 { v1.b }[4], [x8]
    ld1 { v2.b }[4], [x6]
    ld1 { v3.b }[4], [x5]
    ld1 { v4.b }[4], [x4]
    ld1 { v5.b }[4], [x2]
    ld1 { v6.b }[4], [x1]
    b label_264
KAI_ASM_LABEL(label_262)  // Height 6: Multiply loop: Ragged operand read: partial_1_0
    tbz x7, #1, label_263
    ldr h1, [x8], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    ldr h4, [x4], #0x2
    ldr h5, [x2], #0x2
    ldr h6, [x1], #0x2
    tbz x7, #0, label_264
    ld1 { v1.b }[2], [x8]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    ld1 { v4.b }[2], [x4]
    ld1 { v5.b }[2], [x2]
    ld1 { v6.b }[2], [x1]
    b label_264
KAI_ASM_LABEL(label_263)  // Height 6: Multiply loop: Ragged operand read: partial_0_0
    ldr b1, [x8, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
    ldr b4, [x4, #0]
    ldr b5, [x2, #0]
    ldr b6, [x1, #0]
KAI_ASM_LABEL(label_264)  // Height 6: Multiply loop: Ragged operand read: Done
    ldr q7, [x21, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x4e87ac08)  // usmmla v8.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac50)  // usmmla v16.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac98)  // usmmla v24.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x4e86ac0c)  // usmmla v12.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac54)  // usmmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9c)  // usmmla v28.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4e87ac09)  // usmmla v9.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac51)  // usmmla v17.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac99)  // usmmla v25.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x40]
    KAI_ASM_INST(0x4e86ac0d)  // usmmla v13.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac55)  // usmmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9d)  // usmmla v29.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4e87ac0a)  // usmmla v10.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac52)  // usmmla v18.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9a)  // usmmla v26.4s, v4.16b, v7.16b
    ldr q7, [x21, #0x60]
    KAI_ASM_INST(0x4e86ac0e)  // usmmla v14.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac56)  // usmmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9e)  // usmmla v30.4s, v4.16b, v6.16b
    ldr q6, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e87ac0b)  // usmmla v11.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e87ac53)  // usmmla v19.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e87ac9b)  // usmmla v27.4s, v4.16b, v7.16b
    KAI_ASM_INST(0x4e86ac0f)  // usmmla v15.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e86ac57)  // usmmla v23.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e86ac9f)  // usmmla v31.4s, v4.16b, v6.16b
KAI_ASM_LABEL(label_265)  // Height 6: Multiply loop: No odd multiplies
    tbz x25, #3, label_266
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_254
KAI_ASM_LABEL(label_266)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    prfm pstl1keep, [x23, #0]
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x6, x23, x1
    add x5, x6, x1
    add x4, x5, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    prfm pstl1keep, [x6, #0]
    add x3, x4, x1
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    prfm pstl1keep, [x5, #0]
    add x2, x3, x1
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    prfm pstl1keep, [x4, #0]
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    prfm pstl1keep, [x3, #0]
    prfm pstl1keep, [x2, #0]
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    uzp1 v23.2d, v24.2d, v28.2d
    uzp2 v24.2d, v24.2d, v28.2d
    uzp1 v28.2d, v25.2d, v29.2d
    uzp2 v25.2d, v25.2d, v29.2d
    uzp1 v29.2d, v26.2d, v30.2d
    uzp2 v26.2d, v26.2d, v30.2d
    uzp1 v30.2d, v27.2d, v31.2d
    uzp2 v27.2d, v27.2d, v31.2d
    bge label_275
    tbz x22, #3, label_270
    st1 { v7.4s }, [x23], #0x10
    st1 { v12.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v9.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    st1 { v17.4s }, [x4], #0x10
    st1 { v23.4s }, [x3], #0x10
    st1 { v28.4s }, [x3], #0x10
    st1 { v24.4s }, [x2], #0x10
    st1 { v25.4s }, [x2], #0x10
    tbz x22, #2, label_268
    st1 { v13.4s }, [x23], #0x10
    st1 { v10.4s }, [x6], #0x10
    st1 { v21.4s }, [x5], #0x10
    st1 { v18.4s }, [x4], #0x10
    st1 { v29.4s }, [x3], #0x10
    st1 { v26.4s }, [x2], #0x10
    tbz x22, #1, label_267
    str d14, [x23], #0x8
    str d11, [x6], #0x8
    str d22, [x5], #0x8
    str d19, [x4], #0x8
    str d30, [x3], #0x8
    str d27, [x2], #0x8
    tbz x22, #0, label_274
    st1 { v14.s }[2], [x23]
    st1 { v11.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    st1 { v19.s }[2], [x4]
    st1 { v30.s }[2], [x3]
    st1 { v27.s }[2], [x2]
    b label_274
KAI_ASM_LABEL(label_267)  // Height 6: Partial direct writeback: partial_0_12
    tbz x22, #0, label_274
    str s14, [x23, #0]
    str s11, [x6, #0]
    str s22, [x5, #0]
    str s19, [x4, #0]
    str s30, [x3, #0]
    str s27, [x2, #0]
    b label_274
KAI_ASM_LABEL(label_268)  // Height 6: Partial direct writeback: partial_1_8
    tbz x22, #1, label_269
    str d13, [x23], #0x8
    str d10, [x6], #0x8
    str d21, [x5], #0x8
    str d18, [x4], #0x8
    str d29, [x3], #0x8
    str d26, [x2], #0x8
    tbz x22, #0, label_274
    st1 { v13.s }[2], [x23]
    st1 { v10.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    st1 { v18.s }[2], [x4]
    st1 { v29.s }[2], [x3]
    st1 { v26.s }[2], [x2]
    b label_274
KAI_ASM_LABEL(label_269)  // Height 6: Partial direct writeback: partial_0_8
    tbz x22, #0, label_274
    str s13, [x23, #0]
    str s10, [x6, #0]
    str s21, [x5, #0]
    str s18, [x4, #0]
    str s29, [x3, #0]
    str s26, [x2, #0]
    b label_274
KAI_ASM_LABEL(label_270)  // Height 6: Partial direct writeback: partial_2_0
    tbz x22, #2, label_272
    st1 { v7.4s }, [x23], #0x10
    st1 { v8.4s }, [x6], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x4], #0x10
    st1 { v23.4s }, [x3], #0x10
    st1 { v24.4s }, [x2], #0x10
    tbz x22, #1, label_271
    str d12, [x23], #0x8
    str d9, [x6], #0x8
    str d20, [x5], #0x8
    str d17, [x4], #0x8
    str d28, [x3], #0x8
    str d25, [x2], #0x8
    tbz x22, #0, label_274
    st1 { v12.s }[2], [x23]
    st1 { v9.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    st1 { v17.s }[2], [x4]
    st1 { v28.s }[2], [x3]
    st1 { v25.s }[2], [x2]
    b label_274
KAI_ASM_LABEL(label_271)  // Height 6: Partial direct writeback: partial_0_4
    tbz x22, #0, label_274
    str s12, [x23, #0]
    str s9, [x6, #0]
    str s20, [x5, #0]
    str s17, [x4, #0]
    str s28, [x3, #0]
    str s25, [x2, #0]
    b label_274
KAI_ASM_LABEL(label_272)  // Height 6: Partial direct writeback: partial_1_0
    tbz x22, #1, label_273
    str d7, [x23], #0x8
    str d8, [x6], #0x8
    str d15, [x5], #0x8
    str d16, [x4], #0x8
    str d23, [x3], #0x8
    str d24, [x2], #0x8
    tbz x22, #0, label_274
    st1 { v7.s }[2], [x23]
    st1 { v8.s }[2], [x6]
    st1 { v15.s }[2], [x5]
    st1 { v16.s }[2], [x4]
    st1 { v23.s }[2], [x3]
    st1 { v24.s }[2], [x2]
    b label_274
KAI_ASM_LABEL(label_273)  // Height 6: Partial direct writeback: partial_0_0
    str s7, [x23, #0]
    str s8, [x6, #0]
    str s15, [x5, #0]
    str s16, [x4, #0]
    str s23, [x3, #0]
    str s24, [x2, #0]
KAI_ASM_LABEL(label_274)  // Height 6: Partial direct writeback: Done
    b label_276
KAI_ASM_LABEL(label_275)  // Height 6: Full writeback
    str q7, [x23, #0]
    str q12, [x23, #0x10]
    str q13, [x23, #0x20]
    str q14, [x23, #0x30]
    add x23, x23, #0x40
    str q8, [x6, #0]
    str q9, [x6, #0x10]
    str q10, [x6, #0x20]
    str q11, [x6, #0x30]
    str q15, [x5, #0]
    str q20, [x5, #0x10]
    str q21, [x5, #0x20]
    str q22, [x5, #0x30]
    str q16, [x4, #0]
    str q17, [x4, #0x10]
    str q18, [x4, #0x20]
    str q19, [x4, #0x30]
    str q23, [x3, #0]
    str q28, [x3, #0x10]
    str q29, [x3, #0x20]
    str q30, [x3, #0x30]
    str q24, [x2, #0]
    str q25, [x2, #0x10]
    str q26, [x2, #0x20]
    str q27, [x2, #0x30]
KAI_ASM_LABEL(label_276)  // Height 6: Writeback done
    subs x22, x22, #0x10
    bgt label_232
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_277)  // Exit
    add SP, SP, #0x40
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_u8s8s32_mmla_6x16)

    KAI_ASM_END
