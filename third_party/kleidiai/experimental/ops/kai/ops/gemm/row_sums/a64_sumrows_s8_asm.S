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

    KAI_ASM_CODE(a64_sumrows_s8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_sumrows_s8)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_sumrows_s8)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_sumrows_s8)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    mov x25, #0
    ldr x3, [x0, #0x18]
    cmp x3, #0
    cset x3, NE
    dup v28.4s, w2
    ldr x24, [x0, #0x20]
    ldr x23, [x0, #0x28]
    ldr x22, [x0, #0]
    neg v28.4s, v28.4s
    lsl x21, x3, #0x3
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x22, #0x4
    bge label_55
    cmp x22, #0x2
    bgt label_37
    beq label_19
    movi v27.8h, #0
    movi v26.4s, #0
    mov x20, #0
    ldr x4, [x0, #0x10]
    mov x8, #0
    tbz x21, #3, label_3
KAI_ASM_LABEL(label_2)  // Height 1: String loop
    ldr x3, [x24, x8, LSL #0x3]
    ldr x2, [x0, #0x18]
    add x3, x3, x25, LSL #3
    ldr x7, [x3, #0]
    ldr x3, [x2, x8, LSL #0x3]
    cbnz x8, label_4
    add x7, x7, x23
    b label_4
KAI_ASM_LABEL(label_3)  // Height 1: setup direct input
    mov x3, x4
    mov x7, x24
KAI_ASM_LABEL(label_4)  // Height 1: input setup done
    cmp x3, #0x10
    blt label_8
    cmp x3, #0x20
    blt label_7
KAI_ASM_LABEL(label_5)  // Height 1: Multiply loop: Main loop head
    ldr q19, [x7, #0]
    cmp x20, #0x7e
    add x7, x7, #0x10
    blt label_6
    sadalp v26.4s, v27.8h
    movi v27.8h, #0
    mov x20, #0
KAI_ASM_LABEL(label_6)  // Height 1: Multiply loop: unique 1: no collapse
    sub x3, x3, #0x10
    sadalp v27.8h, v19.16b
    add x20, x20, #0x1
    cmp x3, #0x20
    bge label_5
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Single iteration only
    ldr q16, [x7, #0]
    sub x3, x3, #0x10
    add x7, x7, #0x10
    sadalp v27.8h, v16.16b
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: Main loop skip
    cbz x3, label_17
    tbz x3, #3, label_12
    ldr d19, [x7], #0x8
    tbz x3, #2, label_10
    ld1 { v19.s }[2], [x7], #0x4
    tbz x3, #1, label_9
    ld1 { v19.h }[6], [x7], #0x2
    tbz x3, #0, label_16
    ld1 { v19.b }[14], [x7]
    b label_16
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Ragged operand read: partial_1_12
    tbz x3, #0, label_16
    ld1 { v19.b }[12], [x7]
    b label_16
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Ragged operand read: partial_2_8
    tbz x3, #1, label_11
    ld1 { v19.h }[4], [x7], #0x2
    tbz x3, #0, label_16
    ld1 { v19.b }[10], [x7]
    b label_16
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Ragged operand read: partial_1_8
    tbz x3, #0, label_16
    ld1 { v19.b }[8], [x7]
    b label_16
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: Ragged operand read: partial_4_0
    tbz x3, #2, label_14
    ldr s19, [x7], #0x4
    tbz x3, #1, label_13
    ld1 { v19.h }[2], [x7], #0x2
    tbz x3, #0, label_16
    ld1 { v19.b }[6], [x7]
    b label_16
KAI_ASM_LABEL(label_13)  // Height 1: Multiply loop: Ragged operand read: partial_1_4
    tbz x3, #0, label_16
    ld1 { v19.b }[4], [x7]
    b label_16
KAI_ASM_LABEL(label_14)  // Height 1: Multiply loop: Ragged operand read: partial_2_0
    tbz x3, #1, label_15
    ldr h19, [x7], #0x2
    tbz x3, #0, label_16
    ld1 { v19.b }[2], [x7]
    b label_16
KAI_ASM_LABEL(label_15)  // Height 1: Multiply loop: Ragged operand read: partial_1_0
    ldr b19, [x7, #0]
KAI_ASM_LABEL(label_16)  // Height 1: Multiply loop: Ragged operand read: Done
    sadalp v27.8h, v19.16b
KAI_ASM_LABEL(label_17)  // Height 1: Multiply loop: No odd multiplies
    tbz x21, #3, label_18
    add x8, x8, #0x1
    cmp x8, x4
    bne label_2
KAI_ASM_LABEL(label_18)  // Height 1: no string loop
    sadalp v26.4s, v27.8h
    addp v26.4s, v26.4s, v26.4s
    addp v26.4s, v26.4s, v26.4s
    mul v26.4s, v26.4s, v28.4s
    str s26, [x1], #0x4
    b label_74
KAI_ASM_LABEL(label_19)  // Height 2
    movi v27.8h, #0
    movi v26.4s, #0
    mov x20, #0
    ldr x4, [x0, #0x10]
    movi v25.8h, #0
    movi v24.4s, #0
    mov x8, #0
    tbz x21, #3, label_21
KAI_ASM_LABEL(label_20)  // Height 2: String loop
    ldr x3, [x24, x8, LSL #0x3]
    ldr x2, [x0, #0x18]
    add x3, x3, x25, LSL #3
    ldr x7, [x3, #0]
    ldr x6, [x3, #0x8]
    ldr x3, [x2, x8, LSL #0x3]
    cbnz x8, label_22
    add x7, x7, x23
    add x6, x6, x23
    b label_22
KAI_ASM_LABEL(label_21)  // Height 2: setup direct input
    mov x7, x24
    mov x3, x4
    add x6, x7, x23
KAI_ASM_LABEL(label_22)  // Height 2: input setup done
    cmp x3, #0x10
    blt label_26
    cmp x3, #0x20
    blt label_25
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Main loop head
    ldr q19, [x7, #0]
    ldr q18, [x6, #0]
    cmp x20, #0x7e
    add x7, x7, #0x10
    add x6, x6, #0x10
    blt label_24
    sadalp v26.4s, v27.8h
    movi v27.8h, #0
    mov x20, #0
    sadalp v24.4s, v25.8h
    movi v25.8h, #0
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: unique 2: no collapse
    sub x3, x3, #0x10
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    add x20, x20, #0x1
    cmp x3, #0x20
    bge label_23
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: Single iteration only
    ldr q17, [x7, #0]
    ldr q16, [x6, #0]
    sub x3, x3, #0x10
    add x7, x7, #0x10
    add x6, x6, #0x10
    sadalp v27.8h, v17.16b
    sadalp v25.8h, v16.16b
KAI_ASM_LABEL(label_26)  // Height 2: Multiply loop: Main loop skip
    cbz x3, label_35
    tbz x3, #3, label_30
    ldr d19, [x7], #0x8
    ldr d18, [x6], #0x8
    tbz x3, #2, label_28
    ld1 { v19.s }[2], [x7], #0x4
    ld1 { v18.s }[2], [x6], #0x4
    tbz x3, #1, label_27
    ld1 { v19.h }[6], [x7], #0x2
    ld1 { v18.h }[6], [x6], #0x2
    tbz x3, #0, label_34
    ld1 { v19.b }[14], [x7]
    ld1 { v18.b }[14], [x6]
    b label_34
KAI_ASM_LABEL(label_27)  // Height 2: Multiply loop: Ragged operand read: partial_1_12
    tbz x3, #0, label_34
    ld1 { v19.b }[12], [x7]
    ld1 { v18.b }[12], [x6]
    b label_34
KAI_ASM_LABEL(label_28)  // Height 2: Multiply loop: Ragged operand read: partial_2_8
    tbz x3, #1, label_29
    ld1 { v19.h }[4], [x7], #0x2
    ld1 { v18.h }[4], [x6], #0x2
    tbz x3, #0, label_34
    ld1 { v19.b }[10], [x7]
    ld1 { v18.b }[10], [x6]
    b label_34
KAI_ASM_LABEL(label_29)  // Height 2: Multiply loop: Ragged operand read: partial_1_8
    tbz x3, #0, label_34
    ld1 { v19.b }[8], [x7]
    ld1 { v18.b }[8], [x6]
    b label_34
KAI_ASM_LABEL(label_30)  // Height 2: Multiply loop: Ragged operand read: partial_4_0
    tbz x3, #2, label_32
    ldr s19, [x7], #0x4
    ldr s18, [x6], #0x4
    tbz x3, #1, label_31
    ld1 { v19.h }[2], [x7], #0x2
    ld1 { v18.h }[2], [x6], #0x2
    tbz x3, #0, label_34
    ld1 { v19.b }[6], [x7]
    ld1 { v18.b }[6], [x6]
    b label_34
KAI_ASM_LABEL(label_31)  // Height 2: Multiply loop: Ragged operand read: partial_1_4
    tbz x3, #0, label_34
    ld1 { v19.b }[4], [x7]
    ld1 { v18.b }[4], [x6]
    b label_34
KAI_ASM_LABEL(label_32)  // Height 2: Multiply loop: Ragged operand read: partial_2_0
    tbz x3, #1, label_33
    ldr h19, [x7], #0x2
    ldr h18, [x6], #0x2
    tbz x3, #0, label_34
    ld1 { v19.b }[2], [x7]
    ld1 { v18.b }[2], [x6]
    b label_34
KAI_ASM_LABEL(label_33)  // Height 2: Multiply loop: Ragged operand read: partial_1_0
    ldr b19, [x7, #0]
    ldr b18, [x6, #0]
KAI_ASM_LABEL(label_34)  // Height 2: Multiply loop: Ragged operand read: Done
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
KAI_ASM_LABEL(label_35)  // Height 2: Multiply loop: No odd multiplies
    tbz x21, #3, label_36
    add x8, x8, #0x1
    cmp x8, x4
    bne label_20
KAI_ASM_LABEL(label_36)  // Height 2: no string loop
    sadalp v26.4s, v27.8h
    sadalp v24.4s, v25.8h
    addp v26.4s, v26.4s, v24.4s
    addp v26.4s, v26.4s, v26.4s
    mul v26.4s, v26.4s, v28.4s
    str d26, [x1], #0x8
    b label_74
KAI_ASM_LABEL(label_37)  // Height 3
    movi v27.8h, #0
    movi v26.4s, #0
    mov x20, #0
    ldr x4, [x0, #0x10]
    movi v25.8h, #0
    movi v24.4s, #0
    mov x8, #0
    movi v23.8h, #0
    movi v22.4s, #0
    tbz x21, #3, label_39
KAI_ASM_LABEL(label_38)  // Height 3: String loop
    ldr x3, [x24, x8, LSL #0x3]
    ldr x2, [x0, #0x18]
    add x3, x3, x25, LSL #3
    ldr x7, [x3, #0]
    ldr x6, [x3, #0x8]
    ldr x5, [x3, #0x10]
    ldr x3, [x2, x8, LSL #0x3]
    cbnz x8, label_40
    add x7, x7, x23
    add x6, x6, x23
    add x5, x5, x23
    b label_40
KAI_ASM_LABEL(label_39)  // Height 3: setup direct input
    mov x7, x24
    mov x3, x4
    add x6, x7, x23
    add x5, x6, x23
KAI_ASM_LABEL(label_40)  // Height 3: input setup done
    cmp x3, #0x10
    blt label_44
    cmp x3, #0x20
    blt label_43
KAI_ASM_LABEL(label_41)  // Height 3: Multiply loop: Main loop head
    ldr q19, [x7, #0]
    ldr q18, [x6, #0]
    cmp x20, #0x7e
    add x7, x7, #0x10
    ldr q17, [x5, #0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    blt label_42
    sadalp v26.4s, v27.8h
    movi v27.8h, #0
    mov x20, #0
    sadalp v24.4s, v25.8h
    movi v25.8h, #0
    sadalp v22.4s, v23.8h
    movi v23.8h, #0
KAI_ASM_LABEL(label_42)  // Height 3: Multiply loop: unique 3: no collapse
    sub x3, x3, #0x10
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    add x20, x20, #0x1
    cmp x3, #0x20
    sadalp v23.8h, v17.16b
    bge label_41
KAI_ASM_LABEL(label_43)  // Height 3: Multiply loop: Single iteration only
    ldr q18, [x7, #0]
    ldr q17, [x6, #0]
    sub x3, x3, #0x10
    add x7, x7, #0x10
    ldr q16, [x5, #0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    sadalp v27.8h, v18.16b
    sadalp v25.8h, v17.16b
    sadalp v23.8h, v16.16b
KAI_ASM_LABEL(label_44)  // Height 3: Multiply loop: Main loop skip
    cbz x3, label_53
    tbz x3, #3, label_48
    ldr d19, [x7], #0x8
    ldr d18, [x6], #0x8
    ldr d17, [x5], #0x8
    tbz x3, #2, label_46
    ld1 { v19.s }[2], [x7], #0x4
    ld1 { v18.s }[2], [x6], #0x4
    ld1 { v17.s }[2], [x5], #0x4
    tbz x3, #1, label_45
    ld1 { v19.h }[6], [x7], #0x2
    ld1 { v18.h }[6], [x6], #0x2
    ld1 { v17.h }[6], [x5], #0x2
    tbz x3, #0, label_52
    ld1 { v19.b }[14], [x7]
    ld1 { v18.b }[14], [x6]
    ld1 { v17.b }[14], [x5]
    b label_52
KAI_ASM_LABEL(label_45)  // Height 3: Multiply loop: Ragged operand read: partial_1_12
    tbz x3, #0, label_52
    ld1 { v19.b }[12], [x7]
    ld1 { v18.b }[12], [x6]
    ld1 { v17.b }[12], [x5]
    b label_52
KAI_ASM_LABEL(label_46)  // Height 3: Multiply loop: Ragged operand read: partial_2_8
    tbz x3, #1, label_47
    ld1 { v19.h }[4], [x7], #0x2
    ld1 { v18.h }[4], [x6], #0x2
    ld1 { v17.h }[4], [x5], #0x2
    tbz x3, #0, label_52
    ld1 { v19.b }[10], [x7]
    ld1 { v18.b }[10], [x6]
    ld1 { v17.b }[10], [x5]
    b label_52
KAI_ASM_LABEL(label_47)  // Height 3: Multiply loop: Ragged operand read: partial_1_8
    tbz x3, #0, label_52
    ld1 { v19.b }[8], [x7]
    ld1 { v18.b }[8], [x6]
    ld1 { v17.b }[8], [x5]
    b label_52
KAI_ASM_LABEL(label_48)  // Height 3: Multiply loop: Ragged operand read: partial_4_0
    tbz x3, #2, label_50
    ldr s19, [x7], #0x4
    ldr s18, [x6], #0x4
    ldr s17, [x5], #0x4
    tbz x3, #1, label_49
    ld1 { v19.h }[2], [x7], #0x2
    ld1 { v18.h }[2], [x6], #0x2
    ld1 { v17.h }[2], [x5], #0x2
    tbz x3, #0, label_52
    ld1 { v19.b }[6], [x7]
    ld1 { v18.b }[6], [x6]
    ld1 { v17.b }[6], [x5]
    b label_52
KAI_ASM_LABEL(label_49)  // Height 3: Multiply loop: Ragged operand read: partial_1_4
    tbz x3, #0, label_52
    ld1 { v19.b }[4], [x7]
    ld1 { v18.b }[4], [x6]
    ld1 { v17.b }[4], [x5]
    b label_52
KAI_ASM_LABEL(label_50)  // Height 3: Multiply loop: Ragged operand read: partial_2_0
    tbz x3, #1, label_51
    ldr h19, [x7], #0x2
    ldr h18, [x6], #0x2
    ldr h17, [x5], #0x2
    tbz x3, #0, label_52
    ld1 { v19.b }[2], [x7]
    ld1 { v18.b }[2], [x6]
    ld1 { v17.b }[2], [x5]
    b label_52
KAI_ASM_LABEL(label_51)  // Height 3: Multiply loop: Ragged operand read: partial_1_0
    ldr b19, [x7, #0]
    ldr b18, [x6, #0]
    ldr b17, [x5, #0]
KAI_ASM_LABEL(label_52)  // Height 3: Multiply loop: Ragged operand read: Done
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    sadalp v23.8h, v17.16b
KAI_ASM_LABEL(label_53)  // Height 3: Multiply loop: No odd multiplies
    tbz x21, #3, label_54
    add x8, x8, #0x1
    cmp x8, x4
    bne label_38
KAI_ASM_LABEL(label_54)  // Height 3: no string loop
    sadalp v26.4s, v27.8h
    sadalp v24.4s, v25.8h
    sadalp v22.4s, v23.8h
    addp v26.4s, v26.4s, v24.4s
    addp v22.4s, v22.4s, v22.4s
    addp v26.4s, v26.4s, v26.4s
    addp v22.4s, v22.4s, v22.4s
    mul v26.4s, v26.4s, v28.4s
    str d26, [x1], #0x8
    mul v22.4s, v22.4s, v28.4s
    str s22, [x1], #0x4
    b label_74
KAI_ASM_LABEL(label_55)  // Height 4
    movi v27.8h, #0
    movi v26.4s, #0
    mov x20, #0
    ldr x4, [x0, #0x10]
    movi v25.8h, #0
    movi v24.4s, #0
    mov x8, #0
    movi v23.8h, #0
    movi v22.4s, #0
    movi v21.8h, #0
    movi v20.4s, #0
    tbz x21, #3, label_57
KAI_ASM_LABEL(label_56)  // Height 4: String loop
    ldr x2, [x24, x8, LSL #0x3]
    ldr x3, [x0, #0x18]
    add x2, x2, x25, LSL #3
    ldr x7, [x2, #0]
    ldr x6, [x2, #0x8]
    ldr x5, [x2, #0x10]
    ldr x2, [x2, #0x18]
    ldr x3, [x3, x8, LSL #0x3]
    cbnz x8, label_58
    add x7, x7, x23
    add x6, x6, x23
    add x5, x5, x23
    add x2, x2, x23
    b label_58
KAI_ASM_LABEL(label_57)  // Height 4: setup direct input
    mov x7, x24
    mov x3, x4
    add x6, x7, x23
    add x5, x6, x23
    add x2, x5, x23
KAI_ASM_LABEL(label_58)  // Height 4: input setup done
    cmp x3, #0x10
    blt label_62
    cmp x3, #0x20
    blt label_61
KAI_ASM_LABEL(label_59)  // Height 4: Multiply loop: Main loop head
    ldr q19, [x7, #0]
    ldr q18, [x6, #0]
    cmp x20, #0x7e
    add x7, x7, #0x10
    ldr q17, [x5, #0]
    ldr q16, [x2, #0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x2, x2, #0x10
    blt label_60
    sadalp v26.4s, v27.8h
    movi v27.8h, #0
    mov x20, #0
    sadalp v24.4s, v25.8h
    movi v25.8h, #0
    sadalp v22.4s, v23.8h
    movi v23.8h, #0
    sadalp v20.4s, v21.8h
    movi v21.8h, #0
KAI_ASM_LABEL(label_60)  // Height 4: Multiply loop: unique 4: no collapse
    sub x3, x3, #0x10
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    add x20, x20, #0x1
    cmp x3, #0x20
    sadalp v23.8h, v17.16b
    sadalp v21.8h, v16.16b
    bge label_59
KAI_ASM_LABEL(label_61)  // Height 4: Multiply loop: Single iteration only
    ldr q19, [x7, #0]
    ldr q18, [x6, #0]
    sub x3, x3, #0x10
    add x7, x7, #0x10
    ldr q17, [x5, #0]
    ldr q16, [x2, #0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x2, x2, #0x10
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    sadalp v23.8h, v17.16b
    sadalp v21.8h, v16.16b
KAI_ASM_LABEL(label_62)  // Height 4: Multiply loop: Main loop skip
    cbz x3, label_71
    tbz x3, #3, label_66
    ldr d19, [x7], #0x8
    ldr d18, [x6], #0x8
    ldr d17, [x5], #0x8
    ldr d16, [x2], #0x8
    tbz x3, #2, label_64
    ld1 { v19.s }[2], [x7], #0x4
    ld1 { v18.s }[2], [x6], #0x4
    ld1 { v17.s }[2], [x5], #0x4
    ld1 { v16.s }[2], [x2], #0x4
    tbz x3, #1, label_63
    ld1 { v19.h }[6], [x7], #0x2
    ld1 { v18.h }[6], [x6], #0x2
    ld1 { v17.h }[6], [x5], #0x2
    ld1 { v16.h }[6], [x2], #0x2
    tbz x3, #0, label_70
    ld1 { v19.b }[14], [x7]
    ld1 { v18.b }[14], [x6]
    ld1 { v17.b }[14], [x5]
    ld1 { v16.b }[14], [x2]
    b label_70
KAI_ASM_LABEL(label_63)  // Height 4: Multiply loop: Ragged operand read: partial_1_12
    tbz x3, #0, label_70
    ld1 { v19.b }[12], [x7]
    ld1 { v18.b }[12], [x6]
    ld1 { v17.b }[12], [x5]
    ld1 { v16.b }[12], [x2]
    b label_70
KAI_ASM_LABEL(label_64)  // Height 4: Multiply loop: Ragged operand read: partial_2_8
    tbz x3, #1, label_65
    ld1 { v19.h }[4], [x7], #0x2
    ld1 { v18.h }[4], [x6], #0x2
    ld1 { v17.h }[4], [x5], #0x2
    ld1 { v16.h }[4], [x2], #0x2
    tbz x3, #0, label_70
    ld1 { v19.b }[10], [x7]
    ld1 { v18.b }[10], [x6]
    ld1 { v17.b }[10], [x5]
    ld1 { v16.b }[10], [x2]
    b label_70
KAI_ASM_LABEL(label_65)  // Height 4: Multiply loop: Ragged operand read: partial_1_8
    tbz x3, #0, label_70
    ld1 { v19.b }[8], [x7]
    ld1 { v18.b }[8], [x6]
    ld1 { v17.b }[8], [x5]
    ld1 { v16.b }[8], [x2]
    b label_70
KAI_ASM_LABEL(label_66)  // Height 4: Multiply loop: Ragged operand read: partial_4_0
    tbz x3, #2, label_68
    ldr s19, [x7], #0x4
    ldr s18, [x6], #0x4
    ldr s17, [x5], #0x4
    ldr s16, [x2], #0x4
    tbz x3, #1, label_67
    ld1 { v19.h }[2], [x7], #0x2
    ld1 { v18.h }[2], [x6], #0x2
    ld1 { v17.h }[2], [x5], #0x2
    ld1 { v16.h }[2], [x2], #0x2
    tbz x3, #0, label_70
    ld1 { v19.b }[6], [x7]
    ld1 { v18.b }[6], [x6]
    ld1 { v17.b }[6], [x5]
    ld1 { v16.b }[6], [x2]
    b label_70
KAI_ASM_LABEL(label_67)  // Height 4: Multiply loop: Ragged operand read: partial_1_4
    tbz x3, #0, label_70
    ld1 { v19.b }[4], [x7]
    ld1 { v18.b }[4], [x6]
    ld1 { v17.b }[4], [x5]
    ld1 { v16.b }[4], [x2]
    b label_70
KAI_ASM_LABEL(label_68)  // Height 4: Multiply loop: Ragged operand read: partial_2_0
    tbz x3, #1, label_69
    ldr h19, [x7], #0x2
    ldr h18, [x6], #0x2
    ldr h17, [x5], #0x2
    ldr h16, [x2], #0x2
    tbz x3, #0, label_70
    ld1 { v19.b }[2], [x7]
    ld1 { v18.b }[2], [x6]
    ld1 { v17.b }[2], [x5]
    ld1 { v16.b }[2], [x2]
    b label_70
KAI_ASM_LABEL(label_69)  // Height 4: Multiply loop: Ragged operand read: partial_1_0
    ldr b19, [x7, #0]
    ldr b18, [x6, #0]
    ldr b17, [x5, #0]
    ldr b16, [x2, #0]
KAI_ASM_LABEL(label_70)  // Height 4: Multiply loop: Ragged operand read: Done
    sadalp v27.8h, v19.16b
    sadalp v25.8h, v18.16b
    sadalp v23.8h, v17.16b
    sadalp v21.8h, v16.16b
KAI_ASM_LABEL(label_71)  // Height 4: Multiply loop: No odd multiplies
    tbz x21, #3, label_72
    add x8, x8, #0x1
    cmp x8, x4
    bne label_56
KAI_ASM_LABEL(label_72)  // Height 4: no string loop
    sadalp v26.4s, v27.8h
    sadalp v24.4s, v25.8h
    subs x22, x22, #0x4
    sadalp v22.4s, v23.8h
    sadalp v20.4s, v21.8h
    addp v26.4s, v26.4s, v24.4s
    addp v16.4s, v22.4s, v20.4s
    addp v26.4s, v26.4s, v16.4s
    mul v26.4s, v26.4s, v28.4s
    st1 { v26.4s }, [x1], #0x10
    beq label_74
    tbz x21, #3, label_73
    add x25, x25, #0x4
    b label_1
KAI_ASM_LABEL(label_73)  // Update direct input
    mov x2, #0x4
    madd x24, x2, x23, x24
    b label_1
KAI_ASM_LABEL(label_74)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_sumrows_s8)

    KAI_ASM_END
