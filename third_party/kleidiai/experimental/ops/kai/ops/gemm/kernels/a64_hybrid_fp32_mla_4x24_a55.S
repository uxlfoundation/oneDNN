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

    KAI_ASM_CODE(a64_hybrid_fp32_mla_4x24_a55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_fp32_mla_4x24_a55)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_fp32_mla_4x24_a55)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_fp32_mla_4x24_a55)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x5, #0xff800000
    mov x4, #0x7f800000
    sub SP, SP, #0x60
    mov x13, #0
    ldr x1, [x0, #0x60]
    ldr w3, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    cmp x1, #0
    ldr x1, [x0, #0x18]
    cset x12, NE
    cmp x3, x5
    cset x3, NE
    cmp x2, x4
    cset x2, NE
    cmp x1, #0
    cset x1, NE
    ldr x11, [x0, #0]
    orr x12, x12, x3, LSL #1
    orr x12, x12, x2, LSL #1
    orr x12, x12, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x11, x13
    cmp x1, #0x4
    bge label_148
    cmp x1, #0x2
    bgt label_99
    beq label_50
    ldr x1, [x0, #0x58]
    ldr x9, [x0, #0x50]
    ldr x28, [x0, #0x8]
    ldr x27, [x0, #0x30]
    ldr x26, [x0, #0x40]
    madd x9, x13, x1, x9
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x26, label_10
    cmp x28, #0x18
    bge label_8
    cbnz x13, label_7
    mov x2, x28
    mov x1, SP
    cmp x2, #0x4
    blt label_4
KAI_ASM_LABEL(label_3)  // Height 1: Partial bias load: vector loop
    ld1 { v16.4s }, [x26], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_3
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_6
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: element loop
    ldr s16, [x26], #0x4
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
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_9
KAI_ASM_LABEL(label_8)  // Height 1: full width bias
    ldr q8, [x26, #0]
    ldr q9, [x26, #0x10]
    ldr q10, [x26, #0x20]
    ldr q11, [x26, #0x30]
    ldr q12, [x26, #0x40]
    ldr q13, [x26, #0x50]
    add x26, x26, #0x60
KAI_ASM_LABEL(label_9)  // Height 1: bias load done
    b label_25
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x12, #0, label_24
    cmp x28, #0x18
    bge label_23
    tbz x28, #4, label_14
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v11.4s }, [x9], #0x10
    tbz x28, #2, label_12
    ld1 { v12.4s }, [x9], #0x10
    tbz x28, #1, label_11
    ldr d13, [x9], #0x8
    mov x1, #0x58
    tbz x28, #0, label_22
    ld1 { v13.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x28, #0, label_22
    ldr s13, [x9, #0]
    b label_22
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_16
    tbz x28, #1, label_13
    ldr d12, [x9], #0x8
    mov x1, #0x48
    tbz x28, #0, label_22
    ld1 { v12.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x28, #0, label_22
    ldr s12, [x9, #0]
    b label_22
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_3_0
    tbz x28, #3, label_18
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v9.4s }, [x9], #0x10
    tbz x28, #2, label_16
    ld1 { v10.4s }, [x9], #0x10
    tbz x28, #1, label_15
    ldr d11, [x9], #0x8
    mov x1, #0x38
    tbz x28, #0, label_22
    ld1 { v11.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x28, #0, label_22
    ldr s11, [x9, #0]
    b label_22
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_8
    tbz x28, #1, label_17
    ldr d10, [x9], #0x8
    mov x1, #0x28
    tbz x28, #0, label_22
    ld1 { v10.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x28, #0, label_22
    ldr s10, [x9, #0]
    b label_22
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_2_0
    tbz x28, #2, label_20
    ld1 { v8.4s }, [x9], #0x10
    tbz x28, #1, label_19
    ldr d9, [x9], #0x8
    mov x1, #0x18
    tbz x28, #0, label_22
    ld1 { v9.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x28, #0, label_22
    ldr s9, [x9, #0]
    b label_22
KAI_ASM_LABEL(label_20)  // Height 1: Partial accumulate: partial_1_0
    tbz x28, #1, label_21
    ldr d8, [x9], #0x8
    mov x1, #0x8
    tbz x28, #0, label_22
    ld1 { v8.s }[2], [x9]
    b label_22
KAI_ASM_LABEL(label_21)  // Height 1: Partial accumulate: partial_0_0
    ldr s8, [x9, #0]
    mov x1, #0
KAI_ASM_LABEL(label_22)  // Height 1: Partial accumulate: Done
    sub x9, x9, x1
    b label_25
KAI_ASM_LABEL(label_23)  // Height 1: full accumulate
    ldr q8, [x9, #0]
    ldr q9, [x9, #0x10]
    ldr q10, [x9, #0x20]
    ldr q11, [x9, #0x30]
    ldr q12, [x9, #0x40]
    ldr q13, [x9, #0x50]
    b label_25
KAI_ASM_LABEL(label_24)  // Height 1: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
KAI_ASM_LABEL(label_25)  // Height 1: setup done
    ldr x4, [x0, #0x28]
    tbnz x12, #3, label_26
    ldr x25, [x0, #0x20]
    ldr x24, [x0, #0x10]
    madd x25, x13, x4, x25
    b label_28
KAI_ASM_LABEL(label_26)  // Height 1: setup indirect input
    mov x10, #0
KAI_ASM_LABEL(label_27)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x10, LSL #0x3]
    ldr x24, [x2, x10, LSL #0x3]
    add x1, x1, x13, LSL #3
    ldr x25, [x1, #0]
    cbnz x10, label_28
    add x25, x25, x4, LSL #2
KAI_ASM_LABEL(label_28)  // Height 1: input setup done
    cmp x24, #0x4
    blt label_31
    ldr q0, [x25, #0]
    cmp x24, #0x8
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    blt label_30
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: Main loop head
    fmla v8.4s, v4.4s, v0.s[0]
    ldr d4, [x27, #0x40]
    ldr x1, [x27, #0x48]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr d5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr d6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr d7, [x27, #0x70]
    mov v4.d[1], x1
    ldr x3, [x27, #0x58]
    add x25, x25, #0x10
    ldr x1, [x27, #0x68]
    fmla v12.4s, v4.4s, v0.s[0]
    ldr d4, [x27, #0x80]
    sub x24, x24, #0x4
    ldr x2, [x27, #0x78]
    mov v5.d[1], x3
    mov v6.d[1], x1
    ldr x1, [x27, #0x88]
    fmla v13.4s, v5.4s, v0.s[0]
    ldr d5, [x27, #0x90]
    mov v7.d[1], x2
    ldr x3, [x27, #0x98]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr d6, [x27, #0xa0]
    ldr x2, [x27, #0xa8]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr d7, [x27, #0xb0]
    mov v4.d[1], x1
    ldr x1, [x27, #0xb8]
    mov v5.d[1], x3
    mov v6.d[1], x2
    fmla v10.4s, v4.4s, v0.s[1]
    ldr d4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    ldr d5, [x27, #0xd0]
    mov v7.d[1], x1
    ldr x1, [x27, #0xc8]
    fmla v12.4s, v6.4s, v0.s[1]
    ldr d6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    ldr d7, [x27, #0xf0]
    cmp x24, #0x8
    ldr x3, [x27, #0xd8]
    mov v4.d[1], x1
    ldr x1, [x27, #0xe8]
    ldr x2, [x27, #0xf8]
    fmla v8.4s, v4.4s, v0.s[2]
    ldr d4, [x27, #0x100]
    mov v5.d[1], x3
    mov v6.d[1], x1
    ldr x1, [x27, #0x108]
    mov v7.d[1], x2
    fmla v9.4s, v5.4s, v0.s[2]
    ldr d5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr d6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr d7, [x27, #0x130]
    mov v4.d[1], x1
    ldr x3, [x27, #0x118]
    ldr x1, [x27, #0x128]
    fmla v12.4s, v4.4s, v0.s[2]
    ldr d4, [x27, #0x140]
    ldr x2, [x27, #0x138]
    mov v5.d[1], x3
    mov v6.d[1], x1
    ldr x1, [x27, #0x148]
    fmla v13.4s, v5.4s, v0.s[2]
    ldr d5, [x27, #0x150]
    mov v7.d[1], x2
    ldr x3, [x27, #0x158]
    fmla v8.4s, v6.4s, v0.s[3]
    ldr d6, [x27, #0x160]
    ldr x2, [x27, #0x168]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr d7, [x27, #0x170]
    mov v4.d[1], x1
    ldr x1, [x27, #0x178]
    mov v5.d[1], x3
    mov v6.d[1], x2
    add x27, x27, #0x180
    fmla v10.4s, v4.4s, v0.s[3]
    ldr d4, [x27, #0]
    mov v7.d[1], x1
    ldr x2, [x27, #0x8]
    fmla v11.4s, v5.4s, v0.s[3]
    ldr d5, [x27, #0x10]
    ldr x1, [x27, #0x18]
    fmla v12.4s, v6.4s, v0.s[3]
    ldr d6, [x27, #0x20]
    fmla v13.4s, v7.4s, v0.s[3]
    ldr d0, [x25, #0]
    mov v4.d[1], x2
    ldr d7, [x27, #0x30]
    mov v5.d[1], x1
    ldr x3, [x27, #0x28]
    ldr x2, [x25, #0x8]
    ldr x1, [x27, #0x38]
    mov v6.d[1], x3
    prfm pldl1keep, [x25, #0x80]
    mov v0.d[1], x2
    mov v7.d[1], x1
    bge label_29
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: Single iteration only
    fmla v8.4s, v4.4s, v0.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr q6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr q7, [x27, #0x70]
    fmla v12.4s, v4.4s, v0.s[0]
    ldr q4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    ldr q5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr q6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr q7, [x27, #0xb0]
    fmla v10.4s, v4.4s, v0.s[1]
    ldr q4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    ldr q5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    ldr q6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    ldr q7, [x27, #0xf0]
    fmla v8.4s, v4.4s, v0.s[2]
    ldr q4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    ldr q5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr q6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr q7, [x27, #0x130]
    fmla v12.4s, v4.4s, v0.s[2]
    ldr q4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    ldr q5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    ldr q6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr q7, [x27, #0x170]
    add x25, x25, #0x10
    sub x24, x24, #0x4
    fmla v10.4s, v4.4s, v0.s[3]
    prfm pldl1keep, [x25, #0x80]
    fmla v11.4s, v5.4s, v0.s[3]
    add x27, x27, #0x180
    fmla v12.4s, v6.4s, v0.s[3]
    fmla v13.4s, v7.4s, v0.s[3]
KAI_ASM_LABEL(label_31)  // Height 1: Multiply loop: Main loop skip
    cbz x24, label_33
KAI_ASM_LABEL(label_32)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x25], #0x4
    sub x24, x24, #0x1
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    fmla v8.4s, v4.4s, v0.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    add x27, x27, #0x60
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v13.4s, v5.4s, v0.s[0]
    cbnz x24, label_32
KAI_ASM_LABEL(label_33)  // Height 1: Multiply loop: No odd multiplies
    tbz x12, #3, label_34
    ldr x1, [x0, #0x10]
    add x10, x10, #0x1
    cmp x10, x1
    bne label_27
KAI_ASM_LABEL(label_34)  // Height 1: no string loop
    prfm pstl1keep, [x9, #0]
    tbz x12, #1, label_35
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmin v12.4s, v12.4s, v17.4s
    fmin v13.4s, v13.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
    fmax v12.4s, v12.4s, v16.4s
    fmax v13.4s, v13.4s, v16.4s
KAI_ASM_LABEL(label_35)  // Height 1: No activation
    cmp x28, #0x18
    bge label_48
    tbz x28, #4, label_39
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v10.4s }, [x9], #0x10
    st1 { v11.4s }, [x9], #0x10
    tbz x28, #2, label_37
    st1 { v12.4s }, [x9], #0x10
    tbz x28, #1, label_36
    str d13, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v13.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_36)  // Height 1: Partial direct writeback: partial_0_20
    tbz x28, #0, label_47
    str s13, [x9, #0]
    b label_47
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_1_16
    tbz x28, #1, label_38
    str d12, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v12.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_0_16
    tbz x28, #0, label_47
    str s12, [x9, #0]
    b label_47
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_3_0
    tbz x28, #3, label_43
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    tbz x28, #2, label_41
    st1 { v10.4s }, [x9], #0x10
    tbz x28, #1, label_40
    str d11, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v11.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_0_12
    tbz x28, #0, label_47
    str s11, [x9, #0]
    b label_47
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_1_8
    tbz x28, #1, label_42
    str d10, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v10.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_0_8
    tbz x28, #0, label_47
    str s10, [x9, #0]
    b label_47
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_2_0
    tbz x28, #2, label_45
    st1 { v8.4s }, [x9], #0x10
    tbz x28, #1, label_44
    str d9, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v9.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: partial_0_4
    tbz x28, #0, label_47
    str s9, [x9, #0]
    b label_47
KAI_ASM_LABEL(label_45)  // Height 1: Partial direct writeback: partial_1_0
    tbz x28, #1, label_46
    str d8, [x9], #0x8
    tbz x28, #0, label_47
    st1 { v8.s }[2], [x9]
    b label_47
KAI_ASM_LABEL(label_46)  // Height 1: Partial direct writeback: partial_0_0
    str s8, [x9, #0]
KAI_ASM_LABEL(label_47)  // Height 1: Partial direct writeback: Done
    b label_49
KAI_ASM_LABEL(label_48)  // Height 1: Full writeback
    str q8, [x9, #0]
    str q9, [x9, #0x10]
    str q10, [x9, #0x20]
    str q11, [x9, #0x30]
    str q12, [x9, #0x40]
    str q13, [x9, #0x50]
    add x9, x9, #0x60
KAI_ASM_LABEL(label_49)  // Height 1: Writeback done
    subs x28, x28, #0x18
    bgt label_2
    b label_197
KAI_ASM_LABEL(label_50)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x9, [x0, #0x50]
    ldr x28, [x0, #0x8]
    ldr x27, [x0, #0x30]
    ldr x26, [x0, #0x40]
    madd x9, x13, x1, x9
KAI_ASM_LABEL(label_51)  // Height 2: Column loop
    cbz x26, label_59
    cmp x28, #0x18
    bge label_57
    cbnz x13, label_56
    mov x2, x28
    mov x1, SP
    cmp x2, #0x4
    blt label_53
KAI_ASM_LABEL(label_52)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x26], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_52
KAI_ASM_LABEL(label_53)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_55
KAI_ASM_LABEL(label_54)  // Height 2: Partial bias load: element loop
    ldr s16, [x26], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_54
KAI_ASM_LABEL(label_55)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_56)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_58
KAI_ASM_LABEL(label_57)  // Height 2: full width bias
    ldr q8, [x26, #0]
    ldr q9, [x26, #0x10]
    ldr q10, [x26, #0x20]
    ldr q11, [x26, #0x30]
    ldr q12, [x26, #0x40]
    ldr q13, [x26, #0x50]
    add x26, x26, #0x60
KAI_ASM_LABEL(label_58)  // Height 2: bias load done
    mov v14.16b, v8.16b
    mov v15.16b, v9.16b
    mov v16.16b, v10.16b
    mov v17.16b, v11.16b
    mov v18.16b, v12.16b
    mov v19.16b, v13.16b
    b label_74
KAI_ASM_LABEL(label_59)  // Height 2: no bias
    tbz x12, #0, label_73
    ldr x1, [x0, #0x58]
    cmp x28, #0x18
    add x5, x9, x1
    bge label_72
    tbz x28, #4, label_63
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v11.4s }, [x9], #0x10
    ld1 { v17.4s }, [x5], #0x10
    tbz x28, #2, label_61
    ld1 { v12.4s }, [x9], #0x10
    ld1 { v18.4s }, [x5], #0x10
    tbz x28, #1, label_60
    ldr d13, [x9], #0x8
    mov x1, #0x58
    ldr d19, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v13.s }[2], [x9]
    ld1 { v19.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_60)  // Height 2: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x28, #0, label_71
    ldr s13, [x9, #0]
    ldr s19, [x5, #0]
    b label_71
KAI_ASM_LABEL(label_61)  // Height 2: Partial accumulate: partial_1_16
    tbz x28, #1, label_62
    ldr d12, [x9], #0x8
    mov x1, #0x48
    ldr d18, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v12.s }[2], [x9]
    ld1 { v18.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_62)  // Height 2: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x28, #0, label_71
    ldr s12, [x9, #0]
    ldr s18, [x5, #0]
    b label_71
KAI_ASM_LABEL(label_63)  // Height 2: Partial accumulate: partial_3_0
    tbz x28, #3, label_67
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    tbz x28, #2, label_65
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    tbz x28, #1, label_64
    ldr d11, [x9], #0x8
    mov x1, #0x38
    ldr d17, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v11.s }[2], [x9]
    ld1 { v17.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_64)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x28, #0, label_71
    ldr s11, [x9, #0]
    ldr s17, [x5, #0]
    b label_71
KAI_ASM_LABEL(label_65)  // Height 2: Partial accumulate: partial_1_8
    tbz x28, #1, label_66
    ldr d10, [x9], #0x8
    mov x1, #0x28
    ldr d16, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v10.s }[2], [x9]
    ld1 { v16.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_66)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x28, #0, label_71
    ldr s10, [x9, #0]
    ldr s16, [x5, #0]
    b label_71
KAI_ASM_LABEL(label_67)  // Height 2: Partial accumulate: partial_2_0
    tbz x28, #2, label_69
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    tbz x28, #1, label_68
    ldr d9, [x9], #0x8
    mov x1, #0x18
    ldr d15, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v9.s }[2], [x9]
    ld1 { v15.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_68)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x28, #0, label_71
    ldr s9, [x9, #0]
    ldr s15, [x5, #0]
    b label_71
KAI_ASM_LABEL(label_69)  // Height 2: Partial accumulate: partial_1_0
    tbz x28, #1, label_70
    ldr d8, [x9], #0x8
    mov x1, #0x8
    ldr d14, [x5], #0x8
    tbz x28, #0, label_71
    ld1 { v8.s }[2], [x9]
    ld1 { v14.s }[2], [x5]
    b label_71
KAI_ASM_LABEL(label_70)  // Height 2: Partial accumulate: partial_0_0
    ldr s8, [x9, #0]
    mov x1, #0
    ldr s14, [x5, #0]
KAI_ASM_LABEL(label_71)  // Height 2: Partial accumulate: Done
    sub x9, x9, x1
    b label_74
KAI_ASM_LABEL(label_72)  // Height 2: full accumulate
    ldr q8, [x9, #0]
    ldr q9, [x9, #0x10]
    ldr q10, [x9, #0x20]
    ldr q11, [x9, #0x30]
    ldr q12, [x9, #0x40]
    ldr q13, [x9, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
    b label_74
KAI_ASM_LABEL(label_73)  // Height 2: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
KAI_ASM_LABEL(label_74)  // Height 2: setup done
    ldr x5, [x0, #0x28]
    tbnz x12, #3, label_75
    ldr x25, [x0, #0x20]
    ldr x24, [x0, #0x10]
    madd x25, x13, x5, x25
    add x23, x25, x5
    b label_77
KAI_ASM_LABEL(label_75)  // Height 2: setup indirect input
    mov x10, #0
KAI_ASM_LABEL(label_76)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x10, LSL #0x3]
    ldr x24, [x2, x10, LSL #0x3]
    add x1, x1, x13, LSL #3
    ldr x25, [x1, #0]
    ldr x23, [x1, #0x8]
    cbnz x10, label_77
    add x25, x25, x5, LSL #2
    add x23, x23, x5, LSL #2
KAI_ASM_LABEL(label_77)  // Height 2: input setup done
    cmp x24, #0x4
    blt label_80
    ldr q0, [x25, #0]
    cmp x24, #0x8
    ldr q1, [x23, #0]
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    blt label_79
KAI_ASM_LABEL(label_78)  // Height 2: Multiply loop: Main loop head
    fmla v8.4s, v4.4s, v0.s[0]
    ldr x4, [x27, #0x48]
    fmla v14.4s, v4.4s, v1.s[0]
    ldr d4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr x3, [x27, #0x58]
    fmla v15.4s, v5.4s, v1.s[0]
    ldr d5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr x2, [x27, #0x68]
    fmla v16.4s, v6.4s, v1.s[0]
    ldr d6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x1, [x27, #0x78]
    fmla v17.4s, v7.4s, v1.s[0]
    ldr d7, [x27, #0x70]
    mov v4.d[1], x4
    ldr x4, [x27, #0x88]
    mov v5.d[1], x3
    ldr x3, [x27, #0x98]
    mov v6.d[1], x2
    fmla v12.4s, v4.4s, v0.s[0]
    mov v7.d[1], x1
    fmla v18.4s, v4.4s, v1.s[0]
    ldr d4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    ldr d5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr x2, [x27, #0xa8]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr d6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr x1, [x27, #0xb8]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr d7, [x27, #0xb0]
    mov v4.d[1], x4
    ldr x4, [x27, #0xc8]
    mov v5.d[1], x3
    ldr x3, [x27, #0xd8]
    mov v6.d[1], x2
    fmla v10.4s, v4.4s, v0.s[1]
    mov v7.d[1], x1
    fmla v16.4s, v4.4s, v1.s[1]
    ldr d4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    fmla v17.4s, v5.4s, v1.s[1]
    ldr d5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    ldr x2, [x27, #0xe8]
    fmla v18.4s, v6.4s, v1.s[1]
    ldr d6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    ldr x1, [x27, #0xf8]
    fmla v19.4s, v7.4s, v1.s[1]
    ldr d7, [x27, #0xf0]
    mov v4.d[1], x4
    ldr x4, [x27, #0x108]
    mov v5.d[1], x3
    ldr x3, [x27, #0x118]
    mov v6.d[1], x2
    fmla v8.4s, v4.4s, v0.s[2]
    mov v7.d[1], x1
    fmla v14.4s, v4.4s, v1.s[2]
    ldr d4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    fmla v15.4s, v5.4s, v1.s[2]
    ldr d5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr x2, [x27, #0x128]
    fmla v16.4s, v6.4s, v1.s[2]
    ldr d6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr x1, [x27, #0x138]
    fmla v17.4s, v7.4s, v1.s[2]
    ldr d7, [x27, #0x130]
    mov v4.d[1], x4
    ldr x4, [x27, #0x148]
    mov v5.d[1], x3
    ldr x3, [x27, #0x158]
    mov v6.d[1], x2
    fmla v12.4s, v4.4s, v0.s[2]
    mov v7.d[1], x1
    fmla v18.4s, v4.4s, v1.s[2]
    ldr d4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v1.s[2]
    ldr d5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    ldr x2, [x27, #0x168]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr d6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr x1, [x27, #0x178]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d7, [x27, #0x170]
    mov v4.d[1], x4
    add x25, x25, #0x10
    mov v5.d[1], x3
    add x23, x23, #0x10
    mov v6.d[1], x2
    add x27, x27, #0x180
    mov v7.d[1], x1
    fmla v10.4s, v4.4s, v0.s[3]
    fmla v16.4s, v4.4s, v1.s[3]
    ldr d4, [x27, #0]
    ldr x2, [x27, #0x8]
    fmla v11.4s, v5.4s, v0.s[3]
    fmla v17.4s, v5.4s, v1.s[3]
    ldr d5, [x27, #0x10]
    ldr x1, [x27, #0x18]
    fmla v12.4s, v6.4s, v0.s[3]
    fmla v18.4s, v6.4s, v1.s[3]
    ldr d6, [x27, #0x20]
    ldr x4, [x27, #0x28]
    fmla v13.4s, v7.4s, v0.s[3]
    ldr d0, [x25, #0]
    fmla v19.4s, v7.4s, v1.s[3]
    ldr d1, [x23, #0]
    sub x24, x24, #0x4
    ldr d7, [x27, #0x30]
    cmp x24, #0x8
    ldr x3, [x25, #0x8]
    mov v4.d[1], x2
    ldr x2, [x23, #0x8]
    mov v5.d[1], x1
    ldr x1, [x27, #0x38]
    mov v6.d[1], x4
    prfm pldl1keep, [x25, #0x80]
    mov v0.d[1], x3
    prfm pldl1keep, [x23, #0x80]
    mov v1.d[1], x2
    mov v7.d[1], x1
    bge label_78
KAI_ASM_LABEL(label_79)  // Height 2: Multiply loop: Single iteration only
    fmla v8.4s, v4.4s, v0.s[0]
    add x25, x25, #0x10
    fmla v14.4s, v4.4s, v1.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    add x23, x23, #0x10
    fmla v15.4s, v5.4s, v1.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    sub x24, x24, #0x4
    fmla v16.4s, v6.4s, v1.s[0]
    ldr q6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    prfm pldl1keep, [x25, #0x80]
    fmla v17.4s, v7.4s, v1.s[0]
    ldr q7, [x27, #0x70]
    fmla v12.4s, v4.4s, v0.s[0]
    prfm pldl1keep, [x23, #0x80]
    fmla v18.4s, v4.4s, v1.s[0]
    ldr q4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    ldr q5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr q6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr q7, [x27, #0xb0]
    fmla v10.4s, v4.4s, v0.s[1]
    fmla v16.4s, v4.4s, v1.s[1]
    ldr q4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    fmla v17.4s, v5.4s, v1.s[1]
    ldr q5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    fmla v18.4s, v6.4s, v1.s[1]
    ldr q6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    fmla v19.4s, v7.4s, v1.s[1]
    ldr q7, [x27, #0xf0]
    fmla v8.4s, v4.4s, v0.s[2]
    fmla v14.4s, v4.4s, v1.s[2]
    ldr q4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    fmla v15.4s, v5.4s, v1.s[2]
    ldr q5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v16.4s, v6.4s, v1.s[2]
    ldr q6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v17.4s, v7.4s, v1.s[2]
    ldr q7, [x27, #0x130]
    fmla v12.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v1.s[2]
    ldr q4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v1.s[2]
    ldr q5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr q6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q7, [x27, #0x170]
    fmla v10.4s, v4.4s, v0.s[3]
    add x27, x27, #0x180
    fmla v16.4s, v4.4s, v1.s[3]
    fmla v11.4s, v5.4s, v0.s[3]
    fmla v17.4s, v5.4s, v1.s[3]
    fmla v12.4s, v6.4s, v0.s[3]
    fmla v18.4s, v6.4s, v1.s[3]
    fmla v13.4s, v7.4s, v0.s[3]
    fmla v19.4s, v7.4s, v1.s[3]
KAI_ASM_LABEL(label_80)  // Height 2: Multiply loop: Main loop skip
    cbz x24, label_82
KAI_ASM_LABEL(label_81)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x25], #0x4
    sub x24, x24, #0x1
    ldr s1, [x23], #0x4
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v14.4s, v4.4s, v1.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v15.4s, v5.4s, v1.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    add x27, x27, #0x60
    fmla v16.4s, v6.4s, v1.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v17.4s, v7.4s, v1.s[0]
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    cbnz x24, label_81
KAI_ASM_LABEL(label_82)  // Height 2: Multiply loop: No odd multiplies
    tbz x12, #3, label_83
    ldr x1, [x0, #0x10]
    add x10, x10, #0x1
    cmp x10, x1
    bne label_76
KAI_ASM_LABEL(label_83)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x9, #0]
    add x5, x9, x1
    prfm pstl1keep, [x5, #0]
    tbz x12, #1, label_84
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v21.4s }, [x2]
    ld1r { v20.4s }, [x1]
    fmin v8.4s, v8.4s, v21.4s
    fmin v9.4s, v9.4s, v21.4s
    fmin v10.4s, v10.4s, v21.4s
    fmin v11.4s, v11.4s, v21.4s
    fmin v12.4s, v12.4s, v21.4s
    fmin v13.4s, v13.4s, v21.4s
    fmin v14.4s, v14.4s, v21.4s
    fmin v15.4s, v15.4s, v21.4s
    fmin v16.4s, v16.4s, v21.4s
    fmin v17.4s, v17.4s, v21.4s
    fmin v18.4s, v18.4s, v21.4s
    fmin v19.4s, v19.4s, v21.4s
    fmax v8.4s, v8.4s, v20.4s
    fmax v9.4s, v9.4s, v20.4s
    fmax v10.4s, v10.4s, v20.4s
    fmax v11.4s, v11.4s, v20.4s
    fmax v12.4s, v12.4s, v20.4s
    fmax v13.4s, v13.4s, v20.4s
    fmax v14.4s, v14.4s, v20.4s
    fmax v15.4s, v15.4s, v20.4s
    fmax v16.4s, v16.4s, v20.4s
    fmax v17.4s, v17.4s, v20.4s
    fmax v18.4s, v18.4s, v20.4s
    fmax v19.4s, v19.4s, v20.4s
KAI_ASM_LABEL(label_84)  // Height 2: No activation
    cmp x28, #0x18
    bge label_97
    tbz x28, #4, label_88
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v10.4s }, [x9], #0x10
    st1 { v11.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    tbz x28, #2, label_86
    st1 { v12.4s }, [x9], #0x10
    st1 { v18.4s }, [x5], #0x10
    tbz x28, #1, label_85
    str d13, [x9], #0x8
    str d19, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v13.s }[2], [x9]
    st1 { v19.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_85)  // Height 2: Partial direct writeback: partial_0_20
    tbz x28, #0, label_96
    str s13, [x9, #0]
    str s19, [x5, #0]
    b label_96
KAI_ASM_LABEL(label_86)  // Height 2: Partial direct writeback: partial_1_16
    tbz x28, #1, label_87
    str d12, [x9], #0x8
    str d18, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v12.s }[2], [x9]
    st1 { v18.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_87)  // Height 2: Partial direct writeback: partial_0_16
    tbz x28, #0, label_96
    str s12, [x9, #0]
    str s18, [x5, #0]
    b label_96
KAI_ASM_LABEL(label_88)  // Height 2: Partial direct writeback: partial_3_0
    tbz x28, #3, label_92
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    tbz x28, #2, label_90
    st1 { v10.4s }, [x9], #0x10
    st1 { v16.4s }, [x5], #0x10
    tbz x28, #1, label_89
    str d11, [x9], #0x8
    str d17, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v11.s }[2], [x9]
    st1 { v17.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_89)  // Height 2: Partial direct writeback: partial_0_12
    tbz x28, #0, label_96
    str s11, [x9, #0]
    str s17, [x5, #0]
    b label_96
KAI_ASM_LABEL(label_90)  // Height 2: Partial direct writeback: partial_1_8
    tbz x28, #1, label_91
    str d10, [x9], #0x8
    str d16, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v10.s }[2], [x9]
    st1 { v16.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_91)  // Height 2: Partial direct writeback: partial_0_8
    tbz x28, #0, label_96
    str s10, [x9, #0]
    str s16, [x5, #0]
    b label_96
KAI_ASM_LABEL(label_92)  // Height 2: Partial direct writeback: partial_2_0
    tbz x28, #2, label_94
    st1 { v8.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    tbz x28, #1, label_93
    str d9, [x9], #0x8
    str d15, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v9.s }[2], [x9]
    st1 { v15.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_93)  // Height 2: Partial direct writeback: partial_0_4
    tbz x28, #0, label_96
    str s9, [x9, #0]
    str s15, [x5, #0]
    b label_96
KAI_ASM_LABEL(label_94)  // Height 2: Partial direct writeback: partial_1_0
    tbz x28, #1, label_95
    str d8, [x9], #0x8
    str d14, [x5], #0x8
    tbz x28, #0, label_96
    st1 { v8.s }[2], [x9]
    st1 { v14.s }[2], [x5]
    b label_96
KAI_ASM_LABEL(label_95)  // Height 2: Partial direct writeback: partial_0_0
    str s8, [x9, #0]
    str s14, [x5, #0]
KAI_ASM_LABEL(label_96)  // Height 2: Partial direct writeback: Done
    b label_98
KAI_ASM_LABEL(label_97)  // Height 2: Full writeback
    str q8, [x9, #0]
    str q9, [x9, #0x10]
    str q10, [x9, #0x20]
    str q11, [x9, #0x30]
    str q12, [x9, #0x40]
    str q13, [x9, #0x50]
    add x9, x9, #0x60
    str q14, [x5, #0]
    str q15, [x5, #0x10]
    str q16, [x5, #0x20]
    str q17, [x5, #0x30]
    str q18, [x5, #0x40]
    str q19, [x5, #0x50]
KAI_ASM_LABEL(label_98)  // Height 2: Writeback done
    subs x28, x28, #0x18
    bgt label_51
    b label_197
KAI_ASM_LABEL(label_99)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x9, [x0, #0x50]
    ldr x28, [x0, #0x8]
    ldr x27, [x0, #0x30]
    ldr x26, [x0, #0x40]
    madd x9, x13, x1, x9
KAI_ASM_LABEL(label_100)  // Height 3: Column loop
    cbz x26, label_108
    cmp x28, #0x18
    bge label_106
    cbnz x13, label_105
    mov x2, x28
    mov x1, SP
    cmp x2, #0x4
    blt label_102
KAI_ASM_LABEL(label_101)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x26], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_101
KAI_ASM_LABEL(label_102)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_104
KAI_ASM_LABEL(label_103)  // Height 3: Partial bias load: element loop
    ldr s16, [x26], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_103
KAI_ASM_LABEL(label_104)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_105)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_107
KAI_ASM_LABEL(label_106)  // Height 3: full width bias
    ldr q8, [x26, #0]
    ldr q9, [x26, #0x10]
    ldr q10, [x26, #0x20]
    ldr q11, [x26, #0x30]
    ldr q12, [x26, #0x40]
    ldr q13, [x26, #0x50]
    add x26, x26, #0x60
KAI_ASM_LABEL(label_107)  // Height 3: bias load done
    mov v14.16b, v8.16b
    mov v15.16b, v9.16b
    mov v16.16b, v10.16b
    mov v17.16b, v11.16b
    mov v18.16b, v12.16b
    mov v19.16b, v13.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v12.16b
    mov v25.16b, v13.16b
    b label_123
KAI_ASM_LABEL(label_108)  // Height 3: no bias
    tbz x12, #0, label_122
    ldr x1, [x0, #0x58]
    cmp x28, #0x18
    add x5, x9, x1
    add x4, x5, x1
    bge label_121
    tbz x28, #4, label_112
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v11.4s }, [x9], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    tbz x28, #2, label_110
    ld1 { v12.4s }, [x9], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    tbz x28, #1, label_109
    ldr d13, [x9], #0x8
    mov x1, #0x58
    ldr d19, [x5], #0x8
    ldr d25, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v13.s }[2], [x9]
    ld1 { v19.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_109)  // Height 3: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x28, #0, label_120
    ldr s13, [x9, #0]
    ldr s19, [x5, #0]
    ldr s25, [x4, #0]
    b label_120
KAI_ASM_LABEL(label_110)  // Height 3: Partial accumulate: partial_1_16
    tbz x28, #1, label_111
    ldr d12, [x9], #0x8
    mov x1, #0x48
    ldr d18, [x5], #0x8
    ldr d24, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v12.s }[2], [x9]
    ld1 { v18.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_111)  // Height 3: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x28, #0, label_120
    ldr s12, [x9, #0]
    ldr s18, [x5, #0]
    ldr s24, [x4, #0]
    b label_120
KAI_ASM_LABEL(label_112)  // Height 3: Partial accumulate: partial_3_0
    tbz x28, #3, label_116
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    tbz x28, #2, label_114
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    tbz x28, #1, label_113
    ldr d11, [x9], #0x8
    mov x1, #0x38
    ldr d17, [x5], #0x8
    ldr d23, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v11.s }[2], [x9]
    ld1 { v17.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_113)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x28, #0, label_120
    ldr s11, [x9, #0]
    ldr s17, [x5, #0]
    ldr s23, [x4, #0]
    b label_120
KAI_ASM_LABEL(label_114)  // Height 3: Partial accumulate: partial_1_8
    tbz x28, #1, label_115
    ldr d10, [x9], #0x8
    mov x1, #0x28
    ldr d16, [x5], #0x8
    ldr d22, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v10.s }[2], [x9]
    ld1 { v16.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_115)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x28, #0, label_120
    ldr s10, [x9, #0]
    ldr s16, [x5, #0]
    ldr s22, [x4, #0]
    b label_120
KAI_ASM_LABEL(label_116)  // Height 3: Partial accumulate: partial_2_0
    tbz x28, #2, label_118
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    tbz x28, #1, label_117
    ldr d9, [x9], #0x8
    mov x1, #0x18
    ldr d15, [x5], #0x8
    ldr d21, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v9.s }[2], [x9]
    ld1 { v15.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_117)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x28, #0, label_120
    ldr s9, [x9, #0]
    ldr s15, [x5, #0]
    ldr s21, [x4, #0]
    b label_120
KAI_ASM_LABEL(label_118)  // Height 3: Partial accumulate: partial_1_0
    tbz x28, #1, label_119
    ldr d8, [x9], #0x8
    mov x1, #0x8
    ldr d14, [x5], #0x8
    ldr d20, [x4], #0x8
    tbz x28, #0, label_120
    ld1 { v8.s }[2], [x9]
    ld1 { v14.s }[2], [x5]
    ld1 { v20.s }[2], [x4]
    b label_120
KAI_ASM_LABEL(label_119)  // Height 3: Partial accumulate: partial_0_0
    ldr s8, [x9, #0]
    mov x1, #0
    ldr s14, [x5, #0]
    ldr s20, [x4, #0]
KAI_ASM_LABEL(label_120)  // Height 3: Partial accumulate: Done
    sub x9, x9, x1
    b label_123
KAI_ASM_LABEL(label_121)  // Height 3: full accumulate
    ldr q8, [x9, #0]
    ldr q9, [x9, #0x10]
    ldr q10, [x9, #0x20]
    ldr q11, [x9, #0x30]
    ldr q12, [x9, #0x40]
    ldr q13, [x9, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
    ldr q20, [x4, #0]
    ldr q21, [x4, #0x10]
    ldr q22, [x4, #0x20]
    ldr q23, [x4, #0x30]
    ldr q24, [x4, #0x40]
    ldr q25, [x4, #0x50]
    b label_123
KAI_ASM_LABEL(label_122)  // Height 3: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
    movi v23.16b, #0
    movi v24.16b, #0
    movi v25.16b, #0
KAI_ASM_LABEL(label_123)  // Height 3: setup done
    ldr x7, [x0, #0x28]
    tbnz x12, #3, label_124
    ldr x25, [x0, #0x20]
    ldr x24, [x0, #0x10]
    madd x25, x13, x7, x25
    add x23, x25, x7
    add x22, x23, x7
    b label_126
KAI_ASM_LABEL(label_124)  // Height 3: setup indirect input
    mov x10, #0
KAI_ASM_LABEL(label_125)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x10, LSL #0x3]
    ldr x24, [x2, x10, LSL #0x3]
    add x1, x1, x13, LSL #3
    ldr x25, [x1, #0]
    ldr x23, [x1, #0x8]
    ldr x22, [x1, #0x10]
    cbnz x10, label_126
    add x25, x25, x7, LSL #2
    add x23, x23, x7, LSL #2
    add x22, x22, x7, LSL #2
KAI_ASM_LABEL(label_126)  // Height 3: input setup done
    cmp x24, #0x4
    blt label_129
    ldr q0, [x25, #0]
    cmp x24, #0x8
    ldr q1, [x23, #0]
    ldr q2, [x22, #0]
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    blt label_128
KAI_ASM_LABEL(label_127)  // Height 3: Multiply loop: Main loop head
    fmla v8.4s, v4.4s, v0.s[0]
    ldr x4, [x27, #0x48]
    fmla v14.4s, v4.4s, v1.s[0]
    ldr x3, [x27, #0x58]
    fmla v20.4s, v4.4s, v2.s[0]
    ldr d4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr x2, [x27, #0x68]
    fmla v15.4s, v5.4s, v1.s[0]
    ldr x1, [x27, #0x78]
    fmla v21.4s, v5.4s, v2.s[0]
    ldr d5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    mov v4.d[1], x4
    fmla v16.4s, v6.4s, v1.s[0]
    ldr x4, [x27, #0x88]
    fmla v22.4s, v6.4s, v2.s[0]
    ldr d6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    mov v5.d[1], x3
    fmla v17.4s, v7.4s, v1.s[0]
    ldr x3, [x27, #0x98]
    fmla v23.4s, v7.4s, v2.s[0]
    ldr d7, [x27, #0x70]
    mov v6.d[1], x2
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    ldr x2, [x27, #0xa8]
    mov v7.d[1], x1
    fmla v24.4s, v4.4s, v2.s[0]
    ldr d4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    ldr x1, [x27, #0xb8]
    fmla v25.4s, v5.4s, v2.s[0]
    ldr d5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    mov v4.d[1], x4
    fmla v14.4s, v6.4s, v1.s[1]
    ldr x4, [x27, #0xc8]
    fmla v20.4s, v6.4s, v2.s[1]
    ldr d6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    mov v5.d[1], x3
    fmla v15.4s, v7.4s, v1.s[1]
    ldr x3, [x27, #0xd8]
    fmla v21.4s, v7.4s, v2.s[1]
    ldr d7, [x27, #0xb0]
    mov v6.d[1], x2
    fmla v10.4s, v4.4s, v0.s[1]
    fmla v16.4s, v4.4s, v1.s[1]
    ldr x2, [x27, #0xe8]
    mov v7.d[1], x1
    fmla v22.4s, v4.4s, v2.s[1]
    ldr d4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    fmla v17.4s, v5.4s, v1.s[1]
    ldr x1, [x27, #0xf8]
    fmla v23.4s, v5.4s, v2.s[1]
    ldr d5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    mov v4.d[1], x4
    fmla v18.4s, v6.4s, v1.s[1]
    ldr x4, [x27, #0x108]
    fmla v24.4s, v6.4s, v2.s[1]
    ldr d6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    mov v5.d[1], x3
    fmla v19.4s, v7.4s, v1.s[1]
    ldr x3, [x27, #0x118]
    fmla v25.4s, v7.4s, v2.s[1]
    ldr d7, [x27, #0xf0]
    mov v6.d[1], x2
    fmla v8.4s, v4.4s, v0.s[2]
    fmla v14.4s, v4.4s, v1.s[2]
    ldr x2, [x27, #0x128]
    mov v7.d[1], x1
    fmla v20.4s, v4.4s, v2.s[2]
    ldr d4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    fmla v15.4s, v5.4s, v1.s[2]
    ldr x1, [x27, #0x138]
    fmla v21.4s, v5.4s, v2.s[2]
    ldr d5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    mov v4.d[1], x4
    fmla v16.4s, v6.4s, v1.s[2]
    ldr x4, [x27, #0x148]
    fmla v22.4s, v6.4s, v2.s[2]
    ldr d6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    mov v5.d[1], x3
    fmla v17.4s, v7.4s, v1.s[2]
    ldr x3, [x27, #0x158]
    fmla v23.4s, v7.4s, v2.s[2]
    ldr d7, [x27, #0x130]
    mov v6.d[1], x2
    fmla v12.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v1.s[2]
    ldr x2, [x27, #0x168]
    mov v7.d[1], x1
    fmla v24.4s, v4.4s, v2.s[2]
    ldr d4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v1.s[2]
    ldr x1, [x27, #0x178]
    fmla v25.4s, v5.4s, v2.s[2]
    ldr d5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    mov v4.d[1], x4
    fmla v14.4s, v6.4s, v1.s[3]
    add x25, x25, #0x10
    fmla v20.4s, v6.4s, v2.s[3]
    ldr d6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    mov v5.d[1], x3
    fmla v15.4s, v7.4s, v1.s[3]
    add x23, x23, #0x10
    fmla v21.4s, v7.4s, v2.s[3]
    ldr d7, [x27, #0x170]
    mov v6.d[1], x2
    add x22, x22, #0x10
    add x27, x27, #0x180
    fmla v10.4s, v4.4s, v0.s[3]
    mov v7.d[1], x1
    fmla v16.4s, v4.4s, v1.s[3]
    fmla v22.4s, v4.4s, v2.s[3]
    ldr d4, [x27, #0]
    ldr x1, [x27, #0x8]
    fmla v11.4s, v5.4s, v0.s[3]
    fmla v17.4s, v5.4s, v1.s[3]
    ldr x6, [x27, #0x18]
    fmla v23.4s, v5.4s, v2.s[3]
    ldr d5, [x27, #0x10]
    fmla v12.4s, v6.4s, v0.s[3]
    ldr x5, [x27, #0x28]
    fmla v18.4s, v6.4s, v1.s[3]
    ldr x4, [x25, #0x8]
    fmla v24.4s, v6.4s, v2.s[3]
    ldr d6, [x27, #0x20]
    fmla v13.4s, v7.4s, v0.s[3]
    ldr d0, [x25, #0]
    fmla v19.4s, v7.4s, v1.s[3]
    ldr d1, [x23, #0]
    ldr x3, [x23, #0x8]
    fmla v25.4s, v7.4s, v2.s[3]
    ldr d2, [x22, #0]
    sub x24, x24, #0x4
    ldr d7, [x27, #0x30]
    cmp x24, #0x8
    ldr x2, [x22, #0x8]
    mov v4.d[1], x1
    ldr x1, [x27, #0x38]
    mov v5.d[1], x6
    prfm pldl1keep, [x25, #0x80]
    mov v6.d[1], x5
    prfm pldl1keep, [x23, #0x80]
    mov v0.d[1], x4
    prfm pldl1keep, [x22, #0x80]
    mov v1.d[1], x3
    mov v2.d[1], x2
    mov v7.d[1], x1
    bge label_127
KAI_ASM_LABEL(label_128)  // Height 3: Multiply loop: Single iteration only
    fmla v8.4s, v4.4s, v0.s[0]
    add x25, x25, #0x10
    fmla v14.4s, v4.4s, v1.s[0]
    add x23, x23, #0x10
    fmla v20.4s, v4.4s, v2.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    add x22, x22, #0x10
    fmla v15.4s, v5.4s, v1.s[0]
    sub x24, x24, #0x4
    fmla v21.4s, v5.4s, v2.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x25, #0x80]
    fmla v16.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x23, #0x80]
    fmla v22.4s, v6.4s, v2.s[0]
    ldr q6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    prfm pldl1keep, [x22, #0x80]
    fmla v17.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v2.s[0]
    ldr q7, [x27, #0x70]
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v2.s[0]
    ldr q4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v2.s[0]
    ldr q5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v20.4s, v6.4s, v2.s[1]
    ldr q6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v21.4s, v7.4s, v2.s[1]
    ldr q7, [x27, #0xb0]
    fmla v10.4s, v4.4s, v0.s[1]
    fmla v16.4s, v4.4s, v1.s[1]
    fmla v22.4s, v4.4s, v2.s[1]
    ldr q4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    fmla v17.4s, v5.4s, v1.s[1]
    fmla v23.4s, v5.4s, v2.s[1]
    ldr q5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    fmla v18.4s, v6.4s, v1.s[1]
    fmla v24.4s, v6.4s, v2.s[1]
    ldr q6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    fmla v19.4s, v7.4s, v1.s[1]
    fmla v25.4s, v7.4s, v2.s[1]
    ldr q7, [x27, #0xf0]
    fmla v8.4s, v4.4s, v0.s[2]
    fmla v14.4s, v4.4s, v1.s[2]
    fmla v20.4s, v4.4s, v2.s[2]
    ldr q4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    fmla v15.4s, v5.4s, v1.s[2]
    fmla v21.4s, v5.4s, v2.s[2]
    ldr q5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v16.4s, v6.4s, v1.s[2]
    fmla v22.4s, v6.4s, v2.s[2]
    ldr q6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v17.4s, v7.4s, v1.s[2]
    fmla v23.4s, v7.4s, v2.s[2]
    ldr q7, [x27, #0x130]
    fmla v12.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v1.s[2]
    fmla v24.4s, v4.4s, v2.s[2]
    ldr q4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v1.s[2]
    fmla v25.4s, v5.4s, v2.s[2]
    ldr q5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v20.4s, v6.4s, v2.s[3]
    ldr q6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v21.4s, v7.4s, v2.s[3]
    ldr q7, [x27, #0x170]
    fmla v10.4s, v4.4s, v0.s[3]
    add x27, x27, #0x180
    fmla v16.4s, v4.4s, v1.s[3]
    fmla v22.4s, v4.4s, v2.s[3]
    fmla v11.4s, v5.4s, v0.s[3]
    fmla v17.4s, v5.4s, v1.s[3]
    fmla v23.4s, v5.4s, v2.s[3]
    fmla v12.4s, v6.4s, v0.s[3]
    fmla v18.4s, v6.4s, v1.s[3]
    fmla v24.4s, v6.4s, v2.s[3]
    fmla v13.4s, v7.4s, v0.s[3]
    fmla v19.4s, v7.4s, v1.s[3]
    fmla v25.4s, v7.4s, v2.s[3]
KAI_ASM_LABEL(label_129)  // Height 3: Multiply loop: Main loop skip
    cbz x24, label_131
KAI_ASM_LABEL(label_130)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x25], #0x4
    sub x24, x24, #0x1
    ldr s1, [x23], #0x4
    ldr s2, [x22], #0x4
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v14.4s, v4.4s, v1.s[0]
    fmla v20.4s, v4.4s, v2.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v15.4s, v5.4s, v1.s[0]
    fmla v21.4s, v5.4s, v2.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    add x27, x27, #0x60
    fmla v16.4s, v6.4s, v1.s[0]
    fmla v22.4s, v6.4s, v2.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v17.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v2.s[0]
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v2.s[0]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v2.s[0]
    cbnz x24, label_130
KAI_ASM_LABEL(label_131)  // Height 3: Multiply loop: No odd multiplies
    tbz x12, #3, label_132
    ldr x1, [x0, #0x10]
    add x10, x10, #0x1
    cmp x10, x1
    bne label_125
KAI_ASM_LABEL(label_132)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x9, #0]
    add x5, x9, x1
    add x4, x5, x1
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    tbz x12, #1, label_133
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v27.4s }, [x2]
    ld1r { v26.4s }, [x1]
    fmin v8.4s, v8.4s, v27.4s
    fmin v9.4s, v9.4s, v27.4s
    fmin v10.4s, v10.4s, v27.4s
    fmin v11.4s, v11.4s, v27.4s
    fmin v12.4s, v12.4s, v27.4s
    fmin v13.4s, v13.4s, v27.4s
    fmin v14.4s, v14.4s, v27.4s
    fmin v15.4s, v15.4s, v27.4s
    fmin v16.4s, v16.4s, v27.4s
    fmin v17.4s, v17.4s, v27.4s
    fmin v18.4s, v18.4s, v27.4s
    fmin v19.4s, v19.4s, v27.4s
    fmin v20.4s, v20.4s, v27.4s
    fmin v21.4s, v21.4s, v27.4s
    fmin v22.4s, v22.4s, v27.4s
    fmin v23.4s, v23.4s, v27.4s
    fmin v24.4s, v24.4s, v27.4s
    fmin v25.4s, v25.4s, v27.4s
    fmax v8.4s, v8.4s, v26.4s
    fmax v9.4s, v9.4s, v26.4s
    fmax v10.4s, v10.4s, v26.4s
    fmax v11.4s, v11.4s, v26.4s
    fmax v12.4s, v12.4s, v26.4s
    fmax v13.4s, v13.4s, v26.4s
    fmax v14.4s, v14.4s, v26.4s
    fmax v15.4s, v15.4s, v26.4s
    fmax v16.4s, v16.4s, v26.4s
    fmax v17.4s, v17.4s, v26.4s
    fmax v18.4s, v18.4s, v26.4s
    fmax v19.4s, v19.4s, v26.4s
    fmax v20.4s, v20.4s, v26.4s
    fmax v21.4s, v21.4s, v26.4s
    fmax v22.4s, v22.4s, v26.4s
    fmax v23.4s, v23.4s, v26.4s
    fmax v24.4s, v24.4s, v26.4s
    fmax v25.4s, v25.4s, v26.4s
KAI_ASM_LABEL(label_133)  // Height 3: No activation
    cmp x28, #0x18
    bge label_146
    tbz x28, #4, label_137
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v10.4s }, [x9], #0x10
    st1 { v11.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    st1 { v22.4s }, [x4], #0x10
    st1 { v23.4s }, [x4], #0x10
    tbz x28, #2, label_135
    st1 { v12.4s }, [x9], #0x10
    st1 { v18.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    tbz x28, #1, label_134
    str d13, [x9], #0x8
    str d19, [x5], #0x8
    str d25, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v13.s }[2], [x9]
    st1 { v19.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_134)  // Height 3: Partial direct writeback: partial_0_20
    tbz x28, #0, label_145
    str s13, [x9, #0]
    str s19, [x5, #0]
    str s25, [x4, #0]
    b label_145
KAI_ASM_LABEL(label_135)  // Height 3: Partial direct writeback: partial_1_16
    tbz x28, #1, label_136
    str d12, [x9], #0x8
    str d18, [x5], #0x8
    str d24, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v12.s }[2], [x9]
    st1 { v18.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_136)  // Height 3: Partial direct writeback: partial_0_16
    tbz x28, #0, label_145
    str s12, [x9, #0]
    str s18, [x5, #0]
    str s24, [x4, #0]
    b label_145
KAI_ASM_LABEL(label_137)  // Height 3: Partial direct writeback: partial_3_0
    tbz x28, #3, label_141
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    tbz x28, #2, label_139
    st1 { v10.4s }, [x9], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v22.4s }, [x4], #0x10
    tbz x28, #1, label_138
    str d11, [x9], #0x8
    str d17, [x5], #0x8
    str d23, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v11.s }[2], [x9]
    st1 { v17.s }[2], [x5]
    st1 { v23.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_138)  // Height 3: Partial direct writeback: partial_0_12
    tbz x28, #0, label_145
    str s11, [x9, #0]
    str s17, [x5, #0]
    str s23, [x4, #0]
    b label_145
KAI_ASM_LABEL(label_139)  // Height 3: Partial direct writeback: partial_1_8
    tbz x28, #1, label_140
    str d10, [x9], #0x8
    str d16, [x5], #0x8
    str d22, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v10.s }[2], [x9]
    st1 { v16.s }[2], [x5]
    st1 { v22.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_140)  // Height 3: Partial direct writeback: partial_0_8
    tbz x28, #0, label_145
    str s10, [x9, #0]
    str s16, [x5, #0]
    str s22, [x4, #0]
    b label_145
KAI_ASM_LABEL(label_141)  // Height 3: Partial direct writeback: partial_2_0
    tbz x28, #2, label_143
    st1 { v8.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    tbz x28, #1, label_142
    str d9, [x9], #0x8
    str d15, [x5], #0x8
    str d21, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v9.s }[2], [x9]
    st1 { v15.s }[2], [x5]
    st1 { v21.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_142)  // Height 3: Partial direct writeback: partial_0_4
    tbz x28, #0, label_145
    str s9, [x9, #0]
    str s15, [x5, #0]
    str s21, [x4, #0]
    b label_145
KAI_ASM_LABEL(label_143)  // Height 3: Partial direct writeback: partial_1_0
    tbz x28, #1, label_144
    str d8, [x9], #0x8
    str d14, [x5], #0x8
    str d20, [x4], #0x8
    tbz x28, #0, label_145
    st1 { v8.s }[2], [x9]
    st1 { v14.s }[2], [x5]
    st1 { v20.s }[2], [x4]
    b label_145
KAI_ASM_LABEL(label_144)  // Height 3: Partial direct writeback: partial_0_0
    str s8, [x9, #0]
    str s14, [x5, #0]
    str s20, [x4, #0]
KAI_ASM_LABEL(label_145)  // Height 3: Partial direct writeback: Done
    b label_147
KAI_ASM_LABEL(label_146)  // Height 3: Full writeback
    str q8, [x9, #0]
    str q9, [x9, #0x10]
    str q10, [x9, #0x20]
    str q11, [x9, #0x30]
    str q12, [x9, #0x40]
    str q13, [x9, #0x50]
    add x9, x9, #0x60
    str q14, [x5, #0]
    str q15, [x5, #0x10]
    str q16, [x5, #0x20]
    str q17, [x5, #0x30]
    str q18, [x5, #0x40]
    str q19, [x5, #0x50]
    str q20, [x4, #0]
    str q21, [x4, #0x10]
    str q22, [x4, #0x20]
    str q23, [x4, #0x30]
    str q24, [x4, #0x40]
    str q25, [x4, #0x50]
KAI_ASM_LABEL(label_147)  // Height 3: Writeback done
    subs x28, x28, #0x18
    bgt label_100
    b label_197
KAI_ASM_LABEL(label_148)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x9, [x0, #0x50]
    ldr x28, [x0, #0x8]
    ldr x27, [x0, #0x30]
    ldr x26, [x0, #0x40]
    madd x9, x13, x1, x9
KAI_ASM_LABEL(label_149)  // Height 4: Column loop
    cbz x26, label_157
    cmp x28, #0x18
    bge label_155
    cbnz x13, label_154
    mov x2, x28
    mov x1, SP
    cmp x2, #0x4
    blt label_151
KAI_ASM_LABEL(label_150)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x26], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_150
KAI_ASM_LABEL(label_151)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_153
KAI_ASM_LABEL(label_152)  // Height 4: Partial bias load: element loop
    ldr s16, [x26], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_152
KAI_ASM_LABEL(label_153)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_154)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_156
KAI_ASM_LABEL(label_155)  // Height 4: full width bias
    ldr q8, [x26, #0]
    ldr q9, [x26, #0x10]
    ldr q10, [x26, #0x20]
    ldr q11, [x26, #0x30]
    ldr q12, [x26, #0x40]
    ldr q13, [x26, #0x50]
    add x26, x26, #0x60
KAI_ASM_LABEL(label_156)  // Height 4: bias load done
    mov v14.16b, v8.16b
    mov v15.16b, v9.16b
    mov v16.16b, v10.16b
    mov v17.16b, v11.16b
    mov v18.16b, v12.16b
    mov v19.16b, v13.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v12.16b
    mov v25.16b, v13.16b
    mov v26.16b, v8.16b
    mov v27.16b, v9.16b
    mov v28.16b, v10.16b
    mov v29.16b, v11.16b
    mov v30.16b, v12.16b
    mov v31.16b, v13.16b
    b label_172
KAI_ASM_LABEL(label_157)  // Height 4: no bias
    tbz x12, #0, label_171
    ldr x1, [x0, #0x58]
    cmp x28, #0x18
    add x5, x9, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_170
    tbz x28, #4, label_161
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    ld1 { v11.4s }, [x9], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    ld1 { v29.4s }, [x3], #0x10
    tbz x28, #2, label_159
    ld1 { v12.4s }, [x9], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v30.4s }, [x3], #0x10
    tbz x28, #1, label_158
    ldr d13, [x9], #0x8
    mov x1, #0x58
    ldr d19, [x5], #0x8
    ldr d25, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v13.s }[2], [x9]
    ld1 { v19.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    ld1 { v31.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_158)  // Height 4: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x28, #0, label_169
    ldr s13, [x9, #0]
    ldr s19, [x5, #0]
    ldr s25, [x4, #0]
    ldr s31, [x3, #0]
    b label_169
KAI_ASM_LABEL(label_159)  // Height 4: Partial accumulate: partial_1_16
    tbz x28, #1, label_160
    ldr d12, [x9], #0x8
    mov x1, #0x48
    ldr d18, [x5], #0x8
    ldr d24, [x4], #0x8
    ldr d30, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v12.s }[2], [x9]
    ld1 { v18.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    ld1 { v30.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_160)  // Height 4: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x28, #0, label_169
    ldr s12, [x9, #0]
    ldr s18, [x5, #0]
    ldr s24, [x4, #0]
    ldr s30, [x3, #0]
    b label_169
KAI_ASM_LABEL(label_161)  // Height 4: Partial accumulate: partial_3_0
    tbz x28, #3, label_165
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    ld1 { v9.4s }, [x9], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    tbz x28, #2, label_163
    ld1 { v10.4s }, [x9], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    tbz x28, #1, label_162
    ldr d11, [x9], #0x8
    mov x1, #0x38
    ldr d17, [x5], #0x8
    ldr d23, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v11.s }[2], [x9]
    ld1 { v17.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    ld1 { v29.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_162)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x28, #0, label_169
    ldr s11, [x9, #0]
    ldr s17, [x5, #0]
    ldr s23, [x4, #0]
    ldr s29, [x3, #0]
    b label_169
KAI_ASM_LABEL(label_163)  // Height 4: Partial accumulate: partial_1_8
    tbz x28, #1, label_164
    ldr d10, [x9], #0x8
    mov x1, #0x28
    ldr d16, [x5], #0x8
    ldr d22, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v10.s }[2], [x9]
    ld1 { v16.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    ld1 { v28.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_164)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x28, #0, label_169
    ldr s10, [x9, #0]
    ldr s16, [x5, #0]
    ldr s22, [x4, #0]
    ldr s28, [x3, #0]
    b label_169
KAI_ASM_LABEL(label_165)  // Height 4: Partial accumulate: partial_2_0
    tbz x28, #2, label_167
    ld1 { v8.4s }, [x9], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v20.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    tbz x28, #1, label_166
    ldr d9, [x9], #0x8
    mov x1, #0x18
    ldr d15, [x5], #0x8
    ldr d21, [x4], #0x8
    ldr d27, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v9.s }[2], [x9]
    ld1 { v15.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    ld1 { v27.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_166)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x28, #0, label_169
    ldr s9, [x9, #0]
    ldr s15, [x5, #0]
    ldr s21, [x4, #0]
    ldr s27, [x3, #0]
    b label_169
KAI_ASM_LABEL(label_167)  // Height 4: Partial accumulate: partial_1_0
    tbz x28, #1, label_168
    ldr d8, [x9], #0x8
    mov x1, #0x8
    ldr d14, [x5], #0x8
    ldr d20, [x4], #0x8
    ldr d26, [x3], #0x8
    tbz x28, #0, label_169
    ld1 { v8.s }[2], [x9]
    ld1 { v14.s }[2], [x5]
    ld1 { v20.s }[2], [x4]
    ld1 { v26.s }[2], [x3]
    b label_169
KAI_ASM_LABEL(label_168)  // Height 4: Partial accumulate: partial_0_0
    ldr s8, [x9, #0]
    mov x1, #0
    ldr s14, [x5, #0]
    ldr s20, [x4, #0]
    ldr s26, [x3, #0]
KAI_ASM_LABEL(label_169)  // Height 4: Partial accumulate: Done
    sub x9, x9, x1
    b label_172
KAI_ASM_LABEL(label_170)  // Height 4: full accumulate
    ldr q8, [x9, #0]
    ldr q9, [x9, #0x10]
    ldr q10, [x9, #0x20]
    ldr q11, [x9, #0x30]
    ldr q12, [x9, #0x40]
    ldr q13, [x9, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
    ldr q20, [x4, #0]
    ldr q21, [x4, #0x10]
    ldr q22, [x4, #0x20]
    ldr q23, [x4, #0x30]
    ldr q24, [x4, #0x40]
    ldr q25, [x4, #0x50]
    ldr q26, [x3, #0]
    ldr q27, [x3, #0x10]
    ldr q28, [x3, #0x20]
    ldr q29, [x3, #0x30]
    ldr q30, [x3, #0x40]
    ldr q31, [x3, #0x50]
    b label_172
KAI_ASM_LABEL(label_171)  // Height 4: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
    movi v23.16b, #0
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0
KAI_ASM_LABEL(label_172)  // Height 4: setup done
    ldr x21, [x0, #0x28]
    tbnz x12, #3, label_173
    ldr x25, [x0, #0x20]
    ldr x24, [x0, #0x10]
    madd x25, x13, x21, x25
    add x23, x25, x21
    add x22, x23, x21
    add x20, x22, x21
    b label_175
KAI_ASM_LABEL(label_173)  // Height 4: setup indirect input
    mov x10, #0
KAI_ASM_LABEL(label_174)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x10, LSL #0x3]
    ldr x24, [x2, x10, LSL #0x3]
    add x1, x1, x13, LSL #3
    ldr x25, [x1, #0]
    ldr x23, [x1, #0x8]
    ldr x22, [x1, #0x10]
    ldr x20, [x1, #0x18]
    cbnz x10, label_175
    add x25, x25, x21, LSL #2
    add x23, x23, x21, LSL #2
    add x22, x22, x21, LSL #2
    add x20, x20, x21, LSL #2
KAI_ASM_LABEL(label_175)  // Height 4: input setup done
    cmp x24, #0x4
    blt label_178
    ldr q0, [x25, #0]
    cmp x24, #0x8
    ldr q1, [x23, #0]
    ldr q2, [x22, #0]
    ldr q3, [x20, #0]
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    blt label_177
KAI_ASM_LABEL(label_176)  // Height 4: Multiply loop: Main loop head
    fmla v8.4s, v4.4s, v0.s[0]
    ldr x4, [x27, #0x48]
    fmla v14.4s, v4.4s, v1.s[0]
    ldr x3, [x27, #0x58]
    fmla v20.4s, v4.4s, v2.s[0]
    ldr x2, [x27, #0x68]
    fmla v26.4s, v4.4s, v3.s[0]
    ldr d4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    ldr x1, [x27, #0x78]
    fmla v15.4s, v5.4s, v1.s[0]
    add x25, x25, #0x10
    fmla v21.4s, v5.4s, v2.s[0]
    mov v4.d[1], x4
    fmla v27.4s, v5.4s, v3.s[0]
    ldr d5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr x4, [x27, #0x88]
    fmla v16.4s, v6.4s, v1.s[0]
    add x23, x23, #0x10
    fmla v22.4s, v6.4s, v2.s[0]
    mov v5.d[1], x3
    fmla v28.4s, v6.4s, v3.s[0]
    ldr d6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x3, [x27, #0x98]
    fmla v17.4s, v7.4s, v1.s[0]
    add x22, x22, #0x10
    fmla v23.4s, v7.4s, v2.s[0]
    mov v6.d[1], x2
    fmla v29.4s, v7.4s, v3.s[0]
    ldr d7, [x27, #0x70]
    fmla v12.4s, v4.4s, v0.s[0]
    ldr x2, [x27, #0xa8]
    fmla v18.4s, v4.4s, v1.s[0]
    add x20, x20, #0x10
    mov v7.d[1], x1
    fmla v24.4s, v4.4s, v2.s[0]
    fmla v30.4s, v4.4s, v3.s[0]
    ldr d4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    ldr x1, [x27, #0xb8]
    fmla v19.4s, v5.4s, v1.s[0]
    ldr x8, [x25, #0x8]
    fmla v25.4s, v5.4s, v2.s[0]
    mov v4.d[1], x4
    fmla v31.4s, v5.4s, v3.s[0]
    ldr d5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr x4, [x27, #0xc8]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr x7, [x23, #0x8]
    fmla v20.4s, v6.4s, v2.s[1]
    mov v5.d[1], x3
    fmla v26.4s, v6.4s, v3.s[1]
    ldr d6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr x3, [x27, #0xd8]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr x6, [x22, #0x8]
    fmla v21.4s, v7.4s, v2.s[1]
    mov v6.d[1], x2
    fmla v27.4s, v7.4s, v3.s[1]
    ldr d7, [x27, #0xb0]
    fmla v10.4s, v4.4s, v0.s[1]
    ldr x2, [x27, #0xe8]
    fmla v16.4s, v4.4s, v1.s[1]
    ldr x5, [x20, #0x8]
    mov v7.d[1], x1
    fmla v22.4s, v4.4s, v2.s[1]
    fmla v28.4s, v4.4s, v3.s[1]
    ldr d4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    ldr x1, [x27, #0xf8]
    fmla v17.4s, v5.4s, v1.s[1]
    sub x24, x24, #0x4
    fmla v23.4s, v5.4s, v2.s[1]
    mov v4.d[1], x4
    fmla v29.4s, v5.4s, v3.s[1]
    ldr d5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    ldr x4, [x27, #0x108]
    fmla v18.4s, v6.4s, v1.s[1]
    cmp x24, #0x8
    fmla v24.4s, v6.4s, v2.s[1]
    mov v5.d[1], x3
    fmla v30.4s, v6.4s, v3.s[1]
    ldr d6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    ldr x3, [x27, #0x118]
    fmla v19.4s, v7.4s, v1.s[1]
    prfm pldl1keep, [x25, #0x80]
    fmla v25.4s, v7.4s, v2.s[1]
    mov v6.d[1], x2
    fmla v31.4s, v7.4s, v3.s[1]
    ldr d7, [x27, #0xf0]
    fmla v8.4s, v4.4s, v0.s[2]
    ldr x2, [x27, #0x128]
    fmla v14.4s, v4.4s, v1.s[2]
    prfm pldl1keep, [x23, #0x80]
    mov v7.d[1], x1
    fmla v20.4s, v4.4s, v2.s[2]
    fmla v26.4s, v4.4s, v3.s[2]
    ldr d4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    ldr x1, [x27, #0x138]
    fmla v15.4s, v5.4s, v1.s[2]
    prfm pldl1keep, [x22, #0x80]
    fmla v21.4s, v5.4s, v2.s[2]
    mov v4.d[1], x4
    fmla v27.4s, v5.4s, v3.s[2]
    ldr d5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr x4, [x27, #0x148]
    fmla v16.4s, v6.4s, v1.s[2]
    prfm pldl1keep, [x20, #0x80]
    fmla v22.4s, v6.4s, v2.s[2]
    mov v5.d[1], x3
    fmla v28.4s, v6.4s, v3.s[2]
    ldr d6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr x3, [x27, #0x158]
    fmla v17.4s, v7.4s, v1.s[2]
    fmla v23.4s, v7.4s, v2.s[2]
    mov v6.d[1], x2
    fmla v29.4s, v7.4s, v3.s[2]
    ldr d7, [x27, #0x130]
    fmla v12.4s, v4.4s, v0.s[2]
    ldr x2, [x27, #0x168]
    fmla v18.4s, v4.4s, v1.s[2]
    mov v7.d[1], x1
    fmla v24.4s, v4.4s, v2.s[2]
    fmla v30.4s, v4.4s, v3.s[2]
    ldr d4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    ldr x1, [x27, #0x178]
    fmla v19.4s, v5.4s, v1.s[2]
    fmla v25.4s, v5.4s, v2.s[2]
    mov v4.d[1], x4
    fmla v31.4s, v5.4s, v3.s[2]
    ldr d5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v20.4s, v6.4s, v2.s[3]
    mov v5.d[1], x3
    fmla v26.4s, v6.4s, v3.s[3]
    ldr d6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v21.4s, v7.4s, v2.s[3]
    mov v6.d[1], x2
    fmla v27.4s, v7.4s, v3.s[3]
    ldr d7, [x27, #0x170]
    add x27, x27, #0x180
    fmla v10.4s, v4.4s, v0.s[3]
    fmla v16.4s, v4.4s, v1.s[3]
    ldr x4, [x27, #0x8]
    mov v7.d[1], x1
    fmla v22.4s, v4.4s, v2.s[3]
    fmla v28.4s, v4.4s, v3.s[3]
    ldr d4, [x27, #0]
    fmla v11.4s, v5.4s, v0.s[3]
    ldr x3, [x27, #0x18]
    fmla v17.4s, v5.4s, v1.s[3]
    ldr x2, [x27, #0x28]
    fmla v23.4s, v5.4s, v2.s[3]
    ldr x1, [x27, #0x38]
    fmla v29.4s, v5.4s, v3.s[3]
    ldr d5, [x27, #0x10]
    fmla v12.4s, v6.4s, v0.s[3]
    mov v4.d[1], x4
    fmla v18.4s, v6.4s, v1.s[3]
    fmla v24.4s, v6.4s, v2.s[3]
    mov v5.d[1], x3
    fmla v30.4s, v6.4s, v3.s[3]
    ldr d6, [x27, #0x20]
    fmla v13.4s, v7.4s, v0.s[3]
    ldr d0, [x25, #0]
    fmla v19.4s, v7.4s, v1.s[3]
    ldr d1, [x23, #0]
    fmla v25.4s, v7.4s, v2.s[3]
    ldr d2, [x22, #0]
    fmla v31.4s, v7.4s, v3.s[3]
    ldr d3, [x20, #0]
    ldr d7, [x27, #0x30]
    mov v6.d[1], x2
    mov v0.d[1], x8
    mov v1.d[1], x7
    mov v2.d[1], x6
    mov v3.d[1], x5
    mov v7.d[1], x1
    bge label_176
KAI_ASM_LABEL(label_177)  // Height 4: Multiply loop: Single iteration only
    fmla v8.4s, v4.4s, v0.s[0]
    add x25, x25, #0x10
    fmla v14.4s, v4.4s, v1.s[0]
    add x23, x23, #0x10
    fmla v20.4s, v4.4s, v2.s[0]
    add x22, x22, #0x10
    fmla v26.4s, v4.4s, v3.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    add x20, x20, #0x10
    fmla v15.4s, v5.4s, v1.s[0]
    sub x24, x24, #0x4
    fmla v21.4s, v5.4s, v2.s[0]
    prfm pldl1keep, [x25, #0x80]
    fmla v27.4s, v5.4s, v3.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x23, #0x80]
    fmla v16.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x22, #0x80]
    fmla v22.4s, v6.4s, v2.s[0]
    prfm pldl1keep, [x20, #0x80]
    fmla v28.4s, v6.4s, v3.s[0]
    ldr q6, [x27, #0x60]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v17.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v2.s[0]
    fmla v29.4s, v7.4s, v3.s[0]
    ldr q7, [x27, #0x70]
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v2.s[0]
    fmla v30.4s, v4.4s, v3.s[0]
    ldr q4, [x27, #0x80]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v2.s[0]
    fmla v31.4s, v5.4s, v3.s[0]
    ldr q5, [x27, #0x90]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v20.4s, v6.4s, v2.s[1]
    fmla v26.4s, v6.4s, v3.s[1]
    ldr q6, [x27, #0xa0]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v21.4s, v7.4s, v2.s[1]
    fmla v27.4s, v7.4s, v3.s[1]
    ldr q7, [x27, #0xb0]
    fmla v10.4s, v4.4s, v0.s[1]
    fmla v16.4s, v4.4s, v1.s[1]
    fmla v22.4s, v4.4s, v2.s[1]
    fmla v28.4s, v4.4s, v3.s[1]
    ldr q4, [x27, #0xc0]
    fmla v11.4s, v5.4s, v0.s[1]
    fmla v17.4s, v5.4s, v1.s[1]
    fmla v23.4s, v5.4s, v2.s[1]
    fmla v29.4s, v5.4s, v3.s[1]
    ldr q5, [x27, #0xd0]
    fmla v12.4s, v6.4s, v0.s[1]
    fmla v18.4s, v6.4s, v1.s[1]
    fmla v24.4s, v6.4s, v2.s[1]
    fmla v30.4s, v6.4s, v3.s[1]
    ldr q6, [x27, #0xe0]
    fmla v13.4s, v7.4s, v0.s[1]
    fmla v19.4s, v7.4s, v1.s[1]
    fmla v25.4s, v7.4s, v2.s[1]
    fmla v31.4s, v7.4s, v3.s[1]
    ldr q7, [x27, #0xf0]
    fmla v8.4s, v4.4s, v0.s[2]
    fmla v14.4s, v4.4s, v1.s[2]
    fmla v20.4s, v4.4s, v2.s[2]
    fmla v26.4s, v4.4s, v3.s[2]
    ldr q4, [x27, #0x100]
    fmla v9.4s, v5.4s, v0.s[2]
    fmla v15.4s, v5.4s, v1.s[2]
    fmla v21.4s, v5.4s, v2.s[2]
    fmla v27.4s, v5.4s, v3.s[2]
    ldr q5, [x27, #0x110]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v16.4s, v6.4s, v1.s[2]
    fmla v22.4s, v6.4s, v2.s[2]
    fmla v28.4s, v6.4s, v3.s[2]
    ldr q6, [x27, #0x120]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v17.4s, v7.4s, v1.s[2]
    fmla v23.4s, v7.4s, v2.s[2]
    fmla v29.4s, v7.4s, v3.s[2]
    ldr q7, [x27, #0x130]
    fmla v12.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v1.s[2]
    fmla v24.4s, v4.4s, v2.s[2]
    fmla v30.4s, v4.4s, v3.s[2]
    ldr q4, [x27, #0x140]
    fmla v13.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v1.s[2]
    fmla v25.4s, v5.4s, v2.s[2]
    fmla v31.4s, v5.4s, v3.s[2]
    ldr q5, [x27, #0x150]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v20.4s, v6.4s, v2.s[3]
    fmla v26.4s, v6.4s, v3.s[3]
    ldr q6, [x27, #0x160]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v21.4s, v7.4s, v2.s[3]
    fmla v27.4s, v7.4s, v3.s[3]
    ldr q7, [x27, #0x170]
    fmla v10.4s, v4.4s, v0.s[3]
    add x27, x27, #0x180
    fmla v16.4s, v4.4s, v1.s[3]
    fmla v22.4s, v4.4s, v2.s[3]
    fmla v28.4s, v4.4s, v3.s[3]
    fmla v11.4s, v5.4s, v0.s[3]
    fmla v17.4s, v5.4s, v1.s[3]
    fmla v23.4s, v5.4s, v2.s[3]
    fmla v29.4s, v5.4s, v3.s[3]
    fmla v12.4s, v6.4s, v0.s[3]
    fmla v18.4s, v6.4s, v1.s[3]
    fmla v24.4s, v6.4s, v2.s[3]
    fmla v30.4s, v6.4s, v3.s[3]
    fmla v13.4s, v7.4s, v0.s[3]
    fmla v19.4s, v7.4s, v1.s[3]
    fmla v25.4s, v7.4s, v2.s[3]
    fmla v31.4s, v7.4s, v3.s[3]
KAI_ASM_LABEL(label_178)  // Height 4: Multiply loop: Main loop skip
    cbz x24, label_180
KAI_ASM_LABEL(label_179)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x25], #0x4
    sub x24, x24, #0x1
    ldr s1, [x23], #0x4
    ldr s2, [x22], #0x4
    ldr s3, [x20], #0x4
    ldr q4, [x27, #0]
    ldr q5, [x27, #0x10]
    ldr q6, [x27, #0x20]
    ldr q7, [x27, #0x30]
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v14.4s, v4.4s, v1.s[0]
    fmla v20.4s, v4.4s, v2.s[0]
    fmla v26.4s, v4.4s, v3.s[0]
    ldr q4, [x27, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v15.4s, v5.4s, v1.s[0]
    fmla v21.4s, v5.4s, v2.s[0]
    fmla v27.4s, v5.4s, v3.s[0]
    ldr q5, [x27, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    add x27, x27, #0x60
    fmla v16.4s, v6.4s, v1.s[0]
    fmla v22.4s, v6.4s, v2.s[0]
    fmla v28.4s, v6.4s, v3.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v17.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v2.s[0]
    fmla v29.4s, v7.4s, v3.s[0]
    fmla v12.4s, v4.4s, v0.s[0]
    fmla v18.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v2.s[0]
    fmla v30.4s, v4.4s, v3.s[0]
    fmla v13.4s, v5.4s, v0.s[0]
    fmla v19.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v2.s[0]
    fmla v31.4s, v5.4s, v3.s[0]
    cbnz x24, label_179
KAI_ASM_LABEL(label_180)  // Height 4: Multiply loop: No odd multiplies
    tbz x12, #3, label_181
    ldr x1, [x0, #0x10]
    add x10, x10, #0x1
    cmp x10, x1
    bne label_174
KAI_ASM_LABEL(label_181)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x9, #0]
    add x5, x9, x1
    add x4, x5, x1
    prfm pstl1keep, [x5, #0]
    add x3, x4, x1
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    tbz x12, #1, label_182
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v1.4s }, [x2]
    ld1r { v0.4s }, [x1]
    fmin v8.4s, v8.4s, v1.4s
    fmin v9.4s, v9.4s, v1.4s
    fmin v10.4s, v10.4s, v1.4s
    fmin v11.4s, v11.4s, v1.4s
    fmin v12.4s, v12.4s, v1.4s
    fmin v13.4s, v13.4s, v1.4s
    fmin v14.4s, v14.4s, v1.4s
    fmin v15.4s, v15.4s, v1.4s
    fmin v16.4s, v16.4s, v1.4s
    fmin v17.4s, v17.4s, v1.4s
    fmin v18.4s, v18.4s, v1.4s
    fmin v19.4s, v19.4s, v1.4s
    fmin v20.4s, v20.4s, v1.4s
    fmin v21.4s, v21.4s, v1.4s
    fmin v22.4s, v22.4s, v1.4s
    fmin v23.4s, v23.4s, v1.4s
    fmin v24.4s, v24.4s, v1.4s
    fmin v25.4s, v25.4s, v1.4s
    fmin v26.4s, v26.4s, v1.4s
    fmin v27.4s, v27.4s, v1.4s
    fmin v28.4s, v28.4s, v1.4s
    fmin v29.4s, v29.4s, v1.4s
    fmin v30.4s, v30.4s, v1.4s
    fmin v31.4s, v31.4s, v1.4s
    fmax v8.4s, v8.4s, v0.4s
    fmax v9.4s, v9.4s, v0.4s
    fmax v10.4s, v10.4s, v0.4s
    fmax v11.4s, v11.4s, v0.4s
    fmax v12.4s, v12.4s, v0.4s
    fmax v13.4s, v13.4s, v0.4s
    fmax v14.4s, v14.4s, v0.4s
    fmax v15.4s, v15.4s, v0.4s
    fmax v16.4s, v16.4s, v0.4s
    fmax v17.4s, v17.4s, v0.4s
    fmax v18.4s, v18.4s, v0.4s
    fmax v19.4s, v19.4s, v0.4s
    fmax v20.4s, v20.4s, v0.4s
    fmax v21.4s, v21.4s, v0.4s
    fmax v22.4s, v22.4s, v0.4s
    fmax v23.4s, v23.4s, v0.4s
    fmax v24.4s, v24.4s, v0.4s
    fmax v25.4s, v25.4s, v0.4s
    fmax v26.4s, v26.4s, v0.4s
    fmax v27.4s, v27.4s, v0.4s
    fmax v28.4s, v28.4s, v0.4s
    fmax v29.4s, v29.4s, v0.4s
    fmax v30.4s, v30.4s, v0.4s
    fmax v31.4s, v31.4s, v0.4s
KAI_ASM_LABEL(label_182)  // Height 4: No activation
    cmp x28, #0x18
    bge label_195
    tbz x28, #4, label_186
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v10.4s }, [x9], #0x10
    st1 { v11.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    st1 { v22.4s }, [x4], #0x10
    st1 { v23.4s }, [x4], #0x10
    st1 { v26.4s }, [x3], #0x10
    st1 { v27.4s }, [x3], #0x10
    st1 { v28.4s }, [x3], #0x10
    st1 { v29.4s }, [x3], #0x10
    tbz x28, #2, label_184
    st1 { v12.4s }, [x9], #0x10
    st1 { v18.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v30.4s }, [x3], #0x10
    tbz x28, #1, label_183
    str d13, [x9], #0x8
    str d19, [x5], #0x8
    str d25, [x4], #0x8
    str d31, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v13.s }[2], [x9]
    st1 { v19.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    st1 { v31.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_183)  // Height 4: Partial direct writeback: partial_0_20
    tbz x28, #0, label_194
    str s13, [x9, #0]
    str s19, [x5, #0]
    str s25, [x4, #0]
    str s31, [x3, #0]
    b label_194
KAI_ASM_LABEL(label_184)  // Height 4: Partial direct writeback: partial_1_16
    tbz x28, #1, label_185
    str d12, [x9], #0x8
    str d18, [x5], #0x8
    str d24, [x4], #0x8
    str d30, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v12.s }[2], [x9]
    st1 { v18.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    st1 { v30.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_185)  // Height 4: Partial direct writeback: partial_0_16
    tbz x28, #0, label_194
    str s12, [x9, #0]
    str s18, [x5, #0]
    str s24, [x4, #0]
    str s30, [x3, #0]
    b label_194
KAI_ASM_LABEL(label_186)  // Height 4: Partial direct writeback: partial_3_0
    tbz x28, #3, label_190
    st1 { v8.4s }, [x9], #0x10
    st1 { v9.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v15.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    st1 { v26.4s }, [x3], #0x10
    st1 { v27.4s }, [x3], #0x10
    tbz x28, #2, label_188
    st1 { v10.4s }, [x9], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v22.4s }, [x4], #0x10
    st1 { v28.4s }, [x3], #0x10
    tbz x28, #1, label_187
    str d11, [x9], #0x8
    str d17, [x5], #0x8
    str d23, [x4], #0x8
    str d29, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v11.s }[2], [x9]
    st1 { v17.s }[2], [x5]
    st1 { v23.s }[2], [x4]
    st1 { v29.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_187)  // Height 4: Partial direct writeback: partial_0_12
    tbz x28, #0, label_194
    str s11, [x9, #0]
    str s17, [x5, #0]
    str s23, [x4, #0]
    str s29, [x3, #0]
    b label_194
KAI_ASM_LABEL(label_188)  // Height 4: Partial direct writeback: partial_1_8
    tbz x28, #1, label_189
    str d10, [x9], #0x8
    str d16, [x5], #0x8
    str d22, [x4], #0x8
    str d28, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v10.s }[2], [x9]
    st1 { v16.s }[2], [x5]
    st1 { v22.s }[2], [x4]
    st1 { v28.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_189)  // Height 4: Partial direct writeback: partial_0_8
    tbz x28, #0, label_194
    str s10, [x9, #0]
    str s16, [x5, #0]
    str s22, [x4, #0]
    str s28, [x3, #0]
    b label_194
KAI_ASM_LABEL(label_190)  // Height 4: Partial direct writeback: partial_2_0
    tbz x28, #2, label_192
    st1 { v8.4s }, [x9], #0x10
    st1 { v14.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v26.4s }, [x3], #0x10
    tbz x28, #1, label_191
    str d9, [x9], #0x8
    str d15, [x5], #0x8
    str d21, [x4], #0x8
    str d27, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v9.s }[2], [x9]
    st1 { v15.s }[2], [x5]
    st1 { v21.s }[2], [x4]
    st1 { v27.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_191)  // Height 4: Partial direct writeback: partial_0_4
    tbz x28, #0, label_194
    str s9, [x9, #0]
    str s15, [x5, #0]
    str s21, [x4, #0]
    str s27, [x3, #0]
    b label_194
KAI_ASM_LABEL(label_192)  // Height 4: Partial direct writeback: partial_1_0
    tbz x28, #1, label_193
    str d8, [x9], #0x8
    str d14, [x5], #0x8
    str d20, [x4], #0x8
    str d26, [x3], #0x8
    tbz x28, #0, label_194
    st1 { v8.s }[2], [x9]
    st1 { v14.s }[2], [x5]
    st1 { v20.s }[2], [x4]
    st1 { v26.s }[2], [x3]
    b label_194
KAI_ASM_LABEL(label_193)  // Height 4: Partial direct writeback: partial_0_0
    str s8, [x9, #0]
    str s14, [x5, #0]
    str s20, [x4, #0]
    str s26, [x3, #0]
KAI_ASM_LABEL(label_194)  // Height 4: Partial direct writeback: Done
    b label_196
KAI_ASM_LABEL(label_195)  // Height 4: Full writeback
    str q8, [x9, #0]
    str q9, [x9, #0x10]
    str q10, [x9, #0x20]
    str q11, [x9, #0x30]
    str q12, [x9, #0x40]
    str q13, [x9, #0x50]
    add x9, x9, #0x60
    str q14, [x5, #0]
    str q15, [x5, #0x10]
    str q16, [x5, #0x20]
    str q17, [x5, #0x30]
    str q18, [x5, #0x40]
    str q19, [x5, #0x50]
    str q20, [x4, #0]
    str q21, [x4, #0x10]
    str q22, [x4, #0x20]
    str q23, [x4, #0x30]
    str q24, [x4, #0x40]
    str q25, [x4, #0x50]
    str q26, [x3, #0]
    str q27, [x3, #0x10]
    str q28, [x3, #0x20]
    str q29, [x3, #0x30]
    str q30, [x3, #0x40]
    str q31, [x3, #0x50]
KAI_ASM_LABEL(label_196)  // Height 4: Writeback done
    subs x28, x28, #0x18
    bgt label_149
    add x13, x13, #0x4
    cmp x13, x11
    blt label_1
KAI_ASM_LABEL(label_197)  // Exit
    add SP, SP, #0x60
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_fp32_mla_4x24_a55)

    KAI_ASM_END
