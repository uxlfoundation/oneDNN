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

    KAI_ASM_CODE(a64_hybrid_fp16_mla_6x32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_fp16_mla_6x32)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_fp16_mla_6x32)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_fp16_mla_6x32)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    mov x6, #0xfc00
    mov x5, #0x7c00
    sub SP, SP, #0x40
    mov x27, #0
    ldr x4, [x0, #0x60]
    ldrh w1, [x0, #0xa0]
    ldrh w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x26, [x0, #0]
    cmp x4, #0
    cset x25, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x25, x25, x1, LSL #1
    cset x1, NE
    orr x25, x25, x2, LSL #1
    orr x25, x25, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x26, x27
    cmp x1, #0x6
    bge label_286
    cmp x1, #0x4
    bgt label_229
    beq label_172
    cmp x1, #0x2
    bgt label_115
    beq label_58
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x20, label_10
    cmp x22, #0x20
    bge label_8
    cbnz x27, label_7
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_4
KAI_ASM_LABEL(label_3)  // Height 1: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_3
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_6
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
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
    b label_29
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x25, #0, label_28
    cmp x22, #0x20
    bge label_27
    tbz x22, #4, label_18
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v9.8h }, [x23], #0x10
    tbz x22, #3, label_14
    ld1 { v10.8h }, [x23], #0x10
    tbz x22, #2, label_12
    ldr d11, [x23], #0x8
    tbz x22, #1, label_11
    ld1 { v11.s }[2], [x23], #0x4
    mov x1, #0x3c
    tbz x22, #0, label_26
    ld1 { v11.h }[6], [x23]
    b label_26
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_26
    ld1 { v11.h }[4], [x23]
    b label_26
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_24
    tbz x22, #1, label_13
    ldr s11, [x23], #0x4
    mov x1, #0x34
    tbz x22, #0, label_26
    ld1 { v11.h }[2], [x23]
    b label_26
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_26
    ldr h11, [x23, #0]
    b label_26
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_2_16
    tbz x22, #2, label_16
    ldr d10, [x23], #0x8
    tbz x22, #1, label_15
    ld1 { v10.s }[2], [x23], #0x4
    mov x1, #0x2c
    tbz x22, #0, label_26
    ld1 { v10.h }[6], [x23]
    b label_26
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_26
    ld1 { v10.h }[4], [x23]
    b label_26
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_16
    tbz x22, #1, label_17
    ldr s10, [x23], #0x4
    mov x1, #0x24
    tbz x22, #0, label_26
    ld1 { v10.h }[2], [x23]
    b label_26
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_26
    ldr h10, [x23, #0]
    b label_26
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_3_0
    tbz x22, #3, label_22
    ld1 { v8.8h }, [x23], #0x10
    tbz x22, #2, label_20
    ldr d9, [x23], #0x8
    tbz x22, #1, label_19
    ld1 { v9.s }[2], [x23], #0x4
    mov x1, #0x1c
    tbz x22, #0, label_26
    ld1 { v9.h }[6], [x23]
    b label_26
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_26
    ld1 { v9.h }[4], [x23]
    b label_26
KAI_ASM_LABEL(label_20)  // Height 1: Partial accumulate: partial_1_8
    tbz x22, #1, label_21
    ldr s9, [x23], #0x4
    mov x1, #0x14
    tbz x22, #0, label_26
    ld1 { v9.h }[2], [x23]
    b label_26
KAI_ASM_LABEL(label_21)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_26
    ldr h9, [x23, #0]
    b label_26
KAI_ASM_LABEL(label_22)  // Height 1: Partial accumulate: partial_2_0
    tbz x22, #2, label_24
    ldr d8, [x23], #0x8
    tbz x22, #1, label_23
    ld1 { v8.s }[2], [x23], #0x4
    mov x1, #0xc
    tbz x22, #0, label_26
    ld1 { v8.h }[6], [x23]
    b label_26
KAI_ASM_LABEL(label_23)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_26
    ld1 { v8.h }[4], [x23]
    b label_26
KAI_ASM_LABEL(label_24)  // Height 1: Partial accumulate: partial_1_0
    tbz x22, #1, label_25
    ldr s8, [x23], #0x4
    mov x1, #0x4
    tbz x22, #0, label_26
    ld1 { v8.h }[2], [x23]
    b label_26
KAI_ASM_LABEL(label_25)  // Height 1: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    mov x1, #0
KAI_ASM_LABEL(label_26)  // Height 1: Partial accumulate: Done
    sub x23, x23, x1
    b label_29
KAI_ASM_LABEL(label_27)  // Height 1: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    b label_29
KAI_ASM_LABEL(label_28)  // Height 1: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
KAI_ASM_LABEL(label_29)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_30
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    b label_32
KAI_ASM_LABEL(label_30)  // Height 1: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_31)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    cbnz x24, label_32
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_32)  // Height 1: input setup done
    cmp x7, #0x8
    blt label_35
    ldr q0, [x8, #0]
    ldr q6, [x21, #0]
    cmp x7, #0x10
    ldr q7, [x21, #0x10]
    blt label_34
KAI_ASM_LABEL(label_33)  // Height 1: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr q7, [x21, #0x30]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    ldr q7, [x21, #0x10]
    bge label_33
KAI_ASM_LABEL(label_34)  // Height 1: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr q7, [x21, #0x30]
    add x8, x8, #0x10
    sub x7, x7, #0x8
    prfm pldl1keep, [x8, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
KAI_ASM_LABEL(label_35)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_37
KAI_ASM_LABEL(label_36)  // Height 1: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr q6, [x21, #0]
    sub x7, x7, #0x1
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    cbnz x7, label_36
KAI_ASM_LABEL(label_37)  // Height 1: Multiply loop: No odd multiplies
    tbz x25, #3, label_38
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_31
KAI_ASM_LABEL(label_38)  // Height 1: no string loop
    prfm pstl1keep, [x23, #0]
    tbz x25, #1, label_39
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.8h }, [x2]
    ld1r { v16.8h }, [x1]
    fmin v8.8h, v8.8h, v17.8h
    fmin v9.8h, v9.8h, v17.8h
    fmin v10.8h, v10.8h, v17.8h
    fmin v11.8h, v11.8h, v17.8h
    fmax v8.8h, v8.8h, v16.8h
    fmax v9.8h, v9.8h, v16.8h
    fmax v10.8h, v10.8h, v16.8h
    fmax v11.8h, v11.8h, v16.8h
KAI_ASM_LABEL(label_39)  // Height 1: No activation
    cmp x22, #0x20
    bge label_56
    tbz x22, #4, label_47
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    tbz x22, #3, label_43
    st1 { v10.8h }, [x23], #0x10
    tbz x22, #2, label_41
    str d11, [x23], #0x8
    tbz x22, #1, label_40
    st1 { v11.s }[2], [x23], #0x4
    tbz x22, #0, label_55
    st1 { v11.h }[6], [x23]
    b label_55
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_0_28
    tbz x22, #0, label_55
    st1 { v11.h }[4], [x23]
    b label_55
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_1_24
    tbz x22, #1, label_42
    str s11, [x23], #0x4
    tbz x22, #0, label_55
    st1 { v11.h }[2], [x23]
    b label_55
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_0_24
    tbz x22, #0, label_55
    str h11, [x23, #0]
    b label_55
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_2_16
    tbz x22, #2, label_45
    str d10, [x23], #0x8
    tbz x22, #1, label_44
    st1 { v10.s }[2], [x23], #0x4
    tbz x22, #0, label_55
    st1 { v10.h }[6], [x23]
    b label_55
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: partial_0_20
    tbz x22, #0, label_55
    st1 { v10.h }[4], [x23]
    b label_55
KAI_ASM_LABEL(label_45)  // Height 1: Partial direct writeback: partial_1_16
    tbz x22, #1, label_46
    str s10, [x23], #0x4
    tbz x22, #0, label_55
    st1 { v10.h }[2], [x23]
    b label_55
KAI_ASM_LABEL(label_46)  // Height 1: Partial direct writeback: partial_0_16
    tbz x22, #0, label_55
    str h10, [x23, #0]
    b label_55
KAI_ASM_LABEL(label_47)  // Height 1: Partial direct writeback: partial_3_0
    tbz x22, #3, label_51
    st1 { v8.8h }, [x23], #0x10
    tbz x22, #2, label_49
    str d9, [x23], #0x8
    tbz x22, #1, label_48
    st1 { v9.s }[2], [x23], #0x4
    tbz x22, #0, label_55
    st1 { v9.h }[6], [x23]
    b label_55
KAI_ASM_LABEL(label_48)  // Height 1: Partial direct writeback: partial_0_12
    tbz x22, #0, label_55
    st1 { v9.h }[4], [x23]
    b label_55
KAI_ASM_LABEL(label_49)  // Height 1: Partial direct writeback: partial_1_8
    tbz x22, #1, label_50
    str s9, [x23], #0x4
    tbz x22, #0, label_55
    st1 { v9.h }[2], [x23]
    b label_55
KAI_ASM_LABEL(label_50)  // Height 1: Partial direct writeback: partial_0_8
    tbz x22, #0, label_55
    str h9, [x23, #0]
    b label_55
KAI_ASM_LABEL(label_51)  // Height 1: Partial direct writeback: partial_2_0
    tbz x22, #2, label_53
    str d8, [x23], #0x8
    tbz x22, #1, label_52
    st1 { v8.s }[2], [x23], #0x4
    tbz x22, #0, label_55
    st1 { v8.h }[6], [x23]
    b label_55
KAI_ASM_LABEL(label_52)  // Height 1: Partial direct writeback: partial_0_4
    tbz x22, #0, label_55
    st1 { v8.h }[4], [x23]
    b label_55
KAI_ASM_LABEL(label_53)  // Height 1: Partial direct writeback: partial_1_0
    tbz x22, #1, label_54
    str s8, [x23], #0x4
    tbz x22, #0, label_55
    st1 { v8.h }[2], [x23]
    b label_55
KAI_ASM_LABEL(label_54)  // Height 1: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
KAI_ASM_LABEL(label_55)  // Height 1: Partial direct writeback: Done
    b label_57
KAI_ASM_LABEL(label_56)  // Height 1: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
KAI_ASM_LABEL(label_57)  // Height 1: Writeback done
    subs x22, x22, #0x20
    bgt label_2
    b label_343
KAI_ASM_LABEL(label_58)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_59)  // Height 2: Column loop
    cbz x20, label_67
    cmp x22, #0x20
    bge label_65
    cbnz x27, label_64
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_61
KAI_ASM_LABEL(label_60)  // Height 2: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_60
KAI_ASM_LABEL(label_61)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_63
KAI_ASM_LABEL(label_62)  // Height 2: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_62
KAI_ASM_LABEL(label_63)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_64)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_66
KAI_ASM_LABEL(label_65)  // Height 2: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_66)  // Height 2: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    b label_86
KAI_ASM_LABEL(label_67)  // Height 2: no bias
    tbz x25, #0, label_85
    ldr x1, [x0, #0x58]
    cmp x22, #0x20
    add x7, x23, x1
    bge label_84
    tbz x22, #4, label_75
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v9.8h }, [x23], #0x10
    ld1 { v13.8h }, [x7], #0x10
    tbz x22, #3, label_71
    ld1 { v10.8h }, [x23], #0x10
    ld1 { v14.8h }, [x7], #0x10
    tbz x22, #2, label_69
    ldr d11, [x23], #0x8
    ldr d15, [x7], #0x8
    tbz x22, #1, label_68
    ld1 { v11.s }[2], [x23], #0x4
    ld1 { v15.s }[2], [x7], #0x4
    mov x1, #0x3c
    tbz x22, #0, label_83
    ld1 { v11.h }[6], [x23]
    ld1 { v15.h }[6], [x7]
    b label_83
KAI_ASM_LABEL(label_68)  // Height 2: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_83
    ld1 { v11.h }[4], [x23]
    ld1 { v15.h }[4], [x7]
    b label_83
KAI_ASM_LABEL(label_69)  // Height 2: Partial accumulate: partial_1_24
    tbz x22, #1, label_70
    ldr s11, [x23], #0x4
    ldr s15, [x7], #0x4
    mov x1, #0x34
    tbz x22, #0, label_83
    ld1 { v11.h }[2], [x23]
    ld1 { v15.h }[2], [x7]
    b label_83
KAI_ASM_LABEL(label_70)  // Height 2: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_83
    ldr h11, [x23, #0]
    ldr h15, [x7, #0]
    b label_83
KAI_ASM_LABEL(label_71)  // Height 2: Partial accumulate: partial_2_16
    tbz x22, #2, label_73
    ldr d10, [x23], #0x8
    ldr d14, [x7], #0x8
    tbz x22, #1, label_72
    ld1 { v10.s }[2], [x23], #0x4
    ld1 { v14.s }[2], [x7], #0x4
    mov x1, #0x2c
    tbz x22, #0, label_83
    ld1 { v10.h }[6], [x23]
    ld1 { v14.h }[6], [x7]
    b label_83
KAI_ASM_LABEL(label_72)  // Height 2: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_83
    ld1 { v10.h }[4], [x23]
    ld1 { v14.h }[4], [x7]
    b label_83
KAI_ASM_LABEL(label_73)  // Height 2: Partial accumulate: partial_1_16
    tbz x22, #1, label_74
    ldr s10, [x23], #0x4
    ldr s14, [x7], #0x4
    mov x1, #0x24
    tbz x22, #0, label_83
    ld1 { v10.h }[2], [x23]
    ld1 { v14.h }[2], [x7]
    b label_83
KAI_ASM_LABEL(label_74)  // Height 2: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_83
    ldr h10, [x23, #0]
    ldr h14, [x7, #0]
    b label_83
KAI_ASM_LABEL(label_75)  // Height 2: Partial accumulate: partial_3_0
    tbz x22, #3, label_79
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    tbz x22, #2, label_77
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    tbz x22, #1, label_76
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    tbz x22, #0, label_83
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    b label_83
KAI_ASM_LABEL(label_76)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_83
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    b label_83
KAI_ASM_LABEL(label_77)  // Height 2: Partial accumulate: partial_1_8
    tbz x22, #1, label_78
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    tbz x22, #0, label_83
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    b label_83
KAI_ASM_LABEL(label_78)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_83
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    b label_83
KAI_ASM_LABEL(label_79)  // Height 2: Partial accumulate: partial_2_0
    tbz x22, #2, label_81
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    tbz x22, #1, label_80
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    tbz x22, #0, label_83
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    b label_83
KAI_ASM_LABEL(label_80)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_83
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    b label_83
KAI_ASM_LABEL(label_81)  // Height 2: Partial accumulate: partial_1_0
    tbz x22, #1, label_82
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    tbz x22, #0, label_83
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    b label_83
KAI_ASM_LABEL(label_82)  // Height 2: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
KAI_ASM_LABEL(label_83)  // Height 2: Partial accumulate: Done
    sub x23, x23, x1
    b label_86
KAI_ASM_LABEL(label_84)  // Height 2: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    b label_86
KAI_ASM_LABEL(label_85)  // Height 2: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
KAI_ASM_LABEL(label_86)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_87
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_89
KAI_ASM_LABEL(label_87)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_88)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_89
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_89)  // Height 2: input setup done
    cmp x7, #0x8
    blt label_92
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_91
KAI_ASM_LABEL(label_90)  // Height 2: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr q6, [x21, #0x20]
    sub x7, x7, #0x8
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    ldr q7, [x21, #0x30]
    add x8, x8, #0x10
    add x6, x6, #0x10
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    ldr q6, [x21, #0x40]
    prfm pldl1keep, [x6, #0x80]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr q1, [x6, #0]
    ldr q7, [x21, #0x10]
    bge label_90
KAI_ASM_LABEL(label_91)  // Height 2: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr q6, [x21, #0x20]
    add x8, x8, #0x10
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    ldr q7, [x21, #0x30]
    add x6, x6, #0x10
    sub x7, x7, #0x8
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
KAI_ASM_LABEL(label_92)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_94
KAI_ASM_LABEL(label_93)  // Height 2: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    cbnz x7, label_93
KAI_ASM_LABEL(label_94)  // Height 2: Multiply loop: No odd multiplies
    tbz x25, #3, label_95
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_88
KAI_ASM_LABEL(label_95)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    tbz x25, #1, label_96
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.8h }, [x2]
    ld1r { v16.8h }, [x1]
    fmin v8.8h, v8.8h, v17.8h
    fmin v9.8h, v9.8h, v17.8h
    fmin v10.8h, v10.8h, v17.8h
    fmin v11.8h, v11.8h, v17.8h
    fmin v12.8h, v12.8h, v17.8h
    fmin v13.8h, v13.8h, v17.8h
    fmin v14.8h, v14.8h, v17.8h
    fmin v15.8h, v15.8h, v17.8h
    fmax v8.8h, v8.8h, v16.8h
    fmax v9.8h, v9.8h, v16.8h
    fmax v10.8h, v10.8h, v16.8h
    fmax v11.8h, v11.8h, v16.8h
    fmax v12.8h, v12.8h, v16.8h
    fmax v13.8h, v13.8h, v16.8h
    fmax v14.8h, v14.8h, v16.8h
    fmax v15.8h, v15.8h, v16.8h
KAI_ASM_LABEL(label_96)  // Height 2: No activation
    cmp x22, #0x20
    bge label_113
    tbz x22, #4, label_104
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v13.8h }, [x7], #0x10
    tbz x22, #3, label_100
    st1 { v10.8h }, [x23], #0x10
    st1 { v14.8h }, [x7], #0x10
    tbz x22, #2, label_98
    str d11, [x23], #0x8
    str d15, [x7], #0x8
    tbz x22, #1, label_97
    st1 { v11.s }[2], [x23], #0x4
    st1 { v15.s }[2], [x7], #0x4
    tbz x22, #0, label_112
    st1 { v11.h }[6], [x23]
    st1 { v15.h }[6], [x7]
    b label_112
KAI_ASM_LABEL(label_97)  // Height 2: Partial direct writeback: partial_0_28
    tbz x22, #0, label_112
    st1 { v11.h }[4], [x23]
    st1 { v15.h }[4], [x7]
    b label_112
KAI_ASM_LABEL(label_98)  // Height 2: Partial direct writeback: partial_1_24
    tbz x22, #1, label_99
    str s11, [x23], #0x4
    str s15, [x7], #0x4
    tbz x22, #0, label_112
    st1 { v11.h }[2], [x23]
    st1 { v15.h }[2], [x7]
    b label_112
KAI_ASM_LABEL(label_99)  // Height 2: Partial direct writeback: partial_0_24
    tbz x22, #0, label_112
    str h11, [x23, #0]
    str h15, [x7, #0]
    b label_112
KAI_ASM_LABEL(label_100)  // Height 2: Partial direct writeback: partial_2_16
    tbz x22, #2, label_102
    str d10, [x23], #0x8
    str d14, [x7], #0x8
    tbz x22, #1, label_101
    st1 { v10.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x7], #0x4
    tbz x22, #0, label_112
    st1 { v10.h }[6], [x23]
    st1 { v14.h }[6], [x7]
    b label_112
KAI_ASM_LABEL(label_101)  // Height 2: Partial direct writeback: partial_0_20
    tbz x22, #0, label_112
    st1 { v10.h }[4], [x23]
    st1 { v14.h }[4], [x7]
    b label_112
KAI_ASM_LABEL(label_102)  // Height 2: Partial direct writeback: partial_1_16
    tbz x22, #1, label_103
    str s10, [x23], #0x4
    str s14, [x7], #0x4
    tbz x22, #0, label_112
    st1 { v10.h }[2], [x23]
    st1 { v14.h }[2], [x7]
    b label_112
KAI_ASM_LABEL(label_103)  // Height 2: Partial direct writeback: partial_0_16
    tbz x22, #0, label_112
    str h10, [x23, #0]
    str h14, [x7, #0]
    b label_112
KAI_ASM_LABEL(label_104)  // Height 2: Partial direct writeback: partial_3_0
    tbz x22, #3, label_108
    st1 { v8.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    tbz x22, #2, label_106
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    tbz x22, #1, label_105
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    tbz x22, #0, label_112
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    b label_112
KAI_ASM_LABEL(label_105)  // Height 2: Partial direct writeback: partial_0_12
    tbz x22, #0, label_112
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    b label_112
KAI_ASM_LABEL(label_106)  // Height 2: Partial direct writeback: partial_1_8
    tbz x22, #1, label_107
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    tbz x22, #0, label_112
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    b label_112
KAI_ASM_LABEL(label_107)  // Height 2: Partial direct writeback: partial_0_8
    tbz x22, #0, label_112
    str h9, [x23, #0]
    str h13, [x7, #0]
    b label_112
KAI_ASM_LABEL(label_108)  // Height 2: Partial direct writeback: partial_2_0
    tbz x22, #2, label_110
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    tbz x22, #1, label_109
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    tbz x22, #0, label_112
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    b label_112
KAI_ASM_LABEL(label_109)  // Height 2: Partial direct writeback: partial_0_4
    tbz x22, #0, label_112
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    b label_112
KAI_ASM_LABEL(label_110)  // Height 2: Partial direct writeback: partial_1_0
    tbz x22, #1, label_111
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    tbz x22, #0, label_112
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    b label_112
KAI_ASM_LABEL(label_111)  // Height 2: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
KAI_ASM_LABEL(label_112)  // Height 2: Partial direct writeback: Done
    b label_114
KAI_ASM_LABEL(label_113)  // Height 2: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
KAI_ASM_LABEL(label_114)  // Height 2: Writeback done
    subs x22, x22, #0x20
    bgt label_59
    b label_343
KAI_ASM_LABEL(label_115)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_116)  // Height 3: Column loop
    cbz x20, label_124
    cmp x22, #0x20
    bge label_122
    cbnz x27, label_121
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_118
KAI_ASM_LABEL(label_117)  // Height 3: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_117
KAI_ASM_LABEL(label_118)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_120
KAI_ASM_LABEL(label_119)  // Height 3: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_119
KAI_ASM_LABEL(label_120)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_121)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_123
KAI_ASM_LABEL(label_122)  // Height 3: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_123)  // Height 3: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    b label_143
KAI_ASM_LABEL(label_124)  // Height 3: no bias
    tbz x25, #0, label_142
    ldr x1, [x0, #0x58]
    cmp x22, #0x20
    add x7, x23, x1
    add x6, x7, x1
    bge label_141
    tbz x22, #4, label_132
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v9.8h }, [x23], #0x10
    ld1 { v13.8h }, [x7], #0x10
    ld1 { v17.8h }, [x6], #0x10
    tbz x22, #3, label_128
    ld1 { v10.8h }, [x23], #0x10
    ld1 { v14.8h }, [x7], #0x10
    ld1 { v18.8h }, [x6], #0x10
    tbz x22, #2, label_126
    ldr d11, [x23], #0x8
    ldr d15, [x7], #0x8
    ldr d19, [x6], #0x8
    tbz x22, #1, label_125
    ld1 { v11.s }[2], [x23], #0x4
    ld1 { v15.s }[2], [x7], #0x4
    mov x1, #0x3c
    ld1 { v19.s }[2], [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v11.h }[6], [x23]
    ld1 { v15.h }[6], [x7]
    ld1 { v19.h }[6], [x6]
    b label_140
KAI_ASM_LABEL(label_125)  // Height 3: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_140
    ld1 { v11.h }[4], [x23]
    ld1 { v15.h }[4], [x7]
    ld1 { v19.h }[4], [x6]
    b label_140
KAI_ASM_LABEL(label_126)  // Height 3: Partial accumulate: partial_1_24
    tbz x22, #1, label_127
    ldr s11, [x23], #0x4
    ldr s15, [x7], #0x4
    mov x1, #0x34
    ldr s19, [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v11.h }[2], [x23]
    ld1 { v15.h }[2], [x7]
    ld1 { v19.h }[2], [x6]
    b label_140
KAI_ASM_LABEL(label_127)  // Height 3: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_140
    ldr h11, [x23, #0]
    ldr h15, [x7, #0]
    ldr h19, [x6, #0]
    b label_140
KAI_ASM_LABEL(label_128)  // Height 3: Partial accumulate: partial_2_16
    tbz x22, #2, label_130
    ldr d10, [x23], #0x8
    ldr d14, [x7], #0x8
    ldr d18, [x6], #0x8
    tbz x22, #1, label_129
    ld1 { v10.s }[2], [x23], #0x4
    ld1 { v14.s }[2], [x7], #0x4
    mov x1, #0x2c
    ld1 { v18.s }[2], [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v10.h }[6], [x23]
    ld1 { v14.h }[6], [x7]
    ld1 { v18.h }[6], [x6]
    b label_140
KAI_ASM_LABEL(label_129)  // Height 3: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_140
    ld1 { v10.h }[4], [x23]
    ld1 { v14.h }[4], [x7]
    ld1 { v18.h }[4], [x6]
    b label_140
KAI_ASM_LABEL(label_130)  // Height 3: Partial accumulate: partial_1_16
    tbz x22, #1, label_131
    ldr s10, [x23], #0x4
    ldr s14, [x7], #0x4
    mov x1, #0x24
    ldr s18, [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v10.h }[2], [x23]
    ld1 { v14.h }[2], [x7]
    ld1 { v18.h }[2], [x6]
    b label_140
KAI_ASM_LABEL(label_131)  // Height 3: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_140
    ldr h10, [x23, #0]
    ldr h14, [x7, #0]
    ldr h18, [x6, #0]
    b label_140
KAI_ASM_LABEL(label_132)  // Height 3: Partial accumulate: partial_3_0
    tbz x22, #3, label_136
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    tbz x22, #2, label_134
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    tbz x22, #1, label_133
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    b label_140
KAI_ASM_LABEL(label_133)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_140
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    b label_140
KAI_ASM_LABEL(label_134)  // Height 3: Partial accumulate: partial_1_8
    tbz x22, #1, label_135
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    b label_140
KAI_ASM_LABEL(label_135)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_140
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    b label_140
KAI_ASM_LABEL(label_136)  // Height 3: Partial accumulate: partial_2_0
    tbz x22, #2, label_138
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    tbz x22, #1, label_137
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    b label_140
KAI_ASM_LABEL(label_137)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_140
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    b label_140
KAI_ASM_LABEL(label_138)  // Height 3: Partial accumulate: partial_1_0
    tbz x22, #1, label_139
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    tbz x22, #0, label_140
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    b label_140
KAI_ASM_LABEL(label_139)  // Height 3: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
KAI_ASM_LABEL(label_140)  // Height 3: Partial accumulate: Done
    sub x23, x23, x1
    b label_143
KAI_ASM_LABEL(label_141)  // Height 3: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q16, [x6, #0]
    ldr q17, [x6, #0x10]
    ldr q18, [x6, #0x20]
    ldr q19, [x6, #0x30]
    b label_143
KAI_ASM_LABEL(label_142)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_143)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_144
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_146
KAI_ASM_LABEL(label_144)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_145)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_146
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_146)  // Height 3: input setup done
    cmp x7, #0x8
    blt label_149
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_148
KAI_ASM_LABEL(label_147)  // Height 3: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x6, x6, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    ldr q7, [x21, #0x30]
    add x5, x5, #0x10
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    ldr q6, [x21, #0x40]
    prfm pldl1keep, [x5, #0x80]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr q1, [x6, #0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr q2, [x5, #0]
    ldr q7, [x21, #0x10]
    bge label_147
KAI_ASM_LABEL(label_148)  // Height 3: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x5, x5, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    ldr q7, [x21, #0x30]
    sub x7, x7, #0x8
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    ldr q6, [x21, #0x40]
    prfm pldl1keep, [x5, #0x80]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
KAI_ASM_LABEL(label_149)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_151
KAI_ASM_LABEL(label_150)  // Height 3: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    cbnz x7, label_150
KAI_ASM_LABEL(label_151)  // Height 3: Multiply loop: No odd multiplies
    tbz x25, #3, label_152
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_145
KAI_ASM_LABEL(label_152)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    tbz x25, #1, label_153
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v21.8h }, [x2]
    ld1r { v20.8h }, [x1]
    fmin v8.8h, v8.8h, v21.8h
    fmin v9.8h, v9.8h, v21.8h
    fmin v10.8h, v10.8h, v21.8h
    fmin v11.8h, v11.8h, v21.8h
    fmin v12.8h, v12.8h, v21.8h
    fmin v13.8h, v13.8h, v21.8h
    fmin v14.8h, v14.8h, v21.8h
    fmin v15.8h, v15.8h, v21.8h
    fmin v16.8h, v16.8h, v21.8h
    fmin v17.8h, v17.8h, v21.8h
    fmin v18.8h, v18.8h, v21.8h
    fmin v19.8h, v19.8h, v21.8h
    fmax v8.8h, v8.8h, v20.8h
    fmax v9.8h, v9.8h, v20.8h
    fmax v10.8h, v10.8h, v20.8h
    fmax v11.8h, v11.8h, v20.8h
    fmax v12.8h, v12.8h, v20.8h
    fmax v13.8h, v13.8h, v20.8h
    fmax v14.8h, v14.8h, v20.8h
    fmax v15.8h, v15.8h, v20.8h
    fmax v16.8h, v16.8h, v20.8h
    fmax v17.8h, v17.8h, v20.8h
    fmax v18.8h, v18.8h, v20.8h
    fmax v19.8h, v19.8h, v20.8h
KAI_ASM_LABEL(label_153)  // Height 3: No activation
    cmp x22, #0x20
    bge label_170
    tbz x22, #4, label_161
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v13.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v17.8h }, [x6], #0x10
    tbz x22, #3, label_157
    st1 { v10.8h }, [x23], #0x10
    st1 { v14.8h }, [x7], #0x10
    st1 { v18.8h }, [x6], #0x10
    tbz x22, #2, label_155
    str d11, [x23], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    tbz x22, #1, label_154
    st1 { v11.s }[2], [x23], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v19.s }[2], [x6], #0x4
    tbz x22, #0, label_169
    st1 { v11.h }[6], [x23]
    st1 { v15.h }[6], [x7]
    st1 { v19.h }[6], [x6]
    b label_169
KAI_ASM_LABEL(label_154)  // Height 3: Partial direct writeback: partial_0_28
    tbz x22, #0, label_169
    st1 { v11.h }[4], [x23]
    st1 { v15.h }[4], [x7]
    st1 { v19.h }[4], [x6]
    b label_169
KAI_ASM_LABEL(label_155)  // Height 3: Partial direct writeback: partial_1_24
    tbz x22, #1, label_156
    str s11, [x23], #0x4
    str s15, [x7], #0x4
    str s19, [x6], #0x4
    tbz x22, #0, label_169
    st1 { v11.h }[2], [x23]
    st1 { v15.h }[2], [x7]
    st1 { v19.h }[2], [x6]
    b label_169
KAI_ASM_LABEL(label_156)  // Height 3: Partial direct writeback: partial_0_24
    tbz x22, #0, label_169
    str h11, [x23, #0]
    str h15, [x7, #0]
    str h19, [x6, #0]
    b label_169
KAI_ASM_LABEL(label_157)  // Height 3: Partial direct writeback: partial_2_16
    tbz x22, #2, label_159
    str d10, [x23], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    tbz x22, #1, label_158
    st1 { v10.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x7], #0x4
    st1 { v18.s }[2], [x6], #0x4
    tbz x22, #0, label_169
    st1 { v10.h }[6], [x23]
    st1 { v14.h }[6], [x7]
    st1 { v18.h }[6], [x6]
    b label_169
KAI_ASM_LABEL(label_158)  // Height 3: Partial direct writeback: partial_0_20
    tbz x22, #0, label_169
    st1 { v10.h }[4], [x23]
    st1 { v14.h }[4], [x7]
    st1 { v18.h }[4], [x6]
    b label_169
KAI_ASM_LABEL(label_159)  // Height 3: Partial direct writeback: partial_1_16
    tbz x22, #1, label_160
    str s10, [x23], #0x4
    str s14, [x7], #0x4
    str s18, [x6], #0x4
    tbz x22, #0, label_169
    st1 { v10.h }[2], [x23]
    st1 { v14.h }[2], [x7]
    st1 { v18.h }[2], [x6]
    b label_169
KAI_ASM_LABEL(label_160)  // Height 3: Partial direct writeback: partial_0_16
    tbz x22, #0, label_169
    str h10, [x23, #0]
    str h14, [x7, #0]
    str h18, [x6, #0]
    b label_169
KAI_ASM_LABEL(label_161)  // Height 3: Partial direct writeback: partial_3_0
    tbz x22, #3, label_165
    st1 { v8.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    tbz x22, #2, label_163
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    tbz x22, #1, label_162
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    tbz x22, #0, label_169
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    b label_169
KAI_ASM_LABEL(label_162)  // Height 3: Partial direct writeback: partial_0_12
    tbz x22, #0, label_169
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    b label_169
KAI_ASM_LABEL(label_163)  // Height 3: Partial direct writeback: partial_1_8
    tbz x22, #1, label_164
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    tbz x22, #0, label_169
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    b label_169
KAI_ASM_LABEL(label_164)  // Height 3: Partial direct writeback: partial_0_8
    tbz x22, #0, label_169
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    b label_169
KAI_ASM_LABEL(label_165)  // Height 3: Partial direct writeback: partial_2_0
    tbz x22, #2, label_167
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    tbz x22, #1, label_166
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    tbz x22, #0, label_169
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    b label_169
KAI_ASM_LABEL(label_166)  // Height 3: Partial direct writeback: partial_0_4
    tbz x22, #0, label_169
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    b label_169
KAI_ASM_LABEL(label_167)  // Height 3: Partial direct writeback: partial_1_0
    tbz x22, #1, label_168
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    tbz x22, #0, label_169
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    b label_169
KAI_ASM_LABEL(label_168)  // Height 3: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
KAI_ASM_LABEL(label_169)  // Height 3: Partial direct writeback: Done
    b label_171
KAI_ASM_LABEL(label_170)  // Height 3: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
KAI_ASM_LABEL(label_171)  // Height 3: Writeback done
    subs x22, x22, #0x20
    bgt label_116
    b label_343
KAI_ASM_LABEL(label_172)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_173)  // Height 4: Column loop
    cbz x20, label_181
    cmp x22, #0x20
    bge label_179
    cbnz x27, label_178
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_175
KAI_ASM_LABEL(label_174)  // Height 4: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_174
KAI_ASM_LABEL(label_175)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_177
KAI_ASM_LABEL(label_176)  // Height 4: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_176
KAI_ASM_LABEL(label_177)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_178)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_180
KAI_ASM_LABEL(label_179)  // Height 4: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_180)  // Height 4: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    b label_200
KAI_ASM_LABEL(label_181)  // Height 4: no bias
    tbz x25, #0, label_199
    ldr x1, [x0, #0x58]
    cmp x22, #0x20
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    bge label_198
    tbz x22, #4, label_189
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    ld1 { v9.8h }, [x23], #0x10
    ld1 { v13.8h }, [x7], #0x10
    ld1 { v17.8h }, [x6], #0x10
    ld1 { v21.8h }, [x5], #0x10
    tbz x22, #3, label_185
    ld1 { v10.8h }, [x23], #0x10
    ld1 { v14.8h }, [x7], #0x10
    ld1 { v18.8h }, [x6], #0x10
    ld1 { v22.8h }, [x5], #0x10
    tbz x22, #2, label_183
    ldr d11, [x23], #0x8
    ldr d15, [x7], #0x8
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    tbz x22, #1, label_182
    ld1 { v11.s }[2], [x23], #0x4
    ld1 { v15.s }[2], [x7], #0x4
    mov x1, #0x3c
    ld1 { v19.s }[2], [x6], #0x4
    ld1 { v23.s }[2], [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v11.h }[6], [x23]
    ld1 { v15.h }[6], [x7]
    ld1 { v19.h }[6], [x6]
    ld1 { v23.h }[6], [x5]
    b label_197
KAI_ASM_LABEL(label_182)  // Height 4: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_197
    ld1 { v11.h }[4], [x23]
    ld1 { v15.h }[4], [x7]
    ld1 { v19.h }[4], [x6]
    ld1 { v23.h }[4], [x5]
    b label_197
KAI_ASM_LABEL(label_183)  // Height 4: Partial accumulate: partial_1_24
    tbz x22, #1, label_184
    ldr s11, [x23], #0x4
    ldr s15, [x7], #0x4
    mov x1, #0x34
    ldr s19, [x6], #0x4
    ldr s23, [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v11.h }[2], [x23]
    ld1 { v15.h }[2], [x7]
    ld1 { v19.h }[2], [x6]
    ld1 { v23.h }[2], [x5]
    b label_197
KAI_ASM_LABEL(label_184)  // Height 4: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_197
    ldr h11, [x23, #0]
    ldr h15, [x7, #0]
    ldr h19, [x6, #0]
    ldr h23, [x5, #0]
    b label_197
KAI_ASM_LABEL(label_185)  // Height 4: Partial accumulate: partial_2_16
    tbz x22, #2, label_187
    ldr d10, [x23], #0x8
    ldr d14, [x7], #0x8
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    tbz x22, #1, label_186
    ld1 { v10.s }[2], [x23], #0x4
    ld1 { v14.s }[2], [x7], #0x4
    mov x1, #0x2c
    ld1 { v18.s }[2], [x6], #0x4
    ld1 { v22.s }[2], [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v10.h }[6], [x23]
    ld1 { v14.h }[6], [x7]
    ld1 { v18.h }[6], [x6]
    ld1 { v22.h }[6], [x5]
    b label_197
KAI_ASM_LABEL(label_186)  // Height 4: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_197
    ld1 { v10.h }[4], [x23]
    ld1 { v14.h }[4], [x7]
    ld1 { v18.h }[4], [x6]
    ld1 { v22.h }[4], [x5]
    b label_197
KAI_ASM_LABEL(label_187)  // Height 4: Partial accumulate: partial_1_16
    tbz x22, #1, label_188
    ldr s10, [x23], #0x4
    ldr s14, [x7], #0x4
    mov x1, #0x24
    ldr s18, [x6], #0x4
    ldr s22, [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v10.h }[2], [x23]
    ld1 { v14.h }[2], [x7]
    ld1 { v18.h }[2], [x6]
    ld1 { v22.h }[2], [x5]
    b label_197
KAI_ASM_LABEL(label_188)  // Height 4: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_197
    ldr h10, [x23, #0]
    ldr h14, [x7, #0]
    ldr h18, [x6, #0]
    ldr h22, [x5, #0]
    b label_197
KAI_ASM_LABEL(label_189)  // Height 4: Partial accumulate: partial_3_0
    tbz x22, #3, label_193
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    tbz x22, #2, label_191
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    tbz x22, #1, label_190
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    b label_197
KAI_ASM_LABEL(label_190)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_197
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    b label_197
KAI_ASM_LABEL(label_191)  // Height 4: Partial accumulate: partial_1_8
    tbz x22, #1, label_192
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    b label_197
KAI_ASM_LABEL(label_192)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_197
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    b label_197
KAI_ASM_LABEL(label_193)  // Height 4: Partial accumulate: partial_2_0
    tbz x22, #2, label_195
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    tbz x22, #1, label_194
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    b label_197
KAI_ASM_LABEL(label_194)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_197
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    b label_197
KAI_ASM_LABEL(label_195)  // Height 4: Partial accumulate: partial_1_0
    tbz x22, #1, label_196
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    tbz x22, #0, label_197
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    b label_197
KAI_ASM_LABEL(label_196)  // Height 4: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
KAI_ASM_LABEL(label_197)  // Height 4: Partial accumulate: Done
    sub x23, x23, x1
    b label_200
KAI_ASM_LABEL(label_198)  // Height 4: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q16, [x6, #0]
    ldr q17, [x6, #0x10]
    ldr q18, [x6, #0x20]
    ldr q19, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
    b label_200
KAI_ASM_LABEL(label_199)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_200)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_201
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_203
KAI_ASM_LABEL(label_201)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_202)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_203
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_203)  // Height 4: input setup done
    cmp x7, #0x8
    blt label_206
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_205
KAI_ASM_LABEL(label_204)  // Height 4: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    ldr q6, [x21, #0x20]
    add x6, x6, #0x10
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    ldr q7, [x21, #0x30]
    cmp x7, #0x10
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr q6, [x21, #0x40]
    prfm pldl1keep, [x5, #0x80]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    prfm pldl1keep, [x4, #0x80]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr q1, [x6, #0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr q2, [x5, #0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr q3, [x4, #0]
    ldr q7, [x21, #0x10]
    bge label_204
KAI_ASM_LABEL(label_205)  // Height 4: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    ldr q6, [x21, #0x20]
    add x5, x5, #0x10
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    add x4, x4, #0x10
    prfm pldl1keep, [x8, #0x80]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    ldr q7, [x21, #0x30]
    prfm pldl1keep, [x6, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    prfm pldl1keep, [x5, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr q6, [x21, #0x40]
    prfm pldl1keep, [x4, #0x80]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
KAI_ASM_LABEL(label_206)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_208
KAI_ASM_LABEL(label_207)  // Height 4: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    cbnz x7, label_207
KAI_ASM_LABEL(label_208)  // Height 4: Multiply loop: No odd multiplies
    tbz x25, #3, label_209
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_202
KAI_ASM_LABEL(label_209)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    add x5, x6, x1
    prfm pstl1keep, [x5, #0]
    tbz x25, #1, label_210
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v25.8h }, [x2]
    ld1r { v24.8h }, [x1]
    fmin v8.8h, v8.8h, v25.8h
    fmin v9.8h, v9.8h, v25.8h
    fmin v10.8h, v10.8h, v25.8h
    fmin v11.8h, v11.8h, v25.8h
    fmin v12.8h, v12.8h, v25.8h
    fmin v13.8h, v13.8h, v25.8h
    fmin v14.8h, v14.8h, v25.8h
    fmin v15.8h, v15.8h, v25.8h
    fmin v16.8h, v16.8h, v25.8h
    fmin v17.8h, v17.8h, v25.8h
    fmin v18.8h, v18.8h, v25.8h
    fmin v19.8h, v19.8h, v25.8h
    fmin v20.8h, v20.8h, v25.8h
    fmin v21.8h, v21.8h, v25.8h
    fmin v22.8h, v22.8h, v25.8h
    fmin v23.8h, v23.8h, v25.8h
    fmax v8.8h, v8.8h, v24.8h
    fmax v9.8h, v9.8h, v24.8h
    fmax v10.8h, v10.8h, v24.8h
    fmax v11.8h, v11.8h, v24.8h
    fmax v12.8h, v12.8h, v24.8h
    fmax v13.8h, v13.8h, v24.8h
    fmax v14.8h, v14.8h, v24.8h
    fmax v15.8h, v15.8h, v24.8h
    fmax v16.8h, v16.8h, v24.8h
    fmax v17.8h, v17.8h, v24.8h
    fmax v18.8h, v18.8h, v24.8h
    fmax v19.8h, v19.8h, v24.8h
    fmax v20.8h, v20.8h, v24.8h
    fmax v21.8h, v21.8h, v24.8h
    fmax v22.8h, v22.8h, v24.8h
    fmax v23.8h, v23.8h, v24.8h
KAI_ASM_LABEL(label_210)  // Height 4: No activation
    cmp x22, #0x20
    bge label_227
    tbz x22, #4, label_218
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v13.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v17.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    st1 { v21.8h }, [x5], #0x10
    tbz x22, #3, label_214
    st1 { v10.8h }, [x23], #0x10
    st1 { v14.8h }, [x7], #0x10
    st1 { v18.8h }, [x6], #0x10
    st1 { v22.8h }, [x5], #0x10
    tbz x22, #2, label_212
    str d11, [x23], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    tbz x22, #1, label_211
    st1 { v11.s }[2], [x23], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v19.s }[2], [x6], #0x4
    st1 { v23.s }[2], [x5], #0x4
    tbz x22, #0, label_226
    st1 { v11.h }[6], [x23]
    st1 { v15.h }[6], [x7]
    st1 { v19.h }[6], [x6]
    st1 { v23.h }[6], [x5]
    b label_226
KAI_ASM_LABEL(label_211)  // Height 4: Partial direct writeback: partial_0_28
    tbz x22, #0, label_226
    st1 { v11.h }[4], [x23]
    st1 { v15.h }[4], [x7]
    st1 { v19.h }[4], [x6]
    st1 { v23.h }[4], [x5]
    b label_226
KAI_ASM_LABEL(label_212)  // Height 4: Partial direct writeback: partial_1_24
    tbz x22, #1, label_213
    str s11, [x23], #0x4
    str s15, [x7], #0x4
    str s19, [x6], #0x4
    str s23, [x5], #0x4
    tbz x22, #0, label_226
    st1 { v11.h }[2], [x23]
    st1 { v15.h }[2], [x7]
    st1 { v19.h }[2], [x6]
    st1 { v23.h }[2], [x5]
    b label_226
KAI_ASM_LABEL(label_213)  // Height 4: Partial direct writeback: partial_0_24
    tbz x22, #0, label_226
    str h11, [x23, #0]
    str h15, [x7, #0]
    str h19, [x6, #0]
    str h23, [x5, #0]
    b label_226
KAI_ASM_LABEL(label_214)  // Height 4: Partial direct writeback: partial_2_16
    tbz x22, #2, label_216
    str d10, [x23], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    tbz x22, #1, label_215
    st1 { v10.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x7], #0x4
    st1 { v18.s }[2], [x6], #0x4
    st1 { v22.s }[2], [x5], #0x4
    tbz x22, #0, label_226
    st1 { v10.h }[6], [x23]
    st1 { v14.h }[6], [x7]
    st1 { v18.h }[6], [x6]
    st1 { v22.h }[6], [x5]
    b label_226
KAI_ASM_LABEL(label_215)  // Height 4: Partial direct writeback: partial_0_20
    tbz x22, #0, label_226
    st1 { v10.h }[4], [x23]
    st1 { v14.h }[4], [x7]
    st1 { v18.h }[4], [x6]
    st1 { v22.h }[4], [x5]
    b label_226
KAI_ASM_LABEL(label_216)  // Height 4: Partial direct writeback: partial_1_16
    tbz x22, #1, label_217
    str s10, [x23], #0x4
    str s14, [x7], #0x4
    str s18, [x6], #0x4
    str s22, [x5], #0x4
    tbz x22, #0, label_226
    st1 { v10.h }[2], [x23]
    st1 { v14.h }[2], [x7]
    st1 { v18.h }[2], [x6]
    st1 { v22.h }[2], [x5]
    b label_226
KAI_ASM_LABEL(label_217)  // Height 4: Partial direct writeback: partial_0_16
    tbz x22, #0, label_226
    str h10, [x23, #0]
    str h14, [x7, #0]
    str h18, [x6, #0]
    str h22, [x5, #0]
    b label_226
KAI_ASM_LABEL(label_218)  // Height 4: Partial direct writeback: partial_3_0
    tbz x22, #3, label_222
    st1 { v8.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    tbz x22, #2, label_220
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    tbz x22, #1, label_219
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    tbz x22, #0, label_226
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    b label_226
KAI_ASM_LABEL(label_219)  // Height 4: Partial direct writeback: partial_0_12
    tbz x22, #0, label_226
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    b label_226
KAI_ASM_LABEL(label_220)  // Height 4: Partial direct writeback: partial_1_8
    tbz x22, #1, label_221
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    tbz x22, #0, label_226
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    b label_226
KAI_ASM_LABEL(label_221)  // Height 4: Partial direct writeback: partial_0_8
    tbz x22, #0, label_226
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    b label_226
KAI_ASM_LABEL(label_222)  // Height 4: Partial direct writeback: partial_2_0
    tbz x22, #2, label_224
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    tbz x22, #1, label_223
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    tbz x22, #0, label_226
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    b label_226
KAI_ASM_LABEL(label_223)  // Height 4: Partial direct writeback: partial_0_4
    tbz x22, #0, label_226
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    b label_226
KAI_ASM_LABEL(label_224)  // Height 4: Partial direct writeback: partial_1_0
    tbz x22, #1, label_225
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    tbz x22, #0, label_226
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    b label_226
KAI_ASM_LABEL(label_225)  // Height 4: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
KAI_ASM_LABEL(label_226)  // Height 4: Partial direct writeback: Done
    b label_228
KAI_ASM_LABEL(label_227)  // Height 4: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
    str q22, [x5, #0x20]
    str q23, [x5, #0x30]
KAI_ASM_LABEL(label_228)  // Height 4: Writeback done
    subs x22, x22, #0x20
    bgt label_173
    b label_343
KAI_ASM_LABEL(label_229)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_230)  // Height 5: Column loop
    cbz x20, label_238
    cmp x22, #0x20
    bge label_236
    cbnz x27, label_235
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_232
KAI_ASM_LABEL(label_231)  // Height 5: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_231
KAI_ASM_LABEL(label_232)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_234
KAI_ASM_LABEL(label_233)  // Height 5: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_233
KAI_ASM_LABEL(label_234)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_235)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_237
KAI_ASM_LABEL(label_236)  // Height 5: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_237)  // Height 5: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    b label_257
KAI_ASM_LABEL(label_238)  // Height 5: no bias
    tbz x25, #0, label_256
    ldr x1, [x0, #0x58]
    cmp x22, #0x20
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_255
    tbz x22, #4, label_246
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    ld1 { v24.8h }, [x4], #0x10
    ld1 { v9.8h }, [x23], #0x10
    ld1 { v13.8h }, [x7], #0x10
    ld1 { v17.8h }, [x6], #0x10
    ld1 { v21.8h }, [x5], #0x10
    ld1 { v25.8h }, [x4], #0x10
    tbz x22, #3, label_242
    ld1 { v10.8h }, [x23], #0x10
    ld1 { v14.8h }, [x7], #0x10
    ld1 { v18.8h }, [x6], #0x10
    ld1 { v22.8h }, [x5], #0x10
    ld1 { v26.8h }, [x4], #0x10
    tbz x22, #2, label_240
    ldr d11, [x23], #0x8
    ldr d15, [x7], #0x8
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d27, [x4], #0x8
    tbz x22, #1, label_239
    ld1 { v11.s }[2], [x23], #0x4
    ld1 { v15.s }[2], [x7], #0x4
    mov x1, #0x3c
    ld1 { v19.s }[2], [x6], #0x4
    ld1 { v23.s }[2], [x5], #0x4
    ld1 { v27.s }[2], [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v11.h }[6], [x23]
    ld1 { v15.h }[6], [x7]
    ld1 { v19.h }[6], [x6]
    ld1 { v23.h }[6], [x5]
    ld1 { v27.h }[6], [x4]
    b label_254
KAI_ASM_LABEL(label_239)  // Height 5: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_254
    ld1 { v11.h }[4], [x23]
    ld1 { v15.h }[4], [x7]
    ld1 { v19.h }[4], [x6]
    ld1 { v23.h }[4], [x5]
    ld1 { v27.h }[4], [x4]
    b label_254
KAI_ASM_LABEL(label_240)  // Height 5: Partial accumulate: partial_1_24
    tbz x22, #1, label_241
    ldr s11, [x23], #0x4
    ldr s15, [x7], #0x4
    mov x1, #0x34
    ldr s19, [x6], #0x4
    ldr s23, [x5], #0x4
    ldr s27, [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v11.h }[2], [x23]
    ld1 { v15.h }[2], [x7]
    ld1 { v19.h }[2], [x6]
    ld1 { v23.h }[2], [x5]
    ld1 { v27.h }[2], [x4]
    b label_254
KAI_ASM_LABEL(label_241)  // Height 5: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_254
    ldr h11, [x23, #0]
    ldr h15, [x7, #0]
    ldr h19, [x6, #0]
    ldr h23, [x5, #0]
    ldr h27, [x4, #0]
    b label_254
KAI_ASM_LABEL(label_242)  // Height 5: Partial accumulate: partial_2_16
    tbz x22, #2, label_244
    ldr d10, [x23], #0x8
    ldr d14, [x7], #0x8
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d26, [x4], #0x8
    tbz x22, #1, label_243
    ld1 { v10.s }[2], [x23], #0x4
    ld1 { v14.s }[2], [x7], #0x4
    mov x1, #0x2c
    ld1 { v18.s }[2], [x6], #0x4
    ld1 { v22.s }[2], [x5], #0x4
    ld1 { v26.s }[2], [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v10.h }[6], [x23]
    ld1 { v14.h }[6], [x7]
    ld1 { v18.h }[6], [x6]
    ld1 { v22.h }[6], [x5]
    ld1 { v26.h }[6], [x4]
    b label_254
KAI_ASM_LABEL(label_243)  // Height 5: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_254
    ld1 { v10.h }[4], [x23]
    ld1 { v14.h }[4], [x7]
    ld1 { v18.h }[4], [x6]
    ld1 { v22.h }[4], [x5]
    ld1 { v26.h }[4], [x4]
    b label_254
KAI_ASM_LABEL(label_244)  // Height 5: Partial accumulate: partial_1_16
    tbz x22, #1, label_245
    ldr s10, [x23], #0x4
    ldr s14, [x7], #0x4
    mov x1, #0x24
    ldr s18, [x6], #0x4
    ldr s22, [x5], #0x4
    ldr s26, [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v10.h }[2], [x23]
    ld1 { v14.h }[2], [x7]
    ld1 { v18.h }[2], [x6]
    ld1 { v22.h }[2], [x5]
    ld1 { v26.h }[2], [x4]
    b label_254
KAI_ASM_LABEL(label_245)  // Height 5: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_254
    ldr h10, [x23, #0]
    ldr h14, [x7, #0]
    ldr h18, [x6, #0]
    ldr h22, [x5, #0]
    ldr h26, [x4, #0]
    b label_254
KAI_ASM_LABEL(label_246)  // Height 5: Partial accumulate: partial_3_0
    tbz x22, #3, label_250
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    ld1 { v24.8h }, [x4], #0x10
    tbz x22, #2, label_248
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    tbz x22, #1, label_247
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    ld1 { v25.s }[2], [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    ld1 { v25.h }[6], [x4]
    b label_254
KAI_ASM_LABEL(label_247)  // Height 5: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_254
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    ld1 { v25.h }[4], [x4]
    b label_254
KAI_ASM_LABEL(label_248)  // Height 5: Partial accumulate: partial_1_8
    tbz x22, #1, label_249
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    ldr s25, [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    ld1 { v25.h }[2], [x4]
    b label_254
KAI_ASM_LABEL(label_249)  // Height 5: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_254
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    ldr h25, [x4, #0]
    b label_254
KAI_ASM_LABEL(label_250)  // Height 5: Partial accumulate: partial_2_0
    tbz x22, #2, label_252
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    tbz x22, #1, label_251
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    ld1 { v24.s }[2], [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    ld1 { v24.h }[6], [x4]
    b label_254
KAI_ASM_LABEL(label_251)  // Height 5: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_254
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    ld1 { v24.h }[4], [x4]
    b label_254
KAI_ASM_LABEL(label_252)  // Height 5: Partial accumulate: partial_1_0
    tbz x22, #1, label_253
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    ldr s24, [x4], #0x4
    tbz x22, #0, label_254
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    ld1 { v24.h }[2], [x4]
    b label_254
KAI_ASM_LABEL(label_253)  // Height 5: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
    ldr h24, [x4, #0]
KAI_ASM_LABEL(label_254)  // Height 5: Partial accumulate: Done
    sub x23, x23, x1
    b label_257
KAI_ASM_LABEL(label_255)  // Height 5: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q16, [x6, #0]
    ldr q17, [x6, #0x10]
    ldr q18, [x6, #0x20]
    ldr q19, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
    ldr q24, [x4, #0]
    ldr q25, [x4, #0x10]
    ldr q26, [x4, #0x20]
    ldr q27, [x4, #0x30]
    b label_257
KAI_ASM_LABEL(label_256)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_257)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_258
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_260
KAI_ASM_LABEL(label_258)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_259)  // Height 5: String loop
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
    cbnz x24, label_260
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_260)  // Height 5: input setup done
    cmp x7, #0x8
    blt label_263
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_262
KAI_ASM_LABEL(label_261)  // Height 5: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x4, x4, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    add x2, x2, #0x10
    cmp x7, #0x10
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr q7, [x21, #0x30]
    prfm pldl1keep, [x8, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    fmla v26.8h, v6.8h, v4.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr q1, [x6, #0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr q2, [x5, #0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr q3, [x4, #0]
    fmla v27.8h, v7.8h, v4.h[7]
    ldr q4, [x2, #0]
    ldr q7, [x21, #0x10]
    bge label_261
KAI_ASM_LABEL(label_262)  // Height 5: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x2, x2, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr q7, [x21, #0x30]
    sub x7, x7, #0x8
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    prfm pldl1keep, [x2, #0x80]
    fmla v26.8h, v6.8h, v4.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
    fmla v27.8h, v7.8h, v4.h[7]
KAI_ASM_LABEL(label_263)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_265
KAI_ASM_LABEL(label_264)  // Height 5: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr h4, [x2], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    fmla v24.8h, v6.8h, v4.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    cbnz x7, label_264
KAI_ASM_LABEL(label_265)  // Height 5: Multiply loop: No odd multiplies
    tbz x25, #3, label_266
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_259
KAI_ASM_LABEL(label_266)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    add x5, x6, x1
    prfm pstl1keep, [x5, #0]
    add x4, x5, x1
    prfm pstl1keep, [x4, #0]
    tbz x25, #1, label_267
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v29.8h }, [x2]
    ld1r { v28.8h }, [x1]
    fmin v8.8h, v8.8h, v29.8h
    fmin v9.8h, v9.8h, v29.8h
    fmin v10.8h, v10.8h, v29.8h
    fmin v11.8h, v11.8h, v29.8h
    fmin v12.8h, v12.8h, v29.8h
    fmin v13.8h, v13.8h, v29.8h
    fmin v14.8h, v14.8h, v29.8h
    fmin v15.8h, v15.8h, v29.8h
    fmin v16.8h, v16.8h, v29.8h
    fmin v17.8h, v17.8h, v29.8h
    fmin v18.8h, v18.8h, v29.8h
    fmin v19.8h, v19.8h, v29.8h
    fmin v20.8h, v20.8h, v29.8h
    fmin v21.8h, v21.8h, v29.8h
    fmin v22.8h, v22.8h, v29.8h
    fmin v23.8h, v23.8h, v29.8h
    fmin v24.8h, v24.8h, v29.8h
    fmin v25.8h, v25.8h, v29.8h
    fmin v26.8h, v26.8h, v29.8h
    fmin v27.8h, v27.8h, v29.8h
    fmax v8.8h, v8.8h, v28.8h
    fmax v9.8h, v9.8h, v28.8h
    fmax v10.8h, v10.8h, v28.8h
    fmax v11.8h, v11.8h, v28.8h
    fmax v12.8h, v12.8h, v28.8h
    fmax v13.8h, v13.8h, v28.8h
    fmax v14.8h, v14.8h, v28.8h
    fmax v15.8h, v15.8h, v28.8h
    fmax v16.8h, v16.8h, v28.8h
    fmax v17.8h, v17.8h, v28.8h
    fmax v18.8h, v18.8h, v28.8h
    fmax v19.8h, v19.8h, v28.8h
    fmax v20.8h, v20.8h, v28.8h
    fmax v21.8h, v21.8h, v28.8h
    fmax v22.8h, v22.8h, v28.8h
    fmax v23.8h, v23.8h, v28.8h
    fmax v24.8h, v24.8h, v28.8h
    fmax v25.8h, v25.8h, v28.8h
    fmax v26.8h, v26.8h, v28.8h
    fmax v27.8h, v27.8h, v28.8h
KAI_ASM_LABEL(label_267)  // Height 5: No activation
    cmp x22, #0x20
    bge label_284
    tbz x22, #4, label_275
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v13.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v17.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    st1 { v21.8h }, [x5], #0x10
    st1 { v24.8h }, [x4], #0x10
    st1 { v25.8h }, [x4], #0x10
    tbz x22, #3, label_271
    st1 { v10.8h }, [x23], #0x10
    st1 { v14.8h }, [x7], #0x10
    st1 { v18.8h }, [x6], #0x10
    st1 { v22.8h }, [x5], #0x10
    st1 { v26.8h }, [x4], #0x10
    tbz x22, #2, label_269
    str d11, [x23], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    str d27, [x4], #0x8
    tbz x22, #1, label_268
    st1 { v11.s }[2], [x23], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v19.s }[2], [x6], #0x4
    st1 { v23.s }[2], [x5], #0x4
    st1 { v27.s }[2], [x4], #0x4
    tbz x22, #0, label_283
    st1 { v11.h }[6], [x23]
    st1 { v15.h }[6], [x7]
    st1 { v19.h }[6], [x6]
    st1 { v23.h }[6], [x5]
    st1 { v27.h }[6], [x4]
    b label_283
KAI_ASM_LABEL(label_268)  // Height 5: Partial direct writeback: partial_0_28
    tbz x22, #0, label_283
    st1 { v11.h }[4], [x23]
    st1 { v15.h }[4], [x7]
    st1 { v19.h }[4], [x6]
    st1 { v23.h }[4], [x5]
    st1 { v27.h }[4], [x4]
    b label_283
KAI_ASM_LABEL(label_269)  // Height 5: Partial direct writeback: partial_1_24
    tbz x22, #1, label_270
    str s11, [x23], #0x4
    str s15, [x7], #0x4
    str s19, [x6], #0x4
    str s23, [x5], #0x4
    str s27, [x4], #0x4
    tbz x22, #0, label_283
    st1 { v11.h }[2], [x23]
    st1 { v15.h }[2], [x7]
    st1 { v19.h }[2], [x6]
    st1 { v23.h }[2], [x5]
    st1 { v27.h }[2], [x4]
    b label_283
KAI_ASM_LABEL(label_270)  // Height 5: Partial direct writeback: partial_0_24
    tbz x22, #0, label_283
    str h11, [x23, #0]
    str h15, [x7, #0]
    str h19, [x6, #0]
    str h23, [x5, #0]
    str h27, [x4, #0]
    b label_283
KAI_ASM_LABEL(label_271)  // Height 5: Partial direct writeback: partial_2_16
    tbz x22, #2, label_273
    str d10, [x23], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    str d26, [x4], #0x8
    tbz x22, #1, label_272
    st1 { v10.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x7], #0x4
    st1 { v18.s }[2], [x6], #0x4
    st1 { v22.s }[2], [x5], #0x4
    st1 { v26.s }[2], [x4], #0x4
    tbz x22, #0, label_283
    st1 { v10.h }[6], [x23]
    st1 { v14.h }[6], [x7]
    st1 { v18.h }[6], [x6]
    st1 { v22.h }[6], [x5]
    st1 { v26.h }[6], [x4]
    b label_283
KAI_ASM_LABEL(label_272)  // Height 5: Partial direct writeback: partial_0_20
    tbz x22, #0, label_283
    st1 { v10.h }[4], [x23]
    st1 { v14.h }[4], [x7]
    st1 { v18.h }[4], [x6]
    st1 { v22.h }[4], [x5]
    st1 { v26.h }[4], [x4]
    b label_283
KAI_ASM_LABEL(label_273)  // Height 5: Partial direct writeback: partial_1_16
    tbz x22, #1, label_274
    str s10, [x23], #0x4
    str s14, [x7], #0x4
    str s18, [x6], #0x4
    str s22, [x5], #0x4
    str s26, [x4], #0x4
    tbz x22, #0, label_283
    st1 { v10.h }[2], [x23]
    st1 { v14.h }[2], [x7]
    st1 { v18.h }[2], [x6]
    st1 { v22.h }[2], [x5]
    st1 { v26.h }[2], [x4]
    b label_283
KAI_ASM_LABEL(label_274)  // Height 5: Partial direct writeback: partial_0_16
    tbz x22, #0, label_283
    str h10, [x23, #0]
    str h14, [x7, #0]
    str h18, [x6, #0]
    str h22, [x5, #0]
    str h26, [x4, #0]
    b label_283
KAI_ASM_LABEL(label_275)  // Height 5: Partial direct writeback: partial_3_0
    tbz x22, #3, label_279
    st1 { v8.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    st1 { v24.8h }, [x4], #0x10
    tbz x22, #2, label_277
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    tbz x22, #1, label_276
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    st1 { v25.s }[2], [x4], #0x4
    tbz x22, #0, label_283
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    st1 { v25.h }[6], [x4]
    b label_283
KAI_ASM_LABEL(label_276)  // Height 5: Partial direct writeback: partial_0_12
    tbz x22, #0, label_283
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    st1 { v25.h }[4], [x4]
    b label_283
KAI_ASM_LABEL(label_277)  // Height 5: Partial direct writeback: partial_1_8
    tbz x22, #1, label_278
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    str s25, [x4], #0x4
    tbz x22, #0, label_283
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    st1 { v25.h }[2], [x4]
    b label_283
KAI_ASM_LABEL(label_278)  // Height 5: Partial direct writeback: partial_0_8
    tbz x22, #0, label_283
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    str h25, [x4, #0]
    b label_283
KAI_ASM_LABEL(label_279)  // Height 5: Partial direct writeback: partial_2_0
    tbz x22, #2, label_281
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    tbz x22, #1, label_280
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    tbz x22, #0, label_283
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    st1 { v24.h }[6], [x4]
    b label_283
KAI_ASM_LABEL(label_280)  // Height 5: Partial direct writeback: partial_0_4
    tbz x22, #0, label_283
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    st1 { v24.h }[4], [x4]
    b label_283
KAI_ASM_LABEL(label_281)  // Height 5: Partial direct writeback: partial_1_0
    tbz x22, #1, label_282
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    tbz x22, #0, label_283
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    st1 { v24.h }[2], [x4]
    b label_283
KAI_ASM_LABEL(label_282)  // Height 5: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
    str h24, [x4, #0]
KAI_ASM_LABEL(label_283)  // Height 5: Partial direct writeback: Done
    b label_285
KAI_ASM_LABEL(label_284)  // Height 5: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
    str q22, [x5, #0x20]
    str q23, [x5, #0x30]
    str q24, [x4, #0]
    str q25, [x4, #0x10]
    str q26, [x4, #0x20]
    str q27, [x4, #0x30]
KAI_ASM_LABEL(label_285)  // Height 5: Writeback done
    subs x22, x22, #0x20
    bgt label_230
    b label_343
KAI_ASM_LABEL(label_286)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_287)  // Height 6: Column loop
    cbz x20, label_295
    cmp x22, #0x20
    bge label_293
    cbnz x27, label_292
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_289
KAI_ASM_LABEL(label_288)  // Height 6: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_288
KAI_ASM_LABEL(label_289)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_291
KAI_ASM_LABEL(label_290)  // Height 6: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_290
KAI_ASM_LABEL(label_291)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_292)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_294
KAI_ASM_LABEL(label_293)  // Height 6: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    ldr q10, [x20, #0x20]
    ldr q11, [x20, #0x30]
    add x20, x20, #0x40
KAI_ASM_LABEL(label_294)  // Height 6: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    mov v28.16b, v8.16b
    mov v29.16b, v9.16b
    mov v30.16b, v10.16b
    mov v31.16b, v11.16b
    b label_314
KAI_ASM_LABEL(label_295)  // Height 6: no bias
    tbz x25, #0, label_313
    ldr x1, [x0, #0x58]
    cmp x22, #0x20
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_312
    tbz x22, #4, label_303
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    ld1 { v24.8h }, [x4], #0x10
    ld1 { v28.8h }, [x3], #0x10
    ld1 { v9.8h }, [x23], #0x10
    ld1 { v13.8h }, [x7], #0x10
    ld1 { v17.8h }, [x6], #0x10
    ld1 { v21.8h }, [x5], #0x10
    ld1 { v25.8h }, [x4], #0x10
    ld1 { v29.8h }, [x3], #0x10
    tbz x22, #3, label_299
    ld1 { v10.8h }, [x23], #0x10
    ld1 { v14.8h }, [x7], #0x10
    ld1 { v18.8h }, [x6], #0x10
    ld1 { v22.8h }, [x5], #0x10
    ld1 { v26.8h }, [x4], #0x10
    ld1 { v30.8h }, [x3], #0x10
    tbz x22, #2, label_297
    ldr d11, [x23], #0x8
    ldr d15, [x7], #0x8
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d27, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x22, #1, label_296
    ld1 { v11.s }[2], [x23], #0x4
    ld1 { v15.s }[2], [x7], #0x4
    mov x1, #0x3c
    ld1 { v19.s }[2], [x6], #0x4
    ld1 { v23.s }[2], [x5], #0x4
    ld1 { v27.s }[2], [x4], #0x4
    ld1 { v31.s }[2], [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v11.h }[6], [x23]
    ld1 { v15.h }[6], [x7]
    ld1 { v19.h }[6], [x6]
    ld1 { v23.h }[6], [x5]
    ld1 { v27.h }[6], [x4]
    ld1 { v31.h }[6], [x3]
    b label_311
KAI_ASM_LABEL(label_296)  // Height 6: Partial accumulate: partial_0_28
    mov x1, #0x38
    tbz x22, #0, label_311
    ld1 { v11.h }[4], [x23]
    ld1 { v15.h }[4], [x7]
    ld1 { v19.h }[4], [x6]
    ld1 { v23.h }[4], [x5]
    ld1 { v27.h }[4], [x4]
    ld1 { v31.h }[4], [x3]
    b label_311
KAI_ASM_LABEL(label_297)  // Height 6: Partial accumulate: partial_1_24
    tbz x22, #1, label_298
    ldr s11, [x23], #0x4
    ldr s15, [x7], #0x4
    mov x1, #0x34
    ldr s19, [x6], #0x4
    ldr s23, [x5], #0x4
    ldr s27, [x4], #0x4
    ldr s31, [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v11.h }[2], [x23]
    ld1 { v15.h }[2], [x7]
    ld1 { v19.h }[2], [x6]
    ld1 { v23.h }[2], [x5]
    ld1 { v27.h }[2], [x4]
    ld1 { v31.h }[2], [x3]
    b label_311
KAI_ASM_LABEL(label_298)  // Height 6: Partial accumulate: partial_0_24
    mov x1, #0x30
    tbz x22, #0, label_311
    ldr h11, [x23, #0]
    ldr h15, [x7, #0]
    ldr h19, [x6, #0]
    ldr h23, [x5, #0]
    ldr h27, [x4, #0]
    ldr h31, [x3, #0]
    b label_311
KAI_ASM_LABEL(label_299)  // Height 6: Partial accumulate: partial_2_16
    tbz x22, #2, label_301
    ldr d10, [x23], #0x8
    ldr d14, [x7], #0x8
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d26, [x4], #0x8
    ldr d30, [x3], #0x8
    tbz x22, #1, label_300
    ld1 { v10.s }[2], [x23], #0x4
    ld1 { v14.s }[2], [x7], #0x4
    mov x1, #0x2c
    ld1 { v18.s }[2], [x6], #0x4
    ld1 { v22.s }[2], [x5], #0x4
    ld1 { v26.s }[2], [x4], #0x4
    ld1 { v30.s }[2], [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v10.h }[6], [x23]
    ld1 { v14.h }[6], [x7]
    ld1 { v18.h }[6], [x6]
    ld1 { v22.h }[6], [x5]
    ld1 { v26.h }[6], [x4]
    ld1 { v30.h }[6], [x3]
    b label_311
KAI_ASM_LABEL(label_300)  // Height 6: Partial accumulate: partial_0_20
    mov x1, #0x28
    tbz x22, #0, label_311
    ld1 { v10.h }[4], [x23]
    ld1 { v14.h }[4], [x7]
    ld1 { v18.h }[4], [x6]
    ld1 { v22.h }[4], [x5]
    ld1 { v26.h }[4], [x4]
    ld1 { v30.h }[4], [x3]
    b label_311
KAI_ASM_LABEL(label_301)  // Height 6: Partial accumulate: partial_1_16
    tbz x22, #1, label_302
    ldr s10, [x23], #0x4
    ldr s14, [x7], #0x4
    mov x1, #0x24
    ldr s18, [x6], #0x4
    ldr s22, [x5], #0x4
    ldr s26, [x4], #0x4
    ldr s30, [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v10.h }[2], [x23]
    ld1 { v14.h }[2], [x7]
    ld1 { v18.h }[2], [x6]
    ld1 { v22.h }[2], [x5]
    ld1 { v26.h }[2], [x4]
    ld1 { v30.h }[2], [x3]
    b label_311
KAI_ASM_LABEL(label_302)  // Height 6: Partial accumulate: partial_0_16
    mov x1, #0x20
    tbz x22, #0, label_311
    ldr h10, [x23, #0]
    ldr h14, [x7, #0]
    ldr h18, [x6, #0]
    ldr h22, [x5, #0]
    ldr h26, [x4, #0]
    ldr h30, [x3, #0]
    b label_311
KAI_ASM_LABEL(label_303)  // Height 6: Partial accumulate: partial_3_0
    tbz x22, #3, label_307
    ld1 { v8.8h }, [x23], #0x10
    ld1 { v12.8h }, [x7], #0x10
    ld1 { v16.8h }, [x6], #0x10
    ld1 { v20.8h }, [x5], #0x10
    ld1 { v24.8h }, [x4], #0x10
    ld1 { v28.8h }, [x3], #0x10
    tbz x22, #2, label_305
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x22, #1, label_304
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    ld1 { v25.s }[2], [x4], #0x4
    ld1 { v29.s }[2], [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    ld1 { v25.h }[6], [x4]
    ld1 { v29.h }[6], [x3]
    b label_311
KAI_ASM_LABEL(label_304)  // Height 6: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_311
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    ld1 { v25.h }[4], [x4]
    ld1 { v29.h }[4], [x3]
    b label_311
KAI_ASM_LABEL(label_305)  // Height 6: Partial accumulate: partial_1_8
    tbz x22, #1, label_306
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    ldr s25, [x4], #0x4
    ldr s29, [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    ld1 { v25.h }[2], [x4]
    ld1 { v29.h }[2], [x3]
    b label_311
KAI_ASM_LABEL(label_306)  // Height 6: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_311
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    ldr h25, [x4, #0]
    ldr h29, [x3, #0]
    b label_311
KAI_ASM_LABEL(label_307)  // Height 6: Partial accumulate: partial_2_0
    tbz x22, #2, label_309
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x22, #1, label_308
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    ld1 { v24.s }[2], [x4], #0x4
    ld1 { v28.s }[2], [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    ld1 { v24.h }[6], [x4]
    ld1 { v28.h }[6], [x3]
    b label_311
KAI_ASM_LABEL(label_308)  // Height 6: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_311
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    ld1 { v24.h }[4], [x4]
    ld1 { v28.h }[4], [x3]
    b label_311
KAI_ASM_LABEL(label_309)  // Height 6: Partial accumulate: partial_1_0
    tbz x22, #1, label_310
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    ldr s24, [x4], #0x4
    ldr s28, [x3], #0x4
    tbz x22, #0, label_311
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    ld1 { v24.h }[2], [x4]
    ld1 { v28.h }[2], [x3]
    b label_311
KAI_ASM_LABEL(label_310)  // Height 6: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
    ldr h24, [x4, #0]
    ldr h28, [x3, #0]
KAI_ASM_LABEL(label_311)  // Height 6: Partial accumulate: Done
    sub x23, x23, x1
    b label_314
KAI_ASM_LABEL(label_312)  // Height 6: full accumulate
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q16, [x6, #0]
    ldr q17, [x6, #0x10]
    ldr q18, [x6, #0x20]
    ldr q19, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
    ldr q24, [x4, #0]
    ldr q25, [x4, #0x10]
    ldr q26, [x4, #0x20]
    ldr q27, [x4, #0x30]
    ldr q28, [x3, #0]
    ldr q29, [x3, #0x10]
    ldr q30, [x3, #0x20]
    ldr q31, [x3, #0x30]
    b label_314
KAI_ASM_LABEL(label_313)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_314)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_315
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_317
KAI_ASM_LABEL(label_315)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_316)  // Height 6: String loop
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
    cbnz x24, label_317
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_317)  // Height 6: input setup done
    cmp x7, #0x8
    blt label_320
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q5, [x1, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_319
KAI_ASM_LABEL(label_318)  // Height 6: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    fmla v28.8h, v6.8h, v5.h[0]
    ldr q6, [x21, #0x20]
    add x4, x4, #0x10
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    fmla v25.8h, v7.8h, v4.h[0]
    fmla v29.8h, v7.8h, v5.h[0]
    ldr q7, [x21, #0x30]
    prfm pldl1keep, [x6, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v30.8h, v6.8h, v5.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    fmla v31.8h, v7.8h, v5.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    fmla v28.8h, v6.8h, v5.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    fmla v29.8h, v7.8h, v5.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    fmla v30.8h, v6.8h, v5.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    fmla v31.8h, v7.8h, v5.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    fmla v28.8h, v6.8h, v5.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    fmla v29.8h, v7.8h, v5.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    fmla v30.8h, v6.8h, v5.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    fmla v31.8h, v7.8h, v5.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    fmla v28.8h, v6.8h, v5.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    fmla v29.8h, v7.8h, v5.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    fmla v30.8h, v6.8h, v5.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    fmla v31.8h, v7.8h, v5.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    fmla v28.8h, v6.8h, v5.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    fmla v29.8h, v7.8h, v5.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    fmla v30.8h, v6.8h, v5.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    fmla v31.8h, v7.8h, v5.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    fmla v28.8h, v6.8h, v5.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    fmla v29.8h, v7.8h, v5.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    fmla v30.8h, v6.8h, v5.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    fmla v31.8h, v7.8h, v5.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    fmla v28.8h, v6.8h, v5.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    fmla v29.8h, v7.8h, v5.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    fmla v30.8h, v6.8h, v5.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    fmla v31.8h, v7.8h, v5.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    fmla v28.8h, v6.8h, v5.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    fmla v29.8h, v7.8h, v5.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v30.8h, v6.8h, v5.h[7]
    ldr q6, [x21, #0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr q0, [x8, #0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr q1, [x6, #0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr q2, [x5, #0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr q3, [x4, #0]
    fmla v27.8h, v7.8h, v4.h[7]
    ldr q4, [x2, #0]
    fmla v31.8h, v7.8h, v5.h[7]
    ldr q5, [x1, #0]
    ldr q7, [x21, #0x10]
    bge label_318
KAI_ASM_LABEL(label_319)  // Height 6: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    fmla v28.8h, v6.8h, v5.h[0]
    ldr q6, [x21, #0x20]
    add x2, x2, #0x10
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    add x1, x1, #0x10
    prfm pldl1keep, [x8, #0x80]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    fmla v25.8h, v7.8h, v4.h[0]
    fmla v29.8h, v7.8h, v5.h[0]
    ldr q7, [x21, #0x30]
    prfm pldl1keep, [x4, #0x80]
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    sub x7, x7, #0x8
    prfm pldl1keep, [x2, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    prfm pldl1keep, [x1, #0x80]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v30.8h, v6.8h, v5.h[0]
    ldr q6, [x21, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    fmla v31.8h, v7.8h, v5.h[0]
    ldr q7, [x21, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    fmla v28.8h, v6.8h, v5.h[1]
    ldr q6, [x21, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    fmla v29.8h, v7.8h, v5.h[1]
    ldr q7, [x21, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    fmla v30.8h, v6.8h, v5.h[1]
    ldr q6, [x21, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    fmla v31.8h, v7.8h, v5.h[1]
    ldr q7, [x21, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    fmla v28.8h, v6.8h, v5.h[2]
    ldr q6, [x21, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    fmla v29.8h, v7.8h, v5.h[2]
    ldr q7, [x21, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    fmla v30.8h, v6.8h, v5.h[2]
    ldr q6, [x21, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    fmla v31.8h, v7.8h, v5.h[2]
    ldr q7, [x21, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    fmla v28.8h, v6.8h, v5.h[3]
    ldr q6, [x21, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    fmla v29.8h, v7.8h, v5.h[3]
    ldr q7, [x21, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    fmla v30.8h, v6.8h, v5.h[3]
    ldr q6, [x21, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    fmla v31.8h, v7.8h, v5.h[3]
    ldr q7, [x21, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    fmla v28.8h, v6.8h, v5.h[4]
    ldr q6, [x21, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    fmla v29.8h, v7.8h, v5.h[4]
    ldr q7, [x21, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    fmla v30.8h, v6.8h, v5.h[4]
    ldr q6, [x21, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    fmla v31.8h, v7.8h, v5.h[4]
    ldr q7, [x21, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    fmla v28.8h, v6.8h, v5.h[5]
    ldr q6, [x21, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    fmla v29.8h, v7.8h, v5.h[5]
    ldr q7, [x21, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    fmla v30.8h, v6.8h, v5.h[5]
    ldr q6, [x21, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    fmla v31.8h, v7.8h, v5.h[5]
    ldr q7, [x21, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    fmla v28.8h, v6.8h, v5.h[6]
    ldr q6, [x21, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    fmla v29.8h, v7.8h, v5.h[6]
    ldr q7, [x21, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    fmla v30.8h, v6.8h, v5.h[6]
    ldr q6, [x21, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    fmla v31.8h, v7.8h, v5.h[6]
    ldr q7, [x21, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    fmla v28.8h, v6.8h, v5.h[7]
    ldr q6, [x21, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    fmla v29.8h, v7.8h, v5.h[7]
    ldr q7, [x21, #0x1f0]
    add x21, x21, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v30.8h, v6.8h, v5.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
    fmla v27.8h, v7.8h, v4.h[7]
    fmla v31.8h, v7.8h, v5.h[7]
KAI_ASM_LABEL(label_320)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_322
KAI_ASM_LABEL(label_321)  // Height 6: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr h4, [x2], #0x2
    ldr h5, [x1], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    fmla v24.8h, v6.8h, v4.h[0]
    fmla v28.8h, v6.8h, v5.h[0]
    ldr q6, [x21, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    fmla v29.8h, v7.8h, v5.h[0]
    ldr q7, [x21, #0x30]
    add x21, x21, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v30.8h, v6.8h, v5.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    fmla v31.8h, v7.8h, v5.h[0]
    cbnz x7, label_321
KAI_ASM_LABEL(label_322)  // Height 6: Multiply loop: No odd multiplies
    tbz x25, #3, label_323
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_316
KAI_ASM_LABEL(label_323)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    add x5, x6, x1
    prfm pstl1keep, [x5, #0]
    add x4, x5, x1
    add x3, x4, x1
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    tbz x25, #1, label_324
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v1.8h }, [x2]
    ld1r { v0.8h }, [x1]
    fmin v8.8h, v8.8h, v1.8h
    fmin v9.8h, v9.8h, v1.8h
    fmin v10.8h, v10.8h, v1.8h
    fmin v11.8h, v11.8h, v1.8h
    fmin v12.8h, v12.8h, v1.8h
    fmin v13.8h, v13.8h, v1.8h
    fmin v14.8h, v14.8h, v1.8h
    fmin v15.8h, v15.8h, v1.8h
    fmin v16.8h, v16.8h, v1.8h
    fmin v17.8h, v17.8h, v1.8h
    fmin v18.8h, v18.8h, v1.8h
    fmin v19.8h, v19.8h, v1.8h
    fmin v20.8h, v20.8h, v1.8h
    fmin v21.8h, v21.8h, v1.8h
    fmin v22.8h, v22.8h, v1.8h
    fmin v23.8h, v23.8h, v1.8h
    fmin v24.8h, v24.8h, v1.8h
    fmin v25.8h, v25.8h, v1.8h
    fmin v26.8h, v26.8h, v1.8h
    fmin v27.8h, v27.8h, v1.8h
    fmin v28.8h, v28.8h, v1.8h
    fmin v29.8h, v29.8h, v1.8h
    fmin v30.8h, v30.8h, v1.8h
    fmin v31.8h, v31.8h, v1.8h
    fmax v8.8h, v8.8h, v0.8h
    fmax v9.8h, v9.8h, v0.8h
    fmax v10.8h, v10.8h, v0.8h
    fmax v11.8h, v11.8h, v0.8h
    fmax v12.8h, v12.8h, v0.8h
    fmax v13.8h, v13.8h, v0.8h
    fmax v14.8h, v14.8h, v0.8h
    fmax v15.8h, v15.8h, v0.8h
    fmax v16.8h, v16.8h, v0.8h
    fmax v17.8h, v17.8h, v0.8h
    fmax v18.8h, v18.8h, v0.8h
    fmax v19.8h, v19.8h, v0.8h
    fmax v20.8h, v20.8h, v0.8h
    fmax v21.8h, v21.8h, v0.8h
    fmax v22.8h, v22.8h, v0.8h
    fmax v23.8h, v23.8h, v0.8h
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
KAI_ASM_LABEL(label_324)  // Height 6: No activation
    cmp x22, #0x20
    bge label_341
    tbz x22, #4, label_332
    st1 { v8.8h }, [x23], #0x10
    st1 { v9.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v13.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v17.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    st1 { v21.8h }, [x5], #0x10
    st1 { v24.8h }, [x4], #0x10
    st1 { v25.8h }, [x4], #0x10
    st1 { v28.8h }, [x3], #0x10
    st1 { v29.8h }, [x3], #0x10
    tbz x22, #3, label_328
    st1 { v10.8h }, [x23], #0x10
    st1 { v14.8h }, [x7], #0x10
    st1 { v18.8h }, [x6], #0x10
    st1 { v22.8h }, [x5], #0x10
    st1 { v26.8h }, [x4], #0x10
    st1 { v30.8h }, [x3], #0x10
    tbz x22, #2, label_326
    str d11, [x23], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    str d27, [x4], #0x8
    str d31, [x3], #0x8
    tbz x22, #1, label_325
    st1 { v11.s }[2], [x23], #0x4
    st1 { v15.s }[2], [x7], #0x4
    st1 { v19.s }[2], [x6], #0x4
    st1 { v23.s }[2], [x5], #0x4
    st1 { v27.s }[2], [x4], #0x4
    st1 { v31.s }[2], [x3], #0x4
    tbz x22, #0, label_340
    st1 { v11.h }[6], [x23]
    st1 { v15.h }[6], [x7]
    st1 { v19.h }[6], [x6]
    st1 { v23.h }[6], [x5]
    st1 { v27.h }[6], [x4]
    st1 { v31.h }[6], [x3]
    b label_340
KAI_ASM_LABEL(label_325)  // Height 6: Partial direct writeback: partial_0_28
    tbz x22, #0, label_340
    st1 { v11.h }[4], [x23]
    st1 { v15.h }[4], [x7]
    st1 { v19.h }[4], [x6]
    st1 { v23.h }[4], [x5]
    st1 { v27.h }[4], [x4]
    st1 { v31.h }[4], [x3]
    b label_340
KAI_ASM_LABEL(label_326)  // Height 6: Partial direct writeback: partial_1_24
    tbz x22, #1, label_327
    str s11, [x23], #0x4
    str s15, [x7], #0x4
    str s19, [x6], #0x4
    str s23, [x5], #0x4
    str s27, [x4], #0x4
    str s31, [x3], #0x4
    tbz x22, #0, label_340
    st1 { v11.h }[2], [x23]
    st1 { v15.h }[2], [x7]
    st1 { v19.h }[2], [x6]
    st1 { v23.h }[2], [x5]
    st1 { v27.h }[2], [x4]
    st1 { v31.h }[2], [x3]
    b label_340
KAI_ASM_LABEL(label_327)  // Height 6: Partial direct writeback: partial_0_24
    tbz x22, #0, label_340
    str h11, [x23, #0]
    str h15, [x7, #0]
    str h19, [x6, #0]
    str h23, [x5, #0]
    str h27, [x4, #0]
    str h31, [x3, #0]
    b label_340
KAI_ASM_LABEL(label_328)  // Height 6: Partial direct writeback: partial_2_16
    tbz x22, #2, label_330
    str d10, [x23], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    str d26, [x4], #0x8
    str d30, [x3], #0x8
    tbz x22, #1, label_329
    st1 { v10.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x7], #0x4
    st1 { v18.s }[2], [x6], #0x4
    st1 { v22.s }[2], [x5], #0x4
    st1 { v26.s }[2], [x4], #0x4
    st1 { v30.s }[2], [x3], #0x4
    tbz x22, #0, label_340
    st1 { v10.h }[6], [x23]
    st1 { v14.h }[6], [x7]
    st1 { v18.h }[6], [x6]
    st1 { v22.h }[6], [x5]
    st1 { v26.h }[6], [x4]
    st1 { v30.h }[6], [x3]
    b label_340
KAI_ASM_LABEL(label_329)  // Height 6: Partial direct writeback: partial_0_20
    tbz x22, #0, label_340
    st1 { v10.h }[4], [x23]
    st1 { v14.h }[4], [x7]
    st1 { v18.h }[4], [x6]
    st1 { v22.h }[4], [x5]
    st1 { v26.h }[4], [x4]
    st1 { v30.h }[4], [x3]
    b label_340
KAI_ASM_LABEL(label_330)  // Height 6: Partial direct writeback: partial_1_16
    tbz x22, #1, label_331
    str s10, [x23], #0x4
    str s14, [x7], #0x4
    str s18, [x6], #0x4
    str s22, [x5], #0x4
    str s26, [x4], #0x4
    str s30, [x3], #0x4
    tbz x22, #0, label_340
    st1 { v10.h }[2], [x23]
    st1 { v14.h }[2], [x7]
    st1 { v18.h }[2], [x6]
    st1 { v22.h }[2], [x5]
    st1 { v26.h }[2], [x4]
    st1 { v30.h }[2], [x3]
    b label_340
KAI_ASM_LABEL(label_331)  // Height 6: Partial direct writeback: partial_0_16
    tbz x22, #0, label_340
    str h10, [x23, #0]
    str h14, [x7, #0]
    str h18, [x6, #0]
    str h22, [x5, #0]
    str h26, [x4, #0]
    str h30, [x3, #0]
    b label_340
KAI_ASM_LABEL(label_332)  // Height 6: Partial direct writeback: partial_3_0
    tbz x22, #3, label_336
    st1 { v8.8h }, [x23], #0x10
    st1 { v12.8h }, [x7], #0x10
    st1 { v16.8h }, [x6], #0x10
    st1 { v20.8h }, [x5], #0x10
    st1 { v24.8h }, [x4], #0x10
    st1 { v28.8h }, [x3], #0x10
    tbz x22, #2, label_334
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    str d29, [x3], #0x8
    tbz x22, #1, label_333
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    st1 { v25.s }[2], [x4], #0x4
    st1 { v29.s }[2], [x3], #0x4
    tbz x22, #0, label_340
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    st1 { v25.h }[6], [x4]
    st1 { v29.h }[6], [x3]
    b label_340
KAI_ASM_LABEL(label_333)  // Height 6: Partial direct writeback: partial_0_12
    tbz x22, #0, label_340
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    st1 { v25.h }[4], [x4]
    st1 { v29.h }[4], [x3]
    b label_340
KAI_ASM_LABEL(label_334)  // Height 6: Partial direct writeback: partial_1_8
    tbz x22, #1, label_335
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    str s25, [x4], #0x4
    str s29, [x3], #0x4
    tbz x22, #0, label_340
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    st1 { v25.h }[2], [x4]
    st1 { v29.h }[2], [x3]
    b label_340
KAI_ASM_LABEL(label_335)  // Height 6: Partial direct writeback: partial_0_8
    tbz x22, #0, label_340
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    str h25, [x4, #0]
    str h29, [x3, #0]
    b label_340
KAI_ASM_LABEL(label_336)  // Height 6: Partial direct writeback: partial_2_0
    tbz x22, #2, label_338
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    str d28, [x3], #0x8
    tbz x22, #1, label_337
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    st1 { v28.s }[2], [x3], #0x4
    tbz x22, #0, label_340
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    st1 { v24.h }[6], [x4]
    st1 { v28.h }[6], [x3]
    b label_340
KAI_ASM_LABEL(label_337)  // Height 6: Partial direct writeback: partial_0_4
    tbz x22, #0, label_340
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    st1 { v24.h }[4], [x4]
    st1 { v28.h }[4], [x3]
    b label_340
KAI_ASM_LABEL(label_338)  // Height 6: Partial direct writeback: partial_1_0
    tbz x22, #1, label_339
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    str s28, [x3], #0x4
    tbz x22, #0, label_340
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    st1 { v24.h }[2], [x4]
    st1 { v28.h }[2], [x3]
    b label_340
KAI_ASM_LABEL(label_339)  // Height 6: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
    str h24, [x4, #0]
    str h28, [x3, #0]
KAI_ASM_LABEL(label_340)  // Height 6: Partial direct writeback: Done
    b label_342
KAI_ASM_LABEL(label_341)  // Height 6: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    str q10, [x23, #0x20]
    str q11, [x23, #0x30]
    add x23, x23, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
    str q22, [x5, #0x20]
    str q23, [x5, #0x30]
    str q24, [x4, #0]
    str q25, [x4, #0x10]
    str q26, [x4, #0x20]
    str q27, [x4, #0x30]
    str q28, [x3, #0]
    str q29, [x3, #0x10]
    str q30, [x3, #0x20]
    str q31, [x3, #0x30]
KAI_ASM_LABEL(label_342)  // Height 6: Writeback done
    subs x22, x22, #0x20
    bgt label_287
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_343)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_fp16_mla_6x32)

    KAI_ASM_END
