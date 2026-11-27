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

    KAI_ASM_CODE(a64_ffhybrid_bf16fp32_mmla_6x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_ffhybrid_bf16fp32_mmla_6x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_ffhybrid_bf16fp32_mmla_6x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_ffhybrid_bf16fp32_mmla_6x16)
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
    sub SP, SP, #0x40
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
    bge label_231
    cmp x1, #0x4
    bgt label_185
    beq label_139
    cmp x1, #0x2
    bgt label_93
    beq label_47
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_3
    cmp x26, #0x8
    mov x20, x23
    bgt label_3
    cmp x26, #0x4
    mov x21, x23
    bgt label_3
    mov x22, x23
KAI_ASM_LABEL(label_3)  // Height 1: B setup done
    cbz x24, label_11
    cmp x26, #0x10
    bge label_9
    cbnz x11, label_8
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_5
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_4
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_7
KAI_ASM_LABEL(label_6)  // Height 1: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_6
KAI_ASM_LABEL(label_7)  // Height 1: Partial bias load: element loop skip
KAI_ASM_LABEL(label_8)  // Height 1: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_10
KAI_ASM_LABEL(label_9)  // Height 1: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_10)  // Height 1: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    b label_23
KAI_ASM_LABEL(label_11)  // Height 1: no bias
    tbz x9, #0, label_22
    cmp x26, #0x10
    bge label_20
    tbz x26, #3, label_15
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v10.4s }, [x27], #0x10
    tbz x26, #2, label_13
    ld1 { v11.4s }, [x27], #0x10
    tbz x26, #1, label_12
    ldr d16, [x27], #0x8
    mov x1, #0x38
    tbz x26, #0, label_19
    ld1 { v16.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_19
    ldr s16, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_1_8
    tbz x26, #1, label_14
    ldr d11, [x27], #0x8
    mov x1, #0x28
    tbz x26, #0, label_19
    ld1 { v11.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_19
    ldr s11, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_2_0
    tbz x26, #2, label_17
    ld1 { v9.4s }, [x27], #0x10
    tbz x26, #1, label_16
    ldr d10, [x27], #0x8
    mov x1, #0x18
    tbz x26, #0, label_19
    ld1 { v10.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_19
    ldr s10, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_1_0
    tbz x26, #1, label_18
    ldr d9, [x27], #0x8
    mov x1, #0x8
    tbz x26, #0, label_19
    ld1 { v9.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    mov x1, #0
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: Done
    sub x27, x27, x1
    b label_21
KAI_ASM_LABEL(label_20)  // Height 1: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
KAI_ASM_LABEL(label_21)  // Height 1: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    b label_23
KAI_ASM_LABEL(label_22)  // Height 1: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
KAI_ASM_LABEL(label_23)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_24
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    b label_26
KAI_ASM_LABEL(label_24)  // Height 1: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_25)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    cbnz x28, label_26
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_26)  // Height 1: input setup done
    cmp x7, #0x8
    blt label_29
    ldr q1, [x8, #0]
    ldr q7, [x23, #0]
    cmp x7, #0x10
    ldr q6, [x23, #0x10]
    blt label_28
KAI_ASM_LABEL(label_27)  // Height 1: Multiply loop: Main loop head
    sub x7, x7, #0x8
    cmp x7, #0x10
    add x8, x8, #0x10
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    ldr q6, [x23, #0x10]
    bge label_27
KAI_ASM_LABEL(label_28)  // Height 1: Multiply loop: Single iteration only
    sub x7, x7, #0x8
    add x8, x8, #0x10
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_34
    cmp x7, #0x4
    blt label_31
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr q6, [x23, #0]
    sub x7, x7, #0x4
    ldr q7, [x23, #0x10]
    cmp x7, #0x4
    add x23, x23, #0x20
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    bge label_30
KAI_ASM_LABEL(label_31)  // Height 1: Multiply loop: Skip odd blocks
    cbz x7, label_34
    tbz x7, #1, label_32
    ldr s1, [x8], #0x4
    tbz x7, #0, label_33
    ld1 { v1.h }[2], [x8]
    b label_33
KAI_ASM_LABEL(label_32)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
KAI_ASM_LABEL(label_33)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    add x23, x23, #0x20
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
KAI_ASM_LABEL(label_34)  // Height 1: Multiply loop: No odd multiplies
    tbz x9, #3, label_35
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_25
KAI_ASM_LABEL(label_35)  // Height 1: no string loop
    uzp1 v8.2d, v8.2d, v12.2d
    uzp1 v9.2d, v9.2d, v13.2d
    uzp1 v10.2d, v10.2d, v14.2d
    uzp1 v11.2d, v11.2d, v15.2d
    tbz x9, #1, label_36
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
KAI_ASM_LABEL(label_36)  // Height 1: No activation
    cmp x26, #0x10
    bge label_45
    tbz x26, #3, label_40
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    tbz x26, #2, label_38
    st1 { v10.4s }, [x27], #0x10
    tbz x26, #1, label_37
    str d11, [x27], #0x8
    tbz x26, #0, label_44
    st1 { v11.s }[2], [x27]
    b label_44
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_0_12
    tbz x26, #0, label_44
    str s11, [x27, #0]
    b label_44
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_1_8
    tbz x26, #1, label_39
    str d10, [x27], #0x8
    tbz x26, #0, label_44
    st1 { v10.s }[2], [x27]
    b label_44
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_0_8
    tbz x26, #0, label_44
    str s10, [x27, #0]
    b label_44
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_2_0
    tbz x26, #2, label_42
    st1 { v8.4s }, [x27], #0x10
    tbz x26, #1, label_41
    str d9, [x27], #0x8
    tbz x26, #0, label_44
    st1 { v9.s }[2], [x27]
    b label_44
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_0_4
    tbz x26, #0, label_44
    str s9, [x27, #0]
    b label_44
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_1_0
    tbz x26, #1, label_43
    str d8, [x27], #0x8
    tbz x26, #0, label_44
    st1 { v8.s }[2], [x27]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: Done
    b label_46
KAI_ASM_LABEL(label_45)  // Height 1: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
KAI_ASM_LABEL(label_46)  // Height 1: Writeback done
    subs x26, x26, #0x10
    bgt label_2
    b label_277
KAI_ASM_LABEL(label_47)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_48)  // Height 2: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_49
    cmp x26, #0x8
    mov x20, x23
    bgt label_49
    cmp x26, #0x4
    mov x21, x23
    bgt label_49
    mov x22, x23
KAI_ASM_LABEL(label_49)  // Height 2: B setup done
    cbz x24, label_57
    cmp x26, #0x10
    bge label_55
    cbnz x11, label_54
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_51
KAI_ASM_LABEL(label_50)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_50
KAI_ASM_LABEL(label_51)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_53
KAI_ASM_LABEL(label_52)  // Height 2: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_52
KAI_ASM_LABEL(label_53)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_54)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_56
KAI_ASM_LABEL(label_55)  // Height 2: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_56)  // Height 2: bias load done
    zip2 v12.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v13.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v14.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v15.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    b label_69
KAI_ASM_LABEL(label_57)  // Height 2: no bias
    tbz x9, #0, label_68
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    bge label_66
    tbz x26, #3, label_61
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    tbz x26, #2, label_59
    ld1 { v11.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    tbz x26, #1, label_58
    ldr d16, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    tbz x26, #0, label_65
    ld1 { v16.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    b label_65
KAI_ASM_LABEL(label_58)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_65
    ldr s16, [x27, #0]
    ldr s15, [x7, #0]
    b label_65
KAI_ASM_LABEL(label_59)  // Height 2: Partial accumulate: partial_1_8
    tbz x26, #1, label_60
    ldr d11, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    tbz x26, #0, label_65
    ld1 { v11.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    b label_65
KAI_ASM_LABEL(label_60)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_65
    ldr s11, [x27, #0]
    ldr s14, [x7, #0]
    b label_65
KAI_ASM_LABEL(label_61)  // Height 2: Partial accumulate: partial_2_0
    tbz x26, #2, label_63
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    tbz x26, #1, label_62
    ldr d10, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    tbz x26, #0, label_65
    ld1 { v10.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    b label_65
KAI_ASM_LABEL(label_62)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_65
    ldr s10, [x27, #0]
    ldr s13, [x7, #0]
    b label_65
KAI_ASM_LABEL(label_63)  // Height 2: Partial accumulate: partial_1_0
    tbz x26, #1, label_64
    ldr d9, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    tbz x26, #0, label_65
    ld1 { v9.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    b label_65
KAI_ASM_LABEL(label_64)  // Height 2: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
KAI_ASM_LABEL(label_65)  // Height 2: Partial accumulate: Done
    sub x27, x27, x1
    b label_67
KAI_ASM_LABEL(label_66)  // Height 2: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
KAI_ASM_LABEL(label_67)  // Height 2: MMLA fixup
    zip1 v8.2d, v9.2d, v12.2d
    zip2 v12.2d, v9.2d, v12.2d
    zip1 v9.2d, v10.2d, v13.2d
    zip2 v13.2d, v10.2d, v13.2d
    zip1 v10.2d, v11.2d, v14.2d
    zip2 v14.2d, v11.2d, v14.2d
    zip1 v11.2d, v16.2d, v15.2d
    zip2 v15.2d, v16.2d, v15.2d
    b label_69
KAI_ASM_LABEL(label_68)  // Height 2: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
KAI_ASM_LABEL(label_69)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_70
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    b label_72
KAI_ASM_LABEL(label_70)  // Height 2: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_71)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x28, label_72
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_72)  // Height 2: input setup done
    cmp x7, #0x8
    blt label_75
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x10
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    blt label_74
KAI_ASM_LABEL(label_73)  // Height 2: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    cmp x7, #0x10
    add x6, x6, #0x10
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    ldr q6, [x23, #0x10]
    bge label_73
KAI_ASM_LABEL(label_74)  // Height 2: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
KAI_ASM_LABEL(label_75)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_80
    cmp x7, #0x4
    blt label_77
KAI_ASM_LABEL(label_76)  // Height 2: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x4
    ldr q6, [x23, #0]
    ldr q7, [x23, #0x10]
    cmp x7, #0x4
    add x23, x23, #0x20
    trn1 v0.2d, v1.2d, v2.2d
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    bge label_76
KAI_ASM_LABEL(label_77)  // Height 2: Multiply loop: Skip odd blocks
    cbz x7, label_80
    tbz x7, #1, label_78
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    tbz x7, #0, label_79
    ld1 { v1.h }[2], [x8]
    ld1 { v2.h }[2], [x6]
    b label_79
KAI_ASM_LABEL(label_78)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
    ldr h2, [x6, #0]
KAI_ASM_LABEL(label_79)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
KAI_ASM_LABEL(label_80)  // Height 2: Multiply loop: No odd multiplies
    tbz x9, #3, label_81
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_71
KAI_ASM_LABEL(label_81)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x7, x27, x1
    tbz x9, #1, label_82
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v7.4s, v7.4s, v17.4s
    fmin v12.4s, v12.4s, v17.4s
    fmin v13.4s, v13.4s, v17.4s
    fmin v14.4s, v14.4s, v17.4s
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmax v7.4s, v7.4s, v16.4s
    fmax v12.4s, v12.4s, v16.4s
    fmax v13.4s, v13.4s, v16.4s
    fmax v14.4s, v14.4s, v16.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
KAI_ASM_LABEL(label_82)  // Height 2: No activation
    cmp x26, #0x10
    bge label_91
    tbz x26, #3, label_86
    st1 { v7.4s }, [x27], #0x10
    st1 { v12.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v9.4s }, [x7], #0x10
    tbz x26, #2, label_84
    st1 { v13.4s }, [x27], #0x10
    st1 { v10.4s }, [x7], #0x10
    tbz x26, #1, label_83
    str d14, [x27], #0x8
    str d11, [x7], #0x8
    tbz x26, #0, label_90
    st1 { v14.s }[2], [x27]
    st1 { v11.s }[2], [x7]
    b label_90
KAI_ASM_LABEL(label_83)  // Height 2: Partial direct writeback: partial_0_12
    tbz x26, #0, label_90
    str s14, [x27, #0]
    str s11, [x7, #0]
    b label_90
KAI_ASM_LABEL(label_84)  // Height 2: Partial direct writeback: partial_1_8
    tbz x26, #1, label_85
    str d13, [x27], #0x8
    str d10, [x7], #0x8
    tbz x26, #0, label_90
    st1 { v13.s }[2], [x27]
    st1 { v10.s }[2], [x7]
    b label_90
KAI_ASM_LABEL(label_85)  // Height 2: Partial direct writeback: partial_0_8
    tbz x26, #0, label_90
    str s13, [x27, #0]
    str s10, [x7, #0]
    b label_90
KAI_ASM_LABEL(label_86)  // Height 2: Partial direct writeback: partial_2_0
    tbz x26, #2, label_88
    st1 { v7.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    tbz x26, #1, label_87
    str d12, [x27], #0x8
    str d9, [x7], #0x8
    tbz x26, #0, label_90
    st1 { v12.s }[2], [x27]
    st1 { v9.s }[2], [x7]
    b label_90
KAI_ASM_LABEL(label_87)  // Height 2: Partial direct writeback: partial_0_4
    tbz x26, #0, label_90
    str s12, [x27, #0]
    str s9, [x7, #0]
    b label_90
KAI_ASM_LABEL(label_88)  // Height 2: Partial direct writeback: partial_1_0
    tbz x26, #1, label_89
    str d7, [x27], #0x8
    str d8, [x7], #0x8
    tbz x26, #0, label_90
    st1 { v7.s }[2], [x27]
    st1 { v8.s }[2], [x7]
    b label_90
KAI_ASM_LABEL(label_89)  // Height 2: Partial direct writeback: partial_0_0
    str s7, [x27, #0]
    str s8, [x7, #0]
KAI_ASM_LABEL(label_90)  // Height 2: Partial direct writeback: Done
    b label_92
KAI_ASM_LABEL(label_91)  // Height 2: Full writeback
    str q7, [x27, #0]
    str q12, [x27, #0x10]
    str q13, [x27, #0x20]
    str q14, [x27, #0x30]
    add x27, x27, #0x40
    str q8, [x7, #0]
    str q9, [x7, #0x10]
    str q10, [x7, #0x20]
    str q11, [x7, #0x30]
KAI_ASM_LABEL(label_92)  // Height 2: Writeback done
    subs x26, x26, #0x10
    bgt label_48
    b label_277
KAI_ASM_LABEL(label_93)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_94)  // Height 3: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_95
    cmp x26, #0x8
    mov x20, x23
    bgt label_95
    cmp x26, #0x4
    mov x21, x23
    bgt label_95
    mov x22, x23
KAI_ASM_LABEL(label_95)  // Height 3: B setup done
    cbz x24, label_103
    cmp x26, #0x10
    bge label_101
    cbnz x11, label_100
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_97
KAI_ASM_LABEL(label_96)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_96
KAI_ASM_LABEL(label_97)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_99
KAI_ASM_LABEL(label_98)  // Height 3: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_98
KAI_ASM_LABEL(label_99)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_100)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_102
KAI_ASM_LABEL(label_101)  // Height 3: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_102)  // Height 3: bias load done
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
    b label_115
KAI_ASM_LABEL(label_103)  // Height 3: no bias
    tbz x9, #0, label_114
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    bge label_112
    tbz x26, #3, label_107
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    tbz x26, #2, label_105
    ld1 { v11.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v19.4s }, [x6], #0x10
    tbz x26, #1, label_104
    ldr d16, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d24, [x6], #0x8
    tbz x26, #0, label_111
    ld1 { v16.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v24.s }[2], [x6]
    b label_111
KAI_ASM_LABEL(label_104)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_111
    ldr s16, [x27, #0]
    ldr s15, [x7, #0]
    ldr s24, [x6, #0]
    b label_111
KAI_ASM_LABEL(label_105)  // Height 3: Partial accumulate: partial_1_8
    tbz x26, #1, label_106
    ldr d11, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d19, [x6], #0x8
    tbz x26, #0, label_111
    ld1 { v11.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    b label_111
KAI_ASM_LABEL(label_106)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_111
    ldr s11, [x27, #0]
    ldr s14, [x7, #0]
    ldr s19, [x6, #0]
    b label_111
KAI_ASM_LABEL(label_107)  // Height 3: Partial accumulate: partial_2_0
    tbz x26, #2, label_109
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    tbz x26, #1, label_108
    ldr d10, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d18, [x6], #0x8
    tbz x26, #0, label_111
    ld1 { v10.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    b label_111
KAI_ASM_LABEL(label_108)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_111
    ldr s10, [x27, #0]
    ldr s13, [x7, #0]
    ldr s18, [x6, #0]
    b label_111
KAI_ASM_LABEL(label_109)  // Height 3: Partial accumulate: partial_1_0
    tbz x26, #1, label_110
    ldr d9, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d17, [x6], #0x8
    tbz x26, #0, label_111
    ld1 { v9.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    b label_111
KAI_ASM_LABEL(label_110)  // Height 3: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s17, [x6, #0]
KAI_ASM_LABEL(label_111)  // Height 3: Partial accumulate: Done
    sub x27, x27, x1
    b label_113
KAI_ASM_LABEL(label_112)  // Height 3: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q17, [x6, #0]
    ldr q18, [x6, #0x10]
    ldr q19, [x6, #0x20]
    ldr q24, [x6, #0x30]
KAI_ASM_LABEL(label_113)  // Height 3: MMLA fixup
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
    b label_115
KAI_ASM_LABEL(label_114)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_115)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_116
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_118
KAI_ASM_LABEL(label_116)  // Height 3: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_117)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x28, label_118
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_118)  // Height 3: input setup done
    cmp x7, #0x8
    blt label_121
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x10
    ldr q3, [x5, #0]
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    blt label_120
KAI_ASM_LABEL(label_119)  // Height 3: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    cmp x7, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x23, #0x30]
    ldr q2, [x6, #0]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    ldr q3, [x5, #0]
    ldr q6, [x23, #0x10]
    bge label_119
KAI_ASM_LABEL(label_120)  // Height 3: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    trn2 v3.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
KAI_ASM_LABEL(label_121)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_126
    cmp x7, #0x4
    blt label_123
KAI_ASM_LABEL(label_122)  // Height 3: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x4
    ldr d3, [x5], #0x8
    ldr q6, [x23, #0]
    cmp x7, #0x4
    ldr q7, [x23, #0x10]
    add x23, x23, #0x20
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec50)  // bfmmla v16.4s, v2.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec54)  // bfmmla v20.4s, v2.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec51)  // bfmmla v17.4s, v2.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec55)  // bfmmla v21.4s, v2.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec52)  // bfmmla v18.4s, v2.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec56)  // bfmmla v22.4s, v2.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec53)  // bfmmla v19.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    bge label_122
KAI_ASM_LABEL(label_123)  // Height 3: Multiply loop: Skip odd blocks
    cbz x7, label_126
    tbz x7, #1, label_124
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    tbz x7, #0, label_125
    ld1 { v1.h }[2], [x8]
    ld1 { v2.h }[2], [x6]
    ld1 { v3.h }[2], [x5]
    b label_125
KAI_ASM_LABEL(label_124)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
    ldr h2, [x6, #0]
    ldr h3, [x5, #0]
KAI_ASM_LABEL(label_125)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    add x23, x23, #0x20
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
KAI_ASM_LABEL(label_126)  // Height 3: Multiply loop: No odd multiplies
    tbz x9, #3, label_127
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_117
KAI_ASM_LABEL(label_127)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x7, x27, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x6, x7, x1
    uzp1 v16.2d, v16.2d, v20.2d
    uzp1 v17.2d, v17.2d, v21.2d
    uzp1 v18.2d, v18.2d, v22.2d
    uzp1 v19.2d, v19.2d, v23.2d
    tbz x9, #1, label_128
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v21.4s }, [x2]
    ld1r { v20.4s }, [x1]
    fmin v7.4s, v7.4s, v21.4s
    fmin v12.4s, v12.4s, v21.4s
    fmin v13.4s, v13.4s, v21.4s
    fmin v14.4s, v14.4s, v21.4s
    fmin v8.4s, v8.4s, v21.4s
    fmin v9.4s, v9.4s, v21.4s
    fmin v10.4s, v10.4s, v21.4s
    fmin v11.4s, v11.4s, v21.4s
    fmin v16.4s, v16.4s, v21.4s
    fmin v17.4s, v17.4s, v21.4s
    fmin v18.4s, v18.4s, v21.4s
    fmin v19.4s, v19.4s, v21.4s
    fmax v7.4s, v7.4s, v20.4s
    fmax v12.4s, v12.4s, v20.4s
    fmax v13.4s, v13.4s, v20.4s
    fmax v14.4s, v14.4s, v20.4s
    fmax v8.4s, v8.4s, v20.4s
    fmax v9.4s, v9.4s, v20.4s
    fmax v10.4s, v10.4s, v20.4s
    fmax v11.4s, v11.4s, v20.4s
    fmax v16.4s, v16.4s, v20.4s
    fmax v17.4s, v17.4s, v20.4s
    fmax v18.4s, v18.4s, v20.4s
    fmax v19.4s, v19.4s, v20.4s
KAI_ASM_LABEL(label_128)  // Height 3: No activation
    cmp x26, #0x10
    bge label_137
    tbz x26, #3, label_132
    st1 { v7.4s }, [x27], #0x10
    st1 { v12.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v9.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v17.4s }, [x6], #0x10
    tbz x26, #2, label_130
    st1 { v13.4s }, [x27], #0x10
    st1 { v10.4s }, [x7], #0x10
    st1 { v18.4s }, [x6], #0x10
    tbz x26, #1, label_129
    str d14, [x27], #0x8
    str d11, [x7], #0x8
    str d19, [x6], #0x8
    tbz x26, #0, label_136
    st1 { v14.s }[2], [x27]
    st1 { v11.s }[2], [x7]
    st1 { v19.s }[2], [x6]
    b label_136
KAI_ASM_LABEL(label_129)  // Height 3: Partial direct writeback: partial_0_12
    tbz x26, #0, label_136
    str s14, [x27, #0]
    str s11, [x7, #0]
    str s19, [x6, #0]
    b label_136
KAI_ASM_LABEL(label_130)  // Height 3: Partial direct writeback: partial_1_8
    tbz x26, #1, label_131
    str d13, [x27], #0x8
    str d10, [x7], #0x8
    str d18, [x6], #0x8
    tbz x26, #0, label_136
    st1 { v13.s }[2], [x27]
    st1 { v10.s }[2], [x7]
    st1 { v18.s }[2], [x6]
    b label_136
KAI_ASM_LABEL(label_131)  // Height 3: Partial direct writeback: partial_0_8
    tbz x26, #0, label_136
    str s13, [x27, #0]
    str s10, [x7, #0]
    str s18, [x6, #0]
    b label_136
KAI_ASM_LABEL(label_132)  // Height 3: Partial direct writeback: partial_2_0
    tbz x26, #2, label_134
    st1 { v7.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    tbz x26, #1, label_133
    str d12, [x27], #0x8
    str d9, [x7], #0x8
    str d17, [x6], #0x8
    tbz x26, #0, label_136
    st1 { v12.s }[2], [x27]
    st1 { v9.s }[2], [x7]
    st1 { v17.s }[2], [x6]
    b label_136
KAI_ASM_LABEL(label_133)  // Height 3: Partial direct writeback: partial_0_4
    tbz x26, #0, label_136
    str s12, [x27, #0]
    str s9, [x7, #0]
    str s17, [x6, #0]
    b label_136
KAI_ASM_LABEL(label_134)  // Height 3: Partial direct writeback: partial_1_0
    tbz x26, #1, label_135
    str d7, [x27], #0x8
    str d8, [x7], #0x8
    str d16, [x6], #0x8
    tbz x26, #0, label_136
    st1 { v7.s }[2], [x27]
    st1 { v8.s }[2], [x7]
    st1 { v16.s }[2], [x6]
    b label_136
KAI_ASM_LABEL(label_135)  // Height 3: Partial direct writeback: partial_0_0
    str s7, [x27, #0]
    str s8, [x7, #0]
    str s16, [x6, #0]
KAI_ASM_LABEL(label_136)  // Height 3: Partial direct writeback: Done
    b label_138
KAI_ASM_LABEL(label_137)  // Height 3: Full writeback
    str q7, [x27, #0]
    str q12, [x27, #0x10]
    str q13, [x27, #0x20]
    str q14, [x27, #0x30]
    add x27, x27, #0x40
    str q8, [x7, #0]
    str q9, [x7, #0x10]
    str q10, [x7, #0x20]
    str q11, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
KAI_ASM_LABEL(label_138)  // Height 3: Writeback done
    subs x26, x26, #0x10
    bgt label_94
    b label_277
KAI_ASM_LABEL(label_139)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_140)  // Height 4: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_141
    cmp x26, #0x8
    mov x20, x23
    bgt label_141
    cmp x26, #0x4
    mov x21, x23
    bgt label_141
    mov x22, x23
KAI_ASM_LABEL(label_141)  // Height 4: B setup done
    cbz x24, label_149
    cmp x26, #0x10
    bge label_147
    cbnz x11, label_146
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_143
KAI_ASM_LABEL(label_142)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_142
KAI_ASM_LABEL(label_143)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_145
KAI_ASM_LABEL(label_144)  // Height 4: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_144
KAI_ASM_LABEL(label_145)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_146)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_148
KAI_ASM_LABEL(label_147)  // Height 4: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_148)  // Height 4: bias load done
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
    b label_161
KAI_ASM_LABEL(label_149)  // Height 4: no bias
    tbz x9, #0, label_160
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    bge label_158
    tbz x26, #3, label_153
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    tbz x26, #2, label_151
    ld1 { v11.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v19.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    tbz x26, #1, label_150
    ldr d16, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d24, [x6], #0x8
    ldr d23, [x5], #0x8
    tbz x26, #0, label_157
    ld1 { v16.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v24.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    b label_157
KAI_ASM_LABEL(label_150)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_157
    ldr s16, [x27, #0]
    ldr s15, [x7, #0]
    ldr s24, [x6, #0]
    ldr s23, [x5, #0]
    b label_157
KAI_ASM_LABEL(label_151)  // Height 4: Partial accumulate: partial_1_8
    tbz x26, #1, label_152
    ldr d11, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d19, [x6], #0x8
    ldr d22, [x5], #0x8
    tbz x26, #0, label_157
    ld1 { v11.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    b label_157
KAI_ASM_LABEL(label_152)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_157
    ldr s11, [x27, #0]
    ldr s14, [x7, #0]
    ldr s19, [x6, #0]
    ldr s22, [x5, #0]
    b label_157
KAI_ASM_LABEL(label_153)  // Height 4: Partial accumulate: partial_2_0
    tbz x26, #2, label_155
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    tbz x26, #1, label_154
    ldr d10, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d18, [x6], #0x8
    ldr d21, [x5], #0x8
    tbz x26, #0, label_157
    ld1 { v10.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    b label_157
KAI_ASM_LABEL(label_154)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_157
    ldr s10, [x27, #0]
    ldr s13, [x7, #0]
    ldr s18, [x6, #0]
    ldr s21, [x5, #0]
    b label_157
KAI_ASM_LABEL(label_155)  // Height 4: Partial accumulate: partial_1_0
    tbz x26, #1, label_156
    ldr d9, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d17, [x6], #0x8
    ldr d20, [x5], #0x8
    tbz x26, #0, label_157
    ld1 { v9.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    b label_157
KAI_ASM_LABEL(label_156)  // Height 4: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s17, [x6, #0]
    ldr s20, [x5, #0]
KAI_ASM_LABEL(label_157)  // Height 4: Partial accumulate: Done
    sub x27, x27, x1
    b label_159
KAI_ASM_LABEL(label_158)  // Height 4: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q17, [x6, #0]
    ldr q18, [x6, #0x10]
    ldr q19, [x6, #0x20]
    ldr q24, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
KAI_ASM_LABEL(label_159)  // Height 4: MMLA fixup
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
    b label_161
KAI_ASM_LABEL(label_160)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_161)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_162
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_164
KAI_ASM_LABEL(label_162)  // Height 4: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_163)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x28, label_164
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_164)  // Height 4: input setup done
    cmp x7, #0x8
    blt label_167
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x10
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    blt label_166
KAI_ASM_LABEL(label_165)  // Height 4: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    cmp x7, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    ldr q4, [x4, #0]
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x23, #0x30]
    ldr q2, [x6, #0]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    ldr q3, [x5, #0]
    ldr q6, [x23, #0x10]
    bge label_165
KAI_ASM_LABEL(label_166)  // Height 4: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
KAI_ASM_LABEL(label_167)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_172
    cmp x7, #0x4
    blt label_169
KAI_ASM_LABEL(label_168)  // Height 4: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x4
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x4
    ldr q6, [x23, #0]
    ldr q7, [x23, #0x10]
    add x23, x23, #0x20
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec50)  // bfmmla v16.4s, v2.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec54)  // bfmmla v20.4s, v2.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec51)  // bfmmla v17.4s, v2.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec55)  // bfmmla v21.4s, v2.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec52)  // bfmmla v18.4s, v2.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec56)  // bfmmla v22.4s, v2.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec53)  // bfmmla v19.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    bge label_168
KAI_ASM_LABEL(label_169)  // Height 4: Multiply loop: Skip odd blocks
    cbz x7, label_172
    tbz x7, #1, label_170
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    tbz x7, #0, label_171
    ld1 { v1.h }[2], [x8]
    ld1 { v2.h }[2], [x6]
    ld1 { v3.h }[2], [x5]
    ld1 { v4.h }[2], [x4]
    b label_171
KAI_ASM_LABEL(label_170)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
    ldr h2, [x6, #0]
    ldr h3, [x5, #0]
    ldr h4, [x4, #0]
KAI_ASM_LABEL(label_171)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    ldr q6, [x23, #0x10]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
KAI_ASM_LABEL(label_172)  // Height 4: Multiply loop: No odd multiplies
    tbz x9, #3, label_173
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_163
KAI_ASM_LABEL(label_173)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x7, x27, x1
    add x6, x7, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x5, x6, x1
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    tbz x9, #1, label_174
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v24.4s }, [x2]
    ld1r { v23.4s }, [x1]
    fmin v7.4s, v7.4s, v24.4s
    fmin v12.4s, v12.4s, v24.4s
    fmin v13.4s, v13.4s, v24.4s
    fmin v14.4s, v14.4s, v24.4s
    fmin v8.4s, v8.4s, v24.4s
    fmin v9.4s, v9.4s, v24.4s
    fmin v10.4s, v10.4s, v24.4s
    fmin v11.4s, v11.4s, v24.4s
    fmin v15.4s, v15.4s, v24.4s
    fmin v20.4s, v20.4s, v24.4s
    fmin v21.4s, v21.4s, v24.4s
    fmin v22.4s, v22.4s, v24.4s
    fmin v16.4s, v16.4s, v24.4s
    fmin v17.4s, v17.4s, v24.4s
    fmin v18.4s, v18.4s, v24.4s
    fmin v19.4s, v19.4s, v24.4s
    fmax v7.4s, v7.4s, v23.4s
    fmax v12.4s, v12.4s, v23.4s
    fmax v13.4s, v13.4s, v23.4s
    fmax v14.4s, v14.4s, v23.4s
    fmax v8.4s, v8.4s, v23.4s
    fmax v9.4s, v9.4s, v23.4s
    fmax v10.4s, v10.4s, v23.4s
    fmax v11.4s, v11.4s, v23.4s
    fmax v15.4s, v15.4s, v23.4s
    fmax v20.4s, v20.4s, v23.4s
    fmax v21.4s, v21.4s, v23.4s
    fmax v22.4s, v22.4s, v23.4s
    fmax v16.4s, v16.4s, v23.4s
    fmax v17.4s, v17.4s, v23.4s
    fmax v18.4s, v18.4s, v23.4s
    fmax v19.4s, v19.4s, v23.4s
KAI_ASM_LABEL(label_174)  // Height 4: No activation
    cmp x26, #0x10
    bge label_183
    tbz x26, #3, label_178
    st1 { v7.4s }, [x27], #0x10
    st1 { v12.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v9.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v20.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    tbz x26, #2, label_176
    st1 { v13.4s }, [x27], #0x10
    st1 { v10.4s }, [x7], #0x10
    st1 { v21.4s }, [x6], #0x10
    st1 { v18.4s }, [x5], #0x10
    tbz x26, #1, label_175
    str d14, [x27], #0x8
    str d11, [x7], #0x8
    str d22, [x6], #0x8
    str d19, [x5], #0x8
    tbz x26, #0, label_182
    st1 { v14.s }[2], [x27]
    st1 { v11.s }[2], [x7]
    st1 { v22.s }[2], [x6]
    st1 { v19.s }[2], [x5]
    b label_182
KAI_ASM_LABEL(label_175)  // Height 4: Partial direct writeback: partial_0_12
    tbz x26, #0, label_182
    str s14, [x27, #0]
    str s11, [x7, #0]
    str s22, [x6, #0]
    str s19, [x5, #0]
    b label_182
KAI_ASM_LABEL(label_176)  // Height 4: Partial direct writeback: partial_1_8
    tbz x26, #1, label_177
    str d13, [x27], #0x8
    str d10, [x7], #0x8
    str d21, [x6], #0x8
    str d18, [x5], #0x8
    tbz x26, #0, label_182
    st1 { v13.s }[2], [x27]
    st1 { v10.s }[2], [x7]
    st1 { v21.s }[2], [x6]
    st1 { v18.s }[2], [x5]
    b label_182
KAI_ASM_LABEL(label_177)  // Height 4: Partial direct writeback: partial_0_8
    tbz x26, #0, label_182
    str s13, [x27, #0]
    str s10, [x7, #0]
    str s21, [x6, #0]
    str s18, [x5, #0]
    b label_182
KAI_ASM_LABEL(label_178)  // Height 4: Partial direct writeback: partial_2_0
    tbz x26, #2, label_180
    st1 { v7.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    tbz x26, #1, label_179
    str d12, [x27], #0x8
    str d9, [x7], #0x8
    str d20, [x6], #0x8
    str d17, [x5], #0x8
    tbz x26, #0, label_182
    st1 { v12.s }[2], [x27]
    st1 { v9.s }[2], [x7]
    st1 { v20.s }[2], [x6]
    st1 { v17.s }[2], [x5]
    b label_182
KAI_ASM_LABEL(label_179)  // Height 4: Partial direct writeback: partial_0_4
    tbz x26, #0, label_182
    str s12, [x27, #0]
    str s9, [x7, #0]
    str s20, [x6, #0]
    str s17, [x5, #0]
    b label_182
KAI_ASM_LABEL(label_180)  // Height 4: Partial direct writeback: partial_1_0
    tbz x26, #1, label_181
    str d7, [x27], #0x8
    str d8, [x7], #0x8
    str d15, [x6], #0x8
    str d16, [x5], #0x8
    tbz x26, #0, label_182
    st1 { v7.s }[2], [x27]
    st1 { v8.s }[2], [x7]
    st1 { v15.s }[2], [x6]
    st1 { v16.s }[2], [x5]
    b label_182
KAI_ASM_LABEL(label_181)  // Height 4: Partial direct writeback: partial_0_0
    str s7, [x27, #0]
    str s8, [x7, #0]
    str s15, [x6, #0]
    str s16, [x5, #0]
KAI_ASM_LABEL(label_182)  // Height 4: Partial direct writeback: Done
    b label_184
KAI_ASM_LABEL(label_183)  // Height 4: Full writeback
    str q7, [x27, #0]
    str q12, [x27, #0x10]
    str q13, [x27, #0x20]
    str q14, [x27, #0x30]
    add x27, x27, #0x40
    str q8, [x7, #0]
    str q9, [x7, #0x10]
    str q10, [x7, #0x20]
    str q11, [x7, #0x30]
    str q15, [x6, #0]
    str q20, [x6, #0x10]
    str q21, [x6, #0x20]
    str q22, [x6, #0x30]
    str q16, [x5, #0]
    str q17, [x5, #0x10]
    str q18, [x5, #0x20]
    str q19, [x5, #0x30]
KAI_ASM_LABEL(label_184)  // Height 4: Writeback done
    subs x26, x26, #0x10
    bgt label_140
    b label_277
KAI_ASM_LABEL(label_185)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_186)  // Height 5: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_187
    cmp x26, #0x8
    mov x20, x23
    bgt label_187
    cmp x26, #0x4
    mov x21, x23
    bgt label_187
    mov x22, x23
KAI_ASM_LABEL(label_187)  // Height 5: B setup done
    cbz x24, label_195
    cmp x26, #0x10
    bge label_193
    cbnz x11, label_192
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_189
KAI_ASM_LABEL(label_188)  // Height 5: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_188
KAI_ASM_LABEL(label_189)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_191
KAI_ASM_LABEL(label_190)  // Height 5: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_190
KAI_ASM_LABEL(label_191)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_192)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_194
KAI_ASM_LABEL(label_193)  // Height 5: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_194)  // Height 5: bias load done
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
    b label_207
KAI_ASM_LABEL(label_195)  // Height 5: no bias
    tbz x9, #0, label_206
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_204
    tbz x26, #3, label_199
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    ld1 { v26.4s }, [x4], #0x10
    tbz x26, #2, label_197
    ld1 { v11.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v19.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    ld1 { v27.4s }, [x4], #0x10
    tbz x26, #1, label_196
    ldr d16, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d24, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d6, [x4], #0x8
    tbz x26, #0, label_203
    ld1 { v16.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v24.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    ld1 { v6.s }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_196)  // Height 5: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_203
    ldr s16, [x27, #0]
    ldr s15, [x7, #0]
    ldr s24, [x6, #0]
    ldr s23, [x5, #0]
    ldr s6, [x4, #0]
    b label_203
KAI_ASM_LABEL(label_197)  // Height 5: Partial accumulate: partial_1_8
    tbz x26, #1, label_198
    ldr d11, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d19, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d27, [x4], #0x8
    tbz x26, #0, label_203
    ld1 { v11.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    ld1 { v27.s }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_198)  // Height 5: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_203
    ldr s11, [x27, #0]
    ldr s14, [x7, #0]
    ldr s19, [x6, #0]
    ldr s22, [x5, #0]
    ldr s27, [x4, #0]
    b label_203
KAI_ASM_LABEL(label_199)  // Height 5: Partial accumulate: partial_2_0
    tbz x26, #2, label_201
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    tbz x26, #1, label_200
    ldr d10, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d18, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d26, [x4], #0x8
    tbz x26, #0, label_203
    ld1 { v10.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    ld1 { v26.s }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_200)  // Height 5: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_203
    ldr s10, [x27, #0]
    ldr s13, [x7, #0]
    ldr s18, [x6, #0]
    ldr s21, [x5, #0]
    ldr s26, [x4, #0]
    b label_203
KAI_ASM_LABEL(label_201)  // Height 5: Partial accumulate: partial_1_0
    tbz x26, #1, label_202
    ldr d9, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d17, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d25, [x4], #0x8
    tbz x26, #0, label_203
    ld1 { v9.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_202)  // Height 5: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s17, [x6, #0]
    ldr s20, [x5, #0]
    ldr s25, [x4, #0]
KAI_ASM_LABEL(label_203)  // Height 5: Partial accumulate: Done
    sub x27, x27, x1
    b label_205
KAI_ASM_LABEL(label_204)  // Height 5: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q17, [x6, #0]
    ldr q18, [x6, #0x10]
    ldr q19, [x6, #0x20]
    ldr q24, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
    ldr q25, [x4, #0]
    ldr q26, [x4, #0x10]
    ldr q27, [x4, #0x20]
    ldr q6, [x4, #0x30]
KAI_ASM_LABEL(label_205)  // Height 5: MMLA fixup
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
    b label_207
KAI_ASM_LABEL(label_206)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_207)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_208
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_210
KAI_ASM_LABEL(label_208)  // Height 5: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_209)  // Height 5: String loop
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
    cbnz x28, label_210
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_210)  // Height 5: input setup done
    cmp x7, #0x8
    blt label_213
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x10
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q7, [x23, #0]
    blt label_212
KAI_ASM_LABEL(label_211)  // Height 5: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x8, x8, #0x10
    cmp x7, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    add x4, x4, #0x10
    add x2, x2, #0x10
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
    ldr q6, [x23, #0x30]
    ldr q4, [x4, #0]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb8)  // bfmmla v24.4s, v5.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbc)  // bfmmla v28.4s, v5.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb9)  // bfmmla v25.4s, v5.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbd)  // bfmmla v29.4s, v5.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecba)  // bfmmla v26.4s, v5.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbe)  // bfmmla v30.4s, v5.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecbb)  // bfmmla v27.4s, v5.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x6e46ecbf)  // bfmmla v31.4s, v5.8h, v6.8h
    ldr q5, [x2, #0]
    bge label_211
KAI_ASM_LABEL(label_212)  // Height 5: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x8, x8, #0x10
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    add x2, x2, #0x10
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb8)  // bfmmla v24.4s, v5.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbc)  // bfmmla v28.4s, v5.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb9)  // bfmmla v25.4s, v5.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbd)  // bfmmla v29.4s, v5.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecba)  // bfmmla v26.4s, v5.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbe)  // bfmmla v30.4s, v5.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecbb)  // bfmmla v27.4s, v5.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbf)  // bfmmla v31.4s, v5.8h, v6.8h
KAI_ASM_LABEL(label_213)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_218
    cmp x7, #0x4
    blt label_215
KAI_ASM_LABEL(label_214)  // Height 5: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x4
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x4
    ldr d5, [x2], #0x8
    ldr q6, [x23, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v7.2d
    ldr q7, [x23, #0x10]
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec50)  // bfmmla v16.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec98)  // bfmmla v24.4s, v4.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec54)  // bfmmla v20.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9c)  // bfmmla v28.4s, v4.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec51)  // bfmmla v17.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec99)  // bfmmla v25.4s, v4.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec55)  // bfmmla v21.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9d)  // bfmmla v29.4s, v4.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec52)  // bfmmla v18.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9a)  // bfmmla v26.4s, v4.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec56)  // bfmmla v22.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9e)  // bfmmla v30.4s, v4.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec53)  // bfmmla v19.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9b)  // bfmmla v27.4s, v4.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9f)  // bfmmla v31.4s, v4.8h, v7.8h
    bge label_214
KAI_ASM_LABEL(label_215)  // Height 5: Multiply loop: Skip odd blocks
    cbz x7, label_218
    tbz x7, #1, label_216
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    tbz x7, #0, label_217
    ld1 { v1.h }[2], [x8]
    ld1 { v2.h }[2], [x6]
    ld1 { v3.h }[2], [x5]
    ld1 { v4.h }[2], [x4]
    ld1 { v5.h }[2], [x2]
    b label_217
KAI_ASM_LABEL(label_216)  // Height 5: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
    ldr h2, [x6, #0]
    ldr h3, [x5, #0]
    ldr h4, [x4, #0]
    ldr h5, [x2, #0]
KAI_ASM_LABEL(label_217)  // Height 5: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
KAI_ASM_LABEL(label_218)  // Height 5: Multiply loop: No odd multiplies
    tbz x9, #3, label_219
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_209
KAI_ASM_LABEL(label_219)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    add x4, x5, x1
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
    uzp1 v22.2d, v19.2d, v23.2d
    uzp2 v19.2d, v19.2d, v23.2d
    uzp1 v24.2d, v24.2d, v28.2d
    uzp1 v25.2d, v25.2d, v29.2d
    uzp1 v26.2d, v26.2d, v30.2d
    uzp1 v27.2d, v27.2d, v31.2d
    tbz x9, #1, label_220
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v28.4s }, [x2]
    ld1r { v23.4s }, [x1]
    fmin v7.4s, v7.4s, v28.4s
    fmin v12.4s, v12.4s, v28.4s
    fmin v13.4s, v13.4s, v28.4s
    fmin v14.4s, v14.4s, v28.4s
    fmin v8.4s, v8.4s, v28.4s
    fmin v9.4s, v9.4s, v28.4s
    fmin v10.4s, v10.4s, v28.4s
    fmin v11.4s, v11.4s, v28.4s
    fmin v15.4s, v15.4s, v28.4s
    fmin v20.4s, v20.4s, v28.4s
    fmin v21.4s, v21.4s, v28.4s
    fmin v22.4s, v22.4s, v28.4s
    fmin v16.4s, v16.4s, v28.4s
    fmin v17.4s, v17.4s, v28.4s
    fmin v18.4s, v18.4s, v28.4s
    fmin v19.4s, v19.4s, v28.4s
    fmin v24.4s, v24.4s, v28.4s
    fmin v25.4s, v25.4s, v28.4s
    fmin v26.4s, v26.4s, v28.4s
    fmin v27.4s, v27.4s, v28.4s
    fmax v7.4s, v7.4s, v23.4s
    fmax v12.4s, v12.4s, v23.4s
    fmax v13.4s, v13.4s, v23.4s
    fmax v14.4s, v14.4s, v23.4s
    fmax v8.4s, v8.4s, v23.4s
    fmax v9.4s, v9.4s, v23.4s
    fmax v10.4s, v10.4s, v23.4s
    fmax v11.4s, v11.4s, v23.4s
    fmax v15.4s, v15.4s, v23.4s
    fmax v20.4s, v20.4s, v23.4s
    fmax v21.4s, v21.4s, v23.4s
    fmax v22.4s, v22.4s, v23.4s
    fmax v16.4s, v16.4s, v23.4s
    fmax v17.4s, v17.4s, v23.4s
    fmax v18.4s, v18.4s, v23.4s
    fmax v19.4s, v19.4s, v23.4s
    fmax v24.4s, v24.4s, v23.4s
    fmax v25.4s, v25.4s, v23.4s
    fmax v26.4s, v26.4s, v23.4s
    fmax v27.4s, v27.4s, v23.4s
KAI_ASM_LABEL(label_220)  // Height 5: No activation
    cmp x26, #0x10
    bge label_229
    tbz x26, #3, label_224
    st1 { v7.4s }, [x27], #0x10
    st1 { v12.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v9.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v20.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v25.4s }, [x4], #0x10
    tbz x26, #2, label_222
    st1 { v13.4s }, [x27], #0x10
    st1 { v10.4s }, [x7], #0x10
    st1 { v21.4s }, [x6], #0x10
    st1 { v18.4s }, [x5], #0x10
    st1 { v26.4s }, [x4], #0x10
    tbz x26, #1, label_221
    str d14, [x27], #0x8
    str d11, [x7], #0x8
    str d22, [x6], #0x8
    str d19, [x5], #0x8
    str d27, [x4], #0x8
    tbz x26, #0, label_228
    st1 { v14.s }[2], [x27]
    st1 { v11.s }[2], [x7]
    st1 { v22.s }[2], [x6]
    st1 { v19.s }[2], [x5]
    st1 { v27.s }[2], [x4]
    b label_228
KAI_ASM_LABEL(label_221)  // Height 5: Partial direct writeback: partial_0_12
    tbz x26, #0, label_228
    str s14, [x27, #0]
    str s11, [x7, #0]
    str s22, [x6, #0]
    str s19, [x5, #0]
    str s27, [x4, #0]
    b label_228
KAI_ASM_LABEL(label_222)  // Height 5: Partial direct writeback: partial_1_8
    tbz x26, #1, label_223
    str d13, [x27], #0x8
    str d10, [x7], #0x8
    str d21, [x6], #0x8
    str d18, [x5], #0x8
    str d26, [x4], #0x8
    tbz x26, #0, label_228
    st1 { v13.s }[2], [x27]
    st1 { v10.s }[2], [x7]
    st1 { v21.s }[2], [x6]
    st1 { v18.s }[2], [x5]
    st1 { v26.s }[2], [x4]
    b label_228
KAI_ASM_LABEL(label_223)  // Height 5: Partial direct writeback: partial_0_8
    tbz x26, #0, label_228
    str s13, [x27, #0]
    str s10, [x7, #0]
    str s21, [x6, #0]
    str s18, [x5, #0]
    str s26, [x4, #0]
    b label_228
KAI_ASM_LABEL(label_224)  // Height 5: Partial direct writeback: partial_2_0
    tbz x26, #2, label_226
    st1 { v7.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    tbz x26, #1, label_225
    str d12, [x27], #0x8
    str d9, [x7], #0x8
    str d20, [x6], #0x8
    str d17, [x5], #0x8
    str d25, [x4], #0x8
    tbz x26, #0, label_228
    st1 { v12.s }[2], [x27]
    st1 { v9.s }[2], [x7]
    st1 { v20.s }[2], [x6]
    st1 { v17.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    b label_228
KAI_ASM_LABEL(label_225)  // Height 5: Partial direct writeback: partial_0_4
    tbz x26, #0, label_228
    str s12, [x27, #0]
    str s9, [x7, #0]
    str s20, [x6, #0]
    str s17, [x5, #0]
    str s25, [x4, #0]
    b label_228
KAI_ASM_LABEL(label_226)  // Height 5: Partial direct writeback: partial_1_0
    tbz x26, #1, label_227
    str d7, [x27], #0x8
    str d8, [x7], #0x8
    str d15, [x6], #0x8
    str d16, [x5], #0x8
    str d24, [x4], #0x8
    tbz x26, #0, label_228
    st1 { v7.s }[2], [x27]
    st1 { v8.s }[2], [x7]
    st1 { v15.s }[2], [x6]
    st1 { v16.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    b label_228
KAI_ASM_LABEL(label_227)  // Height 5: Partial direct writeback: partial_0_0
    str s7, [x27, #0]
    str s8, [x7, #0]
    str s15, [x6, #0]
    str s16, [x5, #0]
    str s24, [x4, #0]
KAI_ASM_LABEL(label_228)  // Height 5: Partial direct writeback: Done
    b label_230
KAI_ASM_LABEL(label_229)  // Height 5: Full writeback
    str q7, [x27, #0]
    str q12, [x27, #0x10]
    str q13, [x27, #0x20]
    str q14, [x27, #0x30]
    add x27, x27, #0x40
    str q8, [x7, #0]
    str q9, [x7, #0x10]
    str q10, [x7, #0x20]
    str q11, [x7, #0x30]
    str q15, [x6, #0]
    str q20, [x6, #0x10]
    str q21, [x6, #0x20]
    str q22, [x6, #0x30]
    str q16, [x5, #0]
    str q17, [x5, #0x10]
    str q18, [x5, #0x20]
    str q19, [x5, #0x30]
    str q24, [x4, #0]
    str q25, [x4, #0x10]
    str q26, [x4, #0x20]
    str q27, [x4, #0x30]
KAI_ASM_LABEL(label_230)  // Height 5: Writeback done
    subs x26, x26, #0x10
    bgt label_186
    b label_277
KAI_ASM_LABEL(label_231)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_232)  // Height 6: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_233
    cmp x26, #0x8
    mov x20, x23
    bgt label_233
    cmp x26, #0x4
    mov x21, x23
    bgt label_233
    mov x22, x23
KAI_ASM_LABEL(label_233)  // Height 6: B setup done
    cbz x24, label_241
    cmp x26, #0x10
    bge label_239
    cbnz x11, label_238
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_235
KAI_ASM_LABEL(label_234)  // Height 6: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_234
KAI_ASM_LABEL(label_235)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_237
KAI_ASM_LABEL(label_236)  // Height 6: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_236
KAI_ASM_LABEL(label_237)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_238)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_240
KAI_ASM_LABEL(label_239)  // Height 6: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_240)  // Height 6: bias load done
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
    b label_253
KAI_ASM_LABEL(label_241)  // Height 6: no bias
    tbz x9, #0, label_252
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_250
    tbz x26, #3, label_245
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    ld1 { v26.4s }, [x4], #0x10
    ld1 { v29.4s }, [x3], #0x10
    tbz x26, #2, label_243
    ld1 { v11.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v19.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    ld1 { v27.4s }, [x4], #0x10
    ld1 { v30.4s }, [x3], #0x10
    tbz x26, #1, label_242
    ldr d16, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d24, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d6, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x26, #0, label_249
    ld1 { v16.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v24.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    ld1 { v6.s }[2], [x4]
    ld1 { v31.s }[2], [x3]
    b label_249
KAI_ASM_LABEL(label_242)  // Height 6: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_249
    ldr s16, [x27, #0]
    ldr s15, [x7, #0]
    ldr s24, [x6, #0]
    ldr s23, [x5, #0]
    ldr s6, [x4, #0]
    ldr s31, [x3, #0]
    b label_249
KAI_ASM_LABEL(label_243)  // Height 6: Partial accumulate: partial_1_8
    tbz x26, #1, label_244
    ldr d11, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d19, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d27, [x4], #0x8
    ldr d30, [x3], #0x8
    tbz x26, #0, label_249
    ld1 { v11.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    ld1 { v27.s }[2], [x4]
    ld1 { v30.s }[2], [x3]
    b label_249
KAI_ASM_LABEL(label_244)  // Height 6: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_249
    ldr s11, [x27, #0]
    ldr s14, [x7, #0]
    ldr s19, [x6, #0]
    ldr s22, [x5, #0]
    ldr s27, [x4, #0]
    ldr s30, [x3, #0]
    b label_249
KAI_ASM_LABEL(label_245)  // Height 6: Partial accumulate: partial_2_0
    tbz x26, #2, label_247
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    tbz x26, #1, label_246
    ldr d10, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d18, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d26, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x26, #0, label_249
    ld1 { v10.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    ld1 { v26.s }[2], [x4]
    ld1 { v29.s }[2], [x3]
    b label_249
KAI_ASM_LABEL(label_246)  // Height 6: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_249
    ldr s10, [x27, #0]
    ldr s13, [x7, #0]
    ldr s18, [x6, #0]
    ldr s21, [x5, #0]
    ldr s26, [x4, #0]
    ldr s29, [x3, #0]
    b label_249
KAI_ASM_LABEL(label_247)  // Height 6: Partial accumulate: partial_1_0
    tbz x26, #1, label_248
    ldr d9, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d17, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d25, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x26, #0, label_249
    ld1 { v9.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    ld1 { v28.s }[2], [x3]
    b label_249
KAI_ASM_LABEL(label_248)  // Height 6: Partial accumulate: partial_0_0
    ldr s9, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s17, [x6, #0]
    ldr s20, [x5, #0]
    ldr s25, [x4, #0]
    ldr s28, [x3, #0]
KAI_ASM_LABEL(label_249)  // Height 6: Partial accumulate: Done
    sub x27, x27, x1
    b label_251
KAI_ASM_LABEL(label_250)  // Height 6: full accumulate
    ldr q9, [x27, #0]
    ldr q10, [x27, #0x10]
    ldr q11, [x27, #0x20]
    ldr q16, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q17, [x6, #0]
    ldr q18, [x6, #0x10]
    ldr q19, [x6, #0x20]
    ldr q24, [x6, #0x30]
    ldr q20, [x5, #0]
    ldr q21, [x5, #0x10]
    ldr q22, [x5, #0x20]
    ldr q23, [x5, #0x30]
    ldr q25, [x4, #0]
    ldr q26, [x4, #0x10]
    ldr q27, [x4, #0x20]
    ldr q6, [x4, #0x30]
    ldr q28, [x3, #0]
    ldr q29, [x3, #0x10]
    ldr q30, [x3, #0x20]
    ldr q31, [x3, #0x30]
KAI_ASM_LABEL(label_251)  // Height 6: MMLA fixup
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
    b label_253
KAI_ASM_LABEL(label_252)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_253)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_254
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_256
KAI_ASM_LABEL(label_254)  // Height 6: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_255)  // Height 6: String loop
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
    cbnz x28, label_256
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_256)  // Height 6: input setup done
    cmp x7, #0x8
    blt label_259
    ldr q1, [x8, #0]
    ldr q2, [x6, #0]
    cmp x7, #0x10
    ldr q3, [x5, #0]
    ldr q4, [x4, #0]
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    ldr q7, [x23, #0]
    blt label_258
KAI_ASM_LABEL(label_257)  // Height 6: Multiply loop: Main loop head
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    cmp x7, #0x10
    add x6, x6, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    add x5, x5, #0x10
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    add x4, x4, #0x10
    add x2, x2, #0x10
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    add x1, x1, #0x10
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
    ldr q6, [x23, #0x30]
    ldr q4, [x4, #0]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb8)  // bfmmla v24.4s, v5.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbc)  // bfmmla v28.4s, v5.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb9)  // bfmmla v25.4s, v5.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbd)  // bfmmla v29.4s, v5.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecba)  // bfmmla v26.4s, v5.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbe)  // bfmmla v30.4s, v5.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecbb)  // bfmmla v27.4s, v5.8h, v7.8h
    ldr q7, [x23, #0]
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    ldr q1, [x8, #0]
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    ldr q3, [x5, #0]
    KAI_ASM_INST(0x6e46ecbf)  // bfmmla v31.4s, v5.8h, v6.8h
    ldr q5, [x2, #0]
    ldr q6, [x1, #0]
    bge label_257
KAI_ASM_LABEL(label_258)  // Height 6: Multiply loop: Single iteration only
    trn1 v0.2d, v1.2d, v2.2d
    trn2 v1.2d, v1.2d, v2.2d
    sub x7, x7, #0x8
    add x8, x8, #0x10
    trn1 v2.2d, v3.2d, v4.2d
    trn2 v3.2d, v3.2d, v4.2d
    add x6, x6, #0x10
    add x5, x5, #0x10
    trn1 v4.2d, v5.2d, v6.2d
    trn2 v5.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    add x4, x4, #0x10
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    add x2, x2, #0x10
    add x1, x1, #0x10
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    ldr q7, [x23, #0x20]
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x6e47ec28)  // bfmmla v8.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec70)  // bfmmla v16.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb8)  // bfmmla v24.4s, v5.8h, v7.8h
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x6e46ec2c)  // bfmmla v12.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec74)  // bfmmla v20.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbc)  // bfmmla v28.4s, v5.8h, v6.8h
    ldr q6, [x22, #0x30]
    add x22, x22, #0x40
    KAI_ASM_INST(0x6e47ec29)  // bfmmla v9.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec71)  // bfmmla v17.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecb9)  // bfmmla v25.4s, v5.8h, v7.8h
    ldr q7, [x21, #0x20]
    KAI_ASM_INST(0x6e46ec2d)  // bfmmla v13.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec75)  // bfmmla v21.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbd)  // bfmmla v29.4s, v5.8h, v6.8h
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    KAI_ASM_INST(0x6e47ec2a)  // bfmmla v10.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec72)  // bfmmla v18.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecba)  // bfmmla v26.4s, v5.8h, v7.8h
    ldr q7, [x20, #0x20]
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec76)  // bfmmla v22.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbe)  // bfmmla v30.4s, v5.8h, v6.8h
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6e47ec2b)  // bfmmla v11.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e47ec73)  // bfmmla v19.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e47ecbb)  // bfmmla v27.4s, v5.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e46ec77)  // bfmmla v23.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e46ecbf)  // bfmmla v31.4s, v5.8h, v6.8h
KAI_ASM_LABEL(label_259)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_264
    cmp x7, #0x4
    blt label_261
KAI_ASM_LABEL(label_260)  // Height 6: Multiply loop: Odd block loop
    ldr d1, [x8], #0x8
    ldr d2, [x6], #0x8
    sub x7, x7, #0x4
    ldr d3, [x5], #0x8
    ldr d4, [x4], #0x8
    cmp x7, #0x4
    ldr d5, [x2], #0x8
    ldr d7, [x1], #0x8
    ldr q6, [x23, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v7.2d
    ldr q7, [x23, #0x10]
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec50)  // bfmmla v16.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec98)  // bfmmla v24.4s, v4.8h, v6.8h
    ldr q6, [x22, #0]
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec54)  // bfmmla v20.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9c)  // bfmmla v28.4s, v4.8h, v7.8h
    ldr q7, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec51)  // bfmmla v17.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec99)  // bfmmla v25.4s, v4.8h, v6.8h
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec55)  // bfmmla v21.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9d)  // bfmmla v29.4s, v4.8h, v7.8h
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec52)  // bfmmla v18.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9a)  // bfmmla v26.4s, v4.8h, v6.8h
    ldr q6, [x20, #0]
    KAI_ASM_INST(0x6e47ec0e)  // bfmmla v14.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec56)  // bfmmla v22.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9e)  // bfmmla v30.4s, v4.8h, v7.8h
    ldr q7, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec53)  // bfmmla v19.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9b)  // bfmmla v27.4s, v4.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9f)  // bfmmla v31.4s, v4.8h, v7.8h
    bge label_260
KAI_ASM_LABEL(label_261)  // Height 6: Multiply loop: Skip odd blocks
    cbz x7, label_264
    tbz x7, #1, label_262
    ldr s1, [x8], #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    ldr s4, [x4], #0x4
    ldr s5, [x2], #0x4
    ldr s6, [x1], #0x4
    tbz x7, #0, label_263
    ld1 { v1.h }[2], [x8]
    ld1 { v2.h }[2], [x6]
    ld1 { v3.h }[2], [x5]
    ld1 { v4.h }[2], [x4]
    ld1 { v5.h }[2], [x2]
    ld1 { v6.h }[2], [x1]
    b label_263
KAI_ASM_LABEL(label_262)  // Height 6: Multiply loop: Ragged operand read: partial_0_0
    ldr h1, [x8, #0]
    ldr h2, [x6, #0]
    ldr h3, [x5, #0]
    ldr h4, [x4, #0]
    ldr h5, [x2, #0]
    ldr h6, [x1, #0]
KAI_ASM_LABEL(label_263)  // Height 6: Multiply loop: Ragged operand read: Done
    ldr q7, [x23, #0]
    trn1 v0.2d, v1.2d, v2.2d
    trn1 v2.2d, v3.2d, v4.2d
    trn1 v4.2d, v5.2d, v6.2d
    ldr q6, [x23, #0x10]
    add x23, x23, #0x20
    KAI_ASM_INST(0x6e47ec08)  // bfmmla v8.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec50)  // bfmmla v16.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec98)  // bfmmla v24.4s, v4.8h, v7.8h
    ldr q7, [x22, #0]
    KAI_ASM_INST(0x6e46ec0c)  // bfmmla v12.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9c)  // bfmmla v28.4s, v4.8h, v6.8h
    ldr q6, [x22, #0x10]
    add x22, x22, #0x20
    KAI_ASM_INST(0x6e47ec09)  // bfmmla v9.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec51)  // bfmmla v17.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec99)  // bfmmla v25.4s, v4.8h, v7.8h
    ldr q7, [x21, #0]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9d)  // bfmmla v29.4s, v4.8h, v6.8h
    ldr q6, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x6e47ec0a)  // bfmmla v10.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec52)  // bfmmla v18.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9a)  // bfmmla v26.4s, v4.8h, v7.8h
    ldr q7, [x20, #0]
    KAI_ASM_INST(0x6e46ec0e)  // bfmmla v14.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9e)  // bfmmla v30.4s, v4.8h, v6.8h
    ldr q6, [x20, #0x10]
    add x20, x20, #0x20
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec53)  // bfmmla v19.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e47ec9b)  // bfmmla v27.4s, v4.8h, v7.8h
    KAI_ASM_INST(0x6e46ec0f)  // bfmmla v15.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e46ec9f)  // bfmmla v31.4s, v4.8h, v6.8h
KAI_ASM_LABEL(label_264)  // Height 6: Multiply loop: No odd multiplies
    tbz x9, #3, label_265
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_255
KAI_ASM_LABEL(label_265)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v7.2d, v8.2d, v12.2d
    uzp2 v8.2d, v8.2d, v12.2d
    uzp1 v12.2d, v9.2d, v13.2d
    uzp2 v9.2d, v9.2d, v13.2d
    uzp1 v13.2d, v10.2d, v14.2d
    uzp2 v10.2d, v10.2d, v14.2d
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    uzp1 v14.2d, v11.2d, v15.2d
    uzp2 v11.2d, v11.2d, v15.2d
    add x4, x5, x1
    uzp1 v15.2d, v16.2d, v20.2d
    uzp2 v16.2d, v16.2d, v20.2d
    add x3, x4, x1
    uzp1 v20.2d, v17.2d, v21.2d
    uzp2 v17.2d, v17.2d, v21.2d
    uzp1 v21.2d, v18.2d, v22.2d
    uzp2 v18.2d, v18.2d, v22.2d
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
    tbz x9, #1, label_266
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v0.4s }, [x2]
    ld1r { v31.4s }, [x1]
    fmin v7.4s, v7.4s, v0.4s
    fmin v12.4s, v12.4s, v0.4s
    fmin v13.4s, v13.4s, v0.4s
    fmin v14.4s, v14.4s, v0.4s
    fmin v8.4s, v8.4s, v0.4s
    fmin v9.4s, v9.4s, v0.4s
    fmin v10.4s, v10.4s, v0.4s
    fmin v11.4s, v11.4s, v0.4s
    fmin v15.4s, v15.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmax v7.4s, v7.4s, v31.4s
    fmax v12.4s, v12.4s, v31.4s
    fmax v13.4s, v13.4s, v31.4s
    fmax v14.4s, v14.4s, v31.4s
    fmax v8.4s, v8.4s, v31.4s
    fmax v9.4s, v9.4s, v31.4s
    fmax v10.4s, v10.4s, v31.4s
    fmax v11.4s, v11.4s, v31.4s
    fmax v15.4s, v15.4s, v31.4s
    fmax v20.4s, v20.4s, v31.4s
    fmax v21.4s, v21.4s, v31.4s
    fmax v22.4s, v22.4s, v31.4s
    fmax v16.4s, v16.4s, v31.4s
    fmax v17.4s, v17.4s, v31.4s
    fmax v18.4s, v18.4s, v31.4s
    fmax v19.4s, v19.4s, v31.4s
    fmax v23.4s, v23.4s, v31.4s
    fmax v28.4s, v28.4s, v31.4s
    fmax v29.4s, v29.4s, v31.4s
    fmax v30.4s, v30.4s, v31.4s
    fmax v24.4s, v24.4s, v31.4s
    fmax v25.4s, v25.4s, v31.4s
    fmax v26.4s, v26.4s, v31.4s
    fmax v27.4s, v27.4s, v31.4s
KAI_ASM_LABEL(label_266)  // Height 6: No activation
    cmp x26, #0x10
    bge label_275
    tbz x26, #3, label_270
    st1 { v7.4s }, [x27], #0x10
    st1 { v12.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v9.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v20.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v17.4s }, [x5], #0x10
    st1 { v23.4s }, [x4], #0x10
    st1 { v28.4s }, [x4], #0x10
    st1 { v24.4s }, [x3], #0x10
    st1 { v25.4s }, [x3], #0x10
    tbz x26, #2, label_268
    st1 { v13.4s }, [x27], #0x10
    st1 { v10.4s }, [x7], #0x10
    st1 { v21.4s }, [x6], #0x10
    st1 { v18.4s }, [x5], #0x10
    st1 { v29.4s }, [x4], #0x10
    st1 { v26.4s }, [x3], #0x10
    tbz x26, #1, label_267
    str d14, [x27], #0x8
    str d11, [x7], #0x8
    str d22, [x6], #0x8
    str d19, [x5], #0x8
    str d30, [x4], #0x8
    str d27, [x3], #0x8
    tbz x26, #0, label_274
    st1 { v14.s }[2], [x27]
    st1 { v11.s }[2], [x7]
    st1 { v22.s }[2], [x6]
    st1 { v19.s }[2], [x5]
    st1 { v30.s }[2], [x4]
    st1 { v27.s }[2], [x3]
    b label_274
KAI_ASM_LABEL(label_267)  // Height 6: Partial direct writeback: partial_0_12
    tbz x26, #0, label_274
    str s14, [x27, #0]
    str s11, [x7, #0]
    str s22, [x6, #0]
    str s19, [x5, #0]
    str s30, [x4, #0]
    str s27, [x3, #0]
    b label_274
KAI_ASM_LABEL(label_268)  // Height 6: Partial direct writeback: partial_1_8
    tbz x26, #1, label_269
    str d13, [x27], #0x8
    str d10, [x7], #0x8
    str d21, [x6], #0x8
    str d18, [x5], #0x8
    str d29, [x4], #0x8
    str d26, [x3], #0x8
    tbz x26, #0, label_274
    st1 { v13.s }[2], [x27]
    st1 { v10.s }[2], [x7]
    st1 { v21.s }[2], [x6]
    st1 { v18.s }[2], [x5]
    st1 { v29.s }[2], [x4]
    st1 { v26.s }[2], [x3]
    b label_274
KAI_ASM_LABEL(label_269)  // Height 6: Partial direct writeback: partial_0_8
    tbz x26, #0, label_274
    str s13, [x27, #0]
    str s10, [x7, #0]
    str s21, [x6, #0]
    str s18, [x5, #0]
    str s29, [x4, #0]
    str s26, [x3, #0]
    b label_274
KAI_ASM_LABEL(label_270)  // Height 6: Partial direct writeback: partial_2_0
    tbz x26, #2, label_272
    st1 { v7.4s }, [x27], #0x10
    st1 { v8.4s }, [x7], #0x10
    st1 { v15.4s }, [x6], #0x10
    st1 { v16.4s }, [x5], #0x10
    st1 { v23.4s }, [x4], #0x10
    st1 { v24.4s }, [x3], #0x10
    tbz x26, #1, label_271
    str d12, [x27], #0x8
    str d9, [x7], #0x8
    str d20, [x6], #0x8
    str d17, [x5], #0x8
    str d28, [x4], #0x8
    str d25, [x3], #0x8
    tbz x26, #0, label_274
    st1 { v12.s }[2], [x27]
    st1 { v9.s }[2], [x7]
    st1 { v20.s }[2], [x6]
    st1 { v17.s }[2], [x5]
    st1 { v28.s }[2], [x4]
    st1 { v25.s }[2], [x3]
    b label_274
KAI_ASM_LABEL(label_271)  // Height 6: Partial direct writeback: partial_0_4
    tbz x26, #0, label_274
    str s12, [x27, #0]
    str s9, [x7, #0]
    str s20, [x6, #0]
    str s17, [x5, #0]
    str s28, [x4, #0]
    str s25, [x3, #0]
    b label_274
KAI_ASM_LABEL(label_272)  // Height 6: Partial direct writeback: partial_1_0
    tbz x26, #1, label_273
    str d7, [x27], #0x8
    str d8, [x7], #0x8
    str d15, [x6], #0x8
    str d16, [x5], #0x8
    str d23, [x4], #0x8
    str d24, [x3], #0x8
    tbz x26, #0, label_274
    st1 { v7.s }[2], [x27]
    st1 { v8.s }[2], [x7]
    st1 { v15.s }[2], [x6]
    st1 { v16.s }[2], [x5]
    st1 { v23.s }[2], [x4]
    st1 { v24.s }[2], [x3]
    b label_274
KAI_ASM_LABEL(label_273)  // Height 6: Partial direct writeback: partial_0_0
    str s7, [x27, #0]
    str s8, [x7, #0]
    str s15, [x6, #0]
    str s16, [x5, #0]
    str s23, [x4, #0]
    str s24, [x3, #0]
KAI_ASM_LABEL(label_274)  // Height 6: Partial direct writeback: Done
    b label_276
KAI_ASM_LABEL(label_275)  // Height 6: Full writeback
    str q7, [x27, #0]
    str q12, [x27, #0x10]
    str q13, [x27, #0x20]
    str q14, [x27, #0x30]
    add x27, x27, #0x40
    str q8, [x7, #0]
    str q9, [x7, #0x10]
    str q10, [x7, #0x20]
    str q11, [x7, #0x30]
    str q15, [x6, #0]
    str q20, [x6, #0x10]
    str q21, [x6, #0x20]
    str q22, [x6, #0x30]
    str q16, [x5, #0]
    str q17, [x5, #0x10]
    str q18, [x5, #0x20]
    str q19, [x5, #0x30]
    str q23, [x4, #0]
    str q28, [x4, #0x10]
    str q29, [x4, #0x20]
    str q30, [x4, #0x30]
    str q24, [x3, #0]
    str q25, [x3, #0x10]
    str q26, [x3, #0x20]
    str q27, [x3, #0x30]
KAI_ASM_LABEL(label_276)  // Height 6: Writeback done
    subs x26, x26, #0x10
    bgt label_232
    add x11, x11, #0x6
    cmp x11, x10
    blt label_1
KAI_ASM_LABEL(label_277)  // Exit
    add SP, SP, #0x40
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_ffhybrid_bf16fp32_mmla_6x16)

    KAI_ASM_END
