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

    KAI_ASM_CODE(a64_hybrid_fp32bf16fp32_mmla_4x24)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_fp32bf16fp32_mmla_4x24)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_fp32bf16fp32_mmla_4x24)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_fp32bf16fp32_mmla_4x24)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    sub SP, SP, #0x60
    mov x25, #0
    ldr x4, [x0, #0x60]
    ldr w1, [x0, #0xa0]
    ldr w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x24, [x0, #0]
    cmp x4, #0
    cset x23, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x23, x23, x1, LSL #1
    cset x1, NE
    orr x23, x23, x2, LSL #1
    orr x23, x23, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x24, x25
    cmp x1, #0x4
    bge label_154
    cmp x1, #0x2
    bgt label_103
    beq label_52
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x7, label_10
    cmp x20, #0x18
    bge label_8
    cbnz x25, label_7
    mov x2, x20
    mov x1, SP
    cmp x2, #0x4
    blt label_4
KAI_ASM_LABEL(label_3)  // Height 1: Partial bias load: vector loop
    ld1 { v16.4s }, [x7], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_3
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_6
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: element loop
    ldr s16, [x7], #0x4
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
    ldr q8, [x7, #0]
    ldr q9, [x7, #0x10]
    ldr q10, [x7, #0x20]
    ldr q11, [x7, #0x30]
    ldr q12, [x7, #0x40]
    ldr q13, [x7, #0x50]
    add x7, x7, #0x60
KAI_ASM_LABEL(label_9)  // Height 1: bias load done
    zip2 v14.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v15.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v16.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v17.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    zip2 v18.2d, v12.2d, v12.2d
    zip1 v12.2d, v12.2d, v12.2d
    zip2 v19.2d, v13.2d, v13.2d
    zip1 v13.2d, v13.2d, v13.2d
    b label_26
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x23, #0, label_25
    cmp x20, #0x18
    bge label_23
    tbz x20, #4, label_14
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v12.4s }, [x21], #0x10
    tbz x20, #2, label_12
    ld1 { v13.4s }, [x21], #0x10
    tbz x20, #1, label_11
    ldr d20, [x21], #0x8
    mov x1, #0x58
    tbz x20, #0, label_22
    ld1 { v20.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x20, #0, label_22
    ldr s20, [x21, #0]
    b label_22
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_16
    tbz x20, #1, label_13
    ldr d13, [x21], #0x8
    mov x1, #0x48
    tbz x20, #0, label_22
    ld1 { v13.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x20, #0, label_22
    ldr s13, [x21, #0]
    b label_22
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_3_0
    tbz x20, #3, label_18
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v10.4s }, [x21], #0x10
    tbz x20, #2, label_16
    ld1 { v11.4s }, [x21], #0x10
    tbz x20, #1, label_15
    ldr d12, [x21], #0x8
    mov x1, #0x38
    tbz x20, #0, label_22
    ld1 { v12.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x20, #0, label_22
    ldr s12, [x21, #0]
    b label_22
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_8
    tbz x20, #1, label_17
    ldr d11, [x21], #0x8
    mov x1, #0x28
    tbz x20, #0, label_22
    ld1 { v11.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x20, #0, label_22
    ldr s11, [x21, #0]
    b label_22
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_2_0
    tbz x20, #2, label_20
    ld1 { v9.4s }, [x21], #0x10
    tbz x20, #1, label_19
    ldr d10, [x21], #0x8
    mov x1, #0x18
    tbz x20, #0, label_22
    ld1 { v10.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x20, #0, label_22
    ldr s10, [x21, #0]
    b label_22
KAI_ASM_LABEL(label_20)  // Height 1: Partial accumulate: partial_1_0
    tbz x20, #1, label_21
    ldr d9, [x21], #0x8
    mov x1, #0x8
    tbz x20, #0, label_22
    ld1 { v9.s }[2], [x21]
    b label_22
KAI_ASM_LABEL(label_21)  // Height 1: Partial accumulate: partial_0_0
    ldr s9, [x21, #0]
    mov x1, #0
KAI_ASM_LABEL(label_22)  // Height 1: Partial accumulate: Done
    sub x21, x21, x1
    b label_24
KAI_ASM_LABEL(label_23)  // Height 1: full accumulate
    ldr q9, [x21, #0]
    ldr q10, [x21, #0x10]
    ldr q11, [x21, #0x20]
    ldr q12, [x21, #0x30]
    ldr q13, [x21, #0x40]
    ldr q20, [x21, #0x50]
KAI_ASM_LABEL(label_24)  // Height 1: MMLA fixup
    zip1 v8.2d, v9.2d, v14.2d
    zip2 v14.2d, v9.2d, v14.2d
    zip1 v9.2d, v10.2d, v15.2d
    zip2 v15.2d, v10.2d, v15.2d
    zip1 v10.2d, v11.2d, v16.2d
    zip2 v16.2d, v11.2d, v16.2d
    zip1 v11.2d, v12.2d, v17.2d
    zip2 v17.2d, v12.2d, v17.2d
    zip1 v12.2d, v13.2d, v18.2d
    zip2 v18.2d, v13.2d, v18.2d
    zip1 v13.2d, v20.2d, v19.2d
    zip2 v19.2d, v20.2d, v19.2d
    b label_26
KAI_ASM_LABEL(label_25)  // Height 1: no accumulate
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
KAI_ASM_LABEL(label_26)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_27
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    b label_29
KAI_ASM_LABEL(label_27)  // Height 1: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_28)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    cbnz x22, label_29
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_29)  // Height 1: input setup done
    cmp x5, #0x4
    blt label_32
    ld1 { v0.4s }, [x6], #0x10
    ldr q4, [x8, #0]
    cmp x5, #0x8
    ldr q5, [x8, #0x10]
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    blt label_31
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    cmp x5, #0x8
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    ldr q4, [x8, #0]
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x20]
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    ld1 { v0.4s }, [x6], #0x10
    ldr q7, [x8, #0x30]
    bge label_30
KAI_ASM_LABEL(label_31)  // Height 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
KAI_ASM_LABEL(label_32)  // Height 1: Multiply loop: Main loop skip
    cbz x5, label_35
    cbz x5, label_35
    tbz x5, #1, label_33
    ldr d0, [x6], #0x8
    tbz x5, #0, label_34
    ld1 { v0.s }[2], [x6]
    b label_34
KAI_ASM_LABEL(label_33)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr s0, [x6, #0]
KAI_ASM_LABEL(label_34)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
KAI_ASM_LABEL(label_35)  // Height 1: Multiply loop: No odd multiplies
    tbz x23, #3, label_36
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_28
KAI_ASM_LABEL(label_36)  // Height 1: no string loop
    uzp1 v8.2d, v8.2d, v14.2d
    uzp1 v9.2d, v9.2d, v15.2d
    prfm pstl1keep, [x21, #0]
    uzp1 v10.2d, v10.2d, v16.2d
    uzp1 v11.2d, v11.2d, v17.2d
    uzp1 v12.2d, v12.2d, v18.2d
    uzp1 v13.2d, v13.2d, v19.2d
    tbz x23, #1, label_37
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
KAI_ASM_LABEL(label_37)  // Height 1: No activation
    cmp x20, #0x18
    bge label_50
    tbz x20, #4, label_41
    st1 { v8.4s }, [x21], #0x10
    st1 { v9.4s }, [x21], #0x10
    st1 { v10.4s }, [x21], #0x10
    st1 { v11.4s }, [x21], #0x10
    tbz x20, #2, label_39
    st1 { v12.4s }, [x21], #0x10
    tbz x20, #1, label_38
    str d13, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v13.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_0_20
    tbz x20, #0, label_49
    str s13, [x21, #0]
    b label_49
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_1_16
    tbz x20, #1, label_40
    str d12, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v12.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_0_16
    tbz x20, #0, label_49
    str s12, [x21, #0]
    b label_49
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_3_0
    tbz x20, #3, label_45
    st1 { v8.4s }, [x21], #0x10
    st1 { v9.4s }, [x21], #0x10
    tbz x20, #2, label_43
    st1 { v10.4s }, [x21], #0x10
    tbz x20, #1, label_42
    str d11, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v11.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_0_12
    tbz x20, #0, label_49
    str s11, [x21, #0]
    b label_49
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_1_8
    tbz x20, #1, label_44
    str d10, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v10.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: partial_0_8
    tbz x20, #0, label_49
    str s10, [x21, #0]
    b label_49
KAI_ASM_LABEL(label_45)  // Height 1: Partial direct writeback: partial_2_0
    tbz x20, #2, label_47
    st1 { v8.4s }, [x21], #0x10
    tbz x20, #1, label_46
    str d9, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v9.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_46)  // Height 1: Partial direct writeback: partial_0_4
    tbz x20, #0, label_49
    str s9, [x21, #0]
    b label_49
KAI_ASM_LABEL(label_47)  // Height 1: Partial direct writeback: partial_1_0
    tbz x20, #1, label_48
    str d8, [x21], #0x8
    tbz x20, #0, label_49
    st1 { v8.s }[2], [x21]
    b label_49
KAI_ASM_LABEL(label_48)  // Height 1: Partial direct writeback: partial_0_0
    str s8, [x21, #0]
KAI_ASM_LABEL(label_49)  // Height 1: Partial direct writeback: Done
    b label_51
KAI_ASM_LABEL(label_50)  // Height 1: Full writeback
    str q8, [x21, #0]
    str q9, [x21, #0x10]
    str q10, [x21, #0x20]
    str q11, [x21, #0x30]
    str q12, [x21, #0x40]
    str q13, [x21, #0x50]
    add x21, x21, #0x60
KAI_ASM_LABEL(label_51)  // Height 1: Writeback done
    subs x20, x20, #0x18
    bgt label_2
    b label_205
KAI_ASM_LABEL(label_52)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_53)  // Height 2: Column loop
    cbz x7, label_61
    cmp x20, #0x18
    bge label_59
    cbnz x25, label_58
    mov x2, x20
    mov x1, SP
    cmp x2, #0x4
    blt label_55
KAI_ASM_LABEL(label_54)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x7], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_54
KAI_ASM_LABEL(label_55)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_57
KAI_ASM_LABEL(label_56)  // Height 2: Partial bias load: element loop
    ldr s16, [x7], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_56
KAI_ASM_LABEL(label_57)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_58)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_60
KAI_ASM_LABEL(label_59)  // Height 2: full width bias
    ldr q8, [x7, #0]
    ldr q9, [x7, #0x10]
    ldr q10, [x7, #0x20]
    ldr q11, [x7, #0x30]
    ldr q12, [x7, #0x40]
    ldr q13, [x7, #0x50]
    add x7, x7, #0x60
KAI_ASM_LABEL(label_60)  // Height 2: bias load done
    zip2 v14.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v15.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v16.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v17.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    zip2 v18.2d, v12.2d, v12.2d
    zip1 v12.2d, v12.2d, v12.2d
    zip2 v19.2d, v13.2d, v13.2d
    zip1 v13.2d, v13.2d, v13.2d
    b label_77
KAI_ASM_LABEL(label_61)  // Height 2: no bias
    tbz x23, #0, label_76
    ldr x1, [x0, #0x58]
    cmp x20, #0x18
    add x5, x21, x1
    bge label_74
    tbz x20, #4, label_65
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v12.4s }, [x21], #0x10
    ld1 { v17.4s }, [x5], #0x10
    tbz x20, #2, label_63
    ld1 { v13.4s }, [x21], #0x10
    ld1 { v18.4s }, [x5], #0x10
    tbz x20, #1, label_62
    ldr d20, [x21], #0x8
    ldr d19, [x5], #0x8
    mov x1, #0x58
    tbz x20, #0, label_73
    ld1 { v20.s }[2], [x21]
    ld1 { v19.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_62)  // Height 2: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x20, #0, label_73
    ldr s20, [x21, #0]
    ldr s19, [x5, #0]
    b label_73
KAI_ASM_LABEL(label_63)  // Height 2: Partial accumulate: partial_1_16
    tbz x20, #1, label_64
    ldr d13, [x21], #0x8
    ldr d18, [x5], #0x8
    mov x1, #0x48
    tbz x20, #0, label_73
    ld1 { v13.s }[2], [x21]
    ld1 { v18.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_64)  // Height 2: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x20, #0, label_73
    ldr s13, [x21, #0]
    ldr s18, [x5, #0]
    b label_73
KAI_ASM_LABEL(label_65)  // Height 2: Partial accumulate: partial_3_0
    tbz x20, #3, label_69
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    tbz x20, #2, label_67
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    tbz x20, #1, label_66
    ldr d12, [x21], #0x8
    ldr d17, [x5], #0x8
    mov x1, #0x38
    tbz x20, #0, label_73
    ld1 { v12.s }[2], [x21]
    ld1 { v17.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_66)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x20, #0, label_73
    ldr s12, [x21, #0]
    ldr s17, [x5, #0]
    b label_73
KAI_ASM_LABEL(label_67)  // Height 2: Partial accumulate: partial_1_8
    tbz x20, #1, label_68
    ldr d11, [x21], #0x8
    ldr d16, [x5], #0x8
    mov x1, #0x28
    tbz x20, #0, label_73
    ld1 { v11.s }[2], [x21]
    ld1 { v16.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_68)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x20, #0, label_73
    ldr s11, [x21, #0]
    ldr s16, [x5, #0]
    b label_73
KAI_ASM_LABEL(label_69)  // Height 2: Partial accumulate: partial_2_0
    tbz x20, #2, label_71
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    tbz x20, #1, label_70
    ldr d10, [x21], #0x8
    ldr d15, [x5], #0x8
    mov x1, #0x18
    tbz x20, #0, label_73
    ld1 { v10.s }[2], [x21]
    ld1 { v15.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_70)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x20, #0, label_73
    ldr s10, [x21, #0]
    ldr s15, [x5, #0]
    b label_73
KAI_ASM_LABEL(label_71)  // Height 2: Partial accumulate: partial_1_0
    tbz x20, #1, label_72
    ldr d9, [x21], #0x8
    ldr d14, [x5], #0x8
    mov x1, #0x8
    tbz x20, #0, label_73
    ld1 { v9.s }[2], [x21]
    ld1 { v14.s }[2], [x5]
    b label_73
KAI_ASM_LABEL(label_72)  // Height 2: Partial accumulate: partial_0_0
    ldr s9, [x21, #0]
    ldr s14, [x5, #0]
    mov x1, #0
KAI_ASM_LABEL(label_73)  // Height 2: Partial accumulate: Done
    sub x21, x21, x1
    b label_75
KAI_ASM_LABEL(label_74)  // Height 2: full accumulate
    ldr q9, [x21, #0]
    ldr q10, [x21, #0x10]
    ldr q11, [x21, #0x20]
    ldr q12, [x21, #0x30]
    ldr q13, [x21, #0x40]
    ldr q20, [x21, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
KAI_ASM_LABEL(label_75)  // Height 2: MMLA fixup
    zip1 v8.2d, v9.2d, v14.2d
    zip2 v14.2d, v9.2d, v14.2d
    zip1 v9.2d, v10.2d, v15.2d
    zip2 v15.2d, v10.2d, v15.2d
    zip1 v10.2d, v11.2d, v16.2d
    zip2 v16.2d, v11.2d, v16.2d
    zip1 v11.2d, v12.2d, v17.2d
    zip2 v17.2d, v12.2d, v17.2d
    zip1 v12.2d, v13.2d, v18.2d
    zip2 v18.2d, v13.2d, v18.2d
    zip1 v13.2d, v20.2d, v19.2d
    zip2 v19.2d, v20.2d, v19.2d
    b label_77
KAI_ASM_LABEL(label_76)  // Height 2: no accumulate
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
KAI_ASM_LABEL(label_77)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_78
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    b label_80
KAI_ASM_LABEL(label_78)  // Height 2: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_79)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    cbnz x22, label_80
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_80)  // Height 2: input setup done
    cmp x5, #0x4
    blt label_83
    ld1 { v0.4s }, [x6], #0x10
    ld1 { v1.4s }, [x4], #0x10
    cmp x5, #0x8
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    blt label_82
KAI_ASM_LABEL(label_81)  // Height 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    cmp x5, #0x8
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    ld1 { v1.4s }, [x4], #0x10
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    ldr q4, [x8, #0]
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x20]
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    ld1 { v0.4s }, [x6], #0x10
    ldr q7, [x8, #0x30]
    bge label_81
KAI_ASM_LABEL(label_82)  // Height 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
KAI_ASM_LABEL(label_83)  // Height 2: Multiply loop: Main loop skip
    cbz x5, label_86
    cbz x5, label_86
    tbz x5, #1, label_84
    ldr d0, [x6], #0x8
    ldr d1, [x4], #0x8
    tbz x5, #0, label_85
    ld1 { v0.s }[2], [x6]
    ld1 { v1.s }[2], [x4]
    b label_85
KAI_ASM_LABEL(label_84)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr s0, [x6, #0]
    ldr s1, [x4, #0]
KAI_ASM_LABEL(label_85)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
KAI_ASM_LABEL(label_86)  // Height 2: Multiply loop: No odd multiplies
    tbz x23, #3, label_87
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_79
KAI_ASM_LABEL(label_87)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v4.2d, v8.2d, v14.2d
    uzp2 v8.2d, v8.2d, v14.2d
    prfm pstl1keep, [x21, #0]
    uzp1 v14.2d, v9.2d, v15.2d
    uzp2 v9.2d, v9.2d, v15.2d
    uzp1 v15.2d, v10.2d, v16.2d
    uzp2 v10.2d, v10.2d, v16.2d
    add x5, x21, x1
    uzp1 v16.2d, v11.2d, v17.2d
    uzp2 v11.2d, v11.2d, v17.2d
    prfm pstl1keep, [x5, #0]
    uzp1 v17.2d, v12.2d, v18.2d
    uzp2 v12.2d, v12.2d, v18.2d
    uzp1 v18.2d, v13.2d, v19.2d
    uzp2 v13.2d, v13.2d, v19.2d
    tbz x23, #1, label_88
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v20.4s }, [x2]
    ld1r { v19.4s }, [x1]
    fmin v4.4s, v4.4s, v20.4s
    fmin v14.4s, v14.4s, v20.4s
    fmin v15.4s, v15.4s, v20.4s
    fmin v16.4s, v16.4s, v20.4s
    fmin v17.4s, v17.4s, v20.4s
    fmin v18.4s, v18.4s, v20.4s
    fmin v8.4s, v8.4s, v20.4s
    fmin v9.4s, v9.4s, v20.4s
    fmin v10.4s, v10.4s, v20.4s
    fmin v11.4s, v11.4s, v20.4s
    fmin v12.4s, v12.4s, v20.4s
    fmin v13.4s, v13.4s, v20.4s
    fmax v4.4s, v4.4s, v19.4s
    fmax v14.4s, v14.4s, v19.4s
    fmax v15.4s, v15.4s, v19.4s
    fmax v16.4s, v16.4s, v19.4s
    fmax v17.4s, v17.4s, v19.4s
    fmax v18.4s, v18.4s, v19.4s
    fmax v8.4s, v8.4s, v19.4s
    fmax v9.4s, v9.4s, v19.4s
    fmax v10.4s, v10.4s, v19.4s
    fmax v11.4s, v11.4s, v19.4s
    fmax v12.4s, v12.4s, v19.4s
    fmax v13.4s, v13.4s, v19.4s
KAI_ASM_LABEL(label_88)  // Height 2: No activation
    cmp x20, #0x18
    bge label_101
    tbz x20, #4, label_92
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v15.4s }, [x21], #0x10
    st1 { v16.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    st1 { v10.4s }, [x5], #0x10
    st1 { v11.4s }, [x5], #0x10
    tbz x20, #2, label_90
    st1 { v17.4s }, [x21], #0x10
    st1 { v12.4s }, [x5], #0x10
    tbz x20, #1, label_89
    str d18, [x21], #0x8
    str d13, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v18.s }[2], [x21]
    st1 { v13.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_89)  // Height 2: Partial direct writeback: partial_0_20
    tbz x20, #0, label_100
    str s18, [x21, #0]
    str s13, [x5, #0]
    b label_100
KAI_ASM_LABEL(label_90)  // Height 2: Partial direct writeback: partial_1_16
    tbz x20, #1, label_91
    str d17, [x21], #0x8
    str d12, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v17.s }[2], [x21]
    st1 { v12.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_91)  // Height 2: Partial direct writeback: partial_0_16
    tbz x20, #0, label_100
    str s17, [x21, #0]
    str s12, [x5, #0]
    b label_100
KAI_ASM_LABEL(label_92)  // Height 2: Partial direct writeback: partial_3_0
    tbz x20, #3, label_96
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    tbz x20, #2, label_94
    st1 { v15.4s }, [x21], #0x10
    st1 { v10.4s }, [x5], #0x10
    tbz x20, #1, label_93
    str d16, [x21], #0x8
    str d11, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v16.s }[2], [x21]
    st1 { v11.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_93)  // Height 2: Partial direct writeback: partial_0_12
    tbz x20, #0, label_100
    str s16, [x21, #0]
    str s11, [x5, #0]
    b label_100
KAI_ASM_LABEL(label_94)  // Height 2: Partial direct writeback: partial_1_8
    tbz x20, #1, label_95
    str d15, [x21], #0x8
    str d10, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v15.s }[2], [x21]
    st1 { v10.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_95)  // Height 2: Partial direct writeback: partial_0_8
    tbz x20, #0, label_100
    str s15, [x21, #0]
    str s10, [x5, #0]
    b label_100
KAI_ASM_LABEL(label_96)  // Height 2: Partial direct writeback: partial_2_0
    tbz x20, #2, label_98
    st1 { v4.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    tbz x20, #1, label_97
    str d14, [x21], #0x8
    str d9, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v14.s }[2], [x21]
    st1 { v9.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_97)  // Height 2: Partial direct writeback: partial_0_4
    tbz x20, #0, label_100
    str s14, [x21, #0]
    str s9, [x5, #0]
    b label_100
KAI_ASM_LABEL(label_98)  // Height 2: Partial direct writeback: partial_1_0
    tbz x20, #1, label_99
    str d4, [x21], #0x8
    str d8, [x5], #0x8
    tbz x20, #0, label_100
    st1 { v4.s }[2], [x21]
    st1 { v8.s }[2], [x5]
    b label_100
KAI_ASM_LABEL(label_99)  // Height 2: Partial direct writeback: partial_0_0
    str s4, [x21, #0]
    str s8, [x5, #0]
KAI_ASM_LABEL(label_100)  // Height 2: Partial direct writeback: Done
    b label_102
KAI_ASM_LABEL(label_101)  // Height 2: Full writeback
    str q4, [x21, #0]
    str q14, [x21, #0x10]
    str q15, [x21, #0x20]
    str q16, [x21, #0x30]
    str q17, [x21, #0x40]
    str q18, [x21, #0x50]
    add x21, x21, #0x60
    str q8, [x5, #0]
    str q9, [x5, #0x10]
    str q10, [x5, #0x20]
    str q11, [x5, #0x30]
    str q12, [x5, #0x40]
    str q13, [x5, #0x50]
KAI_ASM_LABEL(label_102)  // Height 2: Writeback done
    subs x20, x20, #0x18
    bgt label_53
    b label_205
KAI_ASM_LABEL(label_103)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_104)  // Height 3: Column loop
    cbz x7, label_112
    cmp x20, #0x18
    bge label_110
    cbnz x25, label_109
    mov x2, x20
    mov x1, SP
    cmp x2, #0x4
    blt label_106
KAI_ASM_LABEL(label_105)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x7], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_105
KAI_ASM_LABEL(label_106)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_108
KAI_ASM_LABEL(label_107)  // Height 3: Partial bias load: element loop
    ldr s16, [x7], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_107
KAI_ASM_LABEL(label_108)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_109)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_111
KAI_ASM_LABEL(label_110)  // Height 3: full width bias
    ldr q8, [x7, #0]
    ldr q9, [x7, #0x10]
    ldr q10, [x7, #0x20]
    ldr q11, [x7, #0x30]
    ldr q12, [x7, #0x40]
    ldr q13, [x7, #0x50]
    add x7, x7, #0x60
KAI_ASM_LABEL(label_111)  // Height 3: bias load done
    zip2 v14.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v15.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v16.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v17.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    zip2 v18.2d, v12.2d, v12.2d
    zip1 v12.2d, v12.2d, v12.2d
    zip2 v19.2d, v13.2d, v13.2d
    zip1 v13.2d, v13.2d, v13.2d
    mov v20.16b, v8.16b
    mov v26.16b, v14.16b
    mov v21.16b, v9.16b
    mov v27.16b, v15.16b
    mov v22.16b, v10.16b
    mov v28.16b, v16.16b
    mov v23.16b, v11.16b
    mov v29.16b, v17.16b
    mov v24.16b, v12.16b
    mov v30.16b, v18.16b
    mov v25.16b, v13.16b
    mov v31.16b, v19.16b
    b label_128
KAI_ASM_LABEL(label_112)  // Height 3: no bias
    tbz x23, #0, label_127
    ldr x1, [x0, #0x58]
    cmp x20, #0x18
    add x5, x21, x1
    add x4, x5, x1
    bge label_125
    tbz x20, #4, label_116
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    ld1 { v12.4s }, [x21], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    tbz x20, #2, label_114
    ld1 { v13.4s }, [x21], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    tbz x20, #1, label_113
    ldr d20, [x21], #0x8
    ldr d19, [x5], #0x8
    mov x1, #0x58
    ldr d4, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v20.s }[2], [x21]
    ld1 { v19.s }[2], [x5]
    ld1 { v4.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_113)  // Height 3: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x20, #0, label_124
    ldr s20, [x21, #0]
    ldr s19, [x5, #0]
    ldr s4, [x4, #0]
    b label_124
KAI_ASM_LABEL(label_114)  // Height 3: Partial accumulate: partial_1_16
    tbz x20, #1, label_115
    ldr d13, [x21], #0x8
    ldr d18, [x5], #0x8
    mov x1, #0x48
    ldr d25, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v13.s }[2], [x21]
    ld1 { v18.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_115)  // Height 3: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x20, #0, label_124
    ldr s13, [x21, #0]
    ldr s18, [x5, #0]
    ldr s25, [x4, #0]
    b label_124
KAI_ASM_LABEL(label_116)  // Height 3: Partial accumulate: partial_3_0
    tbz x20, #3, label_120
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    tbz x20, #2, label_118
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    tbz x20, #1, label_117
    ldr d12, [x21], #0x8
    ldr d17, [x5], #0x8
    mov x1, #0x38
    ldr d24, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v12.s }[2], [x21]
    ld1 { v17.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_117)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x20, #0, label_124
    ldr s12, [x21, #0]
    ldr s17, [x5, #0]
    ldr s24, [x4, #0]
    b label_124
KAI_ASM_LABEL(label_118)  // Height 3: Partial accumulate: partial_1_8
    tbz x20, #1, label_119
    ldr d11, [x21], #0x8
    ldr d16, [x5], #0x8
    mov x1, #0x28
    ldr d23, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v11.s }[2], [x21]
    ld1 { v16.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_119)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x20, #0, label_124
    ldr s11, [x21, #0]
    ldr s16, [x5, #0]
    ldr s23, [x4, #0]
    b label_124
KAI_ASM_LABEL(label_120)  // Height 3: Partial accumulate: partial_2_0
    tbz x20, #2, label_122
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    tbz x20, #1, label_121
    ldr d10, [x21], #0x8
    ldr d15, [x5], #0x8
    mov x1, #0x18
    ldr d22, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v10.s }[2], [x21]
    ld1 { v15.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_121)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x20, #0, label_124
    ldr s10, [x21, #0]
    ldr s15, [x5, #0]
    ldr s22, [x4, #0]
    b label_124
KAI_ASM_LABEL(label_122)  // Height 3: Partial accumulate: partial_1_0
    tbz x20, #1, label_123
    ldr d9, [x21], #0x8
    ldr d14, [x5], #0x8
    mov x1, #0x8
    ldr d21, [x4], #0x8
    tbz x20, #0, label_124
    ld1 { v9.s }[2], [x21]
    ld1 { v14.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    b label_124
KAI_ASM_LABEL(label_123)  // Height 3: Partial accumulate: partial_0_0
    ldr s9, [x21, #0]
    ldr s14, [x5, #0]
    mov x1, #0
    ldr s21, [x4, #0]
KAI_ASM_LABEL(label_124)  // Height 3: Partial accumulate: Done
    sub x21, x21, x1
    b label_126
KAI_ASM_LABEL(label_125)  // Height 3: full accumulate
    ldr q9, [x21, #0]
    ldr q10, [x21, #0x10]
    ldr q11, [x21, #0x20]
    ldr q12, [x21, #0x30]
    ldr q13, [x21, #0x40]
    ldr q20, [x21, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
    ldr q21, [x4, #0]
    ldr q22, [x4, #0x10]
    ldr q23, [x4, #0x20]
    ldr q24, [x4, #0x30]
    ldr q25, [x4, #0x40]
    ldr q4, [x4, #0x50]
KAI_ASM_LABEL(label_126)  // Height 3: MMLA fixup
    zip1 v8.2d, v9.2d, v14.2d
    zip2 v14.2d, v9.2d, v14.2d
    zip1 v9.2d, v10.2d, v15.2d
    zip2 v15.2d, v10.2d, v15.2d
    zip1 v10.2d, v11.2d, v16.2d
    zip2 v16.2d, v11.2d, v16.2d
    zip1 v11.2d, v12.2d, v17.2d
    zip2 v17.2d, v12.2d, v17.2d
    zip1 v12.2d, v13.2d, v18.2d
    zip2 v18.2d, v13.2d, v18.2d
    zip1 v13.2d, v20.2d, v19.2d
    zip2 v19.2d, v20.2d, v19.2d
    zip1 v20.2d, v21.2d, v26.2d
    zip2 v26.2d, v21.2d, v26.2d
    zip1 v21.2d, v22.2d, v27.2d
    zip2 v27.2d, v22.2d, v27.2d
    zip1 v22.2d, v23.2d, v28.2d
    zip2 v28.2d, v23.2d, v28.2d
    zip1 v23.2d, v24.2d, v29.2d
    zip2 v29.2d, v24.2d, v29.2d
    zip1 v24.2d, v25.2d, v30.2d
    zip2 v30.2d, v25.2d, v30.2d
    zip1 v25.2d, v4.2d, v31.2d
    zip2 v31.2d, v4.2d, v31.2d
    b label_128
KAI_ASM_LABEL(label_127)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_128)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_129
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    b label_131
KAI_ASM_LABEL(label_129)  // Height 3: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_130)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    cbnz x22, label_131
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_131)  // Height 3: input setup done
    cmp x5, #0x4
    blt label_134
    ld1 { v0.4s }, [x6], #0x10
    ld1 { v1.4s }, [x4], #0x10
    cmp x5, #0x8
    ld1 { v2.4s }, [x2], #0x10
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    blt label_133
KAI_ASM_LABEL(label_132)  // Height 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    cmp x5, #0x8
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    ld1 { v1.4s }, [x4], #0x10
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    ldr q4, [x8, #0]
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x20]
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    ld1 { v0.4s }, [x6], #0x10
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
    ld1 { v2.4s }, [x2], #0x10
    ldr q7, [x8, #0x30]
    bge label_132
KAI_ASM_LABEL(label_133)  // Height 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
KAI_ASM_LABEL(label_134)  // Height 3: Multiply loop: Main loop skip
    cbz x5, label_137
    cbz x5, label_137
    tbz x5, #1, label_135
    ldr d0, [x6], #0x8
    ldr d1, [x4], #0x8
    ldr d2, [x2], #0x8
    tbz x5, #0, label_136
    ld1 { v0.s }[2], [x6]
    ld1 { v1.s }[2], [x4]
    ld1 { v2.s }[2], [x2]
    b label_136
KAI_ASM_LABEL(label_135)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr s0, [x6, #0]
    ldr s1, [x4, #0]
    ldr s2, [x2, #0]
KAI_ASM_LABEL(label_136)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
KAI_ASM_LABEL(label_137)  // Height 3: Multiply loop: No odd multiplies
    tbz x23, #3, label_138
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_130
KAI_ASM_LABEL(label_138)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v4.2d, v8.2d, v14.2d
    uzp2 v8.2d, v8.2d, v14.2d
    prfm pstl1keep, [x21, #0]
    uzp1 v14.2d, v9.2d, v15.2d
    uzp2 v9.2d, v9.2d, v15.2d
    uzp1 v15.2d, v10.2d, v16.2d
    uzp2 v10.2d, v10.2d, v16.2d
    add x5, x21, x1
    add x4, x5, x1
    uzp1 v16.2d, v11.2d, v17.2d
    uzp2 v11.2d, v11.2d, v17.2d
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    uzp1 v17.2d, v12.2d, v18.2d
    uzp2 v12.2d, v12.2d, v18.2d
    uzp1 v18.2d, v13.2d, v19.2d
    uzp2 v13.2d, v13.2d, v19.2d
    uzp1 v20.2d, v20.2d, v26.2d
    uzp1 v21.2d, v21.2d, v27.2d
    uzp1 v22.2d, v22.2d, v28.2d
    uzp1 v23.2d, v23.2d, v29.2d
    uzp1 v24.2d, v24.2d, v30.2d
    uzp1 v25.2d, v25.2d, v31.2d
    tbz x23, #1, label_139
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v26.4s }, [x2]
    ld1r { v19.4s }, [x1]
    fmin v4.4s, v4.4s, v26.4s
    fmin v14.4s, v14.4s, v26.4s
    fmin v15.4s, v15.4s, v26.4s
    fmin v16.4s, v16.4s, v26.4s
    fmin v17.4s, v17.4s, v26.4s
    fmin v18.4s, v18.4s, v26.4s
    fmin v8.4s, v8.4s, v26.4s
    fmin v9.4s, v9.4s, v26.4s
    fmin v10.4s, v10.4s, v26.4s
    fmin v11.4s, v11.4s, v26.4s
    fmin v12.4s, v12.4s, v26.4s
    fmin v13.4s, v13.4s, v26.4s
    fmin v20.4s, v20.4s, v26.4s
    fmin v21.4s, v21.4s, v26.4s
    fmin v22.4s, v22.4s, v26.4s
    fmin v23.4s, v23.4s, v26.4s
    fmin v24.4s, v24.4s, v26.4s
    fmin v25.4s, v25.4s, v26.4s
    fmax v4.4s, v4.4s, v19.4s
    fmax v14.4s, v14.4s, v19.4s
    fmax v15.4s, v15.4s, v19.4s
    fmax v16.4s, v16.4s, v19.4s
    fmax v17.4s, v17.4s, v19.4s
    fmax v18.4s, v18.4s, v19.4s
    fmax v8.4s, v8.4s, v19.4s
    fmax v9.4s, v9.4s, v19.4s
    fmax v10.4s, v10.4s, v19.4s
    fmax v11.4s, v11.4s, v19.4s
    fmax v12.4s, v12.4s, v19.4s
    fmax v13.4s, v13.4s, v19.4s
    fmax v20.4s, v20.4s, v19.4s
    fmax v21.4s, v21.4s, v19.4s
    fmax v22.4s, v22.4s, v19.4s
    fmax v23.4s, v23.4s, v19.4s
    fmax v24.4s, v24.4s, v19.4s
    fmax v25.4s, v25.4s, v19.4s
KAI_ASM_LABEL(label_139)  // Height 3: No activation
    cmp x20, #0x18
    bge label_152
    tbz x20, #4, label_143
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v15.4s }, [x21], #0x10
    st1 { v16.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    st1 { v10.4s }, [x5], #0x10
    st1 { v11.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    st1 { v22.4s }, [x4], #0x10
    st1 { v23.4s }, [x4], #0x10
    tbz x20, #2, label_141
    st1 { v17.4s }, [x21], #0x10
    st1 { v12.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    tbz x20, #1, label_140
    str d18, [x21], #0x8
    str d13, [x5], #0x8
    str d25, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v18.s }[2], [x21]
    st1 { v13.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_140)  // Height 3: Partial direct writeback: partial_0_20
    tbz x20, #0, label_151
    str s18, [x21, #0]
    str s13, [x5, #0]
    str s25, [x4, #0]
    b label_151
KAI_ASM_LABEL(label_141)  // Height 3: Partial direct writeback: partial_1_16
    tbz x20, #1, label_142
    str d17, [x21], #0x8
    str d12, [x5], #0x8
    str d24, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v17.s }[2], [x21]
    st1 { v12.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_142)  // Height 3: Partial direct writeback: partial_0_16
    tbz x20, #0, label_151
    str s17, [x21, #0]
    str s12, [x5, #0]
    str s24, [x4, #0]
    b label_151
KAI_ASM_LABEL(label_143)  // Height 3: Partial direct writeback: partial_3_0
    tbz x20, #3, label_147
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    st1 { v21.4s }, [x4], #0x10
    tbz x20, #2, label_145
    st1 { v15.4s }, [x21], #0x10
    st1 { v10.4s }, [x5], #0x10
    st1 { v22.4s }, [x4], #0x10
    tbz x20, #1, label_144
    str d16, [x21], #0x8
    str d11, [x5], #0x8
    str d23, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v16.s }[2], [x21]
    st1 { v11.s }[2], [x5]
    st1 { v23.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_144)  // Height 3: Partial direct writeback: partial_0_12
    tbz x20, #0, label_151
    str s16, [x21, #0]
    str s11, [x5, #0]
    str s23, [x4, #0]
    b label_151
KAI_ASM_LABEL(label_145)  // Height 3: Partial direct writeback: partial_1_8
    tbz x20, #1, label_146
    str d15, [x21], #0x8
    str d10, [x5], #0x8
    str d22, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v15.s }[2], [x21]
    st1 { v10.s }[2], [x5]
    st1 { v22.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_146)  // Height 3: Partial direct writeback: partial_0_8
    tbz x20, #0, label_151
    str s15, [x21, #0]
    str s10, [x5, #0]
    str s22, [x4, #0]
    b label_151
KAI_ASM_LABEL(label_147)  // Height 3: Partial direct writeback: partial_2_0
    tbz x20, #2, label_149
    st1 { v4.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v20.4s }, [x4], #0x10
    tbz x20, #1, label_148
    str d14, [x21], #0x8
    str d9, [x5], #0x8
    str d21, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v14.s }[2], [x21]
    st1 { v9.s }[2], [x5]
    st1 { v21.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_148)  // Height 3: Partial direct writeback: partial_0_4
    tbz x20, #0, label_151
    str s14, [x21, #0]
    str s9, [x5, #0]
    str s21, [x4, #0]
    b label_151
KAI_ASM_LABEL(label_149)  // Height 3: Partial direct writeback: partial_1_0
    tbz x20, #1, label_150
    str d4, [x21], #0x8
    str d8, [x5], #0x8
    str d20, [x4], #0x8
    tbz x20, #0, label_151
    st1 { v4.s }[2], [x21]
    st1 { v8.s }[2], [x5]
    st1 { v20.s }[2], [x4]
    b label_151
KAI_ASM_LABEL(label_150)  // Height 3: Partial direct writeback: partial_0_0
    str s4, [x21, #0]
    str s8, [x5, #0]
    str s20, [x4, #0]
KAI_ASM_LABEL(label_151)  // Height 3: Partial direct writeback: Done
    b label_153
KAI_ASM_LABEL(label_152)  // Height 3: Full writeback
    str q4, [x21, #0]
    str q14, [x21, #0x10]
    str q15, [x21, #0x20]
    str q16, [x21, #0x30]
    str q17, [x21, #0x40]
    str q18, [x21, #0x50]
    add x21, x21, #0x60
    str q8, [x5, #0]
    str q9, [x5, #0x10]
    str q10, [x5, #0x20]
    str q11, [x5, #0x30]
    str q12, [x5, #0x40]
    str q13, [x5, #0x50]
    str q20, [x4, #0]
    str q21, [x4, #0x10]
    str q22, [x4, #0x20]
    str q23, [x4, #0x30]
    str q24, [x4, #0x40]
    str q25, [x4, #0x50]
KAI_ASM_LABEL(label_153)  // Height 3: Writeback done
    subs x20, x20, #0x18
    bgt label_104
    b label_205
KAI_ASM_LABEL(label_154)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x21, [x0, #0x50]
    ldr x20, [x0, #0x8]
    ldr x8, [x0, #0x30]
    ldr x7, [x0, #0x40]
    madd x21, x25, x1, x21
KAI_ASM_LABEL(label_155)  // Height 4: Column loop
    cbz x7, label_163
    cmp x20, #0x18
    bge label_161
    cbnz x25, label_160
    mov x2, x20
    mov x1, SP
    cmp x2, #0x4
    blt label_157
KAI_ASM_LABEL(label_156)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x7], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_156
KAI_ASM_LABEL(label_157)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_159
KAI_ASM_LABEL(label_158)  // Height 4: Partial bias load: element loop
    ldr s16, [x7], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_158
KAI_ASM_LABEL(label_159)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_160)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    ldr q12, [x1, #0x40]
    ldr q13, [x1, #0x50]
    b label_162
KAI_ASM_LABEL(label_161)  // Height 4: full width bias
    ldr q8, [x7, #0]
    ldr q9, [x7, #0x10]
    ldr q10, [x7, #0x20]
    ldr q11, [x7, #0x30]
    ldr q12, [x7, #0x40]
    ldr q13, [x7, #0x50]
    add x7, x7, #0x60
KAI_ASM_LABEL(label_162)  // Height 4: bias load done
    zip2 v14.2d, v8.2d, v8.2d
    zip1 v8.2d, v8.2d, v8.2d
    zip2 v15.2d, v9.2d, v9.2d
    zip1 v9.2d, v9.2d, v9.2d
    zip2 v16.2d, v10.2d, v10.2d
    zip1 v10.2d, v10.2d, v10.2d
    zip2 v17.2d, v11.2d, v11.2d
    zip1 v11.2d, v11.2d, v11.2d
    zip2 v18.2d, v12.2d, v12.2d
    zip1 v12.2d, v12.2d, v12.2d
    zip2 v19.2d, v13.2d, v13.2d
    zip1 v13.2d, v13.2d, v13.2d
    mov v20.16b, v8.16b
    mov v26.16b, v14.16b
    mov v21.16b, v9.16b
    mov v27.16b, v15.16b
    mov v22.16b, v10.16b
    mov v28.16b, v16.16b
    mov v23.16b, v11.16b
    mov v29.16b, v17.16b
    mov v24.16b, v12.16b
    mov v30.16b, v18.16b
    mov v25.16b, v13.16b
    mov v31.16b, v19.16b
    b label_179
KAI_ASM_LABEL(label_163)  // Height 4: no bias
    tbz x23, #0, label_178
    ldr x1, [x0, #0x58]
    cmp x20, #0x18
    add x5, x21, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_176
    tbz x20, #4, label_167
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    ld1 { v12.4s }, [x21], #0x10
    ld1 { v17.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v29.4s }, [x3], #0x10
    tbz x20, #2, label_165
    ld1 { v13.4s }, [x21], #0x10
    ld1 { v18.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    ld1 { v30.4s }, [x3], #0x10
    tbz x20, #1, label_164
    ldr d20, [x21], #0x8
    ldr d19, [x5], #0x8
    mov x1, #0x58
    ldr d4, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v20.s }[2], [x21]
    ld1 { v19.s }[2], [x5]
    ld1 { v4.s }[2], [x4]
    ld1 { v31.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_164)  // Height 4: Partial accumulate: partial_0_20
    mov x1, #0x50
    tbz x20, #0, label_175
    ldr s20, [x21, #0]
    ldr s19, [x5, #0]
    ldr s4, [x4, #0]
    ldr s31, [x3, #0]
    b label_175
KAI_ASM_LABEL(label_165)  // Height 4: Partial accumulate: partial_1_16
    tbz x20, #1, label_166
    ldr d13, [x21], #0x8
    ldr d18, [x5], #0x8
    mov x1, #0x48
    ldr d25, [x4], #0x8
    ldr d30, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v13.s }[2], [x21]
    ld1 { v18.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    ld1 { v30.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_166)  // Height 4: Partial accumulate: partial_0_16
    mov x1, #0x40
    tbz x20, #0, label_175
    ldr s13, [x21, #0]
    ldr s18, [x5, #0]
    ldr s25, [x4, #0]
    ldr s30, [x3, #0]
    b label_175
KAI_ASM_LABEL(label_167)  // Height 4: Partial accumulate: partial_3_0
    tbz x20, #3, label_171
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    ld1 { v10.4s }, [x21], #0x10
    ld1 { v15.4s }, [x5], #0x10
    ld1 { v22.4s }, [x4], #0x10
    ld1 { v27.4s }, [x3], #0x10
    tbz x20, #2, label_169
    ld1 { v11.4s }, [x21], #0x10
    ld1 { v16.4s }, [x5], #0x10
    ld1 { v23.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    tbz x20, #1, label_168
    ldr d12, [x21], #0x8
    ldr d17, [x5], #0x8
    mov x1, #0x38
    ldr d24, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v12.s }[2], [x21]
    ld1 { v17.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    ld1 { v29.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_168)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x20, #0, label_175
    ldr s12, [x21, #0]
    ldr s17, [x5, #0]
    ldr s24, [x4, #0]
    ldr s29, [x3, #0]
    b label_175
KAI_ASM_LABEL(label_169)  // Height 4: Partial accumulate: partial_1_8
    tbz x20, #1, label_170
    ldr d11, [x21], #0x8
    ldr d16, [x5], #0x8
    mov x1, #0x28
    ldr d23, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v11.s }[2], [x21]
    ld1 { v16.s }[2], [x5]
    ld1 { v23.s }[2], [x4]
    ld1 { v28.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_170)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x20, #0, label_175
    ldr s11, [x21, #0]
    ldr s16, [x5, #0]
    ldr s23, [x4, #0]
    ldr s28, [x3, #0]
    b label_175
KAI_ASM_LABEL(label_171)  // Height 4: Partial accumulate: partial_2_0
    tbz x20, #2, label_173
    ld1 { v9.4s }, [x21], #0x10
    ld1 { v14.4s }, [x5], #0x10
    ld1 { v21.4s }, [x4], #0x10
    ld1 { v26.4s }, [x3], #0x10
    tbz x20, #1, label_172
    ldr d10, [x21], #0x8
    ldr d15, [x5], #0x8
    mov x1, #0x18
    ldr d22, [x4], #0x8
    ldr d27, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v10.s }[2], [x21]
    ld1 { v15.s }[2], [x5]
    ld1 { v22.s }[2], [x4]
    ld1 { v27.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_172)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x20, #0, label_175
    ldr s10, [x21, #0]
    ldr s15, [x5, #0]
    ldr s22, [x4, #0]
    ldr s27, [x3, #0]
    b label_175
KAI_ASM_LABEL(label_173)  // Height 4: Partial accumulate: partial_1_0
    tbz x20, #1, label_174
    ldr d9, [x21], #0x8
    ldr d14, [x5], #0x8
    mov x1, #0x8
    ldr d21, [x4], #0x8
    ldr d26, [x3], #0x8
    tbz x20, #0, label_175
    ld1 { v9.s }[2], [x21]
    ld1 { v14.s }[2], [x5]
    ld1 { v21.s }[2], [x4]
    ld1 { v26.s }[2], [x3]
    b label_175
KAI_ASM_LABEL(label_174)  // Height 4: Partial accumulate: partial_0_0
    ldr s9, [x21, #0]
    ldr s14, [x5, #0]
    mov x1, #0
    ldr s21, [x4, #0]
    ldr s26, [x3, #0]
KAI_ASM_LABEL(label_175)  // Height 4: Partial accumulate: Done
    sub x21, x21, x1
    b label_177
KAI_ASM_LABEL(label_176)  // Height 4: full accumulate
    ldr q9, [x21, #0]
    ldr q10, [x21, #0x10]
    ldr q11, [x21, #0x20]
    ldr q12, [x21, #0x30]
    ldr q13, [x21, #0x40]
    ldr q20, [x21, #0x50]
    ldr q14, [x5, #0]
    ldr q15, [x5, #0x10]
    ldr q16, [x5, #0x20]
    ldr q17, [x5, #0x30]
    ldr q18, [x5, #0x40]
    ldr q19, [x5, #0x50]
    ldr q21, [x4, #0]
    ldr q22, [x4, #0x10]
    ldr q23, [x4, #0x20]
    ldr q24, [x4, #0x30]
    ldr q25, [x4, #0x40]
    ldr q4, [x4, #0x50]
    ldr q26, [x3, #0]
    ldr q27, [x3, #0x10]
    ldr q28, [x3, #0x20]
    ldr q29, [x3, #0x30]
    ldr q30, [x3, #0x40]
    ldr q31, [x3, #0x50]
KAI_ASM_LABEL(label_177)  // Height 4: MMLA fixup
    zip1 v8.2d, v9.2d, v14.2d
    zip2 v14.2d, v9.2d, v14.2d
    zip1 v9.2d, v10.2d, v15.2d
    zip2 v15.2d, v10.2d, v15.2d
    zip1 v10.2d, v11.2d, v16.2d
    zip2 v16.2d, v11.2d, v16.2d
    zip1 v11.2d, v12.2d, v17.2d
    zip2 v17.2d, v12.2d, v17.2d
    zip1 v12.2d, v13.2d, v18.2d
    zip2 v18.2d, v13.2d, v18.2d
    zip1 v13.2d, v20.2d, v19.2d
    zip2 v19.2d, v20.2d, v19.2d
    zip1 v20.2d, v21.2d, v26.2d
    zip2 v26.2d, v21.2d, v26.2d
    zip1 v21.2d, v22.2d, v27.2d
    zip2 v27.2d, v22.2d, v27.2d
    zip1 v22.2d, v23.2d, v28.2d
    zip2 v28.2d, v23.2d, v28.2d
    zip1 v23.2d, v24.2d, v29.2d
    zip2 v29.2d, v24.2d, v29.2d
    zip1 v24.2d, v25.2d, v30.2d
    zip2 v30.2d, v25.2d, v30.2d
    zip1 v25.2d, v4.2d, v31.2d
    zip2 v31.2d, v4.2d, v31.2d
    b label_179
KAI_ASM_LABEL(label_178)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_179)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x23, #3, label_180
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x25, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_182
KAI_ASM_LABEL(label_180)  // Height 4: setup indirect input
    mov x22, #0
KAI_ASM_LABEL(label_181)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x22, LSL #0x3]
    ldr x5, [x2, x22, LSL #0x3]
    add x1, x1, x25, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    ldr x1, [x1, #0x18]
    cbnz x22, label_182
    add x6, x6, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_182)  // Height 4: input setup done
    cmp x5, #0x4
    blt label_185
    ld1 { v0.4s }, [x6], #0x10
    ld1 { v2.4s }, [x2], #0x10
    cmp x5, #0x8
    ld1 { v1.4s }, [x4], #0x10
    ld1 { v3.4s }, [x1], #0x10
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    blt label_184
KAI_ASM_LABEL(label_183)  // Height 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    cmp x5, #0x8
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    ld1 { v1.4s }, [x4], #0x10
    KAI_ASM_INST(0x4ea16862)  // bfcvtn2 v2.8h, v3.4s
    ld1 { v3.4s }, [x1], #0x10
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    ldr q4, [x8, #0]
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x20]
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    ld1 { v0.4s }, [x6], #0x10
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
    ld1 { v2.4s }, [x2], #0x10
    ldr q7, [x8, #0x30]
    bge label_183
KAI_ASM_LABEL(label_184)  // Height 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    sub x5, x5, #0x4
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x4ea16862)  // bfcvtn2 v2.8h, v3.4s
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
KAI_ASM_LABEL(label_185)  // Height 4: Multiply loop: Main loop skip
    cbz x5, label_188
    cbz x5, label_188
    tbz x5, #1, label_186
    ldr d0, [x6], #0x8
    ldr d1, [x4], #0x8
    ldr d2, [x2], #0x8
    ldr d3, [x1], #0x8
    tbz x5, #0, label_187
    ld1 { v0.s }[2], [x6]
    ld1 { v1.s }[2], [x4]
    ld1 { v2.s }[2], [x2]
    ld1 { v3.s }[2], [x1]
    b label_187
KAI_ASM_LABEL(label_186)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr s0, [x6, #0]
    ldr s1, [x4, #0]
    ldr s2, [x2, #0]
    ldr s3, [x1, #0]
KAI_ASM_LABEL(label_187)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q4, [x8, #0]
    ldr q5, [x8, #0x10]
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    ldr q6, [x8, #0x20]
    ldr q7, [x8, #0x30]
    KAI_ASM_INST(0x4ea16820)  // bfcvtn2 v0.8h, v1.4s
    KAI_ASM_INST(0x4ea16862)  // bfcvtn2 v2.8h, v3.4s
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x40]
    KAI_ASM_INST(0x6e45ec0e)  // bfmmla v14.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5a)  // bfmmla v26.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x50]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    ldr q6, [x8, #0x60]
    KAI_ASM_INST(0x6e47ec0f)  // bfmmla v15.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5b)  // bfmmla v27.4s, v2.8h, v7.8h
    ldr q7, [x8, #0x70]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    ldr q4, [x8, #0x80]
    KAI_ASM_INST(0x6e45ec10)  // bfmmla v16.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5c)  // bfmmla v28.4s, v2.8h, v5.8h
    ldr q5, [x8, #0x90]
    KAI_ASM_INST(0x6e46ec0b)  // bfmmla v11.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec57)  // bfmmla v23.4s, v2.8h, v6.8h
    ldr q6, [x8, #0xa0]
    KAI_ASM_INST(0x6e47ec11)  // bfmmla v17.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5d)  // bfmmla v29.4s, v2.8h, v7.8h
    ldr q7, [x8, #0xb0]
    add x8, x8, #0xc0
    KAI_ASM_INST(0x6e44ec0c)  // bfmmla v12.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e44ec58)  // bfmmla v24.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec12)  // bfmmla v18.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e45ec5e)  // bfmmla v30.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0d)  // bfmmla v13.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e46ec59)  // bfmmla v25.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec13)  // bfmmla v19.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e47ec5f)  // bfmmla v31.4s, v2.8h, v7.8h
KAI_ASM_LABEL(label_188)  // Height 4: Multiply loop: No odd multiplies
    tbz x23, #3, label_189
    ldr x1, [x0, #0x10]
    add x22, x22, #0x1
    cmp x22, x1
    bne label_181
KAI_ASM_LABEL(label_189)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    uzp1 v4.2d, v8.2d, v14.2d
    uzp2 v8.2d, v8.2d, v14.2d
    prfm pstl1keep, [x21, #0]
    uzp1 v14.2d, v9.2d, v15.2d
    uzp2 v9.2d, v9.2d, v15.2d
    uzp1 v15.2d, v10.2d, v16.2d
    uzp2 v10.2d, v10.2d, v16.2d
    add x5, x21, x1
    add x4, x5, x1
    add x3, x4, x1
    uzp1 v16.2d, v11.2d, v17.2d
    uzp2 v11.2d, v11.2d, v17.2d
    prfm pstl1keep, [x5, #0]
    uzp1 v17.2d, v12.2d, v18.2d
    uzp2 v12.2d, v12.2d, v18.2d
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    uzp1 v18.2d, v13.2d, v19.2d
    uzp2 v13.2d, v13.2d, v19.2d
    uzp1 v19.2d, v20.2d, v26.2d
    uzp2 v20.2d, v20.2d, v26.2d
    uzp1 v26.2d, v21.2d, v27.2d
    uzp2 v21.2d, v21.2d, v27.2d
    uzp1 v27.2d, v22.2d, v28.2d
    uzp2 v22.2d, v22.2d, v28.2d
    uzp1 v28.2d, v23.2d, v29.2d
    uzp2 v23.2d, v23.2d, v29.2d
    uzp1 v29.2d, v24.2d, v30.2d
    uzp2 v24.2d, v24.2d, v30.2d
    uzp1 v30.2d, v25.2d, v31.2d
    uzp2 v25.2d, v25.2d, v31.2d
    tbz x23, #1, label_190
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v0.4s }, [x2]
    ld1r { v31.4s }, [x1]
    fmin v4.4s, v4.4s, v0.4s
    fmin v14.4s, v14.4s, v0.4s
    fmin v15.4s, v15.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v8.4s, v8.4s, v0.4s
    fmin v9.4s, v9.4s, v0.4s
    fmin v10.4s, v10.4s, v0.4s
    fmin v11.4s, v11.4s, v0.4s
    fmin v12.4s, v12.4s, v0.4s
    fmin v13.4s, v13.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmax v4.4s, v4.4s, v31.4s
    fmax v14.4s, v14.4s, v31.4s
    fmax v15.4s, v15.4s, v31.4s
    fmax v16.4s, v16.4s, v31.4s
    fmax v17.4s, v17.4s, v31.4s
    fmax v18.4s, v18.4s, v31.4s
    fmax v8.4s, v8.4s, v31.4s
    fmax v9.4s, v9.4s, v31.4s
    fmax v10.4s, v10.4s, v31.4s
    fmax v11.4s, v11.4s, v31.4s
    fmax v12.4s, v12.4s, v31.4s
    fmax v13.4s, v13.4s, v31.4s
    fmax v19.4s, v19.4s, v31.4s
    fmax v26.4s, v26.4s, v31.4s
    fmax v27.4s, v27.4s, v31.4s
    fmax v28.4s, v28.4s, v31.4s
    fmax v29.4s, v29.4s, v31.4s
    fmax v30.4s, v30.4s, v31.4s
    fmax v20.4s, v20.4s, v31.4s
    fmax v21.4s, v21.4s, v31.4s
    fmax v22.4s, v22.4s, v31.4s
    fmax v23.4s, v23.4s, v31.4s
    fmax v24.4s, v24.4s, v31.4s
    fmax v25.4s, v25.4s, v31.4s
KAI_ASM_LABEL(label_190)  // Height 4: No activation
    cmp x20, #0x18
    bge label_203
    tbz x20, #4, label_194
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v15.4s }, [x21], #0x10
    st1 { v16.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    st1 { v10.4s }, [x5], #0x10
    st1 { v11.4s }, [x5], #0x10
    st1 { v19.4s }, [x4], #0x10
    st1 { v26.4s }, [x4], #0x10
    st1 { v27.4s }, [x4], #0x10
    st1 { v28.4s }, [x4], #0x10
    st1 { v20.4s }, [x3], #0x10
    st1 { v21.4s }, [x3], #0x10
    st1 { v22.4s }, [x3], #0x10
    st1 { v23.4s }, [x3], #0x10
    tbz x20, #2, label_192
    st1 { v17.4s }, [x21], #0x10
    st1 { v12.4s }, [x5], #0x10
    st1 { v29.4s }, [x4], #0x10
    st1 { v24.4s }, [x3], #0x10
    tbz x20, #1, label_191
    str d18, [x21], #0x8
    str d13, [x5], #0x8
    str d30, [x4], #0x8
    str d25, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v18.s }[2], [x21]
    st1 { v13.s }[2], [x5]
    st1 { v30.s }[2], [x4]
    st1 { v25.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_191)  // Height 4: Partial direct writeback: partial_0_20
    tbz x20, #0, label_202
    str s18, [x21, #0]
    str s13, [x5, #0]
    str s30, [x4, #0]
    str s25, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_192)  // Height 4: Partial direct writeback: partial_1_16
    tbz x20, #1, label_193
    str d17, [x21], #0x8
    str d12, [x5], #0x8
    str d29, [x4], #0x8
    str d24, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v17.s }[2], [x21]
    st1 { v12.s }[2], [x5]
    st1 { v29.s }[2], [x4]
    st1 { v24.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_193)  // Height 4: Partial direct writeback: partial_0_16
    tbz x20, #0, label_202
    str s17, [x21, #0]
    str s12, [x5, #0]
    str s29, [x4, #0]
    str s24, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_194)  // Height 4: Partial direct writeback: partial_3_0
    tbz x20, #3, label_198
    st1 { v4.4s }, [x21], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v9.4s }, [x5], #0x10
    st1 { v19.4s }, [x4], #0x10
    st1 { v26.4s }, [x4], #0x10
    st1 { v20.4s }, [x3], #0x10
    st1 { v21.4s }, [x3], #0x10
    tbz x20, #2, label_196
    st1 { v15.4s }, [x21], #0x10
    st1 { v10.4s }, [x5], #0x10
    st1 { v27.4s }, [x4], #0x10
    st1 { v22.4s }, [x3], #0x10
    tbz x20, #1, label_195
    str d16, [x21], #0x8
    str d11, [x5], #0x8
    str d28, [x4], #0x8
    str d23, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v16.s }[2], [x21]
    st1 { v11.s }[2], [x5]
    st1 { v28.s }[2], [x4]
    st1 { v23.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_195)  // Height 4: Partial direct writeback: partial_0_12
    tbz x20, #0, label_202
    str s16, [x21, #0]
    str s11, [x5, #0]
    str s28, [x4, #0]
    str s23, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_196)  // Height 4: Partial direct writeback: partial_1_8
    tbz x20, #1, label_197
    str d15, [x21], #0x8
    str d10, [x5], #0x8
    str d27, [x4], #0x8
    str d22, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v15.s }[2], [x21]
    st1 { v10.s }[2], [x5]
    st1 { v27.s }[2], [x4]
    st1 { v22.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_197)  // Height 4: Partial direct writeback: partial_0_8
    tbz x20, #0, label_202
    str s15, [x21, #0]
    str s10, [x5, #0]
    str s27, [x4, #0]
    str s22, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_198)  // Height 4: Partial direct writeback: partial_2_0
    tbz x20, #2, label_200
    st1 { v4.4s }, [x21], #0x10
    st1 { v8.4s }, [x5], #0x10
    st1 { v19.4s }, [x4], #0x10
    st1 { v20.4s }, [x3], #0x10
    tbz x20, #1, label_199
    str d14, [x21], #0x8
    str d9, [x5], #0x8
    str d26, [x4], #0x8
    str d21, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v14.s }[2], [x21]
    st1 { v9.s }[2], [x5]
    st1 { v26.s }[2], [x4]
    st1 { v21.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_199)  // Height 4: Partial direct writeback: partial_0_4
    tbz x20, #0, label_202
    str s14, [x21, #0]
    str s9, [x5, #0]
    str s26, [x4, #0]
    str s21, [x3, #0]
    b label_202
KAI_ASM_LABEL(label_200)  // Height 4: Partial direct writeback: partial_1_0
    tbz x20, #1, label_201
    str d4, [x21], #0x8
    str d8, [x5], #0x8
    str d19, [x4], #0x8
    str d20, [x3], #0x8
    tbz x20, #0, label_202
    st1 { v4.s }[2], [x21]
    st1 { v8.s }[2], [x5]
    st1 { v19.s }[2], [x4]
    st1 { v20.s }[2], [x3]
    b label_202
KAI_ASM_LABEL(label_201)  // Height 4: Partial direct writeback: partial_0_0
    str s4, [x21, #0]
    str s8, [x5, #0]
    str s19, [x4, #0]
    str s20, [x3, #0]
KAI_ASM_LABEL(label_202)  // Height 4: Partial direct writeback: Done
    b label_204
KAI_ASM_LABEL(label_203)  // Height 4: Full writeback
    str q4, [x21, #0]
    str q14, [x21, #0x10]
    str q15, [x21, #0x20]
    str q16, [x21, #0x30]
    str q17, [x21, #0x40]
    str q18, [x21, #0x50]
    add x21, x21, #0x60
    str q8, [x5, #0]
    str q9, [x5, #0x10]
    str q10, [x5, #0x20]
    str q11, [x5, #0x30]
    str q12, [x5, #0x40]
    str q13, [x5, #0x50]
    str q19, [x4, #0]
    str q26, [x4, #0x10]
    str q27, [x4, #0x20]
    str q28, [x4, #0x30]
    str q29, [x4, #0x40]
    str q30, [x4, #0x50]
    str q20, [x3, #0]
    str q21, [x3, #0x10]
    str q22, [x3, #0x20]
    str q23, [x3, #0x30]
    str q24, [x3, #0x40]
    str q25, [x3, #0x50]
KAI_ASM_LABEL(label_204)  // Height 4: Writeback done
    subs x20, x20, #0x18
    bgt label_155
    add x25, x25, #0x4
    cmp x25, x24
    blt label_1
KAI_ASM_LABEL(label_205)  // Exit
    add SP, SP, #0x60
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_fp32bf16fp32_mmla_4x24)

    KAI_ASM_END
