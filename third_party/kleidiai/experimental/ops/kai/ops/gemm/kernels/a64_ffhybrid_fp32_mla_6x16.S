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

    KAI_ASM_CODE(a64_ffhybrid_fp32_mla_6x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_ffhybrid_fp32_mla_6x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_ffhybrid_fp32_mla_6x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_ffhybrid_fp32_mla_6x16)
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
    bge label_211
    cmp x1, #0x4
    bgt label_169
    beq label_127
    cmp x1, #0x2
    bgt label_85
    beq label_43
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
    b label_22
KAI_ASM_LABEL(label_11)  // Height 1: no bias
    tbz x9, #0, label_21
    cmp x26, #0x10
    bge label_20
    tbz x26, #3, label_15
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v9.4s }, [x27], #0x10
    tbz x26, #2, label_13
    ld1 { v10.4s }, [x27], #0x10
    tbz x26, #1, label_12
    ldr d11, [x27], #0x8
    mov x1, #0x38
    tbz x26, #0, label_19
    ld1 { v11.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_19
    ldr s11, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_1_8
    tbz x26, #1, label_14
    ldr d10, [x27], #0x8
    mov x1, #0x28
    tbz x26, #0, label_19
    ld1 { v10.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_19
    ldr s10, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_2_0
    tbz x26, #2, label_17
    ld1 { v8.4s }, [x27], #0x10
    tbz x26, #1, label_16
    ldr d9, [x27], #0x8
    mov x1, #0x18
    tbz x26, #0, label_19
    ld1 { v9.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_19
    ldr s9, [x27, #0]
    b label_19
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_1_0
    tbz x26, #1, label_18
    ldr d8, [x27], #0x8
    mov x1, #0x8
    tbz x26, #0, label_19
    ld1 { v8.s }[2], [x27]
    b label_19
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    mov x1, #0
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: Done
    sub x27, x27, x1
    b label_22
KAI_ASM_LABEL(label_20)  // Height 1: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
    b label_22
KAI_ASM_LABEL(label_21)  // Height 1: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
KAI_ASM_LABEL(label_22)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_23
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    b label_25
KAI_ASM_LABEL(label_23)  // Height 1: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_24)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    cbnz x28, label_25
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_25)  // Height 1: input setup done
    cmp x7, #0x4
    blt label_28
    ldr q0, [x8, #0]
    ldr q6, [x23, #0]
    cmp x7, #0x8
    ldr q7, [x22, #0]
    blt label_27
KAI_ASM_LABEL(label_26)  // Height 1: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr q7, [x20, #0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    cmp x7, #0x8
    fmla v10.4s, v6.4s, v0.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    ldr q7, [x22, #0]
    bge label_26
KAI_ASM_LABEL(label_27)  // Height 1: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr q7, [x20, #0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
KAI_ASM_LABEL(label_28)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_30
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr q6, [x23, #0]
    sub x7, x7, #0x1
    add x23, x23, #0x10
    ldr q7, [x22, #0]
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    cbnz x7, label_29
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: No odd multiplies
    tbz x9, #3, label_31
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_24
KAI_ASM_LABEL(label_31)  // Height 1: no string loop
    tbz x9, #1, label_32
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
KAI_ASM_LABEL(label_32)  // Height 1: No activation
    cmp x26, #0x10
    bge label_41
    tbz x26, #3, label_36
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    tbz x26, #2, label_34
    st1 { v10.4s }, [x27], #0x10
    tbz x26, #1, label_33
    str d11, [x27], #0x8
    tbz x26, #0, label_40
    st1 { v11.s }[2], [x27]
    b label_40
KAI_ASM_LABEL(label_33)  // Height 1: Partial direct writeback: partial_0_12
    tbz x26, #0, label_40
    str s11, [x27, #0]
    b label_40
KAI_ASM_LABEL(label_34)  // Height 1: Partial direct writeback: partial_1_8
    tbz x26, #1, label_35
    str d10, [x27], #0x8
    tbz x26, #0, label_40
    st1 { v10.s }[2], [x27]
    b label_40
KAI_ASM_LABEL(label_35)  // Height 1: Partial direct writeback: partial_0_8
    tbz x26, #0, label_40
    str s10, [x27, #0]
    b label_40
KAI_ASM_LABEL(label_36)  // Height 1: Partial direct writeback: partial_2_0
    tbz x26, #2, label_38
    st1 { v8.4s }, [x27], #0x10
    tbz x26, #1, label_37
    str d9, [x27], #0x8
    tbz x26, #0, label_40
    st1 { v9.s }[2], [x27]
    b label_40
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_0_4
    tbz x26, #0, label_40
    str s9, [x27, #0]
    b label_40
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_1_0
    tbz x26, #1, label_39
    str d8, [x27], #0x8
    tbz x26, #0, label_40
    st1 { v8.s }[2], [x27]
    b label_40
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: Done
    b label_42
KAI_ASM_LABEL(label_41)  // Height 1: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
KAI_ASM_LABEL(label_42)  // Height 1: Writeback done
    subs x26, x26, #0x10
    bgt label_2
    b label_253
KAI_ASM_LABEL(label_43)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_44)  // Height 2: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_45
    cmp x26, #0x8
    mov x20, x23
    bgt label_45
    cmp x26, #0x4
    mov x21, x23
    bgt label_45
    mov x22, x23
KAI_ASM_LABEL(label_45)  // Height 2: B setup done
    cbz x24, label_53
    cmp x26, #0x10
    bge label_51
    cbnz x11, label_50
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_47
KAI_ASM_LABEL(label_46)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_46
KAI_ASM_LABEL(label_47)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_49
KAI_ASM_LABEL(label_48)  // Height 2: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_48
KAI_ASM_LABEL(label_49)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_50)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_52
KAI_ASM_LABEL(label_51)  // Height 2: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_52)  // Height 2: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    b label_64
KAI_ASM_LABEL(label_53)  // Height 2: no bias
    tbz x9, #0, label_63
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    bge label_62
    tbz x26, #3, label_57
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    tbz x26, #2, label_55
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    tbz x26, #1, label_54
    ldr d11, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    tbz x26, #0, label_61
    ld1 { v11.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    b label_61
KAI_ASM_LABEL(label_54)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_61
    ldr s11, [x27, #0]
    ldr s15, [x7, #0]
    b label_61
KAI_ASM_LABEL(label_55)  // Height 2: Partial accumulate: partial_1_8
    tbz x26, #1, label_56
    ldr d10, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    tbz x26, #0, label_61
    ld1 { v10.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    b label_61
KAI_ASM_LABEL(label_56)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_61
    ldr s10, [x27, #0]
    ldr s14, [x7, #0]
    b label_61
KAI_ASM_LABEL(label_57)  // Height 2: Partial accumulate: partial_2_0
    tbz x26, #2, label_59
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    tbz x26, #1, label_58
    ldr d9, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    tbz x26, #0, label_61
    ld1 { v9.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    b label_61
KAI_ASM_LABEL(label_58)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_61
    ldr s9, [x27, #0]
    ldr s13, [x7, #0]
    b label_61
KAI_ASM_LABEL(label_59)  // Height 2: Partial accumulate: partial_1_0
    tbz x26, #1, label_60
    ldr d8, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    tbz x26, #0, label_61
    ld1 { v8.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    b label_61
KAI_ASM_LABEL(label_60)  // Height 2: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
KAI_ASM_LABEL(label_61)  // Height 2: Partial accumulate: Done
    sub x27, x27, x1
    b label_64
KAI_ASM_LABEL(label_62)  // Height 2: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    b label_64
KAI_ASM_LABEL(label_63)  // Height 2: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
KAI_ASM_LABEL(label_64)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_65
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    b label_67
KAI_ASM_LABEL(label_65)  // Height 2: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_66)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x28, label_67
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_67)  // Height 2: input setup done
    cmp x7, #0x4
    blt label_70
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x8
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    blt label_69
KAI_ASM_LABEL(label_68)  // Height 2: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr q6, [x21, #0]
    sub x7, x7, #0x4
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    ldr q7, [x20, #0]
    add x8, x8, #0x10
    cmp x7, #0x8
    add x6, x6, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q1, [x6, #0]
    ldr q7, [x22, #0]
    bge label_68
KAI_ASM_LABEL(label_69)  // Height 2: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr q6, [x21, #0]
    sub x7, x7, #0x4
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    ldr q7, [x20, #0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
KAI_ASM_LABEL(label_70)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_72
KAI_ASM_LABEL(label_71)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x6], #0x4
    sub x7, x7, #0x1
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    cbnz x7, label_71
KAI_ASM_LABEL(label_72)  // Height 2: Multiply loop: No odd multiplies
    tbz x9, #3, label_73
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_66
KAI_ASM_LABEL(label_73)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    tbz x9, #1, label_74
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
    fmin v14.4s, v14.4s, v17.4s
    fmin v15.4s, v15.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
    fmax v12.4s, v12.4s, v16.4s
    fmax v13.4s, v13.4s, v16.4s
    fmax v14.4s, v14.4s, v16.4s
    fmax v15.4s, v15.4s, v16.4s
KAI_ASM_LABEL(label_74)  // Height 2: No activation
    cmp x26, #0x10
    bge label_83
    tbz x26, #3, label_78
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v13.4s }, [x7], #0x10
    tbz x26, #2, label_76
    st1 { v10.4s }, [x27], #0x10
    st1 { v14.4s }, [x7], #0x10
    tbz x26, #1, label_75
    str d11, [x27], #0x8
    str d15, [x7], #0x8
    tbz x26, #0, label_82
    st1 { v11.s }[2], [x27]
    st1 { v15.s }[2], [x7]
    b label_82
KAI_ASM_LABEL(label_75)  // Height 2: Partial direct writeback: partial_0_12
    tbz x26, #0, label_82
    str s11, [x27, #0]
    str s15, [x7, #0]
    b label_82
KAI_ASM_LABEL(label_76)  // Height 2: Partial direct writeback: partial_1_8
    tbz x26, #1, label_77
    str d10, [x27], #0x8
    str d14, [x7], #0x8
    tbz x26, #0, label_82
    st1 { v10.s }[2], [x27]
    st1 { v14.s }[2], [x7]
    b label_82
KAI_ASM_LABEL(label_77)  // Height 2: Partial direct writeback: partial_0_8
    tbz x26, #0, label_82
    str s10, [x27, #0]
    str s14, [x7, #0]
    b label_82
KAI_ASM_LABEL(label_78)  // Height 2: Partial direct writeback: partial_2_0
    tbz x26, #2, label_80
    st1 { v8.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    tbz x26, #1, label_79
    str d9, [x27], #0x8
    str d13, [x7], #0x8
    tbz x26, #0, label_82
    st1 { v9.s }[2], [x27]
    st1 { v13.s }[2], [x7]
    b label_82
KAI_ASM_LABEL(label_79)  // Height 2: Partial direct writeback: partial_0_4
    tbz x26, #0, label_82
    str s9, [x27, #0]
    str s13, [x7, #0]
    b label_82
KAI_ASM_LABEL(label_80)  // Height 2: Partial direct writeback: partial_1_0
    tbz x26, #1, label_81
    str d8, [x27], #0x8
    str d12, [x7], #0x8
    tbz x26, #0, label_82
    st1 { v8.s }[2], [x27]
    st1 { v12.s }[2], [x7]
    b label_82
KAI_ASM_LABEL(label_81)  // Height 2: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
    str s12, [x7, #0]
KAI_ASM_LABEL(label_82)  // Height 2: Partial direct writeback: Done
    b label_84
KAI_ASM_LABEL(label_83)  // Height 2: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
KAI_ASM_LABEL(label_84)  // Height 2: Writeback done
    subs x26, x26, #0x10
    bgt label_44
    b label_253
KAI_ASM_LABEL(label_85)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_86)  // Height 3: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_87
    cmp x26, #0x8
    mov x20, x23
    bgt label_87
    cmp x26, #0x4
    mov x21, x23
    bgt label_87
    mov x22, x23
KAI_ASM_LABEL(label_87)  // Height 3: B setup done
    cbz x24, label_95
    cmp x26, #0x10
    bge label_93
    cbnz x11, label_92
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_89
KAI_ASM_LABEL(label_88)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_88
KAI_ASM_LABEL(label_89)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_91
KAI_ASM_LABEL(label_90)  // Height 3: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_90
KAI_ASM_LABEL(label_91)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_92)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_94
KAI_ASM_LABEL(label_93)  // Height 3: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_94)  // Height 3: bias load done
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    b label_106
KAI_ASM_LABEL(label_95)  // Height 3: no bias
    tbz x9, #0, label_105
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    bge label_104
    tbz x26, #3, label_99
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    tbz x26, #2, label_97
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    tbz x26, #1, label_96
    ldr d11, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d19, [x6], #0x8
    tbz x26, #0, label_103
    ld1 { v11.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    b label_103
KAI_ASM_LABEL(label_96)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_103
    ldr s11, [x27, #0]
    ldr s15, [x7, #0]
    ldr s19, [x6, #0]
    b label_103
KAI_ASM_LABEL(label_97)  // Height 3: Partial accumulate: partial_1_8
    tbz x26, #1, label_98
    ldr d10, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d18, [x6], #0x8
    tbz x26, #0, label_103
    ld1 { v10.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    b label_103
KAI_ASM_LABEL(label_98)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_103
    ldr s10, [x27, #0]
    ldr s14, [x7, #0]
    ldr s18, [x6, #0]
    b label_103
KAI_ASM_LABEL(label_99)  // Height 3: Partial accumulate: partial_2_0
    tbz x26, #2, label_101
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    tbz x26, #1, label_100
    ldr d9, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d17, [x6], #0x8
    tbz x26, #0, label_103
    ld1 { v9.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    b label_103
KAI_ASM_LABEL(label_100)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_103
    ldr s9, [x27, #0]
    ldr s13, [x7, #0]
    ldr s17, [x6, #0]
    b label_103
KAI_ASM_LABEL(label_101)  // Height 3: Partial accumulate: partial_1_0
    tbz x26, #1, label_102
    ldr d8, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d16, [x6], #0x8
    tbz x26, #0, label_103
    ld1 { v8.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v16.s }[2], [x6]
    b label_103
KAI_ASM_LABEL(label_102)  // Height 3: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s16, [x6, #0]
KAI_ASM_LABEL(label_103)  // Height 3: Partial accumulate: Done
    sub x27, x27, x1
    b label_106
KAI_ASM_LABEL(label_104)  // Height 3: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
    ldr q12, [x7, #0]
    ldr q13, [x7, #0x10]
    ldr q14, [x7, #0x20]
    ldr q15, [x7, #0x30]
    ldr q16, [x6, #0]
    ldr q17, [x6, #0x10]
    ldr q18, [x6, #0x20]
    ldr q19, [x6, #0x30]
    b label_106
KAI_ASM_LABEL(label_105)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_106)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_107
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_109
KAI_ASM_LABEL(label_107)  // Height 3: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_108)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x28, label_109
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_109)  // Height 3: input setup done
    cmp x7, #0x4
    blt label_112
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x8
    ldr q2, [x5, #0]
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    blt label_111
KAI_ASM_LABEL(label_110)  // Height 3: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    cmp x7, #0x8
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    ldr q7, [x20, #0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q1, [x6, #0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr q2, [x5, #0]
    ldr q7, [x22, #0]
    bge label_110
KAI_ASM_LABEL(label_111)  // Height 3: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    add x6, x6, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    ldr q7, [x20, #0]
    add x5, x5, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
KAI_ASM_LABEL(label_112)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_114
KAI_ASM_LABEL(label_113)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x6], #0x4
    sub x7, x7, #0x1
    ldr s2, [x5], #0x4
    ldr q6, [x23, #0]
    add x23, x23, #0x10
    ldr q7, [x22, #0]
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    cbnz x7, label_113
KAI_ASM_LABEL(label_114)  // Height 3: Multiply loop: No odd multiplies
    tbz x9, #3, label_115
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_108
KAI_ASM_LABEL(label_115)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    tbz x9, #1, label_116
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
KAI_ASM_LABEL(label_116)  // Height 3: No activation
    cmp x26, #0x10
    bge label_125
    tbz x26, #3, label_120
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v13.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v17.4s }, [x6], #0x10
    tbz x26, #2, label_118
    st1 { v10.4s }, [x27], #0x10
    st1 { v14.4s }, [x7], #0x10
    st1 { v18.4s }, [x6], #0x10
    tbz x26, #1, label_117
    str d11, [x27], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    tbz x26, #0, label_124
    st1 { v11.s }[2], [x27]
    st1 { v15.s }[2], [x7]
    st1 { v19.s }[2], [x6]
    b label_124
KAI_ASM_LABEL(label_117)  // Height 3: Partial direct writeback: partial_0_12
    tbz x26, #0, label_124
    str s11, [x27, #0]
    str s15, [x7, #0]
    str s19, [x6, #0]
    b label_124
KAI_ASM_LABEL(label_118)  // Height 3: Partial direct writeback: partial_1_8
    tbz x26, #1, label_119
    str d10, [x27], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    tbz x26, #0, label_124
    st1 { v10.s }[2], [x27]
    st1 { v14.s }[2], [x7]
    st1 { v18.s }[2], [x6]
    b label_124
KAI_ASM_LABEL(label_119)  // Height 3: Partial direct writeback: partial_0_8
    tbz x26, #0, label_124
    str s10, [x27, #0]
    str s14, [x7, #0]
    str s18, [x6, #0]
    b label_124
KAI_ASM_LABEL(label_120)  // Height 3: Partial direct writeback: partial_2_0
    tbz x26, #2, label_122
    st1 { v8.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    tbz x26, #1, label_121
    str d9, [x27], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    tbz x26, #0, label_124
    st1 { v9.s }[2], [x27]
    st1 { v13.s }[2], [x7]
    st1 { v17.s }[2], [x6]
    b label_124
KAI_ASM_LABEL(label_121)  // Height 3: Partial direct writeback: partial_0_4
    tbz x26, #0, label_124
    str s9, [x27, #0]
    str s13, [x7, #0]
    str s17, [x6, #0]
    b label_124
KAI_ASM_LABEL(label_122)  // Height 3: Partial direct writeback: partial_1_0
    tbz x26, #1, label_123
    str d8, [x27], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    tbz x26, #0, label_124
    st1 { v8.s }[2], [x27]
    st1 { v12.s }[2], [x7]
    st1 { v16.s }[2], [x6]
    b label_124
KAI_ASM_LABEL(label_123)  // Height 3: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
    str s12, [x7, #0]
    str s16, [x6, #0]
KAI_ASM_LABEL(label_124)  // Height 3: Partial direct writeback: Done
    b label_126
KAI_ASM_LABEL(label_125)  // Height 3: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q14, [x7, #0x20]
    str q15, [x7, #0x30]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q18, [x6, #0x20]
    str q19, [x6, #0x30]
KAI_ASM_LABEL(label_126)  // Height 3: Writeback done
    subs x26, x26, #0x10
    bgt label_86
    b label_253
KAI_ASM_LABEL(label_127)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_128)  // Height 4: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_129
    cmp x26, #0x8
    mov x20, x23
    bgt label_129
    cmp x26, #0x4
    mov x21, x23
    bgt label_129
    mov x22, x23
KAI_ASM_LABEL(label_129)  // Height 4: B setup done
    cbz x24, label_137
    cmp x26, #0x10
    bge label_135
    cbnz x11, label_134
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_131
KAI_ASM_LABEL(label_130)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_130
KAI_ASM_LABEL(label_131)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_133
KAI_ASM_LABEL(label_132)  // Height 4: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_132
KAI_ASM_LABEL(label_133)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_134)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_136
KAI_ASM_LABEL(label_135)  // Height 4: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_136)  // Height 4: bias load done
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
    b label_148
KAI_ASM_LABEL(label_137)  // Height 4: no bias
    tbz x9, #0, label_147
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    bge label_146
    tbz x26, #3, label_141
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    tbz x26, #2, label_139
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    tbz x26, #1, label_138
    ldr d11, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    tbz x26, #0, label_145
    ld1 { v11.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    b label_145
KAI_ASM_LABEL(label_138)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_145
    ldr s11, [x27, #0]
    ldr s15, [x7, #0]
    ldr s19, [x6, #0]
    ldr s23, [x5, #0]
    b label_145
KAI_ASM_LABEL(label_139)  // Height 4: Partial accumulate: partial_1_8
    tbz x26, #1, label_140
    ldr d10, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    tbz x26, #0, label_145
    ld1 { v10.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    b label_145
KAI_ASM_LABEL(label_140)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_145
    ldr s10, [x27, #0]
    ldr s14, [x7, #0]
    ldr s18, [x6, #0]
    ldr s22, [x5, #0]
    b label_145
KAI_ASM_LABEL(label_141)  // Height 4: Partial accumulate: partial_2_0
    tbz x26, #2, label_143
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    tbz x26, #1, label_142
    ldr d9, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    tbz x26, #0, label_145
    ld1 { v9.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    b label_145
KAI_ASM_LABEL(label_142)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_145
    ldr s9, [x27, #0]
    ldr s13, [x7, #0]
    ldr s17, [x6, #0]
    ldr s21, [x5, #0]
    b label_145
KAI_ASM_LABEL(label_143)  // Height 4: Partial accumulate: partial_1_0
    tbz x26, #1, label_144
    ldr d8, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    tbz x26, #0, label_145
    ld1 { v8.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v16.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    b label_145
KAI_ASM_LABEL(label_144)  // Height 4: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s16, [x6, #0]
    ldr s20, [x5, #0]
KAI_ASM_LABEL(label_145)  // Height 4: Partial accumulate: Done
    sub x27, x27, x1
    b label_148
KAI_ASM_LABEL(label_146)  // Height 4: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
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
    b label_148
KAI_ASM_LABEL(label_147)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_148)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_149
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_151
KAI_ASM_LABEL(label_149)  // Height 4: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_150)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x28, LSL #0x3]
    ldr x7, [x2, x28, LSL #0x3]
    add x1, x1, x11, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x28, label_151
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_151)  // Height 4: input setup done
    cmp x7, #0x4
    blt label_154
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x8
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    blt label_153
KAI_ASM_LABEL(label_152)  // Height 4: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    ldr q6, [x21, #0]
    cmp x7, #0x8
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    ldr q7, [x20, #0]
    add x4, x4, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q1, [x6, #0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr q2, [x5, #0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr q3, [x4, #0]
    ldr q7, [x22, #0]
    bge label_152
KAI_ASM_LABEL(label_153)  // Height 4: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    ldr q6, [x21, #0]
    add x6, x6, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    ldr q7, [x20, #0]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
KAI_ASM_LABEL(label_154)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_156
KAI_ASM_LABEL(label_155)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x6], #0x4
    sub x7, x7, #0x1
    ldr s2, [x5], #0x4
    ldr s3, [x4], #0x4
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    cbnz x7, label_155
KAI_ASM_LABEL(label_156)  // Height 4: Multiply loop: No odd multiplies
    tbz x9, #3, label_157
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_150
KAI_ASM_LABEL(label_157)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x9, #1, label_158
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v25.4s }, [x2]
    ld1r { v24.4s }, [x1]
    fmin v8.4s, v8.4s, v25.4s
    fmin v9.4s, v9.4s, v25.4s
    fmin v10.4s, v10.4s, v25.4s
    fmin v11.4s, v11.4s, v25.4s
    fmin v12.4s, v12.4s, v25.4s
    fmin v13.4s, v13.4s, v25.4s
    fmin v14.4s, v14.4s, v25.4s
    fmin v15.4s, v15.4s, v25.4s
    fmin v16.4s, v16.4s, v25.4s
    fmin v17.4s, v17.4s, v25.4s
    fmin v18.4s, v18.4s, v25.4s
    fmin v19.4s, v19.4s, v25.4s
    fmin v20.4s, v20.4s, v25.4s
    fmin v21.4s, v21.4s, v25.4s
    fmin v22.4s, v22.4s, v25.4s
    fmin v23.4s, v23.4s, v25.4s
    fmax v8.4s, v8.4s, v24.4s
    fmax v9.4s, v9.4s, v24.4s
    fmax v10.4s, v10.4s, v24.4s
    fmax v11.4s, v11.4s, v24.4s
    fmax v12.4s, v12.4s, v24.4s
    fmax v13.4s, v13.4s, v24.4s
    fmax v14.4s, v14.4s, v24.4s
    fmax v15.4s, v15.4s, v24.4s
    fmax v16.4s, v16.4s, v24.4s
    fmax v17.4s, v17.4s, v24.4s
    fmax v18.4s, v18.4s, v24.4s
    fmax v19.4s, v19.4s, v24.4s
    fmax v20.4s, v20.4s, v24.4s
    fmax v21.4s, v21.4s, v24.4s
    fmax v22.4s, v22.4s, v24.4s
    fmax v23.4s, v23.4s, v24.4s
KAI_ASM_LABEL(label_158)  // Height 4: No activation
    cmp x26, #0x10
    bge label_167
    tbz x26, #3, label_162
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v13.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v17.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v21.4s }, [x5], #0x10
    tbz x26, #2, label_160
    st1 { v10.4s }, [x27], #0x10
    st1 { v14.4s }, [x7], #0x10
    st1 { v18.4s }, [x6], #0x10
    st1 { v22.4s }, [x5], #0x10
    tbz x26, #1, label_159
    str d11, [x27], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    tbz x26, #0, label_166
    st1 { v11.s }[2], [x27]
    st1 { v15.s }[2], [x7]
    st1 { v19.s }[2], [x6]
    st1 { v23.s }[2], [x5]
    b label_166
KAI_ASM_LABEL(label_159)  // Height 4: Partial direct writeback: partial_0_12
    tbz x26, #0, label_166
    str s11, [x27, #0]
    str s15, [x7, #0]
    str s19, [x6, #0]
    str s23, [x5, #0]
    b label_166
KAI_ASM_LABEL(label_160)  // Height 4: Partial direct writeback: partial_1_8
    tbz x26, #1, label_161
    str d10, [x27], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    tbz x26, #0, label_166
    st1 { v10.s }[2], [x27]
    st1 { v14.s }[2], [x7]
    st1 { v18.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    b label_166
KAI_ASM_LABEL(label_161)  // Height 4: Partial direct writeback: partial_0_8
    tbz x26, #0, label_166
    str s10, [x27, #0]
    str s14, [x7, #0]
    str s18, [x6, #0]
    str s22, [x5, #0]
    b label_166
KAI_ASM_LABEL(label_162)  // Height 4: Partial direct writeback: partial_2_0
    tbz x26, #2, label_164
    st1 { v8.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    tbz x26, #1, label_163
    str d9, [x27], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    tbz x26, #0, label_166
    st1 { v9.s }[2], [x27]
    st1 { v13.s }[2], [x7]
    st1 { v17.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    b label_166
KAI_ASM_LABEL(label_163)  // Height 4: Partial direct writeback: partial_0_4
    tbz x26, #0, label_166
    str s9, [x27, #0]
    str s13, [x7, #0]
    str s17, [x6, #0]
    str s21, [x5, #0]
    b label_166
KAI_ASM_LABEL(label_164)  // Height 4: Partial direct writeback: partial_1_0
    tbz x26, #1, label_165
    str d8, [x27], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    tbz x26, #0, label_166
    st1 { v8.s }[2], [x27]
    st1 { v12.s }[2], [x7]
    st1 { v16.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    b label_166
KAI_ASM_LABEL(label_165)  // Height 4: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
    str s12, [x7, #0]
    str s16, [x6, #0]
    str s20, [x5, #0]
KAI_ASM_LABEL(label_166)  // Height 4: Partial direct writeback: Done
    b label_168
KAI_ASM_LABEL(label_167)  // Height 4: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
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
KAI_ASM_LABEL(label_168)  // Height 4: Writeback done
    subs x26, x26, #0x10
    bgt label_128
    b label_253
KAI_ASM_LABEL(label_169)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_170)  // Height 5: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_171
    cmp x26, #0x8
    mov x20, x23
    bgt label_171
    cmp x26, #0x4
    mov x21, x23
    bgt label_171
    mov x22, x23
KAI_ASM_LABEL(label_171)  // Height 5: B setup done
    cbz x24, label_179
    cmp x26, #0x10
    bge label_177
    cbnz x11, label_176
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_173
KAI_ASM_LABEL(label_172)  // Height 5: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_172
KAI_ASM_LABEL(label_173)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_175
KAI_ASM_LABEL(label_174)  // Height 5: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_174
KAI_ASM_LABEL(label_175)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_176)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_178
KAI_ASM_LABEL(label_177)  // Height 5: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_178)  // Height 5: bias load done
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
    b label_190
KAI_ASM_LABEL(label_179)  // Height 5: no bias
    tbz x9, #0, label_189
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_188
    tbz x26, #3, label_183
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    tbz x26, #2, label_181
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    ld1 { v26.4s }, [x4], #0x10
    tbz x26, #1, label_180
    ldr d11, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d27, [x4], #0x8
    tbz x26, #0, label_187
    ld1 { v11.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    ld1 { v27.s }[2], [x4]
    b label_187
KAI_ASM_LABEL(label_180)  // Height 5: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_187
    ldr s11, [x27, #0]
    ldr s15, [x7, #0]
    ldr s19, [x6, #0]
    ldr s23, [x5, #0]
    ldr s27, [x4, #0]
    b label_187
KAI_ASM_LABEL(label_181)  // Height 5: Partial accumulate: partial_1_8
    tbz x26, #1, label_182
    ldr d10, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d26, [x4], #0x8
    tbz x26, #0, label_187
    ld1 { v10.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    ld1 { v26.s }[2], [x4]
    b label_187
KAI_ASM_LABEL(label_182)  // Height 5: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_187
    ldr s10, [x27, #0]
    ldr s14, [x7, #0]
    ldr s18, [x6, #0]
    ldr s22, [x5, #0]
    ldr s26, [x4, #0]
    b label_187
KAI_ASM_LABEL(label_183)  // Height 5: Partial accumulate: partial_2_0
    tbz x26, #2, label_185
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    tbz x26, #1, label_184
    ldr d9, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    tbz x26, #0, label_187
    ld1 { v9.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    b label_187
KAI_ASM_LABEL(label_184)  // Height 5: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_187
    ldr s9, [x27, #0]
    ldr s13, [x7, #0]
    ldr s17, [x6, #0]
    ldr s21, [x5, #0]
    ldr s25, [x4, #0]
    b label_187
KAI_ASM_LABEL(label_185)  // Height 5: Partial accumulate: partial_1_0
    tbz x26, #1, label_186
    ldr d8, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    tbz x26, #0, label_187
    ld1 { v8.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v16.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    b label_187
KAI_ASM_LABEL(label_186)  // Height 5: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s16, [x6, #0]
    ldr s20, [x5, #0]
    ldr s24, [x4, #0]
KAI_ASM_LABEL(label_187)  // Height 5: Partial accumulate: Done
    sub x27, x27, x1
    b label_190
KAI_ASM_LABEL(label_188)  // Height 5: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
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
    b label_190
KAI_ASM_LABEL(label_189)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_190)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_191
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_193
KAI_ASM_LABEL(label_191)  // Height 5: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_192)  // Height 5: String loop
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
    cbnz x28, label_193
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_193)  // Height 5: input setup done
    cmp x7, #0x4
    blt label_196
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x8
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    blt label_195
KAI_ASM_LABEL(label_194)  // Height 5: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    cmp x7, #0x8
    add x6, x6, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    add x5, x5, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr q7, [x20, #0]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    ldr q6, [x23, #0x30]
    fmla v11.4s, v7.4s, v0.s[2]
    add x23, x23, #0x40
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    ldr q6, [x21, #0x30]
    fmla v9.4s, v7.4s, v0.s[3]
    add x21, x21, #0x40
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q1, [x6, #0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr q2, [x5, #0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr q3, [x4, #0]
    fmla v27.4s, v7.4s, v4.s[3]
    ldr q4, [x2, #0]
    ldr q7, [x22, #0]
    bge label_194
KAI_ASM_LABEL(label_195)  // Height 5: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    ldr q6, [x21, #0]
    fmla v9.4s, v7.4s, v0.s[0]
    add x4, x4, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    add x2, x2, #0x10
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr q7, [x20, #0]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    ldr q6, [x23, #0x30]
    fmla v11.4s, v7.4s, v0.s[2]
    add x23, x23, #0x40
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    ldr q6, [x21, #0x30]
    fmla v9.4s, v7.4s, v0.s[3]
    add x21, x21, #0x40
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
    fmla v27.4s, v7.4s, v4.s[3]
KAI_ASM_LABEL(label_196)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_198
KAI_ASM_LABEL(label_197)  // Height 5: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x6], #0x4
    sub x7, x7, #0x1
    ldr s2, [x5], #0x4
    ldr s3, [x4], #0x4
    ldr s4, [x2], #0x4
    ldr q6, [x23, #0]
    add x23, x23, #0x10
    ldr q7, [x22, #0]
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v4.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    cbnz x7, label_197
KAI_ASM_LABEL(label_198)  // Height 5: Multiply loop: No odd multiplies
    tbz x9, #3, label_199
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_192
KAI_ASM_LABEL(label_199)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x9, #1, label_200
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v29.4s }, [x2]
    ld1r { v28.4s }, [x1]
    fmin v8.4s, v8.4s, v29.4s
    fmin v9.4s, v9.4s, v29.4s
    fmin v10.4s, v10.4s, v29.4s
    fmin v11.4s, v11.4s, v29.4s
    fmin v12.4s, v12.4s, v29.4s
    fmin v13.4s, v13.4s, v29.4s
    fmin v14.4s, v14.4s, v29.4s
    fmin v15.4s, v15.4s, v29.4s
    fmin v16.4s, v16.4s, v29.4s
    fmin v17.4s, v17.4s, v29.4s
    fmin v18.4s, v18.4s, v29.4s
    fmin v19.4s, v19.4s, v29.4s
    fmin v20.4s, v20.4s, v29.4s
    fmin v21.4s, v21.4s, v29.4s
    fmin v22.4s, v22.4s, v29.4s
    fmin v23.4s, v23.4s, v29.4s
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s
    fmin v26.4s, v26.4s, v29.4s
    fmin v27.4s, v27.4s, v29.4s
    fmax v8.4s, v8.4s, v28.4s
    fmax v9.4s, v9.4s, v28.4s
    fmax v10.4s, v10.4s, v28.4s
    fmax v11.4s, v11.4s, v28.4s
    fmax v12.4s, v12.4s, v28.4s
    fmax v13.4s, v13.4s, v28.4s
    fmax v14.4s, v14.4s, v28.4s
    fmax v15.4s, v15.4s, v28.4s
    fmax v16.4s, v16.4s, v28.4s
    fmax v17.4s, v17.4s, v28.4s
    fmax v18.4s, v18.4s, v28.4s
    fmax v19.4s, v19.4s, v28.4s
    fmax v20.4s, v20.4s, v28.4s
    fmax v21.4s, v21.4s, v28.4s
    fmax v22.4s, v22.4s, v28.4s
    fmax v23.4s, v23.4s, v28.4s
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    fmax v26.4s, v26.4s, v28.4s
    fmax v27.4s, v27.4s, v28.4s
KAI_ASM_LABEL(label_200)  // Height 5: No activation
    cmp x26, #0x10
    bge label_209
    tbz x26, #3, label_204
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v13.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v17.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v21.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v25.4s }, [x4], #0x10
    tbz x26, #2, label_202
    st1 { v10.4s }, [x27], #0x10
    st1 { v14.4s }, [x7], #0x10
    st1 { v18.4s }, [x6], #0x10
    st1 { v22.4s }, [x5], #0x10
    st1 { v26.4s }, [x4], #0x10
    tbz x26, #1, label_201
    str d11, [x27], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    str d27, [x4], #0x8
    tbz x26, #0, label_208
    st1 { v11.s }[2], [x27]
    st1 { v15.s }[2], [x7]
    st1 { v19.s }[2], [x6]
    st1 { v23.s }[2], [x5]
    st1 { v27.s }[2], [x4]
    b label_208
KAI_ASM_LABEL(label_201)  // Height 5: Partial direct writeback: partial_0_12
    tbz x26, #0, label_208
    str s11, [x27, #0]
    str s15, [x7, #0]
    str s19, [x6, #0]
    str s23, [x5, #0]
    str s27, [x4, #0]
    b label_208
KAI_ASM_LABEL(label_202)  // Height 5: Partial direct writeback: partial_1_8
    tbz x26, #1, label_203
    str d10, [x27], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    str d26, [x4], #0x8
    tbz x26, #0, label_208
    st1 { v10.s }[2], [x27]
    st1 { v14.s }[2], [x7]
    st1 { v18.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    st1 { v26.s }[2], [x4]
    b label_208
KAI_ASM_LABEL(label_203)  // Height 5: Partial direct writeback: partial_0_8
    tbz x26, #0, label_208
    str s10, [x27, #0]
    str s14, [x7, #0]
    str s18, [x6, #0]
    str s22, [x5, #0]
    str s26, [x4, #0]
    b label_208
KAI_ASM_LABEL(label_204)  // Height 5: Partial direct writeback: partial_2_0
    tbz x26, #2, label_206
    st1 { v8.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    tbz x26, #1, label_205
    str d9, [x27], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    tbz x26, #0, label_208
    st1 { v9.s }[2], [x27]
    st1 { v13.s }[2], [x7]
    st1 { v17.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    b label_208
KAI_ASM_LABEL(label_205)  // Height 5: Partial direct writeback: partial_0_4
    tbz x26, #0, label_208
    str s9, [x27, #0]
    str s13, [x7, #0]
    str s17, [x6, #0]
    str s21, [x5, #0]
    str s25, [x4, #0]
    b label_208
KAI_ASM_LABEL(label_206)  // Height 5: Partial direct writeback: partial_1_0
    tbz x26, #1, label_207
    str d8, [x27], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    tbz x26, #0, label_208
    st1 { v8.s }[2], [x27]
    st1 { v12.s }[2], [x7]
    st1 { v16.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    b label_208
KAI_ASM_LABEL(label_207)  // Height 5: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
    str s12, [x7, #0]
    str s16, [x6, #0]
    str s20, [x5, #0]
    str s24, [x4, #0]
KAI_ASM_LABEL(label_208)  // Height 5: Partial direct writeback: Done
    b label_210
KAI_ASM_LABEL(label_209)  // Height 5: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
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
KAI_ASM_LABEL(label_210)  // Height 5: Writeback done
    subs x26, x26, #0x10
    bgt label_170
    b label_253
KAI_ASM_LABEL(label_211)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x27, [x0, #0x50]
    ldr x26, [x0, #0x8]
    ldr x25, [x0, #0x30]
    ldr x24, [x0, #0x40]
    madd x27, x11, x1, x27
KAI_ASM_LABEL(label_212)  // Height 6: Column loop
    mov x23, x25
    ldr x1, [x0, #0x38]
    cmp x26, #0xc
    add x22, x23, x1
    add x21, x22, x1
    add x20, x21, x1
    add x25, x20, x1
    bgt label_213
    cmp x26, #0x8
    mov x20, x23
    bgt label_213
    cmp x26, #0x4
    mov x21, x23
    bgt label_213
    mov x22, x23
KAI_ASM_LABEL(label_213)  // Height 6: B setup done
    cbz x24, label_221
    cmp x26, #0x10
    bge label_219
    cbnz x11, label_218
    mov x2, x26
    mov x1, SP
    cmp x2, #0x4
    blt label_215
KAI_ASM_LABEL(label_214)  // Height 6: Partial bias load: vector loop
    ld1 { v16.4s }, [x24], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_214
KAI_ASM_LABEL(label_215)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_217
KAI_ASM_LABEL(label_216)  // Height 6: Partial bias load: element loop
    ldr s16, [x24], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_216
KAI_ASM_LABEL(label_217)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_218)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    ldr q11, [x1, #0x30]
    b label_220
KAI_ASM_LABEL(label_219)  // Height 6: full width bias
    ldr q8, [x24, #0]
    ldr q9, [x24, #0x10]
    ldr q10, [x24, #0x20]
    ldr q11, [x24, #0x30]
    add x24, x24, #0x40
KAI_ASM_LABEL(label_220)  // Height 6: bias load done
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
    b label_232
KAI_ASM_LABEL(label_221)  // Height 6: no bias
    tbz x9, #0, label_231
    ldr x1, [x0, #0x58]
    cmp x26, #0x10
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_230
    tbz x26, #3, label_225
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    ld1 { v9.4s }, [x27], #0x10
    ld1 { v13.4s }, [x7], #0x10
    ld1 { v17.4s }, [x6], #0x10
    ld1 { v21.4s }, [x5], #0x10
    ld1 { v25.4s }, [x4], #0x10
    ld1 { v29.4s }, [x3], #0x10
    tbz x26, #2, label_223
    ld1 { v10.4s }, [x27], #0x10
    ld1 { v14.4s }, [x7], #0x10
    ld1 { v18.4s }, [x6], #0x10
    ld1 { v22.4s }, [x5], #0x10
    ld1 { v26.4s }, [x4], #0x10
    ld1 { v30.4s }, [x3], #0x10
    tbz x26, #1, label_222
    ldr d11, [x27], #0x8
    ldr d15, [x7], #0x8
    mov x1, #0x38
    ldr d19, [x6], #0x8
    ldr d23, [x5], #0x8
    ldr d27, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x26, #0, label_229
    ld1 { v11.s }[2], [x27]
    ld1 { v15.s }[2], [x7]
    ld1 { v19.s }[2], [x6]
    ld1 { v23.s }[2], [x5]
    ld1 { v27.s }[2], [x4]
    ld1 { v31.s }[2], [x3]
    b label_229
KAI_ASM_LABEL(label_222)  // Height 6: Partial accumulate: partial_0_12
    mov x1, #0x30
    tbz x26, #0, label_229
    ldr s11, [x27, #0]
    ldr s15, [x7, #0]
    ldr s19, [x6, #0]
    ldr s23, [x5, #0]
    ldr s27, [x4, #0]
    ldr s31, [x3, #0]
    b label_229
KAI_ASM_LABEL(label_223)  // Height 6: Partial accumulate: partial_1_8
    tbz x26, #1, label_224
    ldr d10, [x27], #0x8
    ldr d14, [x7], #0x8
    mov x1, #0x28
    ldr d18, [x6], #0x8
    ldr d22, [x5], #0x8
    ldr d26, [x4], #0x8
    ldr d30, [x3], #0x8
    tbz x26, #0, label_229
    ld1 { v10.s }[2], [x27]
    ld1 { v14.s }[2], [x7]
    ld1 { v18.s }[2], [x6]
    ld1 { v22.s }[2], [x5]
    ld1 { v26.s }[2], [x4]
    ld1 { v30.s }[2], [x3]
    b label_229
KAI_ASM_LABEL(label_224)  // Height 6: Partial accumulate: partial_0_8
    mov x1, #0x20
    tbz x26, #0, label_229
    ldr s10, [x27, #0]
    ldr s14, [x7, #0]
    ldr s18, [x6, #0]
    ldr s22, [x5, #0]
    ldr s26, [x4, #0]
    ldr s30, [x3, #0]
    b label_229
KAI_ASM_LABEL(label_225)  // Height 6: Partial accumulate: partial_2_0
    tbz x26, #2, label_227
    ld1 { v8.4s }, [x27], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    tbz x26, #1, label_226
    ldr d9, [x27], #0x8
    ldr d13, [x7], #0x8
    mov x1, #0x18
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x26, #0, label_229
    ld1 { v9.s }[2], [x27]
    ld1 { v13.s }[2], [x7]
    ld1 { v17.s }[2], [x6]
    ld1 { v21.s }[2], [x5]
    ld1 { v25.s }[2], [x4]
    ld1 { v29.s }[2], [x3]
    b label_229
KAI_ASM_LABEL(label_226)  // Height 6: Partial accumulate: partial_0_4
    mov x1, #0x10
    tbz x26, #0, label_229
    ldr s9, [x27, #0]
    ldr s13, [x7, #0]
    ldr s17, [x6, #0]
    ldr s21, [x5, #0]
    ldr s25, [x4, #0]
    ldr s29, [x3, #0]
    b label_229
KAI_ASM_LABEL(label_227)  // Height 6: Partial accumulate: partial_1_0
    tbz x26, #1, label_228
    ldr d8, [x27], #0x8
    ldr d12, [x7], #0x8
    mov x1, #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x26, #0, label_229
    ld1 { v8.s }[2], [x27]
    ld1 { v12.s }[2], [x7]
    ld1 { v16.s }[2], [x6]
    ld1 { v20.s }[2], [x5]
    ld1 { v24.s }[2], [x4]
    ld1 { v28.s }[2], [x3]
    b label_229
KAI_ASM_LABEL(label_228)  // Height 6: Partial accumulate: partial_0_0
    ldr s8, [x27, #0]
    ldr s12, [x7, #0]
    mov x1, #0
    ldr s16, [x6, #0]
    ldr s20, [x5, #0]
    ldr s24, [x4, #0]
    ldr s28, [x3, #0]
KAI_ASM_LABEL(label_229)  // Height 6: Partial accumulate: Done
    sub x27, x27, x1
    b label_232
KAI_ASM_LABEL(label_230)  // Height 6: full accumulate
    ldr q8, [x27, #0]
    ldr q9, [x27, #0x10]
    ldr q10, [x27, #0x20]
    ldr q11, [x27, #0x30]
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
    b label_232
KAI_ASM_LABEL(label_231)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_232)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x9, #3, label_233
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x11, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_235
KAI_ASM_LABEL(label_233)  // Height 6: setup indirect input
    mov x28, #0
KAI_ASM_LABEL(label_234)  // Height 6: String loop
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
    cbnz x28, label_235
    add x8, x8, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_235)  // Height 6: input setup done
    cmp x7, #0x4
    blt label_238
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x8
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q5, [x1, #0]
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    blt label_237
KAI_ASM_LABEL(label_236)  // Height 6: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    cmp x7, #0x8
    add x6, x6, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    fmla v28.4s, v6.4s, v5.s[0]
    ldr q6, [x21, #0]
    add x5, x5, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    add x1, x1, #0x10
    fmla v25.4s, v7.4s, v4.s[0]
    fmla v29.4s, v7.4s, v5.s[0]
    ldr q7, [x20, #0]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    fmla v28.4s, v6.4s, v5.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    fmla v29.4s, v7.4s, v5.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    fmla v30.4s, v6.4s, v5.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    fmla v31.4s, v7.4s, v5.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    fmla v28.4s, v6.4s, v5.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    fmla v29.4s, v7.4s, v5.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    fmla v30.4s, v6.4s, v5.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    fmla v31.4s, v7.4s, v5.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    fmla v28.4s, v6.4s, v5.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    fmla v29.4s, v7.4s, v5.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v30.4s, v6.4s, v5.s[3]
    ldr q6, [x23, #0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr q0, [x8, #0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr q1, [x6, #0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr q2, [x5, #0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr q3, [x4, #0]
    fmla v27.4s, v7.4s, v4.s[3]
    ldr q4, [x2, #0]
    fmla v31.4s, v7.4s, v5.s[3]
    ldr q5, [x1, #0]
    ldr q7, [x22, #0]
    bge label_236
KAI_ASM_LABEL(label_237)  // Height 6: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    sub x7, x7, #0x4
    add x8, x8, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    fmla v28.4s, v6.4s, v5.s[0]
    ldr q6, [x21, #0]
    add x4, x4, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    fmla v29.4s, v7.4s, v5.s[0]
    ldr q7, [x20, #0]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    ldr q6, [x23, #0x10]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    ldr q7, [x22, #0x10]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    fmla v28.4s, v6.4s, v5.s[1]
    ldr q6, [x21, #0x10]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    fmla v29.4s, v7.4s, v5.s[1]
    ldr q7, [x20, #0x10]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    fmla v30.4s, v6.4s, v5.s[1]
    ldr q6, [x23, #0x20]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    fmla v31.4s, v7.4s, v5.s[1]
    ldr q7, [x22, #0x20]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    fmla v28.4s, v6.4s, v5.s[2]
    ldr q6, [x21, #0x20]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    fmla v29.4s, v7.4s, v5.s[2]
    ldr q7, [x20, #0x20]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    fmla v30.4s, v6.4s, v5.s[2]
    ldr q6, [x23, #0x30]
    add x23, x23, #0x40
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    fmla v31.4s, v7.4s, v5.s[2]
    ldr q7, [x22, #0x30]
    add x22, x22, #0x40
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    fmla v28.4s, v6.4s, v5.s[3]
    ldr q6, [x21, #0x30]
    add x21, x21, #0x40
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    fmla v29.4s, v7.4s, v5.s[3]
    ldr q7, [x20, #0x30]
    add x20, x20, #0x40
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v30.4s, v6.4s, v5.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
    fmla v27.4s, v7.4s, v4.s[3]
    fmla v31.4s, v7.4s, v5.s[3]
KAI_ASM_LABEL(label_238)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_240
KAI_ASM_LABEL(label_239)  // Height 6: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x6], #0x4
    sub x7, x7, #0x1
    ldr s2, [x5], #0x4
    ldr s3, [x4], #0x4
    ldr s4, [x2], #0x4
    ldr s5, [x1], #0x4
    ldr q6, [x23, #0]
    ldr q7, [x22, #0]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v4.s[0]
    fmla v28.4s, v6.4s, v5.s[0]
    ldr q6, [x21, #0]
    add x21, x21, #0x10
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    fmla v29.4s, v7.4s, v5.s[0]
    ldr q7, [x20, #0]
    add x20, x20, #0x10
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    cbnz x7, label_239
KAI_ASM_LABEL(label_240)  // Height 6: Multiply loop: No odd multiplies
    tbz x9, #3, label_241
    ldr x1, [x0, #0x10]
    add x28, x28, #0x1
    cmp x28, x1
    bne label_234
KAI_ASM_LABEL(label_241)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x7, x27, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x9, #1, label_242
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
KAI_ASM_LABEL(label_242)  // Height 6: No activation
    cmp x26, #0x10
    bge label_251
    tbz x26, #3, label_246
    st1 { v8.4s }, [x27], #0x10
    st1 { v9.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v13.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v17.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v21.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v25.4s }, [x4], #0x10
    st1 { v28.4s }, [x3], #0x10
    st1 { v29.4s }, [x3], #0x10
    tbz x26, #2, label_244
    st1 { v10.4s }, [x27], #0x10
    st1 { v14.4s }, [x7], #0x10
    st1 { v18.4s }, [x6], #0x10
    st1 { v22.4s }, [x5], #0x10
    st1 { v26.4s }, [x4], #0x10
    st1 { v30.4s }, [x3], #0x10
    tbz x26, #1, label_243
    str d11, [x27], #0x8
    str d15, [x7], #0x8
    str d19, [x6], #0x8
    str d23, [x5], #0x8
    str d27, [x4], #0x8
    str d31, [x3], #0x8
    tbz x26, #0, label_250
    st1 { v11.s }[2], [x27]
    st1 { v15.s }[2], [x7]
    st1 { v19.s }[2], [x6]
    st1 { v23.s }[2], [x5]
    st1 { v27.s }[2], [x4]
    st1 { v31.s }[2], [x3]
    b label_250
KAI_ASM_LABEL(label_243)  // Height 6: Partial direct writeback: partial_0_12
    tbz x26, #0, label_250
    str s11, [x27, #0]
    str s15, [x7, #0]
    str s19, [x6, #0]
    str s23, [x5, #0]
    str s27, [x4, #0]
    str s31, [x3, #0]
    b label_250
KAI_ASM_LABEL(label_244)  // Height 6: Partial direct writeback: partial_1_8
    tbz x26, #1, label_245
    str d10, [x27], #0x8
    str d14, [x7], #0x8
    str d18, [x6], #0x8
    str d22, [x5], #0x8
    str d26, [x4], #0x8
    str d30, [x3], #0x8
    tbz x26, #0, label_250
    st1 { v10.s }[2], [x27]
    st1 { v14.s }[2], [x7]
    st1 { v18.s }[2], [x6]
    st1 { v22.s }[2], [x5]
    st1 { v26.s }[2], [x4]
    st1 { v30.s }[2], [x3]
    b label_250
KAI_ASM_LABEL(label_245)  // Height 6: Partial direct writeback: partial_0_8
    tbz x26, #0, label_250
    str s10, [x27, #0]
    str s14, [x7, #0]
    str s18, [x6, #0]
    str s22, [x5, #0]
    str s26, [x4, #0]
    str s30, [x3, #0]
    b label_250
KAI_ASM_LABEL(label_246)  // Height 6: Partial direct writeback: partial_2_0
    tbz x26, #2, label_248
    st1 { v8.4s }, [x27], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v28.4s }, [x3], #0x10
    tbz x26, #1, label_247
    str d9, [x27], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    str d29, [x3], #0x8
    tbz x26, #0, label_250
    st1 { v9.s }[2], [x27]
    st1 { v13.s }[2], [x7]
    st1 { v17.s }[2], [x6]
    st1 { v21.s }[2], [x5]
    st1 { v25.s }[2], [x4]
    st1 { v29.s }[2], [x3]
    b label_250
KAI_ASM_LABEL(label_247)  // Height 6: Partial direct writeback: partial_0_4
    tbz x26, #0, label_250
    str s9, [x27, #0]
    str s13, [x7, #0]
    str s17, [x6, #0]
    str s21, [x5, #0]
    str s25, [x4, #0]
    str s29, [x3, #0]
    b label_250
KAI_ASM_LABEL(label_248)  // Height 6: Partial direct writeback: partial_1_0
    tbz x26, #1, label_249
    str d8, [x27], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    str d28, [x3], #0x8
    tbz x26, #0, label_250
    st1 { v8.s }[2], [x27]
    st1 { v12.s }[2], [x7]
    st1 { v16.s }[2], [x6]
    st1 { v20.s }[2], [x5]
    st1 { v24.s }[2], [x4]
    st1 { v28.s }[2], [x3]
    b label_250
KAI_ASM_LABEL(label_249)  // Height 6: Partial direct writeback: partial_0_0
    str s8, [x27, #0]
    str s12, [x7, #0]
    str s16, [x6, #0]
    str s20, [x5, #0]
    str s24, [x4, #0]
    str s28, [x3, #0]
KAI_ASM_LABEL(label_250)  // Height 6: Partial direct writeback: Done
    b label_252
KAI_ASM_LABEL(label_251)  // Height 6: Full writeback
    str q8, [x27, #0]
    str q9, [x27, #0x10]
    str q10, [x27, #0x20]
    str q11, [x27, #0x30]
    add x27, x27, #0x40
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
KAI_ASM_LABEL(label_252)  // Height 6: Writeback done
    subs x26, x26, #0x10
    bgt label_212
    add x11, x11, #0x6
    cmp x11, x10
    blt label_1
KAI_ASM_LABEL(label_253)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_ffhybrid_fp32_mla_6x16)

    KAI_ASM_END
