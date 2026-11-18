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

    KAI_ASM_CODE(a64_hybrid_fp32_mla_8x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_fp32_mla_8x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_fp32_mla_8x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_fp32_mla_8x4)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    str d12, [sp, 104]
    mov x6, #0xff800000
    mov x5, #0x7f800000
    sub SP, SP, #0x10
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
    bge label_204
    cmp x1, #0x6
    bgt label_175
    beq label_146
    cmp x1, #0x4
    bgt label_117
    beq label_88
    cmp x1, #0x2
    bgt label_59
    beq label_30
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x22, label_10
    cmp x24, #0x4
    bge label_8
    cbnz x9, label_7
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_4
KAI_ASM_LABEL(label_3)  // Height 1: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_3
KAI_ASM_LABEL(label_4)  // Height 1: Partial bias load: vector loop skip
    cbz x2, label_6
KAI_ASM_LABEL(label_5)  // Height 1: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_5
KAI_ASM_LABEL(label_6)  // Height 1: Partial bias load: element loop skip
KAI_ASM_LABEL(label_7)  // Height 1: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_9
KAI_ASM_LABEL(label_8)  // Height 1: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_9)  // Height 1: bias load done
    b label_15
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x27, #0, label_14
    cmp x24, #0x4
    bge label_13
    tbz x24, #1, label_11
    ldr d24, [x25], #0x8
    mov x1, #0x8
    tbz x24, #0, label_12
    ld1 { v24.s }[2], [x25]
    b label_12
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    mov x1, #0
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: Done
    sub x25, x25, x1
    b label_15
KAI_ASM_LABEL(label_13)  // Height 1: full accumulate
    ldr q24, [x25, #0]
    b label_15
KAI_ASM_LABEL(label_14)  // Height 1: no accumulate
    movi v24.16b, #0
KAI_ASM_LABEL(label_15)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_16
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    b label_18
KAI_ASM_LABEL(label_16)  // Height 1: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_17)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    cbnz x26, label_18
    add x21, x21, x3, LSL #2
KAI_ASM_LABEL(label_18)  // Height 1: input setup done
    cmp x20, #0x4
    blt label_21
    ldr q0, [x21, #0]
    ldr q8, [x23, #0]
    cmp x20, #0x8
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_20
KAI_ASM_LABEL(label_19)  // Height 1: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    cmp x20, #0x8
    add x23, x23, #0x40
    prfm pldl1keep, [x21, #0x80]
    ldr q8, [x23, #0]
    fmla v24.4s, v9.4s, v0.s[1]
    ldr q9, [x23, #0x10]
    fmla v24.4s, v10.4s, v0.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    ldr q11, [x23, #0x30]
    bge label_19
KAI_ASM_LABEL(label_20)  // Height 1: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    add x21, x21, #0x10
    sub x20, x20, #0x4
    add x23, x23, #0x40
    prfm pldl1keep, [x21, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
KAI_ASM_LABEL(label_21)  // Height 1: Multiply loop: Main loop skip
    cbz x20, label_23
KAI_ASM_LABEL(label_22)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr q12, [x23, #0]
    sub x20, x20, #0x1
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    cbnz x20, label_22
KAI_ASM_LABEL(label_23)  // Height 1: Multiply loop: No odd multiplies
    tbz x27, #3, label_24
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_17
KAI_ASM_LABEL(label_24)  // Height 1: no string loop
    prfm pstl1keep, [x25, #0]
    tbz x27, #1, label_25
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
KAI_ASM_LABEL(label_25)  // Height 1: No activation
    cmp x24, #0x4
    bge label_28
    tbz x24, #1, label_26
    str d24, [x25], #0x8
    tbz x24, #0, label_27
    st1 { v24.s }[2], [x25]
    b label_27
KAI_ASM_LABEL(label_26)  // Height 1: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
KAI_ASM_LABEL(label_27)  // Height 1: Partial direct writeback: Done
    b label_29
KAI_ASM_LABEL(label_28)  // Height 1: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
KAI_ASM_LABEL(label_29)  // Height 1: Writeback done
    subs x24, x24, #0x4
    bgt label_2
    b label_233
KAI_ASM_LABEL(label_30)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_31)  // Height 2: Column loop
    cbz x22, label_39
    cmp x24, #0x4
    bge label_37
    cbnz x9, label_36
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_33
KAI_ASM_LABEL(label_32)  // Height 2: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_32
KAI_ASM_LABEL(label_33)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_35
KAI_ASM_LABEL(label_34)  // Height 2: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_34
KAI_ASM_LABEL(label_35)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_36)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_38
KAI_ASM_LABEL(label_37)  // Height 2: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_38)  // Height 2: bias load done
    mov v25.16b, v24.16b
    b label_44
KAI_ASM_LABEL(label_39)  // Height 2: no bias
    tbz x27, #0, label_43
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    bge label_42
    tbz x24, #1, label_40
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    tbz x24, #0, label_41
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    b label_41
KAI_ASM_LABEL(label_40)  // Height 2: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
KAI_ASM_LABEL(label_41)  // Height 2: Partial accumulate: Done
    sub x25, x25, x1
    b label_44
KAI_ASM_LABEL(label_42)  // Height 2: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 2: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
KAI_ASM_LABEL(label_44)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_45
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    b label_47
KAI_ASM_LABEL(label_45)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_46)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    cbnz x26, label_47
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
KAI_ASM_LABEL(label_47)  // Height 2: input setup done
    cmp x20, #0x4
    blt label_50
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_49
KAI_ASM_LABEL(label_48)  // Height 2: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    add x8, x8, #0x10
    cmp x20, #0x8
    prfm pldl1keep, [x21, #0x80]
    add x23, x23, #0x40
    prfm pldl1keep, [x8, #0x80]
    ldr q8, [x23, #0]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    ldr q9, [x23, #0x10]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    ldr q11, [x23, #0x30]
    bge label_48
KAI_ASM_LABEL(label_49)  // Height 2: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    sub x20, x20, #0x4
    prfm pldl1keep, [x21, #0x80]
    add x23, x23, #0x40
    prfm pldl1keep, [x8, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
KAI_ASM_LABEL(label_50)  // Height 2: Multiply loop: Main loop skip
    cbz x20, label_52
KAI_ASM_LABEL(label_51)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    cbnz x20, label_51
KAI_ASM_LABEL(label_52)  // Height 2: Multiply loop: No odd multiplies
    tbz x27, #3, label_53
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_46
KAI_ASM_LABEL(label_53)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    tbz x27, #1, label_54
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
KAI_ASM_LABEL(label_54)  // Height 2: No activation
    cmp x24, #0x4
    bge label_57
    tbz x24, #1, label_55
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    tbz x24, #0, label_56
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    b label_56
KAI_ASM_LABEL(label_55)  // Height 2: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
KAI_ASM_LABEL(label_56)  // Height 2: Partial direct writeback: Done
    b label_58
KAI_ASM_LABEL(label_57)  // Height 2: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
KAI_ASM_LABEL(label_58)  // Height 2: Writeback done
    subs x24, x24, #0x4
    bgt label_31
    b label_233
KAI_ASM_LABEL(label_59)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_60)  // Height 3: Column loop
    cbz x22, label_68
    cmp x24, #0x4
    bge label_66
    cbnz x9, label_65
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_62
KAI_ASM_LABEL(label_61)  // Height 3: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_61
KAI_ASM_LABEL(label_62)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_64
KAI_ASM_LABEL(label_63)  // Height 3: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_63
KAI_ASM_LABEL(label_64)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_65)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_67
KAI_ASM_LABEL(label_66)  // Height 3: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_67)  // Height 3: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    b label_73
KAI_ASM_LABEL(label_68)  // Height 3: no bias
    tbz x27, #0, label_72
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    bge label_71
    tbz x24, #1, label_69
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    tbz x24, #0, label_70
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    b label_70
KAI_ASM_LABEL(label_69)  // Height 3: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
KAI_ASM_LABEL(label_70)  // Height 3: Partial accumulate: Done
    sub x25, x25, x1
    b label_73
KAI_ASM_LABEL(label_71)  // Height 3: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    b label_73
KAI_ASM_LABEL(label_72)  // Height 3: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
KAI_ASM_LABEL(label_73)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_74
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    b label_76
KAI_ASM_LABEL(label_74)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_75)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    cbnz x26, label_76
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
KAI_ASM_LABEL(label_76)  // Height 3: input setup done
    cmp x20, #0x4
    blt label_79
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_78
KAI_ASM_LABEL(label_77)  // Height 3: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    prfm pldl1keep, [x21, #0x80]
    cmp x20, #0x8
    add x23, x23, #0x40
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x7, #0x80]
    ldr q8, [x23, #0]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    fmla v26.4s, v9.4s, v2.s[1]
    ldr q9, [x23, #0x10]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    ldr q11, [x23, #0x30]
    bge label_77
KAI_ASM_LABEL(label_78)  // Height 3: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    add x7, x7, #0x10
    prfm pldl1keep, [x21, #0x80]
    sub x20, x20, #0x4
    prfm pldl1keep, [x8, #0x80]
    add x23, x23, #0x40
    prfm pldl1keep, [x7, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
KAI_ASM_LABEL(label_79)  // Height 3: Multiply loop: Main loop skip
    cbz x20, label_81
KAI_ASM_LABEL(label_80)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    cbnz x20, label_80
KAI_ASM_LABEL(label_81)  // Height 3: Multiply loop: No odd multiplies
    tbz x27, #3, label_82
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_75
KAI_ASM_LABEL(label_82)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    tbz x27, #1, label_83
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
KAI_ASM_LABEL(label_83)  // Height 3: No activation
    cmp x24, #0x4
    bge label_86
    tbz x24, #1, label_84
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    tbz x24, #0, label_85
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    b label_85
KAI_ASM_LABEL(label_84)  // Height 3: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
KAI_ASM_LABEL(label_85)  // Height 3: Partial direct writeback: Done
    b label_87
KAI_ASM_LABEL(label_86)  // Height 3: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
KAI_ASM_LABEL(label_87)  // Height 3: Writeback done
    subs x24, x24, #0x4
    bgt label_60
    b label_233
KAI_ASM_LABEL(label_88)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_89)  // Height 4: Column loop
    cbz x22, label_97
    cmp x24, #0x4
    bge label_95
    cbnz x9, label_94
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_91
KAI_ASM_LABEL(label_90)  // Height 4: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_90
KAI_ASM_LABEL(label_91)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_93
KAI_ASM_LABEL(label_92)  // Height 4: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_92
KAI_ASM_LABEL(label_93)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_94)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_96
KAI_ASM_LABEL(label_95)  // Height 4: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_96)  // Height 4: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    mov v27.16b, v24.16b
    b label_102
KAI_ASM_LABEL(label_97)  // Height 4: no bias
    tbz x27, #0, label_101
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    bge label_100
    tbz x24, #1, label_98
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    ldr d27, [x7], #0x8
    tbz x24, #0, label_99
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    ld1 { v27.s }[2], [x7]
    b label_99
KAI_ASM_LABEL(label_98)  // Height 4: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
    ldr s27, [x7, #0]
KAI_ASM_LABEL(label_99)  // Height 4: Partial accumulate: Done
    sub x25, x25, x1
    b label_102
KAI_ASM_LABEL(label_100)  // Height 4: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    ldr q27, [x7, #0]
    b label_102
KAI_ASM_LABEL(label_101)  // Height 4: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
KAI_ASM_LABEL(label_102)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_103
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    b label_105
KAI_ASM_LABEL(label_103)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_104)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x20, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x21, [x1, #0]
    ldr x8, [x1, #0x8]
    ldr x7, [x1, #0x10]
    ldr x6, [x1, #0x18]
    cbnz x26, label_105
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
KAI_ASM_LABEL(label_105)  // Height 4: input setup done
    cmp x20, #0x4
    blt label_108
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q3, [x6, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_107
KAI_ASM_LABEL(label_106)  // Height 4: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    add x6, x6, #0x10
    cmp x20, #0x8
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    add x23, x23, #0x40
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    ldr q8, [x23, #0]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    ldr q9, [x23, #0x10]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    fmla v27.4s, v11.4s, v3.s[3]
    ldr q3, [x6, #0]
    ldr q11, [x23, #0x30]
    bge label_106
KAI_ASM_LABEL(label_107)  // Height 4: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x7, x7, #0x10
    add x6, x6, #0x10
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    sub x20, x20, #0x4
    add x23, x23, #0x40
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
    fmla v27.4s, v11.4s, v3.s[3]
KAI_ASM_LABEL(label_108)  // Height 4: Multiply loop: Main loop skip
    cbz x20, label_110
KAI_ASM_LABEL(label_109)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr s3, [x6], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    cbnz x20, label_109
KAI_ASM_LABEL(label_110)  // Height 4: Multiply loop: No odd multiplies
    tbz x27, #3, label_111
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_104
KAI_ASM_LABEL(label_111)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    add x7, x8, x1
    prfm pstl1keep, [x7, #0]
    tbz x27, #1, label_112
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
KAI_ASM_LABEL(label_112)  // Height 4: No activation
    cmp x24, #0x4
    bge label_115
    tbz x24, #1, label_113
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    str d27, [x7], #0x8
    tbz x24, #0, label_114
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    st1 { v27.s }[2], [x7]
    b label_114
KAI_ASM_LABEL(label_113)  // Height 4: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
    str s27, [x7, #0]
KAI_ASM_LABEL(label_114)  // Height 4: Partial direct writeback: Done
    b label_116
KAI_ASM_LABEL(label_115)  // Height 4: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
    str q27, [x7, #0]
KAI_ASM_LABEL(label_116)  // Height 4: Writeback done
    subs x24, x24, #0x4
    bgt label_89
    b label_233
KAI_ASM_LABEL(label_117)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_118)  // Height 5: Column loop
    cbz x22, label_126
    cmp x24, #0x4
    bge label_124
    cbnz x9, label_123
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_120
KAI_ASM_LABEL(label_119)  // Height 5: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_119
KAI_ASM_LABEL(label_120)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_122
KAI_ASM_LABEL(label_121)  // Height 5: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_121
KAI_ASM_LABEL(label_122)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_123)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_125
KAI_ASM_LABEL(label_124)  // Height 5: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_125)  // Height 5: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    mov v27.16b, v24.16b
    mov v28.16b, v24.16b
    b label_131
KAI_ASM_LABEL(label_126)  // Height 5: no bias
    tbz x27, #0, label_130
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    bge label_129
    tbz x24, #1, label_127
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    ldr d27, [x7], #0x8
    ldr d28, [x6], #0x8
    tbz x24, #0, label_128
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    ld1 { v27.s }[2], [x7]
    ld1 { v28.s }[2], [x6]
    b label_128
KAI_ASM_LABEL(label_127)  // Height 5: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
    ldr s27, [x7, #0]
    ldr s28, [x6, #0]
KAI_ASM_LABEL(label_128)  // Height 5: Partial accumulate: Done
    sub x25, x25, x1
    b label_131
KAI_ASM_LABEL(label_129)  // Height 5: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    ldr q27, [x7, #0]
    ldr q28, [x6, #0]
    b label_131
KAI_ASM_LABEL(label_130)  // Height 5: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
KAI_ASM_LABEL(label_131)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_132
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    b label_134
KAI_ASM_LABEL(label_132)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_133)  // Height 5: String loop
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
    cbnz x26, label_134
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
KAI_ASM_LABEL(label_134)  // Height 5: input setup done
    cmp x20, #0x4
    blt label_137
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q3, [x6, #0]
    ldr q4, [x5, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_136
KAI_ASM_LABEL(label_135)  // Height 5: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    prfm pldl1keep, [x21, #0x80]
    cmp x20, #0x8
    add x23, x23, #0x40
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x7, #0x80]
    ldr q8, [x23, #0]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    prfm pldl1keep, [x6, #0x80]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    prfm pldl1keep, [x5, #0x80]
    fmla v28.4s, v9.4s, v4.s[1]
    ldr q9, [x23, #0x10]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    fmla v27.4s, v11.4s, v3.s[3]
    ldr q3, [x6, #0]
    fmla v28.4s, v11.4s, v4.s[3]
    ldr q4, [x5, #0]
    ldr q11, [x23, #0x30]
    bge label_135
KAI_ASM_LABEL(label_136)  // Height 5: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x7, x7, #0x10
    add x6, x6, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    add x5, x5, #0x10
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x7, #0x80]
    sub x20, x20, #0x4
    add x23, x23, #0x40
    prfm pldl1keep, [x6, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    prfm pldl1keep, [x5, #0x80]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
    fmla v27.4s, v11.4s, v3.s[3]
    fmla v28.4s, v11.4s, v4.s[3]
KAI_ASM_LABEL(label_137)  // Height 5: Multiply loop: Main loop skip
    cbz x20, label_139
KAI_ASM_LABEL(label_138)  // Height 5: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr s3, [x6], #0x4
    ldr s4, [x5], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    fmla v28.4s, v12.4s, v4.s[0]
    cbnz x20, label_138
KAI_ASM_LABEL(label_139)  // Height 5: Multiply loop: No odd multiplies
    tbz x27, #3, label_140
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_133
KAI_ASM_LABEL(label_140)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    add x7, x8, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    tbz x27, #1, label_141
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
KAI_ASM_LABEL(label_141)  // Height 5: No activation
    cmp x24, #0x4
    bge label_144
    tbz x24, #1, label_142
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    str d27, [x7], #0x8
    str d28, [x6], #0x8
    tbz x24, #0, label_143
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    st1 { v27.s }[2], [x7]
    st1 { v28.s }[2], [x6]
    b label_143
KAI_ASM_LABEL(label_142)  // Height 5: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
    str s27, [x7, #0]
    str s28, [x6, #0]
KAI_ASM_LABEL(label_143)  // Height 5: Partial direct writeback: Done
    b label_145
KAI_ASM_LABEL(label_144)  // Height 5: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
    str q27, [x7, #0]
    str q28, [x6, #0]
KAI_ASM_LABEL(label_145)  // Height 5: Writeback done
    subs x24, x24, #0x4
    bgt label_118
    b label_233
KAI_ASM_LABEL(label_146)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_147)  // Height 6: Column loop
    cbz x22, label_155
    cmp x24, #0x4
    bge label_153
    cbnz x9, label_152
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_149
KAI_ASM_LABEL(label_148)  // Height 6: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_148
KAI_ASM_LABEL(label_149)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_151
KAI_ASM_LABEL(label_150)  // Height 6: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_150
KAI_ASM_LABEL(label_151)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_152)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_154
KAI_ASM_LABEL(label_153)  // Height 6: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_154)  // Height 6: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    mov v27.16b, v24.16b
    mov v28.16b, v24.16b
    mov v29.16b, v24.16b
    b label_160
KAI_ASM_LABEL(label_155)  // Height 6: no bias
    tbz x27, #0, label_159
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    bge label_158
    tbz x24, #1, label_156
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    ldr d27, [x7], #0x8
    ldr d28, [x6], #0x8
    ldr d29, [x5], #0x8
    tbz x24, #0, label_157
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    ld1 { v27.s }[2], [x7]
    ld1 { v28.s }[2], [x6]
    ld1 { v29.s }[2], [x5]
    b label_157
KAI_ASM_LABEL(label_156)  // Height 6: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
    ldr s27, [x7, #0]
    ldr s28, [x6, #0]
    ldr s29, [x5, #0]
KAI_ASM_LABEL(label_157)  // Height 6: Partial accumulate: Done
    sub x25, x25, x1
    b label_160
KAI_ASM_LABEL(label_158)  // Height 6: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    ldr q27, [x7, #0]
    ldr q28, [x6, #0]
    ldr q29, [x5, #0]
    b label_160
KAI_ASM_LABEL(label_159)  // Height 6: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
KAI_ASM_LABEL(label_160)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_161
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_163
KAI_ASM_LABEL(label_161)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_162)  // Height 6: String loop
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
    cbnz x26, label_163
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
KAI_ASM_LABEL(label_163)  // Height 6: input setup done
    cmp x20, #0x4
    blt label_166
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q3, [x6, #0]
    ldr q4, [x5, #0]
    ldr q5, [x4, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_165
KAI_ASM_LABEL(label_164)  // Height 6: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    add x4, x4, #0x10
    cmp x20, #0x8
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    add x23, x23, #0x40
    prfm pldl1keep, [x7, #0x80]
    ldr q8, [x23, #0]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    prfm pldl1keep, [x6, #0x80]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    ldr q9, [x23, #0x10]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    fmla v27.4s, v11.4s, v3.s[3]
    ldr q3, [x6, #0]
    fmla v28.4s, v11.4s, v4.s[3]
    ldr q4, [x5, #0]
    fmla v29.4s, v11.4s, v5.s[3]
    ldr q5, [x4, #0]
    ldr q11, [x23, #0x30]
    bge label_164
KAI_ASM_LABEL(label_165)  // Height 6: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x7, x7, #0x10
    add x6, x6, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    sub x20, x20, #0x4
    add x23, x23, #0x40
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
    fmla v27.4s, v11.4s, v3.s[3]
    fmla v28.4s, v11.4s, v4.s[3]
    fmla v29.4s, v11.4s, v5.s[3]
KAI_ASM_LABEL(label_166)  // Height 6: Multiply loop: Main loop skip
    cbz x20, label_168
KAI_ASM_LABEL(label_167)  // Height 6: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr s3, [x6], #0x4
    ldr s4, [x5], #0x4
    ldr s5, [x4], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    fmla v28.4s, v12.4s, v4.s[0]
    fmla v29.4s, v12.4s, v5.s[0]
    cbnz x20, label_167
KAI_ASM_LABEL(label_168)  // Height 6: Multiply loop: No odd multiplies
    tbz x27, #3, label_169
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_162
KAI_ASM_LABEL(label_169)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    add x7, x8, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    prfm pstl1keep, [x5, #0]
    tbz x27, #1, label_170
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmin v29.4s, v29.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
    fmax v29.4s, v29.4s, v16.4s
KAI_ASM_LABEL(label_170)  // Height 6: No activation
    cmp x24, #0x4
    bge label_173
    tbz x24, #1, label_171
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    str d27, [x7], #0x8
    str d28, [x6], #0x8
    str d29, [x5], #0x8
    tbz x24, #0, label_172
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    st1 { v27.s }[2], [x7]
    st1 { v28.s }[2], [x6]
    st1 { v29.s }[2], [x5]
    b label_172
KAI_ASM_LABEL(label_171)  // Height 6: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
    str s27, [x7, #0]
    str s28, [x6, #0]
    str s29, [x5, #0]
KAI_ASM_LABEL(label_172)  // Height 6: Partial direct writeback: Done
    b label_174
KAI_ASM_LABEL(label_173)  // Height 6: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
    str q27, [x7, #0]
    str q28, [x6, #0]
    str q29, [x5, #0]
KAI_ASM_LABEL(label_174)  // Height 6: Writeback done
    subs x24, x24, #0x4
    bgt label_147
    b label_233
KAI_ASM_LABEL(label_175)  // Height 7
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_176)  // Height 7: Column loop
    cbz x22, label_184
    cmp x24, #0x4
    bge label_182
    cbnz x9, label_181
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_178
KAI_ASM_LABEL(label_177)  // Height 7: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_177
KAI_ASM_LABEL(label_178)  // Height 7: Partial bias load: vector loop skip
    cbz x2, label_180
KAI_ASM_LABEL(label_179)  // Height 7: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_179
KAI_ASM_LABEL(label_180)  // Height 7: Partial bias load: element loop skip
KAI_ASM_LABEL(label_181)  // Height 7: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_183
KAI_ASM_LABEL(label_182)  // Height 7: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_183)  // Height 7: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    mov v27.16b, v24.16b
    mov v28.16b, v24.16b
    mov v29.16b, v24.16b
    mov v30.16b, v24.16b
    b label_189
KAI_ASM_LABEL(label_184)  // Height 7: no bias
    tbz x27, #0, label_188
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_187
    tbz x24, #1, label_185
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    ldr d27, [x7], #0x8
    ldr d28, [x6], #0x8
    ldr d29, [x5], #0x8
    ldr d30, [x4], #0x8
    tbz x24, #0, label_186
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    ld1 { v27.s }[2], [x7]
    ld1 { v28.s }[2], [x6]
    ld1 { v29.s }[2], [x5]
    ld1 { v30.s }[2], [x4]
    b label_186
KAI_ASM_LABEL(label_185)  // Height 7: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
    ldr s27, [x7, #0]
    ldr s28, [x6, #0]
    ldr s29, [x5, #0]
    ldr s30, [x4, #0]
KAI_ASM_LABEL(label_186)  // Height 7: Partial accumulate: Done
    sub x25, x25, x1
    b label_189
KAI_ASM_LABEL(label_187)  // Height 7: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    ldr q27, [x7, #0]
    ldr q28, [x6, #0]
    ldr q29, [x5, #0]
    ldr q30, [x4, #0]
    b label_189
KAI_ASM_LABEL(label_188)  // Height 7: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
KAI_ASM_LABEL(label_189)  // Height 7: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_190
    ldr x21, [x0, #0x20]
    ldr x20, [x0, #0x10]
    madd x21, x9, x3, x21
    add x8, x21, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_192
KAI_ASM_LABEL(label_190)  // Height 7: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_191)  // Height 7: String loop
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
    cbnz x26, label_192
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
KAI_ASM_LABEL(label_192)  // Height 7: input setup done
    cmp x20, #0x4
    blt label_195
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q3, [x6, #0]
    ldr q4, [x5, #0]
    ldr q5, [x4, #0]
    ldr q6, [x2, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_194
KAI_ASM_LABEL(label_193)  // Height 7: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v30.4s, v8.4s, v6.s[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    prfm pldl1keep, [x21, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    cmp x20, #0x8
    add x23, x23, #0x40
    ldr q8, [x23, #0]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    prfm pldl1keep, [x8, #0x80]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v30.4s, v9.4s, v6.s[1]
    ldr q9, [x23, #0x10]
    prfm pldl1keep, [x5, #0x80]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    fmla v30.4s, v10.4s, v6.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    fmla v27.4s, v11.4s, v3.s[3]
    ldr q3, [x6, #0]
    fmla v28.4s, v11.4s, v4.s[3]
    ldr q4, [x5, #0]
    fmla v29.4s, v11.4s, v5.s[3]
    ldr q5, [x4, #0]
    fmla v30.4s, v11.4s, v6.s[3]
    ldr q6, [x2, #0]
    ldr q11, [x23, #0x30]
    bge label_193
KAI_ASM_LABEL(label_194)  // Height 7: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x7, x7, #0x10
    add x6, x6, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v30.4s, v8.4s, v6.s[0]
    add x2, x2, #0x10
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    sub x20, x20, #0x4
    add x23, x23, #0x40
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v30.4s, v9.4s, v6.s[1]
    prfm pldl1keep, [x2, #0x80]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    fmla v30.4s, v10.4s, v6.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
    fmla v27.4s, v11.4s, v3.s[3]
    fmla v28.4s, v11.4s, v4.s[3]
    fmla v29.4s, v11.4s, v5.s[3]
    fmla v30.4s, v11.4s, v6.s[3]
KAI_ASM_LABEL(label_195)  // Height 7: Multiply loop: Main loop skip
    cbz x20, label_197
KAI_ASM_LABEL(label_196)  // Height 7: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr s3, [x6], #0x4
    ldr s4, [x5], #0x4
    ldr s5, [x4], #0x4
    ldr s6, [x2], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    fmla v28.4s, v12.4s, v4.s[0]
    fmla v29.4s, v12.4s, v5.s[0]
    fmla v30.4s, v12.4s, v6.s[0]
    cbnz x20, label_196
KAI_ASM_LABEL(label_197)  // Height 7: Multiply loop: No odd multiplies
    tbz x27, #3, label_198
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_191
KAI_ASM_LABEL(label_198)  // Height 7: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    add x7, x8, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    prfm pstl1keep, [x6, #0]
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    tbz x27, #1, label_199
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmin v29.4s, v29.4s, v17.4s
    fmin v30.4s, v30.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
    fmax v29.4s, v29.4s, v16.4s
    fmax v30.4s, v30.4s, v16.4s
KAI_ASM_LABEL(label_199)  // Height 7: No activation
    cmp x24, #0x4
    bge label_202
    tbz x24, #1, label_200
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    str d27, [x7], #0x8
    str d28, [x6], #0x8
    str d29, [x5], #0x8
    str d30, [x4], #0x8
    tbz x24, #0, label_201
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    st1 { v27.s }[2], [x7]
    st1 { v28.s }[2], [x6]
    st1 { v29.s }[2], [x5]
    st1 { v30.s }[2], [x4]
    b label_201
KAI_ASM_LABEL(label_200)  // Height 7: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
    str s27, [x7, #0]
    str s28, [x6, #0]
    str s29, [x5, #0]
    str s30, [x4, #0]
KAI_ASM_LABEL(label_201)  // Height 7: Partial direct writeback: Done
    b label_203
KAI_ASM_LABEL(label_202)  // Height 7: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
    str q27, [x7, #0]
    str q28, [x6, #0]
    str q29, [x5, #0]
    str q30, [x4, #0]
KAI_ASM_LABEL(label_203)  // Height 7: Writeback done
    subs x24, x24, #0x4
    bgt label_176
    b label_233
KAI_ASM_LABEL(label_204)  // Height 8
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_205)  // Height 8: Column loop
    cbz x22, label_213
    cmp x24, #0x4
    bge label_211
    cbnz x9, label_210
    mov x2, x24
    mov x1, SP
    cmp x2, #0x4
    blt label_207
KAI_ASM_LABEL(label_206)  // Height 8: Partial bias load: vector loop
    ld1 { v16.4s }, [x22], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_206
KAI_ASM_LABEL(label_207)  // Height 8: Partial bias load: vector loop skip
    cbz x2, label_209
KAI_ASM_LABEL(label_208)  // Height 8: Partial bias load: element loop
    ldr s16, [x22], #0x4
    subs x2, x2, #0x1
    str s16, [x1, #0]
    add x1, x1, #0x4
    bgt label_208
KAI_ASM_LABEL(label_209)  // Height 8: Partial bias load: element loop skip
KAI_ASM_LABEL(label_210)  // Height 8: Partial bias load: read skip
    mov x1, SP
    ldr q24, [x1, #0]
    b label_212
KAI_ASM_LABEL(label_211)  // Height 8: full width bias
    ldr q24, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_212)  // Height 8: bias load done
    mov v25.16b, v24.16b
    mov v26.16b, v24.16b
    mov v27.16b, v24.16b
    mov v28.16b, v24.16b
    mov v29.16b, v24.16b
    mov v30.16b, v24.16b
    mov v31.16b, v24.16b
    b label_218
KAI_ASM_LABEL(label_213)  // Height 8: no bias
    tbz x27, #0, label_217
    ldr x1, [x0, #0x58]
    cmp x24, #0x4
    add x20, x25, x1
    add x8, x20, x1
    add x7, x8, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_216
    tbz x24, #1, label_214
    ldr d24, [x25], #0x8
    ldr d25, [x20], #0x8
    mov x1, #0x8
    ldr d26, [x8], #0x8
    ldr d27, [x7], #0x8
    ldr d28, [x6], #0x8
    ldr d29, [x5], #0x8
    ldr d30, [x4], #0x8
    ldr d31, [x3], #0x8
    tbz x24, #0, label_215
    ld1 { v24.s }[2], [x25]
    ld1 { v25.s }[2], [x20]
    ld1 { v26.s }[2], [x8]
    ld1 { v27.s }[2], [x7]
    ld1 { v28.s }[2], [x6]
    ld1 { v29.s }[2], [x5]
    ld1 { v30.s }[2], [x4]
    ld1 { v31.s }[2], [x3]
    b label_215
KAI_ASM_LABEL(label_214)  // Height 8: Partial accumulate: partial_0_0
    ldr s24, [x25, #0]
    ldr s25, [x20, #0]
    mov x1, #0
    ldr s26, [x8, #0]
    ldr s27, [x7, #0]
    ldr s28, [x6, #0]
    ldr s29, [x5, #0]
    ldr s30, [x4, #0]
    ldr s31, [x3, #0]
KAI_ASM_LABEL(label_215)  // Height 8: Partial accumulate: Done
    sub x25, x25, x1
    b label_218
KAI_ASM_LABEL(label_216)  // Height 8: full accumulate
    ldr q24, [x25, #0]
    ldr q25, [x20, #0]
    ldr q26, [x8, #0]
    ldr q27, [x7, #0]
    ldr q28, [x6, #0]
    ldr q29, [x5, #0]
    ldr q30, [x4, #0]
    ldr q31, [x3, #0]
    b label_218
KAI_ASM_LABEL(label_217)  // Height 8: no accumulate
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0
KAI_ASM_LABEL(label_218)  // Height 8: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_219
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
    b label_221
KAI_ASM_LABEL(label_219)  // Height 8: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_220)  // Height 8: String loop
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
    cbnz x26, label_221
    add x21, x21, x3, LSL #2
    add x8, x8, x3, LSL #2
    add x7, x7, x3, LSL #2
    add x6, x6, x3, LSL #2
    add x5, x5, x3, LSL #2
    add x4, x4, x3, LSL #2
    add x2, x2, x3, LSL #2
    add x1, x1, x3, LSL #2
KAI_ASM_LABEL(label_221)  // Height 8: input setup done
    cmp x20, #0x4
    blt label_224
    ldr q0, [x21, #0]
    ldr q1, [x8, #0]
    cmp x20, #0x8
    ldr q2, [x7, #0]
    ldr q3, [x6, #0]
    ldr q4, [x5, #0]
    ldr q5, [x4, #0]
    ldr q6, [x2, #0]
    ldr q7, [x1, #0]
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    blt label_223
KAI_ASM_LABEL(label_222)  // Height 8: Multiply loop: Main loop head
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    sub x20, x20, #0x4
    add x21, x21, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    fmla v30.4s, v8.4s, v6.s[0]
    fmla v31.4s, v8.4s, v7.s[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    add x1, x1, #0x10
    cmp x20, #0x8
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    add x23, x23, #0x40
    prfm pldl1keep, [x21, #0x80]
    ldr q8, [x23, #0]
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    prfm pldl1keep, [x8, #0x80]
    fmla v30.4s, v9.4s, v6.s[1]
    fmla v31.4s, v9.4s, v7.s[1]
    ldr q9, [x23, #0x10]
    prfm pldl1keep, [x7, #0x80]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    prfm pldl1keep, [x1, #0x80]
    fmla v30.4s, v10.4s, v6.s[2]
    fmla v31.4s, v10.4s, v7.s[2]
    ldr q10, [x23, #0x20]
    fmla v24.4s, v11.4s, v0.s[3]
    ldr q0, [x21, #0]
    fmla v25.4s, v11.4s, v1.s[3]
    ldr q1, [x8, #0]
    fmla v26.4s, v11.4s, v2.s[3]
    ldr q2, [x7, #0]
    fmla v27.4s, v11.4s, v3.s[3]
    ldr q3, [x6, #0]
    fmla v28.4s, v11.4s, v4.s[3]
    ldr q4, [x5, #0]
    fmla v29.4s, v11.4s, v5.s[3]
    ldr q5, [x4, #0]
    fmla v30.4s, v11.4s, v6.s[3]
    ldr q6, [x2, #0]
    fmla v31.4s, v11.4s, v7.s[3]
    ldr q7, [x1, #0]
    ldr q11, [x23, #0x30]
    bge label_222
KAI_ASM_LABEL(label_223)  // Height 8: Multiply loop: Single iteration only
    fmla v24.4s, v8.4s, v0.s[0]
    fmla v25.4s, v8.4s, v1.s[0]
    add x21, x21, #0x10
    add x8, x8, #0x10
    fmla v26.4s, v8.4s, v2.s[0]
    fmla v27.4s, v8.4s, v3.s[0]
    add x7, x7, #0x10
    add x6, x6, #0x10
    fmla v28.4s, v8.4s, v4.s[0]
    fmla v29.4s, v8.4s, v5.s[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    fmla v30.4s, v8.4s, v6.s[0]
    fmla v31.4s, v8.4s, v7.s[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    fmla v24.4s, v9.4s, v0.s[1]
    fmla v25.4s, v9.4s, v1.s[1]
    prfm pldl1keep, [x21, #0x80]
    prfm pldl1keep, [x8, #0x80]
    fmla v26.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    sub x20, x20, #0x4
    add x23, x23, #0x40
    fmla v28.4s, v9.4s, v4.s[1]
    fmla v29.4s, v9.4s, v5.s[1]
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    fmla v30.4s, v9.4s, v6.s[1]
    fmla v31.4s, v9.4s, v7.s[1]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    fmla v24.4s, v10.4s, v0.s[2]
    fmla v25.4s, v10.4s, v1.s[2]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    fmla v26.4s, v10.4s, v2.s[2]
    fmla v27.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v29.4s, v10.4s, v5.s[2]
    fmla v30.4s, v10.4s, v6.s[2]
    fmla v31.4s, v10.4s, v7.s[2]
    fmla v24.4s, v11.4s, v0.s[3]
    fmla v25.4s, v11.4s, v1.s[3]
    fmla v26.4s, v11.4s, v2.s[3]
    fmla v27.4s, v11.4s, v3.s[3]
    fmla v28.4s, v11.4s, v4.s[3]
    fmla v29.4s, v11.4s, v5.s[3]
    fmla v30.4s, v11.4s, v6.s[3]
    fmla v31.4s, v11.4s, v7.s[3]
KAI_ASM_LABEL(label_224)  // Height 8: Multiply loop: Main loop skip
    cbz x20, label_226
KAI_ASM_LABEL(label_225)  // Height 8: Multiply loop: Odd block loop
    ldr s0, [x21], #0x4
    ldr s1, [x8], #0x4
    sub x20, x20, #0x1
    ldr s2, [x7], #0x4
    ldr s3, [x6], #0x4
    ldr s4, [x5], #0x4
    ldr s5, [x4], #0x4
    ldr s6, [x2], #0x4
    ldr s7, [x1], #0x4
    ldr q12, [x23, #0]
    add x23, x23, #0x10
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    fmla v28.4s, v12.4s, v4.s[0]
    fmla v29.4s, v12.4s, v5.s[0]
    fmla v30.4s, v12.4s, v6.s[0]
    fmla v31.4s, v12.4s, v7.s[0]
    cbnz x20, label_225
KAI_ASM_LABEL(label_226)  // Height 8: Multiply loop: No odd multiplies
    tbz x27, #3, label_227
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_220
KAI_ASM_LABEL(label_227)  // Height 8: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x25, #0]
    add x20, x25, x1
    prfm pstl1keep, [x20, #0]
    add x8, x20, x1
    prfm pstl1keep, [x8, #0]
    add x7, x8, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    prfm pstl1keep, [x6, #0]
    prfm pstl1keep, [x5, #0]
    add x3, x4, x1
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    tbz x27, #1, label_228
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.4s }, [x2]
    ld1r { v16.4s }, [x1]
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmin v29.4s, v29.4s, v17.4s
    fmin v30.4s, v30.4s, v17.4s
    fmin v31.4s, v31.4s, v17.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
    fmax v29.4s, v29.4s, v16.4s
    fmax v30.4s, v30.4s, v16.4s
    fmax v31.4s, v31.4s, v16.4s
KAI_ASM_LABEL(label_228)  // Height 8: No activation
    cmp x24, #0x4
    bge label_231
    tbz x24, #1, label_229
    str d24, [x25], #0x8
    str d25, [x20], #0x8
    str d26, [x8], #0x8
    str d27, [x7], #0x8
    str d28, [x6], #0x8
    str d29, [x5], #0x8
    str d30, [x4], #0x8
    str d31, [x3], #0x8
    tbz x24, #0, label_230
    st1 { v24.s }[2], [x25]
    st1 { v25.s }[2], [x20]
    st1 { v26.s }[2], [x8]
    st1 { v27.s }[2], [x7]
    st1 { v28.s }[2], [x6]
    st1 { v29.s }[2], [x5]
    st1 { v30.s }[2], [x4]
    st1 { v31.s }[2], [x3]
    b label_230
KAI_ASM_LABEL(label_229)  // Height 8: Partial direct writeback: partial_0_0
    str s24, [x25, #0]
    str s25, [x20, #0]
    str s26, [x8, #0]
    str s27, [x7, #0]
    str s28, [x6, #0]
    str s29, [x5, #0]
    str s30, [x4, #0]
    str s31, [x3, #0]
KAI_ASM_LABEL(label_230)  // Height 8: Partial direct writeback: Done
    b label_232
KAI_ASM_LABEL(label_231)  // Height 8: Full writeback
    str q24, [x25, #0]
    add x25, x25, #0x10
    str q25, [x20, #0]
    str q26, [x8, #0]
    str q27, [x7, #0]
    str q28, [x6, #0]
    str q29, [x5, #0]
    str q30, [x4, #0]
    str q31, [x3, #0]
KAI_ASM_LABEL(label_232)  // Height 8: Writeback done
    subs x24, x24, #0x4
    bgt label_205
    add x9, x9, #0x8
    cmp x9, x28
    blt label_1
KAI_ASM_LABEL(label_233)  // Exit
    add SP, SP, #0x10
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldr d12, [sp, 104]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_fp32_mla_8x4)

    KAI_ASM_END
