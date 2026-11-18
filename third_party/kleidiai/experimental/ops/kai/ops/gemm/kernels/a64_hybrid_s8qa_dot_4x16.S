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

    KAI_ASM_CODE(a64_hybrid_s8qa_dot_4x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_s8qa_dot_4x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_s8qa_dot_4x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_s8qa_dot_4x16)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    stp d8, d9, [sp, 56]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    mov x26, #0
    ldr x2, [x0, #0x60]
    ldr x1, [x0, #0x18]
    cmp x2, #0
    cset x25, NE
    cmp x1, #0
    ldr x1, [x0, #0x78]
    cset x2, NE
    ldr x24, [x0, #0]
    cmp x1, #0
    cset x1, NE
    orr x25, x25, x2, LSL #3
    orr x25, x25, x1, LSL #4
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x24, x26
    cmp x1, #0x4
    bge label_91
    cmp x1, #0x2
    bgt label_61
    beq label_31
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    movi v11.4s, #0
    movi v15.16b, #0x1
    bic x25, x25, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_4
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_5)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    cbnz x23, label_6
    add x6, x6, x3
KAI_ASM_LABEL(label_6)  // Height 1: input setup done
    cmp x5, #0x10
    blt label_11
    ldr q0, [x6, #0]
    ldr q4, [x20, #0]
    cmp x5, #0x20
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    blt label_9
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    ldr q4, [x20, #0x70]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    ldr q5, [x20, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    ldr q7, [x20, #0xa0]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    tbnz x25, #31, label_8
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: unique 1: skip row sum
    ldr q0, [x6, #0]
    ldr q4, [x20, #0]
    sub x5, x5, #0x10
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    cmp x5, #0x20
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    prfm pldl1keep, [x6, #0x80]
    bge label_7
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    ldr q4, [x20, #0x70]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    ldr q5, [x20, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    ldr q7, [x20, #0xa0]
    sub x5, x5, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    tbnz x25, #31, label_10
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: unique 2: skip row sum
    prfm pldl1keep, [x6, #0x80]
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Main loop skip
    cbz x5, label_18
    cmp x5, #0x4
    blt label_14
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x6], #0x4
    tbnz x25, #31, label_13
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
KAI_ASM_LABEL(label_13)  // Height 1: Multiply loop: unique 3: skip row sum
    ldr q6, [x20, #0]
    ldr q7, [x20, #0x10]
    sub x5, x5, #0x4
    ldr q8, [x20, #0x20]
    ldr q9, [x20, #0x30]
    cmp x5, #0x4
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e0d0)  // sdot v16.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0f1)  // sdot v17.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e112)  // sdot v18.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e133)  // sdot v19.4s, v9.16b, v0.4b[0]
    bge label_12
KAI_ASM_LABEL(label_14)  // Height 1: Multiply loop: Skip odd blocks
    cbz x5, label_18
    tbz x5, #1, label_15
    ldr h0, [x6], #0x2
    tbz x5, #0, label_16
    ld1 { v0.b }[2], [x6]
    b label_16
KAI_ASM_LABEL(label_15)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x6, #0]
KAI_ASM_LABEL(label_16)  // Height 1: Multiply loop: Ragged operand read: Done
    tbnz x25, #31, label_17
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
KAI_ASM_LABEL(label_17)  // Height 1: Multiply loop: unique 4: skip row sum
    ldr q10, [x20, #0]
    ldr q4, [x20, #0x10]
    ldr q5, [x20, #0x20]
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e150)  // sdot v16.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e091)  // sdot v17.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0b2)  // sdot v18.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0d3)  // sdot v19.4s, v6.16b, v0.4b[0]
KAI_ASM_LABEL(label_18)  // Height 1: Multiply loop: No odd multiplies
    tbz x25, #3, label_19
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_5
KAI_ASM_LABEL(label_19)  // Height 1: no string loop
    prfm pstl1keep, [x22, #0]
    tbnz x25, #31, label_20
    add x1, x0, #0x90
    addp v11.4s, v11.4s, v11.4s
    ld1r { v1.4s }, [x1]
    neg v1.4s, v1.4s
    addp v11.4s, v11.4s, v11.4s
    mul v11.4s, v11.4s, v1.4s
KAI_ASM_LABEL(label_20)  // Height 1: skip row sum fixup
    ldr q0, [x8, #0]
    ldr q1, [x8, #0x10]
    add v16.4s, v16.4s, v11.4s
    add v17.4s, v17.4s, v11.4s
    ldr q2, [x8, #0x20]
    ldr q3, [x8, #0x30]
    add v18.4s, v18.4s, v11.4s
    add v19.4s, v19.4s, v11.4s
    add x2, x0, #0x98
    add x1, x0, #0x9c
    ld1r { v4.4s }, [x2]
    add v16.4s, v16.4s, v0.4s
    ld1r { v0.4s }, [x1]
    add v17.4s, v17.4s, v1.4s
    add x2, x0, #0x94
    add x1, x0, #0xa4
    ld1r { v6.4s }, [x1]
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add x1, x0, #0xa0
    ld1r { v5.4s }, [x1]
    sqdmulh v16.4s, v16.4s, v4.4s
    cmp x21, #0x10
    orr x25, x25, #0x80000000
    sqdmulh v17.4s, v17.4s, v4.4s
    add x8, x8, #0x40
    sqdmulh v18.4s, v18.4s, v4.4s
    sqdmulh v19.4s, v19.4s, v4.4s
    ld1r { v4.4s }, [x2]
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v0.4s
    srshl v18.4s, v18.4s, v0.4s
    srshl v19.4s, v19.4s, v0.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v16.16b, v16.16b, v17.16b
    bge label_29
    tbz x21, #3, label_24
    str d16, [x22], #0x8
    tbz x21, #2, label_22
    st1 { v16.s }[2], [x22], #0x4
    tbz x21, #1, label_21
    st1 { v16.h }[6], [x22], #0x2
    tbz x21, #0, label_28
    st1 { v16.b }[14], [x22]
    b label_28
KAI_ASM_LABEL(label_21)  // Height 1: Partial direct writeback: partial_0_12
    tbz x21, #0, label_28
    st1 { v16.b }[12], [x22]
    b label_28
KAI_ASM_LABEL(label_22)  // Height 1: Partial direct writeback: partial_1_8
    tbz x21, #1, label_23
    st1 { v16.h }[4], [x22], #0x2
    tbz x21, #0, label_28
    st1 { v16.b }[10], [x22]
    b label_28
KAI_ASM_LABEL(label_23)  // Height 1: Partial direct writeback: partial_0_8
    tbz x21, #0, label_28
    st1 { v16.b }[8], [x22]
    b label_28
KAI_ASM_LABEL(label_24)  // Height 1: Partial direct writeback: partial_2_0
    tbz x21, #2, label_26
    str s16, [x22], #0x4
    tbz x21, #1, label_25
    st1 { v16.h }[2], [x22], #0x2
    tbz x21, #0, label_28
    st1 { v16.b }[6], [x22]
    b label_28
KAI_ASM_LABEL(label_25)  // Height 1: Partial direct writeback: partial_0_4
    tbz x21, #0, label_28
    st1 { v16.b }[4], [x22]
    b label_28
KAI_ASM_LABEL(label_26)  // Height 1: Partial direct writeback: partial_1_0
    tbz x21, #1, label_27
    str h16, [x22], #0x2
    tbz x21, #0, label_28
    st1 { v16.b }[2], [x22]
    b label_28
KAI_ASM_LABEL(label_27)  // Height 1: Partial direct writeback: partial_0_0
    str b16, [x22, #0]
KAI_ASM_LABEL(label_28)  // Height 1: Partial direct writeback: Done
    b label_30
KAI_ASM_LABEL(label_29)  // Height 1: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_30)  // Height 1: Writeback done
    subs x21, x21, #0x10
    bgt label_2
    b label_121
KAI_ASM_LABEL(label_31)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    movi v11.4s, #0
    movi v12.4s, #0
    movi v15.16b, #0x1
    bic x25, x25, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_32)  // Height 2: Column loop
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_34
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    b label_36
KAI_ASM_LABEL(label_34)  // Height 2: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_35)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    cbnz x23, label_36
    add x6, x6, x3
    add x4, x4, x3
KAI_ASM_LABEL(label_36)  // Height 2: input setup done
    cmp x5, #0x10
    blt label_41
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    cmp x5, #0x20
    ldr q4, [x20, #0]
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    blt label_39
KAI_ASM_LABEL(label_37)  // Height 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    ldr q4, [x20, #0x70]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    ldr q5, [x20, #0x80]
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    tbnz x25, #31, label_38
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
KAI_ASM_LABEL(label_38)  // Height 2: Multiply loop: unique 5: skip row sum
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    sub x5, x5, #0x10
    ldr q4, [x20, #0]
    ldr q5, [x20, #0x10]
    cmp x5, #0x20
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    bge label_37
KAI_ASM_LABEL(label_39)  // Height 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    ldr q4, [x20, #0x70]
    sub x5, x5, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    ldr q5, [x20, #0x80]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    ldr q6, [x20, #0x90]
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    tbnz x25, #31, label_40
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
KAI_ASM_LABEL(label_40)  // Height 2: Multiply loop: unique 6: skip row sum
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
KAI_ASM_LABEL(label_41)  // Height 2: Multiply loop: Main loop skip
    cbz x5, label_48
    cmp x5, #0x4
    blt label_44
KAI_ASM_LABEL(label_42)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x6], #0x4
    ldr s1, [x4], #0x4
    tbnz x25, #31, label_43
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
KAI_ASM_LABEL(label_43)  // Height 2: Multiply loop: unique 7: skip row sum
    ldr q6, [x20, #0]
    ldr q7, [x20, #0x10]
    sub x5, x5, #0x4
    ldr q8, [x20, #0x20]
    ldr q9, [x20, #0x30]
    cmp x5, #0x4
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e0d0)  // sdot v16.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d4)  // sdot v20.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e0f1)  // sdot v17.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f5)  // sdot v21.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e112)  // sdot v18.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e116)  // sdot v22.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e133)  // sdot v19.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e137)  // sdot v23.4s, v9.16b, v1.4b[0]
    bge label_42
KAI_ASM_LABEL(label_44)  // Height 2: Multiply loop: Skip odd blocks
    cbz x5, label_48
    tbz x5, #1, label_45
    ldr h0, [x6], #0x2
    ldr h1, [x4], #0x2
    tbz x5, #0, label_46
    ld1 { v0.b }[2], [x6]
    ld1 { v1.b }[2], [x4]
    b label_46
KAI_ASM_LABEL(label_45)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x6, #0]
    ldr b1, [x4, #0]
KAI_ASM_LABEL(label_46)  // Height 2: Multiply loop: Ragged operand read: Done
    tbnz x25, #31, label_47
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
KAI_ASM_LABEL(label_47)  // Height 2: Multiply loop: unique 8: skip row sum
    ldr q10, [x20, #0]
    ldr q4, [x20, #0x10]
    ldr q5, [x20, #0x20]
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e150)  // sdot v16.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e154)  // sdot v20.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e091)  // sdot v17.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e095)  // sdot v21.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e0b2)  // sdot v18.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b6)  // sdot v22.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e0d3)  // sdot v19.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d7)  // sdot v23.4s, v6.16b, v1.4b[0]
KAI_ASM_LABEL(label_48)  // Height 2: Multiply loop: No odd multiplies
    tbz x25, #3, label_49
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_35
KAI_ASM_LABEL(label_49)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x22, #0]
    add x7, x22, x1
    prfm pstl1keep, [x7, #0]
    tbnz x25, #31, label_50
    add x1, x0, #0x90
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    ld1r { v2.4s }, [x1]
    neg v2.4s, v2.4s
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    mul v11.4s, v11.4s, v2.4s
    mul v12.4s, v12.4s, v2.4s
KAI_ASM_LABEL(label_50)  // Height 2: skip row sum fixup
    ldr q0, [x8, #0]
    ldr q1, [x8, #0x10]
    add v16.4s, v16.4s, v11.4s
    add v17.4s, v17.4s, v11.4s
    ldr q2, [x8, #0x20]
    ldr q3, [x8, #0x30]
    add v18.4s, v18.4s, v11.4s
    add v19.4s, v19.4s, v11.4s
    add v20.4s, v20.4s, v12.4s
    add v21.4s, v21.4s, v12.4s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    ld1r { v4.4s }, [x1]
    add v22.4s, v22.4s, v12.4s
    add v23.4s, v23.4s, v12.4s
    add x3, x0, #0x94
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    ld1r { v5.4s }, [x1]
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add v20.4s, v20.4s, v0.4s
    ld1r { v0.4s }, [x4]
    add v21.4s, v21.4s, v1.4s
    cmp x21, #0x10
    add v22.4s, v22.4s, v2.4s
    add v23.4s, v23.4s, v3.4s
    orr x25, x25, #0x80000000
    add x8, x8, #0x40
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v4.4s
    sqdmulh v18.4s, v18.4s, v4.4s
    sqdmulh v19.4s, v19.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v4.4s
    sqdmulh v22.4s, v22.4s, v4.4s
    sqdmulh v23.4s, v23.4s, v4.4s
    ld1r { v4.4s }, [x3]
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v0.4s
    srshl v18.4s, v18.4s, v0.4s
    srshl v19.4s, v19.4s, v0.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v0.4s
    srshl v22.4s, v22.4s, v0.4s
    srshl v23.4s, v23.4s, v0.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v23.4s, v23.4s, v4.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v23.4s, v23.4s, v6.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v23.4s, v23.4s, v5.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    bge label_59
    tbz x21, #3, label_54
    str d16, [x22], #0x8
    str d20, [x7], #0x8
    tbz x21, #2, label_52
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x7], #0x4
    tbz x21, #1, label_51
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x7], #0x2
    tbz x21, #0, label_58
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x7]
    b label_58
KAI_ASM_LABEL(label_51)  // Height 2: Partial direct writeback: partial_0_12
    tbz x21, #0, label_58
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x7]
    b label_58
KAI_ASM_LABEL(label_52)  // Height 2: Partial direct writeback: partial_1_8
    tbz x21, #1, label_53
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x7], #0x2
    tbz x21, #0, label_58
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x7]
    b label_58
KAI_ASM_LABEL(label_53)  // Height 2: Partial direct writeback: partial_0_8
    tbz x21, #0, label_58
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x7]
    b label_58
KAI_ASM_LABEL(label_54)  // Height 2: Partial direct writeback: partial_2_0
    tbz x21, #2, label_56
    str s16, [x22], #0x4
    str s20, [x7], #0x4
    tbz x21, #1, label_55
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x7], #0x2
    tbz x21, #0, label_58
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x7]
    b label_58
KAI_ASM_LABEL(label_55)  // Height 2: Partial direct writeback: partial_0_4
    tbz x21, #0, label_58
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x7]
    b label_58
KAI_ASM_LABEL(label_56)  // Height 2: Partial direct writeback: partial_1_0
    tbz x21, #1, label_57
    str h16, [x22], #0x2
    str h20, [x7], #0x2
    tbz x21, #0, label_58
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x7]
    b label_58
KAI_ASM_LABEL(label_57)  // Height 2: Partial direct writeback: partial_0_0
    str b16, [x22, #0]
    str b20, [x7, #0]
KAI_ASM_LABEL(label_58)  // Height 2: Partial direct writeback: Done
    b label_60
KAI_ASM_LABEL(label_59)  // Height 2: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x7, #0]
KAI_ASM_LABEL(label_60)  // Height 2: Writeback done
    subs x21, x21, #0x10
    bgt label_32
    b label_121
KAI_ASM_LABEL(label_61)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v15.16b, #0x1
    bic x25, x25, #0x80000000
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_62)  // Height 3: Column loop
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
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_64
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    b label_66
KAI_ASM_LABEL(label_64)  // Height 3: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_65)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    cbnz x23, label_66
    add x6, x6, x3
    add x4, x4, x3
    add x2, x2, x3
KAI_ASM_LABEL(label_66)  // Height 3: input setup done
    cmp x5, #0x10
    blt label_71
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    cmp x5, #0x20
    ldr q2, [x2, #0]
    ldr q4, [x20, #0]
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    blt label_69
KAI_ASM_LABEL(label_67)  // Height 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    add x6, x6, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    ldr q4, [x20, #0x70]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    add x2, x2, #0x10
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    ldr q5, [x20, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e09b)  // sdot v27.4s, v4.16b, v2.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8b8)  // sdot v24.4s, v5.16b, v2.4b[2]
    ldr q5, [x20, #0xf0]
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d9)  // sdot v25.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8fa)  // sdot v26.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e91b)  // sdot v27.4s, v8.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e938)  // sdot v24.4s, v9.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e959)  // sdot v25.4s, v10.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e89a)  // sdot v26.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8bb)  // sdot v27.4s, v5.16b, v2.4b[3]
    tbnz x25, #31, label_68
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
KAI_ASM_LABEL(label_68)  // Height 3: Multiply loop: unique 9: skip row sum
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    sub x5, x5, #0x10
    ldr q2, [x2, #0]
    ldr q4, [x20, #0]
    cmp x5, #0x20
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    bge label_67
KAI_ASM_LABEL(label_69)  // Height 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    sub x5, x5, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    ldr q4, [x20, #0x70]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    ldr q5, [x20, #0x80]
    add x2, x2, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e09b)  // sdot v27.4s, v4.16b, v2.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8b8)  // sdot v24.4s, v5.16b, v2.4b[2]
    ldr q5, [x20, #0xf0]
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d9)  // sdot v25.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8fa)  // sdot v26.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e91b)  // sdot v27.4s, v8.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e938)  // sdot v24.4s, v9.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e959)  // sdot v25.4s, v10.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e89a)  // sdot v26.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8bb)  // sdot v27.4s, v5.16b, v2.4b[3]
    tbnz x25, #31, label_70
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
KAI_ASM_LABEL(label_70)  // Height 3: Multiply loop: unique 10: skip row sum
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
KAI_ASM_LABEL(label_71)  // Height 3: Multiply loop: Main loop skip
    cbz x5, label_78
    cmp x5, #0x4
    blt label_74
KAI_ASM_LABEL(label_72)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x6], #0x4
    ldr s1, [x4], #0x4
    ldr s2, [x2], #0x4
    tbnz x25, #31, label_73
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
KAI_ASM_LABEL(label_73)  // Height 3: Multiply loop: unique 11: skip row sum
    ldr q6, [x20, #0]
    ldr q7, [x20, #0x10]
    sub x5, x5, #0x4
    ldr q8, [x20, #0x20]
    ldr q9, [x20, #0x30]
    cmp x5, #0x4
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e0d0)  // sdot v16.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d4)  // sdot v20.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d8)  // sdot v24.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e0f1)  // sdot v17.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f5)  // sdot v21.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f9)  // sdot v25.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e112)  // sdot v18.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e116)  // sdot v22.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e11a)  // sdot v26.4s, v8.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e133)  // sdot v19.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e137)  // sdot v23.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e13b)  // sdot v27.4s, v9.16b, v2.4b[0]
    bge label_72
KAI_ASM_LABEL(label_74)  // Height 3: Multiply loop: Skip odd blocks
    cbz x5, label_78
    tbz x5, #1, label_75
    ldr h0, [x6], #0x2
    ldr h1, [x4], #0x2
    ldr h2, [x2], #0x2
    tbz x5, #0, label_76
    ld1 { v0.b }[2], [x6]
    ld1 { v1.b }[2], [x4]
    ld1 { v2.b }[2], [x2]
    b label_76
KAI_ASM_LABEL(label_75)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x6, #0]
    ldr b1, [x4, #0]
    ldr b2, [x2, #0]
KAI_ASM_LABEL(label_76)  // Height 3: Multiply loop: Ragged operand read: Done
    tbnz x25, #31, label_77
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
KAI_ASM_LABEL(label_77)  // Height 3: Multiply loop: unique 12: skip row sum
    ldr q10, [x20, #0]
    ldr q4, [x20, #0x10]
    ldr q5, [x20, #0x20]
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e150)  // sdot v16.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e154)  // sdot v20.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e158)  // sdot v24.4s, v10.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e091)  // sdot v17.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e095)  // sdot v21.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e099)  // sdot v25.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e0b2)  // sdot v18.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b6)  // sdot v22.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0ba)  // sdot v26.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e0d3)  // sdot v19.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d7)  // sdot v23.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0db)  // sdot v27.4s, v6.16b, v2.4b[0]
KAI_ASM_LABEL(label_78)  // Height 3: Multiply loop: No odd multiplies
    tbz x25, #3, label_79
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_65
KAI_ASM_LABEL(label_79)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x22, #0]
    add x7, x22, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    tbnz x25, #31, label_80
    add x1, x0, #0x90
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    ld1r { v3.4s }, [x1]
    addp v13.4s, v13.4s, v13.4s
    neg v3.4s, v3.4s
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    addp v13.4s, v13.4s, v13.4s
    mul v11.4s, v11.4s, v3.4s
    mul v12.4s, v12.4s, v3.4s
    mul v13.4s, v13.4s, v3.4s
KAI_ASM_LABEL(label_80)  // Height 3: skip row sum fixup
    ldr q0, [x8, #0]
    ldr q1, [x8, #0x10]
    add v16.4s, v16.4s, v11.4s
    add v17.4s, v17.4s, v11.4s
    ldr q2, [x8, #0x20]
    ldr q3, [x8, #0x30]
    add v18.4s, v18.4s, v11.4s
    add v19.4s, v19.4s, v11.4s
    add v20.4s, v20.4s, v12.4s
    add v21.4s, v21.4s, v12.4s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    ld1r { v4.4s }, [x1]
    add v22.4s, v22.4s, v12.4s
    add v23.4s, v23.4s, v12.4s
    add x3, x0, #0x94
    add v24.4s, v24.4s, v13.4s
    add v25.4s, v25.4s, v13.4s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    ld1r { v5.4s }, [x1]
    add v26.4s, v26.4s, v13.4s
    add v27.4s, v27.4s, v13.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    cmp x21, #0x10
    orr x25, x25, #0x80000000
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add x8, x8, #0x40
    add v20.4s, v20.4s, v0.4s
    add v21.4s, v21.4s, v1.4s
    add v22.4s, v22.4s, v2.4s
    add v23.4s, v23.4s, v3.4s
    add v24.4s, v24.4s, v0.4s
    ld1r { v0.4s }, [x4]
    add v25.4s, v25.4s, v1.4s
    add v26.4s, v26.4s, v2.4s
    add v27.4s, v27.4s, v3.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v4.4s
    sqdmulh v18.4s, v18.4s, v4.4s
    sqdmulh v19.4s, v19.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v4.4s
    sqdmulh v22.4s, v22.4s, v4.4s
    sqdmulh v23.4s, v23.4s, v4.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    sqdmulh v25.4s, v25.4s, v4.4s
    sqdmulh v26.4s, v26.4s, v4.4s
    sqdmulh v27.4s, v27.4s, v4.4s
    ld1r { v4.4s }, [x3]
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v0.4s
    srshl v18.4s, v18.4s, v0.4s
    srshl v19.4s, v19.4s, v0.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v0.4s
    srshl v22.4s, v22.4s, v0.4s
    srshl v23.4s, v23.4s, v0.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v0.4s
    srshl v26.4s, v26.4s, v0.4s
    srshl v27.4s, v27.4s, v0.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v23.4s, v23.4s, v4.4s
    add v24.4s, v24.4s, v4.4s
    add v25.4s, v25.4s, v4.4s
    add v26.4s, v26.4s, v4.4s
    add v27.4s, v27.4s, v4.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v23.4s, v23.4s, v6.4s
    smin v24.4s, v24.4s, v6.4s
    smin v25.4s, v25.4s, v6.4s
    smin v26.4s, v26.4s, v6.4s
    smin v27.4s, v27.4s, v6.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v23.4s, v23.4s, v5.4s
    smax v24.4s, v24.4s, v5.4s
    smax v25.4s, v25.4s, v5.4s
    smax v26.4s, v26.4s, v5.4s
    smax v27.4s, v27.4s, v5.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    bge label_89
    tbz x21, #3, label_84
    str d16, [x22], #0x8
    str d20, [x7], #0x8
    str d24, [x6], #0x8
    tbz x21, #2, label_82
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x7], #0x4
    st1 { v24.s }[2], [x6], #0x4
    tbz x21, #1, label_81
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x7], #0x2
    st1 { v24.h }[6], [x6], #0x2
    tbz x21, #0, label_88
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x7]
    st1 { v24.b }[14], [x6]
    b label_88
KAI_ASM_LABEL(label_81)  // Height 3: Partial direct writeback: partial_0_12
    tbz x21, #0, label_88
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x7]
    st1 { v24.b }[12], [x6]
    b label_88
KAI_ASM_LABEL(label_82)  // Height 3: Partial direct writeback: partial_1_8
    tbz x21, #1, label_83
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x7], #0x2
    st1 { v24.h }[4], [x6], #0x2
    tbz x21, #0, label_88
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x7]
    st1 { v24.b }[10], [x6]
    b label_88
KAI_ASM_LABEL(label_83)  // Height 3: Partial direct writeback: partial_0_8
    tbz x21, #0, label_88
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x7]
    st1 { v24.b }[8], [x6]
    b label_88
KAI_ASM_LABEL(label_84)  // Height 3: Partial direct writeback: partial_2_0
    tbz x21, #2, label_86
    str s16, [x22], #0x4
    str s20, [x7], #0x4
    str s24, [x6], #0x4
    tbz x21, #1, label_85
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x7], #0x2
    st1 { v24.h }[2], [x6], #0x2
    tbz x21, #0, label_88
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x7]
    st1 { v24.b }[6], [x6]
    b label_88
KAI_ASM_LABEL(label_85)  // Height 3: Partial direct writeback: partial_0_4
    tbz x21, #0, label_88
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x7]
    st1 { v24.b }[4], [x6]
    b label_88
KAI_ASM_LABEL(label_86)  // Height 3: Partial direct writeback: partial_1_0
    tbz x21, #1, label_87
    str h16, [x22], #0x2
    str h20, [x7], #0x2
    str h24, [x6], #0x2
    tbz x21, #0, label_88
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x7]
    st1 { v24.b }[2], [x6]
    b label_88
KAI_ASM_LABEL(label_87)  // Height 3: Partial direct writeback: partial_0_0
    str b16, [x22, #0]
    str b20, [x7, #0]
    str b24, [x6, #0]
KAI_ASM_LABEL(label_88)  // Height 3: Partial direct writeback: Done
    b label_90
KAI_ASM_LABEL(label_89)  // Height 3: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x7, #0]
    str q24, [x6, #0]
KAI_ASM_LABEL(label_90)  // Height 3: Writeback done
    subs x21, x21, #0x10
    bgt label_62
    b label_121
KAI_ASM_LABEL(label_91)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x22, [x0, #0x50]
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    bic x25, x25, #0x80000000
    ldr x21, [x0, #0x8]
    movi v15.16b, #0x1
    ldr x20, [x0, #0x30]
    ldr x8, [x0, #0x40]
    madd x22, x26, x1, x22
KAI_ASM_LABEL(label_92)  // Height 4: Column loop
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
    tbnz x25, #3, label_94
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x10]
    madd x6, x26, x3, x6
    add x4, x6, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_96
KAI_ASM_LABEL(label_94)  // Height 4: setup indirect input
    mov x23, #0
KAI_ASM_LABEL(label_95)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x23, LSL #0x3]
    ldr x5, [x2, x23, LSL #0x3]
    add x1, x1, x26, LSL #3
    ldr x6, [x1, #0]
    ldr x4, [x1, #0x8]
    ldr x2, [x1, #0x10]
    ldr x1, [x1, #0x18]
    cbnz x23, label_96
    add x6, x6, x3
    add x4, x4, x3
    add x2, x2, x3
    add x1, x1, x3
KAI_ASM_LABEL(label_96)  // Height 4: input setup done
    cmp x5, #0x10
    blt label_101
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    cmp x5, #0x20
    ldr q2, [x2, #0]
    ldr q3, [x1, #0]
    ldr q4, [x20, #0]
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    blt label_99
KAI_ASM_LABEL(label_97)  // Height 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    add x6, x6, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e09c)  // sdot v28.4s, v4.16b, v3.4b[0]
    ldr q4, [x20, #0x70]
    add x2, x2, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    add x1, x1, #0x10
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0bd)  // sdot v29.4s, v5.16b, v3.4b[0]
    ldr q5, [x20, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0de)  // sdot v30.4s, v6.16b, v3.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0ff)  // sdot v31.4s, v7.16b, v3.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e11c)  // sdot v28.4s, v8.16b, v3.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e13d)  // sdot v29.4s, v9.16b, v3.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e15e)  // sdot v30.4s, v10.16b, v3.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e09b)  // sdot v27.4s, v4.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e09f)  // sdot v31.4s, v4.16b, v3.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8b8)  // sdot v24.4s, v5.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8bc)  // sdot v28.4s, v5.16b, v3.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d9)  // sdot v25.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8dd)  // sdot v29.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8fa)  // sdot v26.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8fe)  // sdot v30.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e91b)  // sdot v27.4s, v8.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e91f)  // sdot v31.4s, v8.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e938)  // sdot v24.4s, v9.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e93c)  // sdot v28.4s, v9.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e959)  // sdot v25.4s, v10.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e95d)  // sdot v29.4s, v10.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e89a)  // sdot v26.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e89e)  // sdot v30.4s, v4.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8bb)  // sdot v27.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8bf)  // sdot v31.4s, v5.16b, v3.4b[3]
    tbnz x25, #31, label_98
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
    KAI_ASM_INST(0x4e8f946e)  // sdot v14.4s, v3.16b, v15.16b
KAI_ASM_LABEL(label_98)  // Height 4: Multiply loop: unique 13: skip row sum
    ldr q0, [x6, #0]
    ldr q1, [x4, #0]
    sub x5, x5, #0x10
    ldr q2, [x2, #0]
    ldr q3, [x1, #0]
    cmp x5, #0x20
    ldr q4, [x20, #0]
    ldr q5, [x20, #0x10]
    ldr q6, [x20, #0x20]
    ldr q7, [x20, #0x30]
    ldr q8, [x20, #0x40]
    ldr q9, [x20, #0x50]
    ldr q10, [x20, #0x60]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    bge label_97
KAI_ASM_LABEL(label_99)  // Height 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    sub x5, x5, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e09c)  // sdot v28.4s, v4.16b, v3.4b[0]
    ldr q4, [x20, #0x70]
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0bd)  // sdot v29.4s, v5.16b, v3.4b[0]
    ldr q5, [x20, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0de)  // sdot v30.4s, v6.16b, v3.4b[0]
    ldr q6, [x20, #0x90]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0ff)  // sdot v31.4s, v7.16b, v3.4b[0]
    ldr q7, [x20, #0xa0]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e11c)  // sdot v28.4s, v8.16b, v3.4b[1]
    ldr q8, [x20, #0xb0]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e13d)  // sdot v29.4s, v9.16b, v3.4b[1]
    ldr q9, [x20, #0xc0]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e15e)  // sdot v30.4s, v10.16b, v3.4b[1]
    ldr q10, [x20, #0xd0]
    KAI_ASM_INST(0x4fa0e093)  // sdot v19.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e09b)  // sdot v27.4s, v4.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e09f)  // sdot v31.4s, v4.16b, v3.4b[1]
    ldr q4, [x20, #0xe0]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8b4)  // sdot v20.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8b8)  // sdot v24.4s, v5.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8bc)  // sdot v28.4s, v5.16b, v3.4b[2]
    ldr q5, [x20, #0xf0]
    add x20, x20, #0x100
    KAI_ASM_INST(0x4f80e8d1)  // sdot v17.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8d5)  // sdot v21.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d9)  // sdot v25.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8dd)  // sdot v29.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e8f2)  // sdot v18.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8f6)  // sdot v22.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8fa)  // sdot v26.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8fe)  // sdot v30.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e913)  // sdot v19.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e917)  // sdot v23.4s, v8.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e91b)  // sdot v27.4s, v8.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e91f)  // sdot v31.4s, v8.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa0e930)  // sdot v16.4s, v9.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e934)  // sdot v20.4s, v9.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e938)  // sdot v24.4s, v9.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e93c)  // sdot v28.4s, v9.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e951)  // sdot v17.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e955)  // sdot v21.4s, v10.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e959)  // sdot v25.4s, v10.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e95d)  // sdot v29.4s, v10.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e896)  // sdot v22.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e89a)  // sdot v26.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e89e)  // sdot v30.4s, v4.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b7)  // sdot v23.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8bb)  // sdot v27.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8bf)  // sdot v31.4s, v5.16b, v3.4b[3]
    tbnz x25, #31, label_100
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
    KAI_ASM_INST(0x4e8f946e)  // sdot v14.4s, v3.16b, v15.16b
KAI_ASM_LABEL(label_100)  // Height 4: Multiply loop: unique 14: skip row sum
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
KAI_ASM_LABEL(label_101)  // Height 4: Multiply loop: Main loop skip
    cbz x5, label_108
    cmp x5, #0x4
    blt label_104
KAI_ASM_LABEL(label_102)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x6], #0x4
    ldr s1, [x4], #0x4
    ldr s2, [x2], #0x4
    ldr s3, [x1], #0x4
    tbnz x25, #31, label_103
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
    KAI_ASM_INST(0x4e8f946e)  // sdot v14.4s, v3.16b, v15.16b
KAI_ASM_LABEL(label_103)  // Height 4: Multiply loop: unique 15: skip row sum
    ldr q6, [x20, #0]
    ldr q7, [x20, #0x10]
    sub x5, x5, #0x4
    ldr q8, [x20, #0x20]
    ldr q9, [x20, #0x30]
    cmp x5, #0x4
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e0d0)  // sdot v16.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d4)  // sdot v20.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d8)  // sdot v24.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0dc)  // sdot v28.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e0f1)  // sdot v17.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f5)  // sdot v21.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f9)  // sdot v25.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0fd)  // sdot v29.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e112)  // sdot v18.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e116)  // sdot v22.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e11a)  // sdot v26.4s, v8.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e11e)  // sdot v30.4s, v8.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e133)  // sdot v19.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e137)  // sdot v23.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e13b)  // sdot v27.4s, v9.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e13f)  // sdot v31.4s, v9.16b, v3.4b[0]
    bge label_102
KAI_ASM_LABEL(label_104)  // Height 4: Multiply loop: Skip odd blocks
    cbz x5, label_108
    tbz x5, #1, label_105
    ldr h0, [x6], #0x2
    ldr h1, [x4], #0x2
    ldr h2, [x2], #0x2
    ldr h3, [x1], #0x2
    tbz x5, #0, label_106
    ld1 { v0.b }[2], [x6]
    ld1 { v1.b }[2], [x4]
    ld1 { v2.b }[2], [x2]
    ld1 { v3.b }[2], [x1]
    b label_106
KAI_ASM_LABEL(label_105)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x6, #0]
    ldr b1, [x4, #0]
    ldr b2, [x2, #0]
    ldr b3, [x1, #0]
KAI_ASM_LABEL(label_106)  // Height 4: Multiply loop: Ragged operand read: Done
    tbnz x25, #31, label_107
    KAI_ASM_INST(0x4e8f940b)  // sdot v11.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8f942c)  // sdot v12.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8f944d)  // sdot v13.4s, v2.16b, v15.16b
    KAI_ASM_INST(0x4e8f946e)  // sdot v14.4s, v3.16b, v15.16b
KAI_ASM_LABEL(label_107)  // Height 4: Multiply loop: unique 16: skip row sum
    ldr q10, [x20, #0]
    ldr q4, [x20, #0x10]
    ldr q5, [x20, #0x20]
    ldr q6, [x20, #0x30]
    add x20, x20, #0x40
    KAI_ASM_INST(0x4f80e150)  // sdot v16.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e154)  // sdot v20.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e158)  // sdot v24.4s, v10.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e15c)  // sdot v28.4s, v10.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e091)  // sdot v17.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e095)  // sdot v21.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e099)  // sdot v25.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e09d)  // sdot v29.4s, v4.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e0b2)  // sdot v18.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b6)  // sdot v22.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0ba)  // sdot v26.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0be)  // sdot v30.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e0d3)  // sdot v19.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d7)  // sdot v23.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0db)  // sdot v27.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0df)  // sdot v31.4s, v6.16b, v3.4b[0]
KAI_ASM_LABEL(label_108)  // Height 4: Multiply loop: No odd multiplies
    tbz x25, #3, label_109
    ldr x1, [x0, #0x10]
    add x23, x23, #0x1
    cmp x23, x1
    bne label_95
KAI_ASM_LABEL(label_109)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x22, #0]
    add x7, x22, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    add x5, x6, x1
    prfm pstl1keep, [x5, #0]
    tbnz x25, #31, label_110
    add x1, x0, #0x90
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    ld1r { v4.4s }, [x1]
    addp v13.4s, v13.4s, v13.4s
    addp v14.4s, v14.4s, v14.4s
    neg v4.4s, v4.4s
    addp v11.4s, v11.4s, v11.4s
    addp v12.4s, v12.4s, v12.4s
    addp v13.4s, v13.4s, v13.4s
    addp v14.4s, v14.4s, v14.4s
    mul v11.4s, v11.4s, v4.4s
    mul v12.4s, v12.4s, v4.4s
    mul v13.4s, v13.4s, v4.4s
    mul v14.4s, v14.4s, v4.4s
KAI_ASM_LABEL(label_110)  // Height 4: skip row sum fixup
    ldr q0, [x8, #0]
    ldr q1, [x8, #0x10]
    add v16.4s, v16.4s, v11.4s
    add v17.4s, v17.4s, v11.4s
    ldr q2, [x8, #0x20]
    ldr q3, [x8, #0x30]
    add v18.4s, v18.4s, v11.4s
    add v19.4s, v19.4s, v11.4s
    add v20.4s, v20.4s, v12.4s
    add v21.4s, v21.4s, v12.4s
    add x1, x0, #0x98
    add x4, x0, #0x9c
    ld1r { v4.4s }, [x1]
    add v22.4s, v22.4s, v12.4s
    add v23.4s, v23.4s, v12.4s
    add x3, x0, #0x94
    add v24.4s, v24.4s, v13.4s
    add v25.4s, v25.4s, v13.4s
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    ld1r { v5.4s }, [x1]
    add v26.4s, v26.4s, v13.4s
    add v27.4s, v27.4s, v13.4s
    add v28.4s, v28.4s, v14.4s
    add v29.4s, v29.4s, v14.4s
    cmp x21, #0x10
    orr x25, x25, #0x80000000
    add v30.4s, v30.4s, v14.4s
    add v31.4s, v31.4s, v14.4s
    add x8, x8, #0x40
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add v20.4s, v20.4s, v0.4s
    add v21.4s, v21.4s, v1.4s
    add v22.4s, v22.4s, v2.4s
    add v23.4s, v23.4s, v3.4s
    add v24.4s, v24.4s, v0.4s
    add v25.4s, v25.4s, v1.4s
    add v26.4s, v26.4s, v2.4s
    add v27.4s, v27.4s, v3.4s
    add v28.4s, v28.4s, v0.4s
    ld1r { v0.4s }, [x4]
    add v29.4s, v29.4s, v1.4s
    add v30.4s, v30.4s, v2.4s
    add v31.4s, v31.4s, v3.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v4.4s
    sqdmulh v18.4s, v18.4s, v4.4s
    sqdmulh v19.4s, v19.4s, v4.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v4.4s
    sqdmulh v22.4s, v22.4s, v4.4s
    sqdmulh v23.4s, v23.4s, v4.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    sqdmulh v25.4s, v25.4s, v4.4s
    sqdmulh v26.4s, v26.4s, v4.4s
    sqdmulh v27.4s, v27.4s, v4.4s
    sqdmulh v28.4s, v28.4s, v4.4s
    sqdmulh v29.4s, v29.4s, v4.4s
    sqdmulh v30.4s, v30.4s, v4.4s
    sqdmulh v31.4s, v31.4s, v4.4s
    ld1r { v4.4s }, [x3]
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v0.4s
    srshl v18.4s, v18.4s, v0.4s
    srshl v19.4s, v19.4s, v0.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v0.4s
    srshl v22.4s, v22.4s, v0.4s
    srshl v23.4s, v23.4s, v0.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v0.4s
    srshl v26.4s, v26.4s, v0.4s
    srshl v27.4s, v27.4s, v0.4s
    srshl v28.4s, v28.4s, v0.4s
    srshl v29.4s, v29.4s, v0.4s
    srshl v30.4s, v30.4s, v0.4s
    srshl v31.4s, v31.4s, v0.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v23.4s, v23.4s, v4.4s
    add v24.4s, v24.4s, v4.4s
    add v25.4s, v25.4s, v4.4s
    add v26.4s, v26.4s, v4.4s
    add v27.4s, v27.4s, v4.4s
    add v28.4s, v28.4s, v4.4s
    add v29.4s, v29.4s, v4.4s
    add v30.4s, v30.4s, v4.4s
    add v31.4s, v31.4s, v4.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v23.4s, v23.4s, v6.4s
    smin v24.4s, v24.4s, v6.4s
    smin v25.4s, v25.4s, v6.4s
    smin v26.4s, v26.4s, v6.4s
    smin v27.4s, v27.4s, v6.4s
    smin v28.4s, v28.4s, v6.4s
    smin v29.4s, v29.4s, v6.4s
    smin v30.4s, v30.4s, v6.4s
    smin v31.4s, v31.4s, v6.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v23.4s, v23.4s, v5.4s
    smax v24.4s, v24.4s, v5.4s
    smax v25.4s, v25.4s, v5.4s
    smax v26.4s, v26.4s, v5.4s
    smax v27.4s, v27.4s, v5.4s
    smax v28.4s, v28.4s, v5.4s
    smax v29.4s, v29.4s, v5.4s
    smax v30.4s, v30.4s, v5.4s
    smax v31.4s, v31.4s, v5.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v29.8h, v30.8h, v31.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    uzp1 v28.16b, v28.16b, v29.16b
    bge label_119
    tbz x21, #3, label_114
    str d16, [x22], #0x8
    str d20, [x7], #0x8
    str d24, [x6], #0x8
    str d28, [x5], #0x8
    tbz x21, #2, label_112
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x7], #0x4
    st1 { v24.s }[2], [x6], #0x4
    st1 { v28.s }[2], [x5], #0x4
    tbz x21, #1, label_111
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x7], #0x2
    st1 { v24.h }[6], [x6], #0x2
    st1 { v28.h }[6], [x5], #0x2
    tbz x21, #0, label_118
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x7]
    st1 { v24.b }[14], [x6]
    st1 { v28.b }[14], [x5]
    b label_118
KAI_ASM_LABEL(label_111)  // Height 4: Partial direct writeback: partial_0_12
    tbz x21, #0, label_118
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x7]
    st1 { v24.b }[12], [x6]
    st1 { v28.b }[12], [x5]
    b label_118
KAI_ASM_LABEL(label_112)  // Height 4: Partial direct writeback: partial_1_8
    tbz x21, #1, label_113
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x7], #0x2
    st1 { v24.h }[4], [x6], #0x2
    st1 { v28.h }[4], [x5], #0x2
    tbz x21, #0, label_118
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x7]
    st1 { v24.b }[10], [x6]
    st1 { v28.b }[10], [x5]
    b label_118
KAI_ASM_LABEL(label_113)  // Height 4: Partial direct writeback: partial_0_8
    tbz x21, #0, label_118
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x7]
    st1 { v24.b }[8], [x6]
    st1 { v28.b }[8], [x5]
    b label_118
KAI_ASM_LABEL(label_114)  // Height 4: Partial direct writeback: partial_2_0
    tbz x21, #2, label_116
    str s16, [x22], #0x4
    str s20, [x7], #0x4
    str s24, [x6], #0x4
    str s28, [x5], #0x4
    tbz x21, #1, label_115
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x7], #0x2
    st1 { v24.h }[2], [x6], #0x2
    st1 { v28.h }[2], [x5], #0x2
    tbz x21, #0, label_118
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x7]
    st1 { v24.b }[6], [x6]
    st1 { v28.b }[6], [x5]
    b label_118
KAI_ASM_LABEL(label_115)  // Height 4: Partial direct writeback: partial_0_4
    tbz x21, #0, label_118
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x7]
    st1 { v24.b }[4], [x6]
    st1 { v28.b }[4], [x5]
    b label_118
KAI_ASM_LABEL(label_116)  // Height 4: Partial direct writeback: partial_1_0
    tbz x21, #1, label_117
    str h16, [x22], #0x2
    str h20, [x7], #0x2
    str h24, [x6], #0x2
    str h28, [x5], #0x2
    tbz x21, #0, label_118
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x7]
    st1 { v24.b }[2], [x6]
    st1 { v28.b }[2], [x5]
    b label_118
KAI_ASM_LABEL(label_117)  // Height 4: Partial direct writeback: partial_0_0
    str b16, [x22, #0]
    str b20, [x7, #0]
    str b24, [x6, #0]
    str b28, [x5, #0]
KAI_ASM_LABEL(label_118)  // Height 4: Partial direct writeback: Done
    b label_120
KAI_ASM_LABEL(label_119)  // Height 4: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x7, #0]
    str q24, [x6, #0]
    str q28, [x5, #0]
KAI_ASM_LABEL(label_120)  // Height 4: Writeback done
    subs x21, x21, #0x10
    bgt label_92
    add x26, x26, #0x4
    cmp x26, x24
    blt label_1
KAI_ASM_LABEL(label_121)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp d8, d9, [sp, 56]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_s8qa_dot_4x16)

    KAI_ASM_END
