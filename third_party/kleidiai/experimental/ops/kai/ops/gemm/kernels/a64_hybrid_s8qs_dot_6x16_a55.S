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

    KAI_ASM_CODE(a64_hybrid_s8qs_dot_6x16_a55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_s8qs_dot_6x16_a55)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_s8qs_dot_6x16_a55)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_s8qs_dot_6x16_a55)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x17, #0
    ldr x3, [x0, #0x60]
    ldr x2, [x0, #0x18]
    ldr x1, [x0, #0x78]
    cmp x3, #0
    ldr x16, [x0, #0]
    cset x15, NE
    cmp x2, #0
    cset x2, NE
    cmp x1, #0
    cset x1, NE
    orr x15, x15, x2, LSL #3
    orr x15, x15, x1, LSL #4
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x16, x17
    cmp x1, #0x6
    bge label_136
    cmp x1, #0x4
    bgt label_109
    beq label_82
    cmp x1, #0x2
    bgt label_55
    beq label_28
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    ldr x4, [x0, #0x28]
    tbnz x15, #3, label_4
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x4, x27
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_5)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    cbnz x14, label_6
    add x27, x27, x4
KAI_ASM_LABEL(label_6)  // Height 1: input setup done
    cmp x26, #0x10
    blt label_9
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_8
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr d6, [x9, #0x20]
    ldr x2, [x9, #0x28]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    ldr d7, [x9, #0x30]
    add x27, x27, #0x10
    ldr x1, [x9, #0x38]
    sub x26, x26, #0x10
    mov v6.d[1], x2
    ldr x2, [x9, #0x48]
    ldr x3, [x27, #0x8]
    cmp x26, #0x20
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr d7, [x9, #0x50]
    ldr x1, [x9, #0x58]
    mov v6.d[1], x2
    ldr x2, [x9, #0x68]
    prfm pldl1keep, [x27, #0x80]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    ldr d7, [x9, #0x70]
    ldr x1, [x9, #0x78]
    mov v6.d[1], x2
    ldr x2, [x9, #0x88]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    ldr d7, [x9, #0x90]
    ldr x1, [x9, #0x98]
    mov v6.d[1], x2
    ldr x2, [x9, #0xa8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    ldr d7, [x9, #0xb0]
    ldr x1, [x9, #0xb8]
    mov v6.d[1], x2
    ldr x2, [x9, #0xc8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    ldr d7, [x9, #0xd0]
    ldr x1, [x9, #0xd8]
    mov v6.d[1], x2
    ldr x2, [x9, #0xe8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    ldr d7, [x9, #0xf0]
    ldr x1, [x9, #0xf8]
    add x9, x9, #0x100
    mov v6.d[1], x2
    ldr x2, [x9, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    ldr d7, [x9, #0x10]
    ldr x1, [x9, #0x18]
    mov v6.d[1], x2
    mov v0.d[1], x3
    mov v7.d[1], x1
    bge label_7
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    ldr q7, [x9, #0x30]
    add x27, x27, #0x10
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr q7, [x9, #0x50]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    ldr q7, [x9, #0xf0]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop skip
    cbz x26, label_14
    cmp x26, #0x4
    blt label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr q6, [x9, #0]
    cmp x26, #0x4
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    ldr q7, [x9, #0x30]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    bge label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Skip odd blocks
    cbz x26, label_14
    tbz x26, #1, label_12
    ldr h0, [x27], #0x2
    tbz x26, #0, label_13
    ld1 { v0.b }[2], [x27]
    b label_13
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
KAI_ASM_LABEL(label_13)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    ldr q7, [x9, #0x30]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
KAI_ASM_LABEL(label_14)  // Height 1: Multiply loop: No odd multiplies
    tbz x15, #3, label_15
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_5
KAI_ASM_LABEL(label_15)  // Height 1: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    prfm pstl1keep, [x13, #0]
    add v9.4s, v9.4s, v1.4s
    add v10.4s, v10.4s, v2.4s
    add x28, x28, #0x40
    add v11.4s, v11.4s, v3.4s
    tbz x15, #4, label_16
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_17
KAI_ASM_LABEL(label_16)  // Height 1: per layer parameters
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
KAI_ASM_LABEL(label_17)  // Height 1: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    add x1, x0, #0x94
    add x2, x0, #0xa4
    ld1r { v4.4s }, [x1]
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    cmp x10, #0x10
    ld1r { v5.4s }, [x1]
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v8.16b, v8.16b, v9.16b
    bge label_26
    tbz x10, #3, label_21
    str d8, [x13], #0x8
    tbz x10, #2, label_19
    st1 { v8.s }[2], [x13], #0x4
    tbz x10, #1, label_18
    st1 { v8.h }[6], [x13], #0x2
    tbz x10, #0, label_25
    st1 { v8.b }[14], [x13]
    b label_25
KAI_ASM_LABEL(label_18)  // Height 1: Partial direct writeback: partial_0_12
    tbz x10, #0, label_25
    st1 { v8.b }[12], [x13]
    b label_25
KAI_ASM_LABEL(label_19)  // Height 1: Partial direct writeback: partial_1_8
    tbz x10, #1, label_20
    st1 { v8.h }[4], [x13], #0x2
    tbz x10, #0, label_25
    st1 { v8.b }[10], [x13]
    b label_25
KAI_ASM_LABEL(label_20)  // Height 1: Partial direct writeback: partial_0_8
    tbz x10, #0, label_25
    st1 { v8.b }[8], [x13]
    b label_25
KAI_ASM_LABEL(label_21)  // Height 1: Partial direct writeback: partial_2_0
    tbz x10, #2, label_23
    str s8, [x13], #0x4
    tbz x10, #1, label_22
    st1 { v8.h }[2], [x13], #0x2
    tbz x10, #0, label_25
    st1 { v8.b }[6], [x13]
    b label_25
KAI_ASM_LABEL(label_22)  // Height 1: Partial direct writeback: partial_0_4
    tbz x10, #0, label_25
    st1 { v8.b }[4], [x13]
    b label_25
KAI_ASM_LABEL(label_23)  // Height 1: Partial direct writeback: partial_1_0
    tbz x10, #1, label_24
    str h8, [x13], #0x2
    tbz x10, #0, label_25
    st1 { v8.b }[2], [x13]
    b label_25
KAI_ASM_LABEL(label_24)  // Height 1: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
KAI_ASM_LABEL(label_25)  // Height 1: Partial direct writeback: Done
    b label_27
KAI_ASM_LABEL(label_26)  // Height 1: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
KAI_ASM_LABEL(label_27)  // Height 1: Writeback done
    subs x10, x10, #0x10
    bgt label_2
    b label_163
KAI_ASM_LABEL(label_28)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_29)  // Height 2: Column loop
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    ldr x5, [x0, #0x28]
    tbnz x15, #3, label_31
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x5, x27
    add x25, x27, x5
    b label_33
KAI_ASM_LABEL(label_31)  // Height 2: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_32)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    ldr x25, [x1, #0x8]
    cbnz x14, label_33
    add x27, x27, x5
    add x25, x25, x5
KAI_ASM_LABEL(label_33)  // Height 2: input setup done
    cmp x26, #0x10
    blt label_36
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q1, [x25, #0]
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_35
KAI_ASM_LABEL(label_34)  // Height 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr x1, [x9, #0x28]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr d6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    ldr x2, [x9, #0x38]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    ldr d7, [x9, #0x30]
    mov v6.d[1], x1
    ldr x1, [x9, #0x48]
    add x27, x27, #0x10
    add x25, x25, #0x10
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr x2, [x9, #0x58]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr d7, [x9, #0x50]
    mov v6.d[1], x1
    ldr x1, [x9, #0x68]
    ldr x4, [x27, #0x8]
    sub x26, x26, #0x10
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    ldr x2, [x9, #0x78]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    ldr d7, [x9, #0x70]
    mov v6.d[1], x1
    ldr x1, [x9, #0x88]
    ldr x3, [x25, #0x8]
    cmp x26, #0x20
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    ldr x2, [x9, #0x98]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    ldr d7, [x9, #0x90]
    mov v6.d[1], x1
    ldr x1, [x9, #0xa8]
    prfm pldl1keep, [x27, #0x80]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    ldr x2, [x9, #0xb8]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    ldr d7, [x9, #0xb0]
    mov v6.d[1], x1
    ldr x1, [x9, #0xc8]
    prfm pldl1keep, [x25, #0x80]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    ldr x2, [x9, #0xd8]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    ldr d7, [x9, #0xd0]
    mov v6.d[1], x1
    ldr x1, [x9, #0xe8]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    ldr x2, [x9, #0xf8]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    ldr d7, [x9, #0xf0]
    mov v6.d[1], x1
    add x9, x9, #0x100
    ldr x1, [x9, #0x8]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    ldr d1, [x25, #0]
    ldr d7, [x9, #0x10]
    mov v6.d[1], x1
    ldr x1, [x9, #0x18]
    mov v0.d[1], x4
    mov v1.d[1], x3
    mov v7.d[1], x1
    bge label_34
KAI_ASM_LABEL(label_35)  // Height 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr q7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    ldr q7, [x9, #0xf0]
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
KAI_ASM_LABEL(label_36)  // Height 2: Multiply loop: Main loop skip
    cbz x26, label_41
    cmp x26, #0x4
    blt label_38
KAI_ASM_LABEL(label_37)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr s1, [x25], #0x4
    cmp x26, #0x4
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    bge label_37
KAI_ASM_LABEL(label_38)  // Height 2: Multiply loop: Skip odd blocks
    cbz x26, label_41
    tbz x26, #1, label_39
    ldr h0, [x27], #0x2
    ldr h1, [x25], #0x2
    tbz x26, #0, label_40
    ld1 { v0.b }[2], [x27]
    ld1 { v1.b }[2], [x25]
    b label_40
KAI_ASM_LABEL(label_39)  // Height 2: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
    ldr b1, [x25, #0]
KAI_ASM_LABEL(label_40)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
KAI_ASM_LABEL(label_41)  // Height 2: Multiply loop: No odd multiplies
    tbz x15, #3, label_42
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_32
KAI_ASM_LABEL(label_42)  // Height 2: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    ldr x1, [x0, #0x58]
    add v9.4s, v9.4s, v1.4s
    prfm pstl1keep, [x13, #0]
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v12.4s, v12.4s, v0.4s
    add x7, x13, x1
    add v13.4s, v13.4s, v1.4s
    prfm pstl1keep, [x7, #0]
    add v14.4s, v14.4s, v2.4s
    add v15.4s, v15.4s, v3.4s
    add x28, x28, #0x40
    tbz x15, #4, label_43
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_44
KAI_ASM_LABEL(label_43)  // Height 2: per layer parameters
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
KAI_ASM_LABEL(label_44)  // Height 2: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v12.4s, v12.4s, v4.4s
    sqdmulh v13.4s, v13.4s, v5.4s
    sqdmulh v14.4s, v14.4s, v6.4s
    sqdmulh v15.4s, v15.4s, v7.4s
    add x2, x0, #0x94
    add x1, x0, #0xa4
    ld1r { v4.4s }, [x2]
    srshl v8.4s, v8.4s, v0.4s
    ld1r { v6.4s }, [x1]
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v12.4s, v12.4s, v0.4s
    srshl v13.4s, v13.4s, v1.4s
    srshl v14.4s, v14.4s, v2.4s
    srshl v15.4s, v15.4s, v3.4s
    add v8.4s, v8.4s, v4.4s
    add x1, x0, #0xa0
    ld1r { v5.4s }, [x1]
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    smin v8.4s, v8.4s, v6.4s
    cmp x10, #0x10
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v12.8h, v12.8h, v13.8h
    uzp1 v13.8h, v14.8h, v15.8h
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v12.16b, v12.16b, v13.16b
    bge label_53
    tbz x10, #3, label_48
    str d8, [x13], #0x8
    str d12, [x7], #0x8
    tbz x10, #2, label_46
    st1 { v8.s }[2], [x13], #0x4
    st1 { v12.s }[2], [x7], #0x4
    tbz x10, #1, label_45
    st1 { v8.h }[6], [x13], #0x2
    st1 { v12.h }[6], [x7], #0x2
    tbz x10, #0, label_52
    st1 { v8.b }[14], [x13]
    st1 { v12.b }[14], [x7]
    b label_52
KAI_ASM_LABEL(label_45)  // Height 2: Partial direct writeback: partial_0_12
    tbz x10, #0, label_52
    st1 { v8.b }[12], [x13]
    st1 { v12.b }[12], [x7]
    b label_52
KAI_ASM_LABEL(label_46)  // Height 2: Partial direct writeback: partial_1_8
    tbz x10, #1, label_47
    st1 { v8.h }[4], [x13], #0x2
    st1 { v12.h }[4], [x7], #0x2
    tbz x10, #0, label_52
    st1 { v8.b }[10], [x13]
    st1 { v12.b }[10], [x7]
    b label_52
KAI_ASM_LABEL(label_47)  // Height 2: Partial direct writeback: partial_0_8
    tbz x10, #0, label_52
    st1 { v8.b }[8], [x13]
    st1 { v12.b }[8], [x7]
    b label_52
KAI_ASM_LABEL(label_48)  // Height 2: Partial direct writeback: partial_2_0
    tbz x10, #2, label_50
    str s8, [x13], #0x4
    str s12, [x7], #0x4
    tbz x10, #1, label_49
    st1 { v8.h }[2], [x13], #0x2
    st1 { v12.h }[2], [x7], #0x2
    tbz x10, #0, label_52
    st1 { v8.b }[6], [x13]
    st1 { v12.b }[6], [x7]
    b label_52
KAI_ASM_LABEL(label_49)  // Height 2: Partial direct writeback: partial_0_4
    tbz x10, #0, label_52
    st1 { v8.b }[4], [x13]
    st1 { v12.b }[4], [x7]
    b label_52
KAI_ASM_LABEL(label_50)  // Height 2: Partial direct writeback: partial_1_0
    tbz x10, #1, label_51
    str h8, [x13], #0x2
    str h12, [x7], #0x2
    tbz x10, #0, label_52
    st1 { v8.b }[2], [x13]
    st1 { v12.b }[2], [x7]
    b label_52
KAI_ASM_LABEL(label_51)  // Height 2: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
    str b12, [x7, #0]
KAI_ASM_LABEL(label_52)  // Height 2: Partial direct writeback: Done
    b label_54
KAI_ASM_LABEL(label_53)  // Height 2: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
    str q12, [x7, #0]
KAI_ASM_LABEL(label_54)  // Height 2: Writeback done
    subs x10, x10, #0x10
    bgt label_29
    b label_163
KAI_ASM_LABEL(label_55)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_56)  // Height 3: Column loop
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
    ldr x6, [x0, #0x28]
    tbnz x15, #3, label_58
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x6, x27
    add x25, x27, x6
    add x24, x25, x6
    b label_60
KAI_ASM_LABEL(label_58)  // Height 3: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_59)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    ldr x25, [x1, #0x8]
    ldr x24, [x1, #0x10]
    cbnz x14, label_60
    add x27, x27, x6
    add x25, x25, x6
    add x24, x24, x6
KAI_ASM_LABEL(label_60)  // Height 3: input setup done
    cmp x26, #0x10
    blt label_63
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q1, [x25, #0]
    ldr q2, [x24, #0]
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_62
KAI_ASM_LABEL(label_61)  // Height 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr x2, [x9, #0x28]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr x1, [x9, #0x38]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    ldr d6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    ldr d7, [x9, #0x30]
    mov v6.d[1], x2
    ldr x2, [x9, #0x48]
    add x24, x24, #0x10
    ldr x5, [x27, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    ldr x1, [x9, #0x58]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr x4, [x25, #0x8]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr x3, [x24, #0x8]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    ldr d7, [x9, #0x50]
    mov v6.d[1], x2
    ldr x2, [x9, #0x68]
    sub x26, x26, #0x10
    prfm pldl1keep, [x27, #0x80]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    ldr x1, [x9, #0x78]
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    cmp x26, #0x20
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    ldr d7, [x9, #0x70]
    mov v6.d[1], x2
    ldr x2, [x9, #0x88]
    prfm pldl1keep, [x24, #0x80]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    ldr x1, [x9, #0x98]
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    ldr d7, [x9, #0x90]
    mov v6.d[1], x2
    ldr x2, [x9, #0xa8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    ldr x1, [x9, #0xb8]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    ldr d7, [x9, #0xb0]
    mov v6.d[1], x2
    ldr x2, [x9, #0xc8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    ldr x1, [x9, #0xd8]
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    ldr d7, [x9, #0xd0]
    mov v6.d[1], x2
    ldr x2, [x9, #0xe8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    ldr x1, [x9, #0xf8]
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    ldr d7, [x9, #0xf0]
    mov v6.d[1], x2
    add x9, x9, #0x100
    ldr x2, [x9, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    ldr x1, [x9, #0x18]
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    ldr d1, [x25, #0]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    ldr d2, [x24, #0]
    ldr d7, [x9, #0x10]
    mov v6.d[1], x2
    mov v0.d[1], x5
    mov v1.d[1], x4
    mov v2.d[1], x3
    mov v7.d[1], x1
    bge label_61
KAI_ASM_LABEL(label_62)  // Height 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    ldr q7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    ldr q7, [x9, #0xf0]
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
KAI_ASM_LABEL(label_63)  // Height 3: Multiply loop: Main loop skip
    cbz x26, label_68
    cmp x26, #0x4
    blt label_65
KAI_ASM_LABEL(label_64)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr s1, [x25], #0x4
    cmp x26, #0x4
    ldr s2, [x24], #0x4
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    bge label_64
KAI_ASM_LABEL(label_65)  // Height 3: Multiply loop: Skip odd blocks
    cbz x26, label_68
    tbz x26, #1, label_66
    ldr h0, [x27], #0x2
    ldr h1, [x25], #0x2
    ldr h2, [x24], #0x2
    tbz x26, #0, label_67
    ld1 { v0.b }[2], [x27]
    ld1 { v1.b }[2], [x25]
    ld1 { v2.b }[2], [x24]
    b label_67
KAI_ASM_LABEL(label_66)  // Height 3: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
    ldr b1, [x25, #0]
    ldr b2, [x24, #0]
KAI_ASM_LABEL(label_67)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
KAI_ASM_LABEL(label_68)  // Height 3: Multiply loop: No odd multiplies
    tbz x15, #3, label_69
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_59
KAI_ASM_LABEL(label_69)  // Height 3: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    ldr x1, [x0, #0x58]
    add v9.4s, v9.4s, v1.4s
    prfm pstl1keep, [x13, #0]
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v12.4s, v12.4s, v0.4s
    add x7, x13, x1
    add v13.4s, v13.4s, v1.4s
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    prfm pstl1keep, [x6, #0]
    add v14.4s, v14.4s, v2.4s
    add v15.4s, v15.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add x28, x28, #0x40
    tbz x15, #4, label_70
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_71
KAI_ASM_LABEL(label_70)  // Height 3: per layer parameters
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
KAI_ASM_LABEL(label_71)  // Height 3: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v12.4s, v12.4s, v4.4s
    sqdmulh v13.4s, v13.4s, v5.4s
    sqdmulh v14.4s, v14.4s, v6.4s
    sqdmulh v15.4s, v15.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v12.4s, v12.4s, v0.4s
    srshl v13.4s, v13.4s, v1.4s
    srshl v14.4s, v14.4s, v2.4s
    srshl v15.4s, v15.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    add x1, x0, #0x94
    add x2, x0, #0xa4
    ld1r { v4.4s }, [x1]
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    cmp x10, #0x10
    ld1r { v5.4s }, [x1]
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v12.8h, v12.8h, v13.8h
    uzp1 v13.8h, v14.8h, v15.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v12.16b, v12.16b, v13.16b
    uzp1 v16.16b, v16.16b, v17.16b
    bge label_80
    tbz x10, #3, label_75
    str d8, [x13], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    tbz x10, #2, label_73
    st1 { v8.s }[2], [x13], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    tbz x10, #1, label_72
    st1 { v8.h }[6], [x13], #0x2
    st1 { v12.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    tbz x10, #0, label_79
    st1 { v8.b }[14], [x13]
    st1 { v12.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    b label_79
KAI_ASM_LABEL(label_72)  // Height 3: Partial direct writeback: partial_0_12
    tbz x10, #0, label_79
    st1 { v8.b }[12], [x13]
    st1 { v12.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    b label_79
KAI_ASM_LABEL(label_73)  // Height 3: Partial direct writeback: partial_1_8
    tbz x10, #1, label_74
    st1 { v8.h }[4], [x13], #0x2
    st1 { v12.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    tbz x10, #0, label_79
    st1 { v8.b }[10], [x13]
    st1 { v12.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    b label_79
KAI_ASM_LABEL(label_74)  // Height 3: Partial direct writeback: partial_0_8
    tbz x10, #0, label_79
    st1 { v8.b }[8], [x13]
    st1 { v12.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    b label_79
KAI_ASM_LABEL(label_75)  // Height 3: Partial direct writeback: partial_2_0
    tbz x10, #2, label_77
    str s8, [x13], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    tbz x10, #1, label_76
    st1 { v8.h }[2], [x13], #0x2
    st1 { v12.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    tbz x10, #0, label_79
    st1 { v8.b }[6], [x13]
    st1 { v12.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    b label_79
KAI_ASM_LABEL(label_76)  // Height 3: Partial direct writeback: partial_0_4
    tbz x10, #0, label_79
    st1 { v8.b }[4], [x13]
    st1 { v12.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    b label_79
KAI_ASM_LABEL(label_77)  // Height 3: Partial direct writeback: partial_1_0
    tbz x10, #1, label_78
    str h8, [x13], #0x2
    str h12, [x7], #0x2
    str h16, [x6], #0x2
    tbz x10, #0, label_79
    st1 { v8.b }[2], [x13]
    st1 { v12.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    b label_79
KAI_ASM_LABEL(label_78)  // Height 3: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
    str b12, [x7, #0]
    str b16, [x6, #0]
KAI_ASM_LABEL(label_79)  // Height 3: Partial direct writeback: Done
    b label_81
KAI_ASM_LABEL(label_80)  // Height 3: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
    str q12, [x7, #0]
    str q16, [x6, #0]
KAI_ASM_LABEL(label_81)  // Height 3: Writeback done
    subs x10, x10, #0x10
    bgt label_56
    b label_163
KAI_ASM_LABEL(label_82)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_83)  // Height 4: Column loop
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
    ldr x7, [x0, #0x28]
    tbnz x15, #3, label_85
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x7, x27
    add x25, x27, x7
    add x24, x25, x7
    add x23, x24, x7
    b label_87
KAI_ASM_LABEL(label_85)  // Height 4: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_86)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    ldr x25, [x1, #0x8]
    ldr x24, [x1, #0x10]
    ldr x23, [x1, #0x18]
    cbnz x14, label_87
    add x27, x27, x7
    add x25, x25, x7
    add x24, x24, x7
    add x23, x23, x7
KAI_ASM_LABEL(label_87)  // Height 4: input setup done
    cmp x26, #0x10
    blt label_90
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q1, [x25, #0]
    ldr q2, [x24, #0]
    ldr q3, [x23, #0]
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_89
KAI_ASM_LABEL(label_88)  // Height 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr x1, [x9, #0x28]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr x2, [x9, #0x38]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    ldr d6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr d7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    ldr x1, [x9, #0x48]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    add x23, x23, #0x10
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr x2, [x9, #0x58]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr x6, [x27, #0x8]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    ldr d7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    ldr x1, [x9, #0x68]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    ldr x5, [x25, #0x8]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    ldr x2, [x9, #0x78]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    ldr x4, [x24, #0x8]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    ldr d7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    ldr x1, [x9, #0x88]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    ldr x3, [x23, #0x8]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    ldr x2, [x9, #0x98]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    ldr d7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    ldr x1, [x9, #0xa8]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    cmp x26, #0x20
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    ldr x2, [x9, #0xb8]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    ldr d7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    ldr x1, [x9, #0xc8]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    prfm pldl1keep, [x25, #0x80]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    ldr x2, [x9, #0xd8]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    ldr d7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    ldr x1, [x9, #0xe8]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    prfm pldl1keep, [x23, #0x80]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    ldr x2, [x9, #0xf8]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    mov v6.d[1], x1
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    ldr d7, [x9, #0xf0]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    ldr x1, [x9, #0x8]
    mov v7.d[1], x2
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    ldr d1, [x25, #0]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    ldr d2, [x24, #0]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
    ldr d3, [x23, #0]
    ldr d7, [x9, #0x10]
    mov v6.d[1], x1
    ldr x1, [x9, #0x18]
    mov v0.d[1], x6
    mov v1.d[1], x5
    mov v2.d[1], x4
    mov v3.d[1], x3
    mov v7.d[1], x1
    bge label_88
KAI_ASM_LABEL(label_89)  // Height 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x23, x23, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    prfm pldl1keep, [x23, #0x80]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    ldr q7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    ldr q7, [x9, #0xf0]
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
KAI_ASM_LABEL(label_90)  // Height 4: Multiply loop: Main loop skip
    cbz x26, label_95
    cmp x26, #0x4
    blt label_92
KAI_ASM_LABEL(label_91)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr s1, [x25], #0x4
    cmp x26, #0x4
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    bge label_91
KAI_ASM_LABEL(label_92)  // Height 4: Multiply loop: Skip odd blocks
    cbz x26, label_95
    tbz x26, #1, label_93
    ldr h0, [x27], #0x2
    ldr h1, [x25], #0x2
    ldr h2, [x24], #0x2
    ldr h3, [x23], #0x2
    tbz x26, #0, label_94
    ld1 { v0.b }[2], [x27]
    ld1 { v1.b }[2], [x25]
    ld1 { v2.b }[2], [x24]
    ld1 { v3.b }[2], [x23]
    b label_94
KAI_ASM_LABEL(label_93)  // Height 4: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
    ldr b1, [x25, #0]
    ldr b2, [x24, #0]
    ldr b3, [x23, #0]
KAI_ASM_LABEL(label_94)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
KAI_ASM_LABEL(label_95)  // Height 4: Multiply loop: No odd multiplies
    tbz x15, #3, label_96
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_86
KAI_ASM_LABEL(label_96)  // Height 4: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    ldr x1, [x0, #0x58]
    add v9.4s, v9.4s, v1.4s
    prfm pstl1keep, [x13, #0]
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v12.4s, v12.4s, v0.4s
    add x7, x13, x1
    add v13.4s, v13.4s, v1.4s
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    prfm pstl1keep, [x5, #0]
    add v14.4s, v14.4s, v2.4s
    add v15.4s, v15.4s, v3.4s
    add v16.4s, v16.4s, v0.4s
    add v17.4s, v17.4s, v1.4s
    add v18.4s, v18.4s, v2.4s
    add v19.4s, v19.4s, v3.4s
    add v20.4s, v20.4s, v0.4s
    add v21.4s, v21.4s, v1.4s
    add v22.4s, v22.4s, v2.4s
    add v23.4s, v23.4s, v3.4s
    add x28, x28, #0x40
    tbz x15, #4, label_97
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_98
KAI_ASM_LABEL(label_97)  // Height 4: per layer parameters
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
KAI_ASM_LABEL(label_98)  // Height 4: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v12.4s, v12.4s, v4.4s
    sqdmulh v13.4s, v13.4s, v5.4s
    sqdmulh v14.4s, v14.4s, v6.4s
    sqdmulh v15.4s, v15.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v5.4s
    sqdmulh v22.4s, v22.4s, v6.4s
    sqdmulh v23.4s, v23.4s, v7.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v12.4s, v12.4s, v0.4s
    srshl v13.4s, v13.4s, v1.4s
    srshl v14.4s, v14.4s, v2.4s
    srshl v15.4s, v15.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v1.4s
    srshl v22.4s, v22.4s, v2.4s
    srshl v23.4s, v23.4s, v3.4s
    add x1, x0, #0x94
    add x2, x0, #0xa4
    ld1r { v4.4s }, [x1]
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    cmp x10, #0x10
    ld1r { v5.4s }, [x1]
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
    add v16.4s, v16.4s, v4.4s
    add v17.4s, v17.4s, v4.4s
    add v18.4s, v18.4s, v4.4s
    add v19.4s, v19.4s, v4.4s
    add v20.4s, v20.4s, v4.4s
    add v21.4s, v21.4s, v4.4s
    add v22.4s, v22.4s, v4.4s
    add v23.4s, v23.4s, v4.4s
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
    smin v16.4s, v16.4s, v6.4s
    smin v17.4s, v17.4s, v6.4s
    smin v18.4s, v18.4s, v6.4s
    smin v19.4s, v19.4s, v6.4s
    smin v20.4s, v20.4s, v6.4s
    smin v21.4s, v21.4s, v6.4s
    smin v22.4s, v22.4s, v6.4s
    smin v23.4s, v23.4s, v6.4s
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
    smax v16.4s, v16.4s, v5.4s
    smax v17.4s, v17.4s, v5.4s
    smax v18.4s, v18.4s, v5.4s
    smax v19.4s, v19.4s, v5.4s
    smax v20.4s, v20.4s, v5.4s
    smax v21.4s, v21.4s, v5.4s
    smax v22.4s, v22.4s, v5.4s
    smax v23.4s, v23.4s, v5.4s
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v12.8h, v12.8h, v13.8h
    uzp1 v13.8h, v14.8h, v15.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v12.16b, v12.16b, v13.16b
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    bge label_107
    tbz x10, #3, label_102
    str d8, [x13], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    tbz x10, #2, label_100
    st1 { v8.s }[2], [x13], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    tbz x10, #1, label_99
    st1 { v8.h }[6], [x13], #0x2
    st1 { v12.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    st1 { v20.h }[6], [x5], #0x2
    tbz x10, #0, label_106
    st1 { v8.b }[14], [x13]
    st1 { v12.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    st1 { v20.b }[14], [x5]
    b label_106
KAI_ASM_LABEL(label_99)  // Height 4: Partial direct writeback: partial_0_12
    tbz x10, #0, label_106
    st1 { v8.b }[12], [x13]
    st1 { v12.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    st1 { v20.b }[12], [x5]
    b label_106
KAI_ASM_LABEL(label_100)  // Height 4: Partial direct writeback: partial_1_8
    tbz x10, #1, label_101
    st1 { v8.h }[4], [x13], #0x2
    st1 { v12.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    st1 { v20.h }[4], [x5], #0x2
    tbz x10, #0, label_106
    st1 { v8.b }[10], [x13]
    st1 { v12.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    st1 { v20.b }[10], [x5]
    b label_106
KAI_ASM_LABEL(label_101)  // Height 4: Partial direct writeback: partial_0_8
    tbz x10, #0, label_106
    st1 { v8.b }[8], [x13]
    st1 { v12.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    st1 { v20.b }[8], [x5]
    b label_106
KAI_ASM_LABEL(label_102)  // Height 4: Partial direct writeback: partial_2_0
    tbz x10, #2, label_104
    str s8, [x13], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    tbz x10, #1, label_103
    st1 { v8.h }[2], [x13], #0x2
    st1 { v12.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    st1 { v20.h }[2], [x5], #0x2
    tbz x10, #0, label_106
    st1 { v8.b }[6], [x13]
    st1 { v12.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    st1 { v20.b }[6], [x5]
    b label_106
KAI_ASM_LABEL(label_103)  // Height 4: Partial direct writeback: partial_0_4
    tbz x10, #0, label_106
    st1 { v8.b }[4], [x13]
    st1 { v12.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    st1 { v20.b }[4], [x5]
    b label_106
KAI_ASM_LABEL(label_104)  // Height 4: Partial direct writeback: partial_1_0
    tbz x10, #1, label_105
    str h8, [x13], #0x2
    str h12, [x7], #0x2
    str h16, [x6], #0x2
    str h20, [x5], #0x2
    tbz x10, #0, label_106
    st1 { v8.b }[2], [x13]
    st1 { v12.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    st1 { v20.b }[2], [x5]
    b label_106
KAI_ASM_LABEL(label_105)  // Height 4: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
    str b12, [x7, #0]
    str b16, [x6, #0]
    str b20, [x5, #0]
KAI_ASM_LABEL(label_106)  // Height 4: Partial direct writeback: Done
    b label_108
KAI_ASM_LABEL(label_107)  // Height 4: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
    str q12, [x7, #0]
    str q16, [x6, #0]
    str q20, [x5, #0]
KAI_ASM_LABEL(label_108)  // Height 4: Writeback done
    subs x10, x10, #0x10
    bgt label_83
    b label_163
KAI_ASM_LABEL(label_109)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_110)  // Height 5: Column loop
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
    ldr x8, [x0, #0x28]
    tbnz x15, #3, label_112
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x8, x27
    add x25, x27, x8
    add x24, x25, x8
    add x23, x24, x8
    add x22, x23, x8
    b label_114
KAI_ASM_LABEL(label_112)  // Height 5: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_113)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    ldr x25, [x1, #0x8]
    ldr x24, [x1, #0x10]
    ldr x23, [x1, #0x18]
    ldr x22, [x1, #0x20]
    cbnz x14, label_114
    add x27, x27, x8
    add x25, x25, x8
    add x24, x24, x8
    add x23, x23, x8
    add x22, x22, x8
KAI_ASM_LABEL(label_114)  // Height 5: input setup done
    cmp x26, #0x10
    blt label_117
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q1, [x25, #0]
    ldr q2, [x24, #0]
    ldr q3, [x23, #0]
    ldr q4, [x22, #0]
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_116
KAI_ASM_LABEL(label_115)  // Height 5: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr x2, [x9, #0x28]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr x1, [x9, #0x38]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    ldr d6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    add x23, x23, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr x2, [x9, #0x48]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    ldr d7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x22, x22, #0x10
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    ldr x7, [x27, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    ldr x1, [x9, #0x58]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr x6, [x25, #0x8]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr x5, [x24, #0x8]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    ldr x2, [x9, #0x68]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    ldr d7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    ldr x4, [x23, #0x8]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    ldr x3, [x22, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    ldr x1, [x9, #0x78]
    KAI_ASM_INST(0x4fa4e0d8)  // sdot v24.4s, v6.16b, v4.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    cmp x26, #0x20
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    ldr x2, [x9, #0x88]
    KAI_ASM_INST(0x4fa4e0f9)  // sdot v25.4s, v7.16b, v4.4b[1]
    ldr d7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    prfm pldl1keep, [x25, #0x80]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    ldr x1, [x9, #0x98]
    KAI_ASM_INST(0x4fa4e0da)  // sdot v26.4s, v6.16b, v4.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    prfm pldl1keep, [x23, #0x80]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    ldr x2, [x9, #0xa8]
    KAI_ASM_INST(0x4fa4e0fb)  // sdot v27.4s, v7.16b, v4.4b[1]
    ldr d7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    prfm pldl1keep, [x22, #0x80]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    ldr x1, [x9, #0xb8]
    KAI_ASM_INST(0x4f84e8d8)  // sdot v24.4s, v6.16b, v4.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    ldr x2, [x9, #0xc8]
    KAI_ASM_INST(0x4f84e8f9)  // sdot v25.4s, v7.16b, v4.4b[2]
    ldr d7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    ldr x1, [x9, #0xd8]
    KAI_ASM_INST(0x4f84e8da)  // sdot v26.4s, v6.16b, v4.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    ldr x2, [x9, #0xe8]
    KAI_ASM_INST(0x4f84e8fb)  // sdot v27.4s, v7.16b, v4.4b[2]
    ldr d7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    ldr x1, [x9, #0xf8]
    KAI_ASM_INST(0x4fa4e8d8)  // sdot v24.4s, v6.16b, v4.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8f9)  // sdot v25.4s, v7.16b, v4.4b[3]
    ldr d7, [x9, #0xf0]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    ldr x2, [x9, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    ldr x1, [x9, #0x18]
    KAI_ASM_INST(0x4fa4e8da)  // sdot v26.4s, v6.16b, v4.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    ldr d1, [x25, #0]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    ldr d2, [x24, #0]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
    ldr d3, [x23, #0]
    KAI_ASM_INST(0x4fa4e8fb)  // sdot v27.4s, v7.16b, v4.4b[3]
    ldr d4, [x22, #0]
    ldr d7, [x9, #0x10]
    mov v6.d[1], x2
    mov v0.d[1], x7
    mov v1.d[1], x6
    mov v2.d[1], x5
    mov v3.d[1], x4
    mov v4.d[1], x3
    mov v7.d[1], x1
    bge label_115
KAI_ASM_LABEL(label_116)  // Height 5: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    add x23, x23, #0x10
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x22, x22, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x23, #0x80]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    prfm pldl1keep, [x22, #0x80]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    ldr q7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0d8)  // sdot v24.4s, v6.16b, v4.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0f9)  // sdot v25.4s, v7.16b, v4.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0da)  // sdot v26.4s, v6.16b, v4.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0fb)  // sdot v27.4s, v7.16b, v4.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8d8)  // sdot v24.4s, v6.16b, v4.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8f9)  // sdot v25.4s, v7.16b, v4.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8da)  // sdot v26.4s, v6.16b, v4.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8fb)  // sdot v27.4s, v7.16b, v4.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8d8)  // sdot v24.4s, v6.16b, v4.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8f9)  // sdot v25.4s, v7.16b, v4.4b[3]
    ldr q7, [x9, #0xf0]
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8da)  // sdot v26.4s, v6.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8fb)  // sdot v27.4s, v7.16b, v4.4b[3]
KAI_ASM_LABEL(label_117)  // Height 5: Multiply loop: Main loop skip
    cbz x26, label_122
    cmp x26, #0x4
    blt label_119
KAI_ASM_LABEL(label_118)  // Height 5: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr s1, [x25], #0x4
    cmp x26, #0x4
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr s4, [x22], #0x4
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    bge label_118
KAI_ASM_LABEL(label_119)  // Height 5: Multiply loop: Skip odd blocks
    cbz x26, label_122
    tbz x26, #1, label_120
    ldr h0, [x27], #0x2
    ldr h1, [x25], #0x2
    ldr h2, [x24], #0x2
    ldr h3, [x23], #0x2
    ldr h4, [x22], #0x2
    tbz x26, #0, label_121
    ld1 { v0.b }[2], [x27]
    ld1 { v1.b }[2], [x25]
    ld1 { v2.b }[2], [x24]
    ld1 { v3.b }[2], [x23]
    ld1 { v4.b }[2], [x22]
    b label_121
KAI_ASM_LABEL(label_120)  // Height 5: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
    ldr b1, [x25, #0]
    ldr b2, [x24, #0]
    ldr b3, [x23, #0]
    ldr b4, [x22, #0]
KAI_ASM_LABEL(label_121)  // Height 5: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
KAI_ASM_LABEL(label_122)  // Height 5: Multiply loop: No odd multiplies
    tbz x15, #3, label_123
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_113
KAI_ASM_LABEL(label_123)  // Height 5: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    ldr x1, [x0, #0x58]
    add v9.4s, v9.4s, v1.4s
    prfm pstl1keep, [x13, #0]
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v12.4s, v12.4s, v0.4s
    add x7, x13, x1
    add v13.4s, v13.4s, v1.4s
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    add x4, x5, x1
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    add v14.4s, v14.4s, v2.4s
    add v15.4s, v15.4s, v3.4s
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
    add x28, x28, #0x40
    tbz x15, #4, label_124
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_125
KAI_ASM_LABEL(label_124)  // Height 5: per layer parameters
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
KAI_ASM_LABEL(label_125)  // Height 5: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v12.4s, v12.4s, v4.4s
    sqdmulh v13.4s, v13.4s, v5.4s
    sqdmulh v14.4s, v14.4s, v6.4s
    sqdmulh v15.4s, v15.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v5.4s
    sqdmulh v22.4s, v22.4s, v6.4s
    sqdmulh v23.4s, v23.4s, v7.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    sqdmulh v25.4s, v25.4s, v5.4s
    sqdmulh v26.4s, v26.4s, v6.4s
    sqdmulh v27.4s, v27.4s, v7.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v12.4s, v12.4s, v0.4s
    srshl v13.4s, v13.4s, v1.4s
    srshl v14.4s, v14.4s, v2.4s
    srshl v15.4s, v15.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v1.4s
    srshl v22.4s, v22.4s, v2.4s
    srshl v23.4s, v23.4s, v3.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v1.4s
    srshl v26.4s, v26.4s, v2.4s
    srshl v27.4s, v27.4s, v3.4s
    add x1, x0, #0x94
    add x2, x0, #0xa4
    ld1r { v4.4s }, [x1]
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    cmp x10, #0x10
    ld1r { v5.4s }, [x1]
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
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
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
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
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
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
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v12.8h, v12.8h, v13.8h
    uzp1 v13.8h, v14.8h, v15.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v12.16b, v12.16b, v13.16b
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    bge label_134
    tbz x10, #3, label_129
    str d8, [x13], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    tbz x10, #2, label_127
    st1 { v8.s }[2], [x13], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    tbz x10, #1, label_126
    st1 { v8.h }[6], [x13], #0x2
    st1 { v12.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    st1 { v20.h }[6], [x5], #0x2
    st1 { v24.h }[6], [x4], #0x2
    tbz x10, #0, label_133
    st1 { v8.b }[14], [x13]
    st1 { v12.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    st1 { v20.b }[14], [x5]
    st1 { v24.b }[14], [x4]
    b label_133
KAI_ASM_LABEL(label_126)  // Height 5: Partial direct writeback: partial_0_12
    tbz x10, #0, label_133
    st1 { v8.b }[12], [x13]
    st1 { v12.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    st1 { v20.b }[12], [x5]
    st1 { v24.b }[12], [x4]
    b label_133
KAI_ASM_LABEL(label_127)  // Height 5: Partial direct writeback: partial_1_8
    tbz x10, #1, label_128
    st1 { v8.h }[4], [x13], #0x2
    st1 { v12.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    st1 { v20.h }[4], [x5], #0x2
    st1 { v24.h }[4], [x4], #0x2
    tbz x10, #0, label_133
    st1 { v8.b }[10], [x13]
    st1 { v12.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    st1 { v20.b }[10], [x5]
    st1 { v24.b }[10], [x4]
    b label_133
KAI_ASM_LABEL(label_128)  // Height 5: Partial direct writeback: partial_0_8
    tbz x10, #0, label_133
    st1 { v8.b }[8], [x13]
    st1 { v12.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    st1 { v20.b }[8], [x5]
    st1 { v24.b }[8], [x4]
    b label_133
KAI_ASM_LABEL(label_129)  // Height 5: Partial direct writeback: partial_2_0
    tbz x10, #2, label_131
    str s8, [x13], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    tbz x10, #1, label_130
    st1 { v8.h }[2], [x13], #0x2
    st1 { v12.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    st1 { v20.h }[2], [x5], #0x2
    st1 { v24.h }[2], [x4], #0x2
    tbz x10, #0, label_133
    st1 { v8.b }[6], [x13]
    st1 { v12.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    st1 { v20.b }[6], [x5]
    st1 { v24.b }[6], [x4]
    b label_133
KAI_ASM_LABEL(label_130)  // Height 5: Partial direct writeback: partial_0_4
    tbz x10, #0, label_133
    st1 { v8.b }[4], [x13]
    st1 { v12.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    st1 { v20.b }[4], [x5]
    st1 { v24.b }[4], [x4]
    b label_133
KAI_ASM_LABEL(label_131)  // Height 5: Partial direct writeback: partial_1_0
    tbz x10, #1, label_132
    str h8, [x13], #0x2
    str h12, [x7], #0x2
    str h16, [x6], #0x2
    str h20, [x5], #0x2
    str h24, [x4], #0x2
    tbz x10, #0, label_133
    st1 { v8.b }[2], [x13]
    st1 { v12.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    st1 { v20.b }[2], [x5]
    st1 { v24.b }[2], [x4]
    b label_133
KAI_ASM_LABEL(label_132)  // Height 5: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
    str b12, [x7, #0]
    str b16, [x6, #0]
    str b20, [x5, #0]
    str b24, [x4, #0]
KAI_ASM_LABEL(label_133)  // Height 5: Partial direct writeback: Done
    b label_135
KAI_ASM_LABEL(label_134)  // Height 5: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
    str q12, [x7, #0]
    str q16, [x6, #0]
    str q20, [x5, #0]
    str q24, [x4, #0]
KAI_ASM_LABEL(label_135)  // Height 5: Writeback done
    subs x10, x10, #0x10
    bgt label_110
    b label_163
KAI_ASM_LABEL(label_136)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x13, [x0, #0x50]
    ldr x12, [x0, #0x78]
    ldr x11, [x0, #0x88]
    ldr x10, [x0, #0x8]
    madd x13, x17, x1, x13
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x40]
KAI_ASM_LABEL(label_137)  // Height 6: Column loop
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
    ldr x21, [x0, #0x28]
    tbnz x15, #3, label_139
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x10]
    madd x27, x17, x21, x27
    add x25, x27, x21
    add x24, x25, x21
    add x23, x24, x21
    add x22, x23, x21
    add x20, x22, x21
    b label_141
KAI_ASM_LABEL(label_139)  // Height 6: setup indirect input
    mov x14, #0
KAI_ASM_LABEL(label_140)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x14, LSL #0x3]
    ldr x26, [x2, x14, LSL #0x3]
    add x1, x1, x17, LSL #3
    ldr x27, [x1, #0]
    ldr x25, [x1, #0x8]
    ldr x24, [x1, #0x10]
    ldr x23, [x1, #0x18]
    ldr x22, [x1, #0x20]
    ldr x20, [x1, #0x28]
    cbnz x14, label_141
    add x27, x27, x21
    add x25, x25, x21
    add x24, x24, x21
    add x23, x23, x21
    add x22, x22, x21
    add x20, x20, x21
KAI_ASM_LABEL(label_141)  // Height 6: input setup done
    cmp x26, #0x10
    blt label_144
    ldr q0, [x27, #0]
    cmp x26, #0x20
    ldr q1, [x25, #0]
    ldr q2, [x24, #0]
    ldr q3, [x23, #0]
    ldr q4, [x22, #0]
    ldr q5, [x20, #0]
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    blt label_143
KAI_ASM_LABEL(label_142)  // Height 6: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    ldr x2, [x9, #0x28]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    ldr x1, [x9, #0x38]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f85e0dc)  // sdot v28.4s, v6.16b, v5.4b[0]
    ldr d6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x23, x23, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    add x22, x22, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    ldr x2, [x9, #0x48]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    add x20, x20, #0x10
    KAI_ASM_INST(0x4f85e0fd)  // sdot v29.4s, v7.16b, v5.4b[0]
    ldr d7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    ldr x8, [x27, #0x8]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    ldr x7, [x25, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    ldr x1, [x9, #0x58]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    ldr x6, [x24, #0x8]
    KAI_ASM_INST(0x4f85e0de)  // sdot v30.4s, v6.16b, v5.4b[0]
    ldr d6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    ldr x4, [x22, #0x8]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    ldr x2, [x9, #0x68]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    ldr x3, [x20, #0x8]
    KAI_ASM_INST(0x4f85e0ff)  // sdot v31.4s, v7.16b, v5.4b[0]
    ldr d7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    cmp x26, #0x20
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    ldr x1, [x9, #0x78]
    KAI_ASM_INST(0x4fa4e0d8)  // sdot v24.4s, v6.16b, v4.4b[1]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4fa5e0dc)  // sdot v28.4s, v6.16b, v5.4b[1]
    ldr d6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    ldr x2, [x9, #0x88]
    KAI_ASM_INST(0x4fa4e0f9)  // sdot v25.4s, v7.16b, v4.4b[1]
    prfm pldl1keep, [x23, #0x80]
    KAI_ASM_INST(0x4fa5e0fd)  // sdot v29.4s, v7.16b, v5.4b[1]
    ldr d7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    prfm pldl1keep, [x22, #0x80]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    prfm pldl1keep, [x20, #0x80]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    ldr x1, [x9, #0x98]
    KAI_ASM_INST(0x4fa4e0da)  // sdot v26.4s, v6.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0de)  // sdot v30.4s, v6.16b, v5.4b[1]
    ldr d6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    ldr x2, [x9, #0xa8]
    KAI_ASM_INST(0x4fa4e0fb)  // sdot v27.4s, v7.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0ff)  // sdot v31.4s, v7.16b, v5.4b[1]
    ldr d7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    ldr x1, [x9, #0xb8]
    KAI_ASM_INST(0x4f84e8d8)  // sdot v24.4s, v6.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8dc)  // sdot v28.4s, v6.16b, v5.4b[2]
    ldr d6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    ldr x2, [x9, #0xc8]
    KAI_ASM_INST(0x4f84e8f9)  // sdot v25.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8fd)  // sdot v29.4s, v7.16b, v5.4b[2]
    ldr d7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    ldr x1, [x9, #0xd8]
    KAI_ASM_INST(0x4f84e8da)  // sdot v26.4s, v6.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8de)  // sdot v30.4s, v6.16b, v5.4b[2]
    ldr d6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    ldr x2, [x9, #0xe8]
    KAI_ASM_INST(0x4f84e8fb)  // sdot v27.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8ff)  // sdot v31.4s, v7.16b, v5.4b[2]
    ldr d7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    ldr x1, [x9, #0xf8]
    KAI_ASM_INST(0x4fa4e8d8)  // sdot v24.4s, v6.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8dc)  // sdot v28.4s, v6.16b, v5.4b[3]
    ldr d6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8f9)  // sdot v25.4s, v7.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8fd)  // sdot v29.4s, v7.16b, v5.4b[3]
    ldr d7, [x9, #0xf0]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    ldr x2, [x9, #0x8]
    mov v7.d[1], x1
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    ldr x1, [x9, #0x18]
    KAI_ASM_INST(0x4fa4e8da)  // sdot v26.4s, v6.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8de)  // sdot v30.4s, v6.16b, v5.4b[3]
    ldr d6, [x9, #0]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    ldr d0, [x27, #0]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    ldr d1, [x25, #0]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    ldr d2, [x24, #0]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
    ldr d3, [x23, #0]
    KAI_ASM_INST(0x4fa4e8fb)  // sdot v27.4s, v7.16b, v4.4b[3]
    ldr d4, [x22, #0]
    KAI_ASM_INST(0x4fa5e8ff)  // sdot v31.4s, v7.16b, v5.4b[3]
    ldr d5, [x20, #0]
    ldr d7, [x9, #0x10]
    mov v6.d[1], x2
    mov v0.d[1], x8
    mov v1.d[1], x7
    mov v2.d[1], x6
    mov v3.d[1], x5
    mov v4.d[1], x4
    mov v5.d[1], x3
    mov v7.d[1], x1
    bge label_142
KAI_ASM_LABEL(label_143)  // Height 6: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    add x27, x27, #0x10
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    add x25, x25, #0x10
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    add x24, x24, #0x10
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    add x23, x23, #0x10
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    add x22, x22, #0x10
    KAI_ASM_INST(0x4f85e0dc)  // sdot v28.4s, v6.16b, v5.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    add x20, x20, #0x10
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    sub x26, x26, #0x10
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    prfm pldl1keep, [x27, #0x80]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    prfm pldl1keep, [x25, #0x80]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    prfm pldl1keep, [x24, #0x80]
    KAI_ASM_INST(0x4f85e0fd)  // sdot v29.4s, v7.16b, v5.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    prfm pldl1keep, [x23, #0x80]
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x22, #0x80]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    prfm pldl1keep, [x20, #0x80]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0de)  // sdot v30.4s, v6.16b, v5.4b[0]
    ldr q6, [x9, #0x40]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0ff)  // sdot v31.4s, v7.16b, v5.4b[0]
    ldr q7, [x9, #0x50]
    KAI_ASM_INST(0x4fa0e0c8)  // sdot v8.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0cc)  // sdot v12.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d0)  // sdot v16.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d4)  // sdot v20.4s, v6.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0d8)  // sdot v24.4s, v6.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0dc)  // sdot v28.4s, v6.16b, v5.4b[1]
    ldr q6, [x9, #0x60]
    KAI_ASM_INST(0x4fa0e0e9)  // sdot v9.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ed)  // sdot v13.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f1)  // sdot v17.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f5)  // sdot v21.4s, v7.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0f9)  // sdot v25.4s, v7.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0fd)  // sdot v29.4s, v7.16b, v5.4b[1]
    ldr q7, [x9, #0x70]
    KAI_ASM_INST(0x4fa0e0ca)  // sdot v10.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ce)  // sdot v14.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0d2)  // sdot v18.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0d6)  // sdot v22.4s, v6.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0da)  // sdot v26.4s, v6.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0de)  // sdot v30.4s, v6.16b, v5.4b[1]
    ldr q6, [x9, #0x80]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e0ef)  // sdot v15.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e0f3)  // sdot v19.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e0f7)  // sdot v23.4s, v7.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa4e0fb)  // sdot v27.4s, v7.16b, v4.4b[1]
    KAI_ASM_INST(0x4fa5e0ff)  // sdot v31.4s, v7.16b, v5.4b[1]
    ldr q7, [x9, #0x90]
    KAI_ASM_INST(0x4f80e8c8)  // sdot v8.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8cc)  // sdot v12.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d4)  // sdot v20.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8d8)  // sdot v24.4s, v6.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8dc)  // sdot v28.4s, v6.16b, v5.4b[2]
    ldr q6, [x9, #0xa0]
    KAI_ASM_INST(0x4f80e8e9)  // sdot v9.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ed)  // sdot v13.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f1)  // sdot v17.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f5)  // sdot v21.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8f9)  // sdot v25.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8fd)  // sdot v29.4s, v7.16b, v5.4b[2]
    ldr q7, [x9, #0xb0]
    KAI_ASM_INST(0x4f80e8ca)  // sdot v10.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ce)  // sdot v14.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8d2)  // sdot v18.4s, v6.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8d6)  // sdot v22.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8da)  // sdot v26.4s, v6.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8de)  // sdot v30.4s, v6.16b, v5.4b[2]
    ldr q6, [x9, #0xc0]
    KAI_ASM_INST(0x4f80e8eb)  // sdot v11.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e8ef)  // sdot v15.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e8f3)  // sdot v19.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e8f7)  // sdot v23.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4f84e8fb)  // sdot v27.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4f85e8ff)  // sdot v31.4s, v7.16b, v5.4b[2]
    ldr q7, [x9, #0xd0]
    KAI_ASM_INST(0x4fa0e8c8)  // sdot v8.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8cc)  // sdot v12.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d0)  // sdot v16.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d4)  // sdot v20.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8d8)  // sdot v24.4s, v6.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8dc)  // sdot v28.4s, v6.16b, v5.4b[3]
    ldr q6, [x9, #0xe0]
    KAI_ASM_INST(0x4fa0e8e9)  // sdot v9.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ed)  // sdot v13.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f1)  // sdot v17.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f5)  // sdot v21.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8f9)  // sdot v25.4s, v7.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8fd)  // sdot v29.4s, v7.16b, v5.4b[3]
    ldr q7, [x9, #0xf0]
    KAI_ASM_INST(0x4fa0e8ca)  // sdot v10.4s, v6.16b, v0.4b[3]
    add x9, x9, #0x100
    KAI_ASM_INST(0x4fa1e8ce)  // sdot v14.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8d2)  // sdot v18.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8d6)  // sdot v22.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8da)  // sdot v26.4s, v6.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8de)  // sdot v30.4s, v6.16b, v5.4b[3]
    KAI_ASM_INST(0x4fa0e8eb)  // sdot v11.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8ef)  // sdot v15.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8f3)  // sdot v19.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8f7)  // sdot v23.4s, v7.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa4e8fb)  // sdot v27.4s, v7.16b, v4.4b[3]
    KAI_ASM_INST(0x4fa5e8ff)  // sdot v31.4s, v7.16b, v5.4b[3]
KAI_ASM_LABEL(label_144)  // Height 6: Multiply loop: Main loop skip
    cbz x26, label_149
    cmp x26, #0x4
    blt label_146
KAI_ASM_LABEL(label_145)  // Height 6: Multiply loop: Odd block loop
    ldr s0, [x27], #0x4
    sub x26, x26, #0x4
    ldr s1, [x25], #0x4
    cmp x26, #0x4
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr s4, [x22], #0x4
    ldr s5, [x20], #0x4
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0dc)  // sdot v28.4s, v6.16b, v5.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0fd)  // sdot v29.4s, v7.16b, v5.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0de)  // sdot v30.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0ff)  // sdot v31.4s, v7.16b, v5.4b[0]
    bge label_145
KAI_ASM_LABEL(label_146)  // Height 6: Multiply loop: Skip odd blocks
    cbz x26, label_149
    tbz x26, #1, label_147
    ldr h0, [x27], #0x2
    ldr h1, [x25], #0x2
    ldr h2, [x24], #0x2
    ldr h3, [x23], #0x2
    ldr h4, [x22], #0x2
    ldr h5, [x20], #0x2
    tbz x26, #0, label_148
    ld1 { v0.b }[2], [x27]
    ld1 { v1.b }[2], [x25]
    ld1 { v2.b }[2], [x24]
    ld1 { v3.b }[2], [x23]
    ld1 { v4.b }[2], [x22]
    ld1 { v5.b }[2], [x20]
    b label_148
KAI_ASM_LABEL(label_147)  // Height 6: Multiply loop: Ragged operand read: partial_0_0
    ldr b0, [x27, #0]
    ldr b1, [x25, #0]
    ldr b2, [x24, #0]
    ldr b3, [x23, #0]
    ldr b4, [x22, #0]
    ldr b5, [x20, #0]
KAI_ASM_LABEL(label_148)  // Height 6: Multiply loop: Ragged operand read: Done
    ldr q6, [x9, #0]
    ldr q7, [x9, #0x10]
    KAI_ASM_INST(0x4f80e0c8)  // sdot v8.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0cc)  // sdot v12.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d0)  // sdot v16.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d4)  // sdot v20.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0d8)  // sdot v24.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0dc)  // sdot v28.4s, v6.16b, v5.4b[0]
    ldr q6, [x9, #0x20]
    KAI_ASM_INST(0x4f80e0e9)  // sdot v9.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ed)  // sdot v13.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f1)  // sdot v17.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f5)  // sdot v21.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0f9)  // sdot v25.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0fd)  // sdot v29.4s, v7.16b, v5.4b[0]
    ldr q7, [x9, #0x30]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    add x9, x9, #0x40
    KAI_ASM_INST(0x4f81e0ce)  // sdot v14.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0d2)  // sdot v18.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0da)  // sdot v26.4s, v6.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0de)  // sdot v30.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4f80e0eb)  // sdot v11.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0ef)  // sdot v15.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0f3)  // sdot v19.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0f7)  // sdot v23.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4f85e0ff)  // sdot v31.4s, v7.16b, v5.4b[0]
KAI_ASM_LABEL(label_149)  // Height 6: Multiply loop: No odd multiplies
    tbz x15, #3, label_150
    ldr x1, [x0, #0x10]
    add x14, x14, #0x1
    cmp x14, x1
    bne label_140
KAI_ASM_LABEL(label_150)  // Height 6: no string loop
    ldr q0, [x28, #0]
    ldr q1, [x28, #0x10]
    ldr q2, [x28, #0x20]
    ldr q3, [x28, #0x30]
    add v8.4s, v8.4s, v0.4s
    ldr x1, [x0, #0x58]
    add v9.4s, v9.4s, v1.4s
    prfm pstl1keep, [x13, #0]
    add v10.4s, v10.4s, v2.4s
    add v11.4s, v11.4s, v3.4s
    add v12.4s, v12.4s, v0.4s
    add x7, x13, x1
    add v13.4s, v13.4s, v1.4s
    add x6, x7, x1
    prfm pstl1keep, [x7, #0]
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    add x4, x5, x1
    prfm pstl1keep, [x5, #0]
    add x3, x4, x1
    prfm pstl1keep, [x4, #0]
    prfm pstl1keep, [x3, #0]
    add v14.4s, v14.4s, v2.4s
    add v15.4s, v15.4s, v3.4s
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
    add v29.4s, v29.4s, v1.4s
    add v30.4s, v30.4s, v2.4s
    add v31.4s, v31.4s, v3.4s
    add x28, x28, #0x40
    tbz x15, #4, label_151
    ldr q0, [x11, #0]
    ldr q4, [x12, #0]
    ldr q1, [x11, #0x10]
    ldr q5, [x12, #0x10]
    ldr q2, [x11, #0x20]
    ldr q6, [x12, #0x20]
    ldr q3, [x11, #0x30]
    add x11, x11, #0x40
    ldr q7, [x12, #0x30]
    add x12, x12, #0x40
    b label_152
KAI_ASM_LABEL(label_151)  // Height 6: per layer parameters
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
KAI_ASM_LABEL(label_152)  // Height 6: parameters loaded
    sqdmulh v8.4s, v8.4s, v4.4s
    sqdmulh v9.4s, v9.4s, v5.4s
    sqdmulh v10.4s, v10.4s, v6.4s
    sqdmulh v11.4s, v11.4s, v7.4s
    sqdmulh v12.4s, v12.4s, v4.4s
    sqdmulh v13.4s, v13.4s, v5.4s
    sqdmulh v14.4s, v14.4s, v6.4s
    sqdmulh v15.4s, v15.4s, v7.4s
    sqdmulh v16.4s, v16.4s, v4.4s
    sqdmulh v17.4s, v17.4s, v5.4s
    sqdmulh v18.4s, v18.4s, v6.4s
    sqdmulh v19.4s, v19.4s, v7.4s
    sqdmulh v20.4s, v20.4s, v4.4s
    sqdmulh v21.4s, v21.4s, v5.4s
    sqdmulh v22.4s, v22.4s, v6.4s
    sqdmulh v23.4s, v23.4s, v7.4s
    sqdmulh v24.4s, v24.4s, v4.4s
    sqdmulh v25.4s, v25.4s, v5.4s
    sqdmulh v26.4s, v26.4s, v6.4s
    sqdmulh v27.4s, v27.4s, v7.4s
    sqdmulh v28.4s, v28.4s, v4.4s
    sqdmulh v29.4s, v29.4s, v5.4s
    sqdmulh v30.4s, v30.4s, v6.4s
    sqdmulh v31.4s, v31.4s, v7.4s
    srshl v8.4s, v8.4s, v0.4s
    srshl v9.4s, v9.4s, v1.4s
    srshl v10.4s, v10.4s, v2.4s
    srshl v11.4s, v11.4s, v3.4s
    srshl v12.4s, v12.4s, v0.4s
    srshl v13.4s, v13.4s, v1.4s
    srshl v14.4s, v14.4s, v2.4s
    srshl v15.4s, v15.4s, v3.4s
    srshl v16.4s, v16.4s, v0.4s
    srshl v17.4s, v17.4s, v1.4s
    srshl v18.4s, v18.4s, v2.4s
    srshl v19.4s, v19.4s, v3.4s
    srshl v20.4s, v20.4s, v0.4s
    srshl v21.4s, v21.4s, v1.4s
    srshl v22.4s, v22.4s, v2.4s
    srshl v23.4s, v23.4s, v3.4s
    srshl v24.4s, v24.4s, v0.4s
    srshl v25.4s, v25.4s, v1.4s
    srshl v26.4s, v26.4s, v2.4s
    srshl v27.4s, v27.4s, v3.4s
    srshl v28.4s, v28.4s, v0.4s
    srshl v29.4s, v29.4s, v1.4s
    srshl v30.4s, v30.4s, v2.4s
    srshl v31.4s, v31.4s, v3.4s
    add x1, x0, #0x94
    add x2, x0, #0xa4
    ld1r { v4.4s }, [x1]
    add x1, x0, #0xa0
    ld1r { v6.4s }, [x2]
    cmp x10, #0x10
    ld1r { v5.4s }, [x1]
    add v8.4s, v8.4s, v4.4s
    add v9.4s, v9.4s, v4.4s
    add v10.4s, v10.4s, v4.4s
    add v11.4s, v11.4s, v4.4s
    add v12.4s, v12.4s, v4.4s
    add v13.4s, v13.4s, v4.4s
    add v14.4s, v14.4s, v4.4s
    add v15.4s, v15.4s, v4.4s
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
    smin v8.4s, v8.4s, v6.4s
    smin v9.4s, v9.4s, v6.4s
    smin v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v6.4s
    smin v14.4s, v14.4s, v6.4s
    smin v15.4s, v15.4s, v6.4s
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
    smax v8.4s, v8.4s, v5.4s
    smax v9.4s, v9.4s, v5.4s
    smax v10.4s, v10.4s, v5.4s
    smax v11.4s, v11.4s, v5.4s
    smax v12.4s, v12.4s, v5.4s
    smax v13.4s, v13.4s, v5.4s
    smax v14.4s, v14.4s, v5.4s
    smax v15.4s, v15.4s, v5.4s
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
    uzp1 v8.8h, v8.8h, v9.8h
    uzp1 v9.8h, v10.8h, v11.8h
    uzp1 v12.8h, v12.8h, v13.8h
    uzp1 v13.8h, v14.8h, v15.8h
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v29.8h, v30.8h, v31.8h
    uzp1 v8.16b, v8.16b, v9.16b
    uzp1 v12.16b, v12.16b, v13.16b
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    uzp1 v28.16b, v28.16b, v29.16b
    bge label_161
    tbz x10, #3, label_156
    str d8, [x13], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    str d28, [x3], #0x8
    tbz x10, #2, label_154
    st1 { v8.s }[2], [x13], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    st1 { v28.s }[2], [x3], #0x4
    tbz x10, #1, label_153
    st1 { v8.h }[6], [x13], #0x2
    st1 { v12.h }[6], [x7], #0x2
    st1 { v16.h }[6], [x6], #0x2
    st1 { v20.h }[6], [x5], #0x2
    st1 { v24.h }[6], [x4], #0x2
    st1 { v28.h }[6], [x3], #0x2
    tbz x10, #0, label_160
    st1 { v8.b }[14], [x13]
    st1 { v12.b }[14], [x7]
    st1 { v16.b }[14], [x6]
    st1 { v20.b }[14], [x5]
    st1 { v24.b }[14], [x4]
    st1 { v28.b }[14], [x3]
    b label_160
KAI_ASM_LABEL(label_153)  // Height 6: Partial direct writeback: partial_0_12
    tbz x10, #0, label_160
    st1 { v8.b }[12], [x13]
    st1 { v12.b }[12], [x7]
    st1 { v16.b }[12], [x6]
    st1 { v20.b }[12], [x5]
    st1 { v24.b }[12], [x4]
    st1 { v28.b }[12], [x3]
    b label_160
KAI_ASM_LABEL(label_154)  // Height 6: Partial direct writeback: partial_1_8
    tbz x10, #1, label_155
    st1 { v8.h }[4], [x13], #0x2
    st1 { v12.h }[4], [x7], #0x2
    st1 { v16.h }[4], [x6], #0x2
    st1 { v20.h }[4], [x5], #0x2
    st1 { v24.h }[4], [x4], #0x2
    st1 { v28.h }[4], [x3], #0x2
    tbz x10, #0, label_160
    st1 { v8.b }[10], [x13]
    st1 { v12.b }[10], [x7]
    st1 { v16.b }[10], [x6]
    st1 { v20.b }[10], [x5]
    st1 { v24.b }[10], [x4]
    st1 { v28.b }[10], [x3]
    b label_160
KAI_ASM_LABEL(label_155)  // Height 6: Partial direct writeback: partial_0_8
    tbz x10, #0, label_160
    st1 { v8.b }[8], [x13]
    st1 { v12.b }[8], [x7]
    st1 { v16.b }[8], [x6]
    st1 { v20.b }[8], [x5]
    st1 { v24.b }[8], [x4]
    st1 { v28.b }[8], [x3]
    b label_160
KAI_ASM_LABEL(label_156)  // Height 6: Partial direct writeback: partial_2_0
    tbz x10, #2, label_158
    str s8, [x13], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    str s28, [x3], #0x4
    tbz x10, #1, label_157
    st1 { v8.h }[2], [x13], #0x2
    st1 { v12.h }[2], [x7], #0x2
    st1 { v16.h }[2], [x6], #0x2
    st1 { v20.h }[2], [x5], #0x2
    st1 { v24.h }[2], [x4], #0x2
    st1 { v28.h }[2], [x3], #0x2
    tbz x10, #0, label_160
    st1 { v8.b }[6], [x13]
    st1 { v12.b }[6], [x7]
    st1 { v16.b }[6], [x6]
    st1 { v20.b }[6], [x5]
    st1 { v24.b }[6], [x4]
    st1 { v28.b }[6], [x3]
    b label_160
KAI_ASM_LABEL(label_157)  // Height 6: Partial direct writeback: partial_0_4
    tbz x10, #0, label_160
    st1 { v8.b }[4], [x13]
    st1 { v12.b }[4], [x7]
    st1 { v16.b }[4], [x6]
    st1 { v20.b }[4], [x5]
    st1 { v24.b }[4], [x4]
    st1 { v28.b }[4], [x3]
    b label_160
KAI_ASM_LABEL(label_158)  // Height 6: Partial direct writeback: partial_1_0
    tbz x10, #1, label_159
    str h8, [x13], #0x2
    str h12, [x7], #0x2
    str h16, [x6], #0x2
    str h20, [x5], #0x2
    str h24, [x4], #0x2
    str h28, [x3], #0x2
    tbz x10, #0, label_160
    st1 { v8.b }[2], [x13]
    st1 { v12.b }[2], [x7]
    st1 { v16.b }[2], [x6]
    st1 { v20.b }[2], [x5]
    st1 { v24.b }[2], [x4]
    st1 { v28.b }[2], [x3]
    b label_160
KAI_ASM_LABEL(label_159)  // Height 6: Partial direct writeback: partial_0_0
    str b8, [x13, #0]
    str b12, [x7, #0]
    str b16, [x6, #0]
    str b20, [x5, #0]
    str b24, [x4, #0]
    str b28, [x3, #0]
KAI_ASM_LABEL(label_160)  // Height 6: Partial direct writeback: Done
    b label_162
KAI_ASM_LABEL(label_161)  // Height 6: Full writeback
    str q8, [x13, #0]
    add x13, x13, #0x10
    str q12, [x7, #0]
    str q16, [x6, #0]
    str q20, [x5, #0]
    str q24, [x4, #0]
    str q28, [x3, #0]
KAI_ASM_LABEL(label_162)  // Height 6: Writeback done
    subs x10, x10, #0x10
    bgt label_137
    add x17, x17, #0x6
    cmp x17, x16
    blt label_1
KAI_ASM_LABEL(label_163)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_s8qs_dot_6x16_a55)

    KAI_ASM_END
