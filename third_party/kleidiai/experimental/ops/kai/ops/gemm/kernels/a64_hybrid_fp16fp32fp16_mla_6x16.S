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

    KAI_ASM_CODE(a64_hybrid_fp16fp32fp16_mla_6x16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_hybrid_fp16fp32fp16_mla_6x16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_hybrid_fp16fp32fp16_mla_6x16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_hybrid_fp16fp32fp16_mla_6x16)
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
    sub SP, SP, #0x20
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
    bge label_206
    cmp x1, #0x4
    bgt label_165
    beq label_124
    cmp x1, #0x2
    bgt label_83
    beq label_42
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
    b label_9
KAI_ASM_LABEL(label_8)  // Height 1: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_9)  // Height 1: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    b label_21
KAI_ASM_LABEL(label_10)  // Height 1: no bias
    tbz x25, #0, label_20
    cmp x22, #0x10
    bge label_19
    tbz x22, #3, label_14
    ld1 { v8.4s }, [x23], #0x10
    tbz x22, #2, label_12
    ldr d9, [x23], #0x8
    tbz x22, #1, label_11
    ld1 { v9.s }[2], [x23], #0x4
    mov x1, #0x1c
    tbz x22, #0, label_18
    ld1 { v9.h }[6], [x23]
    b label_18
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_18
    ld1 { v9.h }[4], [x23]
    b label_18
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_8
    tbz x22, #1, label_13
    ldr s9, [x23], #0x4
    mov x1, #0x14
    tbz x22, #0, label_18
    ld1 { v9.h }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_18
    ldr h9, [x23, #0]
    b label_18
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_2_0
    tbz x22, #2, label_16
    ldr d8, [x23], #0x8
    tbz x22, #1, label_15
    ld1 { v8.s }[2], [x23], #0x4
    mov x1, #0xc
    tbz x22, #0, label_18
    ld1 { v8.h }[6], [x23]
    b label_18
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_18
    ld1 { v8.h }[4], [x23]
    b label_18
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_0
    tbz x22, #1, label_17
    ldr s8, [x23], #0x4
    mov x1, #0x4
    tbz x22, #0, label_18
    ld1 { v8.h }[2], [x23]
    b label_18
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    mov x1, #0
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    b label_21
KAI_ASM_LABEL(label_19)  // Height 1: full accumulate
    ldr q17, [x23, #0]
    ldr q16, [x23, #0x10]
    fcvtl v8.4s, v17.4h
    fcvtl2 v9.4s, v17.8h
    fcvtl v10.4s, v16.4h
    fcvtl2 v11.4s, v16.8h
    b label_21
KAI_ASM_LABEL(label_20)  // Height 1: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
KAI_ASM_LABEL(label_21)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_22
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    b label_24
KAI_ASM_LABEL(label_22)  // Height 1: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_23)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    cbnz x24, label_24
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_24)  // Height 1: input setup done
    cmp x7, #0x8
    blt label_27
    ldr q0, [x8, #0]
    ldr q6, [x21, #0]
    cmp x7, #0x10
    ldr q7, [x21, #0x10]
    blt label_26
KAI_ASM_LABEL(label_25)  // Height 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    ldr q6, [x21, #0x20]
    sub x7, x7, #0x8
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    ldr q7, [x21, #0x30]
    add x8, x8, #0x10
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    ldr q7, [x21, #0x10]
    bge label_25
KAI_ASM_LABEL(label_26)  // Height 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    ldr q6, [x21, #0x20]
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    ldr q7, [x21, #0x30]
    sub x7, x7, #0x8
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
KAI_ASM_LABEL(label_27)  // Height 1: Multiply loop: Main loop skip
    cbz x7, label_29
KAI_ASM_LABEL(label_28)  // Height 1: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr q6, [x21, #0]
    sub x7, x7, #0x1
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    cbnz x7, label_28
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: No odd multiplies
    tbz x25, #3, label_30
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_23
KAI_ASM_LABEL(label_30)  // Height 1: no string loop
    prfm pstl1keep, [x23, #0]
    tbz x25, #1, label_31
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.8h }, [x2]
    ld1r { v16.8h }, [x1]
    fcvtl v17.4s, v17.4h
    fcvtl v16.4s, v16.4h
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
KAI_ASM_LABEL(label_31)  // Height 1: No activation
    fcvtn v8.4h, v8.4s
    cmp x22, #0x10
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v9.8h, v11.4s
    bge label_40
    tbz x22, #3, label_35
    st1 { v8.4s }, [x23], #0x10
    tbz x22, #2, label_33
    str d9, [x23], #0x8
    tbz x22, #1, label_32
    st1 { v9.s }[2], [x23], #0x4
    tbz x22, #0, label_39
    st1 { v9.h }[6], [x23]
    b label_39
KAI_ASM_LABEL(label_32)  // Height 1: Partial direct writeback: partial_0_12
    tbz x22, #0, label_39
    st1 { v9.h }[4], [x23]
    b label_39
KAI_ASM_LABEL(label_33)  // Height 1: Partial direct writeback: partial_1_8
    tbz x22, #1, label_34
    str s9, [x23], #0x4
    tbz x22, #0, label_39
    st1 { v9.h }[2], [x23]
    b label_39
KAI_ASM_LABEL(label_34)  // Height 1: Partial direct writeback: partial_0_8
    tbz x22, #0, label_39
    str h9, [x23, #0]
    b label_39
KAI_ASM_LABEL(label_35)  // Height 1: Partial direct writeback: partial_2_0
    tbz x22, #2, label_37
    str d8, [x23], #0x8
    tbz x22, #1, label_36
    st1 { v8.s }[2], [x23], #0x4
    tbz x22, #0, label_39
    st1 { v8.h }[6], [x23]
    b label_39
KAI_ASM_LABEL(label_36)  // Height 1: Partial direct writeback: partial_0_4
    tbz x22, #0, label_39
    st1 { v8.h }[4], [x23]
    b label_39
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_1_0
    tbz x22, #1, label_38
    str s8, [x23], #0x4
    tbz x22, #0, label_39
    st1 { v8.h }[2], [x23]
    b label_39
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: Done
    b label_41
KAI_ASM_LABEL(label_40)  // Height 1: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
KAI_ASM_LABEL(label_41)  // Height 1: Writeback done
    subs x22, x22, #0x10
    bgt label_2
    b label_247
KAI_ASM_LABEL(label_42)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_43)  // Height 2: Column loop
    cbz x20, label_51
    cmp x22, #0x10
    bge label_49
    cbnz x27, label_48
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_45
KAI_ASM_LABEL(label_44)  // Height 2: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_44
KAI_ASM_LABEL(label_45)  // Height 2: Partial bias load: vector loop skip
    cbz x2, label_47
KAI_ASM_LABEL(label_46)  // Height 2: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_46
KAI_ASM_LABEL(label_47)  // Height 2: Partial bias load: element loop skip
KAI_ASM_LABEL(label_48)  // Height 2: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    b label_50
KAI_ASM_LABEL(label_49)  // Height 2: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_50)  // Height 2: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    b label_62
KAI_ASM_LABEL(label_51)  // Height 2: no bias
    tbz x25, #0, label_61
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x7, x23, x1
    bge label_60
    tbz x22, #3, label_55
    ld1 { v8.4s }, [x23], #0x10
    ld1 { v12.4s }, [x7], #0x10
    tbz x22, #2, label_53
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    tbz x22, #1, label_52
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    tbz x22, #0, label_59
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    b label_59
KAI_ASM_LABEL(label_52)  // Height 2: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_59
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    b label_59
KAI_ASM_LABEL(label_53)  // Height 2: Partial accumulate: partial_1_8
    tbz x22, #1, label_54
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    tbz x22, #0, label_59
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    b label_59
KAI_ASM_LABEL(label_54)  // Height 2: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_59
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    b label_59
KAI_ASM_LABEL(label_55)  // Height 2: Partial accumulate: partial_2_0
    tbz x22, #2, label_57
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    tbz x22, #1, label_56
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    tbz x22, #0, label_59
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    b label_59
KAI_ASM_LABEL(label_56)  // Height 2: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_59
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    b label_59
KAI_ASM_LABEL(label_57)  // Height 2: Partial accumulate: partial_1_0
    tbz x22, #1, label_58
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    tbz x22, #0, label_59
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    b label_59
KAI_ASM_LABEL(label_58)  // Height 2: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
KAI_ASM_LABEL(label_59)  // Height 2: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    fcvtl2 v15.4s, v13.8h
    fcvtl v14.4s, v13.4h
    fcvtl2 v13.4s, v12.8h
    fcvtl v12.4s, v12.4h
    b label_62
KAI_ASM_LABEL(label_60)  // Height 2: full accumulate
    ldr q19, [x23, #0]
    ldr q18, [x23, #0x10]
    ldr q17, [x7, #0]
    ldr q16, [x7, #0x10]
    fcvtl v8.4s, v19.4h
    fcvtl2 v9.4s, v19.8h
    fcvtl v10.4s, v18.4h
    fcvtl2 v11.4s, v18.8h
    fcvtl v12.4s, v17.4h
    fcvtl2 v13.4s, v17.8h
    fcvtl v14.4s, v16.4h
    fcvtl2 v15.4s, v16.8h
    b label_62
KAI_ASM_LABEL(label_61)  // Height 2: no accumulate
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
KAI_ASM_LABEL(label_62)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_63
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    b label_65
KAI_ASM_LABEL(label_63)  // Height 2: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_64)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x24, label_65
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_65)  // Height 2: input setup done
    cmp x7, #0x8
    blt label_68
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_67
KAI_ASM_LABEL(label_66)  // Height 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    ldr q6, [x21, #0x20]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    ldr q7, [x21, #0x30]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    ldr q1, [x6, #0]
    ldr q7, [x21, #0x10]
    bge label_66
KAI_ASM_LABEL(label_67)  // Height 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    ldr q6, [x21, #0x20]
    sub x7, x7, #0x8
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
KAI_ASM_LABEL(label_68)  // Height 2: Multiply loop: Main loop skip
    cbz x7, label_70
KAI_ASM_LABEL(label_69)  // Height 2: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    cbnz x7, label_69
KAI_ASM_LABEL(label_70)  // Height 2: Multiply loop: No odd multiplies
    tbz x25, #3, label_71
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_64
KAI_ASM_LABEL(label_71)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    tbz x25, #1, label_72
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v17.8h }, [x2]
    ld1r { v16.8h }, [x1]
    fcvtl v17.4s, v17.4h
    fcvtl v16.4s, v16.4h
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
KAI_ASM_LABEL(label_72)  // Height 2: No activation
    fcvtn v8.4h, v8.4s
    fcvtn v12.4h, v12.4s
    cmp x22, #0x10
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v12.8h, v13.4s
    fcvtn v13.4h, v14.4s
    fcvtn2 v9.8h, v11.4s
    fcvtn2 v13.8h, v15.4s
    bge label_81
    tbz x22, #3, label_76
    st1 { v8.4s }, [x23], #0x10
    st1 { v12.4s }, [x7], #0x10
    tbz x22, #2, label_74
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    tbz x22, #1, label_73
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    tbz x22, #0, label_80
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    b label_80
KAI_ASM_LABEL(label_73)  // Height 2: Partial direct writeback: partial_0_12
    tbz x22, #0, label_80
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    b label_80
KAI_ASM_LABEL(label_74)  // Height 2: Partial direct writeback: partial_1_8
    tbz x22, #1, label_75
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    tbz x22, #0, label_80
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    b label_80
KAI_ASM_LABEL(label_75)  // Height 2: Partial direct writeback: partial_0_8
    tbz x22, #0, label_80
    str h9, [x23, #0]
    str h13, [x7, #0]
    b label_80
KAI_ASM_LABEL(label_76)  // Height 2: Partial direct writeback: partial_2_0
    tbz x22, #2, label_78
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    tbz x22, #1, label_77
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    tbz x22, #0, label_80
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    b label_80
KAI_ASM_LABEL(label_77)  // Height 2: Partial direct writeback: partial_0_4
    tbz x22, #0, label_80
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    b label_80
KAI_ASM_LABEL(label_78)  // Height 2: Partial direct writeback: partial_1_0
    tbz x22, #1, label_79
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    tbz x22, #0, label_80
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    b label_80
KAI_ASM_LABEL(label_79)  // Height 2: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
KAI_ASM_LABEL(label_80)  // Height 2: Partial direct writeback: Done
    b label_82
KAI_ASM_LABEL(label_81)  // Height 2: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
    str q12, [x7, #0]
    str q13, [x7, #0x10]
KAI_ASM_LABEL(label_82)  // Height 2: Writeback done
    subs x22, x22, #0x10
    bgt label_43
    b label_247
KAI_ASM_LABEL(label_83)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_84)  // Height 3: Column loop
    cbz x20, label_92
    cmp x22, #0x10
    bge label_90
    cbnz x27, label_89
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_86
KAI_ASM_LABEL(label_85)  // Height 3: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_85
KAI_ASM_LABEL(label_86)  // Height 3: Partial bias load: vector loop skip
    cbz x2, label_88
KAI_ASM_LABEL(label_87)  // Height 3: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_87
KAI_ASM_LABEL(label_88)  // Height 3: Partial bias load: element loop skip
KAI_ASM_LABEL(label_89)  // Height 3: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    b label_91
KAI_ASM_LABEL(label_90)  // Height 3: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_91)  // Height 3: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    b label_103
KAI_ASM_LABEL(label_92)  // Height 3: no bias
    tbz x25, #0, label_102
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x7, x23, x1
    add x6, x7, x1
    bge label_101
    tbz x22, #3, label_96
    ld1 { v8.4s }, [x23], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    tbz x22, #2, label_94
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    tbz x22, #1, label_93
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    tbz x22, #0, label_100
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    b label_100
KAI_ASM_LABEL(label_93)  // Height 3: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_100
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    b label_100
KAI_ASM_LABEL(label_94)  // Height 3: Partial accumulate: partial_1_8
    tbz x22, #1, label_95
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    tbz x22, #0, label_100
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    b label_100
KAI_ASM_LABEL(label_95)  // Height 3: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_100
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    b label_100
KAI_ASM_LABEL(label_96)  // Height 3: Partial accumulate: partial_2_0
    tbz x22, #2, label_98
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    tbz x22, #1, label_97
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    tbz x22, #0, label_100
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    b label_100
KAI_ASM_LABEL(label_97)  // Height 3: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_100
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    b label_100
KAI_ASM_LABEL(label_98)  // Height 3: Partial accumulate: partial_1_0
    tbz x22, #1, label_99
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    tbz x22, #0, label_100
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    b label_100
KAI_ASM_LABEL(label_99)  // Height 3: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
KAI_ASM_LABEL(label_100)  // Height 3: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    fcvtl2 v15.4s, v13.8h
    fcvtl v14.4s, v13.4h
    fcvtl2 v13.4s, v12.8h
    fcvtl v12.4s, v12.4h
    fcvtl2 v19.4s, v17.8h
    fcvtl v18.4s, v17.4h
    fcvtl2 v17.4s, v16.8h
    fcvtl v16.4s, v16.4h
    b label_103
KAI_ASM_LABEL(label_101)  // Height 3: full accumulate
    ldr q21, [x23, #0]
    ldr q20, [x23, #0x10]
    ldr q18, [x7, #0]
    ldr q16, [x7, #0x10]
    ldr q17, [x6, #0]
    ldr q19, [x6, #0x10]
    fcvtl v8.4s, v21.4h
    fcvtl2 v9.4s, v21.8h
    fcvtl v10.4s, v20.4h
    fcvtl2 v11.4s, v20.8h
    fcvtl v12.4s, v18.4h
    fcvtl2 v13.4s, v18.8h
    fcvtl v14.4s, v16.4h
    fcvtl2 v15.4s, v16.8h
    fcvtl v16.4s, v17.4h
    fcvtl2 v17.4s, v17.8h
    fcvtl v18.4s, v19.4h
    fcvtl2 v19.4s, v19.8h
    b label_103
KAI_ASM_LABEL(label_102)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_103)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_104
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_106
KAI_ASM_LABEL(label_104)  // Height 3: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_105)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x24, label_106
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_106)  // Height 3: input setup done
    cmp x7, #0x8
    blt label_109
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_108
KAI_ASM_LABEL(label_107)  // Height 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    ldr q6, [x21, #0x20]
    cmp x7, #0x10
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    ldr q1, [x6, #0]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    ldr q2, [x5, #0]
    ldr q7, [x21, #0x10]
    bge label_107
KAI_ASM_LABEL(label_108)  // Height 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    add x5, x5, #0x10
    sub x7, x7, #0x8
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    ldr q6, [x21, #0x20]
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
KAI_ASM_LABEL(label_109)  // Height 3: Multiply loop: Main loop skip
    cbz x7, label_111
KAI_ASM_LABEL(label_110)  // Height 3: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    cbnz x7, label_110
KAI_ASM_LABEL(label_111)  // Height 3: Multiply loop: No odd multiplies
    tbz x25, #3, label_112
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_105
KAI_ASM_LABEL(label_112)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    tbz x25, #1, label_113
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v21.8h }, [x2]
    ld1r { v20.8h }, [x1]
    fcvtl v21.4s, v21.4h
    fcvtl v20.4s, v20.4h
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
KAI_ASM_LABEL(label_113)  // Height 3: No activation
    fcvtn v8.4h, v8.4s
    fcvtn v12.4h, v12.4s
    cmp x22, #0x10
    fcvtn v16.4h, v16.4s
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v12.8h, v13.4s
    fcvtn v13.4h, v14.4s
    fcvtn2 v16.8h, v17.4s
    fcvtn v17.4h, v18.4s
    fcvtn2 v9.8h, v11.4s
    fcvtn2 v13.8h, v15.4s
    fcvtn2 v17.8h, v19.4s
    bge label_122
    tbz x22, #3, label_117
    st1 { v8.4s }, [x23], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    tbz x22, #2, label_115
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    tbz x22, #1, label_114
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    tbz x22, #0, label_121
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    b label_121
KAI_ASM_LABEL(label_114)  // Height 3: Partial direct writeback: partial_0_12
    tbz x22, #0, label_121
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    b label_121
KAI_ASM_LABEL(label_115)  // Height 3: Partial direct writeback: partial_1_8
    tbz x22, #1, label_116
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    tbz x22, #0, label_121
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    b label_121
KAI_ASM_LABEL(label_116)  // Height 3: Partial direct writeback: partial_0_8
    tbz x22, #0, label_121
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    b label_121
KAI_ASM_LABEL(label_117)  // Height 3: Partial direct writeback: partial_2_0
    tbz x22, #2, label_119
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    tbz x22, #1, label_118
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    tbz x22, #0, label_121
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    b label_121
KAI_ASM_LABEL(label_118)  // Height 3: Partial direct writeback: partial_0_4
    tbz x22, #0, label_121
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    b label_121
KAI_ASM_LABEL(label_119)  // Height 3: Partial direct writeback: partial_1_0
    tbz x22, #1, label_120
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    tbz x22, #0, label_121
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    b label_121
KAI_ASM_LABEL(label_120)  // Height 3: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
KAI_ASM_LABEL(label_121)  // Height 3: Partial direct writeback: Done
    b label_123
KAI_ASM_LABEL(label_122)  // Height 3: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
KAI_ASM_LABEL(label_123)  // Height 3: Writeback done
    subs x22, x22, #0x10
    bgt label_84
    b label_247
KAI_ASM_LABEL(label_124)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_125)  // Height 4: Column loop
    cbz x20, label_133
    cmp x22, #0x10
    bge label_131
    cbnz x27, label_130
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_127
KAI_ASM_LABEL(label_126)  // Height 4: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_126
KAI_ASM_LABEL(label_127)  // Height 4: Partial bias load: vector loop skip
    cbz x2, label_129
KAI_ASM_LABEL(label_128)  // Height 4: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_128
KAI_ASM_LABEL(label_129)  // Height 4: Partial bias load: element loop skip
KAI_ASM_LABEL(label_130)  // Height 4: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    b label_132
KAI_ASM_LABEL(label_131)  // Height 4: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_132)  // Height 4: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    b label_144
KAI_ASM_LABEL(label_133)  // Height 4: no bias
    tbz x25, #0, label_143
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    bge label_142
    tbz x22, #3, label_137
    ld1 { v8.4s }, [x23], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    tbz x22, #2, label_135
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    tbz x22, #1, label_134
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    tbz x22, #0, label_141
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    b label_141
KAI_ASM_LABEL(label_134)  // Height 4: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_141
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    b label_141
KAI_ASM_LABEL(label_135)  // Height 4: Partial accumulate: partial_1_8
    tbz x22, #1, label_136
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    tbz x22, #0, label_141
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    b label_141
KAI_ASM_LABEL(label_136)  // Height 4: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_141
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    b label_141
KAI_ASM_LABEL(label_137)  // Height 4: Partial accumulate: partial_2_0
    tbz x22, #2, label_139
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    tbz x22, #1, label_138
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    tbz x22, #0, label_141
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    b label_141
KAI_ASM_LABEL(label_138)  // Height 4: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_141
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    b label_141
KAI_ASM_LABEL(label_139)  // Height 4: Partial accumulate: partial_1_0
    tbz x22, #1, label_140
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    tbz x22, #0, label_141
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    b label_141
KAI_ASM_LABEL(label_140)  // Height 4: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
KAI_ASM_LABEL(label_141)  // Height 4: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    fcvtl2 v15.4s, v13.8h
    fcvtl v14.4s, v13.4h
    fcvtl2 v13.4s, v12.8h
    fcvtl v12.4s, v12.4h
    fcvtl2 v19.4s, v17.8h
    fcvtl v18.4s, v17.4h
    fcvtl2 v17.4s, v16.8h
    fcvtl v16.4s, v16.4h
    fcvtl2 v23.4s, v21.8h
    fcvtl v22.4s, v21.4h
    fcvtl2 v21.4s, v20.8h
    fcvtl v20.4s, v20.4h
    b label_144
KAI_ASM_LABEL(label_142)  // Height 4: full accumulate
    ldr q22, [x23, #0]
    ldr q20, [x23, #0x10]
    ldr q18, [x7, #0]
    ldr q16, [x7, #0x10]
    ldr q17, [x6, #0]
    ldr q19, [x6, #0x10]
    ldr q21, [x5, #0]
    ldr q23, [x5, #0x10]
    fcvtl v8.4s, v22.4h
    fcvtl2 v9.4s, v22.8h
    fcvtl v10.4s, v20.4h
    fcvtl2 v11.4s, v20.8h
    fcvtl v12.4s, v18.4h
    fcvtl2 v13.4s, v18.8h
    fcvtl v14.4s, v16.4h
    fcvtl2 v15.4s, v16.8h
    fcvtl v16.4s, v17.4h
    fcvtl2 v17.4s, v17.8h
    fcvtl v18.4s, v19.4h
    fcvtl2 v19.4s, v19.8h
    fcvtl v20.4s, v21.4h
    fcvtl2 v21.4s, v21.8h
    fcvtl v22.4s, v23.4h
    fcvtl2 v23.4s, v23.8h
    b label_144
KAI_ASM_LABEL(label_143)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_144)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_145
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_147
KAI_ASM_LABEL(label_145)  // Height 4: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_146)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x24, LSL #0x3]
    ldr x7, [x2, x24, LSL #0x3]
    add x1, x1, x27, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x24, label_147
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_147)  // Height 4: input setup done
    cmp x7, #0x8
    blt label_150
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_149
KAI_ASM_LABEL(label_148)  // Height 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    add x4, x4, #0x10
    cmp x7, #0x10
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    ldr q6, [x21, #0x20]
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x6, #0x80]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    ldr q1, [x6, #0]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    ldr q2, [x5, #0]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
    ldr q3, [x4, #0]
    ldr q7, [x21, #0x10]
    bge label_148
KAI_ASM_LABEL(label_149)  // Height 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    ldr q6, [x21, #0x20]
    sub x7, x7, #0x8
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
KAI_ASM_LABEL(label_150)  // Height 4: Multiply loop: Main loop skip
    cbz x7, label_152
KAI_ASM_LABEL(label_151)  // Height 4: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    cbnz x7, label_151
KAI_ASM_LABEL(label_152)  // Height 4: Multiply loop: No odd multiplies
    tbz x25, #3, label_153
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_146
KAI_ASM_LABEL(label_153)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    prfm pstl1keep, [x23, #0]
    add x7, x23, x1
    prfm pstl1keep, [x7, #0]
    add x6, x7, x1
    prfm pstl1keep, [x6, #0]
    add x5, x6, x1
    prfm pstl1keep, [x5, #0]
    tbz x25, #1, label_154
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v25.8h }, [x2]
    ld1r { v24.8h }, [x1]
    fcvtl v25.4s, v25.4h
    fcvtl v24.4s, v24.4h
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
KAI_ASM_LABEL(label_154)  // Height 4: No activation
    fcvtn v8.4h, v8.4s
    fcvtn v12.4h, v12.4s
    cmp x22, #0x10
    fcvtn v16.4h, v16.4s
    fcvtn v20.4h, v20.4s
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v12.8h, v13.4s
    fcvtn v13.4h, v14.4s
    fcvtn2 v16.8h, v17.4s
    fcvtn v17.4h, v18.4s
    fcvtn2 v20.8h, v21.4s
    fcvtn v21.4h, v22.4s
    fcvtn2 v9.8h, v11.4s
    fcvtn2 v13.8h, v15.4s
    fcvtn2 v17.8h, v19.4s
    fcvtn2 v21.8h, v23.4s
    bge label_163
    tbz x22, #3, label_158
    st1 { v8.4s }, [x23], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    tbz x22, #2, label_156
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    tbz x22, #1, label_155
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    tbz x22, #0, label_162
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    b label_162
KAI_ASM_LABEL(label_155)  // Height 4: Partial direct writeback: partial_0_12
    tbz x22, #0, label_162
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    b label_162
KAI_ASM_LABEL(label_156)  // Height 4: Partial direct writeback: partial_1_8
    tbz x22, #1, label_157
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    tbz x22, #0, label_162
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    b label_162
KAI_ASM_LABEL(label_157)  // Height 4: Partial direct writeback: partial_0_8
    tbz x22, #0, label_162
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    b label_162
KAI_ASM_LABEL(label_158)  // Height 4: Partial direct writeback: partial_2_0
    tbz x22, #2, label_160
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    tbz x22, #1, label_159
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    tbz x22, #0, label_162
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    b label_162
KAI_ASM_LABEL(label_159)  // Height 4: Partial direct writeback: partial_0_4
    tbz x22, #0, label_162
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    b label_162
KAI_ASM_LABEL(label_160)  // Height 4: Partial direct writeback: partial_1_0
    tbz x22, #1, label_161
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    tbz x22, #0, label_162
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    b label_162
KAI_ASM_LABEL(label_161)  // Height 4: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
KAI_ASM_LABEL(label_162)  // Height 4: Partial direct writeback: Done
    b label_164
KAI_ASM_LABEL(label_163)  // Height 4: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
KAI_ASM_LABEL(label_164)  // Height 4: Writeback done
    subs x22, x22, #0x10
    bgt label_125
    b label_247
KAI_ASM_LABEL(label_165)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_166)  // Height 5: Column loop
    cbz x20, label_174
    cmp x22, #0x10
    bge label_172
    cbnz x27, label_171
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_168
KAI_ASM_LABEL(label_167)  // Height 5: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_167
KAI_ASM_LABEL(label_168)  // Height 5: Partial bias load: vector loop skip
    cbz x2, label_170
KAI_ASM_LABEL(label_169)  // Height 5: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_169
KAI_ASM_LABEL(label_170)  // Height 5: Partial bias load: element loop skip
KAI_ASM_LABEL(label_171)  // Height 5: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    b label_173
KAI_ASM_LABEL(label_172)  // Height 5: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_173)  // Height 5: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    b label_185
KAI_ASM_LABEL(label_174)  // Height 5: no bias
    tbz x25, #0, label_184
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    bge label_183
    tbz x22, #3, label_178
    ld1 { v8.4s }, [x23], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    tbz x22, #2, label_176
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    tbz x22, #1, label_175
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    ld1 { v25.s }[2], [x4], #0x4
    tbz x22, #0, label_182
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    ld1 { v25.h }[6], [x4]
    b label_182
KAI_ASM_LABEL(label_175)  // Height 5: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_182
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    ld1 { v25.h }[4], [x4]
    b label_182
KAI_ASM_LABEL(label_176)  // Height 5: Partial accumulate: partial_1_8
    tbz x22, #1, label_177
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    ldr s25, [x4], #0x4
    tbz x22, #0, label_182
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    ld1 { v25.h }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_177)  // Height 5: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_182
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    ldr h25, [x4, #0]
    b label_182
KAI_ASM_LABEL(label_178)  // Height 5: Partial accumulate: partial_2_0
    tbz x22, #2, label_180
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    tbz x22, #1, label_179
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    ld1 { v24.s }[2], [x4], #0x4
    tbz x22, #0, label_182
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    ld1 { v24.h }[6], [x4]
    b label_182
KAI_ASM_LABEL(label_179)  // Height 5: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_182
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    ld1 { v24.h }[4], [x4]
    b label_182
KAI_ASM_LABEL(label_180)  // Height 5: Partial accumulate: partial_1_0
    tbz x22, #1, label_181
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    ldr s24, [x4], #0x4
    tbz x22, #0, label_182
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    ld1 { v24.h }[2], [x4]
    b label_182
KAI_ASM_LABEL(label_181)  // Height 5: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
    ldr h24, [x4, #0]
KAI_ASM_LABEL(label_182)  // Height 5: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    fcvtl2 v15.4s, v13.8h
    fcvtl v14.4s, v13.4h
    fcvtl2 v13.4s, v12.8h
    fcvtl v12.4s, v12.4h
    fcvtl2 v19.4s, v17.8h
    fcvtl v18.4s, v17.4h
    fcvtl2 v17.4s, v16.8h
    fcvtl v16.4s, v16.4h
    fcvtl2 v23.4s, v21.8h
    fcvtl v22.4s, v21.4h
    fcvtl2 v21.4s, v20.8h
    fcvtl v20.4s, v20.4h
    fcvtl2 v27.4s, v25.8h
    fcvtl v26.4s, v25.4h
    fcvtl2 v25.4s, v24.8h
    fcvtl v24.4s, v24.4h
    b label_185
KAI_ASM_LABEL(label_183)  // Height 5: full accumulate
    ldr q22, [x23, #0]
    ldr q20, [x23, #0x10]
    ldr q18, [x7, #0]
    ldr q16, [x7, #0x10]
    ldr q17, [x6, #0]
    ldr q19, [x6, #0x10]
    ldr q21, [x5, #0]
    ldr q23, [x5, #0x10]
    fcvtl v8.4s, v22.4h
    fcvtl2 v9.4s, v22.8h
    ldr q25, [x4, #0]
    ldr q27, [x4, #0x10]
    fcvtl v10.4s, v20.4h
    fcvtl2 v11.4s, v20.8h
    fcvtl v12.4s, v18.4h
    fcvtl2 v13.4s, v18.8h
    fcvtl v14.4s, v16.4h
    fcvtl2 v15.4s, v16.8h
    fcvtl v16.4s, v17.4h
    fcvtl2 v17.4s, v17.8h
    fcvtl v18.4s, v19.4h
    fcvtl2 v19.4s, v19.8h
    fcvtl v20.4s, v21.4h
    fcvtl2 v21.4s, v21.8h
    fcvtl v22.4s, v23.4h
    fcvtl2 v23.4s, v23.8h
    fcvtl v24.4s, v25.4h
    fcvtl2 v25.4s, v25.8h
    fcvtl v26.4s, v27.4h
    fcvtl2 v27.4s, v27.8h
    b label_185
KAI_ASM_LABEL(label_184)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_185)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_186
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_188
KAI_ASM_LABEL(label_186)  // Height 5: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_187)  // Height 5: String loop
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
    cbnz x24, label_188
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_188)  // Height 5: input setup done
    cmp x7, #0x8
    blt label_191
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_190
KAI_ASM_LABEL(label_189)  // Height 5: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    cmp x7, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    ldr q6, [x21, #0x20]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400d8)  // fmlal v24.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480d9)  // fmlal2 v25.4s, v6.4h, v4.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400fa)  // fmlal v26.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480fb)  // fmlal2 v27.4s, v7.4h, v4.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400d8)  // fmlal v24.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480d9)  // fmlal2 v25.4s, v6.4h, v4.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400fa)  // fmlal v26.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480fb)  // fmlal2 v27.4s, v7.4h, v4.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400d8)  // fmlal v24.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480d9)  // fmlal2 v25.4s, v6.4h, v4.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400fa)  // fmlal v26.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480fb)  // fmlal2 v27.4s, v7.4h, v4.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408d8)  // fmlal v24.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488d9)  // fmlal2 v25.4s, v6.4h, v4.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408fa)  // fmlal v26.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488fb)  // fmlal2 v27.4s, v7.4h, v4.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408d8)  // fmlal v24.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488d9)  // fmlal2 v25.4s, v6.4h, v4.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408fa)  // fmlal v26.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488fb)  // fmlal2 v27.4s, v7.4h, v4.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408d8)  // fmlal v24.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488d9)  // fmlal2 v25.4s, v6.4h, v4.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408fa)  // fmlal v26.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488fb)  // fmlal2 v27.4s, v7.4h, v4.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408d8)  // fmlal v24.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488d9)  // fmlal2 v25.4s, v6.4h, v4.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408fa)  // fmlal v26.4s, v7.4h, v4.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    ldr q1, [x6, #0]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    ldr q2, [x5, #0]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
    ldr q3, [x4, #0]
    KAI_ASM_INST(0x6fb488fb)  // fmlal2 v27.4s, v7.4h, v4.h[7]
    ldr q4, [x2, #0]
    ldr q7, [x21, #0x10]
    bge label_189
KAI_ASM_LABEL(label_190)  // Height 5: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    add x2, x2, #0x10
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    prfm pldl1keep, [x6, #0x80]
    sub x7, x7, #0x8
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    ldr q6, [x21, #0x20]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400d8)  // fmlal v24.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480d9)  // fmlal2 v25.4s, v6.4h, v4.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400fa)  // fmlal v26.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480fb)  // fmlal2 v27.4s, v7.4h, v4.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400d8)  // fmlal v24.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480d9)  // fmlal2 v25.4s, v6.4h, v4.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400fa)  // fmlal v26.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480fb)  // fmlal2 v27.4s, v7.4h, v4.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400d8)  // fmlal v24.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480d9)  // fmlal2 v25.4s, v6.4h, v4.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400fa)  // fmlal v26.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480fb)  // fmlal2 v27.4s, v7.4h, v4.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408d8)  // fmlal v24.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488d9)  // fmlal2 v25.4s, v6.4h, v4.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408fa)  // fmlal v26.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488fb)  // fmlal2 v27.4s, v7.4h, v4.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408d8)  // fmlal v24.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488d9)  // fmlal2 v25.4s, v6.4h, v4.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408fa)  // fmlal v26.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488fb)  // fmlal2 v27.4s, v7.4h, v4.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408d8)  // fmlal v24.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488d9)  // fmlal2 v25.4s, v6.4h, v4.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408fa)  // fmlal v26.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488fb)  // fmlal2 v27.4s, v7.4h, v4.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408d8)  // fmlal v24.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488d9)  // fmlal2 v25.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408fa)  // fmlal v26.4s, v7.4h, v4.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488fb)  // fmlal2 v27.4s, v7.4h, v4.h[7]
KAI_ASM_LABEL(label_191)  // Height 5: Multiply loop: Main loop skip
    cbz x7, label_193
KAI_ASM_LABEL(label_192)  // Height 5: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr h4, [x2], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    cbnz x7, label_192
KAI_ASM_LABEL(label_193)  // Height 5: Multiply loop: No odd multiplies
    tbz x25, #3, label_194
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_187
KAI_ASM_LABEL(label_194)  // Height 5: no string loop
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
    tbz x25, #1, label_195
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v29.8h }, [x2]
    ld1r { v28.8h }, [x1]
    fcvtl v29.4s, v29.4h
    fcvtl v28.4s, v28.4h
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
KAI_ASM_LABEL(label_195)  // Height 5: No activation
    fcvtn v8.4h, v8.4s
    fcvtn v12.4h, v12.4s
    cmp x22, #0x10
    fcvtn v16.4h, v16.4s
    fcvtn v20.4h, v20.4s
    fcvtn v24.4h, v24.4s
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v12.8h, v13.4s
    fcvtn v13.4h, v14.4s
    fcvtn2 v16.8h, v17.4s
    fcvtn v17.4h, v18.4s
    fcvtn2 v20.8h, v21.4s
    fcvtn v21.4h, v22.4s
    fcvtn2 v24.8h, v25.4s
    fcvtn v25.4h, v26.4s
    fcvtn2 v9.8h, v11.4s
    fcvtn2 v13.8h, v15.4s
    fcvtn2 v17.8h, v19.4s
    fcvtn2 v21.8h, v23.4s
    fcvtn2 v25.8h, v27.4s
    bge label_204
    tbz x22, #3, label_199
    st1 { v8.4s }, [x23], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    tbz x22, #2, label_197
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    tbz x22, #1, label_196
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    st1 { v25.s }[2], [x4], #0x4
    tbz x22, #0, label_203
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    st1 { v25.h }[6], [x4]
    b label_203
KAI_ASM_LABEL(label_196)  // Height 5: Partial direct writeback: partial_0_12
    tbz x22, #0, label_203
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    st1 { v25.h }[4], [x4]
    b label_203
KAI_ASM_LABEL(label_197)  // Height 5: Partial direct writeback: partial_1_8
    tbz x22, #1, label_198
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    str s25, [x4], #0x4
    tbz x22, #0, label_203
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    st1 { v25.h }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_198)  // Height 5: Partial direct writeback: partial_0_8
    tbz x22, #0, label_203
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    str h25, [x4, #0]
    b label_203
KAI_ASM_LABEL(label_199)  // Height 5: Partial direct writeback: partial_2_0
    tbz x22, #2, label_201
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    tbz x22, #1, label_200
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    tbz x22, #0, label_203
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    st1 { v24.h }[6], [x4]
    b label_203
KAI_ASM_LABEL(label_200)  // Height 5: Partial direct writeback: partial_0_4
    tbz x22, #0, label_203
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    st1 { v24.h }[4], [x4]
    b label_203
KAI_ASM_LABEL(label_201)  // Height 5: Partial direct writeback: partial_1_0
    tbz x22, #1, label_202
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    tbz x22, #0, label_203
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    st1 { v24.h }[2], [x4]
    b label_203
KAI_ASM_LABEL(label_202)  // Height 5: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
    str h24, [x4, #0]
KAI_ASM_LABEL(label_203)  // Height 5: Partial direct writeback: Done
    b label_205
KAI_ASM_LABEL(label_204)  // Height 5: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
    str q24, [x4, #0]
    str q25, [x4, #0x10]
KAI_ASM_LABEL(label_205)  // Height 5: Writeback done
    subs x22, x22, #0x10
    bgt label_166
    b label_247
KAI_ASM_LABEL(label_206)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x23, [x0, #0x50]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0x40]
    madd x23, x27, x1, x23
KAI_ASM_LABEL(label_207)  // Height 6: Column loop
    cbz x20, label_215
    cmp x22, #0x10
    bge label_213
    cbnz x27, label_212
    mov x2, x22
    mov x1, SP
    cmp x2, #0x8
    blt label_209
KAI_ASM_LABEL(label_208)  // Height 6: Partial bias load: vector loop
    ld1 { v16.8h }, [x20], #0x10
    sub x2, x2, #0x8
    cmp x2, #0x8
    str q16, [x1, #0]
    add x1, x1, #0x10
    bge label_208
KAI_ASM_LABEL(label_209)  // Height 6: Partial bias load: vector loop skip
    cbz x2, label_211
KAI_ASM_LABEL(label_210)  // Height 6: Partial bias load: element loop
    ldr h16, [x20], #0x2
    subs x2, x2, #0x1
    str h16, [x1, #0]
    add x1, x1, #0x2
    bgt label_210
KAI_ASM_LABEL(label_211)  // Height 6: Partial bias load: element loop skip
KAI_ASM_LABEL(label_212)  // Height 6: Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    b label_214
KAI_ASM_LABEL(label_213)  // Height 6: full width bias
    ldr q8, [x20, #0]
    ldr q9, [x20, #0x10]
    add x20, x20, #0x20
KAI_ASM_LABEL(label_214)  // Height 6: bias load done
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
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
    b label_226
KAI_ASM_LABEL(label_215)  // Height 6: no bias
    tbz x25, #0, label_225
    ldr x1, [x0, #0x58]
    cmp x22, #0x10
    add x7, x23, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    bge label_224
    tbz x22, #3, label_219
    ld1 { v8.4s }, [x23], #0x10
    ld1 { v12.4s }, [x7], #0x10
    ld1 { v16.4s }, [x6], #0x10
    ld1 { v20.4s }, [x5], #0x10
    ld1 { v24.4s }, [x4], #0x10
    ld1 { v28.4s }, [x3], #0x10
    tbz x22, #2, label_217
    ldr d9, [x23], #0x8
    ldr d13, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d21, [x5], #0x8
    ldr d25, [x4], #0x8
    ldr d29, [x3], #0x8
    tbz x22, #1, label_216
    ld1 { v9.s }[2], [x23], #0x4
    ld1 { v13.s }[2], [x7], #0x4
    mov x1, #0x1c
    ld1 { v17.s }[2], [x6], #0x4
    ld1 { v21.s }[2], [x5], #0x4
    ld1 { v25.s }[2], [x4], #0x4
    ld1 { v29.s }[2], [x3], #0x4
    tbz x22, #0, label_223
    ld1 { v9.h }[6], [x23]
    ld1 { v13.h }[6], [x7]
    ld1 { v17.h }[6], [x6]
    ld1 { v21.h }[6], [x5]
    ld1 { v25.h }[6], [x4]
    ld1 { v29.h }[6], [x3]
    b label_223
KAI_ASM_LABEL(label_216)  // Height 6: Partial accumulate: partial_0_12
    mov x1, #0x18
    tbz x22, #0, label_223
    ld1 { v9.h }[4], [x23]
    ld1 { v13.h }[4], [x7]
    ld1 { v17.h }[4], [x6]
    ld1 { v21.h }[4], [x5]
    ld1 { v25.h }[4], [x4]
    ld1 { v29.h }[4], [x3]
    b label_223
KAI_ASM_LABEL(label_217)  // Height 6: Partial accumulate: partial_1_8
    tbz x22, #1, label_218
    ldr s9, [x23], #0x4
    ldr s13, [x7], #0x4
    mov x1, #0x14
    ldr s17, [x6], #0x4
    ldr s21, [x5], #0x4
    ldr s25, [x4], #0x4
    ldr s29, [x3], #0x4
    tbz x22, #0, label_223
    ld1 { v9.h }[2], [x23]
    ld1 { v13.h }[2], [x7]
    ld1 { v17.h }[2], [x6]
    ld1 { v21.h }[2], [x5]
    ld1 { v25.h }[2], [x4]
    ld1 { v29.h }[2], [x3]
    b label_223
KAI_ASM_LABEL(label_218)  // Height 6: Partial accumulate: partial_0_8
    mov x1, #0x10
    tbz x22, #0, label_223
    ldr h9, [x23, #0]
    ldr h13, [x7, #0]
    ldr h17, [x6, #0]
    ldr h21, [x5, #0]
    ldr h25, [x4, #0]
    ldr h29, [x3, #0]
    b label_223
KAI_ASM_LABEL(label_219)  // Height 6: Partial accumulate: partial_2_0
    tbz x22, #2, label_221
    ldr d8, [x23], #0x8
    ldr d12, [x7], #0x8
    ldr d16, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d24, [x4], #0x8
    ldr d28, [x3], #0x8
    tbz x22, #1, label_220
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v12.s }[2], [x7], #0x4
    mov x1, #0xc
    ld1 { v16.s }[2], [x6], #0x4
    ld1 { v20.s }[2], [x5], #0x4
    ld1 { v24.s }[2], [x4], #0x4
    ld1 { v28.s }[2], [x3], #0x4
    tbz x22, #0, label_223
    ld1 { v8.h }[6], [x23]
    ld1 { v12.h }[6], [x7]
    ld1 { v16.h }[6], [x6]
    ld1 { v20.h }[6], [x5]
    ld1 { v24.h }[6], [x4]
    ld1 { v28.h }[6], [x3]
    b label_223
KAI_ASM_LABEL(label_220)  // Height 6: Partial accumulate: partial_0_4
    mov x1, #0x8
    tbz x22, #0, label_223
    ld1 { v8.h }[4], [x23]
    ld1 { v12.h }[4], [x7]
    ld1 { v16.h }[4], [x6]
    ld1 { v20.h }[4], [x5]
    ld1 { v24.h }[4], [x4]
    ld1 { v28.h }[4], [x3]
    b label_223
KAI_ASM_LABEL(label_221)  // Height 6: Partial accumulate: partial_1_0
    tbz x22, #1, label_222
    ldr s8, [x23], #0x4
    ldr s12, [x7], #0x4
    mov x1, #0x4
    ldr s16, [x6], #0x4
    ldr s20, [x5], #0x4
    ldr s24, [x4], #0x4
    ldr s28, [x3], #0x4
    tbz x22, #0, label_223
    ld1 { v8.h }[2], [x23]
    ld1 { v12.h }[2], [x7]
    ld1 { v16.h }[2], [x6]
    ld1 { v20.h }[2], [x5]
    ld1 { v24.h }[2], [x4]
    ld1 { v28.h }[2], [x3]
    b label_223
KAI_ASM_LABEL(label_222)  // Height 6: Partial accumulate: partial_0_0
    ldr h8, [x23, #0]
    ldr h12, [x7, #0]
    mov x1, #0
    ldr h16, [x6, #0]
    ldr h20, [x5, #0]
    ldr h24, [x4, #0]
    ldr h28, [x3, #0]
KAI_ASM_LABEL(label_223)  // Height 6: Partial accumulate: Done
    sub x23, x23, x1
    fcvtl2 v11.4s, v9.8h
    fcvtl v10.4s, v9.4h
    fcvtl2 v9.4s, v8.8h
    fcvtl v8.4s, v8.4h
    fcvtl2 v15.4s, v13.8h
    fcvtl v14.4s, v13.4h
    fcvtl2 v13.4s, v12.8h
    fcvtl v12.4s, v12.4h
    fcvtl2 v19.4s, v17.8h
    fcvtl v18.4s, v17.4h
    fcvtl2 v17.4s, v16.8h
    fcvtl v16.4s, v16.4h
    fcvtl2 v23.4s, v21.8h
    fcvtl v22.4s, v21.4h
    fcvtl2 v21.4s, v20.8h
    fcvtl v20.4s, v20.4h
    fcvtl2 v27.4s, v25.8h
    fcvtl v26.4s, v25.4h
    fcvtl2 v25.4s, v24.8h
    fcvtl v24.4s, v24.4h
    fcvtl2 v31.4s, v29.8h
    fcvtl v30.4s, v29.4h
    fcvtl2 v29.4s, v28.8h
    fcvtl v28.4s, v28.4h
    b label_226
KAI_ASM_LABEL(label_224)  // Height 6: full accumulate
    ldr q22, [x23, #0]
    ldr q20, [x23, #0x10]
    ldr q18, [x7, #0]
    ldr q16, [x7, #0x10]
    ldr q17, [x6, #0]
    ldr q19, [x6, #0x10]
    ldr q21, [x5, #0]
    ldr q23, [x5, #0x10]
    fcvtl v8.4s, v22.4h
    fcvtl2 v9.4s, v22.8h
    ldr q25, [x4, #0]
    ldr q27, [x4, #0x10]
    fcvtl v10.4s, v20.4h
    fcvtl2 v11.4s, v20.8h
    ldr q29, [x3, #0]
    ldr q31, [x3, #0x10]
    fcvtl v12.4s, v18.4h
    fcvtl2 v13.4s, v18.8h
    fcvtl v14.4s, v16.4h
    fcvtl2 v15.4s, v16.8h
    fcvtl v16.4s, v17.4h
    fcvtl2 v17.4s, v17.8h
    fcvtl v18.4s, v19.4h
    fcvtl2 v19.4s, v19.8h
    fcvtl v20.4s, v21.4h
    fcvtl2 v21.4s, v21.8h
    fcvtl v22.4s, v23.4h
    fcvtl2 v23.4s, v23.8h
    fcvtl v24.4s, v25.4h
    fcvtl2 v25.4s, v25.8h
    fcvtl v26.4s, v27.4h
    fcvtl2 v27.4s, v27.8h
    fcvtl v28.4s, v29.4h
    fcvtl2 v29.4s, v29.8h
    fcvtl v30.4s, v31.4h
    fcvtl2 v31.4s, v31.8h
    b label_226
KAI_ASM_LABEL(label_225)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_226)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x25, #3, label_227
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x27, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_229
KAI_ASM_LABEL(label_227)  // Height 6: setup indirect input
    mov x24, #0
KAI_ASM_LABEL(label_228)  // Height 6: String loop
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
    cbnz x24, label_229
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_229)  // Height 6: input setup done
    cmp x7, #0x8
    blt label_232
    ldr q0, [x8, #0]
    ldr q1, [x6, #0]
    cmp x7, #0x10
    ldr q2, [x5, #0]
    ldr q3, [x4, #0]
    ldr q4, [x2, #0]
    ldr q5, [x1, #0]
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    blt label_231
KAI_ASM_LABEL(label_230)  // Height 6: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    sub x7, x7, #0x8
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500dc)  // fmlal v28.4s, v6.4h, v5.h[0]
    add x4, x4, #0x10
    add x2, x2, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    add x1, x1, #0x10
    cmp x7, #0x10
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580dd)  // fmlal2 v29.4s, v6.4h, v5.h[0]
    ldr q6, [x21, #0x20]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x2, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500fe)  // fmlal v30.4s, v7.4h, v5.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580ff)  // fmlal2 v31.4s, v7.4h, v5.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400d8)  // fmlal v24.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x4f9500dc)  // fmlal v28.4s, v6.4h, v5.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480d9)  // fmlal2 v25.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x6f9580dd)  // fmlal2 v29.4s, v6.4h, v5.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400fa)  // fmlal v26.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x4f9500fe)  // fmlal v30.4s, v7.4h, v5.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480fb)  // fmlal2 v27.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x6f9580ff)  // fmlal2 v31.4s, v7.4h, v5.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400d8)  // fmlal v24.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x4fa500dc)  // fmlal v28.4s, v6.4h, v5.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480d9)  // fmlal2 v25.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x6fa580dd)  // fmlal2 v29.4s, v6.4h, v5.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400fa)  // fmlal v26.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x4fa500fe)  // fmlal v30.4s, v7.4h, v5.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480fb)  // fmlal2 v27.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x6fa580ff)  // fmlal2 v31.4s, v7.4h, v5.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400d8)  // fmlal v24.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x4fb500dc)  // fmlal v28.4s, v6.4h, v5.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480d9)  // fmlal2 v25.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x6fb580dd)  // fmlal2 v29.4s, v6.4h, v5.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400fa)  // fmlal v26.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x4fb500fe)  // fmlal v30.4s, v7.4h, v5.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480fb)  // fmlal2 v27.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x6fb580ff)  // fmlal2 v31.4s, v7.4h, v5.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408d8)  // fmlal v24.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x4f8508dc)  // fmlal v28.4s, v6.4h, v5.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488d9)  // fmlal2 v25.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x6f8588dd)  // fmlal2 v29.4s, v6.4h, v5.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408fa)  // fmlal v26.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x4f8508fe)  // fmlal v30.4s, v7.4h, v5.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488fb)  // fmlal2 v27.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x6f8588ff)  // fmlal2 v31.4s, v7.4h, v5.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408d8)  // fmlal v24.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x4f9508dc)  // fmlal v28.4s, v6.4h, v5.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488d9)  // fmlal2 v25.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x6f9588dd)  // fmlal2 v29.4s, v6.4h, v5.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408fa)  // fmlal v26.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x4f9508fe)  // fmlal v30.4s, v7.4h, v5.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488fb)  // fmlal2 v27.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x6f9588ff)  // fmlal2 v31.4s, v7.4h, v5.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408d8)  // fmlal v24.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x4fa508dc)  // fmlal v28.4s, v6.4h, v5.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488d9)  // fmlal2 v25.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x6fa588dd)  // fmlal2 v29.4s, v6.4h, v5.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408fa)  // fmlal v26.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x4fa508fe)  // fmlal v30.4s, v7.4h, v5.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488fb)  // fmlal2 v27.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x6fa588ff)  // fmlal2 v31.4s, v7.4h, v5.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408d8)  // fmlal v24.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x4fb508dc)  // fmlal v28.4s, v6.4h, v5.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488d9)  // fmlal2 v25.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x6fb588dd)  // fmlal2 v29.4s, v6.4h, v5.h[7]
    ldr q6, [x21, #0]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408fa)  // fmlal v26.4s, v7.4h, v4.h[7]
    KAI_ASM_INST(0x4fb508fe)  // fmlal v30.4s, v7.4h, v5.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    ldr q1, [x6, #0]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    ldr q2, [x5, #0]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
    ldr q3, [x4, #0]
    KAI_ASM_INST(0x6fb488fb)  // fmlal2 v27.4s, v7.4h, v4.h[7]
    ldr q4, [x2, #0]
    KAI_ASM_INST(0x6fb588ff)  // fmlal2 v31.4s, v7.4h, v5.h[7]
    ldr q5, [x1, #0]
    ldr q7, [x21, #0x10]
    bge label_230
KAI_ASM_LABEL(label_231)  // Height 6: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    add x8, x8, #0x10
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500dc)  // fmlal v28.4s, v6.4h, v5.h[0]
    add x2, x2, #0x10
    add x1, x1, #0x10
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    prfm pldl1keep, [x5, #0x80]
    prfm pldl1keep, [x4, #0x80]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580dd)  // fmlal2 v29.4s, v6.4h, v5.h[0]
    ldr q6, [x21, #0x20]
    sub x7, x7, #0x8
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    prfm pldl1keep, [x2, #0x80]
    prfm pldl1keep, [x1, #0x80]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500fe)  // fmlal v30.4s, v7.4h, v5.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580ff)  // fmlal2 v31.4s, v7.4h, v5.h[0]
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4f9000c8)  // fmlal v8.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100cc)  // fmlal v12.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200d0)  // fmlal v16.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300d4)  // fmlal v20.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400d8)  // fmlal v24.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x4f9500dc)  // fmlal v28.4s, v6.4h, v5.h[1]
    KAI_ASM_INST(0x6f9080c9)  // fmlal2 v9.4s, v6.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180cd)  // fmlal2 v13.4s, v6.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280d1)  // fmlal2 v17.4s, v6.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380d5)  // fmlal2 v21.4s, v6.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480d9)  // fmlal2 v25.4s, v6.4h, v4.h[1]
    KAI_ASM_INST(0x6f9580dd)  // fmlal2 v29.4s, v6.4h, v5.h[1]
    ldr q6, [x21, #0x40]
    KAI_ASM_INST(0x4f9000ea)  // fmlal v10.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x4f9100ee)  // fmlal v14.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x4f9200f2)  // fmlal v18.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x4f9300f6)  // fmlal v22.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x4f9400fa)  // fmlal v26.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x4f9500fe)  // fmlal v30.4s, v7.4h, v5.h[1]
    KAI_ASM_INST(0x6f9080eb)  // fmlal2 v11.4s, v7.4h, v0.h[1]
    KAI_ASM_INST(0x6f9180ef)  // fmlal2 v15.4s, v7.4h, v1.h[1]
    KAI_ASM_INST(0x6f9280f3)  // fmlal2 v19.4s, v7.4h, v2.h[1]
    KAI_ASM_INST(0x6f9380f7)  // fmlal2 v23.4s, v7.4h, v3.h[1]
    KAI_ASM_INST(0x6f9480fb)  // fmlal2 v27.4s, v7.4h, v4.h[1]
    KAI_ASM_INST(0x6f9580ff)  // fmlal2 v31.4s, v7.4h, v5.h[1]
    ldr q7, [x21, #0x50]
    KAI_ASM_INST(0x4fa000c8)  // fmlal v8.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100cc)  // fmlal v12.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200d0)  // fmlal v16.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300d4)  // fmlal v20.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400d8)  // fmlal v24.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x4fa500dc)  // fmlal v28.4s, v6.4h, v5.h[2]
    KAI_ASM_INST(0x6fa080c9)  // fmlal2 v9.4s, v6.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180cd)  // fmlal2 v13.4s, v6.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280d1)  // fmlal2 v17.4s, v6.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380d5)  // fmlal2 v21.4s, v6.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480d9)  // fmlal2 v25.4s, v6.4h, v4.h[2]
    KAI_ASM_INST(0x6fa580dd)  // fmlal2 v29.4s, v6.4h, v5.h[2]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4fa000ea)  // fmlal v10.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x4fa100ee)  // fmlal v14.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x4fa200f2)  // fmlal v18.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x4fa300f6)  // fmlal v22.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x4fa400fa)  // fmlal v26.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x4fa500fe)  // fmlal v30.4s, v7.4h, v5.h[2]
    KAI_ASM_INST(0x6fa080eb)  // fmlal2 v11.4s, v7.4h, v0.h[2]
    KAI_ASM_INST(0x6fa180ef)  // fmlal2 v15.4s, v7.4h, v1.h[2]
    KAI_ASM_INST(0x6fa280f3)  // fmlal2 v19.4s, v7.4h, v2.h[2]
    KAI_ASM_INST(0x6fa380f7)  // fmlal2 v23.4s, v7.4h, v3.h[2]
    KAI_ASM_INST(0x6fa480fb)  // fmlal2 v27.4s, v7.4h, v4.h[2]
    KAI_ASM_INST(0x6fa580ff)  // fmlal2 v31.4s, v7.4h, v5.h[2]
    ldr q7, [x21, #0x70]
    KAI_ASM_INST(0x4fb000c8)  // fmlal v8.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100cc)  // fmlal v12.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200d0)  // fmlal v16.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300d4)  // fmlal v20.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400d8)  // fmlal v24.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x4fb500dc)  // fmlal v28.4s, v6.4h, v5.h[3]
    KAI_ASM_INST(0x6fb080c9)  // fmlal2 v9.4s, v6.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180cd)  // fmlal2 v13.4s, v6.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280d1)  // fmlal2 v17.4s, v6.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380d5)  // fmlal2 v21.4s, v6.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480d9)  // fmlal2 v25.4s, v6.4h, v4.h[3]
    KAI_ASM_INST(0x6fb580dd)  // fmlal2 v29.4s, v6.4h, v5.h[3]
    ldr q6, [x21, #0x80]
    KAI_ASM_INST(0x4fb000ea)  // fmlal v10.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x4fb100ee)  // fmlal v14.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x4fb200f2)  // fmlal v18.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x4fb300f6)  // fmlal v22.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x4fb400fa)  // fmlal v26.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x4fb500fe)  // fmlal v30.4s, v7.4h, v5.h[3]
    KAI_ASM_INST(0x6fb080eb)  // fmlal2 v11.4s, v7.4h, v0.h[3]
    KAI_ASM_INST(0x6fb180ef)  // fmlal2 v15.4s, v7.4h, v1.h[3]
    KAI_ASM_INST(0x6fb280f3)  // fmlal2 v19.4s, v7.4h, v2.h[3]
    KAI_ASM_INST(0x6fb380f7)  // fmlal2 v23.4s, v7.4h, v3.h[3]
    KAI_ASM_INST(0x6fb480fb)  // fmlal2 v27.4s, v7.4h, v4.h[3]
    KAI_ASM_INST(0x6fb580ff)  // fmlal2 v31.4s, v7.4h, v5.h[3]
    ldr q7, [x21, #0x90]
    KAI_ASM_INST(0x4f8008c8)  // fmlal v8.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108cc)  // fmlal v12.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208d0)  // fmlal v16.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308d4)  // fmlal v20.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408d8)  // fmlal v24.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x4f8508dc)  // fmlal v28.4s, v6.4h, v5.h[4]
    KAI_ASM_INST(0x6f8088c9)  // fmlal2 v9.4s, v6.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188cd)  // fmlal2 v13.4s, v6.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288d1)  // fmlal2 v17.4s, v6.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388d5)  // fmlal2 v21.4s, v6.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488d9)  // fmlal2 v25.4s, v6.4h, v4.h[4]
    KAI_ASM_INST(0x6f8588dd)  // fmlal2 v29.4s, v6.4h, v5.h[4]
    ldr q6, [x21, #0xa0]
    KAI_ASM_INST(0x4f8008ea)  // fmlal v10.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x4f8108ee)  // fmlal v14.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x4f8208f2)  // fmlal v18.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x4f8308f6)  // fmlal v22.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x4f8408fa)  // fmlal v26.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x4f8508fe)  // fmlal v30.4s, v7.4h, v5.h[4]
    KAI_ASM_INST(0x6f8088eb)  // fmlal2 v11.4s, v7.4h, v0.h[4]
    KAI_ASM_INST(0x6f8188ef)  // fmlal2 v15.4s, v7.4h, v1.h[4]
    KAI_ASM_INST(0x6f8288f3)  // fmlal2 v19.4s, v7.4h, v2.h[4]
    KAI_ASM_INST(0x6f8388f7)  // fmlal2 v23.4s, v7.4h, v3.h[4]
    KAI_ASM_INST(0x6f8488fb)  // fmlal2 v27.4s, v7.4h, v4.h[4]
    KAI_ASM_INST(0x6f8588ff)  // fmlal2 v31.4s, v7.4h, v5.h[4]
    ldr q7, [x21, #0xb0]
    KAI_ASM_INST(0x4f9008c8)  // fmlal v8.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108cc)  // fmlal v12.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208d0)  // fmlal v16.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308d4)  // fmlal v20.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408d8)  // fmlal v24.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x4f9508dc)  // fmlal v28.4s, v6.4h, v5.h[5]
    KAI_ASM_INST(0x6f9088c9)  // fmlal2 v9.4s, v6.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188cd)  // fmlal2 v13.4s, v6.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288d1)  // fmlal2 v17.4s, v6.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388d5)  // fmlal2 v21.4s, v6.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488d9)  // fmlal2 v25.4s, v6.4h, v4.h[5]
    KAI_ASM_INST(0x6f9588dd)  // fmlal2 v29.4s, v6.4h, v5.h[5]
    ldr q6, [x21, #0xc0]
    KAI_ASM_INST(0x4f9008ea)  // fmlal v10.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x4f9108ee)  // fmlal v14.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x4f9208f2)  // fmlal v18.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x4f9308f6)  // fmlal v22.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x4f9408fa)  // fmlal v26.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x4f9508fe)  // fmlal v30.4s, v7.4h, v5.h[5]
    KAI_ASM_INST(0x6f9088eb)  // fmlal2 v11.4s, v7.4h, v0.h[5]
    KAI_ASM_INST(0x6f9188ef)  // fmlal2 v15.4s, v7.4h, v1.h[5]
    KAI_ASM_INST(0x6f9288f3)  // fmlal2 v19.4s, v7.4h, v2.h[5]
    KAI_ASM_INST(0x6f9388f7)  // fmlal2 v23.4s, v7.4h, v3.h[5]
    KAI_ASM_INST(0x6f9488fb)  // fmlal2 v27.4s, v7.4h, v4.h[5]
    KAI_ASM_INST(0x6f9588ff)  // fmlal2 v31.4s, v7.4h, v5.h[5]
    ldr q7, [x21, #0xd0]
    KAI_ASM_INST(0x4fa008c8)  // fmlal v8.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108cc)  // fmlal v12.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208d0)  // fmlal v16.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308d4)  // fmlal v20.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408d8)  // fmlal v24.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x4fa508dc)  // fmlal v28.4s, v6.4h, v5.h[6]
    KAI_ASM_INST(0x6fa088c9)  // fmlal2 v9.4s, v6.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188cd)  // fmlal2 v13.4s, v6.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288d1)  // fmlal2 v17.4s, v6.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388d5)  // fmlal2 v21.4s, v6.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488d9)  // fmlal2 v25.4s, v6.4h, v4.h[6]
    KAI_ASM_INST(0x6fa588dd)  // fmlal2 v29.4s, v6.4h, v5.h[6]
    ldr q6, [x21, #0xe0]
    KAI_ASM_INST(0x4fa008ea)  // fmlal v10.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x4fa108ee)  // fmlal v14.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x4fa208f2)  // fmlal v18.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x4fa308f6)  // fmlal v22.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x4fa408fa)  // fmlal v26.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x4fa508fe)  // fmlal v30.4s, v7.4h, v5.h[6]
    KAI_ASM_INST(0x6fa088eb)  // fmlal2 v11.4s, v7.4h, v0.h[6]
    KAI_ASM_INST(0x6fa188ef)  // fmlal2 v15.4s, v7.4h, v1.h[6]
    KAI_ASM_INST(0x6fa288f3)  // fmlal2 v19.4s, v7.4h, v2.h[6]
    KAI_ASM_INST(0x6fa388f7)  // fmlal2 v23.4s, v7.4h, v3.h[6]
    KAI_ASM_INST(0x6fa488fb)  // fmlal2 v27.4s, v7.4h, v4.h[6]
    KAI_ASM_INST(0x6fa588ff)  // fmlal2 v31.4s, v7.4h, v5.h[6]
    ldr q7, [x21, #0xf0]
    add x21, x21, #0x100
    KAI_ASM_INST(0x4fb008c8)  // fmlal v8.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108cc)  // fmlal v12.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208d0)  // fmlal v16.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308d4)  // fmlal v20.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408d8)  // fmlal v24.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x4fb508dc)  // fmlal v28.4s, v6.4h, v5.h[7]
    KAI_ASM_INST(0x6fb088c9)  // fmlal2 v9.4s, v6.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188cd)  // fmlal2 v13.4s, v6.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288d1)  // fmlal2 v17.4s, v6.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388d5)  // fmlal2 v21.4s, v6.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488d9)  // fmlal2 v25.4s, v6.4h, v4.h[7]
    KAI_ASM_INST(0x6fb588dd)  // fmlal2 v29.4s, v6.4h, v5.h[7]
    KAI_ASM_INST(0x4fb008ea)  // fmlal v10.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x4fb108ee)  // fmlal v14.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x4fb208f2)  // fmlal v18.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x4fb308f6)  // fmlal v22.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x4fb408fa)  // fmlal v26.4s, v7.4h, v4.h[7]
    KAI_ASM_INST(0x4fb508fe)  // fmlal v30.4s, v7.4h, v5.h[7]
    KAI_ASM_INST(0x6fb088eb)  // fmlal2 v11.4s, v7.4h, v0.h[7]
    KAI_ASM_INST(0x6fb188ef)  // fmlal2 v15.4s, v7.4h, v1.h[7]
    KAI_ASM_INST(0x6fb288f3)  // fmlal2 v19.4s, v7.4h, v2.h[7]
    KAI_ASM_INST(0x6fb388f7)  // fmlal2 v23.4s, v7.4h, v3.h[7]
    KAI_ASM_INST(0x6fb488fb)  // fmlal2 v27.4s, v7.4h, v4.h[7]
    KAI_ASM_INST(0x6fb588ff)  // fmlal2 v31.4s, v7.4h, v5.h[7]
KAI_ASM_LABEL(label_232)  // Height 6: Multiply loop: Main loop skip
    cbz x7, label_234
KAI_ASM_LABEL(label_233)  // Height 6: Multiply loop: Odd block loop
    ldr h0, [x8], #0x2
    ldr h1, [x6], #0x2
    sub x7, x7, #0x1
    ldr h2, [x5], #0x2
    ldr h3, [x4], #0x2
    ldr h4, [x2], #0x2
    ldr h5, [x1], #0x2
    ldr q6, [x21, #0]
    ldr q7, [x21, #0x10]
    add x21, x21, #0x20
    KAI_ASM_INST(0x4f8000c8)  // fmlal v8.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100cc)  // fmlal v12.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200d0)  // fmlal v16.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300d4)  // fmlal v20.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400d8)  // fmlal v24.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500dc)  // fmlal v28.4s, v6.4h, v5.h[0]
    KAI_ASM_INST(0x6f8080c9)  // fmlal2 v9.4s, v6.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180cd)  // fmlal2 v13.4s, v6.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280d1)  // fmlal2 v17.4s, v6.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380d5)  // fmlal2 v21.4s, v6.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480d9)  // fmlal2 v25.4s, v6.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580dd)  // fmlal2 v29.4s, v6.4h, v5.h[0]
    KAI_ASM_INST(0x4f8000ea)  // fmlal v10.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x4f8100ee)  // fmlal v14.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x4f8200f2)  // fmlal v18.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x4f8300f6)  // fmlal v22.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x4f8400fa)  // fmlal v26.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x4f8500fe)  // fmlal v30.4s, v7.4h, v5.h[0]
    KAI_ASM_INST(0x6f8080eb)  // fmlal2 v11.4s, v7.4h, v0.h[0]
    KAI_ASM_INST(0x6f8180ef)  // fmlal2 v15.4s, v7.4h, v1.h[0]
    KAI_ASM_INST(0x6f8280f3)  // fmlal2 v19.4s, v7.4h, v2.h[0]
    KAI_ASM_INST(0x6f8380f7)  // fmlal2 v23.4s, v7.4h, v3.h[0]
    KAI_ASM_INST(0x6f8480fb)  // fmlal2 v27.4s, v7.4h, v4.h[0]
    KAI_ASM_INST(0x6f8580ff)  // fmlal2 v31.4s, v7.4h, v5.h[0]
    cbnz x7, label_233
KAI_ASM_LABEL(label_234)  // Height 6: Multiply loop: No odd multiplies
    tbz x25, #3, label_235
    ldr x1, [x0, #0x10]
    add x24, x24, #0x1
    cmp x24, x1
    bne label_228
KAI_ASM_LABEL(label_235)  // Height 6: no string loop
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
    tbz x25, #1, label_236
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v1.8h }, [x2]
    ld1r { v0.8h }, [x1]
    fcvtl v1.4s, v1.4h
    fcvtl v0.4s, v0.4h
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
KAI_ASM_LABEL(label_236)  // Height 6: No activation
    fcvtn v8.4h, v8.4s
    fcvtn v12.4h, v12.4s
    cmp x22, #0x10
    fcvtn v16.4h, v16.4s
    fcvtn v20.4h, v20.4s
    fcvtn v24.4h, v24.4s
    fcvtn v28.4h, v28.4s
    fcvtn2 v8.8h, v9.4s
    fcvtn v9.4h, v10.4s
    fcvtn2 v12.8h, v13.4s
    fcvtn v13.4h, v14.4s
    fcvtn2 v16.8h, v17.4s
    fcvtn v17.4h, v18.4s
    fcvtn2 v20.8h, v21.4s
    fcvtn v21.4h, v22.4s
    fcvtn2 v24.8h, v25.4s
    fcvtn v25.4h, v26.4s
    fcvtn2 v28.8h, v29.4s
    fcvtn v29.4h, v30.4s
    fcvtn2 v9.8h, v11.4s
    fcvtn2 v13.8h, v15.4s
    fcvtn2 v17.8h, v19.4s
    fcvtn2 v21.8h, v23.4s
    fcvtn2 v25.8h, v27.4s
    fcvtn2 v29.8h, v31.4s
    bge label_245
    tbz x22, #3, label_240
    st1 { v8.4s }, [x23], #0x10
    st1 { v12.4s }, [x7], #0x10
    st1 { v16.4s }, [x6], #0x10
    st1 { v20.4s }, [x5], #0x10
    st1 { v24.4s }, [x4], #0x10
    st1 { v28.4s }, [x3], #0x10
    tbz x22, #2, label_238
    str d9, [x23], #0x8
    str d13, [x7], #0x8
    str d17, [x6], #0x8
    str d21, [x5], #0x8
    str d25, [x4], #0x8
    str d29, [x3], #0x8
    tbz x22, #1, label_237
    st1 { v9.s }[2], [x23], #0x4
    st1 { v13.s }[2], [x7], #0x4
    st1 { v17.s }[2], [x6], #0x4
    st1 { v21.s }[2], [x5], #0x4
    st1 { v25.s }[2], [x4], #0x4
    st1 { v29.s }[2], [x3], #0x4
    tbz x22, #0, label_244
    st1 { v9.h }[6], [x23]
    st1 { v13.h }[6], [x7]
    st1 { v17.h }[6], [x6]
    st1 { v21.h }[6], [x5]
    st1 { v25.h }[6], [x4]
    st1 { v29.h }[6], [x3]
    b label_244
KAI_ASM_LABEL(label_237)  // Height 6: Partial direct writeback: partial_0_12
    tbz x22, #0, label_244
    st1 { v9.h }[4], [x23]
    st1 { v13.h }[4], [x7]
    st1 { v17.h }[4], [x6]
    st1 { v21.h }[4], [x5]
    st1 { v25.h }[4], [x4]
    st1 { v29.h }[4], [x3]
    b label_244
KAI_ASM_LABEL(label_238)  // Height 6: Partial direct writeback: partial_1_8
    tbz x22, #1, label_239
    str s9, [x23], #0x4
    str s13, [x7], #0x4
    str s17, [x6], #0x4
    str s21, [x5], #0x4
    str s25, [x4], #0x4
    str s29, [x3], #0x4
    tbz x22, #0, label_244
    st1 { v9.h }[2], [x23]
    st1 { v13.h }[2], [x7]
    st1 { v17.h }[2], [x6]
    st1 { v21.h }[2], [x5]
    st1 { v25.h }[2], [x4]
    st1 { v29.h }[2], [x3]
    b label_244
KAI_ASM_LABEL(label_239)  // Height 6: Partial direct writeback: partial_0_8
    tbz x22, #0, label_244
    str h9, [x23, #0]
    str h13, [x7, #0]
    str h17, [x6, #0]
    str h21, [x5, #0]
    str h25, [x4, #0]
    str h29, [x3, #0]
    b label_244
KAI_ASM_LABEL(label_240)  // Height 6: Partial direct writeback: partial_2_0
    tbz x22, #2, label_242
    str d8, [x23], #0x8
    str d12, [x7], #0x8
    str d16, [x6], #0x8
    str d20, [x5], #0x8
    str d24, [x4], #0x8
    str d28, [x3], #0x8
    tbz x22, #1, label_241
    st1 { v8.s }[2], [x23], #0x4
    st1 { v12.s }[2], [x7], #0x4
    st1 { v16.s }[2], [x6], #0x4
    st1 { v20.s }[2], [x5], #0x4
    st1 { v24.s }[2], [x4], #0x4
    st1 { v28.s }[2], [x3], #0x4
    tbz x22, #0, label_244
    st1 { v8.h }[6], [x23]
    st1 { v12.h }[6], [x7]
    st1 { v16.h }[6], [x6]
    st1 { v20.h }[6], [x5]
    st1 { v24.h }[6], [x4]
    st1 { v28.h }[6], [x3]
    b label_244
KAI_ASM_LABEL(label_241)  // Height 6: Partial direct writeback: partial_0_4
    tbz x22, #0, label_244
    st1 { v8.h }[4], [x23]
    st1 { v12.h }[4], [x7]
    st1 { v16.h }[4], [x6]
    st1 { v20.h }[4], [x5]
    st1 { v24.h }[4], [x4]
    st1 { v28.h }[4], [x3]
    b label_244
KAI_ASM_LABEL(label_242)  // Height 6: Partial direct writeback: partial_1_0
    tbz x22, #1, label_243
    str s8, [x23], #0x4
    str s12, [x7], #0x4
    str s16, [x6], #0x4
    str s20, [x5], #0x4
    str s24, [x4], #0x4
    str s28, [x3], #0x4
    tbz x22, #0, label_244
    st1 { v8.h }[2], [x23]
    st1 { v12.h }[2], [x7]
    st1 { v16.h }[2], [x6]
    st1 { v20.h }[2], [x5]
    st1 { v24.h }[2], [x4]
    st1 { v28.h }[2], [x3]
    b label_244
KAI_ASM_LABEL(label_243)  // Height 6: Partial direct writeback: partial_0_0
    str h8, [x23, #0]
    str h12, [x7, #0]
    str h16, [x6, #0]
    str h20, [x5, #0]
    str h24, [x4, #0]
    str h28, [x3, #0]
KAI_ASM_LABEL(label_244)  // Height 6: Partial direct writeback: Done
    b label_246
KAI_ASM_LABEL(label_245)  // Height 6: Full writeback
    str q8, [x23, #0]
    str q9, [x23, #0x10]
    add x23, x23, #0x20
    str q12, [x7, #0]
    str q13, [x7, #0x10]
    str q16, [x6, #0]
    str q17, [x6, #0x10]
    str q20, [x5, #0]
    str q21, [x5, #0x10]
    str q24, [x4, #0]
    str q25, [x4, #0x10]
    str q28, [x3, #0]
    str q29, [x3, #0x10]
KAI_ASM_LABEL(label_246)  // Height 6: Writeback done
    subs x22, x22, #0x10
    bgt label_207
    add x27, x27, #0x6
    cmp x27, x26
    blt label_1
KAI_ASM_LABEL(label_247)  // Exit
    add SP, SP, #0x20
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_hybrid_fp16fp32fp16_mla_6x16)

    KAI_ASM_END
