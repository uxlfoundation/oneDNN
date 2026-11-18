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

    KAI_ASM_CODE(sme_interleave1VL_fp32_fp32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_fp32_fp32)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_fp32_fp32)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_fp32_fp32)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x24, #0
    mov x6, x2
    cntw x23
    ldr x22, [x0, #0]
    incw x6
    mov x5, x2
    sub x6, x6, #0x1
    sub x21, x23, #0x1
    udiv x6, x6, x23  // n_passes = ceildiv(width, VL<T>)
    ands x21, x5, x21
    sub x20, x23, #0x2
    mov x8, x1
    sub x7, x6, #0x1
    csel x21, x21, x23, NE
    ldr x6, [x8, #0]
    ptrue p11.s
    whilelt p10.s, XZR, x3
    ldr x5, [x8, #0x8]
    mov x4, x4
    whilelt p9.s, x24, x2
    whilelt p8.s, x24, x2
    add x8, x8, #0x10
    mov x12, #0
    cbz x20, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25306141)  // psel p1.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe08404c0)  // ld1w { za0h.s[x12] }, p1/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe08400a1)  // ld1w { za0h.s[x12, #1] }, p0/Z, [x5, x4, LSL #2]
    add x12, x12, #0x2
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    cmp x12, x20
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25306141)  // psel p1.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    mov x3, x7
    mov x8, x1
    cmp x3, #0x2
    incw x24
    KAI_ASM_INST(0xe08404c0)  // ld1w { za0h.s[x12] }, p1/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe08400a1)  // ld1w { za0h.s[x12, #1] }, p0/Z, [x5, x4, LSL #2]
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    incw x4
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.s, x24, x2
    mov x13, #0
    cbz x20, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25316143)  // psel p3.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716142)  // psel p2.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0x25316d21)  // psel p1.s, p11.s/Z, p9.s[w13]
    KAI_ASM_INST(0x25716d20)  // psel p0.s, p11.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0842cc8)  // ld1w { za2h.s[x13] }, p3/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe08428a9)  // ld1w { za2h.s[x13, #1] }, p2/Z, [x5, x4, LSL #2]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe0bfa6c0)  // st1w { za0v.s[x13] }, p1/Z, [x22, XZR, LSL #2]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe0b7a2c1)  // st1w { za0v.s[x13, #1] }, p0/Z, [x22, x23, LSL #2]
    add x13, x13, #0x2
    addvl x22, x22, #2
    cmp x13, x20
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25316143)  // psel p3.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716142)  // psel p2.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0x25316d21)  // psel p1.s, p11.s/Z, p9.s[w13]
    KAI_ASM_INST(0x25716d20)  // psel p0.s, p11.s/Z, p9.s[w13, #1]
    mov x8, x1
    whilelt p9.s, x24, x2
    KAI_ASM_INST(0xe0842cc8)  // ld1w { za2h.s[x13] }, p3/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    incw x24
    mov x12, #0
    KAI_ASM_INST(0xe08428a9)  // ld1w { za2h.s[x13, #1] }, p2/Z, [x5, x4, LSL #2]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe0bfa6c0)  // st1w { za0v.s[x13] }, p1/Z, [x22, XZR, LSL #2]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe0b7a2c1)  // st1w { za0v.s[x13, #1] }, p0/Z, [x22, x23, LSL #2]
    addvl x22, x22, #2
    incw x4
    whilelt p8.s, x24, x2
    cbz x20, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25306143)  // psel p3.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706142)  // psel p2.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25706d20)  // psel p0.s, p11.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0840cc0)  // ld1w { za0h.s[x12] }, p3/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe08408a1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x5, x4, LSL #2]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe0b782c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x23, LSL #2]
    add x12, x12, #0x2
    addvl x22, x22, #2
    cmp x12, x20
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25306143)  // psel p3.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706142)  // psel p2.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25706d20)  // psel p0.s, p11.s/Z, p9.s[w12, #1]
    mov x8, x1
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe0840cc0)  // ld1w { za0h.s[x12] }, p3/Z, [x6, x4, LSL #2]
    ldr x6, [x8, #0]
    whilelt p9.s, x24, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe08408a1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x5, x4, LSL #2]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe0b782c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x23, LSL #2]
    addvl x22, x22, #2
    incw x24
    incw x4
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x8, x1
    whilelt p8.s, x24, x2
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306140)  // psel p0.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf86c0)  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    addvl x22, x22, #1
    ldr x1, [x8, #0]
    add x8, x8, #0x8
    KAI_ASM_INST(0xe0840028)  // ld1w { za2h.s[x12] }, p0/Z, [x1, x4, LSL #2]
    add x12, x12, #0x1
    cmp x12, x23
    blt label_9
    whilelt p9.s, x24, x2
    whilelt p8.s, x24, x2
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf82c8)  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_10
    whilelt p8.s, x24, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf82c0)  // st1w { za0v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x22, [x0]
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_fp32_fp32)

    KAI_ASM_END
