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

    KAI_ASM_CODE(sme_interleave1VL_block4_u8_u8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_block4_u8_u8)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_block4_u8_u8)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_block4_u8_u8)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x23, #0
    cntb x7
    mov x6, x2
    ldr x22, [x0, #0]
    mov x5, x2
    sub x21, x7, #0x1
    cntw x20
    incb x6
    ands x21, x5, x21
    sub x6, x6, #0x1
    csel x21, x21, x7, NE
    lsl x5, x3, #0x1  // height * 2
    lsl x3, x20, #0x1
    udiv x6, x6, x7  // n_passes = ceildiv(width, VL<T>)
    add x21, x21, #0x3
    sub x8, x20, #0x2
    whilelt p9.b, XZR, x5
    whilelt p8.b, x3, x5
    mov x7, x1
    sub x3, x6, #0x1
    lsr x21, x21, #0x2
    ptrue p11.s
    ldr x6, [x7, #0]
    zip1 p10.b, p9.b, p8.b
    mov x5, x4
    ldr x4, [x7, #0x8]
    whilelt p9.b, x23, x2
    whilelt p8.b, x23, x2
    add x7, x7, #0x10
    mov x12, #0
    cbz x8, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0xe00504c0)  // ld1b { za0h.b[x12] }, p1/Z, [x6, x5]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe0050084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x5]
    add x12, x12, #0x8
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    cmp x12, x8, LSL #2
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    mov x3, x3
    mov x7, x1
    cmp x3, #0x2
    incb x23
    KAI_ASM_INST(0xe00504c0)  // ld1b { za0h.b[x12] }, p1/Z, [x6, x5]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe0050084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x5]
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    incb x5
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.b, x23, x2
    mov x12, #0
    mov x14, #0
    cbz x8, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    KAI_ASM_INST(0xe0050cc2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x6, x5]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe0050886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x5]
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    add x12, x12, #0x8
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    add x14, x14, #0x2
    addvl x22, x22, #2
    cmp x14, x8
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    mov x7, x1
    whilelt p9.b, x23, x2
    KAI_ASM_INST(0xe0050cc2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x6, x5]
    ldr x6, [x7, #0]
    incb x23
    mov x13, #0
    KAI_ASM_INST(0xe0050886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x5]
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    incb x5
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    whilelt p8.b, x23, x2
    mov x12, #0
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    cbz x8, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    KAI_ASM_INST(0xe0052cc0)  // ld1b { za0h.b[x13] }, p3/Z, [x6, x5]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe0052884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x5]
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    add x13, x13, #0x8
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    add x12, x12, #0x2
    addvl x22, x22, #2
    cmp x12, x8
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    mov x7, x1
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe0052cc0)  // ld1b { za0h.b[x13] }, p3/Z, [x6, x5]
    ldr x6, [x7, #0]
    whilelt p9.b, x23, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe0052884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x5]
    ldr x4, [x7, #0x8]
    add x7, x7, #0x10
    incb x23
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    incb x5
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x7, x1
    whilelt p8.b, x23, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356140)  // psel p0.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0xe0bf86c0)  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    ldr x1, [x7, #0]
    cmp x12, x20
    add x7, x7, #0x8
    KAI_ASM_INST(0xe0052022)  // ld1b { za0h.b[x13, #2] }, p0/Z, [x1, x5]
    add x13, x13, #0x4
    blt label_9
    whilelt p9.b, x23, x2
    whilelt p8.b, x23, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    add x1, x1, #0x4
    KAI_ASM_INST(0xe0bf82c8)  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_10
    whilelt p8.b, x23, x2
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
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_block4_u8_u8)

    KAI_ASM_END
