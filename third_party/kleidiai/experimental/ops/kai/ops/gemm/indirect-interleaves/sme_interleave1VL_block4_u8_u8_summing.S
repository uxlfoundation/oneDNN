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

    KAI_ASM_CODE(sme_interleave1VL_block4_u8_u8_summing)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_block4_u8_u8_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_block4_u8_u8_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_block4_u8_u8_summing)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x23, #0
    cntb x8
    mov x7, x2
    ldr x22, [x0, #0]
    dup z19.s, w6
    mov x6, x2
    sub x21, x8, #0x1
    mov z18.b, #0x1
    mov z17.s, #0
    cntw x20
    incb x7
    ands x21, x6, x21
    sub x7, x7, #0x1
    csel x21, x21, x8, NE
    lsl x6, x3, #0x1  // height * 2
    lsl x3, x20, #0x1
    udiv x7, x7, x8  // n_passes = ceildiv(width, VL<T>)
    add x21, x21, #0x3
    whilelt p9.b, XZR, x6
    whilelt p8.b, x3, x6
    ptrue p4.b
    sub x3, x7, #0x1
    lsr x21, x21, #0x2
    sub x8, x20, #0x2
    ptrue p11.s
    zip1 p10.b, p9.b, p8.b
    mov x7, x4
    whilelt p9.b, x23, x2
    whilelt p8.b, x23, x2
    cbnz w5, label_1
    addvl x22, x22, #-1
    ld1w { z17.s }, p4/Z, [x22]
KAI_ASM_LABEL(label_1)  // K loop: Load row sums: End
    mov x6, x1
    mov x12, #0
    ldr x5, [x6, #0]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    cbz x8, label_3
KAI_ASM_LABEL(label_2)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0xe00704a0)  // ld1b { za0h.b[x12] }, p1/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x7]
    add x12, x12, #0x8
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    cmp x12, x8, LSL #2
    blt label_2
KAI_ASM_LABEL(label_3)  // K loop: Charge: End
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    mov x3, x3
    mov x6, x1
    cmp x3, #0x2
    incb x23
    KAI_ASM_INST(0xe00704a0)  // ld1b { za0h.b[x12] }, p1/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x7
    blt label_9
KAI_ASM_LABEL(label_4)  // K loop: Main loop
    whilelt p8.b, x23, x2
    mov x12, #0
    mov x14, #0
    cbz x8, label_6
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    KAI_ASM_INST(0xe0070ca2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    add x12, x12, #0x8
    KAI_ASM_INST(0xc082d010)  // mova z16.s, p4/M, za0v.s[x14]
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    KAI_ASM_INST(0xc082d030)  // mova z16.s, p4/M, za0v.s[x14, #1]
    add x14, x14, #0x2
    cmp x14, x8
    addvl x22, x22, #2
    udot z17.s, z16.b, z18.b
    blt label_5
KAI_ASM_LABEL(label_6)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    mov x6, x1
    whilelt p9.b, x23, x2
    KAI_ASM_INST(0xe0070ca2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    incb x23
    mov x13, #0
    KAI_ASM_INST(0xe0070886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x7
    KAI_ASM_INST(0xc082d010)  // mova z16.s, p4/M, za0v.s[x14]
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    whilelt p8.b, x23, x2
    mov x12, #0
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xc082d030)  // mova z16.s, p4/M, za0v.s[x14, #1]
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    udot z17.s, z16.b, z18.b
    cbz x8, label_8
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    KAI_ASM_INST(0xe0072ca0)  // ld1b { za0h.b[x13] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0072884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    add x13, x13, #0x8
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    KAI_ASM_INST(0xc0829130)  // mova z16.s, p4/M, za2v.s[x12, #1]
    add x12, x12, #0x2
    cmp x12, x8
    addvl x22, x22, #2
    udot z17.s, z16.b, z18.b
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    mov x6, x1
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe0072ca0)  // ld1b { za0h.b[x13] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    whilelt p9.b, x23, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe0072884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x23
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    incb x7
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xc0829130)  // mova z16.s, p4/M, za2v.s[x12, #1]
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    udot z17.s, z16.b, z18.b
    bge label_4
KAI_ASM_LABEL(label_9)  // K loop: Tails
    cbz x3, label_12
    mov x6, x1
    whilelt p8.b, x23, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356140)  // psel p0.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0xc0829010)  // mova z16.s, p4/M, za0v.s[x12]
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf86c0)  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    ldr x1, [x6, #0]
    cmp x12, x20
    add x6, x6, #0x8
    KAI_ASM_INST(0xe0072022)  // ld1b { za0h.b[x13, #2] }, p0/Z, [x1, x7]
    add x13, x13, #0x4
    blt label_10
    whilelt p9.b, x23, x2
    whilelt p8.b, x23, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_11)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    add x1, x1, #0x4
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf82c8)  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_11
    whilelt p8.b, x23, x2
    b label_14
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_13)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0829010)  // mova z16.s, p4/M, za0v.s[x12]
    udot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf82c0)  // st1w { za0v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_13
KAI_ASM_LABEL(label_14)  // K loop: End
    mul z17.s, p4/M, z17.s, z19.s
    st1w { z17.s }, p4, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    str x22, [x0]
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_block4_u8_u8_summing)

    KAI_ASM_END
