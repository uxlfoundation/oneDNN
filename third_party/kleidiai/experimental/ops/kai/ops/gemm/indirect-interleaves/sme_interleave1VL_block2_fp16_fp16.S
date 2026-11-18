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

    KAI_ASM_CODE(sme_interleave1VL_block2_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_block2_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_block2_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_block2_fp16_fp16)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x23, #0
    cnth x7
    mov x6, x2
    ldr x22, [x0, #0]
    mov x5, x2
    sub x21, x7, #0x1
    inch x6
    ands x21, x5, x21
    cntw x20
    sub x6, x6, #0x1
    csel x21, x21, x7, NE
    udiv x6, x6, x7  // n_passes = ceildiv(width, VL<T>)
    add x21, x21, #0x1
    sub x8, x20, #0x2
    lsl x5, x3, #0x1  // height * 2
    mov x7, x1
    sub x3, x6, #0x1
    lsr x21, x21, #0x1
    ldr x6, [x7, #0]
    ptrue p11.s
    whilelt p10.h, XZR, x5
    ldr x5, [x7, #0x8]
    mov x4, x4
    whilelt p9.h, x23, x2
    whilelt p8.h, x23, x2
    add x7, x7, #0x10
    mov x12, #0
    cbz x8, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe04400a2)  // ld1h { za0h.h[x12, #2] }, p0/Z, [x5, x4, LSL #1]
    add x12, x12, #0x4
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    cmp x12, x8, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    mov x3, x3
    mov x7, x1
    cmp x3, #0x2
    inch x23
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe04400a2)  // ld1h { za0h.h[x12, #2] }, p0/Z, [x5, x4, LSL #1]
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    inch x4
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x23, x2
    mov x12, #0
    mov x14, #0
    cbz x8, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25386143)  // psel p3.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25786142)  // psel p2.h, p8.h/Z, p10.h[w12, #3]
    KAI_ASM_INST(0x252a6d21)  // psel p1.h, p11.h/Z, p9.h[w14]
    KAI_ASM_INST(0x253a6d20)  // psel p0.h, p11.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0440cc1)  // ld1h { za0h.h[x12, #1] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe04408a3)  // ld1h { za0h.h[x12, #3] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    add x12, x12, #0x4
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    add x14, x14, #0x2
    addvl x22, x22, #2
    cmp x14, x8
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25386143)  // psel p3.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25786142)  // psel p2.h, p8.h/Z, p10.h[w12, #3]
    KAI_ASM_INST(0x252a6d21)  // psel p1.h, p11.h/Z, p9.h[w14]
    KAI_ASM_INST(0x253a6d20)  // psel p0.h, p11.h/Z, p9.h[w14, #1]
    mov x7, x1
    whilelt p9.h, x23, x2
    KAI_ASM_INST(0xe0440cc1)  // ld1h { za0h.h[x12, #1] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    inch x23
    mov x13, #0
    KAI_ASM_INST(0xe04408a3)  // ld1h { za0h.h[x12, #3] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    inch x4
    KAI_ASM_INST(0xe0bfc6c0)  // st1w { za0v.s[x14] }, p1/Z, [x22, XZR, LSL #2]
    whilelt p8.h, x23, x2
    mov x12, #0
    KAI_ASM_INST(0xe0b4c2c1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    cbz x8, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696142)  // psel p2.h, p8.h/Z, p10.h[w13, #2]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0442cc0)  // ld1h { za0h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    KAI_ASM_INST(0xe04428a2)  // ld1h { za0h.h[x13, #2] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    add x12, x12, #0x2
    addvl x22, x22, #2
    cmp x12, x8
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696142)  // psel p2.h, p8.h/Z, p10.h[w13, #2]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    mov x7, x1
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe0442cc0)  // ld1h { za0h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x7, #0]
    whilelt p9.h, x23, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe04428a2)  // ld1h { za0h.h[x13, #2] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x7, #0x8]
    add x7, x7, #0x10
    inch x23
    KAI_ASM_INST(0xe0bf86c8)  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    inch x4
    KAI_ASM_INST(0xe0b482c9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x22, x20, LSL #2]
    addvl x22, x22, #2
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x7, x1
    whilelt p8.h, x23, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25396140)  // psel p0.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0xe0bf86c0)  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    ldr x1, [x7, #0]
    cmp x12, x20
    add x7, x7, #0x8
    KAI_ASM_INST(0xe0442021)  // ld1h { za0h.h[x13, #1] }, p0/Z, [x1, x4, LSL #1]
    add x13, x13, #0x2
    blt label_9
    whilelt p9.h, x23, x2
    whilelt p8.h, x23, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    add x1, x1, #0x2
    KAI_ASM_INST(0xe0bf82c8)  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_10
    whilelt p8.h, x23, x2
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_block2_fp16_fp16)

    KAI_ASM_END
