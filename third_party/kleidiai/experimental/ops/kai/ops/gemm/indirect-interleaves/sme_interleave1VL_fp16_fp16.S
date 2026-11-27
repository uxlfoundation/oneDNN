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

    KAI_ASM_CODE(sme_interleave1VL_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_fp16_fp16)
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
    cnth x23
    ldr x22, [x0, #0]
    inch x6
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
    ptrue p11.h
    whilelt p10.h, XZR, x3
    ldr x5, [x8, #0x8]
    mov x4, x4
    whilelt p9.h, x24, x2
    whilelt p8.h, x24, x2
    add x8, x8, #0x10
    mov x12, #0
    cbz x20, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04400a1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x5, x4, LSL #1]
    add x12, x12, #0x2
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    cmp x12, x20
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    mov x3, x7
    mov x8, x1
    cmp x3, #0x2
    inch x24
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04400a1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    inch x4
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x24, x2
    mov x13, #0
    cbz x20, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25296d21)  // psel p1.h, p11.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25396d20)  // psel p0.h, p11.h/Z, p9.h[w13, #1]
    KAI_ASM_INST(0xe0442cc8)  // ld1h { za1h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04428a9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07fa6c0)  // st1h { za0v.h[x13] }, p1/Z, [x22, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe077a2c1)  // st1h { za0v.h[x13, #1] }, p0/Z, [x22, x23, LSL #1]
    add x13, x13, #0x2
    addvl x22, x22, #2
    cmp x13, x20
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25296d21)  // psel p1.h, p11.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25396d20)  // psel p0.h, p11.h/Z, p9.h[w13, #1]
    mov x8, x1
    whilelt p9.h, x24, x2
    KAI_ASM_INST(0xe0442cc8)  // ld1h { za1h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    inch x24
    mov x12, #0
    KAI_ASM_INST(0xe04428a9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07fa6c0)  // st1h { za0v.h[x13] }, p1/Z, [x22, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe077a2c1)  // st1h { za0v.h[x13, #1] }, p0/Z, [x22, x23, LSL #1]
    addvl x22, x22, #2
    inch x4
    whilelt p8.h, x24, x2
    cbz x20, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0440cc0)  // ld1h { za0h.h[x12] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04408a1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07f86c8)  // st1h { za1v.h[x12] }, p1/Z, [x22, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe07782c9)  // st1h { za1v.h[x12, #1] }, p0/Z, [x22, x23, LSL #1]
    add x12, x12, #0x2
    addvl x22, x22, #2
    cmp x12, x20
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    mov x8, x1
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe0440cc0)  // ld1h { za0h.h[x12] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    whilelt p9.h, x24, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe04408a1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07f86c8)  // st1h { za1v.h[x12] }, p1/Z, [x22, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe07782c9)  // st1h { za1v.h[x12, #1] }, p0/Z, [x22, x23, LSL #1]
    addvl x22, x22, #2
    inch x24
    inch x4
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x8, x1
    whilelt p8.h, x24, x2
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25286140)  // psel p0.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0xe07f86c0)  // st1h { za0v.h[x12] }, p1/Z, [x22, XZR, LSL #1]
    addvl x22, x22, #1
    ldr x1, [x8, #0]
    add x8, x8, #0x8
    KAI_ASM_INST(0xe0440028)  // ld1h { za1h.h[x12] }, p0/Z, [x1, x4, LSL #1]
    add x12, x12, #0x1
    cmp x12, x23
    blt label_9
    whilelt p9.h, x24, x2
    whilelt p8.h, x24, x2
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25286d20)  // psel p0.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f82c8)  // st1h { za1v.h[x12] }, p0/Z, [x22, XZR, LSL #1]
    add x12, x12, #0x1
    addvl x22, x22, #1
    cmp x12, x21
    blt label_10
    whilelt p8.h, x24, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25286d20)  // psel p0.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f82c0)  // st1h { za0v.h[x12] }, p0/Z, [x22, XZR, LSL #1]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_fp16_fp16)

    KAI_ASM_END
