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

    KAI_ASM_CODE(sme_interleave_1VL_2x1)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave_1VL_2x1)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave_1VL_2x1)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave_1VL_2x1)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x7, x2
    cnth x9
    inch x7
    mov x6, x2
    sub x7, x7, #0x1
    sub x28, x9, #0x1
    udiv x7, x7, x9  // n_passes = ceildiv(width, VL<T>)
    ands x28, x6, x28
    sub x27, x7, #0x1
    csel x28, x28, x9, NE
    sub x26, x9, #0x2
    ptrue p4.h
    mul x25, x9, x5
    sub SP, SP, x9, LSL #3
KAI_ASM_LABEL(label_1)  // Height loop
    mov x24, x5
    mov x23, x0
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x3, #0
    mov x0, x23
    addvl x23, x23, #1
    bgt label_5
    mov z16.b, #0
    mov x3, x27
KAI_ASM_LABEL(label_3)  // Pad empty blocks: Depth loop
    add x2, x0, x25, LSL #1
    mov x1, x24
KAI_ASM_LABEL(label_4)  // Pad empty blocks: Block loop
    subs x1, x1, #0x1
    st1h { z16.h }, p4, [x0]
    addvl x0, x0, #1
    bgt label_4
    subs x3, x3, #0x1
    mov x0, x2
    bge label_3
    b label_20
KAI_ASM_LABEL(label_5)  // Valid rows
    mov x7, SP
    mov x6, x9
KAI_ASM_LABEL(label_6)  // Pointer setup loop
    subs x6, x6, #0x1
    str x1, [x7]
    add x1, x1, x4
    add x7, x7, #0x8
    bne label_6
    mov x22, #0
    mov x21, SP
    whilelt p10.h, XZR, x3
    mov x20, #0
    ldr x8, [x21, #0]
    whilelt p9.h, x22, x2
    whilelt p8.h, x22, x2
    ldr x7, [x21, #0x8]
    add x21, x21, #0x10
    mov x12, #0
    cbz x26, label_8
KAI_ASM_LABEL(label_7)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0xe0540500)  // ld1h { za0h.h[x12] }, p1/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    KAI_ASM_INST(0xe05400e1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x7, x20, LSL #1]
    add x12, x12, #0x2
    ldr x7, [x21, #0x8]
    add x21, x21, #0x10
    cmp x12, x26
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Charge: End
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    mov x6, x27
    mov x21, SP
    cmp x6, #0x2
    inch x22
    KAI_ASM_INST(0xe0540500)  // ld1h { za0h.h[x12] }, p1/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    KAI_ASM_INST(0xe05400e1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x7, x20, LSL #1]
    ldr x7, [x21, #0x8]
    add x21, x21, #0x10
    inch x20
    blt label_14
KAI_ASM_LABEL(label_9)  // K loop: Main loop
    whilelt p8.h, x22, x2
    mov x13, #0
    cbz x26, label_11
KAI_ASM_LABEL(label_10)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25295121)  // psel p1.h, p4.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25395120)  // psel p0.h, p4.h/Z, p9.h[w13, #1]
    KAI_ASM_INST(0xe0542d08)  // ld1h { za1h.h[x13] }, p3/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    KAI_ASM_INST(0xe05428e9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x7, x20, LSL #1]
    ldr x7, [x21, #0x8]
    KAI_ASM_INST(0xe07fa400)  // st1h { za0v.h[x13] }, p1/Z, [x0, XZR, LSL #1]
    add x21, x21, #0x10
    KAI_ASM_INST(0xe079a001)  // st1h { za0v.h[x13, #1] }, p0/Z, [x0, x25, LSL #1]
    add x13, x13, #0x2
    add x0, x0, x25, LSL #2
    cmp x13, x26
    blt label_10
KAI_ASM_LABEL(label_11)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25295121)  // psel p1.h, p4.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25395120)  // psel p0.h, p4.h/Z, p9.h[w13, #1]
    mov x21, SP
    whilelt p9.h, x22, x2
    KAI_ASM_INST(0xe0542d08)  // ld1h { za1h.h[x13] }, p3/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    inch x22
    mov x12, #0
    KAI_ASM_INST(0xe05428e9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x7, x20, LSL #1]
    ldr x7, [x21, #0x8]
    KAI_ASM_INST(0xe07fa400)  // st1h { za0v.h[x13] }, p1/Z, [x0, XZR, LSL #1]
    add x21, x21, #0x10
    KAI_ASM_INST(0xe079a001)  // st1h { za0v.h[x13, #1] }, p0/Z, [x0, x25, LSL #1]
    add x0, x0, x25, LSL #2
    inch x20
    whilelt p8.h, x22, x2
    cbz x26, label_13
KAI_ASM_LABEL(label_12)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25285121)  // psel p1.h, p4.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25385120)  // psel p0.h, p4.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0540d00)  // ld1h { za0h.h[x12] }, p3/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    KAI_ASM_INST(0xe05408e1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x7, x20, LSL #1]
    ldr x7, [x21, #0x8]
    KAI_ASM_INST(0xe07f8408)  // st1h { za1v.h[x12] }, p1/Z, [x0, XZR, LSL #1]
    add x21, x21, #0x10
    KAI_ASM_INST(0xe0798009)  // st1h { za1v.h[x12, #1] }, p0/Z, [x0, x25, LSL #1]
    add x12, x12, #0x2
    add x0, x0, x25, LSL #2
    cmp x12, x26
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25285121)  // psel p1.h, p4.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25385120)  // psel p0.h, p4.h/Z, p9.h[w12, #1]
    mov x21, SP
    sub x6, x6, #0x2
    KAI_ASM_INST(0xe0540d00)  // ld1h { za0h.h[x12] }, p3/Z, [x8, x20, LSL #1]
    ldr x8, [x21, #0]
    whilelt p9.h, x22, x2
    cmp x6, #0x2
    KAI_ASM_INST(0xe05408e1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x7, x20, LSL #1]
    ldr x7, [x21, #0x8]
    KAI_ASM_INST(0xe07f8408)  // st1h { za1v.h[x12] }, p1/Z, [x0, XZR, LSL #1]
    add x21, x21, #0x10
    KAI_ASM_INST(0xe0798009)  // st1h { za1v.h[x12, #1] }, p0/Z, [x0, x25, LSL #1]
    add x0, x0, x25, LSL #2
    inch x22
    inch x20
    bge label_9
KAI_ASM_LABEL(label_14)  // K loop: Tails
    cbz x6, label_17
    mov x21, SP
    whilelt p8.h, x22, x2
    mov x12, #0
KAI_ASM_LABEL(label_15)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25285121)  // psel p1.h, p4.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25286140)  // psel p0.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0xe07f8400)  // st1h { za0v.h[x12] }, p1/Z, [x0, XZR, LSL #1]
    add x0, x0, x25, LSL #1
    ldr x6, [x21, #0]
    add x21, x21, #0x8
    KAI_ASM_INST(0xe05400c8)  // ld1h { za1h.h[x12] }, p0/Z, [x6, x20, LSL #1]
    add x12, x12, #0x1
    cmp x12, x9
    blt label_15
    whilelt p9.h, x22, x2
    whilelt p8.h, x22, x2
    mov x12, #0
KAI_ASM_LABEL(label_16)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25285120)  // psel p0.h, p4.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f8008)  // st1h { za1v.h[x12] }, p0/Z, [x0, XZR, LSL #1]
    add x12, x12, #0x1
    add x0, x0, x25, LSL #1
    cmp x12, x28
    blt label_16
    whilelt p8.h, x22, x2
    b label_19
KAI_ASM_LABEL(label_17)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_18)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25285120)  // psel p0.h, p4.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f8000)  // st1h { za0v.h[x12] }, p0/Z, [x0, XZR, LSL #1]
    add x12, x12, #0x1
    add x0, x0, x25, LSL #1
    cmp x12, x28
    blt label_18
KAI_ASM_LABEL(label_19)  // K loop: End
    subs x24, x24, #0x1
    sub x3, x3, x9
    bne label_2
    addvl x0, x0, #1
    cmp x3, #0
    sub x0, x0, x25, LSL #1
    bgt label_1
KAI_ASM_LABEL(label_20)  // Exit
    add SP, SP, x9, LSL #3
    KAI_ASM_INST(0xd503467f)  // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave_1VL_2x1)

    KAI_ASM_END
