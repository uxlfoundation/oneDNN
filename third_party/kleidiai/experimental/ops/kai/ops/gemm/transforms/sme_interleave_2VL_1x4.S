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

    KAI_ASM_CODE(sme_interleave_2VL_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave_2VL_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave_2VL_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave_2VL_1x4)
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
    cntb x8
    mov x7, x2
    mov x6, x2
    sub x17, x8, #0x1
    cntw x16
    incb x7
    ands x17, x6, x17
    sub x7, x7, #0x1
    csel x17, x17, x8, NE
    lsl x15, x16, #0x1
    udiv x7, x7, x8  // n_passes = ceildiv(width, VL<T>)
    add x17, x17, #0x3
    mul x11, x15, x5
    sub x10, x7, #0x1
    lsr x17, x17, #0x2
    sub x9, x16, #0x2
    ptrue p4.s
    add x28, x11, x16
    sub SP, SP, x15, LSL #3
KAI_ASM_LABEL(label_1)  // Height loop
    mov x27, x5
    mov x26, x0
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x3, #0
    mov x0, x26
    addvl x26, x26, #2
    bgt label_5
    mov z16.b, #0
    mov x3, x10
KAI_ASM_LABEL(label_3)  // Pad empty blocks: Depth loop
    add x2, x0, x11, LSL #2
    mov x1, x27
KAI_ASM_LABEL(label_4)  // Pad empty blocks: Block loop
    subs x1, x1, #0x1
    st1w { z16.s }, p4, [x0]
    st1w { z16.s }, p4, [x0, #1, MUL VL]
    addvl x0, x0, #2
    bgt label_4
    subs x3, x3, #0x1
    mov x0, x2
    bge label_3
    b label_20
KAI_ASM_LABEL(label_5)  // Valid rows
    mov x7, SP
    mov x6, x15
KAI_ASM_LABEL(label_6)  // Pointer setup loop
    subs x6, x6, #0x1
    str x1, [x7]
    add x1, x1, x4
    add x7, x7, #0x8
    bne label_6
    lsl x7, x3, #0x1  // height * 2
    lsl x6, x16, #0x1
    whilelt p9.b, XZR, x7
    whilelt p8.b, x6, x7
    mov x25, #0
    mov x24, SP
    add x23, SP, x16, LSL #3
    zip1 p10.b, p9.b, p8.b
    ldr x22, [x24, #0]
    mov x21, #0
    whilelt p9.b, x25, x2
    ldr x20, [x23, #0]
    whilelt p8.b, x25, x2
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    ldr x7, [x23, #0x8]
    add x23, x23, #0x10
    mov x12, #0
    cbz x9, label_8
KAI_ASM_LABEL(label_7)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    KAI_ASM_INST(0xe0150ec0)  // ld1b { za0h.b[x12] }, p3/Z, [x22, x21]
    ldr x22, [x24, #0]
    KAI_ASM_INST(0xe0150a81)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x20, x21]
    ldr x20, [x23, #0]
    KAI_ASM_INST(0xe0150504)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x8, x21]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01500e5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x7, x21]
    add x12, x12, #0x8
    ldr x7, [x23, #0x8]
    add x23, x23, #0x10
    cmp x12, x9, LSL #2
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Charge: End
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    mov x6, x10
    mov x24, SP
    KAI_ASM_INST(0xe0150ec0)  // ld1b { za0h.b[x12] }, p3/Z, [x22, x21]
    add x23, SP, x16, LSL #3
    cmp x6, #0x2
    ldr x22, [x24, #0]
    KAI_ASM_INST(0xe0150a81)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x20, x21]
    ldr x20, [x23, #0]
    incb x25
    KAI_ASM_INST(0xe0150504)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x8, x21]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01500e5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x7, x21]
    ldr x7, [x23, #0x8]
    add x23, x23, #0x10
    incb x21
    blt label_14
KAI_ASM_LABEL(label_9)  // K loop: Main loop
    whilelt p8.b, x25, x2
    mov x12, #0
    mov x14, #0
    cbz x9, label_11
KAI_ASM_LABEL(label_10)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x253c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #3]
    KAI_ASM_INST(0x25746141)  // psel p1.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x257c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #7]
    KAI_ASM_INST(0xe0150ec2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x22, x21]
    KAI_ASM_INST(0x25265123)  // psel p3.b, p4.b/Z, p9.b[w14]
    ldr x22, [x24, #0]
    KAI_ASM_INST(0xe0150a83)  // ld1b { za0h.b[x12, #3] }, p2/Z, [x20, x21]
    KAI_ASM_INST(0x25265122)  // psel p2.b, p4.b/Z, p9.b[w14]
    ldr x20, [x23, #0]
    KAI_ASM_INST(0xe0150506)  // ld1b { za0h.b[x12, #6] }, p1/Z, [x8, x21]
    KAI_ASM_INST(0x252e5121)  // psel p1.b, p4.b/Z, p9.b[w14, #1]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01500e7)  // ld1b { za0h.b[x12, #7] }, p0/Z, [x7, x21]
    ldr x7, [x23, #0x8]
    KAI_ASM_INST(0x252e5120)  // psel p0.b, p4.b/Z, p9.b[w14, #1]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0bfcc00)  // st1w { za0v.s[x14] }, p3/Z, [x0, XZR, LSL #2]
    add x12, x12, #0x8
    KAI_ASM_INST(0xe0b0c804)  // st1w { za1v.s[x14] }, p2/Z, [x0, x16, LSL #2]
    KAI_ASM_INST(0xe0abc401)  // st1w { za0v.s[x14, #1] }, p1/Z, [x0, x11, LSL #2]
    KAI_ASM_INST(0xe0bcc005)  // st1w { za1v.s[x14, #1] }, p0/Z, [x0, x28, LSL #2]
    add x14, x14, #0x2
    add x0, x0, x11, LSL #3
    cmp x14, x9
    blt label_10
KAI_ASM_LABEL(label_11)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x253c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #3]
    KAI_ASM_INST(0x25746141)  // psel p1.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x257c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #7]
    mov x24, SP
    add x23, SP, x16, LSL #3
    KAI_ASM_INST(0xe0150ec2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x22, x21]
    KAI_ASM_INST(0x25265123)  // psel p3.b, p4.b/Z, p9.b[w14]
    ldr x22, [x24, #0]
    mov x13, #0
    KAI_ASM_INST(0xe0150a83)  // ld1b { za0h.b[x12, #3] }, p2/Z, [x20, x21]
    KAI_ASM_INST(0x25265122)  // psel p2.b, p4.b/Z, p9.b[w14]
    ldr x20, [x23, #0]
    KAI_ASM_INST(0xe0150506)  // ld1b { za0h.b[x12, #6] }, p1/Z, [x8, x21]
    KAI_ASM_INST(0x252e5121)  // psel p1.b, p4.b/Z, p9.b[w14, #1]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01500e7)  // ld1b { za0h.b[x12, #7] }, p0/Z, [x7, x21]
    ldr x7, [x23, #0x8]
    KAI_ASM_INST(0x252e5120)  // psel p0.b, p4.b/Z, p9.b[w14, #1]
    whilelt p9.b, x25, x2
    KAI_ASM_INST(0xe0bfcc00)  // st1w { za0v.s[x14] }, p3/Z, [x0, XZR, LSL #2]
    incb x25
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0b0c804)  // st1w { za1v.s[x14] }, p2/Z, [x0, x16, LSL #2]
    incb x21
    whilelt p8.b, x25, x2
    KAI_ASM_INST(0xe0abc401)  // st1w { za0v.s[x14, #1] }, p1/Z, [x0, x11, LSL #2]
    mov x12, #0
    KAI_ASM_INST(0xe0bcc005)  // st1w { za1v.s[x14, #1] }, p0/Z, [x0, x28, LSL #2]
    add x0, x0, x11, LSL #3
    cbz x9, label_13
KAI_ASM_LABEL(label_12)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    KAI_ASM_INST(0xe0152ec0)  // ld1b { za0h.b[x13] }, p3/Z, [x22, x21]
    KAI_ASM_INST(0x25245123)  // psel p3.b, p4.b/Z, p9.b[w12]
    ldr x22, [x24, #0]
    KAI_ASM_INST(0xe0152a81)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x20, x21]
    KAI_ASM_INST(0x25245122)  // psel p2.b, p4.b/Z, p9.b[w12]
    ldr x20, [x23, #0]
    KAI_ASM_INST(0xe0152504)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x8, x21]
    KAI_ASM_INST(0x252c5121)  // psel p1.b, p4.b/Z, p9.b[w12, #1]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01520e5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x7, x21]
    ldr x7, [x23, #0x8]
    KAI_ASM_INST(0x252c5120)  // psel p0.b, p4.b/Z, p9.b[w12, #1]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0bf8c08)  // st1w { za2v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    add x13, x13, #0x8
    KAI_ASM_INST(0xe0b0880c)  // st1w { za3v.s[x12] }, p2/Z, [x0, x16, LSL #2]
    KAI_ASM_INST(0xe0ab8409)  // st1w { za2v.s[x12, #1] }, p1/Z, [x0, x11, LSL #2]
    KAI_ASM_INST(0xe0bc800d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x0, x28, LSL #2]
    add x12, x12, #0x2
    add x0, x0, x11, LSL #3
    cmp x12, x9
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    mov x24, SP
    add x23, SP, x16, LSL #3
    KAI_ASM_INST(0xe0152ec0)  // ld1b { za0h.b[x13] }, p3/Z, [x22, x21]
    KAI_ASM_INST(0x25245123)  // psel p3.b, p4.b/Z, p9.b[w12]
    ldr x22, [x24, #0]
    sub x6, x6, #0x2
    KAI_ASM_INST(0xe0152a81)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x20, x21]
    KAI_ASM_INST(0x25245122)  // psel p2.b, p4.b/Z, p9.b[w12]
    ldr x20, [x23, #0]
    KAI_ASM_INST(0xe0152504)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x8, x21]
    KAI_ASM_INST(0x252c5121)  // psel p1.b, p4.b/Z, p9.b[w12, #1]
    ldr x8, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe01520e5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x7, x21]
    ldr x7, [x23, #0x8]
    KAI_ASM_INST(0x252c5120)  // psel p0.b, p4.b/Z, p9.b[w12, #1]
    whilelt p9.b, x25, x2
    KAI_ASM_INST(0xe0bf8c08)  // st1w { za2v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    cmp x6, #0x2
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0b0880c)  // st1w { za3v.s[x12] }, p2/Z, [x0, x16, LSL #2]
    incb x25
    incb x21
    KAI_ASM_INST(0xe0ab8409)  // st1w { za2v.s[x12, #1] }, p1/Z, [x0, x11, LSL #2]
    KAI_ASM_INST(0xe0bc800d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x0, x28, LSL #2]
    add x0, x0, x11, LSL #3
    bge label_9
KAI_ASM_LABEL(label_14)  // K loop: Tails
    cbz x6, label_17
    mov x24, SP
    whilelt p8.b, x25, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_15)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25305123)  // psel p3.s, p4.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25305122)  // psel p2.s, p4.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356141)  // psel p1.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0x253d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #3]
    KAI_ASM_INST(0xe0bf8c00)  // st1w { za0v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08804)  // st1w { za1v.s[x12] }, p2/Z, [x0, x16, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x11, LSL #2
    ldr x7, [x24, #0]
    cmp x12, x16
    ldr x6, [x24, x16, LSL #0x3]
    add x24, x24, #0x8
    KAI_ASM_INST(0xe01524e2)  // ld1b { za0h.b[x13, #2] }, p1/Z, [x7, x21]
    KAI_ASM_INST(0xe01520c3)  // ld1b { za0h.b[x13, #3] }, p0/Z, [x6, x21]
    add x13, x13, #0x4
    blt label_15
    whilelt p9.b, x25, x2
    whilelt p8.b, x25, x2
    mov x6, #0
    mov x12, #0
KAI_ASM_LABEL(label_16)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25305121)  // psel p1.s, p4.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25305120)  // psel p0.s, p4.s/Z, p9.s[w12]
    add x6, x6, #0x4
    KAI_ASM_INST(0xe0bf8408)  // st1w { za2v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0b0800c)  // st1w { za3v.s[x12] }, p0/Z, [x0, x16, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x11, LSL #2
    cmp x12, x17
    blt label_16
    whilelt p8.b, x25, x2
    b label_19
KAI_ASM_LABEL(label_17)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_18)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25305121)  // psel p1.s, p4.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25305120)  // psel p0.s, p4.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8400)  // st1w { za0v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08004)  // st1w { za1v.s[x12] }, p0/Z, [x0, x16, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x11, LSL #2
    cmp x12, x17
    blt label_18
KAI_ASM_LABEL(label_19)  // K loop: End
    subs x27, x27, #0x1
    sub x3, x3, x15
    bne label_2
    addvl x0, x0, #2
    cmp x3, #0
    sub x0, x0, x11, LSL #2
    bgt label_1
KAI_ASM_LABEL(label_20)  // Exit
    add SP, SP, x15, LSL #3
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave_2VL_1x4)

    KAI_ASM_END
