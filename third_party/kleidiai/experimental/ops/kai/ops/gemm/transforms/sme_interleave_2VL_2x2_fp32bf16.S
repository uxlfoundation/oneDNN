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

    KAI_ASM_CODE(sme_interleave_2VL_2x2_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave_2VL_2x2_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave_2VL_2x2_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave_2VL_2x2_fp32bf16)
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
    cnth x27
    mov x6, x2
    mov x9, x2
    sub x11, x27, #0x1
    cntw x28
    inch x6
    ands x11, x9, x11
    sub x6, x6, #0x1
    csel x11, x11, x27, NE
    lsl x14, x28, #0x1
    udiv x6, x6, x27  // n_passes = ceildiv(width, VL<T>)
    add x11, x11, #0x1
    mul x16, x14, x5
    ptrue p7.b
    sub x15, x6, #0x1
    lsr x11, x11, #0x1
    sub x17, x28, #0x2
    ptrue p6.s
    add x24, x16, x28
    sub SP, SP, x14, LSL #3
KAI_ASM_LABEL(label_1)  // Height loop
    mov x9, x5
    mov x10, x0
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x3, #0
    mov x0, x10
    addvl x10, x10, #2
    bgt label_5
    mov z16.b, #0
    mov x3, x15
KAI_ASM_LABEL(label_3)  // Pad empty blocks: Depth loop
    add x2, x0, x16, LSL #2
    mov x1, x9
KAI_ASM_LABEL(label_4)  // Pad empty blocks: Block loop
    subs x1, x1, #0x1
    st1w { z16.s }, p6, [x0]
    st1w { z16.s }, p6, [x0, #1, MUL VL]
    addvl x0, x0, #2
    bgt label_4
    subs x3, x3, #0x1
    mov x0, x2
    bge label_3
    b label_20
KAI_ASM_LABEL(label_5)  // Valid rows
    mov x7, SP
    mov x6, x14
KAI_ASM_LABEL(label_6)  // Pointer setup loop
    subs x6, x6, #0x1
    str x1, [x7]
    add x1, x1, x4
    add x7, x7, #0x8
    bne label_6
    mov x26, #0
    lsl x8, x3, #0x1  // height * 2
    lsl x7, x28, #0x1
    mov x27, #0
    mov x12, x26
    whilelt p12.h, XZR, x8
    whilelt p11.h, x7, x8
    mov x8, x27
    whilelt p10.h, x26, x2
    whilelt p9.s, x12, x2
    incw x12
    mov x25, SP
    add x7, SP, x28, LSL #3
    incw x8
    ldr x6, [x25, #0]
    whilelt p8.s, x12, x2
    ldr x21, [x7, #0]
    mov x12, #0
    ldr x23, [x25, #0x8]
    add x25, x25, #0x10
    ldr x22, [x7, #0x8]
    add x7, x7, #0x10
    cbz x17, label_8
KAI_ASM_LABEL(label_7)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286583)  // psel p3.h, p9.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286182)  // psel p2.h, p8.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286561)  // psel p1.h, p9.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286160)  // psel p0.h, p8.h/Z, p11.h[w12]
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25686583)  // psel p3.h, p9.h/Z, p12.h[w12, #2]
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25686182)  // psel p2.h, p8.h/Z, p12.h[w12, #2]
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25686561)  // psel p1.h, p9.h/Z, p11.h[w12, #2]
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25686160)  // psel p0.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    ldr x23, [x25, #0x8]
    add x25, x25, #0x10
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    add x7, x7, #0x10
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0401ee0)  // mova za0h.h[x12], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401ea8)  // mova za1h.h[x12], p7/M, z21.h
    KAI_ASM_INST(0xc0401e62)  // mova za0h.h[x12, #2], p7/M, z19.h
    KAI_ASM_INST(0xc0401e2a)  // mova za1h.h[x12, #2], p7/M, z17.h
    add x12, x12, #0x4
    cmp x12, x17, LSL #1
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Charge: End
    KAI_ASM_INST(0x25286583)  // psel p3.h, p9.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286182)  // psel p2.h, p8.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286561)  // psel p1.h, p9.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286160)  // psel p0.h, p8.h/Z, p11.h[w12]
    mov x20, x15
    mov x25, SP
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25686583)  // psel p3.h, p9.h/Z, p12.h[w12, #2]
    add x7, SP, x28, LSL #3
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25686182)  // psel p2.h, p8.h/Z, p12.h[w12, #2]
    cmp x20, #0x2
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25686561)  // psel p1.h, p9.h/Z, p11.h[w12, #2]
    inch x26
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25686160)  // psel p0.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    ldr x23, [x25, #0x8]
    add x25, x25, #0x10
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    inch x27
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    add x7, x7, #0x10
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    inch x8
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0401ee0)  // mova za0h.h[x12], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401ea8)  // mova za1h.h[x12], p7/M, z21.h
    KAI_ASM_INST(0xc0401e62)  // mova za0h.h[x12, #2], p7/M, z19.h
    KAI_ASM_INST(0xc0401e2a)  // mova za1h.h[x12, #2], p7/M, z17.h
    blt label_14
KAI_ASM_LABEL(label_9)  // K loop: Main loop
    mov x12, x26
    mov x13, #0
    whilelt p9.s, x12, x2
    incw x12
    whilelt p8.s, x12, x2
    mov x12, #0
    cbz x17, label_11
KAI_ASM_LABEL(label_10)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25396583)  // psel p3.h, p9.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396182)  // psel p2.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396561)  // psel p1.h, p9.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25796583)  // psel p3.h, p9.h/Z, p12.h[w13, #3]
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25796182)  // psel p2.h, p8.h/Z, p12.h[w13, #3]
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25796561)  // psel p1.h, p9.h/Z, p11.h[w13, #3]
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25796160)  // psel p0.h, p8.h/Z, p11.h[w13, #3]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    KAI_ASM_INST(0x25285943)  // psel p3.h, p6.h/Z, p10.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    KAI_ASM_INST(0x25285942)  // psel p2.h, p6.h/Z, p10.h[w12]
    ldr x23, [x25, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    KAI_ASM_INST(0x25385941)  // psel p1.h, p6.h/Z, p10.h[w12, #1]
    add x25, x25, #0x10
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    KAI_ASM_INST(0x25385940)  // psel p0.h, p6.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x7, x7, #0x10
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ee1)  // mova za0h.h[x13, #1], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403ea9)  // mova za1h.h[x13, #1], p7/M, z21.h
    KAI_ASM_INST(0xc0403e63)  // mova za0h.h[x13, #3], p7/M, z19.h
    KAI_ASM_INST(0xe0bf8c00)  // st1w { za0v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xc0403e2b)  // mova za1h.h[x13, #3], p7/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bc8804)  // st1w { za1v.s[x12] }, p2/Z, [x0, x28, LSL #2]
    KAI_ASM_INST(0xe0b08401)  // st1w { za0v.s[x12, #1] }, p1/Z, [x0, x16, LSL #2]
    KAI_ASM_INST(0xe0b88005)  // st1w { za1v.s[x12, #1] }, p0/Z, [x0, x24, LSL #2]
    add x12, x12, #0x2
    add x0, x0, x16, LSL #3
    cmp x12, x17
    blt label_10
KAI_ASM_LABEL(label_11)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25396583)  // psel p3.h, p9.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396182)  // psel p2.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396561)  // psel p1.h, p9.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    mov x25, SP
    add x7, SP, x28, LSL #3
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25796583)  // psel p3.h, p9.h/Z, p12.h[w13, #3]
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25796182)  // psel p2.h, p8.h/Z, p12.h[w13, #3]
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25796561)  // psel p1.h, p9.h/Z, p11.h[w13, #3]
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25796160)  // psel p0.h, p8.h/Z, p11.h[w13, #3]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    KAI_ASM_INST(0x25285943)  // psel p3.h, p6.h/Z, p10.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    KAI_ASM_INST(0x25285942)  // psel p2.h, p6.h/Z, p10.h[w12]
    ldr x23, [x25, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    KAI_ASM_INST(0x25385941)  // psel p1.h, p6.h/Z, p10.h[w12, #1]
    add x25, x25, #0x10
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    KAI_ASM_INST(0x25385940)  // psel p0.h, p6.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p10.h, x26, x2
    inch x26
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    add x7, x7, #0x10
    inch x27
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    inch x8
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ee1)  // mova za0h.h[x13, #1], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403ea9)  // mova za1h.h[x13, #1], p7/M, z21.h
    KAI_ASM_INST(0xc0403e63)  // mova za0h.h[x13, #3], p7/M, z19.h
    KAI_ASM_INST(0xe0bf8c00)  // st1w { za0v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xc0403e2b)  // mova za1h.h[x13, #3], p7/M, z17.h
    mov x13, x26
    KAI_ASM_INST(0xe0bc8804)  // st1w { za1v.s[x12] }, p2/Z, [x0, x28, LSL #2]
    whilelt p9.s, x13, x2
    incw x13
    KAI_ASM_INST(0xe0b08401)  // st1w { za0v.s[x12, #1] }, p1/Z, [x0, x16, LSL #2]
    whilelt p8.s, x13, x2
    mov x13, #0
    KAI_ASM_INST(0xe0b88005)  // st1w { za1v.s[x12, #1] }, p0/Z, [x0, x24, LSL #2]
    add x0, x0, x16, LSL #3
    mov x12, #0
    cbz x17, label_13
KAI_ASM_LABEL(label_12)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296583)  // psel p3.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296182)  // psel p2.h, p8.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296561)  // psel p1.h, p9.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25696583)  // psel p3.h, p9.h/Z, p12.h[w13, #2]
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25696182)  // psel p2.h, p8.h/Z, p12.h[w13, #2]
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25696561)  // psel p1.h, p9.h/Z, p11.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25696160)  // psel p0.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    KAI_ASM_INST(0x25285943)  // psel p3.h, p6.h/Z, p10.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    KAI_ASM_INST(0x25285942)  // psel p2.h, p6.h/Z, p10.h[w12]
    ldr x23, [x25, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    KAI_ASM_INST(0x25385941)  // psel p1.h, p6.h/Z, p10.h[w12, #1]
    add x25, x25, #0x10
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    KAI_ASM_INST(0x25385940)  // psel p0.h, p6.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x7, x7, #0x10
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ee0)  // mova za0h.h[x13], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403ea8)  // mova za1h.h[x13], p7/M, z21.h
    KAI_ASM_INST(0xc0403e62)  // mova za0h.h[x13, #2], p7/M, z19.h
    KAI_ASM_INST(0xe0bf8c08)  // st1w { za2v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xc0403e2a)  // mova za1h.h[x13, #2], p7/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bc880c)  // st1w { za3v.s[x12] }, p2/Z, [x0, x28, LSL #2]
    KAI_ASM_INST(0xe0b08409)  // st1w { za2v.s[x12, #1] }, p1/Z, [x0, x16, LSL #2]
    KAI_ASM_INST(0xe0b8800d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x0, x24, LSL #2]
    add x12, x12, #0x2
    add x0, x0, x16, LSL #3
    cmp x12, x17
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296583)  // psel p3.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296182)  // psel p2.h, p8.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296561)  // psel p1.h, p9.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    mov x25, SP
    add x7, SP, x28, LSL #3
    ld1w { z23.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x25696583)  // psel p3.h, p9.h/Z, p12.h[w13, #2]
    sub x20, x20, #0x2
    ld1w { z22.s }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x25696182)  // psel p2.h, p8.h/Z, p12.h[w13, #2]
    ldr x6, [x25, #0]
    ld1w { z21.s }, p1/Z, [x21, x27, LSL #2]
    KAI_ASM_INST(0x25696561)  // psel p1.h, p9.h/Z, p11.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x21, x8, LSL #2]
    KAI_ASM_INST(0x25696160)  // psel p0.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x658abef7)  // bfcvt z23.h, p7/M, z23.s
    ldr x21, [x7, #0]
    ld1w { z19.s }, p3/Z, [x23, x27, LSL #2]
    KAI_ASM_INST(0x658abed6)  // bfcvt z22.h, p7/M, z22.s
    KAI_ASM_INST(0x25285943)  // psel p3.h, p6.h/Z, p10.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x8, LSL #2]
    KAI_ASM_INST(0x658abeb5)  // bfcvt z21.h, p7/M, z21.s
    KAI_ASM_INST(0x25285942)  // psel p2.h, p6.h/Z, p10.h[w12]
    ldr x23, [x25, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x658abe94)  // bfcvt z20.h, p7/M, z20.s
    KAI_ASM_INST(0x25385941)  // psel p1.h, p6.h/Z, p10.h[w12, #1]
    add x25, x25, #0x10
    ld1w { z16.s }, p0/Z, [x22, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    ldr x22, [x7, #0x8]
    KAI_ASM_INST(0x25385940)  // psel p0.h, p6.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p10.h, x26, x2
    cmp x20, #0x2
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    add x7, x7, #0x10
    inch x26
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    inch x27
    inch x8
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ee0)  // mova za0h.h[x13], p7/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403ea8)  // mova za1h.h[x13], p7/M, z21.h
    KAI_ASM_INST(0xc0403e62)  // mova za0h.h[x13, #2], p7/M, z19.h
    KAI_ASM_INST(0xe0bf8c08)  // st1w { za2v.s[x12] }, p3/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xc0403e2a)  // mova za1h.h[x13, #2], p7/M, z17.h
    KAI_ASM_INST(0xe0bc880c)  // st1w { za3v.s[x12] }, p2/Z, [x0, x28, LSL #2]
    KAI_ASM_INST(0xe0b08409)  // st1w { za2v.s[x12, #1] }, p1/Z, [x0, x16, LSL #2]
    KAI_ASM_INST(0xe0b8800d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x0, x24, LSL #2]
    add x0, x0, x16, LSL #3
    bge label_9
KAI_ASM_LABEL(label_14)  // K loop: Tails
    cbz x20, label_17
    mov x6, x26
    mov x25, SP
    whilelt p9.s, x6, x2
    incw x6
    whilelt p8.s, x6, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_15)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25305945)  // psel p5.s, p6.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25305944)  // psel p4.s, p6.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25396583)  // psel p3.h, p9.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396182)  // psel p2.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396561)  // psel p1.h, p9.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0xe0bf9400)  // st1w { za0v.s[x12] }, p5/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc9004)  // st1w { za1v.s[x12] }, p4/Z, [x0, x28, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x16, LSL #2
    ldr x7, [x25, #0]
    cmp x12, x28
    ldr x6, [x25, x28, LSL #0x3]
    add x25, x25, #0x8
    ld1w { z19.s }, p3/Z, [x7, x27, LSL #2]
    ld1w { z18.s }, p2/Z, [x7, x8, LSL #2]
    ld1w { z17.s }, p1/Z, [x6, x27, LSL #2]
    ld1w { z16.s }, p0/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0x658abe73)  // bfcvt z19.h, p7/M, z19.s
    KAI_ASM_INST(0x658abe52)  // bfcvt z18.h, p7/M, z18.s
    KAI_ASM_INST(0x658abe31)  // bfcvt z17.h, p7/M, z17.s
    KAI_ASM_INST(0x658abe10)  // bfcvt z16.h, p7/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403e61)  // mova za0h.h[x13, #1], p7/M, z19.h
    KAI_ASM_INST(0xc0403e29)  // mova za1h.h[x13, #1], p7/M, z17.h
    add x13, x13, #0x2
    blt label_15
    mov x6, x26
    whilelt p10.h, x26, x2
    whilelt p8.s, x6, x2
    incw x6
    whilelt p8.s, x6, x2
    mov x6, #0
    mov x12, #0
KAI_ASM_LABEL(label_16)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25305941)  // psel p1.s, p6.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25305940)  // psel p0.s, p6.s/Z, p10.s[w12]
    add x6, x6, #0x2
    KAI_ASM_INST(0xe0bf8408)  // st1w { za2v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc800c)  // st1w { za3v.s[x12] }, p0/Z, [x0, x28, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x16, LSL #2
    cmp x12, x11
    blt label_16
    whilelt p8.h, x26, x2
    b label_19
KAI_ASM_LABEL(label_17)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_18)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25305941)  // psel p1.s, p6.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25305940)  // psel p0.s, p6.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8400)  // st1w { za0v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8004)  // st1w { za1v.s[x12] }, p0/Z, [x0, x28, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x16, LSL #2
    cmp x12, x11
    blt label_18
KAI_ASM_LABEL(label_19)  // K loop: End
    subs x9, x9, #0x1
    sub x3, x3, x14
    bne label_2
    addvl x0, x0, #2
    cmp x3, #0
    sub x0, x0, x16, LSL #2
    bgt label_1
KAI_ASM_LABEL(label_20)  // Exit
    add SP, SP, x14, LSL #3
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave_2VL_2x2_fp32bf16)

    KAI_ASM_END
