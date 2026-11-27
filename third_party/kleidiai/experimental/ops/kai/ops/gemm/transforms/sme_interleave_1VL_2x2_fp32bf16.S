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

    KAI_ASM_CODE(sme_interleave_1VL_2x2_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave_1VL_2x2_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave_1VL_2x2_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave_1VL_2x2_fp32bf16)
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
    cnth x8
    mov x7, x2
    mov x6, x2
    sub x11, x8, #0x1
    inch x7
    ands x11, x6, x11
    sub x7, x7, #0x1
    csel x11, x11, x8, NE
    cntw x10
    udiv x7, x7, x8  // n_passes = ceildiv(width, VL<T>)
    add x11, x11, #0x1
    ptrue p5.b
    sub x9, x7, #0x1
    lsr x11, x11, #0x1
    sub x28, x10, #0x2
    ptrue p4.s
    mul x27, x10, x5
    sub SP, SP, x10, LSL #3
KAI_ASM_LABEL(label_1)  // Height loop
    mov x26, x5
    mov x25, x0
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x3, #0
    mov x0, x25
    addvl x25, x25, #1
    bgt label_5
    mov z16.b, #0
    mov x3, x9
KAI_ASM_LABEL(label_3)  // Pad empty blocks: Depth loop
    add x2, x0, x27, LSL #2
    mov x1, x26
KAI_ASM_LABEL(label_4)  // Pad empty blocks: Block loop
    subs x1, x1, #0x1
    st1w { z16.s }, p4, [x0]
    addvl x0, x0, #1
    bgt label_4
    subs x3, x3, #0x1
    mov x0, x2
    bge label_3
    b label_20
KAI_ASM_LABEL(label_5)  // Valid rows
    mov x7, SP
    mov x6, x10
KAI_ASM_LABEL(label_6)  // Pointer setup loop
    subs x6, x6, #0x1
    str x1, [x7]
    add x1, x1, x4
    add x7, x7, #0x8
    bne label_6
    mov x24, #0
    lsl x7, x3, #0x1  // height * 2
    mov x23, #0
    mov x6, x24
    whilelt p11.h, XZR, x7
    mov x22, x23
    whilelt p10.h, x24, x2
    whilelt p9.s, x6, x2
    incw x6
    mov x21, SP
    incw x22
    whilelt p8.s, x6, x2
    ldr x20, [x21, #0]
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    mov x12, #0
    cbz x28, label_8
KAI_ASM_LABEL(label_7)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286162)  // psel p2.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25686561)  // psel p1.h, p9.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686160)  // psel p0.h, p8.h/Z, p11.h[w12, #2]
    ld1w { z19.s }, p3/Z, [x20, x23, LSL #2]
    ld1w { z18.s }, p2/Z, [x20, x22, LSL #2]
    ldr x20, [x21, #0]
    ld1w { z17.s }, p1/Z, [x8, x23, LSL #2]
    ld1w { z16.s }, p0/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401660)  // mova za0h.h[x12], p5/M, z19.h
    KAI_ASM_INST(0xc0401622)  // mova za0h.h[x12, #2], p5/M, z17.h
    add x12, x12, #0x4
    cmp x12, x28, LSL #1
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Charge: End
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286162)  // psel p2.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25686561)  // psel p1.h, p9.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686160)  // psel p0.h, p8.h/Z, p11.h[w12, #2]
    mov x7, x9
    mov x21, SP
    ld1w { z19.s }, p3/Z, [x20, x23, LSL #2]
    cmp x7, #0x2
    inch x24
    ld1w { z18.s }, p2/Z, [x20, x22, LSL #2]
    ldr x20, [x21, #0]
    ld1w { z17.s }, p1/Z, [x8, x23, LSL #2]
    inch x23
    ld1w { z16.s }, p0/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    inch x22
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401660)  // mova za0h.h[x12], p5/M, z19.h
    KAI_ASM_INST(0xc0401622)  // mova za0h.h[x12, #2], p5/M, z17.h
    blt label_14
KAI_ASM_LABEL(label_9)  // K loop: Main loop
    mov x6, x24
    mov x15, #0
    whilelt p9.s, x6, x2
    incw x6
    whilelt p8.s, x6, x2
    mov x14, #0
    cbz x28, label_11
KAI_ASM_LABEL(label_10)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b6561)  // psel p1.h, p9.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x257b6563)  // psel p3.h, p9.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6162)  // psel p2.h, p8.h/Z, p11.h[w15, #3]
    ld1w { z19.s }, p1/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0x252a5141)  // psel p1.h, p4.h/Z, p10.h[w14]
    ld1w { z18.s }, p0/Z, [x20, x22, LSL #2]
    KAI_ASM_INST(0x253a5140)  // psel p0.h, p4.h/Z, p10.h[w14, #1]
    ldr x20, [x21, #0]
    ld1w { z17.s }, p3/Z, [x8, x23, LSL #2]
    ld1w { z16.s }, p2/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407661)  // mova za0h.h[x15, #1], p5/M, z19.h
    KAI_ASM_INST(0xc0407623)  // mova za0h.h[x15, #3], p5/M, z17.h
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0bfc400)  // st1w { za0v.s[x14] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bbc001)  // st1w { za0v.s[x14, #1] }, p0/Z, [x0, x27, LSL #2]
    add x14, x14, #0x2
    add x0, x0, x27, LSL #3
    cmp x14, x28
    blt label_10
KAI_ASM_LABEL(label_11)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b6561)  // psel p1.h, p9.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x257b6563)  // psel p3.h, p9.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6162)  // psel p2.h, p8.h/Z, p11.h[w15, #3]
    mov x21, SP
    mov x13, #0
    ld1w { z19.s }, p1/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0x252a5141)  // psel p1.h, p4.h/Z, p10.h[w14]
    mov x12, #0
    ld1w { z18.s }, p0/Z, [x20, x22, LSL #2]
    KAI_ASM_INST(0x253a5140)  // psel p0.h, p4.h/Z, p10.h[w14, #1]
    whilelt p10.h, x24, x2
    ldr x20, [x21, #0]
    ld1w { z17.s }, p3/Z, [x8, x23, LSL #2]
    inch x24
    inch x23
    ld1w { z16.s }, p2/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    mov x6, x24
    ldr x8, [x21, #0x8]
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    whilelt p9.s, x6, x2
    incw x6
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    add x21, x21, #0x10
    inch x22
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    whilelt p8.s, x6, x2
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407661)  // mova za0h.h[x15, #1], p5/M, z19.h
    KAI_ASM_INST(0xc0407623)  // mova za0h.h[x15, #3], p5/M, z17.h
    KAI_ASM_INST(0xe0bfc400)  // st1w { za0v.s[x14] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bbc001)  // st1w { za0v.s[x14, #1] }, p0/Z, [x0, x27, LSL #2]
    add x0, x0, x27, LSL #3
    cbz x28, label_13
KAI_ASM_LABEL(label_12)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296561)  // psel p1.h, p9.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25696563)  // psel p3.h, p9.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696162)  // psel p2.h, p8.h/Z, p11.h[w13, #2]
    ld1w { z19.s }, p1/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0x25285141)  // psel p1.h, p4.h/Z, p10.h[w12]
    ld1w { z18.s }, p0/Z, [x20, x22, LSL #2]
    KAI_ASM_INST(0x25385140)  // psel p0.h, p4.h/Z, p10.h[w12, #1]
    ldr x20, [x21, #0]
    ld1w { z17.s }, p3/Z, [x8, x23, LSL #2]
    ld1w { z16.s }, p2/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403660)  // mova za0h.h[x13], p5/M, z19.h
    KAI_ASM_INST(0xc0403622)  // mova za0h.h[x13, #2], p5/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bf8408)  // st1w { za2v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bb8009)  // st1w { za2v.s[x12, #1] }, p0/Z, [x0, x27, LSL #2]
    add x12, x12, #0x2
    add x0, x0, x27, LSL #3
    cmp x12, x28
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296561)  // psel p1.h, p9.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25696563)  // psel p3.h, p9.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696162)  // psel p2.h, p8.h/Z, p11.h[w13, #2]
    mov x21, SP
    sub x7, x7, #0x2
    ld1w { z19.s }, p1/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0x25285141)  // psel p1.h, p4.h/Z, p10.h[w12]
    ld1w { z18.s }, p0/Z, [x20, x22, LSL #2]
    KAI_ASM_INST(0x25385140)  // psel p0.h, p4.h/Z, p10.h[w12, #1]
    ldr x20, [x21, #0]
    whilelt p10.h, x24, x2
    ld1w { z17.s }, p3/Z, [x8, x23, LSL #2]
    cmp x7, #0x2
    inch x24
    ld1w { z16.s }, p2/Z, [x8, x22, LSL #2]
    KAI_ASM_INST(0x658ab673)  // bfcvt z19.h, p5/M, z19.s
    ldr x8, [x21, #0x8]
    add x21, x21, #0x10
    KAI_ASM_INST(0x658ab652)  // bfcvt z18.h, p5/M, z18.s
    inch x23
    inch x22
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403660)  // mova za0h.h[x13], p5/M, z19.h
    KAI_ASM_INST(0xc0403622)  // mova za0h.h[x13, #2], p5/M, z17.h
    KAI_ASM_INST(0xe0bf8408)  // st1w { za2v.s[x12] }, p1/Z, [x0, XZR, LSL #2]
    KAI_ASM_INST(0xe0bb8009)  // st1w { za2v.s[x12, #1] }, p0/Z, [x0, x27, LSL #2]
    add x0, x0, x27, LSL #3
    bge label_9
KAI_ASM_LABEL(label_14)  // K loop: Tails
    cbz x7, label_17
    mov x6, x24
    mov x21, SP
    whilelt p9.s, x6, x2
    incw x6
    whilelt p8.s, x6, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_15)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25305142)  // psel p2.s, p4.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25396561)  // psel p1.h, p9.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0xe0bf8800)  // st1w { za0v.s[x12] }, p2/Z, [x0, XZR, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x27, LSL #2
    ldr x6, [x21, #0]
    cmp x12, x10
    add x21, x21, #0x8
    ld1w { z17.s }, p1/Z, [x6, x23, LSL #2]
    ld1w { z16.s }, p0/Z, [x6, x22, LSL #2]
    KAI_ASM_INST(0x658ab631)  // bfcvt z17.h, p5/M, z17.s
    KAI_ASM_INST(0x658ab610)  // bfcvt z16.h, p5/M, z16.s
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403621)  // mova za0h.h[x13, #1], p5/M, z17.h
    add x13, x13, #0x2
    blt label_15
    mov x6, x24
    whilelt p10.h, x24, x2
    whilelt p8.s, x6, x2
    incw x6
    whilelt p8.s, x6, x2
    mov x6, #0
    mov x12, #0
KAI_ASM_LABEL(label_16)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25305140)  // psel p0.s, p4.s/Z, p10.s[w12]
    add x6, x6, #0x2
    KAI_ASM_INST(0xe0bf8008)  // st1w { za2v.s[x12] }, p0/Z, [x0, XZR, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x27, LSL #2
    cmp x12, x11
    blt label_16
    whilelt p8.h, x24, x2
    b label_19
KAI_ASM_LABEL(label_17)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_18)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25305140)  // psel p0.s, p4.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8000)  // st1w { za0v.s[x12] }, p0/Z, [x0, XZR, LSL #2]
    add x12, x12, #0x1
    add x0, x0, x27, LSL #2
    cmp x12, x11
    blt label_18
KAI_ASM_LABEL(label_19)  // K loop: End
    subs x26, x26, #0x1
    sub x3, x3, x10
    bne label_2
    addvl x0, x0, #1
    cmp x3, #0
    sub x0, x0, x27, LSL #2
    bgt label_1
KAI_ASM_LABEL(label_20)  // Exit
    add SP, SP, x10, LSL #3
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave_1VL_2x2_fp32bf16)

    KAI_ASM_END
