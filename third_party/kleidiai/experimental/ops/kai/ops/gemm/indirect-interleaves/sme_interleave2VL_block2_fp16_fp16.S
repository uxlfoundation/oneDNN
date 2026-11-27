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

    KAI_ASM_CODE(sme_interleave2VL_block2_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_block2_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_block2_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_block2_fp16_fp16)
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
    mov x9, #0
    cnth x7
    mov x6, x2
    ldr x28, [x0, #0]
    mov x5, x2
    sub x27, x7, #0x1
    inch x6
    ands x27, x5, x27
    cntw x26
    sub x6, x6, #0x1
    csel x27, x27, x7, NE
    udiv x6, x6, x7  // n_passes = ceildiv(width, VL<T>)
    add x27, x27, #0x1
    sub x25, x26, #0x2
    lsl x24, x3, #0x1  // height * 2
    lsl x5, x26, #0x1
    mov x23, x1
    add x22, x1, x26, LSL #3
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    ldr x8, [x23, #0]
    sub x3, x6, #0x1
    lsr x27, x27, #0x1
    ldr x7, [x22, #0]
    ptrue p12.s
    whilelt p11.h, XZR, x24
    ldr x6, [x23, #0x8]
    whilelt p10.h, x5, x24
    mov x5, x4
    ldr x4, [x22, #0x8]
    whilelt p9.h, x9, x2
    whilelt p8.h, x9, x2
    add x23, x23, #0x10
    add x22, x22, #0x10
    mov x12, #0
    cbz x25, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    KAI_ASM_INST(0xe0450d00)  // ld1h { za0h.h[x12] }, p3/Z, [x8, x5, LSL #1]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04508e8)  // ld1h { za1h.h[x12] }, p2/Z, [x7, x5, LSL #1]
    ldr x7, [x22, #0]
    KAI_ASM_INST(0xe04504c2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x6, x5, LSL #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe045008a)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x4, x5, LSL #1]
    add x12, x12, #0x4
    ldr x4, [x22, #0x8]
    add x22, x22, #0x10
    cmp x12, x25, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    mov x3, x3
    mov x23, x1
    KAI_ASM_INST(0xe0450d00)  // ld1h { za0h.h[x12] }, p3/Z, [x8, x5, LSL #1]
    add x22, x1, x26, LSL #3
    cmp x3, #0x2
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04508e8)  // ld1h { za1h.h[x12] }, p2/Z, [x7, x5, LSL #1]
    ldr x7, [x22, #0]
    inch x9
    KAI_ASM_INST(0xe04504c2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x6, x5, LSL #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe045008a)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x4, x5, LSL #1]
    ldr x4, [x22, #0x8]
    add x22, x22, #0x10
    inch x5
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x9, x2
    mov x15, #0
    mov x14, #0
    cbz x25, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    KAI_ASM_INST(0xe0456101)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x8, x5, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04568e9)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x7, x5, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x7, [x22, #0]
    KAI_ASM_INST(0xe04564c3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x6, x5, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0456c8b)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x4, x5, LSL #1]
    ldr x4, [x22, #0x8]
    KAI_ASM_INST(0xe0bfc380)  // st1w { za0v.s[x14] }, p0/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0bacb84)  // st1w { za1v.s[x14] }, p2/Z, [x28, x26, LSL #2]
    add x22, x22, #0x10
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0b5c781)  // st1w { za0v.s[x14, #1] }, p1/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0xe0b4c385)  // st1w { za1v.s[x14, #1] }, p0/Z, [x28, x20, LSL #2]
    add x14, x14, #0x2
    addvl x28, x28, #4
    cmp x14, x25
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    mov x23, x1
    add x22, x1, x26, LSL #3
    KAI_ASM_INST(0xe0456101)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x8, x5, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x8, [x23, #0]
    mov x13, #0
    KAI_ASM_INST(0xe04568e9)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x7, x5, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x7, [x22, #0]
    mov x12, #0
    KAI_ASM_INST(0xe04564c3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x6, x5, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0456c8b)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x4, x5, LSL #1]
    ldr x4, [x22, #0x8]
    KAI_ASM_INST(0xe0bfc380)  // st1w { za0v.s[x14] }, p0/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0bacb84)  // st1w { za1v.s[x14] }, p2/Z, [x28, x26, LSL #2]
    whilelt p9.h, x9, x2
    inch x9
    KAI_ASM_INST(0xe0b5c781)  // st1w { za0v.s[x14, #1] }, p1/Z, [x28, x21, LSL #2]
    add x22, x22, #0x10
    inch x5
    KAI_ASM_INST(0xe0b4c385)  // st1w { za1v.s[x14, #1] }, p0/Z, [x28, x20, LSL #2]
    addvl x28, x28, #4
    whilelt p8.h, x9, x2
    cbz x25, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    KAI_ASM_INST(0xe0452100)  // ld1h { za0h.h[x13] }, p0/Z, [x8, x5, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04528e8)  // ld1h { za1h.h[x13] }, p2/Z, [x7, x5, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x7, [x22, #0]
    KAI_ASM_INST(0xe04524c2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x6, x5, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0452c8a)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x4, x5, LSL #1]
    ldr x4, [x22, #0x8]
    KAI_ASM_INST(0xe0bf8388)  // st1w { za2v.s[x12] }, p0/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0ba8b8c)  // st1w { za3v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    add x22, x22, #0x10
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0b58789)  // st1w { za2v.s[x12, #1] }, p1/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0xe0b4838d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x28, x20, LSL #2]
    add x12, x12, #0x2
    addvl x28, x28, #4
    cmp x12, x25
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    mov x23, x1
    add x22, x1, x26, LSL #3
    KAI_ASM_INST(0xe0452100)  // ld1h { za0h.h[x13] }, p0/Z, [x8, x5, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x8, [x23, #0]
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe04528e8)  // ld1h { za1h.h[x13] }, p2/Z, [x7, x5, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x7, [x22, #0]
    KAI_ASM_INST(0xe04524c2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x6, x5, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x6, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0452c8a)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x4, x5, LSL #1]
    ldr x4, [x22, #0x8]
    KAI_ASM_INST(0xe0bf8388)  // st1w { za2v.s[x12] }, p0/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0ba8b8c)  // st1w { za3v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    whilelt p9.h, x9, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe0b58789)  // st1w { za2v.s[x12, #1] }, p1/Z, [x28, x21, LSL #2]
    add x22, x22, #0x10
    inch x9
    KAI_ASM_INST(0xe0b4838d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x28, x20, LSL #2]
    addvl x28, x28, #4
    inch x5
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x23, x1
    whilelt p8.h, x9, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25396161)  // psel p1.h, p8.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396140)  // psel p0.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0xe0bf8f80)  // st1w { za0v.s[x12] }, p3/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba8b84)  // st1w { za1v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    ldr x3, [x23, #0]
    cmp x12, x26
    ldr x1, [x23, x26, LSL #0x3]
    add x23, x23, #0x8
    KAI_ASM_INST(0xe0452461)  // ld1h { za0h.h[x13, #1] }, p1/Z, [x3, x5, LSL #1]
    KAI_ASM_INST(0xe0452029)  // ld1h { za1h.h[x13, #1] }, p0/Z, [x1, x5, LSL #1]
    add x13, x13, #0x2
    blt label_9
    whilelt p9.h, x9, x2
    whilelt p8.h, x9, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    add x1, x1, #0x2
    KAI_ASM_INST(0xe0bf8788)  // st1w { za2v.s[x12] }, p1/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba838c)  // st1w { za3v.s[x12] }, p0/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    cmp x12, x27
    blt label_10
    whilelt p8.h, x9, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8780)  // st1w { za0v.s[x12] }, p1/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba8384)  // st1w { za1v.s[x12] }, p0/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    cmp x12, x27
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x28, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_block2_fp16_fp16)

    KAI_ASM_END
