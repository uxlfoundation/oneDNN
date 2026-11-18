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

    KAI_ASM_CODE(sme_interleave2VL_fp32_fp32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_fp32_fp32)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_fp32_fp32)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_fp32_fp32)
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
    mov x6, x2
    cntw x28
    ldr x27, [x0, #0]
    incw x6
    mov x5, x2
    sub x6, x6, #0x1
    sub x26, x28, #0x1
    udiv x6, x6, x28  // n_passes = ceildiv(width, VL<T>)
    ands x26, x5, x26
    sub x25, x28, #0x2
    mov x24, x1
    add x23, x1, x28, LSL #3
    cntw x22, ALL, MUL #2
    ldr x21, [x24, #0]
    cntw x20, ALL, MUL #3
    sub x8, x6, #0x1
    ldr x7, [x23, #0]
    csel x26, x26, x28, NE
    ptrue p12.s
    ldr x6, [x24, #0x8]
    whilelt p11.s, XZR, x3
    whilelt p10.s, x28, x3
    ldr x5, [x23, #0x8]
    mov x4, x4
    whilelt p9.s, x9, x2
    whilelt p8.s, x9, x2
    add x24, x24, #0x10
    add x23, x23, #0x10
    mov x12, #0
    cbz x25, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe0840ea0)  // ld1w { za0h.s[x12] }, p3/Z, [x21, x4, LSL #2]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe08408e4)  // ld1w { za1h.s[x12] }, p2/Z, [x7, x4, LSL #2]
    ldr x7, [x23, #0]
    KAI_ASM_INST(0xe08404c1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x4, LSL #2]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe08400a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x5, x4, LSL #2]
    add x12, x12, #0x2
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    cmp x12, x25
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    mov x3, x8
    mov x24, x1
    KAI_ASM_INST(0xe0840ea0)  // ld1w { za0h.s[x12] }, p3/Z, [x21, x4, LSL #2]
    add x23, x1, x28, LSL #3
    cmp x3, #0x2
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe08408e4)  // ld1w { za1h.s[x12] }, p2/Z, [x7, x4, LSL #2]
    ldr x7, [x23, #0]
    incw x9
    KAI_ASM_INST(0xe08404c1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x4, LSL #2]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe08400a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x5, x4, LSL #2]
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    incw x4
    blt label_8
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.s, x9, x2
    mov x13, #0
    cbz x25, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0xe08422a8)  // ld1w { za2h.s[x13] }, p0/Z, [x21, x4, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe08428ec)  // ld1w { za3h.s[x13] }, p2/Z, [x7, x4, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x7, [x23, #0]
    KAI_ASM_INST(0xe08424c9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x6, x4, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0842cad)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x5, x4, LSL #2]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bfa360)  // st1w { za0v.s[x13] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0bcab64)  // st1w { za1v.s[x13] }, p2/Z, [x27, x28, LSL #2]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0b6a761)  // st1w { za0v.s[x13, #1] }, p1/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0xe0b4a365)  // st1w { za1v.s[x13, #1] }, p0/Z, [x27, x20, LSL #2]
    add x13, x13, #0x2
    addvl x27, x27, #4
    cmp x13, x25
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    mov x24, x1
    add x23, x1, x28, LSL #3
    KAI_ASM_INST(0xe08422a8)  // ld1w { za2h.s[x13] }, p0/Z, [x21, x4, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x21, [x24, #0]
    mov x12, #0
    KAI_ASM_INST(0xe08428ec)  // ld1w { za3h.s[x13] }, p2/Z, [x7, x4, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x7, [x23, #0]
    KAI_ASM_INST(0xe08424c9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x6, x4, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0842cad)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x5, x4, LSL #2]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bfa360)  // st1w { za0v.s[x13] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0bcab64)  // st1w { za1v.s[x13] }, p2/Z, [x27, x28, LSL #2]
    whilelt p9.s, x9, x2
    incw x9
    KAI_ASM_INST(0xe0b6a761)  // st1w { za0v.s[x13, #1] }, p1/Z, [x27, x22, LSL #2]
    add x23, x23, #0x10
    incw x4
    KAI_ASM_INST(0xe0b4a365)  // st1w { za1v.s[x13, #1] }, p0/Z, [x27, x20, LSL #2]
    addvl x27, x27, #4
    whilelt p8.s, x9, x2
    cbz x25, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe08402a0)  // ld1w { za0h.s[x12] }, p0/Z, [x21, x4, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe08408e4)  // ld1w { za1h.s[x12] }, p2/Z, [x7, x4, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x7, [x23, #0]
    KAI_ASM_INST(0xe08404c1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x4, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0840ca5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x5, x4, LSL #2]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bf8368)  // st1w { za2v.s[x12] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0bc8b6c)  // st1w { za3v.s[x12] }, p2/Z, [x27, x28, LSL #2]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0b68769)  // st1w { za2v.s[x12, #1] }, p1/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0xe0b4836d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x27, x20, LSL #2]
    add x12, x12, #0x2
    addvl x27, x27, #4
    cmp x12, x25
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    mov x24, x1
    add x23, x1, x28, LSL #3
    KAI_ASM_INST(0xe08402a0)  // ld1w { za0h.s[x12] }, p0/Z, [x21, x4, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x21, [x24, #0]
    sub x3, x3, #0x2
    KAI_ASM_INST(0xe08408e4)  // ld1w { za1h.s[x12] }, p2/Z, [x7, x4, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x7, [x23, #0]
    KAI_ASM_INST(0xe08404c1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x6, x4, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0840ca5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x5, x4, LSL #2]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bf8368)  // st1w { za2v.s[x12] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0bc8b6c)  // st1w { za3v.s[x12] }, p2/Z, [x27, x28, LSL #2]
    whilelt p9.s, x9, x2
    cmp x3, #0x2
    KAI_ASM_INST(0xe0b68769)  // st1w { za2v.s[x12, #1] }, p1/Z, [x27, x22, LSL #2]
    add x23, x23, #0x10
    incw x9
    KAI_ASM_INST(0xe0b4836d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x27, x20, LSL #2]
    addvl x27, x27, #4
    incw x4
    bge label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbz x3, label_11
    mov x24, x1
    whilelt p8.s, x9, x2
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306161)  // psel p1.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306140)  // psel p0.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8f60)  // st1w { za0v.s[x12] }, p3/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8b64)  // st1w { za1v.s[x12] }, p2/Z, [x27, x28, LSL #2]
    addvl x27, x27, #2
    ldr x3, [x24, #0]
    ldr x1, [x24, x28, LSL #0x3]
    add x24, x24, #0x8
    KAI_ASM_INST(0xe0840468)  // ld1w { za2h.s[x12] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084002c)  // ld1w { za3h.s[x12] }, p0/Z, [x1, x4, LSL #2]
    add x12, x12, #0x1
    cmp x12, x28
    blt label_9
    whilelt p9.s, x9, x2
    whilelt p8.s, x9, x2
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8768)  // st1w { za2v.s[x12] }, p1/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc836c)  // st1w { za3v.s[x12] }, p0/Z, [x27, x28, LSL #2]
    add x12, x12, #0x1
    addvl x27, x27, #2
    cmp x12, x26
    blt label_10
    whilelt p8.s, x9, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8760)  // st1w { za0v.s[x12] }, p1/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8364)  // st1w { za1v.s[x12] }, p0/Z, [x27, x28, LSL #2]
    add x12, x12, #0x1
    addvl x27, x27, #2
    cmp x12, x26
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x27, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_fp32_fp32)

    KAI_ASM_END
