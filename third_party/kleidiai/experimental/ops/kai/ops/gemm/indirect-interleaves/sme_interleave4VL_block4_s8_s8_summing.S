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

    KAI_ASM_CODE(sme_interleave4VL_block4_s8_s8_summing)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave4VL_block4_s8_s8_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave4VL_block4_s8_s8_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave4VL_block4_s8_s8_summing)
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
    cntw x9
    cntw x28
    ldr x27, [x0, #0]
    mov z23.b, #0x1
    cntw x26, ALL, MUL #2
    cntw x25, ALL, MUL #3
    dup z22.s, w6
    mov z21.s, #0
    cmp x3, x9
    ptrue p3.b
    mov z20.s, #0
    mov z19.s, #0
    csel x9, x3, x9, LT
    whilelt p12.b, XZR, x3
    mov z18.s, #0
    whilelt p10.b, x28, x3
    whilelt p9.b, x26, x3
    whilelt p8.b, x25, x3
    zip1 p12.b, p12.b, p9.b
    zip1 p10.b, p10.b, p8.b
    cntb x24
    ptrue p11.b
    sub x9, x9, #0x1
    zip1 p10.b, p12.b, p10.b
    mov x23, x4
    cbnz w5, label_1
    addvl x27, x27, #-4
    ld1w { z21.s }, p3/Z, [x27]
    ld1w { z20.s }, p3/Z, [x27, #1, MUL VL]
    ld1w { z19.s }, p3/Z, [x27, #2, MUL VL]
    ld1w { z18.s }, p3/Z, [x27, #3, MUL VL]
KAI_ASM_LABEL(label_1)  // Initialise row sums: End
    mov x22, #0
    whilelt p9.b, x22, x2
    whilelt p8.b, x22, x2
KAI_ASM_LABEL(label_2)  // Width loop
    add x21, x1, XZR, LSL #3
    add x20, x1, x28, LSL #3
    add x8, x1, x26, LSL #3
    add x3, x1, x25, LSL #3
    ldr x7, [x21], #0x8
    mov x13, #0
    ldr x6, [x20], #0x8
    ldr x5, [x8], #0x8
    ldr x4, [x3], #0x8
    cbz x9, label_4
KAI_ASM_LABEL(label_3)  // Loads: Loop
    KAI_ASM_INST(0x25256140)  // psel p0.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25356141)  // psel p1.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0xe01720e0)  // ld1b { za0h.b[x13] }, p0/Z, [x7, x23]
    KAI_ASM_INST(0x253d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #3]
    ldr x7, [x21], #0x8
    KAI_ASM_INST(0xe01728c1)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x6, x23]
    ldr x6, [x20], #0x8
    KAI_ASM_INST(0xe01724a2)  // ld1b { za0h.b[x13, #2] }, p1/Z, [x5, x23]
    ldr x5, [x8], #0x8
    KAI_ASM_INST(0xe0172083)  // ld1b { za0h.b[x13, #3] }, p0/Z, [x4, x23]
    add x13, x13, #0x4
    ldr x4, [x3], #0x8
    cmp x13, x9, LSL #2
    blt label_3
KAI_ASM_LABEL(label_4)  // Loads: Tail
    KAI_ASM_INST(0x25256140)  // psel p0.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25356141)  // psel p1.b, p8.b/Z, p10.b[w13, #2]
    sub x3, x2, x22
    cmp x3, x24
    mov x12, #0
    KAI_ASM_INST(0xe01720e0)  // ld1b { za0h.b[x13] }, p0/Z, [x7, x23]
    KAI_ASM_INST(0x253d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #3]
    csel x3, x3, x24, LT
    KAI_ASM_INST(0xe01728c1)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x6, x23]
    add x3, x3, #0x3
    KAI_ASM_INST(0xe01724a2)  // ld1b { za0h.b[x13, #2] }, p1/Z, [x5, x23]
    lsr x3, x3, #0x2
    KAI_ASM_INST(0xe0172083)  // ld1b { za0h.b[x13, #3] }, p0/Z, [x4, x23]
KAI_ASM_LABEL(label_5)  // Stores: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d22)  // psel p2.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0828c11)  // mova z17.s, p3/M, za0v.s[x12]
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0828c90)  // mova z16.s, p3/M, za1v.s[x12]
    sdot z21.s, z17.b, z23.b
    KAI_ASM_INST(0xc0828d11)  // mova z17.s, p3/M, za2v.s[x12]
    sdot z20.s, z16.b, z23.b
    KAI_ASM_INST(0xe0bf8360)  // st1w { za0v.s[x12] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0828d90)  // mova z16.s, p3/M, za3v.s[x12]
    sdot z19.s, z17.b, z23.b
    KAI_ASM_INST(0xe0bc8b64)  // st1w { za1v.s[x12] }, p2/Z, [x27, x28, LSL #2]
    sdot z18.s, z16.b, z23.b
    KAI_ASM_INST(0xe0ba8768)  // st1w { za2v.s[x12] }, p1/Z, [x27, x26, LSL #2]
    KAI_ASM_INST(0xe0b9836c)  // st1w { za3v.s[x12] }, p0/Z, [x27, x25, LSL #2]
    add x12, x12, #0x1
    addvl x27, x27, #4
    cmp x12, x3
    blt label_5
    incb x22
    incb x23
    whilelt p9.b, x22, x2
    whilelt p8.b, x22, x2
    b.ne label_2
    mul z21.s, p3/M, z21.s, z22.s
    mul z20.s, p3/M, z20.s, z22.s
    mul z19.s, p3/M, z19.s, z22.s
    mul z18.s, p3/M, z18.s, z22.s
    st1w { z21.s }, p3, [x27]
    st1w { z20.s }, p3, [x27, #1, MUL VL]
    st1w { z19.s }, p3, [x27, #2, MUL VL]
    st1w { z18.s }, p3, [x27, #3, MUL VL]
    addvl x27, x27, #4
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    str x27, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave4VL_block4_s8_s8_summing)

    KAI_ASM_END
