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

    KAI_ASM_CODE(sme_interleave4VL_block2_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave4VL_block2_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave4VL_block2_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave4VL_block2_fp16_fp16)
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
    cntw x28
    cntw x27
    ldr x26, [x0, #0]
    cntw x25, ALL, MUL #2
    cntw x24, ALL, MUL #3
    cmp x3, x28
    cnth x23
    csel x28, x3, x28, LT
    whilelt p11.h, XZR, x3
    whilelt p10.h, x27, x3
    whilelt p9.h, x25, x3
    whilelt p8.h, x24, x3
    ptrue p13.b
    sub x28, x28, #0x1
    zip1 p12.h, p11.h, p9.h
    zip1 p11.h, p10.h, p8.h
    mov x22, x4
    whilelt p10.h, x9, x2
    whilelt p9.h, x9, x2
    whilelt p8.h, x9, x2
KAI_ASM_LABEL(label_1)  // Width loop
    add x21, x1, XZR, LSL #3
    add x20, x1, x27, LSL #3
    add x8, x1, x25, LSL #3
    add x3, x1, x24, LSL #3
    ldr x7, [x21], #0x8
    mov x13, #0
    ldr x6, [x20], #0x8
    ldr x5, [x8], #0x8
    ldr x4, [x3], #0x8
    cbz x28, label_3
KAI_ASM_LABEL(label_2)  // Loads: Loop
    KAI_ASM_INST(0x25296580)  // psel p0.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296162)  // psel p2.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25396581)  // psel p1.h, p9.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0xe05620e0)  // ld1h { za0h.h[x13] }, p0/Z, [x7, x22, LSL #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    ldr x7, [x21], #0x8
    KAI_ASM_INST(0xe05628c8)  // ld1h { za1h.h[x13] }, p2/Z, [x6, x22, LSL #1]
    ldr x6, [x20], #0x8
    KAI_ASM_INST(0xe05624a1)  // ld1h { za0h.h[x13, #1] }, p1/Z, [x5, x22, LSL #1]
    ldr x5, [x8], #0x8
    KAI_ASM_INST(0xe0562089)  // ld1h { za1h.h[x13, #1] }, p0/Z, [x4, x22, LSL #1]
    add x13, x13, #0x2
    ldr x4, [x3], #0x8
    cmp x13, x28, LSL #1
    blt label_2
KAI_ASM_LABEL(label_3)  // Loads: Tail
    KAI_ASM_INST(0x25296580)  // psel p0.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296162)  // psel p2.h, p8.h/Z, p11.h[w13]
    sub x3, x2, x9
    KAI_ASM_INST(0x25396581)  // psel p1.h, p9.h/Z, p12.h[w13, #1]
    cmp x3, x23
    mov x12, #0
    KAI_ASM_INST(0xe05620e0)  // ld1h { za0h.h[x13] }, p0/Z, [x7, x22, LSL #1]
    KAI_ASM_INST(0x25396160)  // psel p0.h, p8.h/Z, p11.h[w13, #1]
    csel x3, x3, x23, LT
    KAI_ASM_INST(0xe05628c8)  // ld1h { za1h.h[x13] }, p2/Z, [x6, x22, LSL #1]
    add x3, x3, #0x1
    KAI_ASM_INST(0xe05624a1)  // ld1h { za0h.h[x13, #1] }, p1/Z, [x5, x22, LSL #1]
    lsr x3, x3, #0x1
    KAI_ASM_INST(0xe0562089)  // ld1h { za1h.h[x13, #1] }, p0/Z, [x4, x22, LSL #1]
KAI_ASM_LABEL(label_4)  // Stores: Loop
    KAI_ASM_INST(0x25307540)  // psel p0.s, p13.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25307542)  // psel p2.s, p13.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25307541)  // psel p1.s, p13.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8340)  // st1w { za0v.s[x12] }, p0/Z, [x26, XZR, LSL #2]
    KAI_ASM_INST(0x25307540)  // psel p0.s, p13.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bb8b44)  // st1w { za1v.s[x12] }, p2/Z, [x26, x27, LSL #2]
    KAI_ASM_INST(0xe0b98748)  // st1w { za2v.s[x12] }, p1/Z, [x26, x25, LSL #2]
    KAI_ASM_INST(0xe0b8834c)  // st1w { za3v.s[x12] }, p0/Z, [x26, x24, LSL #2]
    add x12, x12, #0x1
    addvl x26, x26, #4
    cmp x12, x3
    blt label_4
    inch x9
    inch x22
    whilelt p10.h, x9, x2
    whilelt p9.h, x9, x2
    whilelt p8.h, x9, x2
    b.ne label_1
    str x26, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave4VL_block2_fp16_fp16)

    KAI_ASM_END
