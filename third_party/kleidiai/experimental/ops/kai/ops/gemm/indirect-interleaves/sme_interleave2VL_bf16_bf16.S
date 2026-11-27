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

    KAI_ASM_CODE(sme_interleave2VL_block1_bf16_bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_block1_bf16_bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_block1_bf16_bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_block1_bf16_bf16)
    stp x20, x21, [sp, -96]!
    str x22, [sp, 16]
    stp d8, d9, [sp, 24]
    stp d10, d11, [sp, 40]
    stp d12, d13, [sp, 56]
    stp d14, d15, [sp, 72]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x22, #0
    cnth x21
    cnth x20
    ldr x8, [x0, #0]
    cmp x3, x21
    ptrue p13.b
    csel x21, x3, x21, LT
    whilelt p12.h, XZR, x3
    sub x21, x21, #0x1
    whilelt p11.h, x20, x3
    mov x7, x4
    whilelt p10.h, x22, x2
    whilelt p9.h, x22, x2
    whilelt p8.h, x22, x2
KAI_ASM_LABEL(label_1)  // Width loop
    add x6, x1, XZR, LSL #3
    add x3, x1, x20, LSL #3
    mov x13, #0
    ldr x5, [x6], #0x8
    ldr x4, [x3], #0x8
    cbz x21, label_3
KAI_ASM_LABEL(label_2)  // Loads: Loop
    KAI_ASM_INST(0x25296581)  // psel p1.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0xe04724a0)  // ld1h { za0h.h[x13] }, p1/Z, [x5, x7, LSL #1]
    ldr x5, [x6], #0x8
    KAI_ASM_INST(0xe0472088)  // ld1h { za1h.h[x13] }, p0/Z, [x4, x7, LSL #1]
    add x13, x13, #0x1
    ldr x4, [x3], #0x8
    cmp x13, x21
    blt label_2
KAI_ASM_LABEL(label_3)  // Loads: Tail
    KAI_ASM_INST(0x25296581)  // psel p1.h, p9.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    sub x3, x2, x22
    mov x12, #0
    cmp x3, x20
    KAI_ASM_INST(0xe04724a0)  // ld1h { za0h.h[x13] }, p1/Z, [x5, x7, LSL #1]
    csel x3, x3, x20, LT
    KAI_ASM_INST(0xe0472088)  // ld1h { za1h.h[x13] }, p0/Z, [x4, x7, LSL #1]
KAI_ASM_LABEL(label_4)  // Stores: Loop
    KAI_ASM_INST(0x25287541)  // psel p1.h, p13.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25287540)  // psel p0.h, p13.h/Z, p10.h[w12]
    KAI_ASM_INST(0xe07f8500)  // st1h { za0v.h[x12] }, p1/Z, [x8, XZR, LSL #1]
    KAI_ASM_INST(0xe0748108)  // st1h { za1v.h[x12] }, p0/Z, [x8, x20, LSL #1]
    add x12, x12, #0x1
    addvl x8, x8, #2
    cmp x12, x3
    blt label_4
    inch x22
    inch x7
    whilelt p10.h, x22, x2
    whilelt p9.h, x22, x2
    whilelt p8.h, x22, x2
    b.ne label_1
    str x8, [x0]
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldr x22, [sp, 16]
    ldp d8, d9, [sp, 24]
    ldp d10, d11, [sp, 40]
    ldp d12, d13, [sp, 56]
    ldp d14, d15, [sp, 72]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_block1_bf16_bf16)

    KAI_ASM_END
