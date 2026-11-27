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

    KAI_ASM_CODE(sme_interleave4VL_block1_fp32_fp32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave4VL_block1_fp32_fp32)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave4VL_block1_fp32_fp32)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave4VL_block1_fp32_fp32)
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
    mov x28, #0
    cntw x27
    cntw x26
    ldr x25, [x0, #0]
    cmp x3, x27
    cntw x24, ALL, MUL #2
    cntw x23, ALL, MUL #3
    csel x27, x3, x27, LT
    ptrue p4.b
    sub x27, x27, #0x1
    whilelt p3.s, XZR, x3
    whilelt p15.s, x26, x3
    whilelt p14.s, x24, x3
    whilelt p13.s, x23, x3
    mov x22, x4
    whilelt p12.s, x28, x2
    whilelt p11.s, x28, x2
    whilelt p10.s, x28, x2
    whilelt p9.s, x28, x2
    whilelt p8.s, x28, x2
KAI_ASM_LABEL(label_1)  // Width loop
    add x21, x1, XZR, LSL #3
    add x20, x1, x26, LSL #3
    add x8, x1, x24, LSL #3
    add x3, x1, x23, LSL #3
    ldr x7, [x21], #0x8
    mov x13, #0
    ldr x6, [x20], #0x8
    ldr x5, [x8], #0x8
    ldr x4, [x3], #0x8
    cbz x27, label_3
KAI_ASM_LABEL(label_2)  // Loads: Loop
    KAI_ASM_INST(0x25316c60)  // psel p0.s, p11.s/Z, p3.s[w13]
    KAI_ASM_INST(0x253169e2)  // psel p2.s, p10.s/Z, p15.s[w13]
    KAI_ASM_INST(0x253165c1)  // psel p1.s, p9.s/Z, p14.s[w13]
    KAI_ASM_INST(0xe09620e0)  // ld1w { za0h.s[x13] }, p0/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x253161a0)  // psel p0.s, p8.s/Z, p13.s[w13]
    ldr x7, [x21], #0x8
    KAI_ASM_INST(0xe09628c4)  // ld1w { za1h.s[x13] }, p2/Z, [x6, x22, LSL #2]
    ldr x6, [x20], #0x8
    KAI_ASM_INST(0xe09624a8)  // ld1w { za2h.s[x13] }, p1/Z, [x5, x22, LSL #2]
    ldr x5, [x8], #0x8
    KAI_ASM_INST(0xe096208c)  // ld1w { za3h.s[x13] }, p0/Z, [x4, x22, LSL #2]
    add x13, x13, #0x1
    ldr x4, [x3], #0x8
    cmp x13, x27
    blt label_2
KAI_ASM_LABEL(label_3)  // Loads: Tail
    KAI_ASM_INST(0x25316c60)  // psel p0.s, p11.s/Z, p3.s[w13]
    KAI_ASM_INST(0x253169e2)  // psel p2.s, p10.s/Z, p15.s[w13]
    KAI_ASM_INST(0x253165c1)  // psel p1.s, p9.s/Z, p14.s[w13]
    sub x3, x2, x28
    cmp x3, x26
    mov x12, #0
    KAI_ASM_INST(0xe09620e0)  // ld1w { za0h.s[x13] }, p0/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x253161a0)  // psel p0.s, p8.s/Z, p13.s[w13]
    csel x3, x3, x26, LT
    KAI_ASM_INST(0xe09628c4)  // ld1w { za1h.s[x13] }, p2/Z, [x6, x22, LSL #2]
    KAI_ASM_INST(0xe09624a8)  // ld1w { za2h.s[x13] }, p1/Z, [x5, x22, LSL #2]
    KAI_ASM_INST(0xe096208c)  // ld1w { za3h.s[x13] }, p0/Z, [x4, x22, LSL #2]
KAI_ASM_LABEL(label_4)  // Stores: Loop
    KAI_ASM_INST(0x25305180)  // psel p0.s, p4.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25305182)  // psel p2.s, p4.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25305181)  // psel p1.s, p4.s/Z, p12.s[w12]
    KAI_ASM_INST(0xe0bf8320)  // st1w { za0v.s[x12] }, p0/Z, [x25, XZR, LSL #2]
    KAI_ASM_INST(0x25305180)  // psel p0.s, p4.s/Z, p12.s[w12]
    KAI_ASM_INST(0xe0ba8b24)  // st1w { za1v.s[x12] }, p2/Z, [x25, x26, LSL #2]
    KAI_ASM_INST(0xe0b88728)  // st1w { za2v.s[x12] }, p1/Z, [x25, x24, LSL #2]
    KAI_ASM_INST(0xe0b7832c)  // st1w { za3v.s[x12] }, p0/Z, [x25, x23, LSL #2]
    add x12, x12, #0x1
    addvl x25, x25, #4
    cmp x12, x3
    blt label_4
    incw x28
    incw x22
    whilelt p12.s, x28, x2
    whilelt p11.s, x28, x2
    whilelt p10.s, x28, x2
    whilelt p9.s, x28, x2
    whilelt p8.s, x28, x2
    b.ne label_1
    str x25, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave4VL_block1_fp32_fp32)

    KAI_ASM_END
