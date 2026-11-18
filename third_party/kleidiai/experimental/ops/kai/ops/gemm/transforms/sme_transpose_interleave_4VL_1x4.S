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

    KAI_ASM_CODE(sme_transpose_interleave_4VL_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_4VL_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_4VL_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_4VL_1x4)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    stp d8, d9, [sp, 56]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cntb x5, ALL, MUL #4
    mov z16.b, #0
    add x6, x6, #0x3
    ptrue p1.b
    lsr x5, x5, #0x2
    bfm x6, XZR, #0, #0x1  // bfc x6, #0, #0x2
    addvl SP, SP, #-1
    mul x26, x6, x5
    st1b { z16.b }, p1, [SP]
    mov x25, SP
    mov x24, x4
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x23, x1
    cntw x4, ALL, MUL #4
    add x22, x23, x3
    cmp x24, #0x3
    add x21, x22, x3
    csel x20, x4, XZR, GT
    add x8, x21, x3
    csel x21, x21, x25, GE
    add x1, x8, x3
    csel x8, x8, x25, GT
    csel x7, x4, XZR, GE
    cmp x24, #0x1
    mov x6, x0
    csel x22, x22, x25, GT
    csel x5, x4, XZR, GT
    mov x4, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    whilelt p0.b, XZR, x4
    decw x4, ALL, MUL #4
    ld1b { z20.b }, p0/Z, [x23]
    cmp x4, #0
    addvl x23, x23, #1
    ld1b { z19.b }, p0/Z, [x22]
    add x22, x22, x5
    ld1b { z17.b }, p0/Z, [x21]
    add x21, x21, x7
    ld1b { z16.b }, p0/Z, [x8]
    add x8, x8, x20
    zip1 z18.b, z20.b, z17.b
    zip2 z20.b, z20.b, z17.b
    zip1 z17.b, z19.b, z16.b
    zip2 z16.b, z19.b, z16.b
    zip1 z19.b, z18.b, z17.b
    zip2 z18.b, z18.b, z17.b
    zip1 z17.b, z20.b, z16.b
    zip2 z16.b, z20.b, z16.b
    st1b { z19.b }, p1, [x6]
    st1b { z18.b }, p1, [x6, #1, MUL VL]
    st1b { z17.b }, p1, [x6, #2, MUL VL]
    st1b { z16.b }, p1, [x6, #3, MUL VL]
    add x6, x6, x26
    bgt label_2
    sub x24, x24, #0x4
    addvl x0, x0, #4
    cmp x24, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp d8, d9, [sp, 56]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_4VL_1x4)

    KAI_ASM_END
