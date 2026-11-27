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

    KAI_ASM_CODE(sme_transpose_interleave_8VL_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_8VL_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_8VL_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_8VL_1x4)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cntb x5, ALL, MUL #8
    mov z16.b, #0
    add x6, x6, #0x3
    ptrue p2.b
    lsr x5, x5, #0x2
    bfm x6, XZR, #0, #0x1  // bfc x6, #0, #0x2
    addvl SP, SP, #-2
    mul x27, x6, x5
    st1b { z16.b }, p2, [SP]
    mov x26, SP
    mov x25, x4
    st1b { z16.b }, p2, [SP, #1, MUL VL]
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    cntw x4, ALL, MUL #8
    add x23, x24, x3
    cmp x25, #0x3
    add x22, x23, x3
    csel x21, x4, XZR, GT
    add x20, x22, x3
    csel x22, x22, x26, GE
    add x1, x20, x3
    csel x20, x20, x26, GT
    csel x8, x4, XZR, GE
    cmp x25, #0x1
    mov x7, x0
    csel x23, x23, x26, GT
    csel x6, x4, XZR, GT
    mov x5, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x4, x5
    decw x5, ALL, MUL #8
    whilelt p1.b, XZR, x4
    decb x4
    whilelt p0.b, XZR, x4
    ld1b { z20.b }, p1/Z, [x24]
    cmp x5, #0
    ld1b { z24.b }, p0/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ld1b { z23.b }, p1/Z, [x23]
    ld1b { z22.b }, p0/Z, [x23, #1, MUL VL]
    add x23, x23, x6
    ld1b { z19.b }, p1/Z, [x22]
    ld1b { z18.b }, p0/Z, [x22, #1, MUL VL]
    add x22, x22, x8
    ld1b { z17.b }, p1/Z, [x20]
    ld1b { z16.b }, p0/Z, [x20, #1, MUL VL]
    zip1 z21.b, z20.b, z19.b
    zip2 z20.b, z20.b, z19.b
    add x20, x20, x21
    zip1 z25.b, z24.b, z18.b
    zip2 z24.b, z24.b, z18.b
    zip1 z19.b, z23.b, z17.b
    zip2 z18.b, z23.b, z17.b
    zip1 z17.b, z22.b, z16.b
    zip2 z16.b, z22.b, z16.b
    zip1 z23.b, z21.b, z19.b
    zip2 z22.b, z21.b, z19.b
    zip1 z21.b, z20.b, z18.b
    zip2 z20.b, z20.b, z18.b
    zip1 z19.b, z25.b, z17.b
    zip2 z18.b, z25.b, z17.b
    zip1 z17.b, z24.b, z16.b
    zip2 z16.b, z24.b, z16.b
    st1b { z23.b }, p2, [x7]
    st1b { z22.b }, p2, [x7, #1, MUL VL]
    st1b { z21.b }, p2, [x7, #2, MUL VL]
    st1b { z20.b }, p2, [x7, #3, MUL VL]
    st1b { z19.b }, p2, [x7, #4, MUL VL]
    st1b { z18.b }, p2, [x7, #5, MUL VL]
    st1b { z17.b }, p2, [x7, #6, MUL VL]
    st1b { z16.b }, p2, [x7, #7, MUL VL]
    add x7, x7, x27
    bgt label_2
    sub x25, x25, #0x4
    addvl x0, x0, #8
    cmp x25, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #2
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_8VL_1x4)

    KAI_ASM_END
