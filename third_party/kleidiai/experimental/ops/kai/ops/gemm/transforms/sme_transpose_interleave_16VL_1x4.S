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

    KAI_ASM_CODE(sme_transpose_interleave_16VL_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_16VL_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_16VL_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_16VL_1x4)
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
    mov x6, x4
    cntb x5, ALL, MUL #16
    mov z16.b, #0
    add x6, x6, #0x3
    ptrue p4.b
    lsr x5, x5, #0x2
    bfm x6, XZR, #0, #0x1  // bfc x6, #0, #0x2
    addvl SP, SP, #-4
    mul x28, x6, x5
    st1b { z16.b }, p4, [SP]
    mov x27, SP
    mov x26, x4
    st1b { z16.b }, p4, [SP, #1, MUL VL]
    st1b { z16.b }, p4, [SP, #2, MUL VL]
    st1b { z16.b }, p4, [SP, #3, MUL VL]
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x25, x1
    cntw x4, ALL, MUL #16
    add x24, x25, x3
    cmp x26, #0x3
    add x23, x24, x3
    csel x22, x4, XZR, GT
    add x21, x23, x3
    csel x23, x23, x27, GE
    add x1, x21, x3
    csel x21, x21, x27, GT
    csel x20, x4, XZR, GE
    cmp x26, #0x1
    mov x8, x0
    csel x24, x24, x27, GT
    csel x7, x4, XZR, GT
    mov x6, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x5, x6
    mov x4, x8
    whilelt p3.b, XZR, x5
    decb x5
    whilelt p2.b, XZR, x5
    decb x5
    ld1b { z21.b }, p3/Z, [x25]
    whilelt p1.b, XZR, x5
    decb x5
    ld1b { z24.b }, p2/Z, [x25, #1, MUL VL]
    whilelt p0.b, XZR, x5
    ld1b { z23.b }, p3/Z, [x24]
    decw x6, ALL, MUL #16
    ld1b { z20.b }, p2/Z, [x24, #1, MUL VL]
    cmp x6, #0
    add x8, x8, x28
    ld1b { z19.b }, p3/Z, [x23]
    ld1b { z17.b }, p2/Z, [x23, #1, MUL VL]
    ld1b { z16.b }, p3/Z, [x21]
    ld1b { z18.b }, p2/Z, [x21, #1, MUL VL]
    zip1 z22.b, z21.b, z19.b
    zip2 z21.b, z21.b, z19.b
    ld1b { z28.b }, p1/Z, [x25, #2, MUL VL]
    zip1 z1.b, z24.b, z17.b
    zip2 z0.b, z24.b, z17.b
    ld1b { z27.b }, p0/Z, [x25, #3, MUL VL]
    zip1 z17.b, z23.b, z16.b
    zip2 z16.b, z23.b, z16.b
    addvl x25, x25, #4
    ld1b { z26.b }, p1/Z, [x24, #2, MUL VL]
    zip1 z31.b, z20.b, z18.b
    zip2 z30.b, z20.b, z18.b
    ld1b { z25.b }, p0/Z, [x24, #3, MUL VL]
    add x24, x24, x7
    ld1b { z20.b }, p1/Z, [x23, #2, MUL VL]
    ld1b { z19.b }, p0/Z, [x23, #3, MUL VL]
    zip1 z18.b, z22.b, z17.b
    zip2 z24.b, z22.b, z17.b
    add x23, x23, x20
    ld1b { z17.b }, p1/Z, [x21, #2, MUL VL]
    zip1 z23.b, z21.b, z16.b
    zip2 z22.b, z21.b, z16.b
    ld1b { z16.b }, p0/Z, [x21, #3, MUL VL]
    zip1 z21.b, z28.b, z20.b
    zip2 z29.b, z28.b, z20.b
    add x21, x21, x22
    zip1 z28.b, z27.b, z19.b
    zip2 z27.b, z27.b, z19.b
    zip1 z20.b, z26.b, z17.b
    zip2 z19.b, z26.b, z17.b
    st1b { z18.b }, p4, [x4]
    zip1 z18.b, z25.b, z16.b
    zip2 z26.b, z25.b, z16.b
    st1b { z24.b }, p4, [x4, #1, MUL VL]
    zip1 z17.b, z1.b, z31.b
    zip2 z16.b, z1.b, z31.b
    st1b { z23.b }, p4, [x4, #2, MUL VL]
    zip1 z25.b, z0.b, z30.b
    zip2 z24.b, z0.b, z30.b
    st1b { z22.b }, p4, [x4, #3, MUL VL]
    zip1 z23.b, z21.b, z20.b
    zip2 z22.b, z21.b, z20.b
    zip1 z21.b, z29.b, z19.b
    zip2 z20.b, z29.b, z19.b
    st1b { z17.b }, p4, [x4, #4, MUL VL]
    zip1 z19.b, z28.b, z18.b
    zip2 z18.b, z28.b, z18.b
    st1b { z16.b }, p4, [x4, #5, MUL VL]
    zip1 z17.b, z27.b, z26.b
    zip2 z16.b, z27.b, z26.b
    st1b { z25.b }, p4, [x4, #6, MUL VL]
    st1b { z24.b }, p4, [x4, #7, MUL VL]
    addvl x4, x4, #16
    st1b { z23.b }, p4, [x4, #-8, MUL VL]
    st1b { z22.b }, p4, [x4, #-7, MUL VL]
    st1b { z21.b }, p4, [x4, #-6, MUL VL]
    st1b { z20.b }, p4, [x4, #-5, MUL VL]
    st1b { z19.b }, p4, [x4, #-4, MUL VL]
    st1b { z18.b }, p4, [x4, #-3, MUL VL]
    st1b { z17.b }, p4, [x4, #-2, MUL VL]
    st1b { z16.b }, p4, [x4, #-1, MUL VL]
    bgt label_2
    sub x26, x26, #0x4
    addvl x0, x0, #16
    cmp x26, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #4
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_16VL_1x4)

    KAI_ASM_END
