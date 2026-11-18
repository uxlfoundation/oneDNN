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

    KAI_ASM_CODE(sme_transpose_interleave_8VL_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_8VL_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_8VL_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_8VL_2x2)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cnth x5, ALL, MUL #8
    mov z16.h, #0
    add x6, x6, #0x1
    ptrue p4.b
    bfm x6, XZR, #0, #0  // bfc x6, #0, #0x1
    lsr x5, x5, #0x1
    lsl x6, x6, #0x1
    addvl SP, SP, #-4
    mul x23, x6, x5
    st1h { z16.h }, p4, [SP]
    mov x22, SP
    st1h { z16.h }, p4, [SP, #1, MUL VL]
    mov x21, x4
    st1h { z16.h }, p4, [SP, #2, MUL VL]
    st1h { z16.h }, p4, [SP, #3, MUL VL]
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x20, x1
    cntw x4, ALL, MUL #16
    add x8, x20, x3
    cmp x21, #0x1
    add x1, x8, x3
    mov x7, x0
    csel x8, x8, x22, GT
    csel x6, x4, XZR, GT
    mov x5, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x4, x5
    decw x5, ALL, MUL #8
    whilelt p3.h, XZR, x4
    dech x4
    whilelt p2.h, XZR, x4
    dech x4
    ld1h { z21.h }, p3/Z, [x20]
    whilelt p1.h, XZR, x4
    dech x4
    ld1h { z20.h }, p2/Z, [x20, #1, MUL VL]
    whilelt p0.h, XZR, x4
    ld1h { z25.h }, p1/Z, [x20, #2, MUL VL]
    cmp x5, #0
    ld1h { z24.h }, p0/Z, [x20, #3, MUL VL]
    addvl x20, x20, #4
    ld1h { z19.h }, p3/Z, [x8]
    ld1h { z18.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z17.h }, p1/Z, [x8, #2, MUL VL]
    ld1h { z16.h }, p0/Z, [x8, #3, MUL VL]
    add x8, x8, x6
    zip1 z23.h, z21.h, z19.h
    zip2 z22.h, z21.h, z19.h
    zip1 z21.h, z20.h, z18.h
    zip2 z20.h, z20.h, z18.h
    zip1 z19.h, z25.h, z17.h
    zip2 z18.h, z25.h, z17.h
    zip1 z17.h, z24.h, z16.h
    zip2 z16.h, z24.h, z16.h
    st1h { z23.h }, p4, [x7]
    st1h { z22.h }, p4, [x7, #1, MUL VL]
    st1h { z21.h }, p4, [x7, #2, MUL VL]
    st1h { z20.h }, p4, [x7, #3, MUL VL]
    st1h { z19.h }, p4, [x7, #4, MUL VL]
    st1h { z18.h }, p4, [x7, #5, MUL VL]
    st1h { z17.h }, p4, [x7, #6, MUL VL]
    st1h { z16.h }, p4, [x7, #7, MUL VL]
    add x7, x7, x23
    bgt label_2
    sub x21, x21, #0x2
    addvl x0, x0, #8
    cmp x21, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #4
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_8VL_2x2)

    KAI_ASM_END
