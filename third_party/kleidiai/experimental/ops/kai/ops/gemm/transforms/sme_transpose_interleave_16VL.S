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

    KAI_ASM_CODE(sme_transpose_interleave_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_16VL)
    stp x20, x21, [sp, -80]!
    stp d8, d9, [sp, 16]
    stp d10, d11, [sp, 32]
    stp d12, d13, [sp, 48]
    stp d14, d15, [sp, 64]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    cnth x6, ALL, MUL #16
    mov x5, x4
    lsr x6, x6, #0
    lsl x5, x5, #0x1
    ptrue p7.b
    mul x21, x5, x6
    mov x20, x4
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x8, x1
    mov x7, x0
    add x1, x8, x3
    mov x6, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x5, x6
    mov x4, x7
    whilelt p0.h, XZR, x5
    dech x5
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z31.h }, p0/Z, [x8]
    whilelt p0.h, XZR, x5
    dech x5
    ld1h { z30.h }, p1/Z, [x8, #1, MUL VL]
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z29.h }, p0/Z, [x8, #2, MUL VL]
    whilelt p0.h, XZR, x5
    dech x5
    ld1h { z28.h }, p1/Z, [x8, #3, MUL VL]
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z27.h }, p0/Z, [x8, #4, MUL VL]
    whilelt p0.h, XZR, x5
    dech x5
    ld1h { z26.h }, p1/Z, [x8, #5, MUL VL]
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z25.h }, p0/Z, [x8, #6, MUL VL]
    whilelt p0.h, XZR, x5
    dech x5
    ld1h { z24.h }, p1/Z, [x8, #7, MUL VL]
    whilelt p6.h, XZR, x5
    dech x5
    whilelt p5.h, XZR, x5
    dech x5
    whilelt p4.h, XZR, x5
    dech x5
    whilelt p3.h, XZR, x5
    dech x5
    whilelt p2.h, XZR, x5
    dech x5
    whilelt p1.h, XZR, x5
    dech x5
    addvl x8, x8, #16
    dech x6, ALL, MUL #16
    ld1h { z23.h }, p0/Z, [x8, #-8, MUL VL]
    whilelt p0.h, XZR, x5
    cmp x6, #0
    ld1h { z22.h }, p6/Z, [x8, #-7, MUL VL]
    add x7, x7, x21
    ld1h { z21.h }, p5/Z, [x8, #-6, MUL VL]
    ld1h { z20.h }, p4/Z, [x8, #-5, MUL VL]
    ld1h { z19.h }, p3/Z, [x8, #-4, MUL VL]
    ld1h { z18.h }, p2/Z, [x8, #-3, MUL VL]
    ld1h { z17.h }, p1/Z, [x8, #-2, MUL VL]
    ld1h { z16.h }, p0/Z, [x8, #-1, MUL VL]
    st1h { z31.h }, p7, [x4]
    st1h { z30.h }, p7, [x4, #1, MUL VL]
    st1h { z29.h }, p7, [x4, #2, MUL VL]
    st1h { z28.h }, p7, [x4, #3, MUL VL]
    st1h { z27.h }, p7, [x4, #4, MUL VL]
    st1h { z26.h }, p7, [x4, #5, MUL VL]
    st1h { z25.h }, p7, [x4, #6, MUL VL]
    st1h { z24.h }, p7, [x4, #7, MUL VL]
    addvl x4, x4, #16
    st1h { z23.h }, p7, [x4, #-8, MUL VL]
    st1h { z22.h }, p7, [x4, #-7, MUL VL]
    st1h { z21.h }, p7, [x4, #-6, MUL VL]
    st1h { z20.h }, p7, [x4, #-5, MUL VL]
    st1h { z19.h }, p7, [x4, #-4, MUL VL]
    st1h { z18.h }, p7, [x4, #-3, MUL VL]
    st1h { z17.h }, p7, [x4, #-2, MUL VL]
    st1h { z16.h }, p7, [x4, #-1, MUL VL]
    bgt label_2
    sub x20, x20, #0x1
    addvl x0, x0, #16
    cmp x20, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d8, d9, [sp, 16]
    ldp d10, d11, [sp, 32]
    ldp d12, d13, [sp, 48]
    ldp d14, d15, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_16VL)

    KAI_ASM_END
