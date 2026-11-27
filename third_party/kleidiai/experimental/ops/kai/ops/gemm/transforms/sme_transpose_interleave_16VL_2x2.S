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

    KAI_ASM_CODE(sme_transpose_interleave_16VL_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_16VL_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_16VL_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_16VL_2x2)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cnth x5, ALL, MUL #16
    mov z16.h, #0
    add x6, x6, #0x1
    ptrue p6.b
    bfm x6, XZR, #0, #0  // bfc x6, #0, #0x1
    lsr x5, x5, #0x1
    lsl x6, x6, #0x1
    addvl SP, SP, #-8
    mul x24, x6, x5
    st1h { z16.h }, p6, [SP]
    mov x23, SP
    st1h { z16.h }, p6, [SP, #1, MUL VL]
    mov x22, x4
    st1h { z16.h }, p6, [SP, #2, MUL VL]
    st1h { z16.h }, p6, [SP, #3, MUL VL]
    st1h { z16.h }, p6, [SP, #4, MUL VL]
    st1h { z16.h }, p6, [SP, #5, MUL VL]
    st1h { z16.h }, p6, [SP, #6, MUL VL]
    st1h { z16.h }, p6, [SP, #7, MUL VL]
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    cntw x4, ALL, MUL #16
    add x20, x21, x3
    lsl x4, x4, #0x1
    cmp x22, #0x1
    add x1, x20, x3
    mov x8, x0
    csel x20, x20, x23, GT
    csel x7, x4, XZR, GT
    mov x6, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x5, x6
    mov x4, x8
    whilelt p1.h, XZR, x5
    dech x5
    whilelt p0.h, XZR, x5
    dech x5
    ld1h { z21.h }, p1/Z, [x21]
    whilelt p5.h, XZR, x5
    dech x5
    ld1h { z20.h }, p0/Z, [x21, #1, MUL VL]
    whilelt p4.h, XZR, x5
    dech x5
    ld1h { z25.h }, p5/Z, [x21, #2, MUL VL]
    whilelt p3.h, XZR, x5
    dech x5
    ld1h { z24.h }, p4/Z, [x21, #3, MUL VL]
    whilelt p2.h, XZR, x5
    dech x5
    ld1h { z19.h }, p1/Z, [x20]
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z18.h }, p0/Z, [x20, #1, MUL VL]
    whilelt p0.h, XZR, x5
    ld1h { z17.h }, p5/Z, [x20, #2, MUL VL]
    decw x6, ALL, MUL #16
    ld1h { z16.h }, p4/Z, [x20, #3, MUL VL]
    zip1 z23.h, z21.h, z19.h
    zip2 z22.h, z21.h, z19.h
    cmp x6, #0
    ld1h { z21.h }, p3/Z, [x21, #4, MUL VL]
    zip1 z31.h, z20.h, z18.h
    zip2 z30.h, z20.h, z18.h
    add x8, x8, x24
    ld1h { z20.h }, p2/Z, [x21, #5, MUL VL]
    zip1 z29.h, z25.h, z17.h
    zip2 z28.h, z25.h, z17.h
    ld1h { z27.h }, p1/Z, [x21, #6, MUL VL]
    zip1 z26.h, z24.h, z16.h
    zip2 z25.h, z24.h, z16.h
    ld1h { z24.h }, p0/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    ld1h { z19.h }, p3/Z, [x20, #4, MUL VL]
    ld1h { z18.h }, p2/Z, [x20, #5, MUL VL]
    ld1h { z17.h }, p1/Z, [x20, #6, MUL VL]
    ld1h { z16.h }, p0/Z, [x20, #7, MUL VL]
    st1h { z23.h }, p6, [x4]
    add x20, x20, x7
    zip1 z23.h, z21.h, z19.h
    st1h { z22.h }, p6, [x4, #1, MUL VL]
    zip2 z22.h, z21.h, z19.h
    zip1 z21.h, z20.h, z18.h
    st1h { z31.h }, p6, [x4, #2, MUL VL]
    zip2 z20.h, z20.h, z18.h
    zip1 z19.h, z27.h, z17.h
    st1h { z30.h }, p6, [x4, #3, MUL VL]
    zip2 z18.h, z27.h, z17.h
    zip1 z17.h, z24.h, z16.h
    st1h { z29.h }, p6, [x4, #4, MUL VL]
    zip2 z16.h, z24.h, z16.h
    st1h { z28.h }, p6, [x4, #5, MUL VL]
    st1h { z26.h }, p6, [x4, #6, MUL VL]
    st1h { z25.h }, p6, [x4, #7, MUL VL]
    addvl x4, x4, #16
    st1h { z23.h }, p6, [x4, #-8, MUL VL]
    st1h { z22.h }, p6, [x4, #-7, MUL VL]
    st1h { z21.h }, p6, [x4, #-6, MUL VL]
    st1h { z20.h }, p6, [x4, #-5, MUL VL]
    st1h { z19.h }, p6, [x4, #-4, MUL VL]
    st1h { z18.h }, p6, [x4, #-3, MUL VL]
    st1h { z17.h }, p6, [x4, #-2, MUL VL]
    st1h { z16.h }, p6, [x4, #-1, MUL VL]
    bgt label_2
    sub x22, x22, #0x2
    addvl x0, x0, #16
    cmp x22, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #8
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_16VL_2x2)

    KAI_ASM_END
