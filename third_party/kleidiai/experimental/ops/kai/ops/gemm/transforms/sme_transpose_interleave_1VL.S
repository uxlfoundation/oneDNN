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

    KAI_ASM_CODE(sme_transpose_interleave_1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_1VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    cnth x6
    mov x5, x4
    mov x23, x4
    lsr x6, x6, #0
    lsl x5, x5, #0x1
    cmp x23, #0x4
    ptrue p1.b
    mul x22, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    mov x20, x2
    add x8, x21, x3
    cnth x7, ALL, MUL #4
    add x5, x8, x3
    cmp x20, x7
    add x4, x5, x3
    mov x6, x0
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1h { z31.h }, p1/Z, [x21]
    sub x20, x20, x7
    ld1h { z30.h }, p1/Z, [x21, #1, MUL VL]
    cmp x20, x7
    ld1h { z29.h }, p1/Z, [x21, #2, MUL VL]
    ld1h { z28.h }, p1/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    ld1h { z27.h }, p1/Z, [x8]
    ld1h { z26.h }, p1/Z, [x8, #1, MUL VL]
    ld1h { z25.h }, p1/Z, [x8, #2, MUL VL]
    ld1h { z24.h }, p1/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    ld1h { z23.h }, p1/Z, [x5]
    ld1h { z22.h }, p1/Z, [x5, #1, MUL VL]
    ld1h { z21.h }, p1/Z, [x5, #2, MUL VL]
    ld1h { z20.h }, p1/Z, [x5, #3, MUL VL]
    addvl x5, x5, #4
    ld1h { z19.h }, p1/Z, [x4]
    ld1h { z18.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z17.h }, p1/Z, [x4, #2, MUL VL]
    ld1h { z16.h }, p1/Z, [x4, #3, MUL VL]
    st1h { z31.h }, p1, [x6]
    addvl x4, x4, #4
    st1h { z27.h }, p1, [x6, #1, MUL VL]
    st1h { z23.h }, p1, [x6, #2, MUL VL]
    st1h { z19.h }, p1, [x6, #3, MUL VL]
    add x6, x6, x22
    st1h { z30.h }, p1, [x6]
    st1h { z26.h }, p1, [x6, #1, MUL VL]
    st1h { z22.h }, p1, [x6, #2, MUL VL]
    st1h { z18.h }, p1, [x6, #3, MUL VL]
    add x6, x6, x22
    st1h { z29.h }, p1, [x6]
    st1h { z25.h }, p1, [x6, #1, MUL VL]
    st1h { z21.h }, p1, [x6, #2, MUL VL]
    st1h { z17.h }, p1, [x6, #3, MUL VL]
    add x6, x6, x22
    st1h { z28.h }, p1, [x6]
    st1h { z24.h }, p1, [x6, #1, MUL VL]
    st1h { z20.h }, p1, [x6, #2, MUL VL]
    st1h { z16.h }, p1, [x6, #3, MUL VL]
    add x6, x6, x22
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x20, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.h, XZR, x20
    dech x20
    cmp x20, #0
    ld1h { z19.h }, p0/Z, [x21]
    addvl x21, x21, #1
    ld1h { z18.h }, p0/Z, [x8]
    addvl x8, x8, #1
    ld1h { z17.h }, p0/Z, [x5]
    addvl x5, x5, #1
    ld1h { z16.h }, p0/Z, [x4]
    addvl x4, x4, #1
    st1h { z19.h }, p1, [x6]
    st1h { z18.h }, p1, [x6, #1, MUL VL]
    st1h { z17.h }, p1, [x6, #2, MUL VL]
    st1h { z16.h }, p1, [x6, #3, MUL VL]
    add x6, x6, x22
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x23, x23, #0x4
    addvl x0, x0, #4
    cmp x23, #0x4
    bge label_1
    cbz x23, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x5, x2
    cnth x4, ALL, MUL #4
    mov x21, x1
    cmp x5, x4
    add x1, x21, x3
    mov x6, x0
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Unroll column loop
    ld1h { z19.h }, p1/Z, [x21]
    sub x5, x5, x4
    ld1h { z18.h }, p1/Z, [x21, #1, MUL VL]
    cmp x5, x4
    ld1h { z17.h }, p1/Z, [x21, #2, MUL VL]
    ld1h { z16.h }, p1/Z, [x21, #3, MUL VL]
    st1h { z19.h }, p1, [x6]
    add x6, x6, x22
    addvl x21, x21, #4
    st1h { z18.h }, p1, [x6]
    add x6, x6, x22
    st1h { z17.h }, p1, [x6]
    add x6, x6, x22
    st1h { z16.h }, p1, [x6]
    add x6, x6, x22
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x5, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    whilelt p0.h, XZR, x5
    dech x5
    cmp x5, #0
    ld1h { z16.h }, p0/Z, [x21]
    addvl x21, x21, #1
    st1h { z16.h }, p1, [x6]
    add x6, x6, x22
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x23, x23, #0x1
    addvl x0, x0, #1
    cmp x23, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_1VL)

    KAI_ASM_END
