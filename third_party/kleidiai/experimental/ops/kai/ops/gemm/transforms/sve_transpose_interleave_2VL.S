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

    KAI_ASM_CODE(sve_transpose_interleave_2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_2VL)
    stp x20, x21, [sp, -32]!
    stp x22, x23, [sp, 16]
    cnth x6, ALL, MUL #2
    ptrue p2.b
    mov x5, x4
    mov x23, x4
    lsr x6, x6, #0
    lsl x5, x5, #0x1
    cmp x23, #0x4
    mul x22, x5, x6
    blt label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    mov x20, x0
    mov x8, x2
    add x7, x21, x3
    add x6, x7, x3
    add x5, x6, x3
    add x1, x5, x3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x4, x8
    dech x8, ALL, MUL #2
    whilelt p1.h, XZR, x4
    dech x4
    whilelt p0.h, XZR, x4
    cmp x8, #0
    ld1h { z23.h }, p1/Z, [x21]
    ld1h { z22.h }, p1/Z, [x7]
    ld1h { z21.h }, p1/Z, [x6]
    ld1h { z20.h }, p1/Z, [x5]
    ld1h { z19.h }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    ld1h { z18.h }, p0/Z, [x7, #1, MUL VL]
    addvl x7, x7, #2
    ld1h { z17.h }, p0/Z, [x6, #1, MUL VL]
    addvl x6, x6, #2
    ld1h { z16.h }, p0/Z, [x5, #1, MUL VL]
    addvl x5, x5, #2
    st1h { z23.h }, p2, [x20]
    st1h { z19.h }, p2, [x20, #1, MUL VL]
    st1h { z22.h }, p2, [x20, #2, MUL VL]
    st1h { z18.h }, p2, [x20, #3, MUL VL]
    st1h { z21.h }, p2, [x20, #4, MUL VL]
    st1h { z17.h }, p2, [x20, #5, MUL VL]
    st1h { z20.h }, p2, [x20, #6, MUL VL]
    st1h { z16.h }, p2, [x20, #7, MUL VL]
    add x20, x20, x22
    bgt label_2
    sub x23, x23, #0x4
    addvl x0, x0, #8
    cmp x23, #0x4
    bge label_1
    cbz x23, label_8
KAI_ASM_LABEL(label_4)  // Main loop skip
KAI_ASM_LABEL(label_5)  // Tail row loop: Head
    mov x21, x1
    mov x20, x0
    mov x5, x2
    add x1, x21, x3
KAI_ASM_LABEL(label_6)  // Tail row loop: Column loop
    mov x4, x5
    dech x5, ALL, MUL #2
    whilelt p1.h, XZR, x4
    dech x4
    whilelt p0.h, XZR, x4
    cmp x5, #0
    ld1h { z17.h }, p1/Z, [x21]
    ld1h { z16.h }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    st1h { z17.h }, p2, [x20]
    st1h { z16.h }, p2, [x20, #1, MUL VL]
    add x20, x20, x22
    bgt label_6
    sub x23, x23, #0x1
    addvl x0, x0, #2
    cmp x23, #0x1
    bge label_5
KAI_ASM_LABEL(label_8)  // Done
    ldp x22, x23, [sp, 16]
    ldp x20, x21, [sp], 32
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_2VL)

    KAI_ASM_END
