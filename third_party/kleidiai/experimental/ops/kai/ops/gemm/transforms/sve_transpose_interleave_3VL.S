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

    KAI_ASM_CODE(sve_transpose_interleave_3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_3VL)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    cnth x6, ALL, MUL #3
    ptrue p2.b
    mov x5, x4
    mov x24, x4
    lsr x6, x6, #0
    lsl x5, x5, #0x1
    cmp x24, #0x4
    mul x23, x5, x6
    blt label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x22, x1
    mov x21, x0
    mov x20, x2
    add x8, x22, x3
    add x7, x8, x3
    add x6, x7, x3
    add x1, x6, x3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x5, x20
    mov x4, x21
    dech x20, ALL, MUL #3
    add x21, x21, x23
    whilelt p0.h, XZR, x5
    dech x5
    whilelt p1.h, XZR, x5
    dech x5
    ld1h { z27.h }, p0/Z, [x22]
    ld1h { z26.h }, p0/Z, [x8]
    ld1h { z25.h }, p0/Z, [x7]
    ld1h { z24.h }, p0/Z, [x6]
    whilelt p0.h, XZR, x5
    cmp x20, #0
    ld1h { z23.h }, p1/Z, [x22, #1, MUL VL]
    ld1h { z22.h }, p1/Z, [x8, #1, MUL VL]
    ld1h { z21.h }, p1/Z, [x7, #1, MUL VL]
    ld1h { z20.h }, p1/Z, [x6, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x22, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x8, #2, MUL VL]
    addvl x22, x22, #3
    addvl x8, x8, #3
    ld1h { z17.h }, p0/Z, [x7, #2, MUL VL]
    ld1h { z16.h }, p0/Z, [x6, #2, MUL VL]
    st1h { z27.h }, p2, [x4]
    addvl x7, x7, #3
    st1h { z23.h }, p2, [x4, #1, MUL VL]
    addvl x6, x6, #3
    st1h { z19.h }, p2, [x4, #2, MUL VL]
    st1h { z26.h }, p2, [x4, #3, MUL VL]
    st1h { z22.h }, p2, [x4, #4, MUL VL]
    st1h { z18.h }, p2, [x4, #5, MUL VL]
    st1h { z25.h }, p2, [x4, #6, MUL VL]
    st1h { z21.h }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1h { z17.h }, p2, [x4, #-4, MUL VL]
    st1h { z24.h }, p2, [x4, #-3, MUL VL]
    st1h { z20.h }, p2, [x4, #-2, MUL VL]
    st1h { z16.h }, p2, [x4, #-1, MUL VL]
    bgt label_2
    sub x24, x24, #0x4
    addvl x0, x0, #12
    cmp x24, #0x4
    bge label_1
    cbz x24, label_8
KAI_ASM_LABEL(label_4)  // Main loop skip
KAI_ASM_LABEL(label_5)  // Tail row loop: Head
    mov x22, x1
    mov x21, x0
    mov x5, x2
    add x1, x22, x3
KAI_ASM_LABEL(label_6)  // Tail row loop: Column loop
    mov x4, x5
    dech x5, ALL, MUL #3
    whilelt p0.h, XZR, x4
    dech x4
    whilelt p1.h, XZR, x4
    dech x4
    ld1h { z18.h }, p0/Z, [x22]
    whilelt p0.h, XZR, x4
    cmp x5, #0
    ld1h { z17.h }, p1/Z, [x22, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x22, #2, MUL VL]
    addvl x22, x22, #3
    st1h { z18.h }, p2, [x21]
    st1h { z17.h }, p2, [x21, #1, MUL VL]
    st1h { z16.h }, p2, [x21, #2, MUL VL]
    add x21, x21, x23
    bgt label_6
    sub x24, x24, #0x1
    addvl x0, x0, #3
    cmp x24, #0x1
    bge label_5
KAI_ASM_LABEL(label_8)  // Done
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_3VL)

    KAI_ASM_END
