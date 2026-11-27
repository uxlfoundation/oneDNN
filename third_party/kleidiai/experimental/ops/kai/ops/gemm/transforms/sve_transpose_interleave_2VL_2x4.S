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

    KAI_ASM_CODE(sve_transpose_interleave_2VL_2x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_2VL_2x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_2VL_2x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_2VL_2x4)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    cnth x6, ALL, MUL #2
    mov z16.h, #0
    ptrue p1.b
    addvl SP, SP, #-1
    mov x5, x4
    mov x28, x4
    lsr x6, x6, #0x2
    st1h { z16.h }, p1, [SP]
    mov x27, SP
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    cmp x28, #0x8
    lsl x5, x5, #0x1
    mul x26, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x25, x1
    mov x24, x2
    cnth x8
    mov x23, x0
    add x22, x25, x3
    add x21, x22, x3
    add x20, x21, x3
    add x7, x20, x3
    add x6, x7, x3
    cmp x24, x8
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1h { z25.h }, p1/Z, [x25]
    ld1h { z24.h }, p1/Z, [x22]
    sub x24, x24, x8
    addvl x25, x25, #1
    ld1h { z19.h }, p1/Z, [x21]
    ld1h { z18.h }, p1/Z, [x20]
    cmp x24, x8
    addvl x22, x22, #1
    ld1h { z23.h }, p1/Z, [x7]
    ld1h { z22.h }, p1/Z, [x6]
    addvl x21, x21, #1
    addvl x20, x20, #1
    ld1h { z21.h }, p1/Z, [x5]
    ld1h { z17.h }, p1/Z, [x4]
    addvl x7, x7, #1
    addvl x6, x6, #1
    zip1 z20.h, z25.h, z19.h
    zip1 z16.h, z24.h, z18.h
    addvl x5, x5, #1
    addvl x4, x4, #1
    zip2 z25.h, z25.h, z19.h
    zip2 z24.h, z24.h, z18.h
    zip1 z19.h, z23.h, z21.h
    zip1 z18.h, z22.h, z17.h
    zip2 z23.h, z23.h, z21.h
    zip2 z22.h, z22.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    zip1 z21.h, z19.h, z18.h
    zip2 z20.h, z19.h, z18.h
    zip1 z19.h, z25.h, z24.h
    zip2 z18.h, z25.h, z24.h
    st1h { z17.h }, p1, [x23]
    st1h { z16.h }, p1, [x23, #1, MUL VL]
    zip1 z17.h, z23.h, z22.h
    zip2 z16.h, z23.h, z22.h
    st1h { z21.h }, p1, [x23, #2, MUL VL]
    st1h { z20.h }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    st1h { z19.h }, p1, [x23]
    st1h { z18.h }, p1, [x23, #1, MUL VL]
    st1h { z17.h }, p1, [x23, #2, MUL VL]
    st1h { z16.h }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x24, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.h, XZR, x24
    decd x24, ALL, MUL #2
    ld1h { z23.h }, p0/Z, [x25]
    incd x25, ALL, MUL #4
    ld1h { z22.h }, p0/Z, [x22]
    incd x22, ALL, MUL #4
    ld1h { z18.h }, p0/Z, [x21]
    incd x21, ALL, MUL #4
    ld1h { z17.h }, p0/Z, [x20]
    incd x20, ALL, MUL #4
    ld1h { z20.h }, p0/Z, [x7]
    ld1h { z21.h }, p0/Z, [x6]
    ld1h { z19.h }, p0/Z, [x5]
    cmp x24, #0
    incd x7, ALL, MUL #4
    ld1h { z16.h }, p0/Z, [x4]
    zip1 z18.h, z23.h, z18.h
    incd x6, ALL, MUL #4
    incd x5, ALL, MUL #4
    zip1 z17.h, z22.h, z17.h
    incd x4, ALL, MUL #4
    zip1 z20.h, z20.h, z19.h
    zip1 z16.h, z21.h, z16.h
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    st1h { z19.h }, p1, [x23]
    st1h { z18.h }, p1, [x23, #1, MUL VL]
    st1h { z17.h }, p1, [x23, #2, MUL VL]
    st1h { z16.h }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x28, x28, #0x8
    addvl x0, x0, #4
    cmp x28, #0x8
    bge label_1
    cbz x28, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x25, x1
    cntw x4, ALL, MUL #4
    mov x8, x2
    cnth x7
    cmp x28, #0x3
    mov x23, x0
    add x22, x25, x3
    csel x6, x4, XZR, GT
    add x21, x22, x3
    csel x5, x4, XZR, GE
    add x20, x21, x3
    csel x21, x21, x27, GE
    add x1, x20, x3
    csel x20, x20, x27, GT
    cmp x28, #0x1
    csel x22, x22, x27, GT
    csel x4, x4, XZR, GT
    cmp x8, x7
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Unroll column loop
    ld1h { z20.h }, p1/Z, [x25]
    ld1h { z21.h }, p1/Z, [x22]
    sub x8, x8, x7
    addvl x25, x25, #1
    ld1h { z19.h }, p1/Z, [x21]
    ld1h { z16.h }, p1/Z, [x20]
    cmp x8, x7
    add x22, x22, x4
    add x21, x21, x5
    add x20, x20, x6
    zip1 z18.h, z20.h, z19.h
    zip1 z17.h, z21.h, z16.h
    zip2 z20.h, z20.h, z19.h
    zip2 z16.h, z21.h, z16.h
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    st1h { z19.h }, p1, [x23]
    st1h { z18.h }, p1, [x23, #1, MUL VL]
    add x23, x23, x26
    st1h { z17.h }, p1, [x23]
    st1h { z16.h }, p1, [x23, #1, MUL VL]
    add x23, x23, x26
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x8, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    whilelt p0.h, XZR, x8
    decd x8, ALL, MUL #2
    ld1h { z18.h }, p0/Z, [x25]
    incd x25, ALL, MUL #4
    ld1h { z19.h }, p0/Z, [x22]
    incd x22, ALL, MUL #4
    ld1h { z17.h }, p0/Z, [x21]
    incd x21, ALL, MUL #4
    ld1h { z16.h }, p0/Z, [x20]
    incd x20, ALL, MUL #4
    cmp x8, #0
    zip1 z18.h, z18.h, z17.h
    zip1 z16.h, z19.h, z16.h
    zip1 z17.h, z18.h, z16.h
    zip2 z16.h, z18.h, z16.h
    st1h { z17.h }, p1, [x23]
    st1h { z16.h }, p1, [x23, #1, MUL VL]
    add x23, x23, x26
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x28, x28, #0x4
    addvl x0, x0, #2
    cmp x28, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    addvl SP, SP, #1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_2VL_2x4)

    KAI_ASM_END
