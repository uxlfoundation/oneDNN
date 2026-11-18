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

    KAI_ASM_CODE(sve_transpose_interleave_4VL_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_4VL_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_4VL_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_4VL_1x4)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    cntb x6, ALL, MUL #4
    mov z16.b, #0
    ptrue p1.b
    addvl SP, SP, #-2
    mov x5, x4
    mov x28, x4
    lsr x6, x6, #0x2
    st1b { z16.b }, p1, [SP]
    st1b { z16.b }, p1, [SP, #1, MUL VL]
    mov x27, SP
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    cmp x28, #0x8
    mul x26, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x25, x1
    mov x24, x2
    cntb x8, ALL, MUL #2
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
    ld1b { z19.b }, p1/Z, [x25]
    ld1b { z18.b }, p1/Z, [x22]
    sub x24, x24, x8
    ld1b { z17.b }, p1/Z, [x21]
    ld1b { z16.b }, p1/Z, [x20]
    cmp x24, x8
    ld1b { z24.b }, p1/Z, [x7]
    ld1b { z23.b }, p1/Z, [x6]
    ld1b { z22.b }, p1/Z, [x5]
    ld1b { z21.b }, p1/Z, [x4]
    ld1b { z29.b }, p1/Z, [x25, #1, MUL VL]
    ld1b { z28.b }, p1/Z, [x22, #1, MUL VL]
    zip1 z4.b, z19.b, z17.b
    zip1 z3.b, z18.b, z16.b
    ld1b { z27.b }, p1/Z, [x21, #1, MUL VL]
    ld1b { z20.b }, p1/Z, [x20, #1, MUL VL]
    zip2 z2.b, z19.b, z17.b
    zip2 z1.b, z18.b, z16.b
    ld1b { z19.b }, p1/Z, [x7, #1, MUL VL]
    ld1b { z18.b }, p1/Z, [x6, #1, MUL VL]
    zip1 z26.b, z24.b, z22.b
    zip1 z25.b, z23.b, z21.b
    ld1b { z17.b }, p1/Z, [x5, #1, MUL VL]
    ld1b { z16.b }, p1/Z, [x4, #1, MUL VL]
    zip2 z24.b, z24.b, z22.b
    zip2 z23.b, z23.b, z21.b
    zip1 z22.b, z29.b, z27.b
    zip1 z21.b, z28.b, z20.b
    addvl x25, x25, #2
    addvl x22, x22, #2
    zip2 z0.b, z29.b, z27.b
    zip2 z31.b, z28.b, z20.b
    addvl x21, x21, #2
    addvl x20, x20, #2
    zip1 z30.b, z19.b, z17.b
    zip1 z29.b, z18.b, z16.b
    addvl x7, x7, #2
    addvl x6, x6, #2
    zip2 z28.b, z19.b, z17.b
    zip2 z27.b, z18.b, z16.b
    addvl x5, x5, #2
    addvl x4, x4, #2
    zip1 z20.b, z4.b, z3.b
    zip2 z19.b, z4.b, z3.b
    zip1 z18.b, z2.b, z1.b
    zip2 z17.b, z2.b, z1.b
    zip1 z16.b, z26.b, z25.b
    zip2 z26.b, z26.b, z25.b
    zip1 z25.b, z24.b, z23.b
    zip2 z24.b, z24.b, z23.b
    st1b { z20.b }, p1, [x23]
    st1b { z19.b }, p1, [x23, #1, MUL VL]
    zip1 z23.b, z22.b, z21.b
    zip2 z22.b, z22.b, z21.b
    st1b { z18.b }, p1, [x23, #2, MUL VL]
    zip1 z21.b, z0.b, z31.b
    zip2 z20.b, z0.b, z31.b
    st1b { z17.b }, p1, [x23, #3, MUL VL]
    zip1 z19.b, z30.b, z29.b
    zip2 z18.b, z30.b, z29.b
    st1b { z16.b }, p1, [x23, #4, MUL VL]
    zip1 z17.b, z28.b, z27.b
    zip2 z16.b, z28.b, z27.b
    st1b { z26.b }, p1, [x23, #5, MUL VL]
    st1b { z25.b }, p1, [x23, #6, MUL VL]
    st1b { z24.b }, p1, [x23, #7, MUL VL]
    add x23, x23, x26
    st1b { z23.b }, p1, [x23]
    st1b { z22.b }, p1, [x23, #1, MUL VL]
    st1b { z21.b }, p1, [x23, #2, MUL VL]
    st1b { z20.b }, p1, [x23, #3, MUL VL]
    st1b { z19.b }, p1, [x23, #4, MUL VL]
    st1b { z18.b }, p1, [x23, #5, MUL VL]
    st1b { z17.b }, p1, [x23, #6, MUL VL]
    st1b { z16.b }, p1, [x23, #7, MUL VL]
    add x23, x23, x26
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x24, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.b, XZR, x24
    decw x24, ALL, MUL #4
    ld1b { z20.b }, p0/Z, [x25]
    addvl x25, x25, #1
    ld1b { z24.b }, p0/Z, [x22]
    addvl x22, x22, #1
    ld1b { z19.b }, p0/Z, [x21]
    addvl x21, x21, #1
    ld1b { z18.b }, p0/Z, [x20]
    addvl x20, x20, #1
    ld1b { z23.b }, p0/Z, [x7]
    ld1b { z22.b }, p0/Z, [x6]
    ld1b { z17.b }, p0/Z, [x5]
    cmp x24, #0
    addvl x7, x7, #1
    ld1b { z16.b }, p0/Z, [x4]
    zip1 z21.b, z20.b, z19.b
    zip2 z25.b, z20.b, z19.b
    addvl x6, x6, #1
    zip1 z20.b, z24.b, z18.b
    zip2 z19.b, z24.b, z18.b
    addvl x5, x5, #1
    addvl x4, x4, #1
    zip1 z18.b, z23.b, z17.b
    zip2 z24.b, z23.b, z17.b
    zip1 z17.b, z22.b, z16.b
    zip2 z16.b, z22.b, z16.b
    zip1 z23.b, z21.b, z20.b
    zip2 z22.b, z21.b, z20.b
    zip1 z21.b, z25.b, z19.b
    zip2 z20.b, z25.b, z19.b
    zip1 z19.b, z18.b, z17.b
    zip2 z18.b, z18.b, z17.b
    zip1 z17.b, z24.b, z16.b
    zip2 z16.b, z24.b, z16.b
    st1b { z23.b }, p1, [x23]
    st1b { z22.b }, p1, [x23, #1, MUL VL]
    st1b { z21.b }, p1, [x23, #2, MUL VL]
    st1b { z20.b }, p1, [x23, #3, MUL VL]
    st1b { z19.b }, p1, [x23, #4, MUL VL]
    st1b { z18.b }, p1, [x23, #5, MUL VL]
    st1b { z17.b }, p1, [x23, #6, MUL VL]
    st1b { z16.b }, p1, [x23, #7, MUL VL]
    add x23, x23, x26
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x28, x28, #0x8
    addvl x0, x0, #8
    cmp x28, #0x8
    bge label_1
    cbz x28, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x25, x1
    cntw x4, ALL, MUL #8
    mov x8, x2
    cntb x7, ALL, MUL #2
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
    ld1b { z20.b }, p1/Z, [x25]
    ld1b { z24.b }, p1/Z, [x22]
    sub x8, x8, x7
    ld1b { z19.b }, p1/Z, [x21]
    ld1b { z18.b }, p1/Z, [x20]
    cmp x8, x7
    ld1b { z23.b }, p1/Z, [x25, #1, MUL VL]
    ld1b { z25.b }, p1/Z, [x22, #1, MUL VL]
    addvl x25, x25, #2
    add x22, x22, x4
    ld1b { z22.b }, p1/Z, [x21, #1, MUL VL]
    ld1b { z17.b }, p1/Z, [x20, #1, MUL VL]
    add x21, x21, x5
    add x20, x20, x6
    zip1 z21.b, z20.b, z19.b
    zip1 z16.b, z24.b, z18.b
    zip2 z20.b, z20.b, z19.b
    zip2 z19.b, z24.b, z18.b
    zip1 z24.b, z23.b, z22.b
    zip1 z18.b, z25.b, z17.b
    zip2 z23.b, z23.b, z22.b
    zip2 z22.b, z25.b, z17.b
    zip1 z17.b, z21.b, z16.b
    zip2 z16.b, z21.b, z16.b
    zip1 z21.b, z20.b, z19.b
    zip2 z20.b, z20.b, z19.b
    zip1 z19.b, z24.b, z18.b
    zip2 z18.b, z24.b, z18.b
    st1b { z17.b }, p1, [x23]
    st1b { z16.b }, p1, [x23, #1, MUL VL]
    zip1 z17.b, z23.b, z22.b
    zip2 z16.b, z23.b, z22.b
    st1b { z21.b }, p1, [x23, #2, MUL VL]
    st1b { z20.b }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    st1b { z19.b }, p1, [x23]
    st1b { z18.b }, p1, [x23, #1, MUL VL]
    st1b { z17.b }, p1, [x23, #2, MUL VL]
    st1b { z16.b }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x8, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    whilelt p0.b, XZR, x8
    decw x8, ALL, MUL #4
    ld1b { z20.b }, p0/Z, [x25]
    addvl x25, x25, #1
    ld1b { z19.b }, p0/Z, [x22]
    addvl x22, x22, #1
    ld1b { z17.b }, p0/Z, [x21]
    addvl x21, x21, #1
    ld1b { z16.b }, p0/Z, [x20]
    addvl x20, x20, #1
    cmp x8, #0
    zip1 z18.b, z20.b, z17.b
    zip2 z20.b, z20.b, z17.b
    zip1 z17.b, z19.b, z16.b
    zip2 z16.b, z19.b, z16.b
    zip1 z19.b, z18.b, z17.b
    zip2 z18.b, z18.b, z17.b
    zip1 z17.b, z20.b, z16.b
    zip2 z16.b, z20.b, z16.b
    st1b { z19.b }, p1, [x23]
    st1b { z18.b }, p1, [x23, #1, MUL VL]
    st1b { z17.b }, p1, [x23, #2, MUL VL]
    st1b { z16.b }, p1, [x23, #3, MUL VL]
    add x23, x23, x26
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x28, x28, #0x4
    addvl x0, x0, #4
    cmp x28, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    addvl SP, SP, #2
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_4VL_1x4)

    KAI_ASM_END
