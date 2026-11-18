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

    KAI_ASM_CODE(sve_transpose_interleave_6VL_4x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_6VL_4x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_6VL_4x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_6VL_4x2)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    str d8, [sp, 56]
    cntw x6, ALL, MUL #6
    mov z16.s, #0
    ptrue p2.b
    addvl SP, SP, #-6
    mov x5, x4
    mov x26, x4
    lsr x6, x6, #0x1
    st1w { z16.s }, p2, [SP]
    st1w { z16.s }, p2, [SP, #1, MUL VL]
    mov x25, SP
    add x5, x5, #0x1
    bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1
    cmp x26, #0x4
    st1w { z16.s }, p2, [SP, #2, MUL VL]
    st1w { z16.s }, p2, [SP, #3, MUL VL]
    st1w { z16.s }, p2, [SP, #4, MUL VL]
    lsl x5, x5, #0x2
    st1w { z16.s }, p2, [SP, #5, MUL VL]
    mul x24, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x23, x1
    mov x22, x2
    cntw x21, ALL, MUL #6
    mov x20, x0
    add x8, x23, x3
    add x7, x8, x3
    add x6, x7, x3
    cmp x22, x21
    add x1, x6, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1w { z18.s }, p2/Z, [x23]
    ld1w { z28.s }, p2/Z, [x23, #1, MUL VL]
    mov x5, x20
    add x20, x20, x24
    ld1w { z22.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z17.s }, p2/Z, [x8]
    mov x4, x20
    sub x22, x22, x21
    ld1w { z16.s }, p2/Z, [x8, #1, MUL VL]
    ld1w { z21.s }, p2/Z, [x8, #2, MUL VL]
    cmp x22, x21
    add x20, x20, x24
    ld1w { z27.s }, p2/Z, [x7]
    ld1w { z26.s }, p2/Z, [x6]
    ld1w { z20.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z19.s }, p2/Z, [x23, #4, MUL VL]
    zip1 z8.s, z18.s, z17.s
    zip2 z25.s, z18.s, z17.s
    ld1w { z24.s }, p2/Z, [x23, #5, MUL VL]
    ld1w { z18.s }, p2/Z, [x8, #3, MUL VL]
    zip1 z23.s, z28.s, z16.s
    zip2 z7.s, z28.s, z16.s
    ld1w { z17.s }, p2/Z, [x8, #4, MUL VL]
    ld1w { z16.s }, p2/Z, [x8, #5, MUL VL]
    zip1 z6.s, z22.s, z21.s
    zip2 z5.s, z22.s, z21.s
    ld1w { z22.s }, p2/Z, [x7, #1, MUL VL]
    ld1w { z21.s }, p2/Z, [x7, #2, MUL VL]
    zip1 z4.s, z27.s, z26.s
    zip2 z3.s, z27.s, z26.s
    ld1w { z2.s }, p2/Z, [x7, #3, MUL VL]
    ld1w { z1.s }, p2/Z, [x7, #4, MUL VL]
    zip1 z0.s, z20.s, z18.s
    zip2 z31.s, z20.s, z18.s
    ld1w { z30.s }, p2/Z, [x7, #5, MUL VL]
    ld1w { z20.s }, p2/Z, [x6, #1, MUL VL]
    zip1 z29.s, z19.s, z17.s
    zip2 z28.s, z19.s, z17.s
    ld1w { z19.s }, p2/Z, [x6, #2, MUL VL]
    ld1w { z18.s }, p2/Z, [x6, #3, MUL VL]
    zip1 z27.s, z24.s, z16.s
    zip2 z26.s, z24.s, z16.s
    ld1w { z17.s }, p2/Z, [x6, #4, MUL VL]
    ld1w { z16.s }, p2/Z, [x6, #5, MUL VL]
    st1w { z8.s }, p2, [x5]
    addvl x23, x23, #6
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    zip1 z25.s, z22.s, z20.s
    zip2 z24.s, z22.s, z20.s
    addvl x8, x8, #6
    st1w { z23.s }, p2, [x5, #2, MUL VL]
    zip1 z23.s, z21.s, z19.s
    zip2 z22.s, z21.s, z19.s
    addvl x7, x7, #6
    st1w { z7.s }, p2, [x5, #3, MUL VL]
    zip1 z21.s, z2.s, z18.s
    zip2 z20.s, z2.s, z18.s
    addvl x6, x6, #6
    st1w { z6.s }, p2, [x5, #4, MUL VL]
    zip1 z19.s, z1.s, z17.s
    zip2 z18.s, z1.s, z17.s
    st1w { z5.s }, p2, [x5, #5, MUL VL]
    zip1 z17.s, z30.s, z16.s
    zip2 z16.s, z30.s, z16.s
    st1w { z4.s }, p2, [x5, #6, MUL VL]
    st1w { z3.s }, p2, [x5, #7, MUL VL]
    addvl x5, x5, #12
    st1w { z25.s }, p2, [x5, #-4, MUL VL]
    st1w { z24.s }, p2, [x5, #-3, MUL VL]
    st1w { z23.s }, p2, [x5, #-2, MUL VL]
    st1w { z22.s }, p2, [x5, #-1, MUL VL]
    st1w { z0.s }, p2, [x4]
    st1w { z31.s }, p2, [x4, #1, MUL VL]
    st1w { z29.s }, p2, [x4, #2, MUL VL]
    st1w { z28.s }, p2, [x4, #3, MUL VL]
    st1w { z27.s }, p2, [x4, #4, MUL VL]
    st1w { z26.s }, p2, [x4, #5, MUL VL]
    st1w { z21.s }, p2, [x4, #6, MUL VL]
    st1w { z20.s }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1w { z19.s }, p2, [x4, #-4, MUL VL]
    st1w { z18.s }, p2, [x4, #-3, MUL VL]
    st1w { z17.s }, p2, [x4, #-2, MUL VL]
    st1w { z16.s }, p2, [x4, #-1, MUL VL]
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x22, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x5, x22
    mov x4, x20
    decd x22, ALL, MUL #6
    add x20, x20, x24
    whilelt p0.s, XZR, x5
    decw x5
    whilelt p1.s, XZR, x5
    decw x5
    ld1w { z19.s }, p0/Z, [x23]
    ld1w { z18.s }, p0/Z, [x8]
    ld1w { z22.s }, p0/Z, [x7]
    ld1w { z17.s }, p0/Z, [x6]
    whilelt p0.s, XZR, x5
    cmp x22, #0
    ld1w { z21.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x8, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x7, #1, MUL VL]
    ld1w { z27.s }, p1/Z, [x6, #1, MUL VL]
    zip1 z20.s, z19.s, z18.s
    zip2 z26.s, z19.s, z18.s
    ld1w { z19.s }, p0/Z, [x23, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x8, #2, MUL VL]
    zip1 z25.s, z22.s, z17.s
    zip2 z24.s, z22.s, z17.s
    ld1w { z23.s }, p0/Z, [x7, #2, MUL VL]
    ld1w { z22.s }, p0/Z, [x6, #2, MUL VL]
    zip1 z17.s, z21.s, z16.s
    zip2 z16.s, z21.s, z16.s
    st1w { z20.s }, p2, [x4]
    addvl x23, x23, #3
    addvl x8, x8, #3
    zip1 z21.s, z28.s, z27.s
    zip1 z20.s, z19.s, z18.s
    zip2 z19.s, z19.s, z18.s
    st1w { z26.s }, p2, [x4, #1, MUL VL]
    addvl x7, x7, #3
    st1w { z17.s }, p2, [x4, #2, MUL VL]
    addvl x6, x6, #3
    zip2 z18.s, z28.s, z27.s
    zip1 z17.s, z23.s, z22.s
    st1w { z16.s }, p2, [x4, #3, MUL VL]
    zip2 z16.s, z23.s, z22.s
    st1w { z20.s }, p2, [x4, #4, MUL VL]
    st1w { z19.s }, p2, [x4, #5, MUL VL]
    st1w { z25.s }, p2, [x4, #6, MUL VL]
    st1w { z24.s }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1w { z21.s }, p2, [x4, #-4, MUL VL]
    st1w { z18.s }, p2, [x4, #-3, MUL VL]
    st1w { z17.s }, p2, [x4, #-2, MUL VL]
    st1w { z16.s }, p2, [x4, #-1, MUL VL]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x26, x26, #0x4
    addvl x0, x0, #12
    cmp x26, #0x4
    bge label_1
    cbz x26, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    cntw x4, ALL, MUL #12
    mov x23, x1
    mov x6, x2
    cntw x5, ALL, MUL #6
    cmp x26, #0x1
    mov x20, x0
    lsl x4, x4, #0x1
    add x8, x23, x3
    add x1, x8, x3
    csel x8, x8, x25, GT
    csel x4, x4, XZR, GT
    cmp x6, x5
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Unroll column loop
    ld1w { z17.s }, p2/Z, [x23]
    ld1w { z23.s }, p2/Z, [x23, #1, MUL VL]
    sub x6, x6, x5
    ld1w { z29.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p2/Z, [x8]
    cmp x6, x5
    ld1w { z22.s }, p2/Z, [x8, #1, MUL VL]
    ld1w { z21.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z28.s }, p2/Z, [x23, #3, MUL VL]
    ld1w { z27.s }, p2/Z, [x23, #4, MUL VL]
    ld1w { z26.s }, p2/Z, [x23, #5, MUL VL]
    ld1w { z20.s }, p2/Z, [x8, #3, MUL VL]
    zip1 z19.s, z17.s, z16.s
    zip2 z18.s, z17.s, z16.s
    ld1w { z17.s }, p2/Z, [x8, #4, MUL VL]
    ld1w { z25.s }, p2/Z, [x8, #5, MUL VL]
    zip1 z16.s, z23.s, z22.s
    zip2 z24.s, z23.s, z22.s
    zip1 z23.s, z29.s, z21.s
    zip2 z22.s, z29.s, z21.s
    addvl x23, x23, #6
    add x8, x8, x4
    st1w { z19.s }, p2, [x20]
    zip1 z21.s, z28.s, z20.s
    zip2 z20.s, z28.s, z20.s
    st1w { z18.s }, p2, [x20, #1, MUL VL]
    zip1 z19.s, z27.s, z17.s
    zip2 z18.s, z27.s, z17.s
    st1w { z16.s }, p2, [x20, #2, MUL VL]
    zip1 z17.s, z26.s, z25.s
    zip2 z16.s, z26.s, z25.s
    st1w { z24.s }, p2, [x20, #3, MUL VL]
    st1w { z23.s }, p2, [x20, #4, MUL VL]
    st1w { z22.s }, p2, [x20, #5, MUL VL]
    add x20, x20, x24
    st1w { z21.s }, p2, [x20]
    st1w { z20.s }, p2, [x20, #1, MUL VL]
    st1w { z19.s }, p2, [x20, #2, MUL VL]
    st1w { z18.s }, p2, [x20, #3, MUL VL]
    st1w { z17.s }, p2, [x20, #4, MUL VL]
    st1w { z16.s }, p2, [x20, #5, MUL VL]
    add x20, x20, x24
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x6, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    mov x4, x6
    decd x6, ALL, MUL #6
    whilelt p0.s, XZR, x4
    decw x4
    whilelt p1.s, XZR, x4
    decw x4
    ld1w { z17.s }, p0/Z, [x23]
    ld1w { z16.s }, p0/Z, [x8]
    whilelt p0.s, XZR, x4
    cmp x6, #0
    ld1w { z22.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x8, #1, MUL VL]
    zip1 z21.s, z17.s, z16.s
    zip2 z17.s, z17.s, z16.s
    ld1w { z20.s }, p0/Z, [x23, #2, MUL VL]
    addvl x23, x23, #3
    ld1w { z16.s }, p0/Z, [x8, #2, MUL VL]
    addvl x8, x8, #3
    zip1 z19.s, z22.s, z18.s
    zip2 z18.s, z22.s, z18.s
    st1w { z21.s }, p2, [x20]
    st1w { z17.s }, p2, [x20, #1, MUL VL]
    zip1 z17.s, z20.s, z16.s
    zip2 z16.s, z20.s, z16.s
    st1w { z19.s }, p2, [x20, #2, MUL VL]
    st1w { z18.s }, p2, [x20, #3, MUL VL]
    st1w { z17.s }, p2, [x20, #4, MUL VL]
    st1w { z16.s }, p2, [x20, #5, MUL VL]
    add x20, x20, x24
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x26, x26, #0x2
    addvl x0, x0, #6
    cmp x26, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    addvl SP, SP, #6
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldr d8, [sp, 56]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_6VL_4x2)

    KAI_ASM_END
