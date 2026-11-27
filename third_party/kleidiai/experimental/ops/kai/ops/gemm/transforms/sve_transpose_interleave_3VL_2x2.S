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

    KAI_ASM_CODE(sve_transpose_interleave_3VL_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_3VL_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_3VL_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_3VL_2x2)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    str d8, [sp, 72]
    cnth x6, ALL, MUL #3
    mov z16.h, #0
    ptrue p2.b
    addvl SP, SP, #-3
    mov x5, x4
    mov x10, x4
    lsr x6, x6, #0x1
    st1h { z16.h }, p2, [SP]
    st1h { z16.h }, p2, [SP, #1, MUL VL]
    mov x9, SP
    add x5, x5, #0x1
    bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1
    cmp x10, #0x8
    st1h { z16.h }, p2, [SP, #2, MUL VL]
    lsl x5, x5, #0x1
    mul x28, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x27, x1
    mov x26, x2
    cnth x25, ALL, MUL #3
    mov x24, x0
    add x23, x27, x3
    add x22, x23, x3
    add x21, x22, x3
    add x20, x21, x3
    add x8, x20, x3
    cmp x26, x25
    add x7, x8, x3
    add x6, x7, x3
    add x1, x6, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1h { z18.h }, p2/Z, [x27]
    ld1h { z29.h }, p2/Z, [x27, #1, MUL VL]
    mov x5, x24
    add x24, x24, x28
    ld1h { z17.h }, p2/Z, [x23]
    ld1h { z16.h }, p2/Z, [x23, #1, MUL VL]
    mov x4, x24
    sub x26, x26, x25
    ld1h { z21.h }, p2/Z, [x22]
    ld1h { z28.h }, p2/Z, [x22, #1, MUL VL]
    cmp x26, x25
    add x24, x24, x28
    ld1h { z20.h }, p2/Z, [x21]
    ld1h { z27.h }, p2/Z, [x21, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x20]
    ld1h { z19.h }, p2/Z, [x8]
    zip1 z25.h, z18.h, z17.h
    zip2 z24.h, z18.h, z17.h
    ld1h { z23.h }, p2/Z, [x27, #2, MUL VL]
    ld1h { z18.h }, p2/Z, [x23, #2, MUL VL]
    zip1 z22.h, z29.h, z16.h
    zip2 z8.h, z29.h, z16.h
    ld1h { z17.h }, p2/Z, [x22, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x21, #2, MUL VL]
    zip1 z7.h, z21.h, z20.h
    zip2 z6.h, z21.h, z20.h
    ld1h { z21.h }, p2/Z, [x20, #1, MUL VL]
    ld1h { z5.h }, p2/Z, [x20, #2, MUL VL]
    zip1 z4.h, z28.h, z27.h
    zip1 z3.h, z26.h, z19.h
    ld1h { z20.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z2.h }, p2/Z, [x8, #2, MUL VL]
    zip2 z1.h, z26.h, z19.h
    zip1 z0.h, z23.h, z18.h
    ld1h { z19.h }, p2/Z, [x7]
    ld1h { z31.h }, p2/Z, [x7, #1, MUL VL]
    zip2 z30.h, z23.h, z18.h
    zip2 z29.h, z28.h, z27.h
    ld1h { z28.h }, p2/Z, [x7, #2, MUL VL]
    ld1h { z18.h }, p2/Z, [x6]
    zip1 z27.h, z17.h, z16.h
    zip2 z26.h, z17.h, z16.h
    ld1h { z17.h }, p2/Z, [x6, #1, MUL VL]
    ld1h { z16.h }, p2/Z, [x6, #2, MUL VL]
    st1h { z25.h }, p2, [x5]
    zip1 z25.h, z21.h, z20.h
    st1h { z24.h }, p2, [x5, #1, MUL VL]
    zip2 z24.h, z21.h, z20.h
    zip1 z23.h, z5.h, z2.h
    addvl x27, x27, #3
    st1h { z22.h }, p2, [x5, #2, MUL VL]
    zip1 z22.h, z19.h, z18.h
    zip2 z21.h, z19.h, z18.h
    addvl x23, x23, #3
    st1h { z7.h }, p2, [x5, #3, MUL VL]
    zip1 z20.h, z31.h, z17.h
    addvl x22, x22, #3
    addvl x21, x21, #3
    st1h { z6.h }, p2, [x5, #4, MUL VL]
    addvl x20, x20, #3
    addvl x8, x8, #3
    zip2 z19.h, z5.h, z2.h
    st1h { z4.h }, p2, [x5, #5, MUL VL]
    addvl x7, x7, #3
    addvl x6, x6, #3
    zip2 z18.h, z31.h, z17.h
    st1h { z3.h }, p2, [x5, #6, MUL VL]
    zip1 z17.h, z28.h, z16.h
    zip2 z16.h, z28.h, z16.h
    st1h { z1.h }, p2, [x5, #7, MUL VL]
    addvl x5, x5, #12
    st1h { z25.h }, p2, [x5, #-4, MUL VL]
    st1h { z22.h }, p2, [x5, #-3, MUL VL]
    st1h { z21.h }, p2, [x5, #-2, MUL VL]
    st1h { z20.h }, p2, [x5, #-1, MUL VL]
    st1h { z8.h }, p2, [x4]
    st1h { z0.h }, p2, [x4, #1, MUL VL]
    st1h { z30.h }, p2, [x4, #2, MUL VL]
    st1h { z29.h }, p2, [x4, #3, MUL VL]
    st1h { z27.h }, p2, [x4, #4, MUL VL]
    st1h { z26.h }, p2, [x4, #5, MUL VL]
    st1h { z24.h }, p2, [x4, #6, MUL VL]
    st1h { z23.h }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1h { z19.h }, p2, [x4, #-4, MUL VL]
    st1h { z18.h }, p2, [x4, #-3, MUL VL]
    st1h { z17.h }, p2, [x4, #-2, MUL VL]
    st1h { z16.h }, p2, [x4, #-1, MUL VL]
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x26, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x5, x26
    mov x4, x24
    decw x26, ALL, MUL #3
    add x24, x24, x28
    whilelt p1.h, XZR, x5
    dech x5
    whilelt p0.h, XZR, x5
    cmp x26, #0
    ld1h { z19.h }, p1/Z, [x27]
    ld1h { z18.h }, p1/Z, [x23]
    ld1h { z0.h }, p1/Z, [x22]
    ld1h { z31.h }, p1/Z, [x21]
    ld1h { z30.h }, p1/Z, [x20]
    ld1h { z29.h }, p1/Z, [x8]
    ld1h { z17.h }, p0/Z, [x27, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x23, #1, MUL VL]
    addvl x27, x27, #1
    addvl x23, x23, #1
    ld1h { z22.h }, p0/Z, [x22, #1, MUL VL]
    ld1h { z21.h }, p0/Z, [x21, #1, MUL VL]
    addvl x22, x22, #1
    addvl x21, x21, #1
    ld1h { z28.h }, p0/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    ld1h { z27.h }, p0/Z, [x8, #1, MUL VL]
    addvl x8, x8, #1
    ld1h { z26.h }, p1/Z, [x7]
    ld1h { z25.h }, p0/Z, [x7, #1, MUL VL]
    addvl x7, x7, #1
    zip1 z20.h, z19.h, z18.h
    ld1h { z24.h }, p1/Z, [x6]
    ld1h { z23.h }, p0/Z, [x6, #1, MUL VL]
    addvl x6, x6, #1
    zip2 z19.h, z19.h, z18.h
    zip1 z18.h, z17.h, z16.h
    zip1 z17.h, z0.h, z31.h
    incd x27, ALL, MUL #4
    incd x23, ALL, MUL #4
    zip2 z16.h, z0.h, z31.h
    zip1 z22.h, z22.h, z21.h
    st1h { z20.h }, p2, [x4]
    incd x22, ALL, MUL #4
    zip1 z21.h, z30.h, z29.h
    zip2 z20.h, z30.h, z29.h
    st1h { z19.h }, p2, [x4, #1, MUL VL]
    incd x21, ALL, MUL #4
    st1h { z18.h }, p2, [x4, #2, MUL VL]
    incd x20, ALL, MUL #4
    incd x8, ALL, MUL #4
    zip1 z19.h, z28.h, z27.h
    st1h { z17.h }, p2, [x4, #3, MUL VL]
    incd x7, ALL, MUL #4
    incd x6, ALL, MUL #4
    zip1 z18.h, z26.h, z24.h
    st1h { z16.h }, p2, [x4, #4, MUL VL]
    zip2 z17.h, z26.h, z24.h
    zip1 z16.h, z25.h, z23.h
    st1h { z22.h }, p2, [x4, #5, MUL VL]
    st1h { z21.h }, p2, [x4, #6, MUL VL]
    st1h { z20.h }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1h { z19.h }, p2, [x4, #-4, MUL VL]
    st1h { z18.h }, p2, [x4, #-3, MUL VL]
    st1h { z17.h }, p2, [x4, #-2, MUL VL]
    st1h { z16.h }, p2, [x4, #-1, MUL VL]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x10, x10, #0x8
    addvl x0, x0, #12
    cmp x10, #0x8
    bge label_1
    cbz x10, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x27, x1
    cntw x1, ALL, MUL #12
    mov x6, x2
    cnth x5, ALL, MUL #3
    cmp x10, #0x1
    mov x24, x0
    add x23, x27, x3
    csel x4, x1, XZR, GT
    add x1, x23, x3
    csel x23, x23, x9, GT
    cmp x6, x5
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Unroll column loop
    ld1h { z19.h }, p2/Z, [x27]
    ld1h { z23.h }, p2/Z, [x27, #1, MUL VL]
    sub x6, x6, x5
    ld1h { z18.h }, p2/Z, [x23]
    ld1h { z17.h }, p2/Z, [x23, #1, MUL VL]
    cmp x6, x5
    ld1h { z22.h }, p2/Z, [x27, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x23, #2, MUL VL]
    addvl x27, x27, #3
    add x23, x23, x4
    zip1 z21.h, z19.h, z18.h
    zip2 z20.h, z19.h, z18.h
    zip1 z19.h, z23.h, z17.h
    zip2 z18.h, z23.h, z17.h
    zip1 z17.h, z22.h, z16.h
    zip2 z16.h, z22.h, z16.h
    st1h { z21.h }, p2, [x24]
    st1h { z20.h }, p2, [x24, #1, MUL VL]
    st1h { z19.h }, p2, [x24, #2, MUL VL]
    add x24, x24, x28
    st1h { z18.h }, p2, [x24]
    st1h { z17.h }, p2, [x24, #1, MUL VL]
    st1h { z16.h }, p2, [x24, #2, MUL VL]
    add x24, x24, x28
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x6, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    mov x4, x6
    decw x6, ALL, MUL #3
    whilelt p1.h, XZR, x4
    dech x4
    whilelt p0.h, XZR, x4
    cmp x6, #0
    ld1h { z20.h }, p1/Z, [x27]
    ld1h { z17.h }, p1/Z, [x23]
    ld1h { z19.h }, p0/Z, [x27, #1, MUL VL]
    addvl x27, x27, #1
    ld1h { z16.h }, p0/Z, [x23, #1, MUL VL]
    addvl x23, x23, #1
    zip1 z18.h, z20.h, z17.h
    zip2 z17.h, z20.h, z17.h
    incd x27, ALL, MUL #4
    incd x23, ALL, MUL #4
    zip1 z16.h, z19.h, z16.h
    st1h { z18.h }, p2, [x24]
    st1h { z17.h }, p2, [x24, #1, MUL VL]
    st1h { z16.h }, p2, [x24, #2, MUL VL]
    add x24, x24, x28
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x10, x10, #0x2
    addvl x0, x0, #3
    cmp x10, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    addvl SP, SP, #3
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldr d8, [sp, 72]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_3VL_2x2)

    KAI_ASM_END
