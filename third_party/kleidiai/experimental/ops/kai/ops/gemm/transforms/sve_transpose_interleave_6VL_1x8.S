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

    KAI_ASM_CODE(sve_transpose_interleave_6VL_1x8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_6VL_1x8)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_6VL_1x8)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_6VL_1x8)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    cntb x6, ALL, MUL #6
    mov z16.b, #0
    ptrue p1.b
    addvl SP, SP, #-3
    mov x5, x4
    lsr x6, x6, #0x3
    mov x15, x4
    st1b { z16.b }, p1, [SP]
    st1b { z16.b }, p1, [SP, #1, MUL VL]
    mov x14, SP
    add x5, x5, #0x7
    bfm x5, XZR, #0, #0x2  // bfc x5, #0, #0x3
    st1b { z16.b }, p1, [SP, #2, MUL VL]
    mul x13, x5, x6
    cbz x4, label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x12, x1
    cntw x4, ALL, MUL #12
    cmp x15, #0x7
    mov x11, x2
    cntb x10, ALL, MUL #3
    mov x9, x0
    add x28, x12, x3
    csel x27, x4, XZR, GT
    add x26, x28, x3
    csel x25, x4, XZR, GE
    add x24, x26, x3
    add x23, x24, x3
    add x22, x23, x3
    add x21, x22, x3
    add x20, x21, x3
    csel x21, x21, x14, GE
    add x1, x20, x3
    csel x20, x20, x14, GT
    cmp x15, #0x5
    csel x22, x22, x14, GT
    csel x8, x4, XZR, GT
    csel x23, x23, x14, GE
    csel x7, x4, XZR, GE
    cmp x15, #0x3
    csel x24, x24, x14, GT
    csel x6, x4, XZR, GT
    csel x26, x26, x14, GE
    csel x5, x4, XZR, GE
    cmp x15, #0x1
    csel x28, x28, x14, GT
    csel x4, x4, XZR, GT
    cmp x11, x10
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1b { z1.b }, p1/Z, [x12]
    ld1b { z0.b }, p1/Z, [x28]
    sub x11, x11, x10
    ld1b { z31.b }, p1/Z, [x26]
    ld1b { z28.b }, p1/Z, [x24]
    cmp x11, x10
    ld1b { z27.b }, p1/Z, [x23]
    ld1b { z26.b }, p1/Z, [x22]
    ld1b { z25.b }, p1/Z, [x21]
    ld1b { z24.b }, p1/Z, [x20]
    ld1b { z23.b }, p1/Z, [x12, #1, MUL VL]
    ld1b { z30.b }, p1/Z, [x28, #1, MUL VL]
    ld1b { z22.b }, p1/Z, [x26, #1, MUL VL]
    ld1b { z21.b }, p1/Z, [x24, #1, MUL VL]
    zip1 z15.b, z1.b, z27.b
    zip1 z9.b, z0.b, z26.b
    ld1b { z20.b }, p1/Z, [x23, #1, MUL VL]
    ld1b { z19.b }, p1/Z, [x22, #1, MUL VL]
    zip1 z18.b, z31.b, z25.b
    zip1 z29.b, z28.b, z24.b
    ld1b { z17.b }, p1/Z, [x21, #1, MUL VL]
    ld1b { z16.b }, p1/Z, [x20, #1, MUL VL]
    zip2 z14.b, z1.b, z27.b
    zip2 z13.b, z31.b, z25.b
    ld1b { z8.b }, p1/Z, [x12, #2, MUL VL]
    ld1b { z7.b }, p1/Z, [x28, #2, MUL VL]
    zip2 z12.b, z0.b, z26.b
    zip2 z6.b, z28.b, z24.b
    ld1b { z5.b }, p1/Z, [x26, #2, MUL VL]
    ld1b { z4.b }, p1/Z, [x24, #2, MUL VL]
    zip1 z3.b, z23.b, z20.b
    zip1 z11.b, z30.b, z19.b
    ld1b { z28.b }, p1/Z, [x23, #2, MUL VL]
    ld1b { z27.b }, p1/Z, [x22, #2, MUL VL]
    zip1 z2.b, z22.b, z17.b
    zip1 z1.b, z21.b, z16.b
    ld1b { z26.b }, p1/Z, [x21, #2, MUL VL]
    ld1b { z25.b }, p1/Z, [x20, #2, MUL VL]
    zip2 z24.b, z23.b, z20.b
    zip2 z23.b, z22.b, z17.b
    zip2 z22.b, z30.b, z19.b
    zip2 z21.b, z21.b, z16.b
    addvl x12, x12, #3
    add x28, x28, x4
    zip1 z0.b, z8.b, z28.b
    zip1 z10.b, z7.b, z27.b
    add x26, x26, x5
    add x24, x24, x6
    zip1 z31.b, z5.b, z26.b
    zip1 z30.b, z4.b, z25.b
    add x23, x23, x7
    add x22, x22, x8
    zip1 z20.b, z15.b, z18.b
    zip1 z19.b, z9.b, z29.b
    add x21, x21, x25
    add x20, x20, x27
    zip2 z18.b, z15.b, z18.b
    zip2 z16.b, z9.b, z29.b
    zip1 z29.b, z14.b, z13.b
    zip1 z17.b, z12.b, z6.b
    zip2 z9.b, z8.b, z28.b
    zip2 z28.b, z5.b, z26.b
    zip2 z8.b, z7.b, z27.b
    zip2 z27.b, z4.b, z25.b
    zip2 z7.b, z14.b, z13.b
    zip2 z6.b, z12.b, z6.b
    zip1 z5.b, z3.b, z2.b
    zip1 z4.b, z11.b, z1.b
    zip2 z3.b, z3.b, z2.b
    zip2 z2.b, z11.b, z1.b
    zip1 z26.b, z24.b, z23.b
    zip1 z25.b, z22.b, z21.b
    zip2 z24.b, z24.b, z23.b
    zip2 z23.b, z22.b, z21.b
    zip1 z1.b, z0.b, z31.b
    zip1 z22.b, z10.b, z30.b
    zip1 z21.b, z20.b, z19.b
    zip2 z20.b, z20.b, z19.b
    zip1 z19.b, z18.b, z16.b
    zip2 z18.b, z18.b, z16.b
    zip1 z16.b, z29.b, z17.b
    zip2 z17.b, z29.b, z17.b
    st1b { z21.b }, p1, [x9]
    st1b { z20.b }, p1, [x9, #1, MUL VL]
    zip2 z0.b, z0.b, z31.b
    zip2 z31.b, z10.b, z30.b
    st1b { z19.b }, p1, [x9, #2, MUL VL]
    zip1 z30.b, z9.b, z28.b
    zip1 z29.b, z8.b, z27.b
    st1b { z18.b }, p1, [x9, #3, MUL VL]
    zip2 z28.b, z9.b, z28.b
    zip2 z27.b, z8.b, z27.b
    st1b { z16.b }, p1, [x9, #4, MUL VL]
    zip1 z21.b, z7.b, z6.b
    zip2 z16.b, z7.b, z6.b
    st1b { z17.b }, p1, [x9, #5, MUL VL]
    add x9, x9, x13
    zip1 z20.b, z5.b, z4.b
    zip2 z19.b, z5.b, z4.b
    zip1 z18.b, z3.b, z2.b
    zip2 z17.b, z3.b, z2.b
    st1b { z21.b }, p1, [x9]
    st1b { z16.b }, p1, [x9, #1, MUL VL]
    zip1 z16.b, z26.b, z25.b
    zip2 z26.b, z26.b, z25.b
    st1b { z20.b }, p1, [x9, #2, MUL VL]
    zip1 z25.b, z24.b, z23.b
    zip2 z24.b, z24.b, z23.b
    st1b { z19.b }, p1, [x9, #3, MUL VL]
    zip1 z23.b, z1.b, z22.b
    zip2 z22.b, z1.b, z22.b
    st1b { z18.b }, p1, [x9, #4, MUL VL]
    zip1 z21.b, z0.b, z31.b
    zip2 z20.b, z0.b, z31.b
    st1b { z17.b }, p1, [x9, #5, MUL VL]
    add x9, x9, x13
    zip1 z19.b, z30.b, z29.b
    zip2 z18.b, z30.b, z29.b
    st1b { z16.b }, p1, [x9]
    zip1 z17.b, z28.b, z27.b
    zip2 z16.b, z28.b, z27.b
    st1b { z26.b }, p1, [x9, #1, MUL VL]
    st1b { z25.b }, p1, [x9, #2, MUL VL]
    st1b { z24.b }, p1, [x9, #3, MUL VL]
    st1b { z23.b }, p1, [x9, #4, MUL VL]
    st1b { z22.b }, p1, [x9, #5, MUL VL]
    add x9, x9, x13
    st1b { z21.b }, p1, [x9]
    st1b { z20.b }, p1, [x9, #1, MUL VL]
    st1b { z19.b }, p1, [x9, #2, MUL VL]
    st1b { z18.b }, p1, [x9, #3, MUL VL]
    st1b { z17.b }, p1, [x9, #4, MUL VL]
    st1b { z16.b }, p1, [x9, #5, MUL VL]
    add x9, x9, x13
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x11, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.b, XZR, x11
    decd x11, ALL, MUL #6
    ld1b { z21.b }, p0/Z, [x12]
    incd x12, ALL, MUL #6
    ld1b { z26.b }, p0/Z, [x28]
    incd x28, ALL, MUL #6
    ld1b { z20.b }, p0/Z, [x26]
    incd x26, ALL, MUL #6
    ld1b { z25.b }, p0/Z, [x24]
    incd x24, ALL, MUL #6
    ld1b { z18.b }, p0/Z, [x23]
    ld1b { z19.b }, p0/Z, [x22]
    ld1b { z17.b }, p0/Z, [x21]
    cmp x11, #0
    incd x23, ALL, MUL #6
    ld1b { z16.b }, p0/Z, [x20]
    incd x22, ALL, MUL #6
    incd x21, ALL, MUL #6
    zip1 z23.b, z21.b, z18.b
    zip2 z24.b, z21.b, z18.b
    incd x20, ALL, MUL #6
    zip1 z22.b, z20.b, z17.b
    zip1 z21.b, z26.b, z19.b
    zip2 z18.b, z20.b, z17.b
    zip1 z17.b, z25.b, z16.b
    zip2 z20.b, z26.b, z19.b
    zip2 z16.b, z25.b, z16.b
    zip1 z19.b, z23.b, z22.b
    zip2 z23.b, z23.b, z22.b
    zip1 z22.b, z24.b, z18.b
    zip1 z18.b, z21.b, z17.b
    zip2 z17.b, z21.b, z17.b
    zip1 z16.b, z20.b, z16.b
    zip1 z21.b, z19.b, z18.b
    zip2 z20.b, z19.b, z18.b
    zip1 z19.b, z23.b, z17.b
    zip2 z18.b, z23.b, z17.b
    zip1 z17.b, z22.b, z16.b
    zip2 z16.b, z22.b, z16.b
    st1b { z21.b }, p1, [x9]
    st1b { z20.b }, p1, [x9, #1, MUL VL]
    st1b { z19.b }, p1, [x9, #2, MUL VL]
    st1b { z18.b }, p1, [x9, #3, MUL VL]
    st1b { z17.b }, p1, [x9, #4, MUL VL]
    st1b { z16.b }, p1, [x9, #5, MUL VL]
    add x9, x9, x13
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x15, x15, #0x8
    addvl x0, x0, #6
    cmp x15, #0x1
    bge label_1
KAI_ASM_LABEL(label_6)  // Done
    addvl SP, SP, #3
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_6VL_1x8)

    KAI_ASM_END
