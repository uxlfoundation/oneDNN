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

    KAI_ASM_CODE(sve_transpose_interleave_6VL_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_6VL_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_6VL_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_6VL_2x4_fp32bf16)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    str d10, [sp, 80]
    cnth x6, ALL, MUL #6
    mov z16.b, #0
    ptrue p2.b
    addvl SP, SP, #-6
    mov x5, x4
    lsr x6, x6, #0x2
    mov x27, x4
    st1w { z16.s }, p2, [SP]
    st1w { z16.s }, p2, [SP, #1, MUL VL]
    mov x26, SP
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    st1w { z16.s }, p2, [SP, #2, MUL VL]
    st1w { z16.s }, p2, [SP, #3, MUL VL]
    st1w { z16.s }, p2, [SP, #4, MUL VL]
    lsl x5, x5, #0x1
    st1w { z16.s }, p2, [SP, #5, MUL VL]
    mul x25, x5, x6
    cbz x4, label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    cntw x4, ALL, MUL #12
    mov x23, x2
    cnth x22, ALL, MUL #3
    cmp x27, #0x3
    mov x21, x0
    add x20, x24, x3
    lsl x4, x4, #0x1
    add x8, x20, x3
    add x7, x8, x3
    add x1, x7, x3
    csel x7, x7, x26, GT
    csel x6, x4, XZR, GT
    csel x8, x8, x26, GE
    csel x5, x4, XZR, GE
    cmp x27, #0x1
    csel x20, x20, x26, GT
    csel x4, x4, XZR, GT
    cmp x23, x22
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1w { z17.s }, p2/Z, [x24]
    ld1w { z24.s }, p2/Z, [x24, #1, MUL VL]
    sub x23, x23, x22
    ld1w { z23.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z16.s }, p2/Z, [x8]
    cmp x23, x22
    ld1w { z22.s }, p2/Z, [x8, #1, MUL VL]
    ld1w { z21.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z20.s }, p2/Z, [x24, #3, MUL VL]
    ld1w { z19.s }, p2/Z, [x24, #4, MUL VL]
    ld1w { z5.s }, p2/Z, [x24, #5, MUL VL]
    ld1w { z18.s }, p2/Z, [x8, #3, MUL VL]
    zip1 z4.s, z17.s, z16.s
    zip2 z3.s, z17.s, z16.s
    ld1w { z17.s }, p2/Z, [x8, #4, MUL VL]
    ld1w { z16.s }, p2/Z, [x8, #5, MUL VL]
    zip1 z2.s, z24.s, z22.s
    zip2 z1.s, z24.s, z22.s
    ld1w { z0.s }, p2/Z, [x20]
    ld1w { z31.s }, p2/Z, [x20, #1, MUL VL]
    zip1 z30.s, z23.s, z21.s
    zip2 z29.s, z23.s, z21.s
    ld1w { z28.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x7]
    zip1 z26.s, z20.s, z18.s
    zip2 z25.s, z20.s, z18.s
    ld1w { z24.s }, p2/Z, [x7, #1, MUL VL]
    ld1w { z23.s }, p2/Z, [x7, #2, MUL VL]
    zip1 z22.s, z19.s, z17.s
    zip2 z10.s, z19.s, z17.s
    ld1w { z21.s }, p2/Z, [x20, #3, MUL VL]
    ld1w { z20.s }, p2/Z, [x20, #4, MUL VL]
    zip1 z19.s, z5.s, z16.s
    zip2 z9.s, z5.s, z16.s
    ld1w { z8.s }, p2/Z, [x20, #5, MUL VL]
    ld1w { z18.s }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x658aa887)  // bfcvt z7.h, p2/M, z4.s
    zip1 z6.s, z0.s, z27.s
    ld1w { z17.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z16.s }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x658aa865)  // bfcvt z5.h, p2/M, z3.s
    zip2 z4.s, z0.s, z27.s
    KAI_ASM_INST(0x658aa843)  // bfcvt z3.h, p2/M, z2.s
    zip1 z2.s, z31.s, z24.s
    addvl x24, x24, #6
    add x20, x20, x4
    KAI_ASM_INST(0x658aa821)  // bfcvt z1.h, p2/M, z1.s
    zip2 z0.s, z31.s, z24.s
    add x8, x8, x5
    add x7, x7, x6
    KAI_ASM_INST(0x658aabdf)  // bfcvt z31.h, p2/M, z30.s
    zip1 z30.s, z28.s, z23.s
    KAI_ASM_INST(0x658aabbd)  // bfcvt z29.h, p2/M, z29.s
    zip2 z28.s, z28.s, z23.s
    KAI_ASM_INST(0x658aab5b)  // bfcvt z27.h, p2/M, z26.s
    zip1 z26.s, z21.s, z18.s
    KAI_ASM_INST(0x658aab39)  // bfcvt z25.h, p2/M, z25.s
    zip2 z24.s, z21.s, z18.s
    KAI_ASM_INST(0x658aaad7)  // bfcvt z23.h, p2/M, z22.s
    zip1 z22.s, z20.s, z17.s
    KAI_ASM_INST(0x658aa955)  // bfcvt z21.h, p2/M, z10.s
    zip2 z20.s, z20.s, z17.s
    KAI_ASM_INST(0x658aaa73)  // bfcvt z19.h, p2/M, z19.s
    zip1 z18.s, z8.s, z16.s
    KAI_ASM_INST(0x658aa931)  // bfcvt z17.h, p2/M, z9.s
    zip2 z16.s, z8.s, z16.s
    KAI_ASM_INST(0x648aa8c7)  // bfcvtnt z7.h, p2/M, z6.s
    KAI_ASM_INST(0x648aa885)  // bfcvtnt z5.h, p2/M, z4.s
    KAI_ASM_INST(0x648aa843)  // bfcvtnt z3.h, p2/M, z2.s
    KAI_ASM_INST(0x648aa801)  // bfcvtnt z1.h, p2/M, z0.s
    KAI_ASM_INST(0x648aabdf)  // bfcvtnt z31.h, p2/M, z30.s
    KAI_ASM_INST(0x648aab9d)  // bfcvtnt z29.h, p2/M, z28.s
    st1h { z7.h }, p2, [x21]
    st1h { z5.h }, p2, [x21, #1, MUL VL]
    KAI_ASM_INST(0x648aab5b)  // bfcvtnt z27.h, p2/M, z26.s
    KAI_ASM_INST(0x648aab19)  // bfcvtnt z25.h, p2/M, z24.s
    st1h { z3.h }, p2, [x21, #2, MUL VL]
    KAI_ASM_INST(0x648aaad7)  // bfcvtnt z23.h, p2/M, z22.s
    KAI_ASM_INST(0x648aaa95)  // bfcvtnt z21.h, p2/M, z20.s
    st1h { z1.h }, p2, [x21, #3, MUL VL]
    KAI_ASM_INST(0x648aaa53)  // bfcvtnt z19.h, p2/M, z18.s
    KAI_ASM_INST(0x648aaa11)  // bfcvtnt z17.h, p2/M, z16.s
    st1h { z31.h }, p2, [x21, #4, MUL VL]
    st1h { z29.h }, p2, [x21, #5, MUL VL]
    add x21, x21, x25
    st1h { z27.h }, p2, [x21]
    st1h { z25.h }, p2, [x21, #1, MUL VL]
    st1h { z23.h }, p2, [x21, #2, MUL VL]
    st1h { z21.h }, p2, [x21, #3, MUL VL]
    st1h { z19.h }, p2, [x21, #4, MUL VL]
    st1h { z17.h }, p2, [x21, #5, MUL VL]
    add x21, x21, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x23, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x4, x23
    decd x23, ALL, MUL #6
    whilelt p0.s, XZR, x4
    decw x4
    whilelt p1.s, XZR, x4
    decw x4
    ld1w { z17.s }, p0/Z, [x24]
    ld1w { z16.s }, p0/Z, [x8]
    ld1w { z23.s }, p0/Z, [x20]
    ld1w { z19.s }, p0/Z, [x7]
    whilelt p0.s, XZR, x4
    cmp x23, #0
    ld1w { z22.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x8, #1, MUL VL]
    ld1w { z31.s }, p1/Z, [x20, #1, MUL VL]
    ld1w { z30.s }, p1/Z, [x7, #1, MUL VL]
    zip1 z21.s, z17.s, z16.s
    zip2 z17.s, z17.s, z16.s
    ld1w { z20.s }, p0/Z, [x24, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x8, #2, MUL VL]
    zip1 z29.s, z23.s, z19.s
    zip2 z28.s, z23.s, z19.s
    ld1w { z27.s }, p0/Z, [x20, #2, MUL VL]
    ld1w { z26.s }, p0/Z, [x7, #2, MUL VL]
    zip1 z19.s, z22.s, z18.s
    zip2 z18.s, z22.s, z18.s
    KAI_ASM_INST(0x658aaab9)  // bfcvt z25.h, p2/M, z21.s
    KAI_ASM_INST(0x658aaa38)  // bfcvt z24.h, p2/M, z17.s
    addvl x24, x24, #3
    addvl x20, x20, #3
    zip1 z17.s, z20.s, z16.s
    zip2 z16.s, z20.s, z16.s
    addvl x8, x8, #3
    addvl x7, x7, #3
    KAI_ASM_INST(0x658aaa77)  // bfcvt z23.h, p2/M, z19.s
    zip1 z22.s, z31.s, z30.s
    KAI_ASM_INST(0x658aaa55)  // bfcvt z21.h, p2/M, z18.s
    zip2 z20.s, z31.s, z30.s
    KAI_ASM_INST(0x658aaa33)  // bfcvt z19.h, p2/M, z17.s
    zip1 z18.s, z27.s, z26.s
    KAI_ASM_INST(0x658aaa11)  // bfcvt z17.h, p2/M, z16.s
    zip2 z16.s, z27.s, z26.s
    KAI_ASM_INST(0x648aabb9)  // bfcvtnt z25.h, p2/M, z29.s
    KAI_ASM_INST(0x648aab98)  // bfcvtnt z24.h, p2/M, z28.s
    KAI_ASM_INST(0x648aaad7)  // bfcvtnt z23.h, p2/M, z22.s
    KAI_ASM_INST(0x648aaa95)  // bfcvtnt z21.h, p2/M, z20.s
    KAI_ASM_INST(0x648aaa53)  // bfcvtnt z19.h, p2/M, z18.s
    KAI_ASM_INST(0x648aaa11)  // bfcvtnt z17.h, p2/M, z16.s
    st1h { z25.h }, p2, [x21]
    st1h { z24.h }, p2, [x21, #1, MUL VL]
    st1h { z23.h }, p2, [x21, #2, MUL VL]
    st1h { z21.h }, p2, [x21, #3, MUL VL]
    st1h { z19.h }, p2, [x21, #4, MUL VL]
    st1h { z17.h }, p2, [x21, #5, MUL VL]
    add x21, x21, x25
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x27, x27, #0x4
    addvl x0, x0, #6
    cmp x27, #0x1
    bge label_1
KAI_ASM_LABEL(label_6)  // Done
    addvl SP, SP, #6
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldr d10, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_6VL_2x4_fp32bf16)

    KAI_ASM_END
