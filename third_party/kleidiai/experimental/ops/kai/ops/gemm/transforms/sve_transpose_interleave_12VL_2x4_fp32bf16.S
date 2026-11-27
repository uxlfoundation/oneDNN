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

    KAI_ASM_CODE(sve_transpose_interleave_12VL_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_12VL_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_12VL_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_12VL_2x4_fp32bf16)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    cnth x7, ALL, MUL #12
    mov z17.b, #0
    addvl SP, SP, #-12
    ptrue p2.b
    mov x6, x4
    lsr x7, x7, #0x2
    mov x12, x4
    addvl x5, SP, #16
    st1w { z17.s }, p2, [SP]
    st1w { z17.s }, p2, [SP, #1, MUL VL]
    mov x11, SP
    add x6, x6, #0x3
    bfm x6, XZR, #0, #0x1  // bfc x6, #0, #0x2
    st1w { z17.s }, p2, [SP, #2, MUL VL]
    st1w { z17.s }, p2, [SP, #3, MUL VL]
    st1w { z17.s }, p2, [SP, #4, MUL VL]
    lsl x6, x6, #0x1
    st1w { z17.s }, p2, [SP, #5, MUL VL]
    st1w { z17.s }, p2, [SP, #6, MUL VL]
    mul x10, x6, x7
    st1w { z17.s }, p2, [SP, #7, MUL VL]
    st1w { z17.s }, p2, [x5, #-8, MUL VL]
    st1w { z17.s }, p2, [x5, #-7, MUL VL]
    st1w { z17.s }, p2, [x5, #-6, MUL VL]
    st1w { z17.s }, p2, [x5, #-5, MUL VL]
    cbz x4, label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x9, x1
    cntw x4, ALL, MUL #12
    mov x28, x2
    cnth x27, ALL, MUL #6
    cmp x12, #0x3
    mov x26, x0
    add x25, x9, x3
    lsl x4, x4, #0x2
    add x24, x25, x3
    add x23, x24, x3
    add x1, x23, x3
    csel x23, x23, x11, GT
    csel x22, x4, XZR, GT
    csel x24, x24, x11, GE
    csel x21, x4, XZR, GE
    cmp x12, #0x1
    csel x25, x25, x11, GT
    csel x20, x4, XZR, GT
    cmp x28, x27
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    mov x8, x24
    ld1w { z5.s }, p2/Z, [x9]
    ld1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    mov x7, x25
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z12.s }, p2/Z, [x9, #3, MUL VL]
    mov x6, x23
    mov x5, x26
    ld1w { z24.s }, p2/Z, [x9, #4, MUL VL]
    ld1w { z20.s }, p2/Z, [x9, #5, MUL VL]
    add x26, x26, x10
    sub x28, x28, x27
    ld1w { z31.s }, p2/Z, [x9, #6, MUL VL]
    ld1w { z3.s }, p2/Z, [x9, #7, MUL VL]
    addvl x9, x9, #12
    mov x4, x26
    ld1w { z26.s }, p2/Z, [x8]
    ld1w { z11.s }, p2/Z, [x8, #1, MUL VL]
    cmp x28, x27
    add x25, x25, x20
    ld1w { z9.s }, p2/Z, [x8, #2, MUL VL]
    ld1w { z8.s }, p2/Z, [x8, #3, MUL VL]
    add x24, x24, x21
    add x23, x23, x22
    ld1w { z1.s }, p2/Z, [x8, #4, MUL VL]
    ld1w { z16.s }, p2/Z, [x8, #5, MUL VL]
    add x26, x26, x10
    ld1w { z18.s }, p2/Z, [x8, #6, MUL VL]
    ld1w { z21.s }, p2/Z, [x8, #7, MUL VL]
    addvl x8, x8, #12
    zip1 z29.s, z5.s, z26.s
    ld1w { z22.s }, p2/Z, [x9, #-4, MUL VL]
    ld1w { z15.s }, p2/Z, [x9, #-3, MUL VL]
    zip2 z28.s, z5.s, z26.s
    zip1 z30.s, z2.s, z11.s
    ld1w { z27.s }, p2/Z, [x7]
    ld1w { z5.s }, p2/Z, [x7, #1, MUL VL]
    zip2 z6.s, z2.s, z11.s
    zip1 z2.s, z10.s, z9.s
    ld1w { z11.s }, p2/Z, [x7, #2, MUL VL]
    ld1w { z19.s }, p2/Z, [x7, #3, MUL VL]
    zip2 z10.s, z10.s, z9.s
    zip1 z7.s, z12.s, z8.s
    ld1w { z26.s }, p2/Z, [x7, #4, MUL VL]
    ld1w { z25.s }, p2/Z, [x7, #5, MUL VL]
    zip2 z9.s, z12.s, z8.s
    KAI_ASM_INST(0x658aabbd)  // bfcvt z29.h, p2/M, z29.s
    ld1w { z8.s }, p2/Z, [x7, #6, MUL VL]
    ld1w { z0.s }, p2/Z, [x7, #7, MUL VL]
    addvl x7, x7, #12
    KAI_ASM_INST(0x658aab8d)  // bfcvt z13.h, p2/M, z28.s
    ld1w { z14.s }, p2/Z, [x8, #-4, MUL VL]
    ld1w { z28.s }, p2/Z, [x8, #-3, MUL VL]
    KAI_ASM_INST(0x658aabcc)  // bfcvt z12.h, p2/M, z30.s
    KAI_ASM_INST(0x658aa8d7)  // bfcvt z23.h, p2/M, z6.s
    ld1w { z30.s }, p2/Z, [x6]
    ld1w { z4.s }, p2/Z, [x6, #1, MUL VL]
    KAI_ASM_INST(0x658aa846)  // bfcvt z6.h, p2/M, z2.s
    KAI_ASM_INST(0x658aa951)  // bfcvt z17.h, p2/M, z10.s
    ld1w { z2.s }, p2/Z, [x6, #2, MUL VL]
    KAI_ASM_INST(0x658aa8e7)  // bfcvt z7.h, p2/M, z7.s
    KAI_ASM_INST(0x658aa929)  // bfcvt z9.h, p2/M, z9.s
    zip1 z10.s, z24.s, z1.s
    zip2 z24.s, z24.s, z1.s
    zip1 z1.s, z27.s, z30.s
    zip2 z27.s, z27.s, z30.s
    ld1w { z30.s }, p2/Z, [x6, #3, MUL VL]
    KAI_ASM_INST(0x658aa94a)  // bfcvt z10.h, p2/M, z10.s
    KAI_ASM_INST(0x658aab18)  // bfcvt z24.h, p2/M, z24.s
    KAI_ASM_INST(0x648aa83d)  // bfcvtnt z29.h, p2/M, z1.s
    ld1w { z1.s }, p2/Z, [x6, #4, MUL VL]
    KAI_ASM_INST(0x648aab6d)  // bfcvtnt z13.h, p2/M, z27.s
    zip1 z27.s, z5.s, z4.s
    zip2 z5.s, z5.s, z4.s
    ld1w { z4.s }, p2/Z, [x6, #5, MUL VL]
    KAI_ASM_INST(0x648aab6c)  // bfcvtnt z12.h, p2/M, z27.s
    ld1w { z27.s }, p2/Z, [x6, #6, MUL VL]
    KAI_ASM_INST(0x648aa8b7)  // bfcvtnt z23.h, p2/M, z5.s
    zip1 z5.s, z11.s, z2.s
    zip2 z2.s, z11.s, z2.s
    ld1w { z11.s }, p2/Z, [x6, #7, MUL VL]
    addvl x6, x6, #12
    KAI_ASM_INST(0x648aa8a6)  // bfcvtnt z6.h, p2/M, z5.s
    ld1w { z5.s }, p2/Z, [x9, #-2, MUL VL]
    KAI_ASM_INST(0x648aa851)  // bfcvtnt z17.h, p2/M, z2.s
    zip1 z2.s, z19.s, z30.s
    zip2 z19.s, z19.s, z30.s
    ld1w { z30.s }, p2/Z, [x9, #-1, MUL VL]
    KAI_ASM_INST(0x648aa847)  // bfcvtnt z7.h, p2/M, z2.s
    ld1w { z2.s }, p2/Z, [x7, #-4, MUL VL]
    KAI_ASM_INST(0x648aaa69)  // bfcvtnt z9.h, p2/M, z19.s
    zip1 z19.s, z20.s, z16.s
    zip2 z20.s, z20.s, z16.s
    zip1 z16.s, z31.s, z18.s
    zip2 z18.s, z31.s, z18.s
    zip1 z31.s, z3.s, z21.s
    zip2 z3.s, z3.s, z21.s
    zip1 z21.s, z22.s, z14.s
    zip2 z22.s, z22.s, z14.s
    zip1 z14.s, z15.s, z28.s
    zip2 z15.s, z15.s, z28.s
    zip1 z28.s, z26.s, z1.s
    zip2 z26.s, z26.s, z1.s
    ld1w { z1.s }, p2/Z, [x7, #-3, MUL VL]
    KAI_ASM_INST(0x658aaa73)  // bfcvt z19.h, p2/M, z19.s
    KAI_ASM_INST(0x658aaa94)  // bfcvt z20.h, p2/M, z20.s
    KAI_ASM_INST(0x658aaa10)  // bfcvt z16.h, p2/M, z16.s
    KAI_ASM_INST(0x658aaa52)  // bfcvt z18.h, p2/M, z18.s
    KAI_ASM_INST(0x658aabff)  // bfcvt z31.h, p2/M, z31.s
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0x658aaab5)  // bfcvt z21.h, p2/M, z21.s
    KAI_ASM_INST(0x658aaad6)  // bfcvt z22.h, p2/M, z22.s
    KAI_ASM_INST(0x658aa9ce)  // bfcvt z14.h, p2/M, z14.s
    KAI_ASM_INST(0x658aa9ef)  // bfcvt z15.h, p2/M, z15.s
    KAI_ASM_INST(0x648aab8a)  // bfcvtnt z10.h, p2/M, z28.s
    ld1w { z28.s }, p2/Z, [x8, #-2, MUL VL]
    KAI_ASM_INST(0x648aab58)  // bfcvtnt z24.h, p2/M, z26.s
    zip1 z26.s, z25.s, z4.s
    zip2 z4.s, z25.s, z4.s
    ld1w { z25.s }, p2/Z, [x8, #-1, MUL VL]
    KAI_ASM_INST(0x648aab53)  // bfcvtnt z19.h, p2/M, z26.s
    ld1w { z26.s }, p2/Z, [x6, #-4, MUL VL]
    KAI_ASM_INST(0x648aa894)  // bfcvtnt z20.h, p2/M, z4.s
    zip1 z4.s, z8.s, z27.s
    zip2 z8.s, z8.s, z27.s
    ld1w { z27.s }, p2/Z, [x6, #-3, MUL VL]
    KAI_ASM_INST(0x648aa890)  // bfcvtnt z16.h, p2/M, z4.s
    ld1w { z4.s }, p2/Z, [x7, #-2, MUL VL]
    KAI_ASM_INST(0x648aa912)  // bfcvtnt z18.h, p2/M, z8.s
    zip1 z8.s, z0.s, z11.s
    zip2 z0.s, z0.s, z11.s
    ld1w { z11.s }, p2/Z, [x7, #-1, MUL VL]
    KAI_ASM_INST(0x648aa91f)  // bfcvtnt z31.h, p2/M, z8.s
    ld1w { z8.s }, p2/Z, [x6, #-2, MUL VL]
    KAI_ASM_INST(0x648aa803)  // bfcvtnt z3.h, p2/M, z0.s
    ld1w { z0.s }, p2/Z, [x6, #-1, MUL VL]
    st1h { z29.h }, p2, [x5]
    zip1 z29.s, z2.s, z26.s
    zip2 z2.s, z2.s, z26.s
    zip1 z26.s, z1.s, z27.s
    zip2 z1.s, z1.s, z27.s
    st1h { z13.h }, p2, [x5, #1, MUL VL]
    zip1 z27.s, z5.s, z28.s
    zip2 z28.s, z5.s, z28.s
    st1h { z12.h }, p2, [x5, #2, MUL VL]
    zip1 z13.s, z30.s, z25.s
    zip2 z12.s, z30.s, z25.s
    st1h { z23.h }, p2, [x5, #3, MUL VL]
    st1h { z6.h }, p2, [x5, #4, MUL VL]
    KAI_ASM_INST(0x648aabb5)  // bfcvtnt z21.h, p2/M, z29.s
    KAI_ASM_INST(0x648aa856)  // bfcvtnt z22.h, p2/M, z2.s
    st1h { z17.h }, p2, [x5, #5, MUL VL]
    KAI_ASM_INST(0x648aab4e)  // bfcvtnt z14.h, p2/M, z26.s
    KAI_ASM_INST(0x648aa82f)  // bfcvtnt z15.h, p2/M, z1.s
    st1h { z7.h }, p2, [x5, #6, MUL VL]
    KAI_ASM_INST(0x658aab66)  // bfcvt z6.h, p2/M, z27.s
    zip1 z30.s, z4.s, z8.s
    st1h { z9.h }, p2, [x5, #7, MUL VL]
    addvl x5, x5, #12
    KAI_ASM_INST(0x658aab9b)  // bfcvt z27.h, p2/M, z28.s
    zip2 z5.s, z4.s, z8.s
    KAI_ASM_INST(0x658aa9a4)  // bfcvt z4.h, p2/M, z13.s
    zip1 z25.s, z11.s, z0.s
    KAI_ASM_INST(0x658aa997)  // bfcvt z23.h, p2/M, z12.s
    zip2 z17.s, z11.s, z0.s
    st1h { z10.h }, p2, [x5, #-4, MUL VL]
    st1h { z24.h }, p2, [x5, #-3, MUL VL]
    KAI_ASM_INST(0x648aabc6)  // bfcvtnt z6.h, p2/M, z30.s
    KAI_ASM_INST(0x648aa8bb)  // bfcvtnt z27.h, p2/M, z5.s
    st1h { z19.h }, p2, [x5, #-2, MUL VL]
    KAI_ASM_INST(0x648aab24)  // bfcvtnt z4.h, p2/M, z25.s
    st1h { z20.h }, p2, [x5, #-1, MUL VL]
    KAI_ASM_INST(0x648aaa37)  // bfcvtnt z23.h, p2/M, z17.s
    st1h { z16.h }, p2, [x4]
    st1h { z18.h }, p2, [x4, #1, MUL VL]
    st1h { z31.h }, p2, [x4, #2, MUL VL]
    st1h { z3.h }, p2, [x4, #3, MUL VL]
    st1h { z21.h }, p2, [x4, #4, MUL VL]
    st1h { z22.h }, p2, [x4, #5, MUL VL]
    st1h { z14.h }, p2, [x4, #6, MUL VL]
    st1h { z15.h }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    st1h { z6.h }, p2, [x4, #-4, MUL VL]
    st1h { z27.h }, p2, [x4, #-3, MUL VL]
    st1h { z4.h }, p2, [x4, #-2, MUL VL]
    st1h { z23.h }, p2, [x4, #-1, MUL VL]
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x28, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x5, x28
    mov x4, x26
    decd x28, ALL, MUL #12
    add x26, x26, x10
    whilelt p1.s, XZR, x5
    decw x5
    whilelt p0.s, XZR, x5
    decw x5
    ld1w { z19.s }, p1/Z, [x9]
    ld1w { z18.s }, p1/Z, [x24]
    ld1w { z30.s }, p1/Z, [x25]
    ld1w { z29.s }, p1/Z, [x23]
    whilelt p1.s, XZR, x5
    decw x5
    ld1w { z21.s }, p0/Z, [x9, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x24, #1, MUL VL]
    ld1w { z28.s }, p0/Z, [x25, #1, MUL VL]
    ld1w { z27.s }, p0/Z, [x23, #1, MUL VL]
    zip1 z16.s, z19.s, z18.s
    zip2 z26.s, z19.s, z18.s
    whilelt p0.s, XZR, x5
    decw x5
    ld1w { z20.s }, p1/Z, [x9, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z25.s }, p1/Z, [x25, #2, MUL VL]
    ld1w { z24.s }, p1/Z, [x23, #2, MUL VL]
    zip1 z18.s, z21.s, z17.s
    zip2 z23.s, z21.s, z17.s
    KAI_ASM_INST(0x658aaa0a)  // bfcvt z10.h, p2/M, z16.s
    zip1 z9.s, z30.s, z29.s
    whilelt p1.s, XZR, x5
    decw x5
    ld1w { z17.s }, p0/Z, [x9, #3, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #3, MUL VL]
    zip1 z22.s, z20.s, z19.s
    zip2 z21.s, z20.s, z19.s
    ld1w { z20.s }, p0/Z, [x25, #3, MUL VL]
    ld1w { z19.s }, p0/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0x658aab48)  // bfcvt z8.h, p2/M, z26.s
    zip2 z7.s, z30.s, z29.s
    whilelt p0.s, XZR, x5
    ld1w { z6.s }, p1/Z, [x9, #4, MUL VL]
    ld1w { z5.s }, p1/Z, [x24, #4, MUL VL]
    KAI_ASM_INST(0x658aaa44)  // bfcvt z4.h, p2/M, z18.s
    zip1 z18.s, z17.s, z16.s
    zip2 z17.s, z17.s, z16.s
    ld1w { z3.s }, p1/Z, [x25, #4, MUL VL]
    ld1w { z2.s }, p1/Z, [x23, #4, MUL VL]
    zip1 z1.s, z28.s, z27.s
    KAI_ASM_INST(0x658aaae0)  // bfcvt z0.h, p2/M, z23.s
    cmp x28, #0
    ld1w { z31.s }, p0/Z, [x9, #5, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #5, MUL VL]
    ld1w { z30.s }, p0/Z, [x25, #5, MUL VL]
    zip2 z29.s, z28.s, z27.s
    KAI_ASM_INST(0x658aaadc)  // bfcvt z28.h, p2/M, z22.s
    ld1w { z27.s }, p0/Z, [x23, #5, MUL VL]
    zip1 z23.s, z25.s, z24.s
    KAI_ASM_INST(0x658aaaba)  // bfcvt z26.h, p2/M, z21.s
    addvl x9, x9, #6
    zip2 z22.s, z25.s, z24.s
    KAI_ASM_INST(0x658aaa59)  // bfcvt z25.h, p2/M, z18.s
    addvl x25, x25, #6
    addvl x24, x24, #6
    zip1 z21.s, z20.s, z19.s
    KAI_ASM_INST(0x658aaa38)  // bfcvt z24.h, p2/M, z17.s
    addvl x23, x23, #6
    zip2 z20.s, z20.s, z19.s
    zip1 z19.s, z6.s, z5.s
    zip2 z18.s, z6.s, z5.s
    zip1 z17.s, z31.s, z16.s
    zip2 z16.s, z31.s, z16.s
    KAI_ASM_INST(0x648aa92a)  // bfcvtnt z10.h, p2/M, z9.s
    KAI_ASM_INST(0x648aa8e8)  // bfcvtnt z8.h, p2/M, z7.s
    KAI_ASM_INST(0x648aa824)  // bfcvtnt z4.h, p2/M, z1.s
    KAI_ASM_INST(0x648aaba0)  // bfcvtnt z0.h, p2/M, z29.s
    KAI_ASM_INST(0x648aaafc)  // bfcvtnt z28.h, p2/M, z23.s
    KAI_ASM_INST(0x648aaada)  // bfcvtnt z26.h, p2/M, z22.s
    KAI_ASM_INST(0x648aaab9)  // bfcvtnt z25.h, p2/M, z21.s
    st1h { z10.h }, p2, [x4]
    KAI_ASM_INST(0x648aaa98)  // bfcvtnt z24.h, p2/M, z20.s
    KAI_ASM_INST(0x658aaa77)  // bfcvt z23.h, p2/M, z19.s
    st1h { z8.h }, p2, [x4, #1, MUL VL]
    zip1 z22.s, z3.s, z2.s
    KAI_ASM_INST(0x658aaa55)  // bfcvt z21.h, p2/M, z18.s
    st1h { z4.h }, p2, [x4, #2, MUL VL]
    zip2 z20.s, z3.s, z2.s
    KAI_ASM_INST(0x658aaa33)  // bfcvt z19.h, p2/M, z17.s
    st1h { z0.h }, p2, [x4, #3, MUL VL]
    zip1 z18.s, z30.s, z27.s
    KAI_ASM_INST(0x658aaa11)  // bfcvt z17.h, p2/M, z16.s
    st1h { z28.h }, p2, [x4, #4, MUL VL]
    zip2 z16.s, z30.s, z27.s
    st1h { z26.h }, p2, [x4, #5, MUL VL]
    KAI_ASM_INST(0x648aaad7)  // bfcvtnt z23.h, p2/M, z22.s
    st1h { z25.h }, p2, [x4, #6, MUL VL]
    KAI_ASM_INST(0x648aaa95)  // bfcvtnt z21.h, p2/M, z20.s
    st1h { z24.h }, p2, [x4, #7, MUL VL]
    addvl x4, x4, #12
    KAI_ASM_INST(0x648aaa53)  // bfcvtnt z19.h, p2/M, z18.s
    KAI_ASM_INST(0x648aaa11)  // bfcvtnt z17.h, p2/M, z16.s
    st1h { z23.h }, p2, [x4, #-4, MUL VL]
    st1h { z21.h }, p2, [x4, #-3, MUL VL]
    st1h { z19.h }, p2, [x4, #-2, MUL VL]
    st1h { z17.h }, p2, [x4, #-1, MUL VL]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x12, x12, #0x4
    addvl x0, x0, #12
    cmp x12, #0x1
    bge label_1
KAI_ASM_LABEL(label_6)  // Done
    addvl SP, SP, #12
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_12VL_2x4_fp32bf16)

    KAI_ASM_END
