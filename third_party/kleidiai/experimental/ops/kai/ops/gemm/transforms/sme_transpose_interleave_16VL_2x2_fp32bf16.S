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

    KAI_ASM_CODE(sme_transpose_interleave_16VL_2x2_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_16VL_2x2_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_16VL_2x2_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_16VL_2x2_fp32bf16)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x7, x4
    cnth x6, ALL, MUL #16
    fmov z16.s, #0
    add x7, x7, #0x1
    addvl SP, SP, #-16
    bfm x7, XZR, #0, #0  // bfc x7, #0, #0x1
    ptrue p7.b
    lsr x6, x6, #0x1
    lsl x7, x7, #0x1
    addvl x5, SP, #16
    mul x25, x7, x6
    st1w { z16.s }, p7, [SP]
    st1w { z16.s }, p7, [SP, #1, MUL VL]
    mov x24, SP
    mov x23, x4
    st1w { z16.s }, p7, [SP, #2, MUL VL]
    st1w { z16.s }, p7, [SP, #3, MUL VL]
    st1w { z16.s }, p7, [SP, #4, MUL VL]
    st1w { z16.s }, p7, [SP, #5, MUL VL]
    st1w { z16.s }, p7, [SP, #6, MUL VL]
    st1w { z16.s }, p7, [SP, #7, MUL VL]
    st1w { z16.s }, p7, [x5, #-8, MUL VL]
    st1w { z16.s }, p7, [x5, #-7, MUL VL]
    st1w { z16.s }, p7, [x5, #-6, MUL VL]
    st1w { z16.s }, p7, [x5, #-5, MUL VL]
    st1w { z16.s }, p7, [x5, #-4, MUL VL]
    st1w { z16.s }, p7, [x5, #-3, MUL VL]
    st1w { z16.s }, p7, [x5, #-2, MUL VL]
    st1w { z16.s }, p7, [x5, #-1, MUL VL]
    cbz x4, label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x22, x1
    cntw x4, ALL, MUL #16
    add x21, x22, x3
    lsl x4, x4, #0x2
    cmp x23, #0x1
    add x1, x21, x3
    mov x20, x0
    csel x21, x21, x24, GT
    csel x8, x4, XZR, GT
    mov x7, x2
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x6, x7
    mov x5, x21
    whilelt p1.s, XZR, x6
    decw x6
    whilelt p0.s, XZR, x6
    decw x6
    ld1w { z16.s }, p1/Z, [x22]
    whilelt p6.s, XZR, x6
    decw x6
    ld1w { z18.s }, p0/Z, [x22, #1, MUL VL]
    whilelt p5.s, XZR, x6
    decw x6
    ld1w { z17.s }, p6/Z, [x22, #2, MUL VL]
    whilelt p4.s, XZR, x6
    decw x6
    ld1w { z19.s }, p5/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0x658abe03)  // bfcvt z3.h, p7/M, z16.s
    whilelt p3.s, XZR, x6
    decw x6
    ld1w { z16.s }, p4/Z, [x22, #4, MUL VL]
    KAI_ASM_INST(0x658abe42)  // bfcvt z2.h, p7/M, z18.s
    whilelt p2.s, XZR, x6
    decw x6
    ld1w { z18.s }, p3/Z, [x22, #5, MUL VL]
    KAI_ASM_INST(0x658abe21)  // bfcvt z1.h, p7/M, z17.s
    ld1w { z17.s }, p2/Z, [x22, #6, MUL VL]
    KAI_ASM_INST(0x658abe60)  // bfcvt z0.h, p7/M, z19.s
    mov x4, x20
    decw x7, ALL, MUL #16
    KAI_ASM_INST(0x658abe1f)  // bfcvt z31.h, p7/M, z16.s
    ld1w { z19.s }, p1/Z, [x5]
    whilelt p1.s, XZR, x6
    decw x6
    ld1w { z16.s }, p1/Z, [x22, #7, MUL VL]
    addvl x22, x22, #16
    KAI_ASM_INST(0x658abe5e)  // bfcvt z30.h, p7/M, z18.s
    add x21, x21, x8
    KAI_ASM_INST(0x658abe3d)  // bfcvt z29.h, p7/M, z17.s
    ld1w { z18.s }, p0/Z, [x5, #1, MUL VL]
    whilelt p0.s, XZR, x6
    decw x6
    ld1w { z17.s }, p0/Z, [x22, #-8, MUL VL]
    KAI_ASM_INST(0x648abe63)  // bfcvtnt z3.h, p7/M, z19.s
    add x20, x20, x25
    KAI_ASM_INST(0x658abe1c)  // bfcvt z28.h, p7/M, z16.s
    ld1w { z19.s }, p6/Z, [x5, #2, MUL VL]
    whilelt p6.s, XZR, x6
    decw x6
    ld1w { z16.s }, p6/Z, [x22, #-7, MUL VL]
    KAI_ASM_INST(0x648abe42)  // bfcvtnt z2.h, p7/M, z18.s
    ld1w { z18.s }, p5/Z, [x5, #3, MUL VL]
    whilelt p5.s, XZR, x6
    decw x6
    KAI_ASM_INST(0x658abe3b)  // bfcvt z27.h, p7/M, z17.s
    ld1w { z17.s }, p5/Z, [x22, #-6, MUL VL]
    KAI_ASM_INST(0x648abe61)  // bfcvtnt z1.h, p7/M, z19.s
    ld1w { z19.s }, p4/Z, [x5, #4, MUL VL]
    whilelt p4.s, XZR, x6
    decw x6
    KAI_ASM_INST(0x658abe1a)  // bfcvt z26.h, p7/M, z16.s
    ld1w { z16.s }, p4/Z, [x22, #-5, MUL VL]
    KAI_ASM_INST(0x648abe40)  // bfcvtnt z0.h, p7/M, z18.s
    ld1w { z18.s }, p3/Z, [x5, #5, MUL VL]
    whilelt p3.s, XZR, x6
    decw x6
    KAI_ASM_INST(0x658abe39)  // bfcvt z25.h, p7/M, z17.s
    ld1w { z17.s }, p3/Z, [x22, #-4, MUL VL]
    KAI_ASM_INST(0x648abe7f)  // bfcvtnt z31.h, p7/M, z19.s
    ld1w { z19.s }, p2/Z, [x5, #6, MUL VL]
    whilelt p2.s, XZR, x6
    decw x6
    KAI_ASM_INST(0x658abe18)  // bfcvt z24.h, p7/M, z16.s
    ld1w { z16.s }, p2/Z, [x22, #-3, MUL VL]
    KAI_ASM_INST(0x648abe5e)  // bfcvtnt z30.h, p7/M, z18.s
    ld1w { z18.s }, p1/Z, [x5, #7, MUL VL]
    whilelt p1.s, XZR, x6
    decw x6
    KAI_ASM_INST(0x658abe37)  // bfcvt z23.h, p7/M, z17.s
    ld1w { z17.s }, p1/Z, [x22, #-2, MUL VL]
    addvl x5, x5, #16
    KAI_ASM_INST(0x648abe7d)  // bfcvtnt z29.h, p7/M, z19.s
    KAI_ASM_INST(0x658abe16)  // bfcvt z22.h, p7/M, z16.s
    ld1w { z19.s }, p0/Z, [x5, #-8, MUL VL]
    whilelt p0.s, XZR, x6
    cmp x7, #0
    ld1w { z16.s }, p0/Z, [x22, #-1, MUL VL]
    KAI_ASM_INST(0x648abe5c)  // bfcvtnt z28.h, p7/M, z18.s
    KAI_ASM_INST(0x658abe35)  // bfcvt z21.h, p7/M, z17.s
    ld1w { z18.s }, p6/Z, [x5, #-7, MUL VL]
    ld1w { z17.s }, p5/Z, [x5, #-6, MUL VL]
    KAI_ASM_INST(0x648abe7b)  // bfcvtnt z27.h, p7/M, z19.s
    KAI_ASM_INST(0x658abe14)  // bfcvt z20.h, p7/M, z16.s
    ld1w { z16.s }, p4/Z, [x5, #-5, MUL VL]
    ld1w { z19.s }, p3/Z, [x5, #-4, MUL VL]
    KAI_ASM_INST(0x648abe5a)  // bfcvtnt z26.h, p7/M, z18.s
    ld1w { z18.s }, p2/Z, [x5, #-3, MUL VL]
    KAI_ASM_INST(0x648abe39)  // bfcvtnt z25.h, p7/M, z17.s
    ld1w { z17.s }, p1/Z, [x5, #-2, MUL VL]
    KAI_ASM_INST(0x648abe18)  // bfcvtnt z24.h, p7/M, z16.s
    ld1w { z16.s }, p0/Z, [x5, #-1, MUL VL]
    st1h { z3.h }, p7, [x4]
    KAI_ASM_INST(0x648abe77)  // bfcvtnt z23.h, p7/M, z19.s
    st1h { z2.h }, p7, [x4, #1, MUL VL]
    KAI_ASM_INST(0x648abe56)  // bfcvtnt z22.h, p7/M, z18.s
    st1h { z1.h }, p7, [x4, #2, MUL VL]
    KAI_ASM_INST(0x648abe35)  // bfcvtnt z21.h, p7/M, z17.s
    st1h { z0.h }, p7, [x4, #3, MUL VL]
    KAI_ASM_INST(0x648abe14)  // bfcvtnt z20.h, p7/M, z16.s
    st1h { z31.h }, p7, [x4, #4, MUL VL]
    st1h { z30.h }, p7, [x4, #5, MUL VL]
    st1h { z29.h }, p7, [x4, #6, MUL VL]
    st1h { z28.h }, p7, [x4, #7, MUL VL]
    addvl x4, x4, #16
    st1h { z27.h }, p7, [x4, #-8, MUL VL]
    st1h { z26.h }, p7, [x4, #-7, MUL VL]
    st1h { z25.h }, p7, [x4, #-6, MUL VL]
    st1h { z24.h }, p7, [x4, #-5, MUL VL]
    st1h { z23.h }, p7, [x4, #-4, MUL VL]
    st1h { z22.h }, p7, [x4, #-3, MUL VL]
    st1h { z21.h }, p7, [x4, #-2, MUL VL]
    st1h { z20.h }, p7, [x4, #-1, MUL VL]
    bgt label_2
    sub x23, x23, #0x2
    addvl x0, x0, #16
    cmp x23, #0x1
    bge label_1
KAI_ASM_LABEL(label_4)  // Done
    addvl SP, SP, #16
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_16VL_2x2_fp32bf16)

    KAI_ASM_END
