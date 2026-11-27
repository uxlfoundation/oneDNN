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

    KAI_ASM_CODE(sve_transpose_interleave_2VL_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_transpose_interleave_2VL_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_transpose_interleave_2VL_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_transpose_interleave_2VL_2x4_fp32bf16)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    cnth x6, ALL, MUL #2
    mov z16.b, #0
    ptrue p1.b
    addvl SP, SP, #-2
    mov x5, x4
    lsr x6, x6, #0x2
    mov x27, x4
    st1w { z16.s }, p1, [SP]
    st1w { z16.s }, p1, [SP, #1, MUL VL]
    mov x26, SP
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    lsl x5, x5, #0x1
    mul x25, x5, x6
    cbz x4, label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    cntw x4, ALL, MUL #8
    mov x23, x2
    cnth x22
    cmp x27, #0x3
    mov x21, x0
    add x20, x24, x3
    csel x8, x4, XZR, GT
    add x7, x20, x3
    csel x6, x4, XZR, GE
    add x5, x7, x3
    csel x7, x7, x26, GE
    add x1, x5, x3
    csel x5, x5, x26, GT
    cmp x27, #0x1
    csel x20, x20, x26, GT
    csel x4, x4, XZR, GT
    cmp x23, x22
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1w { z18.s }, p1/Z, [x24]
    ld1w { z17.s }, p1/Z, [x7]
    sub x23, x23, x22
    ld1w { z21.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x7, #1, MUL VL]
    cmp x23, x22
    addvl x24, x24, #2
    ld1w { z26.s }, p1/Z, [x20]
    ld1w { z20.s }, p1/Z, [x5]
    add x7, x7, x6
    zip1 z19.s, z18.s, z17.s
    zip2 z18.s, z18.s, z17.s
    ld1w { z25.s }, p1/Z, [x20, #1, MUL VL]
    ld1w { z24.s }, p1/Z, [x5, #1, MUL VL]
    add x20, x20, x4
    zip1 z17.s, z21.s, z16.s
    zip2 z16.s, z21.s, z16.s
    add x5, x5, x8
    KAI_ASM_INST(0x658aa677)  // bfcvt z23.h, p1/M, z19.s
    zip1 z22.s, z26.s, z20.s
    KAI_ASM_INST(0x658aa655)  // bfcvt z21.h, p1/M, z18.s
    zip2 z20.s, z26.s, z20.s
    KAI_ASM_INST(0x658aa633)  // bfcvt z19.h, p1/M, z17.s
    zip1 z18.s, z25.s, z24.s
    KAI_ASM_INST(0x658aa611)  // bfcvt z17.h, p1/M, z16.s
    zip2 z16.s, z25.s, z24.s
    KAI_ASM_INST(0x648aa6d7)  // bfcvtnt z23.h, p1/M, z22.s
    KAI_ASM_INST(0x648aa695)  // bfcvtnt z21.h, p1/M, z20.s
    KAI_ASM_INST(0x648aa653)  // bfcvtnt z19.h, p1/M, z18.s
    KAI_ASM_INST(0x648aa611)  // bfcvtnt z17.h, p1/M, z16.s
    st1h { z23.h }, p1, [x21]
    st1h { z21.h }, p1, [x21, #1, MUL VL]
    add x21, x21, x25
    st1h { z19.h }, p1, [x21]
    st1h { z17.h }, p1, [x21, #1, MUL VL]
    add x21, x21, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x23, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.s, XZR, x23
    decd x23, ALL, MUL #2
    ld1w { z19.s }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1w { z16.s }, p0/Z, [x7]
    addvl x7, x7, #1
    ld1w { z20.s }, p0/Z, [x20]
    addvl x20, x20, #1
    ld1w { z18.s }, p0/Z, [x5]
    addvl x5, x5, #1
    cmp x23, #0
    zip1 z17.s, z19.s, z16.s
    zip2 z16.s, z19.s, z16.s
    zip1 z19.s, z20.s, z18.s
    zip2 z18.s, z20.s, z18.s
    KAI_ASM_INST(0x658aa631)  // bfcvt z17.h, p1/M, z17.s
    KAI_ASM_INST(0x658aa610)  // bfcvt z16.h, p1/M, z16.s
    KAI_ASM_INST(0x648aa671)  // bfcvtnt z17.h, p1/M, z19.s
    KAI_ASM_INST(0x648aa650)  // bfcvtnt z16.h, p1/M, z18.s
    st1h { z17.h }, p1, [x21]
    st1h { z16.h }, p1, [x21, #1, MUL VL]
    add x21, x21, x25
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x27, x27, #0x4
    addvl x0, x0, #2
    cmp x27, #0x1
    bge label_1
KAI_ASM_LABEL(label_6)  // Done
    addvl SP, SP, #2
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_transpose_interleave_2VL_2x4_fp32bf16)

    KAI_ASM_END
