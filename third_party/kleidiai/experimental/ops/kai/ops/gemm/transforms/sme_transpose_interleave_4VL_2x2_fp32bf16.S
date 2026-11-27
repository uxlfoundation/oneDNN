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

    KAI_ASM_CODE(sme_transpose_interleave_4VL_2x2_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_4VL_2x2_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_4VL_2x2_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_4VL_2x2_fp32bf16)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cnth x5, ALL, MUL #4
    fmov z16.s, #0
    add x6, x6, #0x1
    mov x24, x4
    bfm x6, XZR, #0, #0  // bfc x6, #0, #0x1
    ptrue p4.b
    lsr x5, x5, #0x1
    lsl x6, x6, #0x1
    addvl SP, SP, #-4
    cmp x24, #0x4
    mul x23, x6, x5
    st1w { z16.s }, p4, [SP]
    mov x22, SP
    st1w { z16.s }, p4, [SP, #1, MUL VL]
    st1w { z16.s }, p4, [SP, #2, MUL VL]
    st1w { z16.s }, p4, [SP, #3, MUL VL]
    blt label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    mov x20, x0
    add x8, x21, x3
    mov x7, x2
    add x6, x8, x3
    add x5, x6, x3
    add x1, x5, x3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x4, x7
    decw x7, ALL, MUL #4
    whilelt p3.s, XZR, x4
    decw x4
    whilelt p2.s, XZR, x4
    decw x4
    ld1w { z19.s }, p3/Z, [x21]
    whilelt p1.s, XZR, x4
    decw x4
    ld1w { z18.s }, p2/Z, [x21, #1, MUL VL]
    whilelt p0.s, XZR, x4
    ld1w { z17.s }, p1/Z, [x21, #2, MUL VL]
    cmp x7, #0
    ld1w { z16.s }, p0/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x658ab27b)  // bfcvt z27.h, p4/M, z19.s
    addvl x21, x21, #4
    ld1w { z19.s }, p3/Z, [x6]
    KAI_ASM_INST(0x658ab25a)  // bfcvt z26.h, p4/M, z18.s
    ld1w { z18.s }, p2/Z, [x6, #1, MUL VL]
    KAI_ASM_INST(0x658ab239)  // bfcvt z25.h, p4/M, z17.s
    ld1w { z17.s }, p1/Z, [x6, #2, MUL VL]
    KAI_ASM_INST(0x658ab218)  // bfcvt z24.h, p4/M, z16.s
    ld1w { z16.s }, p0/Z, [x6, #3, MUL VL]
    KAI_ASM_INST(0x658ab277)  // bfcvt z23.h, p4/M, z19.s
    addvl x6, x6, #4
    KAI_ASM_INST(0x658ab256)  // bfcvt z22.h, p4/M, z18.s
    ld1w { z19.s }, p3/Z, [x8]
    KAI_ASM_INST(0x658ab235)  // bfcvt z21.h, p4/M, z17.s
    ld1w { z18.s }, p2/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0x658ab214)  // bfcvt z20.h, p4/M, z16.s
    ld1w { z17.s }, p1/Z, [x8, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    KAI_ASM_INST(0x648ab27b)  // bfcvtnt z27.h, p4/M, z19.s
    ld1w { z19.s }, p3/Z, [x5]
    KAI_ASM_INST(0x648ab25a)  // bfcvtnt z26.h, p4/M, z18.s
    ld1w { z18.s }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x648ab239)  // bfcvtnt z25.h, p4/M, z17.s
    ld1w { z17.s }, p1/Z, [x5, #2, MUL VL]
    KAI_ASM_INST(0x648ab218)  // bfcvtnt z24.h, p4/M, z16.s
    ld1w { z16.s }, p0/Z, [x5, #3, MUL VL]
    addvl x5, x5, #4
    KAI_ASM_INST(0x648ab277)  // bfcvtnt z23.h, p4/M, z19.s
    st1h { z27.h }, p4, [x20]
    KAI_ASM_INST(0x648ab256)  // bfcvtnt z22.h, p4/M, z18.s
    st1h { z26.h }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x648ab235)  // bfcvtnt z21.h, p4/M, z17.s
    st1h { z25.h }, p4, [x20, #2, MUL VL]
    KAI_ASM_INST(0x648ab214)  // bfcvtnt z20.h, p4/M, z16.s
    st1h { z24.h }, p4, [x20, #3, MUL VL]
    st1h { z23.h }, p4, [x20, #4, MUL VL]
    st1h { z22.h }, p4, [x20, #5, MUL VL]
    st1h { z21.h }, p4, [x20, #6, MUL VL]
    st1h { z20.h }, p4, [x20, #7, MUL VL]
    add x20, x20, x23
    bgt label_2
    sub x24, x24, #0x4
    addvl x0, x0, #8
    cmp x24, #0x4
    bge label_1
    cbz x24, label_8
KAI_ASM_LABEL(label_4)  // Main loop skip
KAI_ASM_LABEL(label_5)  // Tail row loop: Head
    mov x21, x1
    cntw x4, ALL, MUL #16
    add x8, x21, x3
    cmp x24, #0x1
    add x1, x8, x3
    mov x20, x0
    csel x8, x8, x22, GT
    csel x6, x4, XZR, GT
    mov x5, x2
KAI_ASM_LABEL(label_6)  // Tail row loop: Column loop
    mov x4, x5
    decw x5, ALL, MUL #4
    whilelt p3.s, XZR, x4
    decw x4
    whilelt p2.s, XZR, x4
    decw x4
    ld1w { z19.s }, p3/Z, [x21]
    whilelt p1.s, XZR, x4
    decw x4
    ld1w { z18.s }, p2/Z, [x21, #1, MUL VL]
    whilelt p0.s, XZR, x4
    ld1w { z17.s }, p1/Z, [x21, #2, MUL VL]
    cmp x5, #0
    ld1w { z16.s }, p0/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x658ab277)  // bfcvt z23.h, p4/M, z19.s
    addvl x21, x21, #4
    KAI_ASM_INST(0x658ab256)  // bfcvt z22.h, p4/M, z18.s
    ld1w { z21.s }, p3/Z, [x8]
    KAI_ASM_INST(0x658ab234)  // bfcvt z20.h, p4/M, z17.s
    ld1w { z19.s }, p2/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0x658ab212)  // bfcvt z18.h, p4/M, z16.s
    ld1w { z17.s }, p1/Z, [x8, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x8, #3, MUL VL]
    add x8, x8, x6
    KAI_ASM_INST(0x648ab2b7)  // bfcvtnt z23.h, p4/M, z21.s
    KAI_ASM_INST(0x648ab276)  // bfcvtnt z22.h, p4/M, z19.s
    KAI_ASM_INST(0x648ab234)  // bfcvtnt z20.h, p4/M, z17.s
    KAI_ASM_INST(0x648ab212)  // bfcvtnt z18.h, p4/M, z16.s
    st1h { z23.h }, p4, [x20]
    st1h { z22.h }, p4, [x20, #1, MUL VL]
    st1h { z20.h }, p4, [x20, #2, MUL VL]
    st1h { z18.h }, p4, [x20, #3, MUL VL]
    add x20, x20, x23
    bgt label_6
    sub x24, x24, #0x2
    addvl x0, x0, #4
    cmp x24, #0x1
    bge label_5
KAI_ASM_LABEL(label_8)  // Done
    addvl SP, SP, #4
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_4VL_2x2_fp32bf16)

    KAI_ASM_END
