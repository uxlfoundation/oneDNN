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

    KAI_ASM_CODE(sve_interleaved_fp32_mla_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_interleaved_fp32_mla_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_interleaved_fp32_mla_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_interleaved_fp32_mla_8x3VL)
    stp d8, d9, [sp, -64]!
    stp d10, d11, [sp, 16]
    stp d12, d13, [sp, 32]
    stp d14, d15, [sp, 48]
    ptrue p0.b
    ldr x7, [x0, #0]
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x50]
    cbz x7, label_6
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x30]
    mov x2, x6
KAI_ASM_LABEL(label_2)  // Width loop
    ldr x1, [x0, #0x10]
    mov x6, x2
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    ld1w { z4.s }, p0/Z, [x3]
    mov z11.b, #0
    mov z12.b, #0
    ld1w { z5.s }, p0/Z, [x3, #1, MUL VL]
    sub x1, x1, #0x1
    mov z13.b, #0
    mov z14.b, #0
    cmp x1, #0x2
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    ld1rqw { z0.s }, p0/Z, [x6]
    mov z19.b, #0
    mov z20.b, #0
    ld1rqw { z1.s }, p0/Z, [x6, #16]
    mov z21.b, #0
    mov z22.b, #0
    ld1w { z6.s }, p0/Z, [x3, #2, MUL VL]
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    fmla z8.s, z4.s, z0.s[0]
    fmla z11.s, z4.s, z0.s[1]
    ld1rqw { z2.s }, p0/Z, [x6, #32]
    fmla z14.s, z4.s, z0.s[2]
    fmla z17.s, z4.s, z0.s[3]
    ld1rqw { z3.s }, p0/Z, [x6, #48]
    fmla z20.s, z4.s, z1.s[0]
    fmla z23.s, z4.s, z1.s[1]
    sub x1, x1, #0x2
    fmla z26.s, z4.s, z1.s[2]
    fmla z29.s, z4.s, z1.s[3]
    ld1w { z4.s }, p0/Z, [x3, #3, MUL VL]
    fmla z9.s, z5.s, z0.s[0]
    fmla z12.s, z5.s, z0.s[1]
    cmp x1, #0x2
    fmla z15.s, z5.s, z0.s[2]
    fmla z18.s, z5.s, z0.s[3]
    add x6, x6, #0x40
    fmla z21.s, z5.s, z1.s[0]
    fmla z24.s, z5.s, z1.s[1]
    fmla z27.s, z5.s, z1.s[2]
    fmla z30.s, z5.s, z1.s[3]
    ld1w { z5.s }, p0/Z, [x3, #4, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z13.s, z6.s, z0.s[1]
    fmla z16.s, z6.s, z0.s[2]
    fmla z19.s, z6.s, z0.s[3]
    ld1rqw { z0.s }, p0/Z, [x6]
    fmla z22.s, z6.s, z1.s[0]
    fmla z25.s, z6.s, z1.s[1]
    fmla z28.s, z6.s, z1.s[2]
    fmla z31.s, z6.s, z1.s[3]
    ld1w { z6.s }, p0/Z, [x3, #5, MUL VL]
    addvl x3, x3, #6
    fmla z8.s, z4.s, z2.s[0]
    fmla z11.s, z4.s, z2.s[1]
    ld1rqw { z1.s }, p0/Z, [x6, #16]
    fmla z14.s, z4.s, z2.s[2]
    fmla z17.s, z4.s, z2.s[3]
    fmla z20.s, z4.s, z3.s[0]
    fmla z23.s, z4.s, z3.s[1]
    fmla z26.s, z4.s, z3.s[2]
    fmla z29.s, z4.s, z3.s[3]
    ld1w { z4.s }, p0/Z, [x3]
    fmla z9.s, z5.s, z2.s[0]
    fmla z12.s, z5.s, z2.s[1]
    fmla z15.s, z5.s, z2.s[2]
    fmla z18.s, z5.s, z2.s[3]
    fmla z21.s, z5.s, z3.s[0]
    fmla z24.s, z5.s, z3.s[1]
    fmla z27.s, z5.s, z3.s[2]
    fmla z30.s, z5.s, z3.s[3]
    ld1w { z5.s }, p0/Z, [x3, #1, MUL VL]
    fmla z10.s, z6.s, z2.s[0]
    fmla z13.s, z6.s, z2.s[1]
    fmla z16.s, z6.s, z2.s[2]
    fmla z19.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[0]
    fmla z25.s, z6.s, z3.s[1]
    fmla z28.s, z6.s, z3.s[2]
    fmla z31.s, z6.s, z3.s[3]
    ld1w { z6.s }, p0/Z, [x3, #2, MUL VL]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x20
    addvl x3, x3, #3
    fmla z8.s, z4.s, z0.s[0]
    fmla z11.s, z4.s, z0.s[1]
    fmla z14.s, z4.s, z0.s[2]
    fmla z17.s, z4.s, z0.s[3]
    fmla z20.s, z4.s, z1.s[0]
    fmla z23.s, z4.s, z1.s[1]
    fmla z26.s, z4.s, z1.s[2]
    fmla z29.s, z4.s, z1.s[3]
    fmla z9.s, z5.s, z0.s[0]
    fmla z12.s, z5.s, z0.s[1]
    fmla z15.s, z5.s, z0.s[2]
    fmla z18.s, z5.s, z0.s[3]
    fmla z21.s, z5.s, z1.s[0]
    fmla z24.s, z5.s, z1.s[1]
    fmla z27.s, z5.s, z1.s[2]
    fmla z30.s, z5.s, z1.s[3]
    fmla z10.s, z6.s, z0.s[0]
    fmla z13.s, z6.s, z0.s[1]
    fmla z16.s, z6.s, z0.s[2]
    fmla z19.s, z6.s, z0.s[3]
    fmla z22.s, z6.s, z1.s[0]
    fmla z25.s, z6.s, z1.s[1]
    fmla z28.s, z6.s, z1.s[2]
    fmla z31.s, z6.s, z1.s[3]
    cbz x1, label_5
    ld1rqw { z0.s }, p0/Z, [x6]
    ld1rqw { z1.s }, p0/Z, [x6, #16]
    add x6, x6, #0x20
    ld1w { z7.s }, p0/Z, [x3]
    ld1w { z4.s }, p0/Z, [x3, #1, MUL VL]
    ld1w { z5.s }, p0/Z, [x3, #2, MUL VL]
    addvl x3, x3, #3
    fmla z8.s, z7.s, z0.s[0]
    fmla z11.s, z7.s, z0.s[1]
    fmla z14.s, z7.s, z0.s[2]
    fmla z17.s, z7.s, z0.s[3]
    fmla z20.s, z7.s, z1.s[0]
    fmla z23.s, z7.s, z1.s[1]
    fmla z26.s, z7.s, z1.s[2]
    fmla z29.s, z7.s, z1.s[3]
    fmla z9.s, z4.s, z0.s[0]
    fmla z12.s, z4.s, z0.s[1]
    fmla z15.s, z4.s, z0.s[2]
    fmla z18.s, z4.s, z0.s[3]
    fmla z21.s, z4.s, z1.s[0]
    fmla z24.s, z4.s, z1.s[1]
    fmla z27.s, z4.s, z1.s[2]
    fmla z30.s, z4.s, z1.s[3]
    fmla z10.s, z5.s, z0.s[0]
    fmla z13.s, z5.s, z0.s[1]
    fmla z16.s, z5.s, z0.s[2]
    fmla z19.s, z5.s, z0.s[3]
    fmla z22.s, z5.s, z1.s[0]
    fmla z25.s, z5.s, z1.s[1]
    fmla z28.s, z5.s, z1.s[2]
    fmla z31.s, z5.s, z1.s[3]
KAI_ASM_LABEL(label_5)  // multiply loop done
    decw x4, ALL, MUL #3
    st1w { z8.s }, p0, [x5]
    st1w { z9.s }, p0, [x5, #1, MUL VL]
    st1w { z10.s }, p0, [x5, #2, MUL VL]
    st1w { z11.s }, p0, [x5, #3, MUL VL]
    cmp x4, XZR
    st1w { z12.s }, p0, [x5, #4, MUL VL]
    st1w { z13.s }, p0, [x5, #5, MUL VL]
    st1w { z14.s }, p0, [x5, #6, MUL VL]
    st1w { z15.s }, p0, [x5, #7, MUL VL]
    addvl x5, x5, #16
    st1w { z16.s }, p0, [x5, #-8, MUL VL]
    st1w { z17.s }, p0, [x5, #-7, MUL VL]
    st1w { z18.s }, p0, [x5, #-6, MUL VL]
    st1w { z19.s }, p0, [x5, #-5, MUL VL]
    st1w { z20.s }, p0, [x5, #-4, MUL VL]
    st1w { z21.s }, p0, [x5, #-3, MUL VL]
    st1w { z22.s }, p0, [x5, #-2, MUL VL]
    st1w { z23.s }, p0, [x5, #-1, MUL VL]
    st1w { z24.s }, p0, [x5]
    st1w { z25.s }, p0, [x5, #1, MUL VL]
    st1w { z26.s }, p0, [x5, #2, MUL VL]
    st1w { z27.s }, p0, [x5, #3, MUL VL]
    st1w { z28.s }, p0, [x5, #4, MUL VL]
    st1w { z29.s }, p0, [x5, #5, MUL VL]
    st1w { z30.s }, p0, [x5, #6, MUL VL]
    st1w { z31.s }, p0, [x5, #7, MUL VL]
    addvl x5, x5, #8
    bgt label_2
    subs x7, x7, #0x8
    bgt label_1
KAI_ASM_LABEL(label_6)  // Exit
    ldp d10, d11, [sp, 16]
    ldp d12, d13, [sp, 32]
    ldp d14, d15, [sp, 48]
    ldp d8, d9, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_interleaved_fp32_mla_8x3VL)

    KAI_ASM_END
