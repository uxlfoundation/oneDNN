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

    KAI_ASM_CODE(sve_interleaved_fp32_mla_8x3VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_interleaved_fp32_mla_8x3VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_interleaved_fp32_mla_8x3VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_interleaved_fp32_mla_8x3VL_a64fx)
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
    ld1w { z0.s }, p0/Z, [x3]
    mov z11.b, #0
    mov z12.b, #0
    ld1w { z1.s }, p0/Z, [x3, #1, MUL VL]
    sub x1, x1, #0x1
    mov z13.b, #0
    mov z14.b, #0
    cmp x1, #0x2
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    ld1w { z2.s }, p0/Z, [x3, #2, MUL VL]
    mov z19.b, #0
    mov z20.b, #0
    KAI_ASM_INST(0x8540c0c3)  // ld1rw { z3.s }, p0/Z, [x6]
    mov z21.b, #0
    mov z22.b, #0
    KAI_ASM_INST(0x8541c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #4]
    mov z23.b, #0
    mov z24.b, #0
    KAI_ASM_INST(0x8542c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #8]
    mov z25.b, #0
    mov z26.b, #0
    KAI_ASM_INST(0x8543c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #12]
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    sub x1, x1, #0x2
    fmla z10.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8544c0c3)  // ld1rw { z3.s }, p0/Z, [x6, #16]
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    cmp x1, #0x2
    fmla z16.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x8546c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #24]
    fmla z17.s, p0/M, z0.s, z6.s
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #28]
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    fmla z22.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8548c0c3)  // ld1rw { z3.s }, p0/Z, [x6, #32]
    fmla z23.s, p0/M, z0.s, z4.s
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8549c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #36]
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x854ac0c5)  // ld1rw { z5.s }, p0/Z, [x6, #40]
    fmla z29.s, p0/M, z0.s, z6.s
    ld1w { z0.s }, p0/Z, [x3, #3, MUL VL]
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
    ld1w { z1.s }, p0/Z, [x3, #4, MUL VL]
    ld1w { z2.s }, p0/Z, [x3, #5, MUL VL]
    addvl x3, x3, #6
    KAI_ASM_INST(0x854bc0c6)  // ld1rw { z6.s }, p0/Z, [x6, #44]
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z9.s, p0/M, z1.s, z3.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z10.s, p0/M, z2.s, z3.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x854cc0c3)  // ld1rw { z3.s }, p0/Z, [x6, #48]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    KAI_ASM_INST(0x854dc0c4)  // ld1rw { z4.s }, p0/Z, [x6, #52]
    fmla z16.s, p0/M, z2.s, z5.s
    fmla z17.s, p0/M, z0.s, z6.s
    KAI_ASM_INST(0x854ec0c5)  // ld1rw { z5.s }, p0/Z, [x6, #56]
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x854fc0c6)  // ld1rw { z6.s }, p0/Z, [x6, #60]
    add x6, x6, #0x40
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    fmla z22.s, p0/M, z2.s, z3.s
    fmla z23.s, p0/M, z0.s, z4.s
    KAI_ASM_INST(0x8540c0c3)  // ld1rw { z3.s }, p0/Z, [x6]
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8541c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #4]
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    fmla z29.s, p0/M, z0.s, z6.s
    ld1w { z0.s }, p0/Z, [x3]
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
    ld1w { z1.s }, p0/Z, [x3, #1, MUL VL]
    ld1w { z2.s }, p0/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0x8542c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #8]
    KAI_ASM_INST(0x8543c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #12]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    addvl x3, x3, #3
    fmla z10.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8544c0c3)  // ld1rw { z3.s }, p0/Z, [x6, #16]
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    fmla z16.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x8546c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #24]
    fmla z17.s, p0/M, z0.s, z6.s
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #28]
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    add x6, x6, #0x20
    fmla z22.s, p0/M, z2.s, z3.s
    fmla z23.s, p0/M, z0.s, z4.s
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    fmla z29.s, p0/M, z0.s, z6.s
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
    cbz x1, label_5
    ld1w { z0.s }, p0/Z, [x3]
    ld1w { z1.s }, p0/Z, [x3, #1, MUL VL]
    ld1w { z2.s }, p0/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0x8540c0c3)  // ld1rw { z3.s }, p0/Z, [x6]
    addvl x3, x3, #3
    KAI_ASM_INST(0x8541c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #4]
    KAI_ASM_INST(0x8542c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #8]
    KAI_ASM_INST(0x8543c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #12]
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    fmla z10.s, p0/M, z2.s, z3.s
    fmla z11.s, p0/M, z0.s, z4.s
    KAI_ASM_INST(0x8544c0c3)  // ld1rw { z3.s }, p0/Z, [x6, #16]
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c0c4)  // ld1rw { z4.s }, p0/Z, [x6, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    fmla z16.s, p0/M, z2.s, z5.s
    fmla z17.s, p0/M, z0.s, z6.s
    KAI_ASM_INST(0x8546c0c5)  // ld1rw { z5.s }, p0/Z, [x6, #24]
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c0c6)  // ld1rw { z6.s }, p0/Z, [x6, #28]
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    add x6, x6, #0x20
    fmla z22.s, p0/M, z2.s, z3.s
    fmla z23.s, p0/M, z0.s, z4.s
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    fmla z29.s, p0/M, z0.s, z6.s
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_interleaved_fp32_mla_8x3VL_a64fx)

    KAI_ASM_END
