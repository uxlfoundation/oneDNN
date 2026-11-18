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

    KAI_ASM_CODE(sve_ffinterleaved_fp32_mla_8x3VL_a64fx)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffinterleaved_fp32_mla_8x3VL_a64fx)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffinterleaved_fp32_mla_8x3VL_a64fx)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffinterleaved_fp32_mla_8x3VL_a64fx)
    stp x20, x21, [sp, -80]!
    stp d8, d9, [sp, 16]
    stp d10, d11, [sp, 32]
    stp d12, d13, [sp, 48]
    stp d14, d15, [sp, 64]
    ptrue p0.b
    ldr x21, [x0, #0]
    ldr x20, [x0, #0x20]
    ldr x8, [x0, #0x50]
    cbz x21, label_7
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x1, [x0, #0x30]
    ldr x7, [x0, #0x8]
    mov x6, x20
    mov x4, x1
KAI_ASM_LABEL(label_2)  // Width loop
    mov x5, x4
    ldr x4, [x0, #0x38]
    cntw x1, ALL, MUL #2
    mov x20, x6
    add x3, x5, x4
    cmp x7, x1
    add x2, x3, x4
    add x4, x2, x4
    bgt label_3
    decw x1
    mov x2, x5
    cmp x7, x1
    bgt label_3
    mov x3, x5
KAI_ASM_LABEL(label_3)  // B setup done
    ldr x1, [x0, #0x10]
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    ld1w { z0.s }, p0/Z, [x5]
    mov z12.b, #0
    mov z13.b, #0
    ld1w { z1.s }, p0/Z, [x3]
    sub x1, x1, #0x1
    mov z14.b, #0
    mov z15.b, #0
    cmp x1, #0x2
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    ld1w { z2.s }, p0/Z, [x2]
    mov z20.b, #0
    mov z21.b, #0
    KAI_ASM_INST(0x8540c283)  // ld1rw { z3.s }, p0/Z, [x20]
    mov z22.b, #0
    mov z23.b, #0
    KAI_ASM_INST(0x8541c284)  // ld1rw { z4.s }, p0/Z, [x20, #4]
    mov z24.b, #0
    mov z25.b, #0
    KAI_ASM_INST(0x8542c285)  // ld1rw { z5.s }, p0/Z, [x20, #8]
    mov z26.b, #0
    mov z27.b, #0
    KAI_ASM_INST(0x8543c286)  // ld1rw { z6.s }, p0/Z, [x20, #12]
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
    blt label_5
KAI_ASM_LABEL(label_4)  // main loop head
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    sub x1, x1, #0x2
    fmla z10.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8544c283)  // ld1rw { z3.s }, p0/Z, [x20, #16]
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c284)  // ld1rw { z4.s }, p0/Z, [x20, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    cmp x1, #0x2
    fmla z16.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x8546c285)  // ld1rw { z5.s }, p0/Z, [x20, #24]
    fmla z17.s, p0/M, z0.s, z6.s
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c286)  // ld1rw { z6.s }, p0/Z, [x20, #28]
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    fmla z22.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8548c283)  // ld1rw { z3.s }, p0/Z, [x20, #32]
    fmla z23.s, p0/M, z0.s, z4.s
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8549c284)  // ld1rw { z4.s }, p0/Z, [x20, #36]
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x854ac285)  // ld1rw { z5.s }, p0/Z, [x20, #40]
    fmla z29.s, p0/M, z0.s, z6.s
    ld1w { z0.s }, p0/Z, [x5, #1, MUL VL]
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
    ld1w { z1.s }, p0/Z, [x3, #1, MUL VL]
    ld1w { z2.s }, p0/Z, [x2, #1, MUL VL]
    addvl x5, x5, #2
    KAI_ASM_INST(0x854bc286)  // ld1rw { z6.s }, p0/Z, [x20, #44]
    addvl x3, x3, #2
    addvl x2, x2, #2
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z9.s, p0/M, z1.s, z3.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z10.s, p0/M, z2.s, z3.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x854cc283)  // ld1rw { z3.s }, p0/Z, [x20, #48]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    KAI_ASM_INST(0x854dc284)  // ld1rw { z4.s }, p0/Z, [x20, #52]
    fmla z16.s, p0/M, z2.s, z5.s
    fmla z17.s, p0/M, z0.s, z6.s
    KAI_ASM_INST(0x854ec285)  // ld1rw { z5.s }, p0/Z, [x20, #56]
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x854fc286)  // ld1rw { z6.s }, p0/Z, [x20, #60]
    add x20, x20, #0x40
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    fmla z22.s, p0/M, z2.s, z3.s
    fmla z23.s, p0/M, z0.s, z4.s
    KAI_ASM_INST(0x8540c283)  // ld1rw { z3.s }, p0/Z, [x20]
    fmla z24.s, p0/M, z1.s, z4.s
    fmla z25.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8541c284)  // ld1rw { z4.s }, p0/Z, [x20, #4]
    fmla z26.s, p0/M, z0.s, z5.s
    fmla z27.s, p0/M, z1.s, z5.s
    fmla z28.s, p0/M, z2.s, z5.s
    fmla z29.s, p0/M, z0.s, z6.s
    ld1w { z0.s }, p0/Z, [x5]
    fmla z30.s, p0/M, z1.s, z6.s
    fmla z31.s, p0/M, z2.s, z6.s
    ld1w { z1.s }, p0/Z, [x3]
    ld1w { z2.s }, p0/Z, [x2]
    KAI_ASM_INST(0x8542c285)  // ld1rw { z5.s }, p0/Z, [x20, #8]
    KAI_ASM_INST(0x8543c286)  // ld1rw { z6.s }, p0/Z, [x20, #12]
    bge label_4
KAI_ASM_LABEL(label_5)  // main loop skip
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    addvl x5, x5, #1
    fmla z10.s, p0/M, z2.s, z3.s
    KAI_ASM_INST(0x8544c283)  // ld1rw { z3.s }, p0/Z, [x20, #16]
    fmla z11.s, p0/M, z0.s, z4.s
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c284)  // ld1rw { z4.s }, p0/Z, [x20, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    addvl x3, x3, #1
    fmla z16.s, p0/M, z2.s, z5.s
    KAI_ASM_INST(0x8546c285)  // ld1rw { z5.s }, p0/Z, [x20, #24]
    fmla z17.s, p0/M, z0.s, z6.s
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c286)  // ld1rw { z6.s }, p0/Z, [x20, #28]
    addvl x2, x2, #1
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    add x20, x20, #0x20
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
    cbz x1, label_6
    ld1w { z0.s }, p0/Z, [x5]
    ld1w { z1.s }, p0/Z, [x3]
    ld1w { z2.s }, p0/Z, [x2]
    KAI_ASM_INST(0x8540c283)  // ld1rw { z3.s }, p0/Z, [x20]
    KAI_ASM_INST(0x8541c284)  // ld1rw { z4.s }, p0/Z, [x20, #4]
    KAI_ASM_INST(0x8542c285)  // ld1rw { z5.s }, p0/Z, [x20, #8]
    KAI_ASM_INST(0x8543c286)  // ld1rw { z6.s }, p0/Z, [x20, #12]
    fmla z8.s, p0/M, z0.s, z3.s
    fmla z9.s, p0/M, z1.s, z3.s
    fmla z10.s, p0/M, z2.s, z3.s
    fmla z11.s, p0/M, z0.s, z4.s
    KAI_ASM_INST(0x8544c283)  // ld1rw { z3.s }, p0/Z, [x20, #16]
    fmla z12.s, p0/M, z1.s, z4.s
    fmla z13.s, p0/M, z2.s, z4.s
    KAI_ASM_INST(0x8545c284)  // ld1rw { z4.s }, p0/Z, [x20, #20]
    fmla z14.s, p0/M, z0.s, z5.s
    fmla z15.s, p0/M, z1.s, z5.s
    fmla z16.s, p0/M, z2.s, z5.s
    fmla z17.s, p0/M, z0.s, z6.s
    KAI_ASM_INST(0x8546c285)  // ld1rw { z5.s }, p0/Z, [x20, #24]
    fmla z18.s, p0/M, z1.s, z6.s
    fmla z19.s, p0/M, z2.s, z6.s
    KAI_ASM_INST(0x8547c286)  // ld1rw { z6.s }, p0/Z, [x20, #28]
    fmla z20.s, p0/M, z0.s, z3.s
    fmla z21.s, p0/M, z1.s, z3.s
    add x20, x20, #0x20
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
KAI_ASM_LABEL(label_6)  // multiply loop done
    decw x7, ALL, MUL #3
    st1w { z8.s }, p0, [x8]
    st1w { z9.s }, p0, [x8, #1, MUL VL]
    st1w { z10.s }, p0, [x8, #2, MUL VL]
    st1w { z11.s }, p0, [x8, #3, MUL VL]
    cmp x7, XZR
    st1w { z12.s }, p0, [x8, #4, MUL VL]
    st1w { z13.s }, p0, [x8, #5, MUL VL]
    st1w { z14.s }, p0, [x8, #6, MUL VL]
    st1w { z15.s }, p0, [x8, #7, MUL VL]
    addvl x8, x8, #16
    st1w { z16.s }, p0, [x8, #-8, MUL VL]
    st1w { z17.s }, p0, [x8, #-7, MUL VL]
    st1w { z18.s }, p0, [x8, #-6, MUL VL]
    st1w { z19.s }, p0, [x8, #-5, MUL VL]
    st1w { z20.s }, p0, [x8, #-4, MUL VL]
    st1w { z21.s }, p0, [x8, #-3, MUL VL]
    st1w { z22.s }, p0, [x8, #-2, MUL VL]
    st1w { z23.s }, p0, [x8, #-1, MUL VL]
    st1w { z24.s }, p0, [x8]
    st1w { z25.s }, p0, [x8, #1, MUL VL]
    st1w { z26.s }, p0, [x8, #2, MUL VL]
    st1w { z27.s }, p0, [x8, #3, MUL VL]
    st1w { z28.s }, p0, [x8, #4, MUL VL]
    st1w { z29.s }, p0, [x8, #5, MUL VL]
    st1w { z30.s }, p0, [x8, #6, MUL VL]
    st1w { z31.s }, p0, [x8, #7, MUL VL]
    addvl x8, x8, #8
    bgt label_2
    subs x21, x21, #0x8
    bgt label_1
KAI_ASM_LABEL(label_7)  // Exit
    ldp d8, d9, [sp, 16]
    ldp d10, d11, [sp, 32]
    ldp d12, d13, [sp, 48]
    ldp d14, d15, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffinterleaved_fp32_mla_8x3VL_a64fx)

    KAI_ASM_END
