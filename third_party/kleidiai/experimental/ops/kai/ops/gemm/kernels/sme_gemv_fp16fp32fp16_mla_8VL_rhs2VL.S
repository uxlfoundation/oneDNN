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

    KAI_ASM_CODE(sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    mov x5, #0xfc00
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldrh w1, [x0, #0xa0]
    mov x4, #0x7c00
    cntw x27
    ldr x26, [x0, #0x10]
    mov x25, #0
    cntw x3, ALL, MUL #2
    ldrh w2, [x0, #0xa4]
    ptrue p2.b
    ldr x24, [x0, #0x8]
    cmp x1, x5
    add x23, x26, #0x1
    cset x1, NE
    ldr x22, [x0, #0x30]
    cmp x2, x4
    bic x23, x23, #0x1
    ldr x21, [x0, #0x50]
    add x20, x24, x27
    orr x25, x25, x1, LSL #1
    ldr x8, [x0, #0x40]
    cset x1, NE
    sub x20, x20, #0x1
    lsl x23, x23, #0x1
    orr x25, x25, x1, LSL #1
    udiv x20, x20, x27
    mul x23, x23, x3
KAI_ASM_LABEL(label_1)  // Column loop
    cmp x20, #0x8
    bge label_50
    cmp x20, #0x6
    bgt label_43
    beq label_36
    cmp x20, #0x4
    bgt label_29
    beq label_22
    cmp x20, #0x2
    bgt label_15
    beq label_8
    mov x7, x26
    ldr x6, [x0, #0x20]
    whilelt p1.h, XZR, x24
    cbz x8, label_2
    fmov z17.h, #0
    ld1h { z16.h }, p1/Z, [x8]
    zip1 z24.h, z16.h, z17.h
    fcvt z24.s, p2/m, z24.h
    b label_3
KAI_ASM_LABEL(label_2)  // Width 1: no bias
    fmov z24.s, #0
KAI_ASM_LABEL(label_3)  // Width 1: setup done
    cmp x7, #0x8
    ble label_5
KAI_ASM_LABEL(label_4)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    addvl x22, x22, #2
    ld1rqh { z0.h }, p0/Z, [x6]
    sub x7, x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22]
    addvl x22, x22, #2
    cmp x7, #0x8
    ldnt1h { z3.h }, p2/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    KAI_ASM_INST(0x64a84058)  // fmlalb z24.s, z2.h, z0.h[2]
    KAI_ASM_INST(0x64a84c58)  // fmlalt z24.s, z2.h, z0.h[3]
    KAI_ASM_INST(0x64b04078)  // fmlalb z24.s, z3.h, z0.h[4]
    KAI_ASM_INST(0x64b04c78)  // fmlalt z24.s, z3.h, z0.h[5]
    KAI_ASM_INST(0x64b84098)  // fmlalb z24.s, z4.h, z0.h[6]
    KAI_ASM_INST(0x64b84c98)  // fmlalt z24.s, z4.h, z0.h[7]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z5.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a040b8)  // fmlalb z24.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a04cb8)  // fmlalt z24.s, z5.h, z0.h[1]
    ble label_6
    ldnt1h { z6.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a840d8)  // fmlalb z24.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a84cd8)  // fmlalt z24.s, z6.h, z0.h[3]
    ble label_6
    ldnt1h { z7.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    addvl x22, x22, #2
    KAI_ASM_INST(0x64b040f8)  // fmlalb z24.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b04cf8)  // fmlalt z24.s, z7.h, z0.h[5]
    ble label_6
    ldnt1h { z8.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64b84118)  // fmlalb z24.s, z8.h, z0.h[6]
    KAI_ASM_INST(0x64b84d18)  // fmlalt z24.s, z8.h, z0.h[7]
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    uzp1 z24.h, z24.h, z24.h
    tbz x25, #1, label_7
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
KAI_ASM_LABEL(label_7)  // Width 1: No activation
    st1h { z24.h }, p1, [x21]
    b label_57
KAI_ASM_LABEL(label_8)  // Width 2
    mov x7, x26
    ldr x6, [x0, #0x20]
    whilelt p1.h, XZR, x24
    cbz x8, label_9
    fmov z17.h, #0
    ld1h { z16.h }, p1/Z, [x8]
    zip1 z24.h, z16.h, z17.h
    zip2 z25.h, z16.h, z17.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    b label_10
KAI_ASM_LABEL(label_9)  // Width 2: no bias
    fmov z24.s, #0
    fmov z25.s, #0
KAI_ASM_LABEL(label_10)  // Width 2: setup done
    cmp x7, #0x8
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x22]
    ldnt1h { z6.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z7.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    KAI_ASM_INST(0x64a84078)  // fmlalb z24.s, z3.h, z0.h[2]
    KAI_ASM_INST(0x64a84099)  // fmlalb z25.s, z4.h, z0.h[2]
    KAI_ASM_INST(0x64a84c78)  // fmlalt z24.s, z3.h, z0.h[3]
    KAI_ASM_INST(0x64a84c99)  // fmlalt z25.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x64b040b8)  // fmlalb z24.s, z5.h, z0.h[4]
    KAI_ASM_INST(0x64b040d9)  // fmlalb z25.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b04cb8)  // fmlalt z24.s, z5.h, z0.h[5]
    KAI_ASM_INST(0x64b04cd9)  // fmlalt z25.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b840f8)  // fmlalb z24.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b84119)  // fmlalb z25.s, z8.h, z0.h[6]
    KAI_ASM_INST(0x64b84cf8)  // fmlalt z24.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b84d19)  // fmlalt z25.s, z8.h, z0.h[7]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z9.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z10.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04138)  // fmlalb z24.s, z9.h, z0.h[0]
    KAI_ASM_INST(0x64a04159)  // fmlalb z25.s, z10.h, z0.h[0]
    KAI_ASM_INST(0x64a04d38)  // fmlalt z24.s, z9.h, z0.h[1]
    KAI_ASM_INST(0x64a04d59)  // fmlalt z25.s, z10.h, z0.h[1]
    ble label_13
    ldnt1h { z11.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z12.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a84178)  // fmlalb z24.s, z11.h, z0.h[2]
    KAI_ASM_INST(0x64a84199)  // fmlalb z25.s, z12.h, z0.h[2]
    KAI_ASM_INST(0x64a84d78)  // fmlalt z24.s, z11.h, z0.h[3]
    KAI_ASM_INST(0x64a84d99)  // fmlalt z25.s, z12.h, z0.h[3]
    ble label_13
    ldnt1h { z13.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64b041b8)  // fmlalb z24.s, z13.h, z0.h[4]
    KAI_ASM_INST(0x64b041d9)  // fmlalb z25.s, z14.h, z0.h[4]
    KAI_ASM_INST(0x64b04db8)  // fmlalt z24.s, z13.h, z0.h[5]
    KAI_ASM_INST(0x64b04dd9)  // fmlalt z25.s, z14.h, z0.h[5]
    ble label_13
    ldnt1h { z15.h }, p2/Z, [x22]
    ldnt1h { z16.h }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x64b841f8)  // fmlalb z24.s, z15.h, z0.h[6]
    KAI_ASM_INST(0x64b84219)  // fmlalb z25.s, z16.h, z0.h[6]
    KAI_ASM_INST(0x64b84df8)  // fmlalt z24.s, z15.h, z0.h[7]
    KAI_ASM_INST(0x64b84e19)  // fmlalt z25.s, z16.h, z0.h[7]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    tbz x25, #1, label_14
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    st1h { z24.h }, p1, [x21]
    b label_57
KAI_ASM_LABEL(label_15)  // Width 3
    lsl x1, x27, #0x1
    mov x7, x26
    ldr x6, [x0, #0x20]
    sub x2, x24, x1
    add x1, x22, x23
    whilelt p1.h, XZR, x2
    cbz x8, label_16
    fmov z18.h, #0
    ld1h { z17.h }, p2/Z, [x8]
    ld1h { z16.h }, p1/Z, [x8, #1, MUL VL]
    zip1 z24.h, z17.h, z18.h
    zip2 z25.h, z17.h, z18.h
    zip1 z26.h, z16.h, z18.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    b label_17
KAI_ASM_LABEL(label_16)  // Width 3: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
KAI_ASM_LABEL(label_17)  // Width 3: setup done
    cmp x7, #0x8
    ble label_19
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x1]
    addvl x1, x1, #2
    ldnt1h { z7.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x22]
    ldnt1h { z11.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a84098)  // fmlalb z24.s, z4.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a840b9)  // fmlalb z25.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x64a840da)  // fmlalb z26.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a84c98)  // fmlalt z24.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x64a84cb9)  // fmlalt z25.s, z5.h, z0.h[3]
    KAI_ASM_INST(0x64a84cda)  // fmlalt z26.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64b040f8)  // fmlalb z24.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b04119)  // fmlalb z25.s, z8.h, z0.h[4]
    KAI_ASM_INST(0x64b0413a)  // fmlalb z26.s, z9.h, z0.h[4]
    KAI_ASM_INST(0x64b04cf8)  // fmlalt z24.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b04d19)  // fmlalt z25.s, z8.h, z0.h[5]
    KAI_ASM_INST(0x64b04d3a)  // fmlalt z26.s, z9.h, z0.h[5]
    KAI_ASM_INST(0x64b84158)  // fmlalb z24.s, z10.h, z0.h[6]
    KAI_ASM_INST(0x64b84179)  // fmlalb z25.s, z11.h, z0.h[6]
    KAI_ASM_INST(0x64b8419a)  // fmlalb z26.s, z12.h, z0.h[6]
    KAI_ASM_INST(0x64b84d58)  // fmlalt z24.s, z10.h, z0.h[7]
    KAI_ASM_INST(0x64b84d79)  // fmlalt z25.s, z11.h, z0.h[7]
    KAI_ASM_INST(0x64b84d9a)  // fmlalt z26.s, z12.h, z0.h[7]
    bgt label_18
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z13.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z15.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a041b8)  // fmlalb z24.s, z13.h, z0.h[0]
    KAI_ASM_INST(0x64a041d9)  // fmlalb z25.s, z14.h, z0.h[0]
    KAI_ASM_INST(0x64a041fa)  // fmlalb z26.s, z15.h, z0.h[0]
    KAI_ASM_INST(0x64a04db8)  // fmlalt z24.s, z13.h, z0.h[1]
    KAI_ASM_INST(0x64a04dd9)  // fmlalt z25.s, z14.h, z0.h[1]
    KAI_ASM_INST(0x64a04dfa)  // fmlalt z26.s, z15.h, z0.h[1]
    ble label_20
    ldnt1h { z16.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z17.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z18.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a84218)  // fmlalb z24.s, z16.h, z0.h[2]
    KAI_ASM_INST(0x64a84239)  // fmlalb z25.s, z17.h, z0.h[2]
    KAI_ASM_INST(0x64a8425a)  // fmlalb z26.s, z18.h, z0.h[2]
    KAI_ASM_INST(0x64a84e18)  // fmlalt z24.s, z16.h, z0.h[3]
    KAI_ASM_INST(0x64a84e39)  // fmlalt z25.s, z17.h, z0.h[3]
    KAI_ASM_INST(0x64a84e5a)  // fmlalt z26.s, z18.h, z0.h[3]
    ble label_20
    ldnt1h { z19.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z20.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z21.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b04278)  // fmlalb z24.s, z19.h, z0.h[4]
    KAI_ASM_INST(0x64b04299)  // fmlalb z25.s, z20.h, z0.h[4]
    KAI_ASM_INST(0x64b042ba)  // fmlalb z26.s, z21.h, z0.h[4]
    KAI_ASM_INST(0x64b04e78)  // fmlalt z24.s, z19.h, z0.h[5]
    KAI_ASM_INST(0x64b04e99)  // fmlalt z25.s, z20.h, z0.h[5]
    KAI_ASM_INST(0x64b04eba)  // fmlalt z26.s, z21.h, z0.h[5]
    ble label_20
    ldnt1h { z22.h }, p2/Z, [x22]
    ldnt1h { z23.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b842d8)  // fmlalb z24.s, z22.h, z0.h[6]
    KAI_ASM_INST(0x64b842f9)  // fmlalb z25.s, z23.h, z0.h[6]
    KAI_ASM_INST(0x64b8403a)  // fmlalb z26.s, z1.h, z0.h[6]
    KAI_ASM_INST(0x64b84ed8)  // fmlalt z24.s, z22.h, z0.h[7]
    KAI_ASM_INST(0x64b84ef9)  // fmlalt z25.s, z23.h, z0.h[7]
    KAI_ASM_INST(0x64b84c3a)  // fmlalt z26.s, z1.h, z0.h[7]
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    uzp1 z25.h, z25.h, z25.h
    tbz x25, #1, label_21
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
KAI_ASM_LABEL(label_21)  // Width 3: No activation
    st1h { z24.h }, p2, [x21]
    st1h { z25.h }, p1, [x21, #1, MUL VL]
    b label_57
KAI_ASM_LABEL(label_22)  // Width 4
    lsl x1, x27, #0x1
    mov x7, x26
    ldr x6, [x0, #0x20]
    sub x2, x24, x1
    add x1, x22, x23
    whilelt p1.h, XZR, x2
    cbz x8, label_23
    fmov z18.h, #0
    ld1h { z17.h }, p2/Z, [x8]
    ld1h { z16.h }, p1/Z, [x8, #1, MUL VL]
    zip1 z24.h, z17.h, z18.h
    zip2 z25.h, z17.h, z18.h
    zip1 z26.h, z16.h, z18.h
    zip2 z27.h, z16.h, z18.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    fcvt z27.s, p2/m, z27.h
    b label_24
KAI_ASM_LABEL(label_23)  // Width 4: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
KAI_ASM_LABEL(label_24)  // Width 4: setup done
    cmp x7, #0x8
    ble label_26
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x1]
    ldnt1h { z4.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a0409b)  // fmlalb z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x1, #1, MUL VL]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a04c9b)  // fmlalt z27.s, z4.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a840b8)  // fmlalb z24.s, z5.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x1, #1, MUL VL]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a840d9)  // fmlalb z25.s, z6.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a840fa)  // fmlalb z26.s, z7.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a8411b)  // fmlalb z27.s, z8.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a84cb8)  // fmlalt z24.s, z5.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x1, #1, MUL VL]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a84cd9)  // fmlalt z25.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a84cfa)  // fmlalt z26.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a84d1b)  // fmlalt z27.s, z8.h, z0.h[3]
    KAI_ASM_INST(0x64b04138)  // fmlalb z24.s, z9.h, z0.h[4]
    KAI_ASM_INST(0x64b04159)  // fmlalb z25.s, z10.h, z0.h[4]
    KAI_ASM_INST(0x64b0417a)  // fmlalb z26.s, z11.h, z0.h[4]
    KAI_ASM_INST(0x64b0419b)  // fmlalb z27.s, z12.h, z0.h[4]
    KAI_ASM_INST(0x64b04d38)  // fmlalt z24.s, z9.h, z0.h[5]
    KAI_ASM_INST(0x64b04d59)  // fmlalt z25.s, z10.h, z0.h[5]
    KAI_ASM_INST(0x64b04d7a)  // fmlalt z26.s, z11.h, z0.h[5]
    KAI_ASM_INST(0x64b04d9b)  // fmlalt z27.s, z12.h, z0.h[5]
    KAI_ASM_INST(0x64b841b8)  // fmlalb z24.s, z13.h, z0.h[6]
    KAI_ASM_INST(0x64b841d9)  // fmlalb z25.s, z14.h, z0.h[6]
    KAI_ASM_INST(0x64b841fa)  // fmlalb z26.s, z15.h, z0.h[6]
    KAI_ASM_INST(0x64b8421b)  // fmlalb z27.s, z16.h, z0.h[6]
    KAI_ASM_INST(0x64b84db8)  // fmlalt z24.s, z13.h, z0.h[7]
    KAI_ASM_INST(0x64b84dd9)  // fmlalt z25.s, z14.h, z0.h[7]
    KAI_ASM_INST(0x64b84dfa)  // fmlalt z26.s, z15.h, z0.h[7]
    KAI_ASM_INST(0x64b84e1b)  // fmlalt z27.s, z16.h, z0.h[7]
    bgt label_25
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z17.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z18.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z19.h }, p2/Z, [x1]
    ldnt1h { z20.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a04238)  // fmlalb z24.s, z17.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04259)  // fmlalb z25.s, z18.h, z0.h[0]
    KAI_ASM_INST(0x64a0427a)  // fmlalb z26.s, z19.h, z0.h[0]
    KAI_ASM_INST(0x64a0429b)  // fmlalb z27.s, z20.h, z0.h[0]
    KAI_ASM_INST(0x64a04e38)  // fmlalt z24.s, z17.h, z0.h[1]
    KAI_ASM_INST(0x64a04e59)  // fmlalt z25.s, z18.h, z0.h[1]
    KAI_ASM_INST(0x64a04e7a)  // fmlalt z26.s, z19.h, z0.h[1]
    KAI_ASM_INST(0x64a04e9b)  // fmlalt z27.s, z20.h, z0.h[1]
    ble label_27
    ldnt1h { z21.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z22.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z23.h }, p2/Z, [x1]
    ldnt1h { z1.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a842b8)  // fmlalb z24.s, z21.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a842d9)  // fmlalb z25.s, z22.h, z0.h[2]
    KAI_ASM_INST(0x64a842fa)  // fmlalb z26.s, z23.h, z0.h[2]
    KAI_ASM_INST(0x64a8403b)  // fmlalb z27.s, z1.h, z0.h[2]
    KAI_ASM_INST(0x64a84eb8)  // fmlalt z24.s, z21.h, z0.h[3]
    KAI_ASM_INST(0x64a84ed9)  // fmlalt z25.s, z22.h, z0.h[3]
    KAI_ASM_INST(0x64a84efa)  // fmlalt z26.s, z23.h, z0.h[3]
    KAI_ASM_INST(0x64a84c3b)  // fmlalt z27.s, z1.h, z0.h[3]
    ble label_27
    ldnt1h { z2.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z3.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z4.h }, p2/Z, [x1]
    ldnt1h { z5.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b04058)  // fmlalb z24.s, z2.h, z0.h[4]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b04079)  // fmlalb z25.s, z3.h, z0.h[4]
    KAI_ASM_INST(0x64b0409a)  // fmlalb z26.s, z4.h, z0.h[4]
    KAI_ASM_INST(0x64b040bb)  // fmlalb z27.s, z5.h, z0.h[4]
    KAI_ASM_INST(0x64b04c58)  // fmlalt z24.s, z2.h, z0.h[5]
    KAI_ASM_INST(0x64b04c79)  // fmlalt z25.s, z3.h, z0.h[5]
    KAI_ASM_INST(0x64b04c9a)  // fmlalt z26.s, z4.h, z0.h[5]
    KAI_ASM_INST(0x64b04cbb)  // fmlalt z27.s, z5.h, z0.h[5]
    ble label_27
    ldnt1h { z6.h }, p2/Z, [x22]
    ldnt1h { z7.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x1]
    ldnt1h { z9.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b840d8)  // fmlalb z24.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b840f9)  // fmlalb z25.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b8411a)  // fmlalb z26.s, z8.h, z0.h[6]
    KAI_ASM_INST(0x64b8413b)  // fmlalb z27.s, z9.h, z0.h[6]
    KAI_ASM_INST(0x64b84cd8)  // fmlalt z24.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b84cf9)  // fmlalt z25.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b84d1a)  // fmlalt z26.s, z8.h, z0.h[7]
    KAI_ASM_INST(0x64b84d3b)  // fmlalt z27.s, z9.h, z0.h[7]
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    fcvt z26.h, p2/m, z27.s
    uzp1 z25.h, z25.h, z26.h
    tbz x25, #1, label_28
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
KAI_ASM_LABEL(label_28)  // Width 4: No activation
    st1h { z24.h }, p2, [x21]
    st1h { z25.h }, p1, [x21, #1, MUL VL]
    b label_57
KAI_ASM_LABEL(label_29)  // Width 5
    lsl x2, x27, #0x1
    mov x1, #0x2
    ldr x6, [x0, #0x20]
    msub x3, x2, x1, x24
    mov x7, x26
    add x2, x22, x23
    add x1, x22, x23, LSL #1
    whilelt p1.h, XZR, x3
    cbz x8, label_30
    fmov z19.h, #0
    ld1h { z18.h }, p2/Z, [x8]
    ld1h { z17.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z16.h }, p1/Z, [x8, #2, MUL VL]
    zip1 z24.h, z18.h, z19.h
    zip2 z25.h, z18.h, z19.h
    zip1 z26.h, z17.h, z19.h
    zip2 z27.h, z17.h, z19.h
    zip1 z28.h, z16.h, z19.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    fcvt z27.s, p2/m, z27.h
    fcvt z28.s, p2/m, z28.h
    b label_31
KAI_ASM_LABEL(label_30)  // Width 5: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
KAI_ASM_LABEL(label_31)  // Width 5: setup done
    cmp x7, #0x8
    ble label_33
KAI_ASM_LABEL(label_32)  // Width 5: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x2]
    ldnt1h { z4.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z5.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a0409b)  // fmlalb z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a040bc)  // fmlalb z28.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04c9b)  // fmlalt z27.s, z4.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a04cbc)  // fmlalt z28.s, z5.h, z0.h[1]
    KAI_ASM_INST(0x64a840d8)  // fmlalb z24.s, z6.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a840f9)  // fmlalb z25.s, z7.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a8411a)  // fmlalb z26.s, z8.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a8413b)  // fmlalb z27.s, z9.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a8415c)  // fmlalb z28.s, z10.h, z0.h[2]
    KAI_ASM_INST(0x64a84cd8)  // fmlalt z24.s, z6.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a84cf9)  // fmlalt z25.s, z7.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a84d1a)  // fmlalt z26.s, z8.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a84d3b)  // fmlalt z27.s, z9.h, z0.h[3]
    ldnt1h { z19.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a84d5c)  // fmlalt z28.s, z10.h, z0.h[3]
    KAI_ASM_INST(0x64b04178)  // fmlalb z24.s, z11.h, z0.h[4]
    ldnt1h { z20.h }, p2/Z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b04199)  // fmlalb z25.s, z12.h, z0.h[4]
    KAI_ASM_INST(0x64b041ba)  // fmlalb z26.s, z13.h, z0.h[4]
    KAI_ASM_INST(0x64b041db)  // fmlalb z27.s, z14.h, z0.h[4]
    KAI_ASM_INST(0x64b041fc)  // fmlalb z28.s, z15.h, z0.h[4]
    KAI_ASM_INST(0x64b04d78)  // fmlalt z24.s, z11.h, z0.h[5]
    KAI_ASM_INST(0x64b04d99)  // fmlalt z25.s, z12.h, z0.h[5]
    KAI_ASM_INST(0x64b04dba)  // fmlalt z26.s, z13.h, z0.h[5]
    KAI_ASM_INST(0x64b04ddb)  // fmlalt z27.s, z14.h, z0.h[5]
    KAI_ASM_INST(0x64b04dfc)  // fmlalt z28.s, z15.h, z0.h[5]
    KAI_ASM_INST(0x64b84218)  // fmlalb z24.s, z16.h, z0.h[6]
    KAI_ASM_INST(0x64b84239)  // fmlalb z25.s, z17.h, z0.h[6]
    KAI_ASM_INST(0x64b8425a)  // fmlalb z26.s, z18.h, z0.h[6]
    KAI_ASM_INST(0x64b8427b)  // fmlalb z27.s, z19.h, z0.h[6]
    KAI_ASM_INST(0x64b8429c)  // fmlalb z28.s, z20.h, z0.h[6]
    KAI_ASM_INST(0x64b84e18)  // fmlalt z24.s, z16.h, z0.h[7]
    KAI_ASM_INST(0x64b84e39)  // fmlalt z25.s, z17.h, z0.h[7]
    KAI_ASM_INST(0x64b84e5a)  // fmlalt z26.s, z18.h, z0.h[7]
    KAI_ASM_INST(0x64b84e7b)  // fmlalt z27.s, z19.h, z0.h[7]
    KAI_ASM_INST(0x64b84e9c)  // fmlalt z28.s, z20.h, z0.h[7]
    bgt label_32
KAI_ASM_LABEL(label_33)  // Width 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z21.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z22.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z23.h }, p2/Z, [x2]
    ldnt1h { z1.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a042b8)  // fmlalb z24.s, z21.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z2.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a042d9)  // fmlalb z25.s, z22.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a042fa)  // fmlalb z26.s, z23.h, z0.h[0]
    KAI_ASM_INST(0x64a0403b)  // fmlalb z27.s, z1.h, z0.h[0]
    KAI_ASM_INST(0x64a0405c)  // fmlalb z28.s, z2.h, z0.h[0]
    KAI_ASM_INST(0x64a04eb8)  // fmlalt z24.s, z21.h, z0.h[1]
    KAI_ASM_INST(0x64a04ed9)  // fmlalt z25.s, z22.h, z0.h[1]
    KAI_ASM_INST(0x64a04efa)  // fmlalt z26.s, z23.h, z0.h[1]
    KAI_ASM_INST(0x64a04c3b)  // fmlalt z27.s, z1.h, z0.h[1]
    KAI_ASM_INST(0x64a04c5c)  // fmlalt z28.s, z2.h, z0.h[1]
    ble label_34
    ldnt1h { z3.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z4.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z5.h }, p2/Z, [x2]
    ldnt1h { z6.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a84078)  // fmlalb z24.s, z3.h, z0.h[2]
    addvl x2, x2, #2
    ldnt1h { z7.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a84099)  // fmlalb z25.s, z4.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a840ba)  // fmlalb z26.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x64a840db)  // fmlalb z27.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a840fc)  // fmlalb z28.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a84c78)  // fmlalt z24.s, z3.h, z0.h[3]
    KAI_ASM_INST(0x64a84c99)  // fmlalt z25.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x64a84cba)  // fmlalt z26.s, z5.h, z0.h[3]
    KAI_ASM_INST(0x64a84cdb)  // fmlalt z27.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a84cfc)  // fmlalt z28.s, z7.h, z0.h[3]
    ble label_34
    ldnt1h { z8.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z9.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z10.h }, p2/Z, [x2]
    ldnt1h { z11.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b04118)  // fmlalb z24.s, z8.h, z0.h[4]
    addvl x2, x2, #2
    ldnt1h { z12.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b04139)  // fmlalb z25.s, z9.h, z0.h[4]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b0415a)  // fmlalb z26.s, z10.h, z0.h[4]
    KAI_ASM_INST(0x64b0417b)  // fmlalb z27.s, z11.h, z0.h[4]
    KAI_ASM_INST(0x64b0419c)  // fmlalb z28.s, z12.h, z0.h[4]
    KAI_ASM_INST(0x64b04d18)  // fmlalt z24.s, z8.h, z0.h[5]
    KAI_ASM_INST(0x64b04d39)  // fmlalt z25.s, z9.h, z0.h[5]
    KAI_ASM_INST(0x64b04d5a)  // fmlalt z26.s, z10.h, z0.h[5]
    KAI_ASM_INST(0x64b04d7b)  // fmlalt z27.s, z11.h, z0.h[5]
    KAI_ASM_INST(0x64b04d9c)  // fmlalt z28.s, z12.h, z0.h[5]
    ble label_34
    ldnt1h { z13.h }, p2/Z, [x22]
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x2]
    ldnt1h { z16.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b841b8)  // fmlalb z24.s, z13.h, z0.h[6]
    ldnt1h { z17.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b841d9)  // fmlalb z25.s, z14.h, z0.h[6]
    KAI_ASM_INST(0x64b841fa)  // fmlalb z26.s, z15.h, z0.h[6]
    KAI_ASM_INST(0x64b8421b)  // fmlalb z27.s, z16.h, z0.h[6]
    KAI_ASM_INST(0x64b8423c)  // fmlalb z28.s, z17.h, z0.h[6]
    KAI_ASM_INST(0x64b84db8)  // fmlalt z24.s, z13.h, z0.h[7]
    KAI_ASM_INST(0x64b84dd9)  // fmlalt z25.s, z14.h, z0.h[7]
    KAI_ASM_INST(0x64b84dfa)  // fmlalt z26.s, z15.h, z0.h[7]
    KAI_ASM_INST(0x64b84e1b)  // fmlalt z27.s, z16.h, z0.h[7]
    KAI_ASM_INST(0x64b84e3c)  // fmlalt z28.s, z17.h, z0.h[7]
KAI_ASM_LABEL(label_34)  // Width 5: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    fcvt z26.h, p2/m, z27.s
    uzp1 z25.h, z25.h, z26.h
    fcvt z26.h, p2/m, z28.s
    uzp1 z26.h, z26.h, z26.h
    tbz x25, #1, label_35
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
KAI_ASM_LABEL(label_35)  // Width 5: No activation
    st1h { z24.h }, p2, [x21]
    st1h { z25.h }, p2, [x21, #1, MUL VL]
    st1h { z26.h }, p1, [x21, #2, MUL VL]
    b label_57
KAI_ASM_LABEL(label_36)  // Width 6
    lsl x2, x27, #0x1
    mov x1, #0x2
    ldr x6, [x0, #0x20]
    msub x3, x2, x1, x24
    mov x7, x26
    add x2, x22, x23
    add x1, x22, x23, LSL #1
    whilelt p1.h, XZR, x3
    cbz x8, label_37
    fmov z19.h, #0
    ld1h { z18.h }, p2/Z, [x8]
    ld1h { z17.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z16.h }, p1/Z, [x8, #2, MUL VL]
    zip1 z24.h, z18.h, z19.h
    zip2 z25.h, z18.h, z19.h
    zip1 z26.h, z17.h, z19.h
    zip2 z27.h, z17.h, z19.h
    zip1 z28.h, z16.h, z19.h
    zip2 z29.h, z16.h, z19.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    fcvt z27.s, p2/m, z27.h
    fcvt z28.s, p2/m, z28.h
    fcvt z29.s, p2/m, z29.h
    b label_38
KAI_ASM_LABEL(label_37)  // Width 6: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
    fmov z29.s, #0
KAI_ASM_LABEL(label_38)  // Width 6: setup done
    cmp x7, #0x8
    ble label_40
KAI_ASM_LABEL(label_39)  // Width 6: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x2]
    ldnt1h { z4.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z5.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a0409b)  // fmlalb z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a040bc)  // fmlalb z28.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a040dd)  // fmlalb z29.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a04c9b)  // fmlalt z27.s, z4.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a04cbc)  // fmlalt z28.s, z5.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a840f8)  // fmlalb z24.s, z7.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a04cdd)  // fmlalt z29.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a84119)  // fmlalb z25.s, z8.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a8413a)  // fmlalb z26.s, z9.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a8415b)  // fmlalb z27.s, z10.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a8417c)  // fmlalb z28.s, z11.h, z0.h[2]
    KAI_ASM_INST(0x64a84cf8)  // fmlalt z24.s, z7.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a8419d)  // fmlalb z29.s, z12.h, z0.h[2]
    KAI_ASM_INST(0x64a84d19)  // fmlalt z25.s, z8.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a84d3a)  // fmlalt z26.s, z9.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x1, #1, MUL VL]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a84d5b)  // fmlalt z27.s, z10.h, z0.h[3]
    ldnt1h { z19.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a84d7c)  // fmlalt z28.s, z11.h, z0.h[3]
    KAI_ASM_INST(0x64b041b8)  // fmlalb z24.s, z13.h, z0.h[4]
    ldnt1h { z20.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a84d9d)  // fmlalt z29.s, z12.h, z0.h[3]
    KAI_ASM_INST(0x64b041d9)  // fmlalb z25.s, z14.h, z0.h[4]
    ldnt1h { z21.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64b041fa)  // fmlalb z26.s, z15.h, z0.h[4]
    ldnt1h { z22.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64b0421b)  // fmlalb z27.s, z16.h, z0.h[4]
    ldnt1h { z23.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b0423c)  // fmlalb z28.s, z17.h, z0.h[4]
    KAI_ASM_INST(0x64b04db8)  // fmlalt z24.s, z13.h, z0.h[5]
    ldnt1h { z1.h }, p2/Z, [x1, #1, MUL VL]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b0425d)  // fmlalb z29.s, z18.h, z0.h[4]
    KAI_ASM_INST(0x64b04dd9)  // fmlalt z25.s, z14.h, z0.h[5]
    KAI_ASM_INST(0x64b04dfa)  // fmlalt z26.s, z15.h, z0.h[5]
    KAI_ASM_INST(0x64b04e1b)  // fmlalt z27.s, z16.h, z0.h[5]
    KAI_ASM_INST(0x64b04e3c)  // fmlalt z28.s, z17.h, z0.h[5]
    KAI_ASM_INST(0x64b84278)  // fmlalb z24.s, z19.h, z0.h[6]
    KAI_ASM_INST(0x64b04e5d)  // fmlalt z29.s, z18.h, z0.h[5]
    KAI_ASM_INST(0x64b84299)  // fmlalb z25.s, z20.h, z0.h[6]
    KAI_ASM_INST(0x64b842ba)  // fmlalb z26.s, z21.h, z0.h[6]
    KAI_ASM_INST(0x64b842db)  // fmlalb z27.s, z22.h, z0.h[6]
    KAI_ASM_INST(0x64b842fc)  // fmlalb z28.s, z23.h, z0.h[6]
    KAI_ASM_INST(0x64b84e78)  // fmlalt z24.s, z19.h, z0.h[7]
    KAI_ASM_INST(0x64b8403d)  // fmlalb z29.s, z1.h, z0.h[6]
    KAI_ASM_INST(0x64b84e99)  // fmlalt z25.s, z20.h, z0.h[7]
    KAI_ASM_INST(0x64b84eba)  // fmlalt z26.s, z21.h, z0.h[7]
    KAI_ASM_INST(0x64b84edb)  // fmlalt z27.s, z22.h, z0.h[7]
    KAI_ASM_INST(0x64b84efc)  // fmlalt z28.s, z23.h, z0.h[7]
    KAI_ASM_INST(0x64b84c3d)  // fmlalt z29.s, z1.h, z0.h[7]
    bgt label_39
KAI_ASM_LABEL(label_40)  // Width 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z2.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z3.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z4.h }, p2/Z, [x2]
    ldnt1h { z5.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04058)  // fmlalb z24.s, z2.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z6.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a04079)  // fmlalb z25.s, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a0409a)  // fmlalb z26.s, z4.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a040bb)  // fmlalb z27.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a040dc)  // fmlalb z28.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a04c58)  // fmlalt z24.s, z2.h, z0.h[1]
    KAI_ASM_INST(0x64a040fd)  // fmlalb z29.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a04c79)  // fmlalt z25.s, z3.h, z0.h[1]
    KAI_ASM_INST(0x64a04c9a)  // fmlalt z26.s, z4.h, z0.h[1]
    KAI_ASM_INST(0x64a04cbb)  // fmlalt z27.s, z5.h, z0.h[1]
    KAI_ASM_INST(0x64a04cdc)  // fmlalt z28.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a04cfd)  // fmlalt z29.s, z7.h, z0.h[1]
    ble label_41
    ldnt1h { z8.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z9.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z10.h }, p2/Z, [x2]
    ldnt1h { z11.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a84118)  // fmlalb z24.s, z8.h, z0.h[2]
    addvl x2, x2, #2
    ldnt1h { z12.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a84139)  // fmlalb z25.s, z9.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a8415a)  // fmlalb z26.s, z10.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a8417b)  // fmlalb z27.s, z11.h, z0.h[2]
    KAI_ASM_INST(0x64a8419c)  // fmlalb z28.s, z12.h, z0.h[2]
    KAI_ASM_INST(0x64a84d18)  // fmlalt z24.s, z8.h, z0.h[3]
    KAI_ASM_INST(0x64a841bd)  // fmlalb z29.s, z13.h, z0.h[2]
    KAI_ASM_INST(0x64a84d39)  // fmlalt z25.s, z9.h, z0.h[3]
    KAI_ASM_INST(0x64a84d5a)  // fmlalt z26.s, z10.h, z0.h[3]
    KAI_ASM_INST(0x64a84d7b)  // fmlalt z27.s, z11.h, z0.h[3]
    KAI_ASM_INST(0x64a84d9c)  // fmlalt z28.s, z12.h, z0.h[3]
    KAI_ASM_INST(0x64a84dbd)  // fmlalt z29.s, z13.h, z0.h[3]
    ble label_41
    ldnt1h { z14.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z15.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z16.h }, p2/Z, [x2]
    ldnt1h { z17.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b041d8)  // fmlalb z24.s, z14.h, z0.h[4]
    addvl x2, x2, #2
    ldnt1h { z18.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b041f9)  // fmlalb z25.s, z15.h, z0.h[4]
    ldnt1h { z19.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b0421a)  // fmlalb z26.s, z16.h, z0.h[4]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b0423b)  // fmlalb z27.s, z17.h, z0.h[4]
    KAI_ASM_INST(0x64b0425c)  // fmlalb z28.s, z18.h, z0.h[4]
    KAI_ASM_INST(0x64b04dd8)  // fmlalt z24.s, z14.h, z0.h[5]
    KAI_ASM_INST(0x64b0427d)  // fmlalb z29.s, z19.h, z0.h[4]
    KAI_ASM_INST(0x64b04df9)  // fmlalt z25.s, z15.h, z0.h[5]
    KAI_ASM_INST(0x64b04e1a)  // fmlalt z26.s, z16.h, z0.h[5]
    KAI_ASM_INST(0x64b04e3b)  // fmlalt z27.s, z17.h, z0.h[5]
    KAI_ASM_INST(0x64b04e5c)  // fmlalt z28.s, z18.h, z0.h[5]
    KAI_ASM_INST(0x64b04e7d)  // fmlalt z29.s, z19.h, z0.h[5]
    ble label_41
    ldnt1h { z20.h }, p2/Z, [x22]
    ldnt1h { z21.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x2]
    ldnt1h { z23.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b84298)  // fmlalb z24.s, z20.h, z0.h[6]
    ldnt1h { z1.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b842b9)  // fmlalb z25.s, z21.h, z0.h[6]
    ldnt1h { z2.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b842da)  // fmlalb z26.s, z22.h, z0.h[6]
    KAI_ASM_INST(0x64b842fb)  // fmlalb z27.s, z23.h, z0.h[6]
    KAI_ASM_INST(0x64b8403c)  // fmlalb z28.s, z1.h, z0.h[6]
    KAI_ASM_INST(0x64b84e98)  // fmlalt z24.s, z20.h, z0.h[7]
    KAI_ASM_INST(0x64b8405d)  // fmlalb z29.s, z2.h, z0.h[6]
    KAI_ASM_INST(0x64b84eb9)  // fmlalt z25.s, z21.h, z0.h[7]
    KAI_ASM_INST(0x64b84eda)  // fmlalt z26.s, z22.h, z0.h[7]
    KAI_ASM_INST(0x64b84efb)  // fmlalt z27.s, z23.h, z0.h[7]
    KAI_ASM_INST(0x64b84c3c)  // fmlalt z28.s, z1.h, z0.h[7]
    KAI_ASM_INST(0x64b84c5d)  // fmlalt z29.s, z2.h, z0.h[7]
KAI_ASM_LABEL(label_41)  // Width 6: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    fcvt z26.h, p2/m, z27.s
    fcvt z27.h, p2/m, z29.s
    uzp1 z25.h, z25.h, z26.h
    fcvt z26.h, p2/m, z28.s
    uzp1 z26.h, z26.h, z27.h
    tbz x25, #1, label_42
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
KAI_ASM_LABEL(label_42)  // Width 6: No activation
    st1h { z24.h }, p2, [x21]
    st1h { z25.h }, p2, [x21, #1, MUL VL]
    st1h { z26.h }, p1, [x21, #2, MUL VL]
    b label_57
KAI_ASM_LABEL(label_43)  // Width 7
    lsl x2, x27, #0x1
    mov x1, #0x3
    ldr x6, [x0, #0x20]
    add x4, x22, x23, LSL #1
    msub x3, x2, x1, x24
    mov x7, x26
    add x2, x22, x23
    add x1, x4, x23
    whilelt p1.h, XZR, x3
    cbz x8, label_44
    fmov z20.h, #0
    ld1h { z19.h }, p2/Z, [x8]
    ld1h { z18.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z17.h }, p2/Z, [x8, #2, MUL VL]
    ld1h { z16.h }, p1/Z, [x8, #3, MUL VL]
    zip1 z24.h, z19.h, z20.h
    zip2 z25.h, z19.h, z20.h
    zip1 z26.h, z18.h, z20.h
    zip2 z27.h, z18.h, z20.h
    zip1 z28.h, z17.h, z20.h
    zip2 z29.h, z17.h, z20.h
    zip1 z30.h, z16.h, z20.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    fcvt z27.s, p2/m, z27.h
    fcvt z28.s, p2/m, z28.h
    fcvt z29.s, p2/m, z29.h
    fcvt z30.s, p2/m, z30.h
    b label_45
KAI_ASM_LABEL(label_44)  // Width 7: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
    fmov z29.s, #0
    fmov z30.s, #0
KAI_ASM_LABEL(label_45)  // Width 7: setup done
    cmp x7, #0x8
    ble label_47
KAI_ASM_LABEL(label_46)  // Width 7: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x2]
    ldnt1h { z4.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z5.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    addvl x4, x4, #2
    ldnt1h { z7.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a0409b)  // fmlalb z27.s, z4.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a040bc)  // fmlalb z28.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a040dd)  // fmlalb z29.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a040fe)  // fmlalb z30.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a04c9b)  // fmlalt z27.s, z4.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a04cbc)  // fmlalt z28.s, z5.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64a84118)  // fmlalb z24.s, z8.h, z0.h[2]
    KAI_ASM_INST(0x64a04cdd)  // fmlalt z29.s, z6.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64a84139)  // fmlalb z25.s, z9.h, z0.h[2]
    addvl x4, x4, #2
    KAI_ASM_INST(0x64a04cfe)  // fmlalt z30.s, z7.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a8415a)  // fmlalb z26.s, z10.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a8417b)  // fmlalb z27.s, z11.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a8419c)  // fmlalb z28.s, z12.h, z0.h[2]
    KAI_ASM_INST(0x64a84d18)  // fmlalt z24.s, z8.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a841bd)  // fmlalb z29.s, z13.h, z0.h[2]
    KAI_ASM_INST(0x64a84d39)  // fmlalt z25.s, z9.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a841de)  // fmlalb z30.s, z14.h, z0.h[2]
    KAI_ASM_INST(0x64a84d5a)  // fmlalt z26.s, z10.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a84d7b)  // fmlalt z27.s, z11.h, z0.h[3]
    ldnt1h { z19.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64a84d9c)  // fmlalt z28.s, z12.h, z0.h[3]
    ldnt1h { z20.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64b041f8)  // fmlalb z24.s, z15.h, z0.h[4]
    addvl x4, x4, #2
    KAI_ASM_INST(0x64a84dbd)  // fmlalt z29.s, z13.h, z0.h[3]
    ldnt1h { z21.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b04219)  // fmlalb z25.s, z16.h, z0.h[4]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a84dde)  // fmlalt z30.s, z14.h, z0.h[3]
    KAI_ASM_INST(0x64b0423a)  // fmlalb z26.s, z17.h, z0.h[4]
    ldnt1h { z22.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64b0425b)  // fmlalb z27.s, z18.h, z0.h[4]
    ldnt1h { z23.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64b0427c)  // fmlalb z28.s, z19.h, z0.h[4]
    KAI_ASM_INST(0x64b04df8)  // fmlalt z24.s, z15.h, z0.h[5]
    ldnt1h { z1.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64b0429d)  // fmlalb z29.s, z20.h, z0.h[4]
    KAI_ASM_INST(0x64b04e19)  // fmlalt z25.s, z16.h, z0.h[5]
    ldnt1h { z2.h }, p2/Z, [x2, #1, MUL VL]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64b042be)  // fmlalb z30.s, z21.h, z0.h[4]
    KAI_ASM_INST(0x64b04e3a)  // fmlalt z26.s, z17.h, z0.h[5]
    ldnt1h { z3.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64b04e5b)  // fmlalt z27.s, z18.h, z0.h[5]
    ldnt1h { z4.h }, p2/Z, [x4, #1, MUL VL]
    addvl x4, x4, #2
    KAI_ASM_INST(0x64b04e7c)  // fmlalt z28.s, z19.h, z0.h[5]
    ldnt1h { z5.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b842d8)  // fmlalb z24.s, z22.h, z0.h[6]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b04e9d)  // fmlalt z29.s, z20.h, z0.h[5]
    KAI_ASM_INST(0x64b842f9)  // fmlalb z25.s, z23.h, z0.h[6]
    KAI_ASM_INST(0x64b04ebe)  // fmlalt z30.s, z21.h, z0.h[5]
    KAI_ASM_INST(0x64b8403a)  // fmlalb z26.s, z1.h, z0.h[6]
    KAI_ASM_INST(0x64b8405b)  // fmlalb z27.s, z2.h, z0.h[6]
    KAI_ASM_INST(0x64b8407c)  // fmlalb z28.s, z3.h, z0.h[6]
    KAI_ASM_INST(0x64b84ed8)  // fmlalt z24.s, z22.h, z0.h[7]
    KAI_ASM_INST(0x64b8409d)  // fmlalb z29.s, z4.h, z0.h[6]
    KAI_ASM_INST(0x64b84ef9)  // fmlalt z25.s, z23.h, z0.h[7]
    KAI_ASM_INST(0x64b840be)  // fmlalb z30.s, z5.h, z0.h[6]
    KAI_ASM_INST(0x64b84c3a)  // fmlalt z26.s, z1.h, z0.h[7]
    KAI_ASM_INST(0x64b84c5b)  // fmlalt z27.s, z2.h, z0.h[7]
    KAI_ASM_INST(0x64b84c7c)  // fmlalt z28.s, z3.h, z0.h[7]
    KAI_ASM_INST(0x64b84c9d)  // fmlalt z29.s, z4.h, z0.h[7]
    KAI_ASM_INST(0x64b84cbe)  // fmlalt z30.s, z5.h, z0.h[7]
    bgt label_46
KAI_ASM_LABEL(label_47)  // Width 7: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z6.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z7.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z8.h }, p2/Z, [x2]
    ldnt1h { z9.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a040d8)  // fmlalb z24.s, z6.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z10.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64a040f9)  // fmlalb z25.s, z7.h, z0.h[0]
    ldnt1h { z11.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64a0411a)  // fmlalb z26.s, z8.h, z0.h[0]
    addvl x4, x4, #2
    ldnt1h { z12.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a0413b)  // fmlalb z27.s, z9.h, z0.h[0]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a0415c)  // fmlalb z28.s, z10.h, z0.h[0]
    KAI_ASM_INST(0x64a04cd8)  // fmlalt z24.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a0417d)  // fmlalb z29.s, z11.h, z0.h[0]
    KAI_ASM_INST(0x64a04cf9)  // fmlalt z25.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a0419e)  // fmlalb z30.s, z12.h, z0.h[0]
    KAI_ASM_INST(0x64a04d1a)  // fmlalt z26.s, z8.h, z0.h[1]
    KAI_ASM_INST(0x64a04d3b)  // fmlalt z27.s, z9.h, z0.h[1]
    KAI_ASM_INST(0x64a04d5c)  // fmlalt z28.s, z10.h, z0.h[1]
    KAI_ASM_INST(0x64a04d7d)  // fmlalt z29.s, z11.h, z0.h[1]
    KAI_ASM_INST(0x64a04d9e)  // fmlalt z30.s, z12.h, z0.h[1]
    ble label_48
    ldnt1h { z13.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z14.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z15.h }, p2/Z, [x2]
    ldnt1h { z16.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a841b8)  // fmlalb z24.s, z13.h, z0.h[2]
    addvl x2, x2, #2
    ldnt1h { z17.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64a841d9)  // fmlalb z25.s, z14.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64a841fa)  // fmlalb z26.s, z15.h, z0.h[2]
    addvl x4, x4, #2
    ldnt1h { z19.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a8421b)  // fmlalb z27.s, z16.h, z0.h[2]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a8423c)  // fmlalb z28.s, z17.h, z0.h[2]
    KAI_ASM_INST(0x64a84db8)  // fmlalt z24.s, z13.h, z0.h[3]
    KAI_ASM_INST(0x64a8425d)  // fmlalb z29.s, z18.h, z0.h[2]
    KAI_ASM_INST(0x64a84dd9)  // fmlalt z25.s, z14.h, z0.h[3]
    KAI_ASM_INST(0x64a8427e)  // fmlalb z30.s, z19.h, z0.h[2]
    KAI_ASM_INST(0x64a84dfa)  // fmlalt z26.s, z15.h, z0.h[3]
    KAI_ASM_INST(0x64a84e1b)  // fmlalt z27.s, z16.h, z0.h[3]
    KAI_ASM_INST(0x64a84e3c)  // fmlalt z28.s, z17.h, z0.h[3]
    KAI_ASM_INST(0x64a84e5d)  // fmlalt z29.s, z18.h, z0.h[3]
    KAI_ASM_INST(0x64a84e7e)  // fmlalt z30.s, z19.h, z0.h[3]
    ble label_48
    ldnt1h { z20.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z21.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z22.h }, p2/Z, [x2]
    ldnt1h { z23.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b04298)  // fmlalb z24.s, z20.h, z0.h[4]
    addvl x2, x2, #2
    ldnt1h { z1.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64b042b9)  // fmlalb z25.s, z21.h, z0.h[4]
    ldnt1h { z2.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64b042da)  // fmlalb z26.s, z22.h, z0.h[4]
    addvl x4, x4, #2
    ldnt1h { z3.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b042fb)  // fmlalb z27.s, z23.h, z0.h[4]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b0403c)  // fmlalb z28.s, z1.h, z0.h[4]
    KAI_ASM_INST(0x64b04e98)  // fmlalt z24.s, z20.h, z0.h[5]
    KAI_ASM_INST(0x64b0405d)  // fmlalb z29.s, z2.h, z0.h[4]
    KAI_ASM_INST(0x64b04eb9)  // fmlalt z25.s, z21.h, z0.h[5]
    KAI_ASM_INST(0x64b0407e)  // fmlalb z30.s, z3.h, z0.h[4]
    KAI_ASM_INST(0x64b04eda)  // fmlalt z26.s, z22.h, z0.h[5]
    KAI_ASM_INST(0x64b04efb)  // fmlalt z27.s, z23.h, z0.h[5]
    KAI_ASM_INST(0x64b04c3c)  // fmlalt z28.s, z1.h, z0.h[5]
    KAI_ASM_INST(0x64b04c5d)  // fmlalt z29.s, z2.h, z0.h[5]
    KAI_ASM_INST(0x64b04c7e)  // fmlalt z30.s, z3.h, z0.h[5]
    ble label_48
    ldnt1h { z4.h }, p2/Z, [x22]
    ldnt1h { z5.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x2]
    ldnt1h { z7.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b84098)  // fmlalb z24.s, z4.h, z0.h[6]
    ldnt1h { z8.h }, p2/Z, [x4]
    KAI_ASM_INST(0x64b840b9)  // fmlalb z25.s, z5.h, z0.h[6]
    ldnt1h { z9.h }, p2/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x64b840da)  // fmlalb z26.s, z6.h, z0.h[6]
    ldnt1h { z10.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b840fb)  // fmlalb z27.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b8411c)  // fmlalb z28.s, z8.h, z0.h[6]
    KAI_ASM_INST(0x64b84c98)  // fmlalt z24.s, z4.h, z0.h[7]
    KAI_ASM_INST(0x64b8413d)  // fmlalb z29.s, z9.h, z0.h[6]
    KAI_ASM_INST(0x64b84cb9)  // fmlalt z25.s, z5.h, z0.h[7]
    KAI_ASM_INST(0x64b8415e)  // fmlalb z30.s, z10.h, z0.h[6]
    KAI_ASM_INST(0x64b84cda)  // fmlalt z26.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b84cfb)  // fmlalt z27.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b84d1c)  // fmlalt z28.s, z8.h, z0.h[7]
    KAI_ASM_INST(0x64b84d3d)  // fmlalt z29.s, z9.h, z0.h[7]
    KAI_ASM_INST(0x64b84d5e)  // fmlalt z30.s, z10.h, z0.h[7]
KAI_ASM_LABEL(label_48)  // Width 7: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    fcvt z26.h, p2/m, z27.s
    fcvt z27.h, p2/m, z29.s
    uzp1 z25.h, z25.h, z26.h
    fcvt z26.h, p2/m, z28.s
    uzp1 z26.h, z26.h, z27.h
    fcvt z27.h, p2/m, z30.s
    uzp1 z27.h, z27.h, z27.h
    tbz x25, #1, label_49
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
KAI_ASM_LABEL(label_49)  // Width 7: No activation
    st1h { z24.h }, p2, [x21]
    st1h { z25.h }, p2, [x21, #1, MUL VL]
    st1h { z26.h }, p2, [x21, #2, MUL VL]
    st1h { z27.h }, p1, [x21, #3, MUL VL]
    b label_57
KAI_ASM_LABEL(label_50)  // Width 8
    lsl x2, x27, #0x1
    mov x1, #0x3
    ldr x6, [x0, #0x20]
    add x5, x22, x23, LSL #1
    msub x4, x2, x1, x24
    mov x7, x26
    add x2, x22, x23
    add x1, x5, x23
    add x3, x22, x23, LSL #2
    whilelt p1.h, XZR, x4
    cbz x8, label_51
    fmov z20.h, #0
    ld1h { z19.h }, p2/Z, [x8]
    ld1h { z18.h }, p2/Z, [x8, #1, MUL VL]
    ld1h { z17.h }, p2/Z, [x8, #2, MUL VL]
    ld1h { z16.h }, p1/Z, [x8, #3, MUL VL]
    inch x8, ALL, MUL #8
    zip1 z24.h, z19.h, z20.h
    zip2 z25.h, z19.h, z20.h
    zip1 z26.h, z18.h, z20.h
    zip2 z27.h, z18.h, z20.h
    zip1 z28.h, z17.h, z20.h
    zip2 z29.h, z17.h, z20.h
    zip1 z30.h, z16.h, z20.h
    zip2 z31.h, z16.h, z20.h
    fcvt z24.s, p2/m, z24.h
    fcvt z25.s, p2/m, z25.h
    fcvt z26.s, p2/m, z26.h
    fcvt z27.s, p2/m, z27.h
    fcvt z28.s, p2/m, z28.h
    fcvt z29.s, p2/m, z29.h
    fcvt z30.s, p2/m, z30.h
    fcvt z31.s, p2/m, z31.h
    b label_52
KAI_ASM_LABEL(label_51)  // Width 8: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
    fmov z29.s, #0
    fmov z30.s, #0
    fmov z31.s, #0
KAI_ASM_LABEL(label_52)  // Width 8: setup done
    cmp x7, #0x8
    ble label_54
KAI_ASM_LABEL(label_53)  // Width 8: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ldnt1h { z1.h }, p2/Z, [x22]
    sub x7, x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x6]
    cmp x7, #0x8
    add x6, x6, #0x10
    ldnt1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z3.h }, p2/Z, [x2]
    ldnt1h { z4.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04038)  // fmlalb z24.s, z1.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z5.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64a04059)  // fmlalb z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64a0407a)  // fmlalb z26.s, z3.h, z0.h[0]
    addvl x5, x5, #2
    ldnt1h { z7.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a0409b)  // fmlalb z27.s, z4.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a040bc)  // fmlalb z28.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x64a04c38)  // fmlalt z24.s, z1.h, z0.h[1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a040dd)  // fmlalb z29.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a04c59)  // fmlalt z25.s, z2.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a040fe)  // fmlalb z30.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a04c7a)  // fmlalt z26.s, z3.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a0411f)  // fmlalb z31.s, z8.h, z0.h[0]
    KAI_ASM_INST(0x64a04c9b)  // fmlalt z27.s, z4.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a04cbc)  // fmlalt z28.s, z5.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a84138)  // fmlalb z24.s, z9.h, z0.h[2]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a04cdd)  // fmlalt z29.s, z6.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64a84159)  // fmlalb z25.s, z10.h, z0.h[2]
    KAI_ASM_INST(0x64a04cfe)  // fmlalt z30.s, z7.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64a8417a)  // fmlalb z26.s, z11.h, z0.h[2]
    addvl x5, x5, #2
    KAI_ASM_INST(0x64a04d1f)  // fmlalt z31.s, z8.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a8419b)  // fmlalb z27.s, z12.h, z0.h[2]
    ldnt1h { z16.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a841bc)  // fmlalb z28.s, z13.h, z0.h[2]
    KAI_ASM_INST(0x64a84d38)  // fmlalt z24.s, z9.h, z0.h[3]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a841dd)  // fmlalb z29.s, z14.h, z0.h[2]
    KAI_ASM_INST(0x64a84d59)  // fmlalt z25.s, z10.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64a841fe)  // fmlalb z30.s, z15.h, z0.h[2]
    KAI_ASM_INST(0x64a84d7a)  // fmlalt z26.s, z11.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64a8421f)  // fmlalb z31.s, z16.h, z0.h[2]
    KAI_ASM_INST(0x64a84d9b)  // fmlalt z27.s, z12.h, z0.h[3]
    ldnt1h { z19.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64a84dbc)  // fmlalt z28.s, z13.h, z0.h[3]
    ldnt1h { z20.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b04238)  // fmlalb z24.s, z17.h, z0.h[4]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64a84ddd)  // fmlalt z29.s, z14.h, z0.h[3]
    ldnt1h { z21.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64b04259)  // fmlalb z25.s, z18.h, z0.h[4]
    KAI_ASM_INST(0x64a84dfe)  // fmlalt z30.s, z15.h, z0.h[3]
    ldnt1h { z22.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64b0427a)  // fmlalb z26.s, z19.h, z0.h[4]
    addvl x5, x5, #2
    KAI_ASM_INST(0x64a84e1f)  // fmlalt z31.s, z16.h, z0.h[3]
    ldnt1h { z23.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b0429b)  // fmlalb z27.s, z20.h, z0.h[4]
    ldnt1h { z1.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b042bc)  // fmlalb z28.s, z21.h, z0.h[4]
    KAI_ASM_INST(0x64b04e38)  // fmlalt z24.s, z17.h, z0.h[5]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b042dd)  // fmlalb z29.s, z22.h, z0.h[4]
    KAI_ASM_INST(0x64b04e59)  // fmlalt z25.s, z18.h, z0.h[5]
    ldnt1h { z2.h }, p2/Z, [x22]
    KAI_ASM_INST(0x64b042fe)  // fmlalb z30.s, z23.h, z0.h[4]
    KAI_ASM_INST(0x64b04e7a)  // fmlalt z26.s, z19.h, z0.h[5]
    ldnt1h { z3.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0x64b0403f)  // fmlalb z31.s, z1.h, z0.h[4]
    KAI_ASM_INST(0x64b04e9b)  // fmlalt z27.s, z20.h, z0.h[5]
    ldnt1h { z4.h }, p2/Z, [x2]
    KAI_ASM_INST(0x64b04ebc)  // fmlalt z28.s, z21.h, z0.h[5]
    ldnt1h { z5.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b84058)  // fmlalb z24.s, z2.h, z0.h[6]
    addvl x2, x2, #2
    KAI_ASM_INST(0x64b04edd)  // fmlalt z29.s, z22.h, z0.h[5]
    ldnt1h { z6.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64b84079)  // fmlalb z25.s, z3.h, z0.h[6]
    KAI_ASM_INST(0x64b04efe)  // fmlalt z30.s, z23.h, z0.h[5]
    ldnt1h { z7.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64b8409a)  // fmlalb z26.s, z4.h, z0.h[6]
    addvl x5, x5, #2
    KAI_ASM_INST(0x64b04c3f)  // fmlalt z31.s, z1.h, z0.h[5]
    ldnt1h { z8.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b840bb)  // fmlalb z27.s, z5.h, z0.h[6]
    ldnt1h { z9.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b840dc)  // fmlalb z28.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b84c58)  // fmlalt z24.s, z2.h, z0.h[7]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b840fd)  // fmlalb z29.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b84c79)  // fmlalt z25.s, z3.h, z0.h[7]
    KAI_ASM_INST(0x64b8411e)  // fmlalb z30.s, z8.h, z0.h[6]
    KAI_ASM_INST(0x64b84c9a)  // fmlalt z26.s, z4.h, z0.h[7]
    KAI_ASM_INST(0x64b8413f)  // fmlalb z31.s, z9.h, z0.h[6]
    KAI_ASM_INST(0x64b84cbb)  // fmlalt z27.s, z5.h, z0.h[7]
    KAI_ASM_INST(0x64b84cdc)  // fmlalt z28.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b84cfd)  // fmlalt z29.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b84d1e)  // fmlalt z30.s, z8.h, z0.h[7]
    KAI_ASM_INST(0x64b84d3f)  // fmlalt z31.s, z9.h, z0.h[7]
    bgt label_53
KAI_ASM_LABEL(label_54)  // Width 8: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ldnt1h { z10.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ld1rqh { z0.h }, p0/Z, [x6]
    ldnt1h { z11.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z12.h }, p2/Z, [x2]
    ldnt1h { z13.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a04158)  // fmlalb z24.s, z10.h, z0.h[0]
    addvl x2, x2, #2
    ldnt1h { z14.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64a04179)  // fmlalb z25.s, z11.h, z0.h[0]
    ldnt1h { z15.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64a0419a)  // fmlalb z26.s, z12.h, z0.h[0]
    addvl x5, x5, #2
    ldnt1h { z16.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a041bb)  // fmlalb z27.s, z13.h, z0.h[0]
    ldnt1h { z17.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a041dc)  // fmlalb z28.s, z14.h, z0.h[0]
    KAI_ASM_INST(0x64a04d58)  // fmlalt z24.s, z10.h, z0.h[1]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a041fd)  // fmlalb z29.s, z15.h, z0.h[0]
    KAI_ASM_INST(0x64a04d79)  // fmlalt z25.s, z11.h, z0.h[1]
    KAI_ASM_INST(0x64a0421e)  // fmlalb z30.s, z16.h, z0.h[0]
    KAI_ASM_INST(0x64a04d9a)  // fmlalt z26.s, z12.h, z0.h[1]
    KAI_ASM_INST(0x64a0423f)  // fmlalb z31.s, z17.h, z0.h[0]
    KAI_ASM_INST(0x64a04dbb)  // fmlalt z27.s, z13.h, z0.h[1]
    KAI_ASM_INST(0x64a04ddc)  // fmlalt z28.s, z14.h, z0.h[1]
    KAI_ASM_INST(0x64a04dfd)  // fmlalt z29.s, z15.h, z0.h[1]
    KAI_ASM_INST(0x64a04e1e)  // fmlalt z30.s, z16.h, z0.h[1]
    KAI_ASM_INST(0x64a04e3f)  // fmlalt z31.s, z17.h, z0.h[1]
    ble label_55
    ldnt1h { z18.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z19.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z20.h }, p2/Z, [x2]
    ldnt1h { z21.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64a84258)  // fmlalb z24.s, z18.h, z0.h[2]
    addvl x2, x2, #2
    ldnt1h { z22.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64a84279)  // fmlalb z25.s, z19.h, z0.h[2]
    ldnt1h { z23.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64a8429a)  // fmlalb z26.s, z20.h, z0.h[2]
    addvl x5, x5, #2
    ldnt1h { z1.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64a842bb)  // fmlalb z27.s, z21.h, z0.h[2]
    ldnt1h { z2.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64a842dc)  // fmlalb z28.s, z22.h, z0.h[2]
    KAI_ASM_INST(0x64a84e58)  // fmlalt z24.s, z18.h, z0.h[3]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64a842fd)  // fmlalb z29.s, z23.h, z0.h[2]
    KAI_ASM_INST(0x64a84e79)  // fmlalt z25.s, z19.h, z0.h[3]
    KAI_ASM_INST(0x64a8403e)  // fmlalb z30.s, z1.h, z0.h[2]
    KAI_ASM_INST(0x64a84e9a)  // fmlalt z26.s, z20.h, z0.h[3]
    KAI_ASM_INST(0x64a8405f)  // fmlalb z31.s, z2.h, z0.h[2]
    KAI_ASM_INST(0x64a84ebb)  // fmlalt z27.s, z21.h, z0.h[3]
    KAI_ASM_INST(0x64a84edc)  // fmlalt z28.s, z22.h, z0.h[3]
    KAI_ASM_INST(0x64a84efd)  // fmlalt z29.s, z23.h, z0.h[3]
    KAI_ASM_INST(0x64a84c3e)  // fmlalt z30.s, z1.h, z0.h[3]
    KAI_ASM_INST(0x64a84c5f)  // fmlalt z31.s, z2.h, z0.h[3]
    ble label_55
    ldnt1h { z3.h }, p2/Z, [x22]
    subs x7, x7, #0x2
    ldnt1h { z4.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h { z5.h }, p2/Z, [x2]
    ldnt1h { z6.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b04078)  // fmlalb z24.s, z3.h, z0.h[4]
    addvl x2, x2, #2
    ldnt1h { z7.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64b04099)  // fmlalb z25.s, z4.h, z0.h[4]
    ldnt1h { z8.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64b040ba)  // fmlalb z26.s, z5.h, z0.h[4]
    addvl x5, x5, #2
    ldnt1h { z9.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b040db)  // fmlalb z27.s, z6.h, z0.h[4]
    ldnt1h { z10.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b040fc)  // fmlalb z28.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b04c78)  // fmlalt z24.s, z3.h, z0.h[5]
    addvl x1, x1, #2
    KAI_ASM_INST(0x64b0411d)  // fmlalb z29.s, z8.h, z0.h[4]
    KAI_ASM_INST(0x64b04c99)  // fmlalt z25.s, z4.h, z0.h[5]
    KAI_ASM_INST(0x64b0413e)  // fmlalb z30.s, z9.h, z0.h[4]
    KAI_ASM_INST(0x64b04cba)  // fmlalt z26.s, z5.h, z0.h[5]
    KAI_ASM_INST(0x64b0415f)  // fmlalb z31.s, z10.h, z0.h[4]
    KAI_ASM_INST(0x64b04cdb)  // fmlalt z27.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b04cfc)  // fmlalt z28.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b04d1d)  // fmlalt z29.s, z8.h, z0.h[5]
    KAI_ASM_INST(0x64b04d3e)  // fmlalt z30.s, z9.h, z0.h[5]
    KAI_ASM_INST(0x64b04d5f)  // fmlalt z31.s, z10.h, z0.h[5]
    ble label_55
    ldnt1h { z11.h }, p2/Z, [x22]
    ldnt1h { z12.h }, p2/Z, [x22, #1, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x2]
    ldnt1h { z14.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x64b84178)  // fmlalb z24.s, z11.h, z0.h[6]
    ldnt1h { z15.h }, p2/Z, [x5]
    KAI_ASM_INST(0x64b84199)  // fmlalb z25.s, z12.h, z0.h[6]
    ldnt1h { z16.h }, p2/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x64b841ba)  // fmlalb z26.s, z13.h, z0.h[6]
    ldnt1h { z17.h }, p2/Z, [x1]
    KAI_ASM_INST(0x64b841db)  // fmlalb z27.s, z14.h, z0.h[6]
    ldnt1h { z18.h }, p2/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x64b841fc)  // fmlalb z28.s, z15.h, z0.h[6]
    KAI_ASM_INST(0x64b84d78)  // fmlalt z24.s, z11.h, z0.h[7]
    KAI_ASM_INST(0x64b8421d)  // fmlalb z29.s, z16.h, z0.h[6]
    KAI_ASM_INST(0x64b84d99)  // fmlalt z25.s, z12.h, z0.h[7]
    KAI_ASM_INST(0x64b8423e)  // fmlalb z30.s, z17.h, z0.h[6]
    KAI_ASM_INST(0x64b84dba)  // fmlalt z26.s, z13.h, z0.h[7]
    KAI_ASM_INST(0x64b8425f)  // fmlalb z31.s, z18.h, z0.h[6]
    KAI_ASM_INST(0x64b84ddb)  // fmlalt z27.s, z14.h, z0.h[7]
    KAI_ASM_INST(0x64b84dfc)  // fmlalt z28.s, z15.h, z0.h[7]
    KAI_ASM_INST(0x64b84e1d)  // fmlalt z29.s, z16.h, z0.h[7]
    KAI_ASM_INST(0x64b84e3e)  // fmlalt z30.s, z17.h, z0.h[7]
    KAI_ASM_INST(0x64b84e5f)  // fmlalt z31.s, z18.h, z0.h[7]
KAI_ASM_LABEL(label_55)  // Width 8: Multiply loop: multiply skip
    fcvt z24.h, p2/m, z24.s
    fcvt z25.h, p2/m, z25.s
    uzp1 z24.h, z24.h, z25.h
    fcvt z25.h, p2/m, z26.s
    fcvt z26.h, p2/m, z27.s
    fcvt z27.h, p2/m, z29.s
    uzp1 z25.h, z25.h, z26.h
    fcvt z26.h, p2/m, z28.s
    fcvt z28.h, p2/m, z31.s
    uzp1 z26.h, z26.h, z27.h
    fcvt z27.h, p2/m, z30.s
    uzp1 z27.h, z27.h, z28.h
    tbz x25, #1, label_56
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
KAI_ASM_LABEL(label_56)  // Width 8: No activation
    subs x20, x20, #0x8
    st1h { z24.h }, p2, [x21]
    mov x22, x3
    st1h { z25.h }, p2, [x21, #1, MUL VL]
    sub x24, x24, x27, LSL #3
    st1h { z26.h }, p2, [x21, #2, MUL VL]
    st1h { z27.h }, p1, [x21, #3, MUL VL]
    addvl x21, x21, #4
    bgt label_1
KAI_ASM_LABEL(label_57)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL)

    KAI_ASM_END
