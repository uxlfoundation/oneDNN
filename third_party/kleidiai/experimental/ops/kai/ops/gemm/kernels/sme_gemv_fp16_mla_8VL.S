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

    KAI_ASM_CODE(sme_gemv_fp16_mla_8VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_gemv_fp16_mla_8VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_gemv_fp16_mla_8VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_gemv_fp16_mla_8VL)
    stp x20, x21, [sp, -96]!
    str x22, [sp, 16]
    stp d8, d9, [sp, 24]
    stp d10, d11, [sp, 40]
    stp d12, d13, [sp, 56]
    stp d14, d15, [sp, 72]
    mov x4, #0xfc00
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldrh w1, [x0, #0xa0]
    mov x3, #0x7c00
    cnth x22
    ldrh w2, [x0, #0xa4]
    mov x21, #0
    ptrue p2.b
    ldr x20, [x0, #0x8]
    cmp x1, x4
    ldr x8, [x0, #0x10]
    cset x1, NE
    cmp x2, x3
    ldr x7, [x0, #0x30]
    add x6, x20, x22
    orr x21, x21, x1, LSL #1
    ldr x5, [x0, #0x50]
    cset x1, NE
    sub x6, x6, #0x1
    ldr x4, [x0, #0x40]
    orr x21, x21, x1, LSL #1
    udiv x6, x6, x22
KAI_ASM_LABEL(label_1)  // Column loop
    cmp x6, #0x8
    bge label_50
    cmp x6, #0x6
    bgt label_43
    beq label_36
    cmp x6, #0x4
    bgt label_29
    beq label_22
    cmp x6, #0x2
    bgt label_15
    beq label_8
    mov x3, x8
    ldr x2, [x0, #0x20]
    whilelt p1.h, XZR, x20
    cbz x4, label_2
    ld1h { z24.h }, p1/Z, [x4]
    b label_3
KAI_ASM_LABEL(label_2)  // Width 1: no bias
    fmov z24.h, #0
KAI_ASM_LABEL(label_3)  // Width 1: setup done
    cmp x3, #0x8
    ble label_5
KAI_ASM_LABEL(label_4)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    addvl x7, x7, #8
    ld1rqh { z0.h }, p0/Z, [x2]
    sub x3, x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7]
    addvl x7, x7, #8
    cmp x3, #0x8
    ldnt1h { z3.h }, p2/Z, [x7]
    addvl x7, x7, #8
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7]
    addvl x7, x7, #8
    ldnt1h { z5.h }, p2/Z, [x7]
    addvl x7, x7, #8
    ldnt1h { z6.h }, p2/Z, [x7]
    addvl x7, x7, #8
    ldnt1h { z7.h }, p2/Z, [x7]
    addvl x7, x7, #8
    fmla z24.h, z2.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7]
    addvl x7, x7, #8
    fmla z24.h, z3.h, z0.h[2]
    fmla z24.h, z4.h, z0.h[3]
    fmla z24.h, z5.h, z0.h[4]
    fmla z24.h, z6.h, z0.h[5]
    fmla z24.h, z7.h, z0.h[6]
    fmla z24.h, z8.h, z0.h[7]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z9.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    addvl x7, x7, #8
    fmla z24.h, z9.h, z0.h[0]
    ble label_6
    ldnt1h { z10.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z10.h, z0.h[1]
    ble label_6
    ldnt1h { z11.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z11.h, z0.h[2]
    ble label_6
    ldnt1h { z12.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z12.h, z0.h[3]
    ble label_6
    ldnt1h { z13.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z13.h, z0.h[4]
    ble label_6
    ldnt1h { z14.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z14.h, z0.h[5]
    ble label_6
    ldnt1h { z15.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #8
    fmla z24.h, z15.h, z0.h[6]
    ble label_6
    ldnt1h { z16.h }, p2/Z, [x7]
    fmla z24.h, z16.h, z0.h[7]
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_7
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
KAI_ASM_LABEL(label_7)  // Width 1: No activation
    st1h { z24.h }, p1, [x5]
    b label_57
KAI_ASM_LABEL(label_8)  // Width 2
    sub x1, x20, x22
    mov x3, x8
    ldr x2, [x0, #0x20]
    whilelt p1.h, XZR, x1
    cbz x4, label_9
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p1/Z, [x4, #1, MUL VL]
    b label_10
KAI_ASM_LABEL(label_9)  // Width 2: no bias
    fmov z24.h, #0
    fmov z25.h, #0
KAI_ASM_LABEL(label_10)  // Width 2: setup done
    cmp x3, #0x8
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z3.h }, p2/Z, [x7]
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7]
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z7.h }, p2/Z, [x7]
    fmla z24.h, z3.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z4.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z11.h }, p2/Z, [x7]
    fmla z24.h, z5.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z6.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x7]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z15.h }, p2/Z, [x7]
    fmla z24.h, z7.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z8.h, z0.h[3]
    fmla z24.h, z9.h, z0.h[4]
    fmla z25.h, z10.h, z0.h[4]
    fmla z24.h, z11.h, z0.h[5]
    fmla z25.h, z12.h, z0.h[5]
    fmla z24.h, z13.h, z0.h[6]
    fmla z25.h, z14.h, z0.h[6]
    fmla z24.h, z15.h, z0.h[7]
    fmla z25.h, z16.h, z0.h[7]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z17.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z17.h, z0.h[0]
    fmla z25.h, z18.h, z0.h[0]
    ble label_13
    ldnt1h { z19.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z19.h, z0.h[1]
    fmla z25.h, z20.h, z0.h[1]
    ble label_13
    ldnt1h { z21.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z21.h, z0.h[2]
    fmla z25.h, z22.h, z0.h[2]
    ble label_13
    ldnt1h { z23.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z1.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z23.h, z0.h[3]
    fmla z25.h, z1.h, z0.h[3]
    ble label_13
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z2.h, z0.h[4]
    fmla z25.h, z3.h, z0.h[4]
    ble label_13
    ldnt1h { z4.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z4.h, z0.h[5]
    fmla z25.h, z5.h, z0.h[5]
    ble label_13
    ldnt1h { z6.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z6.h, z0.h[6]
    fmla z25.h, z7.h, z0.h[6]
    ble label_13
    ldnt1h { z8.h }, p2/Z, [x7]
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    fmla z24.h, z8.h, z0.h[7]
    fmla z25.h, z9.h, z0.h[7]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_14
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p1, [x5, #1, MUL VL]
    b label_57
KAI_ASM_LABEL(label_15)  // Width 3
    mov x1, #0x2
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_16
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p1/Z, [x4, #2, MUL VL]
    b label_17
KAI_ASM_LABEL(label_16)  // Width 3: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
KAI_ASM_LABEL(label_17)  // Width 3: setup done
    cmp x3, #0x8
    ble label_19
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7]
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    fmla z26.h, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z7.h }, p2/Z, [x7]
    fmla z24.h, z4.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    fmla z25.h, z5.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z26.h, z6.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7]
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    fmla z24.h, z7.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z8.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x7]
    fmla z26.h, z9.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z10.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x7]
    fmla z25.h, z11.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    fmla z26.h, z12.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z19.h }, p2/Z, [x7]
    fmla z24.h, z13.h, z0.h[4]
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    fmla z25.h, z14.h, z0.h[4]
    ldnt1h { z21.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z26.h, z15.h, z0.h[4]
    ldnt1h { z22.h }, p2/Z, [x7]
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    fmla z24.h, z16.h, z0.h[5]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z25.h, z17.h, z0.h[5]
    fmla z26.h, z18.h, z0.h[5]
    fmla z24.h, z19.h, z0.h[6]
    fmla z25.h, z20.h, z0.h[6]
    fmla z26.h, z21.h, z0.h[6]
    fmla z24.h, z22.h, z0.h[7]
    fmla z25.h, z23.h, z0.h[7]
    fmla z26.h, z1.h, z0.h[7]
    bgt label_18
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z2.h, z0.h[0]
    fmla z25.h, z3.h, z0.h[0]
    fmla z26.h, z4.h, z0.h[0]
    ble label_20
    ldnt1h { z5.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z5.h, z0.h[1]
    fmla z25.h, z6.h, z0.h[1]
    fmla z26.h, z7.h, z0.h[1]
    ble label_20
    ldnt1h { z8.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z8.h, z0.h[2]
    fmla z25.h, z9.h, z0.h[2]
    fmla z26.h, z10.h, z0.h[2]
    ble label_20
    ldnt1h { z11.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z11.h, z0.h[3]
    fmla z25.h, z12.h, z0.h[3]
    fmla z26.h, z13.h, z0.h[3]
    ble label_20
    ldnt1h { z14.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z15.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z14.h, z0.h[4]
    fmla z25.h, z15.h, z0.h[4]
    fmla z26.h, z16.h, z0.h[4]
    ble label_20
    ldnt1h { z17.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z17.h, z0.h[5]
    fmla z25.h, z18.h, z0.h[5]
    fmla z26.h, z19.h, z0.h[5]
    ble label_20
    ldnt1h { z20.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    fmla z24.h, z20.h, z0.h[6]
    fmla z25.h, z21.h, z0.h[6]
    fmla z26.h, z22.h, z0.h[6]
    ble label_20
    ldnt1h { z23.h }, p2/Z, [x7]
    ldnt1h { z1.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #2, MUL VL]
    fmla z24.h, z23.h, z0.h[7]
    fmla z25.h, z1.h, z0.h[7]
    fmla z26.h, z2.h, z0.h[7]
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_21
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
KAI_ASM_LABEL(label_21)  // Width 3: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p1, [x5, #2, MUL VL]
    b label_57
KAI_ASM_LABEL(label_22)  // Width 4
    mov x1, #0x3
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_23
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x4, #2, MUL VL]
    ld1h { z27.h }, p1/Z, [x4, #3, MUL VL]
    b label_24
KAI_ASM_LABEL(label_23)  // Width 4: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
    fmov z27.h, #0
KAI_ASM_LABEL(label_24)  // Width 4: setup done
    cmp x3, #0x8
    ble label_26
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z1.h, z0.h[0]
    addvl x7, x7, #8
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7]
    fmla z26.h, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z5.h, z0.h[1]
    addvl x7, x7, #8
    fmla z25.h, z6.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7]
    fmla z26.h, z7.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z8.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z9.h, z0.h[2]
    addvl x7, x7, #8
    fmla z25.h, z10.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x7]
    fmla z26.h, z11.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z12.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z13.h, z0.h[3]
    addvl x7, x7, #8
    fmla z25.h, z14.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x7]
    fmla z26.h, z15.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z16.h, z0.h[3]
    ldnt1h { z19.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z17.h, z0.h[4]
    addvl x7, x7, #8
    fmla z25.h, z18.h, z0.h[4]
    ldnt1h { z21.h }, p2/Z, [x7]
    fmla z26.h, z19.h, z0.h[4]
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z20.h, z0.h[4]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z21.h, z0.h[5]
    addvl x7, x7, #8
    fmla z25.h, z22.h, z0.h[5]
    ldnt1h { z2.h }, p2/Z, [x7]
    fmla z26.h, z23.h, z0.h[5]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z1.h, z0.h[5]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z2.h, z0.h[6]
    addvl x7, x7, #8
    fmla z25.h, z3.h, z0.h[6]
    ldnt1h { z6.h }, p2/Z, [x7]
    fmla z26.h, z4.h, z0.h[6]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    fmla z27.h, z5.h, z0.h[6]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z6.h, z0.h[7]
    addvl x7, x7, #8
    fmla z25.h, z7.h, z0.h[7]
    fmla z26.h, z8.h, z0.h[7]
    fmla z27.h, z9.h, z0.h[7]
    bgt label_25
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z10.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z10.h, z0.h[0]
    addvl x7, x7, #8
    fmla z25.h, z11.h, z0.h[0]
    fmla z26.h, z12.h, z0.h[0]
    fmla z27.h, z13.h, z0.h[0]
    ble label_27
    ldnt1h { z14.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z15.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z14.h, z0.h[1]
    addvl x7, x7, #8
    fmla z25.h, z15.h, z0.h[1]
    fmla z26.h, z16.h, z0.h[1]
    fmla z27.h, z17.h, z0.h[1]
    ble label_27
    ldnt1h { z18.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z19.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z18.h, z0.h[2]
    addvl x7, x7, #8
    fmla z25.h, z19.h, z0.h[2]
    fmla z26.h, z20.h, z0.h[2]
    fmla z27.h, z21.h, z0.h[2]
    ble label_27
    ldnt1h { z22.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z22.h, z0.h[3]
    addvl x7, x7, #8
    fmla z25.h, z23.h, z0.h[3]
    fmla z26.h, z1.h, z0.h[3]
    fmla z27.h, z2.h, z0.h[3]
    ble label_27
    ldnt1h { z3.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z3.h, z0.h[4]
    addvl x7, x7, #8
    fmla z25.h, z4.h, z0.h[4]
    fmla z26.h, z5.h, z0.h[4]
    fmla z27.h, z6.h, z0.h[4]
    ble label_27
    ldnt1h { z7.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z7.h, z0.h[5]
    addvl x7, x7, #8
    fmla z25.h, z8.h, z0.h[5]
    fmla z26.h, z9.h, z0.h[5]
    fmla z27.h, z10.h, z0.h[5]
    ble label_27
    ldnt1h { z11.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z11.h, z0.h[6]
    addvl x7, x7, #8
    fmla z25.h, z12.h, z0.h[6]
    fmla z26.h, z13.h, z0.h[6]
    fmla z27.h, z14.h, z0.h[6]
    ble label_27
    ldnt1h { z15.h }, p2/Z, [x7]
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z15.h, z0.h[7]
    fmla z25.h, z16.h, z0.h[7]
    fmla z26.h, z17.h, z0.h[7]
    fmla z27.h, z18.h, z0.h[7]
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: multiply skip
    tbz x21, #1, label_28
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
KAI_ASM_LABEL(label_28)  // Width 4: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p2, [x5, #2, MUL VL]
    st1h { z27.h }, p1, [x5, #3, MUL VL]
    b label_57
KAI_ASM_LABEL(label_29)  // Width 5
    mov x1, #0x4
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_30
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x4, #2, MUL VL]
    ld1h { z27.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z28.h }, p1/Z, [x4, #4, MUL VL]
    b label_31
KAI_ASM_LABEL(label_30)  // Width 5: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
    fmov z27.h, #0
    fmov z28.h, #0
KAI_ASM_LABEL(label_31)  // Width 5: setup done
    cmp x3, #0x8
    ble label_33
KAI_ASM_LABEL(label_32)  // Width 5: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z2.h, z0.h[0]
    addvl x7, x7, #8
    fmla z26.h, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7]
    fmla z27.h, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z6.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z7.h, z0.h[1]
    addvl x7, x7, #8
    fmla z26.h, z8.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7]
    fmla z27.h, z9.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z10.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z11.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z12.h, z0.h[2]
    addvl x7, x7, #8
    fmla z26.h, z13.h, z0.h[2]
    ldnt1h { z16.h }, p2/Z, [x7]
    fmla z27.h, z14.h, z0.h[2]
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z15.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z16.h, z0.h[3]
    ldnt1h { z20.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z17.h, z0.h[3]
    addvl x7, x7, #8
    fmla z26.h, z18.h, z0.h[3]
    ldnt1h { z21.h }, p2/Z, [x7]
    fmla z27.h, z19.h, z0.h[3]
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z20.h, z0.h[3]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z21.h, z0.h[4]
    ldnt1h { z2.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z22.h, z0.h[4]
    addvl x7, x7, #8
    fmla z26.h, z23.h, z0.h[4]
    ldnt1h { z3.h }, p2/Z, [x7]
    fmla z27.h, z1.h, z0.h[4]
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z2.h, z0.h[4]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z3.h, z0.h[5]
    ldnt1h { z7.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z4.h, z0.h[5]
    addvl x7, x7, #8
    fmla z26.h, z5.h, z0.h[5]
    ldnt1h { z8.h }, p2/Z, [x7]
    fmla z27.h, z6.h, z0.h[5]
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z7.h, z0.h[5]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z8.h, z0.h[6]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z9.h, z0.h[6]
    addvl x7, x7, #8
    fmla z26.h, z10.h, z0.h[6]
    ldnt1h { z13.h }, p2/Z, [x7]
    fmla z27.h, z11.h, z0.h[6]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    fmla z28.h, z12.h, z0.h[6]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z13.h, z0.h[7]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z14.h, z0.h[7]
    addvl x7, x7, #8
    fmla z26.h, z15.h, z0.h[7]
    fmla z27.h, z16.h, z0.h[7]
    fmla z28.h, z17.h, z0.h[7]
    bgt label_32
KAI_ASM_LABEL(label_33)  // Width 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z18.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z19.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z18.h, z0.h[0]
    ldnt1h { z22.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z19.h, z0.h[0]
    addvl x7, x7, #8
    fmla z26.h, z20.h, z0.h[0]
    fmla z27.h, z21.h, z0.h[0]
    fmla z28.h, z22.h, z0.h[0]
    ble label_34
    ldnt1h { z23.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z1.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z23.h, z0.h[1]
    ldnt1h { z4.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z1.h, z0.h[1]
    addvl x7, x7, #8
    fmla z26.h, z2.h, z0.h[1]
    fmla z27.h, z3.h, z0.h[1]
    fmla z28.h, z4.h, z0.h[1]
    ble label_34
    ldnt1h { z5.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z5.h, z0.h[2]
    ldnt1h { z9.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z6.h, z0.h[2]
    addvl x7, x7, #8
    fmla z26.h, z7.h, z0.h[2]
    fmla z27.h, z8.h, z0.h[2]
    fmla z28.h, z9.h, z0.h[2]
    ble label_34
    ldnt1h { z10.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z10.h, z0.h[3]
    ldnt1h { z14.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z11.h, z0.h[3]
    addvl x7, x7, #8
    fmla z26.h, z12.h, z0.h[3]
    fmla z27.h, z13.h, z0.h[3]
    fmla z28.h, z14.h, z0.h[3]
    ble label_34
    ldnt1h { z15.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z15.h, z0.h[4]
    ldnt1h { z19.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z16.h, z0.h[4]
    addvl x7, x7, #8
    fmla z26.h, z17.h, z0.h[4]
    fmla z27.h, z18.h, z0.h[4]
    fmla z28.h, z19.h, z0.h[4]
    ble label_34
    ldnt1h { z20.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z20.h, z0.h[5]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z21.h, z0.h[5]
    addvl x7, x7, #8
    fmla z26.h, z22.h, z0.h[5]
    fmla z27.h, z23.h, z0.h[5]
    fmla z28.h, z1.h, z0.h[5]
    ble label_34
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z2.h, z0.h[6]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z3.h, z0.h[6]
    addvl x7, x7, #8
    fmla z26.h, z4.h, z0.h[6]
    fmla z27.h, z5.h, z0.h[6]
    fmla z28.h, z6.h, z0.h[6]
    ble label_34
    ldnt1h { z7.h }, p2/Z, [x7]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z7.h, z0.h[7]
    ldnt1h { z11.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z8.h, z0.h[7]
    fmla z26.h, z9.h, z0.h[7]
    fmla z27.h, z10.h, z0.h[7]
    fmla z28.h, z11.h, z0.h[7]
KAI_ASM_LABEL(label_34)  // Width 5: Multiply loop: multiply skip
    tbz x21, #1, label_35
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmin z28.h, p2/M, z28.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
    fmax z28.h, p2/M, z28.h, z16.h
KAI_ASM_LABEL(label_35)  // Width 5: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p2, [x5, #2, MUL VL]
    st1h { z27.h }, p2, [x5, #3, MUL VL]
    st1h { z28.h }, p1, [x5, #4, MUL VL]
    b label_57
KAI_ASM_LABEL(label_36)  // Width 6
    mov x1, #0x5
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_37
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x4, #2, MUL VL]
    ld1h { z27.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z28.h }, p2/Z, [x4, #4, MUL VL]
    ld1h { z29.h }, p1/Z, [x4, #5, MUL VL]
    b label_38
KAI_ASM_LABEL(label_37)  // Width 6: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
    fmov z27.h, #0
    fmov z28.h, #0
    fmov z29.h, #0
KAI_ASM_LABEL(label_38)  // Width 6: setup done
    cmp x3, #0x8
    ble label_40
KAI_ASM_LABEL(label_39)  // Width 6: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z3.h, z0.h[0]
    addvl x7, x7, #8
    fmla z27.h, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7]
    fmla z28.h, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z7.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z8.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z9.h, z0.h[1]
    addvl x7, x7, #8
    fmla z27.h, z10.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7]
    fmla z28.h, z11.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z12.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z13.h, z0.h[2]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z14.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z15.h, z0.h[2]
    addvl x7, x7, #8
    fmla z27.h, z16.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7]
    fmla z28.h, z17.h, z0.h[2]
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z18.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z19.h, z0.h[3]
    ldnt1h { z23.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z20.h, z0.h[3]
    ldnt1h { z1.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z21.h, z0.h[3]
    addvl x7, x7, #8
    fmla z27.h, z22.h, z0.h[3]
    ldnt1h { z2.h }, p2/Z, [x7]
    fmla z28.h, z23.h, z0.h[3]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z1.h, z0.h[3]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z2.h, z0.h[4]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z3.h, z0.h[4]
    ldnt1h { z7.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z4.h, z0.h[4]
    addvl x7, x7, #8
    fmla z27.h, z5.h, z0.h[4]
    ldnt1h { z8.h }, p2/Z, [x7]
    fmla z28.h, z6.h, z0.h[4]
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z7.h, z0.h[4]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z8.h, z0.h[5]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z9.h, z0.h[5]
    ldnt1h { z13.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z10.h, z0.h[5]
    addvl x7, x7, #8
    fmla z27.h, z11.h, z0.h[5]
    ldnt1h { z14.h }, p2/Z, [x7]
    fmla z28.h, z12.h, z0.h[5]
    ldnt1h { z15.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z13.h, z0.h[5]
    ldnt1h { z16.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z14.h, z0.h[6]
    ldnt1h { z18.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z15.h, z0.h[6]
    ldnt1h { z19.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z16.h, z0.h[6]
    addvl x7, x7, #8
    fmla z27.h, z17.h, z0.h[6]
    ldnt1h { z20.h }, p2/Z, [x7]
    fmla z28.h, z18.h, z0.h[6]
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    fmla z29.h, z19.h, z0.h[6]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z20.h, z0.h[7]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z21.h, z0.h[7]
    ldnt1h { z2.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z22.h, z0.h[7]
    addvl x7, x7, #8
    fmla z27.h, z23.h, z0.h[7]
    fmla z28.h, z1.h, z0.h[7]
    fmla z29.h, z2.h, z0.h[7]
    bgt label_39
KAI_ASM_LABEL(label_40)  // Width 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z3.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z4.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z5.h, z0.h[0]
    addvl x7, x7, #8
    fmla z27.h, z6.h, z0.h[0]
    fmla z28.h, z7.h, z0.h[0]
    fmla z29.h, z8.h, z0.h[0]
    ble label_41
    ldnt1h { z9.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z10.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z11.h, z0.h[1]
    addvl x7, x7, #8
    fmla z27.h, z12.h, z0.h[1]
    fmla z28.h, z13.h, z0.h[1]
    fmla z29.h, z14.h, z0.h[1]
    ble label_41
    ldnt1h { z15.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z15.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z16.h, z0.h[2]
    ldnt1h { z20.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z17.h, z0.h[2]
    addvl x7, x7, #8
    fmla z27.h, z18.h, z0.h[2]
    fmla z28.h, z19.h, z0.h[2]
    fmla z29.h, z20.h, z0.h[2]
    ble label_41
    ldnt1h { z21.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z21.h, z0.h[3]
    ldnt1h { z2.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z22.h, z0.h[3]
    ldnt1h { z3.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z23.h, z0.h[3]
    addvl x7, x7, #8
    fmla z27.h, z1.h, z0.h[3]
    fmla z28.h, z2.h, z0.h[3]
    fmla z29.h, z3.h, z0.h[3]
    ble label_41
    ldnt1h { z4.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z4.h, z0.h[4]
    ldnt1h { z8.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z5.h, z0.h[4]
    ldnt1h { z9.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z6.h, z0.h[4]
    addvl x7, x7, #8
    fmla z27.h, z7.h, z0.h[4]
    fmla z28.h, z8.h, z0.h[4]
    fmla z29.h, z9.h, z0.h[4]
    ble label_41
    ldnt1h { z10.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z10.h, z0.h[5]
    ldnt1h { z14.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z11.h, z0.h[5]
    ldnt1h { z15.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z12.h, z0.h[5]
    addvl x7, x7, #8
    fmla z27.h, z13.h, z0.h[5]
    fmla z28.h, z14.h, z0.h[5]
    fmla z29.h, z15.h, z0.h[5]
    ble label_41
    ldnt1h { z16.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z16.h, z0.h[6]
    ldnt1h { z20.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z17.h, z0.h[6]
    ldnt1h { z21.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z18.h, z0.h[6]
    addvl x7, x7, #8
    fmla z27.h, z19.h, z0.h[6]
    fmla z28.h, z20.h, z0.h[6]
    fmla z29.h, z21.h, z0.h[6]
    ble label_41
    ldnt1h { z22.h }, p2/Z, [x7]
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z22.h, z0.h[7]
    ldnt1h { z3.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z23.h, z0.h[7]
    ldnt1h { z4.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z1.h, z0.h[7]
    fmla z27.h, z2.h, z0.h[7]
    fmla z28.h, z3.h, z0.h[7]
    fmla z29.h, z4.h, z0.h[7]
KAI_ASM_LABEL(label_41)  // Width 6: Multiply loop: multiply skip
    tbz x21, #1, label_42
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmin z28.h, p2/M, z28.h, z17.h
    fmin z29.h, p2/M, z29.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
    fmax z28.h, p2/M, z28.h, z16.h
    fmax z29.h, p2/M, z29.h, z16.h
KAI_ASM_LABEL(label_42)  // Width 6: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p2, [x5, #2, MUL VL]
    st1h { z27.h }, p2, [x5, #3, MUL VL]
    st1h { z28.h }, p2, [x5, #4, MUL VL]
    st1h { z29.h }, p1, [x5, #5, MUL VL]
    b label_57
KAI_ASM_LABEL(label_43)  // Width 7
    mov x1, #0x6
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_44
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x4, #2, MUL VL]
    ld1h { z27.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z28.h }, p2/Z, [x4, #4, MUL VL]
    ld1h { z29.h }, p2/Z, [x4, #5, MUL VL]
    ld1h { z30.h }, p1/Z, [x4, #6, MUL VL]
    b label_45
KAI_ASM_LABEL(label_44)  // Width 7: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
    fmov z27.h, #0
    fmov z28.h, #0
    fmov z29.h, #0
    fmov z30.h, #0
KAI_ASM_LABEL(label_45)  // Width 7: setup done
    cmp x3, #0x8
    ble label_47
KAI_ASM_LABEL(label_46)  // Width 7: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z4.h, z0.h[0]
    addvl x7, x7, #8
    fmla z28.h, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7]
    fmla z29.h, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z7.h, z0.h[0]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z8.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z10.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z11.h, z0.h[1]
    addvl x7, x7, #8
    fmla z28.h, z12.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7]
    fmla z29.h, z13.h, z0.h[1]
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z14.h, z0.h[1]
    ldnt1h { z17.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z15.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z16.h, z0.h[2]
    ldnt1h { z20.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z17.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z18.h, z0.h[2]
    addvl x7, x7, #8
    fmla z28.h, z19.h, z0.h[2]
    ldnt1h { z22.h }, p2/Z, [x7]
    fmla z29.h, z20.h, z0.h[2]
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z21.h, z0.h[2]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z22.h, z0.h[3]
    ldnt1h { z3.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z23.h, z0.h[3]
    ldnt1h { z4.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z1.h, z0.h[3]
    ldnt1h { z5.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z2.h, z0.h[3]
    addvl x7, x7, #8
    fmla z28.h, z3.h, z0.h[3]
    ldnt1h { z6.h }, p2/Z, [x7]
    fmla z29.h, z4.h, z0.h[3]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z5.h, z0.h[3]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z6.h, z0.h[4]
    ldnt1h { z10.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z7.h, z0.h[4]
    ldnt1h { z11.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z8.h, z0.h[4]
    ldnt1h { z12.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z9.h, z0.h[4]
    addvl x7, x7, #8
    fmla z28.h, z10.h, z0.h[4]
    ldnt1h { z13.h }, p2/Z, [x7]
    fmla z29.h, z11.h, z0.h[4]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z12.h, z0.h[4]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z13.h, z0.h[5]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z14.h, z0.h[5]
    ldnt1h { z18.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z15.h, z0.h[5]
    ldnt1h { z19.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z16.h, z0.h[5]
    addvl x7, x7, #8
    fmla z28.h, z17.h, z0.h[5]
    ldnt1h { z20.h }, p2/Z, [x7]
    fmla z29.h, z18.h, z0.h[5]
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z19.h, z0.h[5]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z20.h, z0.h[6]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z21.h, z0.h[6]
    ldnt1h { z2.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z22.h, z0.h[6]
    ldnt1h { z3.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z23.h, z0.h[6]
    addvl x7, x7, #8
    fmla z28.h, z1.h, z0.h[6]
    ldnt1h { z4.h }, p2/Z, [x7]
    fmla z29.h, z2.h, z0.h[6]
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    fmla z30.h, z3.h, z0.h[6]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z4.h, z0.h[7]
    ldnt1h { z8.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z5.h, z0.h[7]
    ldnt1h { z9.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z6.h, z0.h[7]
    ldnt1h { z10.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z7.h, z0.h[7]
    addvl x7, x7, #8
    fmla z28.h, z8.h, z0.h[7]
    fmla z29.h, z9.h, z0.h[7]
    fmla z30.h, z10.h, z0.h[7]
    bgt label_46
KAI_ASM_LABEL(label_47)  // Width 7: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z11.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z11.h, z0.h[0]
    ldnt1h { z15.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z12.h, z0.h[0]
    ldnt1h { z16.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z13.h, z0.h[0]
    ldnt1h { z17.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z14.h, z0.h[0]
    addvl x7, x7, #8
    fmla z28.h, z15.h, z0.h[0]
    fmla z29.h, z16.h, z0.h[0]
    fmla z30.h, z17.h, z0.h[0]
    ble label_48
    ldnt1h { z18.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z19.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z18.h, z0.h[1]
    ldnt1h { z22.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z19.h, z0.h[1]
    ldnt1h { z23.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z20.h, z0.h[1]
    ldnt1h { z1.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z21.h, z0.h[1]
    addvl x7, x7, #8
    fmla z28.h, z22.h, z0.h[1]
    fmla z29.h, z23.h, z0.h[1]
    fmla z30.h, z1.h, z0.h[1]
    ble label_48
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z2.h, z0.h[2]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z3.h, z0.h[2]
    ldnt1h { z7.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z4.h, z0.h[2]
    ldnt1h { z8.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z5.h, z0.h[2]
    addvl x7, x7, #8
    fmla z28.h, z6.h, z0.h[2]
    fmla z29.h, z7.h, z0.h[2]
    fmla z30.h, z8.h, z0.h[2]
    ble label_48
    ldnt1h { z9.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z9.h, z0.h[3]
    ldnt1h { z13.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z10.h, z0.h[3]
    ldnt1h { z14.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z11.h, z0.h[3]
    ldnt1h { z15.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z12.h, z0.h[3]
    addvl x7, x7, #8
    fmla z28.h, z13.h, z0.h[3]
    fmla z29.h, z14.h, z0.h[3]
    fmla z30.h, z15.h, z0.h[3]
    ble label_48
    ldnt1h { z16.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z16.h, z0.h[4]
    ldnt1h { z20.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z17.h, z0.h[4]
    ldnt1h { z21.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z18.h, z0.h[4]
    ldnt1h { z22.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z19.h, z0.h[4]
    addvl x7, x7, #8
    fmla z28.h, z20.h, z0.h[4]
    fmla z29.h, z21.h, z0.h[4]
    fmla z30.h, z22.h, z0.h[4]
    ble label_48
    ldnt1h { z23.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z1.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z23.h, z0.h[5]
    ldnt1h { z4.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z1.h, z0.h[5]
    ldnt1h { z5.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z2.h, z0.h[5]
    ldnt1h { z6.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z3.h, z0.h[5]
    addvl x7, x7, #8
    fmla z28.h, z4.h, z0.h[5]
    fmla z29.h, z5.h, z0.h[5]
    fmla z30.h, z6.h, z0.h[5]
    ble label_48
    ldnt1h { z7.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z7.h, z0.h[6]
    ldnt1h { z11.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z8.h, z0.h[6]
    ldnt1h { z12.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z9.h, z0.h[6]
    ldnt1h { z13.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z10.h, z0.h[6]
    addvl x7, x7, #8
    fmla z28.h, z11.h, z0.h[6]
    fmla z29.h, z12.h, z0.h[6]
    fmla z30.h, z13.h, z0.h[6]
    ble label_48
    ldnt1h { z14.h }, p2/Z, [x7]
    ldnt1h { z15.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z14.h, z0.h[7]
    ldnt1h { z18.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z15.h, z0.h[7]
    ldnt1h { z19.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z16.h, z0.h[7]
    ldnt1h { z20.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z17.h, z0.h[7]
    fmla z28.h, z18.h, z0.h[7]
    fmla z29.h, z19.h, z0.h[7]
    fmla z30.h, z20.h, z0.h[7]
KAI_ASM_LABEL(label_48)  // Width 7: Multiply loop: multiply skip
    tbz x21, #1, label_49
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmin z28.h, p2/M, z28.h, z17.h
    fmin z29.h, p2/M, z29.h, z17.h
    fmin z30.h, p2/M, z30.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
    fmax z28.h, p2/M, z28.h, z16.h
    fmax z29.h, p2/M, z29.h, z16.h
    fmax z30.h, p2/M, z30.h, z16.h
KAI_ASM_LABEL(label_49)  // Width 7: No activation
    st1h { z24.h }, p2, [x5]
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p2, [x5, #2, MUL VL]
    st1h { z27.h }, p2, [x5, #3, MUL VL]
    st1h { z28.h }, p2, [x5, #4, MUL VL]
    st1h { z29.h }, p2, [x5, #5, MUL VL]
    st1h { z30.h }, p1, [x5, #6, MUL VL]
    b label_57
KAI_ASM_LABEL(label_50)  // Width 8
    mov x1, #0x7
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.h, XZR, x1
    cbz x4, label_51
    ld1h { z24.h }, p2/Z, [x4]
    ld1h { z25.h }, p2/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p2/Z, [x4, #2, MUL VL]
    ld1h { z27.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z28.h }, p2/Z, [x4, #4, MUL VL]
    ld1h { z29.h }, p2/Z, [x4, #5, MUL VL]
    ld1h { z30.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z31.h }, p1/Z, [x4, #7, MUL VL]
    incb x4, ALL, MUL #8
    b label_52
KAI_ASM_LABEL(label_51)  // Width 8: no bias
    fmov z24.h, #0
    fmov z25.h, #0
    fmov z26.h, #0
    fmov z27.h, #0
    fmov z28.h, #0
    fmov z29.h, #0
    fmov z30.h, #0
    fmov z31.h, #0
KAI_ASM_LABEL(label_52)  // Width 8: setup done
    cmp x3, #0x8
    ble label_54
KAI_ASM_LABEL(label_53)  // Width 8: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    ldnt1h { z1.h }, p2/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z0.h }, p0/Z, [x2]
    cmp x3, #0x8
    add x2, x2, #0x10
    ldnt1h { z2.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z3.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z4.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z5.h, z0.h[0]
    addvl x7, x7, #8
    fmla z29.h, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7]
    fmla z30.h, z7.h, z0.h[0]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z8.h, z0.h[0]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z10.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z11.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z12.h, z0.h[1]
    ldnt1h { z16.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z13.h, z0.h[1]
    addvl x7, x7, #8
    fmla z29.h, z14.h, z0.h[1]
    ldnt1h { z17.h }, p2/Z, [x7]
    fmla z30.h, z15.h, z0.h[1]
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z16.h, z0.h[1]
    ldnt1h { z19.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z17.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z18.h, z0.h[2]
    ldnt1h { z22.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z19.h, z0.h[2]
    ldnt1h { z23.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z20.h, z0.h[2]
    ldnt1h { z1.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z21.h, z0.h[2]
    addvl x7, x7, #8
    fmla z29.h, z22.h, z0.h[2]
    ldnt1h { z2.h }, p2/Z, [x7]
    fmla z30.h, z23.h, z0.h[2]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z1.h, z0.h[2]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z2.h, z0.h[3]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z3.h, z0.h[3]
    ldnt1h { z7.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z4.h, z0.h[3]
    ldnt1h { z8.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z5.h, z0.h[3]
    ldnt1h { z9.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z6.h, z0.h[3]
    addvl x7, x7, #8
    fmla z29.h, z7.h, z0.h[3]
    ldnt1h { z10.h }, p2/Z, [x7]
    fmla z30.h, z8.h, z0.h[3]
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z9.h, z0.h[3]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z10.h, z0.h[4]
    ldnt1h { z14.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z11.h, z0.h[4]
    ldnt1h { z15.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z12.h, z0.h[4]
    ldnt1h { z16.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z13.h, z0.h[4]
    ldnt1h { z17.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z14.h, z0.h[4]
    addvl x7, x7, #8
    fmla z29.h, z15.h, z0.h[4]
    ldnt1h { z18.h }, p2/Z, [x7]
    fmla z30.h, z16.h, z0.h[4]
    ldnt1h { z19.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z17.h, z0.h[4]
    ldnt1h { z20.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z18.h, z0.h[5]
    ldnt1h { z22.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z19.h, z0.h[5]
    ldnt1h { z23.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z20.h, z0.h[5]
    ldnt1h { z1.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z21.h, z0.h[5]
    ldnt1h { z2.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z22.h, z0.h[5]
    addvl x7, x7, #8
    fmla z29.h, z23.h, z0.h[5]
    ldnt1h { z3.h }, p2/Z, [x7]
    fmla z30.h, z1.h, z0.h[5]
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z2.h, z0.h[5]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z3.h, z0.h[6]
    ldnt1h { z7.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z4.h, z0.h[6]
    ldnt1h { z8.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z5.h, z0.h[6]
    ldnt1h { z9.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z6.h, z0.h[6]
    ldnt1h { z10.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z7.h, z0.h[6]
    addvl x7, x7, #8
    fmla z29.h, z8.h, z0.h[6]
    ldnt1h { z11.h }, p2/Z, [x7]
    fmla z30.h, z9.h, z0.h[6]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    fmla z31.h, z10.h, z0.h[6]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z11.h, z0.h[7]
    ldnt1h { z15.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z12.h, z0.h[7]
    ldnt1h { z16.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z13.h, z0.h[7]
    ldnt1h { z17.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z14.h, z0.h[7]
    ldnt1h { z18.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z15.h, z0.h[7]
    addvl x7, x7, #8
    fmla z29.h, z16.h, z0.h[7]
    fmla z30.h, z17.h, z0.h[7]
    fmla z31.h, z18.h, z0.h[7]
    bgt label_53
KAI_ASM_LABEL(label_54)  // Width 8: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z19.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z19.h, z0.h[0]
    ldnt1h { z23.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z20.h, z0.h[0]
    ldnt1h { z1.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z21.h, z0.h[0]
    ldnt1h { z2.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z22.h, z0.h[0]
    ldnt1h { z3.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z23.h, z0.h[0]
    addvl x7, x7, #8
    fmla z29.h, z1.h, z0.h[0]
    fmla z30.h, z2.h, z0.h[0]
    fmla z31.h, z3.h, z0.h[0]
    ble label_55
    ldnt1h { z4.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z4.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z5.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z6.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z7.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z8.h, z0.h[1]
    addvl x7, x7, #8
    fmla z29.h, z9.h, z0.h[1]
    fmla z30.h, z10.h, z0.h[1]
    fmla z31.h, z11.h, z0.h[1]
    ble label_55
    ldnt1h { z12.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z13.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z12.h, z0.h[2]
    ldnt1h { z16.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z13.h, z0.h[2]
    ldnt1h { z17.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z14.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z15.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z16.h, z0.h[2]
    addvl x7, x7, #8
    fmla z29.h, z17.h, z0.h[2]
    fmla z30.h, z18.h, z0.h[2]
    fmla z31.h, z19.h, z0.h[2]
    ble label_55
    ldnt1h { z20.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z20.h, z0.h[3]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z21.h, z0.h[3]
    ldnt1h { z2.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z22.h, z0.h[3]
    ldnt1h { z3.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z23.h, z0.h[3]
    ldnt1h { z4.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z1.h, z0.h[3]
    addvl x7, x7, #8
    fmla z29.h, z2.h, z0.h[3]
    fmla z30.h, z3.h, z0.h[3]
    fmla z31.h, z4.h, z0.h[3]
    ble label_55
    ldnt1h { z5.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z5.h, z0.h[4]
    ldnt1h { z9.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z6.h, z0.h[4]
    ldnt1h { z10.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z7.h, z0.h[4]
    ldnt1h { z11.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z8.h, z0.h[4]
    ldnt1h { z12.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z9.h, z0.h[4]
    addvl x7, x7, #8
    fmla z29.h, z10.h, z0.h[4]
    fmla z30.h, z11.h, z0.h[4]
    fmla z31.h, z12.h, z0.h[4]
    ble label_55
    ldnt1h { z13.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z13.h, z0.h[5]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z14.h, z0.h[5]
    ldnt1h { z18.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z15.h, z0.h[5]
    ldnt1h { z19.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z16.h, z0.h[5]
    ldnt1h { z20.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z17.h, z0.h[5]
    addvl x7, x7, #8
    fmla z29.h, z18.h, z0.h[5]
    fmla z30.h, z19.h, z0.h[5]
    fmla z31.h, z20.h, z0.h[5]
    ble label_55
    ldnt1h { z21.h }, p2/Z, [x7]
    subs x3, x3, #0x1
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z21.h, z0.h[6]
    ldnt1h { z2.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z22.h, z0.h[6]
    ldnt1h { z3.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z23.h, z0.h[6]
    ldnt1h { z4.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z1.h, z0.h[6]
    ldnt1h { z5.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z2.h, z0.h[6]
    addvl x7, x7, #8
    fmla z29.h, z3.h, z0.h[6]
    fmla z30.h, z4.h, z0.h[6]
    fmla z31.h, z5.h, z0.h[6]
    ble label_55
    ldnt1h { z6.h }, p2/Z, [x7]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    fmla z24.h, z6.h, z0.h[7]
    ldnt1h { z10.h }, p2/Z, [x7, #4, MUL VL]
    fmla z25.h, z7.h, z0.h[7]
    ldnt1h { z11.h }, p2/Z, [x7, #5, MUL VL]
    fmla z26.h, z8.h, z0.h[7]
    ldnt1h { z12.h }, p2/Z, [x7, #6, MUL VL]
    fmla z27.h, z9.h, z0.h[7]
    ldnt1h { z13.h }, p2/Z, [x7, #7, MUL VL]
    fmla z28.h, z10.h, z0.h[7]
    addvl x7, x7, #8
    fmla z29.h, z11.h, z0.h[7]
    fmla z30.h, z12.h, z0.h[7]
    fmla z31.h, z13.h, z0.h[7]
KAI_ASM_LABEL(label_55)  // Width 8: Multiply loop: multiply skip
    tbz x21, #1, label_56
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0a851)  // ld1rh { z17.h }, p2/Z, [x2]
    KAI_ASM_INST(0x84c0a830)  // ld1rh { z16.h }, p2/Z, [x1]
    fmin z24.h, p2/M, z24.h, z17.h
    fmin z25.h, p2/M, z25.h, z17.h
    fmin z26.h, p2/M, z26.h, z17.h
    fmin z27.h, p2/M, z27.h, z17.h
    fmin z28.h, p2/M, z28.h, z17.h
    fmin z29.h, p2/M, z29.h, z17.h
    fmin z30.h, p2/M, z30.h, z17.h
    fmin z31.h, p2/M, z31.h, z17.h
    fmax z24.h, p2/M, z24.h, z16.h
    fmax z25.h, p2/M, z25.h, z16.h
    fmax z26.h, p2/M, z26.h, z16.h
    fmax z27.h, p2/M, z27.h, z16.h
    fmax z28.h, p2/M, z28.h, z16.h
    fmax z29.h, p2/M, z29.h, z16.h
    fmax z30.h, p2/M, z30.h, z16.h
    fmax z31.h, p2/M, z31.h, z16.h
KAI_ASM_LABEL(label_56)  // Width 8: No activation
    subs x6, x6, #0x8
    st1h { z24.h }, p2, [x5]
    sub x20, x20, x22, LSL #3
    st1h { z25.h }, p2, [x5, #1, MUL VL]
    st1h { z26.h }, p2, [x5, #2, MUL VL]
    st1h { z27.h }, p2, [x5, #3, MUL VL]
    st1h { z28.h }, p2, [x5, #4, MUL VL]
    st1h { z29.h }, p2, [x5, #5, MUL VL]
    st1h { z30.h }, p2, [x5, #6, MUL VL]
    st1h { z31.h }, p1, [x5, #7, MUL VL]
    addvl x5, x5, #8
    bgt label_1
KAI_ASM_LABEL(label_57)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldr x22, [sp, 16]
    ldp d8, d9, [sp, 24]
    ldp d10, d11, [sp, 40]
    ldp d12, d13, [sp, 56]
    ldp d14, d15, [sp, 72]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_gemv_fp16_mla_8VL)

    KAI_ASM_END
