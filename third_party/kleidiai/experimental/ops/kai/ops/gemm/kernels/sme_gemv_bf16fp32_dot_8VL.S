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

    KAI_ASM_CODE(sme_gemv_bf16fp32_dot_8VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_gemv_bf16fp32_dot_8VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_gemv_bf16fp32_dot_8VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_gemv_bf16fp32_dot_8VL)
    stp x20, x21, [sp, -96]!
    str x22, [sp, 16]
    stp d8, d9, [sp, 24]
    stp d10, d11, [sp, 40]
    stp d12, d13, [sp, 56]
    stp d14, d15, [sp, 72]
    mov x4, #0xff800000
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr w1, [x0, #0xa0]
    mov x3, #0x7f800000
    cntw x22
    ldr w2, [x0, #0xa4]
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
    whilelt p1.s, XZR, x20
    cbz x4, label_2
    ld1w { z24.s }, p1/Z, [x4]
    b label_3
KAI_ASM_LABEL(label_2)  // Width 1: no bias
    fmov z24.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64684058)  // bfdot z24.s, z2.h, z0.h[1]
    KAI_ASM_INST(0x64704078)  // bfdot z24.s, z3.h, z0.h[2]
    KAI_ASM_INST(0x64784098)  // bfdot z24.s, z4.h, z0.h[3]
    bgt label_4
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z5.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646040b8)  // bfdot z24.s, z5.h, z0.h[0]
    ble label_6
    ldnt1h { z6.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #8
    KAI_ASM_INST(0x646840d8)  // bfdot z24.s, z6.h, z0.h[1]
    ble label_6
    ldnt1h { z7.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #8
    KAI_ASM_INST(0x647040f8)  // bfdot z24.s, z7.h, z0.h[2]
    ble label_6
    ldnt1h { z8.h }, p2/Z, [x7]
    KAI_ASM_INST(0x64784118)  // bfdot z24.s, z8.h, z0.h[3]
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_7
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
KAI_ASM_LABEL(label_7)  // Width 1: No activation
    st1w { z24.s }, p1, [x5]
    b label_57
KAI_ASM_LABEL(label_8)  // Width 2
    sub x1, x20, x22
    mov x3, x8
    ldr x2, [x0, #0x20]
    whilelt p1.s, XZR, x1
    cbz x4, label_9
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p1/Z, [x4, #1, MUL VL]
    b label_10
KAI_ASM_LABEL(label_9)  // Width 2: no bias
    fmov z24.s, #0
    fmov z25.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7]
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z7.h }, p2/Z, [x7]
    KAI_ASM_INST(0x64684078)  // bfdot z24.s, z3.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64684099)  // bfdot z25.s, z4.h, z0.h[1]
    KAI_ASM_INST(0x647040b8)  // bfdot z24.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x647040d9)  // bfdot z25.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x647840f8)  // bfdot z24.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64784119)  // bfdot z25.s, z8.h, z0.h[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z9.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64604138)  // bfdot z24.s, z9.h, z0.h[0]
    KAI_ASM_INST(0x64604159)  // bfdot z25.s, z10.h, z0.h[0]
    ble label_13
    ldnt1h { z11.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64684178)  // bfdot z24.s, z11.h, z0.h[1]
    KAI_ASM_INST(0x64684199)  // bfdot z25.s, z12.h, z0.h[1]
    ble label_13
    ldnt1h { z13.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647041b8)  // bfdot z24.s, z13.h, z0.h[2]
    KAI_ASM_INST(0x647041d9)  // bfdot z25.s, z14.h, z0.h[2]
    ble label_13
    ldnt1h { z15.h }, p2/Z, [x7]
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x647841f8)  // bfdot z24.s, z15.h, z0.h[3]
    KAI_ASM_INST(0x64784219)  // bfdot z25.s, z16.h, z0.h[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_14
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p1, [x5, #1, MUL VL]
    b label_57
KAI_ASM_LABEL(label_15)  // Width 3
    mov x1, #0x2
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_16
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p1/Z, [x4, #2, MUL VL]
    b label_17
KAI_ASM_LABEL(label_16)  // Width 3: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z4.h }, p2/Z, [x7]
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    ldnt1h { z7.h }, p2/Z, [x7]
    KAI_ASM_INST(0x64684098)  // bfdot z24.s, z4.h, z0.h[1]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x646840b9)  // bfdot z25.s, z5.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646840da)  // bfdot z26.s, z6.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7]
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x647040f8)  // bfdot z24.s, z7.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64704119)  // bfdot z25.s, z8.h, z0.h[2]
    KAI_ASM_INST(0x6470413a)  // bfdot z26.s, z9.h, z0.h[2]
    KAI_ASM_INST(0x64784158)  // bfdot z24.s, z10.h, z0.h[3]
    KAI_ASM_INST(0x64784179)  // bfdot z25.s, z11.h, z0.h[3]
    KAI_ASM_INST(0x6478419a)  // bfdot z26.s, z12.h, z0.h[3]
    bgt label_18
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z13.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646041b8)  // bfdot z24.s, z13.h, z0.h[0]
    KAI_ASM_INST(0x646041d9)  // bfdot z25.s, z14.h, z0.h[0]
    KAI_ASM_INST(0x646041fa)  // bfdot z26.s, z15.h, z0.h[0]
    ble label_20
    ldnt1h { z16.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64684218)  // bfdot z24.s, z16.h, z0.h[1]
    KAI_ASM_INST(0x64684239)  // bfdot z25.s, z17.h, z0.h[1]
    KAI_ASM_INST(0x6468425a)  // bfdot z26.s, z18.h, z0.h[1]
    ble label_20
    ldnt1h { z19.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #2, MUL VL]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64704278)  // bfdot z24.s, z19.h, z0.h[2]
    KAI_ASM_INST(0x64704299)  // bfdot z25.s, z20.h, z0.h[2]
    KAI_ASM_INST(0x647042ba)  // bfdot z26.s, z21.h, z0.h[2]
    ble label_20
    ldnt1h { z22.h }, p2/Z, [x7]
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    KAI_ASM_INST(0x647842d8)  // bfdot z24.s, z22.h, z0.h[3]
    KAI_ASM_INST(0x647842f9)  // bfdot z25.s, z23.h, z0.h[3]
    KAI_ASM_INST(0x6478403a)  // bfdot z26.s, z1.h, z0.h[3]
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_21
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
KAI_ASM_LABEL(label_21)  // Width 3: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p1, [x5, #2, MUL VL]
    b label_57
KAI_ASM_LABEL(label_22)  // Width 4
    mov x1, #0x3
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_23
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z27.s }, p1/Z, [x4, #3, MUL VL]
    b label_24
KAI_ASM_LABEL(label_23)  // Width 4: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6460409b)  // bfdot z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646840b8)  // bfdot z24.s, z5.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646840d9)  // bfdot z25.s, z6.h, z0.h[1]
    ldnt1h { z9.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646840fa)  // bfdot z26.s, z7.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6468411b)  // bfdot z27.s, z8.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704138)  // bfdot z24.s, z9.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64704159)  // bfdot z25.s, z10.h, z0.h[2]
    ldnt1h { z13.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6470417a)  // bfdot z26.s, z11.h, z0.h[2]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6470419b)  // bfdot z27.s, z12.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647841b8)  // bfdot z24.s, z13.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647841d9)  // bfdot z25.s, z14.h, z0.h[3]
    KAI_ASM_INST(0x647841fa)  // bfdot z26.s, z15.h, z0.h[3]
    KAI_ASM_INST(0x6478421b)  // bfdot z27.s, z16.h, z0.h[3]
    bgt label_25
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z17.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64604238)  // bfdot z24.s, z17.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64604259)  // bfdot z25.s, z18.h, z0.h[0]
    KAI_ASM_INST(0x6460427a)  // bfdot z26.s, z19.h, z0.h[0]
    KAI_ASM_INST(0x6460429b)  // bfdot z27.s, z20.h, z0.h[0]
    ble label_27
    ldnt1h { z21.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646842b8)  // bfdot z24.s, z21.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646842d9)  // bfdot z25.s, z22.h, z0.h[1]
    KAI_ASM_INST(0x646842fa)  // bfdot z26.s, z23.h, z0.h[1]
    KAI_ASM_INST(0x6468403b)  // bfdot z27.s, z1.h, z0.h[1]
    ble label_27
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704058)  // bfdot z24.s, z2.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x64704079)  // bfdot z25.s, z3.h, z0.h[2]
    KAI_ASM_INST(0x6470409a)  // bfdot z26.s, z4.h, z0.h[2]
    KAI_ASM_INST(0x647040bb)  // bfdot z27.s, z5.h, z0.h[2]
    ble label_27
    ldnt1h { z6.h }, p2/Z, [x7]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647840d8)  // bfdot z24.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x647840f9)  // bfdot z25.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x6478411a)  // bfdot z26.s, z8.h, z0.h[3]
    KAI_ASM_INST(0x6478413b)  // bfdot z27.s, z9.h, z0.h[3]
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: multiply skip
    tbz x21, #1, label_28
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
KAI_ASM_LABEL(label_28)  // Width 4: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p2, [x5, #2, MUL VL]
    st1w { z27.s }, p1, [x5, #3, MUL VL]
    b label_57
KAI_ASM_LABEL(label_29)  // Width 5
    mov x1, #0x4
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_30
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z28.s }, p1/Z, [x4, #4, MUL VL]
    b label_31
KAI_ASM_LABEL(label_30)  // Width 5: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6460409b)  // bfdot z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x646040bc)  // bfdot z28.s, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646840d8)  // bfdot z24.s, z6.h, z0.h[1]
    ldnt1h { z10.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x646840f9)  // bfdot z25.s, z7.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6468411a)  // bfdot z26.s, z8.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6468413b)  // bfdot z27.s, z9.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6468415c)  // bfdot z28.s, z10.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704178)  // bfdot z24.s, z11.h, z0.h[2]
    ldnt1h { z15.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64704199)  // bfdot z25.s, z12.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647041ba)  // bfdot z26.s, z13.h, z0.h[2]
    ldnt1h { z16.h }, p2/Z, [x7]
    KAI_ASM_INST(0x647041db)  // bfdot z27.s, z14.h, z0.h[2]
    ldnt1h { z17.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x647041fc)  // bfdot z28.s, z15.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z19.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784218)  // bfdot z24.s, z16.h, z0.h[3]
    ldnt1h { z20.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64784239)  // bfdot z25.s, z17.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6478425a)  // bfdot z26.s, z18.h, z0.h[3]
    KAI_ASM_INST(0x6478427b)  // bfdot z27.s, z19.h, z0.h[3]
    KAI_ASM_INST(0x6478429c)  // bfdot z28.s, z20.h, z0.h[3]
    bgt label_32
KAI_ASM_LABEL(label_33)  // Width 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z21.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z22.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z1.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646042b8)  // bfdot z24.s, z21.h, z0.h[0]
    ldnt1h { z2.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x646042d9)  // bfdot z25.s, z22.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646042fa)  // bfdot z26.s, z23.h, z0.h[0]
    KAI_ASM_INST(0x6460403b)  // bfdot z27.s, z1.h, z0.h[0]
    KAI_ASM_INST(0x6460405c)  // bfdot z28.s, z2.h, z0.h[0]
    ble label_34
    ldnt1h { z3.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64684078)  // bfdot z24.s, z3.h, z0.h[1]
    ldnt1h { z7.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684099)  // bfdot z25.s, z4.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646840ba)  // bfdot z26.s, z5.h, z0.h[1]
    KAI_ASM_INST(0x646840db)  // bfdot z27.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x646840fc)  // bfdot z28.s, z7.h, z0.h[1]
    ble label_34
    ldnt1h { z8.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704118)  // bfdot z24.s, z8.h, z0.h[2]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64704139)  // bfdot z25.s, z9.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470415a)  // bfdot z26.s, z10.h, z0.h[2]
    KAI_ASM_INST(0x6470417b)  // bfdot z27.s, z11.h, z0.h[2]
    KAI_ASM_INST(0x6470419c)  // bfdot z28.s, z12.h, z0.h[2]
    ble label_34
    ldnt1h { z13.h }, p2/Z, [x7]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647841b8)  // bfdot z24.s, z13.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647841d9)  // bfdot z25.s, z14.h, z0.h[3]
    KAI_ASM_INST(0x647841fa)  // bfdot z26.s, z15.h, z0.h[3]
    KAI_ASM_INST(0x6478421b)  // bfdot z27.s, z16.h, z0.h[3]
    KAI_ASM_INST(0x6478423c)  // bfdot z28.s, z17.h, z0.h[3]
KAI_ASM_LABEL(label_34)  // Width 5: Multiply loop: multiply skip
    tbz x21, #1, label_35
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
KAI_ASM_LABEL(label_35)  // Width 5: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p2, [x5, #2, MUL VL]
    st1w { z27.s }, p2, [x5, #3, MUL VL]
    st1w { z28.s }, p1, [x5, #4, MUL VL]
    b label_57
KAI_ASM_LABEL(label_36)  // Width 6
    mov x1, #0x5
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_37
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z28.s }, p2/Z, [x4, #4, MUL VL]
    ld1w { z29.s }, p1/Z, [x4, #5, MUL VL]
    b label_38
KAI_ASM_LABEL(label_37)  // Width 6: no bias
    fmov z24.s, #0
    fmov z25.s, #0
    fmov z26.s, #0
    fmov z27.s, #0
    fmov z28.s, #0
    fmov z29.s, #0
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6460409b)  // bfdot z27.s, z4.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646040bc)  // bfdot z28.s, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x646040dd)  // bfdot z29.s, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646840f8)  // bfdot z24.s, z7.h, z0.h[1]
    ldnt1h { z11.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684119)  // bfdot z25.s, z8.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6468413a)  // bfdot z26.s, z9.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6468415b)  // bfdot z27.s, z10.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6468417c)  // bfdot z28.s, z11.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6468419d)  // bfdot z29.s, z12.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647041b8)  // bfdot z24.s, z13.h, z0.h[2]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647041d9)  // bfdot z25.s, z14.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647041fa)  // bfdot z26.s, z15.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470421b)  // bfdot z27.s, z16.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6470423c)  // bfdot z28.s, z17.h, z0.h[2]
    ldnt1h { z20.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6470425d)  // bfdot z29.s, z18.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784278)  // bfdot z24.s, z19.h, z0.h[3]
    ldnt1h { z23.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64784299)  // bfdot z25.s, z20.h, z0.h[3]
    ldnt1h { z1.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647842ba)  // bfdot z26.s, z21.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647842db)  // bfdot z27.s, z22.h, z0.h[3]
    KAI_ASM_INST(0x647842fc)  // bfdot z28.s, z23.h, z0.h[3]
    KAI_ASM_INST(0x6478403d)  // bfdot z29.s, z1.h, z0.h[3]
    bgt label_39
KAI_ASM_LABEL(label_40)  // Width 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z2.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64604058)  // bfdot z24.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604079)  // bfdot z25.s, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460409a)  // bfdot z26.s, z4.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646040bb)  // bfdot z27.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x646040dc)  // bfdot z28.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x646040fd)  // bfdot z29.s, z7.h, z0.h[0]
    ble label_41
    ldnt1h { z8.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64684118)  // bfdot z24.s, z8.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684139)  // bfdot z25.s, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6468415a)  // bfdot z26.s, z10.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6468417b)  // bfdot z27.s, z11.h, z0.h[1]
    KAI_ASM_INST(0x6468419c)  // bfdot z28.s, z12.h, z0.h[1]
    KAI_ASM_INST(0x646841bd)  // bfdot z29.s, z13.h, z0.h[1]
    ble label_41
    ldnt1h { z14.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z15.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z17.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647041d8)  // bfdot z24.s, z14.h, z0.h[2]
    ldnt1h { z18.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647041f9)  // bfdot z25.s, z15.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6470421a)  // bfdot z26.s, z16.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470423b)  // bfdot z27.s, z17.h, z0.h[2]
    KAI_ASM_INST(0x6470425c)  // bfdot z28.s, z18.h, z0.h[2]
    KAI_ASM_INST(0x6470427d)  // bfdot z29.s, z19.h, z0.h[2]
    ble label_41
    ldnt1h { z20.h }, p2/Z, [x7]
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784298)  // bfdot z24.s, z20.h, z0.h[3]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647842b9)  // bfdot z25.s, z21.h, z0.h[3]
    ldnt1h { z2.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647842da)  // bfdot z26.s, z22.h, z0.h[3]
    KAI_ASM_INST(0x647842fb)  // bfdot z27.s, z23.h, z0.h[3]
    KAI_ASM_INST(0x6478403c)  // bfdot z28.s, z1.h, z0.h[3]
    KAI_ASM_INST(0x6478405d)  // bfdot z29.s, z2.h, z0.h[3]
KAI_ASM_LABEL(label_41)  // Width 6: Multiply loop: multiply skip
    tbz x21, #1, label_42
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
KAI_ASM_LABEL(label_42)  // Width 6: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p2, [x5, #2, MUL VL]
    st1w { z27.s }, p2, [x5, #3, MUL VL]
    st1w { z28.s }, p2, [x5, #4, MUL VL]
    st1w { z29.s }, p1, [x5, #5, MUL VL]
    b label_57
KAI_ASM_LABEL(label_43)  // Width 7
    mov x1, #0x6
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_44
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z28.s }, p2/Z, [x4, #4, MUL VL]
    ld1w { z29.s }, p2/Z, [x4, #5, MUL VL]
    ld1w { z30.s }, p1/Z, [x4, #6, MUL VL]
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6460409b)  // bfdot z27.s, z4.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646040bc)  // bfdot z28.s, z5.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646040dd)  // bfdot z29.s, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x646040fe)  // bfdot z30.s, z7.h, z0.h[0]
    ldnt1h { z10.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z11.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64684118)  // bfdot z24.s, z8.h, z0.h[1]
    ldnt1h { z12.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684139)  // bfdot z25.s, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6468415a)  // bfdot z26.s, z10.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6468417b)  // bfdot z27.s, z11.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6468419c)  // bfdot z28.s, z12.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646841bd)  // bfdot z29.s, z13.h, z0.h[1]
    ldnt1h { z16.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x646841de)  // bfdot z30.s, z14.h, z0.h[1]
    ldnt1h { z17.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z18.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647041f8)  // bfdot z24.s, z15.h, z0.h[2]
    ldnt1h { z19.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64704219)  // bfdot z25.s, z16.h, z0.h[2]
    ldnt1h { z20.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6470423a)  // bfdot z26.s, z17.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6470425b)  // bfdot z27.s, z18.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470427c)  // bfdot z28.s, z19.h, z0.h[2]
    ldnt1h { z22.h }, p2/Z, [x7]
    KAI_ASM_INST(0x6470429d)  // bfdot z29.s, z20.h, z0.h[2]
    ldnt1h { z23.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x647042be)  // bfdot z30.s, z21.h, z0.h[2]
    ldnt1h { z1.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z2.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x647842d8)  // bfdot z24.s, z22.h, z0.h[3]
    ldnt1h { z3.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647842f9)  // bfdot z25.s, z23.h, z0.h[3]
    ldnt1h { z4.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6478403a)  // bfdot z26.s, z1.h, z0.h[3]
    ldnt1h { z5.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6478405b)  // bfdot z27.s, z2.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6478407c)  // bfdot z28.s, z3.h, z0.h[3]
    KAI_ASM_INST(0x6478409d)  // bfdot z29.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x647840be)  // bfdot z30.s, z5.h, z0.h[3]
    bgt label_46
KAI_ASM_LABEL(label_47)  // Width 7: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z6.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z7.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z8.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z9.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646040d8)  // bfdot z24.s, z6.h, z0.h[0]
    ldnt1h { z10.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x646040f9)  // bfdot z25.s, z7.h, z0.h[0]
    ldnt1h { z11.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460411a)  // bfdot z26.s, z8.h, z0.h[0]
    ldnt1h { z12.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6460413b)  // bfdot z27.s, z9.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6460415c)  // bfdot z28.s, z10.h, z0.h[0]
    KAI_ASM_INST(0x6460417d)  // bfdot z29.s, z11.h, z0.h[0]
    KAI_ASM_INST(0x6460419e)  // bfdot z30.s, z12.h, z0.h[0]
    ble label_48
    ldnt1h { z13.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z14.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z15.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z16.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x646841b8)  // bfdot z24.s, z13.h, z0.h[1]
    ldnt1h { z17.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x646841d9)  // bfdot z25.s, z14.h, z0.h[1]
    ldnt1h { z18.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x646841fa)  // bfdot z26.s, z15.h, z0.h[1]
    ldnt1h { z19.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6468421b)  // bfdot z27.s, z16.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6468423c)  // bfdot z28.s, z17.h, z0.h[1]
    KAI_ASM_INST(0x6468425d)  // bfdot z29.s, z18.h, z0.h[1]
    KAI_ASM_INST(0x6468427e)  // bfdot z30.s, z19.h, z0.h[1]
    ble label_48
    ldnt1h { z20.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z21.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z22.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z23.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704298)  // bfdot z24.s, z20.h, z0.h[2]
    ldnt1h { z1.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647042b9)  // bfdot z25.s, z21.h, z0.h[2]
    ldnt1h { z2.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647042da)  // bfdot z26.s, z22.h, z0.h[2]
    ldnt1h { z3.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x647042fb)  // bfdot z27.s, z23.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470403c)  // bfdot z28.s, z1.h, z0.h[2]
    KAI_ASM_INST(0x6470405d)  // bfdot z29.s, z2.h, z0.h[2]
    KAI_ASM_INST(0x6470407e)  // bfdot z30.s, z3.h, z0.h[2]
    ble label_48
    ldnt1h { z4.h }, p2/Z, [x7]
    ldnt1h { z5.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z7.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784098)  // bfdot z24.s, z4.h, z0.h[3]
    ldnt1h { z8.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x647840b9)  // bfdot z25.s, z5.h, z0.h[3]
    ldnt1h { z9.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647840da)  // bfdot z26.s, z6.h, z0.h[3]
    ldnt1h { z10.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x647840fb)  // bfdot z27.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x6478411c)  // bfdot z28.s, z8.h, z0.h[3]
    KAI_ASM_INST(0x6478413d)  // bfdot z29.s, z9.h, z0.h[3]
    KAI_ASM_INST(0x6478415e)  // bfdot z30.s, z10.h, z0.h[3]
KAI_ASM_LABEL(label_48)  // Width 7: Multiply loop: multiply skip
    tbz x21, #1, label_49
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
KAI_ASM_LABEL(label_49)  // Width 7: No activation
    st1w { z24.s }, p2, [x5]
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p2, [x5, #2, MUL VL]
    st1w { z27.s }, p2, [x5, #3, MUL VL]
    st1w { z28.s }, p2, [x5, #4, MUL VL]
    st1w { z29.s }, p2, [x5, #5, MUL VL]
    st1w { z30.s }, p1, [x5, #6, MUL VL]
    b label_57
KAI_ASM_LABEL(label_50)  // Width 8
    mov x1, #0x7
    mov x3, x8
    ldr x2, [x0, #0x20]
    msub x1, x22, x1, x20
    whilelt p1.s, XZR, x1
    cbz x4, label_51
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z25.s }, p2/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x4, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z28.s }, p2/Z, [x4, #4, MUL VL]
    ld1w { z29.s }, p2/Z, [x4, #5, MUL VL]
    ld1w { z30.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z31.s }, p1/Z, [x4, #7, MUL VL]
    incb x4, ALL, MUL #8
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
    KAI_ASM_INST(0x64604038)  // bfdot z24.s, z1.h, z0.h[0]
    ldnt1h { z5.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604059)  // bfdot z25.s, z2.h, z0.h[0]
    ldnt1h { z6.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460407a)  // bfdot z26.s, z3.h, z0.h[0]
    ldnt1h { z7.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6460409b)  // bfdot z27.s, z4.h, z0.h[0]
    ldnt1h { z8.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x646040bc)  // bfdot z28.s, z5.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646040dd)  // bfdot z29.s, z6.h, z0.h[0]
    ldnt1h { z9.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646040fe)  // bfdot z30.s, z7.h, z0.h[0]
    ldnt1h { z10.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6460411f)  // bfdot z31.s, z8.h, z0.h[0]
    ldnt1h { z11.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64684138)  // bfdot z24.s, z9.h, z0.h[1]
    ldnt1h { z13.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684159)  // bfdot z25.s, z10.h, z0.h[1]
    ldnt1h { z14.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6468417a)  // bfdot z26.s, z11.h, z0.h[1]
    ldnt1h { z15.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6468419b)  // bfdot z27.s, z12.h, z0.h[1]
    ldnt1h { z16.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x646841bc)  // bfdot z28.s, z13.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646841dd)  // bfdot z29.s, z14.h, z0.h[1]
    ldnt1h { z17.h }, p2/Z, [x7]
    KAI_ASM_INST(0x646841fe)  // bfdot z30.s, z15.h, z0.h[1]
    ldnt1h { z18.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6468421f)  // bfdot z31.s, z16.h, z0.h[1]
    ldnt1h { z19.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704238)  // bfdot z24.s, z17.h, z0.h[2]
    ldnt1h { z21.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64704259)  // bfdot z25.s, z18.h, z0.h[2]
    ldnt1h { z22.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6470427a)  // bfdot z26.s, z19.h, z0.h[2]
    ldnt1h { z23.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x6470429b)  // bfdot z27.s, z20.h, z0.h[2]
    ldnt1h { z1.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x647042bc)  // bfdot z28.s, z21.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647042dd)  // bfdot z29.s, z22.h, z0.h[2]
    ldnt1h { z2.h }, p2/Z, [x7]
    KAI_ASM_INST(0x647042fe)  // bfdot z30.s, z23.h, z0.h[2]
    ldnt1h { z3.h }, p2/Z, [x7, #1, MUL VL]
    KAI_ASM_INST(0x6470403f)  // bfdot z31.s, z1.h, z0.h[2]
    ldnt1h { z4.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784058)  // bfdot z24.s, z2.h, z0.h[3]
    ldnt1h { z6.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64784079)  // bfdot z25.s, z3.h, z0.h[3]
    ldnt1h { z7.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6478409a)  // bfdot z26.s, z4.h, z0.h[3]
    ldnt1h { z8.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x647840bb)  // bfdot z27.s, z5.h, z0.h[3]
    ldnt1h { z9.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x647840dc)  // bfdot z28.s, z6.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x647840fd)  // bfdot z29.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x6478411e)  // bfdot z30.s, z8.h, z0.h[3]
    KAI_ASM_INST(0x6478413f)  // bfdot z31.s, z9.h, z0.h[3]
    bgt label_53
KAI_ASM_LABEL(label_54)  // Width 8: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    ldnt1h { z10.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x2]
    ldnt1h { z11.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z12.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64604158)  // bfdot z24.s, z10.h, z0.h[0]
    ldnt1h { z14.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64604179)  // bfdot z25.s, z11.h, z0.h[0]
    ldnt1h { z15.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6460419a)  // bfdot z26.s, z12.h, z0.h[0]
    ldnt1h { z16.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x646041bb)  // bfdot z27.s, z13.h, z0.h[0]
    ldnt1h { z17.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x646041dc)  // bfdot z28.s, z14.h, z0.h[0]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646041fd)  // bfdot z29.s, z15.h, z0.h[0]
    KAI_ASM_INST(0x6460421e)  // bfdot z30.s, z16.h, z0.h[0]
    KAI_ASM_INST(0x6460423f)  // bfdot z31.s, z17.h, z0.h[0]
    ble label_55
    ldnt1h { z18.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z19.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z20.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z21.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64684258)  // bfdot z24.s, z18.h, z0.h[1]
    ldnt1h { z22.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64684279)  // bfdot z25.s, z19.h, z0.h[1]
    ldnt1h { z23.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x6468429a)  // bfdot z26.s, z20.h, z0.h[1]
    ldnt1h { z1.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x646842bb)  // bfdot z27.s, z21.h, z0.h[1]
    ldnt1h { z2.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x646842dc)  // bfdot z28.s, z22.h, z0.h[1]
    addvl x7, x7, #8
    KAI_ASM_INST(0x646842fd)  // bfdot z29.s, z23.h, z0.h[1]
    KAI_ASM_INST(0x6468403e)  // bfdot z30.s, z1.h, z0.h[1]
    KAI_ASM_INST(0x6468405f)  // bfdot z31.s, z2.h, z0.h[1]
    ble label_55
    ldnt1h { z3.h }, p2/Z, [x7]
    subs x3, x3, #0x2
    ldnt1h { z4.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z5.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z6.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64704078)  // bfdot z24.s, z3.h, z0.h[2]
    ldnt1h { z7.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64704099)  // bfdot z25.s, z4.h, z0.h[2]
    ldnt1h { z8.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647040ba)  // bfdot z26.s, z5.h, z0.h[2]
    ldnt1h { z9.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x647040db)  // bfdot z27.s, z6.h, z0.h[2]
    ldnt1h { z10.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x647040fc)  // bfdot z28.s, z7.h, z0.h[2]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6470411d)  // bfdot z29.s, z8.h, z0.h[2]
    KAI_ASM_INST(0x6470413e)  // bfdot z30.s, z9.h, z0.h[2]
    KAI_ASM_INST(0x6470415f)  // bfdot z31.s, z10.h, z0.h[2]
    ble label_55
    ldnt1h { z11.h }, p2/Z, [x7]
    ldnt1h { z12.h }, p2/Z, [x7, #1, MUL VL]
    ldnt1h { z13.h }, p2/Z, [x7, #2, MUL VL]
    ldnt1h { z14.h }, p2/Z, [x7, #3, MUL VL]
    KAI_ASM_INST(0x64784178)  // bfdot z24.s, z11.h, z0.h[3]
    ldnt1h { z15.h }, p2/Z, [x7, #4, MUL VL]
    KAI_ASM_INST(0x64784199)  // bfdot z25.s, z12.h, z0.h[3]
    ldnt1h { z16.h }, p2/Z, [x7, #5, MUL VL]
    KAI_ASM_INST(0x647841ba)  // bfdot z26.s, z13.h, z0.h[3]
    ldnt1h { z17.h }, p2/Z, [x7, #6, MUL VL]
    KAI_ASM_INST(0x647841db)  // bfdot z27.s, z14.h, z0.h[3]
    ldnt1h { z18.h }, p2/Z, [x7, #7, MUL VL]
    KAI_ASM_INST(0x647841fc)  // bfdot z28.s, z15.h, z0.h[3]
    addvl x7, x7, #8
    KAI_ASM_INST(0x6478421d)  // bfdot z29.s, z16.h, z0.h[3]
    KAI_ASM_INST(0x6478423e)  // bfdot z30.s, z17.h, z0.h[3]
    KAI_ASM_INST(0x6478425f)  // bfdot z31.s, z18.h, z0.h[3]
KAI_ASM_LABEL(label_55)  // Width 8: Multiply loop: multiply skip
    tbz x21, #1, label_56
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmin z31.s, p2/M, z31.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
    fmax z31.s, p2/M, z31.s, z16.s
KAI_ASM_LABEL(label_56)  // Width 8: No activation
    subs x6, x6, #0x8
    st1w { z24.s }, p2, [x5]
    sub x20, x20, x22, LSL #3
    st1w { z25.s }, p2, [x5, #1, MUL VL]
    st1w { z26.s }, p2, [x5, #2, MUL VL]
    st1w { z27.s }, p2, [x5, #3, MUL VL]
    st1w { z28.s }, p2, [x5, #4, MUL VL]
    st1w { z29.s }, p2, [x5, #5, MUL VL]
    st1w { z30.s }, p2, [x5, #6, MUL VL]
    st1w { z31.s }, p1, [x5, #7, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_gemv_bf16fp32_dot_8VL)

    KAI_ASM_END
