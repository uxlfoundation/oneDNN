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

    KAI_ASM_CODE(sme2_gemv_u8qa_dot_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_u8qa_dot_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_u8qa_dot_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_u8qa_dot_16VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x23, #0
    ldr x22, [x0, #0x8]
    cntw x21, ALL, MUL #4
    mov x9, #0
    ldr x1, [x0, #0x78]
    ptrue p2.b
    mov x3, #0x1
    ldr x20, [x0, #0x10]
    add x8, x22, x21
    ldr x7, [x0, #0x30]
    sub x8, x8, #0x1
    cmp x1, #0
    ldr x6, [x0, #0x50]
    udiv x8, x8, x21
    cset x1, NE
    add x2, x8, #0x3
    orr x23, x23, x1, LSL #4
    and x2, x2, #0xfffffffffffffffc
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    mul x2, x2, x21
    mul x2, x2, x20
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x2, #0x200, LSL #12
    blt label_2
    tbnz x2, #0, label_3
    lsr x2, x2, #0x1
    lsl x3, x3, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x1, x2, #0x26
    sub x3, x3, #0x1
    lsl x3, x3, #0x16
    orr x2, x2, x1
    orr x2, x2, x3
    KAI_ASM_INST(0xf8a248fa)  // rprfm pldonce, x2, [x7]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    ldr x5, [x0, #0x40]
    mov z11.s, #0
    mov z8.b, #0x1
    bic x23, x23, #0x80000000
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x8, #0x4
    bge label_34
    cmp x8, #0x2
    bgt label_24
    beq label_14
    ldr x4, [x0, #0x20]
    mov x2, x20
    mov x1, x22
    mov x3, x20
    whilelt p1.b, XZR, x1
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_5
    KAI_ASM_INST(0xa040c0ac)  // ld1w { z12.s-z15.s }, p8.s/Z, [x5]
    KAI_ASM_INST(0xc0042d80)  // mova za.d[x9, #0], { z12.d-z15.d }
    b label_6
KAI_ASM_LABEL(label_5)  // Width 1: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_6)  // Width 1: setup done
    cmp x3, #0x10
    ble label_9
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    ld1rqb { z9.b }, p0/Z, [x4]
    add x4, x4, #0x10
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa04084f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b030)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[0]
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b730)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[1]
    KAI_ASM_INST(0xc159bab0)  // udot za.s[x9, 0], { z20.b-z23.b }, z9.b[2]
    KAI_ASM_INST(0xc159bfb0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[3]
    tbnz x23, #31, label_8
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_7
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    ld1rqb { z9.b }, p0/Z, [x4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b330)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[0]
    ble label_10
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b430)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[1]
    ble label_10
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159bbb0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[2]
    ble label_10
    KAI_ASM_INST(0xa04084f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159be30)  // udot za.s[x9, 0], { z16.b-z19.b }, z9.b[3]
KAI_ASM_LABEL(label_10)  // Width 1: Multiply loop: multiply skip
    tbnz x23, #31, label_11
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_11)  // Width 1: Multiply loop: unique 2: skip row sum
    tbnz x23, #31, label_12
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d11, p0, z11.s
    mov z11.s, z11.s[0]
    neg z17.s, p2/M, z17.s
    mul z11.s, p2/M, z11.s, z17.s
KAI_ASM_LABEL(label_12)  // Width 1: skip row sum fixup
    KAI_ASM_INST(0xc0904960)  // addha za0.s, p2/M, p2/M, z11.s
    add x2, x0, #0x98
    add x1, x0, #0x9c
    KAI_ASM_INST(0xc0904961)  // addha za1.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c84a)  // ld1rw { z10.s }, p2/Z, [x2]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0xc0904962)  // addha za2.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c826)  // ld1rw { z6.s }, p2/Z, [x1]
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0904963)  // addha za3.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c867)  // ld1rw { z7.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c85e)  // ld1rw { z30.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c837)  // ld1rw { z23.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc1aaac0c)  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z10.s
    KAI_ASM_INST(0xc1a6aa2c)  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z6.s
    KAI_ASM_INST(0xc1a7ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z7.s
    KAI_ASM_INST(0xc1b7cfcc)  // sclamp { z12.s-z15.s }, z30.s, z23.s
    uzp1 z12.h, z12.h, z13.h
    uzp1 z19.h, z14.h, z15.h
    uzp1 z12.b, z12.b, z19.b
    st1b { z12.b }, p1, [x6]
    b label_44
KAI_ASM_LABEL(label_14)  // Width 2
    ldr x4, [x0, #0x20]
    mov x2, x20
    sub x1, x22, x21
    mov x3, x20
    whilelt p1.b, XZR, x1
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_15
    KAI_ASM_INST(0xa040c4a4)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c0a0)  // ld1w { z0.s-z3.s }, p8.s/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xc0042c80)  // mova za.d[x9, #0], { z4.d-z7.d }
    KAI_ASM_INST(0xc0042c01)  // mova za.d[x9, #1], { z0.d-z3.d }
    b label_16
KAI_ASM_LABEL(label_15)  // Width 2: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_16)  // Width 2: setup done
    cmp x3, #0x10
    ble label_19
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7]
    ld1rqb { z9.b }, p0/Z, [x4]
    add x4, x4, #0x10
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa04084f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b2b0)  // udot za.s[x9, 0], { z20.b-z23.b }, z9.b[0]
    KAI_ASM_INST(0xa04184f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b331)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[0]
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa04084ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b630)  // udot za.s[x9, 0], { z16.b-z19.b }, z9.b[1]
    KAI_ASM_INST(0xa04184f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b6b1)  // udot za.s[x9, 1], { z20.b-z23.b }, z9.b[1]
    KAI_ASM_INST(0xc159bbb0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[2]
    KAI_ASM_INST(0xc159b831)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[2]
    KAI_ASM_INST(0xc159bdb0)  // udot za.s[x9, 0], { z12.b-z15.b }, z9.b[3]
    KAI_ASM_INST(0xc159be31)  // udot za.s[x9, 1], { z16.b-z19.b }, z9.b[3]
    tbnz x23, #31, label_18
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_18)  // Width 2: Multiply loop: unique 3: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_17
KAI_ASM_LABEL(label_19)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    ld1rqb { z9.b }, p0/Z, [x4]
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b030)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[0]
    KAI_ASM_INST(0xc159b331)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[0]
    ble label_20
    KAI_ASM_INST(0xa04084f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b630)  // udot za.s[x9, 0], { z16.b-z19.b }, z9.b[1]
    KAI_ASM_INST(0xc159b731)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[1]
    ble label_20
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159bbb0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[2]
    KAI_ASM_INST(0xc159b831)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[2]
    ble label_20
    KAI_ASM_INST(0xa04084f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159beb0)  // udot za.s[x9, 0], { z20.b-z23.b }, z9.b[3]
    KAI_ASM_INST(0xc159bf31)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[3]
KAI_ASM_LABEL(label_20)  // Width 2: Multiply loop: multiply skip
    tbnz x23, #31, label_21
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_21)  // Width 2: Multiply loop: unique 4: skip row sum
    tbnz x23, #31, label_22
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c841)  // ld1rw { z1.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d11, p0, z11.s
    mov z11.s, z11.s[0]
    neg z1.s, p2/M, z1.s
    mul z11.s, p2/M, z11.s, z1.s
KAI_ASM_LABEL(label_22)  // Width 2: skip row sum fixup
    KAI_ASM_INST(0xc0904960)  // addha za0.s, p2/M, p2/M, z11.s
    add x2, x0, #0x98
    add x1, x0, #0x9c
    KAI_ASM_INST(0xc0904961)  // addha za1.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c842)  // ld1rw { z2.s }, p2/Z, [x2]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0xc0904962)  // addha za2.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c823)  // ld1rw { z3.s }, p2/Z, [x1]
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0904963)  // addha za3.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c869)  // ld1rw { z9.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c846)  // ld1rw { z6.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c83d)  // ld1rw { z29.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc0062c14)  // mova { z20.d-z23.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc1a2ac14)  // sqdmulh { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a2ac0c)  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s
    KAI_ASM_INST(0xc1a3aa34)  // srshl { z20.s-z23.s }, { z20.s-z23.s }, z3.s
    KAI_ASM_INST(0xc1a3aa2c)  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z3.s
    KAI_ASM_INST(0xc1a9ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z9.s
    KAI_ASM_INST(0xc1a9ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z9.s
    KAI_ASM_INST(0xc1bdccd4)  // sclamp { z20.s-z23.s }, z6.s, z29.s
    KAI_ASM_INST(0xc1bdcccc)  // sclamp { z12.s-z15.s }, z6.s, z29.s
    uzp1 z20.h, z20.h, z21.h
    uzp1 z16.h, z22.h, z23.h
    uzp1 z12.h, z12.h, z13.h
    uzp1 z24.h, z14.h, z15.h
    uzp1 z20.b, z20.b, z16.b
    uzp1 z12.b, z12.b, z24.b
    st1b { z20.b }, p2, [x6]
    st1b { z12.b }, p1, [x6, #1, MUL VL]
    b label_44
KAI_ASM_LABEL(label_24)  // Width 3
    ldr x4, [x0, #0x20]
    mov x1, #0x2
    mov x2, x20
    msub x1, x21, x1, x22
    mov x3, x20
    whilelt p1.b, XZR, x1
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_25
    KAI_ASM_INST(0xa040c4b4)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4b0)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c0ac)  // ld1w { z12.s-z15.s }, p8.s/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xc0042e80)  // mova za.d[x9, #0], { z20.d-z23.d }
    KAI_ASM_INST(0xc0042e01)  // mova za.d[x9, #1], { z16.d-z19.d }
    KAI_ASM_INST(0xc0042d82)  // mova za.d[x9, #2], { z12.d-z15.d }
    b label_26
KAI_ASM_LABEL(label_25)  // Width 3: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_26)  // Width 3: setup done
    cmp x3, #0x10
    ble label_29
KAI_ASM_LABEL(label_27)  // Width 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    ld1rqb { z9.b }, p0/Z, [x4]
    add x4, x4, #0x10
    KAI_ASM_INST(0xa04184f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b3b0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[0]
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b231)  // udot za.s[x9, 1], { z16.b-z19.b }, z9.b[0]
    KAI_ASM_INST(0xa04184fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b1b2)  // udot za.s[x9, 2], { z12.b-z15.b }, z9.b[0]
    KAI_ASM_INST(0xa04284f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa04084ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b730)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[1]
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b7b1)  // udot za.s[x9, 1], { z28.b-z31.b }, z9.b[1]
    KAI_ASM_INST(0xa04284fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b632)  // udot za.s[x9, 2], { z16.b-z19.b }, z9.b[1]
    KAI_ASM_INST(0xa04084f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b9b0)  // udot za.s[x9, 0], { z12.b-z15.b }, z9.b[2]
    KAI_ASM_INST(0xa04284f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b831)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[2]
    KAI_ASM_INST(0xc159bbb2)  // udot za.s[x9, 2], { z28.b-z31.b }, z9.b[2]
    KAI_ASM_INST(0xc159beb0)  // udot za.s[x9, 0], { z20.b-z23.b }, z9.b[3]
    KAI_ASM_INST(0xc159bf31)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[3]
    KAI_ASM_INST(0xc159be32)  // udot za.s[x9, 2], { z16.b-z19.b }, z9.b[3]
    tbnz x23, #31, label_28
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_28)  // Width 3: Multiply loop: unique 5: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_27
KAI_ASM_LABEL(label_29)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    ld1rqb { z9.b }, p0/Z, [x4]
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b1b0)  // udot za.s[x9, 0], { z12.b-z15.b }, z9.b[0]
    KAI_ASM_INST(0xc159b031)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[0]
    KAI_ASM_INST(0xc159b3b2)  // udot za.s[x9, 2], { z28.b-z31.b }, z9.b[0]
    ble label_30
    KAI_ASM_INST(0xa04084e5)  // ldnt1b { z4.b-z7.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b4b0)  // udot za.s[x9, 0], { z4.b-z7.b }, z9.b[1]
    KAI_ASM_INST(0xc159b731)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[1]
    KAI_ASM_INST(0xc159b632)  // udot za.s[x9, 2], { z16.b-z19.b }, z9.b[1]
    ble label_30
    KAI_ASM_INST(0xa04084fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159bbb0)  // udot za.s[x9, 0], { z28.b-z31.b }, z9.b[2]
    KAI_ASM_INST(0xc159b831)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[2]
    KAI_ASM_INST(0xc159bb32)  // udot za.s[x9, 2], { z24.b-z27.b }, z9.b[2]
    ble label_30
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa04184f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc159bc30)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[3]
    KAI_ASM_INST(0xc159beb1)  // udot za.s[x9, 1], { z20.b-z23.b }, z9.b[3]
    KAI_ASM_INST(0xc159be32)  // udot za.s[x9, 2], { z16.b-z19.b }, z9.b[3]
KAI_ASM_LABEL(label_30)  // Width 3: Multiply loop: multiply skip
    tbnz x23, #31, label_31
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_31)  // Width 3: Multiply loop: unique 6: skip row sum
    tbnz x23, #31, label_32
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c850)  // ld1rw { z16.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    uaddv d11, p0, z11.s
    mov z11.s, z11.s[0]
    neg z16.s, p2/M, z16.s
    mul z11.s, p2/M, z11.s, z16.s
KAI_ASM_LABEL(label_32)  // Width 3: skip row sum fixup
    KAI_ASM_INST(0xc0904960)  // addha za0.s, p2/M, p2/M, z11.s
    add x2, x0, #0x98
    add x1, x0, #0x9c
    KAI_ASM_INST(0xc0904961)  // addha za1.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c842)  // ld1rw { z2.s }, p2/Z, [x2]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0xc0904962)  // addha za2.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c821)  // ld1rw { z1.s }, p2/Z, [x1]
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0904963)  // addha za3.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c860)  // ld1rw { z0.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c855)  // ld1rw { z21.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c834)  // ld1rw { z20.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc0062c1c)  // mova { z28.d-z31.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c50)  // mova { z16.d-z19.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc1a2ac1c)  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z2.s
    KAI_ASM_INST(0xc1a2ac0c)  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s
    KAI_ASM_INST(0xc1a2ac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a1aa3c)  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z1.s
    KAI_ASM_INST(0xc1a1aa2c)  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z1.s
    KAI_ASM_INST(0xc1a1aa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b4cebc)  // sclamp { z28.s-z31.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4ceac)  // sclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4ceb0)  // sclamp { z16.s-z19.s }, z21.s, z20.s
    uzp1 z28.h, z28.h, z29.h
    uzp1 z21.h, z30.h, z31.h
    uzp1 z12.h, z12.h, z13.h
    uzp1 z20.h, z14.h, z15.h
    uzp1 z16.h, z16.h, z17.h
    uzp1 z17.h, z18.h, z19.h
    uzp1 z28.b, z28.b, z21.b
    uzp1 z12.b, z12.b, z20.b
    uzp1 z16.b, z16.b, z17.b
    st1b { z28.b }, p2, [x6]
    st1b { z12.b }, p2, [x6, #1, MUL VL]
    st1b { z16.b }, p1, [x6, #2, MUL VL]
    b label_44
KAI_ASM_LABEL(label_34)  // Width 4
    ldr x4, [x0, #0x20]
    mov x1, #0x3
    mov x2, x20
    msub x1, x21, x1, x22
    mov x3, x20
    whilelt p1.b, XZR, x1
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_35
    KAI_ASM_INST(0xa040c4a0)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4ac)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c4b0)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c0b4)  // ld1w { z20.s-z23.s }, p8.s/Z, [x5, #0xc, MUL VL]
    KAI_ASM_INST(0xc0042c00)  // mova za.d[x9, #0], { z0.d-z3.d }
    addvl x5, x5, #16
    KAI_ASM_INST(0xc0042d81)  // mova za.d[x9, #1], { z12.d-z15.d }
    KAI_ASM_INST(0xc0042e02)  // mova za.d[x9, #2], { z16.d-z19.d }
    KAI_ASM_INST(0xc0042e83)  // mova za.d[x9, #3], { z20.d-z23.d }
    b label_36
KAI_ASM_LABEL(label_35)  // Width 4: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_36)  // Width 4: setup done
    cmp x3, #0x10
    ble label_39
KAI_ASM_LABEL(label_37)  // Width 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    ld1rqb { z9.b }, p0/Z, [x4]
    add x4, x4, #0x10
    KAI_ASM_INST(0xa04184f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159b030)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b2b1)  // udot za.s[x9, 1], { z20.b-z23.b }, z9.b[0]
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b3b2)  // udot za.s[x9, 2], { z28.b-z31.b }, z9.b[0]
    KAI_ASM_INST(0xa04184fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b333)  // udot za.s[x9, 3], { z24.b-z27.b }, z9.b[0]
    KAI_ASM_INST(0xa04284f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159b430)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b7b1)  // udot za.s[x9, 1], { z28.b-z31.b }, z9.b[1]
    KAI_ASM_INST(0xa04084f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b632)  // udot za.s[x9, 2], { z16.b-z19.b }, z9.b[1]
    KAI_ASM_INST(0xa04184fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b5b3)  // udot za.s[x9, 3], { z12.b-z15.b }, z9.b[1]
    KAI_ASM_INST(0xa04284e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159bab0)  // udot za.s[x9, 0], { z20.b-z23.b }, z9.b[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159bbb1)  // udot za.s[x9, 1], { z28.b-z31.b }, z9.b[2]
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc159b832)  // udot za.s[x9, 2], { z0.b-z3.b }, z9.b[2]
    KAI_ASM_INST(0xa04184f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc159b9b3)  // udot za.s[x9, 3], { z12.b-z15.b }, z9.b[2]
    KAI_ASM_INST(0xa04284e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159bf30)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159beb1)  // udot za.s[x9, 1], { z20.b-z23.b }, z9.b[3]
    KAI_ASM_INST(0xc159bc32)  // udot za.s[x9, 2], { z0.b-z3.b }, z9.b[3]
    KAI_ASM_INST(0xc159be33)  // udot za.s[x9, 3], { z16.b-z19.b }, z9.b[3]
    tbnz x23, #31, label_38
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_38)  // Width 4: Multiply loop: unique 7: skip row sum
    sub x3, x3, #0x10
    cmp x3, #0x10
    bgt label_37
KAI_ASM_LABEL(label_39)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x3
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    ld1rqb { z9.b }, p0/Z, [x4]
    KAI_ASM_INST(0xa04184f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159b030)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b2b1)  // udot za.s[x9, 1], { z20.b-z23.b }, z9.b[0]
    KAI_ASM_INST(0xc159b3b2)  // udot za.s[x9, 2], { z28.b-z31.b }, z9.b[0]
    KAI_ASM_INST(0xc159b233)  // udot za.s[x9, 3], { z16.b-z19.b }, z9.b[0]
    ble label_40
    KAI_ASM_INST(0xa04084e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159b430)  // udot za.s[x9, 0], { z0.b-z3.b }, z9.b[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b731)  // udot za.s[x9, 1], { z24.b-z27.b }, z9.b[1]
    KAI_ASM_INST(0xc159b7b2)  // udot za.s[x9, 2], { z28.b-z31.b }, z9.b[1]
    KAI_ASM_INST(0xc159b633)  // udot za.s[x9, 3], { z16.b-z19.b }, z9.b[1]
    ble label_40
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    subs x3, x3, #0x4
    KAI_ASM_INST(0xa04184e1)  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384ed)  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159bb30)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159b831)  // udot za.s[x9, 1], { z0.b-z3.b }, z9.b[2]
    KAI_ASM_INST(0xc159bab2)  // udot za.s[x9, 2], { z20.b-z23.b }, z9.b[2]
    KAI_ASM_INST(0xc159b9b3)  // udot za.s[x9, 3], { z12.b-z15.b }, z9.b[2]
    ble label_40
    KAI_ASM_INST(0xa04084f9)  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa04184fd)  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284f5)  // ldnt1b { z20.b-z23.b }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384f1)  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc159bf30)  // udot za.s[x9, 0], { z24.b-z27.b }, z9.b[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc159bfb1)  // udot za.s[x9, 1], { z28.b-z31.b }, z9.b[3]
    KAI_ASM_INST(0xc159beb2)  // udot za.s[x9, 2], { z20.b-z23.b }, z9.b[3]
    KAI_ASM_INST(0xc159be33)  // udot za.s[x9, 3], { z16.b-z19.b }, z9.b[3]
KAI_ASM_LABEL(label_40)  // Width 4: Multiply loop: multiply skip
    tbnz x23, #31, label_41
    udot z11.s, z9.b, z8.b
KAI_ASM_LABEL(label_41)  // Width 4: Multiply loop: unique 8: skip row sum
    tbnz x23, #31, label_42
    add x2, x0, #0x90
    mov x1, #0x4
    KAI_ASM_INST(0x8540c850)  // ld1rw { z16.s }, p2/Z, [x2]
    whilelt p0.s, XZR, x1
    orr x23, x23, #0x80000000
    uaddv d11, p0, z11.s
    mov z11.s, z11.s[0]
    neg z16.s, p2/M, z16.s
    mul z11.s, p2/M, z11.s, z16.s
KAI_ASM_LABEL(label_42)  // Width 4: skip row sum fixup
    KAI_ASM_INST(0xc0904960)  // addha za0.s, p2/M, p2/M, z11.s
    add x2, x0, #0x98
    add x1, x0, #0x9c
    KAI_ASM_INST(0xc0904961)  // addha za1.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c84a)  // ld1rw { z10.s }, p2/Z, [x2]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0xc0904962)  // addha za2.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c82c)  // ld1rw { z12.s }, p2/Z, [x1]
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0904963)  // addha za3.s, p2/M, p2/M, z11.s
    KAI_ASM_INST(0x8540c86d)  // ld1rw { z13.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c832)  // ld1rw { z18.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc0062c1c)  // mova { z28.d-z31.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c58)  // mova { z24.d-z27.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc0062c74)  // mova { z20.d-z23.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc1aaac1c)  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s
    KAI_ASM_INST(0xc1aaac00)  // sqdmulh { z0.s-z3.s }, { z0.s-z3.s }, z10.s
    KAI_ASM_INST(0xc1aaac18)  // sqdmulh { z24.s-z27.s }, { z24.s-z27.s }, z10.s
    KAI_ASM_INST(0xc1aaac14)  // sqdmulh { z20.s-z23.s }, { z20.s-z23.s }, z10.s
    KAI_ASM_INST(0xc1acaa3c)  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z12.s
    KAI_ASM_INST(0xc1acaa20)  // srshl { z0.s-z3.s }, { z0.s-z3.s }, z12.s
    KAI_ASM_INST(0xc1acaa38)  // srshl { z24.s-z27.s }, { z24.s-z27.s }, z12.s
    KAI_ASM_INST(0xc1acaa34)  // srshl { z20.s-z23.s }, { z20.s-z23.s }, z12.s
    KAI_ASM_INST(0xc1adab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z13.s
    KAI_ASM_INST(0xc1adab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z13.s
    KAI_ASM_INST(0xc1adab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z13.s
    KAI_ASM_INST(0xc1adab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z13.s
    KAI_ASM_INST(0xc1b2ce3c)  // sclamp { z28.s-z31.s }, z17.s, z18.s
    KAI_ASM_INST(0xc1b2ce20)  // sclamp { z0.s-z3.s }, z17.s, z18.s
    KAI_ASM_INST(0xc1b2ce38)  // sclamp { z24.s-z27.s }, z17.s, z18.s
    KAI_ASM_INST(0xc1b2ce34)  // sclamp { z20.s-z23.s }, z17.s, z18.s
    uzp1 z28.h, z28.h, z29.h
    uzp1 z14.h, z30.h, z31.h
    uzp1 z0.h, z0.h, z1.h
    uzp1 z18.h, z2.h, z3.h
    uzp1 z24.h, z24.h, z25.h
    uzp1 z17.h, z26.h, z27.h
    uzp1 z20.h, z20.h, z21.h
    uzp1 z16.h, z22.h, z23.h
    uzp1 z28.b, z28.b, z14.b
    uzp1 z0.b, z0.b, z18.b
    uzp1 z24.b, z24.b, z17.b
    uzp1 z20.b, z20.b, z16.b
    st1b { z28.b }, p2, [x6]
    st1b { z0.b }, p2, [x6, #1, MUL VL]
    st1b { z24.b }, p2, [x6, #2, MUL VL]
    st1b { z20.b }, p1, [x6, #3, MUL VL]
    addvl x6, x6, #4
    subs x8, x8, #0x4
    sub x22, x22, x21, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_44)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_u8qa_dot_16VL)

    KAI_ASM_END
