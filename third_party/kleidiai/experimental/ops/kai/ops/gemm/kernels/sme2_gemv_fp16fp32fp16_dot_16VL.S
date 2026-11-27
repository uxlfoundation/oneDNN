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

    KAI_ASM_CODE(sme2_gemv_fp16fp32fp16_dot_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_fp16fp32fp16_dot_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_fp16fp32fp16_dot_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_fp16fp32fp16_dot_16VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x2, #0xfc00
    ldr x23, [x0, #0x8]
    cntw x22, ALL, MUL #4
    mov x5, #0x7c00
    ldrh w1, [x0, #0xa0]
    mov x21, #0
    mov x9, #0
    ldrh w4, [x0, #0xa4]
    ptrue p1.b
    mov x3, #0x1
    add x20, x23, x22
    ldr x8, [x0, #0x10]
    sub x20, x20, #0x1
    cmp x1, x2
    ldr x7, [x0, #0x30]
    udiv x20, x20, x22
    cset x1, NE
    ldr x6, [x0, #0x50]
    add x2, x20, #0x3
    cmp x4, x5
    and x2, x2, #0xfffffffffffffffc
    orr x21, x21, x1, LSL #1
    mul x2, x2, x22
    cset x1, NE
    mul x2, x2, x8
    orr x21, x21, x1, LSL #1
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    lsl x2, x2, #0x1
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
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x20, #0x4
    bge label_28
    cmp x20, #0x2
    bgt label_20
    beq label_12
    ldr x4, [x0, #0x20]
    lsl x2, x8, #0x1
    mov x1, x23
    mov x3, x8
    KAI_ASM_INST(0x256147f0)  // whilelt p8.h, XZR, x1, VLx2
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_5
    KAI_ASM_INST(0xa04020a6)  // ld1h { z6.h-z7.h }, p8.h/Z, [x5]
    fcvt z0.s, p1/m, z6.h
    KAI_ASM_INST(0x6489a4c2)  // fcvtlt z2.s, p1/m, z6.h
    fcvt z28.s, p1/m, z7.h
    KAI_ASM_INST(0x6489a4f2)  // fcvtlt z18.s, p1/m, z7.h
    zip1 z20.s, z0.s, z2.s
    zip2 z21.s, z0.s, z2.s
    zip1 z22.s, z28.s, z18.s
    zip2 z23.s, z28.s, z18.s
    KAI_ASM_INST(0xc0042e80)  // mova za.d[x9, #0], { z20.d-z23.d }
    b label_6
KAI_ASM_LABEL(label_5)  // Width 1: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_6)  // Width 1: setup done
    cmp x3, #0x8
    ble label_8
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    ld1rqh { z8.h }, p0/Z, [x4]
    sub x3, x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    cmp x3, #0x8
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc158b188)  // fdot za.s[x9, 0], { z12.h-z15.h }, z8.h[0]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc158b688)  // fdot za.s[x9, 0], { z20.h-z23.h }, z8.h[1]
    KAI_ASM_INST(0xc158b808)  // fdot za.s[x9, 0], { z0.h-z3.h }, z8.h[2]
    KAI_ASM_INST(0xc158bf88)  // fdot za.s[x9, 0], { z28.h-z31.h }, z8.h[3]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b388)  // fdot za.s[x9, 0], { z28.h-z31.h }, z0.h[0]
    ble label_9
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b788)  // fdot za.s[x9, 0], { z28.h-z31.h }, z0.h[1]
    ble label_9
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b988)  // fdot za.s[x9, 0], { z12.h-z15.h }, z0.h[2]
    ble label_9
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc150bf08)  // fdot za.s[x9, 0], { z24.h-z27.h }, z0.h[3]
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_10
    KAI_ASM_INST(0xc0062c10)  // mova { z16.d-z19.d }, za.d[x9, #0]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0x84c0a441)  // ld1rh { z1.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a438)  // ld1rh { z24.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc120e216)  // fcvt z22.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e257)  // fcvt z23.h, { z18.s-z19.s }
    KAI_ASM_INST(0xc178c036)  // fclamp { z22.h-z23.h }, z1.h, z24.h
    KAI_ASM_INST(0xa06020d6)  // st1h { z22.h-z23.h }, p8, [x6]
    b label_11
KAI_ASM_LABEL(label_10)  // Width 1: No activation
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc120e01e)  // fcvt z30.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e05f)  // fcvt z31.h, { z2.s-z3.s }
    KAI_ASM_INST(0xa06020de)  // st1h { z30.h-z31.h }, p8, [x6]
KAI_ASM_LABEL(label_11)  // Width 1: Output done
    b label_36
KAI_ASM_LABEL(label_12)  // Width 2
    ldr x4, [x0, #0x20]
    lsl x2, x8, #0x1
    sub x1, x23, x22
    mov x3, x8
    KAI_ASM_INST(0x256147f0)  // whilelt p8.h, XZR, x1, VLx2
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_13
    KAI_ASM_INST(0xa04120a8)  // ld1h { z8.h-z9.h }, p8.h/Z, [x5, #0x2, MUL VL]
    ld1h { z12.s }, p1/Z, [x5]
    ld1h { z13.s }, p1/Z, [x5, #1, MUL VL]
    fcvt z28.s, p1/m, z8.h
    KAI_ASM_INST(0x6489a503)  // fcvtlt z3.s, p1/m, z8.h
    ld1h { z14.s }, p1/Z, [x5, #2, MUL VL]
    fcvt z25.s, p1/m, z9.h
    KAI_ASM_INST(0x6489a53d)  // fcvtlt z29.s, p1/m, z9.h
    ld1h { z15.s }, p1/Z, [x5, #3, MUL VL]
    fcvt z12.s, p1/m, z12.h
    fcvt z13.s, p1/m, z13.h
    fcvt z14.s, p1/m, z14.h
    fcvt z15.s, p1/m, z15.h
    zip1 z4.s, z28.s, z3.s
    zip2 z5.s, z28.s, z3.s
    zip1 z6.s, z25.s, z29.s
    zip2 z7.s, z25.s, z29.s
    KAI_ASM_INST(0xc0042d80)  // mova za.d[x9, #0], { z12.d-z15.d }
    KAI_ASM_INST(0xc0042c81)  // mova za.d[x9, #1], { z4.d-z7.d }
    b label_14
KAI_ASM_LABEL(label_13)  // Width 2: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_14)  // Width 2: setup done
    cmp x3, #0x8
    ble label_16
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z2.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc152b088)  // fdot za.s[x9, 0], { z4.h-z7.h }, z2.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc152b389)  // fdot za.s[x9, 1], { z28.h-z31.h }, z2.h[0]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc152b508)  // fdot za.s[x9, 0], { z8.h-z11.h }, z2.h[1]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc152b589)  // fdot za.s[x9, 1], { z12.h-z15.h }, z2.h[1]
    KAI_ASM_INST(0xc152b888)  // fdot za.s[x9, 0], { z4.h-z7.h }, z2.h[2]
    KAI_ASM_INST(0xc152ba89)  // fdot za.s[x9, 1], { z20.h-z23.h }, z2.h[2]
    KAI_ASM_INST(0xc152bf08)  // fdot za.s[x9, 0], { z24.h-z27.h }, z2.h[3]
    KAI_ASM_INST(0xc152bd09)  // fdot za.s[x9, 1], { z8.h-z11.h }, z2.h[3]
    bgt label_15
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b308)  // fdot za.s[x9, 0], { z24.h-z27.h }, z0.h[0]
    KAI_ASM_INST(0xc150b109)  // fdot za.s[x9, 1], { z8.h-z11.h }, z0.h[0]
    ble label_17
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b508)  // fdot za.s[x9, 0], { z8.h-z11.h }, z0.h[1]
    KAI_ASM_INST(0xc150b789)  // fdot za.s[x9, 1], { z28.h-z31.h }, z0.h[1]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150ba08)  // fdot za.s[x9, 0], { z16.h-z19.h }, z0.h[2]
    KAI_ASM_INST(0xc150b989)  // fdot za.s[x9, 1], { z12.h-z15.h }, z0.h[2]
    ble label_17
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc150bd08)  // fdot za.s[x9, 0], { z8.h-z11.h }, z0.h[3]
    KAI_ASM_INST(0xc150be89)  // fdot za.s[x9, 1], { z20.h-z23.h }, z0.h[3]
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_18
    KAI_ASM_INST(0xc0062c10)  // mova { z16.d-z19.d }, za.d[x9, #0]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c28)  // mova { z8.d-z11.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a445)  // ld1rh { z5.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a42c)  // ld1rh { z12.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc120e214)  // fcvt z20.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e255)  // fcvt z21.h, { z18.s-z19.s }
    KAI_ASM_INST(0xc120e112)  // fcvt z18.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e153)  // fcvt z19.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc16cc0b4)  // fclamp { z20.h-z21.h }, z5.h, z12.h
    KAI_ASM_INST(0xc16cc0b2)  // fclamp { z18.h-z19.h }, z5.h, z12.h
    KAI_ASM_INST(0xa06024d4)  // st1h { z20.h-z21.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa06120d2)  // st1h { z18.h-z19.h }, p8, [x6, #0x2, MUL VL]
    b label_19
KAI_ASM_LABEL(label_18)  // Width 2: No activation
    KAI_ASM_INST(0xc0062c10)  // mova { z16.d-z19.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c3c)  // mova { z28.d-z31.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc120e21a)  // fcvt z26.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e25b)  // fcvt z27.h, { z18.s-z19.s }
    KAI_ASM_INST(0xa06024da)  // st1h { z26.h-z27.h }, pn9.b, [x6]
    KAI_ASM_INST(0xc120e38e)  // fcvt z14.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cf)  // fcvt z15.h, { z30.s-z31.s }
    KAI_ASM_INST(0xa06120ce)  // st1h { z14.h-z15.h }, p8, [x6, #0x2, MUL VL]
KAI_ASM_LABEL(label_19)  // Width 2: Output done
    b label_36
KAI_ASM_LABEL(label_20)  // Width 3
    ldr x4, [x0, #0x20]
    mov x1, #0x2
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x256147f0)  // whilelt p8.h, XZR, x1, VLx2
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_21
    KAI_ASM_INST(0xa04220a2)  // ld1h { z2.h-z3.h }, p8.h/Z, [x5, #0x4, MUL VL]
    ld1h { z12.s }, p1/Z, [x5]
    ld1h { z13.s }, p1/Z, [x5, #1, MUL VL]
    fcvt z10.s, p1/m, z2.h
    KAI_ASM_INST(0x6489a450)  // fcvtlt z16.s, p1/m, z2.h
    ld1h { z14.s }, p1/Z, [x5, #2, MUL VL]
    fcvt z20.s, p1/m, z3.h
    KAI_ASM_INST(0x6489a47b)  // fcvtlt z27.s, p1/m, z3.h
    ld1h { z15.s }, p1/Z, [x5, #3, MUL VL]
    ld1h { z4.s }, p1/Z, [x5, #4, MUL VL]
    fcvt z12.s, p1/m, z12.h
    fcvt z13.s, p1/m, z13.h
    ld1h { z5.s }, p1/Z, [x5, #5, MUL VL]
    fcvt z14.s, p1/m, z14.h
    ld1h { z6.s }, p1/Z, [x5, #6, MUL VL]
    fcvt z15.s, p1/m, z15.h
    zip1 z24.s, z10.s, z16.s
    ld1h { z7.s }, p1/Z, [x5, #7, MUL VL]
    fcvt z4.s, p1/m, z4.h
    zip2 z25.s, z10.s, z16.s
    fcvt z5.s, p1/m, z5.h
    zip1 z26.s, z20.s, z27.s
    zip2 z27.s, z20.s, z27.s
    fcvt z6.s, p1/m, z6.h
    fcvt z7.s, p1/m, z7.h
    KAI_ASM_INST(0xc0042d80)  // mova za.d[x9, #0], { z12.d-z15.d }
    KAI_ASM_INST(0xc0042f02)  // mova za.d[x9, #2], { z24.d-z27.d }
    KAI_ASM_INST(0xc0042c81)  // mova za.d[x9, #1], { z4.d-z7.d }
    b label_22
KAI_ASM_LABEL(label_21)  // Width 3: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_22)  // Width 3: setup done
    cmp x3, #0x8
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z1.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc151b388)  // fdot za.s[x9, 0], { z28.h-z31.h }, z1.h[0]
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc151b109)  // fdot za.s[x9, 1], { z8.h-z11.h }, z1.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc151b08a)  // fdot za.s[x9, 2], { z4.h-z7.h }, z1.h[0]
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc151b708)  // fdot za.s[x9, 0], { z24.h-z27.h }, z1.h[1]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc151b589)  // fdot za.s[x9, 1], { z12.h-z15.h }, z1.h[1]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc151b48a)  // fdot za.s[x9, 2], { z4.h-z7.h }, z1.h[1]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc151bb88)  // fdot za.s[x9, 0], { z28.h-z31.h }, z1.h[2]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc151ba89)  // fdot za.s[x9, 1], { z20.h-z23.h }, z1.h[2]
    KAI_ASM_INST(0xc151b90a)  // fdot za.s[x9, 2], { z8.h-z11.h }, z1.h[2]
    KAI_ASM_INST(0xc151bd88)  // fdot za.s[x9, 0], { z12.h-z15.h }, z1.h[3]
    KAI_ASM_INST(0xc151bc89)  // fdot za.s[x9, 1], { z4.h-z7.h }, z1.h[3]
    KAI_ASM_INST(0xc151bf0a)  // fdot za.s[x9, 2], { z24.h-z27.h }, z1.h[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b188)  // fdot za.s[x9, 0], { z12.h-z15.h }, z0.h[0]
    KAI_ASM_INST(0xc150b389)  // fdot za.s[x9, 1], { z28.h-z31.h }, z0.h[0]
    KAI_ASM_INST(0xc150b10a)  // fdot za.s[x9, 2], { z8.h-z11.h }, z0.h[0]
    ble label_25
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b608)  // fdot za.s[x9, 0], { z16.h-z19.h }, z0.h[1]
    KAI_ASM_INST(0xc150b509)  // fdot za.s[x9, 1], { z8.h-z11.h }, z0.h[1]
    KAI_ASM_INST(0xc150b68a)  // fdot za.s[x9, 2], { z20.h-z23.h }, z0.h[1]
    ble label_25
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150bb88)  // fdot za.s[x9, 0], { z28.h-z31.h }, z0.h[2]
    KAI_ASM_INST(0xc150ba09)  // fdot za.s[x9, 1], { z16.h-z19.h }, z0.h[2]
    KAI_ASM_INST(0xc150ba8a)  // fdot za.s[x9, 2], { z20.h-z23.h }, z0.h[2]
    ble label_25
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc150bd08)  // fdot za.s[x9, 0], { z8.h-z11.h }, z0.h[3]
    KAI_ASM_INST(0xc150bf89)  // fdot za.s[x9, 1], { z28.h-z31.h }, z0.h[3]
    KAI_ASM_INST(0xc150be0a)  // fdot za.s[x9, 2], { z16.h-z19.h }, z0.h[3]
KAI_ASM_LABEL(label_25)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_26
    KAI_ASM_INST(0xc0062c1c)  // mova { z28.d-z31.d }, za.d[x9, #0]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a453)  // ld1rh { z19.h }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c40)  // mova { z0.d-z3.d }, za.d[x9, #2]
    KAI_ASM_INST(0x84c0a432)  // ld1rh { z18.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc120e388)  // fcvt z8.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3c9)  // fcvt z9.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc120e19e)  // fcvt z30.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1df)  // fcvt z31.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc172c268)  // fclamp { z8.h-z9.h }, z19.h, z18.h
    KAI_ASM_INST(0xc120e00e)  // fcvt z14.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e04f)  // fcvt z15.h, { z2.s-z3.s }
    KAI_ASM_INST(0xc172c27e)  // fclamp { z30.h-z31.h }, z19.h, z18.h
    KAI_ASM_INST(0xc172c26e)  // fclamp { z14.h-z15.h }, z19.h, z18.h
    KAI_ASM_INST(0xa06024c8)  // st1h { z8.h-z9.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa06124de)  // st1h { z30.h-z31.h }, pn9.b, [x6, #0x2, MUL VL]
    KAI_ASM_INST(0xa06220ce)  // st1h { z14.h-z15.h }, p8, [x6, #0x4, MUL VL]
    b label_27
KAI_ASM_LABEL(label_26)  // Width 3: No activation
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c24)  // mova { z4.d-z7.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c4c)  // mova { z12.d-z15.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc120e011)  // fcvt z17.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e059)  // fcvt z25.h, { z2.s-z3.s }
    KAI_ASM_INST(0xa16024d1)  // st1h { z17.h, z25.h }, pn9.b, [x6]
    KAI_ASM_INST(0xc120e091)  // fcvt z17.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d9)  // fcvt z25.h, { z6.s-z7.s }
    KAI_ASM_INST(0xa16124d1)  // st1h { z17.h, z25.h }, pn9.b, [x6, #0x2, MUL VL]
    KAI_ASM_INST(0xc120e191)  // fcvt z17.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1d9)  // fcvt z25.h, { z14.s-z15.s }
    KAI_ASM_INST(0xa16220d1)  // st1h { z17.h, z25.h }, p8, [x6, #0x4, MUL VL]
KAI_ASM_LABEL(label_27)  // Width 3: Output done
    b label_36
KAI_ASM_LABEL(label_28)  // Width 4
    ldr x4, [x0, #0x20]
    mov x1, #0x3
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x256147f0)  // whilelt p8.h, XZR, x1, VLx2
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_29
    KAI_ASM_INST(0xa04320b2)  // ld1h { z18.h-z19.h }, p8.h/Z, [x5, #0x6, MUL VL]
    addvl x1, x5, #4
    ld1h { z16.s }, p1/Z, [x5]
    ld1h { z17.s }, p1/Z, [x5, #1, MUL VL]
    fcvt z21.s, p1/m, z18.h
    KAI_ASM_INST(0x6489a654)  // fcvtlt z20.s, p1/m, z18.h
    ld1h { z18.s }, p1/Z, [x5, #2, MUL VL]
    fcvt z23.s, p1/m, z19.h
    KAI_ASM_INST(0x6489a676)  // fcvtlt z22.s, p1/m, z19.h
    ld1h { z19.s }, p1/Z, [x5, #3, MUL VL]
    ld1h { z12.s }, p1/Z, [x5, #4, MUL VL]
    fcvt z16.s, p1/m, z16.h
    fcvt z17.s, p1/m, z17.h
    ld1h { z13.s }, p1/Z, [x5, #5, MUL VL]
    fcvt z18.s, p1/m, z18.h
    ld1h { z14.s }, p1/Z, [x5, #6, MUL VL]
    fcvt z19.s, p1/m, z19.h
    zip1 z28.s, z21.s, z20.s
    ld1h { z15.s }, p1/Z, [x5, #7, MUL VL]
    fcvt z12.s, p1/m, z12.h
    zip2 z29.s, z21.s, z20.s
    addvl x5, x5, #8
    ld1h { z20.s }, p1/Z, [x1]
    fcvt z13.s, p1/m, z13.h
    zip1 z30.s, z23.s, z22.s
    ld1h { z21.s }, p1/Z, [x1, #1, MUL VL]
    fcvt z14.s, p1/m, z14.h
    zip2 z31.s, z23.s, z22.s
    ld1h { z22.s }, p1/Z, [x1, #2, MUL VL]
    fcvt z15.s, p1/m, z15.h
    KAI_ASM_INST(0xc0042e00)  // mova za.d[x9, #0], { z16.d-z19.d }
    ld1h { z23.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z20.s, p1/m, z20.h
    fcvt z21.s, p1/m, z21.h
    fcvt z22.s, p1/m, z22.h
    KAI_ASM_INST(0xc0042f83)  // mova za.d[x9, #3], { z28.d-z31.d }
    fcvt z23.s, p1/m, z23.h
    KAI_ASM_INST(0xc0042d81)  // mova za.d[x9, #1], { z12.d-z15.d }
    KAI_ASM_INST(0xc0042e82)  // mova za.d[x9, #2], { z20.d-z23.d }
    b label_30
KAI_ASM_LABEL(label_29)  // Width 4: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_30)  // Width 4: setup done
    cmp x3, #0x8
    ble label_32
KAI_ASM_LABEL(label_31)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z3.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b188)  // fdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b389)  // fdot za.s[x9, 1], { z28.h-z31.h }, z3.h[0]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b28a)  // fdot za.s[x9, 2], { z20.h-z23.h }, z3.h[0]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b10b)  // fdot za.s[x9, 3], { z8.h-z11.h }, z3.h[0]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b488)  // fdot za.s[x9, 0], { z4.h-z7.h }, z3.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b689)  // fdot za.s[x9, 1], { z20.h-z23.h }, z3.h[1]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b50a)  // fdot za.s[x9, 2], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b70b)  // fdot za.s[x9, 3], { z24.h-z27.h }, z3.h[1]
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153ba08)  // fdot za.s[x9, 0], { z16.h-z19.h }, z3.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b909)  // fdot za.s[x9, 1], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b88a)  // fdot za.s[x9, 2], { z4.h-z7.h }, z3.h[2]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b98b)  // fdot za.s[x9, 3], { z12.h-z15.h }, z3.h[2]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bd08)  // fdot za.s[x9, 0], { z8.h-z11.h }, z3.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bf09)  // fdot za.s[x9, 1], { z24.h-z27.h }, z3.h[3]
    KAI_ASM_INST(0xc153be8a)  // fdot za.s[x9, 2], { z20.h-z23.h }, z3.h[3]
    KAI_ASM_INST(0xc153be0b)  // fdot za.s[x9, 3], { z16.h-z19.h }, z3.h[3]
    bgt label_31
KAI_ASM_LABEL(label_32)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z0.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc150b088)  // fdot za.s[x9, 0], { z4.h-z7.h }, z0.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b189)  // fdot za.s[x9, 1], { z12.h-z15.h }, z0.h[0]
    KAI_ASM_INST(0xc150b20a)  // fdot za.s[x9, 2], { z16.h-z19.h }, z0.h[0]
    KAI_ASM_INST(0xc150b30b)  // fdot za.s[x9, 3], { z24.h-z27.h }, z0.h[0]
    ble label_33
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc150b508)  // fdot za.s[x9, 0], { z8.h-z11.h }, z0.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b689)  // fdot za.s[x9, 1], { z20.h-z23.h }, z0.h[1]
    KAI_ASM_INST(0xc150b70a)  // fdot za.s[x9, 2], { z24.h-z27.h }, z0.h[1]
    KAI_ASM_INST(0xc150b60b)  // fdot za.s[x9, 3], { z16.h-z19.h }, z0.h[1]
    ble label_33
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc150b888)  // fdot za.s[x9, 0], { z4.h-z7.h }, z0.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150b909)  // fdot za.s[x9, 1], { z8.h-z11.h }, z0.h[2]
    KAI_ASM_INST(0xc150b98a)  // fdot za.s[x9, 2], { z12.h-z15.h }, z0.h[2]
    KAI_ASM_INST(0xc150ba0b)  // fdot za.s[x9, 3], { z16.h-z19.h }, z0.h[2]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc150bd88)  // fdot za.s[x9, 0], { z12.h-z15.h }, z0.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc150bf89)  // fdot za.s[x9, 1], { z28.h-z31.h }, z0.h[3]
    KAI_ASM_INST(0xc150be0a)  // fdot za.s[x9, 2], { z16.h-z19.h }, z0.h[3]
    KAI_ASM_INST(0xc150bd0b)  // fdot za.s[x9, 3], { z8.h-z11.h }, z0.h[3]
KAI_ASM_LABEL(label_33)  // Width 4: Multiply loop: multiply skip
    tbz x21, #1, label_34
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a451)  // ld1rh { z17.h }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c58)  // mova { z24.d-z27.d }, za.d[x9, #2]
    KAI_ASM_INST(0x84c0a430)  // ld1rh { z16.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc0062c64)  // mova { z4.d-z7.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc120e18a)  // fcvt z10.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1cb)  // fcvt z11.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc120e00c)  // fcvt z12.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e04d)  // fcvt z13.h, { z2.s-z3.s }
    KAI_ASM_INST(0xc170c22a)  // fclamp { z10.h-z11.h }, z17.h, z16.h
    KAI_ASM_INST(0xc120e30e)  // fcvt z14.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e34f)  // fcvt z15.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc170c22c)  // fclamp { z12.h-z13.h }, z17.h, z16.h
    KAI_ASM_INST(0xc120e094)  // fcvt z20.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d5)  // fcvt z21.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc170c22e)  // fclamp { z14.h-z15.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170c234)  // fclamp { z20.h-z21.h }, z17.h, z16.h
    KAI_ASM_INST(0xa06024ca)  // st1h { z10.h-z11.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa06124cc)  // st1h { z12.h-z13.h }, pn9.b, [x6, #0x2, MUL VL]
    KAI_ASM_INST(0xa06224ce)  // st1h { z14.h-z15.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa06320d4)  // st1h { z20.h-z21.h }, p8, [x6, #0x6, MUL VL]
    addvl x6, x6, #8
    b label_35
KAI_ASM_LABEL(label_34)  // Width 4: No activation
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c3c)  // mova { z28.d-z31.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c48)  // mova { z8.d-z11.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc0062c60)  // mova { z0.d-z3.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc120e187)  // fcvt z7.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1cf)  // fcvt z15.h, { z14.s-z15.s }
    KAI_ASM_INST(0xa16024c7)  // st1h { z7.h, z15.h }, pn9.b, [x6]
    KAI_ASM_INST(0xc120e387)  // fcvt z7.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cf)  // fcvt z15.h, { z30.s-z31.s }
    KAI_ASM_INST(0xa16124c7)  // st1h { z7.h, z15.h }, pn9.b, [x6, #0x2, MUL VL]
    KAI_ASM_INST(0xc120e116)  // fcvt z22.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e15e)  // fcvt z30.h, { z10.s-z11.s }
    KAI_ASM_INST(0xa16224d6)  // st1h { z22.h, z30.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xc120e013)  // fcvt z19.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e05b)  // fcvt z27.h, { z2.s-z3.s }
    KAI_ASM_INST(0xa16320d3)  // st1h { z19.h, z27.h }, p8, [x6, #0x6, MUL VL]
    addvl x6, x6, #8
KAI_ASM_LABEL(label_35)  // Width 4: Output done
    subs x20, x20, #0x4
    sub x23, x23, x22, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_36)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_fp16fp32fp16_dot_16VL)

    KAI_ASM_END
