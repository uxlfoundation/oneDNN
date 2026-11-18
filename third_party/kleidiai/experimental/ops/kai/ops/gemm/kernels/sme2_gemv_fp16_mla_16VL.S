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

    KAI_ASM_CODE(sme2_gemv_fp16_mla_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_fp16_mla_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_fp16_mla_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_fp16_mla_16VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x2, #0xfc00
    ldr x23, [x0, #0x8]
    cnth x22, ALL, MUL #4
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
    KAI_ASM_INST(0x256167f0)  // whilelt p8.h, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_5
    KAI_ASM_INST(0xa040a0b4)  // ld1h { z20.h-z23.h }, p8.h/Z, [x5]
    KAI_ASM_INST(0xc0042e80)  // mova za.d[x9, #0], { z20.d-z23.d }
    b label_6
KAI_ASM_LABEL(label_5)  // Width 1: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_6)  // Width 1: setup done
    cmp x3, #0x8
    ble label_8
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    ld1rqh { z15.h }, p0/Z, [x4]
    sub x3, x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    cmp x3, #0x8
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc11fb300)  // fmla za.h[x9, 0], { z24.h-z27.h }, z15.h[0]
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc11fb088)  // fmla za.h[x9, 0], { z4.h-z7.h }, z15.h[1]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc11fb400)  // fmla za.h[x9, 0], { z0.h-z3.h }, z15.h[2]
    KAI_ASM_INST(0xc11fb688)  // fmla za.h[x9, 0], { z20.h-z23.h }, z15.h[3]
    KAI_ASM_INST(0xc11fba00)  // fmla za.h[x9, 0], { z16.h-z19.h }, z15.h[4]
    KAI_ASM_INST(0xc11fb908)  // fmla za.h[x9, 0], { z8.h-z11.h }, z15.h[5]
    KAI_ASM_INST(0xc11fbf00)  // fmla za.h[x9, 0], { z24.h-z27.h }, z15.h[6]
    KAI_ASM_INST(0xc11fbc88)  // fmla za.h[x9, 0], { z4.h-z7.h }, z15.h[7]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b180)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[0]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b208)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[1]
    ble label_9
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b400)  // fmla za.h[x9, 0], { z0.h-z3.h }, z7.h[2]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b608)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[3]
    ble label_9
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba80)  // fmla za.h[x9, 0], { z20.h-z23.h }, z7.h[4]
    ble label_9
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bb88)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[5]
    ble label_9
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bc00)  // fmla za.h[x9, 0], { z0.h-z3.h }, z7.h[6]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117be08)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[7]
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_10
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0x84c0a456)  // ld1rh { z22.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a427)  // ld1rh { z7.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc167cac0)  // fclamp { z0.h-z3.h }, z22.h, z7.h
    KAI_ASM_INST(0xa060a0c0)  // st1h { z0.h-z3.h }, p8, [x6]
    b label_11
KAI_ASM_LABEL(label_10)  // Width 1: No activation
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0xa060a0c0)  // st1h { z0.h-z3.h }, p8, [x6]
KAI_ASM_LABEL(label_11)  // Width 1: Output done
    b label_36
KAI_ASM_LABEL(label_12)  // Width 2
    ldr x4, [x0, #0x20]
    lsl x2, x8, #0x1
    sub x1, x23, x22
    mov x3, x8
    KAI_ASM_INST(0x256167f0)  // whilelt p8.h, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_13
    KAI_ASM_INST(0xa040a4a0)  // ld1h { z0.h-z3.h }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041a0a4)  // ld1h { z4.h-z7.h }, p8.h/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xc0042c00)  // mova za.d[x9, #0], { z0.d-z3.d }
    KAI_ASM_INST(0xc0042c81)  // mova za.d[x9, #1], { z4.d-z7.d }
    b label_14
KAI_ASM_LABEL(label_13)  // Width 2: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_14)  // Width 2: setup done
    cmp x3, #0x8
    ble label_16
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z5.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b000)  // fmla za.h[x9, 0], { z0.h-z3.h }, z5.h[0]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b381)  // fmla za.h[x9, 1], { z28.h-z31.h }, z5.h[0]
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b208)  // fmla za.h[x9, 0], { z16.h-z19.h }, z5.h[1]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b009)  // fmla za.h[x9, 1], { z0.h-z3.h }, z5.h[1]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b680)  // fmla za.h[x9, 0], { z20.h-z23.h }, z5.h[2]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b501)  // fmla za.h[x9, 1], { z8.h-z11.h }, z5.h[2]
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b788)  // fmla za.h[x9, 0], { z28.h-z31.h }, z5.h[3]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b589)  // fmla za.h[x9, 1], { z12.h-z15.h }, z5.h[3]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115ba00)  // fmla za.h[x9, 0], { z16.h-z19.h }, z5.h[4]
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115bb01)  // fmla za.h[x9, 1], { z24.h-z27.h }, z5.h[4]
    KAI_ASM_INST(0xc115b808)  // fmla za.h[x9, 0], { z0.h-z3.h }, z5.h[5]
    KAI_ASM_INST(0xc115ba89)  // fmla za.h[x9, 1], { z20.h-z23.h }, z5.h[5]
    KAI_ASM_INST(0xc115bd00)  // fmla za.h[x9, 0], { z8.h-z11.h }, z5.h[6]
    KAI_ASM_INST(0xc115bf81)  // fmla za.h[x9, 1], { z28.h-z31.h }, z5.h[6]
    KAI_ASM_INST(0xc115bd88)  // fmla za.h[x9, 0], { z12.h-z15.h }, z5.h[7]
    KAI_ASM_INST(0xc115be09)  // fmla za.h[x9, 1], { z16.h-z19.h }, z5.h[7]
    bgt label_15
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b000)  // fmla za.h[x9, 0], { z0.h-z3.h }, z7.h[0]
    KAI_ASM_INST(0xc117b301)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[0]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b208)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[1]
    KAI_ASM_INST(0xc117b109)  // fmla za.h[x9, 1], { z8.h-z11.h }, z7.h[1]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b600)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b581)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[2]
    ble label_17
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b708)  // fmla za.h[x9, 0], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b609)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[3]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba00)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[4]
    KAI_ASM_INST(0xc117b981)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[4]
    ble label_17
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b808)  // fmla za.h[x9, 0], { z0.h-z3.h }, z7.h[5]
    KAI_ASM_INST(0xc117b989)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[5]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117be00)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[6]
    KAI_ASM_INST(0xc117bd81)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[6]
    ble label_17
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117bf08)  // fmla za.h[x9, 0], { z24.h-z27.h }, z7.h[7]
    KAI_ASM_INST(0xc117bc09)  // fmla za.h[x9, 1], { z0.h-z3.h }, z7.h[7]
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_18
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c10)  // mova { z16.d-z19.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c24)  // mova { z4.d-z7.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a44f)  // ld1rh { z15.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a43b)  // ld1rh { z27.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc17bc9f0)  // fclamp { z16.h-z19.h }, z15.h, z27.h
    KAI_ASM_INST(0xc17bc9e4)  // fclamp { z4.h-z7.h }, z15.h, z27.h
    KAI_ASM_INST(0xa060a4d0)  // st1h { z16.h-z19.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a0c4)  // st1h { z4.h-z7.h }, p8, [x6, #0x4, MUL VL]
    b label_19
KAI_ASM_LABEL(label_18)  // Width 2: No activation
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c34)  // mova { z20.d-z23.d }, za.d[x9, #1]
    KAI_ASM_INST(0xa060a4cc)  // st1h { z12.h-z15.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a0d4)  // st1h { z20.h-z23.h }, p8, [x6, #0x4, MUL VL]
KAI_ASM_LABEL(label_19)  // Width 2: Output done
    b label_36
KAI_ASM_LABEL(label_20)  // Width 3
    ldr x4, [x0, #0x20]
    mov x1, #0x2
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x256167f0)  // whilelt p8.h, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_21
    KAI_ASM_INST(0xa040a4a4)  // ld1h { z4.h-z7.h }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041a4a8)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a0ac)  // ld1h { z12.h-z15.h }, p8.h/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xc0042c80)  // mova za.d[x9, #0], { z4.d-z7.d }
    KAI_ASM_INST(0xc0042d01)  // mova za.d[x9, #1], { z8.d-z11.d }
    KAI_ASM_INST(0xc0042d82)  // mova za.d[x9, #2], { z12.d-z15.d }
    b label_22
KAI_ASM_LABEL(label_21)  // Width 3: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_22)  // Width 3: setup done
    cmp x3, #0x8
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z7.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b280)  // fmla za.h[x9, 0], { z20.h-z23.h }, z7.h[0]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b201)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b102)  // fmla za.h[x9, 2], { z8.h-z11.h }, z7.h[0]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b388)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[1]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b189)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[1]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b10a)  // fmla za.h[x9, 2], { z8.h-z11.h }, z7.h[1]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b700)  // fmla za.h[x9, 0], { z24.h-z27.h }, z7.h[2]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b781)  // fmla za.h[x9, 1], { z28.h-z31.h }, z7.h[2]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b582)  // fmla za.h[x9, 2], { z12.h-z15.h }, z7.h[2]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b608)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[3]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b409)  // fmla za.h[x9, 1], { z0.h-z3.h }, z7.h[3]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b50a)  // fmla za.h[x9, 2], { z8.h-z11.h }, z7.h[3]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117bb80)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[4]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117bb01)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[4]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b982)  // fmla za.h[x9, 2], { z12.h-z15.h }, z7.h[4]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117ba08)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117ba89)  // fmla za.h[x9, 1], { z20.h-z23.h }, z7.h[5]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b90a)  // fmla za.h[x9, 2], { z8.h-z11.h }, z7.h[5]
    KAI_ASM_INST(0xc117bc00)  // fmla za.h[x9, 0], { z0.h-z3.h }, z7.h[6]
    KAI_ASM_INST(0xc117bf81)  // fmla za.h[x9, 1], { z28.h-z31.h }, z7.h[6]
    KAI_ASM_INST(0xc117bf02)  // fmla za.h[x9, 2], { z24.h-z27.h }, z7.h[6]
    KAI_ASM_INST(0xc117bd88)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[7]
    KAI_ASM_INST(0xc117be09)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[7]
    KAI_ASM_INST(0xc117be8a)  // fmla za.h[x9, 2], { z20.h-z23.h }, z7.h[7]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b180)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[0]
    KAI_ASM_INST(0xc117b301)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[0]
    KAI_ASM_INST(0xc117b382)  // fmla za.h[x9, 2], { z28.h-z31.h }, z7.h[0]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b288)  // fmla za.h[x9, 0], { z20.h-z23.h }, z7.h[1]
    KAI_ASM_INST(0xc117b009)  // fmla za.h[x9, 1], { z0.h-z3.h }, z7.h[1]
    KAI_ASM_INST(0xc117b30a)  // fmla za.h[x9, 2], { z24.h-z27.h }, z7.h[1]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b680)  // fmla za.h[x9, 0], { z20.h-z23.h }, z7.h[2]
    KAI_ASM_INST(0xc117b601)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b582)  // fmla za.h[x9, 2], { z12.h-z15.h }, z7.h[2]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b688)  // fmla za.h[x9, 0], { z20.h-z23.h }, z7.h[3]
    KAI_ASM_INST(0xc117b709)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b60a)  // fmla za.h[x9, 2], { z16.h-z19.h }, z7.h[3]
    ble label_25
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bb80)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[4]
    KAI_ASM_INST(0xc117b981)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[4]
    KAI_ASM_INST(0xc117ba02)  // fmla za.h[x9, 2], { z16.h-z19.h }, z7.h[4]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b988)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[5]
    KAI_ASM_INST(0xc117ba09)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xc117bb0a)  // fmla za.h[x9, 2], { z24.h-z27.h }, z7.h[5]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bd80)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[6]
    KAI_ASM_INST(0xc117be81)  // fmla za.h[x9, 1], { z20.h-z23.h }, z7.h[6]
    KAI_ASM_INST(0xc117be02)  // fmla za.h[x9, 2], { z16.h-z19.h }, z7.h[6]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc117bd88)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[7]
    KAI_ASM_INST(0xc117bf89)  // fmla za.h[x9, 1], { z28.h-z31.h }, z7.h[7]
    KAI_ASM_INST(0xc117be8a)  // fmla za.h[x9, 2], { z20.h-z23.h }, z7.h[7]
KAI_ASM_LABEL(label_25)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_26
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a451)  // ld1rh { z17.h }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c44)  // mova { z4.d-z7.d }, za.d[x9, #2]
    KAI_ASM_INST(0x84c0a430)  // ld1rh { z16.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc170ca2c)  // fclamp { z12.h-z15.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170ca20)  // fclamp { z0.h-z3.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170ca24)  // fclamp { z4.h-z7.h }, z17.h, z16.h
    KAI_ASM_INST(0xa060a4cc)  // st1h { z12.h-z15.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a4c0)  // st1h { z0.h-z3.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062a0c4)  // st1h { z4.h-z7.h }, p8, [x6, #0x8, MUL VL]
    b label_27
KAI_ASM_LABEL(label_26)  // Width 3: No activation
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c30)  // mova { z16.d-z19.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c4c)  // mova { z12.d-z15.d }, za.d[x9, #2]
    KAI_ASM_INST(0xa060a4c8)  // st1h { z8.h-z11.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a4d0)  // st1h { z16.h-z19.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062a0cc)  // st1h { z12.h-z15.h }, p8, [x6, #0x8, MUL VL]
KAI_ASM_LABEL(label_27)  // Width 3: Output done
    b label_36
KAI_ASM_LABEL(label_28)  // Width 4
    ldr x4, [x0, #0x20]
    mov x1, #0x3
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x256167f0)  // whilelt p8.h, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_29
    KAI_ASM_INST(0xa040a4bc)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041a4b8)  // ld1h { z24.h-z27.h }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4b0)  // ld1h { z16.h-z19.h }, pn9.b/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a0a4)  // ld1h { z4.h-z7.h }, p8.h/Z, [x5, #0xc, MUL VL]
    KAI_ASM_INST(0xc0042f80)  // mova za.d[x9, #0], { z28.d-z31.d }
    addvl x5, x5, #16
    KAI_ASM_INST(0xc0042f01)  // mova za.d[x9, #1], { z24.d-z27.d }
    KAI_ASM_INST(0xc0042e02)  // mova za.d[x9, #2], { z16.d-z19.d }
    KAI_ASM_INST(0xc0042c83)  // mova za.d[x9, #3], { z4.d-z7.d }
    b label_30
KAI_ASM_LABEL(label_29)  // Width 4: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_30)  // Width 4: setup done
    cmp x3, #0x8
    ble label_32
KAI_ASM_LABEL(label_31)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z2.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b300)  // fmla za.h[x9, 0], { z24.h-z27.h }, z2.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b081)  // fmla za.h[x9, 1], { z4.h-z7.h }, z2.h[0]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b282)  // fmla za.h[x9, 2], { z20.h-z23.h }, z2.h[0]
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b183)  // fmla za.h[x9, 3], { z12.h-z15.h }, z2.h[0]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b208)  // fmla za.h[x9, 0], { z16.h-z19.h }, z2.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b089)  // fmla za.h[x9, 1], { z4.h-z7.h }, z2.h[1]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b30a)  // fmla za.h[x9, 2], { z24.h-z27.h }, z2.h[1]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b38b)  // fmla za.h[x9, 3], { z28.h-z31.h }, z2.h[1]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b480)  // fmla za.h[x9, 0], { z4.h-z7.h }, z2.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b501)  // fmla za.h[x9, 1], { z8.h-z11.h }, z2.h[2]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b782)  // fmla za.h[x9, 2], { z28.h-z31.h }, z2.h[2]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b583)  // fmla za.h[x9, 3], { z12.h-z15.h }, z2.h[2]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b488)  // fmla za.h[x9, 0], { z4.h-z7.h }, z2.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b789)  // fmla za.h[x9, 1], { z28.h-z31.h }, z2.h[3]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b58a)  // fmla za.h[x9, 2], { z12.h-z15.h }, z2.h[3]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b50b)  // fmla za.h[x9, 3], { z8.h-z11.h }, z2.h[3]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bb80)  // fmla za.h[x9, 0], { z28.h-z31.h }, z2.h[4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b981)  // fmla za.h[x9, 1], { z12.h-z15.h }, z2.h[4]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112ba02)  // fmla za.h[x9, 2], { z16.h-z19.h }, z2.h[4]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b883)  // fmla za.h[x9, 3], { z4.h-z7.h }, z2.h[4]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bb88)  // fmla za.h[x9, 0], { z28.h-z31.h }, z2.h[5]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b989)  // fmla za.h[x9, 1], { z12.h-z15.h }, z2.h[5]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112ba0a)  // fmla za.h[x9, 2], { z16.h-z19.h }, z2.h[5]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b90b)  // fmla za.h[x9, 3], { z8.h-z11.h }, z2.h[5]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bc80)  // fmla za.h[x9, 0], { z4.h-z7.h }, z2.h[6]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112bf01)  // fmla za.h[x9, 1], { z24.h-z27.h }, z2.h[6]
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112bd02)  // fmla za.h[x9, 2], { z8.h-z11.h }, z2.h[6]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112be03)  // fmla za.h[x9, 3], { z16.h-z19.h }, z2.h[6]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bf08)  // fmla za.h[x9, 0], { z24.h-z27.h }, z2.h[7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112bf89)  // fmla za.h[x9, 1], { z28.h-z31.h }, z2.h[7]
    KAI_ASM_INST(0xc112be8a)  // fmla za.h[x9, 2], { z20.h-z23.h }, z2.h[7]
    KAI_ASM_INST(0xc112be0b)  // fmla za.h[x9, 3], { z16.h-z19.h }, z2.h[7]
    bgt label_31
KAI_ASM_LABEL(label_32)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b380)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b281)  // fmla za.h[x9, 1], { z20.h-z23.h }, z7.h[0]
    KAI_ASM_INST(0xc117b182)  // fmla za.h[x9, 2], { z12.h-z15.h }, z7.h[0]
    KAI_ASM_INST(0xc117b203)  // fmla za.h[x9, 3], { z16.h-z19.h }, z7.h[0]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b388)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b309)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[1]
    KAI_ASM_INST(0xc117b28a)  // fmla za.h[x9, 2], { z20.h-z23.h }, z7.h[1]
    KAI_ASM_INST(0xc117b18b)  // fmla za.h[x9, 3], { z12.h-z15.h }, z7.h[1]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b580)  // fmla za.h[x9, 0], { z12.h-z15.h }, z7.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b601)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b782)  // fmla za.h[x9, 2], { z28.h-z31.h }, z7.h[2]
    KAI_ASM_INST(0xc117b403)  // fmla za.h[x9, 3], { z0.h-z3.h }, z7.h[2]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b788)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b709)  // fmla za.h[x9, 1], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b58a)  // fmla za.h[x9, 2], { z12.h-z15.h }, z7.h[3]
    KAI_ASM_INST(0xc117b68b)  // fmla za.h[x9, 3], { z20.h-z23.h }, z7.h[3]
    ble label_33
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bb00)  // fmla za.h[x9, 0], { z24.h-z27.h }, z7.h[4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba81)  // fmla za.h[x9, 1], { z20.h-z23.h }, z7.h[4]
    KAI_ASM_INST(0xc117ba02)  // fmla za.h[x9, 2], { z16.h-z19.h }, z7.h[4]
    KAI_ASM_INST(0xc117b803)  // fmla za.h[x9, 3], { z0.h-z3.h }, z7.h[4]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bb88)  // fmla za.h[x9, 0], { z28.h-z31.h }, z7.h[5]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba09)  // fmla za.h[x9, 1], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xc117ba8a)  // fmla za.h[x9, 2], { z20.h-z23.h }, z7.h[5]
    KAI_ASM_INST(0xc117b98b)  // fmla za.h[x9, 3], { z12.h-z15.h }, z7.h[5]
    ble label_33
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117be00)  // fmla za.h[x9, 0], { z16.h-z19.h }, z7.h[6]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bd81)  // fmla za.h[x9, 1], { z12.h-z15.h }, z7.h[6]
    KAI_ASM_INST(0xc117be82)  // fmla za.h[x9, 2], { z20.h-z23.h }, z7.h[6]
    KAI_ASM_INST(0xc117bc03)  // fmla za.h[x9, 3], { z0.h-z3.h }, z7.h[6]
    ble label_33
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bf08)  // fmla za.h[x9, 0], { z24.h-z27.h }, z7.h[7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bf89)  // fmla za.h[x9, 1], { z28.h-z31.h }, z7.h[7]
    KAI_ASM_INST(0xc117be0a)  // fmla za.h[x9, 2], { z16.h-z19.h }, z7.h[7]
    KAI_ASM_INST(0xc117bd8b)  // fmla za.h[x9, 3], { z12.h-z15.h }, z7.h[7]
KAI_ASM_LABEL(label_33)  // Width 4: Multiply loop: multiply skip
    tbz x21, #1, label_34
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a451)  // ld1rh { z17.h }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c40)  // mova { z0.d-z3.d }, za.d[x9, #2]
    KAI_ASM_INST(0x84c0a430)  // ld1rh { z16.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc0062c68)  // mova { z8.d-z11.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc170ca24)  // fclamp { z4.h-z7.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170ca2c)  // fclamp { z12.h-z15.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170ca20)  // fclamp { z0.h-z3.h }, z17.h, z16.h
    KAI_ASM_INST(0xc170ca28)  // fclamp { z8.h-z11.h }, z17.h, z16.h
    KAI_ASM_INST(0xa060a4c4)  // st1h { z4.h-z7.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a4cc)  // st1h { z12.h-z15.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062a4c0)  // st1h { z0.h-z3.h }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063a0c8)  // st1h { z8.h-z11.h }, p8, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
    b label_35
KAI_ASM_LABEL(label_34)  // Width 4: No activation
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c50)  // mova { z16.d-z19.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc0062c60)  // mova { z0.d-z3.d }, za.d[x9, #3]
    KAI_ASM_INST(0xa060a4c8)  // st1h { z8.h-z11.h }, pn9.b, [x6]
    KAI_ASM_INST(0xa061a4cc)  // st1h { z12.h-z15.h }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062a4d0)  // st1h { z16.h-z19.h }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063a0c0)  // st1h { z0.h-z3.h }, p8, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_fp16_mla_16VL)

    KAI_ASM_END
