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

    KAI_ASM_CODE(sme2_gemv_bf16_mla_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_bf16_mla_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_bf16_mla_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_bf16_mla_16VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x2, #0xff80
    ldr x23, [x0, #0x8]
    cnth x22, ALL, MUL #4
    mov x5, #0x7f80
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
    KAI_ASM_INST(0xc11fb320)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z15.h[0]
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc11fb0a8)  // bfmla za.h[x9, 0], { z4.h-z7.h }, z15.h[1]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc11fb420)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z15.h[2]
    KAI_ASM_INST(0xc11fb6a8)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z15.h[3]
    KAI_ASM_INST(0xc11fba20)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z15.h[4]
    KAI_ASM_INST(0xc11fb928)  // bfmla za.h[x9, 0], { z8.h-z11.h }, z15.h[5]
    KAI_ASM_INST(0xc11fbf20)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z15.h[6]
    KAI_ASM_INST(0xc11fbca8)  // bfmla za.h[x9, 0], { z4.h-z7.h }, z15.h[7]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b1a0)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[0]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b228)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[1]
    ble label_9
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b420)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z7.h[2]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b628)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[3]
    ble label_9
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117baa0)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z7.h[4]
    ble label_9
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bba8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[5]
    ble label_9
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bc20)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z7.h[6]
    ble label_9
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117be28)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[7]
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_10
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0x84c0a456)  // ld1rh { z22.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a427)  // ld1rh { z7.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc127cac0)  // bfclamp { z0.h-z3.h }, z22.h, z7.h
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
    KAI_ASM_INST(0xc115b020)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z5.h[0]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b3a1)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z5.h[0]
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b228)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z5.h[1]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b029)  // bfmla za.h[x9, 1], { z0.h-z3.h }, z5.h[1]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b6a0)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z5.h[2]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b521)  // bfmla za.h[x9, 1], { z8.h-z11.h }, z5.h[2]
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115b7a8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z5.h[3]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115b5a9)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z5.h[3]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc115ba20)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z5.h[4]
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc115bb21)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z5.h[4]
    KAI_ASM_INST(0xc115b828)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z5.h[5]
    KAI_ASM_INST(0xc115baa9)  // bfmla za.h[x9, 1], { z20.h-z23.h }, z5.h[5]
    KAI_ASM_INST(0xc115bd20)  // bfmla za.h[x9, 0], { z8.h-z11.h }, z5.h[6]
    KAI_ASM_INST(0xc115bfa1)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z5.h[6]
    KAI_ASM_INST(0xc115bda8)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z5.h[7]
    KAI_ASM_INST(0xc115be29)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z5.h[7]
    bgt label_15
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b020)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z7.h[0]
    KAI_ASM_INST(0xc117b321)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[0]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b228)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[1]
    KAI_ASM_INST(0xc117b129)  // bfmla za.h[x9, 1], { z8.h-z11.h }, z7.h[1]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b620)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b5a1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[2]
    ble label_17
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b728)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b629)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[3]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba20)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[4]
    KAI_ASM_INST(0xc117b9a1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[4]
    ble label_17
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b828)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z7.h[5]
    KAI_ASM_INST(0xc117b9a9)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[5]
    ble label_17
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117be20)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[6]
    KAI_ASM_INST(0xc117bda1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[6]
    ble label_17
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117bf28)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z7.h[7]
    KAI_ASM_INST(0xc117bc29)  // bfmla za.h[x9, 1], { z0.h-z3.h }, z7.h[7]
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_18
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c10)  // mova { z16.d-z19.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c24)  // mova { z4.d-z7.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a44f)  // ld1rh { z15.h }, p1/Z, [x2]
    KAI_ASM_INST(0x84c0a43b)  // ld1rh { z27.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc13bc9f0)  // bfclamp { z16.h-z19.h }, z15.h, z27.h
    KAI_ASM_INST(0xc13bc9e4)  // bfclamp { z4.h-z7.h }, z15.h, z27.h
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
    KAI_ASM_INST(0xc117b2a0)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z7.h[0]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b221)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b122)  // bfmla za.h[x9, 2], { z8.h-z11.h }, z7.h[0]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b3a8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[1]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b1a9)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[1]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b12a)  // bfmla za.h[x9, 2], { z8.h-z11.h }, z7.h[1]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b720)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z7.h[2]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b7a1)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z7.h[2]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b5a2)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z7.h[2]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b628)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[3]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117b429)  // bfmla za.h[x9, 1], { z0.h-z3.h }, z7.h[3]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117b52a)  // bfmla za.h[x9, 2], { z8.h-z11.h }, z7.h[3]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117bba0)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[4]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117bb21)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[4]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b9a2)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z7.h[4]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc117ba28)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc117baa9)  // bfmla za.h[x9, 1], { z20.h-z23.h }, z7.h[5]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b92a)  // bfmla za.h[x9, 2], { z8.h-z11.h }, z7.h[5]
    KAI_ASM_INST(0xc117bc20)  // bfmla za.h[x9, 0], { z0.h-z3.h }, z7.h[6]
    KAI_ASM_INST(0xc117bfa1)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z7.h[6]
    KAI_ASM_INST(0xc117bf22)  // bfmla za.h[x9, 2], { z24.h-z27.h }, z7.h[6]
    KAI_ASM_INST(0xc117bda8)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[7]
    KAI_ASM_INST(0xc117be29)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[7]
    KAI_ASM_INST(0xc117beaa)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z7.h[7]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b1a0)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[0]
    KAI_ASM_INST(0xc117b321)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[0]
    KAI_ASM_INST(0xc117b3a2)  // bfmla za.h[x9, 2], { z28.h-z31.h }, z7.h[0]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b2a8)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z7.h[1]
    KAI_ASM_INST(0xc117b029)  // bfmla za.h[x9, 1], { z0.h-z3.h }, z7.h[1]
    KAI_ASM_INST(0xc117b32a)  // bfmla za.h[x9, 2], { z24.h-z27.h }, z7.h[1]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b6a0)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z7.h[2]
    KAI_ASM_INST(0xc117b621)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b5a2)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z7.h[2]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b6a8)  // bfmla za.h[x9, 0], { z20.h-z23.h }, z7.h[3]
    KAI_ASM_INST(0xc117b729)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b62a)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z7.h[3]
    ble label_25
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bba0)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[4]
    KAI_ASM_INST(0xc117b9a1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[4]
    KAI_ASM_INST(0xc117ba22)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z7.h[4]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b9a8)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[5]
    KAI_ASM_INST(0xc117ba29)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xc117bb2a)  // bfmla za.h[x9, 2], { z24.h-z27.h }, z7.h[5]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bda0)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[6]
    KAI_ASM_INST(0xc117bea1)  // bfmla za.h[x9, 1], { z20.h-z23.h }, z7.h[6]
    KAI_ASM_INST(0xc117be22)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z7.h[6]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc117bda8)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[7]
    KAI_ASM_INST(0xc117bfa9)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z7.h[7]
    KAI_ASM_INST(0xc117beaa)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z7.h[7]
KAI_ASM_LABEL(label_25)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_26
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0x84c0a451)  // ld1rh { z17.h }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c44)  // mova { z4.d-z7.d }, za.d[x9, #2]
    KAI_ASM_INST(0x84c0a430)  // ld1rh { z16.h }, p1/Z, [x1]
    KAI_ASM_INST(0xc130ca2c)  // bfclamp { z12.h-z15.h }, z17.h, z16.h
    KAI_ASM_INST(0xc130ca20)  // bfclamp { z0.h-z3.h }, z17.h, z16.h
    KAI_ASM_INST(0xc130ca24)  // bfclamp { z4.h-z7.h }, z17.h, z16.h
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
    KAI_ASM_INST(0xc112b320)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z2.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b0a1)  // bfmla za.h[x9, 1], { z4.h-z7.h }, z2.h[0]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b2a2)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z2.h[0]
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b1a3)  // bfmla za.h[x9, 3], { z12.h-z15.h }, z2.h[0]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b228)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z2.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b0a9)  // bfmla za.h[x9, 1], { z4.h-z7.h }, z2.h[1]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b32a)  // bfmla za.h[x9, 2], { z24.h-z27.h }, z2.h[1]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b3ab)  // bfmla za.h[x9, 3], { z28.h-z31.h }, z2.h[1]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b4a0)  // bfmla za.h[x9, 0], { z4.h-z7.h }, z2.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b521)  // bfmla za.h[x9, 1], { z8.h-z11.h }, z2.h[2]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b7a2)  // bfmla za.h[x9, 2], { z28.h-z31.h }, z2.h[2]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b5a3)  // bfmla za.h[x9, 3], { z12.h-z15.h }, z2.h[2]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112b4a8)  // bfmla za.h[x9, 0], { z4.h-z7.h }, z2.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b7a9)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z2.h[3]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112b5aa)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z2.h[3]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b52b)  // bfmla za.h[x9, 3], { z8.h-z11.h }, z2.h[3]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bba0)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z2.h[4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b9a1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z2.h[4]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112ba22)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z2.h[4]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b8a3)  // bfmla za.h[x9, 3], { z4.h-z7.h }, z2.h[4]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bba8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z2.h[5]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112b9a9)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z2.h[5]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112ba2a)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z2.h[5]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112b92b)  // bfmla za.h[x9, 3], { z8.h-z11.h }, z2.h[5]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bca0)  // bfmla za.h[x9, 0], { z4.h-z7.h }, z2.h[6]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112bf21)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z2.h[6]
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc112bd22)  // bfmla za.h[x9, 2], { z8.h-z11.h }, z2.h[6]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc112be23)  // bfmla za.h[x9, 3], { z16.h-z19.h }, z2.h[6]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc112bf28)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z2.h[7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc112bfa9)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z2.h[7]
    KAI_ASM_INST(0xc112beaa)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z2.h[7]
    KAI_ASM_INST(0xc112be2b)  // bfmla za.h[x9, 3], { z16.h-z19.h }, z2.h[7]
    bgt label_31
KAI_ASM_LABEL(label_32)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    ld1rqh { z7.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b3a0)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b2a1)  // bfmla za.h[x9, 1], { z20.h-z23.h }, z7.h[0]
    KAI_ASM_INST(0xc117b1a2)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z7.h[0]
    KAI_ASM_INST(0xc117b223)  // bfmla za.h[x9, 3], { z16.h-z19.h }, z7.h[0]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b3a8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b329)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[1]
    KAI_ASM_INST(0xc117b2aa)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z7.h[1]
    KAI_ASM_INST(0xc117b1ab)  // bfmla za.h[x9, 3], { z12.h-z15.h }, z7.h[1]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b5a0)  // bfmla za.h[x9, 0], { z12.h-z15.h }, z7.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b621)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc117b7a2)  // bfmla za.h[x9, 2], { z28.h-z31.h }, z7.h[2]
    KAI_ASM_INST(0xc117b423)  // bfmla za.h[x9, 3], { z0.h-z3.h }, z7.h[2]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117b7a8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117b729)  // bfmla za.h[x9, 1], { z24.h-z27.h }, z7.h[3]
    KAI_ASM_INST(0xc117b5aa)  // bfmla za.h[x9, 2], { z12.h-z15.h }, z7.h[3]
    KAI_ASM_INST(0xc117b6ab)  // bfmla za.h[x9, 3], { z20.h-z23.h }, z7.h[3]
    ble label_33
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bb20)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z7.h[4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117baa1)  // bfmla za.h[x9, 1], { z20.h-z23.h }, z7.h[4]
    KAI_ASM_INST(0xc117ba22)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z7.h[4]
    KAI_ASM_INST(0xc117b823)  // bfmla za.h[x9, 3], { z0.h-z3.h }, z7.h[4]
    ble label_33
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bba8)  // bfmla za.h[x9, 0], { z28.h-z31.h }, z7.h[5]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117ba29)  // bfmla za.h[x9, 1], { z16.h-z19.h }, z7.h[5]
    KAI_ASM_INST(0xc117baaa)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z7.h[5]
    KAI_ASM_INST(0xc117b9ab)  // bfmla za.h[x9, 3], { z12.h-z15.h }, z7.h[5]
    ble label_33
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x1
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117be20)  // bfmla za.h[x9, 0], { z16.h-z19.h }, z7.h[6]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bda1)  // bfmla za.h[x9, 1], { z12.h-z15.h }, z7.h[6]
    KAI_ASM_INST(0xc117bea2)  // bfmla za.h[x9, 2], { z20.h-z23.h }, z7.h[6]
    KAI_ASM_INST(0xc117bc23)  // bfmla za.h[x9, 3], { z0.h-z3.h }, z7.h[6]
    ble label_33
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc117bf28)  // bfmla za.h[x9, 0], { z24.h-z27.h }, z7.h[7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc117bfa9)  // bfmla za.h[x9, 1], { z28.h-z31.h }, z7.h[7]
    KAI_ASM_INST(0xc117be2a)  // bfmla za.h[x9, 2], { z16.h-z19.h }, z7.h[7]
    KAI_ASM_INST(0xc117bdab)  // bfmla za.h[x9, 3], { z12.h-z15.h }, z7.h[7]
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
    KAI_ASM_INST(0xc130ca24)  // bfclamp { z4.h-z7.h }, z17.h, z16.h
    KAI_ASM_INST(0xc130ca2c)  // bfclamp { z12.h-z15.h }, z17.h, z16.h
    KAI_ASM_INST(0xc130ca20)  // bfclamp { z0.h-z3.h }, z17.h, z16.h
    KAI_ASM_INST(0xc130ca28)  // bfclamp { z8.h-z11.h }, z17.h, z16.h
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_bf16_mla_16VL)

    KAI_ASM_END
