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

    KAI_ASM_CODE(sme2_gemv_bf16fp32_dot_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_bf16fp32_dot_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_bf16fp32_dot_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_bf16fp32_dot_16VL)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x2, #0xff800000
    ldr x23, [x0, #0x8]
    cntw x22, ALL, MUL #4
    mov x5, #0x7f800000
    ldr w1, [x0, #0xa0]
    mov x21, #0
    mov x9, #0
    ldr w4, [x0, #0xa4]
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
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_5
    KAI_ASM_INST(0xa040c0b4)  // ld1w { z20.s-z23.s }, p8.s/Z, [x5]
    KAI_ASM_INST(0xc0042e80)  // mova za.d[x9, #0], { z20.d-z23.d }
    b label_6
KAI_ASM_LABEL(label_5)  // Width 1: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_6)  // Width 1: setup done
    cmp x3, #0x8
    ble label_8
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    ld1rqh { z2.h }, p0/Z, [x4]
    sub x3, x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    cmp x3, #0x8
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc152b398)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z2.h[0]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc152b618)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z2.h[1]
    KAI_ASM_INST(0xc152bb18)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z2.h[2]
    KAI_ASM_INST(0xc152bd98)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z2.h[3]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z3.h }, p0/Z, [x4]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b198)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    ble label_9
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b498)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[1]
    ble label_9
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b998)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[2]
    ble label_9
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z3.h[3]
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: multiply skip
    tbz x21, #1, label_10
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0x8540c457)  // ld1rw { z23.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c436)  // ld1rw { z22.s }, p1/Z, [x1]
    KAI_ASM_INST(0xc1b6cae0)  // fclamp { z0.s-z3.s }, z23.s, z22.s
    KAI_ASM_INST(0xa060c0c0)  // st1w { z0.s-z3.s }, p8, [x6]
    b label_11
KAI_ASM_LABEL(label_10)  // Width 1: No activation
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0xa060c0c0)  // st1w { z0.s-z3.s }, p8, [x6]
KAI_ASM_LABEL(label_11)  // Width 1: Output done
    b label_36
KAI_ASM_LABEL(label_12)  // Width 2
    ldr x4, [x0, #0x20]
    lsl x2, x8, #0x1
    sub x1, x23, x22
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_13
    KAI_ASM_INST(0xa040c4bc)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c0a4)  // ld1w { z4.s-z7.s }, p8.s/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xc0042f80)  // mova za.d[x9, #0], { z28.d-z31.d }
    KAI_ASM_INST(0xc0042c81)  // mova za.d[x9, #1], { z4.d-z7.d }
    b label_14
KAI_ASM_LABEL(label_13)  // Width 2: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_14)  // Width 2: setup done
    cmp x3, #0x8
    ble label_16
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z1.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc151b398)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z1.h[0]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc151b199)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z1.h[0]
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc151b618)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z1.h[1]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc151b719)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z1.h[1]
    KAI_ASM_INST(0xc151bb98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z1.h[2]
    KAI_ASM_INST(0xc151b999)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z1.h[2]
    KAI_ASM_INST(0xc151bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z1.h[3]
    KAI_ASM_INST(0xc151be99)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z1.h[3]
    bgt label_15
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z3.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b218)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z3.h[0]
    KAI_ASM_INST(0xc153b399)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[0]
    ble label_17
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b698)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z3.h[1]
    KAI_ASM_INST(0xc153b619)  // bfdot za.s[x9, 1], { z16.h-z19.h }, z3.h[1]
    ble label_17
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b898)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[2]
    KAI_ASM_INST(0xc153ba19)  // bfdot za.s[x9, 1], { z16.h-z19.h }, z3.h[2]
    ble label_17
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153bf98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z3.h[3]
    KAI_ASM_INST(0xc153bd99)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[3]
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: multiply skip
    tbz x21, #1, label_18
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c28)  // mova { z8.d-z11.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c451)  // ld1rw { z17.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c437)  // ld1rw { z23.s }, p1/Z, [x1]
    KAI_ASM_INST(0xc1b7ca24)  // fclamp { z4.s-z7.s }, z17.s, z23.s
    KAI_ASM_INST(0xc1b7ca28)  // fclamp { z8.s-z11.s }, z17.s, z23.s
    KAI_ASM_INST(0xa060c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c0c8)  // st1w { z8.s-z11.s }, p8, [x6, #0x4, MUL VL]
    b label_19
KAI_ASM_LABEL(label_18)  // Width 2: No activation
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c30)  // mova { z16.d-z19.d }, za.d[x9, #1]
    KAI_ASM_INST(0xa060c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c0d0)  // st1w { z16.s-z19.s }, p8, [x6, #0x4, MUL VL]
KAI_ASM_LABEL(label_19)  // Width 2: Output done
    b label_36
KAI_ASM_LABEL(label_20)  // Width 3
    ldr x4, [x0, #0x20]
    mov x1, #0x2
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_21
    KAI_ASM_INST(0xa040c4a8)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4a0)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c0a4)  // ld1w { z4.s-z7.s }, p8.s/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xc0042d00)  // mova za.d[x9, #0], { z8.d-z11.d }
    KAI_ASM_INST(0xc0042c01)  // mova za.d[x9, #1], { z0.d-z3.d }
    KAI_ASM_INST(0xc0042c82)  // mova za.d[x9, #2], { z4.d-z7.d }
    b label_22
KAI_ASM_LABEL(label_21)  // Width 3: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_22)  // Width 3: setup done
    cmp x3, #0x8
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    sub x3, x3, #0x8
    ld1rqh { z3.h }, p0/Z, [x4]
    cmp x3, #0x8
    add x4, x4, #0x10
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b198)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b119)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[0]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b09a)  // bfdot za.s[x9, 2], { z4.h-z7.h }, z3.h[0]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b618)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z3.h[1]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b699)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z3.h[1]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b71a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z3.h[1]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b998)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[2]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bb99)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[2]
    KAI_ASM_INST(0xc153b91a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xc153bc98)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[3]
    KAI_ASM_INST(0xc153be99)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z3.h[3]
    KAI_ASM_INST(0xc153bf1a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z3.h[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z3.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b298)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z3.h[0]
    KAI_ASM_INST(0xc153b199)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xc153b09a)  // bfdot za.s[x9, 2], { z4.h-z7.h }, z3.h[0]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b698)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z3.h[1]
    KAI_ASM_INST(0xc153b519)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xc153b61a)  // bfdot za.s[x9, 2], { z16.h-z19.h }, z3.h[1]
    ble label_25
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bb98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z3.h[2]
    KAI_ASM_INST(0xc153bb19)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z3.h[2]
    KAI_ASM_INST(0xc153b99a)  // bfdot za.s[x9, 2], { z12.h-z15.h }, z3.h[2]
    ble label_25
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc153bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z3.h[3]
    KAI_ASM_INST(0xc153bf99)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[3]
    KAI_ASM_INST(0xc153bd9a)  // bfdot za.s[x9, 2], { z12.h-z15.h }, z3.h[3]
KAI_ASM_LABEL(label_25)  // Width 3: Multiply loop: multiply skip
    tbz x21, #1, label_26
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c455)  // ld1rw { z21.s }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c50)  // mova { z16.d-z19.d }, za.d[x9, #2]
    KAI_ASM_INST(0x8540c434)  // ld1rw { z20.s }, p1/Z, [x1]
    KAI_ASM_INST(0xc1b4caa8)  // fclamp { z8.s-z11.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caac)  // fclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cab0)  // fclamp { z16.s-z19.s }, z21.s, z20.s
    KAI_ASM_INST(0xa060c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c0d0)  // st1w { z16.s-z19.s }, p8, [x6, #0x8, MUL VL]
    b label_27
KAI_ASM_LABEL(label_26)  // Width 3: No activation
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c5c)  // mova { z28.d-z31.d }, za.d[x9, #2]
    KAI_ASM_INST(0xa060c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c0dc)  // st1w { z28.s-z31.s }, p8, [x6, #0x8, MUL VL]
KAI_ASM_LABEL(label_27)  // Width 3: Output done
    b label_36
KAI_ASM_LABEL(label_28)  // Width 4
    ldr x4, [x0, #0x20]
    mov x1, #0x3
    lsl x2, x8, #0x1
    msub x1, x22, x1, x23
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_29
    KAI_ASM_INST(0xa040c4a4)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4ac)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c4bc)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c0b0)  // ld1w { z16.s-z19.s }, p8.s/Z, [x5, #0xc, MUL VL]
    KAI_ASM_INST(0xc0042c80)  // mova za.d[x9, #0], { z4.d-z7.d }
    addvl x5, x5, #16
    KAI_ASM_INST(0xc0042d81)  // mova za.d[x9, #1], { z12.d-z15.d }
    KAI_ASM_INST(0xc0042f82)  // mova za.d[x9, #2], { z28.d-z31.d }
    KAI_ASM_INST(0xc0042e03)  // mova za.d[x9, #3], { z16.d-z19.d }
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
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b198)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b299)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z3.h[0]
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b21a)  // bfdot za.s[x9, 2], { z16.h-z19.h }, z3.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b11b)  // bfdot za.s[x9, 3], { z8.h-z11.h }, z3.h[0]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b718)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z3.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b599)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[1]
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b51a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b49b)  // bfdot za.s[x9, 3], { z4.h-z7.h }, z3.h[1]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bb18)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z3.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b919)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153ba1a)  // bfdot za.s[x9, 2], { z16.h-z19.h }, z3.h[2]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b89b)  // bfdot za.s[x9, 3], { z4.h-z7.h }, z3.h[2]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z3.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bf99)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[3]
    KAI_ASM_INST(0xc153bd9a)  // bfdot za.s[x9, 2], { z12.h-z15.h }, z3.h[3]
    KAI_ASM_INST(0xc153be9b)  // bfdot za.s[x9, 3], { z20.h-z23.h }, z3.h[3]
    bgt label_31
KAI_ASM_LABEL(label_32)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x3
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    ld1rqh { z3.h }, p0/Z, [x4]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b218)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z3.h[0]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b199)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xc153b39a)  // bfdot za.s[x9, 2], { z28.h-z31.h }, z3.h[0]
    KAI_ASM_INST(0xc153b29b)  // bfdot za.s[x9, 3], { z20.h-z23.h }, z3.h[0]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b598)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b499)  // bfdot za.s[x9, 1], { z4.h-z7.h }, z3.h[1]
    KAI_ASM_INST(0xc153b71a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z3.h[1]
    KAI_ASM_INST(0xc153b69b)  // bfdot za.s[x9, 3], { z20.h-z23.h }, z3.h[1]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b998)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b919)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xc153ba9a)  // bfdot za.s[x9, 2], { z20.h-z23.h }, z3.h[2]
    KAI_ASM_INST(0xc153ba1b)  // bfdot za.s[x9, 3], { z16.h-z19.h }, z3.h[2]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bd98)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bd19)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[3]
    KAI_ASM_INST(0xc153be9a)  // bfdot za.s[x9, 2], { z20.h-z23.h }, z3.h[3]
    KAI_ASM_INST(0xc153be1b)  // bfdot za.s[x9, 3], { z16.h-z19.h }, z3.h[3]
KAI_ASM_LABEL(label_33)  // Width 4: Multiply loop: multiply skip
    tbz x21, #1, label_34
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c455)  // ld1rw { z21.s }, p1/Z, [x2]
    KAI_ASM_INST(0xc0062c4c)  // mova { z12.d-z15.d }, za.d[x9, #2]
    KAI_ASM_INST(0x8540c434)  // ld1rw { z20.s }, p1/Z, [x1]
    KAI_ASM_INST(0xc0062c70)  // mova { z16.d-z19.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc1b4caa4)  // fclamp { z4.s-z7.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caa0)  // fclamp { z0.s-z3.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caac)  // fclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cab0)  // fclamp { z16.s-z19.s }, z21.s, z20.s
    KAI_ASM_INST(0xa060c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c0)  // st1w { z0.s-z3.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c0d0)  // st1w { z16.s-z19.s }, p8, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
    b label_35
KAI_ASM_LABEL(label_34)  // Width 4: No activation
    KAI_ASM_INST(0xc0062c0c)  // mova { z12.d-z15.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c50)  // mova { z16.d-z19.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc0062c64)  // mova { z4.d-z7.d }, za.d[x9, #3]
    KAI_ASM_INST(0xa060c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c0)  // st1w { z0.s-z3.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c4d0)  // st1w { z16.s-z19.s }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c0c4)  // st1w { z4.s-z7.s }, p8, [x6, #0xc, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_bf16fp32_dot_16VL)

    KAI_ASM_END
