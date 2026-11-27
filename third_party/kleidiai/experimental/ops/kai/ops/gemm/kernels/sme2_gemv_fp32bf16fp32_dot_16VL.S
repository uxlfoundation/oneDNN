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

    KAI_ASM_CODE(sme2_gemv_fp32bf16fp32_dot_16VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_gemv_fp32bf16fp32_dot_16VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_gemv_fp32bf16fp32_dot_16VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_gemv_fp32bf16fp32_dot_16VL)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x2, #0xff800000
    ldr x24, [x0, #0x8]
    cntw x23, ALL, MUL #4
    mov x5, #0x7f800000
    ldr w1, [x0, #0xa0]
    mov x22, #0
    mov x9, #0
    ldr w4, [x0, #0xa4]
    mov x21, #0x4
    ptrue p2.b
    add x20, x24, x23
    ldr x8, [x0, #0x10]
    mov x3, #0x1
    sub x20, x20, #0x1
    cmp x1, x2
    ldr x7, [x0, #0x30]
    udiv x20, x20, x23
    cset x1, NE
    ldr x6, [x0, #0x50]
    add x2, x20, #0x3
    cmp x4, x5
    and x2, x2, #0xfffffffffffffffc
    orr x22, x22, x1, LSL #1
    mul x2, x2, x23
    cset x1, NE
    mul x2, x2, x8
    orr x22, x22, x1, LSL #1
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
    lsl x2, x8, #0x2
    mov x1, x24
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_5
    KAI_ASM_INST(0xa040c0b0)  // ld1w { z16.s-z19.s }, p8.s/Z, [x5]
    KAI_ASM_INST(0xc0042e00)  // mova za.d[x9, #0], { z16.d-z19.d }
    b label_6
KAI_ASM_LABEL(label_5)  // Width 1: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_6)  // Width 1: setup done
    cmp x3, #0x8
    ble label_8
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Main loop head
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    ld1rqw { z8.s }, p1/Z, [x4]
    addvl x7, x7, #16
    sub x3, x3, #0x8
    ld1rqw { z11.s }, p0/Z, [x4, #16]
    cmp x3, #0x8
    add x4, x4, #0x20
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aa908)  // bfcvt z8.h, p2/M, z8.s
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aa96b)  // bfcvt z11.h, p2/M, z11.s
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    addvl x7, x7, #16
    uzp1 z8.h, z8.h, z8.h
    uzp1 z11.h, z11.h, z11.h
    trn1 z8.d, z8.d, z11.d
    KAI_ASM_INST(0xc158b098)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z8.h[0]
    KAI_ASM_INST(0xc158b698)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z8.h[1]
    KAI_ASM_INST(0xc158ba18)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z8.h[2]
    KAI_ASM_INST(0xc158bd98)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z8.h[3]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: Single iteration only
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    ld1rqw { z3.s }, p1/Z, [x4]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    ld1rqw { z24.s }, p0/Z, [x4, #16]
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0x658aab18)  // bfcvt z24.h, p2/M, z24.s
    uzp1 z3.h, z3.h, z3.h
    uzp1 z24.h, z24.h, z24.h
    trn1 z3.d, z3.d, z24.d
    KAI_ASM_INST(0xc153b198)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    ble label_9
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b598)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[1]
    ble label_9
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b998)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[2]
    ble label_9
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153bf98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z3.h[3]
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: multiply skip
    tbz x22, #1, label_10
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c1c)  // mova { z28.d-z31.d }, za.d[x9, #0]
    KAI_ASM_INST(0x8540c844)  // ld1rw { z4.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c832)  // ld1rw { z18.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc1b2c89c)  // fclamp { z28.s-z31.s }, z4.s, z18.s
    KAI_ASM_INST(0xa060c0dc)  // st1w { z28.s-z31.s }, p8, [x6]
    b label_11
KAI_ASM_LABEL(label_10)  // Width 1: No activation
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xa060c0c4)  // st1w { z4.s-z7.s }, p8, [x6]
KAI_ASM_LABEL(label_11)  // Width 1: Output done
    b label_36
KAI_ASM_LABEL(label_12)  // Width 2
    ldr x4, [x0, #0x20]
    lsl x2, x8, #0x2
    sub x1, x24, x23
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_13
    KAI_ASM_INST(0xa040c4b0)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c0a0)  // ld1w { z0.s-z3.s }, p8.s/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xc0042e00)  // mova za.d[x9, #0], { z16.d-z19.d }
    KAI_ASM_INST(0xc0042c01)  // mova za.d[x9, #1], { z0.d-z3.d }
    b label_14
KAI_ASM_LABEL(label_13)  // Width 2: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_14)  // Width 2: setup done
    cmp x3, #0x8
    ble label_16
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: Main loop head
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    ld1rqw { z8.s }, p1/Z, [x4]
    sub x3, x3, #0x8
    ld1rqw { z9.s }, p0/Z, [x4, #16]
    cmp x3, #0x8
    add x4, x4, #0x20
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aa908)  // bfcvt z8.h, p2/M, z8.s
    KAI_ASM_INST(0xa040a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0x658aa929)  // bfcvt z9.h, p2/M, z9.s
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    uzp1 z8.h, z8.h, z8.h
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    uzp1 z9.h, z9.h, z9.h
    trn1 z8.d, z8.d, z9.d
    KAI_ASM_INST(0xc158b298)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z8.h[0]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc158b319)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z8.h[0]
    KAI_ASM_INST(0xc158b418)  // bfdot za.s[x9, 0], { z0.h-z3.h }, z8.h[1]
    KAI_ASM_INST(0xc158b619)  // bfdot za.s[x9, 1], { z16.h-z19.h }, z8.h[1]
    KAI_ASM_INST(0xc158b898)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z8.h[2]
    KAI_ASM_INST(0xc158b999)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z8.h[2]
    KAI_ASM_INST(0xc158bf98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z8.h[3]
    KAI_ASM_INST(0xc158be99)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z8.h[3]
    bgt label_15
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: Single iteration only
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    ld1rqw { z3.s }, p1/Z, [x4]
    subs x3, x3, #0x2
    ld1rqw { z16.s }, p0/Z, [x4, #16]
    KAI_ASM_INST(0xa041a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0x658aaa10)  // bfcvt z16.h, p2/M, z16.s
    uzp1 z3.h, z3.h, z3.h
    uzp1 z16.h, z16.h, z16.h
    trn1 z3.d, z3.d, z16.d
    KAI_ASM_INST(0xc153b318)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z3.h[0]
    KAI_ASM_INST(0xc153b099)  // bfdot za.s[x9, 1], { z4.h-z7.h }, z3.h[0]
    ble label_17
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b498)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[1]
    KAI_ASM_INST(0xc153b719)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z3.h[1]
    ble label_17
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b898)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[2]
    KAI_ASM_INST(0xc153b999)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[2]
    ble label_17
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153bc98)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[3]
    KAI_ASM_INST(0xc153be99)  // bfdot za.s[x9, 1], { z20.h-z23.h }, z3.h[3]
KAI_ASM_LABEL(label_17)  // Width 2: Multiply loop: multiply skip
    tbz x22, #1, label_18
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c2c)  // mova { z12.d-z15.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c843)  // ld1rw { z3.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c83c)  // ld1rw { z28.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc1bcc868)  // fclamp { z8.s-z11.s }, z3.s, z28.s
    KAI_ASM_INST(0xc1bcc86c)  // fclamp { z12.s-z15.s }, z3.s, z28.s
    KAI_ASM_INST(0xa060c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c0cc)  // st1w { z12.s-z15.s }, p8, [x6, #0x4, MUL VL]
    b label_19
KAI_ASM_LABEL(label_18)  // Width 2: No activation
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c38)  // mova { z24.d-z27.d }, za.d[x9, #1]
    KAI_ASM_INST(0xa060c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c0d8)  // st1w { z24.s-z27.s }, p8, [x6, #0x4, MUL VL]
KAI_ASM_LABEL(label_19)  // Width 2: Output done
    b label_36
KAI_ASM_LABEL(label_20)  // Width 3
    ldr x4, [x0, #0x20]
    mov x1, #0x2
    lsl x2, x8, #0x2
    msub x1, x23, x1, x24
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_21
    KAI_ASM_INST(0xa040c4a0)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4b8)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c0bc)  // ld1w { z28.s-z31.s }, p8.s/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xc0042c00)  // mova za.d[x9, #0], { z0.d-z3.d }
    KAI_ASM_INST(0xc0042f01)  // mova za.d[x9, #1], { z24.d-z27.d }
    KAI_ASM_INST(0xc0042f82)  // mova za.d[x9, #2], { z28.d-z31.d }
    b label_22
KAI_ASM_LABEL(label_21)  // Width 3: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_22)  // Width 3: setup done
    cmp x3, #0x8
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: Main loop head
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    ld1rqw { z3.s }, p1/Z, [x4]
    sub x3, x3, #0x8
    ld1rqw { z0.s }, p0/Z, [x4, #16]
    cmp x3, #0x8
    add x4, x4, #0x20
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0xa042a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aa800)  // bfcvt z0.h, p2/M, z0.s
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    uzp1 z3.h, z3.h, z3.h
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    uzp1 z0.h, z0.h, z0.h
    trn1 z3.d, z3.d, z0.d
    KAI_ASM_INST(0xc153b198)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b319)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z3.h[0]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b09a)  // bfdot za.s[x9, 2], { z4.h-z7.h }, z3.h[0]
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc153b618)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z3.h[1]
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc153b799)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[1]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b51a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xc153ba98)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z3.h[2]
    KAI_ASM_INST(0xc153b999)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[2]
    KAI_ASM_INST(0xc153bb1a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z3.h[2]
    KAI_ASM_INST(0xc153bc98)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[3]
    KAI_ASM_INST(0xc153be19)  // bfdot za.s[x9, 1], { z16.h-z19.h }, z3.h[3]
    KAI_ASM_INST(0xc153bf9a)  // bfdot za.s[x9, 2], { z28.h-z31.h }, z3.h[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 3: Multiply loop: Single iteration only
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4e5)  // ldnt1h { z4.h-z7.h }, pn9.b/Z, [x7]
    ld1rqw { z3.s }, p1/Z, [x4]
    subs x3, x3, #0x2
    ld1rqw { z20.s }, p0/Z, [x4, #16]
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0x658aaa94)  // bfcvt z20.h, p2/M, z20.s
    uzp1 z3.h, z3.h, z3.h
    uzp1 z20.h, z20.h, z20.h
    trn1 z3.d, z3.d, z20.d
    KAI_ASM_INST(0xc153b098)  // bfdot za.s[x9, 0], { z4.h-z7.h }, z3.h[0]
    KAI_ASM_INST(0xc153b199)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xc153b31a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z3.h[0]
    ble label_25
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b698)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z3.h[1]
    KAI_ASM_INST(0xc153b619)  // bfdot za.s[x9, 1], { z16.h-z19.h }, z3.h[1]
    KAI_ASM_INST(0xc153b79a)  // bfdot za.s[x9, 2], { z28.h-z31.h }, z3.h[1]
    ble label_25
    KAI_ASM_INST(0xa040a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bb98)  // bfdot za.s[x9, 0], { z28.h-z31.h }, z3.h[2]
    KAI_ASM_INST(0xc153b919)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xc153b99a)  // bfdot za.s[x9, 2], { z12.h-z15.h }, z3.h[2]
    ble label_25
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc153bd98)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[3]
    KAI_ASM_INST(0xc153bf19)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z3.h[3]
    KAI_ASM_INST(0xc153bd1a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[3]
KAI_ASM_LABEL(label_25)  // Width 3: Multiply loop: multiply skip
    tbz x22, #1, label_26
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c851)  // ld1rw { z17.s }, p2/Z, [x2]
    KAI_ASM_INST(0xc0062c44)  // mova { z4.d-z7.d }, za.d[x9, #2]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc1b0ca28)  // fclamp { z8.s-z11.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca20)  // fclamp { z0.s-z3.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca24)  // fclamp { z4.s-z7.s }, z17.s, z16.s
    KAI_ASM_INST(0xa060c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c0)  // st1w { z0.s-z3.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c0c4)  // st1w { z4.s-z7.s }, p8, [x6, #0x8, MUL VL]
    b label_27
KAI_ASM_LABEL(label_26)  // Width 3: No activation
    KAI_ASM_INST(0xc0062c08)  // mova { z8.d-z11.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c20)  // mova { z0.d-z3.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c50)  // mova { z16.d-z19.d }, za.d[x9, #2]
    KAI_ASM_INST(0xa060c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c0)  // st1w { z0.s-z3.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c0d0)  // st1w { z16.s-z19.s }, p8, [x6, #0x8, MUL VL]
KAI_ASM_LABEL(label_27)  // Width 3: Output done
    b label_36
KAI_ASM_LABEL(label_28)  // Width 4
    ldr x4, [x0, #0x20]
    mov x1, #0x3
    lsl x2, x8, #0x2
    msub x1, x23, x1, x24
    mov x3, x8
    KAI_ASM_INST(0x25a167f0)  // whilelt p8.s, XZR, x1, VLx4
    KAI_ASM_INST(0xf8a24898)  // rprfm pldmany, x2, [x4]
    cbz x5, label_29
    KAI_ASM_INST(0xa040c4ac)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x5]
    KAI_ASM_INST(0xa041c4a4)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x5, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c4b4)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x5, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c0b0)  // ld1w { z16.s-z19.s }, p8.s/Z, [x5, #0xc, MUL VL]
    KAI_ASM_INST(0xc0042d80)  // mova za.d[x9, #0], { z12.d-z15.d }
    addvl x5, x5, #16
    KAI_ASM_INST(0xc0042c81)  // mova za.d[x9, #1], { z4.d-z7.d }
    KAI_ASM_INST(0xc0042e82)  // mova za.d[x9, #2], { z20.d-z23.d }
    KAI_ASM_INST(0xc0042e03)  // mova za.d[x9, #3], { z16.d-z19.d }
    b label_30
KAI_ASM_LABEL(label_29)  // Width 4: no bias
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
KAI_ASM_LABEL(label_30)  // Width 4: setup done
    cmp x3, #0x8
    ble label_32
KAI_ASM_LABEL(label_31)  // Width 4: Multiply loop: Main loop head
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7]
    ld1rqw { z7.s }, p1/Z, [x4]
    sub x3, x3, #0x8
    ld1rqw { z4.s }, p0/Z, [x4, #16]
    cmp x3, #0x8
    add x4, x4, #0x20
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0x658aa8e7)  // bfcvt z7.h, p2/M, z7.s
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0x658aa884)  // bfcvt z4.h, p2/M, z4.s
    KAI_ASM_INST(0xa043a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xa040a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    uzp1 z7.h, z7.h, z7.h
    KAI_ASM_INST(0xa042a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    uzp1 z4.h, z4.h, z4.h
    trn1 z7.d, z7.d, z4.d
    KAI_ASM_INST(0xc157b218)  // bfdot za.s[x9, 0], { z16.h-z19.h }, z7.h[0]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc157b199)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z7.h[0]
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc157b39a)  // bfdot za.s[x9, 2], { z28.h-z31.h }, z7.h[0]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc157b31b)  // bfdot za.s[x9, 3], { z24.h-z27.h }, z7.h[0]
    KAI_ASM_INST(0xa042a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc157b698)  // bfdot za.s[x9, 0], { z20.h-z23.h }, z7.h[1]
    KAI_ASM_INST(0xa043a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc157b519)  // bfdot za.s[x9, 1], { z8.h-z11.h }, z7.h[1]
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xc157b41a)  // bfdot za.s[x9, 2], { z0.h-z3.h }, z7.h[1]
    KAI_ASM_INST(0xa041a4e1)  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xc157b61b)  // bfdot za.s[x9, 3], { z16.h-z19.h }, z7.h[1]
    KAI_ASM_INST(0xa042a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xc157b998)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z7.h[2]
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc157bb99)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z7.h[2]
    KAI_ASM_INST(0xc157bb1a)  // bfdot za.s[x9, 2], { z24.h-z27.h }, z7.h[2]
    KAI_ASM_INST(0xc157ba9b)  // bfdot za.s[x9, 3], { z20.h-z23.h }, z7.h[2]
    KAI_ASM_INST(0xc157bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z7.h[3]
    KAI_ASM_INST(0xc157bc19)  // bfdot za.s[x9, 1], { z0.h-z3.h }, z7.h[3]
    KAI_ASM_INST(0xc157be1a)  // bfdot za.s[x9, 2], { z16.h-z19.h }, z7.h[3]
    KAI_ASM_INST(0xc157bd9b)  // bfdot za.s[x9, 3], { z12.h-z15.h }, z7.h[3]
    bgt label_31
KAI_ASM_LABEL(label_32)  // Width 4: Multiply loop: Single iteration only
    whilelt p1.s, XZR, x3
    whilelt p0.s, x21, x3
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    ld1rqw { z3.s }, p1/Z, [x4]
    subs x3, x3, #0x2
    ld1rqw { z16.s }, p0/Z, [x4, #16]
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0x658aa863)  // bfcvt z3.h, p2/M, z3.s
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0x658aaa10)  // bfcvt z16.h, p2/M, z16.s
    KAI_ASM_INST(0xa043a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    addvl x7, x7, #16
    uzp1 z3.h, z3.h, z3.h
    uzp1 z16.h, z16.h, z16.h
    trn1 z3.d, z3.d, z16.d
    KAI_ASM_INST(0xc153b318)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z3.h[0]
    KAI_ASM_INST(0xc153b399)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[0]
    KAI_ASM_INST(0xc153b11a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[0]
    KAI_ASM_INST(0xc153b19b)  // bfdot za.s[x9, 3], { z12.h-z15.h }, z3.h[0]
    ble label_33
    KAI_ASM_INST(0xa040a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153b598)  // bfdot za.s[x9, 0], { z12.h-z15.h }, z3.h[1]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b799)  // bfdot za.s[x9, 1], { z28.h-z31.h }, z3.h[1]
    KAI_ASM_INST(0xc153b51a)  // bfdot za.s[x9, 2], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xc153b71b)  // bfdot za.s[x9, 3], { z24.h-z27.h }, z3.h[1]
    ble label_33
    KAI_ASM_INST(0xa040a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7]
    subs x3, x3, #0x2
    KAI_ASM_INST(0xa041a4ed)  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4fd)  // ldnt1h { z28.h-z31.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bb18)  // bfdot za.s[x9, 0], { z24.h-z27.h }, z3.h[2]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153b999)  // bfdot za.s[x9, 1], { z12.h-z15.h }, z3.h[2]
    KAI_ASM_INST(0xc153bb9a)  // bfdot za.s[x9, 2], { z28.h-z31.h }, z3.h[2]
    KAI_ASM_INST(0xc153ba1b)  // bfdot za.s[x9, 3], { z16.h-z19.h }, z3.h[2]
    ble label_33
    KAI_ASM_INST(0xa040a4e9)  // ldnt1h { z8.h-z11.h }, pn9.b/Z, [x7]
    KAI_ASM_INST(0xa041a4f9)  // ldnt1h { z24.h-z27.h }, pn9.b/Z, [x7, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a4f5)  // ldnt1h { z20.h-z23.h }, pn9.b/Z, [x7, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a4f1)  // ldnt1h { z16.h-z19.h }, pn9.b/Z, [x7, #0xc, MUL VL]
    KAI_ASM_INST(0xc153bd18)  // bfdot za.s[x9, 0], { z8.h-z11.h }, z3.h[3]
    addvl x7, x7, #16
    KAI_ASM_INST(0xc153bf19)  // bfdot za.s[x9, 1], { z24.h-z27.h }, z3.h[3]
    KAI_ASM_INST(0xc153be9a)  // bfdot za.s[x9, 2], { z20.h-z23.h }, z3.h[3]
    KAI_ASM_INST(0xc153be1b)  // bfdot za.s[x9, 3], { z16.h-z19.h }, z3.h[3]
KAI_ASM_LABEL(label_33)  // Width 4: Multiply loop: multiply skip
    tbz x22, #1, label_34
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0xc0062c04)  // mova { z4.d-z7.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c28)  // mova { z8.d-z11.d }, za.d[x9, #1]
    KAI_ASM_INST(0x8540c855)  // ld1rw { z21.s }, p2/Z, [x2]
    KAI_ASM_INST(0xc0062c4c)  // mova { z12.d-z15.d }, za.d[x9, #2]
    KAI_ASM_INST(0x8540c834)  // ld1rw { z20.s }, p2/Z, [x1]
    KAI_ASM_INST(0xc0062c70)  // mova { z16.d-z19.d }, za.d[x9, #3]
    KAI_ASM_INST(0xc1b4caa4)  // fclamp { z4.s-z7.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caa8)  // fclamp { z8.s-z11.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caac)  // fclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cab0)  // fclamp { z16.s-z19.s }, z21.s, z20.s
    KAI_ASM_INST(0xa060c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c8)  // st1w { z8.s-z11.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c0d0)  // st1w { z16.s-z19.s }, p8, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
    b label_35
KAI_ASM_LABEL(label_34)  // Width 4: No activation
    KAI_ASM_INST(0xc0062c00)  // mova { z0.d-z3.d }, za.d[x9, #0]
    KAI_ASM_INST(0xc0062c24)  // mova { z4.d-z7.d }, za.d[x9, #1]
    KAI_ASM_INST(0xc0062c4c)  // mova { z12.d-z15.d }, za.d[x9, #2]
    KAI_ASM_INST(0xc0062c78)  // mova { z24.d-z27.d }, za.d[x9, #3]
    KAI_ASM_INST(0xa060c4c0)  // st1w { z0.s-z3.s }, pn9.b, [x6]
    KAI_ASM_INST(0xa061c4c4)  // st1w { z4.s-z7.s }, pn9.b, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c4cc)  // st1w { z12.s-z15.s }, pn9.b, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c0d8)  // st1w { z24.s-z27.s }, p8, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
KAI_ASM_LABEL(label_35)  // Width 4: Output done
    subs x20, x20, #0x4
    sub x24, x24, x23, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_36)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_gemv_fp32bf16fp32_dot_16VL)

    KAI_ASM_END
