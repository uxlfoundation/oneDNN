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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x10, #0
    ldr x1, [x0, #0x60]
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    mov x6, #0xff800000
    ldr x2, [x0, #0x50]
    mov x5, #0x7f800000
    ptrue p1.b
    ldr w4, [x0, #0xa0]
    cmp x1, XZR
    ldr w3, [x0, #0xa4]
    cset x1, NE
    cmp x2, XZR
    ldr x9, [x0, #0x68]
    orr x10, x10, x1
    cset x1, EQ
    ldr x2, [x0, #0x70]
    cmp x4, x6
    orr x10, x10, x1, LSL #1
    cset x1, NE
    cmp x3, x5
    orr x10, x10, x1, LSL #2
    cset x1, NE
    orr x10, x10, x1, LSL #2
    mov x28, x9
    add x27, x9, x2
    tbz x10, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c120)  // ld1w { z0.s-z3.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041c12c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c134)  // ld1w { z20.s-z23.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c128)  // ld1w { z8.s-z11.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840682)  // mova za2h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x26, [x0, #0x10]
    mov x25, #0
    mov x24, #0
    ldr x23, [x0, #0]
    ldr x22, [x0, #0x8]
    add x26, x26, #0x1
    ldr x21, [x0, #0x20]
    lsr x26, x26, #0x1
KAI_ASM_LABEL(label_3)  // M loop
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x8, x21
    whilelt p0.s, x24, x22
    tbnz x10, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z28.s, #1.0
    ld1w { z18.s }, p0/Z, [x1, x24, LSL #2]
    KAI_ASM_INST(0x80922780)  // fmopa za0.s, p1/M, p1/M, z28.s, z18.s
    KAI_ASM_INST(0x80922781)  // fmopa za1.s, p1/M, p1/M, z28.s, z18.s
    KAI_ASM_INST(0x80922782)  // fmopa za2.s, p1/M, p1/M, z28.s, z18.s
    KAI_ASM_INST(0x80922783)  // fmopa za3.s, p1/M, p1/M, z28.s, z18.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x24
    mov x2, x25
    incw x1
    incw x2, ALL, MUL #4
    cmp x1, x22
    mov x1, x10
    csel x2, x25, x2, LT
    bfm x10, XZR, #0, #0  // bfc x10, #0, #0x1
    csel x9, x9, x27, LT
    cmp x2, x23
    csel x10, x1, x10, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x26, #0x2
    and x1, x26, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa140a100)  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa142a103)  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn8.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa143a113)  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa140a282)  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81a22400)  // fmopa za0.s, p1/M, p1/M, z0.h, z2.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81a22481)  // fmopa za1.s, p1/M, p1/M, z4.h, z2.h
    KAI_ASM_INST(0x81a22502)  // fmopa za2.s, p1/M, p1/M, z8.h, z2.h
    KAI_ASM_INST(0x81a22583)  // fmopa za3.s, p1/M, p1/M, z12.h, z2.h
    KAI_ASM_INST(0xa140a100)  // ld1h { z0.h, z4.h, z8.h, z12.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0x81a62640)  // fmopa za0.s, p1/M, p1/M, z18.h, z6.h
    KAI_ASM_INST(0x81a626c1)  // fmopa za1.s, p1/M, p1/M, z22.h, z6.h
    KAI_ASM_INST(0x81a62742)  // fmopa za2.s, p1/M, p1/M, z26.h, z6.h
    KAI_ASM_INST(0x81a627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z6.h
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0x81aa2460)  // fmopa za0.s, p1/M, p1/M, z3.h, z10.h
    KAI_ASM_INST(0x81aa24e1)  // fmopa za1.s, p1/M, p1/M, z7.h, z10.h
    KAI_ASM_INST(0x81aa2562)  // fmopa za2.s, p1/M, p1/M, z11.h, z10.h
    KAI_ASM_INST(0x81aa25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z10.h
    KAI_ASM_INST(0xa142a103)  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn8.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0x81ae2660)  // fmopa za0.s, p1/M, p1/M, z19.h, z14.h
    KAI_ASM_INST(0x81ae26e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z14.h
    KAI_ASM_INST(0x81ae2762)  // fmopa za2.s, p1/M, p1/M, z27.h, z14.h
    KAI_ASM_INST(0x81ae27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z14.h
    KAI_ASM_INST(0xa143a113)  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa140a282)  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81a22400)  // fmopa za0.s, p1/M, p1/M, z0.h, z2.h
    KAI_ASM_INST(0x81a22481)  // fmopa za1.s, p1/M, p1/M, z4.h, z2.h
    KAI_ASM_INST(0x81a22502)  // fmopa za2.s, p1/M, p1/M, z8.h, z2.h
    KAI_ASM_INST(0x81a22583)  // fmopa za3.s, p1/M, p1/M, z12.h, z2.h
    KAI_ASM_INST(0x81a62640)  // fmopa za0.s, p1/M, p1/M, z18.h, z6.h
    KAI_ASM_INST(0x81a626c1)  // fmopa za1.s, p1/M, p1/M, z22.h, z6.h
    KAI_ASM_INST(0x81a62742)  // fmopa za2.s, p1/M, p1/M, z26.h, z6.h
    KAI_ASM_INST(0x81a627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z6.h
    KAI_ASM_INST(0x81aa2460)  // fmopa za0.s, p1/M, p1/M, z3.h, z10.h
    KAI_ASM_INST(0x81aa24e1)  // fmopa za1.s, p1/M, p1/M, z7.h, z10.h
    KAI_ASM_INST(0x81aa2562)  // fmopa za2.s, p1/M, p1/M, z11.h, z10.h
    KAI_ASM_INST(0x81aa25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z10.h
    KAI_ASM_INST(0x81ae2660)  // fmopa za0.s, p1/M, p1/M, z19.h, z14.h
    KAI_ASM_INST(0x81ae26e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z14.h
    KAI_ASM_INST(0x81ae2762)  // fmopa za2.s, p1/M, p1/M, z27.h, z14.h
    KAI_ASM_INST(0x81ae27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z14.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa040a104)  // ld1h { z4.h-z7.h }, pn8.b/Z, [x8]
    subs x1, x1, #0x1
    addvl x8, x8, #4
    ld1h { z27.h }, p1/Z, [x20]
    addvl x20, x20, #1
    KAI_ASM_INST(0x81bb2480)  // fmopa za0.s, p1/M, p1/M, z4.h, z27.h
    KAI_ASM_INST(0x81bb24a1)  // fmopa za1.s, p1/M, p1/M, z5.h, z27.h
    KAI_ASM_INST(0x81bb24c2)  // fmopa za2.s, p1/M, p1/M, z6.h, z27.h
    KAI_ASM_INST(0x81bb24e3)  // fmopa za3.s, p1/M, p1/M, z7.h, z27.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x10, #1, label_15
    tbz x10, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c134)  // ld1w { z20.s-z23.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c130)  // ld1w { z16.s-z19.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c120)  // ld1w { z0.s-z3.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c12c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840601)  // mova za1h.s[x12], { z16.s-z19.s }
    KAI_ASM_INST(0xa060c388)  // st1w { z8.s-z11.s }, pn8.b, [x28]
    KAI_ASM_INST(0xc0840402)  // mova za2h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xa061c39c)  // st1w { z28.s-z31.s }, pn8.b, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840583)  // mova za3h.s[x12], { z12.s-z15.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c398)  // st1w { z24.s-z27.s }, pn8.b, [x28, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c384)  // st1w { z4.s-z7.s }, pn8.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_12
    b label_43
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c390)  // st1w { z16.s-z19.s }, pn8.b, [x28]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c38c)  // st1w { z12.s-z15.s }, pn8.b, [x28, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c384)  // st1w { z4.s-z7.s }, pn8.b, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c388)  // st1w { z8.s-z11.s }, pn8.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_14
    b label_43
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x23, x25
    ldr x5, [x0, #0x58]
    add x7, x7, x24, LSL #2  // C += n
    madd x7, x25, x5, x7  // C += m * ldc
    tbnz x10, #2, label_28
    cntw x4
    mov x12, #0
    cmp x6, x4
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Skip activation: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    add x12, x12, #0x4
    st1w { z0.s }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z1.s }, p0, [x7]
    add x7, x7, x5
    st1w { z2.s }, p0, [x7]
    add x7, x7, x5
    st1w { z3.s }, p0, [x7]
    add x7, x7, x5
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Skip activation: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    st1w { z8.s }, p0, [x7]
    add x7, x7, x5
    beq label_18
    subs x1, x1, #0x1
    st1w { z9.s }, p0, [x7]
    add x7, x7, x5
    beq label_18
    st1w { z10.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_18)  // Store to output array: Skip activation: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_20
KAI_ASM_LABEL(label_19)  // Store to output array: Skip activation: Accumulator row 1 loop
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    add x12, x12, #0x4
    st1w { z12.s }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z13.s }, p0, [x7]
    add x7, x7, x5
    st1w { z14.s }, p0, [x7]
    add x7, x7, x5
    st1w { z15.s }, p0, [x7]
    add x7, x7, x5
    blt label_19
KAI_ASM_LABEL(label_20)  // Store to output array: Skip activation: Accumulator row 1 oddments
    cbz x1, label_21
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    st1w { z8.s }, p0, [x7]
    add x7, x7, x5
    beq label_21
    subs x1, x1, #0x1
    st1w { z9.s }, p0, [x7]
    add x7, x7, x5
    beq label_21
    st1w { z10.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_21)  // Store to output array: Skip activation: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_23
KAI_ASM_LABEL(label_22)  // Store to output array: Skip activation: Accumulator row 2 loop
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    add x12, x12, #0x4
    st1w { z20.s }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z21.s }, p0, [x7]
    add x7, x7, x5
    st1w { z22.s }, p0, [x7]
    add x7, x7, x5
    st1w { z23.s }, p0, [x7]
    add x7, x7, x5
    blt label_22
KAI_ASM_LABEL(label_23)  // Store to output array: Skip activation: Accumulator row 2 oddments
    cbz x1, label_24
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    st1w { z12.s }, p0, [x7]
    add x7, x7, x5
    beq label_24
    subs x1, x1, #0x1
    st1w { z13.s }, p0, [x7]
    add x7, x7, x5
    beq label_24
    st1w { z14.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_24)  // Store to output array: Skip activation: Accumulator row 2 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_26
KAI_ASM_LABEL(label_25)  // Store to output array: Skip activation: Accumulator row 3 loop
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    add x12, x12, #0x4
    st1w { z24.s }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z25.s }, p0, [x7]
    add x7, x7, x5
    st1w { z26.s }, p0, [x7]
    add x7, x7, x5
    st1w { z27.s }, p0, [x7]
    add x7, x7, x5
    blt label_25
KAI_ASM_LABEL(label_26)  // Store to output array: Skip activation: Accumulator row 3 oddments
    cbz x1, label_27
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    beq label_27
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_27
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_27)  // Store to output array: Skip activation: Accumulator row 3 oddments: End
    subs x6, x6, x3
    beq label_28
    b label_41
KAI_ASM_LABEL(label_28)  // Store to output array: Skip activation: End
    cntw x4
    add x2, x0, #0xa0
    cmp x6, x4
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c459)  // ld1rw { z25.s }, p1/Z, [x2]
    csel x3, x6, x4, LT
    KAI_ASM_INST(0x8540c438)  // ld1rw { z24.s }, p1/Z, [x1]
    mov x12, #0
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_30
KAI_ASM_LABEL(label_29)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860414)  // mova { z20.s-z23.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b8cb34)  // fclamp { z20.s-z23.s }, z25.s, z24.s
    cmp x12, x2, LSL #2
    st1w { z20.s }, p0, [x7]
    add x7, x7, x5
    st1w { z21.s }, p0, [x7]
    add x7, x7, x5
    st1w { z22.s }, p0, [x7]
    add x7, x7, x5
    st1w { z23.s }, p0, [x7]
    add x7, x7, x5
    blt label_29
KAI_ASM_LABEL(label_30)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_31
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1b8cb30)  // fclamp { z16.s-z19.s }, z25.s, z24.s
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    beq label_31
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_31
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_31)  // Store to output array: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_41
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_33
KAI_ASM_LABEL(label_32)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b8cb30)  // fclamp { z16.s-z19.s }, z25.s, z24.s
    cmp x12, x2, LSL #2
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    blt label_32
KAI_ASM_LABEL(label_33)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_34
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1b8cb30)  // fclamp { z16.s-z19.s }, z25.s, z24.s
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    beq label_34
    subs x1, x1, #0x1
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    beq label_34
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_34)  // Store to output array: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_41
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_36
KAI_ASM_LABEL(label_35)  // Store to output array: Accumulator row 2 loop
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b8cb30)  // fclamp { z16.s-z19.s }, z25.s, z24.s
    cmp x12, x2, LSL #2
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    blt label_35
KAI_ASM_LABEL(label_36)  // Store to output array: Accumulator row 2 oddments
    cbz x1, label_37
    KAI_ASM_INST(0xc086045c)  // mova { z28.s-z31.s }, za2h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1b8cb3c)  // fclamp { z28.s-z31.s }, z25.s, z24.s
    st1w { z28.s }, p0, [x7]
    add x7, x7, x5
    beq label_37
    subs x1, x1, #0x1
    st1w { z29.s }, p0, [x7]
    add x7, x7, x5
    beq label_37
    st1w { z30.s }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_37)  // Store to output array: Accumulator row 2 oddments: End
    subs x6, x6, x3
    beq label_41
    cmp x6, x4
    mov x12, #0
    csel x1, x6, x4, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_39
KAI_ASM_LABEL(label_38)  // Store to output array: Accumulator row 3 loop
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b8cb30)  // fclamp { z16.s-z19.s }, z25.s, z24.s
    cmp x12, x2, LSL #2
    st1w { z16.s }, p0, [x7]
    add x7, x7, x5
    st1w { z17.s }, p0, [x7]
    add x7, x7, x5
    st1w { z18.s }, p0, [x7]
    add x7, x7, x5
    st1w { z19.s }, p0, [x7]
    add x7, x7, x5
    blt label_38
KAI_ASM_LABEL(label_39)  // Store to output array: Accumulator row 3 oddments
    cbz x1, label_40
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1b8cb2c)  // fclamp { z12.s-z15.s }, z25.s, z24.s
    st1w { z12.s }, p0, [x7]
    add x7, x7, x5
    beq label_40
    subs x1, x1, #0x1
    st1w { z13.s }, p0, [x7]
    add x7, x7, x5
    beq label_40
    st1w { z14.s }, p0, [x7]
KAI_ASM_LABEL(label_40)  // Store to output array: Accumulator row 3 oddments: End
KAI_ASM_LABEL(label_41)  // Store to output array: End
    tbz x10, #0, label_43
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_42)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c12c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041c13c)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c124)  // ld1w { z4.s-z7.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c128)  // ld1w { z8.s-z11.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840580)  // mova za0h.s[x12], { z12.s-z15.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840781)  // mova za1h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_42
KAI_ASM_LABEL(label_43)  // End block
    incw x24
    cmp x24, x22
    blt label_4
    ldr x1, [x0, #0x70]
    incw x25, ALL, MUL #4
    mov x28, x27
    cmp x25, x23
    mov x24, #0
    mov x21, x8
    add x27, x28, x1
    blt label_3
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL)

    KAI_ASM_END
