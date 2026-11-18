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

    KAI_ASM_CODE(sme2_interleaved_nomerge_u8q_mopa_4VLx1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_4VLx1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_u8q_mopa_4VLx1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_4VLx1VL)
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
    ptrue p1.b
    ldr x2, [x0, #0x50]
    ldr x3, [x0, #0x78]
    cmp x1, XZR
    ldr x9, [x0, #0x68]
    cset x1, NE
    cmp x2, XZR
    ldr x2, [x0, #0x70]
    orr x10, x10, x1
    cset x1, EQ
    cmp x3, XZR
    orr x10, x10, x1, LSL #1
    cset x1, NE
    mov x28, x9
    orr x10, x10, x1, LSL #2
    add x27, x9, x2
    tbz x10, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c120)  // ld1w { z0.s-z3.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041c138)  // ld1w { z24.s-z27.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c130)  // ld1w { z16.s-z19.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c128)  // ld1w { z8.s-z11.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840701)  // mova za1h.s[x12], { z24.s-z27.s }
    KAI_ASM_INST(0xc0840602)  // mova za2h.s[x12], { z16.s-z19.s }
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
    add x26, x26, #0x3
    ldr x21, [x0, #0x20]
    lsr x26, x26, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x8, x21
    whilelt p0.s, x24, x22
    tbnz x10, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    ld1w { z6.s }, p0/Z, [x1, x24, LSL #2]
    KAI_ASM_INST(0xc09024c0)  // addha za0.s, p1/M, p1/M, z6.s
    KAI_ASM_INST(0xc09024c1)  // addha za1.s, p1/M, p1/M, z6.s
    KAI_ASM_INST(0xc09024c2)  // addha za2.s, p1/M, p1/M, z6.s
    KAI_ASM_INST(0xc09024c3)  // addha za3.s, p1/M, p1/M, z6.s
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
    KAI_ASM_INST(0xa1408112)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x8]
    KAI_ASM_INST(0xa0418100)  // ld1b { z0.b-z3.b }, pn8.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa0428108)  // ld1b { z8.b-z11.b }, pn8.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa0438104)  // ld1b { z4.b-z7.b }, pn8.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa040828c)  // ld1b { z12.b-z15.b }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa1ac2640)  // umopa za0.s, p1/M, p1/M, z18.b, z12.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa1ac26c1)  // umopa za1.s, p1/M, p1/M, z22.b, z12.b
    KAI_ASM_INST(0xa1ac2742)  // umopa za2.s, p1/M, p1/M, z26.b, z12.b
    KAI_ASM_INST(0xa1ac27c3)  // umopa za3.s, p1/M, p1/M, z30.b, z12.b
    KAI_ASM_INST(0xa1408112)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x8]
    KAI_ASM_INST(0xa1ad2400)  // umopa za0.s, p1/M, p1/M, z0.b, z13.b
    KAI_ASM_INST(0xa1ad2421)  // umopa za1.s, p1/M, p1/M, z1.b, z13.b
    KAI_ASM_INST(0xa1ad2442)  // umopa za2.s, p1/M, p1/M, z2.b, z13.b
    KAI_ASM_INST(0xa1ad2463)  // umopa za3.s, p1/M, p1/M, z3.b, z13.b
    KAI_ASM_INST(0xa0418100)  // ld1b { z0.b-z3.b }, pn8.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa1ae2500)  // umopa za0.s, p1/M, p1/M, z8.b, z14.b
    KAI_ASM_INST(0xa1ae2521)  // umopa za1.s, p1/M, p1/M, z9.b, z14.b
    KAI_ASM_INST(0xa1ae2542)  // umopa za2.s, p1/M, p1/M, z10.b, z14.b
    KAI_ASM_INST(0xa1ae2563)  // umopa za3.s, p1/M, p1/M, z11.b, z14.b
    KAI_ASM_INST(0xa0428108)  // ld1b { z8.b-z11.b }, pn8.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa1af2480)  // umopa za0.s, p1/M, p1/M, z4.b, z15.b
    KAI_ASM_INST(0xa1af24a1)  // umopa za1.s, p1/M, p1/M, z5.b, z15.b
    KAI_ASM_INST(0xa1af24c2)  // umopa za2.s, p1/M, p1/M, z6.b, z15.b
    KAI_ASM_INST(0xa1af24e3)  // umopa za3.s, p1/M, p1/M, z7.b, z15.b
    KAI_ASM_INST(0xa0438104)  // ld1b { z4.b-z7.b }, pn8.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa040828c)  // ld1b { z12.b-z15.b }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa1ac2640)  // umopa za0.s, p1/M, p1/M, z18.b, z12.b
    KAI_ASM_INST(0xa1ac26c1)  // umopa za1.s, p1/M, p1/M, z22.b, z12.b
    KAI_ASM_INST(0xa1ac2742)  // umopa za2.s, p1/M, p1/M, z26.b, z12.b
    KAI_ASM_INST(0xa1ac27c3)  // umopa za3.s, p1/M, p1/M, z30.b, z12.b
    KAI_ASM_INST(0xa1ad2400)  // umopa za0.s, p1/M, p1/M, z0.b, z13.b
    KAI_ASM_INST(0xa1ad2421)  // umopa za1.s, p1/M, p1/M, z1.b, z13.b
    KAI_ASM_INST(0xa1ad2442)  // umopa za2.s, p1/M, p1/M, z2.b, z13.b
    KAI_ASM_INST(0xa1ad2463)  // umopa za3.s, p1/M, p1/M, z3.b, z13.b
    KAI_ASM_INST(0xa1ae2500)  // umopa za0.s, p1/M, p1/M, z8.b, z14.b
    KAI_ASM_INST(0xa1ae2521)  // umopa za1.s, p1/M, p1/M, z9.b, z14.b
    KAI_ASM_INST(0xa1ae2542)  // umopa za2.s, p1/M, p1/M, z10.b, z14.b
    KAI_ASM_INST(0xa1ae2563)  // umopa za3.s, p1/M, p1/M, z11.b, z14.b
    KAI_ASM_INST(0xa1af2480)  // umopa za0.s, p1/M, p1/M, z4.b, z15.b
    KAI_ASM_INST(0xa1af24a1)  // umopa za1.s, p1/M, p1/M, z5.b, z15.b
    KAI_ASM_INST(0xa1af24c2)  // umopa za2.s, p1/M, p1/M, z6.b, z15.b
    KAI_ASM_INST(0xa1af24e3)  // umopa za3.s, p1/M, p1/M, z7.b, z15.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa1408112)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x8]
    subs x1, x1, #0x1
    addvl x8, x8, #4
    ld1b { z15.b }, p1/Z, [x20]
    addvl x20, x20, #1
    KAI_ASM_INST(0xa1af2640)  // umopa za0.s, p1/M, p1/M, z18.b, z15.b
    KAI_ASM_INST(0xa1af26c1)  // umopa za1.s, p1/M, p1/M, z22.b, z15.b
    KAI_ASM_INST(0xa1af2742)  // umopa za2.s, p1/M, p1/M, z26.b, z15.b
    KAI_ASM_INST(0xa1af27c3)  // umopa za3.s, p1/M, p1/M, z30.b, z15.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    KAI_ASM_INST(0xa140c103)  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn8.b/Z, [x8]
    addvl x8, x8, #4
    KAI_ASM_INST(0xc0912460)  // addva za0.s, p1/M, p1/M, z3.s
    KAI_ASM_INST(0xc09124e1)  // addva za1.s, p1/M, p1/M, z7.s
    KAI_ASM_INST(0xc0912562)  // addva za2.s, p1/M, p1/M, z11.s
    KAI_ASM_INST(0xc09125e3)  // addva za3.s, p1/M, p1/M, z15.s
    tbz x10, #1, label_15
    tbz x10, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c124)  // ld1w { z4.s-z7.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc0860414)  // mova { z20.s-z23.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c12c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c13c)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c130)  // ld1w { z16.s-z19.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840480)  // mova za0h.s[x12], { z4.s-z7.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xa060c394)  // st1w { z20.s-z23.s }, pn8.b, [x28]
    KAI_ASM_INST(0xc0840782)  // mova za2h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xa061c380)  // st1w { z0.s-z3.s }, pn8.b, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840603)  // mova za3h.s[x12], { z16.s-z19.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c388)  // st1w { z8.s-z11.s }, pn8.b, [x28, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c398)  // st1w { z24.s-z27.s }, pn8.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_12
    b label_31
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c384)  // st1w { z4.s-z7.s }, pn8.b, [x28]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c380)  // st1w { z0.s-z3.s }, pn8.b, [x28, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c394)  // st1w { z20.s-z23.s }, pn8.b, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c390)  // st1w { z16.s-z19.s }, pn8.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_14
    b label_31
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    add x1, x0, #0x98
    add x4, x0, #0x9c
    ldr x6, [x0, #0x58]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0x8540c422)  // ld1rw { z2.s }, p1/Z, [x1]
    add x1, x0, #0xa4
    sub x5, x23, x25
    KAI_ASM_INST(0x8540c481)  // ld1rw { z1.s }, p1/Z, [x4]
    add x7, x7, x24  // C += n
    KAI_ASM_INST(0x8540c460)  // ld1rw { z0.s }, p1/Z, [x3]
    madd x7, x25, x6, x7  // C += m * ldc
    KAI_ASM_INST(0x8540c459)  // ld1rw { z25.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c438)  // ld1rw { z24.s }, p1/Z, [x1]
    tbz x10, #2, label_16
    ldr x2, [x0, #0x78]
    ldr x1, [x0, #0x88]
    add x2, x2, x24, LSL #2
    add x1, x1, x24, LSL #2
    ld1w { z2.s }, p0/Z, [x2]
    ld1w { z1.s }, p0/Z, [x1]
KAI_ASM_LABEL(label_16)  // Store to output array: Load per-channel parameters: End
    cntw x4
    whilelt p0.s, x24, x22
    cmp x5, x4
    mov x12, #0
    csel x3, x5, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a2ac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a1aa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b8cf30)  // sclamp { z16.s-z19.s }, z25.s, z24.s
    st1b { z16.s }, p0, [x7]
    add x7, x7, x6
    st1b { z17.s }, p0, [x7]
    add x7, x7, x6
    st1b { z18.s }, p0, [x7]
    add x7, x7, x6
    st1b { z19.s }, p0, [x7]
    add x7, x7, x6
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1a2ac04)  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a1aa24)  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z1.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1b8cf24)  // sclamp { z4.s-z7.s }, z25.s, z24.s
    st1b { z4.s }, p0, [x7]
    add x7, x7, x6
    beq label_19
    subs x1, x1, #0x1
    st1b { z5.s }, p0, [x7]
    add x7, x7, x6
    beq label_19
    st1b { z6.s }, p0, [x7]
    add x7, x7, x6
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    subs x5, x5, x3
    beq label_29
    cmp x5, x4
    mov x12, #0
    csel x3, x5, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_21
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a2ac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a1aa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b8cf30)  // sclamp { z16.s-z19.s }, z25.s, z24.s
    st1b { z16.s }, p0, [x7]
    add x7, x7, x6
    st1b { z17.s }, p0, [x7]
    add x7, x7, x6
    st1b { z18.s }, p0, [x7]
    add x7, x7, x6
    st1b { z19.s }, p0, [x7]
    add x7, x7, x6
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1a2ac04)  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a1aa24)  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z1.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1b8cf24)  // sclamp { z4.s-z7.s }, z25.s, z24.s
    st1b { z4.s }, p0, [x7]
    add x7, x7, x6
    beq label_22
    subs x1, x1, #0x1
    st1b { z5.s }, p0, [x7]
    add x7, x7, x6
    beq label_22
    st1b { z6.s }, p0, [x7]
    add x7, x7, x6
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 1 oddments: End
    subs x5, x5, x3
    beq label_29
    cmp x5, x4
    mov x12, #0
    csel x3, x5, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_24
KAI_ASM_LABEL(label_23)  // Store to output array: Accumulator row 2 loop
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a2ac0c)  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a1aa2c)  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z1.s
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1b8cf2c)  // sclamp { z12.s-z15.s }, z25.s, z24.s
    st1b { z12.s }, p0, [x7]
    add x7, x7, x6
    st1b { z13.s }, p0, [x7]
    add x7, x7, x6
    st1b { z14.s }, p0, [x7]
    add x7, x7, x6
    st1b { z15.s }, p0, [x7]
    add x7, x7, x6
    blt label_23
KAI_ASM_LABEL(label_24)  // Store to output array: Accumulator row 2 oddments
    cbz x1, label_25
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1a2ac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a1aa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b8cf30)  // sclamp { z16.s-z19.s }, z25.s, z24.s
    st1b { z16.s }, p0, [x7]
    add x7, x7, x6
    beq label_25
    subs x1, x1, #0x1
    st1b { z17.s }, p0, [x7]
    add x7, x7, x6
    beq label_25
    st1b { z18.s }, p0, [x7]
    add x7, x7, x6
KAI_ASM_LABEL(label_25)  // Store to output array: Accumulator row 2 oddments: End
    subs x5, x5, x3
    beq label_29
    cmp x5, x4
    mov x12, #0
    csel x1, x5, x4, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_27
KAI_ASM_LABEL(label_26)  // Store to output array: Accumulator row 3 loop
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a2ac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a1aa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b8cf30)  // sclamp { z16.s-z19.s }, z25.s, z24.s
    st1b { z16.s }, p0, [x7]
    add x7, x7, x6
    st1b { z17.s }, p0, [x7]
    add x7, x7, x6
    st1b { z18.s }, p0, [x7]
    add x7, x7, x6
    st1b { z19.s }, p0, [x7]
    add x7, x7, x6
    blt label_26
KAI_ASM_LABEL(label_27)  // Store to output array: Accumulator row 3 oddments
    cbz x1, label_28
    KAI_ASM_INST(0xc0860474)  // mova { z20.s-z23.s }, za3h.s[x12]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1a2ac14)  // sqdmulh { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a1aa34)  // srshl { z20.s-z23.s }, { z20.s-z23.s }, z1.s
    KAI_ASM_INST(0xc1a0ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z0.s
    KAI_ASM_INST(0xc1b8cf34)  // sclamp { z20.s-z23.s }, z25.s, z24.s
    st1b { z20.s }, p0, [x7]
    add x7, x7, x6
    beq label_28
    subs x1, x1, #0x1
    st1b { z21.s }, p0, [x7]
    add x7, x7, x6
    beq label_28
    st1b { z22.s }, p0, [x7]
KAI_ASM_LABEL(label_28)  // Store to output array: Accumulator row 3 oddments: End
KAI_ASM_LABEL(label_29)  // Store to output array: End
    tbz x10, #0, label_31
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_30)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c138)  // ld1w { z24.s-z27.s }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041c12c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c13c)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c120)  // ld1w { z0.s-z3.s }, pn8.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840782)  // mova za2h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_30
KAI_ASM_LABEL(label_31)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_u8q_mopa_4VLx1VL)

    KAI_ASM_END
