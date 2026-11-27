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

    KAI_ASM_CODE(sme2_interleaved_nomerge_u8q_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_u8q_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_1VLx4VL)
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
    mov x11, #0
    ldr x1, [x0, #0x60]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ptrue p1.b
    ldr x2, [x0, #0x50]
    ldr x3, [x0, #0x78]
    cmp x1, XZR
    ldr x10, [x0, #0x68]
    cset x1, NE
    cmp x2, XZR
    ldr x2, [x0, #0x70]
    orr x11, x11, x1
    cset x1, EQ
    cmp x3, XZR
    orr x11, x11, x1, LSL #1
    cset x1, NE
    mov x9, x10
    orr x11, x11, x1, LSL #2
    add x28, x10, x2
    tbz x11, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c558)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xa041c55c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c554)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840781)  // mova za1h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840682)  // mova za2h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x27, [x0, #0x10]
    mov x26, #0
    mov x25, #0
    ldr x24, [x0, #0]
    ldr x23, [x0, #0x8]
    add x27, x27, #0x3
    ldr x22, [x0, #0x20]
    lsr x27, x27, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x21, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x20, x22
    KAI_ASM_INST(0x25b76730)  // whilelt pn8.s, x25, x23, VLx4
    tbnz x11, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    KAI_ASM_INST(0xa019c030)  // ld1w { z16.s-z19.s }, p8/Z, [x1, x25, LSL #2]
    KAI_ASM_INST(0xc0902600)  // addha za0.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902621)  // addha za1.s, p1/M, p1/M, z17.s
    KAI_ASM_INST(0xc0902642)  // addha za2.s, p1/M, p1/M, z18.s
    KAI_ASM_INST(0xc0902663)  // addha za3.s, p1/M, p1/M, z19.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x25
    mov x2, x26
    incw x1, ALL, MUL #4
    incw x2
    cmp x1, x23
    mov x1, x11
    csel x2, x26, x2, LT
    bfm x11, XZR, #0, #0  // bfc x11, #0, #0x1
    csel x10, x10, x28, LT
    cmp x2, x24
    csel x11, x1, x11, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x27, #0x2
    and x1, x27, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa1408681)  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x20]
    addvl x20, x20, #4
    KAI_ASM_INST(0xa14086b0)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x21]
    KAI_ASM_INST(0xa14186b3)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x21, #0x4, MUL VL]
    KAI_ASM_INST(0xa14286a2)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x21, #0x8, MUL VL]
    KAI_ASM_INST(0xa14386b2)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x21, #0xc, MUL VL]
    addvl x21, x21, #16
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa1b02420)  // umopa za0.s, p1/M, p1/M, z1.b, z16.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa1b42421)  // umopa za1.s, p1/M, p1/M, z1.b, z20.b
    KAI_ASM_INST(0xa1b82422)  // umopa za2.s, p1/M, p1/M, z1.b, z24.b
    KAI_ASM_INST(0xa1bc2423)  // umopa za3.s, p1/M, p1/M, z1.b, z28.b
    KAI_ASM_INST(0xa14086b0)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x21]
    KAI_ASM_INST(0xa1b324a0)  // umopa za0.s, p1/M, p1/M, z5.b, z19.b
    KAI_ASM_INST(0xa1b724a1)  // umopa za1.s, p1/M, p1/M, z5.b, z23.b
    KAI_ASM_INST(0xa1bb24a2)  // umopa za2.s, p1/M, p1/M, z5.b, z27.b
    KAI_ASM_INST(0xa1bf24a3)  // umopa za3.s, p1/M, p1/M, z5.b, z31.b
    KAI_ASM_INST(0xa14186b3)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x21, #0x4, MUL VL]
    KAI_ASM_INST(0xa1a22520)  // umopa za0.s, p1/M, p1/M, z9.b, z2.b
    KAI_ASM_INST(0xa1a62521)  // umopa za1.s, p1/M, p1/M, z9.b, z6.b
    KAI_ASM_INST(0xa1aa2522)  // umopa za2.s, p1/M, p1/M, z9.b, z10.b
    KAI_ASM_INST(0xa1ae2523)  // umopa za3.s, p1/M, p1/M, z9.b, z14.b
    KAI_ASM_INST(0xa14286a2)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x21, #0x8, MUL VL]
    KAI_ASM_INST(0xa1b225a0)  // umopa za0.s, p1/M, p1/M, z13.b, z18.b
    KAI_ASM_INST(0xa1b625a1)  // umopa za1.s, p1/M, p1/M, z13.b, z22.b
    KAI_ASM_INST(0xa1ba25a2)  // umopa za2.s, p1/M, p1/M, z13.b, z26.b
    KAI_ASM_INST(0xa1be25a3)  // umopa za3.s, p1/M, p1/M, z13.b, z30.b
    KAI_ASM_INST(0xa1408681)  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x20]
    addvl x20, x20, #4
    KAI_ASM_INST(0xa14386b2)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x21, #0xc, MUL VL]
    addvl x21, x21, #16
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa1b02420)  // umopa za0.s, p1/M, p1/M, z1.b, z16.b
    KAI_ASM_INST(0xa1b42421)  // umopa za1.s, p1/M, p1/M, z1.b, z20.b
    KAI_ASM_INST(0xa1b82422)  // umopa za2.s, p1/M, p1/M, z1.b, z24.b
    KAI_ASM_INST(0xa1bc2423)  // umopa za3.s, p1/M, p1/M, z1.b, z28.b
    KAI_ASM_INST(0xa1b324a0)  // umopa za0.s, p1/M, p1/M, z5.b, z19.b
    KAI_ASM_INST(0xa1b724a1)  // umopa za1.s, p1/M, p1/M, z5.b, z23.b
    KAI_ASM_INST(0xa1bb24a2)  // umopa za2.s, p1/M, p1/M, z5.b, z27.b
    KAI_ASM_INST(0xa1bf24a3)  // umopa za3.s, p1/M, p1/M, z5.b, z31.b
    KAI_ASM_INST(0xa1a22520)  // umopa za0.s, p1/M, p1/M, z9.b, z2.b
    KAI_ASM_INST(0xa1a62521)  // umopa za1.s, p1/M, p1/M, z9.b, z6.b
    KAI_ASM_INST(0xa1aa2522)  // umopa za2.s, p1/M, p1/M, z9.b, z10.b
    KAI_ASM_INST(0xa1ae2523)  // umopa za3.s, p1/M, p1/M, z9.b, z14.b
    KAI_ASM_INST(0xa1b225a0)  // umopa za0.s, p1/M, p1/M, z13.b, z18.b
    KAI_ASM_INST(0xa1b625a1)  // umopa za1.s, p1/M, p1/M, z13.b, z22.b
    KAI_ASM_INST(0xa1ba25a2)  // umopa za2.s, p1/M, p1/M, z13.b, z26.b
    KAI_ASM_INST(0xa1be25a3)  // umopa za3.s, p1/M, p1/M, z13.b, z30.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z16.b }, p1/Z, [x20]
    subs x1, x1, #0x1
    addvl x20, x20, #1
    KAI_ASM_INST(0xa04086a4)  // ld1b { z4.b-z7.b }, pn9.b/Z, [x21]
    addvl x21, x21, #4
    KAI_ASM_INST(0xa1a42600)  // umopa za0.s, p1/M, p1/M, z16.b, z4.b
    KAI_ASM_INST(0xa1a52601)  // umopa za1.s, p1/M, p1/M, z16.b, z5.b
    KAI_ASM_INST(0xa1a62602)  // umopa za2.s, p1/M, p1/M, z16.b, z6.b
    KAI_ASM_INST(0xa1a72603)  // umopa za3.s, p1/M, p1/M, z16.b, z7.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    ld1w { z15.s }, p1/Z, [x20]
    addvl x20, x20, #1
    KAI_ASM_INST(0xc09125e0)  // addva za0.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc09125e1)  // addva za1.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc09125e2)  // addva za2.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc09125e3)  // addva za3.s, p1/M, p1/M, z15.s
    tbz x11, #1, label_15
    tbz x11, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c544)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860438)  // mova { z24.s-z27.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c550)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c54c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840480)  // mova za0h.s[x12], { z4.s-z7.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840601)  // mova za1h.s[x12], { z16.s-z19.s }
    KAI_ASM_INST(0xa060c53c)  // st1w { z28.s-z31.s }, pn9.b, [x9]
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xa061c538)  // st1w { z24.s-z27.s }, pn9.b, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c534)  // st1w { z20.s-z23.s }, pn9.b, [x9, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c528)  // st1w { z8.s-z11.s }, pn9.b, [x9, #0xc, MUL VL]
    addvl x9, x9, #16
    blt label_12
    b label_22
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c528)  // st1w { z8.s-z11.s }, pn9.b, [x9]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c524)  // st1w { z4.s-z7.s }, pn9.b, [x9, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c52c)  // st1w { z12.s-z15.s }, pn9.b, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c530)  // st1w { z16.s-z19.s }, pn9.b, [x9, #0xc, MUL VL]
    addvl x9, x9, #16
    blt label_14
    b label_22
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x8, [x0, #0x50]
    add x7, x0, #0x98
    add x6, x0, #0x9c
    ldr x5, [x0, #0x58]
    add x4, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0x8540c4e4)  // ld1rw { z4.s }, p1/Z, [x7]
    add x1, x0, #0xa4
    sub x3, x24, x26
    KAI_ASM_INST(0x8540c4e5)  // ld1rw { z5.s }, p1/Z, [x7]
    add x8, x8, x25  // C += n
    KAI_ASM_INST(0x8540c4e6)  // ld1rw { z6.s }, p1/Z, [x7]
    madd x8, x26, x5, x8  // C += m * ldc
    KAI_ASM_INST(0x8540c4e7)  // ld1rw { z7.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c4c0)  // ld1rw { z0.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4c1)  // ld1rw { z1.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4c2)  // ld1rw { z2.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c4c3)  // ld1rw { z3.s }, p1/Z, [x6]
    KAI_ASM_INST(0x8540c488)  // ld1rw { z8.s }, p1/Z, [x4]
    KAI_ASM_INST(0x8540c455)  // ld1rw { z21.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c434)  // ld1rw { z20.s }, p1/Z, [x1]
    tbz x11, #2, label_16
    ldr x2, [x0, #0x78]
    ldr x1, [x0, #0x88]
    add x2, x2, x25, LSL #2
    add x1, x1, x25, LSL #2
    KAI_ASM_INST(0xa040c044)  // ld1w { z4.s-z7.s }, p8/Z, [x2]
    KAI_ASM_INST(0xa040c020)  // ld1w { z0.s-z3.s }, p8/Z, [x1]
KAI_ASM_LABEL(label_16)  // Store to output array: Load per-channel parameters: End
    cntw x1
    whilelt p0.b, x25, x23
    cmp x3, x1
    mov x12, #0
    csel x1, x3, x1, LT
    lsr x2, x1, #0x1
    and x1, x1, #0x1
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860010)  // mova { z16.s-z17.s }, za0h.s[x12, 0:1]
    KAI_ASM_INST(0xc086005e)  // mova { z30.s-z31.s }, za1h.s[x12, 0:1]
    KAI_ASM_INST(0xc086009a)  // mova { z26.s-z27.s }, za2h.s[x12, 0:1]
    KAI_ASM_INST(0xc08600cc)  // mova { z12.s-z13.s }, za3h.s[x12, 0:1]
    KAI_ASM_INST(0xc1a4a410)  // sqdmulh { z16.s-z17.s }, { z16.s-z17.s }, z4.s
    KAI_ASM_INST(0xc1a5a41e)  // sqdmulh { z30.s-z31.s }, { z30.s-z31.s }, z5.s
    add x12, x12, #0x2
    KAI_ASM_INST(0xc1a6a41a)  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z6.s
    cmp x12, x2, LSL #1
    KAI_ASM_INST(0xc1a7a40c)  // sqdmulh { z12.s-z13.s }, { z12.s-z13.s }, z7.s
    KAI_ASM_INST(0xc1a0a230)  // srshl { z16.s-z17.s }, { z16.s-z17.s }, z0.s
    KAI_ASM_INST(0xc1a1a23e)  // srshl { z30.s-z31.s }, { z30.s-z31.s }, z1.s
    KAI_ASM_INST(0xc1a2a23a)  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z2.s
    KAI_ASM_INST(0xc1a3a22c)  // srshl { z12.s-z13.s }, { z12.s-z13.s }, z3.s
    KAI_ASM_INST(0xc1a8a310)  // add { z16.s-z17.s }, { z16.s-z17.s }, z8.s
    KAI_ASM_INST(0xc1a8a31e)  // add { z30.s-z31.s }, { z30.s-z31.s }, z8.s
    KAI_ASM_INST(0xc1a8a31a)  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s
    KAI_ASM_INST(0xc1a8a30c)  // add { z12.s-z13.s }, { z12.s-z13.s }, z8.s
    KAI_ASM_INST(0xc1b4c6b0)  // sclamp { z16.s-z17.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6be)  // sclamp { z30.s-z31.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6ba)  // sclamp { z26.s-z27.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6ac)  // sclamp { z12.s-z13.s }, z21.s, z20.s
    uzp1 z19.b, z16.b, z30.b
    uzp1 z18.b, z17.b, z31.b
    uzp1 z17.b, z26.b, z12.b
    uzp1 z16.b, z27.b, z13.b
    uzp1 z17.b, z19.b, z17.b
    uzp1 z16.b, z18.b, z16.b
    st1b { z17.b }, p0, [x8]
    add x8, x8, x5
    st1b { z16.b }, p0, [x8]
    add x8, x8, x5
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc086000a)  // mova { z10.s-z11.s }, za0h.s[x12, 0:1]
    KAI_ASM_INST(0xc086005a)  // mova { z26.s-z27.s }, za1h.s[x12, 0:1]
    KAI_ASM_INST(0xc086008e)  // mova { z14.s-z15.s }, za2h.s[x12, 0:1]
    KAI_ASM_INST(0xc08600d6)  // mova { z22.s-z23.s }, za3h.s[x12, 0:1]
    KAI_ASM_INST(0xc1a4a40a)  // sqdmulh { z10.s-z11.s }, { z10.s-z11.s }, z4.s
    KAI_ASM_INST(0xc1a5a41a)  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z5.s
    KAI_ASM_INST(0xc1a6a40e)  // sqdmulh { z14.s-z15.s }, { z14.s-z15.s }, z6.s
    KAI_ASM_INST(0xc1a7a416)  // sqdmulh { z22.s-z23.s }, { z22.s-z23.s }, z7.s
    KAI_ASM_INST(0xc1a0a22a)  // srshl { z10.s-z11.s }, { z10.s-z11.s }, z0.s
    KAI_ASM_INST(0xc1a1a23a)  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z1.s
    KAI_ASM_INST(0xc1a2a22e)  // srshl { z14.s-z15.s }, { z14.s-z15.s }, z2.s
    KAI_ASM_INST(0xc1a3a236)  // srshl { z22.s-z23.s }, { z22.s-z23.s }, z3.s
    KAI_ASM_INST(0xc1a8a30a)  // add { z10.s-z11.s }, { z10.s-z11.s }, z8.s
    KAI_ASM_INST(0xc1a8a31a)  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s
    KAI_ASM_INST(0xc1a8a30e)  // add { z14.s-z15.s }, { z14.s-z15.s }, z8.s
    KAI_ASM_INST(0xc1a8a316)  // add { z22.s-z23.s }, { z22.s-z23.s }, z8.s
    KAI_ASM_INST(0xc1b4c6aa)  // sclamp { z10.s-z11.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6ba)  // sclamp { z26.s-z27.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6ae)  // sclamp { z14.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4c6b6)  // sclamp { z22.s-z23.s }, z21.s, z20.s
    uzp1 z17.b, z10.b, z26.b
    uzp1 z16.b, z14.b, z22.b
    uzp1 z16.b, z17.b, z16.b
    st1b { z16.b }, p0, [x8]
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    tbz x11, #0, label_22
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_21)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c558)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xa041c54c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c544)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_21
KAI_ASM_LABEL(label_22)  // End block
    incw x25, ALL, MUL #4
    cmp x25, x23
    blt label_4
    ldr x1, [x0, #0x70]
    incw x26
    mov x9, x28
    cmp x26, x24
    mov x25, #0
    mov x22, x20
    add x28, x9, x1
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_u8q_mopa_1VLx4VL)

    KAI_ASM_END
