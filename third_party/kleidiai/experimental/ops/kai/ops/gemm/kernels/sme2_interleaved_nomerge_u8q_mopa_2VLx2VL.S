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

    KAI_ASM_CODE(sme2_interleaved_nomerge_u8q_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_u8q_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_u8q_mopa_2VLx2VL)
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
    KAI_ASM_INST(0xa040c548)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xa041c554)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c544)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840500)  // mova za0h.s[x12], { z8.s-z11.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840681)  // mova za1h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
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
    KAI_ASM_INST(0x25b74730)  // whilelt pn8.s, x25, x23, VLx2
    tbnz x11, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    KAI_ASM_INST(0xa0194026)  // ld1w { z6.s-z7.s }, p8/Z, [x1, x25, LSL #2]
    KAI_ASM_INST(0xc09024c0)  // addha za0.s, p1/M, p1/M, z6.s
    KAI_ASM_INST(0xc09024e1)  // addha za1.s, p1/M, p1/M, z7.s
    KAI_ASM_INST(0xc09024c2)  // addha za2.s, p1/M, p1/M, z6.s
    KAI_ASM_INST(0xc09024e3)  // addha za3.s, p1/M, p1/M, z7.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x25
    mov x2, x26
    incw x1, ALL, MUL #2
    incw x2, ALL, MUL #2
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
    KAI_ASM_INST(0xa1408680)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x20]
    KAI_ASM_INST(0xa0418694)  // ld1b { z20.b-z23.b }, pn9.b/Z, [x20, #0x4, MUL VL]
    addvl x20, x20, #8
    KAI_ASM_INST(0xa04086b0)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x21]
    KAI_ASM_INST(0xa14186a3)  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x21, #0x4, MUL VL]
    addvl x21, x21, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa1b02400)  // umopa za0.s, p1/M, p1/M, z0.b, z16.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa1b12401)  // umopa za1.s, p1/M, p1/M, z0.b, z17.b
    KAI_ASM_INST(0xa1b02482)  // umopa za2.s, p1/M, p1/M, z4.b, z16.b
    KAI_ASM_INST(0xa1b12483)  // umopa za3.s, p1/M, p1/M, z4.b, z17.b
    KAI_ASM_INST(0xa1b22500)  // umopa za0.s, p1/M, p1/M, z8.b, z18.b
    KAI_ASM_INST(0xa1b32501)  // umopa za1.s, p1/M, p1/M, z8.b, z19.b
    KAI_ASM_INST(0xa1b22582)  // umopa za2.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa1b32583)  // umopa za3.s, p1/M, p1/M, z12.b, z19.b
    KAI_ASM_INST(0xa1408680)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x20]
    KAI_ASM_INST(0xa1a32680)  // umopa za0.s, p1/M, p1/M, z20.b, z3.b
    KAI_ASM_INST(0xa04086b0)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x21]
    KAI_ASM_INST(0xa1a72681)  // umopa za1.s, p1/M, p1/M, z20.b, z7.b
    KAI_ASM_INST(0xa1a326a2)  // umopa za2.s, p1/M, p1/M, z21.b, z3.b
    KAI_ASM_INST(0xa1a726a3)  // umopa za3.s, p1/M, p1/M, z21.b, z7.b
    KAI_ASM_INST(0xa1ab26c0)  // umopa za0.s, p1/M, p1/M, z22.b, z11.b
    KAI_ASM_INST(0xa1af26c1)  // umopa za1.s, p1/M, p1/M, z22.b, z15.b
    KAI_ASM_INST(0xa1ab26e2)  // umopa za2.s, p1/M, p1/M, z23.b, z11.b
    KAI_ASM_INST(0xa1af26e3)  // umopa za3.s, p1/M, p1/M, z23.b, z15.b
    KAI_ASM_INST(0xa0418694)  // ld1b { z20.b-z23.b }, pn9.b/Z, [x20, #0x4, MUL VL]
    addvl x20, x20, #8
    KAI_ASM_INST(0xa14186a3)  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x21, #0x4, MUL VL]
    addvl x21, x21, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa1b02400)  // umopa za0.s, p1/M, p1/M, z0.b, z16.b
    KAI_ASM_INST(0xa1b12401)  // umopa za1.s, p1/M, p1/M, z0.b, z17.b
    KAI_ASM_INST(0xa1b02482)  // umopa za2.s, p1/M, p1/M, z4.b, z16.b
    KAI_ASM_INST(0xa1b12483)  // umopa za3.s, p1/M, p1/M, z4.b, z17.b
    KAI_ASM_INST(0xa1b22500)  // umopa za0.s, p1/M, p1/M, z8.b, z18.b
    KAI_ASM_INST(0xa1b32501)  // umopa za1.s, p1/M, p1/M, z8.b, z19.b
    KAI_ASM_INST(0xa1b22582)  // umopa za2.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa1b32583)  // umopa za3.s, p1/M, p1/M, z12.b, z19.b
    KAI_ASM_INST(0xa1a32680)  // umopa za0.s, p1/M, p1/M, z20.b, z3.b
    KAI_ASM_INST(0xa1a72681)  // umopa za1.s, p1/M, p1/M, z20.b, z7.b
    KAI_ASM_INST(0xa1a326a2)  // umopa za2.s, p1/M, p1/M, z21.b, z3.b
    KAI_ASM_INST(0xa1a726a3)  // umopa za3.s, p1/M, p1/M, z21.b, z7.b
    KAI_ASM_INST(0xa1ab26c0)  // umopa za0.s, p1/M, p1/M, z22.b, z11.b
    KAI_ASM_INST(0xa1af26c1)  // umopa za1.s, p1/M, p1/M, z22.b, z15.b
    KAI_ASM_INST(0xa1ab26e2)  // umopa za2.s, p1/M, p1/M, z23.b, z11.b
    KAI_ASM_INST(0xa1af26e3)  // umopa za3.s, p1/M, p1/M, z23.b, z15.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa040069c)  // ld1b { z28.b-z29.b }, pn9.b/Z, [x20]
    subs x1, x1, #0x1
    addvl x20, x20, #2
    KAI_ASM_INST(0xa04006b4)  // ld1b { z20.b-z21.b }, pn9.b/Z, [x21]
    addvl x21, x21, #2
    KAI_ASM_INST(0xa1b42780)  // umopa za0.s, p1/M, p1/M, z28.b, z20.b
    KAI_ASM_INST(0xa1b52781)  // umopa za1.s, p1/M, p1/M, z28.b, z21.b
    KAI_ASM_INST(0xa1b427a2)  // umopa za2.s, p1/M, p1/M, z29.b, z20.b
    KAI_ASM_INST(0xa1b527a3)  // umopa za3.s, p1/M, p1/M, z29.b, z21.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    KAI_ASM_INST(0xa040469a)  // ld1w { z26.s-z27.s }, pn9.b/Z, [x20]
    addvl x20, x20, #2
    KAI_ASM_INST(0xc0912740)  // addva za0.s, p1/M, p1/M, z26.s
    KAI_ASM_INST(0xc0912741)  // addva za1.s, p1/M, p1/M, z26.s
    KAI_ASM_INST(0xc0912762)  // addva za2.s, p1/M, p1/M, z27.s
    KAI_ASM_INST(0xc0912763)  // addva za3.s, p1/M, p1/M, z27.s
    tbz x11, #1, label_15
    tbz x11, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c544)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xc0860414)  // mova { z20.s-z23.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c55c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c558)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840480)  // mova za0h.s[x12], { z4.s-z7.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840781)  // mova za1h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xa060c534)  // st1w { z20.s-z23.s }, pn9.b, [x9]
    KAI_ASM_INST(0xc0840402)  // mova za2h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xa061c530)  // st1w { z16.s-z19.s }, pn9.b, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840703)  // mova za3h.s[x12], { z24.s-z27.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c52c)  // st1w { z12.s-z15.s }, pn9.b, [x9, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c528)  // st1w { z8.s-z11.s }, pn9.b, [x9, #0xc, MUL VL]
    addvl x9, x9, #16
    blt label_12
    b label_25
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860434)  // mova { z20.s-z23.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860460)  // mova { z0.s-z3.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c524)  // st1w { z4.s-z7.s }, pn9.b, [x9]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c534)  // st1w { z20.s-z23.s }, pn9.b, [x9, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c530)  // st1w { z16.s-z19.s }, pn9.b, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c520)  // st1w { z0.s-z3.s }, pn9.b, [x9, #0xc, MUL VL]
    addvl x9, x9, #16
    blt label_14
    b label_25
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x8, [x0, #0x50]
    add x7, x0, #0x98
    add x4, x0, #0x9c
    ldr x6, [x0, #0x58]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0x8540c4e2)  // ld1rw { z2.s }, p1/Z, [x7]
    add x1, x0, #0xa4
    sub x5, x24, x26
    KAI_ASM_INST(0x8540c4ea)  // ld1rw { z10.s }, p1/Z, [x7]
    add x8, x8, x25  // C += n
    KAI_ASM_INST(0x8540c483)  // ld1rw { z3.s }, p1/Z, [x4]
    madd x8, x26, x6, x8  // C += m * ldc
    KAI_ASM_INST(0x8540c48b)  // ld1rw { z11.s }, p1/Z, [x4]
    KAI_ASM_INST(0x8540c460)  // ld1rw { z0.s }, p1/Z, [x3]
    KAI_ASM_INST(0x8540c455)  // ld1rw { z21.s }, p1/Z, [x2]
    KAI_ASM_INST(0x8540c434)  // ld1rw { z20.s }, p1/Z, [x1]
    tbz x11, #2, label_16
    ldr x2, [x0, #0x78]
    ldr x1, [x0, #0x88]
    add x2, x2, x25, LSL #2
    add x1, x1, x25, LSL #2
    KAI_ASM_INST(0xa1404042)  // ld1w { z2.s, z10.s }, p8/Z, [x2]
    KAI_ASM_INST(0xa1404023)  // ld1w { z3.s, z11.s }, p8/Z, [x1]
KAI_ASM_LABEL(label_16)  // Store to output array: Load per-channel parameters: End
    cntw x4
    whilelt p0.h, x25, x23
    cmp x5, x4
    mov x12, #0
    csel x3, x5, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860418)  // mova { z24.s-z27.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    KAI_ASM_INST(0xc1a2ac18)  // sqdmulh { z24.s-z27.s }, { z24.s-z27.s }, z2.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1aaac1c)  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a3aa38)  // srshl { z24.s-z27.s }, { z24.s-z27.s }, z3.s
    KAI_ASM_INST(0xc1abaa3c)  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s
    KAI_ASM_INST(0xc1a0ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z0.s
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1b4ceb8)  // sclamp { z24.s-z27.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cebc)  // sclamp { z28.s-z31.s }, z21.s, z20.s
    uzp1 z19.h, z24.h, z28.h
    uzp1 z18.h, z25.h, z29.h
    uzp1 z17.h, z26.h, z30.h
    uzp1 z16.h, z27.h, z31.h
    st1b { z19.h }, p0, [x8]
    add x8, x8, x6
    st1b { z18.h }, p0, [x8]
    add x8, x8, x6
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    KAI_ASM_INST(0xc1a2ac04)  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1aaac1c)  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s
    KAI_ASM_INST(0xc1a3aa24)  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z3.s
    KAI_ASM_INST(0xc1abaa3c)  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1b4cea4)  // sclamp { z4.s-z7.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cebc)  // sclamp { z28.s-z31.s }, z21.s, z20.s
    uzp1 z16.h, z4.h, z28.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_19
    subs x1, x1, #0x1
    uzp1 z16.h, z5.h, z29.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_19
    uzp1 z16.h, z6.h, z30.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    subs x5, x5, x3
    beq label_23
    cmp x5, x4
    mov x12, #0
    csel x1, x5, x4, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_21
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1a2ac0c)  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1aaac1c)  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z10.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xc1a3aa2c)  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z3.s
    KAI_ASM_INST(0xc1abaa3c)  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z11.s
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1b4ceac)  // sclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cebc)  // sclamp { z28.s-z31.s }, z21.s, z20.s
    uzp1 z19.h, z12.h, z28.h
    uzp1 z18.h, z13.h, z29.h
    uzp1 z17.h, z14.h, z30.h
    uzp1 z16.h, z15.h, z31.h
    st1b { z19.h }, p0, [x8]
    add x8, x8, x6
    st1b { z18.h }, p0, [x8]
    add x8, x8, x6
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1a2ac04)  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1aaac10)  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z10.s
    KAI_ASM_INST(0xc1a3aa24)  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z3.s
    KAI_ASM_INST(0xc1abaa30)  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z11.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b4cea4)  // sclamp { z4.s-z7.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4ceb0)  // sclamp { z16.s-z19.s }, z21.s, z20.s
    uzp1 z16.h, z4.h, z16.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_22
    subs x1, x1, #0x1
    uzp1 z16.h, z5.h, z17.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_22
    uzp1 z16.h, z6.h, z18.h
    st1b { z16.h }, p0, [x8]
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_23)  // Store to output array: End
    tbz x11, #0, label_25
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_24)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c540)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x10]
    KAI_ASM_INST(0xa041c54c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x10, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c55c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x10, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c544)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x10, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x10, x10, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840782)  // mova za2h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840483)  // mova za3h.s[x12], { z4.s-z7.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_24
KAI_ASM_LABEL(label_25)  // End block
    incw x25, ALL, MUL #2
    cmp x25, x23
    blt label_4
    ldr x1, [x0, #0x70]
    incw x26, ALL, MUL #2
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_u8q_mopa_2VLx2VL)

    KAI_ASM_END
