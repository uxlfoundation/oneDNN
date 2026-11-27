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

    KAI_ASM_CODE(sme_interleaved_nomerge_u8q_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_u8q_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_u8q_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_u8q_mopa_2VLx2VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x14, #0
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x1, [x0, #0x60]
    ptrue p2.b
    ldr x2, [x0, #0x50]
    ldr x3, [x0, #0x78]
    cmp x1, XZR
    ldr x13, [x0, #0x68]
    cset x1, NE
    cmp x2, XZR
    ldr x2, [x0, #0x70]
    orr x14, x14, x1
    cset x1, EQ
    cmp x3, XZR
    orr x14, x14, x1, LSL #1
    cset x1, NE
    mov x11, x13
    orr x14, x14, x1, LSL #2
    add x10, x13, x2
    tbz x14, #0, label_2
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    addvl x3, x13, #4
    addvl x2, x13, #8
    KAI_ASM_INST(0xe09f09a0)  // ld1w { za0h.s[x12] }, p2/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0864)  // ld1w { za1h.s[x12] }, p2/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0848)  // ld1w { za2h.s[x12] }, p2/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f082c)  // ld1w { za3h.s[x12] }, p2/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08609a1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860865)  // ld1w { za1h.s[x12, #1] }, p2/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860849)  // ld1w { za2h.s[x12, #1] }, p2/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086082d)  // ld1w { za3h.s[x12, #1] }, p2/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08509a2)  // ld1w { za0h.s[x12, #2] }, p2/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850866)  // ld1w { za1h.s[x12, #2] }, p2/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085084a)  // ld1w { za2h.s[x12, #2] }, p2/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085082e)  // ld1w { za3h.s[x12, #2] }, p2/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08409a3)  // ld1w { za0h.s[x12, #3] }, p2/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840867)  // ld1w { za1h.s[x12, #3] }, p2/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084084b)  // ld1w { za2h.s[x12, #3] }, p2/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084082f)  // ld1w { za3h.s[x12, #3] }, p2/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x9, [x0, #0x10]
    mov x28, #0
    mov x27, #0
    ldr x26, [x0, #0]
    ldr x25, [x0, #0x8]
    add x9, x9, #0x3
    ldr x24, [x0, #0x20]
    lsr x9, x9, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x23, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x1, x27
    mov x22, x24
    whilelt p1.s, x1, x25
    incw x1
    whilelt p0.s, x1, x25
    tbnz x14, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    add x1, x1, x27, LSL #2
    ld1w { z17.s }, p1/Z, [x1]
    ld1w { z16.s }, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xc0904a20)  // addha za0.s, p2/M, p2/M, z17.s
    KAI_ASM_INST(0xc0904a01)  // addha za1.s, p2/M, p2/M, z16.s
    KAI_ASM_INST(0xc0904a22)  // addha za2.s, p2/M, p2/M, z17.s
    KAI_ASM_INST(0xc0904a03)  // addha za3.s, p2/M, p2/M, z16.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x27
    mov x2, x28
    incw x1, ALL, MUL #2
    incw x2, ALL, MUL #2
    cmp x1, x25
    mov x1, x14
    csel x2, x28, x2, LT
    bfm x14, XZR, #0, #0  // bfc x14, #0, #0x1
    csel x13, x13, x10, LT
    cmp x2, x26
    csel x14, x1, x14, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x9, #0x2
    and x1, x9, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    ld1b { z31.b }, p2/Z, [x22]
    ld1b { z30.b }, p2/Z, [x22, #1, MUL VL]
    ld1b { z29.b }, p2/Z, [x22, #2, MUL VL]
    ld1b { z28.b }, p2/Z, [x22, #3, MUL VL]
    ld1b { z27.b }, p2/Z, [x22, #4, MUL VL]
    ld1b { z26.b }, p2/Z, [x22, #5, MUL VL]
    ld1b { z25.b }, p2/Z, [x22, #6, MUL VL]
    ld1b { z24.b }, p2/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1b { z23.b }, p2/Z, [x23]
    ld1b { z22.b }, p2/Z, [x23, #1, MUL VL]
    ld1b { z21.b }, p2/Z, [x23, #2, MUL VL]
    ld1b { z20.b }, p2/Z, [x23, #3, MUL VL]
    ld1b { z19.b }, p2/Z, [x23, #4, MUL VL]
    ld1b { z18.b }, p2/Z, [x23, #5, MUL VL]
    ld1b { z17.b }, p2/Z, [x23, #6, MUL VL]
    ld1b { z16.b }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa1b74be0)  // umopa za0.s, p2/M, p2/M, z31.b, z23.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa1b64be1)  // umopa za1.s, p2/M, p2/M, z31.b, z22.b
    ld1b { z31.b }, p2/Z, [x22]
    KAI_ASM_INST(0xa1b74bc2)  // umopa za2.s, p2/M, p2/M, z30.b, z23.b
    ld1b { z23.b }, p2/Z, [x23]
    KAI_ASM_INST(0xa1b64bc3)  // umopa za3.s, p2/M, p2/M, z30.b, z22.b
    ld1b { z30.b }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa1b54ba0)  // umopa za0.s, p2/M, p2/M, z29.b, z21.b
    ld1b { z22.b }, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa1b44ba1)  // umopa za1.s, p2/M, p2/M, z29.b, z20.b
    ld1b { z29.b }, p2/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa1b54b82)  // umopa za2.s, p2/M, p2/M, z28.b, z21.b
    ld1b { z21.b }, p2/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa1b44b83)  // umopa za3.s, p2/M, p2/M, z28.b, z20.b
    ld1b { z28.b }, p2/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0xa1b34b60)  // umopa za0.s, p2/M, p2/M, z27.b, z19.b
    ld1b { z20.b }, p2/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0xa1b24b61)  // umopa za1.s, p2/M, p2/M, z27.b, z18.b
    ld1b { z27.b }, p2/Z, [x22, #4, MUL VL]
    KAI_ASM_INST(0xa1b34b42)  // umopa za2.s, p2/M, p2/M, z26.b, z19.b
    ld1b { z19.b }, p2/Z, [x23, #4, MUL VL]
    KAI_ASM_INST(0xa1b24b43)  // umopa za3.s, p2/M, p2/M, z26.b, z18.b
    ld1b { z26.b }, p2/Z, [x22, #5, MUL VL]
    KAI_ASM_INST(0xa1b14b20)  // umopa za0.s, p2/M, p2/M, z25.b, z17.b
    ld1b { z18.b }, p2/Z, [x23, #5, MUL VL]
    KAI_ASM_INST(0xa1b04b21)  // umopa za1.s, p2/M, p2/M, z25.b, z16.b
    ld1b { z25.b }, p2/Z, [x22, #6, MUL VL]
    KAI_ASM_INST(0xa1b14b02)  // umopa za2.s, p2/M, p2/M, z24.b, z17.b
    ld1b { z17.b }, p2/Z, [x23, #6, MUL VL]
    KAI_ASM_INST(0xa1b04b03)  // umopa za3.s, p2/M, p2/M, z24.b, z16.b
    ld1b { z24.b }, p2/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1b { z16.b }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa1b74be0)  // umopa za0.s, p2/M, p2/M, z31.b, z23.b
    KAI_ASM_INST(0xa1b64be1)  // umopa za1.s, p2/M, p2/M, z31.b, z22.b
    KAI_ASM_INST(0xa1b74bc2)  // umopa za2.s, p2/M, p2/M, z30.b, z23.b
    KAI_ASM_INST(0xa1b64bc3)  // umopa za3.s, p2/M, p2/M, z30.b, z22.b
    KAI_ASM_INST(0xa1b54ba0)  // umopa za0.s, p2/M, p2/M, z29.b, z21.b
    KAI_ASM_INST(0xa1b44ba1)  // umopa za1.s, p2/M, p2/M, z29.b, z20.b
    KAI_ASM_INST(0xa1b54b82)  // umopa za2.s, p2/M, p2/M, z28.b, z21.b
    KAI_ASM_INST(0xa1b44b83)  // umopa za3.s, p2/M, p2/M, z28.b, z20.b
    KAI_ASM_INST(0xa1b34b60)  // umopa za0.s, p2/M, p2/M, z27.b, z19.b
    KAI_ASM_INST(0xa1b24b61)  // umopa za1.s, p2/M, p2/M, z27.b, z18.b
    KAI_ASM_INST(0xa1b34b42)  // umopa za2.s, p2/M, p2/M, z26.b, z19.b
    KAI_ASM_INST(0xa1b24b43)  // umopa za3.s, p2/M, p2/M, z26.b, z18.b
    KAI_ASM_INST(0xa1b14b20)  // umopa za0.s, p2/M, p2/M, z25.b, z17.b
    KAI_ASM_INST(0xa1b04b21)  // umopa za1.s, p2/M, p2/M, z25.b, z16.b
    KAI_ASM_INST(0xa1b14b02)  // umopa za2.s, p2/M, p2/M, z24.b, z17.b
    KAI_ASM_INST(0xa1b04b03)  // umopa za3.s, p2/M, p2/M, z24.b, z16.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z19.b }, p2/Z, [x22]
    subs x1, x1, #0x1
    ld1b { z18.b }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ld1b { z17.b }, p2/Z, [x23]
    ld1b { z16.b }, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa1b14a60)  // umopa za0.s, p2/M, p2/M, z19.b, z17.b
    KAI_ASM_INST(0xa1b04a61)  // umopa za1.s, p2/M, p2/M, z19.b, z16.b
    KAI_ASM_INST(0xa1b14a42)  // umopa za2.s, p2/M, p2/M, z18.b, z17.b
    KAI_ASM_INST(0xa1b04a43)  // umopa za3.s, p2/M, p2/M, z18.b, z16.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    ld1w { z17.s }, p2/Z, [x22]
    ld1w { z16.s }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc0914a20)  // addva za0.s, p2/M, p2/M, z17.s
    KAI_ASM_INST(0xc0914a21)  // addva za1.s, p2/M, p2/M, z17.s
    KAI_ASM_INST(0xc0914a02)  // addva za2.s, p2/M, p2/M, z16.s
    KAI_ASM_INST(0xc0914a03)  // addva za3.s, p2/M, p2/M, z16.s
    tbz x14, #1, label_15
    tbz x14, #0, label_13
    mov x12, #0
    cntw x21
    cntw x20
    cntw x8, ALL, MUL #2
    cntw x7, ALL, MUL #3
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xe0bf0960)  // st1w { za0h.s[x12] }, p2/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe09f09a0)  // ld1w { za0h.s[x12] }, p2/Z, [x13, XZR, LSL #2]
    addvl x6, x11, #4
    addvl x5, x13, #4
    KAI_ASM_INST(0xe0b40961)  // st1w { za0h.s[x12, #1] }, p2/Z, [x11, x20, LSL #2]
    KAI_ASM_INST(0xe09409a1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x13, x20, LSL #2]
    addvl x4, x11, #8
    addvl x3, x13, #8
    KAI_ASM_INST(0xe0a80962)  // st1w { za0h.s[x12, #2] }, p2/Z, [x11, x8, LSL #2]
    KAI_ASM_INST(0xe08809a2)  // ld1w { za0h.s[x12, #2] }, p2/Z, [x13, x8, LSL #2]
    addvl x2, x11, #12
    addvl x1, x13, #12
    KAI_ASM_INST(0xe0a70963)  // st1w { za0h.s[x12, #3] }, p2/Z, [x11, x7, LSL #2]
    KAI_ASM_INST(0xe08709a3)  // ld1w { za0h.s[x12, #3] }, p2/Z, [x13, x7, LSL #2]
    addvl x11, x11, #16
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0bf08c4)  // st1w { za1h.s[x12] }, p2/Z, [x6, XZR, LSL #2]
    KAI_ASM_INST(0xe09f08a4)  // ld1w { za1h.s[x12] }, p2/Z, [x5, XZR, LSL #2]
    KAI_ASM_INST(0xe0b408c5)  // st1w { za1h.s[x12, #1] }, p2/Z, [x6, x20, LSL #2]
    KAI_ASM_INST(0xe09408a5)  // ld1w { za1h.s[x12, #1] }, p2/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0xe0a808c6)  // st1w { za1h.s[x12, #2] }, p2/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0xe08808a6)  // ld1w { za1h.s[x12, #2] }, p2/Z, [x5, x8, LSL #2]
    KAI_ASM_INST(0xe0a708c7)  // st1w { za1h.s[x12, #3] }, p2/Z, [x6, x7, LSL #2]
    KAI_ASM_INST(0xe08708a7)  // ld1w { za1h.s[x12, #3] }, p2/Z, [x5, x7, LSL #2]
    KAI_ASM_INST(0xe0bf0888)  // st1w { za2h.s[x12] }, p2/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0868)  // ld1w { za2h.s[x12] }, p2/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b40889)  // st1w { za2h.s[x12, #1] }, p2/Z, [x4, x20, LSL #2]
    KAI_ASM_INST(0xe0940869)  // ld1w { za2h.s[x12, #1] }, p2/Z, [x3, x20, LSL #2]
    KAI_ASM_INST(0xe0a8088a)  // st1w { za2h.s[x12, #2] }, p2/Z, [x4, x8, LSL #2]
    KAI_ASM_INST(0xe088086a)  // ld1w { za2h.s[x12, #2] }, p2/Z, [x3, x8, LSL #2]
    KAI_ASM_INST(0xe0a7088b)  // st1w { za2h.s[x12, #3] }, p2/Z, [x4, x7, LSL #2]
    KAI_ASM_INST(0xe087086b)  // ld1w { za2h.s[x12, #3] }, p2/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bf084c)  // st1w { za3h.s[x12] }, p2/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f082c)  // ld1w { za3h.s[x12] }, p2/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4084d)  // st1w { za3h.s[x12, #1] }, p2/Z, [x2, x20, LSL #2]
    KAI_ASM_INST(0xe094082d)  // ld1w { za3h.s[x12, #1] }, p2/Z, [x1, x20, LSL #2]
    KAI_ASM_INST(0xe0a8084e)  // st1w { za3h.s[x12, #2] }, p2/Z, [x2, x8, LSL #2]
    KAI_ASM_INST(0xe088082e)  // ld1w { za3h.s[x12, #2] }, p2/Z, [x1, x8, LSL #2]
    KAI_ASM_INST(0xe0a7084f)  // st1w { za3h.s[x12, #3] }, p2/Z, [x2, x7, LSL #2]
    KAI_ASM_INST(0xe087082f)  // ld1w { za3h.s[x12, #3] }, p2/Z, [x1, x7, LSL #2]
    add x12, x12, #0x4
    cmp x12, x21
    blt label_12
    b label_25
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xe0bf0960)  // st1w { za0h.s[x12] }, p2/Z, [x11, XZR, LSL #2]
    addvl x3, x11, #4
    addvl x2, x11, #8
    KAI_ASM_INST(0xe0a60961)  // st1w { za0h.s[x12, #1] }, p2/Z, [x11, x6, LSL #2]
    addvl x1, x11, #12
    KAI_ASM_INST(0xe0a50962)  // st1w { za0h.s[x12, #2] }, p2/Z, [x11, x5, LSL #2]
    KAI_ASM_INST(0xe0a40963)  // st1w { za0h.s[x12, #3] }, p2/Z, [x11, x4, LSL #2]
    addvl x11, x11, #16
    KAI_ASM_INST(0xe0bf0864)  // st1w { za1h.s[x12] }, p2/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60865)  // st1w { za1h.s[x12, #1] }, p2/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0a50866)  // st1w { za1h.s[x12, #2] }, p2/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe0a40867)  // st1w { za1h.s[x12, #3] }, p2/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe0bf0848)  // st1w { za2h.s[x12] }, p2/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60849)  // st1w { za2h.s[x12, #1] }, p2/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe0a5084a)  // st1w { za2h.s[x12, #2] }, p2/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe0a4084b)  // st1w { za2h.s[x12, #3] }, p2/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe0bf082c)  // st1w { za3h.s[x12] }, p2/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0a6082d)  // st1w { za3h.s[x12, #1] }, p2/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe0a5082e)  // st1w { za3h.s[x12, #2] }, p2/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe0a4082f)  // st1w { za3h.s[x12, #3] }, p2/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_14
    b label_25
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x8, [x0, #0x50]
    add x7, x0, #0x98
    add x4, x0, #0x9c
    ldr x6, [x0, #0x58]
    add x3, x0, #0x94
    add x2, x0, #0xa0
    KAI_ASM_INST(0x8540c8fe)  // ld1rw { z30.s }, p2/Z, [x7]
    add x1, x0, #0xa4
    sub x5, x26, x28
    KAI_ASM_INST(0x8540c8fd)  // ld1rw { z29.s }, p2/Z, [x7]
    add x8, x8, x27  // C += n
    KAI_ASM_INST(0x8540c89c)  // ld1rw { z28.s }, p2/Z, [x4]
    madd x8, x28, x6, x8  // C += m * ldc
    KAI_ASM_INST(0x8540c89b)  // ld1rw { z27.s }, p2/Z, [x4]
    KAI_ASM_INST(0x8540c87a)  // ld1rw { z26.s }, p2/Z, [x3]
    KAI_ASM_INST(0x8540c859)  // ld1rw { z25.s }, p2/Z, [x2]
    KAI_ASM_INST(0x8540c838)  // ld1rw { z24.s }, p2/Z, [x1]
    tbz x14, #2, label_16
    ldr x2, [x0, #0x78]
    ldr x1, [x0, #0x88]
    add x2, x2, x27, LSL #2
    add x1, x1, x27, LSL #2
    ld1w { z30.s }, p1/Z, [x2]
    ld1w { z29.s }, p0/Z, [x2, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x1]
    ld1w { z27.s }, p0/Z, [x1, #1, MUL VL]
KAI_ASM_LABEL(label_16)  // Store to output array: Load per-channel parameters: End
    cntw x4
    whilelt p0.h, x27, x25
    cmp x5, x4
    mov x12, #0
    csel x3, x5, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820817)  // mova z23.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820891)  // mova z17.s, p2/M, za1h.s[x12]
    KAI_ASM_INST(0x04be72f7)  // sqdmulh z23.s, z23.s, z30.s
    KAI_ASM_INST(0xc0820836)  // mova z22.s, p2/M, za0h.s[x12, #1]
    KAI_ASM_INST(0x04bd7231)  // sqdmulh z17.s, z17.s, z29.s
    KAI_ASM_INST(0xc08208b0)  // mova z16.s, p2/M, za1h.s[x12, #1]
    KAI_ASM_INST(0x04be72d6)  // sqdmulh z22.s, z22.s, z30.s
    KAI_ASM_INST(0xc0820855)  // mova z21.s, p2/M, za0h.s[x12, #2]
    KAI_ASM_INST(0x04bd7210)  // sqdmulh z16.s, z16.s, z29.s
    KAI_ASM_INST(0xc08208d4)  // mova z20.s, p2/M, za1h.s[x12, #2]
    KAI_ASM_INST(0x04be72b5)  // sqdmulh z21.s, z21.s, z30.s
    KAI_ASM_INST(0x44828b97)  // srshl z23.s, p2/M, z23.s, z28.s
    KAI_ASM_INST(0xc0820873)  // mova z19.s, p2/M, za0h.s[x12, #3]
    KAI_ASM_INST(0x04bd7294)  // sqdmulh z20.s, z20.s, z29.s
    KAI_ASM_INST(0x44828b71)  // srshl z17.s, p2/M, z17.s, z27.s
    KAI_ASM_INST(0xc08208f2)  // mova z18.s, p2/M, za1h.s[x12, #3]
    KAI_ASM_INST(0x04be7273)  // sqdmulh z19.s, z19.s, z30.s
    KAI_ASM_INST(0x44828b96)  // srshl z22.s, p2/M, z22.s, z28.s
    add x12, x12, #0x4
    KAI_ASM_INST(0x04bd7252)  // sqdmulh z18.s, z18.s, z29.s
    KAI_ASM_INST(0x44828b70)  // srshl z16.s, p2/M, z16.s, z27.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0x44828b95)  // srshl z21.s, p2/M, z21.s, z28.s
    add z23.s, z23.s, z26.s
    KAI_ASM_INST(0x44828b74)  // srshl z20.s, p2/M, z20.s, z27.s
    add z17.s, z17.s, z26.s
    KAI_ASM_INST(0x44828b93)  // srshl z19.s, p2/M, z19.s, z28.s
    add z22.s, z22.s, z26.s
    KAI_ASM_INST(0x44828b72)  // srshl z18.s, p2/M, z18.s, z27.s
    add z16.s, z16.s, z26.s
    add z21.s, z21.s, z26.s
    smin z23.s, p2/M, z23.s, z24.s
    add z20.s, z20.s, z26.s
    smin z17.s, p2/M, z17.s, z24.s
    add z19.s, z19.s, z26.s
    smin z22.s, p2/M, z22.s, z24.s
    add z18.s, z18.s, z26.s
    smin z16.s, p2/M, z16.s, z24.s
    smin z21.s, p2/M, z21.s, z24.s
    smax z23.s, p2/M, z23.s, z25.s
    smin z20.s, p2/M, z20.s, z24.s
    smax z17.s, p2/M, z17.s, z25.s
    smin z19.s, p2/M, z19.s, z24.s
    smax z22.s, p2/M, z22.s, z25.s
    smin z18.s, p2/M, z18.s, z24.s
    smax z16.s, p2/M, z16.s, z25.s
    smax z21.s, p2/M, z21.s, z25.s
    smax z20.s, p2/M, z20.s, z25.s
    uzp1 z17.h, z23.h, z17.h
    smax z19.s, p2/M, z19.s, z25.s
    smax z18.s, p2/M, z18.s, z25.s
    uzp1 z16.h, z22.h, z16.h
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    uzp1 z17.h, z21.h, z20.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    uzp1 z16.h, z19.h, z18.h
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc0820815)  // mova z21.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820890)  // mova z16.s, p2/M, za1h.s[x12]
    KAI_ASM_INST(0x04be72b5)  // sqdmulh z21.s, z21.s, z30.s
    KAI_ASM_INST(0x04bd7210)  // sqdmulh z16.s, z16.s, z29.s
    KAI_ASM_INST(0xc0820834)  // mova z20.s, p2/M, za0h.s[x12, #1]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    KAI_ASM_INST(0x04be7294)  // sqdmulh z20.s, z20.s, z30.s
    KAI_ASM_INST(0xc08208b2)  // mova z18.s, p2/M, za1h.s[x12, #1]
    KAI_ASM_INST(0x04be7273)  // sqdmulh z19.s, z19.s, z30.s
    KAI_ASM_INST(0xc08208d1)  // mova z17.s, p2/M, za1h.s[x12, #2]
    KAI_ASM_INST(0x04bd7252)  // sqdmulh z18.s, z18.s, z29.s
    KAI_ASM_INST(0x44828b95)  // srshl z21.s, p2/M, z21.s, z28.s
    KAI_ASM_INST(0x04bd7231)  // sqdmulh z17.s, z17.s, z29.s
    KAI_ASM_INST(0x44828b70)  // srshl z16.s, p2/M, z16.s, z27.s
    KAI_ASM_INST(0x44828b94)  // srshl z20.s, p2/M, z20.s, z28.s
    KAI_ASM_INST(0x44828b93)  // srshl z19.s, p2/M, z19.s, z28.s
    KAI_ASM_INST(0x44828b72)  // srshl z18.s, p2/M, z18.s, z27.s
    add z21.s, z21.s, z26.s
    KAI_ASM_INST(0x44828b71)  // srshl z17.s, p2/M, z17.s, z27.s
    add z16.s, z16.s, z26.s
    add z20.s, z20.s, z26.s
    add z19.s, z19.s, z26.s
    add z18.s, z18.s, z26.s
    smin z21.s, p2/M, z21.s, z24.s
    add z17.s, z17.s, z26.s
    smin z16.s, p2/M, z16.s, z24.s
    smin z20.s, p2/M, z20.s, z24.s
    smin z19.s, p2/M, z19.s, z24.s
    smax z21.s, p2/M, z21.s, z25.s
    smin z18.s, p2/M, z18.s, z24.s
    smax z16.s, p2/M, z16.s, z25.s
    smin z17.s, p2/M, z17.s, z24.s
    smax z20.s, p2/M, z20.s, z25.s
    smax z19.s, p2/M, z19.s, z25.s
    smax z18.s, p2/M, z18.s, z25.s
    smax z17.s, p2/M, z17.s, z25.s
    uzp1 z16.h, z21.h, z16.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_19
    subs x1, x1, #0x1
    uzp1 z16.h, z20.h, z18.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_19
    uzp1 z16.h, z19.h, z17.h
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
    KAI_ASM_INST(0xc0820917)  // mova z23.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820991)  // mova z17.s, p2/M, za3h.s[x12]
    KAI_ASM_INST(0x04be72f7)  // sqdmulh z23.s, z23.s, z30.s
    KAI_ASM_INST(0xc0820936)  // mova z22.s, p2/M, za2h.s[x12, #1]
    KAI_ASM_INST(0x04bd7231)  // sqdmulh z17.s, z17.s, z29.s
    KAI_ASM_INST(0xc08209b0)  // mova z16.s, p2/M, za3h.s[x12, #1]
    KAI_ASM_INST(0x04be72d6)  // sqdmulh z22.s, z22.s, z30.s
    KAI_ASM_INST(0xc0820955)  // mova z21.s, p2/M, za2h.s[x12, #2]
    KAI_ASM_INST(0x04bd7210)  // sqdmulh z16.s, z16.s, z29.s
    KAI_ASM_INST(0xc08209d4)  // mova z20.s, p2/M, za3h.s[x12, #2]
    KAI_ASM_INST(0x04be72b5)  // sqdmulh z21.s, z21.s, z30.s
    KAI_ASM_INST(0x44828b97)  // srshl z23.s, p2/M, z23.s, z28.s
    KAI_ASM_INST(0xc0820973)  // mova z19.s, p2/M, za2h.s[x12, #3]
    KAI_ASM_INST(0x04bd7294)  // sqdmulh z20.s, z20.s, z29.s
    KAI_ASM_INST(0x44828b71)  // srshl z17.s, p2/M, z17.s, z27.s
    KAI_ASM_INST(0xc08209f2)  // mova z18.s, p2/M, za3h.s[x12, #3]
    KAI_ASM_INST(0x04be7273)  // sqdmulh z19.s, z19.s, z30.s
    KAI_ASM_INST(0x44828b96)  // srshl z22.s, p2/M, z22.s, z28.s
    add x12, x12, #0x4
    KAI_ASM_INST(0x04bd7252)  // sqdmulh z18.s, z18.s, z29.s
    KAI_ASM_INST(0x44828b70)  // srshl z16.s, p2/M, z16.s, z27.s
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0x44828b95)  // srshl z21.s, p2/M, z21.s, z28.s
    add z23.s, z23.s, z26.s
    KAI_ASM_INST(0x44828b74)  // srshl z20.s, p2/M, z20.s, z27.s
    add z17.s, z17.s, z26.s
    KAI_ASM_INST(0x44828b93)  // srshl z19.s, p2/M, z19.s, z28.s
    add z22.s, z22.s, z26.s
    KAI_ASM_INST(0x44828b72)  // srshl z18.s, p2/M, z18.s, z27.s
    add z16.s, z16.s, z26.s
    add z21.s, z21.s, z26.s
    smin z23.s, p2/M, z23.s, z24.s
    add z20.s, z20.s, z26.s
    smin z17.s, p2/M, z17.s, z24.s
    add z19.s, z19.s, z26.s
    smin z22.s, p2/M, z22.s, z24.s
    add z18.s, z18.s, z26.s
    smin z16.s, p2/M, z16.s, z24.s
    smin z21.s, p2/M, z21.s, z24.s
    smax z23.s, p2/M, z23.s, z25.s
    smin z20.s, p2/M, z20.s, z24.s
    smax z17.s, p2/M, z17.s, z25.s
    smin z19.s, p2/M, z19.s, z24.s
    smax z22.s, p2/M, z22.s, z25.s
    smin z18.s, p2/M, z18.s, z24.s
    smax z16.s, p2/M, z16.s, z25.s
    smax z21.s, p2/M, z21.s, z25.s
    smax z20.s, p2/M, z20.s, z25.s
    uzp1 z17.h, z23.h, z17.h
    smax z19.s, p2/M, z19.s, z25.s
    smax z18.s, p2/M, z18.s, z25.s
    uzp1 z16.h, z22.h, z16.h
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    uzp1 z17.h, z21.h, z20.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    uzp1 z16.h, z19.h, z18.h
    st1b { z17.h }, p0, [x8]
    add x8, x8, x6
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0820915)  // mova z21.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820990)  // mova z16.s, p2/M, za3h.s[x12]
    KAI_ASM_INST(0x04be72b5)  // sqdmulh z21.s, z21.s, z30.s
    KAI_ASM_INST(0x04bd7210)  // sqdmulh z16.s, z16.s, z29.s
    KAI_ASM_INST(0xc0820934)  // mova z20.s, p2/M, za2h.s[x12, #1]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    KAI_ASM_INST(0x04be7294)  // sqdmulh z20.s, z20.s, z30.s
    KAI_ASM_INST(0xc08209b2)  // mova z18.s, p2/M, za3h.s[x12, #1]
    KAI_ASM_INST(0x04be7273)  // sqdmulh z19.s, z19.s, z30.s
    KAI_ASM_INST(0xc08209d1)  // mova z17.s, p2/M, za3h.s[x12, #2]
    KAI_ASM_INST(0x04bd7252)  // sqdmulh z18.s, z18.s, z29.s
    KAI_ASM_INST(0x44828b95)  // srshl z21.s, p2/M, z21.s, z28.s
    KAI_ASM_INST(0x04bd7231)  // sqdmulh z17.s, z17.s, z29.s
    KAI_ASM_INST(0x44828b70)  // srshl z16.s, p2/M, z16.s, z27.s
    KAI_ASM_INST(0x44828b94)  // srshl z20.s, p2/M, z20.s, z28.s
    KAI_ASM_INST(0x44828b93)  // srshl z19.s, p2/M, z19.s, z28.s
    KAI_ASM_INST(0x44828b72)  // srshl z18.s, p2/M, z18.s, z27.s
    add z21.s, z21.s, z26.s
    KAI_ASM_INST(0x44828b71)  // srshl z17.s, p2/M, z17.s, z27.s
    add z16.s, z16.s, z26.s
    add z20.s, z20.s, z26.s
    add z19.s, z19.s, z26.s
    add z18.s, z18.s, z26.s
    smin z21.s, p2/M, z21.s, z24.s
    add z17.s, z17.s, z26.s
    smin z16.s, p2/M, z16.s, z24.s
    smin z20.s, p2/M, z20.s, z24.s
    smin z19.s, p2/M, z19.s, z24.s
    smax z21.s, p2/M, z21.s, z25.s
    smin z18.s, p2/M, z18.s, z24.s
    smax z16.s, p2/M, z16.s, z25.s
    smin z17.s, p2/M, z17.s, z24.s
    smax z20.s, p2/M, z20.s, z25.s
    smax z19.s, p2/M, z19.s, z25.s
    smax z18.s, p2/M, z18.s, z25.s
    smax z17.s, p2/M, z17.s, z25.s
    uzp1 z16.h, z21.h, z16.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_22
    subs x1, x1, #0x1
    uzp1 z16.h, z20.h, z18.h
    st1b { z16.h }, p0, [x8]
    add x8, x8, x6
    beq label_22
    uzp1 z16.h, z19.h, z17.h
    st1b { z16.h }, p0, [x8]
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_23)  // Store to output array: End
    tbz x14, #0, label_25
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_24)  // Store to output array: Refill accumulators: Loop
    addvl x3, x13, #4
    addvl x2, x13, #8
    KAI_ASM_INST(0xe09f09a0)  // ld1w { za0h.s[x12] }, p2/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0864)  // ld1w { za1h.s[x12] }, p2/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0848)  // ld1w { za2h.s[x12] }, p2/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f082c)  // ld1w { za3h.s[x12] }, p2/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08609a1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860865)  // ld1w { za1h.s[x12, #1] }, p2/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860849)  // ld1w { za2h.s[x12, #1] }, p2/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086082d)  // ld1w { za3h.s[x12, #1] }, p2/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08509a2)  // ld1w { za0h.s[x12, #2] }, p2/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850866)  // ld1w { za1h.s[x12, #2] }, p2/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085084a)  // ld1w { za2h.s[x12, #2] }, p2/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085082e)  // ld1w { za3h.s[x12, #2] }, p2/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08409a3)  // ld1w { za0h.s[x12, #3] }, p2/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840867)  // ld1w { za1h.s[x12, #3] }, p2/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084084b)  // ld1w { za2h.s[x12, #3] }, p2/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084082f)  // ld1w { za3h.s[x12, #3] }, p2/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_24
KAI_ASM_LABEL(label_25)  // End block
    incw x27, ALL, MUL #2
    cmp x27, x25
    blt label_4
    ldr x1, [x0, #0x70]
    incw x28, ALL, MUL #2
    mov x11, x10
    cmp x28, x26
    mov x27, #0
    mov x24, x22
    add x10, x11, x1
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_u8q_mopa_2VLx2VL)

    KAI_ASM_END
