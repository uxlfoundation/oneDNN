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

    KAI_ASM_CODE(sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL)
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
    mov x6, #0xff800000
    mov x5, #0x7f800000
    ldr x2, [x0, #0x50]
    ptrue p2.b
    ldr w4, [x0, #0xa0]
    cmp x1, XZR
    ldr w3, [x0, #0xa4]
    cset x1, NE
    cmp x2, XZR
    ldr x13, [x0, #0x68]
    orr x14, x14, x1
    cset x1, EQ
    ldr x2, [x0, #0x70]
    cmp x4, x6
    orr x14, x14, x1, LSL #1
    cset x1, NE
    cmp x3, x5
    orr x14, x14, x1, LSL #2
    cset x1, NE
    orr x14, x14, x1, LSL #2
    mov x11, x13
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
    add x9, x9, #0x1
    ldr x24, [x0, #0x20]
    lsr x9, x9, #0x1
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
    fmov z18.s, #1.0
    ld1w { z17.s }, p1/Z, [x1]
    ld1w { z16.s }, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x80914a40)  // fmopa za0.s, p2/M, p2/M, z18.s, z17.s
    KAI_ASM_INST(0x80904a41)  // fmopa za1.s, p2/M, p2/M, z18.s, z16.s
    KAI_ASM_INST(0x80914a42)  // fmopa za2.s, p2/M, p2/M, z18.s, z17.s
    KAI_ASM_INST(0x80904a43)  // fmopa za3.s, p2/M, p2/M, z18.s, z16.s
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
    ld1h { z31.h }, p2/Z, [x22]
    ld1h { z30.h }, p2/Z, [x22, #1, MUL VL]
    ld1h { z29.h }, p2/Z, [x22, #2, MUL VL]
    ld1h { z28.h }, p2/Z, [x22, #3, MUL VL]
    ld1h { z27.h }, p2/Z, [x22, #4, MUL VL]
    ld1h { z26.h }, p2/Z, [x22, #5, MUL VL]
    ld1h { z25.h }, p2/Z, [x22, #6, MUL VL]
    ld1h { z24.h }, p2/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1h { z23.h }, p2/Z, [x23]
    ld1h { z22.h }, p2/Z, [x23, #1, MUL VL]
    ld1h { z21.h }, p2/Z, [x23, #2, MUL VL]
    ld1h { z20.h }, p2/Z, [x23, #3, MUL VL]
    ld1h { z19.h }, p2/Z, [x23, #4, MUL VL]
    ld1h { z18.h }, p2/Z, [x23, #5, MUL VL]
    ld1h { z17.h }, p2/Z, [x23, #6, MUL VL]
    ld1h { z16.h }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81974be0)  // bfmopa za0.s, p2/M, p2/M, z31.h, z23.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81964be1)  // bfmopa za1.s, p2/M, p2/M, z31.h, z22.h
    ld1h { z31.h }, p2/Z, [x22]
    KAI_ASM_INST(0x81974bc2)  // bfmopa za2.s, p2/M, p2/M, z30.h, z23.h
    ld1h { z23.h }, p2/Z, [x23]
    KAI_ASM_INST(0x81964bc3)  // bfmopa za3.s, p2/M, p2/M, z30.h, z22.h
    ld1h { z30.h }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x81954ba0)  // bfmopa za0.s, p2/M, p2/M, z29.h, z21.h
    ld1h { z22.h }, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x81944ba1)  // bfmopa za1.s, p2/M, p2/M, z29.h, z20.h
    ld1h { z29.h }, p2/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x81954b82)  // bfmopa za2.s, p2/M, p2/M, z28.h, z21.h
    ld1h { z21.h }, p2/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x81944b83)  // bfmopa za3.s, p2/M, p2/M, z28.h, z20.h
    ld1h { z28.h }, p2/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0x81934b60)  // bfmopa za0.s, p2/M, p2/M, z27.h, z19.h
    ld1h { z20.h }, p2/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0x81924b61)  // bfmopa za1.s, p2/M, p2/M, z27.h, z18.h
    ld1h { z27.h }, p2/Z, [x22, #4, MUL VL]
    KAI_ASM_INST(0x81934b42)  // bfmopa za2.s, p2/M, p2/M, z26.h, z19.h
    ld1h { z19.h }, p2/Z, [x23, #4, MUL VL]
    KAI_ASM_INST(0x81924b43)  // bfmopa za3.s, p2/M, p2/M, z26.h, z18.h
    ld1h { z26.h }, p2/Z, [x22, #5, MUL VL]
    KAI_ASM_INST(0x81914b20)  // bfmopa za0.s, p2/M, p2/M, z25.h, z17.h
    ld1h { z18.h }, p2/Z, [x23, #5, MUL VL]
    KAI_ASM_INST(0x81904b21)  // bfmopa za1.s, p2/M, p2/M, z25.h, z16.h
    ld1h { z25.h }, p2/Z, [x22, #6, MUL VL]
    KAI_ASM_INST(0x81914b02)  // bfmopa za2.s, p2/M, p2/M, z24.h, z17.h
    ld1h { z17.h }, p2/Z, [x23, #6, MUL VL]
    KAI_ASM_INST(0x81904b03)  // bfmopa za3.s, p2/M, p2/M, z24.h, z16.h
    ld1h { z24.h }, p2/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1h { z16.h }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81974be0)  // bfmopa za0.s, p2/M, p2/M, z31.h, z23.h
    KAI_ASM_INST(0x81964be1)  // bfmopa za1.s, p2/M, p2/M, z31.h, z22.h
    KAI_ASM_INST(0x81974bc2)  // bfmopa za2.s, p2/M, p2/M, z30.h, z23.h
    KAI_ASM_INST(0x81964bc3)  // bfmopa za3.s, p2/M, p2/M, z30.h, z22.h
    KAI_ASM_INST(0x81954ba0)  // bfmopa za0.s, p2/M, p2/M, z29.h, z21.h
    KAI_ASM_INST(0x81944ba1)  // bfmopa za1.s, p2/M, p2/M, z29.h, z20.h
    KAI_ASM_INST(0x81954b82)  // bfmopa za2.s, p2/M, p2/M, z28.h, z21.h
    KAI_ASM_INST(0x81944b83)  // bfmopa za3.s, p2/M, p2/M, z28.h, z20.h
    KAI_ASM_INST(0x81934b60)  // bfmopa za0.s, p2/M, p2/M, z27.h, z19.h
    KAI_ASM_INST(0x81924b61)  // bfmopa za1.s, p2/M, p2/M, z27.h, z18.h
    KAI_ASM_INST(0x81934b42)  // bfmopa za2.s, p2/M, p2/M, z26.h, z19.h
    KAI_ASM_INST(0x81924b43)  // bfmopa za3.s, p2/M, p2/M, z26.h, z18.h
    KAI_ASM_INST(0x81914b20)  // bfmopa za0.s, p2/M, p2/M, z25.h, z17.h
    KAI_ASM_INST(0x81904b21)  // bfmopa za1.s, p2/M, p2/M, z25.h, z16.h
    KAI_ASM_INST(0x81914b02)  // bfmopa za2.s, p2/M, p2/M, z24.h, z17.h
    KAI_ASM_INST(0x81904b03)  // bfmopa za3.s, p2/M, p2/M, z24.h, z16.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1h { z19.h }, p2/Z, [x22]
    subs x1, x1, #0x1
    ld1h { z18.h }, p2/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ld1h { z17.h }, p2/Z, [x23]
    ld1h { z16.h }, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x81914a60)  // bfmopa za0.s, p2/M, p2/M, z19.h, z17.h
    KAI_ASM_INST(0x81904a61)  // bfmopa za1.s, p2/M, p2/M, z19.h, z16.h
    KAI_ASM_INST(0x81914a42)  // bfmopa za2.s, p2/M, p2/M, z18.h, z17.h
    KAI_ASM_INST(0x81904a43)  // bfmopa za3.s, p2/M, p2/M, z18.h, z16.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
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
    b label_31
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
    b label_31
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x26, x28
    ldr x5, [x0, #0x58]
    add x7, x7, x27, LSL #2  // C += n
    madd x7, x28, x5, x7  // C += m * ldc
    tbnz x14, #2, label_22
    cntw x4
    mov x12, #0
    cmp x6, x4
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Skip activation: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820810)  // mova z16.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820896)  // mova z22.s, p2/M, za1h.s[x12]
    st1w { z16.s }, p1, [x7]
    st1w { z22.s }, p0, [x7, #1, MUL VL]
    KAI_ASM_INST(0xc0820835)  // mova z21.s, p2/M, za0h.s[x12, #1]
    add x7, x7, x5
    KAI_ASM_INST(0xc0820854)  // mova z20.s, p2/M, za0h.s[x12, #2]
    KAI_ASM_INST(0xc0820873)  // mova z19.s, p2/M, za0h.s[x12, #3]
    st1w { z21.s }, p1, [x7]
    KAI_ASM_INST(0xc08208b2)  // mova z18.s, p2/M, za1h.s[x12, #1]
    KAI_ASM_INST(0xc08208d1)  // mova z17.s, p2/M, za1h.s[x12, #2]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    KAI_ASM_INST(0xc08208f0)  // mova z16.s, p2/M, za1h.s[x12, #3]
    add x12, x12, #0x4
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Skip activation: Accumulator row 0 oddments
    cbz x1, label_18
    KAI_ASM_INST(0xc0820815)  // mova z21.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820834)  // mova z20.s, p2/M, za0h.s[x12, #1]
    st1w { z21.s }, p1, [x7]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    KAI_ASM_INST(0xc0820892)  // mova z18.s, p2/M, za1h.s[x12]
    KAI_ASM_INST(0xc08208b1)  // mova z17.s, p2/M, za1h.s[x12, #1]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    KAI_ASM_INST(0xc08208d0)  // mova z16.s, p2/M, za1h.s[x12, #2]
    beq label_18
    subs x1, x1, #0x1
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_18
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
KAI_ASM_LABEL(label_18)  // Store to output array: Skip activation: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_22
    cmp x6, x4
    mov x12, #0
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_20
KAI_ASM_LABEL(label_19)  // Store to output array: Skip activation: Accumulator row 1 loop
    KAI_ASM_INST(0xc0820910)  // mova z16.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820996)  // mova z22.s, p2/M, za3h.s[x12]
    st1w { z16.s }, p1, [x7]
    st1w { z22.s }, p0, [x7, #1, MUL VL]
    KAI_ASM_INST(0xc0820935)  // mova z21.s, p2/M, za2h.s[x12, #1]
    add x7, x7, x5
    KAI_ASM_INST(0xc0820954)  // mova z20.s, p2/M, za2h.s[x12, #2]
    KAI_ASM_INST(0xc0820973)  // mova z19.s, p2/M, za2h.s[x12, #3]
    st1w { z21.s }, p1, [x7]
    KAI_ASM_INST(0xc08209b2)  // mova z18.s, p2/M, za3h.s[x12, #1]
    KAI_ASM_INST(0xc08209d1)  // mova z17.s, p2/M, za3h.s[x12, #2]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    KAI_ASM_INST(0xc08209f0)  // mova z16.s, p2/M, za3h.s[x12, #3]
    add x12, x12, #0x4
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    blt label_19
KAI_ASM_LABEL(label_20)  // Store to output array: Skip activation: Accumulator row 1 oddments
    cbz x1, label_21
    KAI_ASM_INST(0xc0820915)  // mova z21.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820934)  // mova z20.s, p2/M, za2h.s[x12, #1]
    st1w { z21.s }, p1, [x7]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    KAI_ASM_INST(0xc0820992)  // mova z18.s, p2/M, za3h.s[x12]
    KAI_ASM_INST(0xc08209b1)  // mova z17.s, p2/M, za3h.s[x12, #1]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    KAI_ASM_INST(0xc08209d0)  // mova z16.s, p2/M, za3h.s[x12, #2]
    beq label_21
    subs x1, x1, #0x1
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_21
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
KAI_ASM_LABEL(label_21)  // Store to output array: Skip activation: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_22
    b label_29
KAI_ASM_LABEL(label_22)  // Store to output array: Skip activation: End
    cntw x4
    add x2, x0, #0xa0
    cmp x6, x4
    add x1, x0, #0xa4
    KAI_ASM_INST(0x8540c859)  // ld1rw { z25.s }, p2/Z, [x2]
    csel x3, x6, x4, LT
    KAI_ASM_INST(0x8540c838)  // ld1rw { z24.s }, p2/Z, [x1]
    mov x12, #0
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_24
KAI_ASM_LABEL(label_23)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820817)  // mova z23.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820896)  // mova z22.s, p2/M, za1h.s[x12]
    fmin z23.s, p2/M, z23.s, z24.s
    KAI_ASM_INST(0xc0820835)  // mova z21.s, p2/M, za0h.s[x12, #1]
    fmin z22.s, p2/M, z22.s, z24.s
    KAI_ASM_INST(0xc08208b4)  // mova z20.s, p2/M, za1h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    KAI_ASM_INST(0xc08208d2)  // mova z18.s, p2/M, za1h.s[x12, #2]
    fmin z19.s, p2/M, z19.s, z24.s
    fmax z23.s, p2/M, z23.s, z25.s
    KAI_ASM_INST(0xc0820871)  // mova z17.s, p2/M, za0h.s[x12, #3]
    fmin z18.s, p2/M, z18.s, z24.s
    fmax z22.s, p2/M, z22.s, z25.s
    KAI_ASM_INST(0xc08208f0)  // mova z16.s, p2/M, za1h.s[x12, #3]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    add x12, x12, #0x4
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    cmp x12, x2, LSL #2
    st1w { z23.s }, p1, [x7]
    fmax z19.s, p2/M, z19.s, z25.s
    st1w { z22.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    fmax z18.s, p2/M, z18.s, z25.s
    st1w { z21.s }, p1, [x7]
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z20.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z19.s }, p1, [x7]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    st1w { z17.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    blt label_23
KAI_ASM_LABEL(label_24)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_25
    KAI_ASM_INST(0xc0820815)  // mova z21.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820834)  // mova z20.s, p2/M, za0h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820892)  // mova z18.s, p2/M, za1h.s[x12]
    fmin z19.s, p2/M, z19.s, z24.s
    KAI_ASM_INST(0xc08208b1)  // mova z17.s, p2/M, za1h.s[x12, #1]
    fmin z18.s, p2/M, z18.s, z24.s
    KAI_ASM_INST(0xc08208d0)  // mova z16.s, p2/M, za1h.s[x12, #2]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    fmax z19.s, p2/M, z19.s, z25.s
    fmax z18.s, p2/M, z18.s, z25.s
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z21.s }, p1, [x7]
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_25
    subs x1, x1, #0x1
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_25
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
KAI_ASM_LABEL(label_25)  // Store to output array: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_29
    cmp x6, x4
    mov x12, #0
    csel x1, x6, x4, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_27
KAI_ASM_LABEL(label_26)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0820917)  // mova z23.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820996)  // mova z22.s, p2/M, za3h.s[x12]
    fmin z23.s, p2/M, z23.s, z24.s
    KAI_ASM_INST(0xc0820935)  // mova z21.s, p2/M, za2h.s[x12, #1]
    fmin z22.s, p2/M, z22.s, z24.s
    KAI_ASM_INST(0xc08209b4)  // mova z20.s, p2/M, za3h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    KAI_ASM_INST(0xc08209d2)  // mova z18.s, p2/M, za3h.s[x12, #2]
    fmin z19.s, p2/M, z19.s, z24.s
    fmax z23.s, p2/M, z23.s, z25.s
    KAI_ASM_INST(0xc0820971)  // mova z17.s, p2/M, za2h.s[x12, #3]
    fmin z18.s, p2/M, z18.s, z24.s
    fmax z22.s, p2/M, z22.s, z25.s
    KAI_ASM_INST(0xc08209f0)  // mova z16.s, p2/M, za3h.s[x12, #3]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    add x12, x12, #0x4
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    cmp x12, x2, LSL #2
    st1w { z23.s }, p1, [x7]
    fmax z19.s, p2/M, z19.s, z25.s
    st1w { z22.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    fmax z18.s, p2/M, z18.s, z25.s
    st1w { z21.s }, p1, [x7]
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z20.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z19.s }, p1, [x7]
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    st1w { z17.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    blt label_26
KAI_ASM_LABEL(label_27)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_28
    KAI_ASM_INST(0xc0820915)  // mova z21.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820934)  // mova z20.s, p2/M, za2h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0820992)  // mova z18.s, p2/M, za3h.s[x12]
    fmin z19.s, p2/M, z19.s, z24.s
    KAI_ASM_INST(0xc08209b1)  // mova z17.s, p2/M, za3h.s[x12, #1]
    fmin z18.s, p2/M, z18.s, z24.s
    KAI_ASM_INST(0xc08209d0)  // mova z16.s, p2/M, za3h.s[x12, #2]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    fmax z19.s, p2/M, z19.s, z25.s
    fmax z18.s, p2/M, z18.s, z25.s
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z21.s }, p1, [x7]
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z18.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_28
    subs x1, x1, #0x1
    st1w { z20.s }, p1, [x7]
    st1w { z17.s }, p0, [x7, #1, MUL VL]
    add x7, x7, x5
    beq label_28
    st1w { z19.s }, p1, [x7]
    st1w { z16.s }, p0, [x7, #1, MUL VL]
KAI_ASM_LABEL(label_28)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_29)  // Store to output array: End
    tbz x14, #0, label_31
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_30)  // Store to output array: Refill accumulators: Loop
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
    blt label_30
KAI_ASM_LABEL(label_31)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL)

    KAI_ASM_END
