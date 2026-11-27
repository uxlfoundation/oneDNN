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

    KAI_ASM_CODE(sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
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
    ldr x3, [x0, #0x50]
    ldr x13, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x14, x14, x1
    cset x1, EQ
    orr x14, x14, x1, LSL #1
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
    whilelt p8.s, x1, x25
    incw x1
    whilelt p8.s, x1, x25
    incw x1
    whilelt p8.s, x1, x25
    incw x1
    whilelt p8.s, x1, x25
    tbnz x14, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    mov x2, x27
    fmov z21.h, #0.0
    add x1, x1, x27, LSL #1
    fmov z20.h, #1.0
    whilelt p1.h, x2, x25
    inch x2
    whilelt p0.h, x2, x25
    ldnt1h { z17.h }, p1/Z, [x1]
    ldnt1h { z16.h }, p0/Z, [x1, #1, MUL VL]
    zip1 z19.h, z17.h, z21.h
    zip2 z18.h, z17.h, z21.h
    zip1 z17.h, z16.h, z21.h
    zip2 z16.h, z16.h, z21.h
    KAI_ASM_INST(0x81b34a80)  // fmopa za0.s, p2/M, p2/M, z20.h, z19.h
    KAI_ASM_INST(0x81b24a81)  // fmopa za1.s, p2/M, p2/M, z20.h, z18.h
    KAI_ASM_INST(0x81b14a82)  // fmopa za2.s, p2/M, p2/M, z20.h, z17.h
    KAI_ASM_INST(0x81b04a83)  // fmopa za3.s, p2/M, p2/M, z20.h, z16.h
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x27
    mov x2, x28
    incw x1, ALL, MUL #4
    incw x2
    cmp x1, x25
    mov x1, x14
    csel x2, x28, x2, LT
    bfm x14, XZR, #0, #0  // bfc x14, #0, #0x1
    csel x13, x13, x10, LT
    cmp x2, x26
    csel x14, x1, x14, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x4, x9, #0x2
    and x3, x9, #0x3
    cbz x4, label_9
    addvl x2, x23, #8
    addvl x1, x23, #12
    ld1h { z3.h }, p2/Z, [x22]
    subs x4, x4, #0x1
    ld1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    ld1h { z1.h }, p2/Z, [x22, #2, MUL VL]
    ld1h { z0.h }, p2/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    ld1h { z31.h }, p2/Z, [x23]
    ld1h { z30.h }, p2/Z, [x23, #1, MUL VL]
    ld1h { z29.h }, p2/Z, [x23, #2, MUL VL]
    ld1h { z28.h }, p2/Z, [x23, #3, MUL VL]
    ld1h { z27.h }, p2/Z, [x23, #4, MUL VL]
    ld1h { z26.h }, p2/Z, [x23, #5, MUL VL]
    ld1h { z25.h }, p2/Z, [x23, #6, MUL VL]
    ld1h { z24.h }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #16
    ld1h { z23.h }, p2/Z, [x2]
    ld1h { z22.h }, p2/Z, [x2, #1, MUL VL]
    ld1h { z21.h }, p2/Z, [x2, #2, MUL VL]
    ld1h { z20.h }, p2/Z, [x2, #3, MUL VL]
    ld1h { z19.h }, p2/Z, [x1]
    ld1h { z18.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z17.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x1, #3, MUL VL]
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81bf4860)  // fmopa za0.s, p2/M, p2/M, z3.h, z31.h
    addvl x2, x23, #8
    addvl x1, x23, #12
    ld1h { z31.h }, p2/Z, [x23]
    KAI_ASM_INST(0x81be4861)  // fmopa za1.s, p2/M, p2/M, z3.h, z30.h
    subs x4, x4, #0x1
    ld1h { z30.h }, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x81bd4862)  // fmopa za2.s, p2/M, p2/M, z3.h, z29.h
    ld1h { z29.h }, p2/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x81bc4863)  // fmopa za3.s, p2/M, p2/M, z3.h, z28.h
    ld1h { z3.h }, p2/Z, [x22]
    KAI_ASM_INST(0x81bb4840)  // fmopa za0.s, p2/M, p2/M, z2.h, z27.h
    ld1h { z28.h }, p2/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0x81ba4841)  // fmopa za1.s, p2/M, p2/M, z2.h, z26.h
    ld1h { z27.h }, p2/Z, [x23, #4, MUL VL]
    KAI_ASM_INST(0x81b94842)  // fmopa za2.s, p2/M, p2/M, z2.h, z25.h
    ld1h { z26.h }, p2/Z, [x23, #5, MUL VL]
    KAI_ASM_INST(0x81b84843)  // fmopa za3.s, p2/M, p2/M, z2.h, z24.h
    ld1h { z2.h }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x81b74820)  // fmopa za0.s, p2/M, p2/M, z1.h, z23.h
    ld1h { z25.h }, p2/Z, [x23, #6, MUL VL]
    KAI_ASM_INST(0x81b64821)  // fmopa za1.s, p2/M, p2/M, z1.h, z22.h
    ld1h { z24.h }, p2/Z, [x23, #7, MUL VL]
    addvl x23, x23, #16
    KAI_ASM_INST(0x81b54822)  // fmopa za2.s, p2/M, p2/M, z1.h, z21.h
    ld1h { z23.h }, p2/Z, [x2]
    KAI_ASM_INST(0x81b44823)  // fmopa za3.s, p2/M, p2/M, z1.h, z20.h
    ld1h { z1.h }, p2/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x81b34800)  // fmopa za0.s, p2/M, p2/M, z0.h, z19.h
    ld1h { z22.h }, p2/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x81b24801)  // fmopa za1.s, p2/M, p2/M, z0.h, z18.h
    ld1h { z21.h }, p2/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0x81b14802)  // fmopa za2.s, p2/M, p2/M, z0.h, z17.h
    ld1h { z20.h }, p2/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x81b04803)  // fmopa za3.s, p2/M, p2/M, z0.h, z16.h
    ld1h { z0.h }, p2/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    ld1h { z19.h }, p2/Z, [x1]
    ld1h { z18.h }, p2/Z, [x1, #1, MUL VL]
    ld1h { z17.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x1, #3, MUL VL]
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81bf4860)  // fmopa za0.s, p2/M, p2/M, z3.h, z31.h
    KAI_ASM_INST(0x81be4861)  // fmopa za1.s, p2/M, p2/M, z3.h, z30.h
    KAI_ASM_INST(0x81bd4862)  // fmopa za2.s, p2/M, p2/M, z3.h, z29.h
    KAI_ASM_INST(0x81bc4863)  // fmopa za3.s, p2/M, p2/M, z3.h, z28.h
    KAI_ASM_INST(0x81bb4840)  // fmopa za0.s, p2/M, p2/M, z2.h, z27.h
    KAI_ASM_INST(0x81ba4841)  // fmopa za1.s, p2/M, p2/M, z2.h, z26.h
    KAI_ASM_INST(0x81b94842)  // fmopa za2.s, p2/M, p2/M, z2.h, z25.h
    KAI_ASM_INST(0x81b84843)  // fmopa za3.s, p2/M, p2/M, z2.h, z24.h
    KAI_ASM_INST(0x81b74820)  // fmopa za0.s, p2/M, p2/M, z1.h, z23.h
    KAI_ASM_INST(0x81b64821)  // fmopa za1.s, p2/M, p2/M, z1.h, z22.h
    KAI_ASM_INST(0x81b54822)  // fmopa za2.s, p2/M, p2/M, z1.h, z21.h
    KAI_ASM_INST(0x81b44823)  // fmopa za3.s, p2/M, p2/M, z1.h, z20.h
    KAI_ASM_INST(0x81b34800)  // fmopa za0.s, p2/M, p2/M, z0.h, z19.h
    KAI_ASM_INST(0x81b24801)  // fmopa za1.s, p2/M, p2/M, z0.h, z18.h
    KAI_ASM_INST(0x81b14802)  // fmopa za2.s, p2/M, p2/M, z0.h, z17.h
    KAI_ASM_INST(0x81b04803)  // fmopa za3.s, p2/M, p2/M, z0.h, z16.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x3, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1h { z20.h }, p2/Z, [x22]
    subs x3, x3, #0x1
    addvl x22, x22, #1
    ld1h { z19.h }, p2/Z, [x23]
    ld1h { z18.h }, p2/Z, [x23, #1, MUL VL]
    ld1h { z17.h }, p2/Z, [x23, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0x81b34a80)  // fmopa za0.s, p2/M, p2/M, z20.h, z19.h
    KAI_ASM_INST(0x81b24a81)  // fmopa za1.s, p2/M, p2/M, z20.h, z18.h
    KAI_ASM_INST(0x81b14a82)  // fmopa za2.s, p2/M, p2/M, z20.h, z17.h
    KAI_ASM_INST(0x81b04a83)  // fmopa za3.s, p2/M, p2/M, z20.h, z16.h
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
    b label_19
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
    b label_19
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    mov x6, x27
    sub x5, x26, x28
    ldr x4, [x0, #0x58]
    cntw x3
    whilelt p1.h, x6, x25
    inch x6
    add x2, x0, #0xa0
    add x7, x7, x27, LSL #1  // C += n
    add x1, x0, #0xa4
    KAI_ASM_INST(0x84c0a855)  // ld1rh { z21.h }, p2/Z, [x2]
    whilelt p0.h, x6, x25
    cmp x5, x3
    KAI_ASM_INST(0x84c0a834)  // ld1rh { z20.h }, p2/Z, [x1]
    madd x7, x28, x4, x7  // C += m * ldc
    mov x12, #0
    csel x5, x5, x3, LT
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc0020813)  // mova z19.b, p2/M, za0h.b[x12]
    KAI_ASM_INST(0xc0020831)  // mova z17.b, p2/M, za0h.b[x12, #1]
    fcvt z19.h, p2/m, z19.s
    KAI_ASM_INST(0xc0020852)  // mova z18.b, p2/M, za0h.b[x12, #2]
    fcvt z17.h, p2/m, z17.s
    KAI_ASM_INST(0xc0020870)  // mova z16.b, p2/M, za0h.b[x12, #3]
    fcvt z18.h, p2/m, z18.s
    add x12, x12, #0x4
    fcvt z16.h, p2/m, z16.s
    cmp x12, x5, LSL #2
    uzp1 z17.h, z19.h, z17.h
    uzp1 z16.h, z18.h, z16.h
    fmin z17.h, p2/M, z17.h, z20.h
    fmin z16.h, p2/M, z16.h, z20.h
    fmax z17.h, p2/M, z17.h, z21.h
    fmax z16.h, p2/M, z16.h, z21.h
    st1h { z17.h }, p1, [x7]
    st1h { z16.h }, p0, [x7, #1, MUL VL]
    add x7, x7, x4
    blt label_16
    tbz x14, #0, label_19
    mov x12, #0
    cntw x7
    cntw x6
    cntw x5, ALL, MUL #2
    cntw x4, ALL, MUL #3
KAI_ASM_LABEL(label_18)  // Store to output array: Refill accumulators: Loop
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
    blt label_18
KAI_ASM_LABEL(label_19)  // End block
    incw x27, ALL, MUL #4
    cmp x27, x25
    blt label_4
    ldr x1, [x0, #0x70]
    incw x28
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)

    KAI_ASM_END
