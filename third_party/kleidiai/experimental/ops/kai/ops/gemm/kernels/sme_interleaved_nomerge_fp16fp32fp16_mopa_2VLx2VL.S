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

    KAI_ASM_CODE(sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)
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
    ptrue p1.b
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
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0464)  // ld1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0448)  // ld1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08605a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860465)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860449)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08505a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850466)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085044a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08405a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840467)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084044b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
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
    tbnz x14, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z19.h, #0.0
    whilelt p0.h, x27, x25
    fmov z18.h, #1.0
    ld1h { z16.h }, p0/Z, [x1, x27, LSL #1]
    zip1 z17.h, z16.h, z19.h
    zip2 z16.h, z16.h, z19.h
    KAI_ASM_INST(0x81b12640)  // fmopa za0.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02641)  // fmopa za1.s, p1/M, p1/M, z18.h, z16.h
    KAI_ASM_INST(0x81b12642)  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02643)  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h
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
    ld1h { z31.h }, p1/Z, [x22]
    ld1h { z30.h }, p1/Z, [x22, #1, MUL VL]
    ld1h { z29.h }, p1/Z, [x22, #2, MUL VL]
    ld1h { z28.h }, p1/Z, [x22, #3, MUL VL]
    ld1h { z27.h }, p1/Z, [x22, #4, MUL VL]
    ld1h { z26.h }, p1/Z, [x22, #5, MUL VL]
    ld1h { z25.h }, p1/Z, [x22, #6, MUL VL]
    ld1h { z24.h }, p1/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1h { z23.h }, p1/Z, [x23]
    ld1h { z22.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z21.h }, p1/Z, [x23, #2, MUL VL]
    ld1h { z20.h }, p1/Z, [x23, #3, MUL VL]
    ld1h { z19.h }, p1/Z, [x23, #4, MUL VL]
    ld1h { z18.h }, p1/Z, [x23, #5, MUL VL]
    ld1h { z17.h }, p1/Z, [x23, #6, MUL VL]
    ld1h { z16.h }, p1/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81b727e0)  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81b627e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h
    ld1h { z31.h }, p1/Z, [x22]
    KAI_ASM_INST(0x81b727c2)  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h
    ld1h { z23.h }, p1/Z, [x23]
    KAI_ASM_INST(0x81b627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h
    ld1h { z30.h }, p1/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x81b527a0)  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h
    ld1h { z22.h }, p1/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x81b427a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h
    ld1h { z29.h }, p1/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x81b52782)  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h
    ld1h { z21.h }, p1/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0x81b42783)  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h
    ld1h { z28.h }, p1/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0x81b32760)  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h
    ld1h { z20.h }, p1/Z, [x23, #3, MUL VL]
    KAI_ASM_INST(0x81b22761)  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h
    ld1h { z27.h }, p1/Z, [x22, #4, MUL VL]
    KAI_ASM_INST(0x81b32742)  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h
    ld1h { z19.h }, p1/Z, [x23, #4, MUL VL]
    KAI_ASM_INST(0x81b22743)  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h
    ld1h { z26.h }, p1/Z, [x22, #5, MUL VL]
    KAI_ASM_INST(0x81b12720)  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h
    ld1h { z18.h }, p1/Z, [x23, #5, MUL VL]
    KAI_ASM_INST(0x81b02721)  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h
    ld1h { z25.h }, p1/Z, [x22, #6, MUL VL]
    KAI_ASM_INST(0x81b12702)  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h
    ld1h { z17.h }, p1/Z, [x23, #6, MUL VL]
    KAI_ASM_INST(0x81b02703)  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h
    ld1h { z24.h }, p1/Z, [x22, #7, MUL VL]
    addvl x22, x22, #8
    ld1h { z16.h }, p1/Z, [x23, #7, MUL VL]
    addvl x23, x23, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81b727e0)  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h
    KAI_ASM_INST(0x81b627e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h
    KAI_ASM_INST(0x81b727c2)  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h
    KAI_ASM_INST(0x81b627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h
    KAI_ASM_INST(0x81b527a0)  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h
    KAI_ASM_INST(0x81b427a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h
    KAI_ASM_INST(0x81b52782)  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h
    KAI_ASM_INST(0x81b42783)  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h
    KAI_ASM_INST(0x81b32760)  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h
    KAI_ASM_INST(0x81b22761)  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h
    KAI_ASM_INST(0x81b32742)  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h
    KAI_ASM_INST(0x81b22743)  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h
    KAI_ASM_INST(0x81b12720)  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h
    KAI_ASM_INST(0x81b02721)  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h
    KAI_ASM_INST(0x81b12702)  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h
    KAI_ASM_INST(0x81b02703)  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1h { z19.h }, p1/Z, [x22]
    subs x1, x1, #0x1
    ld1h { z18.h }, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ld1h { z17.h }, p1/Z, [x23]
    ld1h { z16.h }, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    KAI_ASM_INST(0x81b02661)  // fmopa za1.s, p1/M, p1/M, z19.h, z16.h
    KAI_ASM_INST(0x81b12642)  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02643)  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h
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
    KAI_ASM_INST(0xe0bf0560)  // st1w { za0h.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x6, x11, #4
    addvl x5, x13, #4
    KAI_ASM_INST(0xe0b40561)  // st1w { za0h.s[x12, #1] }, p1/Z, [x11, x20, LSL #2]
    KAI_ASM_INST(0xe09405a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x20, LSL #2]
    addvl x4, x11, #8
    addvl x3, x13, #8
    KAI_ASM_INST(0xe0a80562)  // st1w { za0h.s[x12, #2] }, p1/Z, [x11, x8, LSL #2]
    KAI_ASM_INST(0xe08805a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x8, LSL #2]
    addvl x2, x11, #12
    addvl x1, x13, #12
    KAI_ASM_INST(0xe0a70563)  // st1w { za0h.s[x12, #3] }, p1/Z, [x11, x7, LSL #2]
    KAI_ASM_INST(0xe08705a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x7, LSL #2]
    addvl x11, x11, #16
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0bf04c4)  // st1w { za1h.s[x12] }, p1/Z, [x6, XZR, LSL #2]
    KAI_ASM_INST(0xe09f04a4)  // ld1w { za1h.s[x12] }, p1/Z, [x5, XZR, LSL #2]
    KAI_ASM_INST(0xe0b404c5)  // st1w { za1h.s[x12, #1] }, p1/Z, [x6, x20, LSL #2]
    KAI_ASM_INST(0xe09404a5)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0xe0a804c6)  // st1w { za1h.s[x12, #2] }, p1/Z, [x6, x8, LSL #2]
    KAI_ASM_INST(0xe08804a6)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x5, x8, LSL #2]
    KAI_ASM_INST(0xe0a704c7)  // st1w { za1h.s[x12, #3] }, p1/Z, [x6, x7, LSL #2]
    KAI_ASM_INST(0xe08704a7)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x5, x7, LSL #2]
    KAI_ASM_INST(0xe0bf0488)  // st1w { za2h.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0468)  // ld1w { za2h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b40489)  // st1w { za2h.s[x12, #1] }, p1/Z, [x4, x20, LSL #2]
    KAI_ASM_INST(0xe0940469)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x3, x20, LSL #2]
    KAI_ASM_INST(0xe0a8048a)  // st1w { za2h.s[x12, #2] }, p1/Z, [x4, x8, LSL #2]
    KAI_ASM_INST(0xe088046a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x3, x8, LSL #2]
    KAI_ASM_INST(0xe0a7048b)  // st1w { za2h.s[x12, #3] }, p1/Z, [x4, x7, LSL #2]
    KAI_ASM_INST(0xe087046b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bf044c)  // st1w { za3h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4044d)  // st1w { za3h.s[x12, #1] }, p1/Z, [x2, x20, LSL #2]
    KAI_ASM_INST(0xe094042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x20, LSL #2]
    KAI_ASM_INST(0xe0a8044e)  // st1w { za3h.s[x12, #2] }, p1/Z, [x2, x8, LSL #2]
    KAI_ASM_INST(0xe088042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x8, LSL #2]
    KAI_ASM_INST(0xe0a7044f)  // st1w { za3h.s[x12, #3] }, p1/Z, [x2, x7, LSL #2]
    KAI_ASM_INST(0xe087042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x7, LSL #2]
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
    KAI_ASM_INST(0xe0bf0560)  // st1w { za0h.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    addvl x3, x11, #4
    addvl x2, x11, #8
    KAI_ASM_INST(0xe0a60561)  // st1w { za0h.s[x12, #1] }, p1/Z, [x11, x6, LSL #2]
    addvl x1, x11, #12
    KAI_ASM_INST(0xe0a50562)  // st1w { za0h.s[x12, #2] }, p1/Z, [x11, x5, LSL #2]
    KAI_ASM_INST(0xe0a40563)  // st1w { za0h.s[x12, #3] }, p1/Z, [x11, x4, LSL #2]
    addvl x11, x11, #16
    KAI_ASM_INST(0xe0bf0464)  // st1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60465)  // st1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0a50466)  // st1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe0a40467)  // st1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe0bf0448)  // st1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe0a60449)  // st1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe0a5044a)  // st1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe0a4044b)  // st1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe0bf042c)  // st1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe0a6042d)  // st1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe0a5042e)  // st1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe0a4042f)  // st1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_14
    b label_19
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x26, x28
    cntw x4, ALL, MUL #2
    ldr x3, [x0, #0x58]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    whilelt p0.h, x27, x25
    cmp x5, x4
    KAI_ASM_INST(0x84c0a453)  // ld1rh { z19.h }, p1/Z, [x2]
    add x6, x6, x27, LSL #1  // C += n
    KAI_ASM_INST(0x84c0a432)  // ld1rh { z18.h }, p1/Z, [x1]
    mov x12, #0
    madd x6, x28, x3, x6  // C += m * ldc
    mov x2, #0
    mov x1, #0x2
    csel x5, x5, x4, LT
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc0020411)  // mova z17.b, p1/M, za0h.b[x12]
    add x2, x2, #0x1
    KAI_ASM_INST(0xc0020430)  // mova z16.b, p1/M, za0h.b[x12, #1]
    fcvt z17.h, p1/m, z17.s
    add x12, x12, #0x4
    fcvt z16.h, p1/m, z16.s
    cmp x12, x4, LSL #1
    csel x12, x12, x1, LT
    cmp x2, x5
    uzp1 z16.h, z17.h, z16.h
    fmin z16.h, p1/M, z16.h, z18.h
    fmax z16.h, p1/M, z16.h, z19.h
    st1h { z16.h }, p0, [x6]
    add x6, x6, x3
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
    KAI_ASM_INST(0xe09f05a0)  // ld1w { za0h.s[x12] }, p1/Z, [x13, XZR, LSL #2]
    addvl x1, x13, #12
    KAI_ASM_INST(0xe09f0464)  // ld1w { za1h.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe09f0448)  // ld1w { za2h.s[x12] }, p1/Z, [x2, XZR, LSL #2]
    KAI_ASM_INST(0xe09f042c)  // ld1w { za3h.s[x12] }, p1/Z, [x1, XZR, LSL #2]
    KAI_ASM_INST(0xe08605a1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x13, x6, LSL #2]
    KAI_ASM_INST(0xe0860465)  // ld1w { za1h.s[x12, #1] }, p1/Z, [x3, x6, LSL #2]
    KAI_ASM_INST(0xe0860449)  // ld1w { za2h.s[x12, #1] }, p1/Z, [x2, x6, LSL #2]
    KAI_ASM_INST(0xe086042d)  // ld1w { za3h.s[x12, #1] }, p1/Z, [x1, x6, LSL #2]
    KAI_ASM_INST(0xe08505a2)  // ld1w { za0h.s[x12, #2] }, p1/Z, [x13, x5, LSL #2]
    KAI_ASM_INST(0xe0850466)  // ld1w { za1h.s[x12, #2] }, p1/Z, [x3, x5, LSL #2]
    KAI_ASM_INST(0xe085044a)  // ld1w { za2h.s[x12, #2] }, p1/Z, [x2, x5, LSL #2]
    KAI_ASM_INST(0xe085042e)  // ld1w { za3h.s[x12, #2] }, p1/Z, [x1, x5, LSL #2]
    KAI_ASM_INST(0xe08405a3)  // ld1w { za0h.s[x12, #3] }, p1/Z, [x13, x4, LSL #2]
    addvl x13, x13, #16
    KAI_ASM_INST(0xe0840467)  // ld1w { za1h.s[x12, #3] }, p1/Z, [x3, x4, LSL #2]
    KAI_ASM_INST(0xe084044b)  // ld1w { za2h.s[x12, #3] }, p1/Z, [x2, x4, LSL #2]
    KAI_ASM_INST(0xe084042f)  // ld1w { za3h.s[x12, #3] }, p1/Z, [x1, x4, LSL #2]
    add x12, x12, #0x4
    cmp x12, x7
    blt label_18
KAI_ASM_LABEL(label_19)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL)

    KAI_ASM_END
