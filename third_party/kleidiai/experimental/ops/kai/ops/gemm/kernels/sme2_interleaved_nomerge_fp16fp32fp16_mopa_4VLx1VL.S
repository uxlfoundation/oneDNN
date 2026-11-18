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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL)
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
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ptrue p1.b
    ldr x3, [x0, #0x50]
    ldr x9, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x10, x10, x1
    cset x1, EQ
    orr x10, x10, x1, LSL #1
    mov x28, x9
    add x27, x9, x2
    tbz x10, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c534)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xa041c52c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c530)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c520)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840602)  // mova za2h.s[x12], { z16.s-z19.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
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
    whilelt p8.s, x24, x22
    tbnz x10, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z17.h, #0.0
    whilelt p0.h, x24, x22
    fmov z30.h, #1.0
    ld1h { z28.h }, p0/Z, [x1, x24, LSL #1]
    zip1 z15.h, z28.h, z17.h
    KAI_ASM_INST(0x81af27c0)  // fmopa za0.s, p1/M, p1/M, z30.h, z15.h
    KAI_ASM_INST(0x81af27c1)  // fmopa za1.s, p1/M, p1/M, z30.h, z15.h
    KAI_ASM_INST(0x81af27c2)  // fmopa za2.s, p1/M, p1/M, z30.h, z15.h
    KAI_ASM_INST(0x81af27c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z15.h
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
    KAI_ASM_INST(0xa040a508)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa041a50c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a518)  // ld1h { z24.h-z27.h }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a514)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa040a69c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x20]
    addvl x20, x20, #4
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81bc2500)  // fmopa za0.s, p1/M, p1/M, z8.h, z28.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81bc2521)  // fmopa za1.s, p1/M, p1/M, z9.h, z28.h
    KAI_ASM_INST(0x81bc2542)  // fmopa za2.s, p1/M, p1/M, z10.h, z28.h
    KAI_ASM_INST(0x81bc2563)  // fmopa za3.s, p1/M, p1/M, z11.h, z28.h
    KAI_ASM_INST(0xa040a508)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0x81bd2580)  // fmopa za0.s, p1/M, p1/M, z12.h, z29.h
    KAI_ASM_INST(0x81bd25a1)  // fmopa za1.s, p1/M, p1/M, z13.h, z29.h
    KAI_ASM_INST(0x81bd25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z29.h
    KAI_ASM_INST(0x81bd25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z29.h
    KAI_ASM_INST(0xa041a50c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0x81be2700)  // fmopa za0.s, p1/M, p1/M, z24.h, z30.h
    KAI_ASM_INST(0x81be2721)  // fmopa za1.s, p1/M, p1/M, z25.h, z30.h
    KAI_ASM_INST(0x81be2742)  // fmopa za2.s, p1/M, p1/M, z26.h, z30.h
    KAI_ASM_INST(0x81be2763)  // fmopa za3.s, p1/M, p1/M, z27.h, z30.h
    KAI_ASM_INST(0xa042a518)  // ld1h { z24.h-z27.h }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0x81bf2680)  // fmopa za0.s, p1/M, p1/M, z20.h, z31.h
    KAI_ASM_INST(0x81bf26a1)  // fmopa za1.s, p1/M, p1/M, z21.h, z31.h
    KAI_ASM_INST(0x81bf26c2)  // fmopa za2.s, p1/M, p1/M, z22.h, z31.h
    KAI_ASM_INST(0x81bf26e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z31.h
    KAI_ASM_INST(0xa043a514)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    KAI_ASM_INST(0xa040a69c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x20]
    addvl x20, x20, #4
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81bc2500)  // fmopa za0.s, p1/M, p1/M, z8.h, z28.h
    KAI_ASM_INST(0x81bc2521)  // fmopa za1.s, p1/M, p1/M, z9.h, z28.h
    KAI_ASM_INST(0x81bc2542)  // fmopa za2.s, p1/M, p1/M, z10.h, z28.h
    KAI_ASM_INST(0x81bc2563)  // fmopa za3.s, p1/M, p1/M, z11.h, z28.h
    KAI_ASM_INST(0x81bd2580)  // fmopa za0.s, p1/M, p1/M, z12.h, z29.h
    KAI_ASM_INST(0x81bd25a1)  // fmopa za1.s, p1/M, p1/M, z13.h, z29.h
    KAI_ASM_INST(0x81bd25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z29.h
    KAI_ASM_INST(0x81bd25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z29.h
    KAI_ASM_INST(0x81be2700)  // fmopa za0.s, p1/M, p1/M, z24.h, z30.h
    KAI_ASM_INST(0x81be2721)  // fmopa za1.s, p1/M, p1/M, z25.h, z30.h
    KAI_ASM_INST(0x81be2742)  // fmopa za2.s, p1/M, p1/M, z26.h, z30.h
    KAI_ASM_INST(0x81be2763)  // fmopa za3.s, p1/M, p1/M, z27.h, z30.h
    KAI_ASM_INST(0x81bf2680)  // fmopa za0.s, p1/M, p1/M, z20.h, z31.h
    KAI_ASM_INST(0x81bf26a1)  // fmopa za1.s, p1/M, p1/M, z21.h, z31.h
    KAI_ASM_INST(0x81bf26c2)  // fmopa za2.s, p1/M, p1/M, z22.h, z31.h
    KAI_ASM_INST(0x81bf26e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z31.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa040a508)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x8]
    subs x1, x1, #0x1
    addvl x8, x8, #4
    ld1h { z6.h }, p1/Z, [x20]
    addvl x20, x20, #1
    KAI_ASM_INST(0x81a62500)  // fmopa za0.s, p1/M, p1/M, z8.h, z6.h
    KAI_ASM_INST(0x81a62521)  // fmopa za1.s, p1/M, p1/M, z9.h, z6.h
    KAI_ASM_INST(0x81a62542)  // fmopa za2.s, p1/M, p1/M, z10.h, z6.h
    KAI_ASM_INST(0x81a62563)  // fmopa za3.s, p1/M, p1/M, z11.h, z6.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x10, #1, label_15
    tbz x10, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c530)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c534)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c524)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c520)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840600)  // mova za0h.s[x12], { z16.s-z19.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840681)  // mova za1h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xa060c79c)  // st1w { z28.s-z31.s }, pn9.b, [x28]
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xa061c78c)  // st1w { z12.s-z15.s }, pn9.b, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c798)  // st1w { z24.s-z27.s }, pn9.b, [x28, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c788)  // st1w { z8.s-z11.s }, pn9.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_12
    b label_30
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c790)  // st1w { z16.s-z19.s }, pn9.b, [x28]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c780)  // st1w { z0.s-z3.s }, pn9.b, [x28, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c798)  // st1w { z24.s-z27.s }, pn9.b, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c78c)  // st1w { z12.s-z15.s }, pn9.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_14
    b label_30
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x23, x25
    cntw x5
    ldr x4, [x0, #0x58]
    whilelt p0.s, x24, x22
    cmp x6, x5
    csel x3, x6, x5, LT
    add x2, x0, #0xa0
    add x7, x7, x24, LSL #1  // C += n
    add x1, x0, #0xa4
    KAI_ASM_INST(0x84c0a455)  // ld1rh { z21.h }, p1/Z, [x2]
    lsr x2, x3, #0x2
    madd x7, x25, x4, x7  // C += m * ldc
    KAI_ASM_INST(0x84c0a434)  // ld1rh { z20.h }, p1/Z, [x1]
    mov x12, #0
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    add x12, x12, #0x4
    fcvt z28.h, p1/m, z28.s
    fcvt z29.h, p1/m, z29.s
    cmp x12, x2, LSL #2
    fcvt z30.h, p1/m, z30.s
    fcvt z31.h, p1/m, z31.s
    KAI_ASM_INST(0xc174cabc)  // fclamp { z28.h-z31.h }, z21.h, z20.h
    st1h { z28.s }, p0, [x7]
    add x7, x7, x4
    st1h { z29.s }, p0, [x7]
    add x7, x7, x4
    st1h { z30.s }, p0, [x7]
    add x7, x7, x4
    st1h { z31.s }, p0, [x7]
    add x7, x7, x4
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_18
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    subs x1, x1, #0x1
    fcvt z16.h, p1/m, z16.s
    fcvt z17.h, p1/m, z17.s
    fcvt z18.h, p1/m, z18.s
    fcvt z19.h, p1/m, z19.s
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    st1h { z16.s }, p0, [x7]
    add x7, x7, x4
    beq label_18
    subs x1, x1, #0x1
    st1h { z17.s }, p0, [x7]
    add x7, x7, x4
    beq label_18
    st1h { z18.s }, p0, [x7]
    add x7, x7, x4
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x5
    mov x12, #0
    csel x3, x6, x5, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_20
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    add x12, x12, #0x4
    fcvt z0.h, p1/m, z0.s
    fcvt z1.h, p1/m, z1.s
    cmp x12, x2, LSL #2
    fcvt z2.h, p1/m, z2.s
    fcvt z3.h, p1/m, z3.s
    KAI_ASM_INST(0xc174caa0)  // fclamp { z0.h-z3.h }, z21.h, z20.h
    st1h { z0.s }, p0, [x7]
    add x7, x7, x4
    st1h { z1.s }, p0, [x7]
    add x7, x7, x4
    st1h { z2.s }, p0, [x7]
    add x7, x7, x4
    st1h { z3.s }, p0, [x7]
    add x7, x7, x4
    blt label_19
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_21
    KAI_ASM_INST(0xc0860438)  // mova { z24.s-z27.s }, za1h.s[x12]
    subs x1, x1, #0x1
    fcvt z24.h, p1/m, z24.s
    fcvt z25.h, p1/m, z25.s
    fcvt z26.h, p1/m, z26.s
    fcvt z27.h, p1/m, z27.s
    KAI_ASM_INST(0xc174cab8)  // fclamp { z24.h-z27.h }, z21.h, z20.h
    st1h { z24.s }, p0, [x7]
    add x7, x7, x4
    beq label_21
    subs x1, x1, #0x1
    st1h { z25.s }, p0, [x7]
    add x7, x7, x4
    beq label_21
    st1h { z26.s }, p0, [x7]
    add x7, x7, x4
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x5
    mov x12, #0
    csel x3, x6, x5, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_23
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 2 loop
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    fcvt z16.h, p1/m, z16.s
    fcvt z17.h, p1/m, z17.s
    cmp x12, x2, LSL #2
    fcvt z18.h, p1/m, z18.s
    fcvt z19.h, p1/m, z19.s
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    st1h { z16.s }, p0, [x7]
    add x7, x7, x4
    st1h { z17.s }, p0, [x7]
    add x7, x7, x4
    st1h { z18.s }, p0, [x7]
    add x7, x7, x4
    st1h { z19.s }, p0, [x7]
    add x7, x7, x4
    blt label_22
KAI_ASM_LABEL(label_23)  // Store to output array: Accumulator row 2 oddments
    cbz x1, label_24
    KAI_ASM_INST(0xc086045c)  // mova { z28.s-z31.s }, za2h.s[x12]
    subs x1, x1, #0x1
    fcvt z28.h, p1/m, z28.s
    fcvt z29.h, p1/m, z29.s
    fcvt z30.h, p1/m, z30.s
    fcvt z31.h, p1/m, z31.s
    KAI_ASM_INST(0xc174cabc)  // fclamp { z28.h-z31.h }, z21.h, z20.h
    st1h { z28.s }, p0, [x7]
    add x7, x7, x4
    beq label_24
    subs x1, x1, #0x1
    st1h { z29.s }, p0, [x7]
    add x7, x7, x4
    beq label_24
    st1h { z30.s }, p0, [x7]
    add x7, x7, x4
KAI_ASM_LABEL(label_24)  // Store to output array: Accumulator row 2 oddments: End
    subs x6, x6, x3
    beq label_28
    cmp x6, x5
    mov x12, #0
    csel x1, x6, x5, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_26
KAI_ASM_LABEL(label_25)  // Store to output array: Accumulator row 3 loop
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z28.h, p1/m, z28.s
    fcvt z29.h, p1/m, z29.s
    cmp x12, x2, LSL #2
    fcvt z30.h, p1/m, z30.s
    fcvt z31.h, p1/m, z31.s
    KAI_ASM_INST(0xc174cabc)  // fclamp { z28.h-z31.h }, z21.h, z20.h
    st1h { z28.s }, p0, [x7]
    add x7, x7, x4
    st1h { z29.s }, p0, [x7]
    add x7, x7, x4
    st1h { z30.s }, p0, [x7]
    add x7, x7, x4
    st1h { z31.s }, p0, [x7]
    add x7, x7, x4
    blt label_25
KAI_ASM_LABEL(label_26)  // Store to output array: Accumulator row 3 oddments
    cbz x1, label_27
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    subs x1, x1, #0x1
    fcvt z28.h, p1/m, z28.s
    fcvt z29.h, p1/m, z29.s
    fcvt z30.h, p1/m, z30.s
    fcvt z31.h, p1/m, z31.s
    KAI_ASM_INST(0xc174cabc)  // fclamp { z28.h-z31.h }, z21.h, z20.h
    st1h { z28.s }, p0, [x7]
    add x7, x7, x4
    beq label_27
    subs x1, x1, #0x1
    st1h { z29.s }, p0, [x7]
    add x7, x7, x4
    beq label_27
    st1h { z30.s }, p0, [x7]
KAI_ASM_LABEL(label_27)  // Store to output array: Accumulator row 3 oddments: End
KAI_ASM_LABEL(label_28)  // Store to output array: End
    tbz x10, #0, label_30
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_29)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c538)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xa041c52c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c520)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c528)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840402)  // mova za2h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_29
KAI_ASM_LABEL(label_30)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL)

    KAI_ASM_END
