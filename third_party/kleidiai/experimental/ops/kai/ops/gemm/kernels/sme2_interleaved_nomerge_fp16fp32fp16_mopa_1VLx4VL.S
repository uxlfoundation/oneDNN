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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)
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
    mov x9, #0
    ldr x1, [x0, #0x60]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ptrue p0.b
    ldr x3, [x0, #0x50]
    ldr x28, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x9, x9, x1
    cset x1, EQ
    orr x9, x9, x1, LSL #1
    mov x27, x28
    add x26, x28, x2
    tbz x9, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c798)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c784)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c790)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840401)  // mova za1h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840603)  // mova za3h.s[x12], { z16.s-z19.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x25, [x0, #0x10]
    mov x24, #0
    mov x23, #0
    ldr x22, [x0, #0]
    ldr x21, [x0, #0x8]
    add x25, x25, #0x1
    ldr x20, [x0, #0x20]
    lsr x25, x25, #0x1
KAI_ASM_LABEL(label_3)  // M loop
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x7, x20
    tbnz x9, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z5.h, #0.0
    KAI_ASM_INST(0x257546f0)  // whilelt pn8.h, x23, x21, VLx2
    fmov z24.h, #1.0
    KAI_ASM_INST(0xa117203c)  // ldnt1h { z20.h, z28.h }, p8/Z, [x1, x23, LSL #1]
    zip1 z27.h, z20.h, z5.h
    zip2 z10.h, z20.h, z5.h
    zip1 z20.h, z28.h, z5.h
    zip2 z18.h, z28.h, z5.h
    KAI_ASM_INST(0x81bb0300)  // fmopa za0.s, p0/M, p0/M, z24.h, z27.h
    KAI_ASM_INST(0x81aa0301)  // fmopa za1.s, p0/M, p0/M, z24.h, z10.h
    KAI_ASM_INST(0x81b40302)  // fmopa za2.s, p0/M, p0/M, z24.h, z20.h
    KAI_ASM_INST(0x81b20303)  // fmopa za3.s, p0/M, p0/M, z24.h, z18.h
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x23
    mov x2, x24
    incw x1, ALL, MUL #4
    incw x2
    cmp x1, x21
    mov x1, x9
    csel x2, x24, x2, LT
    bfm x9, XZR, #0, #0  // bfc x9, #0, #0x1
    csel x28, x28, x26, LT
    cmp x2, x22
    csel x9, x1, x9, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x25, #0x2
    and x1, x25, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa140a4e1)  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa140a502)  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa041a51c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa042a510)  // ld1h { z16.h-z19.h }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa043a514)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81a20020)  // fmopa za0.s, p0/M, p0/M, z1.h, z2.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81a60021)  // fmopa za1.s, p0/M, p0/M, z1.h, z6.h
    KAI_ASM_INST(0x81aa0022)  // fmopa za2.s, p0/M, p0/M, z1.h, z10.h
    KAI_ASM_INST(0x81ae0023)  // fmopa za3.s, p0/M, p0/M, z1.h, z14.h
    KAI_ASM_INST(0xa140a502)  // ld1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0x81bc00a0)  // fmopa za0.s, p0/M, p0/M, z5.h, z28.h
    KAI_ASM_INST(0x81bd00a1)  // fmopa za1.s, p0/M, p0/M, z5.h, z29.h
    KAI_ASM_INST(0x81be00a2)  // fmopa za2.s, p0/M, p0/M, z5.h, z30.h
    KAI_ASM_INST(0x81bf00a3)  // fmopa za3.s, p0/M, p0/M, z5.h, z31.h
    KAI_ASM_INST(0xa041a51c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0x81b00120)  // fmopa za0.s, p0/M, p0/M, z9.h, z16.h
    KAI_ASM_INST(0x81b10121)  // fmopa za1.s, p0/M, p0/M, z9.h, z17.h
    KAI_ASM_INST(0x81b20122)  // fmopa za2.s, p0/M, p0/M, z9.h, z18.h
    KAI_ASM_INST(0x81b30123)  // fmopa za3.s, p0/M, p0/M, z9.h, z19.h
    KAI_ASM_INST(0xa042a510)  // ld1h { z16.h-z19.h }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0x81b401a0)  // fmopa za0.s, p0/M, p0/M, z13.h, z20.h
    KAI_ASM_INST(0x81b501a1)  // fmopa za1.s, p0/M, p0/M, z13.h, z21.h
    KAI_ASM_INST(0x81b601a2)  // fmopa za2.s, p0/M, p0/M, z13.h, z22.h
    KAI_ASM_INST(0x81b701a3)  // fmopa za3.s, p0/M, p0/M, z13.h, z23.h
    KAI_ASM_INST(0xa140a4e1)  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa043a514)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81a20020)  // fmopa za0.s, p0/M, p0/M, z1.h, z2.h
    KAI_ASM_INST(0x81a60021)  // fmopa za1.s, p0/M, p0/M, z1.h, z6.h
    KAI_ASM_INST(0x81aa0022)  // fmopa za2.s, p0/M, p0/M, z1.h, z10.h
    KAI_ASM_INST(0x81ae0023)  // fmopa za3.s, p0/M, p0/M, z1.h, z14.h
    KAI_ASM_INST(0x81bc00a0)  // fmopa za0.s, p0/M, p0/M, z5.h, z28.h
    KAI_ASM_INST(0x81bd00a1)  // fmopa za1.s, p0/M, p0/M, z5.h, z29.h
    KAI_ASM_INST(0x81be00a2)  // fmopa za2.s, p0/M, p0/M, z5.h, z30.h
    KAI_ASM_INST(0x81bf00a3)  // fmopa za3.s, p0/M, p0/M, z5.h, z31.h
    KAI_ASM_INST(0x81b00120)  // fmopa za0.s, p0/M, p0/M, z9.h, z16.h
    KAI_ASM_INST(0x81b10121)  // fmopa za1.s, p0/M, p0/M, z9.h, z17.h
    KAI_ASM_INST(0x81b20122)  // fmopa za2.s, p0/M, p0/M, z9.h, z18.h
    KAI_ASM_INST(0x81b30123)  // fmopa za3.s, p0/M, p0/M, z9.h, z19.h
    KAI_ASM_INST(0x81b401a0)  // fmopa za0.s, p0/M, p0/M, z13.h, z20.h
    KAI_ASM_INST(0x81b501a1)  // fmopa za1.s, p0/M, p0/M, z13.h, z21.h
    KAI_ASM_INST(0x81b601a2)  // fmopa za2.s, p0/M, p0/M, z13.h, z22.h
    KAI_ASM_INST(0x81b701a3)  // fmopa za3.s, p0/M, p0/M, z13.h, z23.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1h { z23.h }, p0/Z, [x7]
    subs x1, x1, #0x1
    addvl x7, x7, #1
    KAI_ASM_INST(0xa040a50c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x8]
    addvl x8, x8, #4
    KAI_ASM_INST(0x81ac02e0)  // fmopa za0.s, p0/M, p0/M, z23.h, z12.h
    KAI_ASM_INST(0x81ad02e1)  // fmopa za1.s, p0/M, p0/M, z23.h, z13.h
    KAI_ASM_INST(0x81ae02e2)  // fmopa za2.s, p0/M, p0/M, z23.h, z14.h
    KAI_ASM_INST(0x81af02e3)  // fmopa za3.s, p0/M, p0/M, z23.h, z15.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x9, #1, label_15
    tbz x9, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c790)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc0860414)  // mova { z20.s-z23.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860438)  // mova { z24.s-z27.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c788)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840600)  // mova za0h.s[x12], { z16.s-z19.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840401)  // mova za1h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xa060c774)  // st1w { z20.s-z23.s }, pn9.b, [x27]
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xa061c778)  // st1w { z24.s-z27.s }, pn9.b, [x27, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c764)  // st1w { z4.s-z7.s }, pn9.b, [x27, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c77c)  // st1w { z28.s-z31.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_12
    b label_19
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c768)  // st1w { z8.s-z11.s }, pn9.b, [x27]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c764)  // st1w { z4.s-z7.s }, pn9.b, [x27, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c76c)  // st1w { z12.s-z15.s }, pn9.b, [x27, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c770)  // st1w { z16.s-z19.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_14
    b label_19
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x22, x24
    cntw x4
    ldr x3, [x0, #0x58]
    add x2, x0, #0xa0
    add x1, x0, #0xa4
    KAI_ASM_INST(0x257546f0)  // whilelt pn8.h, x23, x21, VLx2
    cmp x5, x4
    KAI_ASM_INST(0x84c0a055)  // ld1rh { z21.h }, p0/Z, [x2]
    add x6, x6, x23, LSL #1  // C += n
    KAI_ASM_INST(0x84c0a034)  // ld1rh { z20.h }, p0/Z, [x1]
    mov x12, #0
    madd x6, x24, x3, x6  // C += m * ldc
    csel x5, x5, x4, LT
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc120e20e)  // fcvt z14.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e24f)  // fcvt z15.h, { z18.s-z19.s }
    cmp x12, x5, LSL #2
    KAI_ASM_INST(0xc174c2ae)  // fclamp { z14.h-z15.h }, z21.h, z20.h
    KAI_ASM_INST(0xa06020ce)  // st1h { z14.h-z15.h }, p8, [x6]
    add x6, x6, x3
    blt label_16
    tbz x9, #0, label_19
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_18)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c784)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c79c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840481)  // mova za1h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840782)  // mova za2h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840583)  // mova za3h.s[x12], { z12.s-z15.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_18
KAI_ASM_LABEL(label_19)  // End block
    incw x23, ALL, MUL #4
    cmp x23, x21
    blt label_4
    ldr x1, [x0, #0x70]
    incw x24
    mov x27, x26
    cmp x24, x22
    mov x23, #0
    mov x20, x7
    add x26, x27, x1
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL)

    KAI_ASM_END
