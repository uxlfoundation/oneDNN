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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp32_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL)
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
    mov x6, #0xff800000
    ldr x2, [x0, #0x50]
    mov x5, #0x7f800000
    ptrue p0.b
    ldr w4, [x0, #0xa0]
    cmp x1, XZR
    ldr w3, [x0, #0xa4]
    cset x1, NE
    cmp x2, XZR
    ldr x28, [x0, #0x68]
    orr x9, x9, x1
    cset x1, EQ
    ldr x2, [x0, #0x70]
    cmp x4, x6
    orr x9, x9, x1, LSL #1
    cset x1, NE
    cmp x3, x5
    orr x9, x9, x1, LSL #2
    cset x1, NE
    orr x9, x9, x1, LSL #2
    mov x27, x28
    add x26, x28, x2
    tbz x9, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c784)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c798)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840480)  // mova za0h.s[x12], { z4.s-z7.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840701)  // mova za1h.s[x12], { z24.s-z27.s }
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x25, [x0, #0x10]
    mov x24, #0
    mov x23, #0
    ldr x22, [x0, #0]
    ldr x21, [x0, #0x8]
    ldr x20, [x0, #0x20]
KAI_ASM_LABEL(label_3)  // M loop
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x7, x20
    KAI_ASM_INST(0x25b566f0)  // whilelt pn8.s, x23, x21, VLx4
    tbnz x9, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z18.s, #1.0
    KAI_ASM_INST(0xa117c020)  // ld1w { z0.s, z4.s, z8.s, z12.s }, p8/Z, [x1, x23, LSL #2]
    KAI_ASM_INST(0x80800240)  // fmopa za0.s, p0/M, p0/M, z18.s, z0.s
    KAI_ASM_INST(0x80840241)  // fmopa za1.s, p0/M, p0/M, z18.s, z4.s
    KAI_ASM_INST(0x80880242)  // fmopa za2.s, p0/M, p0/M, z18.s, z8.s
    KAI_ASM_INST(0x808c0243)  // fmopa za3.s, p0/M, p0/M, z18.s, z12.s
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
    KAI_ASM_INST(0xa140c4f3)  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa140c510)  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa041c508)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c50c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa143c512)  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x80900260)  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s
    subs x2, x2, #0x1
    KAI_ASM_INST(0x80940261)  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s
    KAI_ASM_INST(0x80980262)  // fmopa za2.s, p0/M, p0/M, z19.s, z24.s
    KAI_ASM_INST(0x809c0263)  // fmopa za3.s, p0/M, p0/M, z19.s, z28.s
    KAI_ASM_INST(0xa140c510)  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x8]
    KAI_ASM_INST(0x808802e0)  // fmopa za0.s, p0/M, p0/M, z23.s, z8.s
    KAI_ASM_INST(0x808902e1)  // fmopa za1.s, p0/M, p0/M, z23.s, z9.s
    KAI_ASM_INST(0x808a02e2)  // fmopa za2.s, p0/M, p0/M, z23.s, z10.s
    KAI_ASM_INST(0x808b02e3)  // fmopa za3.s, p0/M, p0/M, z23.s, z11.s
    KAI_ASM_INST(0xa041c508)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0x808c0360)  // fmopa za0.s, p0/M, p0/M, z27.s, z12.s
    KAI_ASM_INST(0x808d0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z13.s
    KAI_ASM_INST(0x808e0362)  // fmopa za2.s, p0/M, p0/M, z27.s, z14.s
    KAI_ASM_INST(0x808f0363)  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s
    KAI_ASM_INST(0xa042c50c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0x809203e0)  // fmopa za0.s, p0/M, p0/M, z31.s, z18.s
    KAI_ASM_INST(0x809603e1)  // fmopa za1.s, p0/M, p0/M, z31.s, z22.s
    KAI_ASM_INST(0x809a03e2)  // fmopa za2.s, p0/M, p0/M, z31.s, z26.s
    KAI_ASM_INST(0x809e03e3)  // fmopa za3.s, p0/M, p0/M, z31.s, z30.s
    KAI_ASM_INST(0xa140c4f3)  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa143c512)  // ld1w { z18.s, z22.s, z26.s, z30.s }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x80900260)  // fmopa za0.s, p0/M, p0/M, z19.s, z16.s
    KAI_ASM_INST(0x80940261)  // fmopa za1.s, p0/M, p0/M, z19.s, z20.s
    KAI_ASM_INST(0x80980262)  // fmopa za2.s, p0/M, p0/M, z19.s, z24.s
    KAI_ASM_INST(0x809c0263)  // fmopa za3.s, p0/M, p0/M, z19.s, z28.s
    KAI_ASM_INST(0x808802e0)  // fmopa za0.s, p0/M, p0/M, z23.s, z8.s
    KAI_ASM_INST(0x808902e1)  // fmopa za1.s, p0/M, p0/M, z23.s, z9.s
    KAI_ASM_INST(0x808a02e2)  // fmopa za2.s, p0/M, p0/M, z23.s, z10.s
    KAI_ASM_INST(0x808b02e3)  // fmopa za3.s, p0/M, p0/M, z23.s, z11.s
    KAI_ASM_INST(0x808c0360)  // fmopa za0.s, p0/M, p0/M, z27.s, z12.s
    KAI_ASM_INST(0x808d0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z13.s
    KAI_ASM_INST(0x808e0362)  // fmopa za2.s, p0/M, p0/M, z27.s, z14.s
    KAI_ASM_INST(0x808f0363)  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s
    KAI_ASM_INST(0x809203e0)  // fmopa za0.s, p0/M, p0/M, z31.s, z18.s
    KAI_ASM_INST(0x809603e1)  // fmopa za1.s, p0/M, p0/M, z31.s, z22.s
    KAI_ASM_INST(0x809a03e2)  // fmopa za2.s, p0/M, p0/M, z31.s, z26.s
    KAI_ASM_INST(0x809e03e3)  // fmopa za3.s, p0/M, p0/M, z31.s, z30.s
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1w { z27.s }, p0/Z, [x7]
    subs x1, x1, #0x1
    addvl x7, x7, #1
    KAI_ASM_INST(0xa140c503)  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn9.b/Z, [x8]
    addvl x8, x8, #4
    KAI_ASM_INST(0x80830360)  // fmopa za0.s, p0/M, p0/M, z27.s, z3.s
    KAI_ASM_INST(0x80870361)  // fmopa za1.s, p0/M, p0/M, z27.s, z7.s
    KAI_ASM_INST(0x808b0362)  // fmopa za2.s, p0/M, p0/M, z27.s, z11.s
    KAI_ASM_INST(0x808f0363)  // fmopa za3.s, p0/M, p0/M, z27.s, z15.s
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x9, #1, label_15
    tbz x9, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c79c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c788)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840780)  // mova za0h.s[x12], { z28.s-z31.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840501)  // mova za1h.s[x12], { z8.s-z11.s }
    KAI_ASM_INST(0xa060c764)  // st1w { z4.s-z7.s }, pn9.b, [x27]
    KAI_ASM_INST(0xc0840682)  // mova za2h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xa061c770)  // st1w { z16.s-z19.s }, pn9.b, [x27, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840583)  // mova za3h.s[x12], { z12.s-z15.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c760)  // st1w { z0.s-z3.s }, pn9.b, [x27, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c778)  // st1w { z24.s-z27.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_12
    b label_25
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860460)  // mova { z0.s-z3.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c76c)  // st1w { z12.s-z15.s }, pn9.b, [x27]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c764)  // st1w { z4.s-z7.s }, pn9.b, [x27, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c768)  // st1w { z8.s-z11.s }, pn9.b, [x27, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c760)  // st1w { z0.s-z3.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_14
    b label_25
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x22, x24
    ldr x4, [x0, #0x58]
    add x6, x6, x23, LSL #2  // C += n
    madd x6, x24, x4, x6  // C += m * ldc
    tbnz x9, #2, label_19
    cntw x1
    mov x12, #0
    cmp x5, x1
    csel x3, x5, x1, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Skip activation: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860434)  // mova { z20.s-z23.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xa160c0d0)  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x6]
    add x6, x6, x4
    add x12, x12, #0x4
    KAI_ASM_INST(0xa160c0d1)  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x6]
    add x6, x6, x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa160c0d2)  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0d3)  // st1w { z19.s, z23.s, z27.s, z31.s }, p8, [x6]
    add x6, x6, x4
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Skip activation: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa160c0c0)  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x6]
    add x6, x6, x4
    beq label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa160c0c1)  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x6]
    add x6, x6, x4
    beq label_18
    KAI_ASM_INST(0xa160c0c2)  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x6]
    add x6, x6, x4
KAI_ASM_LABEL(label_18)  // Store to output array: Skip activation: Accumulator row 0 oddments: End
    subs x5, x5, x3
    beq label_19
    b label_23
KAI_ASM_LABEL(label_19)  // Store to output array: Skip activation: End
    cntw x3
    add x1, x0, #0xa0
    cmp x5, x3
    add x2, x0, #0xa4
    KAI_ASM_INST(0x8540c021)  // ld1rw { z1.s }, p0/Z, [x1]
    csel x1, x5, x3, LT
    KAI_ASM_INST(0x8540c040)  // ld1rw { z0.s }, p0/Z, [x2]
    mov x12, #0
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_21
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860434)  // mova { z20.s-z23.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1a0c830)  // fclamp { z16.s-z19.s }, z1.s, z0.s
    KAI_ASM_INST(0xc1a0c834)  // fclamp { z20.s-z23.s }, z1.s, z0.s
    KAI_ASM_INST(0xc1a0c838)  // fclamp { z24.s-z27.s }, z1.s, z0.s
    KAI_ASM_INST(0xc1a0c83c)  // fclamp { z28.s-z31.s }, z1.s, z0.s
    add x12, x12, #0x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa160c0d0)  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0d1)  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0d2)  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0d3)  // st1w { z19.s, z23.s, z27.s, z31.s }, p8, [x6]
    add x6, x6, x4
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860434)  // mova { z20.s-z23.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1a0c830)  // fclamp { z16.s-z19.s }, z1.s, z0.s
    KAI_ASM_INST(0xc1a0c834)  // fclamp { z20.s-z23.s }, z1.s, z0.s
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc1a0c838)  // fclamp { z24.s-z27.s }, z1.s, z0.s
    KAI_ASM_INST(0xc1a0c83c)  // fclamp { z28.s-z31.s }, z1.s, z0.s
    KAI_ASM_INST(0xa160c0d0)  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x6]
    add x6, x6, x4
    beq label_22
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa160c0d1)  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x6]
    add x6, x6, x4
    beq label_22
    KAI_ASM_INST(0xa160c0d2)  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x6]
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 0 oddments: End
KAI_ASM_LABEL(label_23)  // Store to output array: End
    tbz x9, #0, label_25
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_24)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c79c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c784)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840782)  // mova za2h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840483)  // mova za3h.s[x12], { z4.s-z7.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_24
KAI_ASM_LABEL(label_25)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL)

    KAI_ASM_END
