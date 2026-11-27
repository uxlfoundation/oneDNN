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

    KAI_ASM_CODE(sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL)
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
    ptrue p0.b
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
    KAI_ASM_INST(0xa041c524)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c52c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c520)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840481)  // mova za1h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
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
    add x26, x26, #0x3
    ldr x21, [x0, #0x20]
    lsr x26, x26, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x8, x21
    KAI_ASM_INST(0x25b64710)  // whilelt pn8.s, x24, x22, VLx2
    tbnz x10, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    KAI_ASM_INST(0xa1184036)  // ld1w { z22.s, z30.s }, p8/Z, [x1, x24, LSL #2]
    KAI_ASM_INST(0xc09002c0)  // addha za0.s, p0/M, p0/M, z22.s
    KAI_ASM_INST(0xc09003c1)  // addha za1.s, p0/M, p0/M, z30.s
    KAI_ASM_INST(0xc09002c2)  // addha za2.s, p0/M, p0/M, z22.s
    KAI_ASM_INST(0xc09003c3)  // addha za3.s, p0/M, p0/M, z30.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x24
    mov x2, x25
    incw x1, ALL, MUL #2
    incw x2, ALL, MUL #2
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
    KAI_ASM_INST(0xa0408518)  // ld1b { z24.b-z27.b }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa0418514)  // ld1b { z20.b-z23.b }, pn9.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    KAI_ASM_INST(0xa0408688)  // ld1b { z8.b-z11.b }, pn9.b/Z, [x20]
    KAI_ASM_INST(0xa0418690)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x20, #0x4, MUL VL]
    addvl x20, x20, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa0880300)  // smopa za0.s, p0/M, p0/M, z24.b, z8.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa0890301)  // smopa za1.s, p0/M, p0/M, z24.b, z9.b
    KAI_ASM_INST(0xa0880322)  // smopa za2.s, p0/M, p0/M, z25.b, z8.b
    KAI_ASM_INST(0xa0890323)  // smopa za3.s, p0/M, p0/M, z25.b, z9.b
    KAI_ASM_INST(0xa08a0340)  // smopa za0.s, p0/M, p0/M, z26.b, z10.b
    KAI_ASM_INST(0xa08b0341)  // smopa za1.s, p0/M, p0/M, z26.b, z11.b
    KAI_ASM_INST(0xa08a0362)  // smopa za2.s, p0/M, p0/M, z27.b, z10.b
    KAI_ASM_INST(0xa08b0363)  // smopa za3.s, p0/M, p0/M, z27.b, z11.b
    KAI_ASM_INST(0xa0408518)  // ld1b { z24.b-z27.b }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa0900280)  // smopa za0.s, p0/M, p0/M, z20.b, z16.b
    KAI_ASM_INST(0xa0408688)  // ld1b { z8.b-z11.b }, pn9.b/Z, [x20]
    KAI_ASM_INST(0xa0910281)  // smopa za1.s, p0/M, p0/M, z20.b, z17.b
    KAI_ASM_INST(0xa09002a2)  // smopa za2.s, p0/M, p0/M, z21.b, z16.b
    KAI_ASM_INST(0xa09102a3)  // smopa za3.s, p0/M, p0/M, z21.b, z17.b
    KAI_ASM_INST(0xa09202c0)  // smopa za0.s, p0/M, p0/M, z22.b, z18.b
    KAI_ASM_INST(0xa09302c1)  // smopa za1.s, p0/M, p0/M, z22.b, z19.b
    KAI_ASM_INST(0xa09202e2)  // smopa za2.s, p0/M, p0/M, z23.b, z18.b
    KAI_ASM_INST(0xa09302e3)  // smopa za3.s, p0/M, p0/M, z23.b, z19.b
    KAI_ASM_INST(0xa0418514)  // ld1b { z20.b-z23.b }, pn9.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    KAI_ASM_INST(0xa0418690)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x20, #0x4, MUL VL]
    addvl x20, x20, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa0880300)  // smopa za0.s, p0/M, p0/M, z24.b, z8.b
    KAI_ASM_INST(0xa0890301)  // smopa za1.s, p0/M, p0/M, z24.b, z9.b
    KAI_ASM_INST(0xa0880322)  // smopa za2.s, p0/M, p0/M, z25.b, z8.b
    KAI_ASM_INST(0xa0890323)  // smopa za3.s, p0/M, p0/M, z25.b, z9.b
    KAI_ASM_INST(0xa08a0340)  // smopa za0.s, p0/M, p0/M, z26.b, z10.b
    KAI_ASM_INST(0xa08b0341)  // smopa za1.s, p0/M, p0/M, z26.b, z11.b
    KAI_ASM_INST(0xa08a0362)  // smopa za2.s, p0/M, p0/M, z27.b, z10.b
    KAI_ASM_INST(0xa08b0363)  // smopa za3.s, p0/M, p0/M, z27.b, z11.b
    KAI_ASM_INST(0xa0900280)  // smopa za0.s, p0/M, p0/M, z20.b, z16.b
    KAI_ASM_INST(0xa0910281)  // smopa za1.s, p0/M, p0/M, z20.b, z17.b
    KAI_ASM_INST(0xa09002a2)  // smopa za2.s, p0/M, p0/M, z21.b, z16.b
    KAI_ASM_INST(0xa09102a3)  // smopa za3.s, p0/M, p0/M, z21.b, z17.b
    KAI_ASM_INST(0xa09202c0)  // smopa za0.s, p0/M, p0/M, z22.b, z18.b
    KAI_ASM_INST(0xa09302c1)  // smopa za1.s, p0/M, p0/M, z22.b, z19.b
    KAI_ASM_INST(0xa09202e2)  // smopa za2.s, p0/M, p0/M, z23.b, z18.b
    KAI_ASM_INST(0xa09302e3)  // smopa za3.s, p0/M, p0/M, z23.b, z19.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa1400517)  // ld1b { z23.b, z31.b }, pn9.b/Z, [x8]
    subs x1, x1, #0x1
    addvl x8, x8, #2
    KAI_ASM_INST(0xa1400687)  // ld1b { z7.b, z15.b }, pn9.b/Z, [x20]
    addvl x20, x20, #2
    KAI_ASM_INST(0xa08702e0)  // smopa za0.s, p0/M, p0/M, z23.b, z7.b
    KAI_ASM_INST(0xa08f02e1)  // smopa za1.s, p0/M, p0/M, z23.b, z15.b
    KAI_ASM_INST(0xa08703e2)  // smopa za2.s, p0/M, p0/M, z31.b, z7.b
    KAI_ASM_INST(0xa08f03e3)  // smopa za3.s, p0/M, p0/M, z31.b, z15.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x10, #1, label_15
    tbz x10, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c538)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c524)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c534)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c530)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840700)  // mova za0h.s[x12], { z24.s-z27.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840481)  // mova za1h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xa060c79c)  // st1w { z28.s-z31.s }, pn9.b, [x28]
    KAI_ASM_INST(0xc0840682)  // mova za2h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xa061c780)  // st1w { z0.s-z3.s }, pn9.b, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840603)  // mova za3h.s[x12], { z16.s-z19.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c788)  // st1w { z8.s-z11.s }, pn9.b, [x28, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c78c)  // st1w { z12.s-z15.s }, pn9.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_12
    b label_24
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c784)  // st1w { z4.s-z7.s }, pn9.b, [x28]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c788)  // st1w { z8.s-z11.s }, pn9.b, [x28, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c780)  // st1w { z0.s-z3.s }, pn9.b, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c798)  // st1w { z24.s-z27.s }, pn9.b, [x28, #0xc, MUL VL]
    addvl x28, x28, #16
    blt label_14
    b label_24
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x23, x25
    cntw x5
    ldr x4, [x0, #0x58]
    cmp x6, x5
    mov x12, #0
    csel x3, x6, x5, LT
    add x7, x7, x24, LSL #2  // C += n
    lsr x2, x3, #0x2
    madd x7, x25, x4, x7  // C += m * ldc
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860438)  // mova { z24.s-z27.s }, za1h.s[x12]
    KAI_ASM_INST(0xa16040f0)  // st1w { z16.s, z24.s }, p8, [x7]
    add x7, x7, x4
    add x12, x12, #0x4
    KAI_ASM_INST(0xa16040f1)  // st1w { z17.s, z25.s }, p8, [x7]
    add x7, x7, x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa16040f2)  // st1w { z18.s, z26.s }, p8, [x7]
    add x7, x7, x4
    KAI_ASM_INST(0xa16040f3)  // st1w { z19.s, z27.s }, p8, [x7]
    add x7, x7, x4
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xa16040e4)  // st1w { z4.s, z12.s }, p8, [x7]
    add x7, x7, x4
    beq label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa16040e5)  // st1w { z5.s, z13.s }, p8, [x7]
    add x7, x7, x4
    beq label_18
    KAI_ASM_INST(0xa16040e6)  // st1w { z6.s, z14.s }, p8, [x7]
    add x7, x7, x4
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments: End
    subs x6, x6, x3
    beq label_22
    cmp x6, x5
    mov x12, #0
    csel x1, x6, x5, LT
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_20
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    KAI_ASM_INST(0xa16040f0)  // st1w { z16.s, z24.s }, p8, [x7]
    add x7, x7, x4
    add x12, x12, #0x4
    KAI_ASM_INST(0xa16040f1)  // st1w { z17.s, z25.s }, p8, [x7]
    add x7, x7, x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa16040f2)  // st1w { z18.s, z26.s }, p8, [x7]
    add x7, x7, x4
    KAI_ASM_INST(0xa16040f3)  // st1w { z19.s, z27.s }, p8, [x7]
    add x7, x7, x4
    blt label_19
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_21
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    KAI_ASM_INST(0xa16040f0)  // st1w { z16.s, z24.s }, p8, [x7]
    add x7, x7, x4
    beq label_21
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa16040f1)  // st1w { z17.s, z25.s }, p8, [x7]
    add x7, x7, x4
    beq label_21
    KAI_ASM_INST(0xa16040f2)  // st1w { z18.s, z26.s }, p8, [x7]
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_22)  // Store to output array: End
    tbz x10, #0, label_24
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_23)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c528)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xa041c520)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x9, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c52c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x9, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c524)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x9, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840500)  // mova za0h.s[x12], { z8.s-z11.s }
    addvl x9, x9, #16
    KAI_ASM_INST(0xc0840401)  // mova za1h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xc0840582)  // mova za2h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840483)  // mova za3h.s[x12], { z4.s-z7.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_23
KAI_ASM_LABEL(label_24)  // End block
    incw x24, ALL, MUL #2
    cmp x24, x22
    blt label_4
    ldr x1, [x0, #0x70]
    incw x25, ALL, MUL #2
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL)

    KAI_ASM_END
