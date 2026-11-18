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

    KAI_ASM_CODE(sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)
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
    KAI_ASM_INST(0xa040c79c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c790)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c788)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840780)  // mova za0h.s[x12], { z28.s-z31.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840601)  // mova za1h.s[x12], { z16.s-z19.s }
    KAI_ASM_INST(0xc0840502)  // mova za2h.s[x12], { z8.s-z11.s }
    KAI_ASM_INST(0xc0840583)  // mova za3h.s[x12], { z12.s-z15.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x25, [x0, #0x10]
    mov x24, #0
    mov x23, #0
    ldr x22, [x0, #0]
    ldr x21, [x0, #0x8]
    add x25, x25, #0x3
    ldr x20, [x0, #0x20]
    lsr x25, x25, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x7, x20
    KAI_ASM_INST(0x25b566f0)  // whilelt pn8.s, x23, x21, VLx4
    tbnz x9, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    KAI_ASM_INST(0xa117c032)  // ld1w { z18.s, z22.s, z26.s, z30.s }, p8/Z, [x1, x23, LSL #2]
    KAI_ASM_INST(0xc0900240)  // addha za0.s, p0/M, p0/M, z18.s
    KAI_ASM_INST(0xc09002c1)  // addha za1.s, p0/M, p0/M, z22.s
    KAI_ASM_INST(0xc0900342)  // addha za2.s, p0/M, p0/M, z26.s
    KAI_ASM_INST(0xc09003c3)  // addha za3.s, p0/M, p0/M, z30.s
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
    KAI_ASM_INST(0xa14084f1)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa1408510)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa1418513)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa1428512)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa1438500)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa0900220)  // smopa za0.s, p0/M, p0/M, z17.b, z16.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa0940221)  // smopa za1.s, p0/M, p0/M, z17.b, z20.b
    KAI_ASM_INST(0xa0980222)  // smopa za2.s, p0/M, p0/M, z17.b, z24.b
    KAI_ASM_INST(0xa09c0223)  // smopa za3.s, p0/M, p0/M, z17.b, z28.b
    KAI_ASM_INST(0xa1408510)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa09302a0)  // smopa za0.s, p0/M, p0/M, z21.b, z19.b
    KAI_ASM_INST(0xa09702a1)  // smopa za1.s, p0/M, p0/M, z21.b, z23.b
    KAI_ASM_INST(0xa09b02a2)  // smopa za2.s, p0/M, p0/M, z21.b, z27.b
    KAI_ASM_INST(0xa09f02a3)  // smopa za3.s, p0/M, p0/M, z21.b, z31.b
    KAI_ASM_INST(0xa1418513)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x8, #0x4, MUL VL]
    KAI_ASM_INST(0xa0920320)  // smopa za0.s, p0/M, p0/M, z25.b, z18.b
    KAI_ASM_INST(0xa0960321)  // smopa za1.s, p0/M, p0/M, z25.b, z22.b
    KAI_ASM_INST(0xa09a0322)  // smopa za2.s, p0/M, p0/M, z25.b, z26.b
    KAI_ASM_INST(0xa09e0323)  // smopa za3.s, p0/M, p0/M, z25.b, z30.b
    KAI_ASM_INST(0xa1428512)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x8, #0x8, MUL VL]
    KAI_ASM_INST(0xa08003a0)  // smopa za0.s, p0/M, p0/M, z29.b, z0.b
    KAI_ASM_INST(0xa08403a1)  // smopa za1.s, p0/M, p0/M, z29.b, z4.b
    KAI_ASM_INST(0xa08803a2)  // smopa za2.s, p0/M, p0/M, z29.b, z8.b
    KAI_ASM_INST(0xa08c03a3)  // smopa za3.s, p0/M, p0/M, z29.b, z12.b
    KAI_ASM_INST(0xa14084f1)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa1438500)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x8, #0xc, MUL VL]
    addvl x8, x8, #16
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa0900220)  // smopa za0.s, p0/M, p0/M, z17.b, z16.b
    KAI_ASM_INST(0xa0940221)  // smopa za1.s, p0/M, p0/M, z17.b, z20.b
    KAI_ASM_INST(0xa0980222)  // smopa za2.s, p0/M, p0/M, z17.b, z24.b
    KAI_ASM_INST(0xa09c0223)  // smopa za3.s, p0/M, p0/M, z17.b, z28.b
    KAI_ASM_INST(0xa09302a0)  // smopa za0.s, p0/M, p0/M, z21.b, z19.b
    KAI_ASM_INST(0xa09702a1)  // smopa za1.s, p0/M, p0/M, z21.b, z23.b
    KAI_ASM_INST(0xa09b02a2)  // smopa za2.s, p0/M, p0/M, z21.b, z27.b
    KAI_ASM_INST(0xa09f02a3)  // smopa za3.s, p0/M, p0/M, z21.b, z31.b
    KAI_ASM_INST(0xa0920320)  // smopa za0.s, p0/M, p0/M, z25.b, z18.b
    KAI_ASM_INST(0xa0960321)  // smopa za1.s, p0/M, p0/M, z25.b, z22.b
    KAI_ASM_INST(0xa09a0322)  // smopa za2.s, p0/M, p0/M, z25.b, z26.b
    KAI_ASM_INST(0xa09e0323)  // smopa za3.s, p0/M, p0/M, z25.b, z30.b
    KAI_ASM_INST(0xa08003a0)  // smopa za0.s, p0/M, p0/M, z29.b, z0.b
    KAI_ASM_INST(0xa08403a1)  // smopa za1.s, p0/M, p0/M, z29.b, z4.b
    KAI_ASM_INST(0xa08803a2)  // smopa za2.s, p0/M, p0/M, z29.b, z8.b
    KAI_ASM_INST(0xa08c03a3)  // smopa za3.s, p0/M, p0/M, z29.b, z12.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z15.b }, p0/Z, [x7]
    subs x1, x1, #0x1
    addvl x7, x7, #1
    KAI_ASM_INST(0xa040851c)  // ld1b { z28.b-z31.b }, pn9.b/Z, [x8]
    addvl x8, x8, #4
    KAI_ASM_INST(0xa09c01e0)  // smopa za0.s, p0/M, p0/M, z15.b, z28.b
    KAI_ASM_INST(0xa09d01e1)  // smopa za1.s, p0/M, p0/M, z15.b, z29.b
    KAI_ASM_INST(0xa09e01e2)  // smopa za2.s, p0/M, p0/M, z15.b, z30.b
    KAI_ASM_INST(0xa09f01e3)  // smopa za3.s, p0/M, p0/M, z15.b, z31.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x9, #1, label_15
    tbz x9, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c79c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c798)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840781)  // mova za1h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xa060c768)  // st1w { z8.s-z11.s }, pn9.b, [x27]
    KAI_ASM_INST(0xc0840702)  // mova za2h.s[x12], { z24.s-z27.s }
    KAI_ASM_INST(0xa061c76c)  // st1w { z12.s-z15.s }, pn9.b, [x27, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840683)  // mova za3h.s[x12], { z20.s-z23.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c764)  // st1w { z4.s-z7.s }, pn9.b, [x27, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c770)  // st1w { z16.s-z19.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_12
    b label_22
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c764)  // st1w { z4.s-z7.s }, pn9.b, [x27]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c770)  // st1w { z16.s-z19.s }, pn9.b, [x27, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c768)  // st1w { z8.s-z11.s }, pn9.b, [x27, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c76c)  // st1w { z12.s-z15.s }, pn9.b, [x27, #0xc, MUL VL]
    addvl x27, x27, #16
    blt label_14
    b label_22
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x22, x24
    KAI_ASM_INST(0x8566c012)  // ld1rw { z18.s }, p0/Z, [x0, #152]
    mov z20.s, #0
    ldr x4, [x0, #0x58]
    mov z21.s, #0
    mov z22.s, #0
    ldr x1, [x0, #0x48]
    mov z23.s, #0
    add x6, x6, x23, LSL #2  // C += n
    madd x6, x24, x4, x6  // C += m * ldc
    cbz x1, label_16
    add x1, x1, x23, LSL #2
    KAI_ASM_INST(0xa040c034)  // ld1w { z20.s-z23.s }, p8/Z, [x1]
KAI_ASM_LABEL(label_16)  // Store to output array: no late bias
    cntw x3
    add x1, x0, #0xa0
    cmp x5, x3
    add x2, x0, #0xa4
    KAI_ASM_INST(0x8540c031)  // ld1rw { z17.s }, p0/Z, [x1]
    csel x1, x5, x3, LT
    KAI_ASM_INST(0x8540c050)  // ld1rw { z16.s }, p0/Z, [x2]
    mov x12, #0
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_18
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s-z3.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmad z0.s, p0/M, z18.s, z20.s
    fmad z1.s, p0/M, z18.s, z20.s
    fmad z2.s, p0/M, z18.s, z20.s
    fmad z3.s, p0/M, z18.s, z20.s
    add x12, x12, #0x4
    fmad z4.s, p0/M, z18.s, z21.s
    fmad z5.s, p0/M, z18.s, z21.s
    cmp x12, x2, LSL #2
    fmad z6.s, p0/M, z18.s, z21.s
    fmad z7.s, p0/M, z18.s, z21.s
    fmad z8.s, p0/M, z18.s, z22.s
    fmad z9.s, p0/M, z18.s, z22.s
    fmad z10.s, p0/M, z18.s, z22.s
    fmad z11.s, p0/M, z18.s, z22.s
    fmad z12.s, p0/M, z18.s, z23.s
    fmad z13.s, p0/M, z18.s, z23.s
    fmad z14.s, p0/M, z18.s, z23.s
    fmad z15.s, p0/M, z18.s, z23.s
    KAI_ASM_INST(0xc1b0ca20)  // fclamp { z0.s-z3.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca24)  // fclamp { z4.s-z7.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca28)  // fclamp { z8.s-z11.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca2c)  // fclamp { z12.s-z15.s }, z17.s, z16.s
    KAI_ASM_INST(0xa160c0c0)  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0c1)  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0c2)  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa160c0c3)  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x6]
    add x6, x6, x4
    blt label_17
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_19
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s-z3.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmad z0.s, p0/M, z18.s, z20.s
    fmad z1.s, p0/M, z18.s, z20.s
    fmad z2.s, p0/M, z18.s, z20.s
    fmad z3.s, p0/M, z18.s, z20.s
    subs x1, x1, #0x1
    fmad z4.s, p0/M, z18.s, z21.s
    fmad z5.s, p0/M, z18.s, z21.s
    fmad z6.s, p0/M, z18.s, z21.s
    fmad z7.s, p0/M, z18.s, z21.s
    fmad z8.s, p0/M, z18.s, z22.s
    fmad z9.s, p0/M, z18.s, z22.s
    fmad z10.s, p0/M, z18.s, z22.s
    fmad z11.s, p0/M, z18.s, z22.s
    fmad z12.s, p0/M, z18.s, z23.s
    fmad z13.s, p0/M, z18.s, z23.s
    fmad z14.s, p0/M, z18.s, z23.s
    fmad z15.s, p0/M, z18.s, z23.s
    KAI_ASM_INST(0xc1b0ca20)  // fclamp { z0.s-z3.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca24)  // fclamp { z4.s-z7.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca28)  // fclamp { z8.s-z11.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca2c)  // fclamp { z12.s-z15.s }, z17.s, z16.s
    KAI_ASM_INST(0xa160c0c0)  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x6]
    add x6, x6, x4
    beq label_19
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa160c0c1)  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x6]
    add x6, x6, x4
    beq label_19
    KAI_ASM_INST(0xa160c0c2)  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x6]
KAI_ASM_LABEL(label_19)  // Store to output array: Accumulator row 0 oddments: End
    tbz x9, #0, label_22
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_21)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c788)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x28, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x28, x28, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840402)  // mova za2h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_21
KAI_ASM_LABEL(label_22)  // End block
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL)

    KAI_ASM_END
