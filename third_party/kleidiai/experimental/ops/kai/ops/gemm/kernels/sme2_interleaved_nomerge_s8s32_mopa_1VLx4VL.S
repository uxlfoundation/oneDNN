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

    KAI_ASM_CODE(sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x27, #0
    ldr x1, [x0, #0x60]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ptrue p0.b
    ldr x3, [x0, #0x50]
    ldr x26, [x0, #0x68]
    cmp x1, XZR
    ldr x2, [x0, #0x70]
    cset x1, NE
    cmp x3, XZR
    orr x27, x27, x1
    cset x1, EQ
    orr x27, x27, x1, LSL #1
    mov x25, x26
    add x24, x26, x2
    tbz x27, #0, label_2
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040c744)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xa041c74c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x26, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c754)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x26, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c740)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x26, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840480)  // mova za0h.s[x12], { z4.s-z7.s }
    addvl x26, x26, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xc0840682)  // mova za2h.s[x12], { z20.s-z23.s }
    KAI_ASM_INST(0xc0840403)  // mova za3h.s[x12], { z0.s-z3.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x23, [x0, #0x10]
    mov x22, #0
    mov x21, #0
    ldr x20, [x0, #0]
    ldr x8, [x0, #0x8]
    add x23, x23, #0x3
    ldr x7, [x0, #0x20]
    lsr x23, x23, #0x2
KAI_ASM_LABEL(label_3)  // M loop
    ldr x6, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x5, x7
    KAI_ASM_INST(0x25a866b0)  // whilelt pn8.s, x21, x8, VLx4
    tbnz x27, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    KAI_ASM_INST(0xa115c030)  // ld1w { z16.s, z20.s, z24.s, z28.s }, p8/Z, [x1, x21, LSL #2]
    KAI_ASM_INST(0xc0900200)  // addha za0.s, p0/M, p0/M, z16.s
    KAI_ASM_INST(0xc0900281)  // addha za1.s, p0/M, p0/M, z20.s
    KAI_ASM_INST(0xc0900302)  // addha za2.s, p0/M, p0/M, z24.s
    KAI_ASM_INST(0xc0900383)  // addha za3.s, p0/M, p0/M, z28.s
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x21
    mov x2, x22
    incw x1, ALL, MUL #4
    incw x2
    cmp x1, x8
    mov x1, x27
    csel x2, x22, x2, LT
    bfm x27, XZR, #0, #0  // bfc x27, #0, #0x1
    csel x26, x26, x24, LT
    cmp x2, x20
    csel x27, x1, x27, LT
KAI_ASM_LABEL(label_6)  // Prepare accumulators: End
    lsr x2, x23, #0x2
    and x1, x23, #0x3
    cbz x2, label_9
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa04084ac)  // ld1b { z12.b-z15.b }, pn9.b/Z, [x5]
    addvl x5, x5, #4
    KAI_ASM_INST(0xa04084d0)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x6]
    KAI_ASM_INST(0xa04184d8)  // ld1b { z24.b-z27.b }, pn9.b/Z, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa04284dc)  // ld1b { z28.b-z31.b }, pn9.b/Z, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa04384c4)  // ld1b { z4.b-z7.b }, pn9.b/Z, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0xa0900180)  // smopa za0.s, p0/M, p0/M, z12.b, z16.b
    subs x2, x2, #0x1
    KAI_ASM_INST(0xa0910181)  // smopa za1.s, p0/M, p0/M, z12.b, z17.b
    KAI_ASM_INST(0xa0920182)  // smopa za2.s, p0/M, p0/M, z12.b, z18.b
    KAI_ASM_INST(0xa0930183)  // smopa za3.s, p0/M, p0/M, z12.b, z19.b
    KAI_ASM_INST(0xa04084d0)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x6]
    KAI_ASM_INST(0xa09801a0)  // smopa za0.s, p0/M, p0/M, z13.b, z24.b
    KAI_ASM_INST(0xa09901a1)  // smopa za1.s, p0/M, p0/M, z13.b, z25.b
    KAI_ASM_INST(0xa09a01a2)  // smopa za2.s, p0/M, p0/M, z13.b, z26.b
    KAI_ASM_INST(0xa09b01a3)  // smopa za3.s, p0/M, p0/M, z13.b, z27.b
    KAI_ASM_INST(0xa04184d8)  // ld1b { z24.b-z27.b }, pn9.b/Z, [x6, #0x4, MUL VL]
    KAI_ASM_INST(0xa09c01c0)  // smopa za0.s, p0/M, p0/M, z14.b, z28.b
    KAI_ASM_INST(0xa09d01c1)  // smopa za1.s, p0/M, p0/M, z14.b, z29.b
    KAI_ASM_INST(0xa09e01c2)  // smopa za2.s, p0/M, p0/M, z14.b, z30.b
    KAI_ASM_INST(0xa09f01c3)  // smopa za3.s, p0/M, p0/M, z14.b, z31.b
    KAI_ASM_INST(0xa04284dc)  // ld1b { z28.b-z31.b }, pn9.b/Z, [x6, #0x8, MUL VL]
    KAI_ASM_INST(0xa08401e0)  // smopa za0.s, p0/M, p0/M, z15.b, z4.b
    KAI_ASM_INST(0xa08501e1)  // smopa za1.s, p0/M, p0/M, z15.b, z5.b
    KAI_ASM_INST(0xa08601e2)  // smopa za2.s, p0/M, p0/M, z15.b, z6.b
    KAI_ASM_INST(0xa08701e3)  // smopa za3.s, p0/M, p0/M, z15.b, z7.b
    KAI_ASM_INST(0xa04084ac)  // ld1b { z12.b-z15.b }, pn9.b/Z, [x5]
    addvl x5, x5, #4
    KAI_ASM_INST(0xa04384c4)  // ld1b { z4.b-z7.b }, pn9.b/Z, [x6, #0xc, MUL VL]
    addvl x6, x6, #16
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0xa0900180)  // smopa za0.s, p0/M, p0/M, z12.b, z16.b
    KAI_ASM_INST(0xa0910181)  // smopa za1.s, p0/M, p0/M, z12.b, z17.b
    KAI_ASM_INST(0xa0920182)  // smopa za2.s, p0/M, p0/M, z12.b, z18.b
    KAI_ASM_INST(0xa0930183)  // smopa za3.s, p0/M, p0/M, z12.b, z19.b
    KAI_ASM_INST(0xa09801a0)  // smopa za0.s, p0/M, p0/M, z13.b, z24.b
    KAI_ASM_INST(0xa09901a1)  // smopa za1.s, p0/M, p0/M, z13.b, z25.b
    KAI_ASM_INST(0xa09a01a2)  // smopa za2.s, p0/M, p0/M, z13.b, z26.b
    KAI_ASM_INST(0xa09b01a3)  // smopa za3.s, p0/M, p0/M, z13.b, z27.b
    KAI_ASM_INST(0xa09c01c0)  // smopa za0.s, p0/M, p0/M, z14.b, z28.b
    KAI_ASM_INST(0xa09d01c1)  // smopa za1.s, p0/M, p0/M, z14.b, z29.b
    KAI_ASM_INST(0xa09e01c2)  // smopa za2.s, p0/M, p0/M, z14.b, z30.b
    KAI_ASM_INST(0xa09f01c3)  // smopa za3.s, p0/M, p0/M, z14.b, z31.b
    KAI_ASM_INST(0xa08401e0)  // smopa za0.s, p0/M, p0/M, z15.b, z4.b
    KAI_ASM_INST(0xa08501e1)  // smopa za1.s, p0/M, p0/M, z15.b, z5.b
    KAI_ASM_INST(0xa08601e2)  // smopa za2.s, p0/M, p0/M, z15.b, z6.b
    KAI_ASM_INST(0xa08701e3)  // smopa za3.s, p0/M, p0/M, z15.b, z7.b
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1b { z4.b }, p0/Z, [x5]
    subs x1, x1, #0x1
    addvl x5, x5, #1
    KAI_ASM_INST(0xa04084d0)  // ld1b { z16.b-z19.b }, pn9.b/Z, [x6]
    addvl x6, x6, #4
    KAI_ASM_INST(0xa0900080)  // smopa za0.s, p0/M, p0/M, z4.b, z16.b
    KAI_ASM_INST(0xa0910081)  // smopa za1.s, p0/M, p0/M, z4.b, z17.b
    KAI_ASM_INST(0xa0920082)  // smopa za2.s, p0/M, p0/M, z4.b, z18.b
    KAI_ASM_INST(0xa0930083)  // smopa za3.s, p0/M, p0/M, z4.b, z19.b
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x27, #1, label_15
    tbz x27, #0, label_13
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040c754)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc0860418)  // mova { z24.s-z27.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xa041c74c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x26, #0x4, MUL VL]
    KAI_ASM_INST(0xc086045c)  // mova { z28.s-z31.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa042c740)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x26, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c744)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x26, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840680)  // mova za0h.s[x12], { z20.s-z23.s }
    addvl x26, x26, #16
    KAI_ASM_INST(0xc0840581)  // mova za1h.s[x12], { z12.s-z15.s }
    KAI_ASM_INST(0xa060c738)  // st1w { z24.s-z27.s }, pn9.b, [x25]
    KAI_ASM_INST(0xc0840402)  // mova za2h.s[x12], { z0.s-z3.s }
    KAI_ASM_INST(0xa061c730)  // st1w { z16.s-z19.s }, pn9.b, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc0840483)  // mova za3h.s[x12], { z4.s-z7.s }
    add x12, x12, #0x4
    KAI_ASM_INST(0xa062c73c)  // st1w { z28.s-z31.s }, pn9.b, [x25, #0x8, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa063c728)  // st1w { z8.s-z11.s }, pn9.b, [x25, #0xc, MUL VL]
    addvl x25, x25, #16
    blt label_12
    b label_21
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xa060c720)  // st1w { z0.s-z3.s }, pn9.b, [x25]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa061c72c)  // st1w { z12.s-z15.s }, pn9.b, [x25, #0x4, MUL VL]
    cmp x12, x1
    KAI_ASM_INST(0xa062c738)  // st1w { z24.s-z27.s }, pn9.b, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c730)  // st1w { z16.s-z19.s }, pn9.b, [x25, #0xc, MUL VL]
    addvl x25, x25, #16
    blt label_14
    b label_21
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x4, [x0, #0x50]
    sub x2, x20, x22
    cntw x1
    ldr x3, [x0, #0x58]
    cmp x2, x1
    mov x12, #0
    csel x1, x2, x1, LT
    add x4, x4, x21, LSL #2  // C += n
    lsr x2, x1, #0x2
    madd x4, x22, x3, x4  // C += m * ldc
    and x1, x1, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa160c080)  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x4]
    add x4, x4, x3
    add x12, x12, #0x4
    KAI_ASM_INST(0xa160c081)  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x4]
    add x4, x4, x3
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa160c082)  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x4]
    add x4, x4, x3
    KAI_ASM_INST(0xa160c083)  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x4]
    add x4, x4, x3
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xa160c080)  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x4]
    add x4, x4, x3
    beq label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa160c081)  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x4]
    add x4, x4, x3
    beq label_18
    KAI_ASM_INST(0xa160c082)  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x4]
KAI_ASM_LABEL(label_18)  // Store to output array: Accumulator row 0 oddments: End
    tbz x27, #0, label_21
    mov x12, #0
    cntw x1
KAI_ASM_LABEL(label_20)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040c740)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xa041c75c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x26, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c744)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x26, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c748)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x26, #0xc, MUL VL]
    KAI_ASM_INST(0xc0840400)  // mova za0h.s[x12], { z0.s-z3.s }
    addvl x26, x26, #16
    KAI_ASM_INST(0xc0840781)  // mova za1h.s[x12], { z28.s-z31.s }
    KAI_ASM_INST(0xc0840482)  // mova za2h.s[x12], { z4.s-z7.s }
    KAI_ASM_INST(0xc0840503)  // mova za3h.s[x12], { z8.s-z11.s }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_20
KAI_ASM_LABEL(label_21)  // End block
    incw x21, ALL, MUL #4
    cmp x21, x8
    blt label_4
    ldr x1, [x0, #0x70]
    incw x22
    mov x25, x24
    cmp x22, x20
    mov x21, #0
    mov x7, x5
    add x24, x25, x1
    blt label_3
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL)

    KAI_ASM_END
