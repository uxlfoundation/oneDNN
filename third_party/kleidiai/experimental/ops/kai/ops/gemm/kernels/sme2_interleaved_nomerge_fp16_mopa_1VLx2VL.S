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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16_mopa_1VLx2VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16_mopa_1VLx2VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16_mopa_1VLx2VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16_mopa_1VLx2VL)
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
    mov x6, #0xfc00
    ldr x2, [x0, #0x50]
    mov x5, #0x7c00
    ptrue p0.b
    ldrh w4, [x0, #0xa0]
    cmp x1, XZR
    ldrh w3, [x0, #0xa4]
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
    cnth x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040a788)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041a79c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    KAI_ASM_INST(0xc0440500)  // mova za0h.h[x12, 0:3], { z8.h-z11.h }
    KAI_ASM_INST(0xc0440782)  // mova za1h.h[x12, 0:3], { z28.h-z31.h }
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
    KAI_ASM_INST(0x257546f0)  // whilelt pn8.h, x23, x21, VLx2
    tbnz x9, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z8.h, #1.0
    KAI_ASM_INST(0xa0172020)  // ld1h { z0.h-z1.h }, p8/Z, [x1, x23, LSL #1]
    KAI_ASM_INST(0x81800108)  // fmopa za0.h, p0/M, p0/M, z8.h, z0.h
    KAI_ASM_INST(0x81810109)  // fmopa za1.h, p0/M, p0/M, z8.h, z1.h
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x23
    mov x2, x24
    inch x1, ALL, MUL #2
    inch x2
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
    KAI_ASM_INST(0xa040a4f4)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa140a503)  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0xa141a501)  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81830288)  // fmopa za0.h, p0/M, p0/M, z20.h, z3.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x81870289)  // fmopa za1.h, p0/M, p0/M, z20.h, z7.h
    KAI_ASM_INST(0x818b02a8)  // fmopa za0.h, p0/M, p0/M, z21.h, z11.h
    KAI_ASM_INST(0x818f02a9)  // fmopa za1.h, p0/M, p0/M, z21.h, z15.h
    KAI_ASM_INST(0xa140a503)  // ld1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x8]
    KAI_ASM_INST(0x818102c8)  // fmopa za0.h, p0/M, p0/M, z22.h, z1.h
    KAI_ASM_INST(0x818502c9)  // fmopa za1.h, p0/M, p0/M, z22.h, z5.h
    KAI_ASM_INST(0x818902e8)  // fmopa za0.h, p0/M, p0/M, z23.h, z9.h
    KAI_ASM_INST(0x818d02e9)  // fmopa za1.h, p0/M, p0/M, z23.h, z13.h
    KAI_ASM_INST(0xa040a4f4)  // ld1h { z20.h-z23.h }, pn9.b/Z, [x7]
    addvl x7, x7, #4
    KAI_ASM_INST(0xa141a501)  // ld1h { z1.h, z5.h, z9.h, z13.h }, pn9.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81830288)  // fmopa za0.h, p0/M, p0/M, z20.h, z3.h
    KAI_ASM_INST(0x81870289)  // fmopa za1.h, p0/M, p0/M, z20.h, z7.h
    KAI_ASM_INST(0x818b02a8)  // fmopa za0.h, p0/M, p0/M, z21.h, z11.h
    KAI_ASM_INST(0x818f02a9)  // fmopa za1.h, p0/M, p0/M, z21.h, z15.h
    KAI_ASM_INST(0x818102c8)  // fmopa za0.h, p0/M, p0/M, z22.h, z1.h
    KAI_ASM_INST(0x818502c9)  // fmopa za1.h, p0/M, p0/M, z22.h, z5.h
    KAI_ASM_INST(0x818902e8)  // fmopa za0.h, p0/M, p0/M, z23.h, z9.h
    KAI_ASM_INST(0x818d02e9)  // fmopa za1.h, p0/M, p0/M, z23.h, z13.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    ld1h { z16.h }, p0/Z, [x7]
    subs x1, x1, #0x1
    addvl x7, x7, #1
    KAI_ASM_INST(0xa1402507)  // ld1h { z7.h, z15.h }, pn9.b/Z, [x8]
    addvl x8, x8, #2
    KAI_ASM_INST(0x81870208)  // fmopa za0.h, p0/M, p0/M, z16.h, z7.h
    KAI_ASM_INST(0x818f0209)  // fmopa za1.h, p0/M, p0/M, z16.h, z15.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x9, #1, label_15
    tbz x9, #0, label_13
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040a788)  // ld1h { z8.h-z11.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc046041c)  // mova { z28.h-z31.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xa041a78c)  // ld1h { z12.h-z15.h }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    KAI_ASM_INST(0xa060a77c)  // st1h { z28.h-z31.h }, pn9.b, [x27]
    KAI_ASM_INST(0xc0440500)  // mova za0h.h[x12, 0:3], { z8.h-z11.h }
    KAI_ASM_INST(0xa061a770)  // st1h { z16.h-z19.h }, pn9.b, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xc0440582)  // mova za1h.h[x12, 0:3], { z12.h-z15.h }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_12
    b label_25
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460444)  // mova { z4.h-z7.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa060a76c)  // st1h { z12.h-z15.h }, pn9.b, [x27]
    cmp x12, x1
    KAI_ASM_INST(0xa061a764)  // st1h { z4.h-z7.h }, pn9.b, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    blt label_14
    b label_25
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x6, [x0, #0x50]
    sub x5, x22, x24
    ldr x4, [x0, #0x58]
    add x6, x6, x23, LSL #1  // C += n
    madd x6, x24, x4, x6  // C += m * ldc
    tbnz x9, #2, label_19
    cnth x1
    mov x12, #0
    cmp x5, x1
    csel x3, x5, x1, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Skip activation: Accumulator row 0 loop
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xa16020d0)  // st1h { z16.h, z24.h }, p8, [x6]
    add x6, x6, x4
    add x12, x12, #0x4
    KAI_ASM_INST(0xa16020d1)  // st1h { z17.h, z25.h }, p8, [x6]
    add x6, x6, x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa16020d2)  // st1h { z18.h, z26.h }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa16020d3)  // st1h { z19.h, z27.h }, p8, [x6]
    add x6, x6, x4
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Skip activation: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0460404)  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xa16020c4)  // st1h { z4.h, z12.h }, p8, [x6]
    add x6, x6, x4
    beq label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa16020c5)  // st1h { z5.h, z13.h }, p8, [x6]
    add x6, x6, x4
    beq label_18
    KAI_ASM_INST(0xa16020c6)  // st1h { z6.h, z14.h }, p8, [x6]
    add x6, x6, x4
KAI_ASM_LABEL(label_18)  // Store to output array: Skip activation: Accumulator row 0 oddments: End
    subs x5, x5, x3
    beq label_19
    b label_23
KAI_ASM_LABEL(label_19)  // Store to output array: Skip activation: End
    cnth x3
    add x1, x0, #0xa0
    cmp x5, x3
    add x2, x0, #0xa4
    KAI_ASM_INST(0x84c0a035)  // ld1rh { z21.h }, p0/Z, [x1]
    csel x1, x5, x3, LT
    KAI_ASM_INST(0x84c0a054)  // ld1rh { z20.h }, p0/Z, [x2]
    mov x12, #0
    lsr x2, x1, #0x2
    and x1, x1, #0x3
    cbz x2, label_21
KAI_ASM_LABEL(label_20)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    KAI_ASM_INST(0xc174cab8)  // fclamp { z24.h-z27.h }, z21.h, z20.h
    add x12, x12, #0x4
    cmp x12, x2, LSL #2
    KAI_ASM_INST(0xa16020d0)  // st1h { z16.h, z24.h }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa16020d1)  // st1h { z17.h, z25.h }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa16020d2)  // st1h { z18.h, z26.h }, p8, [x6]
    add x6, x6, x4
    KAI_ASM_INST(0xa16020d3)  // st1h { z19.h, z27.h }, p8, [x6]
    add x6, x6, x4
    blt label_20
KAI_ASM_LABEL(label_21)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_22
    KAI_ASM_INST(0xc0460404)  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc174caa4)  // fclamp { z4.h-z7.h }, z21.h, z20.h
    KAI_ASM_INST(0xc174caac)  // fclamp { z12.h-z15.h }, z21.h, z20.h
    KAI_ASM_INST(0xa16020c4)  // st1h { z4.h, z12.h }, p8, [x6]
    add x6, x6, x4
    beq label_22
    subs x1, x1, #0x1
    KAI_ASM_INST(0xa16020c5)  // st1h { z5.h, z13.h }, p8, [x6]
    add x6, x6, x4
    beq label_22
    KAI_ASM_INST(0xa16020c6)  // st1h { z6.h, z14.h }, p8, [x6]
KAI_ASM_LABEL(label_22)  // Store to output array: Accumulator row 0 oddments: End
KAI_ASM_LABEL(label_23)  // Store to output array: End
    tbz x9, #0, label_25
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_24)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040a790)  // ld1h { z16.h-z19.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041a79c)  // ld1h { z28.h-z31.h }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    KAI_ASM_INST(0xc0440600)  // mova za0h.h[x12, 0:3], { z16.h-z19.h }
    KAI_ASM_INST(0xc0440782)  // mova za1h.h[x12, 0:3], { z28.h-z31.h }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_24
KAI_ASM_LABEL(label_25)  // End block
    inch x23, ALL, MUL #2
    cmp x23, x21
    blt label_4
    ldr x1, [x0, #0x70]
    inch x24
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16_mopa_1VLx2VL)

    KAI_ASM_END
