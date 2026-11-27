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

    KAI_ASM_CODE(sme2_interleaved_nomerge_fp16_mopa_2VLx1VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme2_interleaved_nomerge_fp16_mopa_2VLx1VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme2_interleaved_nomerge_fp16_mopa_2VLx1VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme2_interleaved_nomerge_fp16_mopa_2VLx1VL)
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
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    mov x6, #0xfc00
    ldr x2, [x0, #0x50]
    mov x5, #0x7c00
    ptrue p1.b
    ldrh w4, [x0, #0xa0]
    cmp x1, XZR
    ldrh w3, [x0, #0xa4]
    cset x1, NE
    cmp x2, XZR
    ldr x9, [x0, #0x68]
    orr x10, x10, x1
    cset x1, EQ
    ldr x2, [x0, #0x70]
    cmp x4, x6
    orr x10, x10, x1, LSL #1
    cset x1, NE
    cmp x3, x5
    orr x10, x10, x1, LSL #2
    cset x1, NE
    orr x10, x10, x1, LSL #2
    mov x28, x9
    add x27, x9, x2
    tbz x10, #0, label_2
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_1)  // Initial accumulator load from buffer: Loop
    KAI_ASM_INST(0xa040a12c)  // ld1h { z12.h-z15.h }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041a134)  // ld1h { z20.h-z23.h }, pn8.b/Z, [x9, #0x4, MUL VL]
    addvl x9, x9, #8
    KAI_ASM_INST(0xc0440580)  // mova za0h.h[x12, 0:3], { z12.h-z15.h }
    KAI_ASM_INST(0xc0440682)  // mova za1h.h[x12, 0:3], { z20.h-z23.h }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_1
KAI_ASM_LABEL(label_2)  // Initial accumulator load from buffer: End
    ldr x26, [x0, #0x10]
    mov x25, #0
    mov x24, #0
    ldr x23, [x0, #0]
    ldr x22, [x0, #0x8]
    ldr x21, [x0, #0x20]
KAI_ASM_LABEL(label_3)  // M loop
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_4)  // N loop
    mov x8, x21
    whilelt p0.h, x24, x22
    tbnz x10, #0, label_5
    ldr x1, [x0, #0x40]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x1, label_6
    fmov z12.h, #1.0
    ld1h { z13.h }, p0/Z, [x1, x24, LSL #1]
    KAI_ASM_INST(0x818d2588)  // fmopa za0.h, p1/M, p1/M, z12.h, z13.h
    KAI_ASM_INST(0x818d2589)  // fmopa za1.h, p1/M, p1/M, z12.h, z13.h
KAI_ASM_LABEL(label_5)  // Prepare accumulators: Test for last block
    mov x1, x24
    mov x2, x25
    inch x1
    inch x2, ALL, MUL #2
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
    KAI_ASM_INST(0xa040a10c)  // ld1h { z12.h-z15.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    KAI_ASM_INST(0xa040a280)  // ld1h { z0.h-z3.h }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    ble label_8
KAI_ASM_LABEL(label_7)  // K loop
    KAI_ASM_INST(0x81802588)  // fmopa za0.h, p1/M, p1/M, z12.h, z0.h
    subs x2, x2, #0x1
    KAI_ASM_INST(0x818025a9)  // fmopa za1.h, p1/M, p1/M, z13.h, z0.h
    KAI_ASM_INST(0x818125c8)  // fmopa za0.h, p1/M, p1/M, z14.h, z1.h
    KAI_ASM_INST(0x818125e9)  // fmopa za1.h, p1/M, p1/M, z15.h, z1.h
    KAI_ASM_INST(0xa040a10c)  // ld1h { z12.h-z15.h }, pn8.b/Z, [x8]
    KAI_ASM_INST(0x81822648)  // fmopa za0.h, p1/M, p1/M, z18.h, z2.h
    KAI_ASM_INST(0x818226c9)  // fmopa za1.h, p1/M, p1/M, z22.h, z2.h
    KAI_ASM_INST(0x81832748)  // fmopa za0.h, p1/M, p1/M, z26.h, z3.h
    KAI_ASM_INST(0x818327c9)  // fmopa za1.h, p1/M, p1/M, z30.h, z3.h
    KAI_ASM_INST(0xa141a112)  // ld1h { z18.h, z22.h, z26.h, z30.h }, pn8.b/Z, [x8, #0x4, MUL VL]
    addvl x8, x8, #8
    KAI_ASM_INST(0xa040a280)  // ld1h { z0.h-z3.h }, pn8.b/Z, [x20]
    addvl x20, x20, #4
    bgt label_7
KAI_ASM_LABEL(label_8)  // K loop tail
    KAI_ASM_INST(0x81802588)  // fmopa za0.h, p1/M, p1/M, z12.h, z0.h
    KAI_ASM_INST(0x818025a9)  // fmopa za1.h, p1/M, p1/M, z13.h, z0.h
    KAI_ASM_INST(0x818125c8)  // fmopa za0.h, p1/M, p1/M, z14.h, z1.h
    KAI_ASM_INST(0x818125e9)  // fmopa za1.h, p1/M, p1/M, z15.h, z1.h
    KAI_ASM_INST(0x81822648)  // fmopa za0.h, p1/M, p1/M, z18.h, z2.h
    KAI_ASM_INST(0x818226c9)  // fmopa za1.h, p1/M, p1/M, z22.h, z2.h
    KAI_ASM_INST(0x81832748)  // fmopa za0.h, p1/M, p1/M, z26.h, z3.h
    KAI_ASM_INST(0x818327c9)  // fmopa za1.h, p1/M, p1/M, z30.h, z3.h
KAI_ASM_LABEL(label_9)  // K oddments
    cbz x1, label_11
KAI_ASM_LABEL(label_10)  // K oddments: Loop
    KAI_ASM_INST(0xa040211a)  // ld1h { z26.h-z27.h }, pn8.b/Z, [x8]
    subs x1, x1, #0x1
    addvl x8, x8, #2
    ld1h { z1.h }, p1/Z, [x20]
    addvl x20, x20, #1
    KAI_ASM_INST(0x81812748)  // fmopa za0.h, p1/M, p1/M, z26.h, z1.h
    KAI_ASM_INST(0x81812769)  // fmopa za1.h, p1/M, p1/M, z27.h, z1.h
    bgt label_10
KAI_ASM_LABEL(label_11)  // K oddments: End
    tbz x10, #1, label_15
    tbz x10, #0, label_13
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_12)  // Store to partial result buffer: Store and refill: Loop
    KAI_ASM_INST(0xa040a128)  // ld1h { z8.h-z11.h }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc0460418)  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046045c)  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xa041a12c)  // ld1h { z12.h-z15.h }, pn8.b/Z, [x9, #0x4, MUL VL]
    addvl x9, x9, #8
    KAI_ASM_INST(0xa060a398)  // st1h { z24.h-z27.h }, pn8.b, [x28]
    KAI_ASM_INST(0xc0440500)  // mova za0h.h[x12, 0:3], { z8.h-z11.h }
    KAI_ASM_INST(0xa061a39c)  // st1h { z28.h-z31.h }, pn8.b, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    KAI_ASM_INST(0xc0440582)  // mova za1h.h[x12, 0:3], { z12.h-z15.h }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_12
    b label_31
KAI_ASM_LABEL(label_13)  // Store to partial result buffer: Store only
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_14)  // Store to partial result buffer: Store only: Loop
    KAI_ASM_INST(0xc0460418)  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xa060a398)  // st1h { z24.h-z27.h }, pn8.b, [x28]
    cmp x12, x1
    KAI_ASM_INST(0xa061a38c)  // st1h { z12.h-z15.h }, pn8.b, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    blt label_14
    b label_31
KAI_ASM_LABEL(label_15)  // Store to output array
    ldr x7, [x0, #0x50]
    sub x6, x23, x25
    ldr x5, [x0, #0x58]
    add x7, x7, x24, LSL #1  // C += n
    madd x7, x25, x5, x7  // C += m * ldc
    tbnz x10, #2, label_22
    cnth x4
    mov x12, #0
    cmp x6, x4
    csel x3, x6, x4, LT
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_17
KAI_ASM_LABEL(label_16)  // Store to output array: Skip activation: Accumulator row 0 loop
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    add x12, x12, #0x4
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    st1h { z18.h }, p0, [x7]
    add x7, x7, x5
    st1h { z19.h }, p0, [x7]
    add x7, x7, x5
    blt label_16
KAI_ASM_LABEL(label_17)  // Store to output array: Skip activation: Accumulator row 0 oddments
    cbz x1, label_18
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc0460400)  // mova { z0.h-z3.h }, za0h.h[x12, 0:3]
    st1h { z0.h }, p0, [x7]
    add x7, x7, x5
    beq label_18
    subs x1, x1, #0x1
    st1h { z1.h }, p0, [x7]
    add x7, x7, x5
    beq label_18
    st1h { z2.h }, p0, [x7]
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
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    cmp x12, x2, LSL #2
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    st1h { z18.h }, p0, [x7]
    add x7, x7, x5
    st1h { z19.h }, p0, [x7]
    add x7, x7, x5
    blt label_19
KAI_ASM_LABEL(label_20)  // Store to output array: Skip activation: Accumulator row 1 oddments
    cbz x1, label_21
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    st1h { z12.h }, p0, [x7]
    add x7, x7, x5
    beq label_21
    subs x1, x1, #0x1
    st1h { z13.h }, p0, [x7]
    add x7, x7, x5
    beq label_21
    st1h { z14.h }, p0, [x7]
    add x7, x7, x5
KAI_ASM_LABEL(label_21)  // Store to output array: Skip activation: Accumulator row 1 oddments: End
    subs x6, x6, x3
    beq label_22
    b label_29
KAI_ASM_LABEL(label_22)  // Store to output array: Skip activation: End
    cnth x4
    add x2, x0, #0xa0
    cmp x6, x4
    add x1, x0, #0xa4
    KAI_ASM_INST(0x84c0a455)  // ld1rh { z21.h }, p1/Z, [x2]
    csel x3, x6, x4, LT
    KAI_ASM_INST(0x84c0a434)  // ld1rh { z20.h }, p1/Z, [x1]
    mov x12, #0
    lsr x2, x3, #0x2
    and x1, x3, #0x3
    cbz x2, label_24
KAI_ASM_LABEL(label_23)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    cmp x12, x2, LSL #2
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    st1h { z18.h }, p0, [x7]
    add x7, x7, x5
    st1h { z19.h }, p0, [x7]
    add x7, x7, x5
    blt label_23
KAI_ASM_LABEL(label_24)  // Store to output array: Accumulator row 0 oddments
    cbz x1, label_25
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    beq label_25
    subs x1, x1, #0x1
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    beq label_25
    st1h { z18.h }, p0, [x7]
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
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    cmp x12, x2, LSL #2
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    st1h { z18.h }, p0, [x7]
    add x7, x7, x5
    st1h { z19.h }, p0, [x7]
    add x7, x7, x5
    blt label_26
KAI_ASM_LABEL(label_27)  // Store to output array: Accumulator row 1 oddments
    cbz x1, label_28
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    subs x1, x1, #0x1
    KAI_ASM_INST(0xc174cab0)  // fclamp { z16.h-z19.h }, z21.h, z20.h
    st1h { z16.h }, p0, [x7]
    add x7, x7, x5
    beq label_28
    subs x1, x1, #0x1
    st1h { z17.h }, p0, [x7]
    add x7, x7, x5
    beq label_28
    st1h { z18.h }, p0, [x7]
KAI_ASM_LABEL(label_28)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_29)  // Store to output array: End
    tbz x10, #0, label_31
    mov x12, #0
    cnth x1
KAI_ASM_LABEL(label_30)  // Store to output array: Refill accumulators: Loop
    KAI_ASM_INST(0xa040a124)  // ld1h { z4.h-z7.h }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xa041a128)  // ld1h { z8.h-z11.h }, pn8.b/Z, [x9, #0x4, MUL VL]
    addvl x9, x9, #8
    KAI_ASM_INST(0xc0440480)  // mova za0h.h[x12, 0:3], { z4.h-z7.h }
    KAI_ASM_INST(0xc0440502)  // mova za1h.h[x12, 0:3], { z8.h-z11.h }
    add x12, x12, #0x4
    cmp x12, x1
    blt label_30
KAI_ASM_LABEL(label_31)  // End block
    inch x24
    cmp x24, x22
    blt label_4
    ldr x1, [x0, #0x70]
    inch x25, ALL, MUL #2
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
    KAI_ASM_FUNCTION_END(kai_ops_sme2_interleaved_nomerge_fp16_mopa_2VLx1VL)

    KAI_ASM_END
