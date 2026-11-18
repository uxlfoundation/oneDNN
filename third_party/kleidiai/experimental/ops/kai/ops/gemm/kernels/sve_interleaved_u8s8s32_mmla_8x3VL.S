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

    KAI_ASM_CODE(sve_interleaved_u8s8s32_mmla_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_interleaved_u8s8s32_mmla_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_interleaved_u8s8s32_mmla_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_interleaved_u8s8s32_mmla_8x3VL)
    stp d8, d9, [sp, -64]!
    stp d10, d11, [sp, 16]
    stp d12, d13, [sp, 32]
    stp d14, d15, [sp, 48]
    ptrue p0.b
    ldr x7, [x0, #0]
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x50]
    cbz x7, label_6
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x30]
    mov x2, x6
KAI_ASM_LABEL(label_2)  // Width loop
    ldr x1, [x0, #0x10]
    mov x6, x2
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    ld1b { z4.b }, p0/Z, [x3]
    mov z11.s, #0
    mov z12.s, #0
    ld1b { z5.b }, p0/Z, [x3, #1, MUL VL]
    add x1, x1, #0x7
    mov z13.s, #0
    mov z14.s, #0
    lsr x1, x1, #0x3
    mov z15.s, #0
    mov z16.s, #0
    mov z17.s, #0
    mov z18.s, #0
    ld1rqb { z0.b }, p0/Z, [x6]
    mov z19.s, #0
    mov z20.s, #0
    ld1rqb { z1.b }, p0/Z, [x6, #16]
    sub x1, x1, #0x1
    mov z21.s, #0
    mov z22.s, #0
    cmp x1, #0x2
    mov z23.s, #0
    mov z24.s, #0
    mov z25.s, #0
    mov z26.s, #0
    ld1rqb { z2.b }, p0/Z, [x6, #32]
    mov z27.s, #0
    mov z28.s, #0
    addvl x3, x3, #2
    mov z29.s, #0
    mov z30.s, #0
    add x6, x6, #0x30
    mov z31.s, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    ld1rqb { z3.b }, p0/Z, [x6]
    KAI_ASM_INST(0x45849808)  // usmmla z8.s, z0.b, z4.b
    KAI_ASM_INST(0x4585980b)  // usmmla z11.s, z0.b, z5.b
    KAI_ASM_INST(0x4584982e)  // usmmla z14.s, z1.b, z4.b
    KAI_ASM_INST(0x45859831)  // usmmla z17.s, z1.b, z5.b
    ld1b { z6.b }, p0/Z, [x3]
    KAI_ASM_INST(0x45849854)  // usmmla z20.s, z2.b, z4.b
    KAI_ASM_INST(0x45859857)  // usmmla z23.s, z2.b, z5.b
    ld1b { z7.b }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x4584987a)  // usmmla z26.s, z3.b, z4.b
    KAI_ASM_INST(0x4585987d)  // usmmla z29.s, z3.b, z5.b
    ld1b { z4.b }, p0/Z, [x3, #2, MUL VL]
    ld1b { z5.b }, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x45869809)  // usmmla z9.s, z0.b, z6.b
    sub x1, x1, #0x2
    KAI_ASM_INST(0x4587980c)  // usmmla z12.s, z0.b, z7.b
    KAI_ASM_INST(0x4586982f)  // usmmla z15.s, z1.b, z6.b
    cmp x1, #0x2
    KAI_ASM_INST(0x45879832)  // usmmla z18.s, z1.b, z7.b
    KAI_ASM_INST(0x45869855)  // usmmla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45879858)  // usmmla z24.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987b)  // usmmla z27.s, z3.b, z6.b
    ld1b { z6.b }, p0/Z, [x3, #4, MUL VL]
    KAI_ASM_INST(0x4587987e)  // usmmla z30.s, z3.b, z7.b
    KAI_ASM_INST(0x4584980a)  // usmmla z10.s, z0.b, z4.b
    ld1b { z7.b }, p0/Z, [x3, #5, MUL VL]
    KAI_ASM_INST(0x4585980d)  // usmmla z13.s, z0.b, z5.b
    KAI_ASM_INST(0x45849830)  // usmmla z16.s, z1.b, z4.b
    ld1rqb { z0.b }, p0/Z, [x6, #16]
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45849856)  // usmmla z22.s, z2.b, z4.b
    ld1rqb { z1.b }, p0/Z, [x6, #32]
    KAI_ASM_INST(0x45859859)  // usmmla z25.s, z2.b, z5.b
    KAI_ASM_INST(0x4584987c)  // usmmla z28.s, z3.b, z4.b
    ld1rqb { z2.b }, p0/Z, [x6, #48]
    KAI_ASM_INST(0x4585987f)  // usmmla z31.s, z3.b, z5.b
    ld1rqb { z3.b }, p0/Z, [x6, #64]
    ld1b { z4.b }, p0/Z, [x3, #6, MUL VL]
    ld1b { z5.b }, p0/Z, [x3, #7, MUL VL]
    addvl x3, x3, #16
    KAI_ASM_INST(0x45869808)  // usmmla z8.s, z0.b, z6.b
    KAI_ASM_INST(0x4587980b)  // usmmla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x4586982e)  // usmmla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45879831)  // usmmla z17.s, z1.b, z7.b
    KAI_ASM_INST(0x45869854)  // usmmla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45879857)  // usmmla z23.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987a)  // usmmla z26.s, z3.b, z6.b
    ld1b { z6.b }, p0/Z, [x3, #-8, MUL VL]
    KAI_ASM_INST(0x4587987d)  // usmmla z29.s, z3.b, z7.b
    ld1b { z7.b }, p0/Z, [x3, #-7, MUL VL]
    KAI_ASM_INST(0x45849809)  // usmmla z9.s, z0.b, z4.b
    KAI_ASM_INST(0x4585980c)  // usmmla z12.s, z0.b, z5.b
    KAI_ASM_INST(0x4584982f)  // usmmla z15.s, z1.b, z4.b
    KAI_ASM_INST(0x45859832)  // usmmla z18.s, z1.b, z5.b
    KAI_ASM_INST(0x45849855)  // usmmla z21.s, z2.b, z4.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    KAI_ASM_INST(0x4584987b)  // usmmla z27.s, z3.b, z4.b
    ld1b { z4.b }, p0/Z, [x3, #-6, MUL VL]
    KAI_ASM_INST(0x4585987e)  // usmmla z30.s, z3.b, z5.b
    KAI_ASM_INST(0x4586980a)  // usmmla z10.s, z0.b, z6.b
    ld1b { z5.b }, p0/Z, [x3, #-5, MUL VL]
    KAI_ASM_INST(0x4587980d)  // usmmla z13.s, z0.b, z7.b
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    ld1rqb { z0.b }, p0/Z, [x6, #80]
    KAI_ASM_INST(0x45879833)  // usmmla z19.s, z1.b, z7.b
    KAI_ASM_INST(0x45869856)  // usmmla z22.s, z2.b, z6.b
    ld1rqb { z1.b }, p0/Z, [x6, #96]
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987c)  // usmmla z28.s, z3.b, z6.b
    ld1rqb { z2.b }, p0/Z, [x6, #112]
    KAI_ASM_INST(0x4587987f)  // usmmla z31.s, z3.b, z7.b
    add x6, x6, #0x80
    addvl x3, x3, #-4
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    ld1rqb { z3.b }, p0/Z, [x6]
    KAI_ASM_INST(0x45849808)  // usmmla z8.s, z0.b, z4.b
    KAI_ASM_INST(0x4585980b)  // usmmla z11.s, z0.b, z5.b
    KAI_ASM_INST(0x4584982e)  // usmmla z14.s, z1.b, z4.b
    KAI_ASM_INST(0x45859831)  // usmmla z17.s, z1.b, z5.b
    ld1b { z6.b }, p0/Z, [x3]
    KAI_ASM_INST(0x45849854)  // usmmla z20.s, z2.b, z4.b
    KAI_ASM_INST(0x45859857)  // usmmla z23.s, z2.b, z5.b
    ld1b { z7.b }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x4584987a)  // usmmla z26.s, z3.b, z4.b
    KAI_ASM_INST(0x4585987d)  // usmmla z29.s, z3.b, z5.b
    ld1b { z4.b }, p0/Z, [x3, #2, MUL VL]
    ld1b { z5.b }, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x45869809)  // usmmla z9.s, z0.b, z6.b
    add x6, x6, #0x10
    KAI_ASM_INST(0x4587980c)  // usmmla z12.s, z0.b, z7.b
    KAI_ASM_INST(0x4586982f)  // usmmla z15.s, z1.b, z6.b
    addvl x3, x3, #4
    KAI_ASM_INST(0x45879832)  // usmmla z18.s, z1.b, z7.b
    KAI_ASM_INST(0x45869855)  // usmmla z21.s, z2.b, z6.b
    KAI_ASM_INST(0x45879858)  // usmmla z24.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987b)  // usmmla z27.s, z3.b, z6.b
    KAI_ASM_INST(0x4587987e)  // usmmla z30.s, z3.b, z7.b
    KAI_ASM_INST(0x4584980a)  // usmmla z10.s, z0.b, z4.b
    KAI_ASM_INST(0x4585980d)  // usmmla z13.s, z0.b, z5.b
    KAI_ASM_INST(0x45849830)  // usmmla z16.s, z1.b, z4.b
    KAI_ASM_INST(0x45859833)  // usmmla z19.s, z1.b, z5.b
    KAI_ASM_INST(0x45849856)  // usmmla z22.s, z2.b, z4.b
    KAI_ASM_INST(0x45859859)  // usmmla z25.s, z2.b, z5.b
    KAI_ASM_INST(0x4584987c)  // usmmla z28.s, z3.b, z4.b
    KAI_ASM_INST(0x4585987f)  // usmmla z31.s, z3.b, z5.b
    cbz x1, label_5
    ld1b { z6.b }, p0/Z, [x3]
    ld1rqb { z0.b }, p0/Z, [x6]
    ld1rqb { z1.b }, p0/Z, [x6, #16]
    ld1b { z7.b }, p0/Z, [x3, #1, MUL VL]
    ld1rqb { z2.b }, p0/Z, [x6, #32]
    ld1rqb { z3.b }, p0/Z, [x6, #48]
    add x6, x6, #0x40
    KAI_ASM_INST(0x45869808)  // usmmla z8.s, z0.b, z6.b
    ld1b { z4.b }, p0/Z, [x3, #2, MUL VL]
    ld1b { z5.b }, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x4587980b)  // usmmla z11.s, z0.b, z7.b
    KAI_ASM_INST(0x4586982e)  // usmmla z14.s, z1.b, z6.b
    KAI_ASM_INST(0x45879831)  // usmmla z17.s, z1.b, z7.b
    KAI_ASM_INST(0x45869854)  // usmmla z20.s, z2.b, z6.b
    KAI_ASM_INST(0x45879857)  // usmmla z23.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987a)  // usmmla z26.s, z3.b, z6.b
    ld1b { z6.b }, p0/Z, [x3, #4, MUL VL]
    KAI_ASM_INST(0x4587987d)  // usmmla z29.s, z3.b, z7.b
    ld1b { z7.b }, p0/Z, [x3, #5, MUL VL]
    KAI_ASM_INST(0x45849809)  // usmmla z9.s, z0.b, z4.b
    KAI_ASM_INST(0x4585980c)  // usmmla z12.s, z0.b, z5.b
    KAI_ASM_INST(0x4584982f)  // usmmla z15.s, z1.b, z4.b
    addvl x3, x3, #6
    KAI_ASM_INST(0x45859832)  // usmmla z18.s, z1.b, z5.b
    KAI_ASM_INST(0x45849855)  // usmmla z21.s, z2.b, z4.b
    KAI_ASM_INST(0x45859858)  // usmmla z24.s, z2.b, z5.b
    KAI_ASM_INST(0x4584987b)  // usmmla z27.s, z3.b, z4.b
    KAI_ASM_INST(0x4585987e)  // usmmla z30.s, z3.b, z5.b
    KAI_ASM_INST(0x4586980a)  // usmmla z10.s, z0.b, z6.b
    KAI_ASM_INST(0x4587980d)  // usmmla z13.s, z0.b, z7.b
    KAI_ASM_INST(0x45869830)  // usmmla z16.s, z1.b, z6.b
    KAI_ASM_INST(0x45879833)  // usmmla z19.s, z1.b, z7.b
    KAI_ASM_INST(0x45869856)  // usmmla z22.s, z2.b, z6.b
    KAI_ASM_INST(0x45879859)  // usmmla z25.s, z2.b, z7.b
    KAI_ASM_INST(0x4586987c)  // usmmla z28.s, z3.b, z6.b
    KAI_ASM_INST(0x4587987f)  // usmmla z31.s, z3.b, z7.b
KAI_ASM_LABEL(label_5)  // multiply loop done
    decw x4, ALL, MUL #3
    uzp1 z4.d, z8.d, z11.d
    uzp2 z8.d, z8.d, z11.d
    uzp1 z11.d, z9.d, z12.d
    uzp2 z9.d, z9.d, z12.d
    uzp1 z12.d, z10.d, z13.d
    uzp2 z10.d, z10.d, z13.d
    st1w { z4.s }, p0, [x5]
    uzp1 z13.d, z14.d, z17.d
    uzp2 z14.d, z14.d, z17.d
    st1w { z11.s }, p0, [x5, #1, MUL VL]
    uzp1 z17.d, z15.d, z18.d
    cmp x4, XZR
    st1w { z12.s }, p0, [x5, #2, MUL VL]
    uzp2 z15.d, z15.d, z18.d
    uzp1 z18.d, z16.d, z19.d
    st1w { z8.s }, p0, [x5, #3, MUL VL]
    uzp2 z16.d, z16.d, z19.d
    uzp1 z19.d, z20.d, z23.d
    st1w { z9.s }, p0, [x5, #4, MUL VL]
    uzp2 z20.d, z20.d, z23.d
    uzp1 z23.d, z21.d, z24.d
    st1w { z10.s }, p0, [x5, #5, MUL VL]
    uzp2 z21.d, z21.d, z24.d
    uzp1 z24.d, z22.d, z25.d
    st1w { z13.s }, p0, [x5, #6, MUL VL]
    uzp2 z22.d, z22.d, z25.d
    uzp1 z25.d, z26.d, z29.d
    st1w { z17.s }, p0, [x5, #7, MUL VL]
    addvl x5, x5, #16
    uzp2 z26.d, z26.d, z29.d
    uzp1 z29.d, z27.d, z30.d
    uzp2 z27.d, z27.d, z30.d
    uzp1 z30.d, z28.d, z31.d
    uzp2 z28.d, z28.d, z31.d
    st1w { z18.s }, p0, [x5, #-8, MUL VL]
    st1w { z14.s }, p0, [x5, #-7, MUL VL]
    st1w { z15.s }, p0, [x5, #-6, MUL VL]
    st1w { z16.s }, p0, [x5, #-5, MUL VL]
    st1w { z19.s }, p0, [x5, #-4, MUL VL]
    st1w { z23.s }, p0, [x5, #-3, MUL VL]
    st1w { z24.s }, p0, [x5, #-2, MUL VL]
    st1w { z20.s }, p0, [x5, #-1, MUL VL]
    st1w { z21.s }, p0, [x5]
    st1w { z22.s }, p0, [x5, #1, MUL VL]
    st1w { z25.s }, p0, [x5, #2, MUL VL]
    st1w { z29.s }, p0, [x5, #3, MUL VL]
    st1w { z30.s }, p0, [x5, #4, MUL VL]
    st1w { z26.s }, p0, [x5, #5, MUL VL]
    st1w { z27.s }, p0, [x5, #6, MUL VL]
    st1w { z28.s }, p0, [x5, #7, MUL VL]
    addvl x5, x5, #8
    bgt label_2
    subs x7, x7, #0x8
    bgt label_1
KAI_ASM_LABEL(label_6)  // Exit
    ldp d10, d11, [sp, 16]
    ldp d12, d13, [sp, 32]
    ldp d14, d15, [sp, 48]
    ldp d8, d9, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_interleaved_u8s8s32_mmla_8x3VL)

    KAI_ASM_END
