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

    KAI_ASM_CODE(sve_interleaved_u8u32_dot_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_interleaved_u8u32_dot_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_interleaved_u8u32_dot_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_interleaved_u8u32_dot_8x3VL)
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
    add x1, x1, #0x3
    mov z13.s, #0
    mov z14.s, #0
    lsr x1, x1, #0x2
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
    ld1b { z6.b }, p0/Z, [x3, #2, MUL VL]
    mov z27.s, #0
    mov z28.s, #0
    mov z29.s, #0
    mov z30.s, #0
    mov z31.s, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    udot z8.s, z4.b, z0.b[0]
    udot z11.s, z4.b, z0.b[1]
    ld1rqb { z2.b }, p0/Z, [x6, #32]
    udot z14.s, z4.b, z0.b[2]
    udot z17.s, z4.b, z0.b[3]
    ld1rqb { z3.b }, p0/Z, [x6, #48]
    udot z20.s, z4.b, z1.b[0]
    udot z23.s, z4.b, z1.b[1]
    sub x1, x1, #0x2
    udot z26.s, z4.b, z1.b[2]
    udot z29.s, z4.b, z1.b[3]
    ld1b { z4.b }, p0/Z, [x3, #3, MUL VL]
    udot z9.s, z5.b, z0.b[0]
    udot z12.s, z5.b, z0.b[1]
    cmp x1, #0x2
    udot z15.s, z5.b, z0.b[2]
    udot z18.s, z5.b, z0.b[3]
    add x6, x6, #0x40
    udot z21.s, z5.b, z1.b[0]
    udot z24.s, z5.b, z1.b[1]
    udot z27.s, z5.b, z1.b[2]
    udot z30.s, z5.b, z1.b[3]
    ld1b { z5.b }, p0/Z, [x3, #4, MUL VL]
    udot z10.s, z6.b, z0.b[0]
    udot z13.s, z6.b, z0.b[1]
    udot z16.s, z6.b, z0.b[2]
    udot z19.s, z6.b, z0.b[3]
    ld1rqb { z0.b }, p0/Z, [x6]
    udot z22.s, z6.b, z1.b[0]
    udot z25.s, z6.b, z1.b[1]
    udot z28.s, z6.b, z1.b[2]
    udot z31.s, z6.b, z1.b[3]
    ld1b { z6.b }, p0/Z, [x3, #5, MUL VL]
    addvl x3, x3, #6
    udot z8.s, z4.b, z2.b[0]
    udot z11.s, z4.b, z2.b[1]
    ld1rqb { z1.b }, p0/Z, [x6, #16]
    udot z14.s, z4.b, z2.b[2]
    udot z17.s, z4.b, z2.b[3]
    udot z20.s, z4.b, z3.b[0]
    udot z23.s, z4.b, z3.b[1]
    udot z26.s, z4.b, z3.b[2]
    udot z29.s, z4.b, z3.b[3]
    ld1b { z4.b }, p0/Z, [x3]
    udot z9.s, z5.b, z2.b[0]
    udot z12.s, z5.b, z2.b[1]
    udot z15.s, z5.b, z2.b[2]
    udot z18.s, z5.b, z2.b[3]
    udot z21.s, z5.b, z3.b[0]
    udot z24.s, z5.b, z3.b[1]
    udot z27.s, z5.b, z3.b[2]
    udot z30.s, z5.b, z3.b[3]
    ld1b { z5.b }, p0/Z, [x3, #1, MUL VL]
    udot z10.s, z6.b, z2.b[0]
    udot z13.s, z6.b, z2.b[1]
    udot z16.s, z6.b, z2.b[2]
    udot z19.s, z6.b, z2.b[3]
    udot z22.s, z6.b, z3.b[0]
    udot z25.s, z6.b, z3.b[1]
    udot z28.s, z6.b, z3.b[2]
    udot z31.s, z6.b, z3.b[3]
    ld1b { z6.b }, p0/Z, [x3, #2, MUL VL]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x20
    addvl x3, x3, #3
    udot z8.s, z4.b, z0.b[0]
    udot z11.s, z4.b, z0.b[1]
    udot z14.s, z4.b, z0.b[2]
    udot z17.s, z4.b, z0.b[3]
    udot z20.s, z4.b, z1.b[0]
    udot z23.s, z4.b, z1.b[1]
    udot z26.s, z4.b, z1.b[2]
    udot z29.s, z4.b, z1.b[3]
    udot z9.s, z5.b, z0.b[0]
    udot z12.s, z5.b, z0.b[1]
    udot z15.s, z5.b, z0.b[2]
    udot z18.s, z5.b, z0.b[3]
    udot z21.s, z5.b, z1.b[0]
    udot z24.s, z5.b, z1.b[1]
    udot z27.s, z5.b, z1.b[2]
    udot z30.s, z5.b, z1.b[3]
    udot z10.s, z6.b, z0.b[0]
    udot z13.s, z6.b, z0.b[1]
    udot z16.s, z6.b, z0.b[2]
    udot z19.s, z6.b, z0.b[3]
    udot z22.s, z6.b, z1.b[0]
    udot z25.s, z6.b, z1.b[1]
    udot z28.s, z6.b, z1.b[2]
    udot z31.s, z6.b, z1.b[3]
    cbz x1, label_5
    ld1rqb { z0.b }, p0/Z, [x6]
    ld1rqb { z1.b }, p0/Z, [x6, #16]
    add x6, x6, #0x20
    ld1b { z7.b }, p0/Z, [x3]
    ld1b { z4.b }, p0/Z, [x3, #1, MUL VL]
    ld1b { z5.b }, p0/Z, [x3, #2, MUL VL]
    addvl x3, x3, #3
    udot z8.s, z7.b, z0.b[0]
    udot z11.s, z7.b, z0.b[1]
    udot z14.s, z7.b, z0.b[2]
    udot z17.s, z7.b, z0.b[3]
    udot z20.s, z7.b, z1.b[0]
    udot z23.s, z7.b, z1.b[1]
    udot z26.s, z7.b, z1.b[2]
    udot z29.s, z7.b, z1.b[3]
    udot z9.s, z4.b, z0.b[0]
    udot z12.s, z4.b, z0.b[1]
    udot z15.s, z4.b, z0.b[2]
    udot z18.s, z4.b, z0.b[3]
    udot z21.s, z4.b, z1.b[0]
    udot z24.s, z4.b, z1.b[1]
    udot z27.s, z4.b, z1.b[2]
    udot z30.s, z4.b, z1.b[3]
    udot z10.s, z5.b, z0.b[0]
    udot z13.s, z5.b, z0.b[1]
    udot z16.s, z5.b, z0.b[2]
    udot z19.s, z5.b, z0.b[3]
    udot z22.s, z5.b, z1.b[0]
    udot z25.s, z5.b, z1.b[1]
    udot z28.s, z5.b, z1.b[2]
    udot z31.s, z5.b, z1.b[3]
KAI_ASM_LABEL(label_5)  // multiply loop done
    decw x4, ALL, MUL #3
    st1w { z8.s }, p0, [x5]
    st1w { z9.s }, p0, [x5, #1, MUL VL]
    st1w { z10.s }, p0, [x5, #2, MUL VL]
    st1w { z11.s }, p0, [x5, #3, MUL VL]
    cmp x4, XZR
    st1w { z12.s }, p0, [x5, #4, MUL VL]
    st1w { z13.s }, p0, [x5, #5, MUL VL]
    st1w { z14.s }, p0, [x5, #6, MUL VL]
    st1w { z15.s }, p0, [x5, #7, MUL VL]
    addvl x5, x5, #16
    st1w { z16.s }, p0, [x5, #-8, MUL VL]
    st1w { z17.s }, p0, [x5, #-7, MUL VL]
    st1w { z18.s }, p0, [x5, #-6, MUL VL]
    st1w { z19.s }, p0, [x5, #-5, MUL VL]
    st1w { z20.s }, p0, [x5, #-4, MUL VL]
    st1w { z21.s }, p0, [x5, #-3, MUL VL]
    st1w { z22.s }, p0, [x5, #-2, MUL VL]
    st1w { z23.s }, p0, [x5, #-1, MUL VL]
    st1w { z24.s }, p0, [x5]
    st1w { z25.s }, p0, [x5, #1, MUL VL]
    st1w { z26.s }, p0, [x5, #2, MUL VL]
    st1w { z27.s }, p0, [x5, #3, MUL VL]
    st1w { z28.s }, p0, [x5, #4, MUL VL]
    st1w { z29.s }, p0, [x5, #5, MUL VL]
    st1w { z30.s }, p0, [x5, #6, MUL VL]
    st1w { z31.s }, p0, [x5, #7, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_interleaved_u8u32_dot_8x3VL)

    KAI_ASM_END
