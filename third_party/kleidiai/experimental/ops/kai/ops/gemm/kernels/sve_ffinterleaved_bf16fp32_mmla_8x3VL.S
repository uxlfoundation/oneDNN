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

    KAI_ASM_CODE(sve_ffinterleaved_bf16fp32_mmla_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffinterleaved_bf16fp32_mmla_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffinterleaved_bf16fp32_mmla_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffinterleaved_bf16fp32_mmla_8x3VL)
    stp x20, x21, [sp, -80]!
    stp d8, d9, [sp, 16]
    stp d10, d11, [sp, 32]
    stp d12, d13, [sp, 48]
    stp d14, d15, [sp, 64]
    ptrue p0.b
    ldr x21, [x0, #0]
    ldr x20, [x0, #0x20]
    ldr x8, [x0, #0x50]
    cbz x21, label_7
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x1, [x0, #0x30]
    ldr x7, [x0, #0x8]
    mov x6, x20
    mov x4, x1
KAI_ASM_LABEL(label_2)  // Width loop
    mov x5, x4
    ldr x4, [x0, #0x38]
    cntw x1, ALL, MUL #2
    mov x20, x6
    add x3, x5, x4
    cmp x7, x1
    add x2, x3, x4
    add x4, x2, x4
    bgt label_3
    decw x1
    mov x2, x5
    cmp x7, x1
    bgt label_3
    mov x3, x5
KAI_ASM_LABEL(label_3)  // B setup done
    ldr x1, [x0, #0x10]
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    ld1h { z4.h }, p0/Z, [x5]
    mov z12.b, #0
    mov z13.b, #0
    ld1rqh { z0.h }, p0/Z, [x20]
    add x1, x1, #0x3
    mov z14.b, #0
    mov z15.b, #0
    lsr x1, x1, #0x2
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    ld1rqh { z1.h }, p0/Z, [x20, #16]
    mov z20.b, #0
    mov z21.b, #0
    ld1h { z5.h }, p0/Z, [x5, #1, MUL VL]
    sub x1, x1, #0x1
    mov z22.b, #0
    mov z23.b, #0
    cmp x1, #0x2
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    ld1rqh { z2.h }, p0/Z, [x20, #32]
    mov z28.b, #0
    mov z29.b, #0
    addvl x5, x5, #2
    mov z30.b, #0
    mov z31.b, #0
    add x20, x20, #0x30
    blt label_5
KAI_ASM_LABEL(label_4)  // main loop head
    ld1rqh { z3.h }, p0/Z, [x20]
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40b)  // bfmmla z11.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e42e)  // bfmmla z14.s, z1.h, z4.h
    KAI_ASM_INST(0x6465e431)  // bfmmla z17.s, z1.h, z5.h
    ld1h { z6.h }, p0/Z, [x3]
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e457)  // bfmmla z23.s, z2.h, z5.h
    ld1h { z7.h }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x6464e47a)  // bfmmla z26.s, z3.h, z4.h
    KAI_ASM_INST(0x6465e47d)  // bfmmla z29.s, z3.h, z5.h
    ld1h { z4.h }, p0/Z, [x2]
    ld1h { z5.h }, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    sub x1, x1, #0x2
    KAI_ASM_INST(0x6467e40c)  // bfmmla z12.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    cmp x1, #0x2
    KAI_ASM_INST(0x6467e432)  // bfmmla z18.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e458)  // bfmmla z24.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47b)  // bfmmla z27.s, z3.h, z6.h
    ld1h { z6.h }, p0/Z, [x5]
    KAI_ASM_INST(0x6467e47e)  // bfmmla z30.s, z3.h, z7.h
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    ld1h { z7.h }, p0/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0x6465e40d)  // bfmmla z13.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e430)  // bfmmla z16.s, z1.h, z4.h
    ld1rqh { z0.h }, p0/Z, [x20, #16]
    KAI_ASM_INST(0x6465e433)  // bfmmla z19.s, z1.h, z5.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    ld1rqh { z1.h }, p0/Z, [x20, #32]
    KAI_ASM_INST(0x6465e459)  // bfmmla z25.s, z2.h, z5.h
    KAI_ASM_INST(0x6464e47c)  // bfmmla z28.s, z3.h, z4.h
    ld1rqh { z2.h }, p0/Z, [x20, #48]
    KAI_ASM_INST(0x6465e47f)  // bfmmla z31.s, z3.h, z5.h
    ld1rqh { z3.h }, p0/Z, [x20, #64]
    KAI_ASM_INST(0x6466e408)  // bfmmla z8.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    ld1h { z4.h }, p0/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0x6467e431)  // bfmmla z17.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    ld1h { z5.h }, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x6467e457)  // bfmmla z23.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47a)  // bfmmla z26.s, z3.h, z6.h
    ld1h { z6.h }, p0/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0x6467e47d)  // bfmmla z29.s, z3.h, z7.h
    ld1h { z7.h }, p0/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x6464e409)  // bfmmla z9.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40c)  // bfmmla z12.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e42f)  // bfmmla z15.s, z1.h, z4.h
    addvl x3, x3, #4
    KAI_ASM_INST(0x6465e432)  // bfmmla z18.s, z1.h, z5.h
    KAI_ASM_INST(0x6464e455)  // bfmmla z21.s, z2.h, z4.h
    addvl x2, x2, #4
    KAI_ASM_INST(0x6465e458)  // bfmmla z24.s, z2.h, z5.h
    KAI_ASM_INST(0x6464e47b)  // bfmmla z27.s, z3.h, z4.h
    ld1h { z4.h }, p0/Z, [x5, #2, MUL VL]
    KAI_ASM_INST(0x6465e47e)  // bfmmla z30.s, z3.h, z5.h
    KAI_ASM_INST(0x6466e40a)  // bfmmla z10.s, z0.h, z6.h
    ld1h { z5.h }, p0/Z, [x5, #3, MUL VL]
    KAI_ASM_INST(0x6467e40d)  // bfmmla z13.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e430)  // bfmmla z16.s, z1.h, z6.h
    ld1rqh { z0.h }, p0/Z, [x20, #80]
    KAI_ASM_INST(0x6467e433)  // bfmmla z19.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    ld1rqh { z1.h }, p0/Z, [x20, #96]
    KAI_ASM_INST(0x6467e459)  // bfmmla z25.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47c)  // bfmmla z28.s, z3.h, z6.h
    ld1rqh { z2.h }, p0/Z, [x20, #112]
    KAI_ASM_INST(0x6467e47f)  // bfmmla z31.s, z3.h, z7.h
    add x20, x20, #0x80
    addvl x5, x5, #4
    bge label_4
KAI_ASM_LABEL(label_5)  // main loop skip
    ld1rqh { z3.h }, p0/Z, [x20]
    KAI_ASM_INST(0x6464e408)  // bfmmla z8.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40b)  // bfmmla z11.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e42e)  // bfmmla z14.s, z1.h, z4.h
    KAI_ASM_INST(0x6465e431)  // bfmmla z17.s, z1.h, z5.h
    ld1h { z6.h }, p0/Z, [x3]
    KAI_ASM_INST(0x6464e454)  // bfmmla z20.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e457)  // bfmmla z23.s, z2.h, z5.h
    ld1h { z7.h }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x6464e47a)  // bfmmla z26.s, z3.h, z4.h
    KAI_ASM_INST(0x6465e47d)  // bfmmla z29.s, z3.h, z5.h
    ld1h { z4.h }, p0/Z, [x2]
    ld1h { z5.h }, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x6466e409)  // bfmmla z9.s, z0.h, z6.h
    add x20, x20, #0x10
    KAI_ASM_INST(0x6467e40c)  // bfmmla z12.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e42f)  // bfmmla z15.s, z1.h, z6.h
    addvl x3, x3, #2
    KAI_ASM_INST(0x6467e432)  // bfmmla z18.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e455)  // bfmmla z21.s, z2.h, z6.h
    addvl x2, x2, #2
    KAI_ASM_INST(0x6467e458)  // bfmmla z24.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47b)  // bfmmla z27.s, z3.h, z6.h
    KAI_ASM_INST(0x6467e47e)  // bfmmla z30.s, z3.h, z7.h
    KAI_ASM_INST(0x6464e40a)  // bfmmla z10.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40d)  // bfmmla z13.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e430)  // bfmmla z16.s, z1.h, z4.h
    KAI_ASM_INST(0x6465e433)  // bfmmla z19.s, z1.h, z5.h
    KAI_ASM_INST(0x6464e456)  // bfmmla z22.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e459)  // bfmmla z25.s, z2.h, z5.h
    KAI_ASM_INST(0x6464e47c)  // bfmmla z28.s, z3.h, z4.h
    KAI_ASM_INST(0x6465e47f)  // bfmmla z31.s, z3.h, z5.h
    cbz x1, label_6
    ld1h { z6.h }, p0/Z, [x5]
    ld1rqh { z0.h }, p0/Z, [x20]
    ld1rqh { z1.h }, p0/Z, [x20, #16]
    ld1h { z7.h }, p0/Z, [x5, #1, MUL VL]
    ld1rqh { z2.h }, p0/Z, [x20, #32]
    ld1rqh { z3.h }, p0/Z, [x20, #48]
    add x20, x20, #0x40
    KAI_ASM_INST(0x6466e408)  // bfmmla z8.s, z0.h, z6.h
    ld1h { z4.h }, p0/Z, [x3]
    ld1h { z5.h }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x6467e40b)  // bfmmla z11.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e42e)  // bfmmla z14.s, z1.h, z6.h
    KAI_ASM_INST(0x6467e431)  // bfmmla z17.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e454)  // bfmmla z20.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e457)  // bfmmla z23.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47a)  // bfmmla z26.s, z3.h, z6.h
    ld1h { z6.h }, p0/Z, [x2]
    KAI_ASM_INST(0x6467e47d)  // bfmmla z29.s, z3.h, z7.h
    ld1h { z7.h }, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x6464e409)  // bfmmla z9.s, z0.h, z4.h
    KAI_ASM_INST(0x6465e40c)  // bfmmla z12.s, z0.h, z5.h
    KAI_ASM_INST(0x6464e42f)  // bfmmla z15.s, z1.h, z4.h
    KAI_ASM_INST(0x6465e432)  // bfmmla z18.s, z1.h, z5.h
    KAI_ASM_INST(0x6464e455)  // bfmmla z21.s, z2.h, z4.h
    KAI_ASM_INST(0x6465e458)  // bfmmla z24.s, z2.h, z5.h
    KAI_ASM_INST(0x6464e47b)  // bfmmla z27.s, z3.h, z4.h
    KAI_ASM_INST(0x6465e47e)  // bfmmla z30.s, z3.h, z5.h
    KAI_ASM_INST(0x6466e40a)  // bfmmla z10.s, z0.h, z6.h
    KAI_ASM_INST(0x6467e40d)  // bfmmla z13.s, z0.h, z7.h
    KAI_ASM_INST(0x6466e430)  // bfmmla z16.s, z1.h, z6.h
    KAI_ASM_INST(0x6467e433)  // bfmmla z19.s, z1.h, z7.h
    KAI_ASM_INST(0x6466e456)  // bfmmla z22.s, z2.h, z6.h
    KAI_ASM_INST(0x6467e459)  // bfmmla z25.s, z2.h, z7.h
    KAI_ASM_INST(0x6466e47c)  // bfmmla z28.s, z3.h, z6.h
    KAI_ASM_INST(0x6467e47f)  // bfmmla z31.s, z3.h, z7.h
KAI_ASM_LABEL(label_6)  // multiply loop done
    decw x7, ALL, MUL #3
    uzp1 z4.d, z8.d, z11.d
    uzp2 z8.d, z8.d, z11.d
    uzp1 z11.d, z9.d, z12.d
    uzp2 z9.d, z9.d, z12.d
    uzp1 z12.d, z10.d, z13.d
    uzp2 z10.d, z10.d, z13.d
    st1w { z4.s }, p0, [x8]
    uzp1 z13.d, z14.d, z17.d
    uzp2 z14.d, z14.d, z17.d
    st1w { z11.s }, p0, [x8, #1, MUL VL]
    uzp1 z17.d, z15.d, z18.d
    cmp x7, XZR
    st1w { z12.s }, p0, [x8, #2, MUL VL]
    uzp2 z15.d, z15.d, z18.d
    uzp1 z18.d, z16.d, z19.d
    st1w { z8.s }, p0, [x8, #3, MUL VL]
    uzp2 z16.d, z16.d, z19.d
    uzp1 z19.d, z20.d, z23.d
    st1w { z9.s }, p0, [x8, #4, MUL VL]
    uzp2 z20.d, z20.d, z23.d
    uzp1 z23.d, z21.d, z24.d
    st1w { z10.s }, p0, [x8, #5, MUL VL]
    uzp2 z21.d, z21.d, z24.d
    uzp1 z24.d, z22.d, z25.d
    st1w { z13.s }, p0, [x8, #6, MUL VL]
    uzp2 z22.d, z22.d, z25.d
    uzp1 z25.d, z26.d, z29.d
    st1w { z17.s }, p0, [x8, #7, MUL VL]
    addvl x8, x8, #16
    uzp2 z26.d, z26.d, z29.d
    uzp1 z29.d, z27.d, z30.d
    uzp2 z27.d, z27.d, z30.d
    uzp1 z30.d, z28.d, z31.d
    uzp2 z28.d, z28.d, z31.d
    st1w { z18.s }, p0, [x8, #-8, MUL VL]
    st1w { z14.s }, p0, [x8, #-7, MUL VL]
    st1w { z15.s }, p0, [x8, #-6, MUL VL]
    st1w { z16.s }, p0, [x8, #-5, MUL VL]
    st1w { z19.s }, p0, [x8, #-4, MUL VL]
    st1w { z23.s }, p0, [x8, #-3, MUL VL]
    st1w { z24.s }, p0, [x8, #-2, MUL VL]
    st1w { z20.s }, p0, [x8, #-1, MUL VL]
    st1w { z21.s }, p0, [x8]
    st1w { z22.s }, p0, [x8, #1, MUL VL]
    st1w { z25.s }, p0, [x8, #2, MUL VL]
    st1w { z29.s }, p0, [x8, #3, MUL VL]
    st1w { z30.s }, p0, [x8, #4, MUL VL]
    st1w { z26.s }, p0, [x8, #5, MUL VL]
    st1w { z27.s }, p0, [x8, #6, MUL VL]
    st1w { z28.s }, p0, [x8, #7, MUL VL]
    addvl x8, x8, #8
    bgt label_2
    subs x21, x21, #0x8
    bgt label_1
KAI_ASM_LABEL(label_7)  // Exit
    ldp d8, d9, [sp, 16]
    ldp d10, d11, [sp, 32]
    ldp d12, d13, [sp, 48]
    ldp d14, d15, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffinterleaved_bf16fp32_mmla_8x3VL)

    KAI_ASM_END
