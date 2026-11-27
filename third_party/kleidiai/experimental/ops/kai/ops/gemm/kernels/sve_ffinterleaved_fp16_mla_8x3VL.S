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

    KAI_ASM_CODE(sve_ffinterleaved_fp16_mla_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffinterleaved_fp16_mla_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffinterleaved_fp16_mla_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffinterleaved_fp16_mla_8x3VL)
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
    cnth x1, ALL, MUL #2
    mov x20, x6
    add x3, x5, x4
    cmp x7, x1
    add x2, x3, x4
    add x4, x2, x4
    bgt label_3
    dech x1
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
    ld1rqh { z0.h }, p0/Z, [x20]
    mov z12.b, #0
    mov z13.b, #0
    ld1h { z2.h }, p0/Z, [x5]
    sub x1, x1, #0x1
    mov z14.b, #0
    mov z15.b, #0
    cmp x1, #0x2
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    ld1h { z3.h }, p0/Z, [x3]
    mov z20.b, #0
    mov z21.b, #0
    ld1h { z4.h }, p0/Z, [x2]
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
    blt label_5
KAI_ASM_LABEL(label_4)  // main loop head
    fmla z8.h, z2.h, z0.h[0]
    fmla z11.h, z2.h, z0.h[1]
    ld1rqh { z1.h }, p0/Z, [x20, #16]
    fmla z14.h, z2.h, z0.h[2]
    fmla z17.h, z2.h, z0.h[3]
    ld1h { z5.h }, p0/Z, [x5, #1, MUL VL]
    fmla z20.h, z2.h, z0.h[4]
    fmla z23.h, z2.h, z0.h[5]
    ld1h { z6.h }, p0/Z, [x3, #1, MUL VL]
    fmla z26.h, z2.h, z0.h[6]
    fmla z29.h, z2.h, z0.h[7]
    ld1h { z7.h }, p0/Z, [x2, #1, MUL VL]
    fmla z9.h, z3.h, z0.h[0]
    fmla z12.h, z3.h, z0.h[1]
    addvl x5, x5, #2
    fmla z15.h, z3.h, z0.h[2]
    fmla z18.h, z3.h, z0.h[3]
    addvl x3, x3, #2
    fmla z21.h, z3.h, z0.h[4]
    fmla z24.h, z3.h, z0.h[5]
    addvl x2, x2, #2
    fmla z27.h, z3.h, z0.h[6]
    fmla z30.h, z3.h, z0.h[7]
    sub x1, x1, #0x2
    fmla z10.h, z4.h, z0.h[0]
    fmla z13.h, z4.h, z0.h[1]
    cmp x1, #0x2
    fmla z16.h, z4.h, z0.h[2]
    fmla z19.h, z4.h, z0.h[3]
    add x20, x20, #0x20
    fmla z22.h, z4.h, z0.h[4]
    fmla z25.h, z4.h, z0.h[5]
    ld1h { z2.h }, p0/Z, [x5]
    fmla z28.h, z4.h, z0.h[6]
    fmla z31.h, z4.h, z0.h[7]
    ld1rqh { z0.h }, p0/Z, [x20]
    fmla z8.h, z5.h, z1.h[0]
    fmla z11.h, z5.h, z1.h[1]
    ld1h { z3.h }, p0/Z, [x3]
    fmla z14.h, z5.h, z1.h[2]
    fmla z17.h, z5.h, z1.h[3]
    ld1h { z4.h }, p0/Z, [x2]
    fmla z20.h, z5.h, z1.h[4]
    fmla z23.h, z5.h, z1.h[5]
    fmla z26.h, z5.h, z1.h[6]
    fmla z29.h, z5.h, z1.h[7]
    fmla z9.h, z6.h, z1.h[0]
    fmla z12.h, z6.h, z1.h[1]
    fmla z15.h, z6.h, z1.h[2]
    fmla z18.h, z6.h, z1.h[3]
    fmla z21.h, z6.h, z1.h[4]
    fmla z24.h, z6.h, z1.h[5]
    fmla z27.h, z6.h, z1.h[6]
    fmla z30.h, z6.h, z1.h[7]
    fmla z10.h, z7.h, z1.h[0]
    fmla z13.h, z7.h, z1.h[1]
    fmla z16.h, z7.h, z1.h[2]
    fmla z19.h, z7.h, z1.h[3]
    fmla z22.h, z7.h, z1.h[4]
    fmla z25.h, z7.h, z1.h[5]
    fmla z28.h, z7.h, z1.h[6]
    fmla z31.h, z7.h, z1.h[7]
    bge label_4
KAI_ASM_LABEL(label_5)  // main loop skip
    add x20, x20, #0x10
    addvl x5, x5, #1
    fmla z8.h, z2.h, z0.h[0]
    addvl x3, x3, #1
    addvl x2, x2, #1
    fmla z11.h, z2.h, z0.h[1]
    fmla z14.h, z2.h, z0.h[2]
    fmla z17.h, z2.h, z0.h[3]
    fmla z20.h, z2.h, z0.h[4]
    fmla z23.h, z2.h, z0.h[5]
    fmla z26.h, z2.h, z0.h[6]
    fmla z29.h, z2.h, z0.h[7]
    fmla z9.h, z3.h, z0.h[0]
    fmla z12.h, z3.h, z0.h[1]
    fmla z15.h, z3.h, z0.h[2]
    fmla z18.h, z3.h, z0.h[3]
    fmla z21.h, z3.h, z0.h[4]
    fmla z24.h, z3.h, z0.h[5]
    fmla z27.h, z3.h, z0.h[6]
    fmla z30.h, z3.h, z0.h[7]
    fmla z10.h, z4.h, z0.h[0]
    fmla z13.h, z4.h, z0.h[1]
    fmla z16.h, z4.h, z0.h[2]
    fmla z19.h, z4.h, z0.h[3]
    fmla z22.h, z4.h, z0.h[4]
    fmla z25.h, z4.h, z0.h[5]
    fmla z28.h, z4.h, z0.h[6]
    fmla z31.h, z4.h, z0.h[7]
    cbz x1, label_6
    ld1rqh { z0.h }, p0/Z, [x20]
    ld1h { z5.h }, p0/Z, [x5]
    add x20, x20, #0x10
    ld1h { z6.h }, p0/Z, [x3]
    ld1h { z7.h }, p0/Z, [x2]
    fmla z8.h, z5.h, z0.h[0]
    fmla z11.h, z5.h, z0.h[1]
    fmla z14.h, z5.h, z0.h[2]
    fmla z17.h, z5.h, z0.h[3]
    fmla z20.h, z5.h, z0.h[4]
    fmla z23.h, z5.h, z0.h[5]
    fmla z26.h, z5.h, z0.h[6]
    fmla z29.h, z5.h, z0.h[7]
    fmla z9.h, z6.h, z0.h[0]
    fmla z12.h, z6.h, z0.h[1]
    fmla z15.h, z6.h, z0.h[2]
    fmla z18.h, z6.h, z0.h[3]
    fmla z21.h, z6.h, z0.h[4]
    fmla z24.h, z6.h, z0.h[5]
    fmla z27.h, z6.h, z0.h[6]
    fmla z30.h, z6.h, z0.h[7]
    fmla z10.h, z7.h, z0.h[0]
    fmla z13.h, z7.h, z0.h[1]
    fmla z16.h, z7.h, z0.h[2]
    fmla z19.h, z7.h, z0.h[3]
    fmla z22.h, z7.h, z0.h[4]
    fmla z25.h, z7.h, z0.h[5]
    fmla z28.h, z7.h, z0.h[6]
    fmla z31.h, z7.h, z0.h[7]
KAI_ASM_LABEL(label_6)  // multiply loop done
    dech x7, ALL, MUL #3
    st1h { z8.h }, p0, [x8]
    st1h { z9.h }, p0, [x8, #1, MUL VL]
    st1h { z10.h }, p0, [x8, #2, MUL VL]
    st1h { z11.h }, p0, [x8, #3, MUL VL]
    cmp x7, XZR
    st1h { z12.h }, p0, [x8, #4, MUL VL]
    st1h { z13.h }, p0, [x8, #5, MUL VL]
    st1h { z14.h }, p0, [x8, #6, MUL VL]
    st1h { z15.h }, p0, [x8, #7, MUL VL]
    addvl x8, x8, #16
    st1h { z16.h }, p0, [x8, #-8, MUL VL]
    st1h { z17.h }, p0, [x8, #-7, MUL VL]
    st1h { z18.h }, p0, [x8, #-6, MUL VL]
    st1h { z19.h }, p0, [x8, #-5, MUL VL]
    st1h { z20.h }, p0, [x8, #-4, MUL VL]
    st1h { z21.h }, p0, [x8, #-3, MUL VL]
    st1h { z22.h }, p0, [x8, #-2, MUL VL]
    st1h { z23.h }, p0, [x8, #-1, MUL VL]
    st1h { z24.h }, p0, [x8]
    st1h { z25.h }, p0, [x8, #1, MUL VL]
    st1h { z26.h }, p0, [x8, #2, MUL VL]
    st1h { z27.h }, p0, [x8, #3, MUL VL]
    st1h { z28.h }, p0, [x8, #4, MUL VL]
    st1h { z29.h }, p0, [x8, #5, MUL VL]
    st1h { z30.h }, p0, [x8, #6, MUL VL]
    st1h { z31.h }, p0, [x8, #7, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffinterleaved_fp16_mla_8x3VL)

    KAI_ASM_END
