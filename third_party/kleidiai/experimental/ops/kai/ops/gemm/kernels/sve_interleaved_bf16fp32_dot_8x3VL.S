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

    KAI_ASM_CODE(sve_interleaved_bf16fp32_dot_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_interleaved_bf16fp32_dot_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_interleaved_bf16fp32_dot_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_interleaved_bf16fp32_dot_8x3VL)
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
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    ld1h { z4.h }, p0/Z, [x3]
    mov z11.b, #0
    mov z12.b, #0
    ld1h { z5.h }, p0/Z, [x3, #1, MUL VL]
    add x1, x1, #0x1
    mov z13.b, #0
    mov z14.b, #0
    lsr x1, x1, #0x1
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    ld1rqh { z0.h }, p0/Z, [x6]
    mov z19.b, #0
    mov z20.b, #0
    ld1rqh { z1.h }, p0/Z, [x6, #16]
    sub x1, x1, #0x1
    mov z21.b, #0
    mov z22.b, #0
    cmp x1, #0x2
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    ld1h { z6.h }, p0/Z, [x3, #2, MUL VL]
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    KAI_ASM_INST(0x64604088)  // bfdot z8.s, z4.h, z0.h[0]
    KAI_ASM_INST(0x6468408b)  // bfdot z11.s, z4.h, z0.h[1]
    ld1rqh { z2.h }, p0/Z, [x6, #32]
    KAI_ASM_INST(0x6470408e)  // bfdot z14.s, z4.h, z0.h[2]
    KAI_ASM_INST(0x64784091)  // bfdot z17.s, z4.h, z0.h[3]
    ld1rqh { z3.h }, p0/Z, [x6, #48]
    KAI_ASM_INST(0x64614094)  // bfdot z20.s, z4.h, z1.h[0]
    KAI_ASM_INST(0x64694097)  // bfdot z23.s, z4.h, z1.h[1]
    sub x1, x1, #0x2
    KAI_ASM_INST(0x6471409a)  // bfdot z26.s, z4.h, z1.h[2]
    KAI_ASM_INST(0x6479409d)  // bfdot z29.s, z4.h, z1.h[3]
    ld1h { z4.h }, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x646040a9)  // bfdot z9.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x646840ac)  // bfdot z12.s, z5.h, z0.h[1]
    cmp x1, #0x2
    KAI_ASM_INST(0x647040af)  // bfdot z15.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x647840b2)  // bfdot z18.s, z5.h, z0.h[3]
    add x6, x6, #0x40
    KAI_ASM_INST(0x646140b5)  // bfdot z21.s, z5.h, z1.h[0]
    KAI_ASM_INST(0x646940b8)  // bfdot z24.s, z5.h, z1.h[1]
    KAI_ASM_INST(0x647140bb)  // bfdot z27.s, z5.h, z1.h[2]
    KAI_ASM_INST(0x647940be)  // bfdot z30.s, z5.h, z1.h[3]
    ld1h { z5.h }, p0/Z, [x3, #4, MUL VL]
    KAI_ASM_INST(0x646040ca)  // bfdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x646840cd)  // bfdot z13.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x647040d0)  // bfdot z16.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x647840d3)  // bfdot z19.s, z6.h, z0.h[3]
    ld1rqh { z0.h }, p0/Z, [x6]
    KAI_ASM_INST(0x646140d6)  // bfdot z22.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x646940d9)  // bfdot z25.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x647140dc)  // bfdot z28.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x647940df)  // bfdot z31.s, z6.h, z1.h[3]
    ld1h { z6.h }, p0/Z, [x3, #5, MUL VL]
    addvl x3, x3, #6
    KAI_ASM_INST(0x64624088)  // bfdot z8.s, z4.h, z2.h[0]
    KAI_ASM_INST(0x646a408b)  // bfdot z11.s, z4.h, z2.h[1]
    ld1rqh { z1.h }, p0/Z, [x6, #16]
    KAI_ASM_INST(0x6472408e)  // bfdot z14.s, z4.h, z2.h[2]
    KAI_ASM_INST(0x647a4091)  // bfdot z17.s, z4.h, z2.h[3]
    KAI_ASM_INST(0x64634094)  // bfdot z20.s, z4.h, z3.h[0]
    KAI_ASM_INST(0x646b4097)  // bfdot z23.s, z4.h, z3.h[1]
    KAI_ASM_INST(0x6473409a)  // bfdot z26.s, z4.h, z3.h[2]
    KAI_ASM_INST(0x647b409d)  // bfdot z29.s, z4.h, z3.h[3]
    ld1h { z4.h }, p0/Z, [x3]
    KAI_ASM_INST(0x646240a9)  // bfdot z9.s, z5.h, z2.h[0]
    KAI_ASM_INST(0x646a40ac)  // bfdot z12.s, z5.h, z2.h[1]
    KAI_ASM_INST(0x647240af)  // bfdot z15.s, z5.h, z2.h[2]
    KAI_ASM_INST(0x647a40b2)  // bfdot z18.s, z5.h, z2.h[3]
    KAI_ASM_INST(0x646340b5)  // bfdot z21.s, z5.h, z3.h[0]
    KAI_ASM_INST(0x646b40b8)  // bfdot z24.s, z5.h, z3.h[1]
    KAI_ASM_INST(0x647340bb)  // bfdot z27.s, z5.h, z3.h[2]
    KAI_ASM_INST(0x647b40be)  // bfdot z30.s, z5.h, z3.h[3]
    ld1h { z5.h }, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x646240ca)  // bfdot z10.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x646a40cd)  // bfdot z13.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x647240d0)  // bfdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x647a40d3)  // bfdot z19.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x646340d6)  // bfdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x646b40d9)  // bfdot z25.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x647340dc)  // bfdot z28.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x647b40df)  // bfdot z31.s, z6.h, z3.h[3]
    ld1h { z6.h }, p0/Z, [x3, #2, MUL VL]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x20
    addvl x3, x3, #3
    KAI_ASM_INST(0x64604088)  // bfdot z8.s, z4.h, z0.h[0]
    KAI_ASM_INST(0x6468408b)  // bfdot z11.s, z4.h, z0.h[1]
    KAI_ASM_INST(0x6470408e)  // bfdot z14.s, z4.h, z0.h[2]
    KAI_ASM_INST(0x64784091)  // bfdot z17.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x64614094)  // bfdot z20.s, z4.h, z1.h[0]
    KAI_ASM_INST(0x64694097)  // bfdot z23.s, z4.h, z1.h[1]
    KAI_ASM_INST(0x6471409a)  // bfdot z26.s, z4.h, z1.h[2]
    KAI_ASM_INST(0x6479409d)  // bfdot z29.s, z4.h, z1.h[3]
    KAI_ASM_INST(0x646040a9)  // bfdot z9.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x646840ac)  // bfdot z12.s, z5.h, z0.h[1]
    KAI_ASM_INST(0x647040af)  // bfdot z15.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x647840b2)  // bfdot z18.s, z5.h, z0.h[3]
    KAI_ASM_INST(0x646140b5)  // bfdot z21.s, z5.h, z1.h[0]
    KAI_ASM_INST(0x646940b8)  // bfdot z24.s, z5.h, z1.h[1]
    KAI_ASM_INST(0x647140bb)  // bfdot z27.s, z5.h, z1.h[2]
    KAI_ASM_INST(0x647940be)  // bfdot z30.s, z5.h, z1.h[3]
    KAI_ASM_INST(0x646040ca)  // bfdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x646840cd)  // bfdot z13.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x647040d0)  // bfdot z16.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x647840d3)  // bfdot z19.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x646140d6)  // bfdot z22.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x646940d9)  // bfdot z25.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x647140dc)  // bfdot z28.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x647940df)  // bfdot z31.s, z6.h, z1.h[3]
    cbz x1, label_5
    ld1rqh { z0.h }, p0/Z, [x6]
    ld1rqh { z1.h }, p0/Z, [x6, #16]
    add x6, x6, #0x20
    ld1h { z7.h }, p0/Z, [x3]
    ld1h { z4.h }, p0/Z, [x3, #1, MUL VL]
    ld1h { z5.h }, p0/Z, [x3, #2, MUL VL]
    addvl x3, x3, #3
    KAI_ASM_INST(0x646040e8)  // bfdot z8.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x646840eb)  // bfdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x647040ee)  // bfdot z14.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x647840f1)  // bfdot z17.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x646140f4)  // bfdot z20.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x646940f7)  // bfdot z23.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x647140fa)  // bfdot z26.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x647940fd)  // bfdot z29.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64604089)  // bfdot z9.s, z4.h, z0.h[0]
    KAI_ASM_INST(0x6468408c)  // bfdot z12.s, z4.h, z0.h[1]
    KAI_ASM_INST(0x6470408f)  // bfdot z15.s, z4.h, z0.h[2]
    KAI_ASM_INST(0x64784092)  // bfdot z18.s, z4.h, z0.h[3]
    KAI_ASM_INST(0x64614095)  // bfdot z21.s, z4.h, z1.h[0]
    KAI_ASM_INST(0x64694098)  // bfdot z24.s, z4.h, z1.h[1]
    KAI_ASM_INST(0x6471409b)  // bfdot z27.s, z4.h, z1.h[2]
    KAI_ASM_INST(0x6479409e)  // bfdot z30.s, z4.h, z1.h[3]
    KAI_ASM_INST(0x646040aa)  // bfdot z10.s, z5.h, z0.h[0]
    KAI_ASM_INST(0x646840ad)  // bfdot z13.s, z5.h, z0.h[1]
    KAI_ASM_INST(0x647040b0)  // bfdot z16.s, z5.h, z0.h[2]
    KAI_ASM_INST(0x647840b3)  // bfdot z19.s, z5.h, z0.h[3]
    KAI_ASM_INST(0x646140b6)  // bfdot z22.s, z5.h, z1.h[0]
    KAI_ASM_INST(0x646940b9)  // bfdot z25.s, z5.h, z1.h[1]
    KAI_ASM_INST(0x647140bc)  // bfdot z28.s, z5.h, z1.h[2]
    KAI_ASM_INST(0x647940bf)  // bfdot z31.s, z5.h, z1.h[3]
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_interleaved_bf16fp32_dot_8x3VL)

    KAI_ASM_END
