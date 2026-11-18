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

    KAI_ASM_CODE(sve_merge_fp16_fp16_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_merge_fp16_fp16_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_merge_fp16_fp16_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_merge_fp16_fp16_8x3VL)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    str d12, [sp, 104]
    ptrue p3.b
    mov x28, #0x8
    ldr x27, [x0, #0x58]
    ldr x3, [x0, #0x60]
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0ac4c)  // ld1rh { z12.h }, p3/Z, [x2]
    KAI_ASM_INST(0x84c0ac2b)  // ld1rh { z11.h }, p3/Z, [x1]
    mul x28, x27, x28
    cbnz x3, label_34
    ldr x26, [x0, #0]
    ldr x6, [x0, #0x50]
    ldr x5, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // Initial: Row loop
    cmp x26, #0x7
    bgt label_30
    beq label_26
    cmp x26, #0x5
    bgt label_22
    beq label_18
    cmp x26, #0x3
    bgt label_14
    beq label_10
    cmp x26, #0x1
    bgt label_6
    ldr x3, [x0, #0x8]
    mov x25, x6
    ldr x2, [x0, #0x40]
KAI_ASM_LABEL(label_3)  // Initial: Height 1: Block loop
    mov x1, #0
    whilelt p2.h, x1, x3
    inch x1
    whilelt p1.h, x1, x3
    inch x1
    whilelt p0.h, x1, x3
    cbnz x2, label_4
    mov z21.b, #0
    mov z20.b, #0
    mov z19.b, #0
    b label_5
KAI_ASM_LABEL(label_4)  // Initial: Height 1: Width 3: bias
    ld1h { z21.h }, p2/Z, [x2]
    ld1h { z20.h }, p1/Z, [x2, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x2, #2, MUL VL]
    incb x2, ALL, MUL #3
KAI_ASM_LABEL(label_5)  // Initial: Height 1: Width 3: init done
    ld1h { z18.h }, p2/Z, [x5]
    ld1h { z17.h }, p1/Z, [x5, #1, MUL VL]
    dech x3, ALL, MUL #3
    ld1h { z16.h }, p0/Z, [x5, #2, MUL VL]
    addvl x5, x5, #24
    fadd z18.h, z18.h, z21.h
    fadd z17.h, z17.h, z20.h
    cmp x3, XZR
    fadd z16.h, z16.h, z19.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    fmax z18.h, p3/M, z18.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    fmax z16.h, p3/M, z16.h, z11.h
    st1h { z18.h }, p2, [x25]
    st1h { z17.h }, p1, [x25, #1, MUL VL]
    st1h { z16.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    bgt label_3
    b label_52
KAI_ASM_LABEL(label_6)  // Initial: Height 2
    mov x25, x6
    ldr x3, [x0, #0x8]
    ldr x2, [x0, #0x40]
    add x24, x25, x27
KAI_ASM_LABEL(label_7)  // Initial: Height 2: Block loop
    mov x1, #0
    whilelt p2.h, x1, x3
    inch x1
    whilelt p1.h, x1, x3
    inch x1
    whilelt p0.h, x1, x3
    cbnz x2, label_8
    mov z24.b, #0
    mov z23.b, #0
    mov z22.b, #0
    b label_9
KAI_ASM_LABEL(label_8)  // Initial: Height 2: Width 3: bias
    ld1h { z24.h }, p2/Z, [x2]
    ld1h { z23.h }, p1/Z, [x2, #1, MUL VL]
    ld1h { z22.h }, p0/Z, [x2, #2, MUL VL]
    incb x2, ALL, MUL #3
KAI_ASM_LABEL(label_9)  // Initial: Height 2: Width 3: init done
    ld1h { z21.h }, p2/Z, [x5]
    ld1h { z20.h }, p1/Z, [x5, #1, MUL VL]
    dech x3, ALL, MUL #3
    ld1h { z19.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z18.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z17.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z16.h }, p0/Z, [x5, #5, MUL VL]
    addvl x5, x5, #24
    fadd z21.h, z21.h, z24.h
    fadd z20.h, z20.h, z23.h
    cmp x3, XZR
    fadd z19.h, z19.h, z22.h
    fadd z18.h, z18.h, z24.h
    fadd z17.h, z17.h, z23.h
    fadd z16.h, z16.h, z22.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    fmax z18.h, p3/M, z18.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    fmax z16.h, p3/M, z16.h, z11.h
    st1h { z21.h }, p2, [x25]
    st1h { z20.h }, p1, [x25, #1, MUL VL]
    st1h { z19.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1h { z18.h }, p2, [x24]
    st1h { z17.h }, p1, [x24, #1, MUL VL]
    st1h { z16.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    bgt label_7
    b label_52
KAI_ASM_LABEL(label_10)  // Initial: Height 3
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
KAI_ASM_LABEL(label_11)  // Initial: Height 3: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_12
    mov z27.b, #0
    mov z26.b, #0
    mov z25.b, #0
    b label_13
KAI_ASM_LABEL(label_12)  // Initial: Height 3: Width 3: bias
    ld1h { z27.h }, p2/Z, [x3]
    ld1h { z26.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z25.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_13)  // Initial: Height 3: Width 3: init done
    ld1h { z24.h }, p2/Z, [x5]
    ld1h { z23.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z22.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z21.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z20.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z19.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z18.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z17.h }, p1/Z, [x5, #7, MUL VL]
    fadd z24.h, z24.h, z27.h
    fadd z23.h, z23.h, z26.h
    ld1h { z16.h }, p0/Z, [x1, #-8, MUL VL]
    fadd z22.h, z22.h, z25.h
    fadd z21.h, z21.h, z27.h
    cmp x4, XZR
    fadd z20.h, z20.h, z26.h
    fadd z19.h, z19.h, z25.h
    addvl x5, x5, #24
    fadd z18.h, z18.h, z27.h
    fadd z17.h, z17.h, z26.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fadd z16.h, z16.h, z25.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z24.h }, p2, [x25]
    fmax z19.h, p3/M, z19.h, z11.h
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z23.h }, p1, [x25, #1, MUL VL]
    st1h { z22.h }, p0, [x25, #2, MUL VL]
    fmax z17.h, p3/M, z17.h, z11.h
    fmax z16.h, p3/M, z16.h, z11.h
    st1h { z21.h }, p2, [x24]
    incb x25, ALL, MUL #3
    st1h { z20.h }, p1, [x24, #1, MUL VL]
    st1h { z19.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1h { z18.h }, p2, [x23]
    st1h { z17.h }, p1, [x23, #1, MUL VL]
    st1h { z16.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    bgt label_11
    b label_52
KAI_ASM_LABEL(label_14)  // Initial: Height 4
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
KAI_ASM_LABEL(label_15)  // Initial: Height 4: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_16
    mov z30.b, #0
    mov z29.b, #0
    mov z28.b, #0
    b label_17
KAI_ASM_LABEL(label_16)  // Initial: Height 4: Width 3: bias
    ld1h { z30.h }, p2/Z, [x3]
    ld1h { z29.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z28.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_17)  // Initial: Height 4: Width 3: init done
    ld1h { z27.h }, p2/Z, [x5]
    ld1h { z26.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z25.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z24.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z23.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z22.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z21.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z20.h }, p1/Z, [x5, #7, MUL VL]
    fadd z27.h, z27.h, z30.h
    fadd z26.h, z26.h, z29.h
    ld1h { z19.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z18.h }, p2/Z, [x1, #-7, MUL VL]
    fadd z25.h, z25.h, z28.h
    fadd z24.h, z24.h, z30.h
    ld1h { z17.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z16.h }, p0/Z, [x1, #-5, MUL VL]
    fadd z23.h, z23.h, z29.h
    fadd z22.h, z22.h, z28.h
    fadd z21.h, z21.h, z30.h
    fadd z20.h, z20.h, z29.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fadd z19.h, z19.h, z28.h
    fadd z18.h, z18.h, z30.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fadd z17.h, z17.h, z29.h
    fadd z16.h, z16.h, z28.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    cmp x4, XZR
    fmax z27.h, p3/M, z27.h, z11.h
    fmax z26.h, p3/M, z26.h, z11.h
    fmax z25.h, p3/M, z25.h, z11.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    st1h { z27.h }, p2, [x25]
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    st1h { z26.h }, p1, [x25, #1, MUL VL]
    st1h { z25.h }, p0, [x25, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    st1h { z24.h }, p2, [x24]
    fmax z18.h, p3/M, z18.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z23.h }, p1, [x24, #1, MUL VL]
    fmax z16.h, p3/M, z16.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z22.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    addvl x5, x5, #24
    st1h { z21.h }, p2, [x23]
    st1h { z20.h }, p1, [x23, #1, MUL VL]
    st1h { z19.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1h { z18.h }, p2, [x22]
    st1h { z17.h }, p1, [x22, #1, MUL VL]
    st1h { z16.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    bgt label_15
    b label_52
KAI_ASM_LABEL(label_18)  // Initial: Height 5
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
KAI_ASM_LABEL(label_19)  // Initial: Height 5: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_20
    mov z1.b, #0
    mov z0.b, #0
    mov z31.b, #0
    b label_21
KAI_ASM_LABEL(label_20)  // Initial: Height 5: Width 3: bias
    ld1h { z1.h }, p2/Z, [x3]
    ld1h { z0.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z31.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_21)  // Initial: Height 5: Width 3: init done
    ld1h { z30.h }, p2/Z, [x5]
    ld1h { z29.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z28.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z27.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z26.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z25.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z24.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z23.h }, p1/Z, [x5, #7, MUL VL]
    fadd z30.h, z30.h, z1.h
    fadd z29.h, z29.h, z0.h
    ld1h { z22.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z21.h }, p2/Z, [x1, #-7, MUL VL]
    fadd z28.h, z28.h, z31.h
    fadd z27.h, z27.h, z1.h
    ld1h { z20.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z19.h }, p0/Z, [x1, #-5, MUL VL]
    fadd z26.h, z26.h, z0.h
    fadd z25.h, z25.h, z31.h
    ld1h { z18.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z17.h }, p1/Z, [x1, #-3, MUL VL]
    fadd z24.h, z24.h, z1.h
    fadd z23.h, z23.h, z0.h
    ld1h { z16.h }, p0/Z, [x1, #-2, MUL VL]
    fadd z22.h, z22.h, z31.h
    fadd z21.h, z21.h, z1.h
    fmin z30.h, p3/M, z30.h, z12.h
    fadd z20.h, z20.h, z0.h
    fadd z19.h, z19.h, z31.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fadd z18.h, z18.h, z1.h
    fadd z17.h, z17.h, z0.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fadd z16.h, z16.h, z31.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    cmp x4, XZR
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z27.h, p3/M, z27.h, z11.h
    fmax z26.h, p3/M, z26.h, z11.h
    st1h { z30.h }, p2, [x25]
    fmax z25.h, p3/M, z25.h, z11.h
    fmax z24.h, p3/M, z24.h, z11.h
    st1h { z29.h }, p1, [x25, #1, MUL VL]
    st1h { z28.h }, p0, [x25, #2, MUL VL]
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    st1h { z27.h }, p2, [x24]
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z26.h }, p1, [x24, #1, MUL VL]
    fmax z19.h, p3/M, z19.h, z11.h
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z25.h }, p0, [x24, #2, MUL VL]
    fmax z17.h, p3/M, z17.h, z11.h
    fmax z16.h, p3/M, z16.h, z11.h
    st1h { z24.h }, p2, [x23]
    incb x25, ALL, MUL #3
    incb x24, ALL, MUL #3
    st1h { z23.h }, p1, [x23, #1, MUL VL]
    addvl x5, x5, #24
    st1h { z22.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1h { z21.h }, p2, [x22]
    st1h { z20.h }, p1, [x22, #1, MUL VL]
    st1h { z19.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1h { z18.h }, p2, [x21]
    st1h { z17.h }, p1, [x21, #1, MUL VL]
    st1h { z16.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    bgt label_19
    b label_52
KAI_ASM_LABEL(label_22)  // Initial: Height 6
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
KAI_ASM_LABEL(label_23)  // Initial: Height 6: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_24
    mov z4.b, #0
    mov z3.b, #0
    mov z2.b, #0
    b label_25
KAI_ASM_LABEL(label_24)  // Initial: Height 6: Width 3: bias
    ld1h { z4.h }, p2/Z, [x3]
    ld1h { z3.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z2.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_25)  // Initial: Height 6: Width 3: init done
    ld1h { z1.h }, p2/Z, [x5]
    ld1h { z0.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z31.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z30.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z29.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z28.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z27.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z26.h }, p1/Z, [x5, #7, MUL VL]
    fadd z1.h, z1.h, z4.h
    fadd z0.h, z0.h, z3.h
    ld1h { z25.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z24.h }, p2/Z, [x1, #-7, MUL VL]
    fadd z31.h, z31.h, z2.h
    fadd z30.h, z30.h, z4.h
    ld1h { z23.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z22.h }, p0/Z, [x1, #-5, MUL VL]
    fadd z29.h, z29.h, z3.h
    fadd z28.h, z28.h, z2.h
    ld1h { z21.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z20.h }, p1/Z, [x1, #-3, MUL VL]
    fadd z27.h, z27.h, z4.h
    fadd z26.h, z26.h, z3.h
    ld1h { z19.h }, p0/Z, [x1, #-2, MUL VL]
    ld1h { z18.h }, p2/Z, [x1, #-1, MUL VL]
    fadd z25.h, z25.h, z2.h
    fadd z24.h, z24.h, z4.h
    ld1h { z17.h }, p1/Z, [x1]
    ld1h { z16.h }, p0/Z, [x1, #1, MUL VL]
    fadd z23.h, z23.h, z3.h
    fadd z22.h, z22.h, z2.h
    fadd z21.h, z21.h, z4.h
    fadd z20.h, z20.h, z3.h
    fmin z1.h, p3/M, z1.h, z12.h
    fmin z0.h, p3/M, z0.h, z12.h
    fadd z19.h, z19.h, z2.h
    fadd z18.h, z18.h, z4.h
    fmin z31.h, p3/M, z31.h, z12.h
    fmin z30.h, p3/M, z30.h, z12.h
    fadd z17.h, z17.h, z3.h
    fadd z16.h, z16.h, z2.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    cmp x4, XZR
    fmax z1.h, p3/M, z1.h, z11.h
    fmax z0.h, p3/M, z0.h, z11.h
    fmax z31.h, p3/M, z31.h, z11.h
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z29.h, p3/M, z29.h, z11.h
    st1h { z1.h }, p2, [x25]
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z27.h, p3/M, z27.h, z11.h
    st1h { z0.h }, p1, [x25, #1, MUL VL]
    st1h { z31.h }, p0, [x25, #2, MUL VL]
    fmax z26.h, p3/M, z26.h, z11.h
    fmax z25.h, p3/M, z25.h, z11.h
    st1h { z30.h }, p2, [x24]
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    st1h { z29.h }, p1, [x24, #1, MUL VL]
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    st1h { z28.h }, p0, [x24, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    st1h { z27.h }, p2, [x23]
    fmax z18.h, p3/M, z18.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z26.h }, p1, [x23, #1, MUL VL]
    fmax z16.h, p3/M, z16.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z25.h }, p0, [x23, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1h { z24.h }, p2, [x22]
    addvl x5, x5, #24
    st1h { z23.h }, p1, [x22, #1, MUL VL]
    st1h { z22.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1h { z21.h }, p2, [x21]
    st1h { z20.h }, p1, [x21, #1, MUL VL]
    st1h { z19.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1h { z18.h }, p2, [x20]
    st1h { z17.h }, p1, [x20, #1, MUL VL]
    st1h { z16.h }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    bgt label_23
    b label_52
KAI_ASM_LABEL(label_26)  // Initial: Height 7
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
    add x8, x20, x27
KAI_ASM_LABEL(label_27)  // Initial: Height 7: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_28
    mov z7.b, #0
    mov z6.b, #0
    mov z5.b, #0
    b label_29
KAI_ASM_LABEL(label_28)  // Initial: Height 7: Width 3: bias
    ld1h { z7.h }, p2/Z, [x3]
    ld1h { z6.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z5.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_29)  // Initial: Height 7: Width 3: init done
    ld1h { z4.h }, p2/Z, [x5]
    ld1h { z3.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z2.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z1.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z0.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z31.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z30.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z29.h }, p1/Z, [x5, #7, MUL VL]
    fadd z4.h, z4.h, z7.h
    fadd z3.h, z3.h, z6.h
    ld1h { z28.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z27.h }, p2/Z, [x1, #-7, MUL VL]
    fadd z2.h, z2.h, z5.h
    fadd z1.h, z1.h, z7.h
    ld1h { z26.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z25.h }, p0/Z, [x1, #-5, MUL VL]
    fadd z0.h, z0.h, z6.h
    fadd z31.h, z31.h, z5.h
    ld1h { z24.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z23.h }, p1/Z, [x1, #-3, MUL VL]
    fadd z30.h, z30.h, z7.h
    fadd z29.h, z29.h, z6.h
    ld1h { z22.h }, p0/Z, [x1, #-2, MUL VL]
    ld1h { z21.h }, p2/Z, [x1, #-1, MUL VL]
    fadd z28.h, z28.h, z5.h
    fadd z27.h, z27.h, z7.h
    ld1h { z20.h }, p1/Z, [x1]
    ld1h { z19.h }, p0/Z, [x1, #1, MUL VL]
    fadd z26.h, z26.h, z6.h
    fadd z25.h, z25.h, z5.h
    ld1h { z18.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z17.h }, p1/Z, [x1, #3, MUL VL]
    fadd z24.h, z24.h, z7.h
    fadd z23.h, z23.h, z6.h
    ld1h { z16.h }, p0/Z, [x1, #4, MUL VL]
    fadd z22.h, z22.h, z5.h
    fadd z21.h, z21.h, z7.h
    fmin z4.h, p3/M, z4.h, z12.h
    fadd z20.h, z20.h, z6.h
    fadd z19.h, z19.h, z5.h
    fmin z3.h, p3/M, z3.h, z12.h
    fmin z2.h, p3/M, z2.h, z12.h
    fadd z18.h, z18.h, z7.h
    fadd z17.h, z17.h, z6.h
    fmin z1.h, p3/M, z1.h, z12.h
    fmin z0.h, p3/M, z0.h, z12.h
    fadd z16.h, z16.h, z5.h
    fmin z31.h, p3/M, z31.h, z12.h
    fmin z30.h, p3/M, z30.h, z12.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    cmp x4, XZR
    fmax z4.h, p3/M, z4.h, z11.h
    fmax z3.h, p3/M, z3.h, z11.h
    fmax z2.h, p3/M, z2.h, z11.h
    fmax z1.h, p3/M, z1.h, z11.h
    fmax z0.h, p3/M, z0.h, z11.h
    st1h { z4.h }, p2, [x25]
    fmax z31.h, p3/M, z31.h, z11.h
    fmax z30.h, p3/M, z30.h, z11.h
    st1h { z3.h }, p1, [x25, #1, MUL VL]
    st1h { z2.h }, p0, [x25, #2, MUL VL]
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    st1h { z1.h }, p2, [x24]
    fmax z27.h, p3/M, z27.h, z11.h
    fmax z26.h, p3/M, z26.h, z11.h
    st1h { z0.h }, p1, [x24, #1, MUL VL]
    fmax z25.h, p3/M, z25.h, z11.h
    fmax z24.h, p3/M, z24.h, z11.h
    st1h { z31.h }, p0, [x24, #2, MUL VL]
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    st1h { z30.h }, p2, [x23]
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z29.h }, p1, [x23, #1, MUL VL]
    fmax z19.h, p3/M, z19.h, z11.h
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z28.h }, p0, [x23, #2, MUL VL]
    fmax z17.h, p3/M, z17.h, z11.h
    fmax z16.h, p3/M, z16.h, z11.h
    st1h { z27.h }, p2, [x22]
    incb x25, ALL, MUL #3
    incb x24, ALL, MUL #3
    st1h { z26.h }, p1, [x22, #1, MUL VL]
    incb x23, ALL, MUL #3
    addvl x5, x5, #24
    st1h { z25.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1h { z24.h }, p2, [x21]
    st1h { z23.h }, p1, [x21, #1, MUL VL]
    st1h { z22.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1h { z21.h }, p2, [x20]
    st1h { z20.h }, p1, [x20, #1, MUL VL]
    st1h { z19.h }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    st1h { z18.h }, p2, [x8]
    st1h { z17.h }, p1, [x8, #1, MUL VL]
    st1h { z16.h }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    bgt label_27
    b label_52
KAI_ASM_LABEL(label_30)  // Initial: Height 8
    mov x25, x6
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x40]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
    add x8, x20, x27
    add x7, x8, x27
KAI_ASM_LABEL(label_31)  // Initial: Height 8: Block loop
    mov x2, #0
    addvl x1, x5, #16
    whilelt p2.h, x2, x4
    inch x2
    whilelt p1.h, x2, x4
    inch x2
    whilelt p0.h, x2, x4
    cbnz x3, label_32
    mov z10.b, #0
    mov z9.b, #0
    mov z8.b, #0
    b label_33
KAI_ASM_LABEL(label_32)  // Initial: Height 8: Width 3: bias
    ld1h { z10.h }, p2/Z, [x3]
    ld1h { z9.h }, p1/Z, [x3, #1, MUL VL]
    ld1h { z8.h }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_33)  // Initial: Height 8: Width 3: init done
    ld1h { z7.h }, p2/Z, [x5]
    ld1h { z6.h }, p1/Z, [x5, #1, MUL VL]
    dech x4, ALL, MUL #3
    ld1h { z5.h }, p0/Z, [x5, #2, MUL VL]
    ld1h { z4.h }, p2/Z, [x5, #3, MUL VL]
    ld1h { z3.h }, p1/Z, [x5, #4, MUL VL]
    ld1h { z2.h }, p0/Z, [x5, #5, MUL VL]
    ld1h { z1.h }, p2/Z, [x5, #6, MUL VL]
    ld1h { z0.h }, p1/Z, [x5, #7, MUL VL]
    fadd z7.h, z7.h, z10.h
    fadd z6.h, z6.h, z9.h
    ld1h { z31.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z30.h }, p2/Z, [x1, #-7, MUL VL]
    fadd z5.h, z5.h, z8.h
    fadd z4.h, z4.h, z10.h
    ld1h { z29.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z28.h }, p0/Z, [x1, #-5, MUL VL]
    fadd z3.h, z3.h, z9.h
    fadd z2.h, z2.h, z8.h
    ld1h { z27.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z26.h }, p1/Z, [x1, #-3, MUL VL]
    fadd z1.h, z1.h, z10.h
    fadd z0.h, z0.h, z9.h
    ld1h { z25.h }, p0/Z, [x1, #-2, MUL VL]
    ld1h { z24.h }, p2/Z, [x1, #-1, MUL VL]
    fadd z31.h, z31.h, z8.h
    fadd z30.h, z30.h, z10.h
    ld1h { z23.h }, p1/Z, [x1]
    ld1h { z22.h }, p0/Z, [x1, #1, MUL VL]
    fadd z29.h, z29.h, z9.h
    fadd z28.h, z28.h, z8.h
    ld1h { z21.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z20.h }, p1/Z, [x1, #3, MUL VL]
    fadd z27.h, z27.h, z10.h
    fadd z26.h, z26.h, z9.h
    ld1h { z19.h }, p0/Z, [x1, #4, MUL VL]
    ld1h { z18.h }, p2/Z, [x1, #5, MUL VL]
    fadd z25.h, z25.h, z8.h
    fadd z24.h, z24.h, z10.h
    ld1h { z17.h }, p1/Z, [x1, #6, MUL VL]
    ld1h { z16.h }, p0/Z, [x1, #7, MUL VL]
    fadd z23.h, z23.h, z9.h
    fadd z22.h, z22.h, z8.h
    fadd z21.h, z21.h, z10.h
    fadd z20.h, z20.h, z9.h
    fmin z7.h, p3/M, z7.h, z12.h
    fmin z6.h, p3/M, z6.h, z12.h
    fadd z19.h, z19.h, z8.h
    fadd z18.h, z18.h, z10.h
    fmin z5.h, p3/M, z5.h, z12.h
    fmin z4.h, p3/M, z4.h, z12.h
    fadd z17.h, z17.h, z9.h
    fadd z16.h, z16.h, z8.h
    fmin z3.h, p3/M, z3.h, z12.h
    fmin z2.h, p3/M, z2.h, z12.h
    fmin z1.h, p3/M, z1.h, z12.h
    fmin z0.h, p3/M, z0.h, z12.h
    fmin z31.h, p3/M, z31.h, z12.h
    fmin z30.h, p3/M, z30.h, z12.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmin z16.h, p3/M, z16.h, z12.h
    cmp x4, XZR
    fmax z7.h, p3/M, z7.h, z11.h
    fmax z6.h, p3/M, z6.h, z11.h
    fmax z5.h, p3/M, z5.h, z11.h
    fmax z4.h, p3/M, z4.h, z11.h
    fmax z3.h, p3/M, z3.h, z11.h
    st1h { z7.h }, p2, [x25]
    fmax z2.h, p3/M, z2.h, z11.h
    fmax z1.h, p3/M, z1.h, z11.h
    st1h { z6.h }, p1, [x25, #1, MUL VL]
    st1h { z5.h }, p0, [x25, #2, MUL VL]
    fmax z0.h, p3/M, z0.h, z11.h
    fmax z31.h, p3/M, z31.h, z11.h
    st1h { z4.h }, p2, [x24]
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z29.h, p3/M, z29.h, z11.h
    st1h { z3.h }, p1, [x24, #1, MUL VL]
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z27.h, p3/M, z27.h, z11.h
    st1h { z2.h }, p0, [x24, #2, MUL VL]
    fmax z26.h, p3/M, z26.h, z11.h
    fmax z25.h, p3/M, z25.h, z11.h
    st1h { z1.h }, p2, [x23]
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    st1h { z0.h }, p1, [x23, #1, MUL VL]
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    st1h { z31.h }, p0, [x23, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    st1h { z30.h }, p2, [x22]
    fmax z18.h, p3/M, z18.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z29.h }, p1, [x22, #1, MUL VL]
    fmax z16.h, p3/M, z16.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z28.h }, p0, [x22, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1h { z27.h }, p2, [x21]
    incb x22, ALL, MUL #3
    addvl x5, x5, #24
    st1h { z26.h }, p1, [x21, #1, MUL VL]
    st1h { z25.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1h { z24.h }, p2, [x20]
    st1h { z23.h }, p1, [x20, #1, MUL VL]
    st1h { z22.h }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    st1h { z21.h }, p2, [x8]
    st1h { z20.h }, p1, [x8, #1, MUL VL]
    st1h { z19.h }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    st1h { z18.h }, p2, [x7]
    st1h { z17.h }, p1, [x7, #1, MUL VL]
    st1h { z16.h }, p0, [x7, #2, MUL VL]
    incb x7, ALL, MUL #3
    bgt label_31
    subs x26, x26, #0x8
    add x6, x6, x28
    bgt label_1
    b label_52
KAI_ASM_LABEL(label_34)  // Accumulate
    ldr x6, [x0, #0]
    ldr x5, [x0, #0x50]
    ldr x4, [x0, #0x20]
KAI_ASM_LABEL(label_35)  // Accumulate: Row loop
    cmp x6, #0x7
    bgt label_50
    beq label_48
    cmp x6, #0x5
    bgt label_46
    beq label_44
    cmp x6, #0x3
    bgt label_42
    beq label_40
    cmp x6, #0x1
    bgt label_38
    ldr x2, [x0, #0x8]
    mov x25, x5
KAI_ASM_LABEL(label_37)  // Accumulate: Height 1: Block loop
    mov x1, #0
    whilelt p0.h, x1, x2
    ld1h { z17.h }, p0/Z, [x4]
    ld1h { z16.h }, p0/Z, [x25]
    fadd z17.h, z17.h, z16.h
    inch x1
    fmin z17.h, p3/M, z17.h, z12.h
    whilelt p1.h, x1, x2
    fmax z17.h, p3/M, z17.h, z11.h
    inch x1
    st1h { z17.h }, p0, [x25]
    ld1h { z18.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z16.h }, p1/Z, [x25, #1, MUL VL]
    fadd z18.h, z18.h, z16.h
    whilelt p0.h, x1, x2
    dech x2, ALL, MUL #3
    fmin z18.h, p3/M, z18.h, z12.h
    ld1h { z17.h }, p0/Z, [x4, #2, MUL VL]
    addvl x4, x4, #24
    cmp x2, XZR
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z18.h }, p1, [x25, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x25, #2, MUL VL]
    fadd z17.h, z17.h, z16.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z17.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    bgt label_37
    b label_52
KAI_ASM_LABEL(label_38)  // Accumulate: Height 2
    mov x25, x5
    ldr x2, [x0, #0x8]
    add x24, x25, x27
KAI_ASM_LABEL(label_39)  // Accumulate: Height 2: Block loop
    mov x1, #0
    whilelt p2.h, x1, x2
    ld1h { z19.h }, p2/Z, [x4]
    ld1h { z18.h }, p2/Z, [x25]
    ld1h { z17.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z16.h }, p2/Z, [x24]
    fadd z19.h, z19.h, z18.h
    inch x1
    fadd z17.h, z17.h, z16.h
    fmin z19.h, p3/M, z19.h, z12.h
    whilelt p1.h, x1, x2
    inch x1
    fmin z17.h, p3/M, z17.h, z12.h
    fmax z19.h, p3/M, z19.h, z11.h
    ld1h { z21.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z20.h }, p1/Z, [x4, #4, MUL VL]
    fmax z17.h, p3/M, z17.h, z11.h
    whilelt p0.h, x1, x2
    dech x2, ALL, MUL #3
    st1h { z19.h }, p2, [x25]
    st1h { z17.h }, p2, [x24]
    ld1h { z17.h }, p1/Z, [x25, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x4, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    cmp x2, XZR
    ld1h { z16.h }, p1/Z, [x24, #1, MUL VL]
    fadd z21.h, z21.h, z17.h
    fadd z20.h, z20.h, z16.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z21.h }, p1, [x25, #1, MUL VL]
    st1h { z20.h }, p1, [x24, #1, MUL VL]
    ld1h { z17.h }, p0/Z, [x25, #2, MUL VL]
    ld1h { z16.h }, p0/Z, [x24, #2, MUL VL]
    fadd z19.h, z19.h, z17.h
    fadd z18.h, z18.h, z16.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z18.h, p3/M, z18.h, z12.h
    fmax z19.h, p3/M, z19.h, z11.h
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z19.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1h { z18.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    bgt label_39
    b label_52
KAI_ASM_LABEL(label_40)  // Accumulate: Height 3
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
KAI_ASM_LABEL(label_41)  // Accumulate: Height 3: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z20.h }, p2/Z, [x4]
    ld1h { z19.h }, p2/Z, [x25]
    ld1h { z25.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z18.h }, p2/Z, [x24]
    ld1h { z17.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z16.h }, p2/Z, [x23]
    fadd z20.h, z20.h, z19.h
    whilelt p1.h, x2, x3
    inch x2
    fadd z25.h, z25.h, z18.h
    fadd z17.h, z17.h, z16.h
    fmin z20.h, p3/M, z20.h, z12.h
    ld1h { z24.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z23.h }, p1/Z, [x4, #4, MUL VL]
    ld1h { z22.h }, p1/Z, [x4, #7, MUL VL]
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmax z20.h, p3/M, z20.h, z11.h
    whilelt p0.h, x2, x3
    dech x3, ALL, MUL #3
    fmax z25.h, p3/M, z25.h, z11.h
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z20.h }, p2, [x25]
    ld1h { z21.h }, p0/Z, [x4, #2, MUL VL]
    ld1h { z20.h }, p0/Z, [x4, #5, MUL VL]
    ld1h { z19.h }, p0/Z, [x1, #-8, MUL VL]
    addvl x4, x4, #24
    st1h { z25.h }, p2, [x24]
    cmp x3, XZR
    st1h { z17.h }, p2, [x23]
    ld1h { z18.h }, p1/Z, [x25, #1, MUL VL]
    ld1h { z17.h }, p1/Z, [x24, #1, MUL VL]
    ld1h { z16.h }, p1/Z, [x23, #1, MUL VL]
    fadd z24.h, z24.h, z18.h
    fadd z23.h, z23.h, z17.h
    fadd z22.h, z22.h, z16.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    st1h { z24.h }, p1, [x25, #1, MUL VL]
    fmax z22.h, p3/M, z22.h, z11.h
    st1h { z23.h }, p1, [x24, #1, MUL VL]
    st1h { z22.h }, p1, [x23, #1, MUL VL]
    ld1h { z18.h }, p0/Z, [x25, #2, MUL VL]
    ld1h { z17.h }, p0/Z, [x24, #2, MUL VL]
    ld1h { z16.h }, p0/Z, [x23, #2, MUL VL]
    fadd z21.h, z21.h, z18.h
    fadd z20.h, z20.h, z17.h
    fadd z19.h, z19.h, z16.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z21.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    fmax z19.h, p3/M, z19.h, z11.h
    st1h { z20.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1h { z19.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    bgt label_41
    b label_52
KAI_ASM_LABEL(label_42)  // Accumulate: Height 4
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
KAI_ASM_LABEL(label_43)  // Accumulate: Height 4: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z29.h }, p2/Z, [x4]
    ld1h { z21.h }, p2/Z, [x25]
    ld1h { z28.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z20.h }, p2/Z, [x24]
    ld1h { z19.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z18.h }, p2/Z, [x23]
    ld1h { z17.h }, p2/Z, [x1, #-7, MUL VL]
    ld1h { z16.h }, p2/Z, [x22]
    fadd z29.h, z29.h, z21.h
    whilelt p1.h, x2, x3
    fadd z28.h, z28.h, z20.h
    inch x2
    fadd z19.h, z19.h, z18.h
    fadd z17.h, z17.h, z16.h
    fmin z29.h, p3/M, z29.h, z12.h
    ld1h { z27.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z26.h }, p1/Z, [x4, #4, MUL VL]
    ld1h { z25.h }, p1/Z, [x4, #7, MUL VL]
    ld1h { z24.h }, p1/Z, [x1, #-6, MUL VL]
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    whilelt p0.h, x2, x3
    dech x3, ALL, MUL #3
    fmin z17.h, p3/M, z17.h, z12.h
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    ld1h { z23.h }, p0/Z, [x4, #2, MUL VL]
    ld1h { z22.h }, p0/Z, [x4, #5, MUL VL]
    ld1h { z21.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z20.h }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    addvl x4, x4, #24
    st1h { z29.h }, p2, [x25]
    fmax z17.h, p3/M, z17.h, z11.h
    st1h { z28.h }, p2, [x24]
    st1h { z19.h }, p2, [x23]
    st1h { z17.h }, p2, [x22]
    ld1h { z19.h }, p1/Z, [x25, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x24, #1, MUL VL]
    ld1h { z17.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z16.h }, p1/Z, [x22, #1, MUL VL]
    fadd z27.h, z27.h, z19.h
    fadd z26.h, z26.h, z18.h
    fadd z25.h, z25.h, z17.h
    fadd z24.h, z24.h, z16.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmax z27.h, p3/M, z27.h, z11.h
    fmax z26.h, p3/M, z26.h, z11.h
    fmax z25.h, p3/M, z25.h, z11.h
    fmax z24.h, p3/M, z24.h, z11.h
    st1h { z27.h }, p1, [x25, #1, MUL VL]
    st1h { z26.h }, p1, [x24, #1, MUL VL]
    st1h { z25.h }, p1, [x23, #1, MUL VL]
    st1h { z24.h }, p1, [x22, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x25, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x24, #2, MUL VL]
    ld1h { z17.h }, p0/Z, [x23, #2, MUL VL]
    ld1h { z16.h }, p0/Z, [x22, #2, MUL VL]
    fadd z23.h, z23.h, z19.h
    fadd z22.h, z22.h, z18.h
    fadd z21.h, z21.h, z17.h
    fadd z20.h, z20.h, z16.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z23.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1h { z22.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1h { z21.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1h { z20.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    bgt label_43
    b label_52
KAI_ASM_LABEL(label_44)  // Accumulate: Height 5
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
KAI_ASM_LABEL(label_45)  // Accumulate: Height 5: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z0.h }, p2/Z, [x4]
    ld1h { z16.h }, p2/Z, [x25]
    ld1h { z31.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z21.h }, p2/Z, [x24]
    ld1h { z30.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z20.h }, p2/Z, [x23]
    ld1h { z19.h }, p2/Z, [x1, #-7, MUL VL]
    ld1h { z18.h }, p2/Z, [x22]
    fadd z0.h, z0.h, z16.h
    whilelt p1.h, x2, x3
    ld1h { z17.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z16.h }, p2/Z, [x21]
    fadd z31.h, z31.h, z21.h
    inch x2
    fadd z30.h, z30.h, z20.h
    fadd z19.h, z19.h, z18.h
    fmin z0.h, p3/M, z0.h, z12.h
    ld1h { z29.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z28.h }, p1/Z, [x4, #4, MUL VL]
    ld1h { z27.h }, p1/Z, [x4, #7, MUL VL]
    fadd z17.h, z17.h, z16.h
    fmin z31.h, p3/M, z31.h, z12.h
    ld1h { z26.h }, p1/Z, [x1, #-6, MUL VL]
    ld1h { z25.h }, p1/Z, [x1, #-3, MUL VL]
    whilelt p0.h, x2, x3
    dech x3, ALL, MUL #3
    fmin z30.h, p3/M, z30.h, z12.h
    fmin z19.h, p3/M, z19.h, z12.h
    fmin z17.h, p3/M, z17.h, z12.h
    fmax z0.h, p3/M, z0.h, z11.h
    fmax z31.h, p3/M, z31.h, z11.h
    ld1h { z24.h }, p0/Z, [x4, #2, MUL VL]
    ld1h { z23.h }, p0/Z, [x4, #5, MUL VL]
    cmp x3, XZR
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    ld1h { z22.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z21.h }, p0/Z, [x1, #-5, MUL VL]
    st1h { z0.h }, p2, [x25]
    fmax z17.h, p3/M, z17.h, z11.h
    ld1h { z20.h }, p0/Z, [x1, #-2, MUL VL]
    addvl x4, x4, #24
    st1h { z31.h }, p2, [x24]
    st1h { z30.h }, p2, [x23]
    st1h { z19.h }, p2, [x22]
    ld1h { z16.h }, p1/Z, [x25, #1, MUL VL]
    st1h { z17.h }, p2, [x21]
    ld1h { z19.h }, p1/Z, [x24, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z17.h }, p1/Z, [x22, #1, MUL VL]
    fadd z29.h, z29.h, z16.h
    ld1h { z16.h }, p1/Z, [x21, #1, MUL VL]
    fadd z28.h, z28.h, z19.h
    fadd z27.h, z27.h, z18.h
    fadd z26.h, z26.h, z17.h
    fadd z25.h, z25.h, z16.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z27.h, p3/M, z27.h, z11.h
    fmax z26.h, p3/M, z26.h, z11.h
    st1h { z29.h }, p1, [x25, #1, MUL VL]
    fmax z25.h, p3/M, z25.h, z11.h
    st1h { z28.h }, p1, [x24, #1, MUL VL]
    st1h { z27.h }, p1, [x23, #1, MUL VL]
    st1h { z26.h }, p1, [x22, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x25, #2, MUL VL]
    st1h { z25.h }, p1, [x21, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x24, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x23, #2, MUL VL]
    ld1h { z17.h }, p0/Z, [x22, #2, MUL VL]
    fadd z24.h, z24.h, z16.h
    ld1h { z16.h }, p0/Z, [x21, #2, MUL VL]
    fadd z23.h, z23.h, z19.h
    fadd z22.h, z22.h, z18.h
    fadd z21.h, z21.h, z17.h
    fadd z20.h, z20.h, z16.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    st1h { z24.h }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    fmax z20.h, p3/M, z20.h, z11.h
    st1h { z23.h }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1h { z22.h }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1h { z21.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1h { z20.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    bgt label_45
    b label_52
KAI_ASM_LABEL(label_46)  // Accumulate: Height 6
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
KAI_ASM_LABEL(label_47)  // Accumulate: Height 6: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z3.h }, p2/Z, [x4]
    ld1h { z17.h }, p2/Z, [x25]
    ld1h { z2.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z16.h }, p2/Z, [x24]
    ld1h { z1.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z21.h }, p2/Z, [x23]
    ld1h { z0.h }, p2/Z, [x1, #-7, MUL VL]
    ld1h { z20.h }, p2/Z, [x22]
    fadd z3.h, z3.h, z17.h
    whilelt p1.h, x2, x3
    ld1h { z19.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z17.h }, p2/Z, [x21]
    fadd z2.h, z2.h, z16.h
    inch x2
    ld1h { z18.h }, p2/Z, [x1, #-1, MUL VL]
    ld1h { z16.h }, p2/Z, [x20]
    fadd z1.h, z1.h, z21.h
    fadd z0.h, z0.h, z20.h
    fmin z3.h, p3/M, z3.h, z12.h
    ld1h { z31.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z30.h }, p1/Z, [x4, #4, MUL VL]
    fadd z19.h, z19.h, z17.h
    fmin z2.h, p3/M, z2.h, z12.h
    ld1h { z29.h }, p1/Z, [x4, #7, MUL VL]
    ld1h { z28.h }, p1/Z, [x1, #-6, MUL VL]
    whilelt p0.h, x2, x3
    fadd z18.h, z18.h, z16.h
    fmin z1.h, p3/M, z1.h, z12.h
    fmin z0.h, p3/M, z0.h, z12.h
    ld1h { z27.h }, p1/Z, [x1, #-3, MUL VL]
    fmin z19.h, p3/M, z19.h, z12.h
    fmax z3.h, p3/M, z3.h, z11.h
    ld1h { z26.h }, p1/Z, [x1]
    fmax z2.h, p3/M, z2.h, z11.h
    ld1h { z25.h }, p0/Z, [x4, #2, MUL VL]
    ld1h { z24.h }, p0/Z, [x4, #5, MUL VL]
    dech x3, ALL, MUL #3
    fmin z18.h, p3/M, z18.h, z12.h
    fmax z1.h, p3/M, z1.h, z11.h
    ld1h { z23.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z22.h }, p0/Z, [x1, #-5, MUL VL]
    st1h { z3.h }, p2, [x25]
    fmax z0.h, p3/M, z0.h, z11.h
    fmax z19.h, p3/M, z19.h, z11.h
    ld1h { z21.h }, p0/Z, [x1, #-2, MUL VL]
    st1h { z2.h }, p2, [x24]
    ld1h { z20.h }, p0/Z, [x1, #1, MUL VL]
    cmp x3, XZR
    addvl x4, x4, #24
    st1h { z1.h }, p2, [x23]
    fmax z18.h, p3/M, z18.h, z11.h
    st1h { z0.h }, p2, [x22]
    ld1h { z17.h }, p1/Z, [x25, #1, MUL VL]
    st1h { z19.h }, p2, [x21]
    ld1h { z16.h }, p1/Z, [x24, #1, MUL VL]
    st1h { z18.h }, p2, [x20]
    ld1h { z19.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z18.h }, p1/Z, [x22, #1, MUL VL]
    fadd z31.h, z31.h, z17.h
    ld1h { z17.h }, p1/Z, [x21, #1, MUL VL]
    fadd z30.h, z30.h, z16.h
    ld1h { z16.h }, p1/Z, [x20, #1, MUL VL]
    fadd z29.h, z29.h, z19.h
    fadd z28.h, z28.h, z18.h
    fmin z31.h, p3/M, z31.h, z12.h
    fadd z27.h, z27.h, z17.h
    fadd z26.h, z26.h, z16.h
    fmin z30.h, p3/M, z30.h, z12.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmax z31.h, p3/M, z31.h, z11.h
    fmin z26.h, p3/M, z26.h, z12.h
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    fmax z27.h, p3/M, z27.h, z11.h
    st1h { z31.h }, p1, [x25, #1, MUL VL]
    fmax z26.h, p3/M, z26.h, z11.h
    st1h { z30.h }, p1, [x24, #1, MUL VL]
    st1h { z29.h }, p1, [x23, #1, MUL VL]
    st1h { z28.h }, p1, [x22, #1, MUL VL]
    ld1h { z17.h }, p0/Z, [x25, #2, MUL VL]
    st1h { z27.h }, p1, [x21, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x24, #2, MUL VL]
    st1h { z26.h }, p1, [x20, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x23, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x22, #2, MUL VL]
    fadd z25.h, z25.h, z17.h
    ld1h { z17.h }, p0/Z, [x21, #2, MUL VL]
    fadd z24.h, z24.h, z16.h
    ld1h { z16.h }, p0/Z, [x20, #2, MUL VL]
    fadd z23.h, z23.h, z19.h
    fadd z22.h, z22.h, z18.h
    fmin z25.h, p3/M, z25.h, z12.h
    fadd z21.h, z21.h, z17.h
    fadd z20.h, z20.h, z16.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmax z25.h, p3/M, z25.h, z11.h
    fmin z20.h, p3/M, z20.h, z12.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    fmax z21.h, p3/M, z21.h, z11.h
    st1h { z25.h }, p0, [x25, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z24.h }, p0, [x24, #2, MUL VL]
    st1h { z23.h }, p0, [x23, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1h { z22.h }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1h { z21.h }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1h { z20.h }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    bgt label_47
    b label_52
KAI_ASM_LABEL(label_48)  // Accumulate: Height 7
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
    add x8, x20, x27
KAI_ASM_LABEL(label_49)  // Accumulate: Height 7: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z26.h }, p2/Z, [x4]
    ld1h { z18.h }, p2/Z, [x25]
    ld1h { z25.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z17.h }, p2/Z, [x24]
    ld1h { z24.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z16.h }, p2/Z, [x23]
    ld1h { z23.h }, p2/Z, [x1, #-7, MUL VL]
    ld1h { z22.h }, p2/Z, [x22]
    fadd z26.h, z26.h, z18.h
    whilelt p1.h, x2, x3
    ld1h { z21.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z18.h }, p2/Z, [x21]
    fadd z25.h, z25.h, z17.h
    inch x2
    ld1h { z20.h }, p2/Z, [x1, #-1, MUL VL]
    ld1h { z17.h }, p2/Z, [x20]
    fadd z24.h, z24.h, z16.h
    ld1h { z19.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z16.h }, p2/Z, [x8]
    fadd z23.h, z23.h, z22.h
    fmin z26.h, p3/M, z26.h, z12.h
    fadd z21.h, z21.h, z18.h
    fmin z25.h, p3/M, z25.h, z12.h
    ld1h { z1.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z0.h }, p1/Z, [x4, #4, MUL VL]
    fadd z20.h, z20.h, z17.h
    fmin z24.h, p3/M, z24.h, z12.h
    ld1h { z31.h }, p1/Z, [x4, #7, MUL VL]
    ld1h { z30.h }, p1/Z, [x1, #-6, MUL VL]
    fadd z19.h, z19.h, z16.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmax z26.h, p3/M, z26.h, z11.h
    ld1h { z29.h }, p1/Z, [x1, #-3, MUL VL]
    fmin z21.h, p3/M, z21.h, z12.h
    fmax z25.h, p3/M, z25.h, z11.h
    ld1h { z28.h }, p1/Z, [x1]
    ld1h { z27.h }, p1/Z, [x1, #3, MUL VL]
    fmin z20.h, p3/M, z20.h, z12.h
    fmax z24.h, p3/M, z24.h, z11.h
    fmin z19.h, p3/M, z19.h, z12.h
    st1h { z26.h }, p2, [x25]
    fmax z23.h, p3/M, z23.h, z11.h
    st1h { z25.h }, p2, [x24]
    fmax z21.h, p3/M, z21.h, z11.h
    whilelt p0.h, x2, x3
    st1h { z24.h }, p2, [x23]
    fmax z20.h, p3/M, z20.h, z11.h
    dech x3, ALL, MUL #3
    st1h { z23.h }, p2, [x22]
    fmax z19.h, p3/M, z19.h, z11.h
    ld1h { z18.h }, p1/Z, [x25, #1, MUL VL]
    st1h { z21.h }, p2, [x21]
    ld1h { z17.h }, p1/Z, [x24, #1, MUL VL]
    ld1h { z26.h }, p0/Z, [x4, #2, MUL VL]
    st1h { z20.h }, p2, [x20]
    ld1h { z16.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z25.h }, p0/Z, [x4, #5, MUL VL]
    cmp x3, XZR
    st1h { z19.h }, p2, [x8]
    ld1h { z19.h }, p1/Z, [x22, #1, MUL VL]
    fadd z1.h, z1.h, z18.h
    ld1h { z24.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z18.h }, p1/Z, [x21, #1, MUL VL]
    fadd z0.h, z0.h, z17.h
    ld1h { z23.h }, p0/Z, [x1, #-5, MUL VL]
    addvl x4, x4, #24
    ld1h { z17.h }, p1/Z, [x20, #1, MUL VL]
    fadd z31.h, z31.h, z16.h
    ld1h { z22.h }, p0/Z, [x1, #-2, MUL VL]
    ld1h { z16.h }, p1/Z, [x8, #1, MUL VL]
    fadd z30.h, z30.h, z19.h
    fmin z1.h, p3/M, z1.h, z12.h
    ld1h { z21.h }, p0/Z, [x1, #1, MUL VL]
    fadd z29.h, z29.h, z18.h
    fmin z0.h, p3/M, z0.h, z12.h
    ld1h { z20.h }, p0/Z, [x1, #4, MUL VL]
    fadd z28.h, z28.h, z17.h
    fmin z31.h, p3/M, z31.h, z12.h
    fadd z27.h, z27.h, z16.h
    fmin z30.h, p3/M, z30.h, z12.h
    fmax z1.h, p3/M, z1.h, z11.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    fmax z0.h, p3/M, z0.h, z11.h
    fmin z27.h, p3/M, z27.h, z12.h
    fmax z31.h, p3/M, z31.h, z11.h
    st1h { z1.h }, p1, [x25, #1, MUL VL]
    fmax z30.h, p3/M, z30.h, z11.h
    fmax z29.h, p3/M, z29.h, z11.h
    st1h { z0.h }, p1, [x24, #1, MUL VL]
    fmax z28.h, p3/M, z28.h, z11.h
    st1h { z31.h }, p1, [x23, #1, MUL VL]
    fmax z27.h, p3/M, z27.h, z11.h
    ld1h { z18.h }, p0/Z, [x25, #2, MUL VL]
    st1h { z30.h }, p1, [x22, #1, MUL VL]
    st1h { z29.h }, p1, [x21, #1, MUL VL]
    ld1h { z17.h }, p0/Z, [x24, #2, MUL VL]
    st1h { z28.h }, p1, [x20, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x23, #2, MUL VL]
    fadd z26.h, z26.h, z18.h
    st1h { z27.h }, p1, [x8, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x22, #2, MUL VL]
    ld1h { z18.h }, p0/Z, [x21, #2, MUL VL]
    fadd z25.h, z25.h, z17.h
    ld1h { z17.h }, p0/Z, [x20, #2, MUL VL]
    fadd z24.h, z24.h, z16.h
    fmin z26.h, p3/M, z26.h, z12.h
    ld1h { z16.h }, p0/Z, [x8, #2, MUL VL]
    fadd z23.h, z23.h, z19.h
    fadd z22.h, z22.h, z18.h
    fmin z25.h, p3/M, z25.h, z12.h
    fadd z21.h, z21.h, z17.h
    fadd z20.h, z20.h, z16.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmax z26.h, p3/M, z26.h, z11.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmax z25.h, p3/M, z25.h, z11.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmin z20.h, p3/M, z20.h, z12.h
    st1h { z26.h }, p0, [x25, #2, MUL VL]
    fmax z24.h, p3/M, z24.h, z11.h
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    st1h { z25.h }, p0, [x24, #2, MUL VL]
    fmax z21.h, p3/M, z21.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z24.h }, p0, [x23, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    incb x24, ALL, MUL #3
    st1h { z23.h }, p0, [x22, #2, MUL VL]
    st1h { z22.h }, p0, [x21, #2, MUL VL]
    incb x23, ALL, MUL #3
    incb x22, ALL, MUL #3
    st1h { z21.h }, p0, [x20, #2, MUL VL]
    incb x21, ALL, MUL #3
    incb x20, ALL, MUL #3
    st1h { z20.h }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    bgt label_49
    b label_52
KAI_ASM_LABEL(label_50)  // Accumulate: Height 8
    mov x25, x5
    ldr x3, [x0, #0x8]
    add x24, x25, x27
    add x23, x24, x27
    add x22, x23, x27
    add x21, x22, x27
    add x20, x21, x27
    add x8, x20, x27
    add x7, x8, x27
KAI_ASM_LABEL(label_51)  // Accumulate: Height 8: Block loop
    mov x2, #0
    addvl x1, x4, #16
    whilelt p2.h, x2, x3
    inch x2
    ld1h { z27.h }, p2/Z, [x4]
    ld1h { z19.h }, p2/Z, [x25]
    ld1h { z26.h }, p2/Z, [x4, #3, MUL VL]
    ld1h { z18.h }, p2/Z, [x24]
    ld1h { z25.h }, p2/Z, [x4, #6, MUL VL]
    ld1h { z17.h }, p2/Z, [x23]
    ld1h { z24.h }, p2/Z, [x1, #-7, MUL VL]
    ld1h { z16.h }, p2/Z, [x22]
    fadd z27.h, z27.h, z19.h
    whilelt p1.h, x2, x3
    ld1h { z23.h }, p2/Z, [x1, #-4, MUL VL]
    ld1h { z19.h }, p2/Z, [x21]
    fadd z26.h, z26.h, z18.h
    inch x2
    ld1h { z22.h }, p2/Z, [x1, #-1, MUL VL]
    ld1h { z18.h }, p2/Z, [x20]
    fadd z25.h, z25.h, z17.h
    ld1h { z21.h }, p2/Z, [x1, #2, MUL VL]
    ld1h { z17.h }, p2/Z, [x8]
    fadd z24.h, z24.h, z16.h
    fmin z27.h, p3/M, z27.h, z12.h
    ld1h { z20.h }, p2/Z, [x1, #5, MUL VL]
    ld1h { z16.h }, p2/Z, [x7]
    fadd z23.h, z23.h, z19.h
    fmin z26.h, p3/M, z26.h, z12.h
    fadd z22.h, z22.h, z18.h
    fmin z25.h, p3/M, z25.h, z12.h
    ld1h { z3.h }, p1/Z, [x4, #1, MUL VL]
    ld1h { z2.h }, p1/Z, [x4, #4, MUL VL]
    fadd z21.h, z21.h, z17.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmax z27.h, p3/M, z27.h, z11.h
    ld1h { z1.h }, p1/Z, [x4, #7, MUL VL]
    fadd z20.h, z20.h, z16.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmax z26.h, p3/M, z26.h, z11.h
    ld1h { z0.h }, p1/Z, [x1, #-6, MUL VL]
    fmin z22.h, p3/M, z22.h, z12.h
    fmax z25.h, p3/M, z25.h, z11.h
    ld1h { z31.h }, p1/Z, [x1, #-3, MUL VL]
    ld1h { z30.h }, p1/Z, [x1]
    fmin z21.h, p3/M, z21.h, z12.h
    st1h { z27.h }, p2, [x25]
    fmax z24.h, p3/M, z24.h, z11.h
    ld1h { z29.h }, p1/Z, [x1, #3, MUL VL]
    fmin z20.h, p3/M, z20.h, z12.h
    st1h { z26.h }, p2, [x24]
    fmax z23.h, p3/M, z23.h, z11.h
    ld1h { z28.h }, p1/Z, [x1, #6, MUL VL]
    st1h { z25.h }, p2, [x23]
    fmax z22.h, p3/M, z22.h, z11.h
    whilelt p0.h, x2, x3
    st1h { z24.h }, p2, [x22]
    fmax z21.h, p3/M, z21.h, z11.h
    ld1h { z19.h }, p1/Z, [x25, #1, MUL VL]
    dech x3, ALL, MUL #3
    st1h { z23.h }, p2, [x21]
    fmax z20.h, p3/M, z20.h, z11.h
    ld1h { z18.h }, p1/Z, [x24, #1, MUL VL]
    st1h { z22.h }, p2, [x20]
    ld1h { z17.h }, p1/Z, [x23, #1, MUL VL]
    ld1h { z27.h }, p0/Z, [x4, #2, MUL VL]
    st1h { z21.h }, p2, [x8]
    ld1h { z16.h }, p1/Z, [x22, #1, MUL VL]
    fadd z3.h, z3.h, z19.h
    ld1h { z26.h }, p0/Z, [x4, #5, MUL VL]
    st1h { z20.h }, p2, [x7]
    ld1h { z19.h }, p1/Z, [x21, #1, MUL VL]
    fadd z2.h, z2.h, z18.h
    ld1h { z25.h }, p0/Z, [x1, #-8, MUL VL]
    ld1h { z18.h }, p1/Z, [x20, #1, MUL VL]
    fadd z1.h, z1.h, z17.h
    ld1h { z24.h }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    ld1h { z17.h }, p1/Z, [x8, #1, MUL VL]
    fadd z0.h, z0.h, z16.h
    fmin z3.h, p3/M, z3.h, z12.h
    ld1h { z23.h }, p0/Z, [x1, #-2, MUL VL]
    ld1h { z16.h }, p1/Z, [x7, #1, MUL VL]
    fadd z31.h, z31.h, z19.h
    fmin z2.h, p3/M, z2.h, z12.h
    ld1h { z22.h }, p0/Z, [x1, #1, MUL VL]
    fadd z30.h, z30.h, z18.h
    fmin z1.h, p3/M, z1.h, z12.h
    ld1h { z21.h }, p0/Z, [x1, #4, MUL VL]
    ld1h { z20.h }, p0/Z, [x1, #7, MUL VL]
    fadd z29.h, z29.h, z17.h
    fmin z0.h, p3/M, z0.h, z12.h
    fmax z3.h, p3/M, z3.h, z11.h
    fadd z28.h, z28.h, z16.h
    fmin z31.h, p3/M, z31.h, z12.h
    fmax z2.h, p3/M, z2.h, z11.h
    fmin z30.h, p3/M, z30.h, z12.h
    fmax z1.h, p3/M, z1.h, z11.h
    fmin z29.h, p3/M, z29.h, z12.h
    fmin z28.h, p3/M, z28.h, z12.h
    st1h { z3.h }, p1, [x25, #1, MUL VL]
    st1h { z2.h }, p1, [x24, #1, MUL VL]
    fmax z0.h, p3/M, z0.h, z11.h
    fmax z31.h, p3/M, z31.h, z11.h
    st1h { z1.h }, p1, [x23, #1, MUL VL]
    fmax z30.h, p3/M, z30.h, z11.h
    addvl x4, x4, #24
    fmax z29.h, p3/M, z29.h, z11.h
    fmax z28.h, p3/M, z28.h, z11.h
    ld1h { z19.h }, p0/Z, [x25, #2, MUL VL]
    st1h { z0.h }, p1, [x22, #1, MUL VL]
    st1h { z31.h }, p1, [x21, #1, MUL VL]
    ld1h { z18.h }, p0/Z, [x24, #2, MUL VL]
    st1h { z30.h }, p1, [x20, #1, MUL VL]
    ld1h { z17.h }, p0/Z, [x23, #2, MUL VL]
    st1h { z29.h }, p1, [x8, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x22, #2, MUL VL]
    fadd z27.h, z27.h, z19.h
    st1h { z28.h }, p1, [x7, #1, MUL VL]
    ld1h { z19.h }, p0/Z, [x21, #2, MUL VL]
    fadd z26.h, z26.h, z18.h
    ld1h { z18.h }, p0/Z, [x20, #2, MUL VL]
    fadd z25.h, z25.h, z17.h
    ld1h { z17.h }, p0/Z, [x8, #2, MUL VL]
    fadd z24.h, z24.h, z16.h
    fmin z27.h, p3/M, z27.h, z12.h
    ld1h { z16.h }, p0/Z, [x7, #2, MUL VL]
    fadd z23.h, z23.h, z19.h
    fadd z22.h, z22.h, z18.h
    fmin z26.h, p3/M, z26.h, z12.h
    fadd z21.h, z21.h, z17.h
    fmin z25.h, p3/M, z25.h, z12.h
    fmin z24.h, p3/M, z24.h, z12.h
    fmax z27.h, p3/M, z27.h, z11.h
    fadd z20.h, z20.h, z16.h
    fmin z23.h, p3/M, z23.h, z12.h
    fmin z22.h, p3/M, z22.h, z12.h
    fmax z26.h, p3/M, z26.h, z11.h
    fmin z21.h, p3/M, z21.h, z12.h
    fmax z25.h, p3/M, z25.h, z11.h
    fmin z20.h, p3/M, z20.h, z12.h
    st1h { z27.h }, p0, [x25, #2, MUL VL]
    fmax z24.h, p3/M, z24.h, z11.h
    st1h { z26.h }, p0, [x24, #2, MUL VL]
    fmax z23.h, p3/M, z23.h, z11.h
    fmax z22.h, p3/M, z22.h, z11.h
    st1h { z25.h }, p0, [x23, #2, MUL VL]
    fmax z21.h, p3/M, z21.h, z11.h
    incb x25, ALL, MUL #3
    st1h { z24.h }, p0, [x22, #2, MUL VL]
    fmax z20.h, p3/M, z20.h, z11.h
    incb x24, ALL, MUL #3
    st1h { z23.h }, p0, [x21, #2, MUL VL]
    incb x23, ALL, MUL #3
    incb x22, ALL, MUL #3
    st1h { z22.h }, p0, [x20, #2, MUL VL]
    incb x21, ALL, MUL #3
    incb x20, ALL, MUL #3
    st1h { z21.h }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    st1h { z20.h }, p0, [x7, #2, MUL VL]
    incb x7, ALL, MUL #3
    bgt label_51
    subs x6, x6, #0x8
    add x5, x5, x28
    bgt label_35
KAI_ASM_LABEL(label_52)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldr d12, [sp, 104]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_merge_fp16_fp16_8x3VL)

    KAI_ASM_END
