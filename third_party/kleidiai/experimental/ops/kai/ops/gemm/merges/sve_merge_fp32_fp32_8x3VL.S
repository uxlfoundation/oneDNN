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

    KAI_ASM_CODE(sve_merge_fp32_fp32_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_merge_fp32_fp32_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_merge_fp32_fp32_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_merge_fp32_fp32_8x3VL)
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
    KAI_ASM_INST(0x8540cc4c)  // ld1rw { z12.s }, p3/Z, [x2]
    KAI_ASM_INST(0x8540cc2b)  // ld1rw { z11.s }, p3/Z, [x1]
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
    whilelt p2.s, x1, x3
    incw x1
    whilelt p1.s, x1, x3
    incw x1
    whilelt p0.s, x1, x3
    cbnz x2, label_4
    mov z21.b, #0
    mov z20.b, #0
    mov z19.b, #0
    b label_5
KAI_ASM_LABEL(label_4)  // Initial: Height 1: Width 3: bias
    ld1w { z21.s }, p2/Z, [x2]
    ld1w { z20.s }, p1/Z, [x2, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x2, #2, MUL VL]
    incb x2, ALL, MUL #3
KAI_ASM_LABEL(label_5)  // Initial: Height 1: Width 3: init done
    ld1w { z18.s }, p2/Z, [x5]
    ld1w { z17.s }, p1/Z, [x5, #1, MUL VL]
    decw x3, ALL, MUL #3
    ld1w { z16.s }, p0/Z, [x5, #2, MUL VL]
    addvl x5, x5, #24
    fadd z18.s, z18.s, z21.s
    fadd z17.s, z17.s, z20.s
    cmp x3, XZR
    fadd z16.s, z16.s, z19.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    fmax z18.s, p3/M, z18.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    fmax z16.s, p3/M, z16.s, z11.s
    st1w { z18.s }, p2, [x25]
    st1w { z17.s }, p1, [x25, #1, MUL VL]
    st1w { z16.s }, p0, [x25, #2, MUL VL]
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
    whilelt p2.s, x1, x3
    incw x1
    whilelt p1.s, x1, x3
    incw x1
    whilelt p0.s, x1, x3
    cbnz x2, label_8
    mov z24.b, #0
    mov z23.b, #0
    mov z22.b, #0
    b label_9
KAI_ASM_LABEL(label_8)  // Initial: Height 2: Width 3: bias
    ld1w { z24.s }, p2/Z, [x2]
    ld1w { z23.s }, p1/Z, [x2, #1, MUL VL]
    ld1w { z22.s }, p0/Z, [x2, #2, MUL VL]
    incb x2, ALL, MUL #3
KAI_ASM_LABEL(label_9)  // Initial: Height 2: Width 3: init done
    ld1w { z21.s }, p2/Z, [x5]
    ld1w { z20.s }, p1/Z, [x5, #1, MUL VL]
    decw x3, ALL, MUL #3
    ld1w { z19.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z18.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z17.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z16.s }, p0/Z, [x5, #5, MUL VL]
    addvl x5, x5, #24
    fadd z21.s, z21.s, z24.s
    fadd z20.s, z20.s, z23.s
    cmp x3, XZR
    fadd z19.s, z19.s, z22.s
    fadd z18.s, z18.s, z24.s
    fadd z17.s, z17.s, z23.s
    fadd z16.s, z16.s, z22.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    fmax z18.s, p3/M, z18.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    fmax z16.s, p3/M, z16.s, z11.s
    st1w { z21.s }, p2, [x25]
    st1w { z20.s }, p1, [x25, #1, MUL VL]
    st1w { z19.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1w { z18.s }, p2, [x24]
    st1w { z17.s }, p1, [x24, #1, MUL VL]
    st1w { z16.s }, p0, [x24, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_12
    mov z27.b, #0
    mov z26.b, #0
    mov z25.b, #0
    b label_13
KAI_ASM_LABEL(label_12)  // Initial: Height 3: Width 3: bias
    ld1w { z27.s }, p2/Z, [x3]
    ld1w { z26.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z25.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_13)  // Initial: Height 3: Width 3: init done
    ld1w { z24.s }, p2/Z, [x5]
    ld1w { z23.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z22.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z21.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z20.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z19.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z18.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z17.s }, p1/Z, [x5, #7, MUL VL]
    fadd z24.s, z24.s, z27.s
    fadd z23.s, z23.s, z26.s
    ld1w { z16.s }, p0/Z, [x1, #-8, MUL VL]
    fadd z22.s, z22.s, z25.s
    fadd z21.s, z21.s, z27.s
    cmp x4, XZR
    fadd z20.s, z20.s, z26.s
    fadd z19.s, z19.s, z25.s
    addvl x5, x5, #24
    fadd z18.s, z18.s, z27.s
    fadd z17.s, z17.s, z26.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fadd z16.s, z16.s, z25.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z24.s }, p2, [x25]
    fmax z19.s, p3/M, z19.s, z11.s
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z23.s }, p1, [x25, #1, MUL VL]
    st1w { z22.s }, p0, [x25, #2, MUL VL]
    fmax z17.s, p3/M, z17.s, z11.s
    fmax z16.s, p3/M, z16.s, z11.s
    st1w { z21.s }, p2, [x24]
    incb x25, ALL, MUL #3
    st1w { z20.s }, p1, [x24, #1, MUL VL]
    st1w { z19.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z18.s }, p2, [x23]
    st1w { z17.s }, p1, [x23, #1, MUL VL]
    st1w { z16.s }, p0, [x23, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_16
    mov z30.b, #0
    mov z29.b, #0
    mov z28.b, #0
    b label_17
KAI_ASM_LABEL(label_16)  // Initial: Height 4: Width 3: bias
    ld1w { z30.s }, p2/Z, [x3]
    ld1w { z29.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z28.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_17)  // Initial: Height 4: Width 3: init done
    ld1w { z27.s }, p2/Z, [x5]
    ld1w { z26.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z25.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z24.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z23.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z22.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z21.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z20.s }, p1/Z, [x5, #7, MUL VL]
    fadd z27.s, z27.s, z30.s
    fadd z26.s, z26.s, z29.s
    ld1w { z19.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #-7, MUL VL]
    fadd z25.s, z25.s, z28.s
    fadd z24.s, z24.s, z30.s
    ld1w { z17.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z16.s }, p0/Z, [x1, #-5, MUL VL]
    fadd z23.s, z23.s, z29.s
    fadd z22.s, z22.s, z28.s
    fadd z21.s, z21.s, z30.s
    fadd z20.s, z20.s, z29.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fadd z19.s, z19.s, z28.s
    fadd z18.s, z18.s, z30.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fadd z17.s, z17.s, z29.s
    fadd z16.s, z16.s, z28.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    cmp x4, XZR
    fmax z27.s, p3/M, z27.s, z11.s
    fmax z26.s, p3/M, z26.s, z11.s
    fmax z25.s, p3/M, z25.s, z11.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    st1w { z27.s }, p2, [x25]
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    st1w { z26.s }, p1, [x25, #1, MUL VL]
    st1w { z25.s }, p0, [x25, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    st1w { z24.s }, p2, [x24]
    fmax z18.s, p3/M, z18.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z23.s }, p1, [x24, #1, MUL VL]
    fmax z16.s, p3/M, z16.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z22.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    addvl x5, x5, #24
    st1w { z21.s }, p2, [x23]
    st1w { z20.s }, p1, [x23, #1, MUL VL]
    st1w { z19.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z18.s }, p2, [x22]
    st1w { z17.s }, p1, [x22, #1, MUL VL]
    st1w { z16.s }, p0, [x22, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_20
    mov z1.b, #0
    mov z0.b, #0
    mov z31.b, #0
    b label_21
KAI_ASM_LABEL(label_20)  // Initial: Height 5: Width 3: bias
    ld1w { z1.s }, p2/Z, [x3]
    ld1w { z0.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z31.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_21)  // Initial: Height 5: Width 3: init done
    ld1w { z30.s }, p2/Z, [x5]
    ld1w { z29.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z28.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z26.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z25.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z24.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z23.s }, p1/Z, [x5, #7, MUL VL]
    fadd z30.s, z30.s, z1.s
    fadd z29.s, z29.s, z0.s
    ld1w { z22.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z21.s }, p2/Z, [x1, #-7, MUL VL]
    fadd z28.s, z28.s, z31.s
    fadd z27.s, z27.s, z1.s
    ld1w { z20.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #-5, MUL VL]
    fadd z26.s, z26.s, z0.s
    fadd z25.s, z25.s, z31.s
    ld1w { z18.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z17.s }, p1/Z, [x1, #-3, MUL VL]
    fadd z24.s, z24.s, z1.s
    fadd z23.s, z23.s, z0.s
    ld1w { z16.s }, p0/Z, [x1, #-2, MUL VL]
    fadd z22.s, z22.s, z31.s
    fadd z21.s, z21.s, z1.s
    fmin z30.s, p3/M, z30.s, z12.s
    fadd z20.s, z20.s, z0.s
    fadd z19.s, z19.s, z31.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fadd z18.s, z18.s, z1.s
    fadd z17.s, z17.s, z0.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fadd z16.s, z16.s, z31.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    cmp x4, XZR
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z27.s, p3/M, z27.s, z11.s
    fmax z26.s, p3/M, z26.s, z11.s
    st1w { z30.s }, p2, [x25]
    fmax z25.s, p3/M, z25.s, z11.s
    fmax z24.s, p3/M, z24.s, z11.s
    st1w { z29.s }, p1, [x25, #1, MUL VL]
    st1w { z28.s }, p0, [x25, #2, MUL VL]
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    st1w { z27.s }, p2, [x24]
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z26.s }, p1, [x24, #1, MUL VL]
    fmax z19.s, p3/M, z19.s, z11.s
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z25.s }, p0, [x24, #2, MUL VL]
    fmax z17.s, p3/M, z17.s, z11.s
    fmax z16.s, p3/M, z16.s, z11.s
    st1w { z24.s }, p2, [x23]
    incb x25, ALL, MUL #3
    incb x24, ALL, MUL #3
    st1w { z23.s }, p1, [x23, #1, MUL VL]
    addvl x5, x5, #24
    st1w { z22.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z21.s }, p2, [x22]
    st1w { z20.s }, p1, [x22, #1, MUL VL]
    st1w { z19.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z18.s }, p2, [x21]
    st1w { z17.s }, p1, [x21, #1, MUL VL]
    st1w { z16.s }, p0, [x21, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_24
    mov z4.b, #0
    mov z3.b, #0
    mov z2.b, #0
    b label_25
KAI_ASM_LABEL(label_24)  // Initial: Height 6: Width 3: bias
    ld1w { z4.s }, p2/Z, [x3]
    ld1w { z3.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z2.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_25)  // Initial: Height 6: Width 3: init done
    ld1w { z1.s }, p2/Z, [x5]
    ld1w { z0.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z31.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z30.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z29.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z28.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z27.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z26.s }, p1/Z, [x5, #7, MUL VL]
    fadd z1.s, z1.s, z4.s
    fadd z0.s, z0.s, z3.s
    ld1w { z25.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z24.s }, p2/Z, [x1, #-7, MUL VL]
    fadd z31.s, z31.s, z2.s
    fadd z30.s, z30.s, z4.s
    ld1w { z23.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z22.s }, p0/Z, [x1, #-5, MUL VL]
    fadd z29.s, z29.s, z3.s
    fadd z28.s, z28.s, z2.s
    ld1w { z21.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z20.s }, p1/Z, [x1, #-3, MUL VL]
    fadd z27.s, z27.s, z4.s
    fadd z26.s, z26.s, z3.s
    ld1w { z19.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #-1, MUL VL]
    fadd z25.s, z25.s, z2.s
    fadd z24.s, z24.s, z4.s
    ld1w { z17.s }, p1/Z, [x1]
    ld1w { z16.s }, p0/Z, [x1, #1, MUL VL]
    fadd z23.s, z23.s, z3.s
    fadd z22.s, z22.s, z2.s
    fadd z21.s, z21.s, z4.s
    fadd z20.s, z20.s, z3.s
    fmin z1.s, p3/M, z1.s, z12.s
    fmin z0.s, p3/M, z0.s, z12.s
    fadd z19.s, z19.s, z2.s
    fadd z18.s, z18.s, z4.s
    fmin z31.s, p3/M, z31.s, z12.s
    fmin z30.s, p3/M, z30.s, z12.s
    fadd z17.s, z17.s, z3.s
    fadd z16.s, z16.s, z2.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    cmp x4, XZR
    fmax z1.s, p3/M, z1.s, z11.s
    fmax z0.s, p3/M, z0.s, z11.s
    fmax z31.s, p3/M, z31.s, z11.s
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z29.s, p3/M, z29.s, z11.s
    st1w { z1.s }, p2, [x25]
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z27.s, p3/M, z27.s, z11.s
    st1w { z0.s }, p1, [x25, #1, MUL VL]
    st1w { z31.s }, p0, [x25, #2, MUL VL]
    fmax z26.s, p3/M, z26.s, z11.s
    fmax z25.s, p3/M, z25.s, z11.s
    st1w { z30.s }, p2, [x24]
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    st1w { z29.s }, p1, [x24, #1, MUL VL]
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    st1w { z28.s }, p0, [x24, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    st1w { z27.s }, p2, [x23]
    fmax z18.s, p3/M, z18.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z26.s }, p1, [x23, #1, MUL VL]
    fmax z16.s, p3/M, z16.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z25.s }, p0, [x23, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1w { z24.s }, p2, [x22]
    addvl x5, x5, #24
    st1w { z23.s }, p1, [x22, #1, MUL VL]
    st1w { z22.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z21.s }, p2, [x21]
    st1w { z20.s }, p1, [x21, #1, MUL VL]
    st1w { z19.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z18.s }, p2, [x20]
    st1w { z17.s }, p1, [x20, #1, MUL VL]
    st1w { z16.s }, p0, [x20, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_28
    mov z7.b, #0
    mov z6.b, #0
    mov z5.b, #0
    b label_29
KAI_ASM_LABEL(label_28)  // Initial: Height 7: Width 3: bias
    ld1w { z7.s }, p2/Z, [x3]
    ld1w { z6.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z5.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_29)  // Initial: Height 7: Width 3: init done
    ld1w { z4.s }, p2/Z, [x5]
    ld1w { z3.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z2.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z1.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z0.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z31.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z30.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z29.s }, p1/Z, [x5, #7, MUL VL]
    fadd z4.s, z4.s, z7.s
    fadd z3.s, z3.s, z6.s
    ld1w { z28.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z27.s }, p2/Z, [x1, #-7, MUL VL]
    fadd z2.s, z2.s, z5.s
    fadd z1.s, z1.s, z7.s
    ld1w { z26.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z25.s }, p0/Z, [x1, #-5, MUL VL]
    fadd z0.s, z0.s, z6.s
    fadd z31.s, z31.s, z5.s
    ld1w { z24.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #-3, MUL VL]
    fadd z30.s, z30.s, z7.s
    fadd z29.s, z29.s, z6.s
    ld1w { z22.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z21.s }, p2/Z, [x1, #-1, MUL VL]
    fadd z28.s, z28.s, z5.s
    fadd z27.s, z27.s, z7.s
    ld1w { z20.s }, p1/Z, [x1]
    ld1w { z19.s }, p0/Z, [x1, #1, MUL VL]
    fadd z26.s, z26.s, z6.s
    fadd z25.s, z25.s, z5.s
    ld1w { z18.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z17.s }, p1/Z, [x1, #3, MUL VL]
    fadd z24.s, z24.s, z7.s
    fadd z23.s, z23.s, z6.s
    ld1w { z16.s }, p0/Z, [x1, #4, MUL VL]
    fadd z22.s, z22.s, z5.s
    fadd z21.s, z21.s, z7.s
    fmin z4.s, p3/M, z4.s, z12.s
    fadd z20.s, z20.s, z6.s
    fadd z19.s, z19.s, z5.s
    fmin z3.s, p3/M, z3.s, z12.s
    fmin z2.s, p3/M, z2.s, z12.s
    fadd z18.s, z18.s, z7.s
    fadd z17.s, z17.s, z6.s
    fmin z1.s, p3/M, z1.s, z12.s
    fmin z0.s, p3/M, z0.s, z12.s
    fadd z16.s, z16.s, z5.s
    fmin z31.s, p3/M, z31.s, z12.s
    fmin z30.s, p3/M, z30.s, z12.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    cmp x4, XZR
    fmax z4.s, p3/M, z4.s, z11.s
    fmax z3.s, p3/M, z3.s, z11.s
    fmax z2.s, p3/M, z2.s, z11.s
    fmax z1.s, p3/M, z1.s, z11.s
    fmax z0.s, p3/M, z0.s, z11.s
    st1w { z4.s }, p2, [x25]
    fmax z31.s, p3/M, z31.s, z11.s
    fmax z30.s, p3/M, z30.s, z11.s
    st1w { z3.s }, p1, [x25, #1, MUL VL]
    st1w { z2.s }, p0, [x25, #2, MUL VL]
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    st1w { z1.s }, p2, [x24]
    fmax z27.s, p3/M, z27.s, z11.s
    fmax z26.s, p3/M, z26.s, z11.s
    st1w { z0.s }, p1, [x24, #1, MUL VL]
    fmax z25.s, p3/M, z25.s, z11.s
    fmax z24.s, p3/M, z24.s, z11.s
    st1w { z31.s }, p0, [x24, #2, MUL VL]
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    st1w { z30.s }, p2, [x23]
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z29.s }, p1, [x23, #1, MUL VL]
    fmax z19.s, p3/M, z19.s, z11.s
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z28.s }, p0, [x23, #2, MUL VL]
    fmax z17.s, p3/M, z17.s, z11.s
    fmax z16.s, p3/M, z16.s, z11.s
    st1w { z27.s }, p2, [x22]
    incb x25, ALL, MUL #3
    incb x24, ALL, MUL #3
    st1w { z26.s }, p1, [x22, #1, MUL VL]
    incb x23, ALL, MUL #3
    addvl x5, x5, #24
    st1w { z25.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z24.s }, p2, [x21]
    st1w { z23.s }, p1, [x21, #1, MUL VL]
    st1w { z22.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z21.s }, p2, [x20]
    st1w { z20.s }, p1, [x20, #1, MUL VL]
    st1w { z19.s }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    st1w { z18.s }, p2, [x8]
    st1w { z17.s }, p1, [x8, #1, MUL VL]
    st1w { z16.s }, p0, [x8, #2, MUL VL]
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
    whilelt p2.s, x2, x4
    incw x2
    whilelt p1.s, x2, x4
    incw x2
    whilelt p0.s, x2, x4
    cbnz x3, label_32
    mov z10.b, #0
    mov z9.b, #0
    mov z8.b, #0
    b label_33
KAI_ASM_LABEL(label_32)  // Initial: Height 8: Width 3: bias
    ld1w { z10.s }, p2/Z, [x3]
    ld1w { z9.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z8.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_33)  // Initial: Height 8: Width 3: init done
    ld1w { z7.s }, p2/Z, [x5]
    ld1w { z6.s }, p1/Z, [x5, #1, MUL VL]
    decw x4, ALL, MUL #3
    ld1w { z5.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z4.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z3.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z2.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z1.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z0.s }, p1/Z, [x5, #7, MUL VL]
    fadd z7.s, z7.s, z10.s
    fadd z6.s, z6.s, z9.s
    ld1w { z31.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z30.s }, p2/Z, [x1, #-7, MUL VL]
    fadd z5.s, z5.s, z8.s
    fadd z4.s, z4.s, z10.s
    ld1w { z29.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z28.s }, p0/Z, [x1, #-5, MUL VL]
    fadd z3.s, z3.s, z9.s
    fadd z2.s, z2.s, z8.s
    ld1w { z27.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z26.s }, p1/Z, [x1, #-3, MUL VL]
    fadd z1.s, z1.s, z10.s
    fadd z0.s, z0.s, z9.s
    ld1w { z25.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z24.s }, p2/Z, [x1, #-1, MUL VL]
    fadd z31.s, z31.s, z8.s
    fadd z30.s, z30.s, z10.s
    ld1w { z23.s }, p1/Z, [x1]
    ld1w { z22.s }, p0/Z, [x1, #1, MUL VL]
    fadd z29.s, z29.s, z9.s
    fadd z28.s, z28.s, z8.s
    ld1w { z21.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z20.s }, p1/Z, [x1, #3, MUL VL]
    fadd z27.s, z27.s, z10.s
    fadd z26.s, z26.s, z9.s
    ld1w { z19.s }, p0/Z, [x1, #4, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #5, MUL VL]
    fadd z25.s, z25.s, z8.s
    fadd z24.s, z24.s, z10.s
    ld1w { z17.s }, p1/Z, [x1, #6, MUL VL]
    ld1w { z16.s }, p0/Z, [x1, #7, MUL VL]
    fadd z23.s, z23.s, z9.s
    fadd z22.s, z22.s, z8.s
    fadd z21.s, z21.s, z10.s
    fadd z20.s, z20.s, z9.s
    fmin z7.s, p3/M, z7.s, z12.s
    fmin z6.s, p3/M, z6.s, z12.s
    fadd z19.s, z19.s, z8.s
    fadd z18.s, z18.s, z10.s
    fmin z5.s, p3/M, z5.s, z12.s
    fmin z4.s, p3/M, z4.s, z12.s
    fadd z17.s, z17.s, z9.s
    fadd z16.s, z16.s, z8.s
    fmin z3.s, p3/M, z3.s, z12.s
    fmin z2.s, p3/M, z2.s, z12.s
    fmin z1.s, p3/M, z1.s, z12.s
    fmin z0.s, p3/M, z0.s, z12.s
    fmin z31.s, p3/M, z31.s, z12.s
    fmin z30.s, p3/M, z30.s, z12.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmin z16.s, p3/M, z16.s, z12.s
    cmp x4, XZR
    fmax z7.s, p3/M, z7.s, z11.s
    fmax z6.s, p3/M, z6.s, z11.s
    fmax z5.s, p3/M, z5.s, z11.s
    fmax z4.s, p3/M, z4.s, z11.s
    fmax z3.s, p3/M, z3.s, z11.s
    st1w { z7.s }, p2, [x25]
    fmax z2.s, p3/M, z2.s, z11.s
    fmax z1.s, p3/M, z1.s, z11.s
    st1w { z6.s }, p1, [x25, #1, MUL VL]
    st1w { z5.s }, p0, [x25, #2, MUL VL]
    fmax z0.s, p3/M, z0.s, z11.s
    fmax z31.s, p3/M, z31.s, z11.s
    st1w { z4.s }, p2, [x24]
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z29.s, p3/M, z29.s, z11.s
    st1w { z3.s }, p1, [x24, #1, MUL VL]
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z27.s, p3/M, z27.s, z11.s
    st1w { z2.s }, p0, [x24, #2, MUL VL]
    fmax z26.s, p3/M, z26.s, z11.s
    fmax z25.s, p3/M, z25.s, z11.s
    st1w { z1.s }, p2, [x23]
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    st1w { z0.s }, p1, [x23, #1, MUL VL]
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    st1w { z31.s }, p0, [x23, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    st1w { z30.s }, p2, [x22]
    fmax z18.s, p3/M, z18.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z29.s }, p1, [x22, #1, MUL VL]
    fmax z16.s, p3/M, z16.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z28.s }, p0, [x22, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1w { z27.s }, p2, [x21]
    incb x22, ALL, MUL #3
    addvl x5, x5, #24
    st1w { z26.s }, p1, [x21, #1, MUL VL]
    st1w { z25.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z24.s }, p2, [x20]
    st1w { z23.s }, p1, [x20, #1, MUL VL]
    st1w { z22.s }, p0, [x20, #2, MUL VL]
    incb x20, ALL, MUL #3
    st1w { z21.s }, p2, [x8]
    st1w { z20.s }, p1, [x8, #1, MUL VL]
    st1w { z19.s }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    st1w { z18.s }, p2, [x7]
    st1w { z17.s }, p1, [x7, #1, MUL VL]
    st1w { z16.s }, p0, [x7, #2, MUL VL]
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
    whilelt p0.s, x1, x2
    ld1w { z17.s }, p0/Z, [x4]
    ld1w { z16.s }, p0/Z, [x25]
    fadd z17.s, z17.s, z16.s
    incw x1
    fmin z17.s, p3/M, z17.s, z12.s
    whilelt p1.s, x1, x2
    fmax z17.s, p3/M, z17.s, z11.s
    incw x1
    st1w { z17.s }, p0, [x25]
    ld1w { z18.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]
    fadd z18.s, z18.s, z16.s
    whilelt p0.s, x1, x2
    decw x2, ALL, MUL #3
    fmin z18.s, p3/M, z18.s, z12.s
    ld1w { z17.s }, p0/Z, [x4, #2, MUL VL]
    addvl x4, x4, #24
    cmp x2, XZR
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z18.s }, p1, [x25, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]
    fadd z17.s, z17.s, z16.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z17.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    bgt label_37
    b label_52
KAI_ASM_LABEL(label_38)  // Accumulate: Height 2
    mov x25, x5
    ldr x2, [x0, #0x8]
    add x24, x25, x27
KAI_ASM_LABEL(label_39)  // Accumulate: Height 2: Block loop
    mov x1, #0
    whilelt p2.s, x1, x2
    ld1w { z19.s }, p2/Z, [x4]
    ld1w { z18.s }, p2/Z, [x25]
    ld1w { z17.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p2/Z, [x24]
    fadd z19.s, z19.s, z18.s
    incw x1
    fadd z17.s, z17.s, z16.s
    fmin z19.s, p3/M, z19.s, z12.s
    whilelt p1.s, x1, x2
    incw x1
    fmin z17.s, p3/M, z17.s, z12.s
    fmax z19.s, p3/M, z19.s, z11.s
    ld1w { z21.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z20.s }, p1/Z, [x4, #4, MUL VL]
    fmax z17.s, p3/M, z17.s, z11.s
    whilelt p0.s, x1, x2
    decw x2, ALL, MUL #3
    st1w { z19.s }, p2, [x25]
    st1w { z17.s }, p2, [x24]
    ld1w { z17.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    cmp x2, XZR
    ld1w { z16.s }, p1/Z, [x24, #1, MUL VL]
    fadd z21.s, z21.s, z17.s
    fadd z20.s, z20.s, z16.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z21.s }, p1, [x25, #1, MUL VL]
    st1w { z20.s }, p1, [x24, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    fadd z19.s, z19.s, z17.s
    fadd z18.s, z18.s, z16.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z18.s, p3/M, z18.s, z12.s
    fmax z19.s, p3/M, z19.s, z11.s
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z19.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1w { z18.s }, p0, [x24, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z20.s }, p2/Z, [x4]
    ld1w { z19.s }, p2/Z, [x25]
    ld1w { z25.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x24]
    ld1w { z17.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z16.s }, p2/Z, [x23]
    fadd z20.s, z20.s, z19.s
    whilelt p1.s, x2, x3
    incw x2
    fadd z25.s, z25.s, z18.s
    fadd z17.s, z17.s, z16.s
    fmin z20.s, p3/M, z20.s, z12.s
    ld1w { z24.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z23.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z22.s }, p1/Z, [x4, #7, MUL VL]
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmax z20.s, p3/M, z20.s, z11.s
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    fmax z25.s, p3/M, z25.s, z11.s
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z20.s }, p2, [x25]
    ld1w { z21.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z20.s }, p0/Z, [x4, #5, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #-8, MUL VL]
    addvl x4, x4, #24
    st1w { z25.s }, p2, [x24]
    cmp x3, XZR
    st1w { z17.s }, p2, [x23]
    ld1w { z18.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #1, MUL VL]
    fadd z24.s, z24.s, z18.s
    fadd z23.s, z23.s, z17.s
    fadd z22.s, z22.s, z16.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    st1w { z24.s }, p1, [x25, #1, MUL VL]
    fmax z22.s, p3/M, z22.s, z11.s
    st1w { z23.s }, p1, [x24, #1, MUL VL]
    st1w { z22.s }, p1, [x23, #1, MUL VL]
    ld1w { z18.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z17.s }, p0/Z, [x24, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x23, #2, MUL VL]
    fadd z21.s, z21.s, z18.s
    fadd z20.s, z20.s, z17.s
    fadd z19.s, z19.s, z16.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z21.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    fmax z19.s, p3/M, z19.s, z11.s
    st1w { z20.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z19.s }, p0, [x23, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z29.s }, p2/Z, [x4]
    ld1w { z21.s }, p2/Z, [x25]
    ld1w { z28.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z20.s }, p2/Z, [x24]
    ld1w { z19.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z18.s }, p2/Z, [x23]
    ld1w { z17.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z16.s }, p2/Z, [x22]
    fadd z29.s, z29.s, z21.s
    whilelt p1.s, x2, x3
    fadd z28.s, z28.s, z20.s
    incw x2
    fadd z19.s, z19.s, z18.s
    fadd z17.s, z17.s, z16.s
    fmin z29.s, p3/M, z29.s, z12.s
    ld1w { z27.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z26.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z25.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z24.s }, p1/Z, [x1, #-6, MUL VL]
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    fmin z17.s, p3/M, z17.s, z12.s
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    ld1w { z23.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z22.s }, p0/Z, [x4, #5, MUL VL]
    ld1w { z21.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z20.s }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    addvl x4, x4, #24
    st1w { z29.s }, p2, [x25]
    fmax z17.s, p3/M, z17.s, z11.s
    st1w { z28.s }, p2, [x24]
    st1w { z19.s }, p2, [x23]
    st1w { z17.s }, p2, [x22]
    ld1w { z19.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z17.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x22, #1, MUL VL]
    fadd z27.s, z27.s, z19.s
    fadd z26.s, z26.s, z18.s
    fadd z25.s, z25.s, z17.s
    fadd z24.s, z24.s, z16.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmax z27.s, p3/M, z27.s, z11.s
    fmax z26.s, p3/M, z26.s, z11.s
    fmax z25.s, p3/M, z25.s, z11.s
    fmax z24.s, p3/M, z24.s, z11.s
    st1w { z27.s }, p1, [x25, #1, MUL VL]
    st1w { z26.s }, p1, [x24, #1, MUL VL]
    st1w { z25.s }, p1, [x23, #1, MUL VL]
    st1w { z24.s }, p1, [x22, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x24, #2, MUL VL]
    ld1w { z17.s }, p0/Z, [x23, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x22, #2, MUL VL]
    fadd z23.s, z23.s, z19.s
    fadd z22.s, z22.s, z18.s
    fadd z21.s, z21.s, z17.s
    fadd z20.s, z20.s, z16.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z23.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1w { z22.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z21.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z20.s }, p0, [x22, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z0.s }, p2/Z, [x4]
    ld1w { z16.s }, p2/Z, [x25]
    ld1w { z31.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z21.s }, p2/Z, [x24]
    ld1w { z30.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z20.s }, p2/Z, [x23]
    ld1w { z19.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z18.s }, p2/Z, [x22]
    fadd z0.s, z0.s, z16.s
    whilelt p1.s, x2, x3
    ld1w { z17.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z16.s }, p2/Z, [x21]
    fadd z31.s, z31.s, z21.s
    incw x2
    fadd z30.s, z30.s, z20.s
    fadd z19.s, z19.s, z18.s
    fmin z0.s, p3/M, z0.s, z12.s
    ld1w { z29.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z27.s }, p1/Z, [x4, #7, MUL VL]
    fadd z17.s, z17.s, z16.s
    fmin z31.s, p3/M, z31.s, z12.s
    ld1w { z26.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z25.s }, p1/Z, [x1, #-3, MUL VL]
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    fmin z30.s, p3/M, z30.s, z12.s
    fmin z19.s, p3/M, z19.s, z12.s
    fmin z17.s, p3/M, z17.s, z12.s
    fmax z0.s, p3/M, z0.s, z11.s
    fmax z31.s, p3/M, z31.s, z11.s
    ld1w { z24.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z23.s }, p0/Z, [x4, #5, MUL VL]
    cmp x3, XZR
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    ld1w { z22.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z21.s }, p0/Z, [x1, #-5, MUL VL]
    st1w { z0.s }, p2, [x25]
    fmax z17.s, p3/M, z17.s, z11.s
    ld1w { z20.s }, p0/Z, [x1, #-2, MUL VL]
    addvl x4, x4, #24
    st1w { z31.s }, p2, [x24]
    st1w { z30.s }, p2, [x23]
    st1w { z19.s }, p2, [x22]
    ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]
    st1w { z17.s }, p2, [x21]
    ld1w { z19.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z17.s }, p1/Z, [x22, #1, MUL VL]
    fadd z29.s, z29.s, z16.s
    ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]
    fadd z28.s, z28.s, z19.s
    fadd z27.s, z27.s, z18.s
    fadd z26.s, z26.s, z17.s
    fadd z25.s, z25.s, z16.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z27.s, p3/M, z27.s, z11.s
    fmax z26.s, p3/M, z26.s, z11.s
    st1w { z29.s }, p1, [x25, #1, MUL VL]
    fmax z25.s, p3/M, z25.s, z11.s
    st1w { z28.s }, p1, [x24, #1, MUL VL]
    st1w { z27.s }, p1, [x23, #1, MUL VL]
    st1w { z26.s }, p1, [x22, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z25.s }, p1, [x21, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x24, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x23, #2, MUL VL]
    ld1w { z17.s }, p0/Z, [x22, #2, MUL VL]
    fadd z24.s, z24.s, z16.s
    ld1w { z16.s }, p0/Z, [x21, #2, MUL VL]
    fadd z23.s, z23.s, z19.s
    fadd z22.s, z22.s, z18.s
    fadd z21.s, z21.s, z17.s
    fadd z20.s, z20.s, z16.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    st1w { z24.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    fmax z20.s, p3/M, z20.s, z11.s
    st1w { z23.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z22.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z21.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z20.s }, p0, [x21, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z3.s }, p2/Z, [x4]
    ld1w { z17.s }, p2/Z, [x25]
    ld1w { z2.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p2/Z, [x24]
    ld1w { z1.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z21.s }, p2/Z, [x23]
    ld1w { z0.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z20.s }, p2/Z, [x22]
    fadd z3.s, z3.s, z17.s
    whilelt p1.s, x2, x3
    ld1w { z19.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z17.s }, p2/Z, [x21]
    fadd z2.s, z2.s, z16.s
    incw x2
    ld1w { z18.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z16.s }, p2/Z, [x20]
    fadd z1.s, z1.s, z21.s
    fadd z0.s, z0.s, z20.s
    fmin z3.s, p3/M, z3.s, z12.s
    ld1w { z31.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z30.s }, p1/Z, [x4, #4, MUL VL]
    fadd z19.s, z19.s, z17.s
    fmin z2.s, p3/M, z2.s, z12.s
    ld1w { z29.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z28.s }, p1/Z, [x1, #-6, MUL VL]
    whilelt p0.s, x2, x3
    fadd z18.s, z18.s, z16.s
    fmin z1.s, p3/M, z1.s, z12.s
    fmin z0.s, p3/M, z0.s, z12.s
    ld1w { z27.s }, p1/Z, [x1, #-3, MUL VL]
    fmin z19.s, p3/M, z19.s, z12.s
    fmax z3.s, p3/M, z3.s, z11.s
    ld1w { z26.s }, p1/Z, [x1]
    fmax z2.s, p3/M, z2.s, z11.s
    ld1w { z25.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z24.s }, p0/Z, [x4, #5, MUL VL]
    decw x3, ALL, MUL #3
    fmin z18.s, p3/M, z18.s, z12.s
    fmax z1.s, p3/M, z1.s, z11.s
    ld1w { z23.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z22.s }, p0/Z, [x1, #-5, MUL VL]
    st1w { z3.s }, p2, [x25]
    fmax z0.s, p3/M, z0.s, z11.s
    fmax z19.s, p3/M, z19.s, z11.s
    ld1w { z21.s }, p0/Z, [x1, #-2, MUL VL]
    st1w { z2.s }, p2, [x24]
    ld1w { z20.s }, p0/Z, [x1, #1, MUL VL]
    cmp x3, XZR
    addvl x4, x4, #24
    st1w { z1.s }, p2, [x23]
    fmax z18.s, p3/M, z18.s, z11.s
    st1w { z0.s }, p2, [x22]
    ld1w { z17.s }, p1/Z, [x25, #1, MUL VL]
    st1w { z19.s }, p2, [x21]
    ld1w { z16.s }, p1/Z, [x24, #1, MUL VL]
    st1w { z18.s }, p2, [x20]
    ld1w { z19.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z18.s }, p1/Z, [x22, #1, MUL VL]
    fadd z31.s, z31.s, z17.s
    ld1w { z17.s }, p1/Z, [x21, #1, MUL VL]
    fadd z30.s, z30.s, z16.s
    ld1w { z16.s }, p1/Z, [x20, #1, MUL VL]
    fadd z29.s, z29.s, z19.s
    fadd z28.s, z28.s, z18.s
    fmin z31.s, p3/M, z31.s, z12.s
    fadd z27.s, z27.s, z17.s
    fadd z26.s, z26.s, z16.s
    fmin z30.s, p3/M, z30.s, z12.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmax z31.s, p3/M, z31.s, z11.s
    fmin z26.s, p3/M, z26.s, z12.s
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    fmax z27.s, p3/M, z27.s, z11.s
    st1w { z31.s }, p1, [x25, #1, MUL VL]
    fmax z26.s, p3/M, z26.s, z11.s
    st1w { z30.s }, p1, [x24, #1, MUL VL]
    st1w { z29.s }, p1, [x23, #1, MUL VL]
    st1w { z28.s }, p1, [x22, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z27.s }, p1, [x21, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z26.s }, p1, [x20, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x23, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x22, #2, MUL VL]
    fadd z25.s, z25.s, z17.s
    ld1w { z17.s }, p0/Z, [x21, #2, MUL VL]
    fadd z24.s, z24.s, z16.s
    ld1w { z16.s }, p0/Z, [x20, #2, MUL VL]
    fadd z23.s, z23.s, z19.s
    fadd z22.s, z22.s, z18.s
    fmin z25.s, p3/M, z25.s, z12.s
    fadd z21.s, z21.s, z17.s
    fadd z20.s, z20.s, z16.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmax z25.s, p3/M, z25.s, z11.s
    fmin z20.s, p3/M, z20.s, z12.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    fmax z21.s, p3/M, z21.s, z11.s
    st1w { z25.s }, p0, [x25, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z24.s }, p0, [x24, #2, MUL VL]
    st1w { z23.s }, p0, [x23, #2, MUL VL]
    incb x24, ALL, MUL #3
    incb x23, ALL, MUL #3
    st1w { z22.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z21.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z20.s }, p0, [x20, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z26.s }, p2/Z, [x4]
    ld1w { z18.s }, p2/Z, [x25]
    ld1w { z25.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z17.s }, p2/Z, [x24]
    ld1w { z24.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z16.s }, p2/Z, [x23]
    ld1w { z23.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z22.s }, p2/Z, [x22]
    fadd z26.s, z26.s, z18.s
    whilelt p1.s, x2, x3
    ld1w { z21.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z18.s }, p2/Z, [x21]
    fadd z25.s, z25.s, z17.s
    incw x2
    ld1w { z20.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z17.s }, p2/Z, [x20]
    fadd z24.s, z24.s, z16.s
    ld1w { z19.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z16.s }, p2/Z, [x8]
    fadd z23.s, z23.s, z22.s
    fmin z26.s, p3/M, z26.s, z12.s
    fadd z21.s, z21.s, z18.s
    fmin z25.s, p3/M, z25.s, z12.s
    ld1w { z1.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z0.s }, p1/Z, [x4, #4, MUL VL]
    fadd z20.s, z20.s, z17.s
    fmin z24.s, p3/M, z24.s, z12.s
    ld1w { z31.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z30.s }, p1/Z, [x1, #-6, MUL VL]
    fadd z19.s, z19.s, z16.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmax z26.s, p3/M, z26.s, z11.s
    ld1w { z29.s }, p1/Z, [x1, #-3, MUL VL]
    fmin z21.s, p3/M, z21.s, z12.s
    fmax z25.s, p3/M, z25.s, z11.s
    ld1w { z28.s }, p1/Z, [x1]
    ld1w { z27.s }, p1/Z, [x1, #3, MUL VL]
    fmin z20.s, p3/M, z20.s, z12.s
    fmax z24.s, p3/M, z24.s, z11.s
    fmin z19.s, p3/M, z19.s, z12.s
    st1w { z26.s }, p2, [x25]
    fmax z23.s, p3/M, z23.s, z11.s
    st1w { z25.s }, p2, [x24]
    fmax z21.s, p3/M, z21.s, z11.s
    whilelt p0.s, x2, x3
    st1w { z24.s }, p2, [x23]
    fmax z20.s, p3/M, z20.s, z11.s
    decw x3, ALL, MUL #3
    st1w { z23.s }, p2, [x22]
    fmax z19.s, p3/M, z19.s, z11.s
    ld1w { z18.s }, p1/Z, [x25, #1, MUL VL]
    st1w { z21.s }, p2, [x21]
    ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z26.s }, p0/Z, [x4, #2, MUL VL]
    st1w { z20.s }, p2, [x20]
    ld1w { z16.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z25.s }, p0/Z, [x4, #5, MUL VL]
    cmp x3, XZR
    st1w { z19.s }, p2, [x8]
    ld1w { z19.s }, p1/Z, [x22, #1, MUL VL]
    fadd z1.s, z1.s, z18.s
    ld1w { z24.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z18.s }, p1/Z, [x21, #1, MUL VL]
    fadd z0.s, z0.s, z17.s
    ld1w { z23.s }, p0/Z, [x1, #-5, MUL VL]
    addvl x4, x4, #24
    ld1w { z17.s }, p1/Z, [x20, #1, MUL VL]
    fadd z31.s, z31.s, z16.s
    ld1w { z22.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z16.s }, p1/Z, [x8, #1, MUL VL]
    fadd z30.s, z30.s, z19.s
    fmin z1.s, p3/M, z1.s, z12.s
    ld1w { z21.s }, p0/Z, [x1, #1, MUL VL]
    fadd z29.s, z29.s, z18.s
    fmin z0.s, p3/M, z0.s, z12.s
    ld1w { z20.s }, p0/Z, [x1, #4, MUL VL]
    fadd z28.s, z28.s, z17.s
    fmin z31.s, p3/M, z31.s, z12.s
    fadd z27.s, z27.s, z16.s
    fmin z30.s, p3/M, z30.s, z12.s
    fmax z1.s, p3/M, z1.s, z11.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    fmax z0.s, p3/M, z0.s, z11.s
    fmin z27.s, p3/M, z27.s, z12.s
    fmax z31.s, p3/M, z31.s, z11.s
    st1w { z1.s }, p1, [x25, #1, MUL VL]
    fmax z30.s, p3/M, z30.s, z11.s
    fmax z29.s, p3/M, z29.s, z11.s
    st1w { z0.s }, p1, [x24, #1, MUL VL]
    fmax z28.s, p3/M, z28.s, z11.s
    st1w { z31.s }, p1, [x23, #1, MUL VL]
    fmax z27.s, p3/M, z27.s, z11.s
    ld1w { z18.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z30.s }, p1, [x22, #1, MUL VL]
    st1w { z29.s }, p1, [x21, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z28.s }, p1, [x20, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x23, #2, MUL VL]
    fadd z26.s, z26.s, z18.s
    st1w { z27.s }, p1, [x8, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x22, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x21, #2, MUL VL]
    fadd z25.s, z25.s, z17.s
    ld1w { z17.s }, p0/Z, [x20, #2, MUL VL]
    fadd z24.s, z24.s, z16.s
    fmin z26.s, p3/M, z26.s, z12.s
    ld1w { z16.s }, p0/Z, [x8, #2, MUL VL]
    fadd z23.s, z23.s, z19.s
    fadd z22.s, z22.s, z18.s
    fmin z25.s, p3/M, z25.s, z12.s
    fadd z21.s, z21.s, z17.s
    fadd z20.s, z20.s, z16.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmax z26.s, p3/M, z26.s, z11.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmax z25.s, p3/M, z25.s, z11.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmin z20.s, p3/M, z20.s, z12.s
    st1w { z26.s }, p0, [x25, #2, MUL VL]
    fmax z24.s, p3/M, z24.s, z11.s
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    st1w { z25.s }, p0, [x24, #2, MUL VL]
    fmax z21.s, p3/M, z21.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z24.s }, p0, [x23, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    incb x24, ALL, MUL #3
    st1w { z23.s }, p0, [x22, #2, MUL VL]
    st1w { z22.s }, p0, [x21, #2, MUL VL]
    incb x23, ALL, MUL #3
    incb x22, ALL, MUL #3
    st1w { z21.s }, p0, [x20, #2, MUL VL]
    incb x21, ALL, MUL #3
    incb x20, ALL, MUL #3
    st1w { z20.s }, p0, [x8, #2, MUL VL]
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
    whilelt p2.s, x2, x3
    incw x2
    ld1w { z27.s }, p2/Z, [x4]
    ld1w { z19.s }, p2/Z, [x25]
    ld1w { z26.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x24]
    ld1w { z25.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z17.s }, p2/Z, [x23]
    ld1w { z24.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z16.s }, p2/Z, [x22]
    fadd z27.s, z27.s, z19.s
    whilelt p1.s, x2, x3
    ld1w { z23.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z19.s }, p2/Z, [x21]
    fadd z26.s, z26.s, z18.s
    incw x2
    ld1w { z22.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z18.s }, p2/Z, [x20]
    fadd z25.s, z25.s, z17.s
    ld1w { z21.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z17.s }, p2/Z, [x8]
    fadd z24.s, z24.s, z16.s
    fmin z27.s, p3/M, z27.s, z12.s
    ld1w { z20.s }, p2/Z, [x1, #5, MUL VL]
    ld1w { z16.s }, p2/Z, [x7]
    fadd z23.s, z23.s, z19.s
    fmin z26.s, p3/M, z26.s, z12.s
    fadd z22.s, z22.s, z18.s
    fmin z25.s, p3/M, z25.s, z12.s
    ld1w { z3.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z2.s }, p1/Z, [x4, #4, MUL VL]
    fadd z21.s, z21.s, z17.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmax z27.s, p3/M, z27.s, z11.s
    ld1w { z1.s }, p1/Z, [x4, #7, MUL VL]
    fadd z20.s, z20.s, z16.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmax z26.s, p3/M, z26.s, z11.s
    ld1w { z0.s }, p1/Z, [x1, #-6, MUL VL]
    fmin z22.s, p3/M, z22.s, z12.s
    fmax z25.s, p3/M, z25.s, z11.s
    ld1w { z31.s }, p1/Z, [x1, #-3, MUL VL]
    ld1w { z30.s }, p1/Z, [x1]
    fmin z21.s, p3/M, z21.s, z12.s
    st1w { z27.s }, p2, [x25]
    fmax z24.s, p3/M, z24.s, z11.s
    ld1w { z29.s }, p1/Z, [x1, #3, MUL VL]
    fmin z20.s, p3/M, z20.s, z12.s
    st1w { z26.s }, p2, [x24]
    fmax z23.s, p3/M, z23.s, z11.s
    ld1w { z28.s }, p1/Z, [x1, #6, MUL VL]
    st1w { z25.s }, p2, [x23]
    fmax z22.s, p3/M, z22.s, z11.s
    whilelt p0.s, x2, x3
    st1w { z24.s }, p2, [x22]
    fmax z21.s, p3/M, z21.s, z11.s
    ld1w { z19.s }, p1/Z, [x25, #1, MUL VL]
    decw x3, ALL, MUL #3
    st1w { z23.s }, p2, [x21]
    fmax z20.s, p3/M, z20.s, z11.s
    ld1w { z18.s }, p1/Z, [x24, #1, MUL VL]
    st1w { z22.s }, p2, [x20]
    ld1w { z17.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z27.s }, p0/Z, [x4, #2, MUL VL]
    st1w { z21.s }, p2, [x8]
    ld1w { z16.s }, p1/Z, [x22, #1, MUL VL]
    fadd z3.s, z3.s, z19.s
    ld1w { z26.s }, p0/Z, [x4, #5, MUL VL]
    st1w { z20.s }, p2, [x7]
    ld1w { z19.s }, p1/Z, [x21, #1, MUL VL]
    fadd z2.s, z2.s, z18.s
    ld1w { z25.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z18.s }, p1/Z, [x20, #1, MUL VL]
    fadd z1.s, z1.s, z17.s
    ld1w { z24.s }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    ld1w { z17.s }, p1/Z, [x8, #1, MUL VL]
    fadd z0.s, z0.s, z16.s
    fmin z3.s, p3/M, z3.s, z12.s
    ld1w { z23.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z16.s }, p1/Z, [x7, #1, MUL VL]
    fadd z31.s, z31.s, z19.s
    fmin z2.s, p3/M, z2.s, z12.s
    ld1w { z22.s }, p0/Z, [x1, #1, MUL VL]
    fadd z30.s, z30.s, z18.s
    fmin z1.s, p3/M, z1.s, z12.s
    ld1w { z21.s }, p0/Z, [x1, #4, MUL VL]
    ld1w { z20.s }, p0/Z, [x1, #7, MUL VL]
    fadd z29.s, z29.s, z17.s
    fmin z0.s, p3/M, z0.s, z12.s
    fmax z3.s, p3/M, z3.s, z11.s
    fadd z28.s, z28.s, z16.s
    fmin z31.s, p3/M, z31.s, z12.s
    fmax z2.s, p3/M, z2.s, z11.s
    fmin z30.s, p3/M, z30.s, z12.s
    fmax z1.s, p3/M, z1.s, z11.s
    fmin z29.s, p3/M, z29.s, z12.s
    fmin z28.s, p3/M, z28.s, z12.s
    st1w { z3.s }, p1, [x25, #1, MUL VL]
    st1w { z2.s }, p1, [x24, #1, MUL VL]
    fmax z0.s, p3/M, z0.s, z11.s
    fmax z31.s, p3/M, z31.s, z11.s
    st1w { z1.s }, p1, [x23, #1, MUL VL]
    fmax z30.s, p3/M, z30.s, z11.s
    addvl x4, x4, #24
    fmax z29.s, p3/M, z29.s, z11.s
    fmax z28.s, p3/M, z28.s, z11.s
    ld1w { z19.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z0.s }, p1, [x22, #1, MUL VL]
    st1w { z31.s }, p1, [x21, #1, MUL VL]
    ld1w { z18.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z30.s }, p1, [x20, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x23, #2, MUL VL]
    st1w { z29.s }, p1, [x8, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x22, #2, MUL VL]
    fadd z27.s, z27.s, z19.s
    st1w { z28.s }, p1, [x7, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x21, #2, MUL VL]
    fadd z26.s, z26.s, z18.s
    ld1w { z18.s }, p0/Z, [x20, #2, MUL VL]
    fadd z25.s, z25.s, z17.s
    ld1w { z17.s }, p0/Z, [x8, #2, MUL VL]
    fadd z24.s, z24.s, z16.s
    fmin z27.s, p3/M, z27.s, z12.s
    ld1w { z16.s }, p0/Z, [x7, #2, MUL VL]
    fadd z23.s, z23.s, z19.s
    fadd z22.s, z22.s, z18.s
    fmin z26.s, p3/M, z26.s, z12.s
    fadd z21.s, z21.s, z17.s
    fmin z25.s, p3/M, z25.s, z12.s
    fmin z24.s, p3/M, z24.s, z12.s
    fmax z27.s, p3/M, z27.s, z11.s
    fadd z20.s, z20.s, z16.s
    fmin z23.s, p3/M, z23.s, z12.s
    fmin z22.s, p3/M, z22.s, z12.s
    fmax z26.s, p3/M, z26.s, z11.s
    fmin z21.s, p3/M, z21.s, z12.s
    fmax z25.s, p3/M, z25.s, z11.s
    fmin z20.s, p3/M, z20.s, z12.s
    st1w { z27.s }, p0, [x25, #2, MUL VL]
    fmax z24.s, p3/M, z24.s, z11.s
    st1w { z26.s }, p0, [x24, #2, MUL VL]
    fmax z23.s, p3/M, z23.s, z11.s
    fmax z22.s, p3/M, z22.s, z11.s
    st1w { z25.s }, p0, [x23, #2, MUL VL]
    fmax z21.s, p3/M, z21.s, z11.s
    incb x25, ALL, MUL #3
    st1w { z24.s }, p0, [x22, #2, MUL VL]
    fmax z20.s, p3/M, z20.s, z11.s
    incb x24, ALL, MUL #3
    st1w { z23.s }, p0, [x21, #2, MUL VL]
    incb x23, ALL, MUL #3
    incb x22, ALL, MUL #3
    st1w { z22.s }, p0, [x20, #2, MUL VL]
    incb x21, ALL, MUL #3
    incb x20, ALL, MUL #3
    st1w { z21.s }, p0, [x8, #2, MUL VL]
    incb x8, ALL, MUL #3
    st1w { z20.s }, p0, [x7, #2, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_merge_fp32_fp32_8x3VL)

    KAI_ASM_END
