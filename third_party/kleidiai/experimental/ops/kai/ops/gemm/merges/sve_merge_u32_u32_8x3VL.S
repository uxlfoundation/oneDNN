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

    KAI_ASM_CODE(sve_merge_u32_u32_8x3VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_merge_u32_u32_8x3VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_merge_u32_u32_8x3VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_merge_u32_u32_8x3VL)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x28, #0x8
    ldr x27, [x0, #0x58]
    ldr x1, [x0, #0x60]
    mul x28, x27, x28
    cbnz x1, label_34
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
    decw x3, ALL, MUL #3
    ld1w { z18.s }, p2/Z, [x5]
    ld1w { z17.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x5, #2, MUL VL]
    addvl x5, x5, #24
    cmp x3, XZR
    add z18.s, z18.s, z21.s
    add z17.s, z17.s, z20.s
    st1w { z18.s }, p2, [x25]
    add z16.s, z16.s, z19.s
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
    decw x3, ALL, MUL #3
    ld1w { z21.s }, p2/Z, [x5]
    ld1w { z20.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z18.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z17.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z16.s }, p0/Z, [x5, #5, MUL VL]
    addvl x5, x5, #24
    cmp x3, XZR
    add z21.s, z21.s, z24.s
    add z20.s, z20.s, z23.s
    add z19.s, z19.s, z22.s
    add z18.s, z18.s, z24.s
    add z17.s, z17.s, z23.s
    add z16.s, z16.s, z22.s
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
    decw x4, ALL, MUL #3
    ld1w { z24.s }, p2/Z, [x5]
    ld1w { z23.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z22.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z21.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z20.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z19.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z18.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z17.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z24.s, z24.s, z27.s
    ld1w { z16.s }, p0/Z, [x1, #-8, MUL VL]
    add z23.s, z23.s, z26.s
    add z22.s, z22.s, z25.s
    addvl x5, x5, #24
    add z21.s, z21.s, z27.s
    add z20.s, z20.s, z26.s
    st1w { z24.s }, p2, [x25]
    add z19.s, z19.s, z25.s
    add z18.s, z18.s, z27.s
    st1w { z23.s }, p1, [x25, #1, MUL VL]
    st1w { z22.s }, p0, [x25, #2, MUL VL]
    add z17.s, z17.s, z26.s
    add z16.s, z16.s, z25.s
    incb x25, ALL, MUL #3
    st1w { z21.s }, p2, [x24]
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
    decw x4, ALL, MUL #3
    ld1w { z27.s }, p2/Z, [x5]
    ld1w { z26.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z25.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z24.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z23.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z22.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z21.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z20.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z27.s, z27.s, z30.s
    ld1w { z19.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #-7, MUL VL]
    add z26.s, z26.s, z29.s
    add z25.s, z25.s, z28.s
    ld1w { z17.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z16.s }, p0/Z, [x1, #-5, MUL VL]
    add z24.s, z24.s, z30.s
    add z23.s, z23.s, z29.s
    st1w { z27.s }, p2, [x25]
    add z22.s, z22.s, z28.s
    add z21.s, z21.s, z30.s
    addvl x5, x5, #24
    st1w { z26.s }, p1, [x25, #1, MUL VL]
    add z20.s, z20.s, z29.s
    add z19.s, z19.s, z28.s
    st1w { z25.s }, p0, [x25, #2, MUL VL]
    add z18.s, z18.s, z30.s
    add z17.s, z17.s, z29.s
    incb x25, ALL, MUL #3
    st1w { z24.s }, p2, [x24]
    add z16.s, z16.s, z28.s
    st1w { z23.s }, p1, [x24, #1, MUL VL]
    st1w { z22.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
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
    mov z0.b, #0
    mov z31.b, #0
    mov z30.b, #0
    b label_21
KAI_ASM_LABEL(label_20)  // Initial: Height 5: Width 3: bias
    ld1w { z0.s }, p2/Z, [x3]
    ld1w { z31.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z30.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_21)  // Initial: Height 5: Width 3: init done
    decw x4, ALL, MUL #3
    ld1w { z16.s }, p2/Z, [x5]
    ld1w { z29.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z28.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z27.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z26.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z25.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z24.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z23.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z16.s, z16.s, z0.s
    ld1w { z22.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z21.s }, p2/Z, [x1, #-7, MUL VL]
    add z29.s, z29.s, z31.s
    add z28.s, z28.s, z30.s
    ld1w { z20.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #-5, MUL VL]
    add z27.s, z27.s, z0.s
    add z26.s, z26.s, z31.s
    ld1w { z18.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z17.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z16.s }, p2, [x25]
    add z25.s, z25.s, z30.s
    ld1w { z16.s }, p0/Z, [x1, #-2, MUL VL]
    st1w { z29.s }, p1, [x25, #1, MUL VL]
    add z24.s, z24.s, z0.s
    add z23.s, z23.s, z31.s
    st1w { z28.s }, p0, [x25, #2, MUL VL]
    add z22.s, z22.s, z30.s
    add z21.s, z21.s, z0.s
    incb x25, ALL, MUL #3
    st1w { z27.s }, p2, [x24]
    add z20.s, z20.s, z31.s
    add z19.s, z19.s, z30.s
    addvl x5, x5, #24
    st1w { z26.s }, p1, [x24, #1, MUL VL]
    add z18.s, z18.s, z0.s
    add z17.s, z17.s, z31.s
    st1w { z25.s }, p0, [x24, #2, MUL VL]
    add z16.s, z16.s, z30.s
    incb x24, ALL, MUL #3
    st1w { z24.s }, p2, [x23]
    st1w { z23.s }, p1, [x23, #1, MUL VL]
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
    mov z2.b, #0
    mov z1.b, #0
    mov z0.b, #0
    b label_25
KAI_ASM_LABEL(label_24)  // Initial: Height 6: Width 3: bias
    ld1w { z2.s }, p2/Z, [x3]
    ld1w { z1.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z0.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_25)  // Initial: Height 6: Width 3: init done
    decw x4, ALL, MUL #3
    ld1w { z17.s }, p2/Z, [x5]
    ld1w { z16.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z31.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z30.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z29.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z28.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z27.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z26.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z17.s, z17.s, z2.s
    ld1w { z25.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z24.s }, p2/Z, [x1, #-7, MUL VL]
    add z16.s, z16.s, z1.s
    add z31.s, z31.s, z0.s
    ld1w { z23.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z22.s }, p0/Z, [x1, #-5, MUL VL]
    add z30.s, z30.s, z2.s
    add z29.s, z29.s, z1.s
    ld1w { z21.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z20.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z17.s }, p2, [x25]
    add z28.s, z28.s, z0.s
    ld1w { z19.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #-1, MUL VL]
    st1w { z16.s }, p1, [x25, #1, MUL VL]
    add z27.s, z27.s, z2.s
    ld1w { z17.s }, p1/Z, [x1]
    ld1w { z16.s }, p0/Z, [x1, #1, MUL VL]
    st1w { z31.s }, p0, [x25, #2, MUL VL]
    add z26.s, z26.s, z1.s
    st1w { z30.s }, p2, [x24]
    add z25.s, z25.s, z0.s
    add z24.s, z24.s, z2.s
    incb x25, ALL, MUL #3
    st1w { z29.s }, p1, [x24, #1, MUL VL]
    add z23.s, z23.s, z1.s
    add z22.s, z22.s, z0.s
    addvl x5, x5, #24
    st1w { z28.s }, p0, [x24, #2, MUL VL]
    add z21.s, z21.s, z2.s
    add z20.s, z20.s, z1.s
    incb x24, ALL, MUL #3
    st1w { z27.s }, p2, [x23]
    add z19.s, z19.s, z0.s
    add z18.s, z18.s, z2.s
    st1w { z26.s }, p1, [x23, #1, MUL VL]
    add z17.s, z17.s, z1.s
    add z16.s, z16.s, z0.s
    st1w { z25.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z24.s }, p2, [x22]
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
    mov z3.b, #0
    mov z2.b, #0
    mov z1.b, #0
    b label_29
KAI_ASM_LABEL(label_28)  // Initial: Height 7: Width 3: bias
    ld1w { z3.s }, p2/Z, [x3]
    ld1w { z2.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z1.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_29)  // Initial: Height 7: Width 3: init done
    decw x4, ALL, MUL #3
    ld1w { z19.s }, p2/Z, [x5]
    ld1w { z18.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z16.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z0.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z31.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z30.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z29.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z19.s, z19.s, z3.s
    ld1w { z28.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z27.s }, p2/Z, [x1, #-7, MUL VL]
    add z18.s, z18.s, z2.s
    add z17.s, z17.s, z1.s
    ld1w { z26.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z25.s }, p0/Z, [x1, #-5, MUL VL]
    add z16.s, z16.s, z3.s
    add z0.s, z0.s, z2.s
    ld1w { z24.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z19.s }, p2, [x25]
    add z31.s, z31.s, z1.s
    ld1w { z22.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z21.s }, p2/Z, [x1, #-1, MUL VL]
    st1w { z18.s }, p1, [x25, #1, MUL VL]
    add z30.s, z30.s, z3.s
    ld1w { z20.s }, p1/Z, [x1]
    ld1w { z19.s }, p0/Z, [x1, #1, MUL VL]
    st1w { z17.s }, p0, [x25, #2, MUL VL]
    add z29.s, z29.s, z2.s
    ld1w { z18.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z17.s }, p1/Z, [x1, #3, MUL VL]
    st1w { z16.s }, p2, [x24]
    add z28.s, z28.s, z1.s
    ld1w { z16.s }, p0/Z, [x1, #4, MUL VL]
    st1w { z0.s }, p1, [x24, #1, MUL VL]
    add z27.s, z27.s, z3.s
    add z26.s, z26.s, z2.s
    st1w { z31.s }, p0, [x24, #2, MUL VL]
    add z25.s, z25.s, z1.s
    add z24.s, z24.s, z3.s
    incb x25, ALL, MUL #3
    st1w { z30.s }, p2, [x23]
    add z23.s, z23.s, z2.s
    add z22.s, z22.s, z1.s
    incb x24, ALL, MUL #3
    st1w { z29.s }, p1, [x23, #1, MUL VL]
    add z21.s, z21.s, z3.s
    add z20.s, z20.s, z2.s
    addvl x5, x5, #24
    st1w { z28.s }, p0, [x23, #2, MUL VL]
    add z19.s, z19.s, z1.s
    add z18.s, z18.s, z3.s
    incb x23, ALL, MUL #3
    st1w { z27.s }, p2, [x22]
    add z17.s, z17.s, z2.s
    add z16.s, z16.s, z1.s
    st1w { z26.s }, p1, [x22, #1, MUL VL]
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
    mov z5.b, #0
    mov z4.b, #0
    mov z3.b, #0
    b label_33
KAI_ASM_LABEL(label_32)  // Initial: Height 8: Width 3: bias
    ld1w { z5.s }, p2/Z, [x3]
    ld1w { z4.s }, p1/Z, [x3, #1, MUL VL]
    ld1w { z3.s }, p0/Z, [x3, #2, MUL VL]
    incb x3, ALL, MUL #3
KAI_ASM_LABEL(label_33)  // Initial: Height 8: Width 3: init done
    decw x4, ALL, MUL #3
    ld1w { z20.s }, p2/Z, [x5]
    ld1w { z19.s }, p1/Z, [x5, #1, MUL VL]
    ld1w { z18.s }, p0/Z, [x5, #2, MUL VL]
    ld1w { z17.s }, p2/Z, [x5, #3, MUL VL]
    ld1w { z16.s }, p1/Z, [x5, #4, MUL VL]
    ld1w { z2.s }, p0/Z, [x5, #5, MUL VL]
    ld1w { z1.s }, p2/Z, [x5, #6, MUL VL]
    ld1w { z0.s }, p1/Z, [x5, #7, MUL VL]
    cmp x4, XZR
    add z20.s, z20.s, z5.s
    ld1w { z31.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z30.s }, p2/Z, [x1, #-7, MUL VL]
    add z19.s, z19.s, z4.s
    add z18.s, z18.s, z3.s
    ld1w { z29.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z28.s }, p0/Z, [x1, #-5, MUL VL]
    add z17.s, z17.s, z5.s
    add z16.s, z16.s, z4.s
    ld1w { z27.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z26.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z20.s }, p2, [x25]
    add z2.s, z2.s, z3.s
    ld1w { z25.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z24.s }, p2/Z, [x1, #-1, MUL VL]
    st1w { z19.s }, p1, [x25, #1, MUL VL]
    add z1.s, z1.s, z5.s
    ld1w { z23.s }, p1/Z, [x1]
    ld1w { z22.s }, p0/Z, [x1, #1, MUL VL]
    st1w { z18.s }, p0, [x25, #2, MUL VL]
    add z0.s, z0.s, z4.s
    ld1w { z21.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z20.s }, p1/Z, [x1, #3, MUL VL]
    st1w { z17.s }, p2, [x24]
    add z31.s, z31.s, z3.s
    ld1w { z19.s }, p0/Z, [x1, #4, MUL VL]
    ld1w { z18.s }, p2/Z, [x1, #5, MUL VL]
    st1w { z16.s }, p1, [x24, #1, MUL VL]
    add z30.s, z30.s, z5.s
    ld1w { z17.s }, p1/Z, [x1, #6, MUL VL]
    ld1w { z16.s }, p0/Z, [x1, #7, MUL VL]
    st1w { z2.s }, p0, [x24, #2, MUL VL]
    add z29.s, z29.s, z4.s
    st1w { z1.s }, p2, [x23]
    add z28.s, z28.s, z3.s
    add z27.s, z27.s, z5.s
    incb x25, ALL, MUL #3
    st1w { z0.s }, p1, [x23, #1, MUL VL]
    add z26.s, z26.s, z4.s
    add z25.s, z25.s, z3.s
    incb x24, ALL, MUL #3
    st1w { z31.s }, p0, [x23, #2, MUL VL]
    add z24.s, z24.s, z5.s
    add z23.s, z23.s, z4.s
    incb x23, ALL, MUL #3
    st1w { z30.s }, p2, [x22]
    add z22.s, z22.s, z3.s
    add z21.s, z21.s, z5.s
    addvl x5, x5, #24
    st1w { z29.s }, p1, [x22, #1, MUL VL]
    add z20.s, z20.s, z4.s
    add z19.s, z19.s, z3.s
    st1w { z28.s }, p0, [x22, #2, MUL VL]
    add z18.s, z18.s, z5.s
    add z17.s, z17.s, z4.s
    incb x22, ALL, MUL #3
    st1w { z27.s }, p2, [x21]
    add z16.s, z16.s, z3.s
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
    incw x1
    ld1w { z18.s }, p0/Z, [x4]
    ld1w { z16.s }, p0/Z, [x25]
    whilelt p1.s, x1, x2
    add z18.s, z18.s, z16.s
    incw x1
    ld1w { z17.s }, p1/Z, [x4, #1, MUL VL]
    st1w { z18.s }, p0, [x25]
    ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]
    whilelt p0.s, x1, x2
    decw x2, ALL, MUL #3
    add z17.s, z17.s, z16.s
    st1w { z17.s }, p1, [x25, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x4, #2, MUL VL]
    addvl x4, x4, #24
    cmp x2, XZR
    ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]
    add z17.s, z17.s, z16.s
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
    whilelt p0.s, x1, x2
    incw x1
    ld1w { z19.s }, p0/Z, [x4]
    ld1w { z18.s }, p0/Z, [x25]
    ld1w { z17.s }, p0/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p0/Z, [x24]
    whilelt p1.s, x1, x2
    add z19.s, z19.s, z18.s
    incw x1
    st1w { z19.s }, p0, [x25]
    add z17.s, z17.s, z16.s
    ld1w { z21.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z20.s }, p1/Z, [x4, #4, MUL VL]
    st1w { z17.s }, p0, [x24]
    ld1w { z17.s }, p1/Z, [x25, #1, MUL VL]
    whilelt p0.s, x1, x2
    decw x2, ALL, MUL #3
    ld1w { z16.s }, p1/Z, [x24, #1, MUL VL]
    add z21.s, z21.s, z17.s
    ld1w { z19.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z18.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    st1w { z21.s }, p1, [x25, #1, MUL VL]
    add z20.s, z20.s, z16.s
    cmp x2, XZR
    st1w { z20.s }, p1, [x24, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    add z19.s, z19.s, z17.s
    add z18.s, z18.s, z16.s
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
    whilelt p0.s, x2, x3
    incw x2
    ld1w { z21.s }, p0/Z, [x4]
    ld1w { z20.s }, p0/Z, [x25]
    ld1w { z19.s }, p0/Z, [x4, #3, MUL VL]
    ld1w { z18.s }, p0/Z, [x24]
    ld1w { z17.s }, p0/Z, [x4, #6, MUL VL]
    ld1w { z16.s }, p0/Z, [x23]
    whilelt p1.s, x2, x3
    add z21.s, z21.s, z20.s
    incw x2
    add z19.s, z19.s, z18.s
    st1w { z21.s }, p0, [x25]
    add z17.s, z17.s, z16.s
    ld1w { z23.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z22.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z21.s }, p1/Z, [x4, #7, MUL VL]
    st1w { z19.s }, p0, [x24]
    st1w { z17.s }, p0, [x23]
    ld1w { z18.s }, p1/Z, [x25, #1, MUL VL]
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z16.s }, p1/Z, [x23, #1, MUL VL]
    add z23.s, z23.s, z18.s
    ld1w { z20.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    ld1w { z18.s }, p0/Z, [x1, #-8, MUL VL]
    cmp x3, XZR
    st1w { z23.s }, p1, [x25, #1, MUL VL]
    add z22.s, z22.s, z17.s
    add z21.s, z21.s, z16.s
    st1w { z22.s }, p1, [x24, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z21.s }, p1, [x23, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x24, #2, MUL VL]
    add z20.s, z20.s, z16.s
    ld1w { z16.s }, p0/Z, [x23, #2, MUL VL]
    st1w { z20.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    add z19.s, z19.s, z17.s
    add z18.s, z18.s, z16.s
    st1w { z19.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z18.s }, p0, [x23, #2, MUL VL]
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
    ld1w { z23.s }, p2/Z, [x4]
    ld1w { z22.s }, p2/Z, [x25]
    ld1w { z21.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z20.s }, p2/Z, [x24]
    ld1w { z19.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z18.s }, p2/Z, [x23]
    ld1w { z17.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z16.s }, p2/Z, [x22]
    whilelt p1.s, x2, x3
    add z23.s, z23.s, z22.s
    add z21.s, z21.s, z20.s
    incw x2
    add z19.s, z19.s, z18.s
    st1w { z23.s }, p2, [x25]
    add z17.s, z17.s, z16.s
    ld1w { z26.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z25.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z24.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #-6, MUL VL]
    st1w { z21.s }, p2, [x24]
    st1w { z19.s }, p2, [x23]
    ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    st1w { z17.s }, p2, [x22]
    ld1w { z18.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z17.s }, p1/Z, [x23, #1, MUL VL]
    add z26.s, z26.s, z16.s
    ld1w { z22.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z21.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    ld1w { z20.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    ld1w { z16.s }, p1/Z, [x22, #1, MUL VL]
    st1w { z26.s }, p1, [x25, #1, MUL VL]
    add z25.s, z25.s, z18.s
    add z24.s, z24.s, z17.s
    st1w { z25.s }, p1, [x24, #1, MUL VL]
    st1w { z24.s }, p1, [x23, #1, MUL VL]
    add z23.s, z23.s, z16.s
    ld1w { z18.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z23.s }, p1, [x22, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x23, #2, MUL VL]
    add z22.s, z22.s, z18.s
    add z21.s, z21.s, z16.s
    ld1w { z16.s }, p0/Z, [x22, #2, MUL VL]
    st1w { z22.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    st1w { z21.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    add z20.s, z20.s, z17.s
    add z19.s, z19.s, z16.s
    st1w { z20.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z19.s }, p0, [x22, #2, MUL VL]
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
    ld1w { z24.s }, p2/Z, [x4]
    ld1w { z16.s }, p2/Z, [x25]
    ld1w { z23.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z22.s }, p2/Z, [x24]
    ld1w { z21.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z20.s }, p2/Z, [x23]
    ld1w { z19.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z17.s }, p2/Z, [x22]
    whilelt p1.s, x2, x3
    add z24.s, z24.s, z16.s
    ld1w { z18.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z16.s }, p2/Z, [x21]
    add z23.s, z23.s, z22.s
    incw x2
    add z21.s, z21.s, z20.s
    st1w { z24.s }, p2, [x25]
    add z19.s, z19.s, z17.s
    ld1w { z28.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z27.s }, p1/Z, [x4, #4, MUL VL]
    ld1w { z26.s }, p1/Z, [x4, #7, MUL VL]
    st1w { z23.s }, p2, [x24]
    add z18.s, z18.s, z16.s
    ld1w { z25.s }, p1/Z, [x1, #-6, MUL VL]
    ld1w { z24.s }, p1/Z, [x1, #-3, MUL VL]
    whilelt p0.s, x2, x3
    decw x3, ALL, MUL #3
    st1w { z21.s }, p2, [x23]
    ld1w { z17.s }, p1/Z, [x25, #1, MUL VL]
    st1w { z19.s }, p2, [x22]
    ld1w { z16.s }, p1/Z, [x24, #1, MUL VL]
    st1w { z18.s }, p2, [x21]
    ld1w { z23.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z22.s }, p0/Z, [x4, #5, MUL VL]
    cmp x3, XZR
    ld1w { z18.s }, p1/Z, [x23, #1, MUL VL]
    add z28.s, z28.s, z17.s
    ld1w { z21.s }, p0/Z, [x1, #-8, MUL VL]
    addvl x4, x4, #24
    ld1w { z17.s }, p1/Z, [x22, #1, MUL VL]
    add z27.s, z27.s, z16.s
    ld1w { z20.s }, p0/Z, [x1, #-5, MUL VL]
    ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x1, #-2, MUL VL]
    st1w { z28.s }, p1, [x25, #1, MUL VL]
    add z26.s, z26.s, z18.s
    st1w { z27.s }, p1, [x24, #1, MUL VL]
    add z25.s, z25.s, z17.s
    st1w { z26.s }, p1, [x23, #1, MUL VL]
    add z24.s, z24.s, z16.s
    ld1w { z17.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z25.s }, p1, [x22, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z24.s }, p1, [x21, #1, MUL VL]
    ld1w { z18.s }, p0/Z, [x23, #2, MUL VL]
    add z23.s, z23.s, z17.s
    ld1w { z17.s }, p0/Z, [x22, #2, MUL VL]
    add z22.s, z22.s, z16.s
    ld1w { z16.s }, p0/Z, [x21, #2, MUL VL]
    st1w { z23.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    add z21.s, z21.s, z18.s
    add z20.s, z20.s, z17.s
    st1w { z22.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    add z19.s, z19.s, z16.s
    st1w { z21.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    st1w { z20.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z19.s }, p0, [x21, #2, MUL VL]
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
    ld1w { z25.s }, p2/Z, [x4]
    ld1w { z17.s }, p2/Z, [x25]
    ld1w { z24.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z16.s }, p2/Z, [x24]
    ld1w { z23.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z22.s }, p2/Z, [x23]
    ld1w { z21.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z20.s }, p2/Z, [x22]
    whilelt p1.s, x2, x3
    add z25.s, z25.s, z17.s
    ld1w { z19.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z17.s }, p2/Z, [x21]
    add z24.s, z24.s, z16.s
    incw x2
    ld1w { z18.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z16.s }, p2/Z, [x20]
    add z23.s, z23.s, z22.s
    st1w { z25.s }, p2, [x25]
    add z21.s, z21.s, z20.s
    ld1w { z20.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z30.s }, p1/Z, [x4, #4, MUL VL]
    st1w { z24.s }, p2, [x24]
    add z19.s, z19.s, z17.s
    ld1w { z29.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z28.s }, p1/Z, [x1, #-6, MUL VL]
    whilelt p0.s, x2, x3
    st1w { z23.s }, p2, [x23]
    add z18.s, z18.s, z16.s
    ld1w { z17.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z27.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z21.s }, p2, [x22]
    ld1w { z16.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z26.s }, p1/Z, [x1]
    decw x3, ALL, MUL #3
    st1w { z19.s }, p2, [x21]
    ld1w { z25.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z24.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    st1w { z18.s }, p2, [x20]
    ld1w { z19.s }, p1/Z, [x23, #1, MUL VL]
    add z20.s, z20.s, z17.s
    ld1w { z23.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z18.s }, p1/Z, [x22, #1, MUL VL]
    add z30.s, z30.s, z16.s
    ld1w { z22.s }, p0/Z, [x1, #-5, MUL VL]
    cmp x3, XZR
    ld1w { z17.s }, p1/Z, [x21, #1, MUL VL]
    ld1w { z21.s }, p0/Z, [x1, #-2, MUL VL]
    ld1w { z16.s }, p1/Z, [x20, #1, MUL VL]
    st1w { z20.s }, p1, [x25, #1, MUL VL]
    add z29.s, z29.s, z19.s
    ld1w { z20.s }, p0/Z, [x1, #1, MUL VL]
    st1w { z30.s }, p1, [x24, #1, MUL VL]
    add z28.s, z28.s, z18.s
    add z27.s, z27.s, z17.s
    st1w { z29.s }, p1, [x23, #1, MUL VL]
    add z26.s, z26.s, z16.s
    ld1w { z17.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z28.s }, p1, [x22, #1, MUL VL]
    st1w { z27.s }, p1, [x21, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z26.s }, p1, [x20, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x23, #2, MUL VL]
    add z25.s, z25.s, z17.s
    ld1w { z18.s }, p0/Z, [x22, #2, MUL VL]
    ld1w { z17.s }, p0/Z, [x21, #2, MUL VL]
    add z24.s, z24.s, z16.s
    ld1w { z16.s }, p0/Z, [x20, #2, MUL VL]
    st1w { z25.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    add z23.s, z23.s, z19.s
    add z22.s, z22.s, z18.s
    st1w { z24.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    add z21.s, z21.s, z17.s
    add z20.s, z20.s, z16.s
    st1w { z23.s }, p0, [x23, #2, MUL VL]
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
    ld1w { z22.s }, p2/Z, [x4]
    ld1w { z18.s }, p2/Z, [x25]
    ld1w { z21.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z17.s }, p2/Z, [x24]
    ld1w { z26.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z16.s }, p2/Z, [x23]
    ld1w { z25.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z20.s }, p2/Z, [x22]
    whilelt p1.s, x2, x3
    add z22.s, z22.s, z18.s
    ld1w { z24.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z18.s }, p2/Z, [x21]
    add z21.s, z21.s, z17.s
    incw x2
    ld1w { z23.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z17.s }, p2/Z, [x20]
    add z26.s, z26.s, z16.s
    ld1w { z19.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z16.s }, p2/Z, [x8]
    st1w { z22.s }, p2, [x25]
    add z25.s, z25.s, z20.s
    st1w { z21.s }, p2, [x24]
    add z24.s, z24.s, z18.s
    ld1w { z22.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z21.s }, p1/Z, [x4, #4, MUL VL]
    st1w { z26.s }, p2, [x23]
    add z23.s, z23.s, z17.s
    ld1w { z20.s }, p1/Z, [x4, #7, MUL VL]
    ld1w { z30.s }, p1/Z, [x1, #-6, MUL VL]
    st1w { z25.s }, p2, [x22]
    add z19.s, z19.s, z16.s
    ld1w { z18.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z29.s }, p1/Z, [x1, #-3, MUL VL]
    st1w { z24.s }, p2, [x21]
    ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x1]
    whilelt p0.s, x2, x3
    st1w { z23.s }, p2, [x20]
    ld1w { z16.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z27.s }, p1/Z, [x1, #3, MUL VL]
    decw x3, ALL, MUL #3
    st1w { z19.s }, p2, [x8]
    ld1w { z19.s }, p1/Z, [x22, #1, MUL VL]
    add z22.s, z22.s, z18.s
    ld1w { z18.s }, p1/Z, [x21, #1, MUL VL]
    add z21.s, z21.s, z17.s
    ld1w { z26.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z25.s }, p0/Z, [x4, #5, MUL VL]
    addvl x4, x4, #24
    ld1w { z17.s }, p1/Z, [x20, #1, MUL VL]
    add z20.s, z20.s, z16.s
    ld1w { z24.s }, p0/Z, [x1, #-8, MUL VL]
    cmp x3, XZR
    ld1w { z16.s }, p1/Z, [x8, #1, MUL VL]
    st1w { z22.s }, p1, [x25, #1, MUL VL]
    add z30.s, z30.s, z19.s
    add z29.s, z29.s, z18.s
    st1w { z21.s }, p1, [x24, #1, MUL VL]
    ld1w { z23.s }, p0/Z, [x1, #-5, MUL VL]
    ld1w { z22.s }, p0/Z, [x1, #-2, MUL VL]
    st1w { z20.s }, p1, [x23, #1, MUL VL]
    add z28.s, z28.s, z17.s
    ld1w { z21.s }, p0/Z, [x1, #1, MUL VL]
    ld1w { z20.s }, p0/Z, [x1, #4, MUL VL]
    st1w { z30.s }, p1, [x22, #1, MUL VL]
    add z27.s, z27.s, z16.s
    ld1w { z18.s }, p0/Z, [x25, #2, MUL VL]
    st1w { z29.s }, p1, [x21, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x24, #2, MUL VL]
    st1w { z28.s }, p1, [x20, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x23, #2, MUL VL]
    st1w { z27.s }, p1, [x8, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x22, #2, MUL VL]
    add z26.s, z26.s, z18.s
    ld1w { z18.s }, p0/Z, [x21, #2, MUL VL]
    add z25.s, z25.s, z17.s
    ld1w { z17.s }, p0/Z, [x20, #2, MUL VL]
    add z24.s, z24.s, z16.s
    ld1w { z16.s }, p0/Z, [x8, #2, MUL VL]
    st1w { z26.s }, p0, [x25, #2, MUL VL]
    incb x25, ALL, MUL #3
    add z23.s, z23.s, z19.s
    add z22.s, z22.s, z18.s
    st1w { z25.s }, p0, [x24, #2, MUL VL]
    incb x24, ALL, MUL #3
    st1w { z24.s }, p0, [x23, #2, MUL VL]
    add z21.s, z21.s, z17.s
    add z20.s, z20.s, z16.s
    incb x23, ALL, MUL #3
    st1w { z23.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    st1w { z22.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z21.s }, p0, [x20, #2, MUL VL]
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
    ld1w { z20.s }, p2/Z, [x4]
    ld1w { z19.s }, p2/Z, [x25]
    ld1w { z26.s }, p2/Z, [x4, #3, MUL VL]
    ld1w { z18.s }, p2/Z, [x24]
    ld1w { z25.s }, p2/Z, [x4, #6, MUL VL]
    ld1w { z17.s }, p2/Z, [x23]
    ld1w { z24.s }, p2/Z, [x1, #-7, MUL VL]
    ld1w { z16.s }, p2/Z, [x22]
    whilelt p1.s, x2, x3
    add z20.s, z20.s, z19.s
    ld1w { z23.s }, p2/Z, [x1, #-4, MUL VL]
    ld1w { z19.s }, p2/Z, [x21]
    add z26.s, z26.s, z18.s
    incw x2
    ld1w { z22.s }, p2/Z, [x1, #-1, MUL VL]
    ld1w { z18.s }, p2/Z, [x20]
    add z25.s, z25.s, z17.s
    ld1w { z21.s }, p2/Z, [x1, #2, MUL VL]
    ld1w { z17.s }, p2/Z, [x8]
    st1w { z20.s }, p2, [x25]
    add z24.s, z24.s, z16.s
    ld1w { z20.s }, p2/Z, [x1, #5, MUL VL]
    ld1w { z16.s }, p2/Z, [x7]
    st1w { z26.s }, p2, [x24]
    add z23.s, z23.s, z19.s
    st1w { z25.s }, p2, [x23]
    add z22.s, z22.s, z18.s
    ld1w { z26.s }, p1/Z, [x4, #1, MUL VL]
    ld1w { z25.s }, p1/Z, [x4, #4, MUL VL]
    st1w { z24.s }, p2, [x22]
    add z21.s, z21.s, z17.s
    ld1w { z19.s }, p1/Z, [x25, #1, MUL VL]
    ld1w { z24.s }, p1/Z, [x4, #7, MUL VL]
    st1w { z23.s }, p2, [x21]
    add z20.s, z20.s, z16.s
    ld1w { z18.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z23.s }, p1/Z, [x1, #-6, MUL VL]
    st1w { z22.s }, p2, [x20]
    ld1w { z17.s }, p1/Z, [x23, #1, MUL VL]
    ld1w { z22.s }, p1/Z, [x1, #-3, MUL VL]
    whilelt p0.s, x2, x3
    st1w { z21.s }, p2, [x8]
    ld1w { z16.s }, p1/Z, [x22, #1, MUL VL]
    ld1w { z21.s }, p1/Z, [x1]
    add z26.s, z26.s, z19.s
    st1w { z20.s }, p2, [x7]
    ld1w { z19.s }, p1/Z, [x21, #1, MUL VL]
    ld1w { z20.s }, p1/Z, [x1, #3, MUL VL]
    add z25.s, z25.s, z18.s
    ld1w { z18.s }, p1/Z, [x20, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x1, #6, MUL VL]
    add z24.s, z24.s, z17.s
    decw x3, ALL, MUL #3
    ld1w { z17.s }, p1/Z, [x8, #1, MUL VL]
    st1w { z26.s }, p1, [x25, #1, MUL VL]
    add z23.s, z23.s, z16.s
    ld1w { z27.s }, p0/Z, [x4, #2, MUL VL]
    ld1w { z16.s }, p1/Z, [x7, #1, MUL VL]
    st1w { z25.s }, p1, [x24, #1, MUL VL]
    add z22.s, z22.s, z19.s
    ld1w { z26.s }, p0/Z, [x4, #5, MUL VL]
    st1w { z24.s }, p1, [x23, #1, MUL VL]
    add z21.s, z21.s, z18.s
    ld1w { z25.s }, p0/Z, [x1, #-8, MUL VL]
    ld1w { z24.s }, p0/Z, [x1, #-5, MUL VL]
    st1w { z23.s }, p1, [x22, #1, MUL VL]
    add z20.s, z20.s, z17.s
    ld1w { z19.s }, p0/Z, [x25, #2, MUL VL]
    ld1w { z23.s }, p0/Z, [x1, #-2, MUL VL]
    st1w { z22.s }, p1, [x21, #1, MUL VL]
    add z28.s, z28.s, z16.s
    ld1w { z18.s }, p0/Z, [x24, #2, MUL VL]
    ld1w { z22.s }, p0/Z, [x1, #1, MUL VL]
    st1w { z21.s }, p1, [x20, #1, MUL VL]
    ld1w { z17.s }, p0/Z, [x23, #2, MUL VL]
    ld1w { z21.s }, p0/Z, [x1, #4, MUL VL]
    cmp x3, XZR
    st1w { z20.s }, p1, [x8, #1, MUL VL]
    ld1w { z16.s }, p0/Z, [x22, #2, MUL VL]
    ld1w { z20.s }, p0/Z, [x1, #7, MUL VL]
    add z27.s, z27.s, z19.s
    st1w { z28.s }, p1, [x7, #1, MUL VL]
    ld1w { z19.s }, p0/Z, [x21, #2, MUL VL]
    add z26.s, z26.s, z18.s
    addvl x4, x4, #24
    ld1w { z18.s }, p0/Z, [x20, #2, MUL VL]
    add z25.s, z25.s, z17.s
    ld1w { z17.s }, p0/Z, [x8, #2, MUL VL]
    st1w { z27.s }, p0, [x25, #2, MUL VL]
    add z24.s, z24.s, z16.s
    incb x25, ALL, MUL #3
    ld1w { z16.s }, p0/Z, [x7, #2, MUL VL]
    st1w { z26.s }, p0, [x24, #2, MUL VL]
    add z23.s, z23.s, z19.s
    incb x24, ALL, MUL #3
    st1w { z25.s }, p0, [x23, #2, MUL VL]
    incb x23, ALL, MUL #3
    add z22.s, z22.s, z18.s
    add z21.s, z21.s, z17.s
    st1w { z24.s }, p0, [x22, #2, MUL VL]
    incb x22, ALL, MUL #3
    add z20.s, z20.s, z16.s
    st1w { z23.s }, p0, [x21, #2, MUL VL]
    incb x21, ALL, MUL #3
    st1w { z22.s }, p0, [x20, #2, MUL VL]
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
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sve_merge_u32_u32_8x3VL)

    KAI_ASM_END
