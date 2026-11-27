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

    KAI_ASM_CODE(a64_interleave8_block8_s8_s8.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block8_s8_s8)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block8_s8_s8)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block8_s8_s8)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    ldr x24, [x1, #0]
    ldr x23, [x1, #0x8]
    ldr x22, [x1, #0x10]
    ldr x21, [x1, #0x18]
    ldr x20, [x1, #0x20]
    ldr x8, [x1, #0x28]
    ldr x7, [x1, #0x30]
    ldr x6, [x1, #0x38]
    cmp x3, #0x8
    add x24, x24, x4
    ldr x5, [x0, #0]
    add x23, x23, x4
    add x22, x22, x4
    add x21, x21, x4
    add x20, x20, x4
    add x8, x8, x4
    add x7, x7, x4
    add x6, x6, x4
    beq label_1
    cmp x3, #0x2
    mov x6, x24
    csel x23, x23, x24, GE
    csel x22, x22, x24, GT
    cmp x3, #0x4
    csel x21, x21, x24, GE
    csel x20, x20, x24, GT
    cmp x3, #0x6
    csel x8, x8, x24, GE
    csel x7, x7, x24, GT
KAI_ASM_LABEL(label_1)  // no_pointer_adj
    cmp x2, #0x10
    prfm pldl1keep, [x24, #0]
    prfm pldl1keep, [x23, #0]
    prfm pldl1keep, [x22, #0]
    prfm pldl1keep, [x21, #0]
    prfm pldl1keep, [x20, #0]
    prfm pldl1keep, [x8, #0]
    prfm pldl1keep, [x7, #0]
    prfm pldl1keep, [x6, #0]
    prfm pldl1keep, [x24, #0x40]
    prfm pldl1keep, [x23, #0x40]
    prfm pldl1keep, [x22, #0x40]
    prfm pldl1keep, [x21, #0x40]
    prfm pldl1keep, [x20, #0x40]
    prfm pldl1keep, [x8, #0x40]
    prfm pldl1keep, [x7, #0x40]
    prfm pldl1keep, [x6, #0x40]
    blt label_3
KAI_ASM_LABEL(label_2)  // Main loop head
    ldr q21, [x24], #0x10
    ldr q16, [x23], #0x10
    subs x2, x2, #0x10
    ldr q20, [x22], #0x10
    ldr q19, [x21], #0x10
    cmp x2, #0x10
    ldr q24, [x20], #0x10
    ldr q18, [x8], #0x10
    ldr q23, [x7], #0x10
    ldr q22, [x6], #0x10
    zip1 v17.2d, v21.2d, v16.2d
    zip2 v21.2d, v21.2d, v16.2d
    prfm pldl1keep, [x24, #0x70]
    prfm pldl1keep, [x23, #0x70]
    zip1 v16.2d, v20.2d, v19.2d
    zip2 v20.2d, v20.2d, v19.2d
    prfm pldl1keep, [x22, #0x70]
    prfm pldl1keep, [x21, #0x70]
    zip1 v19.2d, v24.2d, v18.2d
    zip2 v18.2d, v24.2d, v18.2d
    prfm pldl1keep, [x20, #0x70]
    prfm pldl1keep, [x8, #0x70]
    str q17, [x5, #0]
    zip1 v17.2d, v23.2d, v22.2d
    prfm pldl1keep, [x7, #0x70]
    prfm pldl1keep, [x6, #0x70]
    str q16, [x5, #0x10]
    zip2 v16.2d, v23.2d, v22.2d
    str q19, [x5, #0x20]
    str q17, [x5, #0x30]
    str q21, [x5, #0x40]
    str q20, [x5, #0x50]
    str q18, [x5, #0x60]
    str q16, [x5, #0x70]
    add x5, x5, #0x80
    bge label_2
KAI_ASM_LABEL(label_3)  // Main loop skip
    cbz x2, label_12
    tbz x2, #3, label_7
    ldr d27, [x24], #0x8
    ldr d26, [x23], #0x8
    ldr d25, [x22], #0x8
    ldr d24, [x21], #0x8
    ldr d23, [x20], #0x8
    ldr d22, [x8], #0x8
    ldr d21, [x7], #0x8
    ldr d20, [x6], #0x8
    tbz x2, #2, label_5
    ld1 { v27.s }[2], [x24], #0x4
    ld1 { v26.s }[2], [x23], #0x4
    ld1 { v25.s }[2], [x22], #0x4
    ld1 { v24.s }[2], [x21], #0x4
    ld1 { v23.s }[2], [x20], #0x4
    ld1 { v22.s }[2], [x8], #0x4
    ld1 { v21.s }[2], [x7], #0x4
    ld1 { v20.s }[2], [x6], #0x4
    tbz x2, #1, label_4
    ld1 { v27.h }[6], [x24], #0x2
    ld1 { v26.h }[6], [x23], #0x2
    mov x1, #0x2
    ld1 { v25.h }[6], [x22], #0x2
    ld1 { v24.h }[6], [x21], #0x2
    ld1 { v23.h }[6], [x20], #0x2
    ld1 { v22.h }[6], [x8], #0x2
    ld1 { v21.h }[6], [x7], #0x2
    ld1 { v20.h }[6], [x6], #0x2
    tbz x2, #0, label_11
    ld1 { v27.b }[14], [x24]
    ld1 { v26.b }[14], [x23]
    ld1 { v25.b }[14], [x22]
    ld1 { v24.b }[14], [x21]
    ld1 { v23.b }[14], [x20]
    ld1 { v22.b }[14], [x8]
    ld1 { v21.b }[14], [x7]
    ld1 { v20.b }[14], [x6]
    b label_11
KAI_ASM_LABEL(label_4)  // odd_loads_1_12
    mov x1, #0x2
    tbz x2, #0, label_11
    ld1 { v27.b }[12], [x24]
    ld1 { v26.b }[12], [x23]
    ld1 { v25.b }[12], [x22]
    ld1 { v24.b }[12], [x21]
    ld1 { v23.b }[12], [x20]
    ld1 { v22.b }[12], [x8]
    ld1 { v21.b }[12], [x7]
    ld1 { v20.b }[12], [x6]
    b label_11
KAI_ASM_LABEL(label_5)  // odd_loads_2_8
    tbz x2, #1, label_6
    ld1 { v27.h }[4], [x24], #0x2
    ld1 { v26.h }[4], [x23], #0x2
    mov x1, #0x2
    ld1 { v25.h }[4], [x22], #0x2
    ld1 { v24.h }[4], [x21], #0x2
    ld1 { v23.h }[4], [x20], #0x2
    ld1 { v22.h }[4], [x8], #0x2
    ld1 { v21.h }[4], [x7], #0x2
    ld1 { v20.h }[4], [x6], #0x2
    tbz x2, #0, label_11
    ld1 { v27.b }[10], [x24]
    ld1 { v26.b }[10], [x23]
    ld1 { v25.b }[10], [x22]
    ld1 { v24.b }[10], [x21]
    ld1 { v23.b }[10], [x20]
    ld1 { v22.b }[10], [x8]
    ld1 { v21.b }[10], [x7]
    ld1 { v20.b }[10], [x6]
    b label_11
KAI_ASM_LABEL(label_6)  // odd_loads_1_8
    mov x1, #0x1
    tbz x2, #0, label_11
    ld1 { v27.b }[8], [x24]
    ld1 { v26.b }[8], [x23]
    mov x1, #0x2
    ld1 { v25.b }[8], [x22]
    ld1 { v24.b }[8], [x21]
    ld1 { v23.b }[8], [x20]
    ld1 { v22.b }[8], [x8]
    ld1 { v21.b }[8], [x7]
    ld1 { v20.b }[8], [x6]
    b label_11
KAI_ASM_LABEL(label_7)  // odd_loads_4_0
    tbz x2, #2, label_9
    ldr s27, [x24], #0x4
    ldr s26, [x23], #0x4
    ldr s25, [x22], #0x4
    ldr s24, [x21], #0x4
    ldr s23, [x20], #0x4
    ldr s22, [x8], #0x4
    ldr s21, [x7], #0x4
    ldr s20, [x6], #0x4
    tbz x2, #1, label_8
    ld1 { v27.h }[2], [x24], #0x2
    ld1 { v26.h }[2], [x23], #0x2
    mov x1, #0x1
    ld1 { v25.h }[2], [x22], #0x2
    ld1 { v24.h }[2], [x21], #0x2
    ld1 { v23.h }[2], [x20], #0x2
    ld1 { v22.h }[2], [x8], #0x2
    ld1 { v21.h }[2], [x7], #0x2
    ld1 { v20.h }[2], [x6], #0x2
    tbz x2, #0, label_11
    ld1 { v27.b }[6], [x24]
    ld1 { v26.b }[6], [x23]
    ld1 { v25.b }[6], [x22]
    ld1 { v24.b }[6], [x21]
    ld1 { v23.b }[6], [x20]
    ld1 { v22.b }[6], [x8]
    ld1 { v21.b }[6], [x7]
    ld1 { v20.b }[6], [x6]
    b label_11
KAI_ASM_LABEL(label_8)  // odd_loads_1_4
    mov x1, #0x1
    tbz x2, #0, label_11
    ld1 { v27.b }[4], [x24]
    ld1 { v26.b }[4], [x23]
    ld1 { v25.b }[4], [x22]
    ld1 { v24.b }[4], [x21]
    ld1 { v23.b }[4], [x20]
    ld1 { v22.b }[4], [x8]
    ld1 { v21.b }[4], [x7]
    ld1 { v20.b }[4], [x6]
    b label_11
KAI_ASM_LABEL(label_9)  // odd_loads_2_0
    tbz x2, #1, label_10
    ldr h27, [x24], #0x2
    ldr h26, [x23], #0x2
    mov x1, #0x1
    ldr h25, [x22], #0x2
    ldr h24, [x21], #0x2
    ldr h23, [x20], #0x2
    ldr h22, [x8], #0x2
    ldr h21, [x7], #0x2
    ldr h20, [x6], #0x2
    tbz x2, #0, label_11
    ld1 { v27.b }[2], [x24]
    ld1 { v26.b }[2], [x23]
    ld1 { v25.b }[2], [x22]
    ld1 { v24.b }[2], [x21]
    ld1 { v23.b }[2], [x20]
    ld1 { v22.b }[2], [x8]
    ld1 { v21.b }[2], [x7]
    ld1 { v20.b }[2], [x6]
    b label_11
KAI_ASM_LABEL(label_10)  // odd_loads_1_0
    ldr b27, [x24, #0]
    ldr b26, [x23, #0]
    mov x1, #0x1
    ldr b25, [x22, #0]
    ldr b24, [x21, #0]
    ldr b23, [x20, #0]
    ldr b22, [x8, #0]
    ldr b21, [x7, #0]
    ldr b20, [x6, #0]
KAI_ASM_LABEL(label_11)  // Odd load end
    subs x1, x1, #0x1
    zip1 v19.2d, v27.2d, v26.2d
    zip1 v18.2d, v25.2d, v24.2d
    zip1 v17.2d, v23.2d, v22.2d
    zip1 v16.2d, v21.2d, v20.2d
    str q19, [x5, #0]
    str q18, [x5, #0x10]
    str q17, [x5, #0x20]
    str q16, [x5, #0x30]
    add x5, x5, #0x40
    beq label_12
    zip2 v19.2d, v27.2d, v26.2d
    zip2 v18.2d, v25.2d, v24.2d
    zip2 v17.2d, v23.2d, v22.2d
    zip2 v16.2d, v21.2d, v20.2d
    str q19, [x5, #0]
    str q18, [x5, #0x10]
    str q17, [x5, #0x20]
    str q16, [x5, #0x30]
    add x5, x5, #0x40
KAI_ASM_LABEL(label_12)  // Odds skip
    str x5, [x0]
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block8_s8_s8)

    KAI_ASM_END
