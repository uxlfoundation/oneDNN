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

    KAI_ASM_CODE(a64_interleave8_block1_s8_s16.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block1_s8_s16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block1_s8_s16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block1_s8_s16)
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
    cmp x2, #0x8
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
    ldr d23, [x24], #0x8
    ldr d25, [x23], #0x8
    subs x2, x2, #0x8
    ldr d22, [x22], #0x8
    ldr d21, [x21], #0x8
    cmp x2, #0x8
    ldr d19, [x20], #0x8
    ldr d18, [x8], #0x8
    ldr d17, [x7], #0x8
    ldr d16, [x6], #0x8
    sshll v23.8h, v23.8b, #0
    sshll v25.8h, v25.8b, #0
    sshll v22.8h, v22.8b, #0
    sshll v21.8h, v21.8b, #0
    prfm pldl1keep, [x24, #0x70]
    prfm pldl1keep, [x23, #0x70]
    sshll v19.8h, v19.8b, #0
    sshll v18.8h, v18.8b, #0
    prfm pldl1keep, [x22, #0x70]
    prfm pldl1keep, [x21, #0x70]
    sshll v17.8h, v17.8b, #0
    sshll v16.8h, v16.8b, #0
    prfm pldl1keep, [x20, #0x70]
    prfm pldl1keep, [x8, #0x70]
    prfm pldl1keep, [x7, #0x70]
    prfm pldl1keep, [x6, #0x70]
    zip1 v20.8h, v23.8h, v19.8h
    zip1 v24.8h, v25.8h, v18.8h
    zip2 v23.8h, v23.8h, v19.8h
    zip2 v25.8h, v25.8h, v18.8h
    zip1 v19.8h, v22.8h, v17.8h
    zip1 v18.8h, v21.8h, v16.8h
    zip2 v22.8h, v22.8h, v17.8h
    zip2 v17.8h, v21.8h, v16.8h
    zip1 v21.8h, v20.8h, v19.8h
    zip1 v16.8h, v24.8h, v18.8h
    zip2 v20.8h, v20.8h, v19.8h
    zip2 v19.8h, v24.8h, v18.8h
    zip1 v24.8h, v23.8h, v22.8h
    zip1 v18.8h, v25.8h, v17.8h
    zip2 v23.8h, v23.8h, v22.8h
    zip2 v22.8h, v25.8h, v17.8h
    zip1 v17.8h, v21.8h, v16.8h
    zip2 v16.8h, v21.8h, v16.8h
    zip1 v21.8h, v20.8h, v19.8h
    zip2 v20.8h, v20.8h, v19.8h
    zip1 v19.8h, v24.8h, v18.8h
    zip2 v18.8h, v24.8h, v18.8h
    str q17, [x5, #0]
    str q16, [x5, #0x10]
    zip1 v17.8h, v23.8h, v22.8h
    zip2 v16.8h, v23.8h, v22.8h
    str q21, [x5, #0x20]
    str q20, [x5, #0x30]
    str q19, [x5, #0x40]
    str q18, [x5, #0x50]
    str q17, [x5, #0x60]
    str q16, [x5, #0x70]
    add x5, x5, #0x80
    bge label_2
KAI_ASM_LABEL(label_3)  // Main loop skip
    cbz x2, label_8
    tbz x2, #2, label_5
    ldr s30, [x24], #0x4
    ldr s29, [x23], #0x4
    ldr s28, [x22], #0x4
    ldr s27, [x21], #0x4
    ldr s26, [x20], #0x4
    ldr s25, [x8], #0x4
    ldr s24, [x7], #0x4
    ldr s23, [x6], #0x4
    tbz x2, #1, label_4
    ld1 { v30.h }[2], [x24], #0x2
    ld1 { v29.h }[2], [x23], #0x2
    mov x1, #0x6
    ld1 { v28.h }[2], [x22], #0x2
    ld1 { v27.h }[2], [x21], #0x2
    ld1 { v26.h }[2], [x20], #0x2
    ld1 { v25.h }[2], [x8], #0x2
    ld1 { v24.h }[2], [x7], #0x2
    ld1 { v23.h }[2], [x6], #0x2
    tbz x2, #0, label_7
    ld1 { v30.b }[6], [x24]
    ld1 { v29.b }[6], [x23]
    mov x1, #0x7
    ld1 { v28.b }[6], [x22]
    ld1 { v27.b }[6], [x21]
    ld1 { v26.b }[6], [x20]
    ld1 { v25.b }[6], [x8]
    ld1 { v24.b }[6], [x7]
    ld1 { v23.b }[6], [x6]
    b label_7
KAI_ASM_LABEL(label_4)  // odd_loads_1_4
    mov x1, #0x4
    tbz x2, #0, label_7
    ld1 { v30.b }[4], [x24]
    ld1 { v29.b }[4], [x23]
    mov x1, #0x5
    ld1 { v28.b }[4], [x22]
    ld1 { v27.b }[4], [x21]
    ld1 { v26.b }[4], [x20]
    ld1 { v25.b }[4], [x8]
    ld1 { v24.b }[4], [x7]
    ld1 { v23.b }[4], [x6]
    b label_7
KAI_ASM_LABEL(label_5)  // odd_loads_2_0
    tbz x2, #1, label_6
    ldr h30, [x24], #0x2
    ldr h29, [x23], #0x2
    mov x1, #0x2
    ldr h28, [x22], #0x2
    ldr h27, [x21], #0x2
    ldr h26, [x20], #0x2
    ldr h25, [x8], #0x2
    ldr h24, [x7], #0x2
    ldr h23, [x6], #0x2
    tbz x2, #0, label_7
    ld1 { v30.b }[2], [x24]
    ld1 { v29.b }[2], [x23]
    mov x1, #0x3
    ld1 { v28.b }[2], [x22]
    ld1 { v27.b }[2], [x21]
    ld1 { v26.b }[2], [x20]
    ld1 { v25.b }[2], [x8]
    ld1 { v24.b }[2], [x7]
    ld1 { v23.b }[2], [x6]
    b label_7
KAI_ASM_LABEL(label_6)  // odd_loads_1_0
    ldr b30, [x24, #0]
    ldr b29, [x23, #0]
    mov x1, #0x1
    ldr b28, [x22, #0]
    ldr b27, [x21, #0]
    ldr b26, [x20, #0]
    ldr b25, [x8, #0]
    ldr b24, [x7, #0]
    ldr b23, [x6, #0]
KAI_ASM_LABEL(label_7)  // Odd load end
    sshll v30.8h, v30.8b, #0
    sshll v29.8h, v29.8b, #0
    subs x1, x1, #0x1
    sshll v28.8h, v28.8b, #0
    sshll v27.8h, v27.8b, #0
    sshll v26.8h, v26.8b, #0
    sshll v25.8h, v25.8b, #0
    sshll v24.8h, v24.8b, #0
    sshll v23.8h, v23.8b, #0
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v29.8h, v25.8h
    zip1 v20.8h, v28.8h, v24.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v20.8h
    zip1 v17.8h, v21.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    subs x1, x1, #0x1
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    zip2 v18.8h, v22.8h, v20.8h
    zip2 v17.8h, v21.8h, v19.8h
    subs x1, x1, #0x1
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    subs x1, x1, #0x1
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    subs x1, x1, #0x1
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    subs x1, x1, #0x1
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
    beq label_8
    zip2 v17.8h, v22.8h, v21.8h
    zip2 v16.8h, v20.8h, v19.8h
    zip1 v16.8h, v17.8h, v16.8h
    str q16, [x5, #0]
    add x5, x5, #0x10
KAI_ASM_LABEL(label_8)  // Odds skip
    str x5, [x0]
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block1_s8_s16)

    KAI_ASM_END
