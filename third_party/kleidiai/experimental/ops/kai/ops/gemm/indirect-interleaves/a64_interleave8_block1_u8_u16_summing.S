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

    KAI_ASM_CODE(a64_interleave8_block1_u8_u16_summing.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block1_u8_u16_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block1_u8_u16_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block1_u8_u16_summing)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    mov x26, #0
    movi v2.8h, #0
    movi v1.4s, #0
    movi v0.4s, #0
    ldr x25, [x1, #0]
    ldr x24, [x1, #0x8]
    ldr x23, [x1, #0x10]
    ldr x22, [x1, #0x18]
    ldr x21, [x1, #0x20]
    ldr x20, [x1, #0x28]
    cmp x3, #0x8
    dup v31.4s, w6
    ldr x8, [x1, #0x30]
    ldr x7, [x1, #0x38]
    add x25, x25, x4
    add x24, x24, x4
    ldr x6, [x0, #0]
    add x23, x23, x4
    add x22, x22, x4
    add x21, x21, x4
    add x20, x20, x4
    add x8, x8, x4
    add x7, x7, x4
    beq label_1
    cmp x3, #0x2
    mov x7, x25
    csel x24, x24, x25, GE
    csel x23, x23, x25, GT
    cmp x3, #0x4
    csel x22, x22, x25, GE
    csel x21, x21, x25, GT
    cmp x3, #0x6
    csel x20, x20, x25, GE
    csel x8, x8, x25, GT
KAI_ASM_LABEL(label_1)  // no_pointer_adj
    prfm pldl1keep, [x25, #0]
    prfm pldl1keep, [x24, #0]
    prfm pldl1keep, [x23, #0]
    prfm pldl1keep, [x22, #0]
    prfm pldl1keep, [x21, #0]
    prfm pldl1keep, [x20, #0]
    prfm pldl1keep, [x8, #0]
    prfm pldl1keep, [x7, #0]
    prfm pldl1keep, [x25, #0x40]
    prfm pldl1keep, [x24, #0x40]
    prfm pldl1keep, [x23, #0x40]
    prfm pldl1keep, [x22, #0x40]
    prfm pldl1keep, [x21, #0x40]
    prfm pldl1keep, [x20, #0x40]
    prfm pldl1keep, [x8, #0x40]
    prfm pldl1keep, [x7, #0x40]
    cbnz w5, label_2
    sub x6, x6, #0x20
    ld1 { v1.4s }, [x6]
    ldr q0, [x6, #0x10]
KAI_ASM_LABEL(label_2)  // first_pass
    cmp x2, #0x8
    blt label_5
KAI_ASM_LABEL(label_3)  // Main loop head
    cmp x26, #0xe
    ble label_4
    uaddw v1.4s, v1.4s, v2.4h
    uaddw2 v0.4s, v0.4s, v2.8h
    mov x26, #0
    movi v2.8h, #0
KAI_ASM_LABEL(label_4)  // no_accumulate_16
    ldr d25, [x25], #0x8
    ldr d24, [x24], #0x8
    subs x2, x2, #0x8
    add x26, x26, #0x1
    ldr d23, [x23], #0x8
    ldr d22, [x22], #0x8
    cmp x2, #0x8
    ldr d18, [x21], #0x8
    ldr d16, [x20], #0x8
    ldr d21, [x8], #0x8
    ldr d17, [x7], #0x8
    ushll v25.8h, v25.8b, #0
    ushll v24.8h, v24.8b, #0
    ushll v23.8h, v23.8b, #0
    ushll v22.8h, v22.8b, #0
    prfm pldl1keep, [x25, #0x70]
    prfm pldl1keep, [x24, #0x70]
    ushll v18.8h, v18.8b, #0
    ushll v16.8h, v16.8b, #0
    prfm pldl1keep, [x23, #0x70]
    prfm pldl1keep, [x22, #0x70]
    ushll v21.8h, v21.8b, #0
    ushll v17.8h, v17.8b, #0
    prfm pldl1keep, [x21, #0x70]
    prfm pldl1keep, [x20, #0x70]
    prfm pldl1keep, [x8, #0x70]
    prfm pldl1keep, [x7, #0x70]
    zip1 v20.8h, v25.8h, v18.8h
    zip1 v19.8h, v24.8h, v16.8h
    zip2 v26.8h, v25.8h, v18.8h
    zip2 v25.8h, v24.8h, v16.8h
    zip1 v18.8h, v23.8h, v21.8h
    zip1 v16.8h, v22.8h, v17.8h
    zip2 v24.8h, v23.8h, v21.8h
    zip2 v23.8h, v22.8h, v17.8h
    zip1 v22.8h, v20.8h, v18.8h
    zip1 v17.8h, v19.8h, v16.8h
    zip2 v21.8h, v20.8h, v18.8h
    zip2 v20.8h, v19.8h, v16.8h
    zip1 v19.8h, v26.8h, v24.8h
    zip1 v18.8h, v25.8h, v23.8h
    zip1 v16.8h, v22.8h, v17.8h
    zip2 v17.8h, v22.8h, v17.8h
    zip1 v22.8h, v21.8h, v20.8h
    zip2 v21.8h, v21.8h, v20.8h
    zip1 v20.8h, v19.8h, v18.8h
    zip2 v19.8h, v19.8h, v18.8h
    add v2.8h, v2.8h, v16.8h
    zip2 v18.8h, v26.8h, v24.8h
    str q16, [x6, #0]
    zip2 v16.8h, v25.8h, v23.8h
    str q17, [x6, #0x10]
    str q22, [x6, #0x20]
    add v2.8h, v2.8h, v17.8h
    str q21, [x6, #0x30]
    zip1 v17.8h, v18.8h, v16.8h
    zip2 v16.8h, v18.8h, v16.8h
    str q20, [x6, #0x40]
    add v2.8h, v2.8h, v22.8h
    str q19, [x6, #0x50]
    str q17, [x6, #0x60]
    add v2.8h, v2.8h, v21.8h
    str q16, [x6, #0x70]
    add x6, x6, #0x80
    add v2.8h, v2.8h, v20.8h
    add v2.8h, v2.8h, v19.8h
    add v2.8h, v2.8h, v17.8h
    add v2.8h, v2.8h, v16.8h
    bge label_3
KAI_ASM_LABEL(label_5)  // Main loop skip
    cbz x2, label_10
    tbz x2, #2, label_7
    ldr s30, [x25], #0x4
    ldr s29, [x24], #0x4
    ldr s28, [x23], #0x4
    ldr s27, [x22], #0x4
    ldr s26, [x21], #0x4
    ldr s25, [x20], #0x4
    ldr s24, [x8], #0x4
    ldr s23, [x7], #0x4
    tbz x2, #1, label_6
    ld1 { v30.h }[2], [x25], #0x2
    ld1 { v29.h }[2], [x24], #0x2
    mov x1, #0x6
    ld1 { v28.h }[2], [x23], #0x2
    ld1 { v27.h }[2], [x22], #0x2
    ld1 { v26.h }[2], [x21], #0x2
    ld1 { v25.h }[2], [x20], #0x2
    ld1 { v24.h }[2], [x8], #0x2
    ld1 { v23.h }[2], [x7], #0x2
    tbz x2, #0, label_9
    ld1 { v30.b }[6], [x25]
    ld1 { v29.b }[6], [x24]
    mov x1, #0x7
    ld1 { v28.b }[6], [x23]
    ld1 { v27.b }[6], [x22]
    ld1 { v26.b }[6], [x21]
    ld1 { v25.b }[6], [x20]
    ld1 { v24.b }[6], [x8]
    ld1 { v23.b }[6], [x7]
    b label_9
KAI_ASM_LABEL(label_6)  // odd_loads_1_4
    mov x1, #0x4
    tbz x2, #0, label_9
    ld1 { v30.b }[4], [x25]
    ld1 { v29.b }[4], [x24]
    mov x1, #0x5
    ld1 { v28.b }[4], [x23]
    ld1 { v27.b }[4], [x22]
    ld1 { v26.b }[4], [x21]
    ld1 { v25.b }[4], [x20]
    ld1 { v24.b }[4], [x8]
    ld1 { v23.b }[4], [x7]
    b label_9
KAI_ASM_LABEL(label_7)  // odd_loads_2_0
    tbz x2, #1, label_8
    ldr h30, [x25], #0x2
    ldr h29, [x24], #0x2
    mov x1, #0x2
    ldr h28, [x23], #0x2
    ldr h27, [x22], #0x2
    ldr h26, [x21], #0x2
    ldr h25, [x20], #0x2
    ldr h24, [x8], #0x2
    ldr h23, [x7], #0x2
    tbz x2, #0, label_9
    ld1 { v30.b }[2], [x25]
    ld1 { v29.b }[2], [x24]
    mov x1, #0x3
    ld1 { v28.b }[2], [x23]
    ld1 { v27.b }[2], [x22]
    ld1 { v26.b }[2], [x21]
    ld1 { v25.b }[2], [x20]
    ld1 { v24.b }[2], [x8]
    ld1 { v23.b }[2], [x7]
    b label_9
KAI_ASM_LABEL(label_8)  // odd_loads_1_0
    ldr b30, [x25, #0]
    ldr b29, [x24, #0]
    mov x1, #0x1
    ldr b28, [x23, #0]
    ldr b27, [x22, #0]
    ldr b26, [x21, #0]
    ldr b25, [x20, #0]
    ldr b24, [x8, #0]
    ldr b23, [x7, #0]
KAI_ASM_LABEL(label_9)  // Odd load end
    ushll v30.8h, v30.8b, #0
    ushll v29.8h, v29.8b, #0
    subs x1, x1, #0x1
    ushll v28.8h, v28.8b, #0
    ushll v27.8h, v27.8b, #0
    ushll v26.8h, v26.8b, #0
    ushll v25.8h, v25.8b, #0
    ushll v24.8h, v24.8b, #0
    ushll v23.8h, v23.8b, #0
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v29.8h, v25.8h
    zip1 v20.8h, v28.8h, v24.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v20.8h
    zip1 v17.8h, v21.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v16.8h, v18.8h, v17.8h
    subs x1, x1, #0x1
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v18.8h, v22.8h, v20.8h
    zip2 v17.8h, v21.8h, v19.8h
    subs x1, x1, #0x1
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v16.8h, v18.8h, v17.8h
    subs x1, x1, #0x1
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    subs x1, x1, #0x1
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v16.8h, v18.8h, v17.8h
    subs x1, x1, #0x1
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
    beq label_10
    zip2 v17.8h, v22.8h, v21.8h
    zip2 v16.8h, v20.8h, v19.8h
    zip1 v16.8h, v17.8h, v16.8h
    str q16, [x6, #0]
    add x6, x6, #0x10
    add v2.8h, v2.8h, v16.8h
KAI_ASM_LABEL(label_10)  // Odds skip
    uaddw v1.4s, v1.4s, v2.4h
    uaddw2 v0.4s, v0.4s, v2.8h
    mul v1.4s, v1.4s, v31.4s
    mul v0.4s, v0.4s, v31.4s
    str q1, [x6, #0]
    str q0, [x6, #0x10]
    add x6, x6, #0x20
    str x6, [x0]
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block1_u8_u16_summing)

    KAI_ASM_END
