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

    KAI_ASM_CODE(a64_interleave8_block8_s8_s8_summing.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block8_s8_s8_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block8_s8_s8_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block8_s8_s8_summing)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    mov x26, #0
    movi v6.8h, #0
    movi v5.8h, #0
    movi v4.8h, #0
    movi v3.8h, #0
    movi v2.4s, #0
    movi v1.4s, #0
    ldr x25, [x1, #0]
    ldr x24, [x1, #0x8]
    ldr x23, [x1, #0x10]
    ldr x22, [x1, #0x18]
    cmp x3, #0x8
    dup v0.4s, w6
    ldr x21, [x1, #0x20]
    ldr x20, [x1, #0x28]
    movi v31.4s, #0
    movi v30.4s, #0
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
    movi v29.4s, #0
    movi v28.4s, #0
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
    ld1 { v29.4s }, [x6]
    ldr q28, [x6, #0x10]
KAI_ASM_LABEL(label_2)  // first_pass
    cmp x2, #0x10
    blt label_5
KAI_ASM_LABEL(label_3)  // Main loop head
    cmp x26, #0x3e
    ble label_4
    sadalp v2.4s, v6.8h
    movi v6.8h, #0
    mov x26, #0
    sadalp v1.4s, v5.8h
    movi v5.8h, #0
    sadalp v31.4s, v4.8h
    movi v4.8h, #0
    sadalp v30.4s, v3.8h
    movi v3.8h, #0
KAI_ASM_LABEL(label_4)  // no_accumulate_16
    ldr q22, [x25], #0x10
    ldr q21, [x24], #0x10
    subs x2, x2, #0x10
    add x26, x26, #0x1
    ldr q20, [x23], #0x10
    ldr q19, [x22], #0x10
    cmp x2, #0x10
    ldr q18, [x21], #0x10
    ldr q17, [x20], #0x10
    ldr q24, [x8], #0x10
    ldr q16, [x7], #0x10
    zip1 v23.2d, v22.2d, v21.2d
    zip2 v22.2d, v22.2d, v21.2d
    zip1 v21.2d, v20.2d, v19.2d
    zip2 v20.2d, v20.2d, v19.2d
    prfm pldl1keep, [x25, #0x70]
    prfm pldl1keep, [x24, #0x70]
    zip1 v19.2d, v18.2d, v17.2d
    zip2 v18.2d, v18.2d, v17.2d
    prfm pldl1keep, [x23, #0x70]
    prfm pldl1keep, [x22, #0x70]
    zip1 v17.2d, v24.2d, v16.2d
    sadalp v6.8h, v23.16b
    prfm pldl1keep, [x21, #0x70]
    prfm pldl1keep, [x20, #0x70]
    sadalp v5.8h, v21.16b
    zip2 v16.2d, v24.2d, v16.2d
    prfm pldl1keep, [x8, #0x70]
    prfm pldl1keep, [x7, #0x70]
    sadalp v4.8h, v19.16b
    str q23, [x6, #0]
    sadalp v3.8h, v17.16b
    str q21, [x6, #0x10]
    sadalp v6.8h, v22.16b
    str q19, [x6, #0x20]
    sadalp v5.8h, v20.16b
    str q17, [x6, #0x30]
    sadalp v4.8h, v18.16b
    sadalp v3.8h, v16.16b
    str q22, [x6, #0x40]
    str q20, [x6, #0x50]
    str q18, [x6, #0x60]
    str q16, [x6, #0x70]
    add x6, x6, #0x80
    bge label_3
KAI_ASM_LABEL(label_5)  // Main loop skip
    cbz x2, label_14
    tbz x2, #3, label_9
    ldr d27, [x25], #0x8
    ldr d26, [x24], #0x8
    ldr d25, [x23], #0x8
    ldr d24, [x22], #0x8
    ldr d23, [x21], #0x8
    ldr d22, [x20], #0x8
    ldr d21, [x8], #0x8
    ldr d20, [x7], #0x8
    tbz x2, #2, label_7
    ld1 { v27.s }[2], [x25], #0x4
    ld1 { v26.s }[2], [x24], #0x4
    ld1 { v25.s }[2], [x23], #0x4
    ld1 { v24.s }[2], [x22], #0x4
    ld1 { v23.s }[2], [x21], #0x4
    ld1 { v22.s }[2], [x20], #0x4
    ld1 { v21.s }[2], [x8], #0x4
    ld1 { v20.s }[2], [x7], #0x4
    tbz x2, #1, label_6
    ld1 { v27.h }[6], [x25], #0x2
    ld1 { v26.h }[6], [x24], #0x2
    mov x1, #0x2
    ld1 { v25.h }[6], [x23], #0x2
    ld1 { v24.h }[6], [x22], #0x2
    ld1 { v23.h }[6], [x21], #0x2
    ld1 { v22.h }[6], [x20], #0x2
    ld1 { v21.h }[6], [x8], #0x2
    ld1 { v20.h }[6], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v27.b }[14], [x25]
    ld1 { v26.b }[14], [x24]
    ld1 { v25.b }[14], [x23]
    ld1 { v24.b }[14], [x22]
    ld1 { v23.b }[14], [x21]
    ld1 { v22.b }[14], [x20]
    ld1 { v21.b }[14], [x8]
    ld1 { v20.b }[14], [x7]
    b label_13
KAI_ASM_LABEL(label_6)  // odd_loads_1_12
    mov x1, #0x2
    tbz x2, #0, label_13
    ld1 { v27.b }[12], [x25]
    ld1 { v26.b }[12], [x24]
    ld1 { v25.b }[12], [x23]
    ld1 { v24.b }[12], [x22]
    ld1 { v23.b }[12], [x21]
    ld1 { v22.b }[12], [x20]
    ld1 { v21.b }[12], [x8]
    ld1 { v20.b }[12], [x7]
    b label_13
KAI_ASM_LABEL(label_7)  // odd_loads_2_8
    tbz x2, #1, label_8
    ld1 { v27.h }[4], [x25], #0x2
    ld1 { v26.h }[4], [x24], #0x2
    mov x1, #0x2
    ld1 { v25.h }[4], [x23], #0x2
    ld1 { v24.h }[4], [x22], #0x2
    ld1 { v23.h }[4], [x21], #0x2
    ld1 { v22.h }[4], [x20], #0x2
    ld1 { v21.h }[4], [x8], #0x2
    ld1 { v20.h }[4], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v27.b }[10], [x25]
    ld1 { v26.b }[10], [x24]
    ld1 { v25.b }[10], [x23]
    ld1 { v24.b }[10], [x22]
    ld1 { v23.b }[10], [x21]
    ld1 { v22.b }[10], [x20]
    ld1 { v21.b }[10], [x8]
    ld1 { v20.b }[10], [x7]
    b label_13
KAI_ASM_LABEL(label_8)  // odd_loads_1_8
    mov x1, #0x1
    tbz x2, #0, label_13
    ld1 { v27.b }[8], [x25]
    ld1 { v26.b }[8], [x24]
    mov x1, #0x2
    ld1 { v25.b }[8], [x23]
    ld1 { v24.b }[8], [x22]
    ld1 { v23.b }[8], [x21]
    ld1 { v22.b }[8], [x20]
    ld1 { v21.b }[8], [x8]
    ld1 { v20.b }[8], [x7]
    b label_13
KAI_ASM_LABEL(label_9)  // odd_loads_4_0
    tbz x2, #2, label_11
    ldr s27, [x25], #0x4
    ldr s26, [x24], #0x4
    ldr s25, [x23], #0x4
    ldr s24, [x22], #0x4
    ldr s23, [x21], #0x4
    ldr s22, [x20], #0x4
    ldr s21, [x8], #0x4
    ldr s20, [x7], #0x4
    tbz x2, #1, label_10
    ld1 { v27.h }[2], [x25], #0x2
    ld1 { v26.h }[2], [x24], #0x2
    mov x1, #0x1
    ld1 { v25.h }[2], [x23], #0x2
    ld1 { v24.h }[2], [x22], #0x2
    ld1 { v23.h }[2], [x21], #0x2
    ld1 { v22.h }[2], [x20], #0x2
    ld1 { v21.h }[2], [x8], #0x2
    ld1 { v20.h }[2], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v27.b }[6], [x25]
    ld1 { v26.b }[6], [x24]
    ld1 { v25.b }[6], [x23]
    ld1 { v24.b }[6], [x22]
    ld1 { v23.b }[6], [x21]
    ld1 { v22.b }[6], [x20]
    ld1 { v21.b }[6], [x8]
    ld1 { v20.b }[6], [x7]
    b label_13
KAI_ASM_LABEL(label_10)  // odd_loads_1_4
    mov x1, #0x1
    tbz x2, #0, label_13
    ld1 { v27.b }[4], [x25]
    ld1 { v26.b }[4], [x24]
    ld1 { v25.b }[4], [x23]
    ld1 { v24.b }[4], [x22]
    ld1 { v23.b }[4], [x21]
    ld1 { v22.b }[4], [x20]
    ld1 { v21.b }[4], [x8]
    ld1 { v20.b }[4], [x7]
    b label_13
KAI_ASM_LABEL(label_11)  // odd_loads_2_0
    tbz x2, #1, label_12
    ldr h27, [x25], #0x2
    ldr h26, [x24], #0x2
    mov x1, #0x1
    ldr h25, [x23], #0x2
    ldr h24, [x22], #0x2
    ldr h23, [x21], #0x2
    ldr h22, [x20], #0x2
    ldr h21, [x8], #0x2
    ldr h20, [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v27.b }[2], [x25]
    ld1 { v26.b }[2], [x24]
    ld1 { v25.b }[2], [x23]
    ld1 { v24.b }[2], [x22]
    ld1 { v23.b }[2], [x21]
    ld1 { v22.b }[2], [x20]
    ld1 { v21.b }[2], [x8]
    ld1 { v20.b }[2], [x7]
    b label_13
KAI_ASM_LABEL(label_12)  // odd_loads_1_0
    ldr b27, [x25, #0]
    ldr b26, [x24, #0]
    mov x1, #0x1
    ldr b25, [x23, #0]
    ldr b24, [x22, #0]
    ldr b23, [x21, #0]
    ldr b22, [x20, #0]
    ldr b21, [x8, #0]
    ldr b20, [x7, #0]
KAI_ASM_LABEL(label_13)  // Odd load end
    zip1 v19.2d, v27.2d, v26.2d
    zip1 v18.2d, v25.2d, v24.2d
    subs x1, x1, #0x1
    zip1 v17.2d, v23.2d, v22.2d
    zip1 v16.2d, v21.2d, v20.2d
    str q19, [x6, #0]
    sadalp v6.8h, v19.16b
    str q18, [x6, #0x10]
    sadalp v5.8h, v18.16b
    str q17, [x6, #0x20]
    sadalp v4.8h, v17.16b
    str q16, [x6, #0x30]
    sadalp v3.8h, v16.16b
    add x6, x6, #0x40
    beq label_14
    zip2 v19.2d, v27.2d, v26.2d
    zip2 v18.2d, v25.2d, v24.2d
    zip2 v17.2d, v23.2d, v22.2d
    zip2 v16.2d, v21.2d, v20.2d
    str q19, [x6, #0]
    sadalp v6.8h, v19.16b
    str q18, [x6, #0x10]
    sadalp v5.8h, v18.16b
    str q17, [x6, #0x20]
    sadalp v4.8h, v17.16b
    str q16, [x6, #0x30]
    sadalp v3.8h, v16.16b
    add x6, x6, #0x40
KAI_ASM_LABEL(label_14)  // Odds skip
    sadalp v2.4s, v6.8h
    sadalp v1.4s, v5.8h
    sadalp v31.4s, v4.8h
    sadalp v30.4s, v3.8h
    addp v2.4s, v2.4s, v1.4s
    addp v16.4s, v31.4s, v30.4s
    add v2.4s, v2.4s, v29.4s
    add v16.4s, v16.4s, v28.4s
    mul v2.4s, v2.4s, v0.4s
    mul v16.4s, v16.4s, v0.4s
    str q2, [x6, #0]
    str q16, [x6, #0x10]
    add x6, x6, #0x20
    str x6, [x0]
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block8_s8_s8_summing)

    KAI_ASM_END
