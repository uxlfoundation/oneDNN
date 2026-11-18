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

    KAI_ASM_CODE(a64_interleave8_block4_s8_s8_summing.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block4_s8_s8_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block4_s8_s8_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block4_s8_s8_summing)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    mov x26, #0
    movi v2.8h, #0
    movi v1.8h, #0
    movi v0.4s, #0
    movi v31.4s, #0
    ldr x25, [x1, #0]
    ldr x24, [x1, #0x8]
    ldr x23, [x1, #0x10]
    ldr x22, [x1, #0x18]
    ldr x21, [x1, #0x20]
    ldr x20, [x1, #0x28]
    cmp x3, #0x8
    dup v30.4s, w6
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
    ld1 { v0.4s }, [x6]
    ldr q31, [x6, #0x10]
KAI_ASM_LABEL(label_2)  // first_pass
    cmp x2, #0x10
    blt label_5
KAI_ASM_LABEL(label_3)  // Main loop head
    cmp x26, #0x1e
    ble label_4
    sadalp v0.4s, v2.8h
    movi v2.8h, #0
    mov x26, #0
    sadalp v31.4s, v1.8h
    movi v1.8h, #0
KAI_ASM_LABEL(label_4)  // no_accumulate_16
    ldr q23, [x25], #0x10
    ldr q19, [x24], #0x10
    subs x2, x2, #0x10
    add x26, x26, #0x1
    ldr q18, [x23], #0x10
    ldr q17, [x22], #0x10
    cmp x2, #0x10
    ldr q27, [x21], #0x10
    ldr q26, [x20], #0x10
    ldr q22, [x8], #0x10
    ldr q21, [x7], #0x10
    zip1 v20.4s, v23.4s, v18.4s
    zip1 v16.4s, v19.4s, v17.4s
    zip2 v25.4s, v23.4s, v18.4s
    zip2 v24.4s, v19.4s, v17.4s
    prfm pldl1keep, [x25, #0x70]
    prfm pldl1keep, [x24, #0x70]
    prfm pldl1keep, [x23, #0x70]
    prfm pldl1keep, [x22, #0x70]
    zip1 v19.4s, v27.4s, v22.4s
    zip1 v18.4s, v26.4s, v21.4s
    zip1 v17.4s, v20.4s, v16.4s
    zip2 v23.4s, v20.4s, v16.4s
    prfm pldl1keep, [x21, #0x70]
    prfm pldl1keep, [x20, #0x70]
    zip2 v22.4s, v27.4s, v22.4s
    zip2 v16.4s, v26.4s, v21.4s
    prfm pldl1keep, [x8, #0x70]
    prfm pldl1keep, [x7, #0x70]
    zip1 v21.4s, v19.4s, v18.4s
    zip2 v20.4s, v19.4s, v18.4s
    sadalp v2.8h, v17.16b
    zip1 v19.4s, v25.4s, v24.4s
    str q17, [x6, #0]
    zip1 v18.4s, v22.4s, v16.4s
    zip2 v17.4s, v25.4s, v24.4s
    sadalp v1.8h, v21.16b
    zip2 v16.4s, v22.4s, v16.4s
    str q21, [x6, #0x10]
    sadalp v2.8h, v23.16b
    str q23, [x6, #0x20]
    str q20, [x6, #0x30]
    sadalp v1.8h, v20.16b
    str q19, [x6, #0x40]
    sadalp v2.8h, v19.16b
    str q18, [x6, #0x50]
    sadalp v1.8h, v18.16b
    str q17, [x6, #0x60]
    sadalp v2.8h, v17.16b
    str q16, [x6, #0x70]
    add x6, x6, #0x80
    sadalp v1.8h, v16.16b
    bge label_3
KAI_ASM_LABEL(label_5)  // Main loop skip
    cbz x2, label_14
    tbz x2, #3, label_9
    ldr d29, [x25], #0x8
    ldr d28, [x24], #0x8
    ldr d27, [x23], #0x8
    ldr d26, [x22], #0x8
    ldr d25, [x21], #0x8
    ldr d24, [x20], #0x8
    ldr d23, [x8], #0x8
    ldr d22, [x7], #0x8
    tbz x2, #2, label_7
    ld1 { v29.s }[2], [x25], #0x4
    ld1 { v28.s }[2], [x24], #0x4
    ld1 { v27.s }[2], [x23], #0x4
    ld1 { v26.s }[2], [x22], #0x4
    ld1 { v25.s }[2], [x21], #0x4
    ld1 { v24.s }[2], [x20], #0x4
    ld1 { v23.s }[2], [x8], #0x4
    ld1 { v22.s }[2], [x7], #0x4
    tbz x2, #1, label_6
    ld1 { v29.h }[6], [x25], #0x2
    ld1 { v28.h }[6], [x24], #0x2
    mov x1, #0x4
    ld1 { v27.h }[6], [x23], #0x2
    ld1 { v26.h }[6], [x22], #0x2
    ld1 { v25.h }[6], [x21], #0x2
    ld1 { v24.h }[6], [x20], #0x2
    ld1 { v23.h }[6], [x8], #0x2
    ld1 { v22.h }[6], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v29.b }[14], [x25]
    ld1 { v28.b }[14], [x24]
    ld1 { v27.b }[14], [x23]
    ld1 { v26.b }[14], [x22]
    ld1 { v25.b }[14], [x21]
    ld1 { v24.b }[14], [x20]
    ld1 { v23.b }[14], [x8]
    ld1 { v22.b }[14], [x7]
    b label_13
KAI_ASM_LABEL(label_6)  // odd_loads_1_12
    mov x1, #0x3
    tbz x2, #0, label_13
    ld1 { v29.b }[12], [x25]
    ld1 { v28.b }[12], [x24]
    mov x1, #0x4
    ld1 { v27.b }[12], [x23]
    ld1 { v26.b }[12], [x22]
    ld1 { v25.b }[12], [x21]
    ld1 { v24.b }[12], [x20]
    ld1 { v23.b }[12], [x8]
    ld1 { v22.b }[12], [x7]
    b label_13
KAI_ASM_LABEL(label_7)  // odd_loads_2_8
    tbz x2, #1, label_8
    ld1 { v29.h }[4], [x25], #0x2
    ld1 { v28.h }[4], [x24], #0x2
    mov x1, #0x3
    ld1 { v27.h }[4], [x23], #0x2
    ld1 { v26.h }[4], [x22], #0x2
    ld1 { v25.h }[4], [x21], #0x2
    ld1 { v24.h }[4], [x20], #0x2
    ld1 { v23.h }[4], [x8], #0x2
    ld1 { v22.h }[4], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v29.b }[10], [x25]
    ld1 { v28.b }[10], [x24]
    ld1 { v27.b }[10], [x23]
    ld1 { v26.b }[10], [x22]
    ld1 { v25.b }[10], [x21]
    ld1 { v24.b }[10], [x20]
    ld1 { v23.b }[10], [x8]
    ld1 { v22.b }[10], [x7]
    b label_13
KAI_ASM_LABEL(label_8)  // odd_loads_1_8
    mov x1, #0x2
    tbz x2, #0, label_13
    ld1 { v29.b }[8], [x25]
    ld1 { v28.b }[8], [x24]
    mov x1, #0x3
    ld1 { v27.b }[8], [x23]
    ld1 { v26.b }[8], [x22]
    ld1 { v25.b }[8], [x21]
    ld1 { v24.b }[8], [x20]
    ld1 { v23.b }[8], [x8]
    ld1 { v22.b }[8], [x7]
    b label_13
KAI_ASM_LABEL(label_9)  // odd_loads_4_0
    tbz x2, #2, label_11
    ldr s29, [x25], #0x4
    ldr s28, [x24], #0x4
    ldr s27, [x23], #0x4
    ldr s26, [x22], #0x4
    ldr s25, [x21], #0x4
    ldr s24, [x20], #0x4
    ldr s23, [x8], #0x4
    ldr s22, [x7], #0x4
    tbz x2, #1, label_10
    ld1 { v29.h }[2], [x25], #0x2
    ld1 { v28.h }[2], [x24], #0x2
    mov x1, #0x2
    ld1 { v27.h }[2], [x23], #0x2
    ld1 { v26.h }[2], [x22], #0x2
    ld1 { v25.h }[2], [x21], #0x2
    ld1 { v24.h }[2], [x20], #0x2
    ld1 { v23.h }[2], [x8], #0x2
    ld1 { v22.h }[2], [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v29.b }[6], [x25]
    ld1 { v28.b }[6], [x24]
    ld1 { v27.b }[6], [x23]
    ld1 { v26.b }[6], [x22]
    ld1 { v25.b }[6], [x21]
    ld1 { v24.b }[6], [x20]
    ld1 { v23.b }[6], [x8]
    ld1 { v22.b }[6], [x7]
    b label_13
KAI_ASM_LABEL(label_10)  // odd_loads_1_4
    mov x1, #0x1
    tbz x2, #0, label_13
    ld1 { v29.b }[4], [x25]
    ld1 { v28.b }[4], [x24]
    mov x1, #0x2
    ld1 { v27.b }[4], [x23]
    ld1 { v26.b }[4], [x22]
    ld1 { v25.b }[4], [x21]
    ld1 { v24.b }[4], [x20]
    ld1 { v23.b }[4], [x8]
    ld1 { v22.b }[4], [x7]
    b label_13
KAI_ASM_LABEL(label_11)  // odd_loads_2_0
    tbz x2, #1, label_12
    ldr h29, [x25], #0x2
    ldr h28, [x24], #0x2
    mov x1, #0x1
    ldr h27, [x23], #0x2
    ldr h26, [x22], #0x2
    ldr h25, [x21], #0x2
    ldr h24, [x20], #0x2
    ldr h23, [x8], #0x2
    ldr h22, [x7], #0x2
    tbz x2, #0, label_13
    ld1 { v29.b }[2], [x25]
    ld1 { v28.b }[2], [x24]
    ld1 { v27.b }[2], [x23]
    ld1 { v26.b }[2], [x22]
    ld1 { v25.b }[2], [x21]
    ld1 { v24.b }[2], [x20]
    ld1 { v23.b }[2], [x8]
    ld1 { v22.b }[2], [x7]
    b label_13
KAI_ASM_LABEL(label_12)  // odd_loads_1_0
    ldr b29, [x25, #0]
    ldr b28, [x24, #0]
    mov x1, #0x1
    ldr b27, [x23, #0]
    ldr b26, [x22, #0]
    ldr b25, [x21, #0]
    ldr b24, [x20, #0]
    ldr b23, [x8, #0]
    ldr b22, [x7, #0]
KAI_ASM_LABEL(label_13)  // Odd load end
    zip1 v21.4s, v29.4s, v27.4s
    zip1 v20.4s, v28.4s, v26.4s
    subs x1, x1, #0x1
    zip1 v19.4s, v25.4s, v23.4s
    zip1 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x6, #0]
    sadalp v2.8h, v17.16b
    str q16, [x6, #0x10]
    add x6, x6, #0x20
    sadalp v1.8h, v16.16b
    beq label_14
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    subs x1, x1, #0x1
    str q17, [x6, #0]
    sadalp v2.8h, v17.16b
    str q16, [x6, #0x10]
    add x6, x6, #0x20
    sadalp v1.8h, v16.16b
    beq label_14
    zip2 v21.4s, v29.4s, v27.4s
    zip2 v20.4s, v28.4s, v26.4s
    subs x1, x1, #0x1
    zip2 v19.4s, v25.4s, v23.4s
    zip2 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x6, #0]
    sadalp v2.8h, v17.16b
    str q16, [x6, #0x10]
    add x6, x6, #0x20
    sadalp v1.8h, v16.16b
    beq label_14
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x6, #0]
    sadalp v2.8h, v17.16b
    str q16, [x6, #0x10]
    add x6, x6, #0x20
    sadalp v1.8h, v16.16b
KAI_ASM_LABEL(label_14)  // Odds skip
    sadalp v0.4s, v2.8h
    sadalp v31.4s, v1.8h
    mul v0.4s, v0.4s, v30.4s
    mul v31.4s, v31.4s, v30.4s
    str q0, [x6, #0]
    str q31, [x6, #0x10]
    add x6, x6, #0x20
    str x6, [x0]
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block4_s8_s8_summing)

    KAI_ASM_END
