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

    KAI_ASM_CODE(a64_transpose_interleave_48_bf16fp32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_48_bf16fp32)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_48_bf16fp32)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_48_bf16fp32)
    stp x20, x21, [sp, -32]!
    str x22, [sp, 16]
    mov x6, #0xc
    mov x5, x4
    mov x22, x4
    lsl x5, x5, #0x2
    cmp x22, #0x4
    mul x21, x5, x6
    blt label_11
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x20, x1
    mov x8, x2
    mov x7, x0
    add x6, x20, x3
    add x5, x6, x3
    add x4, x5, x3
    cmp x8, #0x18
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ldr q19, [x20], #0x10
    ldr q18, [x6], #0x10
    sub x8, x8, #0x18
    ldr q17, [x5], #0x10
    ldr q27, [x4], #0x10
    cmp x8, #0x18
    ldr q26, [x20], #0x10
    ldr q3, [x6], #0x10
    ldr q2, [x5], #0x10
    shll v16.4s, v19.4h, #0x10
    shll2 v25.4s, v19.8h, #0x10
    ldr q1, [x4], #0x10
    ldr q24, [x20], #0x10
    shll v23.4s, v18.4h, #0x10
    shll2 v22.4s, v18.8h, #0x10
    ldr q21, [x6], #0x10
    ldr q0, [x5], #0x10
    shll v20.4s, v26.4h, #0x10
    shll v19.4s, v3.4h, #0x10
    ldr q31, [x4], #0x10
    shll v18.4s, v17.4h, #0x10
    shll2 v17.4s, v17.8h, #0x10
    str q16, [x7, #0]
    shll v16.4s, v2.4h, #0x10
    shll v30.4s, v27.4h, #0x10
    str q25, [x7, #0x10]
    shll2 v29.4s, v27.8h, #0x10
    shll v28.4s, v1.4h, #0x10
    str q20, [x7, #0x20]
    str q23, [x7, #0x30]
    shll2 v27.4s, v26.8h, #0x10
    shll v26.4s, v24.4h, #0x10
    str q22, [x7, #0x40]
    shll2 v25.4s, v24.8h, #0x10
    shll2 v24.4s, v3.8h, #0x10
    str q19, [x7, #0x50]
    shll v23.4s, v21.4h, #0x10
    shll2 v22.4s, v21.8h, #0x10
    str q18, [x7, #0x60]
    shll2 v21.4s, v2.8h, #0x10
    shll v20.4s, v0.4h, #0x10
    str q17, [x7, #0x70]
    shll2 v19.4s, v0.8h, #0x10
    shll2 v18.4s, v1.8h, #0x10
    str q16, [x7, #0x80]
    shll v17.4s, v31.4h, #0x10
    shll2 v16.4s, v31.8h, #0x10
    str q30, [x7, #0x90]
    str q29, [x7, #0xa0]
    str q28, [x7, #0xb0]
    add x7, x7, x21
    str q27, [x7, #0]
    str q26, [x7, #0x10]
    str q25, [x7, #0x20]
    str q24, [x7, #0x30]
    str q23, [x7, #0x40]
    str q22, [x7, #0x50]
    str q21, [x7, #0x60]
    str q20, [x7, #0x70]
    str q19, [x7, #0x80]
    str q18, [x7, #0x90]
    str q17, [x7, #0xa0]
    str q16, [x7, #0xb0]
    add x7, x7, x21
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cmp x8, #0xc
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    ldr q16, [x20], #0x10
    ldr q22, [x6], #0x10
    sub x8, x8, #0xc
    ldr q27, [x5], #0x10
    ldr q26, [x4], #0x10
    cmp x8, #0xc
    ldr d21, [x20], #0x8
    ldr d20, [x6], #0x8
    ldr d19, [x5], #0x8
    shll v18.4s, v16.4h, #0x10
    shll2 v17.4s, v16.8h, #0x10
    ldr d16, [x4], #0x8
    shll v25.4s, v22.4h, #0x10
    shll2 v24.4s, v22.8h, #0x10
    shll v23.4s, v21.4h, #0x10
    shll v22.4s, v20.4h, #0x10
    shll v21.4s, v27.4h, #0x10
    shll2 v20.4s, v27.8h, #0x10
    str q18, [x7, #0]
    shll v19.4s, v19.4h, #0x10
    shll v18.4s, v26.4h, #0x10
    str q17, [x7, #0x10]
    shll2 v17.4s, v26.8h, #0x10
    shll v16.4s, v16.4h, #0x10
    str q23, [x7, #0x20]
    str q25, [x7, #0x30]
    str q24, [x7, #0x40]
    str q22, [x7, #0x50]
    str q21, [x7, #0x60]
    str q20, [x7, #0x70]
    str q19, [x7, #0x80]
    str q18, [x7, #0x90]
    str q17, [x7, #0xa0]
    str q16, [x7, #0xb0]
    add x7, x7, x21
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    cbz x8, label_10
    cmp x8, #0x4
    movi v16.16b, #0
    str q16, [x7, #0]
    str q16, [x7, #0x10]
    str q16, [x7, #0x20]
    str q16, [x7, #0x30]
    str q16, [x7, #0x40]
    str q16, [x7, #0x50]
    str q16, [x7, #0x60]
    str q16, [x7, #0x70]
    str q16, [x7, #0x80]
    str q16, [x7, #0x90]
    str q16, [x7, #0xa0]
    str q16, [x7, #0xb0]
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
    ldr d19, [x20], #0x8
    ldr d18, [x6], #0x8
    sub x8, x8, #0x4
    ldr d17, [x5], #0x8
    ldr d16, [x4], #0x8
    cmp x8, #0x4
    shll v19.4s, v19.4h, #0x10
    shll v18.4s, v18.4h, #0x10
    shll v17.4s, v17.4h, #0x10
    shll v16.4s, v16.4h, #0x10
    str q19, [x7, #0]
    str q18, [x7, #0x30]
    str q17, [x7, #0x60]
    str q16, [x7, #0x90]
    add x7, x7, #0x10
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x8, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
    ldr h19, [x20], #0x2
    ldr h18, [x6], #0x2
    sub x8, x8, #0x1
    ldr h17, [x5], #0x2
    ldr h16, [x4], #0x2
    cmp x8, #0x1
    shll v19.4s, v19.4h, #0x10
    shll v18.4s, v18.4h, #0x10
    shll v17.4s, v17.4h, #0x10
    shll v16.4s, v16.4h, #0x10
    str s19, [x7, #0]
    str s18, [x7, #0x30]
    str s17, [x7, #0x60]
    str s16, [x7, #0x90]
    add x7, x7, #0x4
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x22, x22, #0x4
    add x0, x0, #0xc0
    cmp x22, #0x4
    bge label_1
    cbz x22, label_22
KAI_ASM_LABEL(label_11)  // Main loop skip
KAI_ASM_LABEL(label_12)  // Tail row loop: Head
    mov x4, x2
    mov x20, x1
    mov x7, x0
    cmp x4, #0x18
    add x1, x20, x3
    blt label_14
KAI_ASM_LABEL(label_13)  // Tail row loop: Unroll column loop
    ldr q16, [x20], #0x10
    sub x4, x4, #0x18
    ldr q18, [x20], #0x10
    ldr q20, [x20], #0x10
    cmp x4, #0x18
    shll v17.4s, v16.4h, #0x10
    shll2 v16.4s, v16.8h, #0x10
    shll v19.4s, v18.4h, #0x10
    shll2 v18.4s, v18.8h, #0x10
    str q17, [x7, #0]
    str q16, [x7, #0x10]
    shll v17.4s, v20.4h, #0x10
    shll2 v16.4s, v20.8h, #0x10
    str q19, [x7, #0x20]
    add x7, x7, x21
    str q18, [x7, #0]
    str q17, [x7, #0x10]
    str q16, [x7, #0x20]
    add x7, x7, x21
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: Unroll column loop skip
    cmp x4, #0xc
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: Column loop
    ldr q17, [x20], #0x10
    sub x4, x4, #0xc
    ldr d16, [x20], #0x8
    cmp x4, #0xc
    shll v18.4s, v17.4h, #0x10
    shll2 v17.4s, v17.8h, #0x10
    shll v16.4s, v16.4h, #0x10
    str q18, [x7, #0]
    str q17, [x7, #0x10]
    str q16, [x7, #0x20]
    add x7, x7, x21
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: Column loop skip
    cbz x4, label_21
    cmp x4, #0x4
    movi v16.16b, #0
    str q16, [x7, #0]
    str q16, [x7, #0x10]
    str q16, [x7, #0x20]
    blt label_18
KAI_ASM_LABEL(label_17)  // Tail row loop: width 4 loop: loop
    ldr d16, [x20], #0x8
    sub x4, x4, #0x4
    cmp x4, #0x4
    shll v16.4s, v16.4h, #0x10
    str q16, [x7, #0]
    add x7, x7, #0x10
    bge label_17
KAI_ASM_LABEL(label_18)  // Tail row loop: width 4 loop: skip
    cmp x4, #0x1
    blt label_20
KAI_ASM_LABEL(label_19)  // Tail row loop: width 1 loop: loop
    ldr h16, [x20], #0x2
    sub x4, x4, #0x1
    cmp x4, #0x1
    shll v16.4s, v16.4h, #0x10
    str s16, [x7, #0]
    add x7, x7, #0x4
    bge label_19
KAI_ASM_LABEL(label_20)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_21)  // Tail row loop: odd col skip
    sub x22, x22, #0x1
    add x0, x0, #0x30
    cmp x22, #0x1
    bge label_12
KAI_ASM_LABEL(label_22)  // Done
    ldr x22, [sp, 16]
    ldp x20, x21, [sp], 32
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_48_bf16fp32)

    KAI_ASM_END
