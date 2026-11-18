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

    KAI_ASM_CODE(a64_transpose_interleave_24_u8u16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_24_u8u16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_24_u8u16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_24_u8u16)
    stp x20, x21, [sp, -32]!
    str x22, [sp, 16]
    mov x6, #0xc
    mov x5, x4
    mov x22, x4
    lsl x5, x5, #0x1
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
    ldr q3, [x20], #0x10
    ldr q21, [x6], #0x10
    sub x8, x8, #0x18
    ldr q2, [x5], #0x10
    ldr q20, [x4], #0x10
    cmp x8, #0x18
    ldr d19, [x20], #0x8
    ldr d17, [x5], #0x8
    ldr d1, [x6], #0x8
    ushll2 v18.8h, v3.16b, #0
    ushll v0.8h, v21.8b, #0
    ldr d31, [x4], #0x8
    ushll2 v16.8h, v2.16b, #0
    ushll v30.8h, v20.8b, #0
    ushll v29.8h, v19.8b, #0
    ushll v28.8h, v17.8b, #0
    ushll2 v27.8h, v21.16b, #0
    ushll2 v26.8h, v20.16b, #0
    dup v17.2d, v18.d[0]
    dup v25.2d, v0.d[1]
    dup v24.2d, v16.d[0]
    dup v23.2d, v30.d[1]
    dup v22.2d, v18.d[1]
    dup v21.2d, v29.d[1]
    dup v20.2d, v16.d[1]
    dup v19.2d, v28.d[1]
    ushll v16.8h, v3.8b, #0
    ushll v18.8h, v2.8b, #0
    mov v17.d[1], v0.d[0]
    mov v25.d[1], v27.d[0]
    mov v24.d[1], v30.d[0]
    mov v23.d[1], v26.d[0]
    str q16, [x7, #0]
    str q17, [x7, #0x10]
    ushll v17.8h, v1.8b, #0
    ushll v16.8h, v31.8b, #0
    str q25, [x7, #0x20]
    mov v22.d[1], v29.d[0]
    mov v21.d[1], v27.d[1]
    str q18, [x7, #0x30]
    mov v20.d[1], v28.d[0]
    mov v19.d[1], v26.d[1]
    str q24, [x7, #0x40]
    str q23, [x7, #0x50]
    add x7, x7, x21
    str q22, [x7, #0]
    str q21, [x7, #0x10]
    str q17, [x7, #0x20]
    str q20, [x7, #0x30]
    str q19, [x7, #0x40]
    str q16, [x7, #0x50]
    add x7, x7, x21
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cmp x8, #0xc
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    ldr d19, [x6], #0x8
    ldr d18, [x4], #0x8
    sub x8, x8, #0xc
    ld1 { v19.s }[2], [x6], #0x4
    ldr d17, [x20], #0x8
    cmp x8, #0xc
    ld1 { v18.s }[2], [x4], #0x4
    ldr d16, [x5], #0x8
    ld1 { v17.s }[2], [x20], #0x4
    ld1 { v16.s }[2], [x5], #0x4
    ushll v25.8h, v19.8b, #0
    ushll2 v24.8h, v19.16b, #0
    ushll v23.8h, v18.8b, #0
    ushll2 v22.8h, v18.16b, #0
    ushll2 v21.8h, v17.16b, #0
    ushll2 v20.8h, v16.16b, #0
    dup v19.2d, v25.d[1]
    ushll v18.8h, v17.8b, #0
    dup v17.2d, v23.d[1]
    ushll v16.8h, v16.8b, #0
    mov v21.d[1], v25.d[0]
    mov v19.d[1], v24.d[0]
    mov v20.d[1], v23.d[0]
    mov v17.d[1], v22.d[0]
    str q18, [x7, #0]
    str q21, [x7, #0x10]
    str q19, [x7, #0x20]
    str q16, [x7, #0x30]
    str q20, [x7, #0x40]
    str q17, [x7, #0x50]
    add x7, x7, x21
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    cbz x8, label_10
    cmp x8, #0x4
    movi v16.8h, #0
    str q16, [x7, #0]
    str q16, [x7, #0x10]
    str q16, [x7, #0x20]
    str q16, [x7, #0x30]
    str q16, [x7, #0x40]
    str q16, [x7, #0x50]
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
    ldr s19, [x20], #0x4
    ldr s18, [x6], #0x4
    sub x8, x8, #0x4
    ldr s17, [x5], #0x4
    ldr s16, [x4], #0x4
    cmp x8, #0x4
    ushll v19.8h, v19.8b, #0
    ushll v18.8h, v18.8b, #0
    ushll v17.8h, v17.8b, #0
    ushll v16.8h, v16.8b, #0
    str d19, [x7, #0]
    str d18, [x7, #0x18]
    str d17, [x7, #0x30]
    str d16, [x7, #0x48]
    add x7, x7, #0x8
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x8, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
    ldr b19, [x20], #0x1
    ldr b18, [x6], #0x1
    sub x8, x8, #0x1
    ldr b17, [x5], #0x1
    ldr b16, [x4], #0x1
    cmp x8, #0x1
    ushll v19.8h, v19.8b, #0
    ushll v18.8h, v18.8b, #0
    ushll v17.8h, v17.8b, #0
    ushll v16.8h, v16.8b, #0
    str h19, [x7, #0]
    str h18, [x7, #0x18]
    str h17, [x7, #0x30]
    str h16, [x7, #0x48]
    add x7, x7, #0x2
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x22, x22, #0x4
    add x0, x0, #0x60
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
    ldr q18, [x20], #0x10
    sub x4, x4, #0x18
    ldr d16, [x20], #0x8
    cmp x4, #0x18
    ushll2 v17.8h, v18.16b, #0
    ushll v16.8h, v16.8b, #0
    ushll v19.8h, v18.8b, #0
    dup v18.2d, v17.d[1]
    dup v17.2d, v17.d[0]
    mov v18.d[1], v16.d[0]
    dup v16.2d, v16.d[1]
    str q19, [x7, #0]
    str d17, [x7, #0x10]
    add x7, x7, x21
    str q18, [x7, #0]
    str d16, [x7, #0x10]
    add x7, x7, x21
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: Unroll column loop skip
    cmp x4, #0xc
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: Column loop
    ldr d16, [x20], #0x8
    sub x4, x4, #0xc
    ld1 { v16.s }[2], [x20], #0x4
    cmp x4, #0xc
    ushll v17.8h, v16.8b, #0
    ushll2 v16.8h, v16.16b, #0
    str q17, [x7, #0]
    str d16, [x7, #0x10]
    add x7, x7, x21
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: Column loop skip
    cbz x4, label_21
    cmp x4, #0x4
    movi v16.8h, #0
    str q16, [x7, #0]
    str d16, [x7, #0x10]
    blt label_18
KAI_ASM_LABEL(label_17)  // Tail row loop: width 4 loop: loop
    ldr s16, [x20], #0x4
    sub x4, x4, #0x4
    cmp x4, #0x4
    ushll v16.8h, v16.8b, #0
    str d16, [x7, #0]
    add x7, x7, #0x8
    bge label_17
KAI_ASM_LABEL(label_18)  // Tail row loop: width 4 loop: skip
    cmp x4, #0x1
    blt label_20
KAI_ASM_LABEL(label_19)  // Tail row loop: width 1 loop: loop
    ldr b16, [x20], #0x1
    sub x4, x4, #0x1
    cmp x4, #0x1
    ushll v16.8h, v16.8b, #0
    str h16, [x7, #0]
    add x7, x7, #0x2
    bge label_19
KAI_ASM_LABEL(label_20)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_21)  // Tail row loop: odd col skip
    sub x22, x22, #0x1
    add x0, x0, #0x18
    cmp x22, #0x1
    bge label_12
KAI_ASM_LABEL(label_22)  // Done
    ldr x22, [sp, 16]
    ldp x20, x21, [sp], 32
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_24_u8u16)

    KAI_ASM_END
