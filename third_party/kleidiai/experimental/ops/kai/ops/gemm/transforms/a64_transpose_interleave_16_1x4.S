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

    KAI_ASM_CODE(a64_transpose_interleave_16_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_16_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_16_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_16_1x4)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x6, #0x10
    movi v16.16b, #0
    add SP, SP, #-0x10
    str q16, [SP, #0]
    mov x15, SP
    mov x5, x4
    mov x14, x4
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    cmp x14, #0x10
    mul x13, x5, x6
    blt label_9
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x12, x1
    mov x11, x2
    mov x10, x0
    add x9, x12, x3
    add x28, x9, x3
    add x27, x28, x3
    add x26, x27, x3
    cmp x11, #0x10
    add x25, x26, x3
    add x24, x25, x3
    add x23, x24, x3
    add x22, x23, x3
    add x21, x22, x3
    add x20, x21, x3
    add x8, x20, x3
    add x7, x8, x3
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    ldr q19, [x12], #0x10
    ldr q18, [x9], #0x10
    sub x11, x11, #0x10
    ldr q17, [x28], #0x10
    ldr q16, [x27], #0x10
    cmp x11, #0x10
    ldr q24, [x26], #0x10
    ldr q23, [x25], #0x10
    ldr q22, [x24], #0x10
    ldr q21, [x23], #0x10
    ldr q30, [x22], #0x10
    ldr q29, [x21], #0x10
    zip1 v3.16b, v19.16b, v17.16b
    zip1 v2.16b, v18.16b, v16.16b
    ldr q28, [x20], #0x10
    ldr q20, [x8], #0x10
    zip2 v1.16b, v19.16b, v17.16b
    zip2 v27.16b, v18.16b, v16.16b
    ldr q19, [x7], #0x10
    ldr q18, [x6], #0x10
    zip1 v26.16b, v24.16b, v22.16b
    zip1 v25.16b, v23.16b, v21.16b
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    zip2 v24.16b, v24.16b, v22.16b
    zip2 v23.16b, v23.16b, v21.16b
    zip1 v22.16b, v30.16b, v28.16b
    zip1 v21.16b, v29.16b, v20.16b
    zip2 v0.16b, v30.16b, v28.16b
    zip2 v20.16b, v29.16b, v20.16b
    zip1 v31.16b, v19.16b, v17.16b
    zip1 v30.16b, v18.16b, v16.16b
    zip2 v29.16b, v19.16b, v17.16b
    zip2 v28.16b, v18.16b, v16.16b
    zip1 v19.16b, v3.16b, v2.16b
    zip2 v18.16b, v3.16b, v2.16b
    zip1 v17.16b, v1.16b, v27.16b
    zip2 v16.16b, v1.16b, v27.16b
    zip1 v27.16b, v26.16b, v25.16b
    zip2 v26.16b, v26.16b, v25.16b
    zip1 v25.16b, v24.16b, v23.16b
    zip2 v24.16b, v24.16b, v23.16b
    str q19, [x10, #0]
    zip1 v23.16b, v22.16b, v21.16b
    zip2 v22.16b, v22.16b, v21.16b
    str q18, [x10, #0x10]
    zip1 v21.16b, v0.16b, v20.16b
    zip2 v20.16b, v0.16b, v20.16b
    str q17, [x10, #0x20]
    zip1 v19.16b, v31.16b, v30.16b
    zip2 v18.16b, v31.16b, v30.16b
    str q16, [x10, #0x30]
    zip1 v17.16b, v29.16b, v28.16b
    zip2 v16.16b, v29.16b, v28.16b
    str q27, [x10, #0x40]
    str q26, [x10, #0x50]
    str q25, [x10, #0x60]
    str q24, [x10, #0x70]
    str q23, [x10, #0x80]
    str q22, [x10, #0x90]
    str q21, [x10, #0xa0]
    str q20, [x10, #0xb0]
    str q19, [x10, #0xc0]
    str q18, [x10, #0xd0]
    str q17, [x10, #0xe0]
    str q16, [x10, #0xf0]
    add x10, x10, x13
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop skip
    cbz x11, label_8
    cmp x11, #0x4
    movi v16.16b, #0
    str q16, [x10, #0]
    str q16, [x10, #0x10]
    str q16, [x10, #0x20]
    str q16, [x10, #0x30]
    str q16, [x10, #0x40]
    str q16, [x10, #0x50]
    str q16, [x10, #0x60]
    str q16, [x10, #0x70]
    str q16, [x10, #0x80]
    str q16, [x10, #0x90]
    str q16, [x10, #0xa0]
    str q16, [x10, #0xb0]
    str q16, [x10, #0xc0]
    str q16, [x10, #0xd0]
    str q16, [x10, #0xe0]
    str q16, [x10, #0xf0]
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: width 4 loop: loop
    ldr s23, [x12], #0x4
    ldr s21, [x9], #0x4
    sub x11, x11, #0x4
    ldr s20, [x28], #0x4
    ldr s19, [x27], #0x4
    cmp x11, #0x4
    ldr s22, [x26], #0x4
    ldr s18, [x25], #0x4
    ldr s17, [x24], #0x4
    ldr s16, [x23], #0x4
    ldr s27, [x22], #0x4
    ldr s26, [x21], #0x4
    zip1 v25.16b, v23.16b, v20.16b
    zip1 v21.16b, v21.16b, v19.16b
    ldr s20, [x20], #0x4
    ldr s19, [x8], #0x4
    ldr s24, [x7], #0x4
    ldr s23, [x6], #0x4
    zip1 v22.16b, v22.16b, v17.16b
    zip1 v17.16b, v18.16b, v16.16b
    ldr s18, [x5], #0x4
    ldr s16, [x4], #0x4
    zip1 v21.16b, v25.16b, v21.16b
    zip1 v20.16b, v27.16b, v20.16b
    zip1 v19.16b, v26.16b, v19.16b
    zip1 v17.16b, v22.16b, v17.16b
    zip1 v18.16b, v24.16b, v18.16b
    zip1 v16.16b, v23.16b, v16.16b
    str q21, [x10, #0]
    str q17, [x10, #0x40]
    zip1 v17.16b, v20.16b, v19.16b
    zip1 v16.16b, v18.16b, v16.16b
    str q17, [x10, #0x80]
    str q16, [x10, #0xc0]
    add x10, x10, #0x10
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: width 4 loop: skip
    cmp x11, #0x1
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 1 loop: loop
    ldr b23, [x12], #0x1
    ldr b21, [x9], #0x1
    sub x11, x11, #0x1
    ldr b20, [x28], #0x1
    ldr b19, [x27], #0x1
    cmp x11, #0x1
    ldr b22, [x26], #0x1
    ldr b18, [x25], #0x1
    ldr b17, [x24], #0x1
    ldr b16, [x23], #0x1
    ldr b27, [x22], #0x1
    ldr b26, [x21], #0x1
    zip1 v25.16b, v23.16b, v20.16b
    zip1 v21.16b, v21.16b, v19.16b
    ldr b20, [x20], #0x1
    ldr b19, [x8], #0x1
    ldr b24, [x7], #0x1
    ldr b23, [x6], #0x1
    zip1 v22.16b, v22.16b, v17.16b
    zip1 v17.16b, v18.16b, v16.16b
    ldr b18, [x5], #0x1
    ldr b16, [x4], #0x1
    zip1 v21.16b, v25.16b, v21.16b
    zip1 v20.16b, v27.16b, v20.16b
    zip1 v19.16b, v26.16b, v19.16b
    zip1 v17.16b, v22.16b, v17.16b
    zip1 v18.16b, v24.16b, v18.16b
    zip1 v16.16b, v23.16b, v16.16b
    str s21, [x10, #0]
    str s17, [x10, #0x40]
    zip1 v17.16b, v20.16b, v19.16b
    zip1 v16.16b, v18.16b, v16.16b
    str s17, [x10, #0x80]
    str s16, [x10, #0xc0]
    add x10, x10, #0x4
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_8)  // Main row loop: odd col skip
    sub x14, x14, #0x10
    add x0, x0, #0x100
    cmp x14, #0x10
    bge label_1
    cbz x14, label_18
KAI_ASM_LABEL(label_9)  // Main loop skip
KAI_ASM_LABEL(label_10)  // Tail row loop: Head
    mov x12, x1
    mov x4, #0x10
    mov x7, x2
    cmp x14, #0x3
    mov x10, x0
    add x9, x12, x3
    add x28, x9, x3
    csel x6, x4, XZR, GT
    add x27, x28, x3
    csel x28, x28, x15, GE
    add x1, x27, x3
    csel x27, x27, x15, GT
    csel x5, x4, XZR, GE
    cmp x14, #0x1
    csel x9, x9, x15, GT
    csel x4, x4, XZR, GT
    cmp x7, #0x10
    blt label_12
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop
    ldr q20, [x12], #0x10
    ldr q21, [x9, #0]
    sub x7, x7, #0x10
    add x9, x9, x4
    ldr q19, [x28, #0]
    ldr q16, [x27, #0]
    cmp x7, #0x10
    add x28, x28, x5
    add x27, x27, x6
    zip1 v18.16b, v20.16b, v19.16b
    zip1 v17.16b, v21.16b, v16.16b
    zip2 v20.16b, v20.16b, v19.16b
    zip2 v16.16b, v21.16b, v16.16b
    zip1 v19.16b, v18.16b, v17.16b
    zip2 v18.16b, v18.16b, v17.16b
    zip1 v17.16b, v20.16b, v16.16b
    zip2 v16.16b, v20.16b, v16.16b
    str q19, [x10, #0]
    str q18, [x10, #0x10]
    str q17, [x10, #0x20]
    str q16, [x10, #0x30]
    add x10, x10, x13
    bge label_11
KAI_ASM_LABEL(label_12)  // Tail row loop: Column loop skip
    cbz x7, label_17
    cmp x7, #0x4
    movi v16.16b, #0
    str q16, [x10, #0]
    str q16, [x10, #0x10]
    str q16, [x10, #0x20]
    str q16, [x10, #0x30]
    blt label_14
KAI_ASM_LABEL(label_13)  // Tail row loop: width 4 loop: loop
    ldr s19, [x12], #0x4
    ldr s18, [x9], #0x4
    sub x7, x7, #0x4
    ldr s17, [x28], #0x4
    ldr s16, [x27], #0x4
    cmp x7, #0x4
    zip1 v17.16b, v19.16b, v17.16b
    zip1 v16.16b, v18.16b, v16.16b
    zip1 v16.16b, v17.16b, v16.16b
    str q16, [x10, #0]
    add x10, x10, #0x10
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: width 4 loop: skip
    cmp x7, #0x1
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: width 1 loop: loop
    ldr b19, [x12], #0x1
    ldr b18, [x9], #0x1
    sub x7, x7, #0x1
    ldr b17, [x28], #0x1
    ldr b16, [x27], #0x1
    cmp x7, #0x1
    zip1 v17.16b, v19.16b, v17.16b
    zip1 v16.16b, v18.16b, v16.16b
    zip1 v16.16b, v17.16b, v16.16b
    str s16, [x10, #0]
    add x10, x10, #0x4
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_17)  // Tail row loop: odd col skip
    sub x14, x14, #0x4
    add x0, x0, #0x40
    cmp x14, #0x1
    bge label_10
KAI_ASM_LABEL(label_18)  // Done
    add SP, SP, #0x10
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_16_1x4)

    KAI_ASM_END
