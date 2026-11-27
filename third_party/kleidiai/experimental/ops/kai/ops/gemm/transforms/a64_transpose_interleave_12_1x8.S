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

    KAI_ASM_CODE(a64_transpose_interleave_12_1x8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_12_1x8)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_12_1x8)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_12_1x8)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x6, #0xc
    movi v16.16b, #0
    add SP, SP, #-0x30
    str q16, [SP, #0]
    str q16, [SP, #0x10]
    mov x14, SP
    mov x5, x4
    str q16, [SP, #0x20]
    mov x13, x4
    add x5, x5, #0x7
    bfm x5, XZR, #0, #0x2  // bfc x5, #0, #0x3
    mul x12, x5, x6
    cbz x4, label_11
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x11, x1
    mov x4, #0x30
    cmp x13, #0x7
    mov x10, x2
    mov x9, x0
    add x28, x11, x3
    add x27, x28, x3
    csel x26, x4, XZR, GT
    add x25, x27, x3
    csel x24, x4, XZR, GE
    add x23, x25, x3
    add x22, x23, x3
    add x21, x22, x3
    add x20, x21, x3
    add x1, x20, x3
    csel x20, x20, x14, GT
    csel x21, x21, x14, GE
    cmp x13, #0x5
    csel x22, x22, x14, GT
    csel x8, x4, XZR, GT
    csel x23, x23, x14, GE
    csel x7, x4, XZR, GE
    cmp x13, #0x3
    csel x25, x25, x14, GT
    csel x6, x4, XZR, GT
    csel x27, x27, x14, GE
    csel x5, x4, XZR, GE
    cmp x13, #0x1
    csel x28, x28, x14, GT
    csel x4, x4, XZR, GT
    cmp x10, #0x30
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ldr q1, [x11], #0x10
    ldr q0, [x28, #0]
    sub x10, x10, #0x30
    ldr q31, [x27, #0]
    ldr q28, [x25, #0]
    cmp x10, #0x30
    ldr q27, [x23, #0]
    ldr q26, [x22, #0]
    ldr q25, [x21, #0]
    ldr q24, [x20, #0]
    ldr q23, [x11], #0x10
    ldr q30, [x28, #0x10]
    ldr q22, [x27, #0x10]
    ldr q21, [x25, #0x10]
    zip1 v15.16b, v1.16b, v27.16b
    zip1 v9.16b, v0.16b, v26.16b
    ldr q20, [x23, #0x10]
    ldr q19, [x22, #0x10]
    zip1 v18.16b, v31.16b, v25.16b
    zip1 v29.16b, v28.16b, v24.16b
    ldr q17, [x21, #0x10]
    ldr q16, [x20, #0x10]
    zip2 v14.16b, v1.16b, v27.16b
    zip2 v13.16b, v31.16b, v25.16b
    ldr q8, [x11], #0x10
    ldr q7, [x28, #0x20]
    zip2 v12.16b, v0.16b, v26.16b
    zip2 v6.16b, v28.16b, v24.16b
    ldr q5, [x27, #0x20]
    ldr q4, [x25, #0x20]
    zip1 v3.16b, v23.16b, v20.16b
    zip1 v11.16b, v30.16b, v19.16b
    ldr q28, [x23, #0x20]
    ldr q27, [x22, #0x20]
    zip1 v2.16b, v22.16b, v17.16b
    zip1 v1.16b, v21.16b, v16.16b
    ldr q26, [x21, #0x20]
    ldr q25, [x20, #0x20]
    zip2 v24.16b, v23.16b, v20.16b
    zip2 v23.16b, v22.16b, v17.16b
    zip2 v22.16b, v30.16b, v19.16b
    zip2 v21.16b, v21.16b, v16.16b
    add x28, x28, x4
    add x27, x27, x5
    zip1 v0.16b, v8.16b, v28.16b
    zip1 v10.16b, v7.16b, v27.16b
    add x25, x25, x6
    add x23, x23, x7
    zip1 v31.16b, v5.16b, v26.16b
    zip1 v30.16b, v4.16b, v25.16b
    add x22, x22, x8
    add x21, x21, x24
    zip1 v20.16b, v15.16b, v18.16b
    zip1 v19.16b, v9.16b, v29.16b
    add x20, x20, x26
    zip2 v18.16b, v15.16b, v18.16b
    zip2 v16.16b, v9.16b, v29.16b
    zip1 v29.16b, v14.16b, v13.16b
    zip1 v17.16b, v12.16b, v6.16b
    zip2 v9.16b, v8.16b, v28.16b
    zip2 v28.16b, v5.16b, v26.16b
    zip2 v8.16b, v7.16b, v27.16b
    zip2 v27.16b, v4.16b, v25.16b
    zip2 v7.16b, v14.16b, v13.16b
    zip2 v6.16b, v12.16b, v6.16b
    zip1 v5.16b, v3.16b, v2.16b
    zip1 v4.16b, v11.16b, v1.16b
    zip2 v3.16b, v3.16b, v2.16b
    zip2 v2.16b, v11.16b, v1.16b
    zip1 v26.16b, v24.16b, v23.16b
    zip1 v25.16b, v22.16b, v21.16b
    zip2 v24.16b, v24.16b, v23.16b
    zip2 v23.16b, v22.16b, v21.16b
    zip1 v1.16b, v0.16b, v31.16b
    zip1 v22.16b, v10.16b, v30.16b
    zip1 v21.16b, v20.16b, v19.16b
    zip2 v20.16b, v20.16b, v19.16b
    zip1 v19.16b, v18.16b, v16.16b
    zip2 v18.16b, v18.16b, v16.16b
    zip1 v16.16b, v29.16b, v17.16b
    zip2 v17.16b, v29.16b, v17.16b
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    zip2 v0.16b, v0.16b, v31.16b
    zip2 v31.16b, v10.16b, v30.16b
    str q19, [x9, #0x20]
    zip1 v30.16b, v9.16b, v28.16b
    zip1 v29.16b, v8.16b, v27.16b
    str q18, [x9, #0x30]
    zip2 v28.16b, v9.16b, v28.16b
    zip2 v27.16b, v8.16b, v27.16b
    str q16, [x9, #0x40]
    zip1 v21.16b, v7.16b, v6.16b
    zip2 v16.16b, v7.16b, v6.16b
    str q17, [x9, #0x50]
    add x9, x9, x12
    zip1 v20.16b, v5.16b, v4.16b
    zip2 v19.16b, v5.16b, v4.16b
    zip1 v18.16b, v3.16b, v2.16b
    zip2 v17.16b, v3.16b, v2.16b
    str q21, [x9, #0]
    str q16, [x9, #0x10]
    zip1 v16.16b, v26.16b, v25.16b
    zip2 v26.16b, v26.16b, v25.16b
    str q20, [x9, #0x20]
    zip1 v25.16b, v24.16b, v23.16b
    zip2 v24.16b, v24.16b, v23.16b
    str q19, [x9, #0x30]
    zip1 v23.16b, v1.16b, v22.16b
    zip2 v22.16b, v1.16b, v22.16b
    str q18, [x9, #0x40]
    zip1 v21.16b, v0.16b, v31.16b
    zip2 v20.16b, v0.16b, v31.16b
    str q17, [x9, #0x50]
    add x9, x9, x12
    zip1 v19.16b, v30.16b, v29.16b
    zip2 v18.16b, v30.16b, v29.16b
    str q16, [x9, #0]
    zip1 v17.16b, v28.16b, v27.16b
    zip2 v16.16b, v28.16b, v27.16b
    str q26, [x9, #0x10]
    str q25, [x9, #0x20]
    str q24, [x9, #0x30]
    str q23, [x9, #0x40]
    str q22, [x9, #0x50]
    add x9, x9, x12
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    add x9, x9, x12
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cmp x10, #0xc
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    ldr d22, [x11], #0x8
    ldr d23, [x28], #0x8
    sub x10, x10, #0xc
    ldr d21, [x27], #0x8
    ldr d20, [x25], #0x8
    cmp x10, #0xc
    ldr d18, [x23], #0x8
    ldr d17, [x22], #0x8
    ldr d19, [x21], #0x8
    ldr d16, [x20], #0x8
    ld1 { v22.s }[2], [x11], #0x4
    ld1 { v23.s }[2], [x28], #0x4
    ld1 { v21.s }[2], [x27], #0x4
    ld1 { v20.s }[2], [x25], #0x4
    ld1 { v18.s }[2], [x23], #0x4
    ld1 { v17.s }[2], [x22], #0x4
    ld1 { v19.s }[2], [x21], #0x4
    ld1 { v16.s }[2], [x20], #0x4
    zip1 v25.16b, v22.16b, v18.16b
    zip1 v24.16b, v23.16b, v17.16b
    zip2 v22.16b, v22.16b, v18.16b
    zip2 v23.16b, v23.16b, v17.16b
    zip1 v18.16b, v21.16b, v19.16b
    zip1 v17.16b, v20.16b, v16.16b
    zip2 v21.16b, v21.16b, v19.16b
    zip2 v16.16b, v20.16b, v16.16b
    zip1 v20.16b, v25.16b, v18.16b
    zip1 v19.16b, v24.16b, v17.16b
    zip2 v18.16b, v25.16b, v18.16b
    zip2 v17.16b, v24.16b, v17.16b
    zip1 v22.16b, v22.16b, v21.16b
    zip1 v16.16b, v23.16b, v16.16b
    zip1 v21.16b, v20.16b, v19.16b
    zip2 v20.16b, v20.16b, v19.16b
    zip1 v19.16b, v18.16b, v17.16b
    zip2 v18.16b, v18.16b, v17.16b
    zip1 v17.16b, v22.16b, v16.16b
    zip2 v16.16b, v22.16b, v16.16b
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    add x9, x9, x12
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    cbz x10, label_10
    cmp x10, #0x4
    movi v16.16b, #0
    str q16, [x9, #0]
    str q16, [x9, #0x10]
    str q16, [x9, #0x20]
    str q16, [x9, #0x30]
    str q16, [x9, #0x40]
    str q16, [x9, #0x50]
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
    ldr s23, [x11], #0x4
    ldr s22, [x28], #0x4
    sub x10, x10, #0x4
    ldr s21, [x27], #0x4
    ldr s20, [x25], #0x4
    cmp x10, #0x4
    ldr s18, [x23], #0x4
    ldr s19, [x22], #0x4
    ldr s17, [x21], #0x4
    ldr s16, [x20], #0x4
    zip1 v18.16b, v23.16b, v18.16b
    zip1 v19.16b, v22.16b, v19.16b
    zip1 v17.16b, v21.16b, v17.16b
    zip1 v16.16b, v20.16b, v16.16b
    zip1 v18.16b, v18.16b, v17.16b
    zip1 v16.16b, v19.16b, v16.16b
    zip1 v17.16b, v18.16b, v16.16b
    zip2 v16.16b, v18.16b, v16.16b
    str q17, [x9, #0]
    str q16, [x9, #0x10]
    add x9, x9, #0x20
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x10, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
    ldr b23, [x11], #0x1
    ldr b22, [x28], #0x1
    sub x10, x10, #0x1
    ldr b21, [x27], #0x1
    ldr b20, [x25], #0x1
    cmp x10, #0x1
    ldr b19, [x23], #0x1
    ldr b18, [x22], #0x1
    ldr b17, [x21], #0x1
    ldr b16, [x20], #0x1
    zip1 v19.16b, v23.16b, v19.16b
    zip1 v18.16b, v22.16b, v18.16b
    zip1 v17.16b, v21.16b, v17.16b
    zip1 v16.16b, v20.16b, v16.16b
    zip1 v17.16b, v19.16b, v17.16b
    zip1 v16.16b, v18.16b, v16.16b
    zip1 v16.16b, v17.16b, v16.16b
    str d16, [x9, #0]
    add x9, x9, #0x8
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x13, x13, #0x8
    add x0, x0, #0x60
    cmp x13, #0x1
    bge label_1
KAI_ASM_LABEL(label_11)  // Done
    add SP, SP, #0x30
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_12_1x8)

    KAI_ASM_END
