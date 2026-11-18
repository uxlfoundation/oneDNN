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

    KAI_ASM_CODE(a64_transpose_interleave_16_2x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_16_2x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_16_2x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_16_2x4)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    mov x6, #0x10
    movi v26.8h, #0
    add SP, SP, #-0x40
    str q26, [SP, #0]
    str q26, [SP, #0x10]
    mov x27, SP
    mov x5, x4
    mov x26, x4
    str q26, [SP, #0x20]
    str q26, [SP, #0x30]
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    cmp x26, #0x8
    lsl x5, x5, #0x1
    mul x25, x5, x6
    blt label_11
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    mov x23, x2
    mov x22, x0
    add x21, x24, x3
    add x20, x21, x3
    add x8, x20, x3
    add x7, x8, x3
    cmp x23, #0x20
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ldr q27, [x24], #0x10
    ldr q0, [x21], #0x10
    sub x23, x23, #0x20
    ldr q15, [x20], #0x10
    ldr q14, [x8], #0x10
    cmp x23, #0x20
    ldr q2, [x7], #0x10
    ldr q21, [x6], #0x10
    ldr q9, [x5], #0x10
    ldr q19, [x4], #0x10
    ldr q10, [x24], #0x10
    ldr q13, [x21], #0x10
    zip1 v29.8h, v27.8h, v15.8h
    zip1 v25.8h, v0.8h, v14.8h
    ldr q18, [x20], #0x10
    ldr q22, [x8], #0x10
    zip2 v16.8h, v27.8h, v15.8h
    zip2 v30.8h, v0.8h, v14.8h
    ldr q31, [x7], #0x10
    ldr q7, [x6], #0x10
    zip1 v23.8h, v2.8h, v9.8h
    zip1 v11.8h, v21.8h, v19.8h
    ldr q17, [x5], #0x10
    ldr q6, [x4], #0x10
    zip2 v15.8h, v2.8h, v9.8h
    zip2 v12.8h, v21.8h, v19.8h
    ldr q19, [x24], #0x10
    ldr q9, [x21], #0x10
    zip1 v4.8h, v10.8h, v18.8h
    zip1 v1.8h, v13.8h, v22.8h
    ldr q5, [x20], #0x10
    ldr q8, [x8], #0x10
    zip2 v20.8h, v10.8h, v18.8h
    zip2 v14.8h, v13.8h, v22.8h
    ldr q2, [x7], #0x10
    ldr q13, [x6], #0x10
    zip1 v27.8h, v31.8h, v17.8h
    zip1 v3.8h, v7.8h, v6.8h
    ldr q10, [x5], #0x10
    ldr q28, [x4], #0x10
    zip2 v0.8h, v31.8h, v17.8h
    zip2 v21.8h, v7.8h, v6.8h
    ldr q6, [x24], #0x10
    ldr q22, [x21], #0x10
    zip1 v26.8h, v19.8h, v5.8h
    zip1 v17.8h, v9.8h, v8.8h
    ldr q31, [x20], #0x10
    ldr q18, [x8], #0x10
    zip2 v5.8h, v19.8h, v5.8h
    zip2 v24.8h, v9.8h, v8.8h
    ldr q19, [x7], #0x10
    ldr q9, [x6], #0x10
    zip1 v7.8h, v2.8h, v10.8h
    zip1 v8.8h, v13.8h, v28.8h
    zip2 v10.8h, v2.8h, v10.8h
    ldr q2, [x5], #0x10
    zip2 v13.8h, v13.8h, v28.8h
    zip1 v28.8h, v6.8h, v31.8h
    zip2 v31.8h, v6.8h, v31.8h
    zip1 v6.8h, v22.8h, v18.8h
    zip2 v22.8h, v22.8h, v18.8h
    zip1 v18.8h, v19.8h, v2.8h
    zip2 v2.8h, v19.8h, v2.8h
    zip1 v19.8h, v29.8h, v25.8h
    zip2 v25.8h, v29.8h, v25.8h
    ldr q29, [x4], #0x10
    str q19, [x22, #0]
    zip1 v19.8h, v16.8h, v30.8h
    zip2 v16.8h, v16.8h, v30.8h
    zip1 v30.8h, v4.8h, v1.8h
    zip2 v1.8h, v4.8h, v1.8h
    zip1 v4.8h, v9.8h, v29.8h
    zip2 v29.8h, v9.8h, v29.8h
    str q25, [x22, #0x10]
    zip1 v25.8h, v20.8h, v14.8h
    zip2 v9.8h, v20.8h, v14.8h
    str q19, [x22, #0x20]
    zip1 v14.8h, v23.8h, v11.8h
    zip2 v20.8h, v23.8h, v11.8h
    str q16, [x22, #0x30]
    zip1 v23.8h, v15.8h, v12.8h
    zip2 v16.8h, v15.8h, v12.8h
    str q30, [x22, #0x40]
    zip1 v15.8h, v27.8h, v3.8h
    zip2 v3.8h, v27.8h, v3.8h
    str q1, [x22, #0x50]
    zip1 v12.8h, v0.8h, v21.8h
    zip2 v11.8h, v0.8h, v21.8h
    str q25, [x22, #0x60]
    str q9, [x22, #0x70]
    zip1 v0.8h, v26.8h, v17.8h
    zip2 v30.8h, v26.8h, v17.8h
    str q14, [x22, #0x80]
    zip1 v1.8h, v5.8h, v24.8h
    zip2 v26.8h, v5.8h, v24.8h
    str q20, [x22, #0x90]
    zip1 v27.8h, v28.8h, v6.8h
    zip2 v28.8h, v28.8h, v6.8h
    str q23, [x22, #0xa0]
    zip1 v25.8h, v31.8h, v22.8h
    zip2 v24.8h, v31.8h, v22.8h
    str q16, [x22, #0xb0]
    zip1 v23.8h, v7.8h, v8.8h
    zip2 v22.8h, v7.8h, v8.8h
    str q15, [x22, #0xc0]
    zip1 v21.8h, v10.8h, v13.8h
    zip2 v20.8h, v10.8h, v13.8h
    str q3, [x22, #0xd0]
    zip1 v19.8h, v18.8h, v4.8h
    zip2 v18.8h, v18.8h, v4.8h
    str q12, [x22, #0xe0]
    zip1 v17.8h, v2.8h, v29.8h
    zip2 v16.8h, v2.8h, v29.8h
    str q11, [x22, #0xf0]
    add x22, x22, x25
    str q0, [x22, #0]
    str q30, [x22, #0x10]
    str q1, [x22, #0x20]
    str q26, [x22, #0x30]
    str q27, [x22, #0x40]
    str q28, [x22, #0x50]
    str q25, [x22, #0x60]
    str q24, [x22, #0x70]
    str q23, [x22, #0x80]
    str q22, [x22, #0x90]
    str q21, [x22, #0xa0]
    str q20, [x22, #0xb0]
    str q19, [x22, #0xc0]
    str q18, [x22, #0xd0]
    str q17, [x22, #0xe0]
    str q16, [x22, #0xf0]
    add x22, x22, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cmp x23, #0x10
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    ldr q19, [x24], #0x10
    ldr q18, [x21], #0x10
    sub x23, x23, #0x10
    ldr q17, [x20], #0x10
    ldr q16, [x8], #0x10
    cmp x23, #0x10
    ldr q31, [x7], #0x10
    ldr q30, [x6], #0x10
    ldr q23, [x5], #0x10
    ldr q22, [x4], #0x10
    ldr q29, [x24], #0x10
    ldr q28, [x21], #0x10
    zip1 v27.8h, v19.8h, v17.8h
    zip1 v26.8h, v18.8h, v16.8h
    ldr q21, [x20], #0x10
    ldr q20, [x8], #0x10
    zip2 v25.8h, v19.8h, v17.8h
    zip2 v24.8h, v18.8h, v16.8h
    ldr q19, [x7], #0x10
    ldr q18, [x6], #0x10
    zip1 v3.8h, v31.8h, v23.8h
    zip1 v2.8h, v30.8h, v22.8h
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    zip2 v1.8h, v31.8h, v23.8h
    zip2 v0.8h, v30.8h, v22.8h
    zip1 v23.8h, v29.8h, v21.8h
    zip1 v22.8h, v28.8h, v20.8h
    zip2 v21.8h, v29.8h, v21.8h
    zip2 v20.8h, v28.8h, v20.8h
    zip1 v31.8h, v19.8h, v17.8h
    zip1 v30.8h, v18.8h, v16.8h
    zip2 v29.8h, v19.8h, v17.8h
    zip2 v28.8h, v18.8h, v16.8h
    zip1 v19.8h, v27.8h, v26.8h
    zip2 v18.8h, v27.8h, v26.8h
    zip1 v17.8h, v25.8h, v24.8h
    zip2 v16.8h, v25.8h, v24.8h
    zip1 v27.8h, v23.8h, v22.8h
    zip2 v26.8h, v23.8h, v22.8h
    zip1 v25.8h, v21.8h, v20.8h
    zip2 v24.8h, v21.8h, v20.8h
    str q19, [x22, #0]
    zip1 v23.8h, v3.8h, v2.8h
    zip2 v22.8h, v3.8h, v2.8h
    str q18, [x22, #0x10]
    zip1 v21.8h, v1.8h, v0.8h
    zip2 v20.8h, v1.8h, v0.8h
    str q17, [x22, #0x20]
    zip1 v19.8h, v31.8h, v30.8h
    zip2 v18.8h, v31.8h, v30.8h
    str q16, [x22, #0x30]
    zip1 v17.8h, v29.8h, v28.8h
    zip2 v16.8h, v29.8h, v28.8h
    str q27, [x22, #0x40]
    str q26, [x22, #0x50]
    str q25, [x22, #0x60]
    str q24, [x22, #0x70]
    str q23, [x22, #0x80]
    str q22, [x22, #0x90]
    str q21, [x22, #0xa0]
    str q20, [x22, #0xb0]
    str q19, [x22, #0xc0]
    str q18, [x22, #0xd0]
    str q17, [x22, #0xe0]
    str q16, [x22, #0xf0]
    add x22, x22, x25
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    cbz x23, label_10
    cmp x23, #0x4
    movi v16.8h, #0
    str q16, [x22, #0]
    str q16, [x22, #0x10]
    str q16, [x22, #0x20]
    str q16, [x22, #0x30]
    str q16, [x22, #0x40]
    str q16, [x22, #0x50]
    str q16, [x22, #0x60]
    str q16, [x22, #0x70]
    str q16, [x22, #0x80]
    str q16, [x22, #0x90]
    str q16, [x22, #0xa0]
    str q16, [x22, #0xb0]
    str q16, [x22, #0xc0]
    str q16, [x22, #0xd0]
    str q16, [x22, #0xe0]
    str q16, [x22, #0xf0]
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
    ldr d23, [x24], #0x8
    ldr d22, [x21], #0x8
    sub x23, x23, #0x4
    ldr d18, [x20], #0x8
    ldr d17, [x8], #0x8
    cmp x23, #0x4
    ldr d20, [x7], #0x8
    ldr d21, [x6], #0x8
    ldr d19, [x5], #0x8
    ldr d16, [x4], #0x8
    zip1 v18.8h, v23.8h, v18.8h
    zip1 v17.8h, v22.8h, v17.8h
    zip1 v20.8h, v20.8h, v19.8h
    zip1 v16.8h, v21.8h, v16.8h
    zip1 v19.8h, v18.8h, v17.8h
    zip2 v18.8h, v18.8h, v17.8h
    zip1 v17.8h, v20.8h, v16.8h
    zip2 v16.8h, v20.8h, v16.8h
    str q19, [x22, #0]
    str q18, [x22, #0x10]
    str q17, [x22, #0x80]
    str q16, [x22, #0x90]
    add x22, x22, #0x20
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x23, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
    ldr h23, [x24], #0x2
    ldr h22, [x21], #0x2
    sub x23, x23, #0x1
    ldr h19, [x20], #0x2
    ldr h17, [x8], #0x2
    cmp x23, #0x1
    ldr h21, [x7], #0x2
    ldr h20, [x6], #0x2
    ldr h18, [x5], #0x2
    ldr h16, [x4], #0x2
    zip1 v19.8h, v23.8h, v19.8h
    zip1 v17.8h, v22.8h, v17.8h
    zip1 v18.8h, v21.8h, v18.8h
    zip1 v16.8h, v20.8h, v16.8h
    zip1 v17.8h, v19.8h, v17.8h
    zip1 v16.8h, v18.8h, v16.8h
    str d17, [x22, #0]
    str d16, [x22, #0x80]
    add x22, x22, #0x8
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x26, x26, #0x8
    add x0, x0, #0x100
    cmp x26, #0x8
    bge label_1
    cbz x26, label_22
KAI_ASM_LABEL(label_11)  // Main loop skip
KAI_ASM_LABEL(label_12)  // Tail row loop: Head
    mov x24, x1
    mov x4, #0x40
    mov x7, x2
    cmp x26, #0x3
    mov x22, x0
    add x21, x24, x3
    add x20, x21, x3
    csel x6, x4, XZR, GT
    add x8, x20, x3
    csel x20, x20, x27, GE
    add x1, x8, x3
    csel x8, x8, x27, GT
    csel x5, x4, XZR, GE
    cmp x26, #0x1
    csel x21, x21, x27, GT
    csel x4, x4, XZR, GT
    cmp x7, #0x20
    blt label_14
KAI_ASM_LABEL(label_13)  // Tail row loop: Unroll column loop
    ldr q18, [x24], #0x10
    ldr q25, [x21, #0]
    sub x7, x7, #0x20
    ldr q24, [x24], #0x10
    ldr q23, [x21, #0x10]
    cmp x7, #0x20
    ldr q30, [x24], #0x10
    ldr q17, [x20, #0]
    ldr q22, [x20, #0x10]
    ldr q16, [x8, #0]
    ldr q21, [x8, #0x10]
    ldr q29, [x21, #0x20]
    ldr q28, [x24], #0x10
    ldr q27, [x21, #0x30]
    zip1 v3.8h, v18.8h, v17.8h
    zip2 v2.8h, v18.8h, v17.8h
    ldr q20, [x20, #0x20]
    ldr q19, [x20, #0x30]
    zip1 v18.8h, v25.8h, v16.8h
    zip2 v1.8h, v25.8h, v16.8h
    ldr q17, [x8, #0x20]
    ldr q16, [x8, #0x30]
    zip1 v26.8h, v24.8h, v22.8h
    zip1 v25.8h, v23.8h, v21.8h
    zip2 v24.8h, v24.8h, v22.8h
    zip2 v23.8h, v23.8h, v21.8h
    add x21, x21, x4
    add x20, x20, x5
    zip1 v22.8h, v30.8h, v20.8h
    zip2 v0.8h, v30.8h, v20.8h
    add x8, x8, x6
    zip1 v21.8h, v29.8h, v17.8h
    zip2 v31.8h, v29.8h, v17.8h
    zip1 v30.8h, v28.8h, v19.8h
    zip1 v29.8h, v27.8h, v16.8h
    zip2 v28.8h, v28.8h, v19.8h
    zip2 v27.8h, v27.8h, v16.8h
    zip1 v20.8h, v3.8h, v18.8h
    zip2 v19.8h, v3.8h, v18.8h
    zip1 v18.8h, v2.8h, v1.8h
    zip2 v17.8h, v2.8h, v1.8h
    zip1 v16.8h, v26.8h, v25.8h
    zip2 v26.8h, v26.8h, v25.8h
    zip1 v25.8h, v24.8h, v23.8h
    zip2 v24.8h, v24.8h, v23.8h
    str q20, [x22, #0]
    str q19, [x22, #0x10]
    zip1 v23.8h, v22.8h, v21.8h
    zip2 v22.8h, v22.8h, v21.8h
    str q18, [x22, #0x20]
    zip1 v21.8h, v0.8h, v31.8h
    zip2 v20.8h, v0.8h, v31.8h
    str q17, [x22, #0x30]
    zip1 v19.8h, v30.8h, v29.8h
    zip2 v18.8h, v30.8h, v29.8h
    str q16, [x22, #0x40]
    zip1 v17.8h, v28.8h, v27.8h
    zip2 v16.8h, v28.8h, v27.8h
    str q26, [x22, #0x50]
    str q25, [x22, #0x60]
    str q24, [x22, #0x70]
    add x22, x22, x25
    str q23, [x22, #0]
    str q22, [x22, #0x10]
    str q21, [x22, #0x20]
    str q20, [x22, #0x30]
    str q19, [x22, #0x40]
    str q18, [x22, #0x50]
    str q17, [x22, #0x60]
    str q16, [x22, #0x70]
    add x22, x22, x25
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: Unroll column loop skip
    cmp x7, #0x10
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: Column loop
    ldr q20, [x24], #0x10
    ldr q19, [x21], #0x10
    sub x7, x7, #0x10
    ldr q18, [x20], #0x10
    ldr q17, [x8], #0x10
    cmp x7, #0x10
    ldr q24, [x24], #0x10
    ldr q25, [x21], #0x10
    ldr q23, [x20], #0x10
    ldr q16, [x8], #0x10
    zip1 v22.8h, v20.8h, v18.8h
    zip1 v21.8h, v19.8h, v17.8h
    zip2 v20.8h, v20.8h, v18.8h
    zip2 v19.8h, v19.8h, v17.8h
    zip1 v18.8h, v24.8h, v23.8h
    zip1 v17.8h, v25.8h, v16.8h
    zip2 v24.8h, v24.8h, v23.8h
    zip2 v16.8h, v25.8h, v16.8h
    zip1 v23.8h, v22.8h, v21.8h
    zip2 v22.8h, v22.8h, v21.8h
    zip1 v21.8h, v20.8h, v19.8h
    zip2 v20.8h, v20.8h, v19.8h
    zip1 v19.8h, v18.8h, v17.8h
    zip2 v18.8h, v18.8h, v17.8h
    zip1 v17.8h, v24.8h, v16.8h
    zip2 v16.8h, v24.8h, v16.8h
    str q23, [x22, #0]
    str q22, [x22, #0x10]
    str q21, [x22, #0x20]
    str q20, [x22, #0x30]
    str q19, [x22, #0x40]
    str q18, [x22, #0x50]
    str q17, [x22, #0x60]
    str q16, [x22, #0x70]
    add x22, x22, x25
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: Column loop skip
    cbz x7, label_21
    cmp x7, #0x4
    movi v16.8h, #0
    str q16, [x22, #0]
    str q16, [x22, #0x10]
    str q16, [x22, #0x20]
    str q16, [x22, #0x30]
    str q16, [x22, #0x40]
    str q16, [x22, #0x50]
    str q16, [x22, #0x60]
    str q16, [x22, #0x70]
    blt label_18
KAI_ASM_LABEL(label_17)  // Tail row loop: width 4 loop: loop
    ldr d18, [x24], #0x8
    ldr d19, [x21], #0x8
    sub x7, x7, #0x4
    ldr d17, [x20], #0x8
    ldr d16, [x8], #0x8
    cmp x7, #0x4
    zip1 v18.8h, v18.8h, v17.8h
    zip1 v16.8h, v19.8h, v16.8h
    zip1 v17.8h, v18.8h, v16.8h
    zip2 v16.8h, v18.8h, v16.8h
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    add x22, x22, #0x20
    bge label_17
KAI_ASM_LABEL(label_18)  // Tail row loop: width 4 loop: skip
    cmp x7, #0x1
    blt label_20
KAI_ASM_LABEL(label_19)  // Tail row loop: width 1 loop: loop
    ldr h19, [x24], #0x2
    ldr h18, [x21], #0x2
    sub x7, x7, #0x1
    ldr h17, [x20], #0x2
    ldr h16, [x8], #0x2
    cmp x7, #0x1
    zip1 v17.8h, v19.8h, v17.8h
    zip1 v16.8h, v18.8h, v16.8h
    zip1 v16.8h, v17.8h, v16.8h
    str d16, [x22, #0]
    add x22, x22, #0x8
    bge label_19
KAI_ASM_LABEL(label_20)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_21)  // Tail row loop: odd col skip
    sub x26, x26, #0x4
    add x0, x0, #0x80
    cmp x26, #0x1
    bge label_12
KAI_ASM_LABEL(label_22)  // Done
    add SP, SP, #0x40
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_16_2x4)

    KAI_ASM_END
