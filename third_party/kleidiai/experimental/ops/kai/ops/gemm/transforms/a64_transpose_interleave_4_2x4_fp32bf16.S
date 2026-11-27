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

    KAI_ASM_CODE(a64_transpose_interleave_4_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_4_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_4_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_4_2x4_fp32bf16)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str d8, [sp, 64]
    mov x6, #0x4
    movi v16.16b, #0
    add SP, SP, #-0x20
    str q16, [SP, #0]
    str q16, [SP, #0x10]
    mov x27, SP
    mov x5, x4
    mov x26, x4
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    cmp x26, #0x8
    lsl x5, x5, #0x1
    mul x25, x5, x6
    blt label_8
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    mov x23, x2
    mov x22, x0
    add x21, x24, x3
    add x20, x21, x3
    add x8, x20, x3
    add x7, x8, x3
    cmp x23, #0x8
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ldr q19, [x24], #0x10
    ldr q18, [x21], #0x10
    sub x23, x23, #0x8
    ldr q17, [x20], #0x10
    ldr q16, [x8], #0x10
    cmp x23, #0x8
    ldr q1, [x7], #0x10
    ldr q0, [x6], #0x10
    ldr q31, [x5], #0x10
    ldr q24, [x4], #0x10
    ldr q23, [x24], #0x10
    ldr q22, [x21], #0x10
    zip1 v30.4s, v19.4s, v17.4s
    zip1 v29.4s, v18.4s, v16.4s
    ldr q21, [x20], #0x10
    ldr q20, [x8], #0x10
    zip2 v28.4s, v19.4s, v17.4s
    zip2 v27.4s, v18.4s, v16.4s
    ldr q19, [x7], #0x10
    ldr q18, [x6], #0x10
    zip1 v26.4s, v1.4s, v31.4s
    zip1 v25.4s, v0.4s, v24.4s
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    zip2 v8.4s, v1.4s, v31.4s
    zip2 v24.4s, v0.4s, v24.4s
    zip1 v7.4s, v23.4s, v21.4s
    zip1 v6.4s, v22.4s, v20.4s
    zip2 v5.4s, v23.4s, v21.4s
    zip2 v4.4s, v22.4s, v20.4s
    zip1 v3.4s, v19.4s, v17.4s
    zip1 v2.4s, v18.4s, v16.4s
    zip2 v1.4s, v19.4s, v17.4s
    zip2 v0.4s, v18.4s, v16.4s
    zip1 v23.4s, v30.4s, v29.4s
    zip1 v22.4s, v28.4s, v27.4s
    zip1 v21.4s, v26.4s, v25.4s
    zip1 v20.4s, v8.4s, v24.4s
    zip1 v19.4s, v7.4s, v6.4s
    zip1 v18.4s, v5.4s, v4.4s
    zip1 v17.4s, v3.4s, v2.4s
    zip1 v16.4s, v1.4s, v0.4s
    KAI_ASM_INST(0x0ea16aff)  // bfcvtn v31.4h, v23.4s
    zip2 v30.4s, v30.4s, v29.4s
    KAI_ASM_INST(0x0ea16add)  // bfcvtn v29.4h, v22.4s
    zip2 v28.4s, v28.4s, v27.4s
    KAI_ASM_INST(0x0ea16abb)  // bfcvtn v27.4h, v21.4s
    zip2 v26.4s, v26.4s, v25.4s
    KAI_ASM_INST(0x0ea16a99)  // bfcvtn v25.4h, v20.4s
    zip2 v24.4s, v8.4s, v24.4s
    KAI_ASM_INST(0x0ea16a77)  // bfcvtn v23.4h, v19.4s
    zip2 v22.4s, v7.4s, v6.4s
    KAI_ASM_INST(0x0ea16a55)  // bfcvtn v21.4h, v18.4s
    zip2 v20.4s, v5.4s, v4.4s
    KAI_ASM_INST(0x0ea16a33)  // bfcvtn v19.4h, v17.4s
    zip2 v18.4s, v3.4s, v2.4s
    KAI_ASM_INST(0x0ea16a11)  // bfcvtn v17.4h, v16.4s
    zip2 v16.4s, v1.4s, v0.4s
    KAI_ASM_INST(0x4ea16bdf)  // bfcvtn2 v31.8h, v30.4s
    KAI_ASM_INST(0x4ea16b9d)  // bfcvtn2 v29.8h, v28.4s
    KAI_ASM_INST(0x4ea16b5b)  // bfcvtn2 v27.8h, v26.4s
    KAI_ASM_INST(0x4ea16b19)  // bfcvtn2 v25.8h, v24.4s
    KAI_ASM_INST(0x4ea16ad7)  // bfcvtn2 v23.8h, v22.4s
    KAI_ASM_INST(0x4ea16a95)  // bfcvtn2 v21.8h, v20.4s
    str q31, [x22, #0]
    str q29, [x22, #0x10]
    KAI_ASM_INST(0x4ea16a53)  // bfcvtn2 v19.8h, v18.4s
    KAI_ASM_INST(0x4ea16a11)  // bfcvtn2 v17.8h, v16.4s
    str q27, [x22, #0x20]
    str q25, [x22, #0x30]
    add x22, x22, x25
    str q23, [x22, #0]
    str q21, [x22, #0x10]
    str q19, [x22, #0x20]
    str q17, [x22, #0x30]
    add x22, x22, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cmp x23, #0x4
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    ldr q25, [x24], #0x10
    ldr q24, [x21], #0x10
    sub x23, x23, #0x4
    ldr q21, [x20], #0x10
    ldr q20, [x8], #0x10
    cmp x23, #0x4
    ldr q23, [x7], #0x10
    ldr q19, [x6], #0x10
    ldr q18, [x5], #0x10
    ldr q17, [x4], #0x10
    zip1 v22.4s, v25.4s, v21.4s
    zip1 v16.4s, v24.4s, v20.4s
    zip2 v21.4s, v25.4s, v21.4s
    zip2 v20.4s, v24.4s, v20.4s
    zip1 v27.4s, v23.4s, v18.4s
    zip1 v26.4s, v19.4s, v17.4s
    zip2 v25.4s, v23.4s, v18.4s
    zip2 v24.4s, v19.4s, v17.4s
    zip1 v19.4s, v22.4s, v16.4s
    zip1 v18.4s, v21.4s, v20.4s
    zip1 v17.4s, v27.4s, v26.4s
    zip2 v23.4s, v22.4s, v16.4s
    zip1 v16.4s, v25.4s, v24.4s
    zip2 v22.4s, v21.4s, v20.4s
    KAI_ASM_INST(0x0ea16a75)  // bfcvtn v21.4h, v19.4s
    KAI_ASM_INST(0x0ea16a54)  // bfcvtn v20.4h, v18.4s
    KAI_ASM_INST(0x0ea16a33)  // bfcvtn v19.4h, v17.4s
    zip2 v18.4s, v27.4s, v26.4s
    KAI_ASM_INST(0x0ea16a11)  // bfcvtn v17.4h, v16.4s
    zip2 v16.4s, v25.4s, v24.4s
    KAI_ASM_INST(0x4ea16af5)  // bfcvtn2 v21.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad4)  // bfcvtn2 v20.8h, v22.4s
    KAI_ASM_INST(0x4ea16a53)  // bfcvtn2 v19.8h, v18.4s
    KAI_ASM_INST(0x4ea16a11)  // bfcvtn2 v17.8h, v16.4s
    str q21, [x22, #0]
    str q20, [x22, #0x10]
    str q19, [x22, #0x20]
    str q17, [x22, #0x30]
    add x22, x22, x25
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    cbz x23, label_7
    movi v16.16b, #0
    str q16, [x22, #0]
    str q16, [x22, #0x10]
    str q16, [x22, #0x20]
    str q16, [x22, #0x30]
KAI_ASM_LABEL(label_6)  // Main row loop: width 1 loop: loop
    ldr s23, [x24], #0x4
    ldr s22, [x21], #0x4
    sub x23, x23, #0x1
    ldr s19, [x20], #0x4
    ldr s17, [x8], #0x4
    cmp x23, #0x1
    ldr s21, [x7], #0x4
    ldr s20, [x6], #0x4
    ldr s18, [x5], #0x4
    ldr s16, [x4], #0x4
    zip1 v19.4s, v23.4s, v19.4s
    zip1 v17.4s, v22.4s, v17.4s
    zip1 v18.4s, v21.4s, v18.4s
    zip1 v16.4s, v20.4s, v16.4s
    zip1 v17.4s, v19.4s, v17.4s
    zip1 v16.4s, v18.4s, v16.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d17, [x22, #0]
    str d16, [x22, #0x20]
    add x22, x22, #0x8
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: odd col skip
    sub x26, x26, #0x8
    add x0, x0, #0x40
    cmp x26, #0x8
    bge label_1
    cbz x26, label_16
KAI_ASM_LABEL(label_8)  // Main loop skip
KAI_ASM_LABEL(label_9)  // Tail row loop: Head
    mov x24, x1
    mov x4, #0x20
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
    cmp x7, #0x8
    blt label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Unroll column loop
    ldr q25, [x24], #0x10
    ldr q24, [x21, #0]
    sub x7, x7, #0x8
    ldr q21, [x20, #0]
    ldr q20, [x8, #0]
    cmp x7, #0x8
    ldr q23, [x24], #0x10
    ldr q19, [x21, #0x10]
    add x21, x21, x4
    ldr q18, [x20, #0x10]
    ldr q17, [x8, #0x10]
    add x20, x20, x5
    add x8, x8, x6
    zip1 v22.4s, v25.4s, v21.4s
    zip1 v16.4s, v24.4s, v20.4s
    zip2 v21.4s, v25.4s, v21.4s
    zip2 v20.4s, v24.4s, v20.4s
    zip1 v27.4s, v23.4s, v18.4s
    zip1 v26.4s, v19.4s, v17.4s
    zip2 v25.4s, v23.4s, v18.4s
    zip2 v24.4s, v19.4s, v17.4s
    zip1 v19.4s, v22.4s, v16.4s
    zip1 v18.4s, v21.4s, v20.4s
    zip1 v17.4s, v27.4s, v26.4s
    zip2 v23.4s, v22.4s, v16.4s
    zip1 v16.4s, v25.4s, v24.4s
    zip2 v22.4s, v21.4s, v20.4s
    KAI_ASM_INST(0x0ea16a75)  // bfcvtn v21.4h, v19.4s
    KAI_ASM_INST(0x0ea16a54)  // bfcvtn v20.4h, v18.4s
    KAI_ASM_INST(0x0ea16a33)  // bfcvtn v19.4h, v17.4s
    zip2 v18.4s, v27.4s, v26.4s
    KAI_ASM_INST(0x0ea16a11)  // bfcvtn v17.4h, v16.4s
    zip2 v16.4s, v25.4s, v24.4s
    KAI_ASM_INST(0x4ea16af5)  // bfcvtn2 v21.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad4)  // bfcvtn2 v20.8h, v22.4s
    KAI_ASM_INST(0x4ea16a53)  // bfcvtn2 v19.8h, v18.4s
    KAI_ASM_INST(0x4ea16a11)  // bfcvtn2 v17.8h, v16.4s
    str q21, [x22, #0]
    str q20, [x22, #0x10]
    add x22, x22, x25
    str q19, [x22, #0]
    str q17, [x22, #0x10]
    add x22, x22, x25
    bge label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Unroll column loop skip
    cmp x7, #0x4
    blt label_13
KAI_ASM_LABEL(label_12)  // Tail row loop: Column loop
    ldr q21, [x24], #0x10
    ldr q20, [x21], #0x10
    sub x7, x7, #0x4
    ldr q19, [x20], #0x10
    ldr q17, [x8], #0x10
    cmp x7, #0x4
    zip1 v18.4s, v21.4s, v19.4s
    zip1 v16.4s, v20.4s, v17.4s
    zip2 v21.4s, v21.4s, v19.4s
    zip2 v20.4s, v20.4s, v17.4s
    zip1 v17.4s, v18.4s, v16.4s
    zip2 v19.4s, v18.4s, v16.4s
    zip1 v16.4s, v21.4s, v20.4s
    KAI_ASM_INST(0x0ea16a32)  // bfcvtn v18.4h, v17.4s
    zip2 v17.4s, v21.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a72)  // bfcvtn2 v18.8h, v19.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    str q18, [x22, #0]
    str q16, [x22, #0x10]
    add x22, x22, x25
    bge label_12
KAI_ASM_LABEL(label_13)  // Tail row loop: Column loop skip
    cbz x7, label_15
    movi v16.16b, #0
    str q16, [x22, #0]
    str q16, [x22, #0x10]
KAI_ASM_LABEL(label_14)  // Tail row loop: width 1 loop: loop
    ldr s19, [x24], #0x4
    ldr s18, [x21], #0x4
    sub x7, x7, #0x1
    ldr s17, [x20], #0x4
    ldr s16, [x8], #0x4
    cmp x7, #0x1
    zip1 v17.4s, v19.4s, v17.4s
    zip1 v16.4s, v18.4s, v16.4s
    zip1 v16.4s, v17.4s, v16.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    add x22, x22, #0x8
    bge label_14
KAI_ASM_LABEL(label_15)  // Tail row loop: odd col skip
    sub x26, x26, #0x4
    add x0, x0, #0x20
    cmp x26, #0x1
    bge label_9
KAI_ASM_LABEL(label_16)  // Done
    add SP, SP, #0x20
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr d8, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_4_2x4_fp32bf16)

    KAI_ASM_END
