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

    KAI_ASM_CODE(a64_transpose_interleave_24_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_24_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_24_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_24_2x4_fp32bf16)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    mov x6, #0x18
    movi v15.16b, #0
    add SP, SP, #-0x60
    str q15, [SP, #0]
    str q15, [SP, #0x10]
    mov x27, SP
    mov x5, x4
    mov x26, x4
    str q15, [SP, #0x20]
    str q15, [SP, #0x30]
    add x5, x5, #0x3
    bfm x5, XZR, #0, #0x1  // bfc x5, #0, #0x2
    str q15, [SP, #0x40]
    cmp x26, #0x8
    str q15, [SP, #0x50]
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
    cmp x23, #0x18
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    ldr q2, [x24], #0x10
    ldr q16, [x21], #0x10
    sub x23, x23, #0x18
    ldr q25, [x20], #0x10
    ldr q19, [x8], #0x10
    cmp x23, #0x18
    ldr q10, [x7], #0x10
    ldr q11, [x6], #0x10
    ldr q1, [x5], #0x10
    ldr q6, [x4], #0x10
    ldr q8, [x24], #0x10
    ldr q21, [x21], #0x10
    zip1 v20.4s, v2.4s, v25.4s
    zip1 v7.4s, v16.4s, v19.4s
    ldr q31, [x20], #0x10
    ldr q27, [x8], #0x10
    zip2 v0.4s, v2.4s, v25.4s
    zip2 v28.4s, v16.4s, v19.4s
    ldr q13, [x7], #0x10
    ldr q17, [x6], #0x10
    zip1 v23.4s, v10.4s, v1.4s
    zip1 v3.4s, v11.4s, v6.4s
    ldr q26, [x5], #0x10
    ldr q18, [x4], #0x10
    zip2 v22.4s, v10.4s, v1.4s
    zip2 v6.4s, v11.4s, v6.4s
    ldr q30, [x24], #0x10
    ldr q29, [x21], #0x10
    zip1 v4.4s, v8.4s, v31.4s
    zip1 v1.4s, v21.4s, v27.4s
    ldr q24, [x20], #0x10
    ldr q14, [x8], #0x10
    zip2 v5.4s, v8.4s, v31.4s
    zip2 v27.4s, v21.4s, v27.4s
    ldr q8, [x7], #0x10
    ldr q21, [x6], #0x10
    zip1 v2.4s, v13.4s, v26.4s
    zip1 v15.4s, v17.4s, v18.4s
    ldr q31, [x5], #0x10
    ldr q19, [x4], #0x10
    zip2 v26.4s, v13.4s, v26.4s
    zip2 v13.4s, v17.4s, v18.4s
    ldr q18, [x24], #0x10
    ldr q12, [x21], #0x10
    zip1 v11.4s, v30.4s, v24.4s
    zip1 v10.4s, v29.4s, v14.4s
    ldr q25, [x20], #0x10
    ldr q9, [x8], #0x10
    zip2 v30.4s, v30.4s, v24.4s
    zip2 v14.4s, v29.4s, v14.4s
    ldr q29, [x7], #0x10
    ldr q17, [x6], #0x10
    zip1 v24.4s, v8.4s, v31.4s
    zip1 v16.4s, v21.4s, v19.4s
    zip2 v31.4s, v8.4s, v31.4s
    ldr q8, [x5], #0x10
    zip2 v19.4s, v21.4s, v19.4s
    zip1 v21.4s, v18.4s, v25.4s
    zip2 v25.4s, v18.4s, v25.4s
    zip1 v18.4s, v12.4s, v9.4s
    zip2 v12.4s, v12.4s, v9.4s
    zip1 v9.4s, v29.4s, v8.4s
    zip2 v8.4s, v29.4s, v8.4s
    zip1 v29.4s, v20.4s, v7.4s
    zip2 v7.4s, v20.4s, v7.4s
    ldr q20, [x4], #0x10
    KAI_ASM_INST(0x0ea16bbd)  // bfcvtn v29.4h, v29.4s
    KAI_ASM_INST(0x4ea168fd)  // bfcvtn2 v29.8h, v7.4s
    zip1 v7.4s, v0.4s, v28.4s
    zip2 v28.4s, v0.4s, v28.4s
    ldr q0, [x24], #0x10
    KAI_ASM_INST(0x0ea168e7)  // bfcvtn v7.4h, v7.4s
    KAI_ASM_INST(0x4ea16b87)  // bfcvtn2 v7.8h, v28.4s
    zip1 v28.4s, v17.4s, v20.4s
    zip2 v17.4s, v17.4s, v20.4s
    zip1 v20.4s, v4.4s, v1.4s
    zip2 v4.4s, v4.4s, v1.4s
    ldr q1, [x21], #0x10
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x4ea16894)  // bfcvtn2 v20.8h, v4.4s
    zip1 v4.4s, v5.4s, v27.4s
    zip2 v5.4s, v5.4s, v27.4s
    ldr q27, [x20], #0x10
    KAI_ASM_INST(0x0ea16884)  // bfcvtn v4.4h, v4.4s
    KAI_ASM_INST(0x4ea168a4)  // bfcvtn2 v4.8h, v5.4s
    zip1 v5.4s, v11.4s, v10.4s
    zip2 v10.4s, v11.4s, v10.4s
    ldr q11, [x8], #0x10
    KAI_ASM_INST(0x0ea168a5)  // bfcvtn v5.4h, v5.4s
    KAI_ASM_INST(0x4ea16945)  // bfcvtn2 v5.8h, v10.4s
    zip1 v10.4s, v0.4s, v27.4s
    zip2 v27.4s, v0.4s, v27.4s
    zip1 v0.4s, v1.4s, v11.4s
    zip2 v1.4s, v1.4s, v11.4s
    zip1 v11.4s, v30.4s, v14.4s
    zip2 v14.4s, v30.4s, v14.4s
    ldr q30, [x7], #0x10
    KAI_ASM_INST(0x0ea1696b)  // bfcvtn v11.4h, v11.4s
    KAI_ASM_INST(0x4ea169cb)  // bfcvtn2 v11.8h, v14.4s
    zip1 v14.4s, v21.4s, v18.4s
    zip2 v18.4s, v21.4s, v18.4s
    ldr q21, [x6], #0x10
    KAI_ASM_INST(0x0ea169ce)  // bfcvtn v14.4h, v14.4s
    KAI_ASM_INST(0x4ea16a4e)  // bfcvtn2 v14.8h, v18.4s
    zip1 v18.4s, v25.4s, v12.4s
    zip2 v25.4s, v25.4s, v12.4s
    ldr q12, [x5], #0x10
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x4ea16b32)  // bfcvtn2 v18.8h, v25.4s
    zip1 v25.4s, v10.4s, v0.4s
    zip2 v0.4s, v10.4s, v0.4s
    ldr q10, [x4], #0x10
    KAI_ASM_INST(0x0ea16b39)  // bfcvtn v25.4h, v25.4s
    KAI_ASM_INST(0x4ea16819)  // bfcvtn2 v25.8h, v0.4s
    zip1 v0.4s, v30.4s, v12.4s
    zip2 v30.4s, v30.4s, v12.4s
    zip1 v12.4s, v21.4s, v10.4s
    zip2 v10.4s, v21.4s, v10.4s
    zip1 v21.4s, v27.4s, v1.4s
    zip2 v27.4s, v27.4s, v1.4s
    ldr q1, [x24], #0x10
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x4ea16b75)  // bfcvtn2 v21.8h, v27.4s
    zip1 v27.4s, v23.4s, v3.4s
    zip2 v3.4s, v23.4s, v3.4s
    ldr q23, [x21], #0x10
    KAI_ASM_INST(0x0ea16b7b)  // bfcvtn v27.4h, v27.4s
    KAI_ASM_INST(0x4ea1687b)  // bfcvtn2 v27.8h, v3.4s
    zip1 v3.4s, v22.4s, v6.4s
    zip2 v22.4s, v22.4s, v6.4s
    ldr q6, [x20], #0x10
    KAI_ASM_INST(0x0ea16863)  // bfcvtn v3.4h, v3.4s
    KAI_ASM_INST(0x4ea16ac3)  // bfcvtn2 v3.8h, v22.4s
    zip1 v22.4s, v2.4s, v15.4s
    zip2 v2.4s, v2.4s, v15.4s
    ldr q15, [x8], #0x10
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x4ea16856)  // bfcvtn2 v22.8h, v2.4s
    zip1 v2.4s, v1.4s, v6.4s
    zip2 v1.4s, v1.4s, v6.4s
    zip1 v6.4s, v23.4s, v15.4s
    zip2 v15.4s, v23.4s, v15.4s
    zip1 v23.4s, v26.4s, v13.4s
    zip2 v13.4s, v26.4s, v13.4s
    ldr q26, [x7], #0x10
    KAI_ASM_INST(0x0ea16af7)  // bfcvtn v23.4h, v23.4s
    KAI_ASM_INST(0x4ea169b7)  // bfcvtn2 v23.8h, v13.4s
    zip1 v13.4s, v2.4s, v6.4s
    zip2 v6.4s, v2.4s, v6.4s
    ldr q2, [x6], #0x10
    KAI_ASM_INST(0x0ea169ad)  // bfcvtn v13.4h, v13.4s
    KAI_ASM_INST(0x4ea168cd)  // bfcvtn2 v13.8h, v6.4s
    zip1 v6.4s, v1.4s, v15.4s
    zip2 v15.4s, v1.4s, v15.4s
    ldr q1, [x5], #0x10
    KAI_ASM_INST(0x0ea168c6)  // bfcvtn v6.4h, v6.4s
    KAI_ASM_INST(0x4ea169e6)  // bfcvtn2 v6.8h, v15.4s
    ldr q15, [x4], #0x10
    str q29, [x22, #0]
    zip1 v29.4s, v24.4s, v16.4s
    zip2 v24.4s, v24.4s, v16.4s
    str q7, [x22, #0x10]
    zip1 v7.4s, v26.4s, v1.4s
    zip2 v16.4s, v26.4s, v1.4s
    str q20, [x22, #0x20]
    zip1 v1.4s, v2.4s, v15.4s
    zip2 v26.4s, v2.4s, v15.4s
    str q4, [x22, #0x30]
    zip1 v2.4s, v31.4s, v19.4s
    zip1 v20.4s, v9.4s, v28.4s
    str q5, [x22, #0x40]
    zip1 v15.4s, v8.4s, v17.4s
    zip1 v5.4s, v0.4s, v12.4s
    str q11, [x22, #0x50]
    zip1 v4.4s, v30.4s, v10.4s
    zip1 v11.4s, v7.4s, v1.4s
    str q14, [x22, #0x60]
    zip1 v14.4s, v16.4s, v26.4s
    KAI_ASM_INST(0x0ea16bbd)  // bfcvtn v29.4h, v29.4s
    str q18, [x22, #0x70]
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    zip2 v19.4s, v31.4s, v19.4s
    str q25, [x22, #0x80]
    KAI_ASM_INST(0x0ea16a92)  // bfcvtn v18.4h, v20.4s
    zip2 v20.4s, v9.4s, v28.4s
    str q21, [x22, #0x90]
    KAI_ASM_INST(0x0ea169f5)  // bfcvtn v21.4h, v15.4s
    zip2 v31.4s, v8.4s, v17.4s
    str q13, [x22, #0xa0]
    KAI_ASM_INST(0x0ea168a8)  // bfcvtn v8.4h, v5.4s
    zip2 v25.4s, v0.4s, v12.4s
    str q6, [x22, #0xb0]
    KAI_ASM_INST(0x0ea16885)  // bfcvtn v5.4h, v4.4s
    zip2 v12.4s, v30.4s, v10.4s
    str q27, [x22, #0xc0]
    KAI_ASM_INST(0x0ea16966)  // bfcvtn v6.4h, v11.4s
    zip2 v0.4s, v7.4s, v1.4s
    str q3, [x22, #0xd0]
    KAI_ASM_INST(0x0ea169ca)  // bfcvtn v10.4h, v14.4s
    zip2 v16.4s, v16.4s, v26.4s
    str q22, [x22, #0xe0]
    KAI_ASM_INST(0x4ea16b1d)  // bfcvtn2 v29.8h, v24.4s
    KAI_ASM_INST(0x4ea16a62)  // bfcvtn2 v2.8h, v19.4s
    str q23, [x22, #0xf0]
    KAI_ASM_INST(0x4ea16a92)  // bfcvtn2 v18.8h, v20.4s
    KAI_ASM_INST(0x4ea16bf5)  // bfcvtn2 v21.8h, v31.4s
    KAI_ASM_INST(0x4ea16b28)  // bfcvtn2 v8.8h, v25.4s
    KAI_ASM_INST(0x4ea16985)  // bfcvtn2 v5.8h, v12.4s
    KAI_ASM_INST(0x4ea16806)  // bfcvtn2 v6.8h, v0.4s
    KAI_ASM_INST(0x4ea16a0a)  // bfcvtn2 v10.8h, v16.4s
    str q29, [x22, #0x100]
    str q2, [x22, #0x110]
    str q18, [x22, #0x120]
    str q21, [x22, #0x130]
    str q8, [x22, #0x140]
    str q5, [x22, #0x150]
    str q6, [x22, #0x160]
    str q10, [x22, #0x170]
    add x22, x22, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop skip
    cbz x23, label_10
    cmp x23, #0x10
    movi v16.16b, #0
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
    str q16, [x22, #0x100]
    str q16, [x22, #0x110]
    str q16, [x22, #0x120]
    str q16, [x22, #0x130]
    str q16, [x22, #0x140]
    str q16, [x22, #0x150]
    str q16, [x22, #0x160]
    str q16, [x22, #0x170]
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: width 16 loop: loop
    ldr q31, [x24], #0x10
    ldr q22, [x21], #0x10
    sub x23, x23, #0x10
    ldr q21, [x20], #0x10
    ldr q16, [x8], #0x10
    cmp x23, #0x10
    ldr q19, [x7], #0x10
    ldr q29, [x6], #0x10
    ldr q5, [x5], #0x10
    ldr q18, [x4], #0x10
    ldr q17, [x24], #0x10
    ldr q20, [x21], #0x10
    zip1 v4.4s, v31.4s, v21.4s
    zip1 v15.4s, v22.4s, v16.4s
    ldr q24, [x20], #0x10
    ldr q7, [x8], #0x10
    zip2 v14.4s, v31.4s, v21.4s
    zip2 v31.4s, v22.4s, v16.4s
    ldr q11, [x7], #0x10
    ldr q3, [x6], #0x10
    zip1 v30.4s, v19.4s, v5.4s
    zip1 v16.4s, v29.4s, v18.4s
    ldr q6, [x5], #0x10
    ldr q8, [x4], #0x10
    zip2 v2.4s, v19.4s, v5.4s
    zip2 v13.4s, v29.4s, v18.4s
    ldr q10, [x24], #0x10
    ldr q23, [x21], #0x10
    zip1 v1.4s, v17.4s, v24.4s
    zip1 v25.4s, v20.4s, v7.4s
    ldr q28, [x20], #0x10
    ldr q18, [x8], #0x10
    zip2 v17.4s, v17.4s, v24.4s
    zip2 v27.4s, v20.4s, v7.4s
    ldr q0, [x7], #0x10
    ldr q29, [x6], #0x10
    zip1 v26.4s, v11.4s, v6.4s
    zip1 v22.4s, v3.4s, v8.4s
    ldr q21, [x5], #0x10
    ldr q24, [x4], #0x10
    zip2 v6.4s, v11.4s, v6.4s
    zip2 v20.4s, v3.4s, v8.4s
    ldr q12, [x24], #0x10
    ldr q8, [x21], #0x10
    zip1 v3.4s, v10.4s, v28.4s
    zip1 v5.4s, v23.4s, v18.4s
    ldr q11, [x20], #0x10
    ldr q19, [x8], #0x10
    zip2 v9.4s, v10.4s, v28.4s
    zip2 v7.4s, v23.4s, v18.4s
    ldr q18, [x7], #0x10
    ldr q10, [x6], #0x10
    zip1 v28.4s, v0.4s, v21.4s
    zip1 v23.4s, v29.4s, v24.4s
    zip2 v21.4s, v0.4s, v21.4s
    ldr q0, [x5], #0x10
    zip2 v29.4s, v29.4s, v24.4s
    zip1 v24.4s, v12.4s, v11.4s
    zip2 v11.4s, v12.4s, v11.4s
    zip1 v12.4s, v8.4s, v19.4s
    zip2 v8.4s, v8.4s, v19.4s
    zip1 v19.4s, v18.4s, v0.4s
    zip2 v18.4s, v18.4s, v0.4s
    zip1 v0.4s, v4.4s, v15.4s
    zip2 v4.4s, v4.4s, v15.4s
    ldr q15, [x4], #0x10
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x4ea16880)  // bfcvtn2 v0.8h, v4.4s
    zip1 v4.4s, v14.4s, v31.4s
    zip2 v31.4s, v14.4s, v31.4s
    zip1 v14.4s, v10.4s, v15.4s
    zip2 v10.4s, v10.4s, v15.4s
    zip1 v15.4s, v1.4s, v25.4s
    zip2 v25.4s, v1.4s, v25.4s
    str q0, [x22, #0]
    zip1 v0.4s, v17.4s, v27.4s
    zip1 v1.4s, v3.4s, v5.4s
    KAI_ASM_INST(0x0ea16884)  // bfcvtn v4.4h, v4.4s
    zip2 v17.4s, v17.4s, v27.4s
    zip1 v27.4s, v9.4s, v7.4s
    KAI_ASM_INST(0x0ea169ef)  // bfcvtn v15.4h, v15.4s
    KAI_ASM_INST(0x0ea16800)  // bfcvtn v0.4h, v0.4s
    KAI_ASM_INST(0x0ea16821)  // bfcvtn v1.4h, v1.4s
    zip2 v3.4s, v3.4s, v5.4s
    zip1 v5.4s, v24.4s, v12.4s
    KAI_ASM_INST(0x0ea16b7b)  // bfcvtn v27.4h, v27.4s
    zip2 v9.4s, v9.4s, v7.4s
    zip1 v7.4s, v11.4s, v8.4s
    zip2 v12.4s, v24.4s, v12.4s
    zip1 v24.4s, v30.4s, v16.4s
    KAI_ASM_INST(0x0ea168a5)  // bfcvtn v5.4h, v5.4s
    zip2 v11.4s, v11.4s, v8.4s
    zip1 v8.4s, v2.4s, v13.4s
    KAI_ASM_INST(0x0ea168e7)  // bfcvtn v7.4h, v7.4s
    zip2 v16.4s, v30.4s, v16.4s
    zip1 v30.4s, v26.4s, v22.4s
    KAI_ASM_INST(0x0ea16b18)  // bfcvtn v24.4h, v24.4s
    KAI_ASM_INST(0x0ea16908)  // bfcvtn v8.4h, v8.4s
    zip2 v13.4s, v2.4s, v13.4s
    zip1 v2.4s, v6.4s, v20.4s
    zip2 v22.4s, v26.4s, v22.4s
    zip1 v26.4s, v28.4s, v23.4s
    KAI_ASM_INST(0x0ea16bde)  // bfcvtn v30.4h, v30.4s
    zip2 v20.4s, v6.4s, v20.4s
    zip1 v6.4s, v21.4s, v29.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    zip2 v23.4s, v28.4s, v23.4s
    zip1 v28.4s, v19.4s, v14.4s
    KAI_ASM_INST(0x0ea16b5a)  // bfcvtn v26.4h, v26.4s
    KAI_ASM_INST(0x0ea168c6)  // bfcvtn v6.4h, v6.4s
    zip2 v21.4s, v21.4s, v29.4s
    zip1 v29.4s, v18.4s, v10.4s
    zip2 v14.4s, v19.4s, v14.4s
    KAI_ASM_INST(0x0ea16b9c)  // bfcvtn v28.4h, v28.4s
    zip2 v19.4s, v18.4s, v10.4s
    KAI_ASM_INST(0x4ea16be4)  // bfcvtn2 v4.8h, v31.4s
    KAI_ASM_INST(0x4ea16b2f)  // bfcvtn2 v15.8h, v25.4s
    KAI_ASM_INST(0x0ea16bb9)  // bfcvtn v25.4h, v29.4s
    KAI_ASM_INST(0x4ea16a20)  // bfcvtn2 v0.8h, v17.4s
    KAI_ASM_INST(0x4ea16861)  // bfcvtn2 v1.8h, v3.4s
    KAI_ASM_INST(0x4ea1693b)  // bfcvtn2 v27.8h, v9.4s
    KAI_ASM_INST(0x4ea16985)  // bfcvtn2 v5.8h, v12.4s
    KAI_ASM_INST(0x4ea16967)  // bfcvtn2 v7.8h, v11.4s
    str q4, [x22, #0x10]
    KAI_ASM_INST(0x4ea16a18)  // bfcvtn2 v24.8h, v16.4s
    KAI_ASM_INST(0x4ea169a8)  // bfcvtn2 v8.8h, v13.4s
    str q15, [x22, #0x20]
    KAI_ASM_INST(0x4ea16ade)  // bfcvtn2 v30.8h, v22.4s
    KAI_ASM_INST(0x4ea16a82)  // bfcvtn2 v2.8h, v20.4s
    str q0, [x22, #0x30]
    KAI_ASM_INST(0x4ea16afa)  // bfcvtn2 v26.8h, v23.4s
    KAI_ASM_INST(0x4ea16aa6)  // bfcvtn2 v6.8h, v21.4s
    str q1, [x22, #0x40]
    KAI_ASM_INST(0x4ea169dc)  // bfcvtn2 v28.8h, v14.4s
    KAI_ASM_INST(0x4ea16a79)  // bfcvtn2 v25.8h, v19.4s
    str q27, [x22, #0x50]
    str q5, [x22, #0x60]
    str q7, [x22, #0x70]
    str q24, [x22, #0xc0]
    str q8, [x22, #0xd0]
    str q30, [x22, #0xe0]
    str q2, [x22, #0xf0]
    str q26, [x22, #0x100]
    str q6, [x22, #0x110]
    str q28, [x22, #0x120]
    str q25, [x22, #0x130]
    add x22, x22, #0x80
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: width 16 loop: skip
    cmp x23, #0x4
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
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
    str q19, [x22, #0xc0]
    str q17, [x22, #0xd0]
    add x22, x22, #0x20
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x23, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
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
    str d16, [x22, #0xc0]
    add x22, x22, #0x8
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x26, x26, #0x8
    add x0, x0, #0x180
    cmp x26, #0x8
    bge label_1
    cbz x26, label_22
KAI_ASM_LABEL(label_11)  // Main loop skip
KAI_ASM_LABEL(label_12)  // Tail row loop: Head
    mov x24, x1
    mov x4, #0x60
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
    cmp x7, #0x18
    blt label_14
KAI_ASM_LABEL(label_13)  // Tail row loop: Column loop
    ldr q18, [x24], #0x10
    ldr q4, [x21, #0]
    sub x7, x7, #0x18
    ldr q21, [x24], #0x10
    ldr q31, [x21, #0x10]
    cmp x7, #0x18
    ldr q27, [x24], #0x10
    ldr q25, [x21, #0x20]
    ldr q9, [x24], #0x10
    ldr q3, [x21, #0x30]
    ldr q8, [x24], #0x10
    ldr q7, [x21, #0x40]
    ldr q2, [x24], #0x10
    ldr q1, [x21, #0x50]
    add x21, x21, x4
    ldr q17, [x20, #0]
    ldr q19, [x20, #0x10]
    ldr q24, [x20, #0x20]
    ldr q23, [x20, #0x30]
    ldr q30, [x20, #0x40]
    ldr q29, [x20, #0x50]
    add x20, x20, x5
    ldr q20, [x8, #0]
    ldr q16, [x8, #0x10]
    zip1 v15.4s, v18.4s, v17.4s
    zip2 v0.4s, v18.4s, v17.4s
    ldr q22, [x8, #0x20]
    ldr q18, [x8, #0x30]
    zip1 v17.4s, v21.4s, v19.4s
    zip2 v5.4s, v21.4s, v19.4s
    ldr q26, [x8, #0x40]
    ldr q19, [x8, #0x50]
    zip1 v21.4s, v27.4s, v24.4s
    zip2 v28.4s, v27.4s, v24.4s
    zip1 v6.4s, v4.4s, v20.4s
    zip2 v20.4s, v4.4s, v20.4s
    add x8, x8, x6
    zip1 v4.4s, v31.4s, v16.4s
    zip2 v16.4s, v31.4s, v16.4s
    zip1 v31.4s, v25.4s, v22.4s
    zip2 v25.4s, v25.4s, v22.4s
    zip1 v27.4s, v9.4s, v23.4s
    zip1 v22.4s, v3.4s, v18.4s
    zip2 v12.4s, v9.4s, v23.4s
    zip2 v24.4s, v3.4s, v18.4s
    zip1 v23.4s, v8.4s, v30.4s
    zip1 v3.4s, v7.4s, v26.4s
    zip2 v18.4s, v8.4s, v30.4s
    zip2 v13.4s, v7.4s, v26.4s
    zip1 v14.4s, v2.4s, v29.4s
    zip1 v9.4s, v1.4s, v19.4s
    zip2 v8.4s, v2.4s, v29.4s
    zip2 v10.4s, v1.4s, v19.4s
    zip1 v11.4s, v15.4s, v6.4s
    zip1 v7.4s, v0.4s, v20.4s
    zip1 v1.4s, v17.4s, v4.4s
    zip1 v2.4s, v5.4s, v16.4s
    zip1 v30.4s, v21.4s, v31.4s
    zip1 v29.4s, v28.4s, v25.4s
    zip1 v26.4s, v27.4s, v22.4s
    zip1 v19.4s, v12.4s, v24.4s
    KAI_ASM_INST(0x0ea1696b)  // bfcvtn v11.4h, v11.4s
    zip2 v6.4s, v15.4s, v6.4s
    zip1 v15.4s, v23.4s, v3.4s
    KAI_ASM_INST(0x0ea168e7)  // bfcvtn v7.4h, v7.4s
    zip2 v0.4s, v0.4s, v20.4s
    zip1 v20.4s, v18.4s, v13.4s
    KAI_ASM_INST(0x0ea16821)  // bfcvtn v1.4h, v1.4s
    zip2 v4.4s, v17.4s, v4.4s
    zip1 v17.4s, v14.4s, v9.4s
    KAI_ASM_INST(0x0ea16842)  // bfcvtn v2.4h, v2.4s
    zip2 v5.4s, v5.4s, v16.4s
    zip1 v16.4s, v8.4s, v10.4s
    KAI_ASM_INST(0x0ea16bde)  // bfcvtn v30.4h, v30.4s
    zip2 v31.4s, v21.4s, v31.4s
    KAI_ASM_INST(0x0ea16bb5)  // bfcvtn v21.4h, v29.4s
    zip2 v29.4s, v28.4s, v25.4s
    KAI_ASM_INST(0x0ea16b5c)  // bfcvtn v28.4h, v26.4s
    zip2 v27.4s, v27.4s, v22.4s
    KAI_ASM_INST(0x0ea16a7a)  // bfcvtn v26.4h, v19.4s
    zip2 v25.4s, v12.4s, v24.4s
    KAI_ASM_INST(0x0ea169f8)  // bfcvtn v24.4h, v15.4s
    zip2 v22.4s, v23.4s, v3.4s
    KAI_ASM_INST(0x0ea16a83)  // bfcvtn v3.4h, v20.4s
    zip2 v20.4s, v18.4s, v13.4s
    KAI_ASM_INST(0x0ea16a33)  // bfcvtn v19.4h, v17.4s
    zip2 v18.4s, v14.4s, v9.4s
    KAI_ASM_INST(0x0ea16a11)  // bfcvtn v17.4h, v16.4s
    zip2 v16.4s, v8.4s, v10.4s
    KAI_ASM_INST(0x4ea168cb)  // bfcvtn2 v11.8h, v6.4s
    KAI_ASM_INST(0x4ea16807)  // bfcvtn2 v7.8h, v0.4s
    KAI_ASM_INST(0x4ea16881)  // bfcvtn2 v1.8h, v4.4s
    KAI_ASM_INST(0x4ea168a2)  // bfcvtn2 v2.8h, v5.4s
    KAI_ASM_INST(0x4ea16bfe)  // bfcvtn2 v30.8h, v31.4s
    KAI_ASM_INST(0x4ea16bb5)  // bfcvtn2 v21.8h, v29.4s
    KAI_ASM_INST(0x4ea16b7c)  // bfcvtn2 v28.8h, v27.4s
    KAI_ASM_INST(0x4ea16b3a)  // bfcvtn2 v26.8h, v25.4s
    str q11, [x22, #0]
    KAI_ASM_INST(0x4ea16ad8)  // bfcvtn2 v24.8h, v22.4s
    KAI_ASM_INST(0x4ea16a83)  // bfcvtn2 v3.8h, v20.4s
    str q7, [x22, #0x10]
    KAI_ASM_INST(0x4ea16a53)  // bfcvtn2 v19.8h, v18.4s
    KAI_ASM_INST(0x4ea16a11)  // bfcvtn2 v17.8h, v16.4s
    str q1, [x22, #0x20]
    str q2, [x22, #0x30]
    str q30, [x22, #0x40]
    str q21, [x22, #0x50]
    str q28, [x22, #0x60]
    str q26, [x22, #0x70]
    str q24, [x22, #0x80]
    str q3, [x22, #0x90]
    str q19, [x22, #0xa0]
    str q17, [x22, #0xb0]
    add x22, x22, x25
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: Column loop skip
    cbz x7, label_21
    cmp x7, #0x10
    movi v16.16b, #0
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
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: width 16 loop: loop
    ldr q20, [x24], #0x10
    ldr q19, [x21], #0x10
    sub x7, x7, #0x10
    ldr q18, [x20], #0x10
    ldr q17, [x8], #0x10
    cmp x7, #0x10
    ldr q0, [x24], #0x10
    ldr q31, [x21], #0x10
    ldr q24, [x20], #0x10
    ldr q16, [x8], #0x10
    ldr q23, [x24], #0x10
    zip1 v30.4s, v20.4s, v18.4s
    zip1 v29.4s, v19.4s, v17.4s
    ldr q22, [x21], #0x10
    ldr q21, [x20], #0x10
    zip2 v28.4s, v20.4s, v18.4s
    zip2 v27.4s, v19.4s, v17.4s
    ldr q20, [x8], #0x10
    ldr q19, [x24], #0x10
    zip1 v26.4s, v0.4s, v24.4s
    zip1 v25.4s, v31.4s, v16.4s
    ldr q18, [x21], #0x10
    ldr q17, [x20], #0x10
    zip2 v8.4s, v0.4s, v24.4s
    zip2 v24.4s, v31.4s, v16.4s
    ldr q16, [x8], #0x10
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
    KAI_ASM_INST(0x4ea16a53)  // bfcvtn2 v19.8h, v18.4s
    KAI_ASM_INST(0x4ea16a11)  // bfcvtn2 v17.8h, v16.4s
    str q31, [x22, #0]
    str q29, [x22, #0x10]
    str q27, [x22, #0x20]
    str q25, [x22, #0x30]
    str q23, [x22, #0x40]
    str q21, [x22, #0x50]
    str q19, [x22, #0x60]
    str q17, [x22, #0x70]
    add x22, x22, #0x80
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: width 16 loop: skip
    cmp x7, #0x4
    blt label_18
KAI_ASM_LABEL(label_17)  // Tail row loop: width 4 loop: loop
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
    add x22, x22, #0x20
    bge label_17
KAI_ASM_LABEL(label_18)  // Tail row loop: width 4 loop: skip
    cmp x7, #0x1
    blt label_20
KAI_ASM_LABEL(label_19)  // Tail row loop: width 1 loop: loop
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
    bge label_19
KAI_ASM_LABEL(label_20)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_21)  // Tail row loop: odd col skip
    sub x26, x26, #0x4
    add x0, x0, #0xc0
    cmp x26, #0x1
    bge label_12
KAI_ASM_LABEL(label_22)  // Done
    add SP, SP, #0x60
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_24_2x4_fp32bf16)

    KAI_ASM_END
