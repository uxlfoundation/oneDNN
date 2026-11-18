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

    KAI_ASM_CODE(a64_ffinterleaved_bf16fp32_mmla_8x12)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_ffinterleaved_bf16fp32_mmla_8x12)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_ffinterleaved_bf16fp32_mmla_8x12)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_ffinterleaved_bf16fp32_mmla_8x12)
    stp x20, x21, [sp, -80]!
    stp d8, d9, [sp, 16]
    stp d10, d11, [sp, 32]
    stp d12, d13, [sp, 48]
    stp d14, d15, [sp, 64]
    ldr x21, [x0, #0]
    ldr x20, [x0, #0x20]
    ldr x8, [x0, #0x50]
    cbz x21, label_7
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x1, [x0, #0x30]
    ldr x7, [x0, #0x8]
    mov x6, x20
    mov x5, x1
KAI_ASM_LABEL(label_2)  // Width loop
    mov x4, x5
    ldr x1, [x0, #0x38]
    cmp x7, #0x8
    mov x20, x6
    add x3, x4, x1
    add x2, x3, x1
    add x5, x2, x1
    bgt label_3
    cmp x7, #0x4
    mov x2, x4
    bgt label_3
    mov x3, x4
KAI_ASM_LABEL(label_3)  // B setup done
    ldr q4, [x4, #0]
    ldr q0, [x20, #0]
    movi v8.16b, #0
    ldr q1, [x20, #0x10]
    ldr q5, [x4, #0x10]
    movi v9.16b, #0
    ldr q2, [x20, #0x20]
    ldr x1, [x0, #0x10]
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    add x4, x4, #0x20
    movi v13.16b, #0
    movi v14.16b, #0
    add x20, x20, #0x30
    add x1, x1, #0x3
    movi v15.16b, #0
    movi v16.16b, #0
    lsr x1, x1, #0x2
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
    sub x1, x1, #0x1
    movi v23.16b, #0
    movi v24.16b, #0
    cmp x1, #0x2
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0
    blt label_5
KAI_ASM_LABEL(label_4)  // main loop head
    ldr q3, [x20, #0]
    ldr q6, [x3, #0]
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q7, [x3, #0x10]
    KAI_ASM_INST(0x6e45ec0b)  // bfmmla v11.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e44ec2e)  // bfmmla v14.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec31)  // bfmmla v17.4s, v1.8h, v5.8h
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    sub x1, x1, #0x2
    KAI_ASM_INST(0x6e45ec57)  // bfmmla v23.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec7a)  // bfmmla v26.4s, v3.8h, v4.8h
    ldr q4, [x2, #0]
    KAI_ASM_INST(0x6e45ec7d)  // bfmmla v29.4s, v3.8h, v5.8h
    ldr q5, [x2, #0x10]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    cmp x1, #0x2
    KAI_ASM_INST(0x6e47ec32)  // bfmmla v18.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec58)  // bfmmla v24.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec7b)  // bfmmla v27.4s, v3.8h, v6.8h
    ldr q6, [x4, #0]
    KAI_ASM_INST(0x6e47ec7e)  // bfmmla v30.4s, v3.8h, v7.8h
    ldr q7, [x4, #0x10]
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0d)  // bfmmla v13.4s, v0.8h, v5.8h
    ldr q0, [x20, #0x10]
    KAI_ASM_INST(0x6e44ec30)  // bfmmla v16.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec33)  // bfmmla v19.4s, v1.8h, v5.8h
    ldr q1, [x20, #0x20]
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec59)  // bfmmla v25.4s, v2.8h, v5.8h
    ldr q2, [x20, #0x30]
    KAI_ASM_INST(0x6e44ec7c)  // bfmmla v28.4s, v3.8h, v4.8h
    ldr q4, [x3, #0x20]
    KAI_ASM_INST(0x6e45ec7f)  // bfmmla v31.4s, v3.8h, v5.8h
    ldr q3, [x20, #0x40]
    ldr q5, [x3, #0x30]
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e47ec31)  // bfmmla v17.4s, v1.8h, v7.8h
    add x3, x3, #0x40
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec7a)  // bfmmla v26.4s, v3.8h, v6.8h
    ldr q6, [x2, #0x20]
    KAI_ASM_INST(0x6e47ec7d)  // bfmmla v29.4s, v3.8h, v7.8h
    ldr q7, [x2, #0x30]
    KAI_ASM_INST(0x6e44ec09)  // bfmmla v9.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0c)  // bfmmla v12.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e44ec2f)  // bfmmla v15.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec32)  // bfmmla v18.4s, v1.8h, v5.8h
    add x2, x2, #0x40
    KAI_ASM_INST(0x6e44ec55)  // bfmmla v21.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec58)  // bfmmla v24.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec7b)  // bfmmla v27.4s, v3.8h, v4.8h
    ldr q4, [x4, #0x20]
    KAI_ASM_INST(0x6e45ec7e)  // bfmmla v30.4s, v3.8h, v5.8h
    ldr q5, [x4, #0x30]
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    ldr q0, [x20, #0x50]
    KAI_ASM_INST(0x6e46ec30)  // bfmmla v16.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e47ec33)  // bfmmla v19.4s, v1.8h, v7.8h
    ldr q1, [x20, #0x60]
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec59)  // bfmmla v25.4s, v2.8h, v7.8h
    ldr q2, [x20, #0x70]
    KAI_ASM_INST(0x6e46ec7c)  // bfmmla v28.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e47ec7f)  // bfmmla v31.4s, v3.8h, v7.8h
    add x20, x20, #0x80
    add x4, x4, #0x40
    bge label_4
KAI_ASM_LABEL(label_5)  // main loop skip
    ldr q3, [x20, #0]
    ldr q6, [x3, #0]
    KAI_ASM_INST(0x6e44ec08)  // bfmmla v8.4s, v0.8h, v4.8h
    ldr q7, [x3, #0x10]
    KAI_ASM_INST(0x6e45ec0b)  // bfmmla v11.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e44ec2e)  // bfmmla v14.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec31)  // bfmmla v17.4s, v1.8h, v5.8h
    KAI_ASM_INST(0x6e44ec54)  // bfmmla v20.4s, v2.8h, v4.8h
    add x20, x20, #0x10
    KAI_ASM_INST(0x6e45ec57)  // bfmmla v23.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec7a)  // bfmmla v26.4s, v3.8h, v4.8h
    ldr q4, [x2, #0]
    KAI_ASM_INST(0x6e45ec7d)  // bfmmla v29.4s, v3.8h, v5.8h
    ldr q5, [x2, #0x10]
    KAI_ASM_INST(0x6e46ec09)  // bfmmla v9.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0c)  // bfmmla v12.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2f)  // bfmmla v15.4s, v1.8h, v6.8h
    add x3, x3, #0x20
    KAI_ASM_INST(0x6e47ec32)  // bfmmla v18.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec55)  // bfmmla v21.4s, v2.8h, v6.8h
    add x2, x2, #0x20
    KAI_ASM_INST(0x6e47ec58)  // bfmmla v24.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec7b)  // bfmmla v27.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e47ec7e)  // bfmmla v30.4s, v3.8h, v7.8h
    KAI_ASM_INST(0x6e44ec0a)  // bfmmla v10.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0d)  // bfmmla v13.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e44ec30)  // bfmmla v16.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec33)  // bfmmla v19.4s, v1.8h, v5.8h
    KAI_ASM_INST(0x6e44ec56)  // bfmmla v22.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec59)  // bfmmla v25.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec7c)  // bfmmla v28.4s, v3.8h, v4.8h
    KAI_ASM_INST(0x6e45ec7f)  // bfmmla v31.4s, v3.8h, v5.8h
    cbz x1, label_6
    ldr q6, [x4, #0]
    ldr q0, [x20, #0]
    ldr q1, [x20, #0x10]
    ldr q7, [x4, #0x10]
    ldr q2, [x20, #0x20]
    ldr q3, [x20, #0x30]
    add x20, x20, #0x40
    ldr q4, [x3, #0]
    ldr q5, [x3, #0x10]
    KAI_ASM_INST(0x6e46ec08)  // bfmmla v8.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0b)  // bfmmla v11.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec2e)  // bfmmla v14.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e47ec31)  // bfmmla v17.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec54)  // bfmmla v20.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec57)  // bfmmla v23.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec7a)  // bfmmla v26.4s, v3.8h, v6.8h
    ldr q6, [x2, #0]
    KAI_ASM_INST(0x6e47ec7d)  // bfmmla v29.4s, v3.8h, v7.8h
    ldr q7, [x2, #0x10]
    KAI_ASM_INST(0x6e44ec09)  // bfmmla v9.4s, v0.8h, v4.8h
    KAI_ASM_INST(0x6e45ec0c)  // bfmmla v12.4s, v0.8h, v5.8h
    KAI_ASM_INST(0x6e44ec2f)  // bfmmla v15.4s, v1.8h, v4.8h
    KAI_ASM_INST(0x6e45ec32)  // bfmmla v18.4s, v1.8h, v5.8h
    KAI_ASM_INST(0x6e44ec55)  // bfmmla v21.4s, v2.8h, v4.8h
    KAI_ASM_INST(0x6e45ec58)  // bfmmla v24.4s, v2.8h, v5.8h
    KAI_ASM_INST(0x6e44ec7b)  // bfmmla v27.4s, v3.8h, v4.8h
    KAI_ASM_INST(0x6e45ec7e)  // bfmmla v30.4s, v3.8h, v5.8h
    KAI_ASM_INST(0x6e46ec0a)  // bfmmla v10.4s, v0.8h, v6.8h
    KAI_ASM_INST(0x6e47ec0d)  // bfmmla v13.4s, v0.8h, v7.8h
    KAI_ASM_INST(0x6e46ec30)  // bfmmla v16.4s, v1.8h, v6.8h
    KAI_ASM_INST(0x6e47ec33)  // bfmmla v19.4s, v1.8h, v7.8h
    KAI_ASM_INST(0x6e46ec56)  // bfmmla v22.4s, v2.8h, v6.8h
    KAI_ASM_INST(0x6e47ec59)  // bfmmla v25.4s, v2.8h, v7.8h
    KAI_ASM_INST(0x6e46ec7c)  // bfmmla v28.4s, v3.8h, v6.8h
    KAI_ASM_INST(0x6e47ec7f)  // bfmmla v31.4s, v3.8h, v7.8h
KAI_ASM_LABEL(label_6)  // multiply loop done
    subs x7, x7, #0xc
    uzp1 v4.2d, v8.2d, v11.2d
    uzp2 v8.2d, v8.2d, v11.2d
    uzp1 v11.2d, v9.2d, v12.2d
    uzp2 v9.2d, v9.2d, v12.2d
    uzp1 v12.2d, v10.2d, v13.2d
    uzp2 v10.2d, v10.2d, v13.2d
    str q4, [x8, #0]
    uzp1 v13.2d, v14.2d, v17.2d
    uzp2 v14.2d, v14.2d, v17.2d
    str q11, [x8, #0x10]
    uzp1 v17.2d, v15.2d, v18.2d
    uzp2 v15.2d, v15.2d, v18.2d
    str q12, [x8, #0x20]
    uzp1 v18.2d, v16.2d, v19.2d
    uzp2 v16.2d, v16.2d, v19.2d
    str q8, [x8, #0x30]
    uzp1 v19.2d, v20.2d, v23.2d
    uzp2 v20.2d, v20.2d, v23.2d
    str q9, [x8, #0x40]
    uzp1 v23.2d, v21.2d, v24.2d
    uzp2 v21.2d, v21.2d, v24.2d
    str q10, [x8, #0x50]
    uzp1 v24.2d, v22.2d, v25.2d
    uzp2 v22.2d, v22.2d, v25.2d
    str q13, [x8, #0x60]
    uzp1 v25.2d, v26.2d, v29.2d
    uzp2 v26.2d, v26.2d, v29.2d
    str q17, [x8, #0x70]
    uzp1 v29.2d, v27.2d, v30.2d
    uzp2 v27.2d, v27.2d, v30.2d
    str q18, [x8, #0x80]
    uzp1 v30.2d, v28.2d, v31.2d
    uzp2 v28.2d, v28.2d, v31.2d
    str q14, [x8, #0x90]
    str q15, [x8, #0xa0]
    str q16, [x8, #0xb0]
    str q19, [x8, #0xc0]
    str q23, [x8, #0xd0]
    str q24, [x8, #0xe0]
    str q20, [x8, #0xf0]
    str q21, [x8, #0x100]
    str q22, [x8, #0x110]
    str q25, [x8, #0x120]
    str q29, [x8, #0x130]
    str q30, [x8, #0x140]
    str q26, [x8, #0x150]
    str q27, [x8, #0x160]
    str q28, [x8, #0x170]
    add x8, x8, #0x180
    bgt label_2
    subs x21, x21, #0x8
    bgt label_1
KAI_ASM_LABEL(label_7)  // Exit
    ldp d8, d9, [sp, 16]
    ldp d10, d11, [sp, 32]
    ldp d12, d13, [sp, 48]
    ldp d14, d15, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_ffinterleaved_bf16fp32_mmla_8x12)

    KAI_ASM_END
