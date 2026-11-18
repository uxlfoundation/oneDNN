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

    KAI_ASM_CODE(a64_interleaved_bf16fp32_dot_8x12)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_bf16fp32_dot_8x12)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_bf16fp32_dot_8x12)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_bf16fp32_dot_8x12)
    stp d8, d9, [sp, -64]!
    stp d10, d11, [sp, 16]
    stp d12, d13, [sp, 32]
    stp d14, d15, [sp, 48]
    ldr x7, [x0, #0]
    ldr x6, [x0, #0x20]
    ldr x5, [x0, #0x50]
    cbz x7, label_6
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x4, [x0, #0x8]
    ldr x3, [x0, #0x30]
    mov x2, x6
KAI_ASM_LABEL(label_2)  // Width loop
    ldr q4, [x3, #0]
    ldr q5, [x3, #0x10]
    mov x6, x2
    ldr q6, [x3, #0x20]
    ldr x1, [x0, #0x10]
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    ldr q0, [x6, #0]
    ldr q1, [x6, #0x10]
    movi v11.16b, #0
    add x1, x1, #0x1
    movi v12.16b, #0
    movi v13.16b, #0
    lsr x1, x1, #0x1
    movi v14.16b, #0
    movi v15.16b, #0
    prfm pldl1keep, [x6, #0]
    movi v16.16b, #0
    movi v17.16b, #0
    prfm pldl1keep, [x3, #0]
    movi v18.16b, #0
    movi v19.16b, #0
    sub x1, x1, #0x1
    prfm pldl1keep, [x3, #0x40]
    movi v20.16b, #0
    cmp x1, #0x2
    movi v21.16b, #0
    movi v22.16b, #0
    movi v23.16b, #0
    movi v24.16b, #0
    prfm pldl1keep, [x6, #0x40]
    movi v25.16b, #0
    movi v26.16b, #0
    prfm pldl1keep, [x3, #0x80]
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    ldr q2, [x6, #0x20]
    ldr q3, [x6, #0x30]
    KAI_ASM_INST(0x4f40f088)  // bfdot v8.4s, v4.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f08b)  // bfdot v11.4s, v4.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f88e)  // bfdot v14.4s, v4.8h, v0.h[2]
    sub x1, x1, #0x2
    KAI_ASM_INST(0x4f60f891)  // bfdot v17.4s, v4.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f094)  // bfdot v20.4s, v4.8h, v1.h[0]
    cmp x1, #0x2
    KAI_ASM_INST(0x4f61f097)  // bfdot v23.4s, v4.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f89a)  // bfdot v26.4s, v4.8h, v1.h[2]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f61f89d)  // bfdot v29.4s, v4.8h, v1.h[3]
    ldr q4, [x3, #0x30]
    KAI_ASM_INST(0x4f40f0a9)  // bfdot v9.4s, v5.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0ac)  // bfdot v12.4s, v5.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8af)  // bfdot v15.4s, v5.8h, v0.h[2]
    add x6, x6, #0x40
    KAI_ASM_INST(0x4f60f8b2)  // bfdot v18.4s, v5.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f0b5)  // bfdot v21.4s, v5.8h, v1.h[0]
    prfm pldl1keep, [x3, #0x100]
    KAI_ASM_INST(0x4f61f0b8)  // bfdot v24.4s, v5.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8bb)  // bfdot v27.4s, v5.8h, v1.h[2]
    prfm pldl1keep, [x3, #0x140]
    KAI_ASM_INST(0x4f61f8be)  // bfdot v30.4s, v5.8h, v1.h[3]
    ldr q5, [x3, #0x40]
    KAI_ASM_INST(0x4f40f0ca)  // bfdot v10.4s, v6.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0cd)  // bfdot v13.4s, v6.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8d0)  // bfdot v16.4s, v6.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f8d3)  // bfdot v19.4s, v6.8h, v0.h[3]
    ldr q0, [x6, #0]
    KAI_ASM_INST(0x4f41f0d6)  // bfdot v22.4s, v6.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f0d9)  // bfdot v25.4s, v6.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8dc)  // bfdot v28.4s, v6.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f8df)  // bfdot v31.4s, v6.8h, v1.h[3]
    ldr q6, [x3, #0x50]
    ldr q1, [x6, #0x10]
    add x3, x3, #0x60
    KAI_ASM_INST(0x4f42f088)  // bfdot v8.4s, v4.8h, v2.h[0]
    KAI_ASM_INST(0x4f62f08b)  // bfdot v11.4s, v4.8h, v2.h[1]
    KAI_ASM_INST(0x4f42f88e)  // bfdot v14.4s, v4.8h, v2.h[2]
    KAI_ASM_INST(0x4f62f891)  // bfdot v17.4s, v4.8h, v2.h[3]
    KAI_ASM_INST(0x4f43f094)  // bfdot v20.4s, v4.8h, v3.h[0]
    KAI_ASM_INST(0x4f63f097)  // bfdot v23.4s, v4.8h, v3.h[1]
    KAI_ASM_INST(0x4f43f89a)  // bfdot v26.4s, v4.8h, v3.h[2]
    KAI_ASM_INST(0x4f63f89d)  // bfdot v29.4s, v4.8h, v3.h[3]
    ldr q4, [x3, #0]
    KAI_ASM_INST(0x4f42f0a9)  // bfdot v9.4s, v5.8h, v2.h[0]
    KAI_ASM_INST(0x4f62f0ac)  // bfdot v12.4s, v5.8h, v2.h[1]
    KAI_ASM_INST(0x4f42f8af)  // bfdot v15.4s, v5.8h, v2.h[2]
    KAI_ASM_INST(0x4f62f8b2)  // bfdot v18.4s, v5.8h, v2.h[3]
    KAI_ASM_INST(0x4f43f0b5)  // bfdot v21.4s, v5.8h, v3.h[0]
    KAI_ASM_INST(0x4f63f0b8)  // bfdot v24.4s, v5.8h, v3.h[1]
    KAI_ASM_INST(0x4f43f8bb)  // bfdot v27.4s, v5.8h, v3.h[2]
    KAI_ASM_INST(0x4f63f8be)  // bfdot v30.4s, v5.8h, v3.h[3]
    ldr q5, [x3, #0x10]
    KAI_ASM_INST(0x4f42f0ca)  // bfdot v10.4s, v6.8h, v2.h[0]
    KAI_ASM_INST(0x4f62f0cd)  // bfdot v13.4s, v6.8h, v2.h[1]
    KAI_ASM_INST(0x4f42f8d0)  // bfdot v16.4s, v6.8h, v2.h[2]
    KAI_ASM_INST(0x4f62f8d3)  // bfdot v19.4s, v6.8h, v2.h[3]
    KAI_ASM_INST(0x4f43f0d6)  // bfdot v22.4s, v6.8h, v3.h[0]
    KAI_ASM_INST(0x4f63f0d9)  // bfdot v25.4s, v6.8h, v3.h[1]
    KAI_ASM_INST(0x4f43f8dc)  // bfdot v28.4s, v6.8h, v3.h[2]
    KAI_ASM_INST(0x4f63f8df)  // bfdot v31.4s, v6.8h, v3.h[3]
    ldr q6, [x3, #0x20]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x20
    add x3, x3, #0x30
    KAI_ASM_INST(0x4f40f088)  // bfdot v8.4s, v4.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f08b)  // bfdot v11.4s, v4.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f88e)  // bfdot v14.4s, v4.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f891)  // bfdot v17.4s, v4.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f094)  // bfdot v20.4s, v4.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f097)  // bfdot v23.4s, v4.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f89a)  // bfdot v26.4s, v4.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f89d)  // bfdot v29.4s, v4.8h, v1.h[3]
    KAI_ASM_INST(0x4f40f0a9)  // bfdot v9.4s, v5.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0ac)  // bfdot v12.4s, v5.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8af)  // bfdot v15.4s, v5.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f8b2)  // bfdot v18.4s, v5.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f0b5)  // bfdot v21.4s, v5.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f0b8)  // bfdot v24.4s, v5.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8bb)  // bfdot v27.4s, v5.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f8be)  // bfdot v30.4s, v5.8h, v1.h[3]
    KAI_ASM_INST(0x4f40f0ca)  // bfdot v10.4s, v6.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0cd)  // bfdot v13.4s, v6.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8d0)  // bfdot v16.4s, v6.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f8d3)  // bfdot v19.4s, v6.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f0d6)  // bfdot v22.4s, v6.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f0d9)  // bfdot v25.4s, v6.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8dc)  // bfdot v28.4s, v6.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f8df)  // bfdot v31.4s, v6.8h, v1.h[3]
    cbz x1, label_5
    ldr q0, [x6, #0]
    ldr q1, [x6, #0x10]
    add x6, x6, #0x20
    ldr q7, [x3, #0]
    ldr q4, [x3, #0x10]
    ldr q5, [x3, #0x20]
    add x3, x3, #0x30
    KAI_ASM_INST(0x4f40f0e8)  // bfdot v8.4s, v7.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0eb)  // bfdot v11.4s, v7.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8ee)  // bfdot v14.4s, v7.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f8f1)  // bfdot v17.4s, v7.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f0f4)  // bfdot v20.4s, v7.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f0f7)  // bfdot v23.4s, v7.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8fa)  // bfdot v26.4s, v7.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f8fd)  // bfdot v29.4s, v7.8h, v1.h[3]
    KAI_ASM_INST(0x4f40f089)  // bfdot v9.4s, v4.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f08c)  // bfdot v12.4s, v4.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f88f)  // bfdot v15.4s, v4.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f892)  // bfdot v18.4s, v4.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f095)  // bfdot v21.4s, v4.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f098)  // bfdot v24.4s, v4.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f89b)  // bfdot v27.4s, v4.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f89e)  // bfdot v30.4s, v4.8h, v1.h[3]
    KAI_ASM_INST(0x4f40f0aa)  // bfdot v10.4s, v5.8h, v0.h[0]
    KAI_ASM_INST(0x4f60f0ad)  // bfdot v13.4s, v5.8h, v0.h[1]
    KAI_ASM_INST(0x4f40f8b0)  // bfdot v16.4s, v5.8h, v0.h[2]
    KAI_ASM_INST(0x4f60f8b3)  // bfdot v19.4s, v5.8h, v0.h[3]
    KAI_ASM_INST(0x4f41f0b6)  // bfdot v22.4s, v5.8h, v1.h[0]
    KAI_ASM_INST(0x4f61f0b9)  // bfdot v25.4s, v5.8h, v1.h[1]
    KAI_ASM_INST(0x4f41f8bc)  // bfdot v28.4s, v5.8h, v1.h[2]
    KAI_ASM_INST(0x4f61f8bf)  // bfdot v31.4s, v5.8h, v1.h[3]
KAI_ASM_LABEL(label_5)  // multiply loop done
    subs x4, x4, #0xc
    str q8, [x5, #0]
    str q9, [x5, #0x10]
    str q10, [x5, #0x20]
    str q11, [x5, #0x30]
    str q12, [x5, #0x40]
    str q13, [x5, #0x50]
    str q14, [x5, #0x60]
    str q15, [x5, #0x70]
    str q16, [x5, #0x80]
    str q17, [x5, #0x90]
    str q18, [x5, #0xa0]
    str q19, [x5, #0xb0]
    str q20, [x5, #0xc0]
    str q21, [x5, #0xd0]
    str q22, [x5, #0xe0]
    str q23, [x5, #0xf0]
    str q24, [x5, #0x100]
    str q25, [x5, #0x110]
    str q26, [x5, #0x120]
    str q27, [x5, #0x130]
    str q28, [x5, #0x140]
    str q29, [x5, #0x150]
    str q30, [x5, #0x160]
    str q31, [x5, #0x170]
    add x5, x5, #0x180
    bgt label_2
    subs x7, x7, #0x8
    bgt label_1
KAI_ASM_LABEL(label_6)  // Exit
    ldp d10, d11, [sp, 16]
    ldp d12, d13, [sp, 32]
    ldp d14, d15, [sp, 48]
    ldp d8, d9, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_bf16fp32_dot_8x12)

    KAI_ASM_END
