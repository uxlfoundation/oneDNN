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

    KAI_ASM_CODE(a64_interleaved_s8s32_dot_8x12_a55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_s8s32_dot_8x12_a55)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_s8s32_dot_8x12_a55)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_s8s32_dot_8x12_a55)
    stp x20, x21, [sp, -96]!
    stp x22, x23, [sp, 16]
    stp d8, d9, [sp, 32]
    stp d10, d11, [sp, 48]
    stp d12, d13, [sp, 64]
    stp d14, d15, [sp, 80]
    ldr x23, [x0, #0]
    ldr x22, [x0, #0x20]
    ldr x21, [x0, #0x50]
    cbz x23, label_6
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x20, [x0, #0x8]
    mov x8, x22
    ldr x7, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // Width loop
    ldr q4, [x7, #0]
    mov x22, x8
    ldr q5, [x7, #0x10]
    movi v8.4s, #0
    ldr q6, [x7, #0x20]
    movi v9.4s, #0
    ldr q0, [x22, #0]
    movi v10.4s, #0
    ldr q1, [x22, #0x10]
    movi v11.4s, #0
    ldr x6, [x0, #0x10]
    movi v12.4s, #0
    prfm pldl1keep, [x22, #0]
    movi v13.4s, #0
    prfm pldl1keep, [x7, #0]
    movi v14.4s, #0
    add x6, x6, #0x3
    prfm pldl1keep, [x7, #0x40]
    lsr x6, x6, #0x2
    prfm pldl1keep, [x22, #0x40]
    movi v15.4s, #0
    prfm pldl1keep, [x7, #0x80]
    movi v16.4s, #0
    sub x6, x6, #0x1
    movi v17.4s, #0
    cmp x6, #0x2
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    ldr d2, [x22, #0x20]
    KAI_ASM_INST(0x4f80e088)  // sdot v8.4s, v4.16b, v0.4b[0]
    ldr d3, [x22, #0x30]
    KAI_ASM_INST(0x4fa0e08b)  // sdot v11.4s, v4.16b, v0.4b[1]
    ldr x5, [x22, #0x28]
    KAI_ASM_INST(0x4f80e88e)  // sdot v14.4s, v4.16b, v0.4b[2]
    ldr x4, [x22, #0x38]
    KAI_ASM_INST(0x4fa0e891)  // sdot v17.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    ldr x1, [x7, #0x38]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    ldr x3, [x7, #0x48]
    KAI_ASM_INST(0x4f81e89a)  // sdot v26.4s, v4.16b, v1.4b[2]
    ldr x2, [x7, #0x58]
    KAI_ASM_INST(0x4fa1e89d)  // sdot v29.4s, v4.16b, v1.4b[3]
    ldr d4, [x7, #0x30]
    KAI_ASM_INST(0x4f80e0a9)  // sdot v9.4s, v5.16b, v0.4b[0]
    mov v2.d[1], x5
    KAI_ASM_INST(0x4fa0e0ac)  // sdot v12.4s, v5.16b, v0.4b[1]
    mov v3.d[1], x4
    KAI_ASM_INST(0x4f80e8af)  // sdot v15.4s, v5.16b, v0.4b[2]
    mov v4.d[1], x1
    KAI_ASM_INST(0x4fa0e8b2)  // sdot v18.4s, v5.16b, v0.4b[3]
    prfm pldl1keep, [x22, #0x80]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    add x22, x22, #0x40
    KAI_ASM_INST(0x4fa1e0b8)  // sdot v24.4s, v5.16b, v1.4b[1]
    prfm pldl1keep, [x7, #0x100]
    KAI_ASM_INST(0x4f81e8bb)  // sdot v27.4s, v5.16b, v1.4b[2]
    prfm pldl1keep, [x7, #0x140]
    KAI_ASM_INST(0x4fa1e8be)  // sdot v30.4s, v5.16b, v1.4b[3]
    ldr d5, [x7, #0x40]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    ldr x1, [x22, #0x8]
    KAI_ASM_INST(0x4fa0e0cd)  // sdot v13.4s, v6.16b, v0.4b[1]
    ldr x4, [x22, #0x18]
    KAI_ASM_INST(0x4f80e8d0)  // sdot v16.4s, v6.16b, v0.4b[2]
    mov v5.d[1], x3
    KAI_ASM_INST(0x4fa0e8d3)  // sdot v19.4s, v6.16b, v0.4b[3]
    ldr d0, [x22, #0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    sub x6, x6, #0x2
    KAI_ASM_INST(0x4fa1e0d9)  // sdot v25.4s, v6.16b, v1.4b[1]
    cmp x6, #0x2
    KAI_ASM_INST(0x4f81e8dc)  // sdot v28.4s, v6.16b, v1.4b[2]
    mov v0.d[1], x1
    KAI_ASM_INST(0x4fa1e8df)  // sdot v31.4s, v6.16b, v1.4b[3]
    ldr d6, [x7, #0x50]
    ldr d1, [x22, #0x10]
    add x7, x7, #0x60
    KAI_ASM_INST(0x4f82e088)  // sdot v8.4s, v4.16b, v2.4b[0]
    ldr x3, [x7, #0x8]
    mov v6.d[1], x2
    KAI_ASM_INST(0x4fa2e08b)  // sdot v11.4s, v4.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e88e)  // sdot v14.4s, v4.16b, v2.4b[2]
    ldr x2, [x7, #0x18]
    KAI_ASM_INST(0x4fa2e891)  // sdot v17.4s, v4.16b, v2.4b[3]
    ldr x1, [x7, #0x28]
    KAI_ASM_INST(0x4f83e094)  // sdot v20.4s, v4.16b, v3.4b[0]
    mov v1.d[1], x4
    KAI_ASM_INST(0x4fa3e097)  // sdot v23.4s, v4.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e89a)  // sdot v26.4s, v4.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e89d)  // sdot v29.4s, v4.16b, v3.4b[3]
    ldr d4, [x7, #0]
    KAI_ASM_INST(0x4f82e0a9)  // sdot v9.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e0ac)  // sdot v12.4s, v5.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e8af)  // sdot v15.4s, v5.16b, v2.4b[2]
    mov v4.d[1], x3
    KAI_ASM_INST(0x4fa2e8b2)  // sdot v18.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4f83e0b5)  // sdot v21.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0b8)  // sdot v24.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8bb)  // sdot v27.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8be)  // sdot v30.4s, v5.16b, v3.4b[3]
    ldr d5, [x7, #0x10]
    KAI_ASM_INST(0x4f82e0ca)  // sdot v10.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e0cd)  // sdot v13.4s, v6.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e8d0)  // sdot v16.4s, v6.16b, v2.4b[2]
    mov v5.d[1], x2
    KAI_ASM_INST(0x4fa2e8d3)  // sdot v19.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4f83e0d6)  // sdot v22.4s, v6.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0d9)  // sdot v25.4s, v6.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8dc)  // sdot v28.4s, v6.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8df)  // sdot v31.4s, v6.16b, v3.4b[3]
    ldr d6, [x7, #0x20]
    mov v6.d[1], x1
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x22, x22, #0x20
    add x7, x7, #0x30
    KAI_ASM_INST(0x4f80e088)  // sdot v8.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e08b)  // sdot v11.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e88e)  // sdot v14.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e891)  // sdot v17.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e097)  // sdot v23.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e89a)  // sdot v26.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e89d)  // sdot v29.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e0a9)  // sdot v9.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0ac)  // sdot v12.4s, v5.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8af)  // sdot v15.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8b2)  // sdot v18.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0b8)  // sdot v24.4s, v5.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8bb)  // sdot v27.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8be)  // sdot v30.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e0ca)  // sdot v10.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0cd)  // sdot v13.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8d0)  // sdot v16.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8d3)  // sdot v19.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0d9)  // sdot v25.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8dc)  // sdot v28.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8df)  // sdot v31.4s, v6.16b, v1.4b[3]
    cbz x6, label_5
    ldr q0, [x22, #0]
    ldr q1, [x22, #0x10]
    add x22, x22, #0x20
    ldr q7, [x7, #0]
    ldr q4, [x7, #0x10]
    ldr q5, [x7, #0x20]
    add x7, x7, #0x30
    KAI_ASM_INST(0x4f80e0e8)  // sdot v8.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0eb)  // sdot v11.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8ee)  // sdot v14.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8f1)  // sdot v17.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0f4)  // sdot v20.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0f7)  // sdot v23.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8fa)  // sdot v26.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8fd)  // sdot v29.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e089)  // sdot v9.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e08c)  // sdot v12.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e88f)  // sdot v15.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e892)  // sdot v18.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e095)  // sdot v21.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e098)  // sdot v24.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e89b)  // sdot v27.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e89e)  // sdot v30.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e0aa)  // sdot v10.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0ad)  // sdot v13.4s, v5.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8b0)  // sdot v16.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8b3)  // sdot v19.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0b6)  // sdot v22.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0b9)  // sdot v25.4s, v5.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8bc)  // sdot v28.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8bf)  // sdot v31.4s, v5.16b, v1.4b[3]
KAI_ASM_LABEL(label_5)  // multiply loop done
    subs x20, x20, #0xc
    str q8, [x21, #0]
    str q9, [x21, #0x10]
    str q10, [x21, #0x20]
    str q11, [x21, #0x30]
    str q12, [x21, #0x40]
    str q13, [x21, #0x50]
    str q14, [x21, #0x60]
    str q15, [x21, #0x70]
    str q16, [x21, #0x80]
    str q17, [x21, #0x90]
    str q18, [x21, #0xa0]
    str q19, [x21, #0xb0]
    str q20, [x21, #0xc0]
    str q21, [x21, #0xd0]
    str q22, [x21, #0xe0]
    str q23, [x21, #0xf0]
    str q24, [x21, #0x100]
    str q25, [x21, #0x110]
    str q26, [x21, #0x120]
    str q27, [x21, #0x130]
    str q28, [x21, #0x140]
    str q29, [x21, #0x150]
    str q30, [x21, #0x160]
    str q31, [x21, #0x170]
    add x21, x21, #0x180
    bgt label_2
    subs x23, x23, #0x8
    bgt label_1
KAI_ASM_LABEL(label_6)  // Exit
    ldp x22, x23, [sp, 16]
    ldp d8, d9, [sp, 32]
    ldp d10, d11, [sp, 48]
    ldp d12, d13, [sp, 64]
    ldp d14, d15, [sp, 80]
    ldp x20, x21, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_s8s32_dot_8x12_a55)

    KAI_ASM_END
