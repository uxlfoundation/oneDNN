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

    KAI_ASM_CODE(a64_interleaved_fp32_mla_8x12_a55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_fp32_mla_8x12_a55)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_fp32_mla_8x12_a55)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_fp32_mla_8x12_a55)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    ldr x24, [x0, #0]
    ldr x23, [x0, #0x20]
    ldr x22, [x0, #0x50]
    cbz x24, label_7
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x21, [x0, #0x8]
    mov x20, x23
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // Width loop
    ldr q4, [x8, #0]
    mov x23, x20
    ldr q5, [x8, #0x10]
    movi v8.16b, #0
    ldr q6, [x8, #0x20]
    movi v9.16b, #0
    ldr q0, [x23, #0]
    movi v10.16b, #0
    ldr q1, [x23, #0x10]
    movi v11.16b, #0
    ldr x7, [x0, #0x10]
    movi v12.16b, #0
    prfm pldl1keep, [x23, #0]
    movi v13.16b, #0
    prfm pldl1keep, [x8, #0]
    movi v14.16b, #0
    sub x7, x7, #0x1
    prfm pldl1keep, [x8, #0x40]
    cmp x7, #0x4
    prfm pldl1keep, [x23, #0x40]
    prfm pldl1keep, [x8, #0x80]
    movi v15.16b, #0
    prfm pldl1keep, [x23, #0x80]
    movi v16.16b, #0
    prfm pldl1keep, [x8, #0xc0]
    movi v17.16b, #0
    prfm pldl1keep, [x8, #0x100]
    movi v18.16b, #0
    prfm pldl1keep, [x23, #0xc0]
    movi v19.16b, #0
    prfm pldl1keep, [x8, #0x140]
    movi v20.16b, #0
    movi v21.16b, #0
    movi v22.16b, #0
    movi v23.16b, #0
    movi v24.16b, #0
    movi v25.16b, #0
    movi v26.16b, #0
    movi v27.16b, #0
    movi v28.16b, #0
    movi v29.16b, #0
    movi v30.16b, #0
    movi v31.16b, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    ldr d2, [x23, #0x20]
    fmla v8.4s, v4.4s, v0.s[0]
    ldr d3, [x23, #0x30]
    fmla v11.4s, v4.4s, v0.s[1]
    ldr d7, [x8, #0x30]
    fmla v14.4s, v4.4s, v0.s[2]
    ldr x4, [x23, #0x28]
    fmla v17.4s, v4.4s, v0.s[3]
    ldr x3, [x23, #0x38]
    fmla v20.4s, v4.4s, v1.s[0]
    ldr x2, [x8, #0x38]
    fmla v23.4s, v4.4s, v1.s[1]
    fmla v26.4s, v4.4s, v1.s[2]
    ldr x1, [x8, #0x48]
    fmla v29.4s, v4.4s, v1.s[3]
    ldr d4, [x8, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    mov v2.d[1], x4
    fmla v12.4s, v5.4s, v0.s[1]
    mov v3.d[1], x3
    fmla v15.4s, v5.4s, v0.s[2]
    mov v7.d[1], x2
    fmla v18.4s, v5.4s, v0.s[3]
    mov v4.d[1], x1
    fmla v21.4s, v5.4s, v1.s[0]
    ldr x4, [x8, #0x58]
    fmla v24.4s, v5.4s, v1.s[1]
    ldr x3, [x23, #0x48]
    fmla v27.4s, v5.4s, v1.s[2]
    ldr x2, [x23, #0x58]
    fmla v30.4s, v5.4s, v1.s[3]
    ldr d5, [x8, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr x1, [x8, #0x68]
    fmla v13.4s, v6.4s, v0.s[1]
    ldr x6, [x8, #0x78]
    fmla v16.4s, v6.4s, v0.s[2]
    mov v5.d[1], x4
    fmla v19.4s, v6.4s, v0.s[3]
    ldr d0, [x23, #0x40]
    fmla v22.4s, v6.4s, v1.s[0]
    ldr x5, [x8, #0x88]
    fmla v25.4s, v6.4s, v1.s[1]
    ldr x4, [x23, #0x68]
    fmla v28.4s, v6.4s, v1.s[2]
    mov v0.d[1], x3
    fmla v31.4s, v6.4s, v1.s[3]
    ldr d1, [x23, #0x50]
    ldr d6, [x8, #0x60]
    fmla v8.4s, v7.4s, v2.s[0]
    fmla v11.4s, v7.4s, v2.s[1]
    ldr x3, [x23, #0x78]
    fmla v14.4s, v7.4s, v2.s[2]
    mov v1.d[1], x2
    fmla v17.4s, v7.4s, v2.s[3]
    mov v6.d[1], x1
    fmla v20.4s, v7.4s, v3.s[0]
    ldr x2, [x8, #0x98]
    fmla v23.4s, v7.4s, v3.s[1]
    ldr x1, [x8, #0xa8]
    fmla v26.4s, v7.4s, v3.s[2]
    prfm pldl1keep, [x23, #0x100]
    fmla v29.4s, v7.4s, v3.s[3]
    ldr d7, [x8, #0x70]
    fmla v9.4s, v4.4s, v2.s[0]
    prfm pldl1keep, [x8, #0x180]
    fmla v12.4s, v4.4s, v2.s[1]
    prfm pldl1keep, [x8, #0x1c0]
    fmla v15.4s, v4.4s, v2.s[2]
    mov v7.d[1], x6
    fmla v18.4s, v4.4s, v2.s[3]
    ldr x6, [x8, #0xb8]
    fmla v21.4s, v4.4s, v3.s[0]
    prfm pldl1keep, [x23, #0x140]
    fmla v24.4s, v4.4s, v3.s[1]
    prfm pldl1keep, [x8, #0x200]
    fmla v27.4s, v4.4s, v3.s[2]
    sub x7, x7, #0x4
    fmla v30.4s, v4.4s, v3.s[3]
    ldr d4, [x8, #0x80]
    fmla v10.4s, v5.4s, v2.s[0]
    cmp x7, #0x4
    fmla v13.4s, v5.4s, v2.s[1]
    fmla v16.4s, v5.4s, v2.s[2]
    mov v4.d[1], x5
    fmla v19.4s, v5.4s, v2.s[3]
    ldr d2, [x23, #0x60]
    fmla v22.4s, v5.4s, v3.s[0]
    fmla v25.4s, v5.4s, v3.s[1]
    fmla v28.4s, v5.4s, v3.s[2]
    mov v2.d[1], x4
    fmla v31.4s, v5.4s, v3.s[3]
    ldr d3, [x23, #0x70]
    ldr d5, [x8, #0x90]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v11.4s, v6.4s, v0.s[1]
    add x23, x23, #0x80
    fmla v14.4s, v6.4s, v0.s[2]
    mov v3.d[1], x3
    fmla v17.4s, v6.4s, v0.s[3]
    mov v5.d[1], x2
    fmla v20.4s, v6.4s, v1.s[0]
    ldr x5, [x23, #0x8]
    fmla v23.4s, v6.4s, v1.s[1]
    ldr x4, [x23, #0x18]
    fmla v26.4s, v6.4s, v1.s[2]
    fmla v29.4s, v6.4s, v1.s[3]
    ldr d6, [x8, #0xa0]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v12.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v0.s[2]
    mov v6.d[1], x1
    fmla v18.4s, v7.4s, v0.s[3]
    fmla v21.4s, v7.4s, v1.s[0]
    fmla v24.4s, v7.4s, v1.s[1]
    fmla v27.4s, v7.4s, v1.s[2]
    fmla v30.4s, v7.4s, v1.s[3]
    ldr d7, [x8, #0xb0]
    fmla v10.4s, v4.4s, v0.s[0]
    add x8, x8, #0xc0
    fmla v13.4s, v4.4s, v0.s[1]
    ldr x3, [x8, #0x8]
    fmla v16.4s, v4.4s, v0.s[2]
    ldr x2, [x8, #0x18]
    fmla v19.4s, v4.4s, v0.s[3]
    ldr d0, [x23, #0]
    fmla v22.4s, v4.4s, v1.s[0]
    ldr x1, [x8, #0x28]
    fmla v25.4s, v4.4s, v1.s[1]
    mov v7.d[1], x6
    fmla v28.4s, v4.4s, v1.s[2]
    mov v0.d[1], x5
    fmla v31.4s, v4.4s, v1.s[3]
    ldr d1, [x23, #0x10]
    ldr d4, [x8, #0]
    fmla v8.4s, v5.4s, v2.s[0]
    fmla v11.4s, v5.4s, v2.s[1]
    fmla v14.4s, v5.4s, v2.s[2]
    mov v1.d[1], x4
    fmla v17.4s, v5.4s, v2.s[3]
    mov v4.d[1], x3
    fmla v20.4s, v5.4s, v3.s[0]
    fmla v23.4s, v5.4s, v3.s[1]
    fmla v26.4s, v5.4s, v3.s[2]
    fmla v29.4s, v5.4s, v3.s[3]
    ldr d5, [x8, #0x10]
    fmla v9.4s, v6.4s, v2.s[0]
    fmla v12.4s, v6.4s, v2.s[1]
    fmla v15.4s, v6.4s, v2.s[2]
    mov v5.d[1], x2
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v21.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v3.s[1]
    fmla v27.4s, v6.4s, v3.s[2]
    fmla v30.4s, v6.4s, v3.s[3]
    ldr d6, [x8, #0x20]
    fmla v10.4s, v7.4s, v2.s[0]
    fmla v13.4s, v7.4s, v2.s[1]
    mov v6.d[1], x1
    fmla v16.4s, v7.4s, v2.s[2]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v22.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v3.s[1]
    fmla v28.4s, v7.4s, v3.s[2]
    fmla v31.4s, v7.4s, v3.s[3]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x23, x23, #0x20
    add x8, x8, #0x30
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v11.4s, v4.4s, v0.s[1]
    fmla v14.4s, v4.4s, v0.s[2]
    fmla v17.4s, v4.4s, v0.s[3]
    fmla v20.4s, v4.4s, v1.s[0]
    fmla v23.4s, v4.4s, v1.s[1]
    fmla v26.4s, v4.4s, v1.s[2]
    fmla v29.4s, v4.4s, v1.s[3]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v12.4s, v5.4s, v0.s[1]
    fmla v15.4s, v5.4s, v0.s[2]
    fmla v18.4s, v5.4s, v0.s[3]
    fmla v21.4s, v5.4s, v1.s[0]
    fmla v24.4s, v5.4s, v1.s[1]
    fmla v27.4s, v5.4s, v1.s[2]
    fmla v30.4s, v5.4s, v1.s[3]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v13.4s, v6.4s, v0.s[1]
    fmla v16.4s, v6.4s, v0.s[2]
    fmla v19.4s, v6.4s, v0.s[3]
    fmla v22.4s, v6.4s, v1.s[0]
    fmla v25.4s, v6.4s, v1.s[1]
    fmla v28.4s, v6.4s, v1.s[2]
    fmla v31.4s, v6.4s, v1.s[3]
    cbz x7, label_6
KAI_ASM_LABEL(label_5)  // odd loop
    ldr q0, [x23, #0]
    subs x7, x7, #0x1
    ldr q1, [x23, #0x10]
    add x23, x23, #0x20
    ldr q7, [x8, #0]
    ldr q4, [x8, #0x10]
    ldr q5, [x8, #0x20]
    add x8, x8, #0x30
    fmla v8.4s, v7.4s, v0.s[0]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v14.4s, v7.4s, v0.s[2]
    fmla v17.4s, v7.4s, v0.s[3]
    fmla v20.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v1.s[1]
    fmla v26.4s, v7.4s, v1.s[2]
    fmla v29.4s, v7.4s, v1.s[3]
    fmla v9.4s, v4.4s, v0.s[0]
    fmla v12.4s, v4.4s, v0.s[1]
    fmla v15.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v0.s[3]
    fmla v21.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v1.s[1]
    fmla v27.4s, v4.4s, v1.s[2]
    fmla v30.4s, v4.4s, v1.s[3]
    fmla v10.4s, v5.4s, v0.s[0]
    fmla v13.4s, v5.4s, v0.s[1]
    fmla v16.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v0.s[3]
    fmla v22.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v1.s[1]
    fmla v28.4s, v5.4s, v1.s[2]
    fmla v31.4s, v5.4s, v1.s[3]
    bne label_5
KAI_ASM_LABEL(label_6)  // multiply loop done
    subs x21, x21, #0xc
    str q8, [x22, #0]
    str q9, [x22, #0x10]
    str q10, [x22, #0x20]
    str q11, [x22, #0x30]
    str q12, [x22, #0x40]
    str q13, [x22, #0x50]
    str q14, [x22, #0x60]
    str q15, [x22, #0x70]
    str q16, [x22, #0x80]
    str q17, [x22, #0x90]
    str q18, [x22, #0xa0]
    str q19, [x22, #0xb0]
    str q20, [x22, #0xc0]
    str q21, [x22, #0xd0]
    str q22, [x22, #0xe0]
    str q23, [x22, #0xf0]
    str q24, [x22, #0x100]
    str q25, [x22, #0x110]
    str q26, [x22, #0x120]
    str q27, [x22, #0x130]
    str q28, [x22, #0x140]
    str q29, [x22, #0x150]
    str q30, [x22, #0x160]
    str q31, [x22, #0x170]
    add x22, x22, #0x180
    bgt label_2
    subs x24, x24, #0x8
    bgt label_1
KAI_ASM_LABEL(label_7)  // Exit
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_fp32_mla_8x12_a55)

    KAI_ASM_END
