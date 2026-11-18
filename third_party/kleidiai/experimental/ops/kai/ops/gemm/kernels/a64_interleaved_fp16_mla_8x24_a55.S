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

    KAI_ASM_CODE(a64_interleaved_fp16_mla_8x24_a55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_fp16_mla_8x24_a55)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_fp16_mla_8x24_a55)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_fp16_mla_8x24_a55)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    stp d8, d9, [sp, 56]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    ldr x26, [x0, #0]
    ldr x25, [x0, #0x20]
    ldr x24, [x0, #0x50]
    cbz x26, label_6
KAI_ASM_LABEL(label_1)  // Height loop
    ldr x23, [x0, #0x8]
    mov x22, x25
    ldr x21, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // Width loop
    ldr q2, [x21, #0]
    mov x25, x22
    ldr q3, [x21, #0x10]
    movi v8.16b, #0
    ldr q4, [x21, #0x20]
    movi v9.16b, #0
    ldr q0, [x25, #0]
    movi v10.16b, #0
    ldr x20, [x0, #0x10]
    movi v11.16b, #0
    prfm pldl1keep, [x25, #0]
    movi v12.16b, #0
    prfm pldl1keep, [x21, #0]
    movi v13.16b, #0
    sub x20, x20, #0x1
    prfm pldl1keep, [x21, #0x40]
    cmp x20, #0x2
    movi v14.16b, #0
    prfm pldl1keep, [x21, #0x80]
    movi v15.16b, #0
    movi v16.16b, #0
    movi v17.16b, #0
    movi v18.16b, #0
    movi v19.16b, #0
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
    ldr d1, [x25, #0x10]
    fmla v8.8h, v2.8h, v0.h[0]
    ldr d5, [x21, #0x30]
    fmla v11.8h, v2.8h, v0.h[1]
    ldr d6, [x21, #0x40]
    fmla v14.8h, v2.8h, v0.h[2]
    ldr d7, [x21, #0x50]
    fmla v17.8h, v2.8h, v0.h[3]
    ldr x8, [x25, #0x18]
    fmla v20.8h, v2.8h, v0.h[4]
    ldr x7, [x21, #0x38]
    fmla v23.8h, v2.8h, v0.h[5]
    ldr x6, [x21, #0x48]
    fmla v26.8h, v2.8h, v0.h[6]
    ldr x5, [x21, #0x58]
    fmla v29.8h, v2.8h, v0.h[7]
    prfm pldl1keep, [x25, #0x80]
    add x25, x25, #0x20
    fmla v9.8h, v3.8h, v0.h[0]
    prfm pldl1keep, [x21, #0x100]
    fmla v12.8h, v3.8h, v0.h[1]
    prfm pldl1keep, [x21, #0x140]
    fmla v15.8h, v3.8h, v0.h[2]
    add x21, x21, #0x60
    ldr d2, [x21, #0]
    fmla v18.8h, v3.8h, v0.h[3]
    fmla v21.8h, v3.8h, v0.h[4]
    ldr x4, [x21, #0x8]
    fmla v24.8h, v3.8h, v0.h[5]
    ldr x3, [x21, #0x18]
    fmla v27.8h, v3.8h, v0.h[6]
    ldr x2, [x25, #0x8]
    fmla v30.8h, v3.8h, v0.h[7]
    ldr d3, [x21, #0x10]
    fmla v10.8h, v4.8h, v0.h[0]
    ldr x1, [x21, #0x28]
    fmla v13.8h, v4.8h, v0.h[1]
    mov v1.d[1], x8
    fmla v16.8h, v4.8h, v0.h[2]
    mov v5.d[1], x7
    fmla v19.8h, v4.8h, v0.h[3]
    mov v6.d[1], x6
    fmla v22.8h, v4.8h, v0.h[4]
    mov v7.d[1], x5
    fmla v25.8h, v4.8h, v0.h[5]
    sub x20, x20, #0x2
    fmla v28.8h, v4.8h, v0.h[6]
    cmp x20, #0x2
    fmla v31.8h, v4.8h, v0.h[7]
    ldr d0, [x25, #0]
    ldr d4, [x21, #0x20]
    mov v2.d[1], x4
    mov v3.d[1], x3
    fmla v8.8h, v5.8h, v1.h[0]
    mov v0.d[1], x2
    fmla v11.8h, v5.8h, v1.h[1]
    mov v4.d[1], x1
    fmla v14.8h, v5.8h, v1.h[2]
    fmla v17.8h, v5.8h, v1.h[3]
    fmla v20.8h, v5.8h, v1.h[4]
    fmla v23.8h, v5.8h, v1.h[5]
    fmla v26.8h, v5.8h, v1.h[6]
    fmla v29.8h, v5.8h, v1.h[7]
    fmla v9.8h, v6.8h, v1.h[0]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v15.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v1.h[3]
    fmla v21.8h, v6.8h, v1.h[4]
    fmla v24.8h, v6.8h, v1.h[5]
    fmla v27.8h, v6.8h, v1.h[6]
    fmla v30.8h, v6.8h, v1.h[7]
    fmla v10.8h, v7.8h, v1.h[0]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v16.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v1.h[3]
    fmla v22.8h, v7.8h, v1.h[4]
    fmla v25.8h, v7.8h, v1.h[5]
    fmla v28.8h, v7.8h, v1.h[6]
    fmla v31.8h, v7.8h, v1.h[7]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x25, x25, #0x10
    add x21, x21, #0x30
    fmla v8.8h, v2.8h, v0.h[0]
    fmla v11.8h, v2.8h, v0.h[1]
    fmla v14.8h, v2.8h, v0.h[2]
    fmla v17.8h, v2.8h, v0.h[3]
    fmla v20.8h, v2.8h, v0.h[4]
    fmla v23.8h, v2.8h, v0.h[5]
    fmla v26.8h, v2.8h, v0.h[6]
    fmla v29.8h, v2.8h, v0.h[7]
    fmla v9.8h, v3.8h, v0.h[0]
    fmla v12.8h, v3.8h, v0.h[1]
    fmla v15.8h, v3.8h, v0.h[2]
    fmla v18.8h, v3.8h, v0.h[3]
    fmla v21.8h, v3.8h, v0.h[4]
    fmla v24.8h, v3.8h, v0.h[5]
    fmla v27.8h, v3.8h, v0.h[6]
    fmla v30.8h, v3.8h, v0.h[7]
    fmla v10.8h, v4.8h, v0.h[0]
    fmla v13.8h, v4.8h, v0.h[1]
    fmla v16.8h, v4.8h, v0.h[2]
    fmla v19.8h, v4.8h, v0.h[3]
    fmla v22.8h, v4.8h, v0.h[4]
    fmla v25.8h, v4.8h, v0.h[5]
    fmla v28.8h, v4.8h, v0.h[6]
    fmla v31.8h, v4.8h, v0.h[7]
    cbz x20, label_5
    ldr q0, [x25, #0]
    add x25, x25, #0x10
    ldr q5, [x21, #0]
    ldr q6, [x21, #0x10]
    ldr q7, [x21, #0x20]
    add x21, x21, #0x30
    fmla v8.8h, v5.8h, v0.h[0]
    fmla v11.8h, v5.8h, v0.h[1]
    fmla v14.8h, v5.8h, v0.h[2]
    fmla v17.8h, v5.8h, v0.h[3]
    fmla v20.8h, v5.8h, v0.h[4]
    fmla v23.8h, v5.8h, v0.h[5]
    fmla v26.8h, v5.8h, v0.h[6]
    fmla v29.8h, v5.8h, v0.h[7]
    fmla v9.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v0.h[1]
    fmla v15.8h, v6.8h, v0.h[2]
    fmla v18.8h, v6.8h, v0.h[3]
    fmla v21.8h, v6.8h, v0.h[4]
    fmla v24.8h, v6.8h, v0.h[5]
    fmla v27.8h, v6.8h, v0.h[6]
    fmla v30.8h, v6.8h, v0.h[7]
    fmla v10.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v0.h[1]
    fmla v16.8h, v7.8h, v0.h[2]
    fmla v19.8h, v7.8h, v0.h[3]
    fmla v22.8h, v7.8h, v0.h[4]
    fmla v25.8h, v7.8h, v0.h[5]
    fmla v28.8h, v7.8h, v0.h[6]
    fmla v31.8h, v7.8h, v0.h[7]
KAI_ASM_LABEL(label_5)  // multiply loop done
    subs x23, x23, #0x18
    str q8, [x24, #0]
    str q9, [x24, #0x10]
    str q10, [x24, #0x20]
    str q11, [x24, #0x30]
    str q12, [x24, #0x40]
    str q13, [x24, #0x50]
    str q14, [x24, #0x60]
    str q15, [x24, #0x70]
    str q16, [x24, #0x80]
    str q17, [x24, #0x90]
    str q18, [x24, #0xa0]
    str q19, [x24, #0xb0]
    str q20, [x24, #0xc0]
    str q21, [x24, #0xd0]
    str q22, [x24, #0xe0]
    str q23, [x24, #0xf0]
    str q24, [x24, #0x100]
    str q25, [x24, #0x110]
    str q26, [x24, #0x120]
    str q27, [x24, #0x130]
    str q28, [x24, #0x140]
    str q29, [x24, #0x150]
    str q30, [x24, #0x160]
    str q31, [x24, #0x170]
    add x24, x24, #0x180
    bgt label_2
    subs x26, x26, #0x8
    bgt label_1
KAI_ASM_LABEL(label_6)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp d8, d9, [sp, 56]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_fp16_mla_8x24_a55)

    KAI_ASM_END
