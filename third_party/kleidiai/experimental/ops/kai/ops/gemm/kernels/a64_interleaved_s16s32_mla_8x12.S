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

    KAI_ASM_CODE(a64_interleaved_s16s32_mla_8x12)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_s16s32_mla_8x12)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_s16s32_mla_8x12)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_s16s32_mla_8x12)
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
    ldr d1, [x3, #0]
    ldr d2, [x3, #0x8]
    mov x6, x2
    ldr d3, [x3, #0x10]
    ldr x1, [x0, #0x10]
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    ldr q0, [x6, #0]
    movi v11.4s, #0
    movi v12.4s, #0
    sub x1, x1, #0x1
    movi v13.4s, #0
    movi v14.4s, #0
    cmp x1, #0x2
    movi v15.4s, #0
    movi v16.4s, #0
    prfm pldl1keep, [x6, #0]
    movi v17.4s, #0
    movi v18.4s, #0
    prfm pldl1keep, [x3, #0]
    movi v19.4s, #0
    movi v20.4s, #0
    prfm pldl1keep, [x3, #0x40]
    movi v21.4s, #0
    movi v22.4s, #0
    prfm pldl1keep, [x3, #0x80]
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
    smlal v8.4s, v1.4h, v0.h[0]
    smlal v11.4s, v1.4h, v0.h[1]
    sub x1, x1, #0x2
    smlal v14.4s, v1.4h, v0.h[2]
    smlal v17.4s, v1.4h, v0.h[3]
    cmp x1, #0x2
    smlal v20.4s, v1.4h, v0.h[4]
    smlal v23.4s, v1.4h, v0.h[5]
    prfm pldl1keep, [x6, #0x80]
    smlal v26.4s, v1.4h, v0.h[6]
    smlal v29.4s, v1.4h, v0.h[7]
    ldr d1, [x3, #0x18]
    smlal v9.4s, v2.4h, v0.h[0]
    smlal v12.4s, v2.4h, v0.h[1]
    prfm pldl1keep, [x3, #0x100]
    smlal v15.4s, v2.4h, v0.h[2]
    smlal v18.4s, v2.4h, v0.h[3]
    prfm pldl1keep, [x3, #0x140]
    smlal v21.4s, v2.4h, v0.h[4]
    smlal v24.4s, v2.4h, v0.h[5]
    smlal v27.4s, v2.4h, v0.h[6]
    smlal v30.4s, v2.4h, v0.h[7]
    ldr d2, [x3, #0x20]
    smlal v10.4s, v3.4h, v0.h[0]
    smlal v13.4s, v3.4h, v0.h[1]
    smlal v16.4s, v3.4h, v0.h[2]
    smlal v19.4s, v3.4h, v0.h[3]
    smlal v22.4s, v3.4h, v0.h[4]
    smlal v25.4s, v3.4h, v0.h[5]
    smlal v28.4s, v3.4h, v0.h[6]
    smlal v31.4s, v3.4h, v0.h[7]
    ldr q0, [x6, #0x10]
    ldr d3, [x3, #0x28]
    add x6, x6, #0x20
    add x3, x3, #0x30
    smlal v8.4s, v1.4h, v0.h[0]
    smlal v11.4s, v1.4h, v0.h[1]
    smlal v14.4s, v1.4h, v0.h[2]
    smlal v17.4s, v1.4h, v0.h[3]
    smlal v20.4s, v1.4h, v0.h[4]
    smlal v23.4s, v1.4h, v0.h[5]
    smlal v26.4s, v1.4h, v0.h[6]
    smlal v29.4s, v1.4h, v0.h[7]
    ldr d1, [x3, #0]
    smlal v9.4s, v2.4h, v0.h[0]
    smlal v12.4s, v2.4h, v0.h[1]
    smlal v15.4s, v2.4h, v0.h[2]
    smlal v18.4s, v2.4h, v0.h[3]
    smlal v21.4s, v2.4h, v0.h[4]
    smlal v24.4s, v2.4h, v0.h[5]
    smlal v27.4s, v2.4h, v0.h[6]
    smlal v30.4s, v2.4h, v0.h[7]
    ldr d2, [x3, #0x8]
    smlal v10.4s, v3.4h, v0.h[0]
    smlal v13.4s, v3.4h, v0.h[1]
    smlal v16.4s, v3.4h, v0.h[2]
    smlal v19.4s, v3.4h, v0.h[3]
    smlal v22.4s, v3.4h, v0.h[4]
    smlal v25.4s, v3.4h, v0.h[5]
    smlal v28.4s, v3.4h, v0.h[6]
    smlal v31.4s, v3.4h, v0.h[7]
    ldr q0, [x6, #0]
    ldr d3, [x3, #0x10]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x10
    add x3, x3, #0x18
    smlal v8.4s, v1.4h, v0.h[0]
    smlal v11.4s, v1.4h, v0.h[1]
    smlal v14.4s, v1.4h, v0.h[2]
    smlal v17.4s, v1.4h, v0.h[3]
    smlal v20.4s, v1.4h, v0.h[4]
    smlal v23.4s, v1.4h, v0.h[5]
    smlal v26.4s, v1.4h, v0.h[6]
    smlal v29.4s, v1.4h, v0.h[7]
    smlal v9.4s, v2.4h, v0.h[0]
    smlal v12.4s, v2.4h, v0.h[1]
    smlal v15.4s, v2.4h, v0.h[2]
    smlal v18.4s, v2.4h, v0.h[3]
    smlal v21.4s, v2.4h, v0.h[4]
    smlal v24.4s, v2.4h, v0.h[5]
    smlal v27.4s, v2.4h, v0.h[6]
    smlal v30.4s, v2.4h, v0.h[7]
    smlal v10.4s, v3.4h, v0.h[0]
    smlal v13.4s, v3.4h, v0.h[1]
    smlal v16.4s, v3.4h, v0.h[2]
    smlal v19.4s, v3.4h, v0.h[3]
    smlal v22.4s, v3.4h, v0.h[4]
    smlal v25.4s, v3.4h, v0.h[5]
    smlal v28.4s, v3.4h, v0.h[6]
    smlal v31.4s, v3.4h, v0.h[7]
    cbz x1, label_5
    ldr q0, [x6, #0]
    ldr d4, [x3, #0]
    add x6, x6, #0x10
    ldr d5, [x3, #0x8]
    ldr d6, [x3, #0x10]
    add x3, x3, #0x18
    smlal v8.4s, v4.4h, v0.h[0]
    smlal v11.4s, v4.4h, v0.h[1]
    smlal v14.4s, v4.4h, v0.h[2]
    smlal v17.4s, v4.4h, v0.h[3]
    smlal v20.4s, v4.4h, v0.h[4]
    smlal v23.4s, v4.4h, v0.h[5]
    smlal v26.4s, v4.4h, v0.h[6]
    smlal v29.4s, v4.4h, v0.h[7]
    smlal v9.4s, v5.4h, v0.h[0]
    smlal v12.4s, v5.4h, v0.h[1]
    smlal v15.4s, v5.4h, v0.h[2]
    smlal v18.4s, v5.4h, v0.h[3]
    smlal v21.4s, v5.4h, v0.h[4]
    smlal v24.4s, v5.4h, v0.h[5]
    smlal v27.4s, v5.4h, v0.h[6]
    smlal v30.4s, v5.4h, v0.h[7]
    smlal v10.4s, v6.4h, v0.h[0]
    smlal v13.4s, v6.4h, v0.h[1]
    smlal v16.4s, v6.4h, v0.h[2]
    smlal v19.4s, v6.4h, v0.h[3]
    smlal v22.4s, v6.4h, v0.h[4]
    smlal v25.4s, v6.4h, v0.h[5]
    smlal v28.4s, v6.4h, v0.h[6]
    smlal v31.4s, v6.4h, v0.h[7]
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_s16s32_mla_8x12)

    KAI_ASM_END
