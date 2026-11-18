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

    KAI_ASM_CODE(a64_interleaved_s8s32_mmla_8x12_a510)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_s8s32_mmla_8x12_a510)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_s8s32_mmla_8x12_a510)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_s8s32_mmla_8x12_a510)
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
    ldp q4, q5, [x3], #0x20
    mov x6, x2
    ldr x1, [x0, #0x10]
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    ld1 { v0.16b }, [x6], #0x10
    movi v12.4s, #0
    movi v13.4s, #0
    ld1 { v1.16b }, [x6], #0x10
    add x1, x1, #0x7
    movi v14.4s, #0
    ld1 { v2.16b }, [x6], #0x10
    lsr x1, x1, #0x3
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    sub x1, x1, #0x1
    movi v20.4s, #0
    movi v21.4s, #0
    cmp x1, #0x2
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
    ld1 { v3.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e84a408)  // smmla v8.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40b)  // smmla v11.4s, v0.16b, v5.16b
    ldp q6, q7, [x3], #0x20
    KAI_ASM_INST(0x4e84a42e)  // smmla v14.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a431)  // smmla v17.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a454)  // smmla v20.4s, v2.16b, v4.16b
    sub x1, x1, #0x2
    KAI_ASM_INST(0x4e85a457)  // smmla v23.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47a)  // smmla v26.4s, v3.16b, v4.16b
    cmp x1, #0x2
    KAI_ASM_INST(0x4e85a47d)  // smmla v29.4s, v3.16b, v5.16b
    ldp q4, q5, [x3], #0x20
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a432)  // smmla v18.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a458)  // smmla v24.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47b)  // smmla v27.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e87a47e)  // smmla v30.4s, v3.16b, v7.16b
    ldp q6, q7, [x3], #0x20
    KAI_ASM_INST(0x4e84a40a)  // smmla v10.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40d)  // smmla v13.4s, v0.16b, v5.16b
    ld1 { v0.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e84a430)  // smmla v16.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a433)  // smmla v19.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a456)  // smmla v22.4s, v2.16b, v4.16b
    ld1 { v1.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e85a459)  // smmla v25.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47c)  // smmla v28.4s, v3.16b, v4.16b
    ld1 { v2.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e85a47f)  // smmla v31.4s, v3.16b, v5.16b
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    ld1 { v3.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldp q4, q5, [x3], #0x20
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a431)  // smmla v17.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47a)  // smmla v26.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e87a47d)  // smmla v29.4s, v3.16b, v7.16b
    ldp q6, q7, [x3], #0x20
    KAI_ASM_INST(0x4e84a409)  // smmla v9.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40c)  // smmla v12.4s, v0.16b, v5.16b
    KAI_ASM_INST(0x4e84a42f)  // smmla v15.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a432)  // smmla v18.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a455)  // smmla v21.4s, v2.16b, v4.16b
    KAI_ASM_INST(0x4e85a458)  // smmla v24.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47b)  // smmla v27.4s, v3.16b, v4.16b
    KAI_ASM_INST(0x4e85a47e)  // smmla v30.4s, v3.16b, v5.16b
    ldp q4, q5, [x3], #0x20
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    ld1 { v0.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e86a430)  // smmla v16.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a433)  // smmla v19.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    ld1 { v1.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e87a459)  // smmla v25.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47c)  // smmla v28.4s, v3.16b, v6.16b
    ld1 { v2.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e87a47f)  // smmla v31.4s, v3.16b, v7.16b
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    ld1 { v3.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e84a408)  // smmla v8.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40b)  // smmla v11.4s, v0.16b, v5.16b
    ldp q6, q7, [x3], #0x20
    KAI_ASM_INST(0x4e84a42e)  // smmla v14.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a431)  // smmla v17.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a454)  // smmla v20.4s, v2.16b, v4.16b
    KAI_ASM_INST(0x4e85a457)  // smmla v23.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47a)  // smmla v26.4s, v3.16b, v4.16b
    KAI_ASM_INST(0x4e85a47d)  // smmla v29.4s, v3.16b, v5.16b
    ldp q4, q5, [x3], #0x20
    KAI_ASM_INST(0x4e86a409)  // smmla v9.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40c)  // smmla v12.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a42f)  // smmla v15.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a432)  // smmla v18.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a455)  // smmla v21.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a458)  // smmla v24.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47b)  // smmla v27.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e87a47e)  // smmla v30.4s, v3.16b, v7.16b
    KAI_ASM_INST(0x4e84a40a)  // smmla v10.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40d)  // smmla v13.4s, v0.16b, v5.16b
    KAI_ASM_INST(0x4e84a430)  // smmla v16.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a433)  // smmla v19.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a456)  // smmla v22.4s, v2.16b, v4.16b
    KAI_ASM_INST(0x4e85a459)  // smmla v25.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47c)  // smmla v28.4s, v3.16b, v4.16b
    KAI_ASM_INST(0x4e85a47f)  // smmla v31.4s, v3.16b, v5.16b
    cbz x1, label_5
    ldp q6, q7, [x3], #0x20
    ld1 { v0.16b }, [x6], #0x10
    ld1 { v1.16b }, [x6], #0x10
    ld1 { v2.16b }, [x6], #0x10
    ld1 { v3.16b }, [x6], #0x10
    KAI_ASM_INST(0x4e86a408)  // smmla v8.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40b)  // smmla v11.4s, v0.16b, v7.16b
    ldp q4, q5, [x3], #0x20
    KAI_ASM_INST(0x4e86a42e)  // smmla v14.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a431)  // smmla v17.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a454)  // smmla v20.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a457)  // smmla v23.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47a)  // smmla v26.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e87a47d)  // smmla v29.4s, v3.16b, v7.16b
    ldp q6, q7, [x3], #0x20
    KAI_ASM_INST(0x4e84a409)  // smmla v9.4s, v0.16b, v4.16b
    KAI_ASM_INST(0x4e85a40c)  // smmla v12.4s, v0.16b, v5.16b
    KAI_ASM_INST(0x4e84a42f)  // smmla v15.4s, v1.16b, v4.16b
    KAI_ASM_INST(0x4e85a432)  // smmla v18.4s, v1.16b, v5.16b
    KAI_ASM_INST(0x4e84a455)  // smmla v21.4s, v2.16b, v4.16b
    KAI_ASM_INST(0x4e85a458)  // smmla v24.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e84a47b)  // smmla v27.4s, v3.16b, v4.16b
    KAI_ASM_INST(0x4e85a47e)  // smmla v30.4s, v3.16b, v5.16b
    KAI_ASM_INST(0x4e86a40a)  // smmla v10.4s, v0.16b, v6.16b
    KAI_ASM_INST(0x4e87a40d)  // smmla v13.4s, v0.16b, v7.16b
    KAI_ASM_INST(0x4e86a430)  // smmla v16.4s, v1.16b, v6.16b
    KAI_ASM_INST(0x4e87a433)  // smmla v19.4s, v1.16b, v7.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e87a459)  // smmla v25.4s, v2.16b, v7.16b
    KAI_ASM_INST(0x4e86a47c)  // smmla v28.4s, v3.16b, v6.16b
    KAI_ASM_INST(0x4e87a47f)  // smmla v31.4s, v3.16b, v7.16b
KAI_ASM_LABEL(label_5)  // multiply loop done
    subs x4, x4, #0xc
    uzp1 v4.2d, v8.2d, v11.2d
    uzp2 v8.2d, v8.2d, v11.2d
    uzp1 v11.2d, v9.2d, v12.2d
    uzp2 v9.2d, v9.2d, v12.2d
    uzp1 v12.2d, v10.2d, v13.2d
    uzp2 v10.2d, v10.2d, v13.2d
    str q4, [x5, #0]
    uzp1 v13.2d, v14.2d, v17.2d
    uzp2 v14.2d, v14.2d, v17.2d
    str q11, [x5, #0x10]
    uzp1 v17.2d, v15.2d, v18.2d
    uzp2 v15.2d, v15.2d, v18.2d
    str q12, [x5, #0x20]
    uzp1 v18.2d, v16.2d, v19.2d
    uzp2 v16.2d, v16.2d, v19.2d
    str q8, [x5, #0x30]
    uzp1 v19.2d, v20.2d, v23.2d
    uzp2 v20.2d, v20.2d, v23.2d
    str q9, [x5, #0x40]
    uzp1 v23.2d, v21.2d, v24.2d
    uzp2 v21.2d, v21.2d, v24.2d
    str q10, [x5, #0x50]
    uzp1 v24.2d, v22.2d, v25.2d
    uzp2 v22.2d, v22.2d, v25.2d
    str q13, [x5, #0x60]
    uzp1 v25.2d, v26.2d, v29.2d
    uzp2 v26.2d, v26.2d, v29.2d
    str q17, [x5, #0x70]
    uzp1 v29.2d, v27.2d, v30.2d
    uzp2 v27.2d, v27.2d, v30.2d
    str q18, [x5, #0x80]
    uzp1 v30.2d, v28.2d, v31.2d
    uzp2 v28.2d, v28.2d, v31.2d
    str q14, [x5, #0x90]
    str q15, [x5, #0xa0]
    str q16, [x5, #0xb0]
    str q19, [x5, #0xc0]
    str q23, [x5, #0xd0]
    str q24, [x5, #0xe0]
    str q20, [x5, #0xf0]
    str q21, [x5, #0x100]
    str q22, [x5, #0x110]
    str q25, [x5, #0x120]
    str q29, [x5, #0x130]
    str q30, [x5, #0x140]
    str q26, [x5, #0x150]
    str q27, [x5, #0x160]
    str q28, [x5, #0x170]
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_s8s32_mmla_8x12_a510)

    KAI_ASM_END
