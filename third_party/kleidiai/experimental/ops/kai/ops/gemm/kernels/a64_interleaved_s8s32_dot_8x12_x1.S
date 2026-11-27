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

    KAI_ASM_CODE(a64_interleaved_s8s32_dot_8x12_x1)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_s8s32_dot_8x12_x1)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_s8s32_dot_8x12_x1)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_s8s32_dot_8x12_x1)
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
    ldr q2, [x3, #0]
    ldr q3, [x3, #0x10]
    mov x6, x2
    ldr q4, [x3, #0x20]
    ldr x1, [x0, #0x10]
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    ldr q0, [x6, #0]
    ldr q1, [x6, #0x10]
    movi v11.4s, #0
    add x1, x1, #0x3
    movi v12.4s, #0
    movi v13.4s, #0
    lsr x1, x1, #0x2
    movi v14.4s, #0
    movi v15.4s, #0
    prfm pldl1keep, [x6, #0]
    movi v16.4s, #0
    movi v17.4s, #0
    prfm pldl1keep, [x3, #0]
    movi v18.4s, #0
    movi v19.4s, #0
    sub x1, x1, #0x1
    prfm pldl1keep, [x3, #0x40]
    movi v20.4s, #0
    cmp x1, #0x2
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    prfm pldl1keep, [x6, #0x40]
    movi v25.4s, #0
    movi v26.4s, #0
    prfm pldl1keep, [x3, #0x80]
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    blt label_4
KAI_ASM_LABEL(label_3)  // main loop head
    KAI_ASM_INST(0x4f80e048)  // sdot v8.4s, v2.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e04b)  // sdot v11.4s, v2.16b, v0.4b[1]
    sub x1, x1, #0x2
    KAI_ASM_INST(0x4f80e84e)  // sdot v14.4s, v2.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e851)  // sdot v17.4s, v2.16b, v0.4b[3]
    cmp x1, #0x2
    KAI_ASM_INST(0x4f81e054)  // sdot v20.4s, v2.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e057)  // sdot v23.4s, v2.16b, v1.4b[1]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f81e85a)  // sdot v26.4s, v2.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e85d)  // sdot v29.4s, v2.16b, v1.4b[3]
    ldr q2, [x3, #0x30]
    KAI_ASM_INST(0x4f80e069)  // sdot v9.4s, v3.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e06c)  // sdot v12.4s, v3.16b, v0.4b[1]
    prfm pldl1keep, [x3, #0x100]
    KAI_ASM_INST(0x4f80e86f)  // sdot v15.4s, v3.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e872)  // sdot v18.4s, v3.16b, v0.4b[3]
    prfm pldl1keep, [x3, #0x140]
    KAI_ASM_INST(0x4f81e075)  // sdot v21.4s, v3.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e078)  // sdot v24.4s, v3.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e87b)  // sdot v27.4s, v3.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e87e)  // sdot v30.4s, v3.16b, v1.4b[3]
    ldr q3, [x3, #0x40]
    KAI_ASM_INST(0x4f80e08a)  // sdot v10.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e08d)  // sdot v13.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e890)  // sdot v16.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e893)  // sdot v19.4s, v4.16b, v0.4b[3]
    ldr q0, [x6, #0x20]
    KAI_ASM_INST(0x4f81e096)  // sdot v22.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e099)  // sdot v25.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e89c)  // sdot v28.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e89f)  // sdot v31.4s, v4.16b, v1.4b[3]
    ldr q1, [x6, #0x30]
    ldr q4, [x3, #0x50]
    add x6, x6, #0x40
    add x3, x3, #0x60
    KAI_ASM_INST(0x4f80e048)  // sdot v8.4s, v2.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e04b)  // sdot v11.4s, v2.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e84e)  // sdot v14.4s, v2.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e851)  // sdot v17.4s, v2.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e054)  // sdot v20.4s, v2.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e057)  // sdot v23.4s, v2.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e85a)  // sdot v26.4s, v2.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e85d)  // sdot v29.4s, v2.16b, v1.4b[3]
    ldr q2, [x3, #0]
    KAI_ASM_INST(0x4f80e069)  // sdot v9.4s, v3.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e06c)  // sdot v12.4s, v3.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e86f)  // sdot v15.4s, v3.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e872)  // sdot v18.4s, v3.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e075)  // sdot v21.4s, v3.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e078)  // sdot v24.4s, v3.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e87b)  // sdot v27.4s, v3.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e87e)  // sdot v30.4s, v3.16b, v1.4b[3]
    ldr q3, [x3, #0x10]
    KAI_ASM_INST(0x4f80e08a)  // sdot v10.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e08d)  // sdot v13.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e890)  // sdot v16.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e893)  // sdot v19.4s, v4.16b, v0.4b[3]
    ldr q0, [x6, #0]
    KAI_ASM_INST(0x4f81e096)  // sdot v22.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e099)  // sdot v25.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e89c)  // sdot v28.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e89f)  // sdot v31.4s, v4.16b, v1.4b[3]
    ldr q1, [x6, #0x10]
    ldr q4, [x3, #0x20]
    bge label_3
KAI_ASM_LABEL(label_4)  // main loop skip
    add x6, x6, #0x20
    add x3, x3, #0x30
    KAI_ASM_INST(0x4f80e048)  // sdot v8.4s, v2.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e04b)  // sdot v11.4s, v2.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e84e)  // sdot v14.4s, v2.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e851)  // sdot v17.4s, v2.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e054)  // sdot v20.4s, v2.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e057)  // sdot v23.4s, v2.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e85a)  // sdot v26.4s, v2.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e85d)  // sdot v29.4s, v2.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e069)  // sdot v9.4s, v3.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e06c)  // sdot v12.4s, v3.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e86f)  // sdot v15.4s, v3.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e872)  // sdot v18.4s, v3.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e075)  // sdot v21.4s, v3.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e078)  // sdot v24.4s, v3.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e87b)  // sdot v27.4s, v3.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e87e)  // sdot v30.4s, v3.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e08a)  // sdot v10.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e08d)  // sdot v13.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e890)  // sdot v16.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e893)  // sdot v19.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e096)  // sdot v22.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e099)  // sdot v25.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e89c)  // sdot v28.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e89f)  // sdot v31.4s, v4.16b, v1.4b[3]
    cbz x1, label_5
    ldr q0, [x6, #0]
    ldr q1, [x6, #0x10]
    add x6, x6, #0x20
    ldr q5, [x3, #0]
    ldr q6, [x3, #0x10]
    ldr q7, [x3, #0x20]
    add x3, x3, #0x30
    KAI_ASM_INST(0x4f80e0a8)  // sdot v8.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0ab)  // sdot v11.4s, v5.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8ae)  // sdot v14.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0b4)  // sdot v20.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0b7)  // sdot v23.4s, v5.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8ba)  // sdot v26.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8bd)  // sdot v29.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e0c9)  // sdot v9.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0cc)  // sdot v12.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8cf)  // sdot v15.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0d5)  // sdot v21.4s, v6.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0d8)  // sdot v24.4s, v6.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8db)  // sdot v27.4s, v6.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8de)  // sdot v30.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e0ea)  // sdot v10.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0ed)  // sdot v13.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8f0)  // sdot v16.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e0f6)  // sdot v22.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0f9)  // sdot v25.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8fc)  // sdot v28.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8ff)  // sdot v31.4s, v7.16b, v1.4b[3]
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_s8s32_dot_8x12_x1)

    KAI_ASM_END
