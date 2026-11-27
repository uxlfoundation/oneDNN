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

    KAI_ASM_CODE(a64_interleave_4_4x1_bf16fp32)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave_4_4x1_bf16fp32)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave_4_4x1_bf16fp32)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave_4_4x1_bf16fp32)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    str d12, [sp, 104]
    sub SP, SP, #0x30
    movi v12.4s, #0
    str q12, [SP, #0]
KAI_ASM_LABEL(label_1)  // Row loop
    lsl x16, x5, #0x2
    mov x15, x5
    cmp x16, x3
    csel x16, x16, x3, LT
    sub x3, x3, x16
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x16, #0x10
    blt label_27
    mov x17, x1
    mov x22, x2
    str x3, [SP, #0x10]
    sub x15, x15, #0x4
    subs x16, x16, #0x10
    csel x13, x15, XZR, EQ
    str x16, [SP, #0x18]
    mov x14, x0
    add x12, x17, x4, LSL #3
    add x16, x17, x4, LSL #2
    str x15, [SP, #0x20]
    add x10, x12, x4, LSL #2
    add x9, x17, x4, LSL #1
    add x28, x16, x4, LSL #1
    add x11, x12, x4, LSL #1
    add x27, x10, x4, LSL #1
    cmp x22, #0x4
    add x25, x17, x4
    add x24, x9, x4
    add x23, x16, x4
    add x26, x28, x4
    add x21, x12, x4
    add x20, x11, x4
    add x8, x10, x4
    add x7, x27, x4
    add x1, x17, x4, LSL #4
    blt label_14
KAI_ASM_LABEL(label_3)  // 4 block: Column loop
    ldr d11, [x17], #0x8
    ldr d10, [x25], #0x8
    ldr d9, [x9], #0x8
    ldr d8, [x24], #0x8
    ldr d7, [x16], #0x8
    ldr d6, [x23], #0x8
    ldr d5, [x28], #0x8
    ldr d4, [x26], #0x8
    shll v11.4s, v11.4h, #0x10
    shll v10.4s, v10.4h, #0x10
    ldr d3, [x12], #0x8
    ldr d2, [x21], #0x8
    shll v9.4s, v9.4h, #0x10
    shll v8.4s, v8.4h, #0x10
    ldr d1, [x11], #0x8
    ldr d0, [x20], #0x8
    shll v7.4s, v7.4h, #0x10
    shll v6.4s, v6.4h, #0x10
    ldr d31, [x10], #0x8
    ldr d30, [x8], #0x8
    shll v5.4s, v5.4h, #0x10
    shll v4.4s, v4.4h, #0x10
    ldr d29, [x27], #0x8
    ldr d28, [x7], #0x8
    shll v3.4s, v3.4h, #0x10
    shll v2.4s, v2.4h, #0x10
    shll v1.4s, v1.4h, #0x10
    shll v0.4s, v0.4h, #0x10
    shll v31.4s, v31.4h, #0x10
    shll v30.4s, v30.4h, #0x10
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    zip1 v27.4s, v11.4s, v9.4s
    zip1 v26.4s, v10.4s, v8.4s
    zip1 v25.4s, v7.4s, v5.4s
    zip1 v24.4s, v6.4s, v4.4s
    zip1 v23.4s, v3.4s, v1.4s
    zip1 v22.4s, v2.4s, v0.4s
    zip1 v21.4s, v31.4s, v29.4s
    zip1 v20.4s, v30.4s, v28.4s
    zip1 v19.4s, v27.4s, v26.4s
    zip1 v18.4s, v25.4s, v24.4s
    zip1 v17.4s, v23.4s, v22.4s
    zip1 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_6
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_5)  // 4 block: unique 2: unique 3: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_5
KAI_ASM_LABEL(label_6)  // 4 block: unique 2: unique 3: Padding done
    add x14, x14, x5, LSL #4
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_8
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_7)  // 4 block: unique 2: unique 4: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_7
KAI_ASM_LABEL(label_8)  // 4 block: unique 2: unique 4: Padding done
    zip2 v27.4s, v11.4s, v9.4s
    zip2 v26.4s, v10.4s, v8.4s
    add x14, x14, x5, LSL #4
    zip2 v25.4s, v7.4s, v5.4s
    zip2 v24.4s, v6.4s, v4.4s
    zip2 v23.4s, v3.4s, v1.4s
    zip2 v22.4s, v2.4s, v0.4s
    zip2 v21.4s, v31.4s, v29.4s
    zip2 v20.4s, v30.4s, v28.4s
    zip1 v19.4s, v27.4s, v26.4s
    zip1 v18.4s, v25.4s, v24.4s
    zip1 v17.4s, v23.4s, v22.4s
    zip1 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_10
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_9)  // 4 block: unique 2: unique 5: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_9
KAI_ASM_LABEL(label_10)  // 4 block: unique 2: unique 5: Padding done
    add x14, x14, x5, LSL #4
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_12
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_11)  // 4 block: unique 2: unique 6: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_11
KAI_ASM_LABEL(label_12)  // 4 block: unique 2: unique 6: Padding done
    add x14, x14, x5, LSL #4
    sub x22, x22, #0x4
    cmp x22, #0x4
    bge label_3
KAI_ASM_LABEL(label_14)  // 4 block: Loop skip
    cbz x22, label_26
    tbz x22, #1, label_15
    ldr s11, [x17], #0x4
    ldr s10, [x25], #0x4
    ldr s9, [x9], #0x4
    ldr s8, [x24], #0x4
    ldr s7, [x16], #0x4
    ldr s6, [x23], #0x4
    ldr s5, [x28], #0x4
    ldr s4, [x26], #0x4
    ldr s3, [x12], #0x4
    ldr s2, [x21], #0x4
    ldr s1, [x11], #0x4
    ldr s0, [x20], #0x4
    ldr s31, [x10], #0x4
    ldr s30, [x8], #0x4
    ldr s29, [x27], #0x4
    ldr s28, [x7], #0x4
    tbz x22, #0, label_16
    ld1 { v11.h }[2], [x17]
    ld1 { v10.h }[2], [x25]
    ld1 { v9.h }[2], [x9]
    ld1 { v8.h }[2], [x24]
    ld1 { v7.h }[2], [x16]
    ld1 { v6.h }[2], [x23]
    ld1 { v5.h }[2], [x28]
    ld1 { v4.h }[2], [x26]
    ld1 { v3.h }[2], [x12]
    ld1 { v2.h }[2], [x21]
    ld1 { v1.h }[2], [x11]
    ld1 { v0.h }[2], [x20]
    ld1 { v31.h }[2], [x10]
    ld1 { v30.h }[2], [x8]
    ld1 { v29.h }[2], [x27]
    ld1 { v28.h }[2], [x7]
    b label_16
KAI_ASM_LABEL(label_15)  // 4 block: unique 7: Partial load: partial_0_0
    ldr h11, [x17, #0]
    ldr h10, [x25, #0]
    ldr h9, [x9, #0]
    ldr h8, [x24, #0]
    ldr h7, [x16, #0]
    ldr h6, [x23, #0]
    ldr h5, [x28, #0]
    ldr h4, [x26, #0]
    ldr h3, [x12, #0]
    ldr h2, [x21, #0]
    ldr h1, [x11, #0]
    ldr h0, [x20, #0]
    ldr h31, [x10, #0]
    ldr h30, [x8, #0]
    ldr h29, [x27, #0]
    ldr h28, [x7, #0]
KAI_ASM_LABEL(label_16)  // 4 block: unique 7: Partial load: Done
    shll v11.4s, v11.4h, #0x10
    shll v10.4s, v10.4h, #0x10
    shll v9.4s, v9.4h, #0x10
    shll v8.4s, v8.4h, #0x10
    shll v7.4s, v7.4h, #0x10
    shll v6.4s, v6.4h, #0x10
    shll v5.4s, v5.4h, #0x10
    shll v4.4s, v4.4h, #0x10
    shll v3.4s, v3.4h, #0x10
    shll v2.4s, v2.4h, #0x10
    shll v1.4s, v1.4h, #0x10
    shll v0.4s, v0.4h, #0x10
    shll v31.4s, v31.4h, #0x10
    shll v30.4s, v30.4h, #0x10
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    zip1 v27.4s, v11.4s, v9.4s
    zip1 v26.4s, v10.4s, v8.4s
    zip1 v25.4s, v7.4s, v5.4s
    zip1 v24.4s, v6.4s, v4.4s
    zip1 v23.4s, v3.4s, v1.4s
    zip1 v22.4s, v2.4s, v0.4s
    zip1 v21.4s, v31.4s, v29.4s
    zip1 v20.4s, v30.4s, v28.4s
    zip1 v19.4s, v27.4s, v26.4s
    zip1 v18.4s, v25.4s, v24.4s
    zip1 v17.4s, v23.4s, v22.4s
    zip1 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_18
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_17)  // 4 block: unique 8: unique 9: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_17
KAI_ASM_LABEL(label_18)  // 4 block: unique 8: unique 9: Padding done
    cmp x22, #0x1
    add x14, x14, x5, LSL #4
    ble label_25
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_20
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_19)  // 4 block: unique 8: unique 10: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_19
KAI_ASM_LABEL(label_20)  // 4 block: unique 8: unique 10: Padding done
    cmp x22, #0x2
    add x14, x14, x5, LSL #4
    ble label_25
    zip2 v27.4s, v11.4s, v9.4s
    zip2 v26.4s, v10.4s, v8.4s
    zip2 v25.4s, v7.4s, v5.4s
    zip2 v24.4s, v6.4s, v4.4s
    zip2 v23.4s, v3.4s, v1.4s
    zip2 v22.4s, v2.4s, v0.4s
    zip2 v21.4s, v31.4s, v29.4s
    zip2 v20.4s, v30.4s, v28.4s
    zip1 v19.4s, v27.4s, v26.4s
    zip1 v18.4s, v25.4s, v24.4s
    zip1 v17.4s, v23.4s, v22.4s
    zip1 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_22
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_21)  // 4 block: unique 8: unique 11: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_21
KAI_ASM_LABEL(label_22)  // 4 block: unique 8: unique 11: Padding done
    cmp x22, #0x3
    add x14, x14, x5, LSL #4
    ble label_25
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_24
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_23)  // 4 block: unique 8: unique 12: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_23
KAI_ASM_LABEL(label_24)  // 4 block: unique 8: unique 12: Padding done
    add x14, x14, x5, LSL #4
KAI_ASM_LABEL(label_25)  // 4 block: unique 8: Done
KAI_ASM_LABEL(label_26)  // 4 block: Oddments skip
    add x0, x0, #0x40
    ldr x3, [SP, #0x10]
    ldr x16, [SP, #0x18]
    add x10, x14, #0x40
    ldr x15, [SP, #0x20]
    b label_128
KAI_ASM_LABEL(label_27)  // 4 skip
    cmp x16, #0xc
    blt label_52
    mov x12, x1
    mov x11, x2
    str x3, [SP, #0x10]
    sub x15, x15, #0x3
    subs x16, x16, #0xc
    csel x10, x15, XZR, EQ
    str x16, [SP, #0x18]
    mov x9, x0
    add x28, x12, x4, LSL #2
    add x27, x12, x4, LSL #3
    str x15, [SP, #0x20]
    add x26, x12, x4, LSL #1
    add x25, x28, x4, LSL #1
    add x24, x27, x4, LSL #1
    cmp x11, #0x4
    add x23, x12, x4
    add x22, x26, x4
    add x21, x28, x4
    add x20, x25, x4
    add x8, x27, x4
    add x7, x24, x4
    add x1, x27, x4, LSL #2
    blt label_39
KAI_ASM_LABEL(label_28)  // 3 block: Column loop
    ldr d4, [x12], #0x8
    ldr d3, [x23], #0x8
    ldr d2, [x26], #0x8
    ldr d1, [x22], #0x8
    ldr d0, [x28], #0x8
    ldr d31, [x21], #0x8
    ldr d30, [x25], #0x8
    ldr d29, [x20], #0x8
    shll v4.4s, v4.4h, #0x10
    shll v3.4s, v3.4h, #0x10
    ldr d28, [x27], #0x8
    ldr d27, [x8], #0x8
    shll v2.4s, v2.4h, #0x10
    shll v1.4s, v1.4h, #0x10
    ldr d26, [x24], #0x8
    ldr d25, [x7], #0x8
    shll v0.4s, v0.4h, #0x10
    shll v31.4s, v31.4h, #0x10
    shll v30.4s, v30.4h, #0x10
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    shll v27.4s, v27.4h, #0x10
    shll v26.4s, v26.4h, #0x10
    shll v25.4s, v25.4h, #0x10
    zip1 v24.4s, v4.4s, v2.4s
    zip1 v23.4s, v3.4s, v1.4s
    zip1 v22.4s, v0.4s, v30.4s
    zip1 v21.4s, v31.4s, v29.4s
    zip1 v20.4s, v28.4s, v26.4s
    zip1 v19.4s, v27.4s, v25.4s
    zip1 v18.4s, v24.4s, v23.4s
    zip1 v17.4s, v22.4s, v21.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_31
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_30)  // 3 block: unique 14: unique 15: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_30
KAI_ASM_LABEL(label_31)  // 3 block: unique 14: unique 15: Padding done
    add x9, x9, x5, LSL #4
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_33
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_32)  // 3 block: unique 14: unique 16: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_32
KAI_ASM_LABEL(label_33)  // 3 block: unique 14: unique 16: Padding done
    zip2 v24.4s, v4.4s, v2.4s
    zip2 v23.4s, v3.4s, v1.4s
    add x9, x9, x5, LSL #4
    zip2 v22.4s, v0.4s, v30.4s
    zip2 v21.4s, v31.4s, v29.4s
    zip2 v20.4s, v28.4s, v26.4s
    zip2 v19.4s, v27.4s, v25.4s
    zip1 v18.4s, v24.4s, v23.4s
    zip1 v17.4s, v22.4s, v21.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_35
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_34)  // 3 block: unique 14: unique 17: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_34
KAI_ASM_LABEL(label_35)  // 3 block: unique 14: unique 17: Padding done
    add x9, x9, x5, LSL #4
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_37
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_36)  // 3 block: unique 14: unique 18: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_36
KAI_ASM_LABEL(label_37)  // 3 block: unique 14: unique 18: Padding done
    add x9, x9, x5, LSL #4
    sub x11, x11, #0x4
    cmp x11, #0x4
    bge label_28
KAI_ASM_LABEL(label_39)  // 3 block: Loop skip
    cbz x11, label_51
    tbz x11, #1, label_40
    ldr s4, [x12], #0x4
    ldr s3, [x23], #0x4
    ldr s2, [x26], #0x4
    ldr s1, [x22], #0x4
    ldr s0, [x28], #0x4
    ldr s31, [x21], #0x4
    ldr s30, [x25], #0x4
    ldr s29, [x20], #0x4
    ldr s28, [x27], #0x4
    ldr s27, [x8], #0x4
    ldr s26, [x24], #0x4
    ldr s25, [x7], #0x4
    tbz x11, #0, label_41
    ld1 { v4.h }[2], [x12]
    ld1 { v3.h }[2], [x23]
    ld1 { v2.h }[2], [x26]
    ld1 { v1.h }[2], [x22]
    ld1 { v0.h }[2], [x28]
    ld1 { v31.h }[2], [x21]
    ld1 { v30.h }[2], [x25]
    ld1 { v29.h }[2], [x20]
    ld1 { v28.h }[2], [x27]
    ld1 { v27.h }[2], [x8]
    ld1 { v26.h }[2], [x24]
    ld1 { v25.h }[2], [x7]
    b label_41
KAI_ASM_LABEL(label_40)  // 3 block: unique 19: Partial load: partial_0_0
    ldr h4, [x12, #0]
    ldr h3, [x23, #0]
    ldr h2, [x26, #0]
    ldr h1, [x22, #0]
    ldr h0, [x28, #0]
    ldr h31, [x21, #0]
    ldr h30, [x25, #0]
    ldr h29, [x20, #0]
    ldr h28, [x27, #0]
    ldr h27, [x8, #0]
    ldr h26, [x24, #0]
    ldr h25, [x7, #0]
KAI_ASM_LABEL(label_41)  // 3 block: unique 19: Partial load: Done
    shll v4.4s, v4.4h, #0x10
    shll v3.4s, v3.4h, #0x10
    shll v2.4s, v2.4h, #0x10
    shll v1.4s, v1.4h, #0x10
    shll v0.4s, v0.4h, #0x10
    shll v31.4s, v31.4h, #0x10
    shll v30.4s, v30.4h, #0x10
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    shll v27.4s, v27.4h, #0x10
    shll v26.4s, v26.4h, #0x10
    shll v25.4s, v25.4h, #0x10
    zip1 v24.4s, v4.4s, v2.4s
    zip1 v23.4s, v3.4s, v1.4s
    zip1 v22.4s, v0.4s, v30.4s
    zip1 v21.4s, v31.4s, v29.4s
    zip1 v20.4s, v28.4s, v26.4s
    zip1 v19.4s, v27.4s, v25.4s
    zip1 v18.4s, v24.4s, v23.4s
    zip1 v17.4s, v22.4s, v21.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_43
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_42)  // 3 block: unique 20: unique 21: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_42
KAI_ASM_LABEL(label_43)  // 3 block: unique 20: unique 21: Padding done
    cmp x11, #0x1
    add x9, x9, x5, LSL #4
    ble label_50
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_45
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_44)  // 3 block: unique 20: unique 22: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_44
KAI_ASM_LABEL(label_45)  // 3 block: unique 20: unique 22: Padding done
    cmp x11, #0x2
    add x9, x9, x5, LSL #4
    ble label_50
    zip2 v24.4s, v4.4s, v2.4s
    zip2 v23.4s, v3.4s, v1.4s
    zip2 v22.4s, v0.4s, v30.4s
    zip2 v21.4s, v31.4s, v29.4s
    zip2 v20.4s, v28.4s, v26.4s
    zip2 v19.4s, v27.4s, v25.4s
    zip1 v18.4s, v24.4s, v23.4s
    zip1 v17.4s, v22.4s, v21.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_47
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_46)  // 3 block: unique 20: unique 23: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_46
KAI_ASM_LABEL(label_47)  // 3 block: unique 20: unique 23: Padding done
    cmp x11, #0x3
    add x9, x9, x5, LSL #4
    ble label_50
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_49
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_48)  // 3 block: unique 20: unique 24: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_48
KAI_ASM_LABEL(label_49)  // 3 block: unique 20: unique 24: Padding done
    add x9, x9, x5, LSL #4
KAI_ASM_LABEL(label_50)  // 3 block: unique 20: Done
KAI_ASM_LABEL(label_51)  // 3 block: Oddments skip
    add x0, x0, #0x30
    ldr x3, [SP, #0x10]
    ldr x16, [SP, #0x18]
    add x10, x9, #0x30
    ldr x15, [SP, #0x20]
    b label_128
KAI_ASM_LABEL(label_52)  // 3 skip
    cmp x16, #0x8
    blt label_77
    mov x9, x1
    mov x28, x2
    sub x15, x15, #0x2
    subs x16, x16, #0x8
    csel x27, x15, XZR, EQ
    mov x26, x0
    add x25, x9, x4, LSL #2
    add x24, x9, x4, LSL #1
    add x23, x25, x4, LSL #1
    cmp x28, #0x4
    add x22, x9, x4
    add x21, x24, x4
    add x20, x25, x4
    add x8, x23, x4
    add x1, x9, x4, LSL #3
    blt label_64
KAI_ASM_LABEL(label_53)  // 2 block: Column loop
    ldr d29, [x9], #0x8
    ldr d28, [x22], #0x8
    ldr d27, [x24], #0x8
    ldr d26, [x21], #0x8
    ldr d25, [x25], #0x8
    ldr d24, [x20], #0x8
    ldr d23, [x23], #0x8
    ldr d22, [x8], #0x8
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    shll v27.4s, v27.4h, #0x10
    shll v26.4s, v26.4h, #0x10
    shll v25.4s, v25.4h, #0x10
    shll v24.4s, v24.4h, #0x10
    shll v23.4s, v23.4h, #0x10
    shll v22.4s, v22.4h, #0x10
    zip1 v21.4s, v29.4s, v27.4s
    zip1 v20.4s, v28.4s, v26.4s
    zip1 v19.4s, v25.4s, v23.4s
    zip1 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_56
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_55)  // 2 block: unique 26: unique 27: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_55
KAI_ASM_LABEL(label_56)  // 2 block: unique 26: unique 27: Padding done
    add x26, x26, x5, LSL #4
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_58
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_57)  // 2 block: unique 26: unique 28: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_57
KAI_ASM_LABEL(label_58)  // 2 block: unique 26: unique 28: Padding done
    zip2 v21.4s, v29.4s, v27.4s
    zip2 v20.4s, v28.4s, v26.4s
    add x26, x26, x5, LSL #4
    zip2 v19.4s, v25.4s, v23.4s
    zip2 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_60
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_59)  // 2 block: unique 26: unique 29: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_59
KAI_ASM_LABEL(label_60)  // 2 block: unique 26: unique 29: Padding done
    add x26, x26, x5, LSL #4
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_62
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_61)  // 2 block: unique 26: unique 30: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_61
KAI_ASM_LABEL(label_62)  // 2 block: unique 26: unique 30: Padding done
    add x26, x26, x5, LSL #4
    sub x28, x28, #0x4
    cmp x28, #0x4
    bge label_53
KAI_ASM_LABEL(label_64)  // 2 block: Loop skip
    cbz x28, label_76
    tbz x28, #1, label_65
    ldr s29, [x9], #0x4
    ldr s28, [x22], #0x4
    ldr s27, [x24], #0x4
    ldr s26, [x21], #0x4
    ldr s25, [x25], #0x4
    ldr s24, [x20], #0x4
    ldr s23, [x23], #0x4
    ldr s22, [x8], #0x4
    tbz x28, #0, label_66
    ld1 { v29.h }[2], [x9]
    ld1 { v28.h }[2], [x22]
    ld1 { v27.h }[2], [x24]
    ld1 { v26.h }[2], [x21]
    ld1 { v25.h }[2], [x25]
    ld1 { v24.h }[2], [x20]
    ld1 { v23.h }[2], [x23]
    ld1 { v22.h }[2], [x8]
    b label_66
KAI_ASM_LABEL(label_65)  // 2 block: unique 31: Partial load: partial_0_0
    ldr h29, [x9, #0]
    ldr h28, [x22, #0]
    ldr h27, [x24, #0]
    ldr h26, [x21, #0]
    ldr h25, [x25, #0]
    ldr h24, [x20, #0]
    ldr h23, [x23, #0]
    ldr h22, [x8, #0]
KAI_ASM_LABEL(label_66)  // 2 block: unique 31: Partial load: Done
    shll v29.4s, v29.4h, #0x10
    shll v28.4s, v28.4h, #0x10
    shll v27.4s, v27.4h, #0x10
    shll v26.4s, v26.4h, #0x10
    shll v25.4s, v25.4h, #0x10
    shll v24.4s, v24.4h, #0x10
    shll v23.4s, v23.4h, #0x10
    shll v22.4s, v22.4h, #0x10
    zip1 v21.4s, v29.4s, v27.4s
    zip1 v20.4s, v28.4s, v26.4s
    zip1 v19.4s, v25.4s, v23.4s
    zip1 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_68
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_67)  // 2 block: unique 32: unique 33: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_67
KAI_ASM_LABEL(label_68)  // 2 block: unique 32: unique 33: Padding done
    cmp x28, #0x1
    add x26, x26, x5, LSL #4
    ble label_75
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_70
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_69)  // 2 block: unique 32: unique 34: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_69
KAI_ASM_LABEL(label_70)  // 2 block: unique 32: unique 34: Padding done
    cmp x28, #0x2
    add x26, x26, x5, LSL #4
    ble label_75
    zip2 v21.4s, v29.4s, v27.4s
    zip2 v20.4s, v28.4s, v26.4s
    zip2 v19.4s, v25.4s, v23.4s
    zip2 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_72
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_71)  // 2 block: unique 32: unique 35: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_71
KAI_ASM_LABEL(label_72)  // 2 block: unique 32: unique 35: Padding done
    cmp x28, #0x3
    add x26, x26, x5, LSL #4
    ble label_75
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_74
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_73)  // 2 block: unique 32: unique 36: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_73
KAI_ASM_LABEL(label_74)  // 2 block: unique 32: unique 36: Padding done
    add x26, x26, x5, LSL #4
KAI_ASM_LABEL(label_75)  // 2 block: unique 32: Done
KAI_ASM_LABEL(label_76)  // 2 block: Oddments skip
    add x0, x0, #0x20
    add x10, x26, #0x20
    b label_128
KAI_ASM_LABEL(label_77)  // 2 skip
    cmp x16, #0x4
    blt label_102
    mov x25, x1
    mov x24, x2
    sub x15, x15, #0x1
    subs x16, x16, #0x4
    csel x23, x15, XZR, EQ
    mov x22, x0
    add x21, x25, x4, LSL #1
    cmp x24, #0x4
    add x20, x25, x4
    add x8, x21, x4
    add x1, x25, x4, LSL #2
    blt label_89
KAI_ASM_LABEL(label_78)  // 1 block: Column loop
    ldr d22, [x25], #0x8
    ldr d21, [x20], #0x8
    ldr d20, [x21], #0x8
    ldr d19, [x8], #0x8
    shll v22.4s, v22.4h, #0x10
    shll v21.4s, v21.4h, #0x10
    shll v20.4s, v20.4h, #0x10
    shll v19.4s, v19.4h, #0x10
    zip1 v18.4s, v22.4s, v20.4s
    zip1 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_81
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_80)  // 1 block: unique 38: unique 39: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_80
KAI_ASM_LABEL(label_81)  // 1 block: unique 38: unique 39: Padding done
    add x22, x22, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_83
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_82)  // 1 block: unique 38: unique 40: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_82
KAI_ASM_LABEL(label_83)  // 1 block: unique 38: unique 40: Padding done
    zip2 v18.4s, v22.4s, v20.4s
    zip2 v17.4s, v21.4s, v19.4s
    add x22, x22, x5, LSL #4
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_85
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_84)  // 1 block: unique 38: unique 41: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_84
KAI_ASM_LABEL(label_85)  // 1 block: unique 38: unique 41: Padding done
    add x22, x22, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_87
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_86)  // 1 block: unique 38: unique 42: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_86
KAI_ASM_LABEL(label_87)  // 1 block: unique 38: unique 42: Padding done
    add x22, x22, x5, LSL #4
    sub x24, x24, #0x4
    cmp x24, #0x4
    bge label_78
KAI_ASM_LABEL(label_89)  // 1 block: Loop skip
    cbz x24, label_101
    tbz x24, #1, label_90
    ldr s22, [x25], #0x4
    ldr s21, [x20], #0x4
    ldr s20, [x21], #0x4
    ldr s19, [x8], #0x4
    tbz x24, #0, label_91
    ld1 { v22.h }[2], [x25]
    ld1 { v21.h }[2], [x20]
    ld1 { v20.h }[2], [x21]
    ld1 { v19.h }[2], [x8]
    b label_91
KAI_ASM_LABEL(label_90)  // 1 block: unique 43: Partial load: partial_0_0
    ldr h22, [x25, #0]
    ldr h21, [x20, #0]
    ldr h20, [x21, #0]
    ldr h19, [x8, #0]
KAI_ASM_LABEL(label_91)  // 1 block: unique 43: Partial load: Done
    shll v22.4s, v22.4h, #0x10
    shll v21.4s, v21.4h, #0x10
    shll v20.4s, v20.4h, #0x10
    shll v19.4s, v19.4h, #0x10
    zip1 v18.4s, v22.4s, v20.4s
    zip1 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_93
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_92)  // 1 block: unique 44: unique 45: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_92
KAI_ASM_LABEL(label_93)  // 1 block: unique 44: unique 45: Padding done
    cmp x24, #0x1
    add x22, x22, x5, LSL #4
    ble label_100
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_95
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_94)  // 1 block: unique 44: unique 46: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_94
KAI_ASM_LABEL(label_95)  // 1 block: unique 44: unique 46: Padding done
    cmp x24, #0x2
    add x22, x22, x5, LSL #4
    ble label_100
    zip2 v18.4s, v22.4s, v20.4s
    zip2 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_97
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_96)  // 1 block: unique 44: unique 47: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_96
KAI_ASM_LABEL(label_97)  // 1 block: unique 44: unique 47: Padding done
    cmp x24, #0x3
    add x22, x22, x5, LSL #4
    ble label_100
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_99
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_98)  // 1 block: unique 44: unique 48: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_98
KAI_ASM_LABEL(label_99)  // 1 block: unique 44: unique 48: Padding done
    add x22, x22, x5, LSL #4
KAI_ASM_LABEL(label_100)  // 1 block: unique 44: Done
KAI_ASM_LABEL(label_101)  // 1 block: Oddments skip
    add x0, x0, #0x10
    add x10, x22, #0x10
    b label_128
KAI_ASM_LABEL(label_102)  // 1 skip
    cbz x16, label_127
    mov x23, x2
    mov x22, x1
    movi v22.16b, #0
    movi v21.16b, #0
    sub x15, x15, #0x1
    movi v20.16b, #0
    mov x21, x0
    cmp x23, #0x4
    add x20, x22, x4
    add x8, x22, x4, LSL #1
    add x1, x22, x4, LSL #2
    blt label_114
KAI_ASM_LABEL(label_103)  // Oddments: Column loop
    ldr d19, [x22], #0x8
    cmp x16, #0x2
    shll v19.4s, v19.4h, #0x10
    blt label_104
    ldr d22, [x20], #0x8
    shll v22.4s, v22.4h, #0x10
    beq label_104
    ldr d21, [x8], #0x8
    shll v21.4s, v21.4h, #0x10
KAI_ASM_LABEL(label_104)  // Oddments: unique 49: Done
    zip1 v18.4s, v19.4s, v21.4s
    zip1 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_106
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_105)  // Oddments: unique 50: unique 51: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_105
KAI_ASM_LABEL(label_106)  // Oddments: unique 50: unique 51: Padding done
    add x21, x21, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_108
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_107)  // Oddments: unique 50: unique 52: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_107
KAI_ASM_LABEL(label_108)  // Oddments: unique 50: unique 52: Padding done
    zip2 v18.4s, v19.4s, v21.4s
    zip2 v17.4s, v22.4s, v20.4s
    add x21, x21, x5, LSL #4
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_110
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_109)  // Oddments: unique 50: unique 53: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_109
KAI_ASM_LABEL(label_110)  // Oddments: unique 50: unique 53: Padding done
    add x21, x21, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_112
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_111)  // Oddments: unique 50: unique 54: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_111
KAI_ASM_LABEL(label_112)  // Oddments: unique 50: unique 54: Padding done
    add x21, x21, x5, LSL #4
    sub x23, x23, #0x4
    cmp x23, #0x4
    bge label_103
KAI_ASM_LABEL(label_114)  // Oddments: Loop skip
    cbz x23, label_126
    mov x6, SP
    cmp x16, #0x2
    csel x20, x20, x6, GE
    csel x8, x8, x6, GT
    tbz x23, #1, label_115
    ldr s19, [x22], #0x4
    ldr s22, [x20], #0x4
    ldr s21, [x8], #0x4
    ldr s20, [x6], #0x4
    tbz x23, #0, label_116
    ld1 { v19.h }[2], [x22]
    ld1 { v22.h }[2], [x20]
    ld1 { v21.h }[2], [x8]
    ld1 { v20.h }[2], [x6]
    b label_116
KAI_ASM_LABEL(label_115)  // Oddments: unique 55: Partial load: partial_0_0
    ldr h19, [x22, #0]
    ldr h22, [x20, #0]
    ldr h21, [x8, #0]
    ldr h20, [x6, #0]
KAI_ASM_LABEL(label_116)  // Oddments: unique 55: Partial load: Done
    shll v19.4s, v19.4h, #0x10
    shll v22.4s, v22.4h, #0x10
    shll v21.4s, v21.4h, #0x10
    shll v20.4s, v20.4h, #0x10
    zip1 v18.4s, v19.4s, v21.4s
    zip1 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_118
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_117)  // Oddments: unique 56: unique 57: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_117
KAI_ASM_LABEL(label_118)  // Oddments: unique 56: unique 57: Padding done
    cmp x23, #0x1
    add x21, x21, x5, LSL #4
    ble label_125
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_120
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_119)  // Oddments: unique 56: unique 58: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_119
KAI_ASM_LABEL(label_120)  // Oddments: unique 56: unique 58: Padding done
    cmp x23, #0x2
    add x21, x21, x5, LSL #4
    ble label_125
    zip2 v18.4s, v19.4s, v21.4s
    zip2 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_122
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_121)  // Oddments: unique 56: unique 59: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_121
KAI_ASM_LABEL(label_122)  // Oddments: unique 56: unique 59: Padding done
    cmp x23, #0x3
    add x21, x21, x5, LSL #4
    ble label_125
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_124
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_123)  // Oddments: unique 56: unique 60: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_123
KAI_ASM_LABEL(label_124)  // Oddments: unique 56: unique 60: Padding done
    add x21, x21, x5, LSL #4
KAI_ASM_LABEL(label_125)  // Oddments: unique 56: Done
KAI_ASM_LABEL(label_126)  // Oddments: Oddments skip
    add x0, x0, #0x10
    add x10, x21, #0x10
    mov x16, #0
KAI_ASM_LABEL(label_127)  // No oddments
KAI_ASM_LABEL(label_128)  // Block loop end
    cbnz x16, label_2
    sub x0, x10, x5, LSL #4
    add x0, x0, x15, LSL #4
    cbnz x3, label_1
    add SP, SP, #0x30
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldr d12, [sp, 104]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave_4_4x1_bf16fp32)

    KAI_ASM_END
