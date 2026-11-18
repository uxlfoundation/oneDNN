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

    KAI_ASM_CODE(a64_transpose_interleave_4_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_4_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_4_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_4_2x2)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    mov x6, #0x4
    movi v16.8h, #0
    add SP, SP, #-0x10
    str q16, [SP, #0]
    mov x27, SP
    mov x5, x4
    mov x26, x4
    add x5, x5, #0x1
    bfm x5, XZR, #0, #0  // bfc x5, #0, #0x1
    cmp x26, #0x8
    lsl x5, x5, #0x1
    mul x25, x5, x6
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x24, x1
    mov x23, x2
    mov x22, x0
    add x21, x24, x3
    add x20, x21, x3
    add x8, x20, x3
    add x7, x8, x3
    cmp x23, #0x4
    add x6, x7, x3
    add x5, x6, x3
    add x4, x5, x3
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    ldr d23, [x24], #0x8
    ldr d19, [x21], #0x8
    sub x23, x23, #0x4
    ldr d22, [x20], #0x8
    ldr d18, [x8], #0x8
    cmp x23, #0x4
    ldr d21, [x7], #0x8
    ldr d17, [x6], #0x8
    ldr d20, [x5], #0x8
    ldr d16, [x4], #0x8
    zip1 v19.8h, v23.8h, v19.8h
    zip1 v18.8h, v22.8h, v18.8h
    zip1 v17.8h, v21.8h, v17.8h
    zip1 v16.8h, v20.8h, v16.8h
    str q19, [x22, #0]
    str q18, [x22, #0x10]
    str q17, [x22, #0x20]
    str q16, [x22, #0x30]
    add x22, x22, x25
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop skip
    cbz x23, label_5
    movi v16.8h, #0
    str q16, [x22, #0]
    str q16, [x22, #0x10]
    str q16, [x22, #0x20]
    str q16, [x22, #0x30]
KAI_ASM_LABEL(label_4)  // Main row loop: width 1 loop: loop
    ldr h23, [x24], #0x2
    ldr h18, [x21], #0x2
    sub x23, x23, #0x1
    ldr h22, [x20], #0x2
    ldr h16, [x8], #0x2
    cmp x23, #0x1
    ldr h21, [x7], #0x2
    ldr h17, [x6], #0x2
    ldr h20, [x5], #0x2
    ldr h19, [x4], #0x2
    zip1 v18.8h, v23.8h, v18.8h
    zip1 v16.8h, v22.8h, v16.8h
    zip1 v17.8h, v21.8h, v17.8h
    str s18, [x22, #0]
    str s16, [x22, #0x10]
    zip1 v16.8h, v20.8h, v19.8h
    str s17, [x22, #0x20]
    str s16, [x22, #0x30]
    add x22, x22, #0x4
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: odd col skip
    sub x26, x26, #0x8
    add x0, x0, #0x40
    cmp x26, #0x8
    bge label_1
    cbz x26, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x24, x1
    mov x4, #0x8
    mov x5, x2
    cmp x26, #0x1
    mov x22, x0
    add x21, x24, x3
    add x1, x21, x3
    csel x21, x21, x27, GT
    csel x4, x4, XZR, GT
    cmp x5, #0x4
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Column loop
    ldr d17, [x24], #0x8
    ldr d16, [x21, #0]
    sub x5, x5, #0x4
    add x21, x21, x4
    cmp x5, #0x4
    zip1 v16.8h, v17.8h, v16.8h
    str q16, [x22, #0]
    add x22, x22, x25
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Column loop skip
    cbz x5, label_11
    movi v16.8h, #0
    str q16, [x22, #0]
KAI_ASM_LABEL(label_10)  // Tail row loop: width 1 loop: loop
    ldr h17, [x24], #0x2
    ldr h16, [x21], #0x2
    sub x5, x5, #0x1
    cmp x5, #0x1
    zip1 v16.8h, v17.8h, v16.8h
    str s16, [x22, #0]
    add x22, x22, #0x4
    bge label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: odd col skip
    sub x26, x26, #0x2
    add x0, x0, #0x10
    cmp x26, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    add SP, SP, #0x10
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_4_2x2)

    KAI_ASM_END
