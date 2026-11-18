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

    KAI_ASM_CODE(a64_transpose_interleave_96)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_transpose_interleave_96)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_transpose_interleave_96)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_transpose_interleave_96)
    stp x20, x21, [sp, -32]!
    str x22, [sp, 16]
    mov x6, #0x18
    mov x5, x4
    mov x22, x4
    lsl x5, x5, #0x2
    cmp x22, #0x4
    mul x21, x5, x6
    blt label_11
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x20, x1
    mov x8, x2
    mov x7, x0
    add x6, x20, x3
    add x5, x6, x3
    add x4, x5, x3
    cmp x8, #0x18
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    ldr q7, [x20], #0x10
    ldr q6, [x6], #0x10
    sub x8, x8, #0x18
    ldr q5, [x5], #0x10
    ldr q4, [x4], #0x10
    cmp x8, #0x18
    ldr q3, [x20], #0x10
    ldr q2, [x6], #0x10
    ldr q1, [x5], #0x10
    ldr q0, [x4], #0x10
    ldr q31, [x20], #0x10
    ldr q30, [x6], #0x10
    ldr q29, [x5], #0x10
    ldr q28, [x4], #0x10
    ldr q27, [x20], #0x10
    ldr q26, [x6], #0x10
    ldr q25, [x5], #0x10
    ldr q24, [x4], #0x10
    ldr q23, [x20], #0x10
    ldr q22, [x6], #0x10
    ldr q21, [x5], #0x10
    ldr q20, [x4], #0x10
    ldr q19, [x20], #0x10
    ldr q18, [x6], #0x10
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    str q7, [x7, #0]
    str q3, [x7, #0x10]
    str q31, [x7, #0x20]
    str q27, [x7, #0x30]
    str q23, [x7, #0x40]
    str q19, [x7, #0x50]
    str q6, [x7, #0x60]
    str q2, [x7, #0x70]
    str q30, [x7, #0x80]
    str q26, [x7, #0x90]
    str q22, [x7, #0xa0]
    str q18, [x7, #0xb0]
    str q5, [x7, #0xc0]
    str q1, [x7, #0xd0]
    str q29, [x7, #0xe0]
    str q25, [x7, #0xf0]
    str q21, [x7, #0x100]
    str q17, [x7, #0x110]
    str q4, [x7, #0x120]
    str q0, [x7, #0x130]
    str q28, [x7, #0x140]
    str q24, [x7, #0x150]
    str q20, [x7, #0x160]
    str q16, [x7, #0x170]
    add x7, x7, x21
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop skip
    cbz x8, label_10
    cmp x8, #0x10
    movi v16.4s, #0
    str q16, [x7, #0]
    str q16, [x7, #0x10]
    str q16, [x7, #0x20]
    str q16, [x7, #0x30]
    str q16, [x7, #0x40]
    str q16, [x7, #0x50]
    str q16, [x7, #0x60]
    str q16, [x7, #0x70]
    str q16, [x7, #0x80]
    str q16, [x7, #0x90]
    str q16, [x7, #0xa0]
    str q16, [x7, #0xb0]
    str q16, [x7, #0xc0]
    str q16, [x7, #0xd0]
    str q16, [x7, #0xe0]
    str q16, [x7, #0xf0]
    str q16, [x7, #0x100]
    str q16, [x7, #0x110]
    str q16, [x7, #0x120]
    str q16, [x7, #0x130]
    str q16, [x7, #0x140]
    str q16, [x7, #0x150]
    str q16, [x7, #0x160]
    str q16, [x7, #0x170]
    blt label_5
KAI_ASM_LABEL(label_4)  // Main row loop: width 16 loop: loop
    ldr q31, [x20], #0x10
    ldr q30, [x6], #0x10
    sub x8, x8, #0x10
    ldr q29, [x5], #0x10
    ldr q28, [x4], #0x10
    cmp x8, #0x10
    ldr q27, [x20], #0x10
    ldr q26, [x6], #0x10
    ldr q25, [x5], #0x10
    ldr q24, [x4], #0x10
    ldr q23, [x20], #0x10
    ldr q22, [x6], #0x10
    ldr q21, [x5], #0x10
    ldr q20, [x4], #0x10
    ldr q19, [x20], #0x10
    ldr q18, [x6], #0x10
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    str q31, [x7, #0]
    str q27, [x7, #0x10]
    str q23, [x7, #0x20]
    str q19, [x7, #0x30]
    str q30, [x7, #0x60]
    str q26, [x7, #0x70]
    str q22, [x7, #0x80]
    str q18, [x7, #0x90]
    str q29, [x7, #0xc0]
    str q25, [x7, #0xd0]
    str q21, [x7, #0xe0]
    str q17, [x7, #0xf0]
    str q28, [x7, #0x120]
    str q24, [x7, #0x130]
    str q20, [x7, #0x140]
    str q16, [x7, #0x150]
    add x7, x7, #0x40
    bge label_4
KAI_ASM_LABEL(label_5)  // Main row loop: width 16 loop: skip
    cmp x8, #0x4
    blt label_7
KAI_ASM_LABEL(label_6)  // Main row loop: width 4 loop: loop
    ldr q19, [x20], #0x10
    ldr q18, [x6], #0x10
    sub x8, x8, #0x4
    ldr q17, [x5], #0x10
    ldr q16, [x4], #0x10
    cmp x8, #0x4
    str q19, [x7, #0]
    str q18, [x7, #0x60]
    str q17, [x7, #0xc0]
    str q16, [x7, #0x120]
    add x7, x7, #0x10
    bge label_6
KAI_ASM_LABEL(label_7)  // Main row loop: width 4 loop: skip
    cmp x8, #0x1
    blt label_9
KAI_ASM_LABEL(label_8)  // Main row loop: width 1 loop: loop
    ldr s19, [x20], #0x4
    ldr s18, [x6], #0x4
    sub x8, x8, #0x1
    ldr s17, [x5], #0x4
    ldr s16, [x4], #0x4
    cmp x8, #0x1
    str s19, [x7, #0]
    str s18, [x7, #0x60]
    str s17, [x7, #0xc0]
    str s16, [x7, #0x120]
    add x7, x7, #0x4
    bge label_8
KAI_ASM_LABEL(label_9)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_10)  // Main row loop: odd col skip
    sub x22, x22, #0x4
    add x0, x0, #0x180
    cmp x22, #0x4
    bge label_1
    cbz x22, label_22
KAI_ASM_LABEL(label_11)  // Main loop skip
KAI_ASM_LABEL(label_12)  // Tail row loop: Head
    mov x4, x2
    mov x20, x1
    mov x7, x0
    cmp x4, #0x18
    add x1, x20, x3
    blt label_14
KAI_ASM_LABEL(label_13)  // Tail row loop: Column loop
    ldr q21, [x20], #0x10
    sub x4, x4, #0x18
    ldr q20, [x20], #0x10
    ldr q19, [x20], #0x10
    cmp x4, #0x18
    ldr q18, [x20], #0x10
    ldr q17, [x20], #0x10
    ldr q16, [x20], #0x10
    str q21, [x7, #0]
    str q20, [x7, #0x10]
    str q19, [x7, #0x20]
    str q18, [x7, #0x30]
    str q17, [x7, #0x40]
    str q16, [x7, #0x50]
    add x7, x7, x21
    bge label_13
KAI_ASM_LABEL(label_14)  // Tail row loop: Column loop skip
    cbz x4, label_21
    cmp x4, #0x10
    movi v16.4s, #0
    str q16, [x7, #0]
    str q16, [x7, #0x10]
    str q16, [x7, #0x20]
    str q16, [x7, #0x30]
    str q16, [x7, #0x40]
    str q16, [x7, #0x50]
    blt label_16
KAI_ASM_LABEL(label_15)  // Tail row loop: width 16 loop: loop
    ldr q19, [x20], #0x10
    sub x4, x4, #0x10
    ldr q18, [x20], #0x10
    ldr q17, [x20], #0x10
    cmp x4, #0x10
    ldr q16, [x20], #0x10
    str q19, [x7, #0]
    str q18, [x7, #0x10]
    str q17, [x7, #0x20]
    str q16, [x7, #0x30]
    add x7, x7, #0x40
    bge label_15
KAI_ASM_LABEL(label_16)  // Tail row loop: width 16 loop: skip
    cmp x4, #0x4
    blt label_18
KAI_ASM_LABEL(label_17)  // Tail row loop: width 4 loop: loop
    ldr q16, [x20], #0x10
    sub x4, x4, #0x4
    cmp x4, #0x4
    str q16, [x7, #0]
    add x7, x7, #0x10
    bge label_17
KAI_ASM_LABEL(label_18)  // Tail row loop: width 4 loop: skip
    cmp x4, #0x1
    blt label_20
KAI_ASM_LABEL(label_19)  // Tail row loop: width 1 loop: loop
    ldr s16, [x20], #0x4
    sub x4, x4, #0x1
    cmp x4, #0x1
    str s16, [x7, #0]
    add x7, x7, #0x4
    bge label_19
KAI_ASM_LABEL(label_20)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_21)  // Tail row loop: odd col skip
    sub x22, x22, #0x1
    add x0, x0, #0x60
    cmp x22, #0x1
    bge label_12
KAI_ASM_LABEL(label_22)  // Done
    ldr x22, [sp, 16]
    ldp x20, x21, [sp], 32
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_transpose_interleave_96)

    KAI_ASM_END
