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

    KAI_ASM_CODE(a64_interleave_2_2x4_fp32bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave_2_2x4_fp32bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave_2_2x4_fp32bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave_2_2x4_fp32bf16)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    sub SP, SP, #0x30
    movi v0.4s, #0
    str q0, [SP, #0]
KAI_ASM_LABEL(label_1)  // Row loop
    lsl x17, x5, #0x1
    mov x16, x5
    cmp x17, x3
    csel x17, x17, x3, LT
    sub x3, x3, x17
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x17, #0x10
    blt label_15
    mov x14, x1
    mov x13, x2
    str x3, [SP, #0x10]
    sub x16, x16, #0x8
    subs x17, x17, #0x10
    csel x15, x16, XZR, EQ
    str x17, [SP, #0x18]
    mov x12, x0
    add x17, x14, x4, LSL #3
    add x11, x14, x4, LSL #2
    str x16, [SP, #0x20]
    add x10, x17, x4, LSL #2
    add x9, x14, x4, LSL #1
    add x28, x11, x4, LSL #1
    add x27, x17, x4, LSL #1
    add x26, x10, x4, LSL #1
    cmp x13, #0x4
    add x25, x14, x4
    add x24, x9, x4
    add x23, x11, x4
    add x22, x28, x4
    add x21, x17, x4
    add x20, x27, x4
    add x8, x10, x4
    add x7, x26, x4
    add x1, x14, x4, LSL #4
    blt label_8
KAI_ASM_LABEL(label_3)  // 8 block: Column loop
    ldr q23, [x14], #0x10
    ldr q31, [x25], #0x10
    ldr q22, [x9], #0x10
    ldr q30, [x24], #0x10
    ldr q21, [x11], #0x10
    ldr q29, [x23], #0x10
    ldr q20, [x28], #0x10
    ldr q28, [x22], #0x10
    ldr q19, [x17], #0x10
    ldr q27, [x21], #0x10
    ldr q18, [x27], #0x10
    ldr q26, [x20], #0x10
    ldr q17, [x10], #0x10
    ldr q25, [x8], #0x10
    ldr q16, [x26], #0x10
    ldr q24, [x7], #0x10
    KAI_ASM_INST(0x0ea16af7)  // bfcvtn v23.4h, v23.4s
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16bf7)  // bfcvtn2 v23.8h, v31.4s
    KAI_ASM_INST(0x4ea16bd6)  // bfcvtn2 v22.8h, v30.4s
    KAI_ASM_INST(0x4ea16bb5)  // bfcvtn2 v21.8h, v29.4s
    KAI_ASM_INST(0x4ea16b94)  // bfcvtn2 v20.8h, v28.4s
    KAI_ASM_INST(0x4ea16b73)  // bfcvtn2 v19.8h, v27.4s
    KAI_ASM_INST(0x4ea16b52)  // bfcvtn2 v18.8h, v26.4s
    KAI_ASM_INST(0x4ea16b31)  // bfcvtn2 v17.8h, v25.4s
    KAI_ASM_INST(0x4ea16b10)  // bfcvtn2 v16.8h, v24.4s
    str q23, [x12, #0]
    str q22, [x12, #0x10]
    str q21, [x12, #0x20]
    str q20, [x12, #0x30]
    str q19, [x12, #0x40]
    str q18, [x12, #0x50]
    str q17, [x12, #0x60]
    str q16, [x12, #0x70]
    cbz x15, label_6
    add x6, x12, #0x80
    mov x3, x15
KAI_ASM_LABEL(label_5)  // 8 block: unique 2: unique 3: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_5
KAI_ASM_LABEL(label_6)  // 8 block: unique 2: unique 3: Padding done
    add x12, x12, x5, LSL #4
    sub x13, x13, #0x4
    cmp x13, #0x4
    bge label_3
KAI_ASM_LABEL(label_8)  // 8 block: Loop skip
    cbz x13, label_14
    tbz x13, #1, label_9
    ldr d23, [x14], #0x8
    ldr d31, [x25], #0x8
    ldr d22, [x9], #0x8
    ldr d30, [x24], #0x8
    ldr d21, [x11], #0x8
    ldr d29, [x23], #0x8
    ldr d20, [x28], #0x8
    ldr d28, [x22], #0x8
    ldr d19, [x17], #0x8
    ldr d27, [x21], #0x8
    ldr d18, [x27], #0x8
    ldr d26, [x20], #0x8
    ldr d17, [x10], #0x8
    ldr d25, [x8], #0x8
    ldr d16, [x26], #0x8
    ldr d24, [x7], #0x8
    tbz x13, #0, label_10
    ld1 { v23.s }[2], [x14]
    ld1 { v31.s }[2], [x25]
    ld1 { v22.s }[2], [x9]
    ld1 { v30.s }[2], [x24]
    ld1 { v21.s }[2], [x11]
    ld1 { v29.s }[2], [x23]
    ld1 { v20.s }[2], [x28]
    ld1 { v28.s }[2], [x22]
    ld1 { v19.s }[2], [x17]
    ld1 { v27.s }[2], [x21]
    ld1 { v18.s }[2], [x27]
    ld1 { v26.s }[2], [x20]
    ld1 { v17.s }[2], [x10]
    ld1 { v25.s }[2], [x8]
    ld1 { v16.s }[2], [x26]
    ld1 { v24.s }[2], [x7]
    b label_10
KAI_ASM_LABEL(label_9)  // 8 block: unique 4: Partial load: partial_0_0
    ldr s23, [x14, #0]
    ldr s31, [x25, #0]
    ldr s22, [x9, #0]
    ldr s30, [x24, #0]
    ldr s21, [x11, #0]
    ldr s29, [x23, #0]
    ldr s20, [x28, #0]
    ldr s28, [x22, #0]
    ldr s19, [x17, #0]
    ldr s27, [x21, #0]
    ldr s18, [x27, #0]
    ldr s26, [x20, #0]
    ldr s17, [x10, #0]
    ldr s25, [x8, #0]
    ldr s16, [x26, #0]
    ldr s24, [x7, #0]
KAI_ASM_LABEL(label_10)  // 8 block: unique 4: Partial load: Done
    KAI_ASM_INST(0x0ea16af7)  // bfcvtn v23.4h, v23.4s
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16bf7)  // bfcvtn2 v23.8h, v31.4s
    KAI_ASM_INST(0x4ea16bd6)  // bfcvtn2 v22.8h, v30.4s
    KAI_ASM_INST(0x4ea16bb5)  // bfcvtn2 v21.8h, v29.4s
    KAI_ASM_INST(0x4ea16b94)  // bfcvtn2 v20.8h, v28.4s
    KAI_ASM_INST(0x4ea16b73)  // bfcvtn2 v19.8h, v27.4s
    KAI_ASM_INST(0x4ea16b52)  // bfcvtn2 v18.8h, v26.4s
    KAI_ASM_INST(0x4ea16b31)  // bfcvtn2 v17.8h, v25.4s
    KAI_ASM_INST(0x4ea16b10)  // bfcvtn2 v16.8h, v24.4s
    str q23, [x12, #0]
    str q22, [x12, #0x10]
    str q21, [x12, #0x20]
    str q20, [x12, #0x30]
    str q19, [x12, #0x40]
    str q18, [x12, #0x50]
    str q17, [x12, #0x60]
    str q16, [x12, #0x70]
    cbz x15, label_12
    add x6, x12, #0x80
    mov x3, x15
KAI_ASM_LABEL(label_11)  // 8 block: unique 5: unique 6: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_11
KAI_ASM_LABEL(label_12)  // 8 block: unique 5: unique 6: Padding done
    add x12, x12, x5, LSL #4
KAI_ASM_LABEL(label_14)  // 8 block: Oddments skip
    add x0, x0, #0x80
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x12, #0x80
    ldr x16, [SP, #0x20]
    b label_120
KAI_ASM_LABEL(label_15)  // 8 skip
    cmp x17, #0xe
    blt label_28
    mov x14, x1
    mov x13, x2
    str x3, [SP, #0x10]
    sub x16, x16, #0x7
    subs x17, x17, #0xe
    csel x12, x16, XZR, EQ
    str x17, [SP, #0x18]
    mov x11, x0
    add x10, x14, x4, LSL #2
    add x9, x14, x4, LSL #3
    str x16, [SP, #0x20]
    add x28, x14, x4, LSL #1
    add x27, x10, x4, LSL #1
    add x26, x9, x4, LSL #1
    add x25, x9, x4, LSL #2
    cmp x13, #0x4
    add x24, x14, x4
    add x23, x28, x4
    add x22, x10, x4
    add x21, x27, x4
    add x20, x9, x4
    add x8, x26, x4
    add x7, x25, x4
    add x1, x25, x4, LSL #1
    blt label_21
KAI_ASM_LABEL(label_16)  // 7 block: Column loop
    ldr q22, [x14], #0x10
    ldr q29, [x24], #0x10
    ldr q21, [x28], #0x10
    ldr q28, [x23], #0x10
    ldr q20, [x10], #0x10
    ldr q27, [x22], #0x10
    ldr q19, [x27], #0x10
    ldr q26, [x21], #0x10
    ldr q18, [x9], #0x10
    ldr q25, [x20], #0x10
    ldr q17, [x26], #0x10
    ldr q24, [x8], #0x10
    ldr q16, [x25], #0x10
    ldr q23, [x7], #0x10
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16bb6)  // bfcvtn2 v22.8h, v29.4s
    KAI_ASM_INST(0x4ea16b95)  // bfcvtn2 v21.8h, v28.4s
    KAI_ASM_INST(0x4ea16b74)  // bfcvtn2 v20.8h, v27.4s
    KAI_ASM_INST(0x4ea16b53)  // bfcvtn2 v19.8h, v26.4s
    KAI_ASM_INST(0x4ea16b32)  // bfcvtn2 v18.8h, v25.4s
    KAI_ASM_INST(0x4ea16b11)  // bfcvtn2 v17.8h, v24.4s
    KAI_ASM_INST(0x4ea16af0)  // bfcvtn2 v16.8h, v23.4s
    str q22, [x11, #0]
    str q21, [x11, #0x10]
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_19
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_18)  // 7 block: unique 8: unique 9: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_18
KAI_ASM_LABEL(label_19)  // 7 block: unique 8: unique 9: Padding done
    add x11, x11, x5, LSL #4
    sub x13, x13, #0x4
    cmp x13, #0x4
    bge label_16
KAI_ASM_LABEL(label_21)  // 7 block: Loop skip
    cbz x13, label_27
    tbz x13, #1, label_22
    ldr d22, [x14], #0x8
    ldr d29, [x24], #0x8
    ldr d21, [x28], #0x8
    ldr d28, [x23], #0x8
    ldr d20, [x10], #0x8
    ldr d27, [x22], #0x8
    ldr d19, [x27], #0x8
    ldr d26, [x21], #0x8
    ldr d18, [x9], #0x8
    ldr d25, [x20], #0x8
    ldr d17, [x26], #0x8
    ldr d24, [x8], #0x8
    ldr d16, [x25], #0x8
    ldr d23, [x7], #0x8
    tbz x13, #0, label_23
    ld1 { v22.s }[2], [x14]
    ld1 { v29.s }[2], [x24]
    ld1 { v21.s }[2], [x28]
    ld1 { v28.s }[2], [x23]
    ld1 { v20.s }[2], [x10]
    ld1 { v27.s }[2], [x22]
    ld1 { v19.s }[2], [x27]
    ld1 { v26.s }[2], [x21]
    ld1 { v18.s }[2], [x9]
    ld1 { v25.s }[2], [x20]
    ld1 { v17.s }[2], [x26]
    ld1 { v24.s }[2], [x8]
    ld1 { v16.s }[2], [x25]
    ld1 { v23.s }[2], [x7]
    b label_23
KAI_ASM_LABEL(label_22)  // 7 block: unique 10: Partial load: partial_0_0
    ldr s22, [x14, #0]
    ldr s29, [x24, #0]
    ldr s21, [x28, #0]
    ldr s28, [x23, #0]
    ldr s20, [x10, #0]
    ldr s27, [x22, #0]
    ldr s19, [x27, #0]
    ldr s26, [x21, #0]
    ldr s18, [x9, #0]
    ldr s25, [x20, #0]
    ldr s17, [x26, #0]
    ldr s24, [x8, #0]
    ldr s16, [x25, #0]
    ldr s23, [x7, #0]
KAI_ASM_LABEL(label_23)  // 7 block: unique 10: Partial load: Done
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16bb6)  // bfcvtn2 v22.8h, v29.4s
    KAI_ASM_INST(0x4ea16b95)  // bfcvtn2 v21.8h, v28.4s
    KAI_ASM_INST(0x4ea16b74)  // bfcvtn2 v20.8h, v27.4s
    KAI_ASM_INST(0x4ea16b53)  // bfcvtn2 v19.8h, v26.4s
    KAI_ASM_INST(0x4ea16b32)  // bfcvtn2 v18.8h, v25.4s
    KAI_ASM_INST(0x4ea16b11)  // bfcvtn2 v17.8h, v24.4s
    KAI_ASM_INST(0x4ea16af0)  // bfcvtn2 v16.8h, v23.4s
    str q22, [x11, #0]
    str q21, [x11, #0x10]
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_25
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_24)  // 7 block: unique 11: unique 12: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_24
KAI_ASM_LABEL(label_25)  // 7 block: unique 11: unique 12: Padding done
    add x11, x11, x5, LSL #4
KAI_ASM_LABEL(label_27)  // 7 block: Oddments skip
    add x0, x0, #0x70
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x11, #0x70
    ldr x16, [SP, #0x20]
    b label_120
KAI_ASM_LABEL(label_28)  // 7 skip
    cmp x17, #0xc
    blt label_41
    mov x12, x1
    mov x11, x2
    str x3, [SP, #0x10]
    sub x16, x16, #0x6
    subs x17, x17, #0xc
    csel x10, x16, XZR, EQ
    str x17, [SP, #0x18]
    mov x9, x0
    add x28, x12, x4, LSL #2
    add x27, x12, x4, LSL #3
    str x16, [SP, #0x20]
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
    blt label_34
KAI_ASM_LABEL(label_29)  // 6 block: Column loop
    ldr q21, [x12], #0x10
    ldr q27, [x23], #0x10
    ldr q20, [x26], #0x10
    ldr q26, [x22], #0x10
    ldr q19, [x28], #0x10
    ldr q25, [x21], #0x10
    ldr q18, [x25], #0x10
    ldr q24, [x20], #0x10
    ldr q17, [x27], #0x10
    ldr q23, [x8], #0x10
    ldr q16, [x24], #0x10
    ldr q22, [x7], #0x10
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16b75)  // bfcvtn2 v21.8h, v27.4s
    KAI_ASM_INST(0x4ea16b54)  // bfcvtn2 v20.8h, v26.4s
    KAI_ASM_INST(0x4ea16b33)  // bfcvtn2 v19.8h, v25.4s
    KAI_ASM_INST(0x4ea16b12)  // bfcvtn2 v18.8h, v24.4s
    KAI_ASM_INST(0x4ea16af1)  // bfcvtn2 v17.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad0)  // bfcvtn2 v16.8h, v22.4s
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_32
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_31)  // 6 block: unique 14: unique 15: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_31
KAI_ASM_LABEL(label_32)  // 6 block: unique 14: unique 15: Padding done
    add x9, x9, x5, LSL #4
    sub x11, x11, #0x4
    cmp x11, #0x4
    bge label_29
KAI_ASM_LABEL(label_34)  // 6 block: Loop skip
    cbz x11, label_40
    tbz x11, #1, label_35
    ldr d21, [x12], #0x8
    ldr d27, [x23], #0x8
    ldr d20, [x26], #0x8
    ldr d26, [x22], #0x8
    ldr d19, [x28], #0x8
    ldr d25, [x21], #0x8
    ldr d18, [x25], #0x8
    ldr d24, [x20], #0x8
    ldr d17, [x27], #0x8
    ldr d23, [x8], #0x8
    ldr d16, [x24], #0x8
    ldr d22, [x7], #0x8
    tbz x11, #0, label_36
    ld1 { v21.s }[2], [x12]
    ld1 { v27.s }[2], [x23]
    ld1 { v20.s }[2], [x26]
    ld1 { v26.s }[2], [x22]
    ld1 { v19.s }[2], [x28]
    ld1 { v25.s }[2], [x21]
    ld1 { v18.s }[2], [x25]
    ld1 { v24.s }[2], [x20]
    ld1 { v17.s }[2], [x27]
    ld1 { v23.s }[2], [x8]
    ld1 { v16.s }[2], [x24]
    ld1 { v22.s }[2], [x7]
    b label_36
KAI_ASM_LABEL(label_35)  // 6 block: unique 16: Partial load: partial_0_0
    ldr s21, [x12, #0]
    ldr s27, [x23, #0]
    ldr s20, [x26, #0]
    ldr s26, [x22, #0]
    ldr s19, [x28, #0]
    ldr s25, [x21, #0]
    ldr s18, [x25, #0]
    ldr s24, [x20, #0]
    ldr s17, [x27, #0]
    ldr s23, [x8, #0]
    ldr s16, [x24, #0]
    ldr s22, [x7, #0]
KAI_ASM_LABEL(label_36)  // 6 block: unique 16: Partial load: Done
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16b75)  // bfcvtn2 v21.8h, v27.4s
    KAI_ASM_INST(0x4ea16b54)  // bfcvtn2 v20.8h, v26.4s
    KAI_ASM_INST(0x4ea16b33)  // bfcvtn2 v19.8h, v25.4s
    KAI_ASM_INST(0x4ea16b12)  // bfcvtn2 v18.8h, v24.4s
    KAI_ASM_INST(0x4ea16af1)  // bfcvtn2 v17.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad0)  // bfcvtn2 v16.8h, v22.4s
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_38
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_37)  // 6 block: unique 17: unique 18: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_37
KAI_ASM_LABEL(label_38)  // 6 block: unique 17: unique 18: Padding done
    add x9, x9, x5, LSL #4
KAI_ASM_LABEL(label_40)  // 6 block: Oddments skip
    add x0, x0, #0x60
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x9, #0x60
    ldr x16, [SP, #0x20]
    b label_120
KAI_ASM_LABEL(label_41)  // 6 skip
    cmp x17, #0xa
    blt label_54
    mov x10, x1
    mov x9, x2
    str x3, [SP, #0x10]
    sub x16, x16, #0x5
    subs x17, x17, #0xa
    csel x28, x16, XZR, EQ
    str x17, [SP, #0x18]
    mov x27, x0
    add x26, x10, x4, LSL #2
    add x25, x10, x4, LSL #1
    str x16, [SP, #0x20]
    add x24, x26, x4, LSL #1
    add x23, x10, x4, LSL #3
    cmp x9, #0x4
    add x22, x10, x4
    add x21, x25, x4
    add x20, x26, x4
    add x8, x24, x4
    add x7, x23, x4
    add x1, x23, x4, LSL #1
    blt label_47
KAI_ASM_LABEL(label_42)  // 5 block: Column loop
    ldr q20, [x10], #0x10
    ldr q25, [x22], #0x10
    ldr q19, [x25], #0x10
    ldr q24, [x21], #0x10
    ldr q18, [x26], #0x10
    ldr q23, [x20], #0x10
    ldr q17, [x24], #0x10
    ldr q22, [x8], #0x10
    ldr q16, [x23], #0x10
    ldr q21, [x7], #0x10
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16b34)  // bfcvtn2 v20.8h, v25.4s
    KAI_ASM_INST(0x4ea16b13)  // bfcvtn2 v19.8h, v24.4s
    KAI_ASM_INST(0x4ea16af2)  // bfcvtn2 v18.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad1)  // bfcvtn2 v17.8h, v22.4s
    KAI_ASM_INST(0x4ea16ab0)  // bfcvtn2 v16.8h, v21.4s
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_45
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_44)  // 5 block: unique 20: unique 21: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_44
KAI_ASM_LABEL(label_45)  // 5 block: unique 20: unique 21: Padding done
    add x27, x27, x5, LSL #4
    sub x9, x9, #0x4
    cmp x9, #0x4
    bge label_42
KAI_ASM_LABEL(label_47)  // 5 block: Loop skip
    cbz x9, label_53
    tbz x9, #1, label_48
    ldr d20, [x10], #0x8
    ldr d25, [x22], #0x8
    ldr d19, [x25], #0x8
    ldr d24, [x21], #0x8
    ldr d18, [x26], #0x8
    ldr d23, [x20], #0x8
    ldr d17, [x24], #0x8
    ldr d22, [x8], #0x8
    ldr d16, [x23], #0x8
    ldr d21, [x7], #0x8
    tbz x9, #0, label_49
    ld1 { v20.s }[2], [x10]
    ld1 { v25.s }[2], [x22]
    ld1 { v19.s }[2], [x25]
    ld1 { v24.s }[2], [x21]
    ld1 { v18.s }[2], [x26]
    ld1 { v23.s }[2], [x20]
    ld1 { v17.s }[2], [x24]
    ld1 { v22.s }[2], [x8]
    ld1 { v16.s }[2], [x23]
    ld1 { v21.s }[2], [x7]
    b label_49
KAI_ASM_LABEL(label_48)  // 5 block: unique 22: Partial load: partial_0_0
    ldr s20, [x10, #0]
    ldr s25, [x22, #0]
    ldr s19, [x25, #0]
    ldr s24, [x21, #0]
    ldr s18, [x26, #0]
    ldr s23, [x20, #0]
    ldr s17, [x24, #0]
    ldr s22, [x8, #0]
    ldr s16, [x23, #0]
    ldr s21, [x7, #0]
KAI_ASM_LABEL(label_49)  // 5 block: unique 22: Partial load: Done
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16b34)  // bfcvtn2 v20.8h, v25.4s
    KAI_ASM_INST(0x4ea16b13)  // bfcvtn2 v19.8h, v24.4s
    KAI_ASM_INST(0x4ea16af2)  // bfcvtn2 v18.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad1)  // bfcvtn2 v17.8h, v22.4s
    KAI_ASM_INST(0x4ea16ab0)  // bfcvtn2 v16.8h, v21.4s
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_51
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_50)  // 5 block: unique 23: unique 24: Padding loop
    subs x3, x3, #0x1
    str q0, [x6], #0x10
    bne label_50
KAI_ASM_LABEL(label_51)  // 5 block: unique 23: unique 24: Padding done
    add x27, x27, x5, LSL #4
KAI_ASM_LABEL(label_53)  // 5 block: Oddments skip
    add x0, x0, #0x50
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x27, #0x50
    ldr x16, [SP, #0x20]
    b label_120
KAI_ASM_LABEL(label_54)  // 5 skip
    cmp x17, #0x8
    blt label_67
    mov x9, x1
    mov x28, x2
    sub x16, x16, #0x4
    subs x17, x17, #0x8
    csel x27, x16, XZR, EQ
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
    blt label_60
KAI_ASM_LABEL(label_55)  // 4 block: Column loop
    ldr q19, [x9], #0x10
    ldr q23, [x22], #0x10
    ldr q18, [x24], #0x10
    ldr q22, [x21], #0x10
    ldr q17, [x25], #0x10
    ldr q21, [x20], #0x10
    ldr q16, [x23], #0x10
    ldr q20, [x8], #0x10
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16af3)  // bfcvtn2 v19.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad2)  // bfcvtn2 v18.8h, v22.4s
    KAI_ASM_INST(0x4ea16ab1)  // bfcvtn2 v17.8h, v21.4s
    KAI_ASM_INST(0x4ea16a90)  // bfcvtn2 v16.8h, v20.4s
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_58
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_57)  // 4 block: unique 26: unique 27: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_57
KAI_ASM_LABEL(label_58)  // 4 block: unique 26: unique 27: Padding done
    add x26, x26, x5, LSL #4
    sub x28, x28, #0x4
    cmp x28, #0x4
    bge label_55
KAI_ASM_LABEL(label_60)  // 4 block: Loop skip
    cbz x28, label_66
    tbz x28, #1, label_61
    ldr d19, [x9], #0x8
    ldr d23, [x22], #0x8
    ldr d18, [x24], #0x8
    ldr d22, [x21], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x20], #0x8
    ldr d16, [x23], #0x8
    ldr d20, [x8], #0x8
    tbz x28, #0, label_62
    ld1 { v19.s }[2], [x9]
    ld1 { v23.s }[2], [x22]
    ld1 { v18.s }[2], [x24]
    ld1 { v22.s }[2], [x21]
    ld1 { v17.s }[2], [x25]
    ld1 { v21.s }[2], [x20]
    ld1 { v16.s }[2], [x23]
    ld1 { v20.s }[2], [x8]
    b label_62
KAI_ASM_LABEL(label_61)  // 4 block: unique 28: Partial load: partial_0_0
    ldr s19, [x9, #0]
    ldr s23, [x22, #0]
    ldr s18, [x24, #0]
    ldr s22, [x21, #0]
    ldr s17, [x25, #0]
    ldr s21, [x20, #0]
    ldr s16, [x23, #0]
    ldr s20, [x8, #0]
KAI_ASM_LABEL(label_62)  // 4 block: unique 28: Partial load: Done
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16af3)  // bfcvtn2 v19.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad2)  // bfcvtn2 v18.8h, v22.4s
    KAI_ASM_INST(0x4ea16ab1)  // bfcvtn2 v17.8h, v21.4s
    KAI_ASM_INST(0x4ea16a90)  // bfcvtn2 v16.8h, v20.4s
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_64
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_63)  // 4 block: unique 29: unique 30: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_63
KAI_ASM_LABEL(label_64)  // 4 block: unique 29: unique 30: Padding done
    add x26, x26, x5, LSL #4
KAI_ASM_LABEL(label_66)  // 4 block: Oddments skip
    add x0, x0, #0x40
    add x8, x26, #0x40
    b label_120
KAI_ASM_LABEL(label_67)  // 4 skip
    cmp x17, #0x6
    blt label_80
    mov x27, x1
    mov x26, x2
    sub x16, x16, #0x3
    subs x17, x17, #0x6
    csel x25, x16, XZR, EQ
    mov x24, x0
    add x23, x27, x4, LSL #1
    add x22, x27, x4, LSL #2
    cmp x26, #0x4
    add x21, x27, x4
    add x20, x23, x4
    add x8, x22, x4
    add x1, x22, x4, LSL #1
    blt label_73
KAI_ASM_LABEL(label_68)  // 3 block: Column loop
    ldr q18, [x27], #0x10
    ldr q21, [x21], #0x10
    ldr q17, [x23], #0x10
    ldr q20, [x20], #0x10
    ldr q16, [x22], #0x10
    ldr q19, [x8], #0x10
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16ab2)  // bfcvtn2 v18.8h, v21.4s
    KAI_ASM_INST(0x4ea16a91)  // bfcvtn2 v17.8h, v20.4s
    KAI_ASM_INST(0x4ea16a70)  // bfcvtn2 v16.8h, v19.4s
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_71
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_70)  // 3 block: unique 32: unique 33: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_70
KAI_ASM_LABEL(label_71)  // 3 block: unique 32: unique 33: Padding done
    add x24, x24, x5, LSL #4
    sub x26, x26, #0x4
    cmp x26, #0x4
    bge label_68
KAI_ASM_LABEL(label_73)  // 3 block: Loop skip
    cbz x26, label_79
    tbz x26, #1, label_74
    ldr d18, [x27], #0x8
    ldr d21, [x21], #0x8
    ldr d17, [x23], #0x8
    ldr d20, [x20], #0x8
    ldr d16, [x22], #0x8
    ldr d19, [x8], #0x8
    tbz x26, #0, label_75
    ld1 { v18.s }[2], [x27]
    ld1 { v21.s }[2], [x21]
    ld1 { v17.s }[2], [x23]
    ld1 { v20.s }[2], [x20]
    ld1 { v16.s }[2], [x22]
    ld1 { v19.s }[2], [x8]
    b label_75
KAI_ASM_LABEL(label_74)  // 3 block: unique 34: Partial load: partial_0_0
    ldr s18, [x27, #0]
    ldr s21, [x21, #0]
    ldr s17, [x23, #0]
    ldr s20, [x20, #0]
    ldr s16, [x22, #0]
    ldr s19, [x8, #0]
KAI_ASM_LABEL(label_75)  // 3 block: unique 34: Partial load: Done
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16ab2)  // bfcvtn2 v18.8h, v21.4s
    KAI_ASM_INST(0x4ea16a91)  // bfcvtn2 v17.8h, v20.4s
    KAI_ASM_INST(0x4ea16a70)  // bfcvtn2 v16.8h, v19.4s
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_77
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_76)  // 3 block: unique 35: unique 36: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_76
KAI_ASM_LABEL(label_77)  // 3 block: unique 35: unique 36: Padding done
    add x24, x24, x5, LSL #4
KAI_ASM_LABEL(label_79)  // 3 block: Oddments skip
    add x0, x0, #0x30
    add x8, x24, #0x30
    b label_120
KAI_ASM_LABEL(label_80)  // 3 skip
    cmp x17, #0x4
    blt label_93
    mov x25, x1
    mov x24, x2
    sub x16, x16, #0x2
    subs x17, x17, #0x4
    csel x23, x16, XZR, EQ
    mov x22, x0
    add x21, x25, x4, LSL #1
    cmp x24, #0x4
    add x20, x25, x4
    add x8, x21, x4
    add x1, x25, x4, LSL #2
    blt label_86
KAI_ASM_LABEL(label_81)  // 2 block: Column loop
    ldr q17, [x25], #0x10
    ldr q19, [x20], #0x10
    ldr q16, [x21], #0x10
    ldr q18, [x8], #0x10
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a71)  // bfcvtn2 v17.8h, v19.4s
    KAI_ASM_INST(0x4ea16a50)  // bfcvtn2 v16.8h, v18.4s
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_84
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_83)  // 2 block: unique 38: unique 39: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_83
KAI_ASM_LABEL(label_84)  // 2 block: unique 38: unique 39: Padding done
    add x22, x22, x5, LSL #4
    sub x24, x24, #0x4
    cmp x24, #0x4
    bge label_81
KAI_ASM_LABEL(label_86)  // 2 block: Loop skip
    cbz x24, label_92
    tbz x24, #1, label_87
    ldr d17, [x25], #0x8
    ldr d19, [x20], #0x8
    ldr d16, [x21], #0x8
    ldr d18, [x8], #0x8
    tbz x24, #0, label_88
    ld1 { v17.s }[2], [x25]
    ld1 { v19.s }[2], [x20]
    ld1 { v16.s }[2], [x21]
    ld1 { v18.s }[2], [x8]
    b label_88
KAI_ASM_LABEL(label_87)  // 2 block: unique 40: Partial load: partial_0_0
    ldr s17, [x25, #0]
    ldr s19, [x20, #0]
    ldr s16, [x21, #0]
    ldr s18, [x8, #0]
KAI_ASM_LABEL(label_88)  // 2 block: unique 40: Partial load: Done
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a71)  // bfcvtn2 v17.8h, v19.4s
    KAI_ASM_INST(0x4ea16a50)  // bfcvtn2 v16.8h, v18.4s
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_90
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_89)  // 2 block: unique 41: unique 42: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_89
KAI_ASM_LABEL(label_90)  // 2 block: unique 41: unique 42: Padding done
    add x22, x22, x5, LSL #4
KAI_ASM_LABEL(label_92)  // 2 block: Oddments skip
    add x0, x0, #0x20
    add x8, x22, #0x20
    b label_120
KAI_ASM_LABEL(label_93)  // 2 skip
    cmp x17, #0x2
    blt label_106
    mov x23, x2
    mov x22, x1
    sub x16, x16, #0x1
    subs x17, x17, #0x2
    csel x21, x16, XZR, EQ
    mov x20, x0
    cmp x23, #0x4
    add x8, x22, x4
    add x1, x22, x4, LSL #1
    blt label_99
KAI_ASM_LABEL(label_94)  // 1 block: Column loop
    ldr q16, [x22], #0x10
    ldr q17, [x8], #0x10
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    str q16, [x20, #0]
    cbz x21, label_97
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_96)  // 1 block: unique 44: unique 45: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_96
KAI_ASM_LABEL(label_97)  // 1 block: unique 44: unique 45: Padding done
    add x20, x20, x5, LSL #4
    sub x23, x23, #0x4
    cmp x23, #0x4
    bge label_94
KAI_ASM_LABEL(label_99)  // 1 block: Loop skip
    cbz x23, label_105
    tbz x23, #1, label_100
    ldr d16, [x22], #0x8
    ldr d17, [x8], #0x8
    tbz x23, #0, label_101
    ld1 { v16.s }[2], [x22]
    ld1 { v17.s }[2], [x8]
    b label_101
KAI_ASM_LABEL(label_100)  // 1 block: unique 46: Partial load: partial_0_0
    ldr s16, [x22, #0]
    ldr s17, [x8, #0]
KAI_ASM_LABEL(label_101)  // 1 block: unique 46: Partial load: Done
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    str q16, [x20, #0]
    cbz x21, label_103
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_102)  // 1 block: unique 47: unique 48: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_102
KAI_ASM_LABEL(label_103)  // 1 block: unique 47: unique 48: Padding done
    add x20, x20, x5, LSL #4
KAI_ASM_LABEL(label_105)  // 1 block: Oddments skip
    add x0, x0, #0x10
    add x8, x20, #0x10
    b label_120
KAI_ASM_LABEL(label_106)  // 1 skip
    cbz x17, label_119
    mov x21, x2
    mov x20, x1
    movi v17.16b, #0
    sub x16, x16, #0x1
    mov x8, x0
    cmp x21, #0x4
    add x1, x20, x4, LSL #1
    blt label_112
KAI_ASM_LABEL(label_107)  // Oddments: Column loop
    ldr q16, [x20], #0x10
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    str q16, [x8, #0]
    cbz x16, label_110
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_109)  // Oddments: unique 50: unique 51: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_109
KAI_ASM_LABEL(label_110)  // Oddments: unique 50: unique 51: Padding done
    add x8, x8, x5, LSL #4
    sub x21, x21, #0x4
    cmp x21, #0x4
    bge label_107
KAI_ASM_LABEL(label_112)  // Oddments: Loop skip
    cbz x21, label_118
    mov x6, SP
    tbz x21, #1, label_113
    ldr d16, [x20], #0x8
    ldr d17, [x6], #0x8
    tbz x21, #0, label_114
    ld1 { v16.s }[2], [x20]
    ld1 { v17.s }[2], [x6]
    b label_114
KAI_ASM_LABEL(label_113)  // Oddments: unique 52: Partial load: partial_0_0
    ldr s16, [x20, #0]
    ldr s17, [x6, #0]
KAI_ASM_LABEL(label_114)  // Oddments: unique 52: Partial load: Done
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    str q16, [x8, #0]
    cbz x16, label_116
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_115)  // Oddments: unique 53: unique 54: Padding loop
    subs x6, x6, #0x1
    str q0, [x7], #0x10
    bne label_115
KAI_ASM_LABEL(label_116)  // Oddments: unique 53: unique 54: Padding done
    add x8, x8, x5, LSL #4
KAI_ASM_LABEL(label_118)  // Oddments: Oddments skip
    add x0, x0, #0x10
    add x8, x8, #0x10
    mov x17, #0
KAI_ASM_LABEL(label_119)  // No oddments
KAI_ASM_LABEL(label_120)  // Block loop end
    cbnz x17, label_2
    sub x0, x8, x5, LSL #4
    add x0, x0, x16, LSL #4
    cbnz x3, label_1
    add SP, SP, #0x30
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave_2_2x4_fp32bf16)

    KAI_ASM_END
