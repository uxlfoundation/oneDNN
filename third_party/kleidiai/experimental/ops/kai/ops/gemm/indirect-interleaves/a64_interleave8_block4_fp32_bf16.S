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

    KAI_ASM_CODE(a64_interleave8_block4_fp32_bf16.hpp)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave8_block4_fp32_bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave8_block4_fp32_bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave8_block4_fp32_bf16)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    ldr x24, [x1, #0]
    ldr x23, [x1, #0x8]
    ldr x22, [x1, #0x10]
    ldr x21, [x1, #0x18]
    ldr x20, [x1, #0x20]
    ldr x8, [x1, #0x28]
    ldr x7, [x1, #0x30]
    ldr x6, [x1, #0x38]
    cmp x3, #0x8
    add x24, x24, x4, LSL #2
    ldr x5, [x0, #0]
    add x23, x23, x4, LSL #2
    add x22, x22, x4, LSL #2
    add x21, x21, x4, LSL #2
    add x20, x20, x4, LSL #2
    add x8, x8, x4, LSL #2
    add x7, x7, x4, LSL #2
    add x6, x6, x4, LSL #2
    beq label_1
    cmp x3, #0x2
    mov x6, x24
    csel x23, x23, x24, GE
    csel x22, x22, x24, GT
    cmp x3, #0x4
    csel x21, x21, x24, GE
    csel x20, x20, x24, GT
    cmp x3, #0x6
    csel x8, x8, x24, GE
    csel x7, x7, x24, GT
KAI_ASM_LABEL(label_1)  // no_pointer_adj
    cmp x2, #0x4
    prfm pldl1keep, [x24, #0]
    prfm pldl1keep, [x23, #0]
    prfm pldl1keep, [x22, #0]
    prfm pldl1keep, [x21, #0]
    prfm pldl1keep, [x20, #0]
    prfm pldl1keep, [x8, #0]
    prfm pldl1keep, [x7, #0]
    prfm pldl1keep, [x6, #0]
    prfm pldl1keep, [x24, #0x40]
    prfm pldl1keep, [x23, #0x40]
    prfm pldl1keep, [x22, #0x40]
    prfm pldl1keep, [x21, #0x40]
    prfm pldl1keep, [x20, #0x40]
    prfm pldl1keep, [x8, #0x40]
    prfm pldl1keep, [x7, #0x40]
    prfm pldl1keep, [x6, #0x40]
    blt label_3
KAI_ASM_LABEL(label_2)  // Main loop head
    ldr q19, [x24], #0x10
    ldr q18, [x22], #0x10
    subs x2, x2, #0x4
    ldr q17, [x20], #0x10
    ldr q16, [x7], #0x10
    cmp x2, #0x4
    ldr q23, [x23], #0x10
    ldr q22, [x21], #0x10
    ldr q21, [x8], #0x10
    ldr q20, [x6], #0x10
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    prfm pldl1keep, [x24, #0x70]
    prfm pldl1keep, [x23, #0x70]
    prfm pldl1keep, [x22, #0x70]
    prfm pldl1keep, [x21, #0x70]
    prfm pldl1keep, [x20, #0x70]
    prfm pldl1keep, [x8, #0x70]
    KAI_ASM_INST(0x4ea16af3)  // bfcvtn2 v19.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad2)  // bfcvtn2 v18.8h, v22.4s
    prfm pldl1keep, [x7, #0x70]
    prfm pldl1keep, [x6, #0x70]
    KAI_ASM_INST(0x4ea16ab1)  // bfcvtn2 v17.8h, v21.4s
    KAI_ASM_INST(0x4ea16a90)  // bfcvtn2 v16.8h, v20.4s
    str q19, [x5, #0]
    str q18, [x5, #0x10]
    str q17, [x5, #0x20]
    str q16, [x5, #0x30]
    add x5, x5, #0x40
    bge label_2
KAI_ASM_LABEL(label_3)  // Main loop skip
    cbz x2, label_6
    tbz x2, #1, label_4
    ldr d19, [x24], #0x8
    ldr d23, [x23], #0x8
    mov x1, #0x1
    ldr d18, [x22], #0x8
    ldr d22, [x21], #0x8
    ldr d17, [x20], #0x8
    ldr d21, [x8], #0x8
    ldr d16, [x7], #0x8
    ldr d20, [x6], #0x8
    tbz x2, #0, label_5
    ld1 { v19.s }[2], [x24]
    ld1 { v23.s }[2], [x23]
    ld1 { v18.s }[2], [x22]
    ld1 { v22.s }[2], [x21]
    ld1 { v17.s }[2], [x20]
    ld1 { v21.s }[2], [x8]
    ld1 { v16.s }[2], [x7]
    ld1 { v20.s }[2], [x6]
    b label_5
KAI_ASM_LABEL(label_4)  // odd_loads_1_0
    ldr s19, [x24, #0]
    ldr s23, [x23, #0]
    mov x1, #0x1
    ldr s18, [x22, #0]
    ldr s22, [x21, #0]
    ldr s17, [x20, #0]
    ldr s21, [x8, #0]
    ldr s16, [x7, #0]
    ldr s20, [x6, #0]
KAI_ASM_LABEL(label_5)  // Odd load end
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16af3)  // bfcvtn2 v19.8h, v23.4s
    KAI_ASM_INST(0x4ea16ad2)  // bfcvtn2 v18.8h, v22.4s
    KAI_ASM_INST(0x4ea16ab1)  // bfcvtn2 v17.8h, v21.4s
    KAI_ASM_INST(0x4ea16a90)  // bfcvtn2 v16.8h, v20.4s
    str q19, [x5, #0]
    str q18, [x5, #0x10]
    str q17, [x5, #0x20]
    str q16, [x5, #0x30]
    add x5, x5, #0x40
KAI_ASM_LABEL(label_6)  // Odds skip
    str x5, [x0]
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave8_block4_fp32_bf16)

    KAI_ASM_END
