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

    KAI_ASM_CODE(a64_interleave_2_1x8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave_2_1x8)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave_2_1x8)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave_2_1x8)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    sub SP, SP, #0x30
    movi v6.4s, #0
    str q6, [SP, #0]
KAI_ASM_LABEL(label_1)  // Row loop
    lsl x17, x5, #0x1
    mov x16, x5
    cmp x17, x3
    csel x17, x17, x3, LT
    sub x3, x3, x17
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x17, #0x10
    blt label_25
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
    cmp x13, #0x10
    add x25, x14, x4
    add x24, x9, x4
    add x23, x11, x4
    add x22, x28, x4
    add x21, x17, x4
    add x20, x27, x4
    add x8, x10, x4
    add x7, x26, x4
    add x1, x14, x4, LSL #4
    blt label_10
KAI_ASM_LABEL(label_3)  // 8 block: Column loop
    ldr q5, [x14], #0x10
    ldr q4, [x25], #0x10
    ldr q3, [x9], #0x10
    ldr q2, [x24], #0x10
    ldr q1, [x11], #0x10
    ldr q0, [x23], #0x10
    ldr q31, [x28], #0x10
    ldr q30, [x22], #0x10
    ldr q29, [x17], #0x10
    ldr q28, [x21], #0x10
    ldr q27, [x27], #0x10
    ldr q26, [x20], #0x10
    ldr q25, [x10], #0x10
    ldr q24, [x8], #0x10
    ldr q23, [x26], #0x10
    ldr q22, [x7], #0x10
    zip1 v17.2d, v5.2d, v4.2d
    zip1 v16.2d, v3.2d, v2.2d
    zip1 v21.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    str q17, [x12, #0]
    str q16, [x12, #0x10]
    zip1 v17.2d, v25.2d, v24.2d
    zip1 v16.2d, v23.2d, v22.2d
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
    str q6, [x6], #0x10
    bne label_5
KAI_ASM_LABEL(label_6)  // 8 block: unique 2: unique 3: Padding done
    add x12, x12, x5, LSL #4
    zip2 v17.2d, v5.2d, v4.2d
    zip2 v16.2d, v3.2d, v2.2d
    zip2 v21.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    str q17, [x12, #0]
    str q16, [x12, #0x10]
    zip2 v17.2d, v25.2d, v24.2d
    zip2 v16.2d, v23.2d, v22.2d
    str q21, [x12, #0x20]
    str q20, [x12, #0x30]
    str q19, [x12, #0x40]
    str q18, [x12, #0x50]
    str q17, [x12, #0x60]
    str q16, [x12, #0x70]
    cbz x15, label_8
    add x6, x12, #0x80
    mov x3, x15
KAI_ASM_LABEL(label_7)  // 8 block: unique 2: unique 4: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_7
KAI_ASM_LABEL(label_8)  // 8 block: unique 2: unique 4: Padding done
    add x12, x12, x5, LSL #4
    sub x13, x13, #0x10
    cmp x13, #0x10
    bge label_3
KAI_ASM_LABEL(label_10)  // 8 block: Loop skip
    cbz x13, label_24
    tbz x13, #3, label_14
    ldr d5, [x14], #0x8
    ldr d4, [x25], #0x8
    ldr d3, [x9], #0x8
    ldr d2, [x24], #0x8
    ldr d1, [x11], #0x8
    ldr d0, [x23], #0x8
    ldr d31, [x28], #0x8
    ldr d30, [x22], #0x8
    ldr d29, [x17], #0x8
    ldr d28, [x21], #0x8
    ldr d27, [x27], #0x8
    ldr d26, [x20], #0x8
    ldr d25, [x10], #0x8
    ldr d24, [x8], #0x8
    ldr d23, [x26], #0x8
    ldr d22, [x7], #0x8
    tbz x13, #2, label_12
    ld1 { v5.s }[2], [x14], #0x4
    ld1 { v4.s }[2], [x25], #0x4
    ld1 { v3.s }[2], [x9], #0x4
    ld1 { v2.s }[2], [x24], #0x4
    ld1 { v1.s }[2], [x11], #0x4
    ld1 { v0.s }[2], [x23], #0x4
    ld1 { v31.s }[2], [x28], #0x4
    ld1 { v30.s }[2], [x22], #0x4
    ld1 { v29.s }[2], [x17], #0x4
    ld1 { v28.s }[2], [x21], #0x4
    ld1 { v27.s }[2], [x27], #0x4
    ld1 { v26.s }[2], [x20], #0x4
    ld1 { v25.s }[2], [x10], #0x4
    ld1 { v24.s }[2], [x8], #0x4
    ld1 { v23.s }[2], [x26], #0x4
    ld1 { v22.s }[2], [x7], #0x4
    tbz x13, #1, label_11
    ld1 { v5.h }[6], [x14], #0x2
    ld1 { v4.h }[6], [x25], #0x2
    ld1 { v3.h }[6], [x9], #0x2
    ld1 { v2.h }[6], [x24], #0x2
    ld1 { v1.h }[6], [x11], #0x2
    ld1 { v0.h }[6], [x23], #0x2
    ld1 { v31.h }[6], [x28], #0x2
    ld1 { v30.h }[6], [x22], #0x2
    ld1 { v29.h }[6], [x17], #0x2
    ld1 { v28.h }[6], [x21], #0x2
    ld1 { v27.h }[6], [x27], #0x2
    ld1 { v26.h }[6], [x20], #0x2
    ld1 { v25.h }[6], [x10], #0x2
    ld1 { v24.h }[6], [x8], #0x2
    ld1 { v23.h }[6], [x26], #0x2
    ld1 { v22.h }[6], [x7], #0x2
    tbz x13, #0, label_18
    ld1 { v5.b }[14], [x14]
    ld1 { v4.b }[14], [x25]
    ld1 { v3.b }[14], [x9]
    ld1 { v2.b }[14], [x24]
    ld1 { v1.b }[14], [x11]
    ld1 { v0.b }[14], [x23]
    ld1 { v31.b }[14], [x28]
    ld1 { v30.b }[14], [x22]
    ld1 { v29.b }[14], [x17]
    ld1 { v28.b }[14], [x21]
    ld1 { v27.b }[14], [x27]
    ld1 { v26.b }[14], [x20]
    ld1 { v25.b }[14], [x10]
    ld1 { v24.b }[14], [x8]
    ld1 { v23.b }[14], [x26]
    ld1 { v22.b }[14], [x7]
    b label_18
KAI_ASM_LABEL(label_11)  // 8 block: unique 5: Partial load: partial_0_12
    tbz x13, #0, label_18
    ld1 { v5.b }[12], [x14]
    ld1 { v4.b }[12], [x25]
    ld1 { v3.b }[12], [x9]
    ld1 { v2.b }[12], [x24]
    ld1 { v1.b }[12], [x11]
    ld1 { v0.b }[12], [x23]
    ld1 { v31.b }[12], [x28]
    ld1 { v30.b }[12], [x22]
    ld1 { v29.b }[12], [x17]
    ld1 { v28.b }[12], [x21]
    ld1 { v27.b }[12], [x27]
    ld1 { v26.b }[12], [x20]
    ld1 { v25.b }[12], [x10]
    ld1 { v24.b }[12], [x8]
    ld1 { v23.b }[12], [x26]
    ld1 { v22.b }[12], [x7]
    b label_18
KAI_ASM_LABEL(label_12)  // 8 block: unique 5: Partial load: partial_1_8
    tbz x13, #1, label_13
    ld1 { v5.h }[4], [x14], #0x2
    ld1 { v4.h }[4], [x25], #0x2
    ld1 { v3.h }[4], [x9], #0x2
    ld1 { v2.h }[4], [x24], #0x2
    ld1 { v1.h }[4], [x11], #0x2
    ld1 { v0.h }[4], [x23], #0x2
    ld1 { v31.h }[4], [x28], #0x2
    ld1 { v30.h }[4], [x22], #0x2
    ld1 { v29.h }[4], [x17], #0x2
    ld1 { v28.h }[4], [x21], #0x2
    ld1 { v27.h }[4], [x27], #0x2
    ld1 { v26.h }[4], [x20], #0x2
    ld1 { v25.h }[4], [x10], #0x2
    ld1 { v24.h }[4], [x8], #0x2
    ld1 { v23.h }[4], [x26], #0x2
    ld1 { v22.h }[4], [x7], #0x2
    tbz x13, #0, label_18
    ld1 { v5.b }[10], [x14]
    ld1 { v4.b }[10], [x25]
    ld1 { v3.b }[10], [x9]
    ld1 { v2.b }[10], [x24]
    ld1 { v1.b }[10], [x11]
    ld1 { v0.b }[10], [x23]
    ld1 { v31.b }[10], [x28]
    ld1 { v30.b }[10], [x22]
    ld1 { v29.b }[10], [x17]
    ld1 { v28.b }[10], [x21]
    ld1 { v27.b }[10], [x27]
    ld1 { v26.b }[10], [x20]
    ld1 { v25.b }[10], [x10]
    ld1 { v24.b }[10], [x8]
    ld1 { v23.b }[10], [x26]
    ld1 { v22.b }[10], [x7]
    b label_18
KAI_ASM_LABEL(label_13)  // 8 block: unique 5: Partial load: partial_0_8
    tbz x13, #0, label_18
    ld1 { v5.b }[8], [x14]
    ld1 { v4.b }[8], [x25]
    ld1 { v3.b }[8], [x9]
    ld1 { v2.b }[8], [x24]
    ld1 { v1.b }[8], [x11]
    ld1 { v0.b }[8], [x23]
    ld1 { v31.b }[8], [x28]
    ld1 { v30.b }[8], [x22]
    ld1 { v29.b }[8], [x17]
    ld1 { v28.b }[8], [x21]
    ld1 { v27.b }[8], [x27]
    ld1 { v26.b }[8], [x20]
    ld1 { v25.b }[8], [x10]
    ld1 { v24.b }[8], [x8]
    ld1 { v23.b }[8], [x26]
    ld1 { v22.b }[8], [x7]
    b label_18
KAI_ASM_LABEL(label_14)  // 8 block: unique 5: Partial load: partial_2_0
    tbz x13, #2, label_16
    ldr s5, [x14], #0x4
    ldr s4, [x25], #0x4
    ldr s3, [x9], #0x4
    ldr s2, [x24], #0x4
    ldr s1, [x11], #0x4
    ldr s0, [x23], #0x4
    ldr s31, [x28], #0x4
    ldr s30, [x22], #0x4
    ldr s29, [x17], #0x4
    ldr s28, [x21], #0x4
    ldr s27, [x27], #0x4
    ldr s26, [x20], #0x4
    ldr s25, [x10], #0x4
    ldr s24, [x8], #0x4
    ldr s23, [x26], #0x4
    ldr s22, [x7], #0x4
    tbz x13, #1, label_15
    ld1 { v5.h }[2], [x14], #0x2
    ld1 { v4.h }[2], [x25], #0x2
    ld1 { v3.h }[2], [x9], #0x2
    ld1 { v2.h }[2], [x24], #0x2
    ld1 { v1.h }[2], [x11], #0x2
    ld1 { v0.h }[2], [x23], #0x2
    ld1 { v31.h }[2], [x28], #0x2
    ld1 { v30.h }[2], [x22], #0x2
    ld1 { v29.h }[2], [x17], #0x2
    ld1 { v28.h }[2], [x21], #0x2
    ld1 { v27.h }[2], [x27], #0x2
    ld1 { v26.h }[2], [x20], #0x2
    ld1 { v25.h }[2], [x10], #0x2
    ld1 { v24.h }[2], [x8], #0x2
    ld1 { v23.h }[2], [x26], #0x2
    ld1 { v22.h }[2], [x7], #0x2
    tbz x13, #0, label_18
    ld1 { v5.b }[6], [x14]
    ld1 { v4.b }[6], [x25]
    ld1 { v3.b }[6], [x9]
    ld1 { v2.b }[6], [x24]
    ld1 { v1.b }[6], [x11]
    ld1 { v0.b }[6], [x23]
    ld1 { v31.b }[6], [x28]
    ld1 { v30.b }[6], [x22]
    ld1 { v29.b }[6], [x17]
    ld1 { v28.b }[6], [x21]
    ld1 { v27.b }[6], [x27]
    ld1 { v26.b }[6], [x20]
    ld1 { v25.b }[6], [x10]
    ld1 { v24.b }[6], [x8]
    ld1 { v23.b }[6], [x26]
    ld1 { v22.b }[6], [x7]
    b label_18
KAI_ASM_LABEL(label_15)  // 8 block: unique 5: Partial load: partial_0_4
    tbz x13, #0, label_18
    ld1 { v5.b }[4], [x14]
    ld1 { v4.b }[4], [x25]
    ld1 { v3.b }[4], [x9]
    ld1 { v2.b }[4], [x24]
    ld1 { v1.b }[4], [x11]
    ld1 { v0.b }[4], [x23]
    ld1 { v31.b }[4], [x28]
    ld1 { v30.b }[4], [x22]
    ld1 { v29.b }[4], [x17]
    ld1 { v28.b }[4], [x21]
    ld1 { v27.b }[4], [x27]
    ld1 { v26.b }[4], [x20]
    ld1 { v25.b }[4], [x10]
    ld1 { v24.b }[4], [x8]
    ld1 { v23.b }[4], [x26]
    ld1 { v22.b }[4], [x7]
    b label_18
KAI_ASM_LABEL(label_16)  // 8 block: unique 5: Partial load: partial_1_0
    tbz x13, #1, label_17
    ldr h5, [x14], #0x2
    ldr h4, [x25], #0x2
    ldr h3, [x9], #0x2
    ldr h2, [x24], #0x2
    ldr h1, [x11], #0x2
    ldr h0, [x23], #0x2
    ldr h31, [x28], #0x2
    ldr h30, [x22], #0x2
    ldr h29, [x17], #0x2
    ldr h28, [x21], #0x2
    ldr h27, [x27], #0x2
    ldr h26, [x20], #0x2
    ldr h25, [x10], #0x2
    ldr h24, [x8], #0x2
    ldr h23, [x26], #0x2
    ldr h22, [x7], #0x2
    tbz x13, #0, label_18
    ld1 { v5.b }[2], [x14]
    ld1 { v4.b }[2], [x25]
    ld1 { v3.b }[2], [x9]
    ld1 { v2.b }[2], [x24]
    ld1 { v1.b }[2], [x11]
    ld1 { v0.b }[2], [x23]
    ld1 { v31.b }[2], [x28]
    ld1 { v30.b }[2], [x22]
    ld1 { v29.b }[2], [x17]
    ld1 { v28.b }[2], [x21]
    ld1 { v27.b }[2], [x27]
    ld1 { v26.b }[2], [x20]
    ld1 { v25.b }[2], [x10]
    ld1 { v24.b }[2], [x8]
    ld1 { v23.b }[2], [x26]
    ld1 { v22.b }[2], [x7]
    b label_18
KAI_ASM_LABEL(label_17)  // 8 block: unique 5: Partial load: partial_0_0
    ldr b5, [x14, #0]
    ldr b4, [x25, #0]
    ldr b3, [x9, #0]
    ldr b2, [x24, #0]
    ldr b1, [x11, #0]
    ldr b0, [x23, #0]
    ldr b31, [x28, #0]
    ldr b30, [x22, #0]
    ldr b29, [x17, #0]
    ldr b28, [x21, #0]
    ldr b27, [x27, #0]
    ldr b26, [x20, #0]
    ldr b25, [x10, #0]
    ldr b24, [x8, #0]
    ldr b23, [x26, #0]
    ldr b22, [x7, #0]
KAI_ASM_LABEL(label_18)  // 8 block: unique 5: Partial load: Done
    zip1 v17.2d, v5.2d, v4.2d
    zip1 v16.2d, v3.2d, v2.2d
    zip1 v21.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    str q17, [x12, #0]
    str q16, [x12, #0x10]
    zip1 v17.2d, v25.2d, v24.2d
    zip1 v16.2d, v23.2d, v22.2d
    str q21, [x12, #0x20]
    str q20, [x12, #0x30]
    str q19, [x12, #0x40]
    str q18, [x12, #0x50]
    str q17, [x12, #0x60]
    str q16, [x12, #0x70]
    cbz x15, label_20
    add x6, x12, #0x80
    mov x3, x15
KAI_ASM_LABEL(label_19)  // 8 block: unique 6: unique 7: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_19
KAI_ASM_LABEL(label_20)  // 8 block: unique 6: unique 7: Padding done
    cmp x13, #0x8
    add x12, x12, x5, LSL #4
    ble label_23
    zip2 v17.2d, v5.2d, v4.2d
    zip2 v16.2d, v3.2d, v2.2d
    zip2 v21.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    str q17, [x12, #0]
    str q16, [x12, #0x10]
    zip2 v17.2d, v25.2d, v24.2d
    zip2 v16.2d, v23.2d, v22.2d
    str q21, [x12, #0x20]
    str q20, [x12, #0x30]
    str q19, [x12, #0x40]
    str q18, [x12, #0x50]
    str q17, [x12, #0x60]
    str q16, [x12, #0x70]
    cbz x15, label_22
    add x6, x12, #0x80
    mov x3, x15
KAI_ASM_LABEL(label_21)  // 8 block: unique 6: unique 8: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_21
KAI_ASM_LABEL(label_22)  // 8 block: unique 6: unique 8: Padding done
    add x12, x12, x5, LSL #4
KAI_ASM_LABEL(label_23)  // 8 block: unique 6: Done
KAI_ASM_LABEL(label_24)  // 8 block: Oddments skip
    add x0, x0, #0x80
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x12, #0x80
    ldr x16, [SP, #0x20]
    b label_210
KAI_ASM_LABEL(label_25)  // 8 skip
    cmp x17, #0xe
    blt label_48
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
    cmp x13, #0x10
    add x24, x14, x4
    add x23, x28, x4
    add x22, x10, x4
    add x21, x27, x4
    add x20, x9, x4
    add x8, x26, x4
    add x7, x25, x4
    add x1, x25, x4, LSL #1
    blt label_33
KAI_ASM_LABEL(label_26)  // 7 block: Column loop
    ldr q3, [x14], #0x10
    ldr q2, [x24], #0x10
    ldr q1, [x28], #0x10
    ldr q0, [x23], #0x10
    ldr q31, [x10], #0x10
    ldr q30, [x22], #0x10
    ldr q29, [x27], #0x10
    ldr q28, [x21], #0x10
    ldr q27, [x9], #0x10
    ldr q26, [x20], #0x10
    ldr q25, [x26], #0x10
    ldr q24, [x8], #0x10
    ldr q23, [x25], #0x10
    ldr q22, [x7], #0x10
    zip1 v21.2d, v3.2d, v2.2d
    zip1 v16.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    zip1 v17.2d, v25.2d, v24.2d
    str q21, [x11, #0]
    str q16, [x11, #0x10]
    zip1 v16.2d, v23.2d, v22.2d
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_29
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_28)  // 7 block: unique 10: unique 11: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_28
KAI_ASM_LABEL(label_29)  // 7 block: unique 10: unique 11: Padding done
    add x11, x11, x5, LSL #4
    zip2 v21.2d, v3.2d, v2.2d
    zip2 v16.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    zip2 v17.2d, v25.2d, v24.2d
    str q21, [x11, #0]
    str q16, [x11, #0x10]
    zip2 v16.2d, v23.2d, v22.2d
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_31
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_30)  // 7 block: unique 10: unique 12: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_30
KAI_ASM_LABEL(label_31)  // 7 block: unique 10: unique 12: Padding done
    add x11, x11, x5, LSL #4
    sub x13, x13, #0x10
    cmp x13, #0x10
    bge label_26
KAI_ASM_LABEL(label_33)  // 7 block: Loop skip
    cbz x13, label_47
    tbz x13, #3, label_37
    ldr d3, [x14], #0x8
    ldr d2, [x24], #0x8
    ldr d1, [x28], #0x8
    ldr d0, [x23], #0x8
    ldr d31, [x10], #0x8
    ldr d30, [x22], #0x8
    ldr d29, [x27], #0x8
    ldr d28, [x21], #0x8
    ldr d27, [x9], #0x8
    ldr d26, [x20], #0x8
    ldr d25, [x26], #0x8
    ldr d24, [x8], #0x8
    ldr d23, [x25], #0x8
    ldr d22, [x7], #0x8
    tbz x13, #2, label_35
    ld1 { v3.s }[2], [x14], #0x4
    ld1 { v2.s }[2], [x24], #0x4
    ld1 { v1.s }[2], [x28], #0x4
    ld1 { v0.s }[2], [x23], #0x4
    ld1 { v31.s }[2], [x10], #0x4
    ld1 { v30.s }[2], [x22], #0x4
    ld1 { v29.s }[2], [x27], #0x4
    ld1 { v28.s }[2], [x21], #0x4
    ld1 { v27.s }[2], [x9], #0x4
    ld1 { v26.s }[2], [x20], #0x4
    ld1 { v25.s }[2], [x26], #0x4
    ld1 { v24.s }[2], [x8], #0x4
    ld1 { v23.s }[2], [x25], #0x4
    ld1 { v22.s }[2], [x7], #0x4
    tbz x13, #1, label_34
    ld1 { v3.h }[6], [x14], #0x2
    ld1 { v2.h }[6], [x24], #0x2
    ld1 { v1.h }[6], [x28], #0x2
    ld1 { v0.h }[6], [x23], #0x2
    ld1 { v31.h }[6], [x10], #0x2
    ld1 { v30.h }[6], [x22], #0x2
    ld1 { v29.h }[6], [x27], #0x2
    ld1 { v28.h }[6], [x21], #0x2
    ld1 { v27.h }[6], [x9], #0x2
    ld1 { v26.h }[6], [x20], #0x2
    ld1 { v25.h }[6], [x26], #0x2
    ld1 { v24.h }[6], [x8], #0x2
    ld1 { v23.h }[6], [x25], #0x2
    ld1 { v22.h }[6], [x7], #0x2
    tbz x13, #0, label_41
    ld1 { v3.b }[14], [x14]
    ld1 { v2.b }[14], [x24]
    ld1 { v1.b }[14], [x28]
    ld1 { v0.b }[14], [x23]
    ld1 { v31.b }[14], [x10]
    ld1 { v30.b }[14], [x22]
    ld1 { v29.b }[14], [x27]
    ld1 { v28.b }[14], [x21]
    ld1 { v27.b }[14], [x9]
    ld1 { v26.b }[14], [x20]
    ld1 { v25.b }[14], [x26]
    ld1 { v24.b }[14], [x8]
    ld1 { v23.b }[14], [x25]
    ld1 { v22.b }[14], [x7]
    b label_41
KAI_ASM_LABEL(label_34)  // 7 block: unique 13: Partial load: partial_0_12
    tbz x13, #0, label_41
    ld1 { v3.b }[12], [x14]
    ld1 { v2.b }[12], [x24]
    ld1 { v1.b }[12], [x28]
    ld1 { v0.b }[12], [x23]
    ld1 { v31.b }[12], [x10]
    ld1 { v30.b }[12], [x22]
    ld1 { v29.b }[12], [x27]
    ld1 { v28.b }[12], [x21]
    ld1 { v27.b }[12], [x9]
    ld1 { v26.b }[12], [x20]
    ld1 { v25.b }[12], [x26]
    ld1 { v24.b }[12], [x8]
    ld1 { v23.b }[12], [x25]
    ld1 { v22.b }[12], [x7]
    b label_41
KAI_ASM_LABEL(label_35)  // 7 block: unique 13: Partial load: partial_1_8
    tbz x13, #1, label_36
    ld1 { v3.h }[4], [x14], #0x2
    ld1 { v2.h }[4], [x24], #0x2
    ld1 { v1.h }[4], [x28], #0x2
    ld1 { v0.h }[4], [x23], #0x2
    ld1 { v31.h }[4], [x10], #0x2
    ld1 { v30.h }[4], [x22], #0x2
    ld1 { v29.h }[4], [x27], #0x2
    ld1 { v28.h }[4], [x21], #0x2
    ld1 { v27.h }[4], [x9], #0x2
    ld1 { v26.h }[4], [x20], #0x2
    ld1 { v25.h }[4], [x26], #0x2
    ld1 { v24.h }[4], [x8], #0x2
    ld1 { v23.h }[4], [x25], #0x2
    ld1 { v22.h }[4], [x7], #0x2
    tbz x13, #0, label_41
    ld1 { v3.b }[10], [x14]
    ld1 { v2.b }[10], [x24]
    ld1 { v1.b }[10], [x28]
    ld1 { v0.b }[10], [x23]
    ld1 { v31.b }[10], [x10]
    ld1 { v30.b }[10], [x22]
    ld1 { v29.b }[10], [x27]
    ld1 { v28.b }[10], [x21]
    ld1 { v27.b }[10], [x9]
    ld1 { v26.b }[10], [x20]
    ld1 { v25.b }[10], [x26]
    ld1 { v24.b }[10], [x8]
    ld1 { v23.b }[10], [x25]
    ld1 { v22.b }[10], [x7]
    b label_41
KAI_ASM_LABEL(label_36)  // 7 block: unique 13: Partial load: partial_0_8
    tbz x13, #0, label_41
    ld1 { v3.b }[8], [x14]
    ld1 { v2.b }[8], [x24]
    ld1 { v1.b }[8], [x28]
    ld1 { v0.b }[8], [x23]
    ld1 { v31.b }[8], [x10]
    ld1 { v30.b }[8], [x22]
    ld1 { v29.b }[8], [x27]
    ld1 { v28.b }[8], [x21]
    ld1 { v27.b }[8], [x9]
    ld1 { v26.b }[8], [x20]
    ld1 { v25.b }[8], [x26]
    ld1 { v24.b }[8], [x8]
    ld1 { v23.b }[8], [x25]
    ld1 { v22.b }[8], [x7]
    b label_41
KAI_ASM_LABEL(label_37)  // 7 block: unique 13: Partial load: partial_2_0
    tbz x13, #2, label_39
    ldr s3, [x14], #0x4
    ldr s2, [x24], #0x4
    ldr s1, [x28], #0x4
    ldr s0, [x23], #0x4
    ldr s31, [x10], #0x4
    ldr s30, [x22], #0x4
    ldr s29, [x27], #0x4
    ldr s28, [x21], #0x4
    ldr s27, [x9], #0x4
    ldr s26, [x20], #0x4
    ldr s25, [x26], #0x4
    ldr s24, [x8], #0x4
    ldr s23, [x25], #0x4
    ldr s22, [x7], #0x4
    tbz x13, #1, label_38
    ld1 { v3.h }[2], [x14], #0x2
    ld1 { v2.h }[2], [x24], #0x2
    ld1 { v1.h }[2], [x28], #0x2
    ld1 { v0.h }[2], [x23], #0x2
    ld1 { v31.h }[2], [x10], #0x2
    ld1 { v30.h }[2], [x22], #0x2
    ld1 { v29.h }[2], [x27], #0x2
    ld1 { v28.h }[2], [x21], #0x2
    ld1 { v27.h }[2], [x9], #0x2
    ld1 { v26.h }[2], [x20], #0x2
    ld1 { v25.h }[2], [x26], #0x2
    ld1 { v24.h }[2], [x8], #0x2
    ld1 { v23.h }[2], [x25], #0x2
    ld1 { v22.h }[2], [x7], #0x2
    tbz x13, #0, label_41
    ld1 { v3.b }[6], [x14]
    ld1 { v2.b }[6], [x24]
    ld1 { v1.b }[6], [x28]
    ld1 { v0.b }[6], [x23]
    ld1 { v31.b }[6], [x10]
    ld1 { v30.b }[6], [x22]
    ld1 { v29.b }[6], [x27]
    ld1 { v28.b }[6], [x21]
    ld1 { v27.b }[6], [x9]
    ld1 { v26.b }[6], [x20]
    ld1 { v25.b }[6], [x26]
    ld1 { v24.b }[6], [x8]
    ld1 { v23.b }[6], [x25]
    ld1 { v22.b }[6], [x7]
    b label_41
KAI_ASM_LABEL(label_38)  // 7 block: unique 13: Partial load: partial_0_4
    tbz x13, #0, label_41
    ld1 { v3.b }[4], [x14]
    ld1 { v2.b }[4], [x24]
    ld1 { v1.b }[4], [x28]
    ld1 { v0.b }[4], [x23]
    ld1 { v31.b }[4], [x10]
    ld1 { v30.b }[4], [x22]
    ld1 { v29.b }[4], [x27]
    ld1 { v28.b }[4], [x21]
    ld1 { v27.b }[4], [x9]
    ld1 { v26.b }[4], [x20]
    ld1 { v25.b }[4], [x26]
    ld1 { v24.b }[4], [x8]
    ld1 { v23.b }[4], [x25]
    ld1 { v22.b }[4], [x7]
    b label_41
KAI_ASM_LABEL(label_39)  // 7 block: unique 13: Partial load: partial_1_0
    tbz x13, #1, label_40
    ldr h3, [x14], #0x2
    ldr h2, [x24], #0x2
    ldr h1, [x28], #0x2
    ldr h0, [x23], #0x2
    ldr h31, [x10], #0x2
    ldr h30, [x22], #0x2
    ldr h29, [x27], #0x2
    ldr h28, [x21], #0x2
    ldr h27, [x9], #0x2
    ldr h26, [x20], #0x2
    ldr h25, [x26], #0x2
    ldr h24, [x8], #0x2
    ldr h23, [x25], #0x2
    ldr h22, [x7], #0x2
    tbz x13, #0, label_41
    ld1 { v3.b }[2], [x14]
    ld1 { v2.b }[2], [x24]
    ld1 { v1.b }[2], [x28]
    ld1 { v0.b }[2], [x23]
    ld1 { v31.b }[2], [x10]
    ld1 { v30.b }[2], [x22]
    ld1 { v29.b }[2], [x27]
    ld1 { v28.b }[2], [x21]
    ld1 { v27.b }[2], [x9]
    ld1 { v26.b }[2], [x20]
    ld1 { v25.b }[2], [x26]
    ld1 { v24.b }[2], [x8]
    ld1 { v23.b }[2], [x25]
    ld1 { v22.b }[2], [x7]
    b label_41
KAI_ASM_LABEL(label_40)  // 7 block: unique 13: Partial load: partial_0_0
    ldr b3, [x14, #0]
    ldr b2, [x24, #0]
    ldr b1, [x28, #0]
    ldr b0, [x23, #0]
    ldr b31, [x10, #0]
    ldr b30, [x22, #0]
    ldr b29, [x27, #0]
    ldr b28, [x21, #0]
    ldr b27, [x9, #0]
    ldr b26, [x20, #0]
    ldr b25, [x26, #0]
    ldr b24, [x8, #0]
    ldr b23, [x25, #0]
    ldr b22, [x7, #0]
KAI_ASM_LABEL(label_41)  // 7 block: unique 13: Partial load: Done
    zip1 v21.2d, v3.2d, v2.2d
    zip1 v16.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    zip1 v17.2d, v25.2d, v24.2d
    str q21, [x11, #0]
    str q16, [x11, #0x10]
    zip1 v16.2d, v23.2d, v22.2d
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_43
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_42)  // 7 block: unique 14: unique 15: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_42
KAI_ASM_LABEL(label_43)  // 7 block: unique 14: unique 15: Padding done
    cmp x13, #0x8
    add x11, x11, x5, LSL #4
    ble label_46
    zip2 v21.2d, v3.2d, v2.2d
    zip2 v16.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    zip2 v17.2d, v25.2d, v24.2d
    str q21, [x11, #0]
    str q16, [x11, #0x10]
    zip2 v16.2d, v23.2d, v22.2d
    str q20, [x11, #0x20]
    str q19, [x11, #0x30]
    str q18, [x11, #0x40]
    str q17, [x11, #0x50]
    str q16, [x11, #0x60]
    cbz x12, label_45
    add x6, x11, #0x70
    mov x3, x12
KAI_ASM_LABEL(label_44)  // 7 block: unique 14: unique 16: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_44
KAI_ASM_LABEL(label_45)  // 7 block: unique 14: unique 16: Padding done
    add x11, x11, x5, LSL #4
KAI_ASM_LABEL(label_46)  // 7 block: unique 14: Done
KAI_ASM_LABEL(label_47)  // 7 block: Oddments skip
    add x0, x0, #0x70
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x11, #0x70
    ldr x16, [SP, #0x20]
    b label_210
KAI_ASM_LABEL(label_48)  // 7 skip
    cmp x17, #0xc
    blt label_71
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
    cmp x11, #0x10
    add x23, x12, x4
    add x22, x26, x4
    add x21, x28, x4
    add x20, x25, x4
    add x8, x27, x4
    add x7, x24, x4
    add x1, x27, x4, LSL #2
    blt label_56
KAI_ASM_LABEL(label_49)  // 6 block: Column loop
    ldr q1, [x12], #0x10
    ldr q0, [x23], #0x10
    ldr q31, [x26], #0x10
    ldr q30, [x22], #0x10
    ldr q29, [x28], #0x10
    ldr q28, [x21], #0x10
    ldr q27, [x25], #0x10
    ldr q26, [x20], #0x10
    ldr q25, [x27], #0x10
    ldr q24, [x8], #0x10
    ldr q23, [x24], #0x10
    ldr q22, [x7], #0x10
    zip1 v21.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    zip1 v17.2d, v25.2d, v24.2d
    zip1 v16.2d, v23.2d, v22.2d
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_52
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_51)  // 6 block: unique 18: unique 19: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_51
KAI_ASM_LABEL(label_52)  // 6 block: unique 18: unique 19: Padding done
    add x9, x9, x5, LSL #4
    zip2 v21.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    zip2 v17.2d, v25.2d, v24.2d
    zip2 v16.2d, v23.2d, v22.2d
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_54
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_53)  // 6 block: unique 18: unique 20: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_53
KAI_ASM_LABEL(label_54)  // 6 block: unique 18: unique 20: Padding done
    add x9, x9, x5, LSL #4
    sub x11, x11, #0x10
    cmp x11, #0x10
    bge label_49
KAI_ASM_LABEL(label_56)  // 6 block: Loop skip
    cbz x11, label_70
    tbz x11, #3, label_60
    ldr d1, [x12], #0x8
    ldr d0, [x23], #0x8
    ldr d31, [x26], #0x8
    ldr d30, [x22], #0x8
    ldr d29, [x28], #0x8
    ldr d28, [x21], #0x8
    ldr d27, [x25], #0x8
    ldr d26, [x20], #0x8
    ldr d25, [x27], #0x8
    ldr d24, [x8], #0x8
    ldr d23, [x24], #0x8
    ldr d22, [x7], #0x8
    tbz x11, #2, label_58
    ld1 { v1.s }[2], [x12], #0x4
    ld1 { v0.s }[2], [x23], #0x4
    ld1 { v31.s }[2], [x26], #0x4
    ld1 { v30.s }[2], [x22], #0x4
    ld1 { v29.s }[2], [x28], #0x4
    ld1 { v28.s }[2], [x21], #0x4
    ld1 { v27.s }[2], [x25], #0x4
    ld1 { v26.s }[2], [x20], #0x4
    ld1 { v25.s }[2], [x27], #0x4
    ld1 { v24.s }[2], [x8], #0x4
    ld1 { v23.s }[2], [x24], #0x4
    ld1 { v22.s }[2], [x7], #0x4
    tbz x11, #1, label_57
    ld1 { v1.h }[6], [x12], #0x2
    ld1 { v0.h }[6], [x23], #0x2
    ld1 { v31.h }[6], [x26], #0x2
    ld1 { v30.h }[6], [x22], #0x2
    ld1 { v29.h }[6], [x28], #0x2
    ld1 { v28.h }[6], [x21], #0x2
    ld1 { v27.h }[6], [x25], #0x2
    ld1 { v26.h }[6], [x20], #0x2
    ld1 { v25.h }[6], [x27], #0x2
    ld1 { v24.h }[6], [x8], #0x2
    ld1 { v23.h }[6], [x24], #0x2
    ld1 { v22.h }[6], [x7], #0x2
    tbz x11, #0, label_64
    ld1 { v1.b }[14], [x12]
    ld1 { v0.b }[14], [x23]
    ld1 { v31.b }[14], [x26]
    ld1 { v30.b }[14], [x22]
    ld1 { v29.b }[14], [x28]
    ld1 { v28.b }[14], [x21]
    ld1 { v27.b }[14], [x25]
    ld1 { v26.b }[14], [x20]
    ld1 { v25.b }[14], [x27]
    ld1 { v24.b }[14], [x8]
    ld1 { v23.b }[14], [x24]
    ld1 { v22.b }[14], [x7]
    b label_64
KAI_ASM_LABEL(label_57)  // 6 block: unique 21: Partial load: partial_0_12
    tbz x11, #0, label_64
    ld1 { v1.b }[12], [x12]
    ld1 { v0.b }[12], [x23]
    ld1 { v31.b }[12], [x26]
    ld1 { v30.b }[12], [x22]
    ld1 { v29.b }[12], [x28]
    ld1 { v28.b }[12], [x21]
    ld1 { v27.b }[12], [x25]
    ld1 { v26.b }[12], [x20]
    ld1 { v25.b }[12], [x27]
    ld1 { v24.b }[12], [x8]
    ld1 { v23.b }[12], [x24]
    ld1 { v22.b }[12], [x7]
    b label_64
KAI_ASM_LABEL(label_58)  // 6 block: unique 21: Partial load: partial_1_8
    tbz x11, #1, label_59
    ld1 { v1.h }[4], [x12], #0x2
    ld1 { v0.h }[4], [x23], #0x2
    ld1 { v31.h }[4], [x26], #0x2
    ld1 { v30.h }[4], [x22], #0x2
    ld1 { v29.h }[4], [x28], #0x2
    ld1 { v28.h }[4], [x21], #0x2
    ld1 { v27.h }[4], [x25], #0x2
    ld1 { v26.h }[4], [x20], #0x2
    ld1 { v25.h }[4], [x27], #0x2
    ld1 { v24.h }[4], [x8], #0x2
    ld1 { v23.h }[4], [x24], #0x2
    ld1 { v22.h }[4], [x7], #0x2
    tbz x11, #0, label_64
    ld1 { v1.b }[10], [x12]
    ld1 { v0.b }[10], [x23]
    ld1 { v31.b }[10], [x26]
    ld1 { v30.b }[10], [x22]
    ld1 { v29.b }[10], [x28]
    ld1 { v28.b }[10], [x21]
    ld1 { v27.b }[10], [x25]
    ld1 { v26.b }[10], [x20]
    ld1 { v25.b }[10], [x27]
    ld1 { v24.b }[10], [x8]
    ld1 { v23.b }[10], [x24]
    ld1 { v22.b }[10], [x7]
    b label_64
KAI_ASM_LABEL(label_59)  // 6 block: unique 21: Partial load: partial_0_8
    tbz x11, #0, label_64
    ld1 { v1.b }[8], [x12]
    ld1 { v0.b }[8], [x23]
    ld1 { v31.b }[8], [x26]
    ld1 { v30.b }[8], [x22]
    ld1 { v29.b }[8], [x28]
    ld1 { v28.b }[8], [x21]
    ld1 { v27.b }[8], [x25]
    ld1 { v26.b }[8], [x20]
    ld1 { v25.b }[8], [x27]
    ld1 { v24.b }[8], [x8]
    ld1 { v23.b }[8], [x24]
    ld1 { v22.b }[8], [x7]
    b label_64
KAI_ASM_LABEL(label_60)  // 6 block: unique 21: Partial load: partial_2_0
    tbz x11, #2, label_62
    ldr s1, [x12], #0x4
    ldr s0, [x23], #0x4
    ldr s31, [x26], #0x4
    ldr s30, [x22], #0x4
    ldr s29, [x28], #0x4
    ldr s28, [x21], #0x4
    ldr s27, [x25], #0x4
    ldr s26, [x20], #0x4
    ldr s25, [x27], #0x4
    ldr s24, [x8], #0x4
    ldr s23, [x24], #0x4
    ldr s22, [x7], #0x4
    tbz x11, #1, label_61
    ld1 { v1.h }[2], [x12], #0x2
    ld1 { v0.h }[2], [x23], #0x2
    ld1 { v31.h }[2], [x26], #0x2
    ld1 { v30.h }[2], [x22], #0x2
    ld1 { v29.h }[2], [x28], #0x2
    ld1 { v28.h }[2], [x21], #0x2
    ld1 { v27.h }[2], [x25], #0x2
    ld1 { v26.h }[2], [x20], #0x2
    ld1 { v25.h }[2], [x27], #0x2
    ld1 { v24.h }[2], [x8], #0x2
    ld1 { v23.h }[2], [x24], #0x2
    ld1 { v22.h }[2], [x7], #0x2
    tbz x11, #0, label_64
    ld1 { v1.b }[6], [x12]
    ld1 { v0.b }[6], [x23]
    ld1 { v31.b }[6], [x26]
    ld1 { v30.b }[6], [x22]
    ld1 { v29.b }[6], [x28]
    ld1 { v28.b }[6], [x21]
    ld1 { v27.b }[6], [x25]
    ld1 { v26.b }[6], [x20]
    ld1 { v25.b }[6], [x27]
    ld1 { v24.b }[6], [x8]
    ld1 { v23.b }[6], [x24]
    ld1 { v22.b }[6], [x7]
    b label_64
KAI_ASM_LABEL(label_61)  // 6 block: unique 21: Partial load: partial_0_4
    tbz x11, #0, label_64
    ld1 { v1.b }[4], [x12]
    ld1 { v0.b }[4], [x23]
    ld1 { v31.b }[4], [x26]
    ld1 { v30.b }[4], [x22]
    ld1 { v29.b }[4], [x28]
    ld1 { v28.b }[4], [x21]
    ld1 { v27.b }[4], [x25]
    ld1 { v26.b }[4], [x20]
    ld1 { v25.b }[4], [x27]
    ld1 { v24.b }[4], [x8]
    ld1 { v23.b }[4], [x24]
    ld1 { v22.b }[4], [x7]
    b label_64
KAI_ASM_LABEL(label_62)  // 6 block: unique 21: Partial load: partial_1_0
    tbz x11, #1, label_63
    ldr h1, [x12], #0x2
    ldr h0, [x23], #0x2
    ldr h31, [x26], #0x2
    ldr h30, [x22], #0x2
    ldr h29, [x28], #0x2
    ldr h28, [x21], #0x2
    ldr h27, [x25], #0x2
    ldr h26, [x20], #0x2
    ldr h25, [x27], #0x2
    ldr h24, [x8], #0x2
    ldr h23, [x24], #0x2
    ldr h22, [x7], #0x2
    tbz x11, #0, label_64
    ld1 { v1.b }[2], [x12]
    ld1 { v0.b }[2], [x23]
    ld1 { v31.b }[2], [x26]
    ld1 { v30.b }[2], [x22]
    ld1 { v29.b }[2], [x28]
    ld1 { v28.b }[2], [x21]
    ld1 { v27.b }[2], [x25]
    ld1 { v26.b }[2], [x20]
    ld1 { v25.b }[2], [x27]
    ld1 { v24.b }[2], [x8]
    ld1 { v23.b }[2], [x24]
    ld1 { v22.b }[2], [x7]
    b label_64
KAI_ASM_LABEL(label_63)  // 6 block: unique 21: Partial load: partial_0_0
    ldr b1, [x12, #0]
    ldr b0, [x23, #0]
    ldr b31, [x26, #0]
    ldr b30, [x22, #0]
    ldr b29, [x28, #0]
    ldr b28, [x21, #0]
    ldr b27, [x25, #0]
    ldr b26, [x20, #0]
    ldr b25, [x27, #0]
    ldr b24, [x8, #0]
    ldr b23, [x24, #0]
    ldr b22, [x7, #0]
KAI_ASM_LABEL(label_64)  // 6 block: unique 21: Partial load: Done
    zip1 v21.2d, v1.2d, v0.2d
    zip1 v20.2d, v31.2d, v30.2d
    zip1 v19.2d, v29.2d, v28.2d
    zip1 v18.2d, v27.2d, v26.2d
    zip1 v17.2d, v25.2d, v24.2d
    zip1 v16.2d, v23.2d, v22.2d
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_66
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_65)  // 6 block: unique 22: unique 23: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_65
KAI_ASM_LABEL(label_66)  // 6 block: unique 22: unique 23: Padding done
    cmp x11, #0x8
    add x9, x9, x5, LSL #4
    ble label_69
    zip2 v21.2d, v1.2d, v0.2d
    zip2 v20.2d, v31.2d, v30.2d
    zip2 v19.2d, v29.2d, v28.2d
    zip2 v18.2d, v27.2d, v26.2d
    zip2 v17.2d, v25.2d, v24.2d
    zip2 v16.2d, v23.2d, v22.2d
    str q21, [x9, #0]
    str q20, [x9, #0x10]
    str q19, [x9, #0x20]
    str q18, [x9, #0x30]
    str q17, [x9, #0x40]
    str q16, [x9, #0x50]
    cbz x10, label_68
    add x6, x9, #0x60
    mov x3, x10
KAI_ASM_LABEL(label_67)  // 6 block: unique 22: unique 24: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_67
KAI_ASM_LABEL(label_68)  // 6 block: unique 22: unique 24: Padding done
    add x9, x9, x5, LSL #4
KAI_ASM_LABEL(label_69)  // 6 block: unique 22: Done
KAI_ASM_LABEL(label_70)  // 6 block: Oddments skip
    add x0, x0, #0x60
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x9, #0x60
    ldr x16, [SP, #0x20]
    b label_210
KAI_ASM_LABEL(label_71)  // 6 skip
    cmp x17, #0xa
    blt label_94
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
    cmp x9, #0x10
    add x22, x10, x4
    add x21, x25, x4
    add x20, x26, x4
    add x8, x24, x4
    add x7, x23, x4
    add x1, x23, x4, LSL #1
    blt label_79
KAI_ASM_LABEL(label_72)  // 5 block: Column loop
    ldr q30, [x10], #0x10
    ldr q29, [x22], #0x10
    ldr q28, [x25], #0x10
    ldr q27, [x21], #0x10
    ldr q26, [x26], #0x10
    ldr q25, [x20], #0x10
    ldr q24, [x24], #0x10
    ldr q23, [x8], #0x10
    ldr q22, [x23], #0x10
    ldr q21, [x7], #0x10
    zip1 v20.2d, v30.2d, v29.2d
    zip1 v19.2d, v28.2d, v27.2d
    zip1 v18.2d, v26.2d, v25.2d
    zip1 v17.2d, v24.2d, v23.2d
    zip1 v16.2d, v22.2d, v21.2d
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_75
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_74)  // 5 block: unique 26: unique 27: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_74
KAI_ASM_LABEL(label_75)  // 5 block: unique 26: unique 27: Padding done
    add x27, x27, x5, LSL #4
    zip2 v20.2d, v30.2d, v29.2d
    zip2 v19.2d, v28.2d, v27.2d
    zip2 v18.2d, v26.2d, v25.2d
    zip2 v17.2d, v24.2d, v23.2d
    zip2 v16.2d, v22.2d, v21.2d
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_77
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_76)  // 5 block: unique 26: unique 28: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_76
KAI_ASM_LABEL(label_77)  // 5 block: unique 26: unique 28: Padding done
    add x27, x27, x5, LSL #4
    sub x9, x9, #0x10
    cmp x9, #0x10
    bge label_72
KAI_ASM_LABEL(label_79)  // 5 block: Loop skip
    cbz x9, label_93
    tbz x9, #3, label_83
    ldr d30, [x10], #0x8
    ldr d29, [x22], #0x8
    ldr d28, [x25], #0x8
    ldr d27, [x21], #0x8
    ldr d26, [x26], #0x8
    ldr d25, [x20], #0x8
    ldr d24, [x24], #0x8
    ldr d23, [x8], #0x8
    ldr d22, [x23], #0x8
    ldr d21, [x7], #0x8
    tbz x9, #2, label_81
    ld1 { v30.s }[2], [x10], #0x4
    ld1 { v29.s }[2], [x22], #0x4
    ld1 { v28.s }[2], [x25], #0x4
    ld1 { v27.s }[2], [x21], #0x4
    ld1 { v26.s }[2], [x26], #0x4
    ld1 { v25.s }[2], [x20], #0x4
    ld1 { v24.s }[2], [x24], #0x4
    ld1 { v23.s }[2], [x8], #0x4
    ld1 { v22.s }[2], [x23], #0x4
    ld1 { v21.s }[2], [x7], #0x4
    tbz x9, #1, label_80
    ld1 { v30.h }[6], [x10], #0x2
    ld1 { v29.h }[6], [x22], #0x2
    ld1 { v28.h }[6], [x25], #0x2
    ld1 { v27.h }[6], [x21], #0x2
    ld1 { v26.h }[6], [x26], #0x2
    ld1 { v25.h }[6], [x20], #0x2
    ld1 { v24.h }[6], [x24], #0x2
    ld1 { v23.h }[6], [x8], #0x2
    ld1 { v22.h }[6], [x23], #0x2
    ld1 { v21.h }[6], [x7], #0x2
    tbz x9, #0, label_87
    ld1 { v30.b }[14], [x10]
    ld1 { v29.b }[14], [x22]
    ld1 { v28.b }[14], [x25]
    ld1 { v27.b }[14], [x21]
    ld1 { v26.b }[14], [x26]
    ld1 { v25.b }[14], [x20]
    ld1 { v24.b }[14], [x24]
    ld1 { v23.b }[14], [x8]
    ld1 { v22.b }[14], [x23]
    ld1 { v21.b }[14], [x7]
    b label_87
KAI_ASM_LABEL(label_80)  // 5 block: unique 29: Partial load: partial_0_12
    tbz x9, #0, label_87
    ld1 { v30.b }[12], [x10]
    ld1 { v29.b }[12], [x22]
    ld1 { v28.b }[12], [x25]
    ld1 { v27.b }[12], [x21]
    ld1 { v26.b }[12], [x26]
    ld1 { v25.b }[12], [x20]
    ld1 { v24.b }[12], [x24]
    ld1 { v23.b }[12], [x8]
    ld1 { v22.b }[12], [x23]
    ld1 { v21.b }[12], [x7]
    b label_87
KAI_ASM_LABEL(label_81)  // 5 block: unique 29: Partial load: partial_1_8
    tbz x9, #1, label_82
    ld1 { v30.h }[4], [x10], #0x2
    ld1 { v29.h }[4], [x22], #0x2
    ld1 { v28.h }[4], [x25], #0x2
    ld1 { v27.h }[4], [x21], #0x2
    ld1 { v26.h }[4], [x26], #0x2
    ld1 { v25.h }[4], [x20], #0x2
    ld1 { v24.h }[4], [x24], #0x2
    ld1 { v23.h }[4], [x8], #0x2
    ld1 { v22.h }[4], [x23], #0x2
    ld1 { v21.h }[4], [x7], #0x2
    tbz x9, #0, label_87
    ld1 { v30.b }[10], [x10]
    ld1 { v29.b }[10], [x22]
    ld1 { v28.b }[10], [x25]
    ld1 { v27.b }[10], [x21]
    ld1 { v26.b }[10], [x26]
    ld1 { v25.b }[10], [x20]
    ld1 { v24.b }[10], [x24]
    ld1 { v23.b }[10], [x8]
    ld1 { v22.b }[10], [x23]
    ld1 { v21.b }[10], [x7]
    b label_87
KAI_ASM_LABEL(label_82)  // 5 block: unique 29: Partial load: partial_0_8
    tbz x9, #0, label_87
    ld1 { v30.b }[8], [x10]
    ld1 { v29.b }[8], [x22]
    ld1 { v28.b }[8], [x25]
    ld1 { v27.b }[8], [x21]
    ld1 { v26.b }[8], [x26]
    ld1 { v25.b }[8], [x20]
    ld1 { v24.b }[8], [x24]
    ld1 { v23.b }[8], [x8]
    ld1 { v22.b }[8], [x23]
    ld1 { v21.b }[8], [x7]
    b label_87
KAI_ASM_LABEL(label_83)  // 5 block: unique 29: Partial load: partial_2_0
    tbz x9, #2, label_85
    ldr s30, [x10], #0x4
    ldr s29, [x22], #0x4
    ldr s28, [x25], #0x4
    ldr s27, [x21], #0x4
    ldr s26, [x26], #0x4
    ldr s25, [x20], #0x4
    ldr s24, [x24], #0x4
    ldr s23, [x8], #0x4
    ldr s22, [x23], #0x4
    ldr s21, [x7], #0x4
    tbz x9, #1, label_84
    ld1 { v30.h }[2], [x10], #0x2
    ld1 { v29.h }[2], [x22], #0x2
    ld1 { v28.h }[2], [x25], #0x2
    ld1 { v27.h }[2], [x21], #0x2
    ld1 { v26.h }[2], [x26], #0x2
    ld1 { v25.h }[2], [x20], #0x2
    ld1 { v24.h }[2], [x24], #0x2
    ld1 { v23.h }[2], [x8], #0x2
    ld1 { v22.h }[2], [x23], #0x2
    ld1 { v21.h }[2], [x7], #0x2
    tbz x9, #0, label_87
    ld1 { v30.b }[6], [x10]
    ld1 { v29.b }[6], [x22]
    ld1 { v28.b }[6], [x25]
    ld1 { v27.b }[6], [x21]
    ld1 { v26.b }[6], [x26]
    ld1 { v25.b }[6], [x20]
    ld1 { v24.b }[6], [x24]
    ld1 { v23.b }[6], [x8]
    ld1 { v22.b }[6], [x23]
    ld1 { v21.b }[6], [x7]
    b label_87
KAI_ASM_LABEL(label_84)  // 5 block: unique 29: Partial load: partial_0_4
    tbz x9, #0, label_87
    ld1 { v30.b }[4], [x10]
    ld1 { v29.b }[4], [x22]
    ld1 { v28.b }[4], [x25]
    ld1 { v27.b }[4], [x21]
    ld1 { v26.b }[4], [x26]
    ld1 { v25.b }[4], [x20]
    ld1 { v24.b }[4], [x24]
    ld1 { v23.b }[4], [x8]
    ld1 { v22.b }[4], [x23]
    ld1 { v21.b }[4], [x7]
    b label_87
KAI_ASM_LABEL(label_85)  // 5 block: unique 29: Partial load: partial_1_0
    tbz x9, #1, label_86
    ldr h30, [x10], #0x2
    ldr h29, [x22], #0x2
    ldr h28, [x25], #0x2
    ldr h27, [x21], #0x2
    ldr h26, [x26], #0x2
    ldr h25, [x20], #0x2
    ldr h24, [x24], #0x2
    ldr h23, [x8], #0x2
    ldr h22, [x23], #0x2
    ldr h21, [x7], #0x2
    tbz x9, #0, label_87
    ld1 { v30.b }[2], [x10]
    ld1 { v29.b }[2], [x22]
    ld1 { v28.b }[2], [x25]
    ld1 { v27.b }[2], [x21]
    ld1 { v26.b }[2], [x26]
    ld1 { v25.b }[2], [x20]
    ld1 { v24.b }[2], [x24]
    ld1 { v23.b }[2], [x8]
    ld1 { v22.b }[2], [x23]
    ld1 { v21.b }[2], [x7]
    b label_87
KAI_ASM_LABEL(label_86)  // 5 block: unique 29: Partial load: partial_0_0
    ldr b30, [x10, #0]
    ldr b29, [x22, #0]
    ldr b28, [x25, #0]
    ldr b27, [x21, #0]
    ldr b26, [x26, #0]
    ldr b25, [x20, #0]
    ldr b24, [x24, #0]
    ldr b23, [x8, #0]
    ldr b22, [x23, #0]
    ldr b21, [x7, #0]
KAI_ASM_LABEL(label_87)  // 5 block: unique 29: Partial load: Done
    zip1 v20.2d, v30.2d, v29.2d
    zip1 v19.2d, v28.2d, v27.2d
    zip1 v18.2d, v26.2d, v25.2d
    zip1 v17.2d, v24.2d, v23.2d
    zip1 v16.2d, v22.2d, v21.2d
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_89
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_88)  // 5 block: unique 30: unique 31: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_88
KAI_ASM_LABEL(label_89)  // 5 block: unique 30: unique 31: Padding done
    cmp x9, #0x8
    add x27, x27, x5, LSL #4
    ble label_92
    zip2 v20.2d, v30.2d, v29.2d
    zip2 v19.2d, v28.2d, v27.2d
    zip2 v18.2d, v26.2d, v25.2d
    zip2 v17.2d, v24.2d, v23.2d
    zip2 v16.2d, v22.2d, v21.2d
    str q20, [x27, #0]
    str q19, [x27, #0x10]
    str q18, [x27, #0x20]
    str q17, [x27, #0x30]
    str q16, [x27, #0x40]
    cbz x28, label_91
    add x6, x27, #0x50
    mov x3, x28
KAI_ASM_LABEL(label_90)  // 5 block: unique 30: unique 32: Padding loop
    subs x3, x3, #0x1
    str q6, [x6], #0x10
    bne label_90
KAI_ASM_LABEL(label_91)  // 5 block: unique 30: unique 32: Padding done
    add x27, x27, x5, LSL #4
KAI_ASM_LABEL(label_92)  // 5 block: unique 30: Done
KAI_ASM_LABEL(label_93)  // 5 block: Oddments skip
    add x0, x0, #0x50
    ldr x3, [SP, #0x10]
    ldr x17, [SP, #0x18]
    add x8, x27, #0x50
    ldr x16, [SP, #0x20]
    b label_210
KAI_ASM_LABEL(label_94)  // 5 skip
    cmp x17, #0x8
    blt label_117
    mov x9, x1
    mov x28, x2
    sub x16, x16, #0x4
    subs x17, x17, #0x8
    csel x27, x16, XZR, EQ
    mov x26, x0
    add x25, x9, x4, LSL #2
    add x24, x9, x4, LSL #1
    add x23, x25, x4, LSL #1
    cmp x28, #0x10
    add x22, x9, x4
    add x21, x24, x4
    add x20, x25, x4
    add x8, x23, x4
    add x1, x9, x4, LSL #3
    blt label_102
KAI_ASM_LABEL(label_95)  // 4 block: Column loop
    ldr q27, [x9], #0x10
    ldr q26, [x22], #0x10
    ldr q25, [x24], #0x10
    ldr q24, [x21], #0x10
    ldr q23, [x25], #0x10
    ldr q22, [x20], #0x10
    ldr q21, [x23], #0x10
    ldr q20, [x8], #0x10
    zip1 v19.2d, v27.2d, v26.2d
    zip1 v18.2d, v25.2d, v24.2d
    zip1 v17.2d, v23.2d, v22.2d
    zip1 v16.2d, v21.2d, v20.2d
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_98
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_97)  // 4 block: unique 34: unique 35: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_97
KAI_ASM_LABEL(label_98)  // 4 block: unique 34: unique 35: Padding done
    add x26, x26, x5, LSL #4
    zip2 v19.2d, v27.2d, v26.2d
    zip2 v18.2d, v25.2d, v24.2d
    zip2 v17.2d, v23.2d, v22.2d
    zip2 v16.2d, v21.2d, v20.2d
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_100
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_99)  // 4 block: unique 34: unique 36: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_99
KAI_ASM_LABEL(label_100)  // 4 block: unique 34: unique 36: Padding done
    add x26, x26, x5, LSL #4
    sub x28, x28, #0x10
    cmp x28, #0x10
    bge label_95
KAI_ASM_LABEL(label_102)  // 4 block: Loop skip
    cbz x28, label_116
    tbz x28, #3, label_106
    ldr d27, [x9], #0x8
    ldr d26, [x22], #0x8
    ldr d25, [x24], #0x8
    ldr d24, [x21], #0x8
    ldr d23, [x25], #0x8
    ldr d22, [x20], #0x8
    ldr d21, [x23], #0x8
    ldr d20, [x8], #0x8
    tbz x28, #2, label_104
    ld1 { v27.s }[2], [x9], #0x4
    ld1 { v26.s }[2], [x22], #0x4
    ld1 { v25.s }[2], [x24], #0x4
    ld1 { v24.s }[2], [x21], #0x4
    ld1 { v23.s }[2], [x25], #0x4
    ld1 { v22.s }[2], [x20], #0x4
    ld1 { v21.s }[2], [x23], #0x4
    ld1 { v20.s }[2], [x8], #0x4
    tbz x28, #1, label_103
    ld1 { v27.h }[6], [x9], #0x2
    ld1 { v26.h }[6], [x22], #0x2
    ld1 { v25.h }[6], [x24], #0x2
    ld1 { v24.h }[6], [x21], #0x2
    ld1 { v23.h }[6], [x25], #0x2
    ld1 { v22.h }[6], [x20], #0x2
    ld1 { v21.h }[6], [x23], #0x2
    ld1 { v20.h }[6], [x8], #0x2
    tbz x28, #0, label_110
    ld1 { v27.b }[14], [x9]
    ld1 { v26.b }[14], [x22]
    ld1 { v25.b }[14], [x24]
    ld1 { v24.b }[14], [x21]
    ld1 { v23.b }[14], [x25]
    ld1 { v22.b }[14], [x20]
    ld1 { v21.b }[14], [x23]
    ld1 { v20.b }[14], [x8]
    b label_110
KAI_ASM_LABEL(label_103)  // 4 block: unique 37: Partial load: partial_0_12
    tbz x28, #0, label_110
    ld1 { v27.b }[12], [x9]
    ld1 { v26.b }[12], [x22]
    ld1 { v25.b }[12], [x24]
    ld1 { v24.b }[12], [x21]
    ld1 { v23.b }[12], [x25]
    ld1 { v22.b }[12], [x20]
    ld1 { v21.b }[12], [x23]
    ld1 { v20.b }[12], [x8]
    b label_110
KAI_ASM_LABEL(label_104)  // 4 block: unique 37: Partial load: partial_1_8
    tbz x28, #1, label_105
    ld1 { v27.h }[4], [x9], #0x2
    ld1 { v26.h }[4], [x22], #0x2
    ld1 { v25.h }[4], [x24], #0x2
    ld1 { v24.h }[4], [x21], #0x2
    ld1 { v23.h }[4], [x25], #0x2
    ld1 { v22.h }[4], [x20], #0x2
    ld1 { v21.h }[4], [x23], #0x2
    ld1 { v20.h }[4], [x8], #0x2
    tbz x28, #0, label_110
    ld1 { v27.b }[10], [x9]
    ld1 { v26.b }[10], [x22]
    ld1 { v25.b }[10], [x24]
    ld1 { v24.b }[10], [x21]
    ld1 { v23.b }[10], [x25]
    ld1 { v22.b }[10], [x20]
    ld1 { v21.b }[10], [x23]
    ld1 { v20.b }[10], [x8]
    b label_110
KAI_ASM_LABEL(label_105)  // 4 block: unique 37: Partial load: partial_0_8
    tbz x28, #0, label_110
    ld1 { v27.b }[8], [x9]
    ld1 { v26.b }[8], [x22]
    ld1 { v25.b }[8], [x24]
    ld1 { v24.b }[8], [x21]
    ld1 { v23.b }[8], [x25]
    ld1 { v22.b }[8], [x20]
    ld1 { v21.b }[8], [x23]
    ld1 { v20.b }[8], [x8]
    b label_110
KAI_ASM_LABEL(label_106)  // 4 block: unique 37: Partial load: partial_2_0
    tbz x28, #2, label_108
    ldr s27, [x9], #0x4
    ldr s26, [x22], #0x4
    ldr s25, [x24], #0x4
    ldr s24, [x21], #0x4
    ldr s23, [x25], #0x4
    ldr s22, [x20], #0x4
    ldr s21, [x23], #0x4
    ldr s20, [x8], #0x4
    tbz x28, #1, label_107
    ld1 { v27.h }[2], [x9], #0x2
    ld1 { v26.h }[2], [x22], #0x2
    ld1 { v25.h }[2], [x24], #0x2
    ld1 { v24.h }[2], [x21], #0x2
    ld1 { v23.h }[2], [x25], #0x2
    ld1 { v22.h }[2], [x20], #0x2
    ld1 { v21.h }[2], [x23], #0x2
    ld1 { v20.h }[2], [x8], #0x2
    tbz x28, #0, label_110
    ld1 { v27.b }[6], [x9]
    ld1 { v26.b }[6], [x22]
    ld1 { v25.b }[6], [x24]
    ld1 { v24.b }[6], [x21]
    ld1 { v23.b }[6], [x25]
    ld1 { v22.b }[6], [x20]
    ld1 { v21.b }[6], [x23]
    ld1 { v20.b }[6], [x8]
    b label_110
KAI_ASM_LABEL(label_107)  // 4 block: unique 37: Partial load: partial_0_4
    tbz x28, #0, label_110
    ld1 { v27.b }[4], [x9]
    ld1 { v26.b }[4], [x22]
    ld1 { v25.b }[4], [x24]
    ld1 { v24.b }[4], [x21]
    ld1 { v23.b }[4], [x25]
    ld1 { v22.b }[4], [x20]
    ld1 { v21.b }[4], [x23]
    ld1 { v20.b }[4], [x8]
    b label_110
KAI_ASM_LABEL(label_108)  // 4 block: unique 37: Partial load: partial_1_0
    tbz x28, #1, label_109
    ldr h27, [x9], #0x2
    ldr h26, [x22], #0x2
    ldr h25, [x24], #0x2
    ldr h24, [x21], #0x2
    ldr h23, [x25], #0x2
    ldr h22, [x20], #0x2
    ldr h21, [x23], #0x2
    ldr h20, [x8], #0x2
    tbz x28, #0, label_110
    ld1 { v27.b }[2], [x9]
    ld1 { v26.b }[2], [x22]
    ld1 { v25.b }[2], [x24]
    ld1 { v24.b }[2], [x21]
    ld1 { v23.b }[2], [x25]
    ld1 { v22.b }[2], [x20]
    ld1 { v21.b }[2], [x23]
    ld1 { v20.b }[2], [x8]
    b label_110
KAI_ASM_LABEL(label_109)  // 4 block: unique 37: Partial load: partial_0_0
    ldr b27, [x9, #0]
    ldr b26, [x22, #0]
    ldr b25, [x24, #0]
    ldr b24, [x21, #0]
    ldr b23, [x25, #0]
    ldr b22, [x20, #0]
    ldr b21, [x23, #0]
    ldr b20, [x8, #0]
KAI_ASM_LABEL(label_110)  // 4 block: unique 37: Partial load: Done
    zip1 v19.2d, v27.2d, v26.2d
    zip1 v18.2d, v25.2d, v24.2d
    zip1 v17.2d, v23.2d, v22.2d
    zip1 v16.2d, v21.2d, v20.2d
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_112
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_111)  // 4 block: unique 38: unique 39: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_111
KAI_ASM_LABEL(label_112)  // 4 block: unique 38: unique 39: Padding done
    cmp x28, #0x8
    add x26, x26, x5, LSL #4
    ble label_115
    zip2 v19.2d, v27.2d, v26.2d
    zip2 v18.2d, v25.2d, v24.2d
    zip2 v17.2d, v23.2d, v22.2d
    zip2 v16.2d, v21.2d, v20.2d
    str q19, [x26, #0]
    str q18, [x26, #0x10]
    str q17, [x26, #0x20]
    str q16, [x26, #0x30]
    cbz x27, label_114
    add x7, x26, #0x40
    mov x6, x27
KAI_ASM_LABEL(label_113)  // 4 block: unique 38: unique 40: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_113
KAI_ASM_LABEL(label_114)  // 4 block: unique 38: unique 40: Padding done
    add x26, x26, x5, LSL #4
KAI_ASM_LABEL(label_115)  // 4 block: unique 38: Done
KAI_ASM_LABEL(label_116)  // 4 block: Oddments skip
    add x0, x0, #0x40
    add x8, x26, #0x40
    b label_210
KAI_ASM_LABEL(label_117)  // 4 skip
    cmp x17, #0x6
    blt label_140
    mov x27, x1
    mov x26, x2
    sub x16, x16, #0x3
    subs x17, x17, #0x6
    csel x25, x16, XZR, EQ
    mov x24, x0
    add x23, x27, x4, LSL #1
    add x22, x27, x4, LSL #2
    cmp x26, #0x10
    add x21, x27, x4
    add x20, x23, x4
    add x8, x22, x4
    add x1, x22, x4, LSL #1
    blt label_125
KAI_ASM_LABEL(label_118)  // 3 block: Column loop
    ldr q24, [x27], #0x10
    ldr q23, [x21], #0x10
    ldr q22, [x23], #0x10
    ldr q21, [x20], #0x10
    ldr q20, [x22], #0x10
    ldr q19, [x8], #0x10
    zip1 v18.2d, v24.2d, v23.2d
    zip1 v17.2d, v22.2d, v21.2d
    zip1 v16.2d, v20.2d, v19.2d
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_121
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_120)  // 3 block: unique 42: unique 43: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_120
KAI_ASM_LABEL(label_121)  // 3 block: unique 42: unique 43: Padding done
    add x24, x24, x5, LSL #4
    zip2 v18.2d, v24.2d, v23.2d
    zip2 v17.2d, v22.2d, v21.2d
    zip2 v16.2d, v20.2d, v19.2d
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_123
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_122)  // 3 block: unique 42: unique 44: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_122
KAI_ASM_LABEL(label_123)  // 3 block: unique 42: unique 44: Padding done
    add x24, x24, x5, LSL #4
    sub x26, x26, #0x10
    cmp x26, #0x10
    bge label_118
KAI_ASM_LABEL(label_125)  // 3 block: Loop skip
    cbz x26, label_139
    tbz x26, #3, label_129
    ldr d24, [x27], #0x8
    ldr d23, [x21], #0x8
    ldr d22, [x23], #0x8
    ldr d21, [x20], #0x8
    ldr d20, [x22], #0x8
    ldr d19, [x8], #0x8
    tbz x26, #2, label_127
    ld1 { v24.s }[2], [x27], #0x4
    ld1 { v23.s }[2], [x21], #0x4
    ld1 { v22.s }[2], [x23], #0x4
    ld1 { v21.s }[2], [x20], #0x4
    ld1 { v20.s }[2], [x22], #0x4
    ld1 { v19.s }[2], [x8], #0x4
    tbz x26, #1, label_126
    ld1 { v24.h }[6], [x27], #0x2
    ld1 { v23.h }[6], [x21], #0x2
    ld1 { v22.h }[6], [x23], #0x2
    ld1 { v21.h }[6], [x20], #0x2
    ld1 { v20.h }[6], [x22], #0x2
    ld1 { v19.h }[6], [x8], #0x2
    tbz x26, #0, label_133
    ld1 { v24.b }[14], [x27]
    ld1 { v23.b }[14], [x21]
    ld1 { v22.b }[14], [x23]
    ld1 { v21.b }[14], [x20]
    ld1 { v20.b }[14], [x22]
    ld1 { v19.b }[14], [x8]
    b label_133
KAI_ASM_LABEL(label_126)  // 3 block: unique 45: Partial load: partial_0_12
    tbz x26, #0, label_133
    ld1 { v24.b }[12], [x27]
    ld1 { v23.b }[12], [x21]
    ld1 { v22.b }[12], [x23]
    ld1 { v21.b }[12], [x20]
    ld1 { v20.b }[12], [x22]
    ld1 { v19.b }[12], [x8]
    b label_133
KAI_ASM_LABEL(label_127)  // 3 block: unique 45: Partial load: partial_1_8
    tbz x26, #1, label_128
    ld1 { v24.h }[4], [x27], #0x2
    ld1 { v23.h }[4], [x21], #0x2
    ld1 { v22.h }[4], [x23], #0x2
    ld1 { v21.h }[4], [x20], #0x2
    ld1 { v20.h }[4], [x22], #0x2
    ld1 { v19.h }[4], [x8], #0x2
    tbz x26, #0, label_133
    ld1 { v24.b }[10], [x27]
    ld1 { v23.b }[10], [x21]
    ld1 { v22.b }[10], [x23]
    ld1 { v21.b }[10], [x20]
    ld1 { v20.b }[10], [x22]
    ld1 { v19.b }[10], [x8]
    b label_133
KAI_ASM_LABEL(label_128)  // 3 block: unique 45: Partial load: partial_0_8
    tbz x26, #0, label_133
    ld1 { v24.b }[8], [x27]
    ld1 { v23.b }[8], [x21]
    ld1 { v22.b }[8], [x23]
    ld1 { v21.b }[8], [x20]
    ld1 { v20.b }[8], [x22]
    ld1 { v19.b }[8], [x8]
    b label_133
KAI_ASM_LABEL(label_129)  // 3 block: unique 45: Partial load: partial_2_0
    tbz x26, #2, label_131
    ldr s24, [x27], #0x4
    ldr s23, [x21], #0x4
    ldr s22, [x23], #0x4
    ldr s21, [x20], #0x4
    ldr s20, [x22], #0x4
    ldr s19, [x8], #0x4
    tbz x26, #1, label_130
    ld1 { v24.h }[2], [x27], #0x2
    ld1 { v23.h }[2], [x21], #0x2
    ld1 { v22.h }[2], [x23], #0x2
    ld1 { v21.h }[2], [x20], #0x2
    ld1 { v20.h }[2], [x22], #0x2
    ld1 { v19.h }[2], [x8], #0x2
    tbz x26, #0, label_133
    ld1 { v24.b }[6], [x27]
    ld1 { v23.b }[6], [x21]
    ld1 { v22.b }[6], [x23]
    ld1 { v21.b }[6], [x20]
    ld1 { v20.b }[6], [x22]
    ld1 { v19.b }[6], [x8]
    b label_133
KAI_ASM_LABEL(label_130)  // 3 block: unique 45: Partial load: partial_0_4
    tbz x26, #0, label_133
    ld1 { v24.b }[4], [x27]
    ld1 { v23.b }[4], [x21]
    ld1 { v22.b }[4], [x23]
    ld1 { v21.b }[4], [x20]
    ld1 { v20.b }[4], [x22]
    ld1 { v19.b }[4], [x8]
    b label_133
KAI_ASM_LABEL(label_131)  // 3 block: unique 45: Partial load: partial_1_0
    tbz x26, #1, label_132
    ldr h24, [x27], #0x2
    ldr h23, [x21], #0x2
    ldr h22, [x23], #0x2
    ldr h21, [x20], #0x2
    ldr h20, [x22], #0x2
    ldr h19, [x8], #0x2
    tbz x26, #0, label_133
    ld1 { v24.b }[2], [x27]
    ld1 { v23.b }[2], [x21]
    ld1 { v22.b }[2], [x23]
    ld1 { v21.b }[2], [x20]
    ld1 { v20.b }[2], [x22]
    ld1 { v19.b }[2], [x8]
    b label_133
KAI_ASM_LABEL(label_132)  // 3 block: unique 45: Partial load: partial_0_0
    ldr b24, [x27, #0]
    ldr b23, [x21, #0]
    ldr b22, [x23, #0]
    ldr b21, [x20, #0]
    ldr b20, [x22, #0]
    ldr b19, [x8, #0]
KAI_ASM_LABEL(label_133)  // 3 block: unique 45: Partial load: Done
    zip1 v18.2d, v24.2d, v23.2d
    zip1 v17.2d, v22.2d, v21.2d
    zip1 v16.2d, v20.2d, v19.2d
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_135
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_134)  // 3 block: unique 46: unique 47: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_134
KAI_ASM_LABEL(label_135)  // 3 block: unique 46: unique 47: Padding done
    cmp x26, #0x8
    add x24, x24, x5, LSL #4
    ble label_138
    zip2 v18.2d, v24.2d, v23.2d
    zip2 v17.2d, v22.2d, v21.2d
    zip2 v16.2d, v20.2d, v19.2d
    str q18, [x24, #0]
    str q17, [x24, #0x10]
    str q16, [x24, #0x20]
    cbz x25, label_137
    add x7, x24, #0x30
    mov x6, x25
KAI_ASM_LABEL(label_136)  // 3 block: unique 46: unique 48: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_136
KAI_ASM_LABEL(label_137)  // 3 block: unique 46: unique 48: Padding done
    add x24, x24, x5, LSL #4
KAI_ASM_LABEL(label_138)  // 3 block: unique 46: Done
KAI_ASM_LABEL(label_139)  // 3 block: Oddments skip
    add x0, x0, #0x30
    add x8, x24, #0x30
    b label_210
KAI_ASM_LABEL(label_140)  // 3 skip
    cmp x17, #0x4
    blt label_163
    mov x25, x1
    mov x24, x2
    sub x16, x16, #0x2
    subs x17, x17, #0x4
    csel x23, x16, XZR, EQ
    mov x22, x0
    add x21, x25, x4, LSL #1
    cmp x24, #0x10
    add x20, x25, x4
    add x8, x21, x4
    add x1, x25, x4, LSL #2
    blt label_148
KAI_ASM_LABEL(label_141)  // 2 block: Column loop
    ldr q21, [x25], #0x10
    ldr q20, [x20], #0x10
    ldr q19, [x21], #0x10
    ldr q18, [x8], #0x10
    zip1 v17.2d, v21.2d, v20.2d
    zip1 v16.2d, v19.2d, v18.2d
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_144
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_143)  // 2 block: unique 50: unique 51: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_143
KAI_ASM_LABEL(label_144)  // 2 block: unique 50: unique 51: Padding done
    add x22, x22, x5, LSL #4
    zip2 v17.2d, v21.2d, v20.2d
    zip2 v16.2d, v19.2d, v18.2d
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_146
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_145)  // 2 block: unique 50: unique 52: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_145
KAI_ASM_LABEL(label_146)  // 2 block: unique 50: unique 52: Padding done
    add x22, x22, x5, LSL #4
    sub x24, x24, #0x10
    cmp x24, #0x10
    bge label_141
KAI_ASM_LABEL(label_148)  // 2 block: Loop skip
    cbz x24, label_162
    tbz x24, #3, label_152
    ldr d21, [x25], #0x8
    ldr d20, [x20], #0x8
    ldr d19, [x21], #0x8
    ldr d18, [x8], #0x8
    tbz x24, #2, label_150
    ld1 { v21.s }[2], [x25], #0x4
    ld1 { v20.s }[2], [x20], #0x4
    ld1 { v19.s }[2], [x21], #0x4
    ld1 { v18.s }[2], [x8], #0x4
    tbz x24, #1, label_149
    ld1 { v21.h }[6], [x25], #0x2
    ld1 { v20.h }[6], [x20], #0x2
    ld1 { v19.h }[6], [x21], #0x2
    ld1 { v18.h }[6], [x8], #0x2
    tbz x24, #0, label_156
    ld1 { v21.b }[14], [x25]
    ld1 { v20.b }[14], [x20]
    ld1 { v19.b }[14], [x21]
    ld1 { v18.b }[14], [x8]
    b label_156
KAI_ASM_LABEL(label_149)  // 2 block: unique 53: Partial load: partial_0_12
    tbz x24, #0, label_156
    ld1 { v21.b }[12], [x25]
    ld1 { v20.b }[12], [x20]
    ld1 { v19.b }[12], [x21]
    ld1 { v18.b }[12], [x8]
    b label_156
KAI_ASM_LABEL(label_150)  // 2 block: unique 53: Partial load: partial_1_8
    tbz x24, #1, label_151
    ld1 { v21.h }[4], [x25], #0x2
    ld1 { v20.h }[4], [x20], #0x2
    ld1 { v19.h }[4], [x21], #0x2
    ld1 { v18.h }[4], [x8], #0x2
    tbz x24, #0, label_156
    ld1 { v21.b }[10], [x25]
    ld1 { v20.b }[10], [x20]
    ld1 { v19.b }[10], [x21]
    ld1 { v18.b }[10], [x8]
    b label_156
KAI_ASM_LABEL(label_151)  // 2 block: unique 53: Partial load: partial_0_8
    tbz x24, #0, label_156
    ld1 { v21.b }[8], [x25]
    ld1 { v20.b }[8], [x20]
    ld1 { v19.b }[8], [x21]
    ld1 { v18.b }[8], [x8]
    b label_156
KAI_ASM_LABEL(label_152)  // 2 block: unique 53: Partial load: partial_2_0
    tbz x24, #2, label_154
    ldr s21, [x25], #0x4
    ldr s20, [x20], #0x4
    ldr s19, [x21], #0x4
    ldr s18, [x8], #0x4
    tbz x24, #1, label_153
    ld1 { v21.h }[2], [x25], #0x2
    ld1 { v20.h }[2], [x20], #0x2
    ld1 { v19.h }[2], [x21], #0x2
    ld1 { v18.h }[2], [x8], #0x2
    tbz x24, #0, label_156
    ld1 { v21.b }[6], [x25]
    ld1 { v20.b }[6], [x20]
    ld1 { v19.b }[6], [x21]
    ld1 { v18.b }[6], [x8]
    b label_156
KAI_ASM_LABEL(label_153)  // 2 block: unique 53: Partial load: partial_0_4
    tbz x24, #0, label_156
    ld1 { v21.b }[4], [x25]
    ld1 { v20.b }[4], [x20]
    ld1 { v19.b }[4], [x21]
    ld1 { v18.b }[4], [x8]
    b label_156
KAI_ASM_LABEL(label_154)  // 2 block: unique 53: Partial load: partial_1_0
    tbz x24, #1, label_155
    ldr h21, [x25], #0x2
    ldr h20, [x20], #0x2
    ldr h19, [x21], #0x2
    ldr h18, [x8], #0x2
    tbz x24, #0, label_156
    ld1 { v21.b }[2], [x25]
    ld1 { v20.b }[2], [x20]
    ld1 { v19.b }[2], [x21]
    ld1 { v18.b }[2], [x8]
    b label_156
KAI_ASM_LABEL(label_155)  // 2 block: unique 53: Partial load: partial_0_0
    ldr b21, [x25, #0]
    ldr b20, [x20, #0]
    ldr b19, [x21, #0]
    ldr b18, [x8, #0]
KAI_ASM_LABEL(label_156)  // 2 block: unique 53: Partial load: Done
    zip1 v17.2d, v21.2d, v20.2d
    zip1 v16.2d, v19.2d, v18.2d
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_158
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_157)  // 2 block: unique 54: unique 55: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_157
KAI_ASM_LABEL(label_158)  // 2 block: unique 54: unique 55: Padding done
    cmp x24, #0x8
    add x22, x22, x5, LSL #4
    ble label_161
    zip2 v17.2d, v21.2d, v20.2d
    zip2 v16.2d, v19.2d, v18.2d
    str q17, [x22, #0]
    str q16, [x22, #0x10]
    cbz x23, label_160
    add x7, x22, #0x20
    mov x6, x23
KAI_ASM_LABEL(label_159)  // 2 block: unique 54: unique 56: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_159
KAI_ASM_LABEL(label_160)  // 2 block: unique 54: unique 56: Padding done
    add x22, x22, x5, LSL #4
KAI_ASM_LABEL(label_161)  // 2 block: unique 54: Done
KAI_ASM_LABEL(label_162)  // 2 block: Oddments skip
    add x0, x0, #0x20
    add x8, x22, #0x20
    b label_210
KAI_ASM_LABEL(label_163)  // 2 skip
    cmp x17, #0x2
    blt label_186
    mov x23, x2
    mov x22, x1
    sub x16, x16, #0x1
    subs x17, x17, #0x2
    csel x21, x16, XZR, EQ
    mov x20, x0
    cmp x23, #0x10
    add x8, x22, x4
    add x1, x22, x4, LSL #1
    blt label_171
KAI_ASM_LABEL(label_164)  // 1 block: Column loop
    ldr q18, [x22], #0x10
    ldr q17, [x8], #0x10
    zip1 v16.2d, v18.2d, v17.2d
    str q16, [x20, #0]
    cbz x21, label_167
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_166)  // 1 block: unique 58: unique 59: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_166
KAI_ASM_LABEL(label_167)  // 1 block: unique 58: unique 59: Padding done
    add x20, x20, x5, LSL #4
    zip2 v16.2d, v18.2d, v17.2d
    str q16, [x20, #0]
    cbz x21, label_169
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_168)  // 1 block: unique 58: unique 60: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_168
KAI_ASM_LABEL(label_169)  // 1 block: unique 58: unique 60: Padding done
    add x20, x20, x5, LSL #4
    sub x23, x23, #0x10
    cmp x23, #0x10
    bge label_164
KAI_ASM_LABEL(label_171)  // 1 block: Loop skip
    cbz x23, label_185
    tbz x23, #3, label_175
    ldr d18, [x22], #0x8
    ldr d17, [x8], #0x8
    tbz x23, #2, label_173
    ld1 { v18.s }[2], [x22], #0x4
    ld1 { v17.s }[2], [x8], #0x4
    tbz x23, #1, label_172
    ld1 { v18.h }[6], [x22], #0x2
    ld1 { v17.h }[6], [x8], #0x2
    tbz x23, #0, label_179
    ld1 { v18.b }[14], [x22]
    ld1 { v17.b }[14], [x8]
    b label_179
KAI_ASM_LABEL(label_172)  // 1 block: unique 61: Partial load: partial_0_12
    tbz x23, #0, label_179
    ld1 { v18.b }[12], [x22]
    ld1 { v17.b }[12], [x8]
    b label_179
KAI_ASM_LABEL(label_173)  // 1 block: unique 61: Partial load: partial_1_8
    tbz x23, #1, label_174
    ld1 { v18.h }[4], [x22], #0x2
    ld1 { v17.h }[4], [x8], #0x2
    tbz x23, #0, label_179
    ld1 { v18.b }[10], [x22]
    ld1 { v17.b }[10], [x8]
    b label_179
KAI_ASM_LABEL(label_174)  // 1 block: unique 61: Partial load: partial_0_8
    tbz x23, #0, label_179
    ld1 { v18.b }[8], [x22]
    ld1 { v17.b }[8], [x8]
    b label_179
KAI_ASM_LABEL(label_175)  // 1 block: unique 61: Partial load: partial_2_0
    tbz x23, #2, label_177
    ldr s18, [x22], #0x4
    ldr s17, [x8], #0x4
    tbz x23, #1, label_176
    ld1 { v18.h }[2], [x22], #0x2
    ld1 { v17.h }[2], [x8], #0x2
    tbz x23, #0, label_179
    ld1 { v18.b }[6], [x22]
    ld1 { v17.b }[6], [x8]
    b label_179
KAI_ASM_LABEL(label_176)  // 1 block: unique 61: Partial load: partial_0_4
    tbz x23, #0, label_179
    ld1 { v18.b }[4], [x22]
    ld1 { v17.b }[4], [x8]
    b label_179
KAI_ASM_LABEL(label_177)  // 1 block: unique 61: Partial load: partial_1_0
    tbz x23, #1, label_178
    ldr h18, [x22], #0x2
    ldr h17, [x8], #0x2
    tbz x23, #0, label_179
    ld1 { v18.b }[2], [x22]
    ld1 { v17.b }[2], [x8]
    b label_179
KAI_ASM_LABEL(label_178)  // 1 block: unique 61: Partial load: partial_0_0
    ldr b18, [x22, #0]
    ldr b17, [x8, #0]
KAI_ASM_LABEL(label_179)  // 1 block: unique 61: Partial load: Done
    zip1 v16.2d, v18.2d, v17.2d
    str q16, [x20, #0]
    cbz x21, label_181
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_180)  // 1 block: unique 62: unique 63: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_180
KAI_ASM_LABEL(label_181)  // 1 block: unique 62: unique 63: Padding done
    cmp x23, #0x8
    add x20, x20, x5, LSL #4
    ble label_184
    zip2 v16.2d, v18.2d, v17.2d
    str q16, [x20, #0]
    cbz x21, label_183
    add x7, x20, #0x10
    mov x6, x21
KAI_ASM_LABEL(label_182)  // 1 block: unique 62: unique 64: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_182
KAI_ASM_LABEL(label_183)  // 1 block: unique 62: unique 64: Padding done
    add x20, x20, x5, LSL #4
KAI_ASM_LABEL(label_184)  // 1 block: unique 62: Done
KAI_ASM_LABEL(label_185)  // 1 block: Oddments skip
    add x0, x0, #0x10
    add x8, x20, #0x10
    b label_210
KAI_ASM_LABEL(label_186)  // 1 skip
    cbz x17, label_209
    mov x21, x2
    mov x20, x1
    movi v18.16b, #0
    sub x16, x16, #0x1
    mov x8, x0
    cmp x21, #0x10
    add x1, x20, x4, LSL #1
    blt label_194
KAI_ASM_LABEL(label_187)  // Oddments: Column loop
    ldr q17, [x20], #0x10
    zip1 v16.2d, v17.2d, v18.2d
    str q16, [x8, #0]
    cbz x16, label_190
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_189)  // Oddments: unique 66: unique 67: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_189
KAI_ASM_LABEL(label_190)  // Oddments: unique 66: unique 67: Padding done
    add x8, x8, x5, LSL #4
    zip2 v16.2d, v17.2d, v18.2d
    str q16, [x8, #0]
    cbz x16, label_192
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_191)  // Oddments: unique 66: unique 68: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_191
KAI_ASM_LABEL(label_192)  // Oddments: unique 66: unique 68: Padding done
    add x8, x8, x5, LSL #4
    sub x21, x21, #0x10
    cmp x21, #0x10
    bge label_187
KAI_ASM_LABEL(label_194)  // Oddments: Loop skip
    cbz x21, label_208
    mov x6, SP
    tbz x21, #3, label_198
    ldr d17, [x20], #0x8
    ldr d18, [x6], #0x8
    tbz x21, #2, label_196
    ld1 { v17.s }[2], [x20], #0x4
    ld1 { v18.s }[2], [x6], #0x4
    tbz x21, #1, label_195
    ld1 { v17.h }[6], [x20], #0x2
    ld1 { v18.h }[6], [x6], #0x2
    tbz x21, #0, label_202
    ld1 { v17.b }[14], [x20]
    ld1 { v18.b }[14], [x6]
    b label_202
KAI_ASM_LABEL(label_195)  // Oddments: unique 69: Partial load: partial_0_12
    tbz x21, #0, label_202
    ld1 { v17.b }[12], [x20]
    ld1 { v18.b }[12], [x6]
    b label_202
KAI_ASM_LABEL(label_196)  // Oddments: unique 69: Partial load: partial_1_8
    tbz x21, #1, label_197
    ld1 { v17.h }[4], [x20], #0x2
    ld1 { v18.h }[4], [x6], #0x2
    tbz x21, #0, label_202
    ld1 { v17.b }[10], [x20]
    ld1 { v18.b }[10], [x6]
    b label_202
KAI_ASM_LABEL(label_197)  // Oddments: unique 69: Partial load: partial_0_8
    tbz x21, #0, label_202
    ld1 { v17.b }[8], [x20]
    ld1 { v18.b }[8], [x6]
    b label_202
KAI_ASM_LABEL(label_198)  // Oddments: unique 69: Partial load: partial_2_0
    tbz x21, #2, label_200
    ldr s17, [x20], #0x4
    ldr s18, [x6], #0x4
    tbz x21, #1, label_199
    ld1 { v17.h }[2], [x20], #0x2
    ld1 { v18.h }[2], [x6], #0x2
    tbz x21, #0, label_202
    ld1 { v17.b }[6], [x20]
    ld1 { v18.b }[6], [x6]
    b label_202
KAI_ASM_LABEL(label_199)  // Oddments: unique 69: Partial load: partial_0_4
    tbz x21, #0, label_202
    ld1 { v17.b }[4], [x20]
    ld1 { v18.b }[4], [x6]
    b label_202
KAI_ASM_LABEL(label_200)  // Oddments: unique 69: Partial load: partial_1_0
    tbz x21, #1, label_201
    ldr h17, [x20], #0x2
    ldr h18, [x6], #0x2
    tbz x21, #0, label_202
    ld1 { v17.b }[2], [x20]
    ld1 { v18.b }[2], [x6]
    b label_202
KAI_ASM_LABEL(label_201)  // Oddments: unique 69: Partial load: partial_0_0
    ldr b17, [x20, #0]
    ldr b18, [x6, #0]
KAI_ASM_LABEL(label_202)  // Oddments: unique 69: Partial load: Done
    zip1 v16.2d, v17.2d, v18.2d
    str q16, [x8, #0]
    cbz x16, label_204
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_203)  // Oddments: unique 70: unique 71: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_203
KAI_ASM_LABEL(label_204)  // Oddments: unique 70: unique 71: Padding done
    cmp x21, #0x8
    add x8, x8, x5, LSL #4
    ble label_207
    zip2 v16.2d, v17.2d, v18.2d
    str q16, [x8, #0]
    cbz x16, label_206
    add x7, x8, #0x10
    mov x6, x16
KAI_ASM_LABEL(label_205)  // Oddments: unique 70: unique 72: Padding loop
    subs x6, x6, #0x1
    str q6, [x7], #0x10
    bne label_205
KAI_ASM_LABEL(label_206)  // Oddments: unique 70: unique 72: Padding done
    add x8, x8, x5, LSL #4
KAI_ASM_LABEL(label_207)  // Oddments: unique 70: Done
KAI_ASM_LABEL(label_208)  // Oddments: Oddments skip
    add x0, x0, #0x10
    add x8, x8, #0x10
    mov x17, #0
KAI_ASM_LABEL(label_209)  // No oddments
KAI_ASM_LABEL(label_210)  // Block loop end
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave_2_1x8)

    KAI_ASM_END
