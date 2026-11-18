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

    KAI_ASM_CODE(a64_interleave_4_1x4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave_4_1x4)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave_4_1x4)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave_4_1x4)
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
    blt label_33
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
    cmp x22, #0x10
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
    ldr q11, [x17], #0x10
    ldr q10, [x25], #0x10
    ldr q9, [x9], #0x10
    ldr q8, [x24], #0x10
    ldr q7, [x16], #0x10
    ldr q6, [x23], #0x10
    ldr q5, [x28], #0x10
    ldr q4, [x26], #0x10
    ldr q3, [x12], #0x10
    ldr q2, [x21], #0x10
    ldr q1, [x11], #0x10
    ldr q0, [x20], #0x10
    ldr q31, [x10], #0x10
    ldr q30, [x8], #0x10
    ldr q29, [x27], #0x10
    ldr q28, [x7], #0x10
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
    sub x22, x22, #0x10
    cmp x22, #0x10
    bge label_3
KAI_ASM_LABEL(label_14)  // 4 block: Loop skip
    cbz x22, label_32
    tbz x22, #3, label_18
    ldr d11, [x17], #0x8
    ldr d10, [x25], #0x8
    ldr d9, [x9], #0x8
    ldr d8, [x24], #0x8
    ldr d7, [x16], #0x8
    ldr d6, [x23], #0x8
    ldr d5, [x28], #0x8
    ldr d4, [x26], #0x8
    ldr d3, [x12], #0x8
    ldr d2, [x21], #0x8
    ldr d1, [x11], #0x8
    ldr d0, [x20], #0x8
    ldr d31, [x10], #0x8
    ldr d30, [x8], #0x8
    ldr d29, [x27], #0x8
    ldr d28, [x7], #0x8
    tbz x22, #2, label_16
    ld1 { v11.s }[2], [x17], #0x4
    ld1 { v10.s }[2], [x25], #0x4
    ld1 { v9.s }[2], [x9], #0x4
    ld1 { v8.s }[2], [x24], #0x4
    ld1 { v7.s }[2], [x16], #0x4
    ld1 { v6.s }[2], [x23], #0x4
    ld1 { v5.s }[2], [x28], #0x4
    ld1 { v4.s }[2], [x26], #0x4
    ld1 { v3.s }[2], [x12], #0x4
    ld1 { v2.s }[2], [x21], #0x4
    ld1 { v1.s }[2], [x11], #0x4
    ld1 { v0.s }[2], [x20], #0x4
    ld1 { v31.s }[2], [x10], #0x4
    ld1 { v30.s }[2], [x8], #0x4
    ld1 { v29.s }[2], [x27], #0x4
    ld1 { v28.s }[2], [x7], #0x4
    tbz x22, #1, label_15
    ld1 { v11.h }[6], [x17], #0x2
    ld1 { v10.h }[6], [x25], #0x2
    ld1 { v9.h }[6], [x9], #0x2
    ld1 { v8.h }[6], [x24], #0x2
    ld1 { v7.h }[6], [x16], #0x2
    ld1 { v6.h }[6], [x23], #0x2
    ld1 { v5.h }[6], [x28], #0x2
    ld1 { v4.h }[6], [x26], #0x2
    ld1 { v3.h }[6], [x12], #0x2
    ld1 { v2.h }[6], [x21], #0x2
    ld1 { v1.h }[6], [x11], #0x2
    ld1 { v0.h }[6], [x20], #0x2
    ld1 { v31.h }[6], [x10], #0x2
    ld1 { v30.h }[6], [x8], #0x2
    ld1 { v29.h }[6], [x27], #0x2
    ld1 { v28.h }[6], [x7], #0x2
    tbz x22, #0, label_22
    ld1 { v11.b }[14], [x17]
    ld1 { v10.b }[14], [x25]
    ld1 { v9.b }[14], [x9]
    ld1 { v8.b }[14], [x24]
    ld1 { v7.b }[14], [x16]
    ld1 { v6.b }[14], [x23]
    ld1 { v5.b }[14], [x28]
    ld1 { v4.b }[14], [x26]
    ld1 { v3.b }[14], [x12]
    ld1 { v2.b }[14], [x21]
    ld1 { v1.b }[14], [x11]
    ld1 { v0.b }[14], [x20]
    ld1 { v31.b }[14], [x10]
    ld1 { v30.b }[14], [x8]
    ld1 { v29.b }[14], [x27]
    ld1 { v28.b }[14], [x7]
    b label_22
KAI_ASM_LABEL(label_15)  // 4 block: unique 7: Partial load: partial_0_12
    tbz x22, #0, label_22
    ld1 { v11.b }[12], [x17]
    ld1 { v10.b }[12], [x25]
    ld1 { v9.b }[12], [x9]
    ld1 { v8.b }[12], [x24]
    ld1 { v7.b }[12], [x16]
    ld1 { v6.b }[12], [x23]
    ld1 { v5.b }[12], [x28]
    ld1 { v4.b }[12], [x26]
    ld1 { v3.b }[12], [x12]
    ld1 { v2.b }[12], [x21]
    ld1 { v1.b }[12], [x11]
    ld1 { v0.b }[12], [x20]
    ld1 { v31.b }[12], [x10]
    ld1 { v30.b }[12], [x8]
    ld1 { v29.b }[12], [x27]
    ld1 { v28.b }[12], [x7]
    b label_22
KAI_ASM_LABEL(label_16)  // 4 block: unique 7: Partial load: partial_1_8
    tbz x22, #1, label_17
    ld1 { v11.h }[4], [x17], #0x2
    ld1 { v10.h }[4], [x25], #0x2
    ld1 { v9.h }[4], [x9], #0x2
    ld1 { v8.h }[4], [x24], #0x2
    ld1 { v7.h }[4], [x16], #0x2
    ld1 { v6.h }[4], [x23], #0x2
    ld1 { v5.h }[4], [x28], #0x2
    ld1 { v4.h }[4], [x26], #0x2
    ld1 { v3.h }[4], [x12], #0x2
    ld1 { v2.h }[4], [x21], #0x2
    ld1 { v1.h }[4], [x11], #0x2
    ld1 { v0.h }[4], [x20], #0x2
    ld1 { v31.h }[4], [x10], #0x2
    ld1 { v30.h }[4], [x8], #0x2
    ld1 { v29.h }[4], [x27], #0x2
    ld1 { v28.h }[4], [x7], #0x2
    tbz x22, #0, label_22
    ld1 { v11.b }[10], [x17]
    ld1 { v10.b }[10], [x25]
    ld1 { v9.b }[10], [x9]
    ld1 { v8.b }[10], [x24]
    ld1 { v7.b }[10], [x16]
    ld1 { v6.b }[10], [x23]
    ld1 { v5.b }[10], [x28]
    ld1 { v4.b }[10], [x26]
    ld1 { v3.b }[10], [x12]
    ld1 { v2.b }[10], [x21]
    ld1 { v1.b }[10], [x11]
    ld1 { v0.b }[10], [x20]
    ld1 { v31.b }[10], [x10]
    ld1 { v30.b }[10], [x8]
    ld1 { v29.b }[10], [x27]
    ld1 { v28.b }[10], [x7]
    b label_22
KAI_ASM_LABEL(label_17)  // 4 block: unique 7: Partial load: partial_0_8
    tbz x22, #0, label_22
    ld1 { v11.b }[8], [x17]
    ld1 { v10.b }[8], [x25]
    ld1 { v9.b }[8], [x9]
    ld1 { v8.b }[8], [x24]
    ld1 { v7.b }[8], [x16]
    ld1 { v6.b }[8], [x23]
    ld1 { v5.b }[8], [x28]
    ld1 { v4.b }[8], [x26]
    ld1 { v3.b }[8], [x12]
    ld1 { v2.b }[8], [x21]
    ld1 { v1.b }[8], [x11]
    ld1 { v0.b }[8], [x20]
    ld1 { v31.b }[8], [x10]
    ld1 { v30.b }[8], [x8]
    ld1 { v29.b }[8], [x27]
    ld1 { v28.b }[8], [x7]
    b label_22
KAI_ASM_LABEL(label_18)  // 4 block: unique 7: Partial load: partial_2_0
    tbz x22, #2, label_20
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
    tbz x22, #1, label_19
    ld1 { v11.h }[2], [x17], #0x2
    ld1 { v10.h }[2], [x25], #0x2
    ld1 { v9.h }[2], [x9], #0x2
    ld1 { v8.h }[2], [x24], #0x2
    ld1 { v7.h }[2], [x16], #0x2
    ld1 { v6.h }[2], [x23], #0x2
    ld1 { v5.h }[2], [x28], #0x2
    ld1 { v4.h }[2], [x26], #0x2
    ld1 { v3.h }[2], [x12], #0x2
    ld1 { v2.h }[2], [x21], #0x2
    ld1 { v1.h }[2], [x11], #0x2
    ld1 { v0.h }[2], [x20], #0x2
    ld1 { v31.h }[2], [x10], #0x2
    ld1 { v30.h }[2], [x8], #0x2
    ld1 { v29.h }[2], [x27], #0x2
    ld1 { v28.h }[2], [x7], #0x2
    tbz x22, #0, label_22
    ld1 { v11.b }[6], [x17]
    ld1 { v10.b }[6], [x25]
    ld1 { v9.b }[6], [x9]
    ld1 { v8.b }[6], [x24]
    ld1 { v7.b }[6], [x16]
    ld1 { v6.b }[6], [x23]
    ld1 { v5.b }[6], [x28]
    ld1 { v4.b }[6], [x26]
    ld1 { v3.b }[6], [x12]
    ld1 { v2.b }[6], [x21]
    ld1 { v1.b }[6], [x11]
    ld1 { v0.b }[6], [x20]
    ld1 { v31.b }[6], [x10]
    ld1 { v30.b }[6], [x8]
    ld1 { v29.b }[6], [x27]
    ld1 { v28.b }[6], [x7]
    b label_22
KAI_ASM_LABEL(label_19)  // 4 block: unique 7: Partial load: partial_0_4
    tbz x22, #0, label_22
    ld1 { v11.b }[4], [x17]
    ld1 { v10.b }[4], [x25]
    ld1 { v9.b }[4], [x9]
    ld1 { v8.b }[4], [x24]
    ld1 { v7.b }[4], [x16]
    ld1 { v6.b }[4], [x23]
    ld1 { v5.b }[4], [x28]
    ld1 { v4.b }[4], [x26]
    ld1 { v3.b }[4], [x12]
    ld1 { v2.b }[4], [x21]
    ld1 { v1.b }[4], [x11]
    ld1 { v0.b }[4], [x20]
    ld1 { v31.b }[4], [x10]
    ld1 { v30.b }[4], [x8]
    ld1 { v29.b }[4], [x27]
    ld1 { v28.b }[4], [x7]
    b label_22
KAI_ASM_LABEL(label_20)  // 4 block: unique 7: Partial load: partial_1_0
    tbz x22, #1, label_21
    ldr h11, [x17], #0x2
    ldr h10, [x25], #0x2
    ldr h9, [x9], #0x2
    ldr h8, [x24], #0x2
    ldr h7, [x16], #0x2
    ldr h6, [x23], #0x2
    ldr h5, [x28], #0x2
    ldr h4, [x26], #0x2
    ldr h3, [x12], #0x2
    ldr h2, [x21], #0x2
    ldr h1, [x11], #0x2
    ldr h0, [x20], #0x2
    ldr h31, [x10], #0x2
    ldr h30, [x8], #0x2
    ldr h29, [x27], #0x2
    ldr h28, [x7], #0x2
    tbz x22, #0, label_22
    ld1 { v11.b }[2], [x17]
    ld1 { v10.b }[2], [x25]
    ld1 { v9.b }[2], [x9]
    ld1 { v8.b }[2], [x24]
    ld1 { v7.b }[2], [x16]
    ld1 { v6.b }[2], [x23]
    ld1 { v5.b }[2], [x28]
    ld1 { v4.b }[2], [x26]
    ld1 { v3.b }[2], [x12]
    ld1 { v2.b }[2], [x21]
    ld1 { v1.b }[2], [x11]
    ld1 { v0.b }[2], [x20]
    ld1 { v31.b }[2], [x10]
    ld1 { v30.b }[2], [x8]
    ld1 { v29.b }[2], [x27]
    ld1 { v28.b }[2], [x7]
    b label_22
KAI_ASM_LABEL(label_21)  // 4 block: unique 7: Partial load: partial_0_0
    ldr b11, [x17, #0]
    ldr b10, [x25, #0]
    ldr b9, [x9, #0]
    ldr b8, [x24, #0]
    ldr b7, [x16, #0]
    ldr b6, [x23, #0]
    ldr b5, [x28, #0]
    ldr b4, [x26, #0]
    ldr b3, [x12, #0]
    ldr b2, [x21, #0]
    ldr b1, [x11, #0]
    ldr b0, [x20, #0]
    ldr b31, [x10, #0]
    ldr b30, [x8, #0]
    ldr b29, [x27, #0]
    ldr b28, [x7, #0]
KAI_ASM_LABEL(label_22)  // 4 block: unique 7: Partial load: Done
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
    cbz x13, label_24
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_23)  // 4 block: unique 8: unique 9: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_23
KAI_ASM_LABEL(label_24)  // 4 block: unique 8: unique 9: Padding done
    cmp x22, #0x4
    add x14, x14, x5, LSL #4
    ble label_31
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_26
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_25)  // 4 block: unique 8: unique 10: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_25
KAI_ASM_LABEL(label_26)  // 4 block: unique 8: unique 10: Padding done
    cmp x22, #0x8
    add x14, x14, x5, LSL #4
    ble label_31
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
    cbz x13, label_28
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_27)  // 4 block: unique 8: unique 11: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_27
KAI_ASM_LABEL(label_28)  // 4 block: unique 8: unique 11: Padding done
    cmp x22, #0xc
    add x14, x14, x5, LSL #4
    ble label_31
    zip2 v19.4s, v27.4s, v26.4s
    zip2 v18.4s, v25.4s, v24.4s
    zip2 v17.4s, v23.4s, v22.4s
    zip2 v16.4s, v21.4s, v20.4s
    str q19, [x14, #0]
    str q18, [x14, #0x10]
    str q17, [x14, #0x20]
    str q16, [x14, #0x30]
    cbz x13, label_30
    add x6, x14, #0x40
    mov x3, x13
KAI_ASM_LABEL(label_29)  // 4 block: unique 8: unique 12: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_29
KAI_ASM_LABEL(label_30)  // 4 block: unique 8: unique 12: Padding done
    add x14, x14, x5, LSL #4
KAI_ASM_LABEL(label_31)  // 4 block: unique 8: Done
KAI_ASM_LABEL(label_32)  // 4 block: Oddments skip
    add x0, x0, #0x40
    ldr x3, [SP, #0x10]
    ldr x16, [SP, #0x18]
    add x10, x14, #0x40
    ldr x15, [SP, #0x20]
    b label_158
KAI_ASM_LABEL(label_33)  // 4 skip
    cmp x16, #0xc
    blt label_64
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
    cmp x11, #0x10
    add x23, x12, x4
    add x22, x26, x4
    add x21, x28, x4
    add x20, x25, x4
    add x8, x27, x4
    add x7, x24, x4
    add x1, x27, x4, LSL #2
    blt label_45
KAI_ASM_LABEL(label_34)  // 3 block: Column loop
    ldr q4, [x12], #0x10
    ldr q3, [x23], #0x10
    ldr q2, [x26], #0x10
    ldr q1, [x22], #0x10
    ldr q0, [x28], #0x10
    ldr q31, [x21], #0x10
    ldr q30, [x25], #0x10
    ldr q29, [x20], #0x10
    ldr q28, [x27], #0x10
    ldr q27, [x8], #0x10
    ldr q26, [x24], #0x10
    ldr q25, [x7], #0x10
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
    cbz x10, label_37
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_36)  // 3 block: unique 14: unique 15: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_36
KAI_ASM_LABEL(label_37)  // 3 block: unique 14: unique 15: Padding done
    add x9, x9, x5, LSL #4
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_39
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_38)  // 3 block: unique 14: unique 16: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_38
KAI_ASM_LABEL(label_39)  // 3 block: unique 14: unique 16: Padding done
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
    cbz x10, label_41
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_40)  // 3 block: unique 14: unique 17: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_40
KAI_ASM_LABEL(label_41)  // 3 block: unique 14: unique 17: Padding done
    add x9, x9, x5, LSL #4
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_43
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_42)  // 3 block: unique 14: unique 18: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_42
KAI_ASM_LABEL(label_43)  // 3 block: unique 14: unique 18: Padding done
    add x9, x9, x5, LSL #4
    sub x11, x11, #0x10
    cmp x11, #0x10
    bge label_34
KAI_ASM_LABEL(label_45)  // 3 block: Loop skip
    cbz x11, label_63
    tbz x11, #3, label_49
    ldr d4, [x12], #0x8
    ldr d3, [x23], #0x8
    ldr d2, [x26], #0x8
    ldr d1, [x22], #0x8
    ldr d0, [x28], #0x8
    ldr d31, [x21], #0x8
    ldr d30, [x25], #0x8
    ldr d29, [x20], #0x8
    ldr d28, [x27], #0x8
    ldr d27, [x8], #0x8
    ldr d26, [x24], #0x8
    ldr d25, [x7], #0x8
    tbz x11, #2, label_47
    ld1 { v4.s }[2], [x12], #0x4
    ld1 { v3.s }[2], [x23], #0x4
    ld1 { v2.s }[2], [x26], #0x4
    ld1 { v1.s }[2], [x22], #0x4
    ld1 { v0.s }[2], [x28], #0x4
    ld1 { v31.s }[2], [x21], #0x4
    ld1 { v30.s }[2], [x25], #0x4
    ld1 { v29.s }[2], [x20], #0x4
    ld1 { v28.s }[2], [x27], #0x4
    ld1 { v27.s }[2], [x8], #0x4
    ld1 { v26.s }[2], [x24], #0x4
    ld1 { v25.s }[2], [x7], #0x4
    tbz x11, #1, label_46
    ld1 { v4.h }[6], [x12], #0x2
    ld1 { v3.h }[6], [x23], #0x2
    ld1 { v2.h }[6], [x26], #0x2
    ld1 { v1.h }[6], [x22], #0x2
    ld1 { v0.h }[6], [x28], #0x2
    ld1 { v31.h }[6], [x21], #0x2
    ld1 { v30.h }[6], [x25], #0x2
    ld1 { v29.h }[6], [x20], #0x2
    ld1 { v28.h }[6], [x27], #0x2
    ld1 { v27.h }[6], [x8], #0x2
    ld1 { v26.h }[6], [x24], #0x2
    ld1 { v25.h }[6], [x7], #0x2
    tbz x11, #0, label_53
    ld1 { v4.b }[14], [x12]
    ld1 { v3.b }[14], [x23]
    ld1 { v2.b }[14], [x26]
    ld1 { v1.b }[14], [x22]
    ld1 { v0.b }[14], [x28]
    ld1 { v31.b }[14], [x21]
    ld1 { v30.b }[14], [x25]
    ld1 { v29.b }[14], [x20]
    ld1 { v28.b }[14], [x27]
    ld1 { v27.b }[14], [x8]
    ld1 { v26.b }[14], [x24]
    ld1 { v25.b }[14], [x7]
    b label_53
KAI_ASM_LABEL(label_46)  // 3 block: unique 19: Partial load: partial_0_12
    tbz x11, #0, label_53
    ld1 { v4.b }[12], [x12]
    ld1 { v3.b }[12], [x23]
    ld1 { v2.b }[12], [x26]
    ld1 { v1.b }[12], [x22]
    ld1 { v0.b }[12], [x28]
    ld1 { v31.b }[12], [x21]
    ld1 { v30.b }[12], [x25]
    ld1 { v29.b }[12], [x20]
    ld1 { v28.b }[12], [x27]
    ld1 { v27.b }[12], [x8]
    ld1 { v26.b }[12], [x24]
    ld1 { v25.b }[12], [x7]
    b label_53
KAI_ASM_LABEL(label_47)  // 3 block: unique 19: Partial load: partial_1_8
    tbz x11, #1, label_48
    ld1 { v4.h }[4], [x12], #0x2
    ld1 { v3.h }[4], [x23], #0x2
    ld1 { v2.h }[4], [x26], #0x2
    ld1 { v1.h }[4], [x22], #0x2
    ld1 { v0.h }[4], [x28], #0x2
    ld1 { v31.h }[4], [x21], #0x2
    ld1 { v30.h }[4], [x25], #0x2
    ld1 { v29.h }[4], [x20], #0x2
    ld1 { v28.h }[4], [x27], #0x2
    ld1 { v27.h }[4], [x8], #0x2
    ld1 { v26.h }[4], [x24], #0x2
    ld1 { v25.h }[4], [x7], #0x2
    tbz x11, #0, label_53
    ld1 { v4.b }[10], [x12]
    ld1 { v3.b }[10], [x23]
    ld1 { v2.b }[10], [x26]
    ld1 { v1.b }[10], [x22]
    ld1 { v0.b }[10], [x28]
    ld1 { v31.b }[10], [x21]
    ld1 { v30.b }[10], [x25]
    ld1 { v29.b }[10], [x20]
    ld1 { v28.b }[10], [x27]
    ld1 { v27.b }[10], [x8]
    ld1 { v26.b }[10], [x24]
    ld1 { v25.b }[10], [x7]
    b label_53
KAI_ASM_LABEL(label_48)  // 3 block: unique 19: Partial load: partial_0_8
    tbz x11, #0, label_53
    ld1 { v4.b }[8], [x12]
    ld1 { v3.b }[8], [x23]
    ld1 { v2.b }[8], [x26]
    ld1 { v1.b }[8], [x22]
    ld1 { v0.b }[8], [x28]
    ld1 { v31.b }[8], [x21]
    ld1 { v30.b }[8], [x25]
    ld1 { v29.b }[8], [x20]
    ld1 { v28.b }[8], [x27]
    ld1 { v27.b }[8], [x8]
    ld1 { v26.b }[8], [x24]
    ld1 { v25.b }[8], [x7]
    b label_53
KAI_ASM_LABEL(label_49)  // 3 block: unique 19: Partial load: partial_2_0
    tbz x11, #2, label_51
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
    tbz x11, #1, label_50
    ld1 { v4.h }[2], [x12], #0x2
    ld1 { v3.h }[2], [x23], #0x2
    ld1 { v2.h }[2], [x26], #0x2
    ld1 { v1.h }[2], [x22], #0x2
    ld1 { v0.h }[2], [x28], #0x2
    ld1 { v31.h }[2], [x21], #0x2
    ld1 { v30.h }[2], [x25], #0x2
    ld1 { v29.h }[2], [x20], #0x2
    ld1 { v28.h }[2], [x27], #0x2
    ld1 { v27.h }[2], [x8], #0x2
    ld1 { v26.h }[2], [x24], #0x2
    ld1 { v25.h }[2], [x7], #0x2
    tbz x11, #0, label_53
    ld1 { v4.b }[6], [x12]
    ld1 { v3.b }[6], [x23]
    ld1 { v2.b }[6], [x26]
    ld1 { v1.b }[6], [x22]
    ld1 { v0.b }[6], [x28]
    ld1 { v31.b }[6], [x21]
    ld1 { v30.b }[6], [x25]
    ld1 { v29.b }[6], [x20]
    ld1 { v28.b }[6], [x27]
    ld1 { v27.b }[6], [x8]
    ld1 { v26.b }[6], [x24]
    ld1 { v25.b }[6], [x7]
    b label_53
KAI_ASM_LABEL(label_50)  // 3 block: unique 19: Partial load: partial_0_4
    tbz x11, #0, label_53
    ld1 { v4.b }[4], [x12]
    ld1 { v3.b }[4], [x23]
    ld1 { v2.b }[4], [x26]
    ld1 { v1.b }[4], [x22]
    ld1 { v0.b }[4], [x28]
    ld1 { v31.b }[4], [x21]
    ld1 { v30.b }[4], [x25]
    ld1 { v29.b }[4], [x20]
    ld1 { v28.b }[4], [x27]
    ld1 { v27.b }[4], [x8]
    ld1 { v26.b }[4], [x24]
    ld1 { v25.b }[4], [x7]
    b label_53
KAI_ASM_LABEL(label_51)  // 3 block: unique 19: Partial load: partial_1_0
    tbz x11, #1, label_52
    ldr h4, [x12], #0x2
    ldr h3, [x23], #0x2
    ldr h2, [x26], #0x2
    ldr h1, [x22], #0x2
    ldr h0, [x28], #0x2
    ldr h31, [x21], #0x2
    ldr h30, [x25], #0x2
    ldr h29, [x20], #0x2
    ldr h28, [x27], #0x2
    ldr h27, [x8], #0x2
    ldr h26, [x24], #0x2
    ldr h25, [x7], #0x2
    tbz x11, #0, label_53
    ld1 { v4.b }[2], [x12]
    ld1 { v3.b }[2], [x23]
    ld1 { v2.b }[2], [x26]
    ld1 { v1.b }[2], [x22]
    ld1 { v0.b }[2], [x28]
    ld1 { v31.b }[2], [x21]
    ld1 { v30.b }[2], [x25]
    ld1 { v29.b }[2], [x20]
    ld1 { v28.b }[2], [x27]
    ld1 { v27.b }[2], [x8]
    ld1 { v26.b }[2], [x24]
    ld1 { v25.b }[2], [x7]
    b label_53
KAI_ASM_LABEL(label_52)  // 3 block: unique 19: Partial load: partial_0_0
    ldr b4, [x12, #0]
    ldr b3, [x23, #0]
    ldr b2, [x26, #0]
    ldr b1, [x22, #0]
    ldr b0, [x28, #0]
    ldr b31, [x21, #0]
    ldr b30, [x25, #0]
    ldr b29, [x20, #0]
    ldr b28, [x27, #0]
    ldr b27, [x8, #0]
    ldr b26, [x24, #0]
    ldr b25, [x7, #0]
KAI_ASM_LABEL(label_53)  // 3 block: unique 19: Partial load: Done
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
    cbz x10, label_55
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_54)  // 3 block: unique 20: unique 21: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_54
KAI_ASM_LABEL(label_55)  // 3 block: unique 20: unique 21: Padding done
    cmp x11, #0x4
    add x9, x9, x5, LSL #4
    ble label_62
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_57
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_56)  // 3 block: unique 20: unique 22: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_56
KAI_ASM_LABEL(label_57)  // 3 block: unique 20: unique 22: Padding done
    cmp x11, #0x8
    add x9, x9, x5, LSL #4
    ble label_62
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
    cbz x10, label_59
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_58)  // 3 block: unique 20: unique 23: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_58
KAI_ASM_LABEL(label_59)  // 3 block: unique 20: unique 23: Padding done
    cmp x11, #0xc
    add x9, x9, x5, LSL #4
    ble label_62
    zip2 v18.4s, v24.4s, v23.4s
    zip2 v17.4s, v22.4s, v21.4s
    zip2 v16.4s, v20.4s, v19.4s
    str q18, [x9, #0]
    str q17, [x9, #0x10]
    str q16, [x9, #0x20]
    cbz x10, label_61
    add x6, x9, #0x30
    mov x3, x10
KAI_ASM_LABEL(label_60)  // 3 block: unique 20: unique 24: Padding loop
    subs x3, x3, #0x1
    str q12, [x6], #0x10
    bne label_60
KAI_ASM_LABEL(label_61)  // 3 block: unique 20: unique 24: Padding done
    add x9, x9, x5, LSL #4
KAI_ASM_LABEL(label_62)  // 3 block: unique 20: Done
KAI_ASM_LABEL(label_63)  // 3 block: Oddments skip
    add x0, x0, #0x30
    ldr x3, [SP, #0x10]
    ldr x16, [SP, #0x18]
    add x10, x9, #0x30
    ldr x15, [SP, #0x20]
    b label_158
KAI_ASM_LABEL(label_64)  // 3 skip
    cmp x16, #0x8
    blt label_95
    mov x9, x1
    mov x28, x2
    sub x15, x15, #0x2
    subs x16, x16, #0x8
    csel x27, x15, XZR, EQ
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
    blt label_76
KAI_ASM_LABEL(label_65)  // 2 block: Column loop
    ldr q29, [x9], #0x10
    ldr q28, [x22], #0x10
    ldr q27, [x24], #0x10
    ldr q26, [x21], #0x10
    ldr q25, [x25], #0x10
    ldr q24, [x20], #0x10
    ldr q23, [x23], #0x10
    ldr q22, [x8], #0x10
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
KAI_ASM_LABEL(label_67)  // 2 block: unique 26: unique 27: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_67
KAI_ASM_LABEL(label_68)  // 2 block: unique 26: unique 27: Padding done
    add x26, x26, x5, LSL #4
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_70
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_69)  // 2 block: unique 26: unique 28: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_69
KAI_ASM_LABEL(label_70)  // 2 block: unique 26: unique 28: Padding done
    zip2 v21.4s, v29.4s, v27.4s
    zip2 v20.4s, v28.4s, v26.4s
    add x26, x26, x5, LSL #4
    zip2 v19.4s, v25.4s, v23.4s
    zip2 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_72
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_71)  // 2 block: unique 26: unique 29: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_71
KAI_ASM_LABEL(label_72)  // 2 block: unique 26: unique 29: Padding done
    add x26, x26, x5, LSL #4
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_74
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_73)  // 2 block: unique 26: unique 30: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_73
KAI_ASM_LABEL(label_74)  // 2 block: unique 26: unique 30: Padding done
    add x26, x26, x5, LSL #4
    sub x28, x28, #0x10
    cmp x28, #0x10
    bge label_65
KAI_ASM_LABEL(label_76)  // 2 block: Loop skip
    cbz x28, label_94
    tbz x28, #3, label_80
    ldr d29, [x9], #0x8
    ldr d28, [x22], #0x8
    ldr d27, [x24], #0x8
    ldr d26, [x21], #0x8
    ldr d25, [x25], #0x8
    ldr d24, [x20], #0x8
    ldr d23, [x23], #0x8
    ldr d22, [x8], #0x8
    tbz x28, #2, label_78
    ld1 { v29.s }[2], [x9], #0x4
    ld1 { v28.s }[2], [x22], #0x4
    ld1 { v27.s }[2], [x24], #0x4
    ld1 { v26.s }[2], [x21], #0x4
    ld1 { v25.s }[2], [x25], #0x4
    ld1 { v24.s }[2], [x20], #0x4
    ld1 { v23.s }[2], [x23], #0x4
    ld1 { v22.s }[2], [x8], #0x4
    tbz x28, #1, label_77
    ld1 { v29.h }[6], [x9], #0x2
    ld1 { v28.h }[6], [x22], #0x2
    ld1 { v27.h }[6], [x24], #0x2
    ld1 { v26.h }[6], [x21], #0x2
    ld1 { v25.h }[6], [x25], #0x2
    ld1 { v24.h }[6], [x20], #0x2
    ld1 { v23.h }[6], [x23], #0x2
    ld1 { v22.h }[6], [x8], #0x2
    tbz x28, #0, label_84
    ld1 { v29.b }[14], [x9]
    ld1 { v28.b }[14], [x22]
    ld1 { v27.b }[14], [x24]
    ld1 { v26.b }[14], [x21]
    ld1 { v25.b }[14], [x25]
    ld1 { v24.b }[14], [x20]
    ld1 { v23.b }[14], [x23]
    ld1 { v22.b }[14], [x8]
    b label_84
KAI_ASM_LABEL(label_77)  // 2 block: unique 31: Partial load: partial_0_12
    tbz x28, #0, label_84
    ld1 { v29.b }[12], [x9]
    ld1 { v28.b }[12], [x22]
    ld1 { v27.b }[12], [x24]
    ld1 { v26.b }[12], [x21]
    ld1 { v25.b }[12], [x25]
    ld1 { v24.b }[12], [x20]
    ld1 { v23.b }[12], [x23]
    ld1 { v22.b }[12], [x8]
    b label_84
KAI_ASM_LABEL(label_78)  // 2 block: unique 31: Partial load: partial_1_8
    tbz x28, #1, label_79
    ld1 { v29.h }[4], [x9], #0x2
    ld1 { v28.h }[4], [x22], #0x2
    ld1 { v27.h }[4], [x24], #0x2
    ld1 { v26.h }[4], [x21], #0x2
    ld1 { v25.h }[4], [x25], #0x2
    ld1 { v24.h }[4], [x20], #0x2
    ld1 { v23.h }[4], [x23], #0x2
    ld1 { v22.h }[4], [x8], #0x2
    tbz x28, #0, label_84
    ld1 { v29.b }[10], [x9]
    ld1 { v28.b }[10], [x22]
    ld1 { v27.b }[10], [x24]
    ld1 { v26.b }[10], [x21]
    ld1 { v25.b }[10], [x25]
    ld1 { v24.b }[10], [x20]
    ld1 { v23.b }[10], [x23]
    ld1 { v22.b }[10], [x8]
    b label_84
KAI_ASM_LABEL(label_79)  // 2 block: unique 31: Partial load: partial_0_8
    tbz x28, #0, label_84
    ld1 { v29.b }[8], [x9]
    ld1 { v28.b }[8], [x22]
    ld1 { v27.b }[8], [x24]
    ld1 { v26.b }[8], [x21]
    ld1 { v25.b }[8], [x25]
    ld1 { v24.b }[8], [x20]
    ld1 { v23.b }[8], [x23]
    ld1 { v22.b }[8], [x8]
    b label_84
KAI_ASM_LABEL(label_80)  // 2 block: unique 31: Partial load: partial_2_0
    tbz x28, #2, label_82
    ldr s29, [x9], #0x4
    ldr s28, [x22], #0x4
    ldr s27, [x24], #0x4
    ldr s26, [x21], #0x4
    ldr s25, [x25], #0x4
    ldr s24, [x20], #0x4
    ldr s23, [x23], #0x4
    ldr s22, [x8], #0x4
    tbz x28, #1, label_81
    ld1 { v29.h }[2], [x9], #0x2
    ld1 { v28.h }[2], [x22], #0x2
    ld1 { v27.h }[2], [x24], #0x2
    ld1 { v26.h }[2], [x21], #0x2
    ld1 { v25.h }[2], [x25], #0x2
    ld1 { v24.h }[2], [x20], #0x2
    ld1 { v23.h }[2], [x23], #0x2
    ld1 { v22.h }[2], [x8], #0x2
    tbz x28, #0, label_84
    ld1 { v29.b }[6], [x9]
    ld1 { v28.b }[6], [x22]
    ld1 { v27.b }[6], [x24]
    ld1 { v26.b }[6], [x21]
    ld1 { v25.b }[6], [x25]
    ld1 { v24.b }[6], [x20]
    ld1 { v23.b }[6], [x23]
    ld1 { v22.b }[6], [x8]
    b label_84
KAI_ASM_LABEL(label_81)  // 2 block: unique 31: Partial load: partial_0_4
    tbz x28, #0, label_84
    ld1 { v29.b }[4], [x9]
    ld1 { v28.b }[4], [x22]
    ld1 { v27.b }[4], [x24]
    ld1 { v26.b }[4], [x21]
    ld1 { v25.b }[4], [x25]
    ld1 { v24.b }[4], [x20]
    ld1 { v23.b }[4], [x23]
    ld1 { v22.b }[4], [x8]
    b label_84
KAI_ASM_LABEL(label_82)  // 2 block: unique 31: Partial load: partial_1_0
    tbz x28, #1, label_83
    ldr h29, [x9], #0x2
    ldr h28, [x22], #0x2
    ldr h27, [x24], #0x2
    ldr h26, [x21], #0x2
    ldr h25, [x25], #0x2
    ldr h24, [x20], #0x2
    ldr h23, [x23], #0x2
    ldr h22, [x8], #0x2
    tbz x28, #0, label_84
    ld1 { v29.b }[2], [x9]
    ld1 { v28.b }[2], [x22]
    ld1 { v27.b }[2], [x24]
    ld1 { v26.b }[2], [x21]
    ld1 { v25.b }[2], [x25]
    ld1 { v24.b }[2], [x20]
    ld1 { v23.b }[2], [x23]
    ld1 { v22.b }[2], [x8]
    b label_84
KAI_ASM_LABEL(label_83)  // 2 block: unique 31: Partial load: partial_0_0
    ldr b29, [x9, #0]
    ldr b28, [x22, #0]
    ldr b27, [x24, #0]
    ldr b26, [x21, #0]
    ldr b25, [x25, #0]
    ldr b24, [x20, #0]
    ldr b23, [x23, #0]
    ldr b22, [x8, #0]
KAI_ASM_LABEL(label_84)  // 2 block: unique 31: Partial load: Done
    zip1 v21.4s, v29.4s, v27.4s
    zip1 v20.4s, v28.4s, v26.4s
    zip1 v19.4s, v25.4s, v23.4s
    zip1 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_86
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_85)  // 2 block: unique 32: unique 33: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_85
KAI_ASM_LABEL(label_86)  // 2 block: unique 32: unique 33: Padding done
    cmp x28, #0x4
    add x26, x26, x5, LSL #4
    ble label_93
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_88
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_87)  // 2 block: unique 32: unique 34: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_87
KAI_ASM_LABEL(label_88)  // 2 block: unique 32: unique 34: Padding done
    cmp x28, #0x8
    add x26, x26, x5, LSL #4
    ble label_93
    zip2 v21.4s, v29.4s, v27.4s
    zip2 v20.4s, v28.4s, v26.4s
    zip2 v19.4s, v25.4s, v23.4s
    zip2 v18.4s, v24.4s, v22.4s
    zip1 v17.4s, v21.4s, v20.4s
    zip1 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_90
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_89)  // 2 block: unique 32: unique 35: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_89
KAI_ASM_LABEL(label_90)  // 2 block: unique 32: unique 35: Padding done
    cmp x28, #0xc
    add x26, x26, x5, LSL #4
    ble label_93
    zip2 v17.4s, v21.4s, v20.4s
    zip2 v16.4s, v19.4s, v18.4s
    str q17, [x26, #0]
    str q16, [x26, #0x10]
    cbz x27, label_92
    add x7, x26, #0x20
    mov x6, x27
KAI_ASM_LABEL(label_91)  // 2 block: unique 32: unique 36: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_91
KAI_ASM_LABEL(label_92)  // 2 block: unique 32: unique 36: Padding done
    add x26, x26, x5, LSL #4
KAI_ASM_LABEL(label_93)  // 2 block: unique 32: Done
KAI_ASM_LABEL(label_94)  // 2 block: Oddments skip
    add x0, x0, #0x20
    add x10, x26, #0x20
    b label_158
KAI_ASM_LABEL(label_95)  // 2 skip
    cmp x16, #0x4
    blt label_126
    mov x25, x1
    mov x24, x2
    sub x15, x15, #0x1
    subs x16, x16, #0x4
    csel x23, x15, XZR, EQ
    mov x22, x0
    add x21, x25, x4, LSL #1
    cmp x24, #0x10
    add x20, x25, x4
    add x8, x21, x4
    add x1, x25, x4, LSL #2
    blt label_107
KAI_ASM_LABEL(label_96)  // 1 block: Column loop
    ldr q22, [x25], #0x10
    ldr q21, [x20], #0x10
    ldr q20, [x21], #0x10
    ldr q19, [x8], #0x10
    zip1 v18.4s, v22.4s, v20.4s
    zip1 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_99
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_98)  // 1 block: unique 38: unique 39: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_98
KAI_ASM_LABEL(label_99)  // 1 block: unique 38: unique 39: Padding done
    add x22, x22, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_101
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_100)  // 1 block: unique 38: unique 40: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_100
KAI_ASM_LABEL(label_101)  // 1 block: unique 38: unique 40: Padding done
    zip2 v18.4s, v22.4s, v20.4s
    zip2 v17.4s, v21.4s, v19.4s
    add x22, x22, x5, LSL #4
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_103
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_102)  // 1 block: unique 38: unique 41: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_102
KAI_ASM_LABEL(label_103)  // 1 block: unique 38: unique 41: Padding done
    add x22, x22, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_105
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_104)  // 1 block: unique 38: unique 42: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_104
KAI_ASM_LABEL(label_105)  // 1 block: unique 38: unique 42: Padding done
    add x22, x22, x5, LSL #4
    sub x24, x24, #0x10
    cmp x24, #0x10
    bge label_96
KAI_ASM_LABEL(label_107)  // 1 block: Loop skip
    cbz x24, label_125
    tbz x24, #3, label_111
    ldr d22, [x25], #0x8
    ldr d21, [x20], #0x8
    ldr d20, [x21], #0x8
    ldr d19, [x8], #0x8
    tbz x24, #2, label_109
    ld1 { v22.s }[2], [x25], #0x4
    ld1 { v21.s }[2], [x20], #0x4
    ld1 { v20.s }[2], [x21], #0x4
    ld1 { v19.s }[2], [x8], #0x4
    tbz x24, #1, label_108
    ld1 { v22.h }[6], [x25], #0x2
    ld1 { v21.h }[6], [x20], #0x2
    ld1 { v20.h }[6], [x21], #0x2
    ld1 { v19.h }[6], [x8], #0x2
    tbz x24, #0, label_115
    ld1 { v22.b }[14], [x25]
    ld1 { v21.b }[14], [x20]
    ld1 { v20.b }[14], [x21]
    ld1 { v19.b }[14], [x8]
    b label_115
KAI_ASM_LABEL(label_108)  // 1 block: unique 43: Partial load: partial_0_12
    tbz x24, #0, label_115
    ld1 { v22.b }[12], [x25]
    ld1 { v21.b }[12], [x20]
    ld1 { v20.b }[12], [x21]
    ld1 { v19.b }[12], [x8]
    b label_115
KAI_ASM_LABEL(label_109)  // 1 block: unique 43: Partial load: partial_1_8
    tbz x24, #1, label_110
    ld1 { v22.h }[4], [x25], #0x2
    ld1 { v21.h }[4], [x20], #0x2
    ld1 { v20.h }[4], [x21], #0x2
    ld1 { v19.h }[4], [x8], #0x2
    tbz x24, #0, label_115
    ld1 { v22.b }[10], [x25]
    ld1 { v21.b }[10], [x20]
    ld1 { v20.b }[10], [x21]
    ld1 { v19.b }[10], [x8]
    b label_115
KAI_ASM_LABEL(label_110)  // 1 block: unique 43: Partial load: partial_0_8
    tbz x24, #0, label_115
    ld1 { v22.b }[8], [x25]
    ld1 { v21.b }[8], [x20]
    ld1 { v20.b }[8], [x21]
    ld1 { v19.b }[8], [x8]
    b label_115
KAI_ASM_LABEL(label_111)  // 1 block: unique 43: Partial load: partial_2_0
    tbz x24, #2, label_113
    ldr s22, [x25], #0x4
    ldr s21, [x20], #0x4
    ldr s20, [x21], #0x4
    ldr s19, [x8], #0x4
    tbz x24, #1, label_112
    ld1 { v22.h }[2], [x25], #0x2
    ld1 { v21.h }[2], [x20], #0x2
    ld1 { v20.h }[2], [x21], #0x2
    ld1 { v19.h }[2], [x8], #0x2
    tbz x24, #0, label_115
    ld1 { v22.b }[6], [x25]
    ld1 { v21.b }[6], [x20]
    ld1 { v20.b }[6], [x21]
    ld1 { v19.b }[6], [x8]
    b label_115
KAI_ASM_LABEL(label_112)  // 1 block: unique 43: Partial load: partial_0_4
    tbz x24, #0, label_115
    ld1 { v22.b }[4], [x25]
    ld1 { v21.b }[4], [x20]
    ld1 { v20.b }[4], [x21]
    ld1 { v19.b }[4], [x8]
    b label_115
KAI_ASM_LABEL(label_113)  // 1 block: unique 43: Partial load: partial_1_0
    tbz x24, #1, label_114
    ldr h22, [x25], #0x2
    ldr h21, [x20], #0x2
    ldr h20, [x21], #0x2
    ldr h19, [x8], #0x2
    tbz x24, #0, label_115
    ld1 { v22.b }[2], [x25]
    ld1 { v21.b }[2], [x20]
    ld1 { v20.b }[2], [x21]
    ld1 { v19.b }[2], [x8]
    b label_115
KAI_ASM_LABEL(label_114)  // 1 block: unique 43: Partial load: partial_0_0
    ldr b22, [x25, #0]
    ldr b21, [x20, #0]
    ldr b20, [x21, #0]
    ldr b19, [x8, #0]
KAI_ASM_LABEL(label_115)  // 1 block: unique 43: Partial load: Done
    zip1 v18.4s, v22.4s, v20.4s
    zip1 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_117
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_116)  // 1 block: unique 44: unique 45: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_116
KAI_ASM_LABEL(label_117)  // 1 block: unique 44: unique 45: Padding done
    cmp x24, #0x4
    add x22, x22, x5, LSL #4
    ble label_124
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_119
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_118)  // 1 block: unique 44: unique 46: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_118
KAI_ASM_LABEL(label_119)  // 1 block: unique 44: unique 46: Padding done
    cmp x24, #0x8
    add x22, x22, x5, LSL #4
    ble label_124
    zip2 v18.4s, v22.4s, v20.4s
    zip2 v17.4s, v21.4s, v19.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_121
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_120)  // 1 block: unique 44: unique 47: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_120
KAI_ASM_LABEL(label_121)  // 1 block: unique 44: unique 47: Padding done
    cmp x24, #0xc
    add x22, x22, x5, LSL #4
    ble label_124
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x22, #0]
    cbz x23, label_123
    add x7, x22, #0x10
    mov x6, x23
KAI_ASM_LABEL(label_122)  // 1 block: unique 44: unique 48: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_122
KAI_ASM_LABEL(label_123)  // 1 block: unique 44: unique 48: Padding done
    add x22, x22, x5, LSL #4
KAI_ASM_LABEL(label_124)  // 1 block: unique 44: Done
KAI_ASM_LABEL(label_125)  // 1 block: Oddments skip
    add x0, x0, #0x10
    add x10, x22, #0x10
    b label_158
KAI_ASM_LABEL(label_126)  // 1 skip
    cbz x16, label_157
    mov x23, x2
    mov x22, x1
    movi v22.16b, #0
    movi v21.16b, #0
    sub x15, x15, #0x1
    movi v20.16b, #0
    mov x21, x0
    cmp x23, #0x10
    add x20, x22, x4
    add x8, x22, x4, LSL #1
    add x1, x22, x4, LSL #2
    blt label_138
KAI_ASM_LABEL(label_127)  // Oddments: Column loop
    ldr q19, [x22], #0x10
    cmp x16, #0x2
    blt label_128
    ldr q22, [x20], #0x10
    beq label_128
    ldr q21, [x8], #0x10
KAI_ASM_LABEL(label_128)  // Oddments: unique 49: Done
    zip1 v18.4s, v19.4s, v21.4s
    zip1 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_130
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_129)  // Oddments: unique 50: unique 51: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_129
KAI_ASM_LABEL(label_130)  // Oddments: unique 50: unique 51: Padding done
    add x21, x21, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_132
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_131)  // Oddments: unique 50: unique 52: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_131
KAI_ASM_LABEL(label_132)  // Oddments: unique 50: unique 52: Padding done
    zip2 v18.4s, v19.4s, v21.4s
    zip2 v17.4s, v22.4s, v20.4s
    add x21, x21, x5, LSL #4
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_134
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_133)  // Oddments: unique 50: unique 53: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_133
KAI_ASM_LABEL(label_134)  // Oddments: unique 50: unique 53: Padding done
    add x21, x21, x5, LSL #4
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_136
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_135)  // Oddments: unique 50: unique 54: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_135
KAI_ASM_LABEL(label_136)  // Oddments: unique 50: unique 54: Padding done
    add x21, x21, x5, LSL #4
    sub x23, x23, #0x10
    cmp x23, #0x10
    bge label_127
KAI_ASM_LABEL(label_138)  // Oddments: Loop skip
    cbz x23, label_156
    mov x6, SP
    cmp x16, #0x2
    csel x20, x20, x6, GE
    csel x8, x8, x6, GT
    tbz x23, #3, label_142
    ldr d19, [x22], #0x8
    ldr d22, [x20], #0x8
    ldr d21, [x8], #0x8
    ldr d20, [x6], #0x8
    tbz x23, #2, label_140
    ld1 { v19.s }[2], [x22], #0x4
    ld1 { v22.s }[2], [x20], #0x4
    ld1 { v21.s }[2], [x8], #0x4
    ld1 { v20.s }[2], [x6], #0x4
    tbz x23, #1, label_139
    ld1 { v19.h }[6], [x22], #0x2
    ld1 { v22.h }[6], [x20], #0x2
    ld1 { v21.h }[6], [x8], #0x2
    ld1 { v20.h }[6], [x6], #0x2
    tbz x23, #0, label_146
    ld1 { v19.b }[14], [x22]
    ld1 { v22.b }[14], [x20]
    ld1 { v21.b }[14], [x8]
    ld1 { v20.b }[14], [x6]
    b label_146
KAI_ASM_LABEL(label_139)  // Oddments: unique 55: Partial load: partial_0_12
    tbz x23, #0, label_146
    ld1 { v19.b }[12], [x22]
    ld1 { v22.b }[12], [x20]
    ld1 { v21.b }[12], [x8]
    ld1 { v20.b }[12], [x6]
    b label_146
KAI_ASM_LABEL(label_140)  // Oddments: unique 55: Partial load: partial_1_8
    tbz x23, #1, label_141
    ld1 { v19.h }[4], [x22], #0x2
    ld1 { v22.h }[4], [x20], #0x2
    ld1 { v21.h }[4], [x8], #0x2
    ld1 { v20.h }[4], [x6], #0x2
    tbz x23, #0, label_146
    ld1 { v19.b }[10], [x22]
    ld1 { v22.b }[10], [x20]
    ld1 { v21.b }[10], [x8]
    ld1 { v20.b }[10], [x6]
    b label_146
KAI_ASM_LABEL(label_141)  // Oddments: unique 55: Partial load: partial_0_8
    tbz x23, #0, label_146
    ld1 { v19.b }[8], [x22]
    ld1 { v22.b }[8], [x20]
    ld1 { v21.b }[8], [x8]
    ld1 { v20.b }[8], [x6]
    b label_146
KAI_ASM_LABEL(label_142)  // Oddments: unique 55: Partial load: partial_2_0
    tbz x23, #2, label_144
    ldr s19, [x22], #0x4
    ldr s22, [x20], #0x4
    ldr s21, [x8], #0x4
    ldr s20, [x6], #0x4
    tbz x23, #1, label_143
    ld1 { v19.h }[2], [x22], #0x2
    ld1 { v22.h }[2], [x20], #0x2
    ld1 { v21.h }[2], [x8], #0x2
    ld1 { v20.h }[2], [x6], #0x2
    tbz x23, #0, label_146
    ld1 { v19.b }[6], [x22]
    ld1 { v22.b }[6], [x20]
    ld1 { v21.b }[6], [x8]
    ld1 { v20.b }[6], [x6]
    b label_146
KAI_ASM_LABEL(label_143)  // Oddments: unique 55: Partial load: partial_0_4
    tbz x23, #0, label_146
    ld1 { v19.b }[4], [x22]
    ld1 { v22.b }[4], [x20]
    ld1 { v21.b }[4], [x8]
    ld1 { v20.b }[4], [x6]
    b label_146
KAI_ASM_LABEL(label_144)  // Oddments: unique 55: Partial load: partial_1_0
    tbz x23, #1, label_145
    ldr h19, [x22], #0x2
    ldr h22, [x20], #0x2
    ldr h21, [x8], #0x2
    ldr h20, [x6], #0x2
    tbz x23, #0, label_146
    ld1 { v19.b }[2], [x22]
    ld1 { v22.b }[2], [x20]
    ld1 { v21.b }[2], [x8]
    ld1 { v20.b }[2], [x6]
    b label_146
KAI_ASM_LABEL(label_145)  // Oddments: unique 55: Partial load: partial_0_0
    ldr b19, [x22, #0]
    ldr b22, [x20, #0]
    ldr b21, [x8, #0]
    ldr b20, [x6, #0]
KAI_ASM_LABEL(label_146)  // Oddments: unique 55: Partial load: Done
    zip1 v18.4s, v19.4s, v21.4s
    zip1 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_148
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_147)  // Oddments: unique 56: unique 57: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_147
KAI_ASM_LABEL(label_148)  // Oddments: unique 56: unique 57: Padding done
    cmp x23, #0x4
    add x21, x21, x5, LSL #4
    ble label_155
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_150
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_149)  // Oddments: unique 56: unique 58: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_149
KAI_ASM_LABEL(label_150)  // Oddments: unique 56: unique 58: Padding done
    cmp x23, #0x8
    add x21, x21, x5, LSL #4
    ble label_155
    zip2 v18.4s, v19.4s, v21.4s
    zip2 v17.4s, v22.4s, v20.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_152
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_151)  // Oddments: unique 56: unique 59: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_151
KAI_ASM_LABEL(label_152)  // Oddments: unique 56: unique 59: Padding done
    cmp x23, #0xc
    add x21, x21, x5, LSL #4
    ble label_155
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0]
    cbz x15, label_154
    add x7, x21, #0x10
    mov x6, x15
KAI_ASM_LABEL(label_153)  // Oddments: unique 56: unique 60: Padding loop
    subs x6, x6, #0x1
    str q12, [x7], #0x10
    bne label_153
KAI_ASM_LABEL(label_154)  // Oddments: unique 56: unique 60: Padding done
    add x21, x21, x5, LSL #4
KAI_ASM_LABEL(label_155)  // Oddments: unique 56: Done
KAI_ASM_LABEL(label_156)  // Oddments: Oddments skip
    add x0, x0, #0x10
    add x10, x21, #0x10
    mov x16, #0
KAI_ASM_LABEL(label_157)  // No oddments
KAI_ASM_LABEL(label_158)  // Block loop end
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
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave_4_1x4)

    KAI_ASM_END
