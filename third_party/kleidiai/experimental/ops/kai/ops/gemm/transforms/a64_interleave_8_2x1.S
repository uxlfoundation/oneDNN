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

    KAI_ASM_CODE(a64_interleave_8_2x1)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleave_8_2x1)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleave_8_2x1)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleave_8_2x1)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    str d14, [sp, 120]
    sub SP, SP, #0x30
    movi v14.4s, #0
    str q14, [SP, #0]
KAI_ASM_LABEL(label_1)  // Row loop
    lsl x11, x5, #0x3
    mov x10, x5
    cmp x11, x3
    csel x11, x11, x3, LT
    sub x3, x3, x11
KAI_ASM_LABEL(label_2)  // Block loop
    cmp x11, #0x10
    blt label_45
    mov x16, x1
    mov x15, x2
    str x3, [SP, #0x10]
    sub x10, x10, #0x2
    subs x11, x11, #0x10
    csel x14, x10, XZR, EQ
    str x11, [SP, #0x18]
    mov x13, x0
    add x12, x16, x4, LSL #3
    add x11, x16, x4, LSL #2
    str x10, [SP, #0x20]
    add x10, x12, x4, LSL #2
    add x9, x16, x4, LSL #1
    add x28, x11, x4, LSL #1
    add x27, x12, x4, LSL #1
    add x26, x10, x4, LSL #1
    cmp x15, #0x8
    add x25, x16, x4
    add x24, x9, x4
    add x23, x11, x4
    add x22, x28, x4
    add x21, x12, x4
    add x20, x27, x4
    add x8, x10, x4
    add x7, x26, x4
    add x1, x16, x4, LSL #4
    blt label_22
KAI_ASM_LABEL(label_3)  // 2 block: Column loop
    ldr q13, [x16], #0x10
    ldr q12, [x25], #0x10
    ldr q11, [x9], #0x10
    ldr q10, [x24], #0x10
    ldr q9, [x11], #0x10
    ldr q8, [x23], #0x10
    ldr q7, [x28], #0x10
    ldr q6, [x22], #0x10
    ldr q5, [x12], #0x10
    ldr q4, [x21], #0x10
    ldr q3, [x27], #0x10
    ldr q2, [x20], #0x10
    ldr q1, [x10], #0x10
    ldr q0, [x8], #0x10
    ldr q31, [x26], #0x10
    ldr q30, [x7], #0x10
    zip1 v29.8h, v13.8h, v9.8h
    zip1 v28.8h, v11.8h, v7.8h
    zip1 v27.8h, v12.8h, v8.8h
    zip1 v26.8h, v10.8h, v6.8h
    zip1 v25.8h, v5.8h, v1.8h
    zip1 v24.8h, v3.8h, v31.8h
    zip1 v23.8h, v4.8h, v0.8h
    zip1 v22.8h, v2.8h, v30.8h
    zip1 v21.8h, v29.8h, v28.8h
    zip1 v20.8h, v27.8h, v26.8h
    zip1 v19.8h, v25.8h, v24.8h
    zip1 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_6
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_5)  // 2 block: unique 2: unique 3: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_5
KAI_ASM_LABEL(label_6)  // 2 block: unique 2: unique 3: Padding done
    add x13, x13, x5, LSL #4
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_8
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_7)  // 2 block: unique 2: unique 4: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_7
KAI_ASM_LABEL(label_8)  // 2 block: unique 2: unique 4: Padding done
    zip2 v21.8h, v29.8h, v28.8h
    zip2 v20.8h, v27.8h, v26.8h
    add x13, x13, x5, LSL #4
    zip2 v19.8h, v25.8h, v24.8h
    zip2 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_10
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_9)  // 2 block: unique 2: unique 5: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_9
KAI_ASM_LABEL(label_10)  // 2 block: unique 2: unique 5: Padding done
    add x13, x13, x5, LSL #4
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_12
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_11)  // 2 block: unique 2: unique 6: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_11
KAI_ASM_LABEL(label_12)  // 2 block: unique 2: unique 6: Padding done
    zip2 v29.8h, v13.8h, v9.8h
    zip2 v28.8h, v11.8h, v7.8h
    add x13, x13, x5, LSL #4
    zip2 v27.8h, v12.8h, v8.8h
    zip2 v26.8h, v10.8h, v6.8h
    zip2 v25.8h, v5.8h, v1.8h
    zip2 v24.8h, v3.8h, v31.8h
    zip2 v23.8h, v4.8h, v0.8h
    zip2 v22.8h, v2.8h, v30.8h
    zip1 v21.8h, v29.8h, v28.8h
    zip1 v20.8h, v27.8h, v26.8h
    zip1 v19.8h, v25.8h, v24.8h
    zip1 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_14
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_13)  // 2 block: unique 2: unique 7: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_13
KAI_ASM_LABEL(label_14)  // 2 block: unique 2: unique 7: Padding done
    add x13, x13, x5, LSL #4
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_16
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_15)  // 2 block: unique 2: unique 8: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_15
KAI_ASM_LABEL(label_16)  // 2 block: unique 2: unique 8: Padding done
    zip2 v21.8h, v29.8h, v28.8h
    zip2 v20.8h, v27.8h, v26.8h
    add x13, x13, x5, LSL #4
    zip2 v19.8h, v25.8h, v24.8h
    zip2 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_18
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_17)  // 2 block: unique 2: unique 9: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_17
KAI_ASM_LABEL(label_18)  // 2 block: unique 2: unique 9: Padding done
    add x13, x13, x5, LSL #4
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_20
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_19)  // 2 block: unique 2: unique 10: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_19
KAI_ASM_LABEL(label_20)  // 2 block: unique 2: unique 10: Padding done
    add x13, x13, x5, LSL #4
    sub x15, x15, #0x8
    cmp x15, #0x8
    bge label_3
KAI_ASM_LABEL(label_22)  // 2 block: Loop skip
    cbz x15, label_44
    tbz x15, #2, label_24
    ldr d13, [x16], #0x8
    ldr d12, [x25], #0x8
    ldr d11, [x9], #0x8
    ldr d10, [x24], #0x8
    ldr d9, [x11], #0x8
    ldr d8, [x23], #0x8
    ldr d7, [x28], #0x8
    ldr d6, [x22], #0x8
    ldr d5, [x12], #0x8
    ldr d4, [x21], #0x8
    ldr d3, [x27], #0x8
    ldr d2, [x20], #0x8
    ldr d1, [x10], #0x8
    ldr d0, [x8], #0x8
    ldr d31, [x26], #0x8
    ldr d30, [x7], #0x8
    tbz x15, #1, label_23
    ld1 { v13.s }[2], [x16], #0x4
    ld1 { v12.s }[2], [x25], #0x4
    ld1 { v11.s }[2], [x9], #0x4
    ld1 { v10.s }[2], [x24], #0x4
    ld1 { v9.s }[2], [x11], #0x4
    ld1 { v8.s }[2], [x23], #0x4
    ld1 { v7.s }[2], [x28], #0x4
    ld1 { v6.s }[2], [x22], #0x4
    ld1 { v5.s }[2], [x12], #0x4
    ld1 { v4.s }[2], [x21], #0x4
    ld1 { v3.s }[2], [x27], #0x4
    ld1 { v2.s }[2], [x20], #0x4
    ld1 { v1.s }[2], [x10], #0x4
    ld1 { v0.s }[2], [x8], #0x4
    ld1 { v31.s }[2], [x26], #0x4
    ld1 { v30.s }[2], [x7], #0x4
    tbz x15, #0, label_26
    ld1 { v13.h }[6], [x16]
    ld1 { v12.h }[6], [x25]
    ld1 { v11.h }[6], [x9]
    ld1 { v10.h }[6], [x24]
    ld1 { v9.h }[6], [x11]
    ld1 { v8.h }[6], [x23]
    ld1 { v7.h }[6], [x28]
    ld1 { v6.h }[6], [x22]
    ld1 { v5.h }[6], [x12]
    ld1 { v4.h }[6], [x21]
    ld1 { v3.h }[6], [x27]
    ld1 { v2.h }[6], [x20]
    ld1 { v1.h }[6], [x10]
    ld1 { v0.h }[6], [x8]
    ld1 { v31.h }[6], [x26]
    ld1 { v30.h }[6], [x7]
    b label_26
KAI_ASM_LABEL(label_23)  // 2 block: unique 11: Partial load: partial_0_4
    tbz x15, #0, label_26
    ld1 { v13.h }[4], [x16]
    ld1 { v12.h }[4], [x25]
    ld1 { v11.h }[4], [x9]
    ld1 { v10.h }[4], [x24]
    ld1 { v9.h }[4], [x11]
    ld1 { v8.h }[4], [x23]
    ld1 { v7.h }[4], [x28]
    ld1 { v6.h }[4], [x22]
    ld1 { v5.h }[4], [x12]
    ld1 { v4.h }[4], [x21]
    ld1 { v3.h }[4], [x27]
    ld1 { v2.h }[4], [x20]
    ld1 { v1.h }[4], [x10]
    ld1 { v0.h }[4], [x8]
    ld1 { v31.h }[4], [x26]
    ld1 { v30.h }[4], [x7]
    b label_26
KAI_ASM_LABEL(label_24)  // 2 block: unique 11: Partial load: partial_1_0
    tbz x15, #1, label_25
    ldr s13, [x16], #0x4
    ldr s12, [x25], #0x4
    ldr s11, [x9], #0x4
    ldr s10, [x24], #0x4
    ldr s9, [x11], #0x4
    ldr s8, [x23], #0x4
    ldr s7, [x28], #0x4
    ldr s6, [x22], #0x4
    ldr s5, [x12], #0x4
    ldr s4, [x21], #0x4
    ldr s3, [x27], #0x4
    ldr s2, [x20], #0x4
    ldr s1, [x10], #0x4
    ldr s0, [x8], #0x4
    ldr s31, [x26], #0x4
    ldr s30, [x7], #0x4
    tbz x15, #0, label_26
    ld1 { v13.h }[2], [x16]
    ld1 { v12.h }[2], [x25]
    ld1 { v11.h }[2], [x9]
    ld1 { v10.h }[2], [x24]
    ld1 { v9.h }[2], [x11]
    ld1 { v8.h }[2], [x23]
    ld1 { v7.h }[2], [x28]
    ld1 { v6.h }[2], [x22]
    ld1 { v5.h }[2], [x12]
    ld1 { v4.h }[2], [x21]
    ld1 { v3.h }[2], [x27]
    ld1 { v2.h }[2], [x20]
    ld1 { v1.h }[2], [x10]
    ld1 { v0.h }[2], [x8]
    ld1 { v31.h }[2], [x26]
    ld1 { v30.h }[2], [x7]
    b label_26
KAI_ASM_LABEL(label_25)  // 2 block: unique 11: Partial load: partial_0_0
    ldr h13, [x16, #0]
    ldr h12, [x25, #0]
    ldr h11, [x9, #0]
    ldr h10, [x24, #0]
    ldr h9, [x11, #0]
    ldr h8, [x23, #0]
    ldr h7, [x28, #0]
    ldr h6, [x22, #0]
    ldr h5, [x12, #0]
    ldr h4, [x21, #0]
    ldr h3, [x27, #0]
    ldr h2, [x20, #0]
    ldr h1, [x10, #0]
    ldr h0, [x8, #0]
    ldr h31, [x26, #0]
    ldr h30, [x7, #0]
KAI_ASM_LABEL(label_26)  // 2 block: unique 11: Partial load: Done
    zip1 v29.8h, v13.8h, v9.8h
    zip1 v28.8h, v11.8h, v7.8h
    zip1 v27.8h, v12.8h, v8.8h
    zip1 v26.8h, v10.8h, v6.8h
    zip1 v25.8h, v5.8h, v1.8h
    zip1 v24.8h, v3.8h, v31.8h
    zip1 v23.8h, v4.8h, v0.8h
    zip1 v22.8h, v2.8h, v30.8h
    zip1 v21.8h, v29.8h, v28.8h
    zip1 v20.8h, v27.8h, v26.8h
    zip1 v19.8h, v25.8h, v24.8h
    zip1 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_28
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_27)  // 2 block: unique 12: unique 13: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_27
KAI_ASM_LABEL(label_28)  // 2 block: unique 12: unique 13: Padding done
    cmp x15, #0x1
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_30
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_29)  // 2 block: unique 12: unique 14: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_29
KAI_ASM_LABEL(label_30)  // 2 block: unique 12: unique 14: Padding done
    cmp x15, #0x2
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v21.8h, v29.8h, v28.8h
    zip2 v20.8h, v27.8h, v26.8h
    zip2 v19.8h, v25.8h, v24.8h
    zip2 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_32
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_31)  // 2 block: unique 12: unique 15: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_31
KAI_ASM_LABEL(label_32)  // 2 block: unique 12: unique 15: Padding done
    cmp x15, #0x3
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_34
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_33)  // 2 block: unique 12: unique 16: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_33
KAI_ASM_LABEL(label_34)  // 2 block: unique 12: unique 16: Padding done
    cmp x15, #0x4
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v29.8h, v13.8h, v9.8h
    zip2 v28.8h, v11.8h, v7.8h
    zip2 v27.8h, v12.8h, v8.8h
    zip2 v26.8h, v10.8h, v6.8h
    zip2 v25.8h, v5.8h, v1.8h
    zip2 v24.8h, v3.8h, v31.8h
    zip2 v23.8h, v4.8h, v0.8h
    zip2 v22.8h, v2.8h, v30.8h
    zip1 v21.8h, v29.8h, v28.8h
    zip1 v20.8h, v27.8h, v26.8h
    zip1 v19.8h, v25.8h, v24.8h
    zip1 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_36
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_35)  // 2 block: unique 12: unique 17: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_35
KAI_ASM_LABEL(label_36)  // 2 block: unique 12: unique 17: Padding done
    cmp x15, #0x5
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_38
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_37)  // 2 block: unique 12: unique 18: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_37
KAI_ASM_LABEL(label_38)  // 2 block: unique 12: unique 18: Padding done
    cmp x15, #0x6
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v21.8h, v29.8h, v28.8h
    zip2 v20.8h, v27.8h, v26.8h
    zip2 v19.8h, v25.8h, v24.8h
    zip2 v18.8h, v23.8h, v22.8h
    zip1 v17.8h, v21.8h, v20.8h
    zip1 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_40
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_39)  // 2 block: unique 12: unique 19: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_39
KAI_ASM_LABEL(label_40)  // 2 block: unique 12: unique 19: Padding done
    cmp x15, #0x7
    add x13, x13, x5, LSL #4
    ble label_43
    zip2 v17.8h, v21.8h, v20.8h
    zip2 v16.8h, v19.8h, v18.8h
    str q17, [x13, #0]
    str q16, [x13, #0x10]
    cbz x14, label_42
    add x6, x13, #0x20
    mov x3, x14
KAI_ASM_LABEL(label_41)  // 2 block: unique 12: unique 20: Padding loop
    subs x3, x3, #0x1
    str q14, [x6], #0x10
    bne label_41
KAI_ASM_LABEL(label_42)  // 2 block: unique 12: unique 20: Padding done
    add x13, x13, x5, LSL #4
KAI_ASM_LABEL(label_43)  // 2 block: unique 12: Done
KAI_ASM_LABEL(label_44)  // 2 block: Oddments skip
    add x0, x0, #0x20
    ldr x3, [SP, #0x10]
    ldr x11, [SP, #0x18]
    add x6, x13, #0x20
    ldr x10, [SP, #0x20]
    b label_132
KAI_ASM_LABEL(label_45)  // 2 skip
    cmp x11, #0x8
    blt label_88
    mov x9, x1
    mov x28, x2
    sub x10, x10, #0x1
    subs x11, x11, #0x8
    csel x27, x10, XZR, EQ
    mov x26, x0
    add x25, x9, x4, LSL #2
    add x24, x9, x4, LSL #1
    add x23, x25, x4, LSL #1
    cmp x28, #0x8
    add x22, x9, x4
    add x21, x24, x4
    add x20, x25, x4
    add x8, x23, x4
    add x1, x9, x4, LSL #3
    blt label_65
KAI_ASM_LABEL(label_46)  // 1 block: Column loop
    ldr q30, [x9], #0x10
    ldr q29, [x22], #0x10
    ldr q28, [x24], #0x10
    ldr q27, [x21], #0x10
    ldr q26, [x25], #0x10
    ldr q25, [x20], #0x10
    ldr q24, [x23], #0x10
    ldr q23, [x8], #0x10
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_49
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_48)  // 1 block: unique 22: unique 23: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_48
KAI_ASM_LABEL(label_49)  // 1 block: unique 22: unique 23: Padding done
    add x26, x26, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_51
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_50)  // 1 block: unique 22: unique 24: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_50
KAI_ASM_LABEL(label_51)  // 1 block: unique 22: unique 24: Padding done
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    add x26, x26, x5, LSL #4
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_53
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_52)  // 1 block: unique 22: unique 25: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_52
KAI_ASM_LABEL(label_53)  // 1 block: unique 22: unique 25: Padding done
    add x26, x26, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_55
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_54)  // 1 block: unique 22: unique 26: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_54
KAI_ASM_LABEL(label_55)  // 1 block: unique 22: unique 26: Padding done
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    add x26, x26, x5, LSL #4
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_57
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_56)  // 1 block: unique 22: unique 27: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_56
KAI_ASM_LABEL(label_57)  // 1 block: unique 22: unique 27: Padding done
    add x26, x26, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_59
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_58)  // 1 block: unique 22: unique 28: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_58
KAI_ASM_LABEL(label_59)  // 1 block: unique 22: unique 28: Padding done
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    add x26, x26, x5, LSL #4
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_61
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_60)  // 1 block: unique 22: unique 29: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_60
KAI_ASM_LABEL(label_61)  // 1 block: unique 22: unique 29: Padding done
    add x26, x26, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_63
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_62)  // 1 block: unique 22: unique 30: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_62
KAI_ASM_LABEL(label_63)  // 1 block: unique 22: unique 30: Padding done
    add x26, x26, x5, LSL #4
    sub x28, x28, #0x8
    cmp x28, #0x8
    bge label_46
KAI_ASM_LABEL(label_65)  // 1 block: Loop skip
    cbz x28, label_87
    tbz x28, #2, label_67
    ldr d30, [x9], #0x8
    ldr d29, [x22], #0x8
    ldr d28, [x24], #0x8
    ldr d27, [x21], #0x8
    ldr d26, [x25], #0x8
    ldr d25, [x20], #0x8
    ldr d24, [x23], #0x8
    ldr d23, [x8], #0x8
    tbz x28, #1, label_66
    ld1 { v30.s }[2], [x9], #0x4
    ld1 { v29.s }[2], [x22], #0x4
    ld1 { v28.s }[2], [x24], #0x4
    ld1 { v27.s }[2], [x21], #0x4
    ld1 { v26.s }[2], [x25], #0x4
    ld1 { v25.s }[2], [x20], #0x4
    ld1 { v24.s }[2], [x23], #0x4
    ld1 { v23.s }[2], [x8], #0x4
    tbz x28, #0, label_69
    ld1 { v30.h }[6], [x9]
    ld1 { v29.h }[6], [x22]
    ld1 { v28.h }[6], [x24]
    ld1 { v27.h }[6], [x21]
    ld1 { v26.h }[6], [x25]
    ld1 { v25.h }[6], [x20]
    ld1 { v24.h }[6], [x23]
    ld1 { v23.h }[6], [x8]
    b label_69
KAI_ASM_LABEL(label_66)  // 1 block: unique 31: Partial load: partial_0_4
    tbz x28, #0, label_69
    ld1 { v30.h }[4], [x9]
    ld1 { v29.h }[4], [x22]
    ld1 { v28.h }[4], [x24]
    ld1 { v27.h }[4], [x21]
    ld1 { v26.h }[4], [x25]
    ld1 { v25.h }[4], [x20]
    ld1 { v24.h }[4], [x23]
    ld1 { v23.h }[4], [x8]
    b label_69
KAI_ASM_LABEL(label_67)  // 1 block: unique 31: Partial load: partial_1_0
    tbz x28, #1, label_68
    ldr s30, [x9], #0x4
    ldr s29, [x22], #0x4
    ldr s28, [x24], #0x4
    ldr s27, [x21], #0x4
    ldr s26, [x25], #0x4
    ldr s25, [x20], #0x4
    ldr s24, [x23], #0x4
    ldr s23, [x8], #0x4
    tbz x28, #0, label_69
    ld1 { v30.h }[2], [x9]
    ld1 { v29.h }[2], [x22]
    ld1 { v28.h }[2], [x24]
    ld1 { v27.h }[2], [x21]
    ld1 { v26.h }[2], [x25]
    ld1 { v25.h }[2], [x20]
    ld1 { v24.h }[2], [x23]
    ld1 { v23.h }[2], [x8]
    b label_69
KAI_ASM_LABEL(label_68)  // 1 block: unique 31: Partial load: partial_0_0
    ldr h30, [x9, #0]
    ldr h29, [x22, #0]
    ldr h28, [x24, #0]
    ldr h27, [x21, #0]
    ldr h26, [x25, #0]
    ldr h25, [x20, #0]
    ldr h24, [x23, #0]
    ldr h23, [x8, #0]
KAI_ASM_LABEL(label_69)  // 1 block: unique 31: Partial load: Done
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_71
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_70)  // 1 block: unique 32: unique 33: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_70
KAI_ASM_LABEL(label_71)  // 1 block: unique 32: unique 33: Padding done
    cmp x28, #0x1
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_73
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_72)  // 1 block: unique 32: unique 34: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_72
KAI_ASM_LABEL(label_73)  // 1 block: unique 32: unique 34: Padding done
    cmp x28, #0x2
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_75
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_74)  // 1 block: unique 32: unique 35: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_74
KAI_ASM_LABEL(label_75)  // 1 block: unique 32: unique 35: Padding done
    cmp x28, #0x3
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_77
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_76)  // 1 block: unique 32: unique 36: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_76
KAI_ASM_LABEL(label_77)  // 1 block: unique 32: unique 36: Padding done
    cmp x28, #0x4
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_79
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_78)  // 1 block: unique 32: unique 37: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_78
KAI_ASM_LABEL(label_79)  // 1 block: unique 32: unique 37: Padding done
    cmp x28, #0x5
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_81
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_80)  // 1 block: unique 32: unique 38: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_80
KAI_ASM_LABEL(label_81)  // 1 block: unique 32: unique 38: Padding done
    cmp x28, #0x6
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_83
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_82)  // 1 block: unique 32: unique 39: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_82
KAI_ASM_LABEL(label_83)  // 1 block: unique 32: unique 39: Padding done
    cmp x28, #0x7
    add x26, x26, x5, LSL #4
    ble label_86
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x26, #0]
    cbz x27, label_85
    add x7, x26, #0x10
    mov x6, x27
KAI_ASM_LABEL(label_84)  // 1 block: unique 32: unique 40: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_84
KAI_ASM_LABEL(label_85)  // 1 block: unique 32: unique 40: Padding done
    add x26, x26, x5, LSL #4
KAI_ASM_LABEL(label_86)  // 1 block: unique 32: Done
KAI_ASM_LABEL(label_87)  // 1 block: Oddments skip
    add x0, x0, #0x10
    add x6, x26, #0x10
    b label_132
KAI_ASM_LABEL(label_88)  // 1 skip
    cbz x11, label_131
    mov x27, x1
    mov x26, x2
    movi v30.16b, #0
    movi v29.16b, #0
    sub x10, x10, #0x1
    movi v28.16b, #0
    movi v27.16b, #0
    mov x25, x0
    movi v26.16b, #0
    movi v25.16b, #0
    add x24, x27, x4, LSL #1
    add x23, x27, x4, LSL #2
    cmp x26, #0x8
    add x22, x27, x4
    movi v24.16b, #0
    add x21, x24, x4
    add x20, x23, x4
    add x8, x23, x4, LSL #1
    add x1, x27, x4, LSL #3
    blt label_108
KAI_ASM_LABEL(label_89)  // Oddments: Column loop
    ldr q23, [x27], #0x10
    cmp x11, #0x2
    blt label_90
    ldr q30, [x22], #0x10
    beq label_90
    ldr q29, [x24], #0x10
    cmp x11, #0x4
    blt label_90
    ldr q28, [x21], #0x10
    beq label_90
    ldr q27, [x23], #0x10
    cmp x11, #0x6
    blt label_90
    ldr q26, [x20], #0x10
    beq label_90
    ldr q25, [x8], #0x10
KAI_ASM_LABEL(label_90)  // Oddments: unique 41: Done
    zip1 v22.8h, v23.8h, v27.8h
    zip1 v21.8h, v29.8h, v25.8h
    zip1 v20.8h, v30.8h, v26.8h
    zip1 v19.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_92
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_91)  // Oddments: unique 42: unique 43: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_91
KAI_ASM_LABEL(label_92)  // Oddments: unique 42: unique 43: Padding done
    add x25, x25, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_94
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_93)  // Oddments: unique 42: unique 44: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_93
KAI_ASM_LABEL(label_94)  // Oddments: unique 42: unique 44: Padding done
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    add x25, x25, x5, LSL #4
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_96
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_95)  // Oddments: unique 42: unique 45: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_95
KAI_ASM_LABEL(label_96)  // Oddments: unique 42: unique 45: Padding done
    add x25, x25, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_98
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_97)  // Oddments: unique 42: unique 46: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_97
KAI_ASM_LABEL(label_98)  // Oddments: unique 42: unique 46: Padding done
    zip2 v22.8h, v23.8h, v27.8h
    zip2 v21.8h, v29.8h, v25.8h
    add x25, x25, x5, LSL #4
    zip2 v20.8h, v30.8h, v26.8h
    zip2 v19.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_100
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_99)  // Oddments: unique 42: unique 47: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_99
KAI_ASM_LABEL(label_100)  // Oddments: unique 42: unique 47: Padding done
    add x25, x25, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_102
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_101)  // Oddments: unique 42: unique 48: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_101
KAI_ASM_LABEL(label_102)  // Oddments: unique 42: unique 48: Padding done
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    add x25, x25, x5, LSL #4
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_104
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_103)  // Oddments: unique 42: unique 49: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_103
KAI_ASM_LABEL(label_104)  // Oddments: unique 42: unique 49: Padding done
    add x25, x25, x5, LSL #4
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_106
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_105)  // Oddments: unique 42: unique 50: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_105
KAI_ASM_LABEL(label_106)  // Oddments: unique 42: unique 50: Padding done
    add x25, x25, x5, LSL #4
    sub x26, x26, #0x8
    cmp x26, #0x8
    bge label_89
KAI_ASM_LABEL(label_108)  // Oddments: Loop skip
    cbz x26, label_130
    mov x6, SP
    cmp x11, #0x2
    csel x22, x22, x6, GE
    csel x24, x24, x6, GT
    cmp x11, #0x4
    csel x21, x21, x6, GE
    csel x23, x23, x6, GT
    cmp x11, #0x6
    csel x20, x20, x6, GE
    csel x8, x8, x6, GT
    tbz x26, #2, label_110
    ldr d23, [x27], #0x8
    ldr d30, [x22], #0x8
    ldr d29, [x24], #0x8
    ldr d28, [x21], #0x8
    ldr d27, [x23], #0x8
    ldr d26, [x20], #0x8
    ldr d25, [x8], #0x8
    ldr d24, [x6], #0x8
    tbz x26, #1, label_109
    ld1 { v23.s }[2], [x27], #0x4
    ld1 { v30.s }[2], [x22], #0x4
    ld1 { v29.s }[2], [x24], #0x4
    ld1 { v28.s }[2], [x21], #0x4
    ld1 { v27.s }[2], [x23], #0x4
    ld1 { v26.s }[2], [x20], #0x4
    ld1 { v25.s }[2], [x8], #0x4
    ld1 { v24.s }[2], [x6], #0x4
    tbz x26, #0, label_112
    ld1 { v23.h }[6], [x27]
    ld1 { v30.h }[6], [x22]
    ld1 { v29.h }[6], [x24]
    ld1 { v28.h }[6], [x21]
    ld1 { v27.h }[6], [x23]
    ld1 { v26.h }[6], [x20]
    ld1 { v25.h }[6], [x8]
    ld1 { v24.h }[6], [x6]
    b label_112
KAI_ASM_LABEL(label_109)  // Oddments: unique 51: Partial load: partial_0_4
    tbz x26, #0, label_112
    ld1 { v23.h }[4], [x27]
    ld1 { v30.h }[4], [x22]
    ld1 { v29.h }[4], [x24]
    ld1 { v28.h }[4], [x21]
    ld1 { v27.h }[4], [x23]
    ld1 { v26.h }[4], [x20]
    ld1 { v25.h }[4], [x8]
    ld1 { v24.h }[4], [x6]
    b label_112
KAI_ASM_LABEL(label_110)  // Oddments: unique 51: Partial load: partial_1_0
    tbz x26, #1, label_111
    ldr s23, [x27], #0x4
    ldr s30, [x22], #0x4
    ldr s29, [x24], #0x4
    ldr s28, [x21], #0x4
    ldr s27, [x23], #0x4
    ldr s26, [x20], #0x4
    ldr s25, [x8], #0x4
    ldr s24, [x6], #0x4
    tbz x26, #0, label_112
    ld1 { v23.h }[2], [x27]
    ld1 { v30.h }[2], [x22]
    ld1 { v29.h }[2], [x24]
    ld1 { v28.h }[2], [x21]
    ld1 { v27.h }[2], [x23]
    ld1 { v26.h }[2], [x20]
    ld1 { v25.h }[2], [x8]
    ld1 { v24.h }[2], [x6]
    b label_112
KAI_ASM_LABEL(label_111)  // Oddments: unique 51: Partial load: partial_0_0
    ldr h23, [x27, #0]
    ldr h30, [x22, #0]
    ldr h29, [x24, #0]
    ldr h28, [x21, #0]
    ldr h27, [x23, #0]
    ldr h26, [x20, #0]
    ldr h25, [x8, #0]
    ldr h24, [x6, #0]
KAI_ASM_LABEL(label_112)  // Oddments: unique 51: Partial load: Done
    zip1 v22.8h, v23.8h, v27.8h
    zip1 v21.8h, v29.8h, v25.8h
    zip1 v20.8h, v30.8h, v26.8h
    zip1 v19.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_114
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_113)  // Oddments: unique 52: unique 53: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_113
KAI_ASM_LABEL(label_114)  // Oddments: unique 52: unique 53: Padding done
    cmp x26, #0x1
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_116
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_115)  // Oddments: unique 52: unique 54: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_115
KAI_ASM_LABEL(label_116)  // Oddments: unique 52: unique 54: Padding done
    cmp x26, #0x2
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_118
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_117)  // Oddments: unique 52: unique 55: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_117
KAI_ASM_LABEL(label_118)  // Oddments: unique 52: unique 55: Padding done
    cmp x26, #0x3
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_120
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_119)  // Oddments: unique 52: unique 56: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_119
KAI_ASM_LABEL(label_120)  // Oddments: unique 52: unique 56: Padding done
    cmp x26, #0x4
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v22.8h, v23.8h, v27.8h
    zip2 v21.8h, v29.8h, v25.8h
    zip2 v20.8h, v30.8h, v26.8h
    zip2 v19.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_122
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_121)  // Oddments: unique 52: unique 57: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_121
KAI_ASM_LABEL(label_122)  // Oddments: unique 52: unique 57: Padding done
    cmp x26, #0x5
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_124
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_123)  // Oddments: unique 52: unique 58: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_123
KAI_ASM_LABEL(label_124)  // Oddments: unique 52: unique 58: Padding done
    cmp x26, #0x6
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_126
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_125)  // Oddments: unique 52: unique 59: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_125
KAI_ASM_LABEL(label_126)  // Oddments: unique 52: unique 59: Padding done
    cmp x26, #0x7
    add x25, x25, x5, LSL #4
    ble label_129
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x25, #0]
    cbz x10, label_128
    add x7, x25, #0x10
    mov x6, x10
KAI_ASM_LABEL(label_127)  // Oddments: unique 52: unique 60: Padding loop
    subs x6, x6, #0x1
    str q14, [x7], #0x10
    bne label_127
KAI_ASM_LABEL(label_128)  // Oddments: unique 52: unique 60: Padding done
    add x25, x25, x5, LSL #4
KAI_ASM_LABEL(label_129)  // Oddments: unique 52: Done
KAI_ASM_LABEL(label_130)  // Oddments: Oddments skip
    add x0, x0, #0x10
    add x6, x25, #0x10
    mov x11, #0
KAI_ASM_LABEL(label_131)  // No oddments
KAI_ASM_LABEL(label_132)  // Block loop end
    cbnz x11, label_2
    sub x0, x6, x5, LSL #4
    add x0, x0, x10, LSL #4
    cbnz x3, label_1
    add SP, SP, #0x30
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldr d14, [sp, 120]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleave_8_2x1)

    KAI_ASM_END
