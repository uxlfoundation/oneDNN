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

    KAI_ASM_CODE(a64_interleaved_nomerge_fp32_mla_8x12)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_interleaved_nomerge_fp32_mla_8x12)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_interleaved_nomerge_fp32_mla_8x12)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_interleaved_nomerge_fp32_mla_8x12)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ldr x10, [x0, #0]
    ldr x9, [x0, #0x20]
    ldr x28, [x0, #0x50]
    ldr x27, [x0, #0x58]
    cbz x10, label_34
    ldr x5, [x0, #0x60]
    ldr w4, [x0, #0xa0]
    mov x2, #0xff800000
    ldr w3, [x0, #0xa4]
    mov x1, #0x7f800000
    sub SP, SP, #0x30
    cmp x5, #0
    cset x26, NE
    cmp x4, x2
    cset x2, NE
    cmp x3, x1
    cset x1, NE
    orr x26, x26, x2, LSL #1
    orr x26, x26, x1, LSL #1
KAI_ASM_LABEL(label_1)  // Height loop
    add x25, x28, x27, LSL #2
    add x24, x28, x27, LSL #1
    ldr x23, [x0, #0x40]
    add x22, x25, x27, LSL #1
    cmp x10, #0x8
    add x21, x28, x27, LSL #3
    add x20, x28, x27
    add x8, x24, x27
    add x7, x25, x27
    add x6, x22, x27
    bge label_2
    cmp x10, #0x2
    mov x6, x28
    csel x20, x20, x28, GE
    csel x24, x24, x28, GT
    cmp x10, #0x4
    csel x8, x8, x28, GE
    csel x25, x25, x28, GT
    cmp x10, #0x6
    csel x7, x7, x28, GE
    csel x22, x22, x28, GT
KAI_ASM_LABEL(label_2)  // all rows valid
    ldr x5, [x0, #0x8]
    ldr x4, [x0, #0x30]
    mov x3, x9
KAI_ASM_LABEL(label_3)  // Width loop
    mov x9, x3
    tbz x26, #0, label_11
    cmp x5, #0xc
    blt label_4
    ldr q8, [x28, #0]
    ldr q9, [x28, #0x10]
    ldr q10, [x28, #0x20]
    ldr q11, [x20, #0]
    ldr q12, [x20, #0x10]
    ldr q13, [x20, #0x20]
    ldr q14, [x24, #0]
    ldr q15, [x24, #0x10]
    ldr q16, [x24, #0x20]
    ldr q17, [x8, #0]
    ldr q18, [x8, #0x10]
    ldr q19, [x8, #0x20]
    ldr q20, [x25, #0]
    ldr q21, [x25, #0x10]
    ldr q22, [x25, #0x20]
    ldr q23, [x7, #0]
    ldr q24, [x7, #0x10]
    ldr q25, [x7, #0x20]
    ldr q26, [x22, #0]
    ldr q27, [x22, #0x10]
    ldr q28, [x22, #0x20]
    ldr q29, [x6, #0]
    ldr q30, [x6, #0x10]
    ldr q31, [x6, #0x20]
    b label_20
KAI_ASM_LABEL(label_4)  // partial input
    tbz x5, #3, label_6
    ld1 { v29.4s }, [x6], #0x10
    ld1 { v26.4s }, [x22], #0x10
    ld1 { v23.4s }, [x7], #0x10
    ld1 { v20.4s }, [x25], #0x10
    ld1 { v17.4s }, [x8], #0x10
    ld1 { v14.4s }, [x24], #0x10
    ld1 { v11.4s }, [x20], #0x10
    ld1 { v8.4s }, [x28], #0x10
    ld1 { v30.4s }, [x6], #0x10
    ld1 { v27.4s }, [x22], #0x10
    ld1 { v24.4s }, [x7], #0x10
    ld1 { v21.4s }, [x25], #0x10
    ld1 { v18.4s }, [x8], #0x10
    ld1 { v15.4s }, [x24], #0x10
    ld1 { v12.4s }, [x20], #0x10
    ld1 { v9.4s }, [x28], #0x10
    tbz x5, #1, label_5
    ldr d31, [x6], #0x8
    ldr d28, [x22], #0x8
    mov x1, #0x28
    ldr d25, [x7], #0x8
    ldr d22, [x25], #0x8
    ldr d19, [x8], #0x8
    ldr d16, [x24], #0x8
    ldr d13, [x20], #0x8
    ldr d10, [x28], #0x8
    tbz x5, #0, label_10
    ld1 { v31.s }[2], [x6]
    ld1 { v28.s }[2], [x22]
    ld1 { v25.s }[2], [x7]
    ld1 { v22.s }[2], [x25]
    ld1 { v19.s }[2], [x8]
    ld1 { v16.s }[2], [x24]
    ld1 { v13.s }[2], [x20]
    ld1 { v10.s }[2], [x28]
    b label_10
KAI_ASM_LABEL(label_5)  // partial result load: partial_1_8
    mov x1, #0x20
    tbz x5, #0, label_10
    ldr s31, [x6, #0]
    ldr s28, [x22, #0]
    ldr s25, [x7, #0]
    ldr s22, [x25, #0]
    ldr s19, [x8, #0]
    ldr s16, [x24, #0]
    ldr s13, [x20, #0]
    ldr s10, [x28, #0]
    b label_10
KAI_ASM_LABEL(label_6)  // partial result load: partial_4_0
    tbz x5, #2, label_8
    ld1 { v29.4s }, [x6], #0x10
    ld1 { v26.4s }, [x22], #0x10
    ld1 { v23.4s }, [x7], #0x10
    ld1 { v20.4s }, [x25], #0x10
    ld1 { v17.4s }, [x8], #0x10
    ld1 { v14.4s }, [x24], #0x10
    ld1 { v11.4s }, [x20], #0x10
    ld1 { v8.4s }, [x28], #0x10
    tbz x5, #1, label_7
    ldr d30, [x6], #0x8
    ldr d27, [x22], #0x8
    mov x1, #0x18
    ldr d24, [x7], #0x8
    ldr d21, [x25], #0x8
    ldr d18, [x8], #0x8
    ldr d15, [x24], #0x8
    ldr d12, [x20], #0x8
    ldr d9, [x28], #0x8
    tbz x5, #0, label_10
    ld1 { v30.s }[2], [x6]
    ld1 { v27.s }[2], [x22]
    ld1 { v24.s }[2], [x7]
    ld1 { v21.s }[2], [x25]
    ld1 { v18.s }[2], [x8]
    ld1 { v15.s }[2], [x24]
    ld1 { v12.s }[2], [x20]
    ld1 { v9.s }[2], [x28]
    b label_10
KAI_ASM_LABEL(label_7)  // partial result load: partial_1_4
    mov x1, #0x10
    tbz x5, #0, label_10
    ldr s30, [x6, #0]
    ldr s27, [x22, #0]
    ldr s24, [x7, #0]
    ldr s21, [x25, #0]
    ldr s18, [x8, #0]
    ldr s15, [x24, #0]
    ldr s12, [x20, #0]
    ldr s9, [x28, #0]
    b label_10
KAI_ASM_LABEL(label_8)  // partial result load: partial_2_0
    tbz x5, #1, label_9
    ldr d29, [x6], #0x8
    ldr d26, [x22], #0x8
    mov x1, #0x8
    ldr d23, [x7], #0x8
    ldr d20, [x25], #0x8
    ldr d17, [x8], #0x8
    ldr d14, [x24], #0x8
    ldr d11, [x20], #0x8
    ldr d8, [x28], #0x8
    tbz x5, #0, label_10
    ld1 { v29.s }[2], [x6]
    ld1 { v26.s }[2], [x22]
    ld1 { v23.s }[2], [x7]
    ld1 { v20.s }[2], [x25]
    ld1 { v17.s }[2], [x8]
    ld1 { v14.s }[2], [x24]
    ld1 { v11.s }[2], [x20]
    ld1 { v8.s }[2], [x28]
    b label_10
KAI_ASM_LABEL(label_9)  // partial result load: partial_1_0
    ldr s29, [x6, #0]
    ldr s26, [x22, #0]
    mov x1, #0
    ldr s23, [x7, #0]
    ldr s20, [x25, #0]
    ldr s17, [x8, #0]
    ldr s14, [x24, #0]
    ldr s11, [x20, #0]
    ldr s8, [x28, #0]
KAI_ASM_LABEL(label_10)  // partial result load: Done
    sub x28, x28, x1
    sub x20, x20, x1
    sub x24, x24, x1
    sub x8, x8, x1
    sub x25, x25, x1
    sub x7, x7, x1
    sub x22, x22, x1
    sub x6, x6, x1
    b label_20
KAI_ASM_LABEL(label_11)  // no accumulate
    cbz x23, label_18
    cmp x5, #0xc
    bge label_16
    tbnz x26, #31, label_15
    mov x2, x5
    orr x26, x26, #0x80000000
    mov x1, SP
    cmp x2, #0x4
    blt label_13
KAI_ASM_LABEL(label_12)  // Partial bias load: vector loop
    ld1 { v8.4s }, [x23], #0x10
    sub x2, x2, #0x4
    cmp x2, #0x4
    st1 { v8.4s }, [x1], #0x10
    bge label_12
KAI_ASM_LABEL(label_13)  // Partial bias load: vector loop skip
    cbz x2, label_15
KAI_ASM_LABEL(label_14)  // Partial bias load: element loop
    ldr s8, [x23], #0x4
    subs x2, x2, #0x1
    str s8, [x1], #0x4
    bgt label_14
KAI_ASM_LABEL(label_15)  // Partial bias load: read skip
    mov x1, SP
    ldr q8, [x1, #0]
    ldr q9, [x1, #0x10]
    ldr q10, [x1, #0x20]
    b label_17
KAI_ASM_LABEL(label_16)  // full width bias
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    add x23, x23, #0x30
KAI_ASM_LABEL(label_17)  // bias load done
    mov v11.16b, v8.16b
    mov v12.16b, v9.16b
    mov v13.16b, v10.16b
    b label_19
KAI_ASM_LABEL(label_18)  // no bias
    movi v8.16b, #0
    movi v9.16b, #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
KAI_ASM_LABEL(label_19)  // init copy rows
    mov v14.16b, v8.16b
    mov v15.16b, v9.16b
    mov v16.16b, v10.16b
    mov v17.16b, v11.16b
    mov v18.16b, v12.16b
    mov v19.16b, v13.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v12.16b
    mov v25.16b, v13.16b
    mov v26.16b, v8.16b
    mov v27.16b, v9.16b
    mov v28.16b, v10.16b
    mov v29.16b, v11.16b
    mov v30.16b, v12.16b
    mov v31.16b, v13.16b
KAI_ASM_LABEL(label_20)  // acc init done
    ldr q0, [x9, #0]
    ldr q1, [x9, #0x10]
    ldr q4, [x4, #0]
    ldr q5, [x4, #0x10]
    ldr q6, [x4, #0x20]
    ldr x1, [x0, #0x10]
    prfm pldl1keep, [x9, #0]
    prfm pldl1keep, [x4, #0]
    prfm pldl1keep, [x4, #0x40]
    prfm pldl1keep, [x9, #0x40]
    sub x1, x1, #0x1
    prfm pldl1keep, [x4, #0x80]
    prfm pldl1keep, [x9, #0x80]
    cmp x1, #0x4
    prfm pldl1keep, [x4, #0xc0]
    prfm pldl1keep, [x4, #0x100]
    prfm pldl1keep, [x9, #0xc0]
    prfm pldl1keep, [x4, #0x140]
    blt label_22
KAI_ASM_LABEL(label_21)  // main loop head
    ldr q2, [x9, #0x20]
    ldr q3, [x9, #0x30]
    fmla v8.4s, v4.4s, v0.s[0]
    ldr q7, [x4, #0x30]
    fmla v11.4s, v4.4s, v0.s[1]
    fmla v14.4s, v4.4s, v0.s[2]
    fmla v17.4s, v4.4s, v0.s[3]
    fmla v20.4s, v4.4s, v1.s[0]
    sub x1, x1, #0x4
    fmla v23.4s, v4.4s, v1.s[1]
    fmla v26.4s, v4.4s, v1.s[2]
    cmp x1, #0x4
    fmla v29.4s, v4.4s, v1.s[3]
    ldr q4, [x4, #0x40]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v12.4s, v5.4s, v0.s[1]
    fmla v15.4s, v5.4s, v0.s[2]
    prfm pldl1keep, [x9, #0x100]
    fmla v18.4s, v5.4s, v0.s[3]
    fmla v21.4s, v5.4s, v1.s[0]
    prfm pldl1keep, [x4, #0x180]
    fmla v24.4s, v5.4s, v1.s[1]
    fmla v27.4s, v5.4s, v1.s[2]
    prfm pldl1keep, [x4, #0x1c0]
    fmla v30.4s, v5.4s, v1.s[3]
    ldr q5, [x4, #0x50]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v13.4s, v6.4s, v0.s[1]
    fmla v16.4s, v6.4s, v0.s[2]
    prfm pldl1keep, [x9, #0x140]
    fmla v19.4s, v6.4s, v0.s[3]
    ldr q0, [x9, #0x40]
    fmla v22.4s, v6.4s, v1.s[0]
    fmla v25.4s, v6.4s, v1.s[1]
    fmla v28.4s, v6.4s, v1.s[2]
    prfm pldl1keep, [x4, #0x200]
    fmla v31.4s, v6.4s, v1.s[3]
    ldr q1, [x9, #0x50]
    ldr q6, [x4, #0x60]
    fmla v8.4s, v7.4s, v2.s[0]
    fmla v11.4s, v7.4s, v2.s[1]
    fmla v14.4s, v7.4s, v2.s[2]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v20.4s, v7.4s, v3.s[0]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v26.4s, v7.4s, v3.s[2]
    fmla v29.4s, v7.4s, v3.s[3]
    ldr q7, [x4, #0x70]
    fmla v9.4s, v4.4s, v2.s[0]
    fmla v12.4s, v4.4s, v2.s[1]
    fmla v15.4s, v4.4s, v2.s[2]
    fmla v18.4s, v4.4s, v2.s[3]
    fmla v21.4s, v4.4s, v3.s[0]
    fmla v24.4s, v4.4s, v3.s[1]
    fmla v27.4s, v4.4s, v3.s[2]
    fmla v30.4s, v4.4s, v3.s[3]
    ldr q4, [x4, #0x80]
    fmla v10.4s, v5.4s, v2.s[0]
    fmla v13.4s, v5.4s, v2.s[1]
    fmla v16.4s, v5.4s, v2.s[2]
    fmla v19.4s, v5.4s, v2.s[3]
    ldr q2, [x9, #0x60]
    fmla v22.4s, v5.4s, v3.s[0]
    fmla v25.4s, v5.4s, v3.s[1]
    fmla v28.4s, v5.4s, v3.s[2]
    fmla v31.4s, v5.4s, v3.s[3]
    ldr q3, [x9, #0x70]
    ldr q5, [x4, #0x90]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v11.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v0.s[2]
    fmla v17.4s, v6.4s, v0.s[3]
    add x9, x9, #0x80
    fmla v20.4s, v6.4s, v1.s[0]
    fmla v23.4s, v6.4s, v1.s[1]
    fmla v26.4s, v6.4s, v1.s[2]
    fmla v29.4s, v6.4s, v1.s[3]
    ldr q6, [x4, #0xa0]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v12.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v0.s[2]
    fmla v18.4s, v7.4s, v0.s[3]
    fmla v21.4s, v7.4s, v1.s[0]
    fmla v24.4s, v7.4s, v1.s[1]
    fmla v27.4s, v7.4s, v1.s[2]
    fmla v30.4s, v7.4s, v1.s[3]
    ldr q7, [x4, #0xb0]
    fmla v10.4s, v4.4s, v0.s[0]
    fmla v13.4s, v4.4s, v0.s[1]
    add x4, x4, #0xc0
    fmla v16.4s, v4.4s, v0.s[2]
    fmla v19.4s, v4.4s, v0.s[3]
    ldr q0, [x9, #0]
    fmla v22.4s, v4.4s, v1.s[0]
    fmla v25.4s, v4.4s, v1.s[1]
    fmla v28.4s, v4.4s, v1.s[2]
    fmla v31.4s, v4.4s, v1.s[3]
    ldr q1, [x9, #0x10]
    ldr q4, [x4, #0]
    fmla v8.4s, v5.4s, v2.s[0]
    fmla v11.4s, v5.4s, v2.s[1]
    fmla v14.4s, v5.4s, v2.s[2]
    fmla v17.4s, v5.4s, v2.s[3]
    fmla v20.4s, v5.4s, v3.s[0]
    fmla v23.4s, v5.4s, v3.s[1]
    fmla v26.4s, v5.4s, v3.s[2]
    fmla v29.4s, v5.4s, v3.s[3]
    ldr q5, [x4, #0x10]
    fmla v9.4s, v6.4s, v2.s[0]
    fmla v12.4s, v6.4s, v2.s[1]
    fmla v15.4s, v6.4s, v2.s[2]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v21.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v3.s[1]
    fmla v27.4s, v6.4s, v3.s[2]
    fmla v30.4s, v6.4s, v3.s[3]
    ldr q6, [x4, #0x20]
    fmla v10.4s, v7.4s, v2.s[0]
    fmla v13.4s, v7.4s, v2.s[1]
    fmla v16.4s, v7.4s, v2.s[2]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v22.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v3.s[1]
    fmla v28.4s, v7.4s, v3.s[2]
    fmla v31.4s, v7.4s, v3.s[3]
    bge label_21
KAI_ASM_LABEL(label_22)  // main loop skip
    add x9, x9, #0x20
    add x4, x4, #0x30
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v11.4s, v4.4s, v0.s[1]
    fmla v14.4s, v4.4s, v0.s[2]
    fmla v17.4s, v4.4s, v0.s[3]
    fmla v20.4s, v4.4s, v1.s[0]
    fmla v23.4s, v4.4s, v1.s[1]
    fmla v26.4s, v4.4s, v1.s[2]
    fmla v29.4s, v4.4s, v1.s[3]
    fmla v9.4s, v5.4s, v0.s[0]
    fmla v12.4s, v5.4s, v0.s[1]
    fmla v15.4s, v5.4s, v0.s[2]
    fmla v18.4s, v5.4s, v0.s[3]
    fmla v21.4s, v5.4s, v1.s[0]
    fmla v24.4s, v5.4s, v1.s[1]
    fmla v27.4s, v5.4s, v1.s[2]
    fmla v30.4s, v5.4s, v1.s[3]
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v13.4s, v6.4s, v0.s[1]
    fmla v16.4s, v6.4s, v0.s[2]
    fmla v19.4s, v6.4s, v0.s[3]
    fmla v22.4s, v6.4s, v1.s[0]
    fmla v25.4s, v6.4s, v1.s[1]
    fmla v28.4s, v6.4s, v1.s[2]
    fmla v31.4s, v6.4s, v1.s[3]
    cbz x1, label_24
KAI_ASM_LABEL(label_23)  // odd loop
    ldr q0, [x9, #0]
    ldr q1, [x9, #0x10]
    subs x1, x1, #0x1
    ldr q7, [x4, #0]
    ldr q4, [x4, #0x10]
    add x9, x9, #0x20
    ldr q5, [x4, #0x20]
    add x4, x4, #0x30
    fmla v8.4s, v7.4s, v0.s[0]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v14.4s, v7.4s, v0.s[2]
    fmla v17.4s, v7.4s, v0.s[3]
    fmla v20.4s, v7.4s, v1.s[0]
    fmla v23.4s, v7.4s, v1.s[1]
    fmla v26.4s, v7.4s, v1.s[2]
    fmla v29.4s, v7.4s, v1.s[3]
    fmla v9.4s, v4.4s, v0.s[0]
    fmla v12.4s, v4.4s, v0.s[1]
    fmla v15.4s, v4.4s, v0.s[2]
    fmla v18.4s, v4.4s, v0.s[3]
    fmla v21.4s, v4.4s, v1.s[0]
    fmla v24.4s, v4.4s, v1.s[1]
    fmla v27.4s, v4.4s, v1.s[2]
    fmla v30.4s, v4.4s, v1.s[3]
    fmla v10.4s, v5.4s, v0.s[0]
    fmla v13.4s, v5.4s, v0.s[1]
    fmla v16.4s, v5.4s, v0.s[2]
    fmla v19.4s, v5.4s, v0.s[3]
    fmla v22.4s, v5.4s, v1.s[0]
    fmla v25.4s, v5.4s, v1.s[1]
    fmla v28.4s, v5.4s, v1.s[2]
    fmla v31.4s, v5.4s, v1.s[3]
    bne label_23
KAI_ASM_LABEL(label_24)  // multiply loop done
    tbz x26, #1, label_25
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    ld1r { v1.4s }, [x2]
    ld1r { v0.4s }, [x1]
    fmin v8.4s, v8.4s, v1.4s
    fmin v9.4s, v9.4s, v1.4s
    fmin v10.4s, v10.4s, v1.4s
    fmin v11.4s, v11.4s, v1.4s
    fmin v12.4s, v12.4s, v1.4s
    fmin v13.4s, v13.4s, v1.4s
    fmin v14.4s, v14.4s, v1.4s
    fmin v15.4s, v15.4s, v1.4s
    fmin v16.4s, v16.4s, v1.4s
    fmin v17.4s, v17.4s, v1.4s
    fmin v18.4s, v18.4s, v1.4s
    fmin v19.4s, v19.4s, v1.4s
    fmin v20.4s, v20.4s, v1.4s
    fmin v21.4s, v21.4s, v1.4s
    fmin v22.4s, v22.4s, v1.4s
    fmin v23.4s, v23.4s, v1.4s
    fmin v24.4s, v24.4s, v1.4s
    fmin v25.4s, v25.4s, v1.4s
    fmin v26.4s, v26.4s, v1.4s
    fmin v27.4s, v27.4s, v1.4s
    fmin v28.4s, v28.4s, v1.4s
    fmin v29.4s, v29.4s, v1.4s
    fmin v30.4s, v30.4s, v1.4s
    fmin v31.4s, v31.4s, v1.4s
    fmax v8.4s, v8.4s, v0.4s
    fmax v9.4s, v9.4s, v0.4s
    fmax v10.4s, v10.4s, v0.4s
    fmax v11.4s, v11.4s, v0.4s
    fmax v12.4s, v12.4s, v0.4s
    fmax v13.4s, v13.4s, v0.4s
    fmax v14.4s, v14.4s, v0.4s
    fmax v15.4s, v15.4s, v0.4s
    fmax v16.4s, v16.4s, v0.4s
    fmax v17.4s, v17.4s, v0.4s
    fmax v18.4s, v18.4s, v0.4s
    fmax v19.4s, v19.4s, v0.4s
    fmax v20.4s, v20.4s, v0.4s
    fmax v21.4s, v21.4s, v0.4s
    fmax v22.4s, v22.4s, v0.4s
    fmax v23.4s, v23.4s, v0.4s
    fmax v24.4s, v24.4s, v0.4s
    fmax v25.4s, v25.4s, v0.4s
    fmax v26.4s, v26.4s, v0.4s
    fmax v27.4s, v27.4s, v0.4s
    fmax v28.4s, v28.4s, v0.4s
    fmax v29.4s, v29.4s, v0.4s
    fmax v30.4s, v30.4s, v0.4s
    fmax v31.4s, v31.4s, v0.4s
KAI_ASM_LABEL(label_25)  // No activation
    cmp x5, #0xc
    blt label_26
    str q29, [x6, #0]
    str q30, [x6, #0x10]
    str q31, [x6, #0x20]
    add x6, x6, #0x30
    str q26, [x22, #0]
    str q27, [x22, #0x10]
    str q28, [x22, #0x20]
    add x22, x22, #0x30
    str q23, [x7, #0]
    str q24, [x7, #0x10]
    str q25, [x7, #0x20]
    add x7, x7, #0x30
    str q20, [x25, #0]
    str q21, [x25, #0x10]
    str q22, [x25, #0x20]
    add x25, x25, #0x30
    str q17, [x8, #0]
    str q18, [x8, #0x10]
    str q19, [x8, #0x20]
    add x8, x8, #0x30
    str q14, [x24, #0]
    str q15, [x24, #0x10]
    str q16, [x24, #0x20]
    add x24, x24, #0x30
    str q11, [x20, #0]
    str q12, [x20, #0x10]
    str q13, [x20, #0x20]
    add x20, x20, #0x30
    str q8, [x28, #0]
    str q9, [x28, #0x10]
    str q10, [x28, #0x20]
    add x28, x28, #0x30
    b label_33
KAI_ASM_LABEL(label_26)  // partial output
    tbz x5, #3, label_28
    st1 { v29.4s }, [x6], #0x10
    st1 { v30.4s }, [x6], #0x10
    st1 { v26.4s }, [x22], #0x10
    st1 { v27.4s }, [x22], #0x10
    st1 { v23.4s }, [x7], #0x10
    st1 { v24.4s }, [x7], #0x10
    st1 { v20.4s }, [x25], #0x10
    st1 { v21.4s }, [x25], #0x10
    st1 { v17.4s }, [x8], #0x10
    st1 { v18.4s }, [x8], #0x10
    st1 { v14.4s }, [x24], #0x10
    st1 { v15.4s }, [x24], #0x10
    st1 { v11.4s }, [x20], #0x10
    st1 { v12.4s }, [x20], #0x10
    st1 { v8.4s }, [x28], #0x10
    st1 { v9.4s }, [x28], #0x10
    tbz x5, #1, label_27
    str d31, [x6], #0x8
    str d28, [x22], #0x8
    str d25, [x7], #0x8
    str d22, [x25], #0x8
    str d19, [x8], #0x8
    str d16, [x24], #0x8
    str d13, [x20], #0x8
    str d10, [x28], #0x8
    tbz x5, #0, label_32
    st1 { v31.s }[2], [x6]
    st1 { v28.s }[2], [x22]
    st1 { v25.s }[2], [x7]
    st1 { v22.s }[2], [x25]
    st1 { v19.s }[2], [x8]
    st1 { v16.s }[2], [x24]
    st1 { v13.s }[2], [x20]
    st1 { v10.s }[2], [x28]
    b label_32
KAI_ASM_LABEL(label_27)  // partial result store: partial_1_8
    tbz x5, #0, label_32
    str s31, [x6, #0]
    str s28, [x22, #0]
    str s25, [x7, #0]
    str s22, [x25, #0]
    str s19, [x8, #0]
    str s16, [x24, #0]
    str s13, [x20, #0]
    str s10, [x28, #0]
    b label_32
KAI_ASM_LABEL(label_28)  // partial result store: partial_4_0
    tbz x5, #2, label_30
    st1 { v29.4s }, [x6], #0x10
    st1 { v26.4s }, [x22], #0x10
    st1 { v23.4s }, [x7], #0x10
    st1 { v20.4s }, [x25], #0x10
    st1 { v17.4s }, [x8], #0x10
    st1 { v14.4s }, [x24], #0x10
    st1 { v11.4s }, [x20], #0x10
    st1 { v8.4s }, [x28], #0x10
    tbz x5, #1, label_29
    str d30, [x6], #0x8
    str d27, [x22], #0x8
    str d24, [x7], #0x8
    str d21, [x25], #0x8
    str d18, [x8], #0x8
    str d15, [x24], #0x8
    str d12, [x20], #0x8
    str d9, [x28], #0x8
    tbz x5, #0, label_32
    st1 { v30.s }[2], [x6]
    st1 { v27.s }[2], [x22]
    st1 { v24.s }[2], [x7]
    st1 { v21.s }[2], [x25]
    st1 { v18.s }[2], [x8]
    st1 { v15.s }[2], [x24]
    st1 { v12.s }[2], [x20]
    st1 { v9.s }[2], [x28]
    b label_32
KAI_ASM_LABEL(label_29)  // partial result store: partial_1_4
    tbz x5, #0, label_32
    str s30, [x6, #0]
    str s27, [x22, #0]
    str s24, [x7, #0]
    str s21, [x25, #0]
    str s18, [x8, #0]
    str s15, [x24, #0]
    str s12, [x20, #0]
    str s9, [x28, #0]
    b label_32
KAI_ASM_LABEL(label_30)  // partial result store: partial_2_0
    tbz x5, #1, label_31
    str d29, [x6], #0x8
    str d26, [x22], #0x8
    str d23, [x7], #0x8
    str d20, [x25], #0x8
    str d17, [x8], #0x8
    str d14, [x24], #0x8
    str d11, [x20], #0x8
    str d8, [x28], #0x8
    tbz x5, #0, label_32
    st1 { v29.s }[2], [x6]
    st1 { v26.s }[2], [x22]
    st1 { v23.s }[2], [x7]
    st1 { v20.s }[2], [x25]
    st1 { v17.s }[2], [x8]
    st1 { v14.s }[2], [x24]
    st1 { v11.s }[2], [x20]
    st1 { v8.s }[2], [x28]
    b label_32
KAI_ASM_LABEL(label_31)  // partial result store: partial_1_0
    str s29, [x6, #0]
    str s26, [x22, #0]
    str s23, [x7, #0]
    str s20, [x25, #0]
    str s17, [x8, #0]
    str s14, [x24, #0]
    str s11, [x20, #0]
    str s8, [x28, #0]
KAI_ASM_LABEL(label_32)  // partial result store: Done
KAI_ASM_LABEL(label_33)  // store done
    subs x5, x5, #0xc
    bgt label_3
    subs x10, x10, #0x8
    mov x28, x21
    bgt label_1
    add SP, SP, #0x30
KAI_ASM_LABEL(label_34)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_interleaved_nomerge_fp32_mla_8x12)

    KAI_ASM_END
