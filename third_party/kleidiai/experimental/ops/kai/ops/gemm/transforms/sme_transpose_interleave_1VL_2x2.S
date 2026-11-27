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

    KAI_ASM_CODE(sme_transpose_interleave_1VL_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_1VL_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_1VL_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_1VL_2x2)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cnth x5
    mov z16.h, #0
    add x6, x6, #0x1
    mov x24, x4
    bfm x6, XZR, #0, #0  // bfc x6, #0, #0x1
    ptrue p1.b
    lsr x5, x5, #0x1
    lsl x6, x6, #0x1
    addvl SP, SP, #-2
    cmp x24, #0x4
    mul x23, x6, x5
    st1h { z16.h }, p1, [SP]
    mov x22, SP
    st1h { z16.h }, p1, [SP, #1, MUL VL]
    blt label_6
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    mov x20, x2
    add x8, x21, x3
    cnth x6, ALL, MUL #2
    add x5, x8, x3
    cmp x20, x6
    add x4, x5, x3
    mov x7, x0
    add x1, x4, x3
    blt label_3
KAI_ASM_LABEL(label_2)  // Main row loop: Unroll column loop
    ld1h { z18.h }, p1/Z, [x21]
    sub x20, x20, x6
    ld1h { z17.h }, p1/Z, [x8]
    cmp x20, x6
    ld1h { z20.h }, p1/Z, [x5]
    ld1h { z16.h }, p1/Z, [x4]
    ld1h { z23.h }, p1/Z, [x21, #1, MUL VL]
    zip1 z19.h, z18.h, z17.h
    zip2 z22.h, z18.h, z17.h
    addvl x21, x21, #2
    ld1h { z18.h }, p1/Z, [x8, #1, MUL VL]
    addvl x8, x8, #2
    ld1h { z21.h }, p1/Z, [x5, #1, MUL VL]
    zip1 z17.h, z20.h, z16.h
    zip2 z20.h, z20.h, z16.h
    addvl x5, x5, #2
    ld1h { z16.h }, p1/Z, [x4, #1, MUL VL]
    addvl x4, x4, #2
    st1h { z19.h }, p1, [x7]
    zip1 z19.h, z23.h, z18.h
    zip2 z18.h, z23.h, z18.h
    st1h { z17.h }, p1, [x7, #1, MUL VL]
    add x7, x7, x23
    zip1 z17.h, z21.h, z16.h
    zip2 z16.h, z21.h, z16.h
    st1h { z22.h }, p1, [x7]
    st1h { z20.h }, p1, [x7, #1, MUL VL]
    add x7, x7, x23
    st1h { z19.h }, p1, [x7]
    st1h { z17.h }, p1, [x7, #1, MUL VL]
    add x7, x7, x23
    st1h { z18.h }, p1, [x7]
    st1h { z16.h }, p1, [x7, #1, MUL VL]
    add x7, x7, x23
    bge label_2
KAI_ASM_LABEL(label_3)  // Main row loop: Unroll column loop skip
    cbz x20, label_5
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    whilelt p0.h, XZR, x20
    decw x20
    ld1h { z19.h }, p0/Z, [x21]
    cmp x20, #0
    incd x21, ALL, MUL #4
    ld1h { z17.h }, p0/Z, [x8]
    incd x8, ALL, MUL #4
    ld1h { z18.h }, p0/Z, [x5]
    incd x5, ALL, MUL #4
    ld1h { z16.h }, p0/Z, [x4]
    incd x4, ALL, MUL #4
    zip1 z17.h, z19.h, z17.h
    zip1 z16.h, z18.h, z16.h
    st1h { z17.h }, p1, [x7]
    st1h { z16.h }, p1, [x7, #1, MUL VL]
    add x7, x7, x23
    bgt label_4
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop skip
    sub x24, x24, #0x4
    addvl x0, x0, #2
    cmp x24, #0x4
    bge label_1
    cbz x24, label_12
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x21, x1
    cntw x4, ALL, MUL #8
    add x8, x21, x3
    cmp x24, #0x1
    mov x6, x2
    cnth x5, ALL, MUL #2
    add x1, x8, x3
    csel x8, x8, x22, GT
    csel x4, x4, XZR, GT
    cmp x6, x5
    mov x7, x0
    blt label_9
KAI_ASM_LABEL(label_8)  // Tail row loop: Unroll column loop
    ld1h { z18.h }, p1/Z, [x21]
    sub x6, x6, x5
    ld1h { z17.h }, p1/Z, [x8]
    cmp x6, x5
    ld1h { z20.h }, p1/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    ld1h { z16.h }, p1/Z, [x8, #1, MUL VL]
    add x8, x8, x4
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    st1h { z19.h }, p1, [x7]
    add x7, x7, x23
    st1h { z18.h }, p1, [x7]
    add x7, x7, x23
    st1h { z17.h }, p1, [x7]
    add x7, x7, x23
    st1h { z16.h }, p1, [x7]
    add x7, x7, x23
    bge label_8
KAI_ASM_LABEL(label_9)  // Tail row loop: Unroll column loop skip
    cbz x6, label_11
KAI_ASM_LABEL(label_10)  // Tail row loop: Column loop
    whilelt p0.h, XZR, x6
    decw x6
    ld1h { z17.h }, p0/Z, [x21]
    cmp x6, #0
    incd x21, ALL, MUL #4
    ld1h { z16.h }, p0/Z, [x8]
    incd x8, ALL, MUL #4
    zip1 z16.h, z17.h, z16.h
    st1h { z16.h }, p1, [x7]
    add x7, x7, x23
    bgt label_10
KAI_ASM_LABEL(label_11)  // Tail row loop: Column loop skip
    sub x24, x24, #0x2
    addvl x0, x0, #1
    cmp x24, #0x1
    bge label_7
KAI_ASM_LABEL(label_12)  // Done
    addvl SP, SP, #2
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_1VL_2x2)

    KAI_ASM_END
