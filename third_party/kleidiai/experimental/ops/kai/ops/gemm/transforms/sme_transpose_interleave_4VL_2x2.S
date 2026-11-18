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

    KAI_ASM_CODE(sme_transpose_interleave_4VL_2x2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_transpose_interleave_4VL_2x2)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_transpose_interleave_4VL_2x2)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_transpose_interleave_4VL_2x2)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x6, x4
    cnth x5, ALL, MUL #4
    mov z16.h, #0
    add x6, x6, #0x1
    mov x24, x4
    bfm x6, XZR, #0, #0  // bfc x6, #0, #0x1
    ptrue p2.b
    lsr x5, x5, #0x1
    lsl x6, x6, #0x1
    addvl SP, SP, #-2
    cmp x24, #0x4
    mul x23, x6, x5
    st1h { z16.h }, p2, [SP]
    mov x22, SP
    st1h { z16.h }, p2, [SP, #1, MUL VL]
    blt label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x21, x1
    mov x20, x0
    add x8, x21, x3
    mov x7, x2
    add x6, x8, x3
    add x5, x6, x3
    add x1, x5, x3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    mov x4, x7
    decw x7, ALL, MUL #4
    whilelt p1.h, XZR, x4
    dech x4
    whilelt p0.h, XZR, x4
    ld1h { z20.h }, p1/Z, [x21]
    cmp x7, #0
    ld1h { z19.h }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    ld1h { z17.h }, p1/Z, [x8]
    ld1h { z16.h }, p0/Z, [x8, #1, MUL VL]
    addvl x8, x8, #2
    ld1h { z18.h }, p1/Z, [x6]
    ld1h { z24.h }, p0/Z, [x6, #1, MUL VL]
    addvl x6, x6, #2
    zip1 z23.h, z20.h, z17.h
    zip2 z22.h, z20.h, z17.h
    ld1h { z17.h }, p1/Z, [x5]
    zip1 z21.h, z19.h, z16.h
    zip2 z20.h, z19.h, z16.h
    ld1h { z16.h }, p0/Z, [x5, #1, MUL VL]
    addvl x5, x5, #2
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    st1h { z23.h }, p2, [x20]
    zip1 z17.h, z24.h, z16.h
    zip2 z16.h, z24.h, z16.h
    st1h { z22.h }, p2, [x20, #1, MUL VL]
    st1h { z21.h }, p2, [x20, #2, MUL VL]
    st1h { z20.h }, p2, [x20, #3, MUL VL]
    st1h { z19.h }, p2, [x20, #4, MUL VL]
    st1h { z18.h }, p2, [x20, #5, MUL VL]
    st1h { z17.h }, p2, [x20, #6, MUL VL]
    st1h { z16.h }, p2, [x20, #7, MUL VL]
    add x20, x20, x23
    bgt label_2
    sub x24, x24, #0x4
    addvl x0, x0, #8
    cmp x24, #0x4
    bge label_1
    cbz x24, label_8
KAI_ASM_LABEL(label_4)  // Main loop skip
KAI_ASM_LABEL(label_5)  // Tail row loop: Head
    mov x21, x1
    cntw x4, ALL, MUL #8
    add x8, x21, x3
    cmp x24, #0x1
    add x1, x8, x3
    mov x20, x0
    csel x8, x8, x22, GT
    csel x6, x4, XZR, GT
    mov x5, x2
KAI_ASM_LABEL(label_6)  // Tail row loop: Column loop
    mov x4, x5
    decw x5, ALL, MUL #4
    whilelt p1.h, XZR, x4
    dech x4
    whilelt p0.h, XZR, x4
    ld1h { z18.h }, p1/Z, [x21]
    cmp x5, #0
    ld1h { z20.h }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    ld1h { z17.h }, p1/Z, [x8]
    ld1h { z16.h }, p0/Z, [x8, #1, MUL VL]
    add x8, x8, x6
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    st1h { z19.h }, p2, [x20]
    st1h { z18.h }, p2, [x20, #1, MUL VL]
    st1h { z17.h }, p2, [x20, #2, MUL VL]
    st1h { z16.h }, p2, [x20, #3, MUL VL]
    add x20, x20, x23
    bgt label_6
    sub x24, x24, #0x2
    addvl x0, x0, #4
    cmp x24, #0x1
    bge label_5
KAI_ASM_LABEL(label_8)  // Done
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_transpose_interleave_4VL_2x2)

    KAI_ASM_END
