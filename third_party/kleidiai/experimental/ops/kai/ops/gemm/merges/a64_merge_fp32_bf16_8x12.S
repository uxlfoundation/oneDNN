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

    KAI_ASM_CODE(a64_merge_fp32_bf16_8x12)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_a64_merge_fp32_bf16_8x12)

KAI_ASM_FUNCTION_TYPE(kai_ops_a64_merge_fp32_bf16_8x12)
KAI_ASM_FUNCTION_LABEL(kai_ops_a64_merge_fp32_bf16_8x12)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    mov x27, #0x8
    add x1, x0, #0xa4
    ld1r { v21.8h }, [x1]
    add x1, x0, #0xa0
    ld1r { v20.8h }, [x1]
    ldr x26, [x0, #0x58]
    ldr x1, [x0, #0x60]
    shll v21.4s, v21.4h, #0x10
    shll v20.4s, v20.4h, #0x10
    mul x27, x26, x27
    cbnz x1, label_66
    ldr x25, [x0, #0]
    ldr x5, [x0, #0x50]
    ldr x4, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // Initial: Row loop
    cmp x25, #0x7
    bgt label_58
    beq label_50
    cmp x25, #0x5
    bgt label_42
    beq label_34
    cmp x25, #0x3
    bgt label_26
    beq label_18
    cmp x25, #0x1
    bgt label_10
    ldr x2, [x0, #0x8]
    mov x24, x5
    ldr x1, [x0, #0x40]
    cmp x2, #0xc
    blt label_6
KAI_ASM_LABEL(label_3)  // Initial: Height 1: Block loop
    cbnz x1, label_4
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_5
KAI_ASM_LABEL(label_4)  // Initial: Height 1: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_5)  // Initial: Height 1: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    bge label_3
KAI_ASM_LABEL(label_6)  // Initial: Height 1: no full blocks
    cbz x2, label_9
KAI_ASM_LABEL(label_7)  // Initial: Height 1: Single loop
    movi v17.16b, #0
    cbz x1, label_8
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_8)  // Initial: Height 1: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    bne label_7
KAI_ASM_LABEL(label_9)  // Initial: Height 1: no oddments
    b label_108
KAI_ASM_LABEL(label_10)  // Initial: Height 2
    ldr x2, [x0, #0x8]
    mov x24, x5
    ldr x1, [x0, #0x40]
    cmp x2, #0xc
    add x23, x24, x26
    blt label_14
KAI_ASM_LABEL(label_11)  // Initial: Height 2: Block loop
    cbnz x1, label_12
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_13
KAI_ASM_LABEL(label_12)  // Initial: Height 2: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_13)  // Initial: Height 2: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    bge label_11
KAI_ASM_LABEL(label_14)  // Initial: Height 2: no full blocks
    cbz x2, label_17
KAI_ASM_LABEL(label_15)  // Initial: Height 2: Single loop
    movi v17.16b, #0
    cbz x1, label_16
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_16)  // Initial: Height 2: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    bne label_15
KAI_ASM_LABEL(label_17)  // Initial: Height 2: no oddments
    b label_108
KAI_ASM_LABEL(label_18)  // Initial: Height 3
    ldr x2, [x0, #0x8]
    mov x24, x5
    ldr x1, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    cmp x2, #0xc
    blt label_22
KAI_ASM_LABEL(label_19)  // Initial: Height 3: Block loop
    cbnz x1, label_20
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_21
KAI_ASM_LABEL(label_20)  // Initial: Height 3: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_21)  // Initial: Height 3: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    bge label_19
KAI_ASM_LABEL(label_22)  // Initial: Height 3: no full blocks
    cbz x2, label_25
KAI_ASM_LABEL(label_23)  // Initial: Height 3: Single loop
    movi v17.16b, #0
    cbz x1, label_24
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_24)  // Initial: Height 3: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    bne label_23
KAI_ASM_LABEL(label_25)  // Initial: Height 3: no oddments
    b label_108
KAI_ASM_LABEL(label_26)  // Initial: Height 4
    mov x24, x5
    ldr x2, [x0, #0x8]
    ldr x1, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    cmp x2, #0xc
    blt label_30
KAI_ASM_LABEL(label_27)  // Initial: Height 4: Block loop
    cbnz x1, label_28
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_29
KAI_ASM_LABEL(label_28)  // Initial: Height 4: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_29)  // Initial: Height 4: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr q16, [x4, #0x90]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0]
    ldr q16, [x4, #0xa0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x8]
    ldr q16, [x4, #0xb0]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    bge label_27
KAI_ASM_LABEL(label_30)  // Initial: Height 4: no full blocks
    cbz x2, label_33
KAI_ASM_LABEL(label_31)  // Initial: Height 4: Single loop
    movi v17.16b, #0
    cbz x1, label_32
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_32)  // Initial: Height 4: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr s16, [x4, #0x90]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    bne label_31
KAI_ASM_LABEL(label_33)  // Initial: Height 4: no oddments
    b label_108
KAI_ASM_LABEL(label_34)  // Initial: Height 5
    mov x24, x5
    ldr x2, [x0, #0x8]
    ldr x1, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    cmp x2, #0xc
    blt label_38
KAI_ASM_LABEL(label_35)  // Initial: Height 5: Block loop
    cbnz x1, label_36
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_37
KAI_ASM_LABEL(label_36)  // Initial: Height 5: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_37)  // Initial: Height 5: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr q16, [x4, #0x90]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0]
    ldr q16, [x4, #0xa0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x8]
    ldr q16, [x4, #0xb0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr q16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0]
    ldr q16, [x4, #0xd0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x8]
    ldr q16, [x4, #0xe0]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    bge label_35
KAI_ASM_LABEL(label_38)  // Initial: Height 5: no full blocks
    cbz x2, label_41
KAI_ASM_LABEL(label_39)  // Initial: Height 5: Single loop
    movi v17.16b, #0
    cbz x1, label_40
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_40)  // Initial: Height 5: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr s16, [x4, #0x90]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr s16, [x4, #0xc0]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    bne label_39
KAI_ASM_LABEL(label_41)  // Initial: Height 5: no oddments
    b label_108
KAI_ASM_LABEL(label_42)  // Initial: Height 6
    mov x24, x5
    ldr x2, [x0, #0x8]
    ldr x1, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    cmp x2, #0xc
    add x8, x20, x26
    blt label_46
KAI_ASM_LABEL(label_43)  // Initial: Height 6: Block loop
    cbnz x1, label_44
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_45
KAI_ASM_LABEL(label_44)  // Initial: Height 6: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_45)  // Initial: Height 6: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr q16, [x4, #0x90]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0]
    ldr q16, [x4, #0xa0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x8]
    ldr q16, [x4, #0xb0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr q16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0]
    ldr q16, [x4, #0xd0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x8]
    ldr q16, [x4, #0xe0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr q16, [x4, #0xf0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0]
    ldr q16, [x4, #0x100]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x8]
    ldr q16, [x4, #0x110]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    bge label_43
KAI_ASM_LABEL(label_46)  // Initial: Height 6: no full blocks
    cbz x2, label_49
KAI_ASM_LABEL(label_47)  // Initial: Height 6: Single loop
    movi v17.16b, #0
    cbz x1, label_48
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_48)  // Initial: Height 6: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr s16, [x4, #0x90]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr s16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr s16, [x4, #0xf0]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    bne label_47
KAI_ASM_LABEL(label_49)  // Initial: Height 6: no oddments
    b label_108
KAI_ASM_LABEL(label_50)  // Initial: Height 7
    mov x24, x5
    ldr x2, [x0, #0x8]
    ldr x1, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    add x8, x20, x26
    cmp x2, #0xc
    add x7, x8, x26
    blt label_54
KAI_ASM_LABEL(label_51)  // Initial: Height 7: Block loop
    cbnz x1, label_52
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_53
KAI_ASM_LABEL(label_52)  // Initial: Height 7: Width 3: bias
    ldr d18, [x1, #0]
    ldr d17, [x1, #0x8]
    ldr d16, [x1, #0x10]
    add x1, x1, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_53)  // Initial: Height 7: Width 3: init done
    ldr q16, [x4, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr q16, [x4, #0x90]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0]
    ldr q16, [x4, #0xa0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x8]
    ldr q16, [x4, #0xb0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr q16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0]
    ldr q16, [x4, #0xd0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x8]
    ldr q16, [x4, #0xe0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr q16, [x4, #0xf0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0]
    ldr q16, [x4, #0x100]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x8]
    ldr q16, [x4, #0x110]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    ldr q16, [x4, #0x120]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0]
    ldr q16, [x4, #0x130]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0x8]
    ldr q16, [x4, #0x140]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0x10]
    add x7, x7, #0x18
    bge label_51
KAI_ASM_LABEL(label_54)  // Initial: Height 7: no full blocks
    cbz x2, label_57
KAI_ASM_LABEL(label_55)  // Initial: Height 7: Single loop
    movi v17.16b, #0
    cbz x1, label_56
    ldr h16, [x1, #0]
    add x1, x1, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_56)  // Initial: Height 7: Scalar: no bias
    ldr s16, [x4, #0]
    subs x2, x2, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr s16, [x4, #0x90]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr s16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr s16, [x4, #0xf0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    ldr s16, [x4, #0x120]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x7, #0]
    add x7, x7, #0x2
    bne label_55
KAI_ASM_LABEL(label_57)  // Initial: Height 7: no oddments
    b label_108
KAI_ASM_LABEL(label_58)  // Initial: Height 8
    mov x24, x5
    ldr x3, [x0, #0x8]
    ldr x2, [x0, #0x40]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    add x8, x20, x26
    add x7, x8, x26
    cmp x3, #0xc
    add x6, x7, x26
    blt label_62
KAI_ASM_LABEL(label_59)  // Initial: Height 8: Block loop
    cbnz x2, label_60
    movi v19.16b, #0
    movi v18.16b, #0
    movi v17.16b, #0
    b label_61
KAI_ASM_LABEL(label_60)  // Initial: Height 8: Width 3: bias
    ldr d18, [x2, #0]
    ldr d17, [x2, #0x8]
    ldr d16, [x2, #0x10]
    add x2, x2, #0x18
    shll v19.4s, v18.4h, #0x10
    shll v18.4s, v17.4h, #0x10
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_61)  // Initial: Height 8: Width 3: init done
    ldr q16, [x4, #0]
    sub x3, x3, #0xc
    cmp x3, #0xc
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0]
    ldr q16, [x4, #0x10]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x8]
    ldr q16, [x4, #0x20]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr q16, [x4, #0x30]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0]
    ldr q16, [x4, #0x40]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x8]
    ldr q16, [x4, #0x50]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr q16, [x4, #0x60]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0]
    ldr q16, [x4, #0x70]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x8]
    ldr q16, [x4, #0x80]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr q16, [x4, #0x90]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0]
    ldr q16, [x4, #0xa0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x8]
    ldr q16, [x4, #0xb0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr q16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0]
    ldr q16, [x4, #0xd0]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x8]
    ldr q16, [x4, #0xe0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr q16, [x4, #0xf0]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0]
    ldr q16, [x4, #0x100]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x8]
    ldr q16, [x4, #0x110]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    ldr q16, [x4, #0x120]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0]
    ldr q16, [x4, #0x130]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0x8]
    ldr q16, [x4, #0x140]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x7, #0x10]
    add x7, x7, #0x18
    ldr q16, [x4, #0x150]
    fadd v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x6, #0]
    ldr q16, [x4, #0x160]
    fadd v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x6, #0x8]
    ldr q16, [x4, #0x170]
    add x4, x4, #0x180
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str d16, [x6, #0x10]
    add x6, x6, #0x18
    bge label_59
KAI_ASM_LABEL(label_62)  // Initial: Height 8: no full blocks
    cbz x3, label_65
    mov x1, x4
KAI_ASM_LABEL(label_63)  // Initial: Height 8: Single loop
    movi v17.16b, #0
    cbz x2, label_64
    ldr h16, [x2, #0]
    add x2, x2, #0x2
    shll v17.4s, v16.4h, #0x10
KAI_ASM_LABEL(label_64)  // Initial: Height 8: Scalar: no bias
    ldr s16, [x4, #0]
    subs x3, x3, #0x1
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr s16, [x4, #0x30]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr s16, [x4, #0x60]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr s16, [x4, #0x90]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr s16, [x4, #0xc0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr s16, [x4, #0xf0]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    ldr s16, [x4, #0x120]
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x7, #0]
    add x7, x7, #0x2
    ldr s16, [x4, #0x150]
    add x4, x4, #0x4
    fadd v16.4s, v16.4s, v17.4s
    fmin v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v20.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    str h16, [x6, #0]
    add x6, x6, #0x2
    bne label_63
    add x4, x1, #0x180
KAI_ASM_LABEL(label_65)  // Initial: Height 8: no oddments
    subs x25, x25, #0x8
    add x5, x5, x27
    bgt label_1
    b label_108
KAI_ASM_LABEL(label_66)  // Accumulate
    ldr x5, [x0, #0]
    ldr x4, [x0, #0x50]
    ldr x3, [x0, #0x20]
KAI_ASM_LABEL(label_67)  // Accumulate: Row loop
    cmp x5, #0x7
    bgt label_103
    beq label_98
    cmp x5, #0x5
    bgt label_93
    beq label_88
    cmp x5, #0x3
    bgt label_83
    beq label_78
    cmp x5, #0x1
    bgt label_73
    ldr x1, [x0, #0x8]
    mov x24, x4
    cmp x1, #0xc
    blt label_70
KAI_ASM_LABEL(label_69)  // Accumulate: Height 1: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    bge label_69
KAI_ASM_LABEL(label_70)  // Accumulate: Height 1: no full blocks
    cbz x1, label_72
KAI_ASM_LABEL(label_71)  // Accumulate: Height 1: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    bne label_71
KAI_ASM_LABEL(label_72)  // Accumulate: Height 1: no oddments
    b label_108
KAI_ASM_LABEL(label_73)  // Accumulate: Height 2
    ldr x1, [x0, #0x8]
    mov x24, x4
    cmp x1, #0xc
    add x23, x24, x26
    blt label_75
KAI_ASM_LABEL(label_74)  // Accumulate: Height 2: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    bge label_74
KAI_ASM_LABEL(label_75)  // Accumulate: Height 2: no full blocks
    cbz x1, label_77
KAI_ASM_LABEL(label_76)  // Accumulate: Height 2: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    bne label_76
KAI_ASM_LABEL(label_77)  // Accumulate: Height 2: no oddments
    b label_108
KAI_ASM_LABEL(label_78)  // Accumulate: Height 3
    ldr x1, [x0, #0x8]
    mov x24, x4
    add x23, x24, x26
    add x22, x23, x26
    cmp x1, #0xc
    blt label_80
KAI_ASM_LABEL(label_79)  // Accumulate: Height 3: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    bge label_79
KAI_ASM_LABEL(label_80)  // Accumulate: Height 3: no full blocks
    cbz x1, label_82
KAI_ASM_LABEL(label_81)  // Accumulate: Height 3: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    bne label_81
KAI_ASM_LABEL(label_82)  // Accumulate: Height 3: no oddments
    b label_108
KAI_ASM_LABEL(label_83)  // Accumulate: Height 4
    mov x24, x4
    ldr x1, [x0, #0x8]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    cmp x1, #0xc
    blt label_85
KAI_ASM_LABEL(label_84)  // Accumulate: Height 4: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr d16, [x21, #0]
    ldr q17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0]
    ldr d16, [x21, #0x8]
    ldr q17, [x3, #0xa0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x8]
    ldr d16, [x21, #0x10]
    ldr q17, [x3, #0xb0]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    bge label_84
KAI_ASM_LABEL(label_85)  // Accumulate: Height 4: no full blocks
    cbz x1, label_87
KAI_ASM_LABEL(label_86)  // Accumulate: Height 4: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr h16, [x21, #0]
    ldr s17, [x3, #0x90]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    bne label_86
KAI_ASM_LABEL(label_87)  // Accumulate: Height 4: no oddments
    b label_108
KAI_ASM_LABEL(label_88)  // Accumulate: Height 5
    mov x24, x4
    ldr x1, [x0, #0x8]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    cmp x1, #0xc
    blt label_90
KAI_ASM_LABEL(label_89)  // Accumulate: Height 5: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr d16, [x21, #0]
    ldr q17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0]
    ldr d16, [x21, #0x8]
    ldr q17, [x3, #0xa0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x8]
    ldr d16, [x21, #0x10]
    ldr q17, [x3, #0xb0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr d16, [x20, #0]
    ldr q17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0]
    ldr d16, [x20, #0x8]
    ldr q17, [x3, #0xd0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x8]
    ldr d16, [x20, #0x10]
    ldr q17, [x3, #0xe0]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    bge label_89
KAI_ASM_LABEL(label_90)  // Accumulate: Height 5: no full blocks
    cbz x1, label_92
KAI_ASM_LABEL(label_91)  // Accumulate: Height 5: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr h16, [x21, #0]
    ldr s17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr h16, [x20, #0]
    ldr s17, [x3, #0xc0]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    bne label_91
KAI_ASM_LABEL(label_92)  // Accumulate: Height 5: no oddments
    b label_108
KAI_ASM_LABEL(label_93)  // Accumulate: Height 6
    mov x24, x4
    ldr x1, [x0, #0x8]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    cmp x1, #0xc
    add x8, x20, x26
    blt label_95
KAI_ASM_LABEL(label_94)  // Accumulate: Height 6: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr d16, [x21, #0]
    ldr q17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0]
    ldr d16, [x21, #0x8]
    ldr q17, [x3, #0xa0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x8]
    ldr d16, [x21, #0x10]
    ldr q17, [x3, #0xb0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr d16, [x20, #0]
    ldr q17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0]
    ldr d16, [x20, #0x8]
    ldr q17, [x3, #0xd0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x8]
    ldr d16, [x20, #0x10]
    ldr q17, [x3, #0xe0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr d16, [x8, #0]
    ldr q17, [x3, #0xf0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0]
    ldr d16, [x8, #0x8]
    ldr q17, [x3, #0x100]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x8]
    ldr d16, [x8, #0x10]
    ldr q17, [x3, #0x110]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    bge label_94
KAI_ASM_LABEL(label_95)  // Accumulate: Height 6: no full blocks
    cbz x1, label_97
KAI_ASM_LABEL(label_96)  // Accumulate: Height 6: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr h16, [x21, #0]
    ldr s17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr h16, [x20, #0]
    ldr s17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr h16, [x8, #0]
    ldr s17, [x3, #0xf0]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    bne label_96
KAI_ASM_LABEL(label_97)  // Accumulate: Height 6: no oddments
    b label_108
KAI_ASM_LABEL(label_98)  // Accumulate: Height 7
    mov x24, x4
    ldr x1, [x0, #0x8]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    add x8, x20, x26
    cmp x1, #0xc
    add x7, x8, x26
    blt label_100
KAI_ASM_LABEL(label_99)  // Accumulate: Height 7: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x1, x1, #0xc
    cmp x1, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr d16, [x21, #0]
    ldr q17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0]
    ldr d16, [x21, #0x8]
    ldr q17, [x3, #0xa0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x8]
    ldr d16, [x21, #0x10]
    ldr q17, [x3, #0xb0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr d16, [x20, #0]
    ldr q17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0]
    ldr d16, [x20, #0x8]
    ldr q17, [x3, #0xd0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x8]
    ldr d16, [x20, #0x10]
    ldr q17, [x3, #0xe0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr d16, [x8, #0]
    ldr q17, [x3, #0xf0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0]
    ldr d16, [x8, #0x8]
    ldr q17, [x3, #0x100]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x8]
    ldr d16, [x8, #0x10]
    ldr q17, [x3, #0x110]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    ldr d16, [x7, #0]
    ldr q17, [x3, #0x120]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0]
    ldr d16, [x7, #0x8]
    ldr q17, [x3, #0x130]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0x8]
    ldr d16, [x7, #0x10]
    ldr q17, [x3, #0x140]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0x10]
    add x7, x7, #0x18
    bge label_99
KAI_ASM_LABEL(label_100)  // Accumulate: Height 7: no full blocks
    cbz x1, label_102
KAI_ASM_LABEL(label_101)  // Accumulate: Height 7: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x1, x1, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr h16, [x21, #0]
    ldr s17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr h16, [x20, #0]
    ldr s17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr h16, [x8, #0]
    ldr s17, [x3, #0xf0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    ldr h16, [x7, #0]
    ldr s17, [x3, #0x120]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x7, #0]
    add x7, x7, #0x2
    bne label_101
KAI_ASM_LABEL(label_102)  // Accumulate: Height 7: no oddments
    b label_108
KAI_ASM_LABEL(label_103)  // Accumulate: Height 8
    mov x24, x4
    ldr x2, [x0, #0x8]
    add x23, x24, x26
    add x22, x23, x26
    add x21, x22, x26
    add x20, x21, x26
    add x8, x20, x26
    add x7, x8, x26
    cmp x2, #0xc
    add x6, x7, x26
    blt label_105
KAI_ASM_LABEL(label_104)  // Accumulate: Height 8: Block loop
    ldr d16, [x24, #0]
    ldr q17, [x3, #0]
    sub x2, x2, #0xc
    cmp x2, #0xc
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0]
    ldr d16, [x24, #0x8]
    ldr q17, [x3, #0x10]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x8]
    ldr d16, [x24, #0x10]
    ldr q17, [x3, #0x20]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x24, #0x10]
    add x24, x24, #0x18
    ldr d16, [x23, #0]
    ldr q17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0]
    ldr d16, [x23, #0x8]
    ldr q17, [x3, #0x40]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x8]
    ldr d16, [x23, #0x10]
    ldr q17, [x3, #0x50]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x23, #0x10]
    add x23, x23, #0x18
    ldr d16, [x22, #0]
    ldr q17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0]
    ldr d16, [x22, #0x8]
    ldr q17, [x3, #0x70]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x8]
    ldr d16, [x22, #0x10]
    ldr q17, [x3, #0x80]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x22, #0x10]
    add x22, x22, #0x18
    ldr d16, [x21, #0]
    ldr q17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0]
    ldr d16, [x21, #0x8]
    ldr q17, [x3, #0xa0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x8]
    ldr d16, [x21, #0x10]
    ldr q17, [x3, #0xb0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x21, #0x10]
    add x21, x21, #0x18
    ldr d16, [x20, #0]
    ldr q17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0]
    ldr d16, [x20, #0x8]
    ldr q17, [x3, #0xd0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x8]
    ldr d16, [x20, #0x10]
    ldr q17, [x3, #0xe0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x20, #0x10]
    add x20, x20, #0x18
    ldr d16, [x8, #0]
    ldr q17, [x3, #0xf0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0]
    ldr d16, [x8, #0x8]
    ldr q17, [x3, #0x100]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x8]
    ldr d16, [x8, #0x10]
    ldr q17, [x3, #0x110]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x8, #0x10]
    add x8, x8, #0x18
    ldr d16, [x7, #0]
    ldr q17, [x3, #0x120]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0]
    ldr d16, [x7, #0x8]
    ldr q17, [x3, #0x130]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0x8]
    ldr d16, [x7, #0x10]
    ldr q17, [x3, #0x140]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x7, #0x10]
    add x7, x7, #0x18
    ldr d16, [x6, #0]
    ldr q17, [x3, #0x150]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x6, #0]
    ldr d16, [x6, #0x8]
    ldr q17, [x3, #0x160]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x6, #0x8]
    ldr d16, [x6, #0x10]
    ldr q17, [x3, #0x170]
    add x3, x3, #0x180
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str d16, [x6, #0x10]
    add x6, x6, #0x18
    bge label_104
KAI_ASM_LABEL(label_105)  // Accumulate: Height 8: no full blocks
    cbz x2, label_107
    mov x1, x3
KAI_ASM_LABEL(label_106)  // Accumulate: Height 8: Single loop
    ldr h16, [x24, #0]
    ldr s17, [x3, #0]
    subs x2, x2, #0x1
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x24, #0]
    add x24, x24, #0x2
    ldr h16, [x23, #0]
    ldr s17, [x3, #0x30]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x23, #0]
    add x23, x23, #0x2
    ldr h16, [x22, #0]
    ldr s17, [x3, #0x60]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x22, #0]
    add x22, x22, #0x2
    ldr h16, [x21, #0]
    ldr s17, [x3, #0x90]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x21, #0]
    add x21, x21, #0x2
    ldr h16, [x20, #0]
    ldr s17, [x3, #0xc0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x20, #0]
    add x20, x20, #0x2
    ldr h16, [x8, #0]
    ldr s17, [x3, #0xf0]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x8, #0]
    add x8, x8, #0x2
    ldr h16, [x7, #0]
    ldr s17, [x3, #0x120]
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x7, #0]
    add x7, x7, #0x2
    ldr h16, [x6, #0]
    ldr s17, [x3, #0x150]
    add x3, x3, #0x4
    shll v16.4s, v16.4h, #0x10
    fadd v17.4s, v17.4s, v16.4s
    fmin v17.4s, v17.4s, v21.4s
    fmax v17.4s, v17.4s, v20.4s
    KAI_ASM_INST(0x0ea16a30)  // bfcvtn v16.4h, v17.4s
    str h16, [x6, #0]
    add x6, x6, #0x2
    bne label_106
    add x3, x1, #0x180
KAI_ASM_LABEL(label_107)  // Accumulate: Height 8: no oddments
    subs x5, x5, #0x8
    add x4, x4, x27
    bgt label_67
KAI_ASM_LABEL(label_108)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_ops_a64_merge_fp32_bf16_8x12)

    KAI_ASM_END
