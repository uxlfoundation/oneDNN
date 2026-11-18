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

    KAI_ASM_CODE(sve_ffhybrid_fp16fp32fp16_mla_6x4VL)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sve_ffhybrid_fp16fp32fp16_mla_6x4VL)

KAI_ASM_FUNCTION_TYPE(kai_ops_sve_ffhybrid_fp16fp32fp16_mla_6x4VL)
KAI_ASM_FUNCTION_LABEL(kai_ops_sve_ffhybrid_fp16fp32fp16_mla_6x4VL)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x6, #0xfc00
    mov x5, #0x7c00
    ptrue p5.b
    mov x9, #0
    ldr x4, [x0, #0x60]
    ldrh w1, [x0, #0xa0]
    ldrh w2, [x0, #0xa4]
    ldr x3, [x0, #0x18]
    ldr x28, [x0, #0]
    cmp x4, #0
    cset x27, NE
    cmp x1, x6
    cset x1, NE
    cmp x2, x5
    cset x2, NE
    cmp x3, #0
    orr x27, x27, x1, LSL #1
    cset x1, NE
    orr x27, x27, x2, LSL #1
    orr x27, x27, x1, LSL #3
KAI_ASM_LABEL(label_1)  // Row loop
    sub x1, x28, x9
    cmp x1, #0x6
    bge label_76
    cmp x1, #0x4
    bgt label_61
    beq label_46
    cmp x1, #0x2
    bgt label_31
    beq label_16
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_3
    mov x20, x21
KAI_ASM_LABEL(label_3)  // Height 1: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_4
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    b label_6
KAI_ASM_LABEL(label_4)  // Height 1: no bias
    tbz x27, #0, label_5
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    b label_6
KAI_ASM_LABEL(label_5)  // Height 1: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
KAI_ASM_LABEL(label_6)  // Height 1: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_7
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    b label_9
KAI_ASM_LABEL(label_7)  // Height 1: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_8)  // Height 1: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    cbnz x26, label_9
    add x8, x8, x3, LSL #1
KAI_ASM_LABEL(label_9)  // Height 1: input setup done
    cmp x7, #0x8
    ble label_11
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    bgt label_10
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    ble label_12
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: multiply skip
    tbz x27, #3, label_13
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_8
KAI_ASM_LABEL(label_13)  // Height 1: no string loop
    tbz x27, #1, label_14
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b451)  // ld1rh { z17.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b430)  // ld1rh { z16.h }, p5/Z, [x1]
    fcvt z17.s, p5/m, z17.h
    fcvt z16.s, p5/m, z16.h
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
KAI_ASM_LABEL(label_14)  // Height 1: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    st1h { z8.s }, p4, [x25]
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_2
    b label_91
KAI_ASM_LABEL(label_16)  // Height 2
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_17)  // Height 2: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_18
    mov x20, x21
KAI_ASM_LABEL(label_18)  // Height 2: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_19
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_21
KAI_ASM_LABEL(label_19)  // Height 2: no bias
    tbz x27, #0, label_20
    ldr x1, [x0, #0x58]
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    add x1, x25, x1
    ld1h { z12.s }, p4/Z, [x1]
    ld1h { z13.s }, p3/Z, [x1, #1, MUL VL]
    ld1h { z14.s }, p2/Z, [x1, #2, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z15.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    b label_21
KAI_ASM_LABEL(label_20)  // Height 2: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_21)  // Height 2: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_22
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    b label_24
KAI_ASM_LABEL(label_22)  // Height 2: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_23)  // Height 2: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    cbnz x26, label_24
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
KAI_ASM_LABEL(label_24)  // Height 2: input setup done
    cmp x7, #0x8
    ble label_26
KAI_ASM_LABEL(label_25)  // Height 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z1.h }, p0/Z, [x6]
    add x6, x6, #0x10
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    bgt label_25
KAI_ASM_LABEL(label_26)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    ld1rqh { z1.h }, p0/Z, [x6]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    ble label_27
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
KAI_ASM_LABEL(label_27)  // Height 2: Multiply loop: multiply skip
    tbz x27, #3, label_28
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_23
KAI_ASM_LABEL(label_28)  // Height 2: no string loop
    ldr x1, [x0, #0x58]
    add x7, x25, x1
    tbz x27, #1, label_29
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b451)  // ld1rh { z17.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b430)  // ld1rh { z16.h }, p5/Z, [x1]
    fcvt z17.s, p5/m, z17.h
    fcvt z16.s, p5/m, z16.h
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmin z12.s, p5/M, z12.s, z17.s
    fmin z13.s, p5/M, z13.s, z17.s
    fmin z14.s, p5/M, z14.s, z17.s
    fmin z15.s, p5/M, z15.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
    fmax z12.s, p5/M, z12.s, z16.s
    fmax z13.s, p5/M, z13.s, z16.s
    fmax z14.s, p5/M, z14.s, z16.s
    fmax z15.s, p5/M, z15.s, z16.s
KAI_ASM_LABEL(label_29)  // Height 2: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    st1h { z8.s }, p4, [x25]
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    st1h { z12.s }, p4, [x7]
    st1h { z13.s }, p3, [x7, #1, MUL VL]
    st1h { z14.s }, p2, [x7, #2, MUL VL]
    st1h { z15.s }, p1, [x7, #3, MUL VL]
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_17
    b label_91
KAI_ASM_LABEL(label_31)  // Height 3
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_32)  // Height 3: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_33
    mov x20, x21
KAI_ASM_LABEL(label_33)  // Height 3: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_34
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z17.d, z9.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    b label_36
KAI_ASM_LABEL(label_34)  // Height 3: no bias
    tbz x27, #0, label_35
    ldr x1, [x0, #0x58]
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    add x2, x25, x1
    add x1, x2, x1
    ld1h { z12.s }, p4/Z, [x2]
    ld1h { z13.s }, p3/Z, [x2, #1, MUL VL]
    ld1h { z14.s }, p2/Z, [x2, #2, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z15.s }, p1/Z, [x2, #3, MUL VL]
    ld1h { z16.s }, p4/Z, [x1]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    ld1h { z17.s }, p3/Z, [x1, #1, MUL VL]
    ld1h { z18.s }, p2/Z, [x1, #2, MUL VL]
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    ld1h { z19.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    b label_36
KAI_ASM_LABEL(label_35)  // Height 3: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
KAI_ASM_LABEL(label_36)  // Height 3: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_37
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    b label_39
KAI_ASM_LABEL(label_37)  // Height 3: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_38)  // Height 3: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    cbnz x26, label_39
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
KAI_ASM_LABEL(label_39)  // Height 3: input setup done
    cmp x7, #0x8
    ble label_41
KAI_ASM_LABEL(label_40)  // Height 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z1.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z2.h }, p0/Z, [x5]
    add x5, x5, #0x10
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    bgt label_40
KAI_ASM_LABEL(label_41)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    ld1rqh { z1.h }, p0/Z, [x6]
    ld1rqh { z2.h }, p0/Z, [x5]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    ble label_42
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
KAI_ASM_LABEL(label_42)  // Height 3: Multiply loop: multiply skip
    tbz x27, #3, label_43
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_38
KAI_ASM_LABEL(label_43)  // Height 3: no string loop
    ldr x1, [x0, #0x58]
    add x7, x25, x1
    add x6, x7, x1
    tbz x27, #1, label_44
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b455)  // ld1rh { z21.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b434)  // ld1rh { z20.h }, p5/Z, [x1]
    fcvt z21.s, p5/m, z21.h
    fcvt z20.s, p5/m, z20.h
    fmin z8.s, p5/M, z8.s, z21.s
    fmin z9.s, p5/M, z9.s, z21.s
    fmin z10.s, p5/M, z10.s, z21.s
    fmin z11.s, p5/M, z11.s, z21.s
    fmin z12.s, p5/M, z12.s, z21.s
    fmin z13.s, p5/M, z13.s, z21.s
    fmin z14.s, p5/M, z14.s, z21.s
    fmin z15.s, p5/M, z15.s, z21.s
    fmin z16.s, p5/M, z16.s, z21.s
    fmin z17.s, p5/M, z17.s, z21.s
    fmin z18.s, p5/M, z18.s, z21.s
    fmin z19.s, p5/M, z19.s, z21.s
    fmax z8.s, p5/M, z8.s, z20.s
    fmax z9.s, p5/M, z9.s, z20.s
    fmax z10.s, p5/M, z10.s, z20.s
    fmax z11.s, p5/M, z11.s, z20.s
    fmax z12.s, p5/M, z12.s, z20.s
    fmax z13.s, p5/M, z13.s, z20.s
    fmax z14.s, p5/M, z14.s, z20.s
    fmax z15.s, p5/M, z15.s, z20.s
    fmax z16.s, p5/M, z16.s, z20.s
    fmax z17.s, p5/M, z17.s, z20.s
    fmax z18.s, p5/M, z18.s, z20.s
    fmax z19.s, p5/M, z19.s, z20.s
KAI_ASM_LABEL(label_44)  // Height 3: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    st1h { z8.s }, p4, [x25]
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    st1h { z12.s }, p4, [x7]
    st1h { z13.s }, p3, [x7, #1, MUL VL]
    st1h { z14.s }, p2, [x7, #2, MUL VL]
    st1h { z15.s }, p1, [x7, #3, MUL VL]
    st1h { z16.s }, p4, [x6]
    st1h { z17.s }, p3, [x6, #1, MUL VL]
    st1h { z18.s }, p2, [x6, #2, MUL VL]
    st1h { z19.s }, p1, [x6, #3, MUL VL]
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_32
    b label_91
KAI_ASM_LABEL(label_46)  // Height 4
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_47)  // Height 4: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_48
    mov x20, x21
KAI_ASM_LABEL(label_48)  // Height 4: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_49
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    b label_51
KAI_ASM_LABEL(label_49)  // Height 4: no bias
    tbz x27, #0, label_50
    ldr x1, [x0, #0x58]
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    add x3, x25, x1
    add x2, x3, x1
    add x1, x2, x1
    ld1h { z12.s }, p4/Z, [x3]
    ld1h { z13.s }, p3/Z, [x3, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z14.s }, p2/Z, [x3, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x3, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    ld1h { z16.s }, p4/Z, [x2]
    ld1h { z17.s }, p3/Z, [x2, #1, MUL VL]
    fcvt z11.s, p5/m, z11.h
    ld1h { z18.s }, p2/Z, [x2, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x2, #3, MUL VL]
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    ld1h { z20.s }, p4/Z, [x1]
    ld1h { z21.s }, p3/Z, [x1, #1, MUL VL]
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    ld1h { z22.s }, p2/Z, [x1, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    fcvt z20.s, p5/m, z20.h
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    fcvt z23.s, p5/m, z23.h
    b label_51
KAI_ASM_LABEL(label_50)  // Height 4: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
KAI_ASM_LABEL(label_51)  // Height 4: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_52
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    b label_54
KAI_ASM_LABEL(label_52)  // Height 4: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_53)  // Height 4: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    cbnz x26, label_54
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
KAI_ASM_LABEL(label_54)  // Height 4: input setup done
    cmp x7, #0x8
    ble label_56
KAI_ASM_LABEL(label_55)  // Height 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z1.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
    bgt label_55
KAI_ASM_LABEL(label_56)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    ld1rqh { z1.h }, p0/Z, [x6]
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    ble label_57
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
KAI_ASM_LABEL(label_57)  // Height 4: Multiply loop: multiply skip
    tbz x27, #3, label_58
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_53
KAI_ASM_LABEL(label_58)  // Height 4: no string loop
    ldr x1, [x0, #0x58]
    add x7, x25, x1
    add x6, x7, x1
    add x5, x6, x1
    tbz x27, #1, label_59
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b459)  // ld1rh { z25.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b438)  // ld1rh { z24.h }, p5/Z, [x1]
    fcvt z25.s, p5/m, z25.h
    fcvt z24.s, p5/m, z24.h
    fmin z8.s, p5/M, z8.s, z25.s
    fmin z9.s, p5/M, z9.s, z25.s
    fmin z10.s, p5/M, z10.s, z25.s
    fmin z11.s, p5/M, z11.s, z25.s
    fmin z12.s, p5/M, z12.s, z25.s
    fmin z13.s, p5/M, z13.s, z25.s
    fmin z14.s, p5/M, z14.s, z25.s
    fmin z15.s, p5/M, z15.s, z25.s
    fmin z16.s, p5/M, z16.s, z25.s
    fmin z17.s, p5/M, z17.s, z25.s
    fmin z18.s, p5/M, z18.s, z25.s
    fmin z19.s, p5/M, z19.s, z25.s
    fmin z20.s, p5/M, z20.s, z25.s
    fmin z21.s, p5/M, z21.s, z25.s
    fmin z22.s, p5/M, z22.s, z25.s
    fmin z23.s, p5/M, z23.s, z25.s
    fmax z8.s, p5/M, z8.s, z24.s
    fmax z9.s, p5/M, z9.s, z24.s
    fmax z10.s, p5/M, z10.s, z24.s
    fmax z11.s, p5/M, z11.s, z24.s
    fmax z12.s, p5/M, z12.s, z24.s
    fmax z13.s, p5/M, z13.s, z24.s
    fmax z14.s, p5/M, z14.s, z24.s
    fmax z15.s, p5/M, z15.s, z24.s
    fmax z16.s, p5/M, z16.s, z24.s
    fmax z17.s, p5/M, z17.s, z24.s
    fmax z18.s, p5/M, z18.s, z24.s
    fmax z19.s, p5/M, z19.s, z24.s
    fmax z20.s, p5/M, z20.s, z24.s
    fmax z21.s, p5/M, z21.s, z24.s
    fmax z22.s, p5/M, z22.s, z24.s
    fmax z23.s, p5/M, z23.s, z24.s
KAI_ASM_LABEL(label_59)  // Height 4: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    st1h { z8.s }, p4, [x25]
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    st1h { z12.s }, p4, [x7]
    st1h { z13.s }, p3, [x7, #1, MUL VL]
    st1h { z14.s }, p2, [x7, #2, MUL VL]
    st1h { z15.s }, p1, [x7, #3, MUL VL]
    st1h { z16.s }, p4, [x6]
    st1h { z17.s }, p3, [x6, #1, MUL VL]
    st1h { z18.s }, p2, [x6, #2, MUL VL]
    st1h { z19.s }, p1, [x6, #3, MUL VL]
    st1h { z20.s }, p4, [x5]
    st1h { z21.s }, p3, [x5, #1, MUL VL]
    st1h { z22.s }, p2, [x5, #2, MUL VL]
    st1h { z23.s }, p1, [x5, #3, MUL VL]
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_47
    b label_91
KAI_ASM_LABEL(label_61)  // Height 5
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_62)  // Height 5: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_63
    mov x20, x21
KAI_ASM_LABEL(label_63)  // Height 5: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_64
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    b label_66
KAI_ASM_LABEL(label_64)  // Height 5: no bias
    tbz x27, #0, label_65
    ldr x1, [x0, #0x58]
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    add x4, x25, x1
    add x3, x4, x1
    add x2, x3, x1
    ld1h { z12.s }, p4/Z, [x4]
    ld1h { z13.s }, p3/Z, [x4, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    add x1, x2, x1
    ld1h { z14.s }, p2/Z, [x4, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x4, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    ld1h { z16.s }, p4/Z, [x3]
    ld1h { z17.s }, p3/Z, [x3, #1, MUL VL]
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    ld1h { z18.s }, p2/Z, [x3, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x3, #3, MUL VL]
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    ld1h { z20.s }, p4/Z, [x2]
    ld1h { z21.s }, p3/Z, [x2, #1, MUL VL]
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    ld1h { z22.s }, p2/Z, [x2, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x2, #3, MUL VL]
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    ld1h { z24.s }, p4/Z, [x1]
    ld1h { z25.s }, p3/Z, [x1, #1, MUL VL]
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    ld1h { z26.s }, p2/Z, [x1, #2, MUL VL]
    ld1h { z27.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z20.s, p5/m, z20.h
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    fcvt z23.s, p5/m, z23.h
    fcvt z24.s, p5/m, z24.h
    fcvt z25.s, p5/m, z25.h
    fcvt z26.s, p5/m, z26.h
    fcvt z27.s, p5/m, z27.h
    b label_66
KAI_ASM_LABEL(label_65)  // Height 5: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
KAI_ASM_LABEL(label_66)  // Height 5: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_67
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    b label_69
KAI_ASM_LABEL(label_67)  // Height 5: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_68)  // Height 5: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    cbnz x26, label_69
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
KAI_ASM_LABEL(label_69)  // Height 5: input setup done
    cmp x7, #0x8
    ble label_71
KAI_ASM_LABEL(label_70)  // Height 5: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z1.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    ld1rqh { z4.h }, p0/Z, [x2]
    add x5, x5, #0x10
    add x4, x4, #0x10
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    add x2, x2, #0x10
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440d8)  // fmlalb z24.s, z6.h, z4.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440f9)  // fmlalb z25.s, z7.h, z4.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440da)  // fmlalb z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440fb)  // fmlalb z27.s, z7.h, z4.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448d8)  // fmlalb z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448f9)  // fmlalb z25.s, z7.h, z4.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448da)  // fmlalb z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448fb)  // fmlalb z27.s, z7.h, z4.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40d8)  // fmlalb z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40f9)  // fmlalb z25.s, z7.h, z4.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40da)  // fmlalb z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40fb)  // fmlalb z27.s, z7.h, z4.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48d8)  // fmlalb z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48f9)  // fmlalb z25.s, z7.h, z4.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48da)  // fmlalb z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48fb)  // fmlalb z27.s, z7.h, z4.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440d8)  // fmlalb z24.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440f9)  // fmlalb z25.s, z7.h, z4.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440da)  // fmlalb z26.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440fb)  // fmlalb z27.s, z7.h, z4.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448d8)  // fmlalb z24.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448f9)  // fmlalb z25.s, z7.h, z4.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448da)  // fmlalb z26.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448fb)  // fmlalb z27.s, z7.h, z4.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40d8)  // fmlalb z24.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40f9)  // fmlalb z25.s, z7.h, z4.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40da)  // fmlalb z26.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40fb)  // fmlalb z27.s, z7.h, z4.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48d8)  // fmlalb z24.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48f9)  // fmlalb z25.s, z7.h, z4.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48da)  // fmlalb z26.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48fb)  // fmlalb z27.s, z7.h, z4.h[7]
    bgt label_70
KAI_ASM_LABEL(label_71)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    ld1rqh { z1.h }, p0/Z, [x6]
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    ld1rqh { z4.h }, p0/Z, [x2]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440d8)  // fmlalb z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440f9)  // fmlalb z25.s, z7.h, z4.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440da)  // fmlalb z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440fb)  // fmlalb z27.s, z7.h, z4.h[0]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448d8)  // fmlalb z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448f9)  // fmlalb z25.s, z7.h, z4.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448da)  // fmlalb z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448fb)  // fmlalb z27.s, z7.h, z4.h[1]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40d8)  // fmlalb z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40f9)  // fmlalb z25.s, z7.h, z4.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40da)  // fmlalb z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40fb)  // fmlalb z27.s, z7.h, z4.h[2]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48d8)  // fmlalb z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48f9)  // fmlalb z25.s, z7.h, z4.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48da)  // fmlalb z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48fb)  // fmlalb z27.s, z7.h, z4.h[3]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440d8)  // fmlalb z24.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440f9)  // fmlalb z25.s, z7.h, z4.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440da)  // fmlalb z26.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440fb)  // fmlalb z27.s, z7.h, z4.h[4]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448d8)  // fmlalb z24.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448f9)  // fmlalb z25.s, z7.h, z4.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448da)  // fmlalb z26.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448fb)  // fmlalb z27.s, z7.h, z4.h[5]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40d8)  // fmlalb z24.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40f9)  // fmlalb z25.s, z7.h, z4.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40da)  // fmlalb z26.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40fb)  // fmlalb z27.s, z7.h, z4.h[6]
    ble label_72
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48d8)  // fmlalb z24.s, z6.h, z4.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48f9)  // fmlalb z25.s, z7.h, z4.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48da)  // fmlalb z26.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48fb)  // fmlalb z27.s, z7.h, z4.h[7]
KAI_ASM_LABEL(label_72)  // Height 5: Multiply loop: multiply skip
    tbz x27, #3, label_73
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_68
KAI_ASM_LABEL(label_73)  // Height 5: no string loop
    ldr x1, [x0, #0x58]
    add x7, x25, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    tbz x27, #1, label_74
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b45d)  // ld1rh { z29.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b43c)  // ld1rh { z28.h }, p5/Z, [x1]
    fcvt z29.s, p5/m, z29.h
    fcvt z28.s, p5/m, z28.h
    fmin z8.s, p5/M, z8.s, z29.s
    fmin z9.s, p5/M, z9.s, z29.s
    fmin z10.s, p5/M, z10.s, z29.s
    fmin z11.s, p5/M, z11.s, z29.s
    fmin z12.s, p5/M, z12.s, z29.s
    fmin z13.s, p5/M, z13.s, z29.s
    fmin z14.s, p5/M, z14.s, z29.s
    fmin z15.s, p5/M, z15.s, z29.s
    fmin z16.s, p5/M, z16.s, z29.s
    fmin z17.s, p5/M, z17.s, z29.s
    fmin z18.s, p5/M, z18.s, z29.s
    fmin z19.s, p5/M, z19.s, z29.s
    fmin z20.s, p5/M, z20.s, z29.s
    fmin z21.s, p5/M, z21.s, z29.s
    fmin z22.s, p5/M, z22.s, z29.s
    fmin z23.s, p5/M, z23.s, z29.s
    fmin z24.s, p5/M, z24.s, z29.s
    fmin z25.s, p5/M, z25.s, z29.s
    fmin z26.s, p5/M, z26.s, z29.s
    fmin z27.s, p5/M, z27.s, z29.s
    fmax z8.s, p5/M, z8.s, z28.s
    fmax z9.s, p5/M, z9.s, z28.s
    fmax z10.s, p5/M, z10.s, z28.s
    fmax z11.s, p5/M, z11.s, z28.s
    fmax z12.s, p5/M, z12.s, z28.s
    fmax z13.s, p5/M, z13.s, z28.s
    fmax z14.s, p5/M, z14.s, z28.s
    fmax z15.s, p5/M, z15.s, z28.s
    fmax z16.s, p5/M, z16.s, z28.s
    fmax z17.s, p5/M, z17.s, z28.s
    fmax z18.s, p5/M, z18.s, z28.s
    fmax z19.s, p5/M, z19.s, z28.s
    fmax z20.s, p5/M, z20.s, z28.s
    fmax z21.s, p5/M, z21.s, z28.s
    fmax z22.s, p5/M, z22.s, z28.s
    fmax z23.s, p5/M, z23.s, z28.s
    fmax z24.s, p5/M, z24.s, z28.s
    fmax z25.s, p5/M, z25.s, z28.s
    fmax z26.s, p5/M, z26.s, z28.s
    fmax z27.s, p5/M, z27.s, z28.s
KAI_ASM_LABEL(label_74)  // Height 5: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    st1h { z8.s }, p4, [x25]
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    st1h { z12.s }, p4, [x7]
    fcvt z24.h, p5/m, z24.s
    fcvt z25.h, p5/m, z25.s
    st1h { z13.s }, p3, [x7, #1, MUL VL]
    fcvt z26.h, p5/m, z26.s
    fcvt z27.h, p5/m, z27.s
    st1h { z14.s }, p2, [x7, #2, MUL VL]
    st1h { z15.s }, p1, [x7, #3, MUL VL]
    st1h { z16.s }, p4, [x6]
    st1h { z17.s }, p3, [x6, #1, MUL VL]
    st1h { z18.s }, p2, [x6, #2, MUL VL]
    st1h { z19.s }, p1, [x6, #3, MUL VL]
    st1h { z20.s }, p4, [x5]
    st1h { z21.s }, p3, [x5, #1, MUL VL]
    st1h { z22.s }, p2, [x5, #2, MUL VL]
    st1h { z23.s }, p1, [x5, #3, MUL VL]
    st1h { z24.s }, p4, [x4]
    st1h { z25.s }, p3, [x4, #1, MUL VL]
    st1h { z26.s }, p2, [x4, #2, MUL VL]
    st1h { z27.s }, p1, [x4, #3, MUL VL]
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_62
    b label_91
KAI_ASM_LABEL(label_76)  // Height 6
    ldr x1, [x0, #0x58]
    ldr x25, [x0, #0x50]
    ldr x24, [x0, #0x8]
    ldr x23, [x0, #0x30]
    ldr x22, [x0, #0x40]
    madd x25, x9, x1, x25
KAI_ASM_LABEL(label_77)  // Height 6: Column loop
    mov x21, x23
    ldr x2, [x0, #0x38]
    cnth x1
    add x20, x21, x2
    cmp x24, x1
    add x23, x20, x2
    bgt label_78
    mov x20, x21
KAI_ASM_LABEL(label_78)  // Height 6: B setup done
    mov x1, #0
    whilelt p4.s, x1, x24
    incw x1
    whilelt p3.s, x1, x24
    incw x1
    whilelt p2.s, x1, x24
    incw x1
    whilelt p1.s, x1, x24
    cbz x22, label_79
    ld1h { z8.s }, p4/Z, [x22]
    ld1h { z9.s }, p3/Z, [x22, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    b label_81
KAI_ASM_LABEL(label_79)  // Height 6: no bias
    tbz x27, #0, label_80
    ldr x5, [x0, #0x58]
    ld1h { z8.s }, p4/Z, [x25]
    ld1h { z9.s }, p3/Z, [x25, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x25, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x25, #3, MUL VL]
    add x1, x25, x5
    add x4, x1, x5
    add x3, x4, x5
    ld1h { z12.s }, p4/Z, [x1]
    ld1h { z13.s }, p3/Z, [x1, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    add x2, x3, x5
    ld1h { z14.s }, p2/Z, [x1, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    add x1, x2, x5
    ld1h { z16.s }, p4/Z, [x4]
    ld1h { z17.s }, p3/Z, [x4, #1, MUL VL]
    fcvt z10.s, p5/m, z10.h
    ld1h { z18.s }, p2/Z, [x4, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x4, #3, MUL VL]
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    ld1h { z20.s }, p4/Z, [x3]
    ld1h { z21.s }, p3/Z, [x3, #1, MUL VL]
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    ld1h { z22.s }, p2/Z, [x3, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x3, #3, MUL VL]
    fcvt z15.s, p5/m, z15.h
    fcvt z16.s, p5/m, z16.h
    ld1h { z24.s }, p4/Z, [x2]
    ld1h { z25.s }, p3/Z, [x2, #1, MUL VL]
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    ld1h { z26.s }, p2/Z, [x2, #2, MUL VL]
    ld1h { z27.s }, p1/Z, [x2, #3, MUL VL]
    fcvt z19.s, p5/m, z19.h
    fcvt z20.s, p5/m, z20.h
    ld1h { z28.s }, p4/Z, [x1]
    ld1h { z29.s }, p3/Z, [x1, #1, MUL VL]
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    ld1h { z30.s }, p2/Z, [x1, #2, MUL VL]
    ld1h { z31.s }, p1/Z, [x1, #3, MUL VL]
    fcvt z23.s, p5/m, z23.h
    fcvt z24.s, p5/m, z24.h
    fcvt z25.s, p5/m, z25.h
    fcvt z26.s, p5/m, z26.h
    fcvt z27.s, p5/m, z27.h
    fcvt z28.s, p5/m, z28.h
    fcvt z29.s, p5/m, z29.h
    fcvt z30.s, p5/m, z30.h
    fcvt z31.s, p5/m, z31.h
    b label_81
KAI_ASM_LABEL(label_80)  // Height 6: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_81)  // Height 6: setup done
    ldr x3, [x0, #0x28]
    tbnz x27, #3, label_82
    ldr x8, [x0, #0x20]
    ldr x7, [x0, #0x10]
    madd x8, x9, x3, x8
    add x6, x8, x3
    add x5, x6, x3
    add x4, x5, x3
    add x2, x4, x3
    add x1, x2, x3
    b label_84
KAI_ASM_LABEL(label_82)  // Height 6: setup indirect input
    mov x26, #0
KAI_ASM_LABEL(label_83)  // Height 6: String loop
    ldr x1, [x0, #0x20]
    ldr x2, [x0, #0x18]
    ldr x1, [x1, x26, LSL #0x3]
    ldr x7, [x2, x26, LSL #0x3]
    add x1, x1, x9, LSL #3
    ldr x8, [x1, #0]
    ldr x6, [x1, #0x8]
    ldr x5, [x1, #0x10]
    ldr x4, [x1, #0x18]
    ldr x2, [x1, #0x20]
    ldr x1, [x1, #0x28]
    cbnz x26, label_84
    add x8, x8, x3, LSL #1
    add x6, x6, x3, LSL #1
    add x5, x5, x3, LSL #1
    add x4, x4, x3, LSL #1
    add x2, x2, x3, LSL #1
    add x1, x1, x3, LSL #1
KAI_ASM_LABEL(label_84)  // Height 6: input setup done
    cmp x7, #0x8
    ble label_86
KAI_ASM_LABEL(label_85)  // Height 6: Multiply loop: Main loop head
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    sub x7, x7, #0x8
    cmp x7, #0x8
    ld1rqh { z0.h }, p0/Z, [x8]
    add x8, x8, #0x10
    ld1rqh { z1.h }, p0/Z, [x6]
    add x6, x6, #0x10
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    ld1rqh { z4.h }, p0/Z, [x2]
    add x5, x5, #0x10
    add x4, x4, #0x10
    ld1rqh { z5.h }, p0/Z, [x1]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    add x2, x2, #0x10
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    add x1, x1, #0x10
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440d8)  // fmlalb z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a540dc)  // fmlalb z28.s, z6.h, z5.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440f9)  // fmlalb z25.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x64a540fd)  // fmlalb z29.s, z7.h, z5.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440da)  // fmlalb z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a540de)  // fmlalb z30.s, z6.h, z5.h[0]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440fb)  // fmlalb z27.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x64a540ff)  // fmlalb z31.s, z7.h, z5.h[0]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448d8)  // fmlalb z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a548dc)  // fmlalb z28.s, z6.h, z5.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448f9)  // fmlalb z25.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x64a548fd)  // fmlalb z29.s, z7.h, z5.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448da)  // fmlalb z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a548de)  // fmlalb z30.s, z6.h, z5.h[1]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448fb)  // fmlalb z27.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x64a548ff)  // fmlalb z31.s, z7.h, z5.h[1]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40d8)  // fmlalb z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64ad40dc)  // fmlalb z28.s, z6.h, z5.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40f9)  // fmlalb z25.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x64ad40fd)  // fmlalb z29.s, z7.h, z5.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40da)  // fmlalb z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64ad40de)  // fmlalb z30.s, z6.h, z5.h[2]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40fb)  // fmlalb z27.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x64ad40ff)  // fmlalb z31.s, z7.h, z5.h[2]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48d8)  // fmlalb z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64ad48dc)  // fmlalb z28.s, z6.h, z5.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48f9)  // fmlalb z25.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x64ad48fd)  // fmlalb z29.s, z7.h, z5.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48da)  // fmlalb z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64ad48de)  // fmlalb z30.s, z6.h, z5.h[3]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48fb)  // fmlalb z27.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x64ad48ff)  // fmlalb z31.s, z7.h, z5.h[3]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440d8)  // fmlalb z24.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b540dc)  // fmlalb z28.s, z6.h, z5.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440f9)  // fmlalb z25.s, z7.h, z4.h[4]
    KAI_ASM_INST(0x64b540fd)  // fmlalb z29.s, z7.h, z5.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440da)  // fmlalb z26.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b540de)  // fmlalb z30.s, z6.h, z5.h[4]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440fb)  // fmlalb z27.s, z7.h, z4.h[4]
    KAI_ASM_INST(0x64b540ff)  // fmlalb z31.s, z7.h, z5.h[4]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448d8)  // fmlalb z24.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b548dc)  // fmlalb z28.s, z6.h, z5.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448f9)  // fmlalb z25.s, z7.h, z4.h[5]
    KAI_ASM_INST(0x64b548fd)  // fmlalb z29.s, z7.h, z5.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448da)  // fmlalb z26.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b548de)  // fmlalb z30.s, z6.h, z5.h[5]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448fb)  // fmlalb z27.s, z7.h, z4.h[5]
    KAI_ASM_INST(0x64b548ff)  // fmlalb z31.s, z7.h, z5.h[5]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40d8)  // fmlalb z24.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64bd40dc)  // fmlalb z28.s, z6.h, z5.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40f9)  // fmlalb z25.s, z7.h, z4.h[6]
    KAI_ASM_INST(0x64bd40fd)  // fmlalb z29.s, z7.h, z5.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40da)  // fmlalb z26.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64bd40de)  // fmlalb z30.s, z6.h, z5.h[6]
    ld1h { z6.s }, p5/Z, [x21]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40fb)  // fmlalb z27.s, z7.h, z4.h[6]
    KAI_ASM_INST(0x64bd40ff)  // fmlalb z31.s, z7.h, z5.h[6]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48d8)  // fmlalb z24.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64bd48dc)  // fmlalb z28.s, z6.h, z5.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48f9)  // fmlalb z25.s, z7.h, z4.h[7]
    KAI_ASM_INST(0x64bd48fd)  // fmlalb z29.s, z7.h, z5.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48da)  // fmlalb z26.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64bd48de)  // fmlalb z30.s, z6.h, z5.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48fb)  // fmlalb z27.s, z7.h, z4.h[7]
    KAI_ASM_INST(0x64bd48ff)  // fmlalb z31.s, z7.h, z5.h[7]
    bgt label_85
KAI_ASM_LABEL(label_86)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x7
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    ld1rqh { z0.h }, p0/Z, [x8]
    ld1rqh { z1.h }, p0/Z, [x6]
    ld1rqh { z2.h }, p0/Z, [x5]
    ld1rqh { z3.h }, p0/Z, [x4]
    ld1rqh { z4.h }, p0/Z, [x2]
    ld1rqh { z5.h }, p0/Z, [x1]
    KAI_ASM_INST(0x64a040c8)  // fmlalb z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140cc)  // fmlalb z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a040e9)  // fmlalb z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ed)  // fmlalb z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240d0)  // fmlalb z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d4)  // fmlalb z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440d8)  // fmlalb z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a540dc)  // fmlalb z28.s, z6.h, z5.h[0]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a240f1)  // fmlalb z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f5)  // fmlalb z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440f9)  // fmlalb z25.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x64a540fd)  // fmlalb z29.s, z7.h, z5.h[0]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a040ca)  // fmlalb z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x64a140ce)  // fmlalb z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x64a240d2)  // fmlalb z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x64a340d6)  // fmlalb z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a440da)  // fmlalb z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x64a540de)  // fmlalb z30.s, z6.h, z5.h[0]
    KAI_ASM_INST(0x64a040eb)  // fmlalb z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x64a140ef)  // fmlalb z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x64a240f3)  // fmlalb z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x64a340f7)  // fmlalb z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a440fb)  // fmlalb z27.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x64a540ff)  // fmlalb z31.s, z7.h, z5.h[0]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a048c8)  // fmlalb z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148cc)  // fmlalb z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d0)  // fmlalb z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d4)  // fmlalb z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448d8)  // fmlalb z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a548dc)  // fmlalb z28.s, z6.h, z5.h[1]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a048e9)  // fmlalb z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ed)  // fmlalb z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f1)  // fmlalb z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f5)  // fmlalb z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448f9)  // fmlalb z25.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x64a548fd)  // fmlalb z29.s, z7.h, z5.h[1]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a048ca)  // fmlalb z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x64a148ce)  // fmlalb z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x64a248d2)  // fmlalb z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x64a348d6)  // fmlalb z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a448da)  // fmlalb z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x64a548de)  // fmlalb z30.s, z6.h, z5.h[1]
    KAI_ASM_INST(0x64a048eb)  // fmlalb z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x64a148ef)  // fmlalb z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x64a248f3)  // fmlalb z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x64a348f7)  // fmlalb z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a448fb)  // fmlalb z27.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x64a548ff)  // fmlalb z31.s, z7.h, z5.h[1]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a840c8)  // fmlalb z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940cc)  // fmlalb z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d0)  // fmlalb z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d4)  // fmlalb z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40d8)  // fmlalb z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64ad40dc)  // fmlalb z28.s, z6.h, z5.h[2]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a840e9)  // fmlalb z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ed)  // fmlalb z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f1)  // fmlalb z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f5)  // fmlalb z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40f9)  // fmlalb z25.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x64ad40fd)  // fmlalb z29.s, z7.h, z5.h[2]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a840ca)  // fmlalb z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x64a940ce)  // fmlalb z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x64aa40d2)  // fmlalb z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x64ab40d6)  // fmlalb z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ac40da)  // fmlalb z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x64ad40de)  // fmlalb z30.s, z6.h, z5.h[2]
    KAI_ASM_INST(0x64a840eb)  // fmlalb z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x64a940ef)  // fmlalb z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x64aa40f3)  // fmlalb z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x64ab40f7)  // fmlalb z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ac40fb)  // fmlalb z27.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x64ad40ff)  // fmlalb z31.s, z7.h, z5.h[2]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64a848c8)  // fmlalb z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948cc)  // fmlalb z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d0)  // fmlalb z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d4)  // fmlalb z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48d8)  // fmlalb z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64ad48dc)  // fmlalb z28.s, z6.h, z5.h[3]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64a848e9)  // fmlalb z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ed)  // fmlalb z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f1)  // fmlalb z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f5)  // fmlalb z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48f9)  // fmlalb z25.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x64ad48fd)  // fmlalb z29.s, z7.h, z5.h[3]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64a848ca)  // fmlalb z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x64a948ce)  // fmlalb z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x64aa48d2)  // fmlalb z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x64ab48d6)  // fmlalb z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ac48da)  // fmlalb z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x64ad48de)  // fmlalb z30.s, z6.h, z5.h[3]
    KAI_ASM_INST(0x64a848eb)  // fmlalb z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x64a948ef)  // fmlalb z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x64aa48f3)  // fmlalb z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x64ab48f7)  // fmlalb z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ac48fb)  // fmlalb z27.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x64ad48ff)  // fmlalb z31.s, z7.h, z5.h[3]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b040c8)  // fmlalb z8.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140cc)  // fmlalb z12.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d0)  // fmlalb z16.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d4)  // fmlalb z20.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440d8)  // fmlalb z24.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b540dc)  // fmlalb z28.s, z6.h, z5.h[4]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b040e9)  // fmlalb z9.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ed)  // fmlalb z13.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f1)  // fmlalb z17.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f5)  // fmlalb z21.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440f9)  // fmlalb z25.s, z7.h, z4.h[4]
    KAI_ASM_INST(0x64b540fd)  // fmlalb z29.s, z7.h, z5.h[4]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b040ca)  // fmlalb z10.s, z6.h, z0.h[4]
    KAI_ASM_INST(0x64b140ce)  // fmlalb z14.s, z6.h, z1.h[4]
    KAI_ASM_INST(0x64b240d2)  // fmlalb z18.s, z6.h, z2.h[4]
    KAI_ASM_INST(0x64b340d6)  // fmlalb z22.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b440da)  // fmlalb z26.s, z6.h, z4.h[4]
    KAI_ASM_INST(0x64b540de)  // fmlalb z30.s, z6.h, z5.h[4]
    KAI_ASM_INST(0x64b040eb)  // fmlalb z11.s, z7.h, z0.h[4]
    KAI_ASM_INST(0x64b140ef)  // fmlalb z15.s, z7.h, z1.h[4]
    KAI_ASM_INST(0x64b240f3)  // fmlalb z19.s, z7.h, z2.h[4]
    KAI_ASM_INST(0x64b340f7)  // fmlalb z23.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b440fb)  // fmlalb z27.s, z7.h, z4.h[4]
    KAI_ASM_INST(0x64b540ff)  // fmlalb z31.s, z7.h, z5.h[4]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b048c8)  // fmlalb z8.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148cc)  // fmlalb z12.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d0)  // fmlalb z16.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d4)  // fmlalb z20.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448d8)  // fmlalb z24.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b548dc)  // fmlalb z28.s, z6.h, z5.h[5]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b048e9)  // fmlalb z9.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ed)  // fmlalb z13.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f1)  // fmlalb z17.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f5)  // fmlalb z21.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448f9)  // fmlalb z25.s, z7.h, z4.h[5]
    KAI_ASM_INST(0x64b548fd)  // fmlalb z29.s, z7.h, z5.h[5]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b048ca)  // fmlalb z10.s, z6.h, z0.h[5]
    KAI_ASM_INST(0x64b148ce)  // fmlalb z14.s, z6.h, z1.h[5]
    KAI_ASM_INST(0x64b248d2)  // fmlalb z18.s, z6.h, z2.h[5]
    KAI_ASM_INST(0x64b348d6)  // fmlalb z22.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b448da)  // fmlalb z26.s, z6.h, z4.h[5]
    KAI_ASM_INST(0x64b548de)  // fmlalb z30.s, z6.h, z5.h[5]
    KAI_ASM_INST(0x64b048eb)  // fmlalb z11.s, z7.h, z0.h[5]
    KAI_ASM_INST(0x64b148ef)  // fmlalb z15.s, z7.h, z1.h[5]
    KAI_ASM_INST(0x64b248f3)  // fmlalb z19.s, z7.h, z2.h[5]
    KAI_ASM_INST(0x64b348f7)  // fmlalb z23.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b448fb)  // fmlalb z27.s, z7.h, z4.h[5]
    KAI_ASM_INST(0x64b548ff)  // fmlalb z31.s, z7.h, z5.h[5]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    subs x7, x7, #0x1
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b840c8)  // fmlalb z8.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940cc)  // fmlalb z12.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d0)  // fmlalb z16.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d4)  // fmlalb z20.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40d8)  // fmlalb z24.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64bd40dc)  // fmlalb z28.s, z6.h, z5.h[6]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b840e9)  // fmlalb z9.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ed)  // fmlalb z13.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f1)  // fmlalb z17.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f5)  // fmlalb z21.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40f9)  // fmlalb z25.s, z7.h, z4.h[6]
    KAI_ASM_INST(0x64bd40fd)  // fmlalb z29.s, z7.h, z5.h[6]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b840ca)  // fmlalb z10.s, z6.h, z0.h[6]
    KAI_ASM_INST(0x64b940ce)  // fmlalb z14.s, z6.h, z1.h[6]
    KAI_ASM_INST(0x64ba40d2)  // fmlalb z18.s, z6.h, z2.h[6]
    KAI_ASM_INST(0x64bb40d6)  // fmlalb z22.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bc40da)  // fmlalb z26.s, z6.h, z4.h[6]
    KAI_ASM_INST(0x64bd40de)  // fmlalb z30.s, z6.h, z5.h[6]
    KAI_ASM_INST(0x64b840eb)  // fmlalb z11.s, z7.h, z0.h[6]
    KAI_ASM_INST(0x64b940ef)  // fmlalb z15.s, z7.h, z1.h[6]
    KAI_ASM_INST(0x64ba40f3)  // fmlalb z19.s, z7.h, z2.h[6]
    KAI_ASM_INST(0x64bb40f7)  // fmlalb z23.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bc40fb)  // fmlalb z27.s, z7.h, z4.h[6]
    KAI_ASM_INST(0x64bd40ff)  // fmlalb z31.s, z7.h, z5.h[6]
    ble label_87
    ld1h { z6.s }, p5/Z, [x21]
    ld1h { z7.s }, p5/Z, [x21, #1, MUL VL]
    addvl x21, x21, #1
    KAI_ASM_INST(0x64b848c8)  // fmlalb z8.s, z6.h, z0.h[7]
    KAI_ASM_INST(0x64b948cc)  // fmlalb z12.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d0)  // fmlalb z16.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d4)  // fmlalb z20.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48d8)  // fmlalb z24.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64bd48dc)  // fmlalb z28.s, z6.h, z5.h[7]
    KAI_ASM_INST(0x64b848e9)  // fmlalb z9.s, z7.h, z0.h[7]
    ld1h { z6.s }, p5/Z, [x20]
    KAI_ASM_INST(0x64b948ed)  // fmlalb z13.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f1)  // fmlalb z17.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f5)  // fmlalb z21.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48f9)  // fmlalb z25.s, z7.h, z4.h[7]
    KAI_ASM_INST(0x64bd48fd)  // fmlalb z29.s, z7.h, z5.h[7]
    ld1h { z7.s }, p5/Z, [x20, #1, MUL VL]
    KAI_ASM_INST(0x64b848ca)  // fmlalb z10.s, z6.h, z0.h[7]
    addvl x20, x20, #1
    KAI_ASM_INST(0x64b948ce)  // fmlalb z14.s, z6.h, z1.h[7]
    KAI_ASM_INST(0x64ba48d2)  // fmlalb z18.s, z6.h, z2.h[7]
    KAI_ASM_INST(0x64bb48d6)  // fmlalb z22.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bc48da)  // fmlalb z26.s, z6.h, z4.h[7]
    KAI_ASM_INST(0x64bd48de)  // fmlalb z30.s, z6.h, z5.h[7]
    KAI_ASM_INST(0x64b848eb)  // fmlalb z11.s, z7.h, z0.h[7]
    KAI_ASM_INST(0x64b948ef)  // fmlalb z15.s, z7.h, z1.h[7]
    KAI_ASM_INST(0x64ba48f3)  // fmlalb z19.s, z7.h, z2.h[7]
    KAI_ASM_INST(0x64bb48f7)  // fmlalb z23.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bc48fb)  // fmlalb z27.s, z7.h, z4.h[7]
    KAI_ASM_INST(0x64bd48ff)  // fmlalb z31.s, z7.h, z5.h[7]
KAI_ASM_LABEL(label_87)  // Height 6: Multiply loop: multiply skip
    tbz x27, #3, label_88
    ldr x1, [x0, #0x10]
    add x26, x26, #0x1
    cmp x26, x1
    bne label_83
KAI_ASM_LABEL(label_88)  // Height 6: no string loop
    ldr x1, [x0, #0x58]
    add x7, x25, x1
    add x6, x7, x1
    add x5, x6, x1
    add x4, x5, x1
    add x3, x4, x1
    tbz x27, #1, label_89
    add x2, x0, #0xa4
    add x1, x0, #0xa0
    KAI_ASM_INST(0x84c0b441)  // ld1rh { z1.h }, p5/Z, [x2]
    KAI_ASM_INST(0x84c0b420)  // ld1rh { z0.h }, p5/Z, [x1]
    fcvt z1.s, p5/m, z1.h
    fcvt z0.s, p5/m, z0.h
    fmin z8.s, p5/M, z8.s, z1.s
    fmin z9.s, p5/M, z9.s, z1.s
    fmin z10.s, p5/M, z10.s, z1.s
    fmin z11.s, p5/M, z11.s, z1.s
    fmin z12.s, p5/M, z12.s, z1.s
    fmin z13.s, p5/M, z13.s, z1.s
    fmin z14.s, p5/M, z14.s, z1.s
    fmin z15.s, p5/M, z15.s, z1.s
    fmin z16.s, p5/M, z16.s, z1.s
    fmin z17.s, p5/M, z17.s, z1.s
    fmin z18.s, p5/M, z18.s, z1.s
    fmin z19.s, p5/M, z19.s, z1.s
    fmin z20.s, p5/M, z20.s, z1.s
    fmin z21.s, p5/M, z21.s, z1.s
    fmin z22.s, p5/M, z22.s, z1.s
    fmin z23.s, p5/M, z23.s, z1.s
    fmin z24.s, p5/M, z24.s, z1.s
    fmin z25.s, p5/M, z25.s, z1.s
    fmin z26.s, p5/M, z26.s, z1.s
    fmin z27.s, p5/M, z27.s, z1.s
    fmin z28.s, p5/M, z28.s, z1.s
    fmin z29.s, p5/M, z29.s, z1.s
    fmin z30.s, p5/M, z30.s, z1.s
    fmin z31.s, p5/M, z31.s, z1.s
    fmax z8.s, p5/M, z8.s, z0.s
    fmax z9.s, p5/M, z9.s, z0.s
    fmax z10.s, p5/M, z10.s, z0.s
    fmax z11.s, p5/M, z11.s, z0.s
    fmax z12.s, p5/M, z12.s, z0.s
    fmax z13.s, p5/M, z13.s, z0.s
    fmax z14.s, p5/M, z14.s, z0.s
    fmax z15.s, p5/M, z15.s, z0.s
    fmax z16.s, p5/M, z16.s, z0.s
    fmax z17.s, p5/M, z17.s, z0.s
    fmax z18.s, p5/M, z18.s, z0.s
    fmax z19.s, p5/M, z19.s, z0.s
    fmax z20.s, p5/M, z20.s, z0.s
    fmax z21.s, p5/M, z21.s, z0.s
    fmax z22.s, p5/M, z22.s, z0.s
    fmax z23.s, p5/M, z23.s, z0.s
    fmax z24.s, p5/M, z24.s, z0.s
    fmax z25.s, p5/M, z25.s, z0.s
    fmax z26.s, p5/M, z26.s, z0.s
    fmax z27.s, p5/M, z27.s, z0.s
    fmax z28.s, p5/M, z28.s, z0.s
    fmax z29.s, p5/M, z29.s, z0.s
    fmax z30.s, p5/M, z30.s, z0.s
    fmax z31.s, p5/M, z31.s, z0.s
KAI_ASM_LABEL(label_89)  // Height 6: No activation
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    st1h { z8.s }, p4, [x25]
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    st1h { z9.s }, p3, [x25, #1, MUL VL]
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    st1h { z10.s }, p2, [x25, #2, MUL VL]
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    st1h { z11.s }, p1, [x25, #3, MUL VL]
    addvl x25, x25, #2
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    st1h { z12.s }, p4, [x7]
    fcvt z24.h, p5/m, z24.s
    fcvt z25.h, p5/m, z25.s
    st1h { z13.s }, p3, [x7, #1, MUL VL]
    fcvt z26.h, p5/m, z26.s
    fcvt z27.h, p5/m, z27.s
    st1h { z14.s }, p2, [x7, #2, MUL VL]
    fcvt z28.h, p5/m, z28.s
    fcvt z29.h, p5/m, z29.s
    st1h { z15.s }, p1, [x7, #3, MUL VL]
    fcvt z30.h, p5/m, z30.s
    fcvt z31.h, p5/m, z31.s
    st1h { z16.s }, p4, [x6]
    st1h { z17.s }, p3, [x6, #1, MUL VL]
    st1h { z18.s }, p2, [x6, #2, MUL VL]
    st1h { z19.s }, p1, [x6, #3, MUL VL]
    st1h { z20.s }, p4, [x5]
    st1h { z21.s }, p3, [x5, #1, MUL VL]
    st1h { z22.s }, p2, [x5, #2, MUL VL]
    st1h { z23.s }, p1, [x5, #3, MUL VL]
    st1h { z24.s }, p4, [x4]
    st1h { z25.s }, p3, [x4, #1, MUL VL]
    st1h { z26.s }, p2, [x4, #2, MUL VL]
    st1h { z27.s }, p1, [x4, #3, MUL VL]
    st1h { z28.s }, p4, [x3]
    st1h { z29.s }, p3, [x3, #1, MUL VL]
    st1h { z30.s }, p2, [x3, #2, MUL VL]
    st1h { z31.s }, p1, [x3, #3, MUL VL]
    decw x24, ALL, MUL #4
    cmp x24, XZR
    bgt label_77
    add x9, x9, #0x6
    cmp x9, x28
    blt label_1
KAI_ASM_LABEL(label_91)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_ops_sve_ffhybrid_fp16fp32fp16_mla_6x4VL)

    KAI_ASM_END
