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

    KAI_ASM_CODE(kai_ops_common)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_get_sve_vector_length_u8)

KAI_ASM_FUNCTION_TYPE(kai_ops_get_sve_vector_length_u8)
KAI_ASM_FUNCTION_LABEL(kai_ops_get_sve_vector_length_u8)
    KAI_ASM_INST(0x0420e3e0) // cntb x0
    ret
    KAI_ASM_FUNCTION_END(kai_ops_get_sve_vector_length_u8)

    KAI_ASM_GLOBAL(kai_ops_get_sme_vector_length_u8)

KAI_ASM_FUNCTION_TYPE(kai_ops_get_sme_vector_length_u8)
KAI_ASM_FUNCTION_LABEL(kai_ops_get_sme_vector_length_u8)
    KAI_ASM_INST(0x04bf5820) // rdsvl x0, #1
    ret
    KAI_ASM_FUNCTION_END(kai_ops_get_sme_vector_length_u8)

    KAI_ASM_GLOBAL(kai_ops_fp16_to_float)
KAI_ASM_FUNCTION_TYPE(kai_ops_fp16_to_float)
KAI_ASM_FUNCTION_LABEL(kai_ops_fp16_to_float)
    dup v0.8h, w0
    fcvt s0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_ops_fp16_to_float)

    KAI_ASM_GLOBAL(kai_ops_float_to_fp16)
KAI_ASM_FUNCTION_TYPE(kai_ops_float_to_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_float_to_fp16)
    fcvt h0, s0
    umov w0, v0.h[0]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_float_to_fp16)

    KAI_ASM_END
