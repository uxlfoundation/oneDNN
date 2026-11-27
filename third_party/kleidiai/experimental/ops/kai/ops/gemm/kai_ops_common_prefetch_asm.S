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

// Each function takes a single const void* address in x0 and returns no value.
// These leaf functions leave x0 and all callee-saved registers unchanged and
// return through the link register without allocating a stack frame.
//
// PRFM PLDL1KEEP requests a prefetch for a future load into the L1 data cache
// with a temporal retention policy. It is a hint: completion and cache residency
// are not guaranteed, and it does not provide memory synchronization.
//
// The suffix counts prefetch instructions at 64-byte intervals from the supplied
// address. This preserves the original helpers' spacing; it does not imply that
// the address is aligned or that every processor has 64-byte cache lines.
    KAI_ASM_CODE(kai_ops_common_prefetch)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_prefetch_1x)
    KAI_ASM_GLOBAL(kai_ops_prefetch_2x)
    KAI_ASM_GLOBAL(kai_ops_prefetch_3x)
    KAI_ASM_GLOBAL(kai_ops_prefetch_4x)
    KAI_ASM_GLOBAL(kai_ops_prefetch_5x)
    KAI_ASM_GLOBAL(kai_ops_prefetch_6x)

// One prefetch at byte offset 0.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_1x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_1x)
    prfm pldl1keep, [x0]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_1x)

// Two prefetches at byte offsets 0 and 64.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_2x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_2x)
    prfm pldl1keep, [x0]
    prfm pldl1keep, [x0, #64]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_2x)

// Three prefetches at byte offsets 0, 64, and 128.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_3x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_3x)
    prfm pldl1keep, [x0]
    prfm pldl1keep, [x0, #64]
    prfm pldl1keep, [x0, #128]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_3x)

// Four prefetches at byte offsets 0, 64, 128, and 192.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_4x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_4x)
    prfm pldl1keep, [x0]
    prfm pldl1keep, [x0, #64]
    prfm pldl1keep, [x0, #128]
    prfm pldl1keep, [x0, #192]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_4x)

// Five prefetches at byte offsets 0, 64, 128, 192, and 256.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_5x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_5x)
    prfm pldl1keep, [x0]
    prfm pldl1keep, [x0, #64]
    prfm pldl1keep, [x0, #128]
    prfm pldl1keep, [x0, #192]
    prfm pldl1keep, [x0, #256]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_5x)

// Six prefetches at byte offsets 0, 64, 128, 192, 256, and 320.
KAI_ASM_FUNCTION_TYPE(kai_ops_prefetch_6x)
KAI_ASM_FUNCTION_LABEL(kai_ops_prefetch_6x)
    prfm pldl1keep, [x0]
    prfm pldl1keep, [x0, #64]
    prfm pldl1keep, [x0, #128]
    prfm pldl1keep, [x0, #192]
    prfm pldl1keep, [x0, #256]
    prfm pldl1keep, [x0, #320]
    ret
    KAI_ASM_FUNCTION_END(kai_ops_prefetch_6x)

    KAI_ASM_END
