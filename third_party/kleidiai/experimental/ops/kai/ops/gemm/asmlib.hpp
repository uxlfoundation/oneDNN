//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2021, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __aarch64__
// Macro to use in assembler to get a preload.  Needed because of various
// workarounds needed to get working preload behaviour.
//
// Code using these macros needs to clobber x20 and x21 as they might be
// used by the workaround.

// "Correct" version
#define ASM_PREFETCH(address)    "PRFM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHU(address)   "PRFUM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHL2(address)  "PRFM PLDL2KEEP, " address "\n"
#define ASM_PREFETCHW(address)   "PRFM PSTL1KEEP, " address "\n"
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

// No preload at all
//#define ASM_PREFETCH(address) ""
#else

// "Correct" versions for AArch32
#define ASM_PREFETCH(address)     "PLD " address "\n"
#define ASM_PREFETCHW(address)    "PLDW " address "\n"

#endif

/*
 * Do some prefetches.
 */
template <typename T>
static inline void prefetch_6x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
        ASM_PREFETCH("[%[pfp], #256]")
        ASM_PREFETCH("[%[pfp], #320]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_5x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
        ASM_PREFETCH("[%[pfp], #256]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_4x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_3x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_2x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_1x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

