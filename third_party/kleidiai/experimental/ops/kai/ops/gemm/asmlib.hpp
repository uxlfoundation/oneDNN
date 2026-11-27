//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2021, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

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

/*
 * Do some prefetches.
 */

extern "C" void kai_ops_prefetch_6x(const void* address);
extern "C" void kai_ops_prefetch_5x(const void* address);
extern "C" void kai_ops_prefetch_4x(const void* address);
extern "C" void kai_ops_prefetch_3x(const void* address);
extern "C" void kai_ops_prefetch_2x(const void* address);
extern "C" void kai_ops_prefetch_1x(const void* address);

template <typename T>
static inline void prefetch_6x(const T *pfp) {
    kai_ops_prefetch_6x(pfp);
}

template <typename T>
static inline void prefetch_5x(const T *pfp) {
    kai_ops_prefetch_5x(pfp);
}

template <typename T>
static inline void prefetch_4x(const T *pfp) {
    kai_ops_prefetch_4x(pfp);
}

template <typename T>
static inline void prefetch_3x(const T *pfp) {
    kai_ops_prefetch_3x(pfp);
}

template <typename T>
static inline void prefetch_2x(const T *pfp) {
    kai_ops_prefetch_2x(pfp);
}

template <typename T>
static inline void prefetch_1x(const T *pfp) {
    kai_ops_prefetch_1x(pfp);
}

