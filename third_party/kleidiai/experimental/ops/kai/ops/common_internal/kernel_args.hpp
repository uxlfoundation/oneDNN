//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Any32 structure - this allows members of kernel_args to contain data of
// any type <= 4 bytes.
struct Any32 {
    uint32_t data = 0;

    Any32() = default;
    Any32(const Any32 &) = default;
    Any32 &operator=(const Any32 &) = default;

    template<typename T>
    Any32 &operator=(T v) {
        static_assert(sizeof(T) <= 4, "Type too large for Any32");
        static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");
        std::memcpy(&data, &v, sizeof(T));
        return *this;
    }

    template<typename T>
    Any32(const T &v) {
        *this = v;
    }
};

struct KernelArgs {
    int64_t         M;                          // Height of problem
    int64_t         N;                          // Width of problem
    int64_t         K;                          // Accumulation depth / number of segments for indirect GEMM
    const int64_t * K_lengths;                  // Length of each segment for indirect GEMM
    const void *    lhs;                        // LHS data pointer (direct or indirect)
    int64_t         lhs_stride;                 // LHS offset - stride for direct cases, per-row offset for indirect
    const void *    rhs;                        // RHS data pointer (direct or indirect)
    int64_t         rhs_stride;                 // RHS offset - stride for direct cases, per-row offset for indirect
    const void *    bias;                       // Pointer to bias data
    const void *    late_bias;                  // Pointer to 'late' bias data, for int->fp kernels with FP bias
    void *          dst;                        // Destination pointer
    int64_t         dst_stride;                 // Destination stride
    int64_t         accumulate;                 // Flag to indicate accumulate mode
    void *          accumulation_buffer;        // Pointer ta accumulation buffer
    int64_t         accumulation_buffer_stride; // Stride between accumulation buffer rows.
    const void *    scale_mul_rhs_ptr;          // Pointer to array of per-channel scale multipliers
    const void *    scale_mul_lhs_ptr;          // Pointer to array of per-activation scale multipliers
    const void *    scale_shift_rhs_ptr;        // Pointer to array of per-channel shift values
    Any32           scale_rhs_offset;           // RHS zero offset
    Any32           scale_result_offset;        // Result zero offset
    Any32           scale_tensor_mul;           // Per tensor scaling multiplier
    Any32           scale_tensor_shift;         // Per tensor shift value
    Any32           clamp_min;                  // Minimum clamp value
    Any32           clamp_max;                  // Maximum clamp value
};

// Check that the layout used by the compiler matches the layout assumed by
// the assembler code.
static_assert(offsetof(KernelArgs, M) == 0, "KernelArgs offset mismatch (M)");
static_assert(offsetof(KernelArgs, N) == 8, "KernelArgs offset mismatch (N)");
static_assert(offsetof(KernelArgs, K) == 16, "KernelArgs offset mismatch (K)");
static_assert(offsetof(KernelArgs, K_lengths) == 24, "KernelArgs offset mismatch (K_lengths)");
static_assert(offsetof(KernelArgs, lhs) == 32, "KernelArgs offset mismatch (lhs)");
static_assert(offsetof(KernelArgs, lhs_stride) == 40, "KernelArgs offset mismatch (lhs_stride)");
static_assert(offsetof(KernelArgs, rhs) == 48, "KernelArgs offset mismatch (rhs)");
static_assert(offsetof(KernelArgs, rhs_stride) == 56, "KernelArgs offset mismatch (rhs_stride)");
static_assert(offsetof(KernelArgs, bias) == 64, "KernelArgs offset mismatch (bias)");
static_assert(offsetof(KernelArgs, late_bias) == 72, "KernelArgs offset mismatch (late_bias)");
static_assert(offsetof(KernelArgs, dst) == 80, "KernelArgs offset mismatch (dst)");
static_assert(offsetof(KernelArgs, dst_stride) == 88, "KernelArgs offset mismatch (dst_stride)");
static_assert(offsetof(KernelArgs, accumulate) == 96, "KernelArgs offset mismatch (accumulate)");
static_assert(offsetof(KernelArgs, accumulation_buffer) == 104, "KernelArgs offset mismatch (accumulation_buffer)");
static_assert(offsetof(KernelArgs, accumulation_buffer_stride) == 112, "KernelArgs offset mismatch (accumulation_buffer_stride)");
static_assert(offsetof(KernelArgs, scale_mul_rhs_ptr) == 120, "KernelArgs offset mismatch (scale_mul_rhs_ptr)");
static_assert(offsetof(KernelArgs, scale_mul_lhs_ptr) == 128, "KernelArgs offset mismatch (scale_mul_lhs_ptr)");
static_assert(offsetof(KernelArgs, scale_shift_rhs_ptr) == 136, "KernelArgs offset mismatch (scale_shift_rhs_ptr)");
static_assert(offsetof(KernelArgs, scale_rhs_offset) == 144, "KernelArgs offset mismatch (scale_rhs_offset)");
static_assert(offsetof(KernelArgs, scale_result_offset) == 148, "KernelArgs offset mismatch (scale_result_offset)");
static_assert(offsetof(KernelArgs, scale_tensor_mul) == 152, "KernelArgs offset mismatch (scale_tensor_mul)");
static_assert(offsetof(KernelArgs, scale_tensor_shift) == 156, "KernelArgs offset mismatch (scale_tensor_shift)");
static_assert(offsetof(KernelArgs, clamp_min) == 160, "KernelArgs offset mismatch (clamp_min)");
static_assert(offsetof(KernelArgs, clamp_max) == 164, "KernelArgs offset mismatch (clamp_max)");
