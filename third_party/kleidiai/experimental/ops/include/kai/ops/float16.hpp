//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// If the compiler does not support the __fp16 type natively, define this
// wrapper class and use #define to use it instead of __fp16.
#ifndef __ARM_FP16_ARGS

#include <cstdint>
#include <type_traits>

extern "C" uint16_t kai_ops_float_to_fp16(float);
extern "C" float kai_ops_fp16_to_float(uint16_t);

class float16 {
private:
    uint16_t value;

public:
    float16() = default;
    float16(float v) : value(kai_ops_float_to_fp16(v)) {}

    operator float() const {
        return kai_ops_fp16_to_float(value);
    }
};

static_assert(sizeof(float16) == 2);
static_assert(std::is_trivially_copyable<float16>::value);

#define __fp16 float16

#endif
