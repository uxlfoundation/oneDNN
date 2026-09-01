//
// SPDX-FileCopyrightText: Copyright 2020, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <cstring>

namespace {
    inline uint16_t float_to_bf16(const float v) {
        const uint32_t *fromptr = reinterpret_cast<const uint32_t *>(&v);

        uint16_t res = (*fromptr >> 16);
        uint16_t error = (*fromptr & 0x0000ffff);
        uint16_t lsb = (res & 0x0001);

        // Implement round-to-nearest, ties-to-even rounding
        if ((error > 0x8000) || ((error == 0x8000) && (lsb==1))) {
            res += 1;
        }

        return res;
}

    inline float bf16_to_float(const uint16_t &v) {
        const uint32_t lv = (static_cast<uint32_t>(v) << 16);
        float f;

        memcpy(&f, &lv, sizeof(float));

        return f;
    }
}

class bfloat16 {
private:
    uint16_t value;

public:
    bfloat16() : value(0) { }
    bfloat16(float v) : value (float_to_bf16(v)) { }

    bfloat16 & operator=(float v) {
        value = float_to_bf16(v);
        return *this;
    }

    float to_float() const {
        return bf16_to_float(value);
    }

    operator float() const {
        return bf16_to_float(value);
    }

    float operator*(const bfloat16 &b) const {
        return this->to_float() * b.to_float();
    }

    float operator+(const bfloat16 &b) const {
        return this->to_float() + b.to_float();
    }

    bfloat16 operator*=(const bfloat16 &b) {
        value = float_to_bf16(this->to_float() * b.to_float());
        return *this;
    }

    bfloat16 operator+=(const bfloat16 &b) {
        value = float_to_bf16(this->to_float() + b.to_float());
        return *this;
    }
};
