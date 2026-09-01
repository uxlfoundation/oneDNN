//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>

#include "kai/ops/gemm/kai_ops.hpp"

namespace kai {
namespace ops {

// Fallback routine to add bias to a block
template<typename T>
inline void bias_adder(T *out, unsigned int stride, const T *bias, unsigned int rows, unsigned int cols) {
    for (unsigned int row=0; row<rows; row++) {
        for (unsigned int col=0; col<cols; col++) {
            out[row * stride + col] += bias[col];
        }
    }
}

template<bool DoBias, typename T>
inline void activator(T *out, unsigned int stride, const T *bias, Activation act, unsigned int rows, unsigned int cols) {
    if (act.type == Activation::Type::None) {
        if (DoBias) {
            bias_adder(out, stride, bias, rows, cols);
        }
        return;
    }

    if (act.type == Activation::Type::ReLU) {
        for (unsigned int row=0; row<rows; row++) {
            for (unsigned int col=0; col<cols; col++) {
                T &v = out[row * stride + col];
                if (DoBias) {
                    v += bias[col];
                }
                v = std::max(static_cast<T>(0), v);
            }
        }
    }

    if (act.type == Activation::Type::BoundedReLU) {
        const T max = static_cast<T>(act.param1);

        for (unsigned int row=0; row<rows; row++) {
            for (unsigned int col=0; col<cols; col++) {
                T &v = out[row * stride + col];
                if (DoBias) {
                    v += bias[col];
                }
                v = std::max(static_cast<T>(0), std::min(v, max));
            }
        }
    }
}

}  // namespace ops
}  // namespace kai
