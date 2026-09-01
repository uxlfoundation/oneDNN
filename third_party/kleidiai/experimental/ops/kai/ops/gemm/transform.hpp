//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

template <unsigned int IntBy, unsigned int BlockBy, bool Transposed, VLType vlt=VLType::None, typename TOut, typename TIn>
void Transform(
  TOut* out, const TIn* const in, const int stride,
  const int k0, const int kmax, const int x0, const int xmax
);

}  // namespace ops
}  // namespace kai
