//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include "winograd_implementations.hpp"

namespace kai {
namespace ops {
namespace winograd {

template bool get_implementation<__fp16>(
  WinogradImpl &,
  const CPUInfo *,
  const ConvolutionArgs &,
  int max_threads,
  bool fast_mode,
  const WinogradConfig *,
  const kai::ops::GemmConfig *
);

}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
