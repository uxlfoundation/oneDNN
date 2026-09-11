//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "winograd_implementations.hpp"

namespace kai {
namespace ops {
namespace winograd {

template bool get_implementation<float>(
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
