//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_internal/kernel_args.hpp"

namespace kai {
namespace ops {

template<unsigned int twidth, unsigned int height, bool sve=false, typename Tin, typename Tout>
void MergeResults(const KernelArgs *args);

}  // namespace ops
}  // namespace kai
