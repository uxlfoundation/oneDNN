//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace kai {
namespace ops {

template<unsigned int twidth, unsigned int height, bool sve=false, typename Tin, typename Tout>
void MergeResults(Tout * out, const Tin * in, int ldc, int y0, int ymax, int x0, int xmax, const Tout *bias, Activation act, bool append);

}  // namespace ops
}  // namespace kai
