//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

/* As some of the merges need these headers, but are all included in the
 * kai::ops namespace, put these headers here.  */
#include <algorithm>

#include <arm_neon.h>

#include "kai/ops/gemm/kai_ops.hpp"
#include "asmlib.hpp"
#include "kai/ops/bfloat.hpp"
#include "common_internal/utils.hpp"

#include "mergeresults.hpp"

namespace kai {
namespace ops {

#include "merges/list-sve.hpp"

}  // namespace ops
}  // namespace kai
