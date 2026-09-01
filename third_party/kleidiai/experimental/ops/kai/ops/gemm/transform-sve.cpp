//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "common_internal/utils.hpp"

#include "kai/ops/bfloat.hpp"
#include "transform.hpp"

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */


namespace kai {
namespace ops {

#include "transforms/list-sve.hpp"

}  // namespace ops
}  // namespace kai
