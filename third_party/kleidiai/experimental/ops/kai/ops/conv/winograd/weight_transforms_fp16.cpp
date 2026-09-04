//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include "winograd_implementations.hpp"
#include "weight_transform.hpp"

namespace kai {
namespace ops {
namespace winograd {
namespace weight_transform {

void *a64_fp16_4x4_3x3(unsigned int, const __fp16 *, size_t, size_t, __fp16 *, size_t);

#define IMPL(KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN) \
  new Transform<__fp16>(#KERN, KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN)

static const TransformImplementation<__fp16> transforms_fp16[] = {
  { IMPL(3, 3, 6, 6, a64_fp16_4x4_3x3) },
  { nullptr }
};

template <>
const TransformImplementation<__fp16> *implementation_list(void)
{
  return transforms_fp16;
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
