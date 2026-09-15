//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include "output_transform.hpp"
#include "winograd_implementations.hpp"

namespace kai {
namespace ops {
namespace winograd {
namespace output_transform {

void a64_fp16_4x4_3x3(unsigned int, const __fp16 *, size_t, const __fp16 *, __fp16 *, size_t, size_t, __fp16, __fp16);

#define IMPL(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <__fp16, __fp16>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC)


static const TransformImplementation<__fp16> transforms_fp16[] = {
  { IMPL(4, 4, 3, 3, a64_fp16_4x4_3x3, Unpadded) },
  { nullptr }
};

template <>
const TransformImplementation<__fp16> *implementation_list(void)
{
  return transforms_fp16;
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
