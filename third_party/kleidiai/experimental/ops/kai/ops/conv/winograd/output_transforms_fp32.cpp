//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "output_transform.hpp"
#include "winograd_implementations.hpp"

namespace kai {
namespace ops {
namespace winograd {
namespace output_transform {

#if defined(__aarch64__)
void sme_fp32_mopa_4x4_3x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
#endif  // defined(__aarch64__)
void arm_fp32_4x4_3x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_2x2_3x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_2x2_5x5(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x6_1x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x4_1x5(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x2_1x7(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);

#define IMPL(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <float, float>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC)

#define IMPL_T(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <float, float>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, Transform ## DRIVER <float, float>::get_transposed_kernel(FUNC))

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
  { IMPL(4, 4, 3, 3, sme_fp32_mopa_4x4_3x3, Unpadded), MethodConstraints::RequiresSME },
#endif  // defined(__aarch64__)
  { IMPL(4, 4, 3, 3, arm_fp32_4x4_3x3, Unpadded), MethodConstraints::LargerShape },
  { IMPL(2, 2, 3, 3, arm_fp32_2x2_3x3, Unpadded) },
  { IMPL(2, 2, 5, 5, arm_fp32_2x2_5x5, Unpadded) },
  { IMPL(1, 6, 1, 3, arm_fp32_1x6_1x3, Unpadded) },
  { IMPL_T(6, 1, 3, 1, arm_fp32_1x6_1x3, Unpadded) },
  { IMPL(1, 4, 1, 5, arm_fp32_1x4_1x5, Unpadded) },
  { IMPL_T(4, 1, 5, 1, arm_fp32_1x4_1x5, Unpadded) },
  { IMPL(1, 2, 1, 7, arm_fp32_1x2_1x7, Unpadded) },
  { IMPL_T(2, 1, 7, 1, arm_fp32_1x2_1x7, Unpadded) },
  { nullptr }
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
