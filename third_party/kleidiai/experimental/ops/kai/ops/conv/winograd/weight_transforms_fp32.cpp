//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "winograd_implementations.hpp"
#include "weight_transform.hpp"

namespace kai {
namespace ops {
namespace winograd {
namespace weight_transform {

#if defined(__aarch64__)
#endif  // defined(__aarch64__)
void arm_fp32_4x4_3x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_2x2_3x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_2x2_5x5(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x6_1x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x4_1x5(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x2_1x7(unsigned int, const float *, size_t, size_t, float *, size_t);

#define IMPL(KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN) \
  new Transform<float>(#KERN, KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN)

#define IMPL_T(KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN) \
  new Transform<float>(#KERN, KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, Transform<float>::get_transposed_kernel(KERN))

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
#endif  // defined(__aarch64__)
  { IMPL(3, 3, 6, 6, arm_fp32_4x4_3x3) },
  { IMPL(3, 3, 4, 4, arm_fp32_2x2_3x3) },
  { IMPL(5, 5, 6, 6, arm_fp32_2x2_5x5) },
  { IMPL(1, 3, 1, 8, cpp_fp32_1x6_1x3) },
  { IMPL_T(3, 1, 8, 1, cpp_fp32_1x6_1x3) },
  { IMPL(1, 5, 1, 8, cpp_fp32_1x4_1x5) },
  { IMPL_T(5, 1, 8, 1, cpp_fp32_1x4_1x5) },
  { IMPL(1, 7, 1, 8, cpp_fp32_1x2_1x7) },
  { IMPL_T(7, 1, 8, 1, cpp_fp32_1x2_1x7) },
  { nullptr }
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
