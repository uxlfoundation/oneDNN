//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "input_transform.hpp"
#include "winograd_implementations.hpp"

#include <memory>
#include <string>

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

#if defined(__aarch64__)
void sme_fp32_mla_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
void sve_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
void a64_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
#else  // defined(__aarch64__)
void arm_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
#endif  // defined(__aarch64__)
void arm_fp32_4x4(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_1x8(unsigned int, const float *, size_t, size_t, float *, size_t);

#define IMPL(HEIGHT, WIDTH, FUNC, DRIVER) new Transform ## DRIVER <float, float>(#FUNC, HEIGHT, WIDTH, FUNC)

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
  { IMPL(6, 6, sme_fp32_mla_6x6, Unpadded), MethodConstraints::RequiresSME },
  { IMPL(6, 6, sve_fp32_6x6, Unpadded), MethodConstraints::RequiresSVE },
  { IMPL(6, 6, a64_fp32_6x6, Unpadded) },
#else  // defined(__aarch64__)
  { IMPL(6, 6, arm_fp32_6x6, Unpadded) },
#endif  // defined(__aarch64__)
  { IMPL(4, 4, arm_fp32_4x4, Unpadded) },
  { IMPL(1, 8, arm_fp32_1x8, Unpadded) },
  { new TransformUnpadded<float, float>("arm_fp32_1x8", 8, 1, TransformUnpadded<float, float>::get_transposed_kernel(arm_fp32_1x8)) },
  { nullptr },
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
