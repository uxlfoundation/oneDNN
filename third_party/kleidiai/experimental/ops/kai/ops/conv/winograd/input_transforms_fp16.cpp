//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include "input_transform.hpp"
#include "winograd_implementations.hpp"

#include <memory>
#include <string>

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

void a64_fp16_6x6(unsigned int, const __fp16 *, size_t, size_t, __fp16 *, size_t);

#define IMPL(HEIGHT, WIDTH, FUNC, DRIVER) new Transform ## DRIVER <__fp16, __fp16>(#FUNC, HEIGHT, WIDTH, FUNC)

static const TransformImplementation<__fp16> transforms_fp16[] = {
  { IMPL(6, 6, a64_fp16_6x6, Unpadded) },
  { nullptr },
};

template <>
const TransformImplementation<__fp16> *implementation_list(void)
{
  return transforms_fp16;
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
