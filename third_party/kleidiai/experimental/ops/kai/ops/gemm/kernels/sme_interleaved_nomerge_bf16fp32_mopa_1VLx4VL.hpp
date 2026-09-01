//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include "kai/ops/bfloat.hpp"
#include "../std_transforms_sme.hpp"

namespace kai {
namespace ops {

// Implementations
void sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL(const bfloat16 *const A, const bfloat16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer);

class cls_sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL
{
public:
  typedef bfloat16 lhs_operand_type;
  typedef bfloat16 rhs_operand_type;
  typedef float result_type;

  typedef void (*kern_type)(const bfloat16 *const A, const bfloat16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer);

  /* Kernel blocking parameters */
  static unsigned int out_height()
  {
    return sme::get_vector_length<float>() * 1;
  }

  static unsigned int out_width()
  {
    return sme::get_vector_length<float>() * 4;
  }

  static constexpr unsigned int k_unroll()
  {
    return 2;
  }

  static constexpr bool supports_bias()
  {
    return true;
  }

  static constexpr bool is_sme()
  {
    return true;
  }

  // Default to the generic kernel
  kern_type kernel = sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL;

  StdTransformsSME<lhs_operand_type, result_type, 1, 4, 2> transforms = {};

  cls_sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL(const CPUInfo *)
  {
  }
};

} // namespace ops
} // namespace kai
