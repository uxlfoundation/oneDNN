//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#pragma once

#include <cstdint>
#include "../std_transforms_sme.hpp"

namespace kai {
namespace ops {

// Implementations
void sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer);

class cls_sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL
{
public:
  typedef int8_t lhs_operand_type;
  typedef int8_t rhs_operand_type;
  typedef float result_type;

  typedef void (*kern_type)(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer);

  /* Kernel blocking parameters */
  static unsigned int out_height()
  {
    return sme::get_vector_length<int32_t>() * 2;
  }

  static unsigned int out_width()
  {
    return sme::get_vector_length<int32_t>() * 2;
  }

  static constexpr unsigned int k_unroll()
  {
    return 4;
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
  kern_type kernel = sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL;

  StdTransformsSME<lhs_operand_type, result_type, 2, 2, 4> transforms = {};

  cls_sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL(const CPUInfo *)
  {
  }
};

} // namespace ops
} // namespace kai
