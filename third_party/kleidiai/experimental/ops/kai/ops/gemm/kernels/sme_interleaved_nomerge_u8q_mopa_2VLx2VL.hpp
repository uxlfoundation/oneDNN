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
void sme_interleaved_nomerge_u8q_mopa_2VLx2VL(const uint8_t *const A, const uint8_t *const B, uint8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer);

class cls_sme_interleaved_nomerge_u8q_mopa_2VLx2VL
{
public:
  typedef uint8_t lhs_operand_type;
  typedef uint8_t rhs_operand_type;
  typedef uint8_t result_type;

  typedef void (*kern_type)(const uint8_t *const A, const uint8_t *const B, uint8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer);

  /* Kernel blocking parameters */
  static unsigned int out_height()
  {
    return sme::get_vector_length<uint32_t>() * 2;
  }

  static unsigned int out_width()
  {
    return sme::get_vector_length<uint32_t>() * 2;
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
  kern_type kernel = sme_interleaved_nomerge_u8q_mopa_2VLx2VL;

  StdTransformsSME<lhs_operand_type, result_type, 2, 2, 4, true> transforms = {};

  cls_sme_interleaved_nomerge_u8q_mopa_2VLx2VL(const CPUInfo *)
  {
  }
};

} // namespace ops
} // namespace kai
