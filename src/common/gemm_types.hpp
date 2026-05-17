/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_GEMM_TYPES_HPP
#define COMMON_GEMM_TYPES_HPP

#include "common/c_types_map.hpp"

// gemm_desc_t / gemm_pd_t are gone — every internal caller routes through
// matmul_desc_t / matmul_pd_t now. This header survives only as the home for
// the BLAS-style enums (transpose_t, offsetc_t, sum_ab_t) that JIT GEMM
// kernels and CPU BLAS gemm kernels still consume.

namespace dnnl {
namespace impl {

enum transpose_t { dnnl_notrans, dnnl_trans };

namespace transpose {
const transpose_t notrans = dnnl_notrans;
const transpose_t trans = dnnl_trans;
} // namespace transpose

enum offsetc_t { dnnl_fixed, dnnl_column, dnnl_row };

namespace offsetc {
const offsetc_t fixed = dnnl_fixed;
const offsetc_t column = dnnl_column;
const offsetc_t row = dnnl_row;
} // namespace offsetc

enum sum_ab_t { dnnl_sum_a_row, dnnl_sum_b_col, dnnl_sum_none };
namespace sum_ab {
const sum_ab_t sum_a_row = dnnl_sum_a_row;
const sum_ab_t sum_b_col = dnnl_sum_b_col;
const sum_ab_t sum_none = dnnl_sum_none;
} // namespace sum_ab

} // namespace impl
} // namespace dnnl

#endif // COMMON_GEMM_TYPES_HPP
