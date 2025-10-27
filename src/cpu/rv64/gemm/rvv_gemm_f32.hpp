/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
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

#ifndef CPU_RV64_GEMM_RVV_GEMM_F32_HPP
#define CPU_RV64_GEMM_RVV_GEMM_F32_HPP

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t rvv_gemm_f32(const char *transa, const char *transb, const dim_t *M,
        const dim_t *N, const dim_t *K, const float *alpha, const float *A,
        const dim_t *lda, const float *B, const dim_t *ldb, const float *beta,
        float *C, const dim_t *ldc, const float *bias);
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVV_GEMM_F32_HPP
