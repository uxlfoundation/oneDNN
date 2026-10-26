/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_BRGEMM_BRGEMM_HPP
#define CPU_RV64_BRGEMM_BRGEMM_HPP

#include "cpu/rv64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t brgemm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, data_type_t dt_a, data_type_t dt_b,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides = nullptr, bool wide_tile = false);

// Fits the kernel of `brg` to weights laid out by `tile`, the M tile they are
// blocked by, and bounds by `n_bound` the N of a call when it is known (0:
// not known). Unimplemented when the kernel's tile does not divide `tile`
status_t brgemm_desc_set_weights_tile(
        brgemm_desc_t *brg, int tile, dim_t n_bound);

// Sets the output type of `brg` (an f16 store) and re-tunes its blocks for it
status_t brgemm_desc_set_dt_c(brgemm_desc_t *brg, data_type_t dt_c);

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_desc_t &brg);

void brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel);

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, const void *ptr_A,
        const void *ptr_B, void *ptr_C, dim_t N, float beta,
        const void *ptr_bias = nullptr);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_BRGEMM_BRGEMM_HPP
