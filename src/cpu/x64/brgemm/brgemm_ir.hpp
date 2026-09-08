/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_BRGEMM_IR_HPP
#define CPU_X64_BRGEMM_BRGEMM_IR_HPP

// IR-based GEMM kernel.

#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Returns `status::success` when the IR GEMM kernel can generate code for
// `brg`, otherwise `status::unimplemented`.
status_t brgemm_ir_supported(const brgemm_desc_t &brg);

// Creates an IR-based GEMM kernel. Caller owns the pointer.
// Returns `nullptr` on failure.
brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
