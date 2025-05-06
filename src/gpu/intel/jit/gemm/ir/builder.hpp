/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_IR_BUILDER_HPP
#define GPU_INTEL_JIT_GEMM_IR_BUILDER_HPP

#include "gemmstone/config.hpp"
#include "gpu/intel/jit/ir/ir.hpp"

namespace gemmstone {

struct gemm_ir_desc_t;

ir::stmt_t build_ir(const gemm_ir_desc_t &desc, ir::constraint_set_t cset = {});

} // namespace gemmstone

#endif
