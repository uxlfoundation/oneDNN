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

#ifndef GPU_INTEL_JIT_DSL_DECL_HPP
#define GPU_INTEL_JIT_DSL_DECL_HPP

#include "gpu/intel/jit/ir/include/ir.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

using jit::operator<<;

static type_t _bool = type_t::_bool();
static type_t s8 = type_t::s8();
static type_t u8 = type_t::u8();
static type_t s16 = type_t::s16();
static type_t u16 = type_t::u16();
static type_t s32 = type_t::s32();
static type_t u32 = type_t::u32();
static type_t s64 = type_t::s64();
static type_t u64 = type_t::u64();
static type_t f32 = type_t::f32();
static type_t f16 = type_t::f16();
static type_t bf16 = type_t::bf16();

using expr_t = jit::expr_t;

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
