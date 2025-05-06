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

#ifndef GPU_INTEL_JIT_GEMM_IR_IR_INTEROP_HPP
#define GPU_INTEL_JIT_GEMM_IR_IR_INTEROP_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/type.hpp"
#include "gpu/intel/jit/ir/core.hpp"

namespace gemmstone {

inline ir::type_t into_ir(Type t, int elems = 1) {
    using namespace ir;
    switch (t) {
        case Type::invalid: return type_t::undef();

        case Type::f4_e3m0: return type_t::f4_e3m0(elems);
        case Type::f4_e2m1: return type_t::f4_e2m1(elems);
        case Type::bf8: return type_t::bf8(elems);
        case Type::hf8: return type_t::hf8(elems);
        case Type::bf16: return type_t::bf16(elems);
        case Type::f16: return type_t::f16(elems);
        case Type::tf32: return type_t::tf32(elems);
        case Type::f32: return type_t::f32(elems);
        case Type::f64: return type_t::f64(elems);

        case Type::u4: return type_t::u4(elems);
        case Type::s4: return type_t::s4(elems);
        case Type::u8: return type_t::u8(elems);
        case Type::s8: return type_t::s8(elems);
        case Type::u16: return type_t::u16(elems);
        case Type::s16: return type_t::s16(elems);
        case Type::u32: return type_t::u32(elems);
        case Type::s32: return type_t::s32(elems);
        case Type::u64: return type_t::u64(elems);
        case Type::s64: return type_t::s64(elems);

        default: stub(); return type_t::undef();
    }
}

} // namespace gemmstone

#endif
