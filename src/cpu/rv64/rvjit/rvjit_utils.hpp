/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVJIT_RVJIT_UTILS_HPP
#define CPU_RV64_RVJIT_RVJIT_UTILS_HPP

#include <cstdint>
#include <memory>
#include <utility>

#include "cpu/rv64/rvjit/rvjit_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// std::make_unique equivalent, since rvjit stays buildable under C++11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// Determines if an integer value is representable as a signed 12-bit value
inline bool is_simm12(int i) {
    return -2048 <= i && i < 2048;
}

/// Determines if a general-purpose register can be written to
inline bool is_writeable(const Reg &r) {
    return r != Xbyak_riscv::x0;
}

/// Determines if a general-purpose register is not the zero register
inline bool is_not_zero(const Reg &r) {
    return r != Xbyak_riscv::x0;
}

/// Determines if a value is a power of two
inline bool is_pow2(int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

/// Base-2 logarithm of a power-of-two value
///
/// @pre `is_pow2(v)` is true
inline int ilog2q(int64_t v) {
    int r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}

/// Maps a narrow rvjit operand type to its natural 2x-wide counterpart
///
/// @note s8/u8 map to s64, not to themselves: there's no s16/u16 in this
///     enum, and returning `dt` unchanged would make fma_t::widening()'s
///     result alias a uniform triple instead of staying unsupported
inline optype_t natural_wide(const optype_t &dt) {
    switch (dt) {
        case optype_t::f16:
        case optype_t::bf16: return optype_t::f32;
        case optype_t::f32: return optype_t::f64;
        case optype_t::s32: return optype_t::s64;
        case optype_t::s8:
        case optype_t::u8: return optype_t::s64;
        default: return dt;
    }
}

/// Determines if an rvjit integer operand type is signed
inline bool is_signed_int(const optype_t &dt) {
    switch (dt) {
        case optype_t::s8:
        case optype_t::s32:
        case optype_t::s64: return true;
        default: return false;
    }
}

/// Determines if an rvjit operand type is an integer type
inline bool is_int_optype(const optype_t &dt) {
    switch (dt) {
        case optype_t::s8:
        case optype_t::u8:
        case optype_t::s32:
        case optype_t::s64: return true;
        default: return false;
    }
}

/// Single-element-width size in bytes for a given rvjit operand type
inline int sewb(const optype_t &dt) {
    switch (dt) {
        case optype_t::s8:
        case optype_t::u8: return 1;
        case optype_t::f16:
        case optype_t::bf16: return 2;
        case optype_t::s32:
        case optype_t::f32: return 4;
        case optype_t::s64:
        case optype_t::f64: return 8;
        default: return 0;
    }
}

/// Obtains the RVV SEW value for a given rvjit operand type
inline SEW sew_for(const optype_t &dt) {
    switch (sewb(dt)) {
        case 8: return SEW::e64;
        case 4: return SEW::e32;
        case 2: return SEW::e16;
        default: return SEW::e8;
    }
}

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
