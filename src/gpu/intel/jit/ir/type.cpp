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

#include "gpu/intel/jit/ir/include/type.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

struct type_internal_accessor_t {
    using kind_t = type_t::kind_t;
    static int mantissa_bits(const type_t &t) { return t.mantissa_bits(); }
};

namespace type {
using kind_t = type_internal_accessor_t::kind_t;

static auto kind_names = nstl::to_array({
        make_enum_name(kind_t::undef, "undef"),
        make_enum_name(kind_t::u4, "u4"),
        make_enum_name(kind_t::s4, "s4"),
        make_enum_name(kind_t::u8, "u8"),
        make_enum_name(kind_t::s8, "s8"),
        make_enum_name(kind_t::u16, "u16"),
        make_enum_name(kind_t::s16, "s16"),
        make_enum_name(kind_t::u32, "u32"),
        make_enum_name(kind_t::s32, "s32"),
        make_enum_name(kind_t::u64, "u64"),
        make_enum_name(kind_t::s64, "s64"),
        make_enum_name(kind_t::f4_e3m0, "f4_e3m0"),
        make_enum_name(kind_t::f4_e2m1, "f4_e2m1"),
        make_enum_name(kind_t::bf8, "bf8"),
        make_enum_name(kind_t::hf8, "hf8"),
        make_enum_name(kind_t::bf16, "bf16"),
        make_enum_name(kind_t::f16, "f16"),
        make_enum_name(kind_t::tf32, "tf32"),
        make_enum_name(kind_t::f32, "f32"),
        make_enum_name(kind_t::f64, "f64"),
        make_enum_name(kind_t::byte, "byte"),
        make_enum_name(kind_t::dword, "dword"),
        make_enum_name(kind_t::qword, "qword"),
        make_enum_name(kind_t::oword, "oword"),
        make_enum_name(kind_t::hword, "hword"),
        make_enum_name(kind_t::_bool, "bool"),
});

const std::string &to_string(kind_t kind) {
    for (auto &entry : kind_names) {
        if (entry.first == kind) return entry.second;
    }
    static const std::string invalid = "(invalid type::kind_t)";
    return invalid;
}

kind_t get_kind(ngen::DataType t) {
    switch (t) {
        case ngen::DataType::uq: return kind_t::u64;
        case ngen::DataType::q: return kind_t::s64;
        case ngen::DataType::ud: return kind_t::u32;
        case ngen::DataType::d: return kind_t::s32;
        case ngen::DataType::uw: return kind_t::u16;
        case ngen::DataType::w: return kind_t::s16;
        case ngen::DataType::ub: return kind_t::u8;
        case ngen::DataType::b: return kind_t::s8;
        case ngen::DataType::u4: return kind_t::u4;
        case ngen::DataType::s4: return kind_t::s4;

        case ngen::DataType::df: return kind_t::f64;
        case ngen::DataType::f: return kind_t::f32;
        case ngen::DataType::tf32: return kind_t::tf32;
        case ngen::DataType::hf: return kind_t::f16;
        case ngen::DataType::bf: return kind_t::bf16;
        case ngen::DataType::bf8: return kind_t::bf8;
        case ngen::DataType::hf8: return kind_t::hf8;
        default: return kind_t::undef;
    }
}

} // namespace type

type_t::type_t(ngen::DataType type, uint32_t elems, attr_t attr)
    : type_t(type::get_kind(type), elems, attr) {}

size_t type_t::get_hash() const {
    return ir_utils::get_hash(kind(), elems(), is_ptr());
}

int type_t::size() const {
    if (is_ptr()) return sizeof(uint64_t);

    if (is_bool()) return utils::div_up(elems(), 8);
    if (is_x4() || is_fp4()) return utils::div_up(elems(), 2);

    if (elems() != 1) return elems() * base().size();

    switch (kind()) {
        case kind_t::u8:
        case kind_t::s8:
        case kind_t::bf8:
        case kind_t::hf8:
        case kind_t::byte: return 1;
        case kind_t::u16:
        case kind_t::s16:
        case kind_t::bf16:
        case kind_t::f16: return 2;
        case kind_t::u32:
        case kind_t::s32:
        case kind_t::tf32:
        case kind_t::f32:
        case kind_t::dword: return 4;
        case kind_t::f64:
        case kind_t::u64:
        case kind_t::s64:
        case kind_t::qword: return 8;
        case kind_t::oword: return 16;
        case kind_t::hword: return 32;
        default: gpu_error_not_expected();
    }
    return 0;
}

int type_t::mantissa_bits() const {
    if (!is_fp()) return 0;

    switch (kind()) {
        case kind_t::f64: return 52;
        case kind_t::f32: return 23;
        case kind_t::tf32:
        case kind_t::f16: return 10;
        case kind_t::bf16: return 7;
        case kind_t::hf8: return 3;
        case kind_t::bf8: return 2;
        case kind_t::f4_e2m1: return 1;
        case kind_t::f4_e3m0: return 0;
        default: gpu_error_not_expected();
    }
    return 0;
}

std::string type_t::str() const {
    ostringstream_t oss;
    oss << type::to_string(kind());
    if (elems() > 1) oss << "x" << elems();
    if (is_ptr()) oss << ".ptr";
    if (is_simd()) oss << ".simd";
    if (is_mutable()) oss << ".mut";
    if (is_slm()) oss << ".slm";
    return oss.str();
}

void type_t::parse(std::istream &in) {
    bool found = false;
    kind_t kind = {};
    for (auto &entry : type::kind_names) {
        if (stream_try_match(in, entry.second)) {
            kind = entry.first;
            found = true;
        }
    }
    if (!found) {
        *this = {};
        return;
    };

    int elems = 1;
    if (stream_try_match(in, "x")) { in >> elems; }

    attr_t attr {};
    if (stream_try_match(in, ".ptr")) { attr |= attr_t::ptr; }
    if (stream_try_match(in, ".simd")) { attr |= attr_t::simd; }
    if (stream_try_match(in, ".mut")) { attr |= attr_t::mut; }
    if (stream_try_match(in, ".slm")) { attr |= attr_t::slm; }
    *this = type_t(kind, elems, attr);
}

bool is_subset(const type_t &a, const type_t &b) {
    auto is_untyped = [](const type_t &t) {
        return t.is_byte() || t.is_dword() || t.is_qword() || t.is_oword()
                || t.is_hword();
    };

    if (a.is_undef() || b.is_undef()) return false;
    if (a.elems() != b.elems()) return false; // unordered
    if (a.is_ptr() && b.is_ptr()) return true; // XXX: consider alignments?
    if (a.is_ptr() || b.is_ptr()) return false; // unordered
    if (a == b) return true;
    if (a.is_tf32() && b.is_f32()) return true;
    if (a.is_fp() && b.is_int()) return false;

    const auto a_bits = a.base().bitsize();
    const auto b_bits = b.base().bitsize();
    if (is_untyped(a) && is_untyped(b)) return a_bits <= b_bits;
    if (is_untyped(a) || is_untyped(b)) return false; // unordered
    if (a.is_int() && b.is_fp())
        return a_bits
                <= type_internal_accessor_t::mantissa_bits(b) + a.is_signed();
    if (a.is_int() && b.is_int())
        // There are 4 cases:
        // 1. sN is not a subset of uM
        // 2. uN is a subset of sM if N <= M - 1
        // 3. sN is a subset of sM if N - 1 <= M - 1
        // 4. uN is a subset of uM if N <= M
        return (!a.is_signed() || b.is_signed())
                && a_bits + b.is_signed() <= b_bits + a.is_signed();
    return a_bits < b_bits;
}

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
