/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_CORE_HPP
#define GPU_INTEL_JIT_IR_CORE_HPP

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "common/math_utils.hpp"
#include "gpu/intel/jit/codegen/register_allocator.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
#define SANITY_CHECK 1
#endif

// All IR expression objects.
#define HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_IR_OBJECT(binary_op_t) \
    HANDLE_IR_OBJECT(bool_imm_t) \
    HANDLE_IR_OBJECT(cast_t) \
    HANDLE_IR_OBJECT(const_var_t) \
    HANDLE_IR_OBJECT(float_imm_t) \
    HANDLE_IR_OBJECT(iif_t) \
    HANDLE_IR_OBJECT(int_imm_t) \
    HANDLE_IR_OBJECT(linear_t) \
    HANDLE_IR_OBJECT(load_t) \
    HANDLE_IR_OBJECT(ptr_t) \
    HANDLE_IR_OBJECT(shuffle_t) \
    HANDLE_IR_OBJECT(ternary_op_t) \
    HANDLE_IR_OBJECT(unary_op_t) \
    HANDLE_IR_OBJECT(var_t) \
    HANDLE_IR_OBJECT(ref_t)

// All IR statement objects.
#define HANDLE_STMT_IR_OBJECTS() \
    HANDLE_IR_OBJECT(alloc_t) \
    HANDLE_IR_OBJECT(for_t) \
    HANDLE_IR_OBJECT(func_call_t) \
    HANDLE_IR_OBJECT(if_t) \
    HANDLE_IR_OBJECT(let_t) \
    HANDLE_IR_OBJECT(stmt_group_t) \
    HANDLE_IR_OBJECT(stmt_seq_t) \
    HANDLE_IR_OBJECT(store_t) \
    HANDLE_IR_OBJECT(while_t)

#define HANDLE_CORE_IR_OBJECTS() \
    HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_STMT_IR_OBJECTS()

// Auxiliary macros to reduce boilerplate.
#define IR_DECL_TYPE_IMPL(type_id, class_name) \
    using self_type = class_name; \
    static constexpr type_info_t _type_info() { \
        return type_info_t(type_id, typeid(class_name), \
                is_expr_t<class_name>::value, is_stmt_t<class_name>::value); \
    }

#define IR_DECL_CORE_TYPE(class_name) \
    IR_DECL_TYPE_IMPL(ir_type_id_t::class_name, class_name)
#define IR_DECL_TYPE(class_name) \
    IR_DECL_TYPE_IMPL(ir_type_id_t::undef, class_name)

#define IR_DECLARE_TRAVERSERS() \
    object_t _mutate(ir_mutator_t &mutator) const override { \
        return mutator._mutate(*this); \
    } \
    void _visit(ir_visitor_t &visitor) const override { visitor._visit(*this); }

// Defines getter for a function argument.
#define IR_DEFINE_ARG_GET(name, index) \
    static const expr_t &arg_##name(const func_call_t &c) { \
        gpu_assert(c.func.is<self_type>()) << c; \
        return c.args[index]; \
    } \
    static const expr_t &arg_##name(const stmt_t &s) { \
        gpu_assert(s.is<func_call_t>()) << s; \
        auto &c = s.as<func_call_t>(); \
        return arg_##name(c); \
    } \
    template <typename T> \
    static T &arg_##name(std::vector<T> &args) { \
        return args[index]; \
    } \
    template <typename T> \
    static const T &arg_##name(const std::vector<T> &args) { \
        return args[index]; \
    }

#if defined(__GNUC__)
// clang-format off
// Defines dump() method for debugging purposes, to pretty print the object.
#define IR_DEFINE_DUMP() \
    __attribute__((noinline)) \
    __attribute__((used)) \
    void dump() const { \
        printf("%s\n", str().c_str()); \
    }
// clang-format on
#else
#define IR_DEFINE_DUMP()
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class type_kind_t {
    undef,
    _bool,

    // Integer types.
    u4,
    s4,
    u8,
    s8,
    u16,
    s16,
    u32,
    s32,
    u64,
    s64,

    // Floating point types.
    f4_e3m0,
    f4_e2m1,
    bf8,
    f8_e5m2 = bf8,
    hf8,
    f8_e4m3 = hf8,
    bf16,
    f16,
    tf32,
    f32,
    f64,

    // Message data types.
    byte,
    dword,
    qword,
    oword,
    hword
};

static auto type_kind_names = nstl::to_array({
        make_enum_name(type_kind_t::undef, "undef"),
        make_enum_name(type_kind_t::u4, "u4"),
        make_enum_name(type_kind_t::s4, "s4"),
        make_enum_name(type_kind_t::u8, "u8"),
        make_enum_name(type_kind_t::s8, "s8"),
        make_enum_name(type_kind_t::u16, "u16"),
        make_enum_name(type_kind_t::s16, "s16"),
        make_enum_name(type_kind_t::u32, "u32"),
        make_enum_name(type_kind_t::s32, "s32"),
        make_enum_name(type_kind_t::u64, "u64"),
        make_enum_name(type_kind_t::s64, "s64"),
        make_enum_name(type_kind_t::f4_e3m0, "f4_e3m0"),
        make_enum_name(type_kind_t::f4_e2m1, "f4_e2m1"),
        make_enum_name(type_kind_t::bf8, "bf8"),
        make_enum_name(type_kind_t::hf8, "hf8"),
        make_enum_name(type_kind_t::bf16, "bf16"),
        make_enum_name(type_kind_t::f16, "f16"),
        make_enum_name(type_kind_t::tf32, "tf32"),
        make_enum_name(type_kind_t::f32, "f32"),
        make_enum_name(type_kind_t::f64, "f64"),
        make_enum_name(type_kind_t::byte, "byte"),
        make_enum_name(type_kind_t::dword, "dword"),
        make_enum_name(type_kind_t::qword, "qword"),
        make_enum_name(type_kind_t::oword, "oword"),
        make_enum_name(type_kind_t::hword, "hword"),
        make_enum_name(type_kind_t::_bool, "bool"),
});
GPU_DEFINE_PARSE_ENUM(type_kind_t, type_kind_names)

enum class type_attr_t : uint32_t {
    undef = 0,
    ptr = 1,
    mut = 2,
    simd = 4,
    slm = 8
};

GPU_DEFINE_BIT_MASK_ENUM_OPS(type_attr_t)
inline type_attr_t &operator|=(type_attr_t &a, type_attr_t b) {
    return a = a | b;
}
inline type_attr_t &operator&=(type_attr_t &a, type_attr_t b) {
    return a = a & b;
}

class type_t {
public:
    static type_t undef() { return type_t(type_kind_t::undef); }

    static type_t _bool(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::_bool, elems, attr);
    }

    static type_t u4(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::u4, elems, attr);
    }
    static type_t s4(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::s4, elems, attr);
    }
    static type_t u8(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::u8, elems, attr);
    }
    static type_t s8(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::s8, elems, attr);
    }
    static type_t u16(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::u16, elems, attr);
    }
    static type_t s16(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::s16, elems, attr);
    }
    static type_t u32(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::u32, elems, attr);
    }
    static type_t s32(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::s32, elems, attr);
    }
    static type_t u64(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::u64, elems, attr);
    }
    static type_t s64(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::s64, elems, attr);
    }

    // Returns unsigned integer type.
    static type_t u(
            int bits, int elems = 1, type_attr_t attr = type_attr_t::undef) {
        switch (bits) {
            case 4: return u4(elems, attr);
            case 8: return u8(elems, attr);
            case 16: return u16(elems, attr);
            case 32: return u32(elems, attr);
            case 64: return u64(elems, attr);
            default: gpu_error_not_expected();
        }
        return type_t::undef();
    }

    // Returns signed integer type.
    static type_t s(
            int bits, int elems = 1, type_attr_t attr = type_attr_t::undef) {
        switch (bits) {
            case 4: return s4(elems, attr);
            case 8: return s8(elems, attr);
            case 16: return s16(elems, attr);
            case 32: return s32(elems, attr);
            case 64: return s64(elems, attr);
            default: gpu_error_not_expected();
        }
        return type_t::undef();
    }

    static type_t f4_e3m0(
            int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::f4_e3m0, elems, attr);
    }
    static type_t f4_e2m1(
            int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::f4_e2m1, elems, attr);
    }
    static type_t bf8(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::bf8, elems, attr);
    }
    static type_t hf8(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::hf8, elems, attr);
    }
    static type_t bf16(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::bf16, elems, attr);
    }
    static type_t f16(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::f16, elems, attr);
    }
    static type_t tf32(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::tf32, elems, attr);
    }
    static type_t f32(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::f32, elems, attr);
    }
    static type_t f64(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::f64, elems, attr);
    }
    static type_t byte(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::byte, elems, attr);
    }
    static type_t byte_ptr(int elems = 1, bool is_slm = false,
            type_attr_t attr = type_attr_t::undef) {
        auto type = type_t(type_kind_t::byte, elems, attr).with_ptr();
        if (is_slm) type = type.slm();
        return type;
    }
    static type_t dword(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::dword, elems, attr);
    }
    static type_t qword(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::qword, elems, attr);
    }
    static type_t oword(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::oword, elems, attr);
    }
    static type_t hword(int elems = 1, type_attr_t attr = type_attr_t::undef) {
        return type_t(type_kind_t::hword, elems, attr);
    }

    template <typename T>
    static type_t from_cpp() {
#define CASE(cpp_type, type) \
    if (std::is_same<T, cpp_type>::value) return type()

        CASE(bool, _bool);
        CASE(float, f32);
        CASE(double, f64);
        CASE(int16_t, s16);
        CASE(int32_t, s32);
        CASE(int64_t, s64);
        CASE(uint16_t, u16);
        CASE(uint32_t, u32);
        CASE(uint64_t, u64);

#undef CASE

        gpu_error_not_expected();

        return undef();
    }

    template <typename T>
    T max() const {
        switch (kind()) {
            case type_kind_t::u4:
            case type_kind_t::s4:
            case type_kind_t::u8:
            case type_kind_t::s8:
            case type_kind_t::u16:
            case type_kind_t::s16:
            case type_kind_t::u32:
            case type_kind_t::s32:
            case type_kind_t::u64:
            case type_kind_t::s64: {
                int bits = scalar().bitsize();
                if (is_signed()) bits--;
                T ret = T(1) << (bits - 1);
                return ret + (ret - 1);
            }
            default: gpu_error_not_expected();
        }
        return 0;
    }

    template <typename T>
    T min() const {
        switch (kind()) {
            case type_kind_t::u4:
            case type_kind_t::s4:
            case type_kind_t::u8:
            case type_kind_t::s8:
            case type_kind_t::u16:
            case type_kind_t::s16:
            case type_kind_t::u32:
            case type_kind_t::s32:
            case type_kind_t::u64:
            case type_kind_t::s64: {
                if (is_unsigned()) return 0;
                return -max<T>() - 1;
            }
            default: gpu_error_not_expected();
        }
        return 0;
    }

    static bool is_vector(int elems) { return elems != 1; }

    type_t() : type_t(type_t::undef()) {}

    type_t(type_kind_t kind, uint32_t elems = 1,
            type_attr_t attr = type_attr_t::undef)
        : kind_(kind), elems_(elems), attr_(attr) {}

    type_t(const std::string &s) : elems_(1) {
#define CASE(x) \
    if (to_string(type_kind_t::x) == s) { \
        kind_ = type_kind_t::x; \
        return; \
    }
        CASE(f4_e3m0);
        CASE(f4_e2m1);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(tf32);
        CASE(f32);
        CASE(f64);

        CASE(s4);
        CASE(s8);
        CASE(s16);
        CASE(s32);
        CASE(s64);

        CASE(u4);
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
#undef CASE
        gpu_error_not_expected();
    }

    // Constructor from dnnl_data_type_t.
    type_t(data_type_t dt) {
        if (dt == data_type::undef) return;
        elems_ = 1;
        switch ((int)dt) {
#define CASE(x) \
    case data_type::x: kind_ = type_kind_t::x; break;
            CASE(f4_e3m0);
            CASE(f4_e2m1);
            CASE(f8_e5m2);
            CASE(f8_e4m3);
            CASE(bf16);
            CASE(f16);
            CASE(tf32);
            CASE(f32);
            CASE(f64);
            CASE(s32);
            CASE(s8);
            CASE(u8);
            CASE(s4);
            CASE(u4);
#undef CASE
            default: gpu_error_not_expected();
        }
    }

    type_kind_t kind() const { return kind_; }

    int elems() const { return elems_; }

    type_attr_t attr() const { return attr_; }

    bool is_ptr() const { return any(attr() & type_attr_t::ptr); }

    bool is_slm() const { return any(attr() & type_attr_t::slm); }

    bool operator==(const type_t &other) const {
        return (kind() == other.kind()) && (elems() == other.elems())
                && (is_ptr() == other.is_ptr());
    }

    bool operator!=(const type_t &other) const { return !operator==(other); }

    bool is_equal(const type_t &other) const { return operator==(other); }

    size_t get_hash() const {
        return ir_utils::get_hash(kind(), elems(), is_ptr());
    }

    static void init_parse_iface(parse_iface_t<type_t> *iface) {
        iface->add<type_kind_t, &type_t::kind_>();
        iface->set_pre_stringify_func([](const type_t &type) {
            gpu_assert(!type.is_ptr() && (type.is_scalar() || type.is_undef()))
                    << "Cannot stringify pointer/non-scalar type.";
        });
    }

    bool is_undef() const { return kind() == type_kind_t::undef; }

    bool is_vector() const { return type_t::is_vector(elems()); }

    bool is_bool() const { return kind() == type_kind_t::_bool; }

    bool is_fp() const {
        return is_fp4() || is_fp8()
                || utils::one_of(kind(), type_kind_t::bf16, type_kind_t::f16,
                        type_kind_t::tf32, type_kind_t::f32, type_kind_t::f64);
    }

    bool is_f4_e3m0() const { return kind() == type_kind_t::f4_e3m0; }
    bool is_f4_e2m1() const { return kind() == type_kind_t::f4_e2m1; }
    bool is_bf8() const { return kind() == type_kind_t::bf8; }
    bool is_hf8() const { return kind() == type_kind_t::hf8; }
    bool is_bf16() const { return kind() == type_kind_t::bf16; }
    bool is_f16() const { return kind() == type_kind_t::f16; }
    bool is_tf32() const { return kind() == type_kind_t::tf32; }
    bool is_f32() const { return kind() == type_kind_t::f32; }
    bool is_f64() const { return kind() == type_kind_t::f64; }

    bool is_fp4() const { return is_f4_e3m0() || is_f4_e2m1(); }
    bool is_fp8() const { return is_bf8() || is_hf8(); }

    bool is_int() const {
        return is_x4() || is_x8() || is_x16() || is_x32() || is_x64();
    }

    bool is_s4() const { return kind() == type_kind_t::s4; }
    bool is_u4() const { return kind() == type_kind_t::u4; }
    bool is_x4() const { return is_s4() || is_u4(); }

    bool is_s8() const { return kind() == type_kind_t::s8; }
    bool is_u8() const { return kind() == type_kind_t::u8; }
    bool is_x8() const { return is_s8() || is_u8(); }

    bool is_s16() const { return kind() == type_kind_t::s16; }
    bool is_u16() const { return kind() == type_kind_t::u16; }
    bool is_x16() const { return is_s16() || is_u16(); }

    bool is_s32() const { return kind() == type_kind_t::s32; }
    bool is_u32() const { return kind() == type_kind_t::u32; }
    bool is_x32() const { return is_s32() || is_u32(); }

    bool is_s64() const { return kind() == type_kind_t::s64; }
    bool is_u64() const { return kind() == type_kind_t::u64; }
    bool is_x64() const { return is_s64() || is_u64(); }

    bool is_byte() const { return kind() == type_kind_t::byte; }
    bool is_dword() const { return kind() == type_kind_t::dword; }
    bool is_qword() const { return kind() == type_kind_t::qword; }
    bool is_oword() const { return kind() == type_kind_t::oword; }
    bool is_hword() const { return kind() == type_kind_t::hword; }

    bool is_signed(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return utils::one_of(kind(), type_kind_t::s4, type_kind_t::s8,
                type_kind_t::s16, type_kind_t::s32, type_kind_t::s64);
    }

    bool is_unsigned(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return utils::one_of(kind(), type_kind_t::u4, type_kind_t::u8,
                type_kind_t::u16, type_kind_t::u32, type_kind_t::u64);
    }

    bool is_scalar() const { return elems() == 1; }

    bool is_mutable() const { return any(attr() & type_attr_t::mut); }

    bool is_simd() const { return any(attr() & type_attr_t::simd); }

    template <typename T>
    bool is_cpp() const {
        return *this == type_t::from_cpp<T>();
    }

    bool is_bitwise_compatible(const type_t &other) const {
        if (*this == other) return true;

        // tf32 is bitwise compatible with f32.
        if (kind() == type_kind_t::f32 && other.kind() == type_kind_t::tf32)
            return elems() == other.elems();

        return false;
    }

    type_t remove_elems() const { return with_elems(1); }

    type_t remove_ptr() const {
        type_t copy = *this;
        copy.attr_ &= ~type_attr_t::ptr;
        return copy;
    }

    type_t with_elems(int new_elems) const {
        type_t copy = *this;
        copy.elems_ = new_elems;
        return copy;
    }

    type_t with_ptr() const {
        type_t copy = *this;
        copy.attr_ |= type_attr_t::ptr;
        return copy;
    }

    type_t with_attr(type_attr_t attr) const {
        type_t copy = *this;
        copy.attr_ = attr;
        return copy;
    }

    type_t simd() const {
        type_t copy = *this;
        copy.attr_ |= type_attr_t::simd;
        return copy;
    }

    type_t slm() const {
        type_t copy = *this;
        copy.attr_ |= type_attr_t::slm;
        return copy;
    }

    type_t scalar() const { return with_elems(1); }

    // Returns size in bytes.
    int size() const;

    // Returns size in bits.
    int bitsize() const {
        // 8 elements occupy the same number of bytes that a single element
        // occupies in bits.
        constexpr int bits_per_byte = 8;
        return with_elems(bits_per_byte * elems()).size();
    }

    // Returns number of elements that fit in `size()` bytes.
    // The size in bytes of `n` packed elements is
    //     `div_up(n * size(), packing())`.
    int packing() const {
        constexpr int bits_per_byte = 8;
        return bits_per_byte * size() / bitsize();
    }

    std::string str() const {
        ostringstream_t oss;
        oss << to_string(kind());
        if (elems() > 1) oss << "x" << elems();
        if (is_ptr()) oss << "*";
        if (is_mutable()) oss << ".mut";
        if (is_slm()) oss << ".slm";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    type_kind_t kind_ = type_kind_t::undef;
    int elems_ = 0;
    type_attr_t attr_ = type_attr_t::undef;
};

// type_t to dnnl_data_type_t convertor.
data_type_t to_dnnl(const type_t &type);

// Reference counter for IR objects.
class ref_count_t {
public:
    ref_count_t() : value_(0) {}
    ref_count_t(const ref_count_t &) = delete;
    ref_count_t &operator=(const ref_count_t &) = delete;
    ~ref_count_t() = default;

    uint32_t increment() { return ++value_; }
    uint32_t decrement() { return --value_; }

private:
    uint32_t value_;
};

// Forward Declare IR objects
class object_t;
class expr_impl_t;
class stmt_impl_t;
class ir_mutator_t;
class ir_visitor_t;

enum class ir_type_id_t : uint8_t {
    undef = 0,

#define HANDLE_IR_OBJECT(type) type,

    HANDLE_CORE_IR_OBJECTS()

#undef HANDLE_IR_OBJECT
};

// clang-tidy doesn't like the semicolon next to the class name.
#define CLASS_DECLARATION(name) class name
#define HANDLE_IR_OBJECT(type) CLASS_DECLARATION(type);
HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
#undef CLASS_DECLARATION

template <typename T, typename = void>
struct is_expr_t {
    static const bool value = false;
};

template <typename T>
struct is_expr_t<T,
        typename std::enable_if<std::is_base_of<expr_impl_t, T>::value>::type> {
    static const bool value = true;
};

template <typename T, typename = void>
struct is_stmt_t {
    static const bool value = false;
};

template <typename T>
struct is_stmt_t<T,
        typename std::enable_if<std::is_base_of<stmt_impl_t, T>::value>::type> {
    static const bool value = true;
};

struct type_info_t {
    constexpr type_info_t(ir_type_id_t type_id, const std::type_info &info,
            bool is_expr, bool is_stmt)
        : type_id(type_id), info(&info), is_expr(is_expr), is_stmt(is_stmt) {}

    ir_type_id_t type_id = ir_type_id_t::undef;
    const std::type_info *info = nullptr;
    bool is_expr = false;
    bool is_stmt = false;

    bool operator==(const type_info_t &other) const {
        if (type_id != ir_type_id_t::undef
                || other.type_id != ir_type_id_t::undef)
            return type_id == other.type_id;
        return (info == other.info) || (*info == *other.info);
    }

    bool operator!=(const type_info_t &other) const {
        return !operator==(other);
    }
};

// Base class for all IR objects. Implemented as an intrusive pointer, with
// the reference counter stored inside the object.
class object_impl_t {
public:
    object_impl_t(type_info_t type_info) : type_info_(type_info) {};

    object_impl_t(const object_impl_t &) = delete;
    object_impl_t &operator=(const object_impl_t &) = delete;

    virtual ~object_impl_t() = default;

    ref_count_t &ref_count() { return ref_count_; }

    // Provides equality semantics.
    virtual bool is_equal(const object_impl_t &obj) const = 0;

    virtual size_t get_hash() const = 0;

    // Type information.
    const type_info_t &type_info() const { return type_info_; };
    bool is_expr() const { return type_info().is_expr; }
    bool is_stmt() const { return type_info().is_stmt; }

    // Downcasts the object to the IR type, returns a reference. The IR type
    // must match the real IR type.
    // N.B.: this can potentially be dangerous if applied to non-const objects,
    //       since assigning a different value to the source object might make
    //       the reference dangling due to the destruction of the former object;
    //       please only call this method on non-const objects if absolutely
    //       necessary, and please don't add a non-const variant of the method!
    template <typename T>
    const T &as() const {
        gpu_assert(is<T>());
        return *as_ptr<T>(); // fails on incorrect casts even in Release
    }

    // Downcasts the object to the IR type, returns a pointer. If the IR type
    // doesn't match the real IR type, returns nullptr.
    // N.B.: this can potentially be dangerous if applied to non-const objects,
    //       since assigning a different value to the source object might make
    //       the reference dangling due to the destruction of the former object;
    //       please only call this method on non-const objects if absolutely
    //       necessary, and please don't add a non-const variant of the method!
    template <typename T>
    const T *as_ptr() const {
        return (is<T>()) ? (const T *)this : nullptr;
    }

    // Returns true if T matches the real IR type.
    template <typename T>
    bool is() const {
        return type_info() == T::_type_info();
    }

    virtual std::string str() const;

    virtual object_t _mutate(ir_mutator_t &mutator) const;
    virtual void _visit(ir_visitor_t &visitor) const;
    IR_DEFINE_DUMP()

private:
    ref_count_t ref_count_;
    type_info_t type_info_;
};

// Base wrapper for IR objects.
class object_t {
public:
    object_t(object_impl_t *impl = nullptr) : impl_(impl) {
        increment(impl_);
#ifdef SANITY_CHECK
        sanity_check();
#endif
    }
    object_t(const object_impl_t &impl)
        : object_t(const_cast<object_impl_t *>(&impl)) {}
    object_t(const object_impl_t *impl)
        : object_t(const_cast<object_impl_t *>(impl)) {}
    object_t(const object_t &obj) : object_t(obj.impl()) {}
    object_t(object_t &&obj) : impl_(obj.impl_) {
        obj.impl_ = nullptr;
#ifdef SANITY_CHECK
        sanity_check();
#endif
    }

#ifdef SANITY_CHECK
    virtual ~object_t() { decrement_and_maybe_destroy(impl_); }
#else
    ~object_t() { decrement_and_maybe_destroy(impl_); }
#endif

    object_t &operator=(const object_t &other) {
        if (&other == this) return *this;
        auto *other_impl = other.impl();
        increment(other_impl);
        decrement_and_maybe_destroy(impl_);
        impl_ = other_impl;
#ifdef SANITY_CHECK
        sanity_check();
#endif
        return *this;
    }

    object_t &operator=(object_t &&other) {
        std::swap(impl_, other.impl_);
#ifdef SANITY_CHECK
        sanity_check();
#endif
        return *this;
    }

    object_impl_t *impl() const { return impl_; }

    bool is_empty() const { return !impl_; }

    explicit operator bool() const { return !is_empty(); }

    const type_info_t &type_info() const { return impl_->type_info(); }

    template <typename T>
    const T &as() const {
        gpu_assert(impl_);
        return impl_->as<T>();
    }

    template <typename T>
    const T *as_ptr() const {
        if (!impl_) return nullptr;
        return impl_->as_ptr<T>();
    }

    template <typename T>
    bool is() const {
        if (is_empty()) return false;
        return impl_->is<T>();
    }

    // Comparison with identity semantics.
    bool is_same(const object_t &other) const { return impl_ == other.impl(); }

    // Comparison with equality semantics.
    bool is_equal(const object_t &other) const {
        if (is_empty() || other.is_empty())
            return is_empty() == other.is_empty();

        return impl_->is_equal(*other.impl());
    }

    size_t get_hash() const {
        if (is_empty()) return 0;
        return impl()->get_hash();
    }

    bool is_expr() const { return impl_ && impl_->is_expr(); }
    bool is_stmt() const { return impl_ && impl_->is_stmt(); }

    std::string str() const {
        if (is_empty()) return "(nil)";
        return impl()->str();
    }

    IR_DEFINE_DUMP()

protected:
#ifdef SANITY_CHECK
    virtual void sanity_check() const {}
#endif

private:
    static void increment(object_impl_t *impl) {
        if (!impl) return;
        impl->ref_count().increment();
    }

    static void decrement_and_maybe_destroy(object_impl_t *impl) {
        if (!impl) return;
        if (impl->ref_count().decrement() == 0) { delete impl; }
    }

    object_impl_t *impl_;
};

// Helper classes for containers to store object_t.
struct object_id_hash_t {
    size_t operator()(const object_t &obj) const {
        return std::hash<const object_impl_t *>()(obj.impl());
    }
};

struct object_eq_hash_t {
    size_t operator()(const object_t &obj) const { return obj.get_hash(); }
};

struct object_id_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_same(b);
    }
};

struct object_eq_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_equal(b);
    }
};

// Containers to store object_t.

// Unordered set, uses identity comparison for keys.
template <typename KeyT>
using object_set_t
        = std::unordered_set<KeyT, object_id_hash_t, object_id_equal_t>;

// Unordered set, uses equality comparison for keys.
template <typename KeyT>
using object_eq_set_t
        = std::unordered_set<KeyT, object_eq_hash_t, object_eq_equal_t>;

// Unordered map, uses identity comparison for keys.
template <typename KeyT, typename ValueT>
using object_map_t
        = std::unordered_map<KeyT, ValueT, object_id_hash_t, object_id_equal_t>;

// Unordered map, uses equality comparison for keys.
template <typename KeyT, typename ValueT>
using object_eq_map_t
        = std::unordered_map<KeyT, ValueT, object_eq_hash_t, object_eq_equal_t>;

// Helper class to mutate IR tree.
class ir_mutator_t {
public:
    virtual ~ir_mutator_t() = default;

    object_t mutate(const object_t &obj) {
        auto impl = obj.impl();
        if (!impl) return impl;
        return impl->_mutate(*this);
    }

    template <typename T>
    std::vector<T> mutate(const std::vector<T> &v) {
        std::vector<T> new_v;
        new_v.reserve(v.size());
        for (auto &e : v)
            new_v.push_back(mutate(e));
        return new_v;
    }

    // To catch missing _mutate() handlers in ir_mutator_t.
    object_t _mutate(const object_impl_t &obj) {
        gpu_error_not_expected() << "Can't handle type: " << object_t(&obj);
        return {};
    }

#define HANDLE_IR_OBJECT(type) virtual object_t _mutate(const type &obj);
    HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
};

// Helper class to walk through IR tree.
class ir_visitor_t {
public:
    virtual ~ir_visitor_t() = default;

    void visit(const object_t &obj) {
        const object_impl_t *impl = obj.impl();
        if (impl) {
            pre_visit(*impl);
            impl->_visit(*this);
            post_visit(*impl);
        };
    }

    template <typename T>
    void visit(const std::vector<T> &v) {
        for (auto &e : v)
            visit(e);
    }

    virtual void pre_visit(const object_impl_t &obj) {}
    virtual void post_visit(const object_impl_t &obj) {}

    // To catch missing _visit() handlers in ir_visitor_t.
    void _visit(const object_impl_t &obj) {
        gpu_error_not_expected() << "Can't handle type: " << object_t(obj);
    }

#define HANDLE_IR_OBJECT(type) virtual void _visit(const type &obj);
    HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
};

// Base class for IR expression objects.
class expr_impl_t : public object_impl_t {
public:
    expr_impl_t(type_info_t type_info, const type_t &type)
        : object_impl_t(type_info), type(type) {}

    type_t type;
};

// Wrapper for IR expression objects.
class expr_t : public object_t {
public:
    using object_t::object_t;

    expr_t() = default;
    expr_t(const object_t &obj) : object_t(obj) {}
    expr_t(object_t &&obj) : object_t(obj) {}
    expr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    expr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    explicit expr_t(bool v);
    expr_t(float v);
    expr_t(double v);
    expr_t(int16_t v);
    expr_t(int32_t v);
    expr_t(int64_t v);
    expr_t(uint16_t v);
    expr_t(uint32_t v);
    expr_t(uint64_t v);

    const type_t &type() const {
        gpu_assert(!is_empty());
        return ((const expr_impl_t *)impl())->type;
    }

#define DECLARE_BINARY_ASSIGN_OPERATOR(op) \
    expr_t &operator op##=(const expr_t &rhs);

    DECLARE_BINARY_ASSIGN_OPERATOR(+)
    DECLARE_BINARY_ASSIGN_OPERATOR(-)
    DECLARE_BINARY_ASSIGN_OPERATOR(*)
    DECLARE_BINARY_ASSIGN_OPERATOR(/)
    DECLARE_BINARY_ASSIGN_OPERATOR(%)
    DECLARE_BINARY_ASSIGN_OPERATOR(&)

#undef DECLARE_BINARY_ASSIGN_OPERATOR

    // Returns a pointer shifted by `off` bytes relative to this pointer. The
    // base expression must be a pointer.
    expr_t operator[](const expr_t &off) const;

private:
#ifdef SANITY_CHECK
    void sanity_check() const override {
        gpu_assert(dynamic_cast<const expr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
#endif
};

// Helper functions.
inline bool is_const(const expr_t &e);
inline bool is_const(const expr_t &e, int value);
inline bool is_var(const expr_t &e);
inline bool is_ref(const expr_t &e);
inline bool all_of(const expr_t &e, const expr_t &value);
inline bool is_zero(const expr_t &e) {
    return is_const(e, 0);
}
inline bool is_one(const expr_t &e) {
    return is_const(e, 1);
}
inline bool is_minus_one(const expr_t &e) {
    return is_const(e, -1);
}

// Unary and binary operators.
enum class op_kind_t {
    undef,

    _minus,
    _add,
    _sub,
    _mul,
    _div,
    _mod,
    _shl,
    _shr,
    _min,
    _max,

    _lt,
    _le,
    _gt,
    _ge,
    _ne,
    _eq,

    _and,
    _or,

    // Ternary operations.
    // Parametric ReLU.
    // if (a > 0) op = a
    // else       op = a * b
    _prelu,
    // Ternary add.
    // op = a + b + c
    _add3,
    // Multiply-accumulate.
    // op = a + b * c
    _mad,
    // Integer division by a constant with rounding up.
    // op = (a + b - 1) / b
    _div_up,
    // Integer division by a non-constant (rounding down behavior).
    // if (a % b < 0) op = a / b - 1
    // else           op = a / b
    // This is ternary operation, c is a pre-computed value.
    _idiv,
    // Integer modulus by a non-constant (rounding down behavior).
    // if (a % b < 0) op = a % b + b
    // else           op = a % b
    // This is ternary operation, c is a pre-computed value.
    _imod,
};

std::string to_string(op_kind_t kind);

inline std::ostream &operator<<(std::ostream &out, op_kind_t kind) {
    out << to_string(kind);
    return out;
}

bool is_cmp_op(op_kind_t op_kind);

bool is_commutative_op(op_kind_t op_kind);

op_kind_t negate_cmp_op(op_kind_t op_kind);

type_t unary_op_type(op_kind_t op_kind, const expr_t &a);

type_t common_int_type(const type_t &_a, const type_t &_b);

type_t common_type(const type_t &a, const type_t &b);

type_t common_type(const expr_t &a, const expr_t &b);

type_t binary_op_type(op_kind_t op_kind, const expr_t &a, const expr_t &b);

type_t ternary_op_type(
        op_kind_t op_kind, const expr_t &a, const expr_t &b, const expr_t &c);

type_t nary_op_type(op_kind_t op_kind, const std::vector<expr_t> &args);

// Binary operation: (a op b).
class binary_op_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(binary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
        return expr_t(new binary_op_t(op_kind, a, b));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (op_kind == other.op_kind)
                && ((a.is_equal(other.a) && b.is_equal(other.b))
                        || (is_commutative_op(op_kind) && b.is_equal(other.a)
                                && a.is_equal(other.b)));
    }

    size_t get_hash() const override {
        if (is_commutative_op(op_kind)) {
            size_t a_hash = ir_utils::get_hash(a);
            size_t b_hash = ir_utils::get_hash(b);
            return ir_utils::get_hash(op_kind, a_hash ^ b_hash);
        }
        return ir_utils::get_hash(op_kind, a, b);
    }

    IR_DECLARE_TRAVERSERS()

    op_kind_t op_kind;
    expr_t a;
    expr_t b;

private:
    binary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b)
        : expr_impl_t(_type_info(), binary_op_type(op_kind, a, b))
        , op_kind(op_kind)
        , a(a)
        , b(b) {}
};

// Boolean immediate value.
class bool_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_CORE_TYPE(bool_imm_t)

    static expr_t make(bool value) { return expr_t(new bool_imm_t(value)); }

    static type_t get_packed_type(int elems) {
        return type_t::u(std::max(elems, 16));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return value == other.value;
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    IR_DECLARE_TRAVERSERS()

    bool value;

private:
    bool_imm_t(bool value)
        : expr_impl_t(_type_info(), type_t::_bool()), value(value) {}
};

// Cast between data types. In general conversion follows the C++ casting
// rules. Several modes/scenarios are supported:
// - Cast with saturation: cast(T, e) = max(T_min, min(T_max, e))
//   By default saturation is disabled and any underflow/overflow is unhandled.
// - Bitwise cast from bool vector to u16 (boolxN -> u16, 2 <= N <= 16):
//   In this case the lower N bits of the resulting value are initialized based
//   on the boolean elements. The upper (16 - N) bits are uninitialized.
class cast_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(cast_t)

    static expr_t make(
            const type_t &type, const expr_t &expr, bool saturate = false) {
        if (expr.type() == type) return expr;
        if (!saturate) {
            auto *expr_cast = expr.as_ptr<cast_t>();
            if (expr_cast && !expr_cast->saturate
                    && type == expr_cast->expr.type())
                return expr_cast->expr;
        }
        return expr_t(new cast_t(type, expr, saturate));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return type.is_equal(other.type) && expr.is_equal(other.expr)
                && (saturate == other.saturate);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(type, expr, saturate);
    }

    bool is_bool_vec_u16() const {
        if (is_bool_vec(expr.type()) && is_u16_or_u32_scalar(type)) return true;
        if (is_bool_vec(type) && is_u16_or_u32_scalar(expr.type())) return true;
        return false;
    }

    IR_DECLARE_TRAVERSERS()

    expr_t expr;
    bool saturate;

private:
    cast_t(const type_t &type, const expr_t &expr, bool saturate)
        : expr_impl_t(_type_info(), type), expr(expr), saturate(saturate) {
        if (!is_bool_vec_u16()) {
            gpu_assert(type.elems() == expr.type().elems())
                    << "Number of elements must match.";
        }
    }

    static bool is_bool_vec(const type_t &type) {
        return type.is_bool() && type.elems() > 1;
    }

    static bool is_u16_or_u32_scalar(const type_t &type) {
        return (type.is_u16() || type.is_u32()) && type.is_scalar();
    }
};

// Constant variable, used as a coefficient in a linear expression.
class const_var_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(const_var_t)

    static expr_t make(const type_t &type, const std::string &name) {
        return expr_t(new const_var_t(type, name));
    }

    bool is_equal(const object_impl_t &obj) const override {
        // Do not allow variable cloning.
        return this == &obj;
    }

    size_t get_hash() const override { return ir_utils::get_hash(name); }

    IR_DECLARE_TRAVERSERS()

    std::string name;

private:
    const_var_t(const type_t &type, const std::string &name)
        : expr_impl_t(_type_info(), type), name(name) {}
};

// Floating-point immediate value.
class float_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_CORE_TYPE(float_imm_t)

    static expr_t make(double value, const type_t &type = type_t::undef()) {
        return expr_t(new float_imm_t(value, type));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return type.is_equal(other.type) && (value == other.value);
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    IR_DECLARE_TRAVERSERS()

    double value;

private:
    float_imm_t(double value, const type_t &type = type_t::undef())
        : expr_impl_t(_type_info(), type.is_undef() ? type_t::f32() : type)
        , value(value) {}
};

// Integer immediate value.
class int_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_CORE_TYPE(int_imm_t);

    template <typename T>
    static expr_t make(T value, const type_t &type = type_t::undef()) {
        return expr_t(new int_imm_t(value, type));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return type.is_equal(other.type) && (value == other.value);
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    static expr_t shrink_type(const expr_t &e) {
        auto &imm = e.as<int_imm_t>();
        type_t new_type = shrink_type(imm.value);
        if (new_type == imm.type) return e;
        return make(imm.value, new_type);
    }

    template <typename T>
    static bool try_shrink_type(int64_t v) {
        if ((v >= 0 && (uint64_t)v <= (uint64_t)std::numeric_limits<T>::max())
                || (v < 0
                        && (int64_t)v
                                >= (int64_t)std::numeric_limits<T>::min()))
            return true;
        return false;
    }

    IR_DECLARE_TRAVERSERS()

    int64_t value;

private:
    int_imm_t(int64_t value, const type_t &type = type_t::undef())
        : expr_impl_t(_type_info(), type.is_undef() ? shrink_type(value) : type)
        , value(value) {}

    static type_t shrink_type(int64_t v) {
        if (try_shrink_type<int32_t>(v)) return type_t::s32();
        return type_t::s64();
    }
};

// Immediate if or the conditional (ternary) operator.
// C++ equivalent: (cond ? true_expr : false_expr).
class iif_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(iif_t);

    static expr_t make(const expr_t &cond, const expr_t &true_expr,
            const expr_t &false_expr) {
        return expr_t(new iif_t(cond, true_expr, false_expr));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return cond.is_equal(other.cond) && true_expr.is_equal(other.true_expr)
                && false_expr.is_equal(other.false_expr);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(cond, true_expr, false_expr);
    }

    IR_DECLARE_TRAVERSERS()

    expr_t cond;
    expr_t true_expr;
    expr_t false_expr;

private:
    iif_t(const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr)
        : expr_impl_t(
                _type_info(), common_type(true_expr.type(), false_expr.type()))
        , cond(cond)
        , true_expr(true_expr)
        , false_expr(false_expr) {}
};

// Linear combination expression:
//   u[0] * v[0] + u[1] * v[1] + ... u[n - 1] * v[n - 1] + c,
// where:
// - c/u[i] is either an integer immediate (int_imm_t) or a constant variable
//  (const_var_t)
// - v[i] is a non-constant variable (var_t)
class linear_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(linear_t)
    static expr_t make(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec) {
        return expr_t(new linear_t(c, u_vec, v_vec));
    }
    static expr_t make(const expr_t &c) { return make(c, {}, {}); }
    static expr_t make(const expr_t &c, const std::vector<expr_t> &v_vec) {
        std::vector<expr_t> ones(v_vec.size(), expr_t(1));
        return make(c, ones, v_vec);
    }
    static expr_t to_expr(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec) {
        auto e = linear_t::make(c, u_vec, v_vec);
        return e.as<linear_t>().to_expr();
    }
    int nargs() const { return int(v_vec.size()); }
    expr_t to_expr() const;

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return c.is_equal(other.c) && ir_utils::is_equal(u_vec, other.u_vec)
                && ir_utils::is_equal(v_vec, other.v_vec);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(c, u_vec, v_vec);
    }

    IR_DECLARE_TRAVERSERS()

    expr_t c;
    std::vector<expr_t> u_vec;
    std::vector<expr_t> v_vec;

private:
    linear_t(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec)
        : expr_impl_t(_type_info(), type_t::s32())
        , c(c)
        , u_vec(u_vec)
        , v_vec(v_vec) {}
};

// Updates `base_expr` and `off` so that after return:
// - base_expr contains a variable of a pointer type
// - off contains an offset
void normalize_ptr(const type_t &type, expr_t &base, expr_t &off);

// Load from a GRF buffer.
// C++ equivalent (when type is scalar):
//     load = *(type *)(&buf[off]);
// C++ equivalent (when type is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         load[i] = *(scalar_type *)(&buf[off + i * _stride]);
//     }
class load_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(load_t)

    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of type.scalar() elements).
    static expr_t make(const type_t &type, const expr_t &buf, const expr_t &off,
            int stride = default_stride) {
        return expr_t(new load_t(type, buf, off, stride));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return type.is_equal(other.type) && buf.is_equal(other.buf)
                && off.is_equal(other.off) && (stride == other.stride);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(type, buf, off, stride);
    }

    bool has_default_stride() const { return stride == default_stride; }

    IR_DECLARE_TRAVERSERS()

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    int stride;

private:
    load_t(const type_t &_type, const expr_t &_buf, const expr_t &_off,
            int _stride)
        : expr_impl_t(_type_info(), _type)
        , buf(_buf)
        , off(_off)
        , stride(_stride) {
        normalize_ptr(type, buf, off);
        gpu_assert(is_var(buf) || is_ref(buf)) << buf;
        if (stride == type.scalar().size()) stride = default_stride;
    }
};

// Pointer expression: (base_ptr + off).
class ptr_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(ptr_t)

    // off - offset in bytes.
    static expr_t make(const expr_t &base, const expr_t &off) {
        return expr_t(new ptr_t(base, off));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return base.is_equal(other.base) && off.is_equal(other.off);
    }

    size_t get_hash() const override { return ir_utils::get_hash(base, off); }

    // Normalizes (base op off) pointer so that the new base is a variable and
    // off is an offset expression.
    // Example:
    //     Before call: base = (base0 + off0), off = off1
    //     After call:  base = base0, off = off0 + off1
    static void normalize(
            expr_t &base, expr_t &off, op_kind_t op_kind = op_kind_t::_add);

    IR_DECLARE_TRAVERSERS()

    expr_t base;
    expr_t off;

private:
    ptr_t(const expr_t &base, const expr_t &off)
        : expr_impl_t(_type_info(), base.type()), base(base), off(off) {
        normalize(this->base, this->off);
    }
};

inline const expr_t &get_base(const expr_t &e) {
    if (e.is_empty()) return e;
    if (e.is<var_t>()) return e;
    if (e.is<ptr_t>()) return e.as<ptr_t>().base;
    gpu_error_not_expected() << e;
    return e;
}

class shuffle_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(shuffle_t)

    static expr_t make(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        if (idx.size() == 1) return vec[idx[0]];
        return expr_t(new shuffle_t(vec, idx));
    }

    static expr_t make(
            const std::vector<expr_t> &_vec, bool find_equal = true) {
        std::vector<expr_t> vec;
        std::vector<int> idx;
        for (auto &v : _vec) {
            bool found = false;
            int size = int(vec.size());
            if (find_equal) {
                for (int i = 0; i < size; i++) {
                    if (v.is_equal(vec[i])) {
                        idx.push_back(i);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                vec.push_back(v);
                idx.push_back(size);
            }
        }
        return make(vec, idx);
    }

    static expr_t make_broadcast(const expr_t &expr, dim_t elems) {
        if (elems == 1) return expr;
        gpu_assert(expr.type().is_scalar()) << expr;
        return make({expr}, std::vector<int>(elems, 0));
    }

    // Slices the existing shuffle expression. For inputs (S, beg, end) returns
    // (S[beg], S[beg + 1], ..., S[end - 1]) vector.
    static expr_t make(const expr_t &_shuffle, int beg, int end) {
        auto &shuffle = _shuffle.as<shuffle_t>();
        gpu_assert(beg >= 0 && beg <= shuffle.elems());
        gpu_assert(end >= 0 && end <= shuffle.elems());
        gpu_assert(beg < end);
        std::vector<expr_t> vec;
        std::vector<int> idx(end - beg, -1);
        for (int i = beg; i < end; i++) {
            if (idx[i - beg] != -1) continue;
            int old_idx = shuffle.idx[i];
            vec.push_back(shuffle.vec[old_idx]);
            for (int j = i; j < end; j++) {
                if (shuffle.idx[j] == old_idx)
                    idx[j - beg] = int(vec.size()) - 1;
            }
        }
        return make(vec, idx);
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return ir_utils::is_equal(vec, other.vec)
                && ir_utils::is_equal(idx, other.idx);
    }

    size_t get_hash() const override { return ir_utils::get_hash(vec, idx); }

    int elems() const { return int(idx.size()); }

    bool is_vector() const {
        for (int i = 0; i < elems(); i++)
            if (idx[i] != i) return false;
        return true;
    }

    bool is_broadcast() const { return vec.size() == 1; }

    IR_DECLARE_TRAVERSERS()

    std::vector<expr_t> vec;
    std::vector<int> idx;

private:
    shuffle_t(const std::vector<expr_t> &vec, const std::vector<int> &idx)
        : expr_impl_t(_type_info(), shuffle_type(vec, idx))
        , vec(vec)
        , idx(idx) {
        gpu_assert(idx.size() > 1) << "Unexpected empty or scalar shuffle.";
    }

    static type_t shuffle_type(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        gpu_assert(!vec.empty() && !idx.empty());

        auto elem_type = vec[0].type();
        for (auto &v : vec)
            elem_type = common_type(elem_type, v.type());

        for (size_t i = 0; i < idx.size(); i++) {
            gpu_assert(idx[i] >= 0 && idx[i] < int(vec.size()))
                    << "Incorrect index.";
            MAYBE_UNUSED(i);
        }

        int elems = int(idx.size());
        return elem_type.with_elems(elems);
    }
};

// Ternary operation: op(a, b, c).
class ternary_op_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(ternary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c) {
        return expr_t(new ternary_op_t(op_kind, a, b, c));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (op_kind == other.op_kind) && a.is_equal(other.a)
                && b.is_equal(other.b) && c.is_equal(other.c);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(op_kind, a, b, c);
    }

    IR_DECLARE_TRAVERSERS()

    op_kind_t op_kind;
    expr_t a;
    expr_t b;
    expr_t c;

private:
    ternary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c)
        : expr_impl_t(_type_info(), ternary_op_type(op_kind, a, b, c))
        , op_kind(op_kind)
        , a(a)
        , b(b)
        , c(c) {}
};

inline expr_t ternary_mad(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_mad, a, b, c);
}

inline expr_t ternary_add3(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_add3, a, b, c);
}

inline expr_t ternary_idiv(
        const expr_t &a, const expr_t &b, const expr_t &magic) {
    return ternary_op_t::make(op_kind_t::_idiv, a, b, magic);
}

// Unary operation: (op a).
class unary_op_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(unary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a) {
        return expr_t(new unary_op_t(op_kind, a));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (op_kind == other.op_kind) && a.is_equal(other.a);
    }

    size_t get_hash() const override { return ir_utils::get_hash(op_kind, a); }

    IR_DECLARE_TRAVERSERS()

    op_kind_t op_kind;
    expr_t a;

private:
    unary_op_t(op_kind_t op_kind, const expr_t &a)
        : expr_impl_t(_type_info(), unary_op_type(op_kind, a))
        , op_kind(op_kind)
        , a(a) {}
};

class var_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(var_t)

    static expr_t make(const type_t &type, const std::string &name,
            bool is_mutable = false) {
        return expr_t(new var_t(type, name, is_mutable));
    }

    bool is_equal(const object_impl_t &obj) const override {
        // Do not allow variable cloning.
        return this == &obj;
    }

    size_t get_hash() const override { return ir_utils::get_hash(name); }

    IR_DECLARE_TRAVERSERS()

    std::string name;
    bool is_mutable = false;

private:
    var_t(const type_t &type, const std::string &name, bool is_mutable)
        : expr_impl_t(_type_info(), type), name(name), is_mutable(is_mutable) {}
};

// Index into a buffer
// off is offset in number of elements
// elems is number of consecutive elements to access starting from off
// off and elems must be GRF aligned
class ref_t : public expr_impl_t {
public:
    IR_DECL_CORE_TYPE(ref_t)

    static expr_t make(const expr_t &var, int off, int elems) {
        return expr_t(new ref_t(var, off, elems));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return other.var.is_equal(var) && other.off == off
                && other.elems == elems;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << var.str() << "[" << off;
        if (elems > 1) oss << ":" << off + elems;
        oss << "]";
        return oss.str();
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(var, off, elems);
    }

    IR_DECLARE_TRAVERSERS()

    expr_t var;
    int off;
    int elems;

private:
    ref_t(const expr_t &var, int off, int elems)
        : expr_impl_t(_type_info(), var.type().with_elems(elems))
        , var(var)
        , off(off)
        , elems(elems) {}
};

// Convertor from C++ type to IR expression.
template <typename T>
expr_t to_expr(T value, const type_t &type) {
#define CASE(ir_type, cpp_type) \
    if (type == type_t::ir_type()) return expr_t(static_cast<cpp_type>(value))

    CASE(_bool, bool);
    CASE(bf16, bfloat16_t);
    CASE(f16, float16_t);
    CASE(f32, float);
    CASE(f64, double);
    CASE(s16, int16_t);
    CASE(s32, int32_t);
    CASE(s64, int64_t);
    CASE(u16, uint16_t);
    CASE(u32, uint32_t);
    CASE(u64, uint64_t);

#undef CASE

    gpu_error_not_expected() << type;

    return expr_t();
}

template <typename T>
expr_t to_expr(T value) {
    return to_expr(value, type_t::from_cpp<T>());
}

inline bool is_binary_op(const expr_t &e) {
    return e.is<binary_op_t>();
}

inline bool is_binary_op(const expr_t &e, op_kind_t op_kind) {
    if (!is_binary_op(e)) return false;
    return e.as<binary_op_t>().op_kind == op_kind;
}

inline bool is_binary_cmp_op(const expr_t &e) {
    if (!is_binary_op(e)) return false;
    return is_cmp_op(e.as<binary_op_t>().op_kind);
}

inline bool is_const(const expr_t &e) {
    return e.is<bool_imm_t>() || e.is<int_imm_t>() || e.is<float_imm_t>();
}

inline bool is_const(const expr_t &e, int value) {
    if (!is_const(e)) return false;
    return e.is_equal(to_expr(value, e.type()));
}

inline bool all_of(const expr_t &e, const expr_t &value) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return e.is_equal(value);
    for (auto &i : shuffle->idx) {
        if (!shuffle->vec[i].is_equal(value)) return false;
    }
    return true;
}

inline bool is_shuffle_const(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return false;
    for (auto &v : shuffle->vec)
        if (!is_const(v)) return false;
    return true;
}

inline bool is_var(const expr_t &e) {
    return e.is<var_t>();
}

inline bool is_ref(const expr_t &e) {
    return e.is<ref_t>();
}

// Convertor from IR expression to C++ constant.
template <typename T>
T to_cpp(const expr_t &e) {
    gpu_assert(is_const(e)) << "Expression must be constant.";

    if (e.is<int_imm_t>()) return (T)e.as<int_imm_t>().value;
    if (e.is<float_imm_t>()) return (T)e.as<float_imm_t>().value;
    if (e.is<bool_imm_t>()) return (T)e.as<bool_imm_t>().value;

    gpu_error_not_expected();
    return 0;
}

inline int to_int(const expr_t &e) {
    return to_cpp<int>(e);
}

expr_t operator-(const expr_t &a);
expr_t div_up(const expr_t &a, const expr_t &b);

#define DECLARE_BINARY_OPERATOR(op, op_kind) \
    expr_t operator op(const expr_t &a, const expr_t &b);

DECLARE_BINARY_OPERATOR(+, op_kind_t::_add)
DECLARE_BINARY_OPERATOR(-, op_kind_t::_sub)
DECLARE_BINARY_OPERATOR(*, op_kind_t::_mul)
DECLARE_BINARY_OPERATOR(/, op_kind_t::_div)
DECLARE_BINARY_OPERATOR(%, op_kind_t::_mod)
DECLARE_BINARY_OPERATOR(<<, op_kind_t::_shl)
DECLARE_BINARY_OPERATOR(>>, op_kind_t::_shr)

DECLARE_BINARY_OPERATOR(==, op_kind_t::_eq)
DECLARE_BINARY_OPERATOR(!=, op_kind_t::_ne)
DECLARE_BINARY_OPERATOR(>, op_kind_t::_gt)
DECLARE_BINARY_OPERATOR(>=, op_kind_t::_ge)
DECLARE_BINARY_OPERATOR(<, op_kind_t::_lt)
DECLARE_BINARY_OPERATOR(<=, op_kind_t::_le)

DECLARE_BINARY_OPERATOR(&, op_kind_t::_and)
DECLARE_BINARY_OPERATOR(|, op_kind_t::_or)

#undef DECLARE_BINARY_OPERATOR

// Returns a shifted pointer with base `a` (pointer) and offset `b` (in bytes).
// shift_ptr(op, a, b) returns &(a op b) in C++ terms (op is either addition or
// subtraction).
expr_t shift_ptr(op_kind_t op_kind, const expr_t &a, const expr_t &b);

// Base class for IR statement objects.
class stmt_impl_t : public object_impl_t {
public:
    stmt_impl_t(type_info_t type_info) : object_impl_t(type_info) {}
};

// Wrapper for IR statement objects.
class stmt_t : public object_t {
public:
    using object_t::object_t;

    stmt_t() = default;
    stmt_t(const object_t &obj) : object_t(obj) {}
    stmt_t(object_t &&obj) : object_t(std::move(obj)) {}
    stmt_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    stmt_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    stmt_t append(const stmt_t &s) const;

private:
#ifdef SANITY_CHECK
    void sanity_check() const override {
        gpu_assert(dynamic_cast<const stmt_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
#endif
};

enum class alloc_kind_t {
    undef,
    grf, // GRF - general register file.
    slm, // SLM - shared local memory.
    global, // Global memory.
};

class alloc_attr_impl_t : public object_impl_t {
public:
    alloc_attr_impl_t(type_info_t type_info) : object_impl_t(type_info) {}
};

class alloc_attr_t : public object_t {
public:
    using object_t::object_t;

    alloc_attr_t() = default;
    alloc_attr_t(const object_t &obj) : object_t(obj) {}
    alloc_attr_t(object_t &&obj) : object_t(obj) {}
    alloc_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    alloc_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

private:
#ifdef SANITY_CHECK
    void sanity_check() const override {
        gpu_assert(dynamic_cast<const alloc_attr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
#endif
};

class grf_permutation_t;

// Allocation attribute specifying permutation for a GRF buffer.
class grf_permute_attr_t : public alloc_attr_impl_t {
public:
    IR_DECL_TYPE(grf_permute_attr_t)

    static alloc_attr_t make(
            const std::shared_ptr<grf_permutation_t> &grf_perm) {
        return alloc_attr_t(new grf_permute_attr_t(grf_perm));
    }

    bool is_equal(const object_impl_t &obj) const override {
        return this == &obj;
    }

    size_t get_hash() const override { return 0; }

    std::shared_ptr<grf_permutation_t> grf_perm;

private:
    grf_permute_attr_t(const std::shared_ptr<grf_permutation_t> &grf_perm)
        : alloc_attr_impl_t(_type_info()), grf_perm(grf_perm) {}
};

// Allocation attribute to store extra information to avoid bank conflicts.
class bank_conflict_attr_t : public alloc_attr_impl_t {
public:
    IR_DECL_TYPE(bank_conflict_attr_t)

    static alloc_attr_t make(const std::vector<expr_t> &bufs,
            const std::vector<int> &buf_sizes,
            const std::vector<int> &buf_min_block_sizes,
            const std::vector<stmt_t> &instructions) {
        return alloc_attr_t(new bank_conflict_attr_t(
                bufs, buf_sizes, buf_min_block_sizes, instructions));
    }

    bool is_equal(const object_impl_t &obj) const override {
        return this == &obj;
    }

    size_t get_hash() const override { return ir_utils::get_hash(buf_sizes); }

    // List of buffers accessed from instructions.
    std::vector<expr_t> bufs;
    // Buffer sizes in bytes.
    std::vector<int> buf_sizes;
    // Minimum power-of-two block sizes for each buffer to avoid unhandled
    // cross-boundary accesses. A buffer may be allocated in fixed-size blocks
    // to avoid bank conflicts however the block size can't be arbitrary - we
    // need to avoid unhandled boundary crossings (e.g. in memory loads).
    std::vector<int> buf_min_block_sizes;
    // List of instructions whose bank conflicts are to be avoided.
    std::vector<stmt_t> instructions;

private:
    bank_conflict_attr_t(const std::vector<expr_t> &bufs,
            const std::vector<int> &buf_sizes,
            const std::vector<int> &buf_min_block_sizes,
            const std::vector<stmt_t> &instructions)
        : alloc_attr_impl_t(_type_info())
        , bufs(bufs)
        , buf_sizes(buf_sizes)
        , buf_min_block_sizes(buf_min_block_sizes)
        , instructions(instructions) {}
};

// Allocation for SLM and GRF buffers.
// C++ equivalent:
//     {
//         byte *buf = new byte[size];
//         body;
//      }
class alloc_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(alloc_t)

    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const std::vector<alloc_attr_t> &attrs, const stmt_t &body = {}) {
        return stmt_t(new alloc_t(buf, size, kind, attrs, body));
    }

    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const alloc_attr_t &attr, const stmt_t &body = {}) {
        std::vector<alloc_attr_t> attrs = {attr};
        return make(buf, size, kind, attrs, body);
    }

    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const stmt_t &body = {}) {
        return make(buf, size, kind, std::vector<alloc_attr_t>(), body);
    }

    static stmt_t make(const expr_t &buf, const stmt_t &body = {}) {
        return stmt_t(new alloc_t(buf, body));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return buf.is_equal(other.buf) && (size == other.size)
                && (kind == other.kind)
                && ir_utils::is_equal(attrs, other.attrs)
                && body.is_equal(other.body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(buf, size, kind, attrs, body);
    }

    template <typename T>
    bool has_attr() const {
        for (auto &a : attrs)
            if (a.is<T>()) return true;
        return false;
    }

    template <typename T>
    const T &get_attr() const {
        for (auto &a : attrs)
            if (a.is<T>()) return a.as<T>();
        gpu_error_not_expected() << "Can't find attribute.";
        return attrs[0].as<T>();
    }

    int register_alloc_size(int grf_size) const {
        return (kind == alloc_kind_t::grf)
                ? into<int>(utils::rnd_up(size, grf_size))
                : 0;
    }

    std::string line_str() const {
        ostringstream_t out;
        out << "alloc " << buf.as<var_t>().name << "[" << size << "]";
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    expr_t buf;
    uint32_t size;
    alloc_kind_t kind;
    std::vector<alloc_attr_t> attrs;
    stmt_t body;

private:
    alloc_t(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const std::vector<alloc_attr_t> &attrs, const stmt_t &body)
        : stmt_impl_t(_type_info())
        , buf(buf)
        , size(size)
        , kind(kind)
        , attrs(attrs)
        , body(body) {
        gpu_assert(buf.type().is_ptr()
                || into<uint32_t>(buf.type().size()) == size)
                << buf;
    }

    alloc_t(const expr_t &buf, const stmt_t &body)
        : stmt_impl_t(_type_info())
        , buf(buf)
        , size(buf.type().size())
        , kind(alloc_kind_t::grf)
        , body(body) {
        gpu_assert(!buf.type().is_ptr()) << buf;
    }
};

// Store to a GRF buffer.
// C++ equivalent (when value is scalar):
//     *(value_type *)(&buf[off]) = value;
// C++ equivalent (when value is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         *(scalar_type *)(&buf[off + i * _stride]) = value[i];
//     }
class store_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(store_t)

    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of value.type().scalar()
    // elements).
    static stmt_t make(const expr_t &buf, const expr_t &off,
            const expr_t &_value, int stride = default_stride,
            const expr_t &_mask = expr_t(), bool fill_mask0 = false) {
        auto mask = _mask;
        auto value = _value;
        if (mask) {
            if (all_of(mask, expr_t(true))) {
                mask = expr_t();
            } else if (all_of(mask, expr_t(false))) {
                // No need to store anything with a false mask,
                // unless explicitly asked to zero-fill the rest.
                if (!fill_mask0) return stmt_t();
                auto type = value.type();
                value = shuffle_t::make_broadcast(
                        cast_t::make(type.scalar(), 0), type.elems());
                mask = expr_t();
            }
        }
        return stmt_t(
                new store_t(buf, off, value, stride, mask, fill_mask0 && mask));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return buf.is_equal(other.buf) && off.is_equal(other.off)
                && value.is_equal(other.value) && mask.is_equal(other.mask)
                && (stride == other.stride) && (fill_mask0 == other.fill_mask0);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(buf, off, value, stride, mask, fill_mask0);
    }

    bool has_default_stride() const { return stride == default_stride; }

    std::string line_str() const {
        ostringstream_t out;
        out << load_t::make(value.type(), buf, off, stride);
        out << " = " << value;
        if (mask) {
            out << ", mask = " << mask.str();
            if (fill_mask0) out << " [FILL]";
        }
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    expr_t value;
    int stride;
    expr_t mask;
    bool fill_mask0;

private:
    store_t(const expr_t &_buf, const expr_t &_off, const expr_t &_value,
            int _stride, const expr_t &_mask, bool _fill_mask0)
        : stmt_impl_t(_type_info())
        , buf(_buf)
        , off(_off)
        , value(_value)
        , stride(_stride)
        , mask(_mask)
        , fill_mask0(_fill_mask0) {
        normalize_ptr(value.type(), buf, off);
        gpu_assert(is_var(buf) || is_ref(buf)) << buf;
        if (stride == value.type().scalar().size()) stride = default_stride;
        if (mask)
            gpu_assert(mask.type() == type_t::_bool(value.type().elems()));
    }
};

// Loop statement with unit increment.
// C++ equivalent:
//    for (var = init; var < bound; var++) {
//        body;
//    }
// unroll specifies the unroll factor, unroll = 1 means no unrolling.
class for_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(for_t)

    static stmt_t make(const expr_t &var, const expr_t &init,
            const expr_t &bound, const stmt_t &body = {},
            const expr_t &step = expr_t(1), int unroll = 1) {
        return stmt_t(new for_t(var, init, bound, body, step, unroll));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return var.is_equal(other.var) && init.is_equal(other.init)
                && bound.is_equal(other.bound) && body.is_equal(other.body)
                && step.is_equal(other.step) && (unroll == other.unroll);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(var, init, bound, body, step, unroll);
    }

    std::string line_str() const {
        ostringstream_t out;
        out << "for (" << var << " = " << init << "; " << var << " < " << bound
            << "; " << var << " += " << step << ") ";
        if (unroll != 1) out << "[unroll: " << unroll << "] ";
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    expr_t var;
    expr_t init;
    expr_t bound;
    stmt_t body;
    expr_t step;
    int unroll;

private:
    for_t(const expr_t &var, const expr_t &init, const expr_t &bound,
            const stmt_t &body, const expr_t &step, int unroll)
        : stmt_impl_t(_type_info())
        , var(var)
        , init(init)
        , bound(bound)
        , body(body)
        , step(step)
        , unroll(unroll) {}
};

// If-else statement.
// C++ equivalent:
//     if (cond) {
//         body;
//     } else {
//         else_body;
//     }
class if_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(if_t)

    static stmt_t make(const expr_t &cond, const stmt_t &body,
            const stmt_t &else_body = stmt_t()) {
        return stmt_t(new if_t(cond, body, else_body));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return cond.is_equal(other.cond) && body.is_equal(other.body)
                && else_body.is_equal(other.else_body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(cond, body, else_body);
    }

    std::string line_str() const {
        ostringstream_t oss;
        oss << "if (" << cond << ")";
        return oss.str();
    }

    IR_DECLARE_TRAVERSERS()

    expr_t cond;
    stmt_t body;
    stmt_t else_body;

private:
    if_t(const expr_t &cond, const stmt_t &body, const stmt_t &else_body)
        : stmt_impl_t(_type_info())
        , cond(cond)
        , body(body)
        , else_body(else_body) {}
};

// Let statement, used to bind a variable to a value within a scope.
// C++ equivalent:
//     {
//         var = value;
//         body;
//     }
class let_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(let_t)

    static stmt_t make(
            const expr_t &var, const expr_t &value, const stmt_t &body = {}) {
        return stmt_t(new let_t(var, value, body));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return var.is_equal(other.var) && value.is_equal(other.value)
                && body.is_equal(other.body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(var, value, body);
    }

    int register_alloc_size() const {
        // Empty objects are allocated in reserved space
        // nGEN only claims subregisters at dword granularity
        if (value.is_empty()) return 0;
        return utils::rnd_up(var.type().size(), reg_allocator_t::granularity);
    };

    std::string line_str() const {
        ostringstream_t out;
        out << var << "." << var.type() << " = " << value;
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    expr_t var;
    expr_t value;
    stmt_t body;

private:
    let_t(const expr_t &var, const expr_t &value, const stmt_t &body)
        : stmt_impl_t(_type_info()), var(var), value(value), body(body) {
        if (value && !is_const(value))
            gpu_assert(var.type() == value.type())
                    << "Variable " << var << " and  value " << value
                    << "have different types. " << var.type()
                    << " != " << value.type() << "\n";
    }
};

// Statement label, specific to GEMM/convolution.
class stmt_label_t {
public:
    static stmt_label_t kernel(int index = -1) {
        return stmt_label_t(kind_t::_kernel, index);
    }
    static stmt_label_t compute_loop(int index = -1) {
        return stmt_label_t(kind_t::_compute_loop, index);
    }
    static stmt_label_t c_store(int index = -1) {
        return stmt_label_t(kind_t::_c_store, index);
    }
    static stmt_label_t c_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_c_zero_out, index);
    }
    static stmt_label_t b_reduced_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_b_reduced_zero_out, index);
    }
    static stmt_label_t g2s_load(int index = -1) {
        return stmt_label_t(kind_t::_g2s_load, index);
    }
    static stmt_label_t g2s_store(int index = -1) {
        return stmt_label_t(kind_t::_g2s_store, index);
    }
    static stmt_label_t g2r_load(int index = -1) {
        return stmt_label_t(kind_t::_g2r_load, index);
    }
    static stmt_label_t s2r_load(int index = -1) {
        return stmt_label_t(kind_t::_s2r_load, index);
    }
    static stmt_label_t prefetch(int index = -1) {
        return stmt_label_t(kind_t::_prefetch, index);
    }
    static stmt_label_t mul(int index = -1) {
        return stmt_label_t(kind_t::_mul, index);
    }

    bool operator==(const stmt_label_t &other) const {
        if (kind_ != other.kind_) return false;
        if (index_ == -1 || other.index_ == -1) return true;
        return index_ == other.index_;
    }

    bool operator!=(const stmt_label_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const { return ir_utils::get_hash(kind_, index_); }

    std::string str() const {
        switch (kind_) {
#define CASE(kind) \
    case kind_t::_##kind: return #kind
            CASE(kernel);
            CASE(compute_loop);
            CASE(c_store);
            CASE(c_zero_out);
            CASE(g2r_load);
            CASE(g2s_load);
            CASE(g2s_store);
            CASE(s2r_load);
            CASE(prefetch);
            CASE(mul);
#undef CASE
            default: gpu_error_not_expected();
        }
        return {};
    }

private:
    enum class kind_t {
        _undef,
        _kernel, // All kernel.
        _compute_loop, // Compute loop.
        _c_store, // GRF to GMEM store of C.
        _c_zero_out, // Zeroing-out of C.
        _b_reduced_zero_out, // Zeroing-out of B reduced buffer.
        _g2r_load, // GMEM to GRF load for further multiplication.
        _g2s_load, // GMEM to GRF load for GMEM -> SLM copy.
        _g2s_store, // GRF to SLM store for GMEM -> SLM copy.
        _s2r_load, // SLM to GRF load for further multiplication.
        _prefetch, // GMEM prefetch.
        _mul, // Multiplication.
    };

    stmt_label_t() : kind_(kind_t::_undef), index_(-1) {}
    stmt_label_t(kind_t kind, int index) : kind_(kind), index_(index) {}

    kind_t kind_;
    int index_; // Used to differentiate groups with the same kind.
};

// Statement group, used to assign a label to a group of statements.
class stmt_group_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(stmt_group_t)

    static stmt_t make(const stmt_label_t &label, const stmt_t &body) {
        return stmt_t(new stmt_group_t(label, body));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (label == other.label) && body.is_equal(other.body);
    }

    size_t get_hash() const override { return ir_utils::get_hash(label, body); }

    IR_DECLARE_TRAVERSERS()

    stmt_label_t label;
    stmt_t body;

private:
    stmt_group_t(const stmt_label_t &label, const stmt_t &body)
        : stmt_impl_t(_type_info()), label(label), body(body) {}
};

// Statement sequence, allows combining multiple statements.
// C++ equivalent:
//     {
//         vec[0];
//         vec[1];
//         ...
//     }
class stmt_seq_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(stmt_seq_t)

    static stmt_t make(const std::vector<stmt_t> &vec);

    static stmt_t make(const stmt_t &head, const stmt_t &tail) {
        return head.append(tail);
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return ir_utils::is_equal(vec, other.vec);
    }

    size_t get_hash() const override { return ir_utils::get_hash(vec); }

    IR_DECLARE_TRAVERSERS()

    std::vector<stmt_t> vec;

private:
    stmt_seq_t(const std::vector<stmt_t> &vec)
        : stmt_impl_t(_type_info()), vec(vec) {}
};

// While loop statement with a condition.
// C++ equivalent:
//    while (cond) {
//        body;
//    }
class while_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(while_t)

    static stmt_t make(const expr_t &cond, const stmt_t &body = {}) {
        return stmt_t(new while_t(cond, body));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return cond.is_equal(other.cond) && body.is_equal(other.body);
    }

    size_t get_hash() const override { return ir_utils::get_hash(cond, body); }

    std::string line_str() const {
        ostringstream_t out;
        out << "while (" << cond << ")";
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    expr_t cond;
    stmt_t body;

private:
    while_t(const expr_t &cond, const stmt_t &body)
        : stmt_impl_t(_type_info()), cond(cond), body(body) {}
};

// Function call attribute.
class func_call_attr_impl_t : public object_impl_t {
public:
    func_call_attr_impl_t(type_info_t type_info) : object_impl_t(type_info) {}
};

class func_call_attr_t : public object_t {
public:
    using object_t::object_t;

    func_call_attr_t() = default;
    func_call_attr_t(const object_t &obj) : object_t(obj) {}
    func_call_attr_t(object_t &&obj) : object_t(obj) {}
    func_call_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_call_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    // Returns a function call with the attribute applied. The input statement
    // must be a function call.
    stmt_t apply_to(const stmt_t &s) const;

private:
#ifdef SANITY_CHECK
    void sanity_check() const override {
        gpu_assert(
                dynamic_cast<const func_call_attr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
#endif
};

// Instruction modifier, relies on nGEN API.
class instruction_modifier_attr_t : public func_call_attr_impl_t {
public:
    IR_DECL_TYPE(instruction_modifier_attr_t)

    static func_call_attr_t make(const ngen::InstructionModifier &mod) {
        return func_call_attr_t(new instruction_modifier_attr_t(mod));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return mod.getAll() == other.mod.getAll();
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(mod.getAll());
    }

    std::string str() const override {
        ostringstream_t oss;
        oss << "{";
        bool is_first = true;
        auto append = [&](const std::string &s) {
            if (!is_first) oss << ", ";
            oss << s;
            is_first = false;
        };
        if (mod.isAtomic()) append("Atomic");
        if (mod.getSWSB().empty()) {
            append(std::string("$") + std::to_string(mod.getSWSB().getToken()));
        }
        oss << "}";
        return oss.str();
    }

    ngen::InstructionModifier mod;

private:
    instruction_modifier_attr_t(const ngen::InstructionModifier &mod)
        : func_call_attr_impl_t(_type_info()), mod(mod) {}
};

// Base class for function IR objects.
class func_impl_t : public object_impl_t {
public:
    func_impl_t(type_info_t type_info) : object_impl_t(type_info) {}

    size_t get_hash() const override {
        gpu_error_not_expected() << "get_hash() is not implemented.";
        return 0;
    }

    bool is_equal(const object_impl_t &obj) const override {
        gpu_error_not_expected() << "is_equal() is not implemented.";
        return false;
    }

    stmt_t call(const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) const;

    IR_DECLARE_TRAVERSERS()
};

// Wrapper for IR function objects.
class func_t : public object_t {
public:
    using object_t::object_t;

    func_t() = default;
    func_t(const object_t &obj) : object_t(obj) {}
    func_t(object_t &&obj) : object_t(obj) {}
    func_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    stmt_t call(const std::vector<expr_t> &args = {},
            const func_call_attr_t &attr = {}) const {
        return ((const func_impl_t *)impl())->call(args, attr);
    }

private:
#ifdef SANITY_CHECK
    void sanity_check() const override {
        gpu_assert(dynamic_cast<const func_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
#endif
};

// Function call.
class func_call_t : public stmt_impl_t {
public:
    IR_DECL_CORE_TYPE(func_call_t)

    static stmt_t make(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) {
        return stmt_t(new func_call_t(func, args, attr));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return func.is_equal(other.func) && ir_utils::is_equal(args, other.args)
                && attr.is_equal(other.attr);
    }

    size_t get_hash() const override { return ir_utils::get_hash(args, attr); }

    std::string line_str() const {
        ostringstream_t out;
        out << func.str() << "(" << ir_utils::make_seq_print_helper(args)
            << ")";
        if (attr) out << " " << attr;
        return out.str();
    }

    IR_DECLARE_TRAVERSERS()

    func_t func;
    std::vector<expr_t> args;
    func_call_attr_t attr;

private:
    func_call_t(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr)
        : stmt_impl_t(_type_info()), func(func), args(args), attr(attr) {
        gpu_assert(func);
    }
};

inline stmt_t func_impl_t::call(
        const std::vector<expr_t> &args, const func_call_attr_t &attr) const {
    return func_call_t::make(this, args, attr);
}

inline stmt_t func_call_attr_t::apply_to(const stmt_t &s) const {
    auto &c = s.as<func_call_t>();
    gpu_assert(c.attr.is_empty())
            << "Merging of attributes is not supported: " << s;
    return func_call_t::make(c.func, c.args, *this);
}

template <typename F>
inline bool is_func_call(const stmt_t &s) {
    auto *c = s.as_ptr<func_call_t>();
    if (!c) return false;
    return c->func.is<F>();
}

// Generic function with a name.
class builtin_t : public func_impl_t {
public:
    IR_DECL_TYPE(builtin_t)

    static func_t make(const std::string &name) {
        return func_t(new builtin_t(name));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return name == other.name;
    }

    std::string str() const override { return name; }

    std::string name;

private:
    builtin_t(const std::string &name)
        : func_impl_t(_type_info()), name(name) {}
};

#ifndef SANITY_CHECK
// The following types are intrusive pointers and, as such, should have the same
// size as a pointer.
static_assert(sizeof(object_t) <= sizeof(void *),
        "intrusive pointer type object_t size is greater than void * "
        "size.");
static_assert(sizeof(expr_t) <= sizeof(void *),
        "intrusive pointer type expr_t size is greater than void * size.");
static_assert(sizeof(stmt_t) <= sizeof(void *),
        "intrusive pointer type stmt_t size is greater than void * size.");
#endif

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
