/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_REF_IO_HELPER_HPP
#define CPU_REF_IO_HELPER_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/sub_byte.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace io {

inline int load_int_value(data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<int>( \
                reinterpret_cast<const typename prec_traits_t<dt>::type *>( \
                        ptr)[idx]);
#define CASE_SUB_BYTE(dt) \
    case dt: \
        return static_cast<int>(typename prec_traits_t<dt>::type( \
                sub_byte_get<sub_byte_bits(dt)>( \
                        static_cast<const uint8_t *>(ptr), idx)));

    using namespace data_type;
    switch (dt) {
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE_SUB_BYTE(s4);
        CASE_SUB_BYTE(u4);
        CASE_SUB_BYTE(u2);
        default: assert(!"bad data_type");
    }

#undef CASE_SUB_BYTE
#undef CASE
    return INT_MAX;
}

inline int64_t load_int64_value(data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
    if (dt == data_type::s64)
        return reinterpret_cast<const int64_t *>(ptr)[idx];
    return static_cast<int64_t>(load_int_value(dt, ptr, idx));
}

ALWAYS_INLINE float load_float_value(
        data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<float>( \
                reinterpret_cast<const typename prec_traits_t<dt>::type *>( \
                        ptr)[idx]);
#define CASE_SUB_BYTE(dt) \
    case dt: \
        return static_cast<float>(typename prec_traits_t<dt>::type( \
                sub_byte_get<sub_byte_bits(dt)>( \
                        static_cast<const uint8_t *>(ptr), idx)));

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(e8m0);
        CASE_SUB_BYTE(s4);
        CASE_SUB_BYTE(u4);
        CASE_SUB_BYTE(u2);
        CASE_SUB_BYTE(f4_e2m1);
        default: assert(!"bad data_type");
    }

#undef CASE_SUB_BYTE
#undef CASE
    return NAN;
}

inline void store_float_value(data_type_t dt, float val, void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: { \
        using type_ = typename prec_traits_t<dt>::type; \
        *(reinterpret_cast<type_ *>(ptr) + idx) \
                = cpu::q10n::saturate_and_round<type_>(val); \
    } break;

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(e8m0);
        case f4_e2m1: {
            float4_e2m1_t f4_val(val);
            sub_byte_set<4>(static_cast<uint8_t *>(ptr), idx, f4_val.raw_bits_);
            break;
        }
        default: assert(!"bad data_type");
    }

#undef CASE
}

} // namespace io
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
