/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/float4.hpp"
#include "src/common/float8.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"

#include "utils/numeric.hpp"

template <>
struct prec_traits<dnnl_f4_e2m1> {
    using type = dnnl::impl::float4_e2m1_t;
};
template <>
struct prec_traits<dnnl_f4_e3m0> {
    using type = dnnl::impl::float4_e3m0_t;
};
template <>
struct prec_traits<dnnl_e8m0> {
    using type = dnnl::impl::float8_e8m0_t;
};
template <>
struct prec_traits<dnnl_f8_e5m2> {
    using type = dnnl::impl::float8_e5m2_t;
};
template <>
struct prec_traits<dnnl_f8_e4m3> {
    using type = dnnl::impl::float8_e4m3_t;
};
template <>
struct prec_traits<dnnl_bf16> {
    using type = dnnl::impl::bfloat16_t;
};
template <>
struct prec_traits<dnnl_f16> {
    using type = dnnl::impl::float16_t;
};
template <>
struct prec_traits<dnnl_f32> {
    using type = float;
};

// XXX: benchdnn infra doesn't support double yet.
// Use float's max/min/epsilon values to avoid following build warnings:
// warning C4756: overflow in constant arithmetic.
// This should be fixed once cpu reference in f64 is added.
template <>
struct prec_traits<dnnl_f64> {
    using type = float;
};
template <>
struct prec_traits<dnnl_s32> {
    using type = int32_t;
};
template <>
struct prec_traits<dnnl_s16> {
    using type = int16_t;
};
template <>
struct prec_traits<dnnl_u16> {
    using type = uint16_t;
};
template <>
struct prec_traits<dnnl_s8> {
    using type = int8_t;
};
template <>
struct prec_traits<dnnl_u8> {
    using type = uint8_t;
};
template <>
struct prec_traits<dnnl_s4> {
    using type = dnnl::impl::int4_t;
};
template <>
struct prec_traits<dnnl_u4> {
    using type = dnnl::impl::uint4_t;
};
#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_f4_e2m1); \
        CASE(dnnl_f4_e3m0); \
        CASE(dnnl_e8m0); \
        CASE(dnnl_f8_e5m2); \
        CASE(dnnl_f8_e4m3); \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_f64); \
        CASE(dnnl_s32); \
        CASE(dnnl_s16); \
        CASE(dnnl_u16); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        CASE(dnnl_s4); \
        CASE(dnnl_u4); \
        default: assert(!"bad data_type"); SAFE_V(FAIL); \
    }

/* std::numeric_limits::digits functionality */
int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

float lowest_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::lowest();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

float max_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::max();

    CASE_ALL(dt);

#undef CASE
    return 0;
}

template <dnnl_data_type_t dt>
float saturate_and_round(float value) {
    static const float dt_max = max_dt(dt);
    static const float dt_min = lowest_dt(dt);
    static const float max_dt_s32 = max_dt(dnnl_s32);
    if (dt == dnnl_s32 && value >= max_dt_s32) return max_dt_s32;
    if (value > dt_max) value = dt_max;
    if (value < dt_min) value = dt_min;
    return mxcsr_cvt(value);
}

bool is_integral_dt(dnnl_data_type_t dt) {
    switch (dt) {
        case dnnl_s32:
        case dnnl_s16:
        case dnnl_u16:
        case dnnl_s8:
        case dnnl_u8:
        case dnnl_s4:
        case dnnl_u4: return true;
        default: return false;
    }
}

template <dnnl_data_type_t dt>
float maybe_saturate_templ(float value) {
    if (!is_integral_dt(dt)) return value;
    return saturate_and_round<dt>(value);
}

float maybe_saturate(dnnl_data_type_t dt, float value) {
#define CASE(dt) \
    case dt: return maybe_saturate_templ<dt>(value)

    CASE_ALL(dt)
#undef CASE
    return value;
}

template <dnnl_data_type_t dt>
float round_to_nearest_representable_templ(float value) {
    switch (dt) {
        case dnnl_f32: break;
        case dnnl_f64: break;
        case dnnl_f4_e2m1:
            value = (float)dnnl::impl::float4_e2m1_t(value);
            break;
        case dnnl_f4_e3m0:
            value = (float)dnnl::impl::float4_e3m0_t(value);
            break;
        case dnnl_e8m0: value = (float)dnnl::impl::float8_e8m0_t(value); break;
        case dnnl_f8_e5m2:
            value = (float)dnnl::impl::float8_e5m2_t(value);
            break;
        case dnnl_f8_e4m3:
            value = (float)dnnl::impl::float8_e4m3_t(value);
            break;
        case dnnl_bf16: value = (float)dnnl::impl::bfloat16_t(value); break;
        case dnnl_f16: value = (float)dnnl::impl::float16_t(value); break;
        case dnnl_s32:
        case dnnl_s16:
        case dnnl_u16:
        case dnnl_s8:
        case dnnl_u8:
        case dnnl_s4:
        case dnnl_u4: value = maybe_saturate_templ<dt>(value); break;
        default: SAFE_V(FAIL);
    }

    return value;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value) {
#define CASE(dt) \
    case dt: return round_to_nearest_representable_templ<dt>(value)

    CASE_ALL(dt)
#undef CASE
    return value;
}

#undef CASE_ALL

bool is_subbyte_type(const dnnl_data_type_t &type) {
    return type == dnnl_f4_e2m1 || type == dnnl_f4_e3m0 || type == dnnl_u4
            || type == dnnl_s4;
}
