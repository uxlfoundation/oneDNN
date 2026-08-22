/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#if !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))

// AVX-512 fused dynamic-quantization kernels ported from AMD ZenDNN's reorder
// dynamic-quant path. The intrinsic kernel bodies are kept verbatim; only the
// threading driver (oneDNN parallel_nd / parallel) and the scalar bf16/f16
// conversion helpers are oneDNN-native.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <immintrin.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/x64/dynamic_quantize_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace dynamic_quantize_kernels {

namespace {

// Scalar bf16 -> f32: bf16 is the upper 16 bits of an IEEE-754 float32.
static inline float dq_bf16_to_f32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Scalar IEEE-754 half -> float (used only on the sub-16 element tail).
static inline float dq_f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// oneDNN-threaded replacement for ZenDNN's zendnnl_parallel_for chunked API.
template <typename F>
static inline void zendnnl_parallel_for(
        int64_t begin, int64_t end, int64_t /*grain*/, const F &f) {
    if (end <= begin) return;
    parallel(0, [&](int ithr, int nthr) {
        int64_t s = 0, e = 0;
        balance211(end - begin, nthr, ithr, s, e);
        if (s < e) f(begin + s, begin + e);
    });
}

} // namespace

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __m512
finite_or_zero_pg(__m512 v) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 absv = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
    return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ), v);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __m512
bf16x16_to_f32_pg(__m256i bf16) {
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) static inline __m512
f16x16_to_f32_pg(__m256i f16) {
    return _mm512_cvtph_ps(f16);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __mmask16
finite_mask_pg(__m512 v, __m512i abs_mask, __m512 vinf) {
    __m512 absv = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
    return _mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ);
}

static inline void compute_symmetric_scale_pg(float absmax, float &scale) {
    constexpr float scale_eps = 1.0e-30f;
    scale = std::max(absmax / 127.0f, scale_eps);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline void
store_4x16_s8_pg(int8_t *dst, __m128i i0, __m128i i1, __m128i i2, __m128i i3,
        bool cacheline_aligned) {
    if (cacheline_aligned) {
        __m256i lo = _mm256_set_m128i(i1, i0);
        __m256i hi = _mm256_set_m128i(i3, i2);
        __m512i pack = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst), pack);
    } else {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), i0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), i1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 32), i2);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 48), i3);
    }
}

template <typename LoadF32>
__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline float
group_absmax(const LoadF32 &load_f32, int64_t group_size, __m512i abs_mask,
        __m512 vinf) {
    __m512 vam0 = _mm512_setzero_ps();
    __m512 vam1 = _mm512_setzero_ps();
    __m512 vam2 = _mm512_setzero_ps();
    __m512 vam3 = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j + 63 < group_size; j += 64) {
        __m512 f0 = load_f32(j);
        __m512 f1 = load_f32(j + 16);
        __m512 f2 = load_f32(j + 32);
        __m512 f3 = load_f32(j + 48);
        __m512 a0 = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
        __m512 a1 = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
        __m512 a2 = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
        __m512 a3 = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
        vam0 = _mm512_mask_max_ps(
                vam0, finite_mask_pg(f0, abs_mask, vinf), vam0, a0);
        vam1 = _mm512_mask_max_ps(
                vam1, finite_mask_pg(f1, abs_mask, vinf), vam1, a1);
        vam2 = _mm512_mask_max_ps(
                vam2, finite_mask_pg(f2, abs_mask, vinf), vam2, a2);
        vam3 = _mm512_mask_max_ps(
                vam3, finite_mask_pg(f3, abs_mask, vinf), vam3, a3);
    }

    for (; j + 15 < group_size; j += 16) {
        __m512 f = load_f32(j);
        __m512 af = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(f), abs_mask));
        vam0 = _mm512_mask_max_ps(
                vam0, finite_mask_pg(f, abs_mask, vinf), vam0, af);
    }

    vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1), _mm512_max_ps(vam2, vam3));
    return _mm512_reduce_max_ps(vam0);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
dynamic_per_group_quant_f32_s8_native(const float *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G) {
    if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

    const int64_t group_size = K / G;
    const int64_t total_groups = M * G;
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    zendnnl_parallel_for(0, total_groups, 1,
            [&](int64_t begin, int64_t end) __attribute__((
                    target("avx512f,avx512bw,avx512vl"))) {
                for (int64_t task = begin; task < end; ++task) {
                    const int64_t m = task / G;
                    const int64_t g = task - m * G;
                    const int64_t offset = m * K + g * group_size;
                    const float *grp_src = src + offset;
                    int8_t *grp_dst = dst + offset;

                    bool quantize_phase = false;
                    auto load_f32 = ([&](int64_t j) __attribute__((
                            target("avx512f,avx512bw,avx512vl"))) {
                        const __m512 v = _mm512_loadu_ps(grp_src + j);
                        return quantize_phase ? finite_or_zero_pg(v) : v;
                    });

                    float absmax = group_absmax(
                            load_f32, group_size, abs_mask, vinf);
                    int64_t j = group_size & ~int64_t {15};
                    for (; j < group_size; ++j) {
                        if (std::isfinite(grp_src[j]))
                            absmax = std::max(absmax, std::abs(grp_src[j]));
                    }

                    float scale;
                    compute_symmetric_scale_pg(absmax, scale);
                    scales[task] = scale;

                    quantize_phase = true;
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const bool cl_ok
                            = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

                    j = 0;
                    for (; j + 63 < group_size; j += 64) {
                        __m512i r0 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        __m512i r1 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 16), vscale));
                        __m512i r2 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 32), vscale));
                        __m512i r3 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 48), vscale));
                        store_4x16_s8_pg(grp_dst + j, _mm512_cvtepi32_epi8(r0),
                                _mm512_cvtepi32_epi8(r1),
                                _mm512_cvtepi32_epi8(r2),
                                _mm512_cvtepi32_epi8(r3), cl_ok);
                    }
                    for (; j + 15 < group_size; j += 16) {
                        __m512i r = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        _mm_storeu_si128(
                                reinterpret_cast<__m128i *>(grp_dst + j),
                                _mm512_cvtepi32_epi8(r));
                    }
                    for (; j < group_size; ++j) {
                        if (!std::isfinite(grp_src[j])) {
                            grp_dst[j] = 0;
                            continue;
                        }
                        int32_t q = static_cast<int32_t>(
                                std::nearbyint(grp_src[j] / scale));
                        grp_dst[j] = static_cast<int8_t>(q);
                    }
                }
            });
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
dynamic_per_group_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G) {
    if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

    const int64_t group_size = K / G;
    const int64_t total_groups = M * G;
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    zendnnl_parallel_for(0, total_groups, 1,
            [&](int64_t begin, int64_t end) __attribute__((
                    target("avx512f,avx512bw,avx512vl"))) {
                for (int64_t task = begin; task < end; ++task) {
                    const int64_t m = task / G;
                    const int64_t g = task - m * G;
                    const int64_t offset = m * K + g * group_size;
                    const uint16_t *grp_src = src + offset;
                    int8_t *grp_dst = dst + offset;

                    bool quantize_phase = false;
                    auto load_f32 = ([&](int64_t j) __attribute__((
                            target("avx512f,avx512bw,avx512vl"))) {
                        const __m512 v = bf16x16_to_f32_pg(_mm256_loadu_si256(
                                reinterpret_cast<const __m256i *>(
                                        grp_src + j)));
                        return quantize_phase ? finite_or_zero_pg(v) : v;
                    });

                    float absmax = group_absmax(
                            load_f32, group_size, abs_mask, vinf);
                    int64_t j = group_size & ~int64_t {15};
                    for (; j < group_size; ++j) {
                        float v = dq_bf16_to_f32(grp_src[j]);
                        if (std::isfinite(v))
                            absmax = std::max(absmax, std::abs(v));
                    }

                    float scale;
                    compute_symmetric_scale_pg(absmax, scale);
                    scales[task] = scale;

                    quantize_phase = true;
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const bool cl_ok
                            = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

                    j = 0;
                    for (; j + 63 < group_size; j += 64) {
                        __m512i r0 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        __m512i r1 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 16), vscale));
                        __m512i r2 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 32), vscale));
                        __m512i r3 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 48), vscale));
                        store_4x16_s8_pg(grp_dst + j, _mm512_cvtepi32_epi8(r0),
                                _mm512_cvtepi32_epi8(r1),
                                _mm512_cvtepi32_epi8(r2),
                                _mm512_cvtepi32_epi8(r3), cl_ok);
                    }
                    for (; j + 15 < group_size; j += 16) {
                        __m512i r = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        _mm_storeu_si128(
                                reinterpret_cast<__m128i *>(grp_dst + j),
                                _mm512_cvtepi32_epi8(r));
                    }
                    for (; j < group_size; ++j) {
                        float v = dq_bf16_to_f32(grp_src[j]);
                        if (!std::isfinite(v)) {
                            grp_dst[j] = 0;
                            continue;
                        }
                        int32_t q = static_cast<int32_t>(
                                std::nearbyint(v / scale));
                        grp_dst[j] = static_cast<int8_t>(q);
                    }
                }
            });
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void
dynamic_per_group_quant_f16_s8_native(const uint16_t *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G) {
    if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

    const int64_t group_size = K / G;
    const int64_t total_groups = M * G;
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    zendnnl_parallel_for(0, total_groups, 1,
            [&](int64_t begin, int64_t end) __attribute__((
                    target("avx512f,avx512bw,avx512vl,f16c"))) {
                for (int64_t task = begin; task < end; ++task) {
                    const int64_t m = task / G;
                    const int64_t g = task - m * G;
                    const int64_t offset = m * K + g * group_size;
                    const uint16_t *grp_src = src + offset;
                    int8_t *grp_dst = dst + offset;

                    bool quantize_phase = false;
                    auto load_f32 = ([&](int64_t j) __attribute__((
                            target("avx512f,avx512bw,avx512vl,f16c"))) {
                        const __m512 v = f16x16_to_f32_pg(_mm256_loadu_si256(
                                reinterpret_cast<const __m256i *>(
                                        grp_src + j)));
                        return quantize_phase ? finite_or_zero_pg(v) : v;
                    });

                    float absmax = group_absmax(
                            load_f32, group_size, abs_mask, vinf);
                    int64_t j = group_size & ~int64_t {15};
                    for (; j < group_size; ++j) {
                        float v = dq_f16_to_f32(grp_src[j]);
                        if (std::isfinite(v))
                            absmax = std::max(absmax, std::abs(v));
                    }

                    float scale;
                    compute_symmetric_scale_pg(absmax, scale);
                    scales[task] = scale;

                    quantize_phase = true;
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const bool cl_ok
                            = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

                    j = 0;
                    for (; j + 63 < group_size; j += 64) {
                        __m512i r0 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        __m512i r1 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 16), vscale));
                        __m512i r2 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 32), vscale));
                        __m512i r3 = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j + 48), vscale));
                        store_4x16_s8_pg(grp_dst + j, _mm512_cvtepi32_epi8(r0),
                                _mm512_cvtepi32_epi8(r1),
                                _mm512_cvtepi32_epi8(r2),
                                _mm512_cvtepi32_epi8(r3), cl_ok);
                    }
                    for (; j + 15 < group_size; j += 16) {
                        __m512i r = _mm512_cvtps_epi32(
                                _mm512_div_ps(load_f32(j), vscale));
                        _mm_storeu_si128(
                                reinterpret_cast<__m128i *>(grp_dst + j),
                                _mm512_cvtepi32_epi8(r));
                    }
                    for (; j < group_size; ++j) {
                        float v = dq_f16_to_f32(grp_src[j]);
                        if (!std::isfinite(v)) {
                            grp_dst[j] = 0;
                            continue;
                        }
                        int32_t q = static_cast<int32_t>(
                                std::nearbyint(v / scale));
                        grp_dst[j] = static_cast<int8_t>(q);
                    }
                }
            });
}

} // namespace dynamic_quantize_kernels
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
