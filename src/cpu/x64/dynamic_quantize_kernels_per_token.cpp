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
finite_or_zero(__m512 v) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 absv = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
    return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ), v);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __m512
bf16x16_to_f32(__m256i bf16) {
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) static inline __m512
f16x16_to_f32(__m256i f16) {
    return _mm512_cvtph_ps(f16);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __m512
load_finite_f32(const float *src) {
    return finite_or_zero(_mm512_loadu_ps(src));
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __mmask16
finite_mask(__m512 v, __m512i abs_mask, __m512 vinf) {
    __m512 absv = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
    return _mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ);
}

static inline void compute_symmetric_scale_from_absmax(
        float absmax, float &scale) {
    constexpr float scale_eps = 1.0e-30f;
    scale = std::max(absmax / 127.0f, scale_eps);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline void
store_4x16_s8(int8_t *dst, __m128i i0, __m128i i1, __m128i i2, __m128i i3,
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

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
dynamic_per_token_quant_bf16_s8_native(
        const uint16_t *src, int8_t *dst, float *scales, int64_t M, int64_t N) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    auto row_loop = ([&](
            int64_t m) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        const uint16_t *row_src = src + m * N;
        int8_t *row_dst = dst + m * N;

        // -- Pass 1: absmax reduction -----------------------------------------
        __m512 vam0 = _mm512_setzero_ps();
        __m512 vam1 = _mm512_setzero_ps();
        __m512 vam2 = _mm512_setzero_ps();
        __m512 vam3 = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j + 63 < N; j += 64) {
            __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 16)));
            __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 32)));
            __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 48)));

            __m512 a0 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
            __m512 a1 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
            __m512 a2 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
            __m512 a3 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
            vam1 = _mm512_mask_max_ps(
                    vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
            vam2 = _mm512_mask_max_ps(
                    vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
            vam3 = _mm512_mask_max_ps(
                    vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
        }

        for (; j + 15 < N; j += 16) {
            __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 af = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f, abs_mask, vinf), vam0, af);
        }

        vam0 = _mm512_max_ps(
                _mm512_max_ps(vam0, vam1), _mm512_max_ps(vam2, vam3));
        float absmax = _mm512_reduce_max_ps(vam0);

        for (; j < N; ++j) {
            float v = dq_bf16_to_f32(row_src[j]);
            if (std::isfinite(v)) absmax = std::max(absmax, std::abs(v));
        }

        // -- Compute per-row scale --------------------------------------------
        float scale;
        compute_symmetric_scale_from_absmax(absmax, scale);
        scales[m] = scale;

        // -- Pass 2: quantize (re-load BF16 from L1, still hot from Pass 1) --
        __m512 vscale = _mm512_set1_ps(scale);
        bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

        j = 0;
        for (; j + 63 < N; j += 64) {
            __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 16)));
            __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 32)));
            __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 48)));
            f0 = finite_or_zero(f0);
            f1 = finite_or_zero(f1);
            f2 = finite_or_zero(f2);
            f3 = finite_or_zero(f3);

            __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(f0, vscale));
            __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(f1, vscale));
            __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(f2, vscale));
            __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(f3, vscale));

            store_4x16_s8(row_dst + j, _mm512_cvtepi32_epi8(r0),
                    _mm512_cvtepi32_epi8(r1), _mm512_cvtepi32_epi8(r2),
                    _mm512_cvtepi32_epi8(r3), cl_ok);
        }
        for (; j + 15 < N; j += 16) {
            __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            f = finite_or_zero(f);
            __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(f, vscale));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                    _mm512_cvtepi32_epi8(r));
        }
        for (; j < N; ++j) {
            float v = dq_bf16_to_f32(row_src[j]);
            if (!std::isfinite(v)) {
                row_dst[j] = 0;
                continue;
            }
            int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
            row_dst[j] = static_cast<int8_t>(q);
        }
    });

    parallel_nd(M, [&](dim_t m) { row_loop(m); });
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
dynamic_per_token_quant_f32_s8_native(
        const float *src, int8_t *dst, float *scales, int64_t M, int64_t N) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    auto row_loop = ([&](
            int64_t m) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        const float *row_src = src + m * N;
        int8_t *row_dst = dst + m * N;

        // -- Pass 1: absmax reduction (skipping non-finite values) ---------------
        __m512 vam0 = _mm512_setzero_ps();
        __m512 vam1 = _mm512_setzero_ps();
        __m512 vam2 = _mm512_setzero_ps();
        __m512 vam3 = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j + 63 < N; j += 64) {

            __m512 f0 = _mm512_loadu_ps(row_src + j);
            __m512 f1 = _mm512_loadu_ps(row_src + j + 16);
            __m512 f2 = _mm512_loadu_ps(row_src + j + 32);
            __m512 f3 = _mm512_loadu_ps(row_src + j + 48);

            __m512 a0 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
            __m512 a1 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
            __m512 a2 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
            __m512 a3 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
            vam1 = _mm512_mask_max_ps(
                    vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
            vam2 = _mm512_mask_max_ps(
                    vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
            vam3 = _mm512_mask_max_ps(
                    vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
        }

        for (; j + 15 < N; j += 16) {
            __m512 f = _mm512_loadu_ps(row_src + j);
            __m512 af = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f, abs_mask, vinf), vam0, af);
        }

        vam0 = _mm512_max_ps(
                _mm512_max_ps(vam0, vam1), _mm512_max_ps(vam2, vam3));
        float absmax = _mm512_reduce_max_ps(vam0);

        for (; j < N; ++j)
            if (std::isfinite(row_src[j]))
                absmax = std::max(absmax, std::abs(row_src[j]));

        float scale;
        compute_symmetric_scale_from_absmax(absmax, scale);
        scales[m] = scale;

        // -- Pass 2: quantize (F32 re-read from L1, no conversion needed) -----
        __m512 vscale = _mm512_set1_ps(scale);
        bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

        j = 0;
        for (; j + 63 < N; j += 64) {
            __m512i r0 = _mm512_cvtps_epi32(
                    _mm512_div_ps(load_finite_f32(row_src + j), vscale));
            __m512i r1 = _mm512_cvtps_epi32(
                    _mm512_div_ps(load_finite_f32(row_src + j + 16), vscale));
            __m512i r2 = _mm512_cvtps_epi32(
                    _mm512_div_ps(load_finite_f32(row_src + j + 32), vscale));
            __m512i r3 = _mm512_cvtps_epi32(
                    _mm512_div_ps(load_finite_f32(row_src + j + 48), vscale));

            store_4x16_s8(row_dst + j, _mm512_cvtepi32_epi8(r0),
                    _mm512_cvtepi32_epi8(r1), _mm512_cvtepi32_epi8(r2),
                    _mm512_cvtepi32_epi8(r3), cl_ok);
        }

        for (; j + 15 < N; j += 16) {
            __m512i r = _mm512_cvtps_epi32(
                    _mm512_div_ps(load_finite_f32(row_src + j), vscale));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                    _mm512_cvtepi32_epi8(r));
        }

        for (; j < N; ++j) {
            if (!std::isfinite(row_src[j])) {
                row_dst[j] = 0;
                continue;
            }
            int32_t q
                    = static_cast<int32_t>(std::nearbyint(row_src[j] / scale));
            row_dst[j] = static_cast<int8_t>(q);
        }
    });

    parallel_nd(M, [&](dim_t m) { row_loop(m); });
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void
dynamic_per_token_quant_f16_s8_native(
        const uint16_t *src, int8_t *dst, float *scales, int64_t M, int64_t N) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    auto row_loop = ([&](int64_t m) __attribute__((
            target("avx512f,avx512bw,avx512vl,f16c"))) {
        const uint16_t *row_src = src + m * N;
        int8_t *row_dst = dst + m * N;

        // -- Pass 1: absmax reduction -----------------------------------------
        __m512 vam0 = _mm512_setzero_ps();
        __m512 vam1 = _mm512_setzero_ps();
        __m512 vam2 = _mm512_setzero_ps();
        __m512 vam3 = _mm512_setzero_ps();

        int64_t j = 0;
        for (; j + 63 < N; j += 64) {
            __m512 f0 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 f1 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 16)));
            __m512 f2 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 32)));
            __m512 f3 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 48)));

            __m512 a0 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
            __m512 a1 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
            __m512 a2 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
            __m512 a3 = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
            vam1 = _mm512_mask_max_ps(
                    vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
            vam2 = _mm512_mask_max_ps(
                    vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
            vam3 = _mm512_mask_max_ps(
                    vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
        }

        for (; j + 15 < N; j += 16) {
            __m512 f = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 af = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(f), abs_mask));
            vam0 = _mm512_mask_max_ps(
                    vam0, finite_mask(f, abs_mask, vinf), vam0, af);
        }

        vam0 = _mm512_max_ps(
                _mm512_max_ps(vam0, vam1), _mm512_max_ps(vam2, vam3));
        float absmax = _mm512_reduce_max_ps(vam0);

        for (; j < N; ++j) {
            float v = dq_f16_to_f32(row_src[j]);
            if (std::isfinite(v)) absmax = std::max(absmax, std::abs(v));
        }

        // -- Compute per-row scale --------------------------------------------
        float scale;
        compute_symmetric_scale_from_absmax(absmax, scale);
        scales[m] = scale;

        // -- Pass 2: quantize (re-load FP16 from L1, still hot from Pass 1) --
        __m512 vscale = _mm512_set1_ps(scale);
        bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

        j = 0;
        for (; j + 63 < N; j += 64) {
            __m512 f0 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            __m512 f1 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 16)));
            __m512 f2 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 32)));
            __m512 f3 = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j + 48)));
            f0 = finite_or_zero(f0);
            f1 = finite_or_zero(f1);
            f2 = finite_or_zero(f2);
            f3 = finite_or_zero(f3);

            __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(f0, vscale));
            __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(f1, vscale));
            __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(f2, vscale));
            __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(f3, vscale));

            store_4x16_s8(row_dst + j, _mm512_cvtepi32_epi8(r0),
                    _mm512_cvtepi32_epi8(r1), _mm512_cvtepi32_epi8(r2),
                    _mm512_cvtepi32_epi8(r3), cl_ok);
        }
        for (; j + 15 < N; j += 16) {
            __m512 f = f16x16_to_f32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_src + j)));
            f = finite_or_zero(f);
            __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(f, vscale));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                    _mm512_cvtepi32_epi8(r));
        }
        for (; j < N; ++j) {
            float v = dq_f16_to_f32(row_src[j]);
            if (!std::isfinite(v)) {
                row_dst[j] = 0;
                continue;
            }
            int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
            row_dst[j] = static_cast<int8_t>(q);
        }
    });

    parallel_nd(M, [&](dim_t m) { row_loop(m); });
}

} // namespace dynamic_quantize_kernels
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
