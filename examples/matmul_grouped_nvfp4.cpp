/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

/// @example matmul_grouped_nvfp4.cpp
/// > Annotated version: @ref matmul_grouped_nvfp4_cpp

/// @page matmul_grouped_nvfp4_cpp_brief
/// @brief This C++ API example demonstrates grouped matrix-matrix multiplication
/// with 4-bit floating-point weights and block scales, the quantization recipe
/// used by Mixture-of-Experts (MoE) checkpoints stored in NVFP4.

/// @page matmul_grouped_nvfp4_cpp MatMul with Grouped Encoding and NVFP4 Weights
/// \copybrief matmul_grouped_nvfp4_cpp_brief
///
/// Steps in this example cover:
/// - How to combine grouped encoding with `f4_e2m1` weights
/// - How to specify `f8_e4m3` block scales over groups of 16 elements along K
/// - Why the scale tensor must be a dense canonical tensor, and what its
///   logical shape is
/// - How to handle a variable token distribution, including an expert that
///   receives no tokens at all
/// - Verifying the result against a reference computed on the host
///
/// @include matmul_grouped_nvfp4.cpp

namespace {

// Group size of the weight block scales along K. NVFP4 is defined at 16.
constexpr memory::dim wei_group_size = 16;

// The eight E2M1 magnitudes, indexed by the low three bits of a nibble. Bit 3
// carries the sign.
constexpr float e2m1_table[8]
        = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float decode_e2m1(uint8_t nibble) {
    const float mag = e2m1_table[nibble & 0x7];
    return (nibble & 0x8) ? -mag : mag;
}

// A handful of E4M3 encodings whose values are exact, so the host reference
// does not need a general E4M3 decoder. E4M3 is sign(1) exponent(4)
// mantissa(3) with an exponent bias of 7.
struct e4m3_value {
    uint8_t bits;
    float value;
};
constexpr e4m3_value e4m3_values[4] = {
        {0x30, 0.5f}, // exponent 6, mantissa 000
        {0x38, 1.0f}, // exponent 7, mantissa 000
        {0x3C, 1.5f}, // exponent 7, mantissa 100
        {0x40, 2.0f}, // exponent 8, mantissa 000
};

// Minimal float -> IEEE half conversion. Sufficient because this example only
// stores values that are exactly representable in f16.
uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = (x >> 13) & 0x3FFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

} // namespace

void grouped_matmul_nvfp4_example(engine::kind engine_kind) {
    engine eng(engine_kind, 0);
    stream engine_stream(eng);

    // A routed MoE layer produces a very uneven token distribution: some
    // experts receive many tokens, some receive none. Both cases are covered
    // here on purpose -- expert 2 gets no tokens.
    const memory::dim num_experts = 4;
    const std::vector<int32_t> tokens_per_expert = {6, 40, 0, 18};

    // Cumulative (exclusive-end) offsets, as required by grouped encoding.
    std::vector<int32_t> offsets(num_experts);
    std::partial_sum(tokens_per_expert.begin(), tokens_per_expert.end(),
            offsets.begin());

    const memory::dim total_tokens = offsets.back();

    // K must be a multiple of the scale group size.
    const memory::dim K = 64;
    const memory::dim N = 32;
    const memory::dim wei_num_groups = K / wei_group_size;

    std::cout << "Experts            : " << num_experts << std::endl;
    std::cout << "Tokens per expert  : ";
    for (memory::dim i = 0; i < num_experts; ++i)
        std::cout << tokens_per_expert[i] << (i + 1 < num_experts ? ", " : "");
    std::cout << " (total " << total_tokens << ")" << std::endl;
    std::cout << "K, N               : " << K << ", " << N << std::endl;
    std::cout << "Weight scales      : f8_e4m3, group " << wei_group_size
              << " along K" << std::endl;
    std::cout << std::endl;

    // src: f16 activations, [total_tokens, K], grouped along dimension 0.
    std::vector<float> src_ref(total_tokens * K);
    std::vector<uint16_t> src_data(total_tokens * K);
    for (memory::dim i = 0; i < total_tokens * K; ++i) {
        // Small multiples of 1/4 are exact in f16, which keeps the host
        // reference free of conversion error.
        src_ref[i] = static_cast<float>((i % 17) - 8) * 0.25f;
        src_data[i] = f32_to_f16(src_ref[i]);
    }

    // weights: f4_e2m1, logical [num_experts, K, N] in `acb` format, so the
    // physical layout is [num_experts, N, K] with K contiguous and two 4-bit
    // values packed per byte along K. The low nibble holds the even K index.
    std::vector<uint8_t> wei_packed(num_experts * N * K / 2);
    for (size_t i = 0; i < wei_packed.size(); ++i)
        wei_packed[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

    // weight scales: f8_e4m3, logical [num_experts, K/group, N].
    //
    // This tensor must be DENSE and in the canonical `abc` layout. The scales
    // are described by mask and group size only -- there is no API to hand the
    // library a differently strided scale tensor -- so a producer holding its
    // scales as [num_experts, N, K/group] has to repack them into the layout
    // below rather than aliasing them with a transposed descriptor.
    std::vector<uint8_t> wei_scale_bits(num_experts * wei_num_groups * N);
    for (size_t i = 0; i < wei_scale_bits.size(); ++i)
        wei_scale_bits[i] = e4m3_values[i % 4].bits;

    std::vector<float> dst_data(total_tokens * N, 0.0f);

    auto src_md = memory::desc::grouped(
            {total_tokens, K}, memory::data_type::f16, 0, num_experts);
    auto dst_md = memory::desc::grouped(
            {total_tokens, N}, memory::data_type::f32, 0, num_experts);
    auto wei_md = memory::desc({num_experts, K, N}, memory::data_type::f4_e2m1,
            memory::format_tag::acb);
    auto wei_scale_md = memory::desc({num_experts, wei_num_groups, N},
            memory::data_type::f8_e4m3, memory::format_tag::abc);

    auto src_mem = memory(src_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto wei_mem = memory(wei_md, eng);
    auto wei_scale_mem = memory(wei_scale_md, eng);

    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(wei_packed.data(), wei_mem);
    write_to_dnnl_memory(wei_scale_bits.data(), wei_scale_mem);

    // Grouped memory carries the offsets in buffer 1. src and dst share the
    // same token distribution and therefore the same offsets.
    write_to_dnnl_memory(offsets.data(), src_mem, 1);
    write_to_dnnl_memory(offsets.data(), dst_mem, 1);

    // Mask bits 0|1|2 mark all three weight dimensions as scaled, and the
    // group specification {wei_group_size, 1} blocks the last two dimensions,
    // i.e. one scale per 16 elements of K per output column.
    primitive_attr matmul_attr;
    matmul_attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2),
            {wei_group_size, 1}, memory::data_type::f8_e4m3);

    auto matmul_pd
            = matmul::primitive_desc(eng, src_md, wei_md, dst_md, matmul_attr);
    auto matmul_prim = matmul(matmul_pd);

    matmul_prim.execute(engine_stream,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_mem}});
    engine_stream.wait();

    read_from_dnnl_memory(dst_data.data(), dst_mem);

    std::cout << "Implementation     : " << matmul_pd.impl_info_str()
              << std::endl;

    // Reference: dequantize each weight on the host and accumulate in f32.
    double max_rel_err = 0.0;
    memory::dim token = 0;
    for (memory::dim e = 0; e < num_experts; ++e) {
        for (int32_t t = 0; t < tokens_per_expert[e]; ++t, ++token) {
            for (memory::dim n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (memory::dim k = 0; k < K; ++k) {
                    // Physical weight byte for (e, n, k): [E, N, K/2].
                    const size_t byte_idx
                            = (static_cast<size_t>(e) * N + n) * (K / 2)
                            + k / 2;
                    const uint8_t byte = wei_packed[byte_idx];
                    const uint8_t nibble
                            = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                    // Canonical scale index for (e, k/group, n).
                    const size_t scale_idx
                            = (static_cast<size_t>(e) * wei_num_groups
                                      + k / wei_group_size)
                                    * N
                            + n;
                    float scale = 0.0f;
                    for (const auto &v : e4m3_values)
                        if (v.bits == wei_scale_bits[scale_idx])
                            scale = v.value;
                    acc += src_ref[token * K + k] * decode_e2m1(nibble) * scale;
                }
                const float got = dst_data[token * N + n];
                const float denom = std::max(std::abs(acc), 1.0f);
                max_rel_err = std::max(
                        max_rel_err, double(std::abs(got - acc) / denom));
            }
        }
    }

    std::cout << "Max relative error : " << max_rel_err << std::endl;
    if (max_rel_err > 1e-5) {
        throw std::runtime_error(
                "grouped NVFP4 matmul result does not match the reference");
    }
    std::cout << "Result             : PASSED" << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            grouped_matmul_nvfp4_example, parse_engine_kind(argc, argv));
}
