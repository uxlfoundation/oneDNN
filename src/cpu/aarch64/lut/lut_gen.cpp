/*******************************************************************************
* Copyright 2021-2022, 2024-2025 Arm Ltd. and affiliates
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static inline uint32_t f32_to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static inline float bits_to_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline float bf16_bits_to_f32(uint16_t b) {
    return bits_to_f32(uint32_t(b) << 16);
}

// Round-to-nearest-even f32 -> bf16 bits
static inline uint16_t f32_to_bf16_bits_rne(float f) {
    uint32_t u = f32_to_bits(f);
    uint32_t exp = (u >> 23) & 0xff, frac = u & 0x7fffff;
    if (exp == 0xff) { // INF/NaN: keep top16; ensure quiet NaN
        uint16_t top = uint16_t(u >> 16);
        if (frac && ((top & 0x0040) == 0)) top |= 0x0040;
        return top;
    }
    uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    u += bias;
    return uint16_t(u >> 16);
}

// Ops
static inline float gelu_erf(float x) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
}

// Describe each LUT
struct LutSpec {
    const char *symbol; // C symbol to generate
    const char *pretty; // comment
    std::function<float(float)> op;
};

static std::vector<LutSpec> registry() {
    return {
            {"lut_eltwise_gelu_erf_bf16", "GELU(erf), bf16 domain", gelu_erf},
    };
}

static std::string chunked_hex(
        const std::vector<uint16_t> &v, int per_line = 16) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i % per_line == 0) oss << "    ";
        oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << v[i]
            << std::dec;
        if (i + 1 != v.size()) oss << ", ";
        if ((i + 1) % per_line == 0) oss << "\n";
    }
    return oss.str();
}

int main(int argc, char **argv) {
    std::string out_dir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    }

    const std::string out_hpp = out_dir + "/lut_tables.hpp";
    const std::string out_cpp = out_dir + "/lut_tables.cpp";

    auto specs = registry();

    // Build all tables
    struct Built {
        std::string symbol;
        std::vector<uint16_t> data;
        std::string pretty;
    };
    std::vector<Built> built;
    built.reserve(specs.size());

    for (const auto &s : specs) {
        std::vector<uint16_t> table(1u << 16);
        for (uint32_t raw = 0; raw < (1u << 16); ++raw) {
            float x = bf16_bits_to_f32(uint16_t(raw));
            float y = s.op(x);
            table[raw] = f32_to_bf16_bits_rne(y);
        }
        built.push_back({s.symbol, std::move(table), s.pretty});
    }

    // Emit header
    {
        std::ofstream hpp(out_hpp, std::ios::binary);
        if (!hpp) {
            std::cerr << "Failed to open " << out_hpp << "\n";
            return 1;
        }
        hpp << "// Auto-generated. Do not edit.\n"
            << "#pragma once\n"
            << "#include <common/bfloat16.hpp>\n#include <cstddef>\n\n"
            << "namespace dnnl { namespace impl { namespace cpu { namespace aarch64{\n";
        for (const auto &b : built) {
            hpp << "// " << b.pretty << "\n";
            hpp << "extern const uint16_t " << b.symbol << "[1u << 16];\n";
            hpp << "inline bfloat16_t  " << b.symbol
                << "_value(uint16_t idx){\n";
            hpp << "bfloat16_t v;\n";
            hpp << "std::memcpy(&v, &" << b.symbol << "[idx], 2);\n";
            hpp << "return v;\n";
            hpp << "}\n";
        }
        hpp << "\n}}}} // namespace dnnl::impl::cpu\n";
    }

    // Emit source
    {
        std::ofstream cpp(out_cpp, std::ios::binary);
        if (!cpp) {
            std::cerr << "Failed to open " << out_cpp << "\n";
            return 1;
        }
        cpp << "// Auto-generated. Do not edit.\n"
            << "#include <common/bfloat16.hpp>\n#include <cstddef>\n"
            << "#include \"lut_tables.hpp\"\n\n"
            << "namespace dnnl { namespace impl { namespace cpu { namespace aarch64{\n";
        for (const auto &b : built) {
            cpp << "alignas(64) const uint16_t " << b.symbol
                << "[1u << 16] = {\n"
                << chunked_hex(b.data, 16) << "\n};\n\n";
        }
        cpp << "}}}} // namespace dnnl::impl::cpu\n";
    }
    std::cout << "Generated " << out_hpp << " and " << out_cpp << "\n";
    return 0;
}
