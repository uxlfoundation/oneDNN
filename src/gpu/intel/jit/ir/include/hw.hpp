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

#ifndef GPU_INTEL_JIT_IR_INCLUDE_HW_HPP
#define GPU_INTEL_JIT_IR_INCLUDE_HW_HPP

#include <cstddef>
#include <cstdint>
#include <string>

// NOLINTBEGIN(readability-identifier-naming)
namespace ngen {
enum class Core;
using HW = Core;
enum class ProductFamily : int;
struct Product;
} // namespace ngen
// NOLINTEND(readability-identifier-naming)

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

namespace hw {
enum class attr_t { none = 0, large_grf = 1, systolic = 2, atomic_fp64 = 4 };
constexpr attr_t operator&(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr attr_t operator|(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr attr_t operator~(attr_t a) {
    return static_cast<attr_t>(~static_cast<uint32_t>(a));
}
constexpr bool any(attr_t a) {
    return a != static_cast<attr_t>(0);
}

inline attr_t &operator|=(attr_t &a, attr_t b) {
    return a = a | b;
}
inline attr_t &operator&=(attr_t &a, attr_t b) {
    return a = a & b;
}

} // namespace hw

// Provides access to HW configuration which includes non-configurable
// properties.
class hw_t {
public:
    using attr_t = hw::attr_t;
    hw_t() = default;
    explicit hw_t(const ngen::Product &product, int eu_count, int max_wg_size,
            size_t l3_cache_size, attr_t attr);

    ngen::Product product() const;
    ngen::ProductFamily family() const;
    int stepping() const;
    ngen::HW ngen_hw() const { return hw_; }
    operator ngen::HW() const { return hw_; }

    bool has_fp64_atomic_support() const {
        return any(attr_ & attr_t::atomic_fp64);
    }
    int eu_count() const { return eu_count_; }
    int large_grf_support() const { return any(attr_ & attr_t::large_grf); }
    int grf_size() const;
    int systolic_support() const { return any(attr_ & attr_t::systolic); }
    size_t l3_cache_size() const { return l3_cache_size_; }

    int max_tg_size(int regs, int simd) const;

    // Number of EUs per Xe core (maps to dual subslice on XeHPG).
    int eus_per_core() const;
    int threads_per_eu(int regs = 128) const;
    int cache_line_size() const;

    std::string str() const;
    std::string brief_str() const;
    void dump() const { printf("%s\n", str().c_str()); }

    bool operator<(ngen::HW rhs) const { return hw_ < rhs; }
    bool operator>(ngen::HW rhs) const { return hw_ > rhs; }
    bool operator<=(ngen::HW rhs) const { return hw_ <= rhs; }
    bool operator>=(ngen::HW rhs) const { return hw_ >= rhs; }
    bool operator==(ngen::HW rhs) const { return hw_ == rhs; }
    bool operator!=(ngen::HW rhs) const { return hw_ != rhs; }
#if __cplusplus >= 202002L
    bool operator==(const hw_t &other) const = default;
#endif

protected:
    // Memory for storing ngen::Product to avoid nGEN header dependency in IR
    struct alignas(int) product_t {
        unsigned char data[12] = {};
        product_t() = default;
        product_t(const ngen::Product &product);
        const ngen::Product &operator()() const;
#if __cplusplus >= 202002L
        bool operator==(const product_t &other) const = default;
#endif
    };

    product_t product_;

private:
    int max_wg_size(int regs = 128) const {
        bool is_large_grf = (regs > 128);
        return is_large_grf ? max_wg_size_ / 2 : max_wg_size_;
    }

    ngen::HW hw_ = {};
    int eu_count_ = 0;
    int max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;
    attr_t attr_ = attr_t::none;
};

} // namespace dsl

namespace hw {
using attr_t = dsl::hw::attr_t;
}
using hw_t = dsl::hw_t;
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
