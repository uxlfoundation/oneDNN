/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_HW_HPP
#define GPU_INTEL_JIT_IR_HW_HPP

#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Provides access to HW configuration which includes non-configurable
// properties.

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

class hw_t {
public:
    using attr_t = hw::attr_t;
    hw_t() = default;
    explicit hw_t(const ngen::Product &product, int eu_count, int max_wg_size,
            size_t l3_cache_size, attr_t attr);
    ngen::HW ngen_hw() const { return hw_; }
    operator ngen::HW() const { return hw_; }
    const ngen::Product &product() const;

    bool is_undef() const { return hw_ == ngen::HW::Unknown; }
    bool has_fp64_atomic_support() const {
        return any(attr_ & attr_t::atomic_fp64);
    }
    ngen::ProductFamily family() const;
    int stepping() const;
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

    IR_DEFINE_DUMP()

    bool operator<(ngen::HW rhs) const { return hw_ < rhs; }
    bool operator>(ngen::HW rhs) const { return hw_ > rhs; }
    bool operator<=(ngen::HW rhs) const { return hw_ <= rhs; }
    bool operator>=(ngen::HW rhs) const { return hw_ >= rhs; }
    bool operator==(ngen::HW rhs) const { return hw_ == rhs; }
    bool operator!=(ngen::HW rhs) const { return hw_ != rhs; }
#if __cplusplus >= 202002L
    bool operator==(const hw_t &other) const = default;
#endif

private:
    int max_wg_size(int regs = 128) const {
        bool is_large_grf = (regs > 128);
        return is_large_grf ? max_wg_size_ / 2 : max_wg_size_;
    }

    ngen::HW hw_ = ngen::HW::Unknown;
    ngen::Product product_ = {};
    int eu_count_ = 0;
    int max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;
    attr_t attr_ = attr_t::none;
};

inline hw_t make_ir_hw(const impl::engine_t *engine) {
    using namespace compute;
    auto intel_engine = utils::downcast<const engine_t *>(engine);

    auto *device_info = intel_engine->device_info();
    auto product = get_ngen_product(*device_info);
    int eu_count = device_info->eu_count();
    int max_wg_size = static_cast<int>(
            device_info->max_wg_size(/*large_grf_mode=*/false));
    size_t l3_cache_size = device_info->l3_cache_size();
    hw::attr_t attr = hw::attr_t::none;
    if (intel_engine->mayiuse_large_grf_mode()) attr |= hw::attr_t::large_grf;
    if (device_info->mayiuse_systolic()) attr |= hw_t::attr_t::systolic;
    if (device_info->mayiuse_float_atomic_add(data_type::f64))
        attr |= hw_t::attr_t::atomic_fp64;

    return hw_t(product, eu_count, max_wg_size, l3_cache_size, attr);
}

inline bool prefer_large_grf(
        const hw_t &hw, const gpu_primitive_attr_t *gpu_attr) {
    if (!gpu_attr || !hw.large_grf_support()) return false;
    return gpu_attr->threads_per_eu() * 2 == hw.threads_per_eu();
}

class exec_config_t {
public:
    exec_config_t() = default;
    exec_config_t(const hw_t &hw) : hw_(hw) {}
    exec_config_t(const hw_t &hw, int regs, int simd)
        : hw_(hw), regs_(regs), simd_(simd) {}

    const hw_t &hw() const { return hw_; }
    int regs() const { return regs_; }
    int simd() const { return simd_; }
    int grf_size() const { return hw_.grf_size(); }
    void set_regs(int regs) { regs_ = regs; }
    void set_simd(int simd) { simd_ = simd; }

    std::string str() const {
        ostringstream_t oss;
        oss << hw_.str();
        oss << ", SIMD: " << simd();
        oss << ", regs: " << regs();
        return oss.str();
    }

private:
    hw_t hw_;
    int regs_ = 0;
    int simd_ = 0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
