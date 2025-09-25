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

#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

hw_t::hw_t(const ngen::Product &product, int eu_count, int max_wg_size,
        size_t l3_cache_size, attr_t attr)
    : hw_(ngen::getCore(product.family))
    , product_(product)
    , eu_count_(eu_count)
    , max_wg_size_(max_wg_size)
    , l3_cache_size_(l3_cache_size)
    , attr_(attr) {}

const ngen::Product &hw_t::product() const {
    return product_;
}

ngen::ProductFamily hw_t::family() const {
    return product().family;
}

int hw_t::stepping() const {
    return product().stepping;
}

int hw_t::grf_size() const {
    return ngen::GRF::bytes(hw_);
}

int hw_t::max_tg_size(int regs, int simd) const {
    int wg_size = max_wg_size(regs);
    int eu_based_tg_size
            = eus_per_core() * utils::rnd_down_pow2(threads_per_eu(regs));
    int wg_based_tg_size = wg_size / simd;
    return std::min(eu_based_tg_size, wg_based_tg_size);
}
int hw_t::eus_per_core() const {
    switch (hw_) {
        case ngen::HW::XeHPG: return 16;
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return 8;
        default: gpu_error_not_expected(); return 8;
    }
}
int hw_t::threads_per_eu(int regs) const {
    bool is_large_grf = (regs > 128);
    switch (hw_) {
        case ngen::HW::XeLP: return 7;
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return is_large_grf ? 4 : 8;
        default: gpu_error_not_expected(); return 8;
    }
}

int hw_t::cache_line_size() const {
    switch (hw_) {
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return 64;
        default: gpu_error_not_expected();
    }
    return 0;
}

std::string hw_t::str() const {
    ostringstream_t oss;
    oss << to_string(hw_);
    oss << ", stepping: " << stepping();
    oss << ", EUs: " << eu_count();
    return oss.str();
}

std::string hw_t::brief_str() const {
    return ir_utils::to_lower(to_string(hw_));
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
