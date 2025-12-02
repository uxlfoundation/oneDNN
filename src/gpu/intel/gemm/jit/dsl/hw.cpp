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

#include "gemmstone/dsl/hw.hpp"
#include "dsl/utils/utils.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

hw_t::hw_t(const ngen::Product &product, int eu_count, size_t max_wg_size,
        size_t l3_cache_size, attr_t attr)
    : product_(product)
    , hw_(ngen::getCore(product.family))
    , eu_count_(eu_count)
    , max_wg_size_(max_wg_size)
    , l3_cache_size_(l3_cache_size)
    , attr_(attr) {}

ngen::Product hw_t::product() const {
    ngen::Product product;
    std::memcpy(&product, &product_, sizeof(product));
    return product;
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

size_t hw_t::max_tg_size(int regs, int simd) const {
    size_t wg_size = max_wg_size(regs);
    size_t eu_based_tg_size
            = eus_per_core() * rounddown_pow2(threads_per_eu(regs));
    size_t wg_based_tg_size = wg_size / simd;
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
        default: stub(); return 8;
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
        default: stub(); return 8;
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
        default: stub();
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

hw_t::product_t::product_t(const ngen::Product &product) {
    static_assert(sizeof(product) == sizeof(*this),
            "ngen::Product and hw_t::product must be binary compatible");
    std::memcpy(this, &product, sizeof(product));
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
