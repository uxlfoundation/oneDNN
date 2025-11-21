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

#include <unordered_map>

#include "dsl/utils/utils.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

const std::unordered_map<ngen::HW, std::string> &hw_names() {
    static const std::unordered_map<ngen::HW, std::string> names {
            {ngen::HW::Unknown, "unknown"},
            {ngen::HW::Gen9, "Gen9"},
            {ngen::HW::Gen10, "Gen10"},
            {ngen::HW::Gen11, "Gen11"},
            {ngen::HW::XeLP, "XeLP"},
            {ngen::HW::XeHP, "XeHP"},
            {ngen::HW::XeHPG, "XeHPG"},
            {ngen::HW::XeHPC, "XeHPC"},
            {ngen::HW::Xe2, "Xe2"},
            {ngen::HW::Xe3, "Xe3"},
    };
    return names;
}

const std::string &to_string(ngen::HW hw) {
    auto ret = hw_names().find(hw);
    if (ret == hw_names().end()) stub();
    return ret->second;
}

const std::unordered_map<ngen::ProductFamily, std::string> &
product_family_names() {
    static const std::unordered_map<ngen::ProductFamily, std::string> names {
            {ngen::ProductFamily::Unknown, "unknown"},
            {ngen::ProductFamily::GenericGen9, "Gen9"},
            {ngen::ProductFamily::GenericGen10, "Gen10"},
            {ngen::ProductFamily::GenericGen11, "Gen11"},
            {ngen::ProductFamily::GenericXeLP, "XeLP"},
            {ngen::ProductFamily::GenericXeHP, "XeHP"},
            {ngen::ProductFamily::GenericXeHPG, "XeHPG"},
            {ngen::ProductFamily::DG2, "DG2"},
            {ngen::ProductFamily::MTL, "MTL"},
            {ngen::ProductFamily::ARL, "ARL"},
            {ngen::ProductFamily::GenericXeHPC, "XeHPC"},
            {ngen::ProductFamily::PVC, "PVC"},
            {ngen::ProductFamily::GenericXe2, "Xe2"},
            {ngen::ProductFamily::GenericXe3, "Xe3"},
    };
    return names;
}

const std::string &to_string(ngen::ProductFamily family) {
    auto ret = product_family_names().find(family);
    if (ret == product_family_names().end()) stub();
    return ret->second;
}

static const std::unordered_map<ngen::PlatformType, std::string> platform_types
        = {
                {ngen::PlatformType::Unknown, "Unknown"},
                {ngen::PlatformType::Integrated, "Integrated"},
                {ngen::PlatformType::Discrete, "Discrete"},
};

const std::string &to_string(ngen::PlatformType type) {
    auto ret = platform_types.find(type);
    if (ret == platform_types.end()) stub();
    return ret->second;
}

std::string to_string(const ngen::Product &product) {
    return to_string(product.family) + ": platform - " + to_string(product.type)
            + ", stepping - " + std::to_string(product.stepping);
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
