/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef NGEN_LEVEL_ZERO_HPP
#define NGEN_LEVEL_ZERO_HPP

#include "ngen_config_internal.hpp"

#include "level_zero/ze_api.h"

#include <sstream>

#include "ngen_elf.hpp"
#include "ngen_interface.hpp"

#ifndef NGEN_LINK_L0
#include "ngen_dynamic.hpp"
#endif

namespace NGEN_NAMESPACE {

// Exceptions.
class level_zero_error : public std::runtime_error {
public:
    level_zero_error(ze_result_t status_ = ZE_RESULT_SUCCESS) : std::runtime_error("A Level Zero error occurred."), status(status_) {}
protected:
    ze_result_t status;
};

// Dynamic loading support.
// By default L0 is loaded dynamically, but direct linking is also possible
//   by #defining the NGEN_LINK_L0 macro.
namespace dynamic {

#ifdef _WIN32
#define NGEN_L0_LIB "ze_loader.dll"
#else
#define NGEN_L0_LIB "libze_loader.so.1"
#endif

#ifdef NGEN_LINK_L0
#define NGEN_L0_INDIRECT_API(f) using ::f;
#else
template <typename F>
F findL0Symbol(const char *symbol) {
    auto f = (F) findSymbol(NGEN_L0_LIB, symbol);
    if (!f) throw level_zero_error{ZE_RESULT_ERROR_UNINITIALIZED};
    return f;
}

#define NGEN_L0_INDIRECT_API(f) \
    template <typename... Args> ze_result_t f(Args&&... args) { \
        static auto f_ = findL0Symbol<decltype(&::f)>(#f);      \
        return f_(std::forward<Args>(args)...);                 \
    }
#endif

NGEN_L0_INDIRECT_API(zeDeviceGetProperties)
NGEN_L0_INDIRECT_API(zeModuleCreate)
NGEN_L0_INDIRECT_API(zeModuleDestroy)
NGEN_L0_INDIRECT_API(zeModuleGetNativeBinary)
NGEN_L0_INDIRECT_API(zeKernelCreate)

#undef NGEN_L0_INDIRECT_API

} // namespace dynamic

// Level Zero program generator class.
template <HW hw>
class LevelZeroCodeGenerator : public ELFCodeGenerator<hw>
{
public:
    explicit LevelZeroCodeGenerator(Product product_, DebugConfig debugConfig = {}) : ELFCodeGenerator<hw>(product_, debugConfig) {
        this->interface_.setInlineGRFCount(0);
    }

    explicit LevelZeroCodeGenerator(int stepping_ = 0, DebugConfig debugConfig = {}) : LevelZeroCodeGenerator({genericProductFamily(hw), stepping_, PlatformType::Unknown}, debugConfig) {}

    explicit LevelZeroCodeGenerator(DebugConfig debugConfig) : LevelZeroCodeGenerator({genericProductFamily(hw), 0}, debugConfig) {}

    inline ze_module_handle_t getModule(ze_context_handle_t context, ze_device_handle_t device, const std::string &options = "");
    static inline HW detectHW(ze_context_handle_t context, ze_device_handle_t device);
    static inline Product detectHWInfo(ze_context_handle_t context, ze_device_handle_t device);

    static bool binaryIsZebin() { return true; }

};

#define NGEN_FORWARD_LEVEL_ZERO(hw) NGEN_FORWARD_ELF(hw)

namespace detail {

static inline void handleL0(ze_result_t result)
{
    if (result != ZE_RESULT_SUCCESS)
        throw level_zero_error{result};
}

static inline std::vector<uint8_t> getDummyModuleBinary(ze_context_handle_t context, ze_device_handle_t device) {
    static const uint8_t dummySPV[] = {0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0E, 0x00, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0B, 0x00, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4F, 0x70, 0x65, 0x6E, 0x43, 0x4C, 0x2E, 0x73, 0x74, 0x64, 0x00, 0x00, 0x0E, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x5F, 0x00, 0x00, 0x00, 0x07, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x6B, 0x65, 0x72, 0x6E, 0x65, 0x6C, 0x5F, 0x61, 0x72, 0x67, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x2E, 0x5F, 0x2E, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x70, 0x8E, 0x01, 0x00, 0x05, 0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x6E, 0x74, 0x72, 0x79, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xF8, 0x00, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00, 0xFD, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00};
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_IL_SPIRV,
        sizeof(dummySPV),
        dummySPV,
        nullptr,
        nullptr
    };

    ze_module_handle_t module;
    handleL0(dynamic::zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    if (module == nullptr)
        throw level_zero_error{};

    std::vector<uint8_t> binary;
    size_t binarySize;
    handleL0(dynamic::zeModuleGetNativeBinary(module, &binarySize, nullptr));
    binary.resize(binarySize);
    handleL0(dynamic::zeModuleGetNativeBinary(module, &binarySize, binary.data()));
    handleL0(dynamic::zeModuleDestroy(module));
    return binary;
}

}; /* namespace detail */

template <HW hw>
ze_module_handle_t LevelZeroCodeGenerator<hw>::getModule(ze_context_handle_t context, ze_device_handle_t device, const std::string &options)
{
    using super = ELFCodeGenerator<hw>;

    auto binary = super::getBinary();

    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_NATIVE,
        binary.size(),
        binary.data(),
        options.c_str(),
        nullptr
    };

    ze_module_handle_t module;
    detail::handleL0(dynamic::zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    if (module == nullptr)
        throw level_zero_error{};

    return module;
}

template <HW hw>
HW LevelZeroCodeGenerator<hw>::detectHW(ze_context_handle_t context, ze_device_handle_t device)
{
    return getCore(detectHWInfo(context, device).family);
}

template <HW hw>
Product LevelZeroCodeGenerator<hw>::detectHWInfo(ze_context_handle_t context, ze_device_handle_t device)
{
    Product product;

    ze_device_properties_t dprop = {};
    dprop.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

#ifdef ZE_DEVICE_IP_VERSION_EXT_NAME
    // Try ZE_extension_device_ip_version first if available.
    ze_device_ip_version_ext_t vprop = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT, nullptr, 0};
    dprop.pNext = &vprop;

    if (dynamic::zeDeviceGetProperties(device, &dprop) == ZE_RESULT_SUCCESS) {
        return npack::decodeHWIPVersion(vprop.ipVersion);
    } else
#endif
    {
        auto binary = detail::getDummyModuleBinary(context, device);

        product = ELFCodeGenerator<hw>::getBinaryHWInfo(binary);
        dprop.pNext = nullptr;
        detail::handleL0(dynamic::zeDeviceGetProperties(device, &dprop));
    }

    product.type = (dprop.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? PlatformType::Integrated : PlatformType::Discrete;

    return product;
}

} /* namespace NGEN_NAMESPACE */

#endif
