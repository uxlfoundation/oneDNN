/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/intel/sycl/l0/utils.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "ngen_level_zero.hpp"

#include "level_zero/ze_api.h"
#include "level_zero/ze_intel_gpu.h"

#if !defined(__SYCL_COMPILER_VERSION)
#error "Unsupported compiler"
#endif

#if (__SYCL_COMPILER_VERSION < 20200818)
#error "Level Zero is not supported with this compiler version"
#endif

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "gpu/intel/sycl/utils.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "gpu/intel/sycl/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {
namespace l0 {

std::string to_string(ze_result_t r) {
#define ZE_STATUS_CASE(status) \
    case status: return #status
    switch (r) {
        ZE_STATUS_CASE(ZE_RESULT_SUCCESS);
        ZE_STATUS_CASE(ZE_RESULT_NOT_READY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNINITIALIZED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNKNOWN);
        ZE_STATUS_CASE(ZE_RESULT_FORCE_UINT32);
        default: return std::to_string((int)r);
    }
#undef ZE_STATUS_CASE
}

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = to_string(res_); \
            VERROR(common, level_zero, "errcode %s", err_str_.c_str()); \
            return status::runtime_error; \
        } \
    } while (false)

#if defined(_WIN32)
#define L0_LIB_NAME "ze_loader.dll"
#elif defined(__linux__)
#define L0_LIB_NAME "libze_loader.so.1"
#endif

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)xpu::find_symbol(L0_LIB_NAME, symbol);
}
#undef L0_LIB_NAME

#define INDIRECT_L0_CALL(f) \
    template <typename... Args> \
    status_t f(Args &&...args) { \
        const ze_init_flags_t default_ze_flags = 0; \
        static auto init_ = find_ze_symbol<decltype(&::zeInit)>("zeInit"); \
        if (!init_) return status::runtime_error; \
        ZE_CHECK(init_(default_ze_flags)); \
        static auto f_ = find_ze_symbol<decltype(&::f)>(#f); \
        if (!f_) return status::runtime_error; \
        ZE_CHECK(f_(std::forward<Args>(args)...)); \
        return status::success; \
    }
INDIRECT_L0_CALL(zeModuleCreate)
INDIRECT_L0_CALL(zeDeviceGetProperties)
INDIRECT_L0_CALL(zeDeviceGetModuleProperties)
INDIRECT_L0_CALL(zeKernelCreate)
INDIRECT_L0_CALL(zeKernelGetBinaryExp)
INDIRECT_L0_CALL(zeModuleGetNativeBinary)
#undef INDIRECT_L0_CALL

} // namespace l0

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
xpu::device_uuid_t get_device_uuid(const ::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    auto ze_device_properties = ze_device_properties_t();
    ze_device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    auto ze_device = xpu::sycl::compat::get_native<ze_device_handle_t>(dev);
    auto status = l0::zeDeviceGetProperties(ze_device, &ze_device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)ze_device_id[i]) << shift);
    }
    return xpu::device_uuid_t(uuid[0], uuid[1]);
}

status_t sycl_create_kernels_with_level_zero(
        std::vector<std::unique_ptr<::sycl::kernel>> &sycl_kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device = xpu::sycl::compat::get_native<ze_device_handle_t>(
            sycl_engine->device());
    auto ze_ctx = xpu::sycl::compat::get_native<ze_context_handle_t>(
            sycl_engine->context());

    CHECK(l0::zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    ::sycl::kernel_bundle<::sycl::bundle_state::executable> kernel_bundle
            = ::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                    ::sycl::bundle_state::executable>(
                    {ze_module}, sycl_engine->context());

    sycl_kernels.resize(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (kernel_names[i] == nullptr) continue;
        ze_kernel_handle_t ze_kernel;
        ze_kernel_desc_t ze_kernel_desc {
                ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_names[i]};
        CHECK(l0::zeKernelCreate(ze_module, &ze_kernel_desc, &ze_kernel));
        auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
                {kernel_bundle, ze_kernel}, sycl_engine->context());
        sycl_kernels[i] = utils::make_unique<::sycl::kernel>(k);
    }

    return status::success;
}

status_t get_l0_kernel_binary(
        const ::sycl::kernel &kernel, xpu::binary_t &binary) {
#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
    auto l0_kernel = ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
            kernel);
    size_t binary_size = 0;
    CHECK(l0::zeKernelGetBinaryExp(l0_kernel, &binary_size, nullptr));
    binary.resize(binary_size);
    CHECK(l0::zeKernelGetBinaryExp(l0_kernel, &binary_size, binary.data()));
#else
    auto bundle = kernel.get_kernel_bundle();
    auto module_vec
            = ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
                    bundle);
    auto module = module_vec[0];
    size_t module_binary_size;
    CHECK(l0::zeModuleGetNativeBinary(module, &module_binary_size, nullptr));
    binary.resize(module_binary_size);
    CHECK(l0::zeModuleGetNativeBinary(
            module, &module_binary_size, binary.data()));
#endif
    return status::success;
}

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(lhs);
    auto rhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(rhs);

    return lhs_ze_handle == rhs_ze_handle;
}

status_t get_device_ip(ze_device_handle_t device, uint32_t &ip_version) {
    auto devicePropsIP = ze_device_ip_version_ext_t();
    devicePropsIP.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;

    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &devicePropsIP;

    CHECK(l0::zeDeviceGetProperties(device, &deviceProps));
    ip_version = devicePropsIP.ipVersion;
    return status::success;
}

status_t get_l0_device_enabled_systolic_intel(
        ze_device_handle_t device, bool &mayiuse_systolic) {
    // Note: supported by Intel Driver 24.05 and onwards
    auto deviceModPropsExt = ze_intel_device_module_dp_exp_properties_t();
    deviceModPropsExt.stype
            = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES;

    auto deviceModProps = ze_device_module_properties_t();
    deviceModProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    deviceModProps.pNext = &deviceModPropsExt;

    CHECK(l0::zeDeviceGetModuleProperties(device, &deviceModProps));
    mayiuse_systolic
            = deviceModPropsExt.flags & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS;
    return status::success;
}

status_t get_l0_device_enabled_native_float_atomics(
        ze_device_handle_t device, uint64_t &native_extensions) {
    using namespace gpu::intel::compute;

    auto fltAtom = ze_float_atomic_ext_properties_t();
    fltAtom.stype = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;

    auto deviceProps = ze_device_module_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    deviceProps.pNext = &fltAtom;

    CHECK(l0::zeDeviceGetModuleProperties(device, &deviceProps));

    ze_device_fp_atomic_ext_flags_t atomic_load_store
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE;
    ze_device_fp_atomic_ext_flags_t atomic_add
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD;
    ze_device_fp_atomic_ext_flags_t atomic_min_max
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX;

    if ((fltAtom.fp16Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_load_store;
    if ((fltAtom.fp16Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_add;
    if ((fltAtom.fp16Flags & atomic_min_max) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_min_max;

    if ((fltAtom.fp32Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_load_store;
    if ((fltAtom.fp32Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_add;
    if ((fltAtom.fp32Flags & atomic_min_max) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_min_max;

    if ((fltAtom.fp64Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_load_store;
    if ((fltAtom.fp64Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_add;
    if ((fltAtom.fp64Flags & atomic_min_max) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_min_max;

    return status::success;
}

status_t get_l0_device_eu_count(ze_device_handle_t device, int &eu_count) {
    auto eucnt = ze_eu_count_ext_t();
    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &eucnt;

    CHECK(l0::zeDeviceGetProperties(device, &deviceProps));
    eu_count = eucnt.numTotalEUs;
    return status::success;
}

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, compute::gpu_product_t &product_,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels) {
    using namespace ngen;
    ngen::Product product = LevelZeroCodeGenerator<HW::Unknown>::detectHWInfo(
            context, device);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(ngen::getCore(product.family));
    std::memcpy(&product_, &product, sizeof(ngen::Product));

    mayiuse_systolic = false;
    if (get_l0_device_enabled_systolic_intel(device, mayiuse_systolic)
            != status::success)
        mayiuse_systolic = false;

    /* Some old drivers do not report systolic availability. Manually override
       systolic availability based on product family. */
    switch (product.family) {
        case ProductFamily::DG2:
        case ProductFamily::ARL:
        case ProductFamily::PVC: mayiuse_systolic = true;
        default: break;
    }

    CHECK(get_l0_device_enabled_native_float_atomics(
            device, native_extensions));

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels, engine);
    if (status != status::success) mayiuse_ngen_kernels = false;

    ip_version = 0;
    return get_device_ip(device, ip_version);
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
