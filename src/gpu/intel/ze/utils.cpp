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

#include "gpu/intel/ze/utils.hpp"

#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"

#include "ngen_level_zero.hpp"

#include "level_zero/ze_intel_gpu.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

namespace {

status_t get_ze_device_enabled_systolic_intel(
        ze_device_handle_t device, bool &mayiuse_systolic) {
    // Note: supported by Intel Driver 24.05 and onwards
    auto deviceModPropsExt = ze_intel_device_module_dp_exp_properties_t();
    deviceModPropsExt.stype
            = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES;

    auto deviceModProps = ze_device_module_properties_t();
    deviceModProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    deviceModProps.pNext = &deviceModPropsExt;

    CHECK(xpu::ze::zeDeviceGetModuleProperties(device, &deviceModProps));
    mayiuse_systolic
            = deviceModPropsExt.flags & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS;
    return status::success;
}

status_t get_ze_device_enabled_native_float_atomics(
        ze_device_handle_t device, uint64_t &native_extensions) {
    using namespace gpu::intel::compute;

    auto fltAtom = ze_float_atomic_ext_properties_t();
    fltAtom.stype = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;

    auto deviceProps = ze_device_module_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    deviceProps.pNext = &fltAtom;

    CHECK(xpu::ze::zeDeviceGetModuleProperties(device, &deviceProps));

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

status_t get_device_ip(ze_device_handle_t device, uint32_t &ip_version) {
    auto devicePropsIP = ze_device_ip_version_ext_t();
    devicePropsIP.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;

    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &devicePropsIP;

    CHECK(xpu::ze::zeDeviceGetProperties(device, &deviceProps));
    ip_version = devicePropsIP.ipVersion;
    return status::success;
}

} // namespace

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
    if (get_ze_device_enabled_systolic_intel(device, mayiuse_systolic)
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

    CHECK(get_ze_device_enabled_native_float_atomics(
            device, native_extensions));

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels, engine);
    if (status != status::success) mayiuse_ngen_kernels = false;

    ip_version = 0;
    CHECK(get_device_ip(device, ip_version));

    return status::success;
}

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
