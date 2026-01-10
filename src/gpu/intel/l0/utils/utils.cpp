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

#include "gpu/intel/l0/utils/utils.hpp"
#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "ngen_level_zero.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

event_wrapper_t::event_wrapper_t(ze_event_handle_t event) : event_(event) {}

event_wrapper_t::~event_wrapper_t() {
    if (event_) {
        l0::zeEventHostSynchronize(event_, UINT64_MAX);
        l0::zeEventDestroy(event_);
    }
}

event_wrapper_t::operator ze_event_handle_t() const {
    return event_;
}

event_pool_wrapper_t::event_pool_wrapper_t(ze_event_pool_handle_t event_pool)
    : event_pool_(event_pool) {}

event_pool_wrapper_t::~event_pool_wrapper_t() {
    if (event_pool_) l0::zeEventPoolDestroy(event_pool_);
}

event_pool_wrapper_t::operator ze_event_pool_handle_t() const {
    return event_pool_;
}

module_wrapper_t::module_wrapper_t(ze_module_handle_t module)
    : module_(module) {}

module_wrapper_t::~module_wrapper_t() {
    if (module_) l0::zeModuleDestroy(module_);
}

module_wrapper_t::operator ze_module_handle_t() const {
    return module_;
}

status_t get_device_ip(ze_device_handle_t device, uint32_t &ip_version) {
    ze_device_ip_version_ext_t device_ip_version_ext = {};
    device_ip_version_ext.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;
    device_ip_version_ext.pNext = nullptr;

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &device_ip_version_ext;

    CHECK(l0::zeDeviceGetProperties(device, &device_properties));

    ip_version = device_ip_version_ext.ipVersion;

    return status::success;
}

status_t get_l0_device_enabled_systolic_intel(
        ze_device_handle_t device, bool &mayiuse_systolic) {
    ze_intel_device_module_dp_exp_properties_t
            intel_device_module_dp_exp_properties
            = {};
    intel_device_module_dp_exp_properties.stype
            = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES;
    intel_device_module_dp_exp_properties.pNext = nullptr;

    ze_device_module_properties_t device_module_properties = {};
    device_module_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    device_module_properties.pNext = &intel_device_module_dp_exp_properties;

    CHECK(l0::zeDeviceGetModuleProperties(device, &device_module_properties));

    mayiuse_systolic = intel_device_module_dp_exp_properties.flags
            & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS;

    return status::success;
}

status_t get_l0_device_enabled_native_float_atomics(
        ze_device_handle_t device, uint64_t &native_extensions) {
    using namespace gpu::intel::compute;

    ze_float_atomic_ext_properties_t float_atomic_ext_properties = {};
    float_atomic_ext_properties.stype
            = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;
    float_atomic_ext_properties.pNext = nullptr;

    ze_device_module_properties_t device_module_properties = {};
    device_module_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    device_module_properties.pNext = &float_atomic_ext_properties;

    CHECK(l0::zeDeviceGetModuleProperties(device, &device_module_properties));

    ze_device_fp_atomic_ext_flags_t atomic_load_store
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE;
    ze_device_fp_atomic_ext_flags_t atomic_add
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD;
    ze_device_fp_atomic_ext_flags_t atomic_min_max
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX;

    if ((float_atomic_ext_properties.fp16Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_load_store;
    if ((float_atomic_ext_properties.fp16Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_add;
    if ((float_atomic_ext_properties.fp16Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_min_max;

    if ((float_atomic_ext_properties.fp32Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_load_store;
    if ((float_atomic_ext_properties.fp32Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_add;
    if ((float_atomic_ext_properties.fp32Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_min_max;

    if ((float_atomic_ext_properties.fp64Flags & atomic_load_store)
            == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_load_store;
    if ((float_atomic_ext_properties.fp64Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_add;
    if ((float_atomic_ext_properties.fp64Flags & atomic_min_max)
            == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_min_max;

    return status::success;
}

status_t get_l0_device_eu_count(ze_device_handle_t device, int &eu_count) {
    ze_eu_count_ext_t eu_count_ext = {};
    eu_count_ext.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;
    eu_count_ext.pNext = nullptr;

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &eu_count_ext;

    CHECK(l0::zeDeviceGetProperties(device, &device_properties));

    eu_count = eu_count_ext.numTotalEUs;

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

xpu::device_uuid_t get_device_uuid(const ze_device_handle_t device) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = nullptr;

    auto status = l0::zeDeviceGetProperties(device, &device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &device_id = device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)device_id[i]) << shift);
    }

    return xpu::device_uuid_t(uuid[0], uuid[1]);
}

status_t get_device_index(const ze_device_handle_t device, size_t *index) {
    uint32_t driver_count = 0;
    CHECK(l0::zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(l0::zeDriverGet(&driver_count, drivers.data()));

    uint32_t device_count = 0;
    CHECK(l0::zeDeviceGet(drivers[0], &device_count, nullptr));

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(l0::zeDeviceGet(drivers[0], &device_count, devices.data()));

    for (size_t i = 0; i < device_count; i++) {
        if (device == devices[i]) {
            *index = i;

            return status::success;
        }
    }

    return status::invalid_arguments;
}

std::string get_kernel_name(const ze_kernel_handle_t kernel) {
    std::string kernel_name;

    size_t kernel_name_size = 0;
    l0::zeKernelGetName(kernel, &kernel_name_size, nullptr);

    kernel_name.resize(kernel_name_size, 0);
    l0::zeKernelGetName(kernel, &kernel_name_size, &kernel_name[0]);

    // Remove the null terminator as std::string already includes it
    kernel_name.resize(kernel_name_size - 1);

    return kernel_name;
}

status_t get_module_binary(
        const ze_module_handle_t module, xpu::binary_t &binary) {
    size_t module_binary_size;
    CHECK(l0::zeModuleGetNativeBinary(module, &module_binary_size, nullptr));

    binary.resize(module_binary_size);
    CHECK(l0::zeModuleGetNativeBinary(
            module, &module_binary_size, binary.data()));

    return status::success;
}

status_t get_kernel_binary(
        const ze_kernel_handle_t kernel, xpu::binary_t &binary) {
    size_t binary_size = 0;
    CHECK(l0::zeKernelGetBinaryExp(kernel, &binary_size, nullptr));

    binary.resize(binary_size);
    CHECK(l0::zeKernelGetBinaryExp(kernel, &binary_size, binary.data()));

    return status::success;
}

static inline ze_result_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    const ze_init_flags_t default_ze_flags = 0;
    static auto init_ = find_ze_symbol<decltype(&::zeInit)>("zeInit");
    if (!init_) return ZE_RESULT_ERROR_NOT_AVAILABLE;
    init_(default_ze_flags);

    static auto f_
            = find_ze_symbol<decltype(&::zeModuleCreate)>("zeModuleCreate");
    if (!f_) return ZE_RESULT_ERROR_NOT_AVAILABLE;
    return f_(hContext, hDevice, desc, phModule, phBuildLog);
}

#define ZE_MODULE_FORMAT_OCLC (ze_module_format_t)3U
static inline ze_module_handle_t compile_ocl_module(
        const ze_device_handle_t device, const ze_context_handle_t context,
        const std::string &code, const std::string &options) {
    ze_module_desc_t module_desc;
    module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_desc.pNext = nullptr;
    module_desc.format = ZE_MODULE_FORMAT_OCLC;
    module_desc.inputSize = code.size();
    module_desc.pInputModule = reinterpret_cast<const uint8_t *>(code.c_str());
    module_desc.pBuildFlags = options.c_str();
    module_desc.pConstants = nullptr;

    ze_module_handle_t module_handle;
    ze_module_build_log_handle_t module_build_log_handle;
    ze_result_t ret = func_zeModuleCreate(context, device, &module_desc,
            &module_handle, &module_build_log_handle);
    if (ret != ZE_RESULT_SUCCESS) return nullptr;
    return module_handle;
}

bool mayiuse_microkernels(const ze_device_handle_t device,
        const ze_context_handle_t context, const std::string &code) {
    ze_module_handle_t module_handle
            = compile_ocl_module(device, context, code, "");
    if (module_handle) {
        l0::zeModuleDestroy(module_handle);
        return true;
    }
    return false;
}

status_t compile_ocl_module_to_binary(const ze_device_handle_t device,
        const ze_context_handle_t context, const std::string &code,
        const std::string &options, xpu::binary_t &binary) {
    ze_module_handle_t module_handle
            = compile_ocl_module(device, context, code, options);
    if (!module_handle) { return status::runtime_error; }
    CHECK(l0::get_module_binary(module_handle, binary));
    CHECK(l0::zeModuleDestroy(module_handle));

    return status::success;
}

status_t create_kernels(const ze_device_handle_t device,
        const ze_context_handle_t context,
        const std::vector<const char *> &kernel_names,
        const xpu::binary_t &binary, ze_module_handle_t *module,
        std::vector<ze_kernel_handle_t> &kernels) {
    ze_module_desc_t module_desc;
    module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_desc.pNext = nullptr;
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = binary.size();
    module_desc.pInputModule = binary.data();
    module_desc.pBuildFlags = "";
    module_desc.pConstants = nullptr;

    CHECK(l0::zeModuleCreate(context, device, &module_desc, module, nullptr));

    kernels.resize(kernel_names.size(), nullptr);
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (kernel_names[i] == nullptr) {
            kernels[i] = nullptr;
            continue;
        }

        ze_kernel_desc_t kernel_desc = {};
        kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        kernel_desc.pNext = nullptr;
        kernel_desc.flags = 0;
        kernel_desc.pKernelName = kernel_names[i];

        ze_kernel_handle_t kernel;
        CHECK(l0::zeKernelCreate(*module, &kernel_desc, &kernel));

        kernels[i] = kernel;
    }

    return status::success;
}

ze_memory_type_t get_pointer_type(
        const ze_context_handle_t context, const void *ptr) {
    ze_memory_allocation_properties_t memory_allocation_properties;
    memory_allocation_properties.stype
            = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    memory_allocation_properties.pNext = nullptr;

    l0::zeMemGetAllocProperties(
            context, ptr, &memory_allocation_properties, nullptr);

    return memory_allocation_properties.type;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
