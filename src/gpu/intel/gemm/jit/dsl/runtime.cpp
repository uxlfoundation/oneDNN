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

#include "gemmstone/runtime.hpp"
#include "gemmstone/dsl/hw.hpp"
#include "gemmstone/dsl/runtime.hpp"
#include "generator_dsl/builder.hpp"
#include "generator_dsl/kernel_desc.hpp"

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
#include "ngen_opencl.hpp"
#include <CL/cl_ext.h>
#endif

GEMMSTONE_NAMESPACE_START

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
std::string make_asm(const GEMMKernelDesc &desc) {
    if (desc.strategy.isDSLGenerator) {
        generator_dsl_desc_t dsl_desc(
                desc.problem, desc.strategy, desc.iface, desc.options);
        auto dsl_kernel = make_kernel(dsl_desc);
        return dsl::make_asm(dsl_kernel);
    }
    stub();
}
#endif

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
std::vector<uint8_t> make_binary(const GEMMKernelDesc &desc) {
    if (desc.strategy.isDSLGenerator) {
        generator_dsl_desc_t dsl_desc(
                desc.problem, desc.strategy, desc.iface, desc.options);
        auto dsl_kernel = make_kernel(dsl_desc);
        return dsl::make_binary(dsl_kernel);
    }
    stub();
}
#endif

#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
::sycl::kernel make_kernel(const GEMMKernelDesc &desc, sycl::device device,
        sycl::context context) {
    if (desc.strategy.isDSLGenerator) {
        generator_dsl_desc_t dsl_desc(
                desc.problem, desc.strategy, desc.iface, desc.options);
        auto dsl_kernel = make_kernel(dsl_desc);
        return dsl::make_kernel(dsl_kernel, context, device);
    }
    stub();
}
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME

#ifndef CL_DEVICE_FEATURE_CAPABILITIES_INTEL
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL 0x4256
#endif

#ifndef CL_DEVICE_ATOMIC_FLAGS
#define CL_DEVICE_ATOMIC_FLAGS
#define CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT (1 << 0)
#define CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT (1 << 1)
#define CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT (1 << 2)
#define CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT (1 << 16)
#define CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT (1 << 17)
#define CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT (1 << 18)
#endif

#ifndef CL_DEVICE_FEATURE_FLAG_DPAS_INTEL
#define CL_DEVICE_FEATURE_FLAG_DPAS_INTEL (1 << 1)
#endif

dsl::hw_t get_hardware(cl_device_id device, cl_context context) {
    auto product = ngen::OpenCLCodeGenerator<ngen::HW::Unknown>::detectHWInfo(
            context, device);

    cl_int err;
    cl_uint eu_count = 0;
    err = ngen::dynamic::clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(eu_count), &eu_count, nullptr);
    if (err) return {};

    size_t max_wg_size;
    err = ngen::dynamic::clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(max_wg_size), &max_wg_size, nullptr);
    if (err) return {};

    cl_ulong l3_cache_size;
    err = ngen::dynamic::clGetDeviceInfo(device,
            CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(l3_cache_size),
            &l3_cache_size, nullptr);
    if (err) return {};

    dsl::hw::attr_t attr;
    cl_bitfield attrs_cl;
    err = ngen::dynamic::clGetDeviceInfo(device,
            CL_DEVICE_FEATURE_CAPABILITIES_INTEL, sizeof(cl_bitfield),
            &attrs_cl, nullptr);
    if (err) return {};

    if (ngen::getCore(product.family) >= ngen::HW::XeHPC)
        attr |= dsl::hw::attr_t::large_grf;

    if (attrs_cl & CL_DEVICE_FEATURE_FLAG_DPAS_INTEL)
        attr |= dsl::hw::attr_t::systolic;
    if (attrs_cl & CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT
            && product.family != ngen::ProductFamily::ARL
            && product.family != ngen::ProductFamily::MTL)
        attr |= dsl::hw::attr_t::atomic_fp64;

    return dsl::hw_t(product, eu_count, max_wg_size, l3_cache_size, attr);
}

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
std::vector<uint8_t> make_binary(
        const GEMMKernelDesc &desc, cl_device_id device, cl_context context) {
    if (desc.strategy.isDSLGenerator) {
        generator_dsl_desc_t dsl_desc(
                desc.problem, desc.strategy, desc.iface, desc.options);
        if (dsl_desc.options.hw() == dsl::hw_t())
            dsl_desc.options.set_hw(get_hardware(device, context));
        auto dsl_kernel = make_kernel(dsl_desc);
        return dsl::make_binary(dsl_kernel);
    }
    stub();
}
#endif

cl_kernel make_kernel(
        const GEMMKernelDesc &desc, cl_device_id device, cl_context context) {
    if (desc.strategy.isDSLGenerator) {
        generator_dsl_desc_t dsl_desc(
                desc.problem, desc.strategy, desc.iface, desc.options);
        if (dsl_desc.options.hw() == dsl::hw_t())
            dsl_desc.options.set_hw(get_hardware(device, context));
        auto dsl_kernel = make_kernel(dsl_desc);
        return dsl::make_kernel(dsl_kernel, context, device);
    }
    stub();
}
#endif

GEMMSTONE_NAMESPACE_END
