/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GENERATOR_HPP
#define GPU_INTEL_JIT_GENERATOR_HPP

#include <memory>

#include "ngen.hpp"
#include "ngen_emulation.hpp"

#include "common/impl_registration.hpp"
#include "common/nstl.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/jit/generator_base.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "xpu/utils.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/interop_kernel.hpp"
#include "ngen_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/kernel.hpp"
#include "ngen_opencl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using gpu_gen_t = ngen::HW;
constexpr gpu_gen_t gpu_xe_lp = ngen::HW::XeLP;
constexpr gpu_gen_t gpu_xe_hp = ngen::HW::XeHP;
constexpr gpu_gen_t gpu_xe_hpg = ngen::HW::XeHPG;
constexpr gpu_gen_t gpu_xe_hpc = ngen::HW::XeHPC;
constexpr gpu_gen_t gpu_xe2 = ngen::HW::Xe2;
constexpr gpu_gen_t gpu_xe3 = ngen::HW::Xe3;

#if (!defined(NDEBUG) || defined(DNNL_DEV_MODE))
#define GENERATOR_NAME __FILE__
#define GENERATOR_LINE __LINE__
#else
#define GENERATOR_NAME "oneDNN"
#define GENERATOR_LINE 0
#endif

struct debug_config_t {
    const char *name;
    uint32_t line;
};

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
template <gpu_gen_t hw>
using ngen_code_generator_t = ngen::SYCLCodeGenerator<hw>;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
template <gpu_gen_t hw>
using ngen_code_generator_t = ngen::OpenCLCodeGenerator<hw>;
#endif

template <gpu_gen_t hw>
class generator_t : public ngen_code_generator_t<hw>, public generator_base_t {
private:
#ifdef DNNL_DEV_MODE
    static constexpr bool enable_debug_lines = true;
#else
    static constexpr bool enable_debug_lines = false;
#endif
public:
    generator_t(const debug_config_t &debug_config)
        : ngen_code_generator_t<hw>(0,
                {debug_config.name, debug_config.line, enable_debug_lines}) {};

    generator_t(
            const ngen::Product &product, const debug_config_t &debug_config)
        : ngen_code_generator_t<hw>(product,
                {debug_config.name, debug_config.line, enable_debug_lines}) {};

    const char *kernel_name() const override {
        return ngen_code_generator_t<hw>::getExternalName().c_str();
    }

    status_t get_kernel(compute::kernel_t &kernel,
            const compute::compute_engine_t *engine) override {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        auto *sycl_engine = utils::downcast<const sycl::engine_t *>(engine);
        auto sycl_kernel = ngen_code_generator_t<hw>::getKernel(
                sycl_engine->context(), sycl_engine->device());
        return sycl::interop_kernel_t::make(kernel, sycl_kernel, {});
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        auto *ocl_engine = utils::downcast<const ocl::engine_t *>(engine);
        auto ocl_kernel = ngen_code_generator_t<hw>::getKernel(
                ocl_engine->context(), ocl_engine->device());
        return ocl::kernel_t::make(kernel, ocl_kernel, {});
#endif
    }
};

void check_kernel_size(
        const std::string &kernel_name, size_t kernel_size, size_t icache_size);

template <template <ngen::HW> class KernelT, ngen::HW arch, typename... ArgsT>
std::unique_ptr<jit::generator_base_t> make_generator(
        const compute::device_info_t &device_info, ArgsT &&...args) {

    auto raw_kernel = new KernelT<arch>(std::forward<ArgsT>(args)...);
    check_kernel_size(raw_kernel->kernel_name(),
            raw_kernel->getRootStreamLength(), device_info.icache_size());
    return std::unique_ptr<jit::generator_base_t>(raw_kernel);
}

template <template <ngen::HW> class KernelT, typename... ArgsT>
compute::kernel_t make_kernel(gpu_primitive_t *primitive, bool register_kernel,
        impl::engine_t *engine, ArgsT &&...args) {
    using namespace compute;
    kernel_t kernel;

    if (primitive->cache_blob()) {
        status_t status = primitive->create_kernel(
                engine, &kernel, nullptr, register_kernel);
        if (status != status::success) return kernel_t();
        return kernel;
    }

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    auto arch = convert_dnnl_arch_to_ngen(device_info->gpu_arch());

    std::unique_ptr<jit::generator_base_t> jit_kernel;
#define CASE(gpu_arch) \
    case gpu_arch: \
        jit_kernel = make_generator<KernelT, gpu_arch>( \
                *device_info, std::forward<ArgsT>(args)...); \
        break;
    switch (arch) {
        REG_XELP_ISA(CASE(gpu_xe_lp));
        REG_XEHP_ISA(CASE(gpu_xe_hp));
        REG_XEHPG_ISA(CASE(gpu_xe_hpg));
        REG_XEHPC_ISA(CASE(gpu_xe_hpc));
        REG_XE2_ISA(CASE(gpu_xe2));
        REG_XE3_ISA(CASE(gpu_xe3));
        default: break;
    }
#undef CASE

    if (!jit_kernel) return kernel_t();

    status_t status = primitive->create_kernel(
            engine, &kernel, jit_kernel.get(), register_kernel);
    if (status != status::success) return kernel_t();
    return kernel;
}

template <template <ngen::HW> class KernelT, typename... ArgsT>
compute::kernel_t make_kernel(
        gpu_primitive_t *primitive, impl::engine_t *engine, ArgsT &&...args) {
    return make_kernel<KernelT>(primitive, /*register_kernel=*/true, engine,
            std::forward<ArgsT>(args)...);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_GENERATOR_HPP
