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

// Must be included before emulation.hpp
#include "ngen.hpp"

#include "common/impl_registration.hpp"
#include "common/nstl.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/jit/emulation.hpp"
#include "gpu/intel/jit/generator_base.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "xpu/utils.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "ngen_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/engine.hpp"
#include "ngen_opencl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using gpu_gen_t = ngen::HW;
constexpr gpu_gen_t gpu_gen9 = ngen::HW::Gen9;
constexpr gpu_gen_t gpu_gen11 = ngen::HW::Gen11;
constexpr gpu_gen_t gpu_xe_lp = ngen::HW::XeLP;
constexpr gpu_gen_t gpu_xe_hp = ngen::HW::XeHP;
constexpr gpu_gen_t gpu_xe_hpg = ngen::HW::XeHPG;
constexpr gpu_gen_t gpu_xe_hpc = ngen::HW::XeHPC;
constexpr gpu_gen_t gpu_xe2 = ngen::HW::Xe2;
constexpr gpu_gen_t gpu_xe3 = ngen::HW::Xe3;

// nGEN jit generator
//
// The main purpose of this header file is to provide extra features for nGEN
// kernel generator, e.g. additional macros and debugging capabilities.
//
// Jit generator provides additional memory to simplify kernel debugging. This
// memory is allocated using Shared Virtual Memory (SVM) feature in OpenCL 2.0.
// SVM enables the host and device portions of an OpenCL application to
// seamlessly share pointers and complex pointer-containing data-structures.
// This memory can be used to dump state of GPU registers or view GPU memory on
// the host in debugger.
//
// In order to use debug memory:
// 1.  Allocate it using 'void generator_t::dbg_alloc(cl_context context)'
// 2.  Get memory pointer using 'void* generator_t::dbg_memory()'
// 3.  Pass it as extra OpenCL kernel argument and define it as new argument in
//     kernel interface at corresponding order.
// 4.  Set a breakpoint after 'dnnl_stream_wait()', memory will be available on
//     the host side after kernel execution.
//
// A short example below demonstrates how to use debug memory:
//
//  ``` c++
//  status_t primitive_impl_t::execute(const exec_ctx_t &ctx) {
//      ...
//      auto gpu_engine = utils::downcast<ocl::engine_t*>(engine);
//      jit_generator->dbg_alloc(gpu_engine->context());
//      void* dbg_mem = jit_generator->dbg_memory();
//      ...
//      compute::kernel_arg_list_t arg_list;
//      arg_list.set(0, src);
//      arg_list.set(1, dst);
//      arg_list.set(2, dbg_mem, kernel_arg_t::kind_t::svm);
//      ...
//      parallel_for(ctx, nd_range, kernel_, arg_list);
//  }
//
//  ngen_kernel_t() : generator_t<...>() {
//      externalName("ngen_kernel");
//      newArgument("src", GlobalPtr);
//      newArgument("dst", GlobalPtr);
//      newArgument("dbg_mem", GlobalPtr);
//      finalizeInterface();
//      ...
//      auto header = r32;
//      auto data = r64;
//      mov<uint64_t>(1, r64, getArgument("dbg_mem"));
//      store(1, scattered_dword(), A64, header, data);
//      ...
//  }
//  ```
//

template <typename ngen_generator_t>
struct eltwise_injector_f32_t;

template <typename ngen_generator_t>
struct reduction_injector_f32_t;

template <typename ngen_generator_t>
struct post_op_injector_t;

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
    friend struct eltwise_injector_f32_t<generator_t>;
    friend struct reduction_injector_f32_t<generator_t>;
    friend struct post_op_injector_t<generator_t>;
    friend struct EmulationImplementation;

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

    const char *kernel_name() const override {
        return ngen_code_generator_t<hw>::getExternalName().c_str();
    }

    xpu::binary_t get_binary(const compute::compute_engine_t *engine) override {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        return ngen_code_generator_t<hw>::getBinary();
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        auto *ocl_engine = utils::downcast<const ocl::engine_t *>(engine);
        return ngen_code_generator_t<hw>::getBinary(
                ocl_engine->context(), ocl_engine->device());
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
        REG_GEN9_ISA(CASE(gpu_gen9));
        REG_GEN11_ISA(CASE(gpu_gen11));
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
