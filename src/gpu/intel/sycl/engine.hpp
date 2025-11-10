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

#ifndef GPU_INTEL_SYCL_ENGINE_HPP
#define GPU_INTEL_SYCL_ENGINE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory_storage.hpp"

#include "xpu/ocl/utils.hpp"
#include "xpu/sycl/engine_impl.hpp"

#include "gpu/intel/engine.hpp"

#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/kernel.hpp"

#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/utils.hpp"

#include "gpu/intel/sycl/interop_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

class engine_t : public gpu::intel::engine_t {
public:
    engine_t(
            const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
        : gpu::intel::engine_t(new xpu::sycl::engine_impl_t(
                engine_kind::gpu, dev, ctx, index)) {}

    status_t init() override {
        CHECK(init_impl());
        CHECK(gpu::intel::engine_t::init());

        return status::success;
    }

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    status_t create_kernel_from_binary(gpu::intel::compute::kernel_t &kernel,
            const xpu::binary_t &binary, const char *kernel_name,
            const gpu::intel::compute::program_src_t &src) const override {
        std::unique_ptr<::sycl::kernel> sycl_kernel;
        VCHECK_KERNEL(gpu::intel::sycl::compat::make_kernel(
                              sycl_kernel, kernel_name, this, binary),
                VERBOSE_KERNEL_CREATION_FAIL, kernel_name);
        CHECK(interop_kernel_t::make(kernel, *sycl_kernel, {}));
        return status::success;
    }

    status_t create_kernel(gpu::intel::compute::kernel_t *kernel,
            gpu::intel::jit::generator_base_t *jitter) const override;

    status_t create_kernel(compute::kernel_t &kernel,
            const jit::kernel_t &kernel_ir) const override;

#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    status_t create_kernels(std::vector<gpu::intel::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::intel::compute::kernel_ctx_t &kernel_ctx) const override;
#else
    status_t convert_to_sycl(
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<gpu::intel::compute::kernel_t> &ocl_kernels,
            const std::vector<const char *> &kernel_names,
            gpu::intel::ocl::engine_t *ocl_engine) const;

    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    status_t convert_to_sycl(
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            cl_program program,
            const gpu::intel::compute::program_src_t &program_src,
            const std::vector<const char *> &kernel_names,
            gpu::intel::ocl::engine_t *ocl_engine) const;

    status_t create_kernels(std::vector<gpu::intel::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::intel::compute::kernel_ctx_t &kernel_ctx) const override;
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

    cl_device_id ocl_device() const {
        if (backend() != xpu::sycl::backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device().is_cpu() || device().is_gpu());
        return xpu::ocl::make_wrapper(
                xpu::sycl::compat::get_native<cl_device_id>(device()));
    }

    cl_context ocl_context() const {
        if (backend() != xpu::sycl::backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device().is_cpu() || device().is_gpu());
        return xpu::ocl::make_wrapper(
                xpu::sycl::compat::get_native<cl_context>(context()));
    }

    gpu::intel::gpu_utils::device_id_t device_id() const override {
        return gpu::intel::sycl::device_id(device());
    }

    DECLARE_COMMON_SYCL_ENGINE_FUNCTIONS();

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }

    ~engine_t() override = default;
    status_t init_device_info() override;
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
