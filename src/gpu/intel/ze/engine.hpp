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

#ifndef GPU_INTEL_ZE_ENGINE_HPP
#define GPU_INTEL_ZE_ENGINE_HPP

#include "gpu/intel/engine.hpp"

#include "xpu/ze/engine_impl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        ze_driver_handle_t dri, ze_device_handle_t dev, ze_context_handle_t ctx,
        size_t index);

class engine_t : public intel::engine_t {
public:
    engine_t() = delete;
    engine_t(ze_driver_handle_t driver, ze_device_handle_t device,
            ze_context_handle_t context, size_t index);

    ~engine_t() override = default;

    status_t init() override;

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    status_t create_kernel(compute::kernel_t *kernel,
            jit::generator_base_t *jitter) const override;
    status_t create_kernel(compute::kernel_t &kernel,
            const jit::dsl::kernel_t &kernel_ir) const override;
    status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const override;
    status_t create_kernel_from_binary(compute::kernel_t &kernel,
            const xpu::binary_t &binary, const char *kernel_name,
            const compute::program_src_t &src) const override;
    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    gpu_utils::device_id_t device_id() const override;

    ze_driver_handle_t driver() const;
    ze_device_handle_t device() const;
    ze_context_handle_t context() const;

    cl_device_id ocl_device() const;
    cl_context ocl_context() const;

private:
    const xpu::ze::engine_impl_t *impl() const {
        return static_cast<const xpu::ze::engine_impl_t *>(
                impl::engine_t::impl());
    }

    status_t init_device_info() override;
    status_t init_device_info(const std::vector<uint8_t> &cache_blob) override;

    status_t convert_to_ze(std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names,
            xpu::binary_t &binary) const;

    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_t);
};

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_ZE_ENGINE_HPP
