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

#ifndef GPU_INTEL_L0_ENGINE_HPP
#define GPU_INTEL_L0_ENGINE_HPP

#include "gpu/intel/engine.hpp"
#include "gpu/intel/l0/utils/utils.hpp"
#include "xpu/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

struct engine_id_impl_t : public impl::engine_id_impl_t {
    engine_id_impl_t(const ze_device_handle_t device,
            const ze_context_handle_t context, engine_kind_t kind,
            runtime_kind_t runtime_kind, size_t index)
        : impl::engine_id_impl_t(kind, runtime_kind, index)
        , device_(device)
        , context_(context) {}
    ~engine_id_impl_t() override = default;

private:
    bool compare_resource(
            const impl::engine_id_impl_t *id_impl) const override {
        const auto *typed_id
                = utils::downcast<const engine_id_impl_t *>(id_impl);
        return device_ == typed_id->device_ && context_ == typed_id->context_;
    }

    size_t hash_resource() const override {
        size_t seed = 0;
        seed = hash_combine(seed, device_);
        seed = hash_combine(seed, context_);
        return seed;
    }

    ze_device_handle_t device_;
    ze_context_handle_t context_;

    engine_id_impl_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_id_impl_t);
};

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ze_driver_handle_t dri, const ze_device_handle_t dev,
        const ze_context_handle_t ctx, size_t index);

class engine_t : public intel::engine_t {
public:
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

    gpu::intel::gpu_utils::device_id_t device_id() const override;

    const ze_driver_handle_t driver() const;
    const ze_device_handle_t device() const;
    const ze_context_handle_t context() const;

    const cl_device_id ocl_device() const;
    const cl_context ocl_context() const;

    bool mayiuse_microkernels() const;

private:
    status_t init_device_info() override;
    status_t init_device_info(const std::vector<uint8_t> &cache_blob) override;

    status_t convert_to_l0(std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names,
            xpu::binary_t &binary) const;

    engine_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_t);
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_ENGINE_HPP
