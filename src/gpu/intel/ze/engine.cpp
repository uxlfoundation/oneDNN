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

#include "gpu/intel/ze/engine.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/ze/device_info.hpp"
#include "gpu/intel/ze/kernel.hpp"
#include "gpu/intel/ze/stream.hpp"
#include "gpu/intel/ze/utils.hpp"

#include "xpu/ze/memory_storage.hpp"

#include "gemmstone/dsl/runtime.hpp"
#include "gemmstone/microkernel/fuser.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        ze_driver_handle_t dri, ze_device_handle_t dev, ze_context_handle_t ctx,
        size_t index) {
    gpu_assert(engine_kind == engine_kind::gpu);
    std::unique_ptr<ze::engine_t, engine_deleter_t> e(
            (new ze::engine_t(dri, dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

engine_t::engine_t(ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context, size_t index)
    : intel::engine_t(new xpu::ze::engine_impl_t(
              engine_kind::gpu, driver, device, context, index)) {}

status_t engine_t::init() {
    CHECK(init_impl());
    CHECK(intel::engine_t::init());

    return status::success;
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return ze::stream_t::create_stream(stream, this, stream_impl);
}

status_t engine_t::create_kernel(
        compute::kernel_t *kernel, jit::generator_base_t *jitter) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }
    return jitter->get_kernel(*kernel, this);
}

status_t engine_t::create_kernel(compute::kernel_t &kernel,
        const gemmstone::dsl::kernel_t &kernel_dsl) const {
    const auto &ze_kernel_and_module
            = gemmstone::dsl::make_kernel(kernel_dsl, context(), device());
    auto ze_module_ptr
            = std::make_shared<xpu::ze::wrapper_t<ze_module_handle_t>>(
                    ze_kernel_and_module.module);
    return kernel_t::make(
            kernel, ze_module_ptr, ze_kernel_and_module.kernel, {});
}

status_t engine_t::convert_to_ze(std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        xpu::binary_t &binary) const {
    std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>> ze_module_ptr;
    std::vector<ze_kernel_handle_t> ze_kernels;
    CHECK(ze::create_kernels(device(), context(), kernel_names, binary,
            &ze_module_ptr->unwrap(), ze_kernels));
    kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!ze_kernels[i]) continue;

        CHECK(kernel_t::make(
                kernels[i], ze_module_ptr, ze_kernels[i], kernel_names[i]));
    }

    return status::success;
}

status_t engine_t::create_kernels(std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    const char *source = nullptr;
    for (size_t i = 0; source == nullptr && i < kernel_names.size(); i++)
        source = intel::get_kernel_source(kernel_names[i]);
    VERROR_ENGINE(source, status::runtime_error,
            "No OpenCL source was found for kernel");

    std::string options = kernel_ctx.options();
    auto *dev_info = utils::downcast<const device_info_t *>(device_info());
    options += " " + dev_info->get_cl_ext_options();

    stringstream_t code_ss;
    CHECK(compute::preprocess_headers(code_ss, source, kernel_ctx));
    std::string code = code_ss.str();

    compute::program_src_t src(code);
    if (src) { options += " -g -s " + std::string(src.name()); }

    compute::debugdump_processed_source(
            code, options, dev_info->get_cl_ext_options());

    const char *code_c = code.c_str();
    xpu::binary_t binary;
    CHECK(ze::compile_ocl_module_to_binary(
            device(), context(), code, options, binary));

    if (kernel_ctx.has_custom_headers()
            && gemmstone::microkernel::hasMicrokernels(code_c)) {
        try {
            gemmstone::microkernel::fuse(binary, code_c);
        } catch (...) { return status::runtime_error; }
    }

    CHECK(convert_to_ze(*kernels, kernel_names, binary));

    return status::success;
}

status_t engine_t::create_kernel_from_binary(compute::kernel_t &kernel,
        const xpu::binary_t &binary, const char *kernel_name,
        const compute::program_src_t &src) const {
    std::vector<const char *> kernel_names = {kernel_name};
    std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>> ze_module_ptr;
    std::vector<ze_kernel_handle_t> ze_kernels;
    CHECK(ze::create_kernels(device(), context(), kernel_names, binary,
            &ze_module_ptr->unwrap(), ze_kernels));

    CHECK(kernel_t::make(kernel, ze_module_ptr, ze_kernels[0], kernel_name));

    return status::success;
}

status_t engine_t::create_kernels_from_cache_blob(
        const cache_blob_t &cache_blob, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!kernel_names[i] && kernel_names.size() > 1) continue;
        std::string kernel_name(kernel_names[i] ? kernel_names[i] : "");

        const uint8_t *binary_data = nullptr;
        size_t binary_size = 0;
        CHECK(cache_blob.get_binary(&binary_data, &binary_size));

        xpu::binary_t binary(binary_data, binary_data + binary_size);
        CHECK(create_kernel_from_binary(
                kernels[i], binary, kernel_names[i], compute::program_src_t()));
    }

    return status::success;
}

gpu_utils::device_id_t engine_t::device_id() const {
    return std::tuple_cat(
            std::make_tuple(1), xpu::ze::get_device_uuid(device()));
}

ze_driver_handle_t engine_t::driver() const {
    return impl()->driver();
}

ze_device_handle_t engine_t::device() const {
    return impl()->device();
}

ze_context_handle_t engine_t::context() const {
    return impl()->context();
}

cl_device_id engine_t::ocl_device() const {
    return impl()->ocl_device();
}

cl_context engine_t::ocl_context() const {
    return impl()->ocl_context();
}

status_t engine_t::init_device_info() {
    device_info_ = std::make_shared<ze::device_info_t>();
    CHECK(device_info_->init(this));

    return status::success;
}

status_t engine_t::init_device_info(const std::vector<uint8_t> &cache_blob) {
    gpu_assert(false) << "unimplemented function init_device_info() called";

    return status::runtime_error;
}

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
