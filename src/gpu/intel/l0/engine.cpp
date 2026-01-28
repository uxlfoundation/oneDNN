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

#include "gpu/intel/l0/engine.hpp"
#include "gpu/intel/l0/compiler.hpp"
#include "gpu/intel/l0/device_info.hpp"
#include "gpu/intel/l0/kernel.hpp"
#include "gpu/intel/l0/memory_storage.hpp"
#include "gpu/intel/l0/stream.hpp"

#include "gemmstone/dsl/runtime.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/microkernels/fuser.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t(engine_kind_t kind, const ze_driver_handle_t driver,
            const ze_device_handle_t device, const ze_context_handle_t context,
            size_t index)
        : impl::engine_impl_t(kind, runtime_kind::l0, index)
        , driver_(driver)
        , device_(device)
        , context_(context) {
        cl_int err;
        std::vector<cl_device_id> ocl_devices;
        xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);

        ocl_device_ = nullptr;
        ocl_context_ = nullptr;
        xpu::device_uuid_t l0_dev_uuid = get_device_uuid(device);
        for (const cl_device_id &d : ocl_devices) {
            xpu::device_uuid_t ocl_dev_uuid;
            xpu::ocl::get_device_uuid(ocl_dev_uuid, d);
            if (l0_dev_uuid == ocl_dev_uuid) {
                ocl_device_ = xpu::ocl::make_wrapper(d);
                ocl_context_ = xpu::ocl::make_wrapper(
                        xpu::ocl::clCreateContext(nullptr, 1,
                                &ocl_device_.unwrap(), nullptr, nullptr, &err));
            }
        }
    }
    ~engine_impl_t() override { l0::zeContextDestroy(context_); }

    const ze_driver_handle_t driver() const { return driver_; }
    const ze_device_handle_t device() const { return device_; }
    const ze_context_handle_t context() const { return context_; }

    const xpu::ocl::wrapper_t<cl_device_id> ocl_device() const {
        return ocl_device_;
    }
    const xpu::ocl::wrapper_t<cl_context> ocl_context() const {
        return ocl_context_;
    }

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override {
        auto *si = new stream_impl_t(flags, context_, device_);
        if (!si) return status::out_of_memory;

        *stream_impl = si;

        return status::success;
    }

    status_t create_memory_storage(impl::memory_storage_t **storage,
            impl::engine_t *engine, unsigned flags, size_t size,
            void *handle) const override {
        std::unique_ptr<memory_storage_t> _storage;
        _storage.reset(
                new memory_storage_t(engine, memory_storage_kind_t::device));
        if (!_storage) return status::out_of_memory;

        status_t status = _storage->init(flags, size, handle);
        if (status != status::success) return status;

        *storage = _storage.release();

        return status::success;
    }

    engine_id_t engine_id() const override {
        return engine_id_t(new engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

    int get_buffer_alignment() const override { return 128; }

private:
    ze_driver_handle_t driver_;
    ze_device_handle_t device_;
    ze_context_handle_t context_;

    xpu::ocl::wrapper_t<cl_device_id> ocl_device_;
    xpu::ocl::wrapper_t<cl_context> ocl_context_;

    engine_impl_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_impl_t);
};

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ze_driver_handle_t dri, const ze_device_handle_t dev,
        const ze_context_handle_t ctx, size_t index) {
    std::unique_ptr<gpu::intel::l0::engine_t, engine_deleter_t> e(
            (new gpu::intel::l0::engine_t(dri, dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

engine_t::engine_t(ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context, size_t index)
    : gpu::intel::engine_t(new engine_impl_t(
              engine_kind::gpu, driver, device, context, index)) {}

status_t engine_t::init() {
    CHECK(init_impl());
    CHECK(gpu::intel::engine_t::init());

    return status::success;
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return gpu::intel::l0::stream_t::create_stream(stream, this, stream_impl);
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
    const auto &module_kernel_pair
            = gemmstone::dsl::make_kernel(kernel_dsl, context(), device());
    auto l0_module_ptr
            = std::make_shared<module_wrapper_t>(module_kernel_pair.first);

    return kernel_t::make(kernel, l0_module_ptr, module_kernel_pair.second, {});
}

status_t engine_t::convert_to_l0(
        std::vector<gpu::intel::compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        xpu::binary_t &binary) const {
    ze_module_handle_t l0_module = nullptr;
    std::vector<ze_kernel_handle_t> l0_kernels;
    CHECK(gpu::intel::l0::create_kernels(
            device(), context(), kernel_names, binary, &l0_module, l0_kernels));
    auto l0_module_ptr = std::make_shared<module_wrapper_t>(l0_module);

    kernels = std::vector<gpu::intel::compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!l0_kernels[i]) continue;
        CHECK(kernel_t::make(
                kernels[i], l0_module_ptr, l0_kernels[i], kernel_names[i]));
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

    std::string options = kernel_ctx.options();
    auto *dev_info = utils::downcast<const device_info_t *>(device_info());
    options += " " + dev_info->get_cl_ext_options();

    stringstream_t code_ss;
    CHECK(compute::preprocess_headers(code_ss, source, kernel_ctx));
    std::string code = code_ss.str();

    gpu::intel::compute::program_src_t src(code);
    if (src) { options += " -g -s " + std::string(src.name()); }

    compute::debugdump_processed_source(
            code, options, dev_info->get_cl_ext_options());

    const char *code_c = code.c_str();
    xpu::binary_t binary;
    if (l0::compile_ocl_module_to_binary(
                device(), context(), code, options, binary)
            != status::success)
        CHECK(ocl_build_kernels(
                ocl_device(), ocl_context(), code_c, options.c_str(), binary));

    if (kernel_ctx.has_custom_headers() && micro::hasMicrokernels(code_c)) {
        try {
            micro::fuseMicrokernels(binary, code_c);
        } catch (...) { return status::runtime_error; }
    }

    CHECK(convert_to_l0(*kernels, kernel_names, binary));

    return status::success;
}

status_t engine_t::create_kernel_from_binary(compute::kernel_t &kernel,
        const xpu::binary_t &binary, const char *kernel_name,
        const compute::program_src_t &src) const {
    std::vector<const char *> kernel_names = {kernel_name};
    ze_module_handle_t l0_module = nullptr;
    std::vector<ze_kernel_handle_t> l0_kernels;
    CHECK(gpu::intel::l0::create_kernels(
            device(), context(), kernel_names, binary, &l0_module, l0_kernels));
    auto l0_module_ptr = std::make_shared<module_wrapper_t>(l0_module);

    CHECK(kernel_t::make(kernel, l0_module_ptr, l0_kernels[0], kernel_name));

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
        CHECK(create_kernel_from_binary(kernels[i], binary, kernel_names[i],
                gpu::intel::compute::program_src_t()));
    }

    return status::success;
}

gpu_utils::device_id_t engine_t::device_id() const {
    return std::tuple_cat(
            std::make_tuple(1), gpu::intel::l0::get_device_uuid(device()));
}

const ze_driver_handle_t engine_t::driver() const {
    return static_cast<const engine_impl_t *>(impl())->driver();
}

const ze_device_handle_t engine_t::device() const {
    return static_cast<const engine_impl_t *>(impl())->device();
}

const ze_context_handle_t engine_t::context() const {
    return static_cast<const engine_impl_t *>(impl())->context();
}

const cl_device_id engine_t::ocl_device() const {
    return static_cast<const engine_impl_t *>(impl())->ocl_device();
}

const cl_context engine_t::ocl_context() const {
    return static_cast<const engine_impl_t *>(impl())->ocl_context();
}

bool engine_t::mayiuse_microkernels() const {
    if (!l0::mayiuse_microkernels(device(), context(),
                std::string(compute::cl_microkernels_check_kernel_code))) {
        return ocl_mayiuse_microkernels(ocl_device(), ocl_context(),
                compute::cl_microkernels_check_kernel_code);
    }
    return true;
}

status_t engine_t::init_device_info() {
    device_info_ = std::make_shared<gpu::intel::l0::device_info_t>();
    CHECK(device_info_->init(this));

    return status::success;
}

status_t engine_t::init_device_info(const std::vector<uint8_t> &cache_blob) {
    gpu_assert(false) << "unimplemented function init_device_info() called";

    return status::runtime_error;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
