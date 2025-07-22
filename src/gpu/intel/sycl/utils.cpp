/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/sycl/utils.hpp"

#include "gpu/intel/l0/utils/utils.hpp"
#include "gpu/intel/ocl/utils/utils.hpp"
#include "gpu/intel/sycl/engine.hpp"
#include "xpu/ocl/engine_factory.hpp"
#include "xpu/ocl/utils.hpp"
#include "xpu/sycl/compat.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
xpu::device_uuid_t get_device_uuid(const ::sycl::device &dev) {
    return gpu::intel::l0::get_device_uuid(
            xpu::sycl::compat::get_native<ze_device_handle_t>(dev));
}

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(lhs);
    auto rhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(rhs);

    return lhs_ze_handle == rhs_ze_handle;
}

status_t sycl_create_kernels_with_level_zero(
        std::vector<std::unique_ptr<::sycl::kernel>> &sycl_kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary) {
    auto ze_device = xpu::sycl::compat::get_native<ze_device_handle_t>(
            sycl_engine->device());
    auto ze_ctx = xpu::sycl::compat::get_native<ze_context_handle_t>(
            sycl_engine->context());
    ze_module_handle_t ze_module = nullptr;
    std::vector<ze_kernel_handle_t> ze_kernels;

    gpu::intel::l0::create_kernels(
            ze_device, ze_ctx, kernel_names, binary, ze_module, ze_kernels);

    ::sycl::kernel_bundle<::sycl::bundle_state::executable> kernel_bundle
            = ::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                    ::sycl::bundle_state::executable>(
                    {ze_module}, sycl_engine->context());

    sycl_kernels.resize(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (kernel_names[i] == nullptr) continue;
        auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
                {kernel_bundle, ze_kernels[i]}, sycl_engine->context());
        sycl_kernels[i] = utils::make_unique<::sycl::kernel>(k);
    }

    return status::success;
}

::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::intel::compute::nd_range_t &range) {
    const auto &local_range = range.local_range();
    const auto &global_range = range.global_range();

    assert(range.ndims() <= 3);
    auto sycl_global_range = ::sycl::range<3>(
            global_range.ndims() >= 3 ? global_range[2] : 1,
            global_range.ndims() >= 2 ? global_range[1] : 1, global_range[0]);

    if (!local_range) {
        assert(!"not expected");
        return ::sycl::nd_range<3>(
                sycl_global_range, ::sycl::range<3>(1, 1, 1));
    }

    auto sycl_local_range = ::sycl::range<3>(
            local_range.ndims() >= 3 ? local_range[2] : 1,
            local_range.ndims() >= 2 ? local_range[1] : 1, local_range[0]);
    return ::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

#ifndef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
status_t sycl_dev2ocl_dev(cl_device_id *ocl_dev, const ::sycl::device &dev) {
    assert(xpu::sycl::get_backend(dev) == xpu::sycl::backend_t::level0);
    if (xpu::sycl::get_backend(dev) != xpu::sycl::backend_t::level0)
        return status::runtime_error;

    CHECK(gpu::intel::l0::l0_dev2ocl_dev(
            gpu::intel::sycl::get_device_uuid(dev), ocl_dev));
    return status::success;
}

static status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const ::sycl::device &sycl_dev,
        const ::sycl::context *sycl_ctx = nullptr) {
    xpu::ocl::engine_factory_t f(engine_kind::gpu);
    const auto backend = xpu::sycl::get_backend(sycl_dev);

    // The SYCL context is always provided for OpenCL backend.
    if (backend == xpu::sycl::backend_t::opencl && !sycl_ctx)
        return status::runtime_error;
    xpu::ocl::wrapper_t<cl_device_id> ocl_dev;
    xpu::ocl::wrapper_t<cl_context> ocl_ctx;

    switch (backend) {
        case xpu::sycl::backend_t::opencl:
            ocl_dev = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_device_id>(sycl_dev));
            ocl_ctx = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_context>(*sycl_ctx));
            break;
        case xpu::sycl::backend_t::level0: {
            cl_device_id d {nullptr};
            CHECK(sycl_dev2ocl_dev(&d, sycl_dev));
            ocl_dev = xpu::ocl::make_wrapper(d, true);

            cl_int err;
            ocl_ctx = xpu::ocl::make_wrapper(
                    clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err));
            OCL_CHECK(err);
            break;
        }
        default: assert(!"not expected"); return status::invalid_arguments;
    }
    impl::engine_t *ocl_engine_ptr;
    size_t index;
    CHECK(xpu::ocl::get_device_index(&index, ocl_dev));
    CHECK(f.engine_create(&ocl_engine_ptr, ocl_dev, ocl_ctx, index));
    ocl_engine->reset(
            utils::downcast<gpu::intel::ocl::engine_t *>(ocl_engine_ptr));
    return status::success;
}

status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const gpu::intel::sycl::engine_t *engine) {
    const auto &sycl_ctx = engine->context();
    return create_ocl_engine(ocl_engine, engine->device(), &sycl_ctx);
}
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

status_t get_kernel_binary(
        const ::sycl::kernel &kernel, xpu::binary_t &binary) {
    auto devs = kernel.get_context().get_devices();
    assert(!devs.empty());
    switch (xpu::sycl::get_backend(devs[0])) {
        case xpu::sycl::backend_t::level0: {

#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
            auto l0_kernel = ::sycl::get_native<
                    ::sycl::backend::ext_oneapi_level_zero>(kernel);
            size_t binary_size = 0;
            CHECK(gpu::intel::l0::func_zeGetKernelBinary(
                    l0_kernel, &binary_size, nullptr));
            binary.resize(binary_size);
            CHECK(gpu::intel::l0::func_zeGetKernelBinary(
                    l0_kernel, &binary_size, binary.data()));
#else
            auto bundle = kernel.get_kernel_bundle();
            auto module_vec = ::sycl::get_native<
                    ::sycl::backend::ext_oneapi_level_zero>(bundle);
            auto module = module_vec[0];
            size_t module_binary_size;
            xpu::binary_t module_binary;
            CHECK(gpu::intel::l0::func_zeModuleGetNativeBinary(
                    module, &module_binary_size, nullptr));
            module_binary.resize(module_binary_size);
            CHECK(gpu::intel::l0::func_zeModuleGetNativeBinary(
                    module, &module_binary_size, module_binary.data()));
            {
                std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                        ocl_engine;
                CHECK(create_ocl_engine(&ocl_engine, devs[0]));
                xpu::ocl::wrapper_t<cl_program> ocl_program;
                CHECK(xpu::ocl::create_program(ocl_program,
                        ocl_engine->device(), ocl_engine->context(),
                        module_binary));

                cl_int err;
                auto name = kernel.get_info<
                        ::sycl::info::kernel::function_name>();
                auto ocl_kernel = xpu::ocl::make_wrapper(
                        clCreateKernel(ocl_program, name.c_str(), &err));
                OCL_CHECK(err);
                CHECK(gpu::intel::ocl::get_ocl_kernel_binary(
                        ocl_kernel, binary));
            }
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
            return status::success;
        }
        case xpu::sycl::backend_t::opencl: {
            auto ocl_kernel
                    = ::sycl::get_native<::sycl::backend::opencl>(kernel);
            CHECK(gpu::intel::ocl::get_ocl_kernel_binary(ocl_kernel, binary));
            return status::success;
        }
        default: return status::runtime_error;
    }
}

gpu_utils::device_id_t device_id(const ::sycl::device &dev) {
    if (xpu::sycl::is_host(dev))
        return std::make_tuple(
                static_cast<int>(xpu::sycl::backend_t::host), 0, 0);

    gpu_utils::device_id_t device_id = gpu_utils::device_id_t {
            static_cast<int>(xpu::sycl::backend_t::unknown), 0, 0};
    switch (xpu::sycl::get_backend(dev)) {
        case xpu::sycl::backend_t::opencl: {
            auto ocl_device = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_device_id>(dev));
            device_id = std::make_tuple(
                    static_cast<int>(xpu::sycl::backend_t::opencl),
                    reinterpret_cast<uint64_t>(ocl_device.get()), 0);
            break;
        }
        case xpu::sycl::backend_t::level0: {
            device_id = std::tuple_cat(std::make_tuple(static_cast<int>(
                                               xpu::sycl::backend_t::level0)),
                    gpu::intel::sycl::get_device_uuid(dev));
            break;
        }
        case xpu::sycl::backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
    assert(std::get<0>(device_id)
            != static_cast<int>(xpu::sycl::backend_t::unknown));
    return device_id;
}

#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
bool mayiuse_microkernels(const gpu::intel::sycl::engine_t *engine) {
    namespace syclex = ::sycl::ext::oneapi::experimental;
    auto kb_src = syclex::create_kernel_bundle_from_source(engine->context(),
            syclex::source_language::opencl,
            compute::cl_microkernels_check_kernel_code);
    try {
        syclex::build(kb_src);
    } catch (const ::sycl::exception &e) { return false; }
    return true;
}
#else
status_t get_sycl_ocl_device_and_context(
        xpu::ocl::wrapper_t<cl_context> &ocl_context,
        xpu::ocl::wrapper_t<cl_device_id> &ocl_device,
        const sycl::engine_t *sycl_engine) {
    auto &device = sycl_engine->device();
    auto be = xpu::sycl::get_backend(device);
    if (be == xpu::sycl::backend_t::opencl) {
        cl_int err = CL_SUCCESS;
        auto ocl_dev = xpu::sycl::compat::get_native<cl_device_id>(device);
        ocl_device = xpu::ocl::make_wrapper(ocl_dev, true);

        ocl_context = xpu::ocl::make_wrapper(
                clCreateContext(nullptr, 1, &ocl_dev, nullptr, nullptr, &err),
                true);
        OCL_CHECK(err);
    } else if (be == xpu::sycl::backend_t::level0) {
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t> ocl_engine;
        CHECK(create_ocl_engine(&ocl_engine, sycl_engine));
        ocl_device = xpu::ocl::make_wrapper(ocl_engine->device(), true);
        ocl_context = xpu::ocl::make_wrapper(ocl_engine->context(), true);
    }
    return status::success;
}

bool mayiuse_microkernels(const gpu::intel::sycl::engine_t *engine) {
    xpu::ocl::wrapper_t<cl_context> ocl_context;
    xpu::ocl::wrapper_t<cl_device_id> ocl_device;
    auto status
            = get_sycl_ocl_device_and_context(ocl_context, ocl_device, engine);
    if (status != status::success) return false;
    return ocl::try_building(ocl_context, ocl_device,
            compute::cl_microkernels_check_kernel_code);
}
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
