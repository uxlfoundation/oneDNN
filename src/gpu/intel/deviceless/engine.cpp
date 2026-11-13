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

#include "gpu/intel/deviceless/engine.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/deviceless/device_info.hpp"
#include "gpu/intel/deviceless/engine_impl.hpp"
#include "gpu/intel/deviceless/kernel.hpp"
#include "gpu/intel/jit/generator_base.hpp"
#include "gpu/intel/utils.hpp"
#include "xpu/utils.hpp"

#include "common/serialization.hpp"
#include "common/verbose.hpp"

#include "gemmstone/microkernel/fuser.hpp"
#include "ngen.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

namespace {
status_t kernels_unimplemented() {
    VERROR(common, runtime, "device-less engine can't create kernels");
    return status::unimplemented;
}

// ocloc is a part of the compute runtime package rather than of the build
// toolchain, so its signatures are declared here instead of including
// `ocloc_api.h`.
using ocloc_invoke_t = int (*)(unsigned int, const char **, uint32_t,
        const uint8_t **, const uint64_t *, const char **, uint32_t,
        const uint8_t **, const uint64_t *, const char **, uint32_t *,
        uint8_t ***, uint64_t **, char ***);
using ocloc_free_output_t
        = int (*)(uint32_t *, uint8_t ***, uint64_t **, char ***);

// ocloc compiles for a target named by its IP version and needs no driver,
// unlike `clBuildProgram` and `zeModuleCreate`.
status_t compile_ocl_module_to_binary(const compute::device_info_t &device_info,
        const std::string &code, const std::string &options,
        xpu::binary_t &binary) {
#if defined(_WIN32)
    const char *ocloc_lib_name = "ocloc64.dll";
#elif defined(__linux__)
    const char *ocloc_lib_name = "libocloc.so";
#endif
    static const auto ocloc_invoke = reinterpret_cast<ocloc_invoke_t>(
            xpu::find_symbol(ocloc_lib_name, "oclocInvoke"));
    static const auto ocloc_free_output = reinterpret_cast<ocloc_free_output_t>(
            xpu::find_symbol(ocloc_lib_name, "oclocFreeOutput"));
    VERROR_ENGINE(ocloc_invoke && ocloc_free_output, status::runtime_error,
            "device-less engine needs ocloc to compile OpenCL C kernels");

    const char *source_name = "source.cl";
    const std::string device = std::to_string(device_info.ip_version());
    // `-exclude_ir` drops the SPIR-V from the binary, so the runtime can't
    // silently rebuild it for a mismatched target and the import fails
    // instead.
    std::vector<const char *> argv
            = {"ocloc", "compile", "-q", "-file", source_name, "-device",
                    device.c_str(), "-exclude_ir", "-options", options.c_str()};
    // ocloc builds DG2 kernels bindless, while the runtime expects them
    // bindful and rejects the binary.
    if (device_info.product().family == ngen::ProductFamily::DG2)
        argv.insert(argv.end(), {"-stateful_address_mode", "bindful"});
    const auto *source_data = reinterpret_cast<const uint8_t *>(code.c_str());
    const uint64_t source_size = code.size() + 1;

    uint32_t n_outputs = 0;
    uint8_t **outputs = nullptr;
    uint64_t *output_sizes = nullptr;
    char **output_names = nullptr;
    const int err = ocloc_invoke(static_cast<unsigned int>(argv.size()),
            argv.data(), 1, &source_data, &source_size, &source_name, 0,
            nullptr, nullptr, nullptr, &n_outputs, &outputs, &output_sizes,
            &output_names);

    // ocloc allocates outputs even on failure, so they are copied out and
    // freed before `err` is checked.
    std::string log;
    for (uint32_t i = 0; i < n_outputs; i++) {
        const std::string name = output_names[i];
        if (name == "stdout.log")
            log.assign(reinterpret_cast<const char *>(outputs[i]),
                    output_sizes[i]);
        else if (name.size() > 4
                && name.compare(name.size() - 4, 4, ".bin") == 0)
            binary.assign(outputs[i], outputs[i] + output_sizes[i]);
    }
    ocloc_free_output(&n_outputs, &outputs, &output_sizes, &output_names);

    VERROR_ENGINE(
            err == 0, status::runtime_error, "ocloc failed: %s", log.c_str());
    VERROR_ENGINE(
            !binary.empty(), status::runtime_error, "ocloc produced no binary");
    return status::success;
}
} // namespace

engine_t::engine_t(
        runtime_kind_t runtime_kind, const std::vector<uint8_t> &cache_blob)
    : intel::engine_t(new engine_impl_t(runtime_kind, cache_blob)) {}

status_t engine_t::init() {
    return status::invalid_arguments;
}

status_t engine_t::init(const std::vector<uint8_t> &cache_blob) {
    CHECK(init_impl());
    return init_device_info(cache_blob);
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    VERROR(common, runtime, "device-less engine has no streams");
    return status::unimplemented;
}

status_t engine_t::create_kernel(
        compute::kernel_t *kernel, jit::generator_base_t *jitter) const {
    if (!jitter) return status::invalid_arguments;
    return jitter->get_kernel(*kernel, this);
}

status_t engine_t::create_kernel(
        compute::kernel_t &kernel, const jit::dsl::kernel_t &kernel_ir) const {
    return kernels_unimplemented();
}

status_t engine_t::create_kernels(std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {
    const char *source = nullptr;
    for (size_t i = 0; source == nullptr && i < kernel_names.size(); i++)
        source = get_kernel_source(kernel_names[i]);
    VERROR_ENGINE(source, status::runtime_error,
            "No OpenCL source was found for kernel");

    std::string options = kernel_ctx.options();
    options += " " + device_info()->get_cl_ext_options();

    // The OpenCL engine compiles with this option too, so the binary matches
    // the one it would build for the same kernel.
    // TODO: consolidate with `ocl::engine_t::build_program_from_source()`.
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kernel_ctx.require_stateless_addressing()
            && device_info()->device_address_bits() >= 64
            && device_info()->max_allocation_size() <= UINT32_MAX)
        options += " -cl-intel-greater-than-4GB-buffer-required";
#endif

    stringstream_t code_ss;
    CHECK(compute::preprocess_headers(code_ss, source, kernel_ctx));
    std::string code = code_ss.str();

    compute::debugdump_processed_source(
            code, options, device_info()->get_cl_ext_options());

    xpu::binary_t binary;
    CHECK(compile_ocl_module_to_binary(*device_info(), code, options, binary));

    if (kernel_ctx.has_custom_headers()
            && gemmstone::microkernel::hasMicrokernels(code.c_str()))
        CHECK(compute::fuse_microkernels(
                binary, code.c_str(), device_info()->grf_size()));

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!kernel_names[i]) continue;
        CHECK(kernel_t::make((*kernels)[i], binary, kernel_names[i]));
    }
    return status::success;
}

status_t engine_t::create_kernel_from_binary(compute::kernel_t &kernel,
        const xpu::binary_t &binary, const char *kernel_name,
        const compute::program_src_t &src) const {
    return kernel_t::make(kernel, binary, kernel_name);
}

status_t engine_t::create_kernels_from_cache_blob(
        const cache_blob_t &cache_blob, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    return kernels_unimplemented();
}

gpu_utils::device_id_t engine_t::device_id() const {
    return std::make_tuple(-1,
            serialization_stream_t::from_data(device_info_->get_cache_blob())
                    .get_hash(),
            0);
}

status_t engine_t::serialize_device(serialization_stream_t &sstream) const {
    if (runtime_kind() == runtime_kind::ocl)
        sstream.append_array(device_info()->platform_name().size(),
                device_info()->platform_name().data());
    sstream.append_array(
            device_info()->name().size(), device_info()->name().data());
    sstream.append(device_info()->runtime_version().major);
    sstream.append(device_info()->runtime_version().minor);
    sstream.append(device_info()->runtime_version().build);
    sstream.append(device_info()->runtime_version().revision);

    return status::success;
}

status_t engine_t::get_cache_blob_size(size_t *size) const {
    return device_info_->get_cache_blob_size(size);
}

status_t engine_t::get_cache_blob(size_t size, uint8_t *cache_blob) const {
    return device_info_->get_cache_blob(size, cache_blob);
}

status_t engine_t::init_device_info() {
    return status::unimplemented;
}

status_t engine_t::init_device_info(const std::vector<uint8_t> &cache_blob) {
    device_info_ = std::make_shared<device_info_t>();
    return device_info_->init(this, cache_blob);
}

status_t engine_create(
        impl::engine_t **engine, const std::vector<uint8_t> &cache_blob) {
    runtime_kind_t rt_kind = runtime_kind::none;
    CHECK(compute::device_info_t::get_runtime_kind(cache_blob, rt_kind));
    VERROR_ENGINE(utils::one_of(rt_kind, runtime_kind::ocl, runtime_kind::ze),
            status::invalid_arguments,
            "device-less engine supports only OpenCL and Level Zero runtimes");

    std::unique_ptr<engine_t, engine_deleter_t> e(
            new engine_t(rt_kind, cache_blob));
    if (!e) return status::out_of_memory;

    CHECK(e->init(cache_blob));
    *engine = e.release();

    return status::success;
}

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
