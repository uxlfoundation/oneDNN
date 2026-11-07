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
#include "gpu/intel/deviceless/device_info.hpp"
#include "gpu/intel/deviceless/engine_impl.hpp"
#include "gpu/intel/deviceless/kernel.hpp"
#include "gpu/intel/jit/generator_base.hpp"

#include "common/serialization.hpp"
#include "common/verbose.hpp"

#include "ngen.hpp"

#include <memory>

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
    return kernels_unimplemented();
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
