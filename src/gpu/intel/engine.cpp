/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <unordered_map>

#include "gpu/intel/engine.hpp"

#include "gpu/intel/logging.hpp"

#include "common/rw_mutex.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

void maybe_print_build_info(const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) {

    gpu_info() << "build kernels:" << [&]() {
        ostringstream_t names;
        for (const char *name : kernel_names)
            if (name) names << " " << name;
        return names.str();
    }();
    gpu_info() << "build options: " << kernel_ctx.options();
}

// Cache for device_info_t objects. Reuse the already initialized
// device_info_t objects to save time on HW detection and nGEN binary
// check.
using device_info_cache_t = std::unordered_map<gpu_utils::device_id_t,
        std::shared_ptr<compute::device_info_t>, gpu_utils::device_id_hash_t>;

utils::rw_mutex_t &device_info_cache_mutex() {
    static utils::rw_mutex_t m;
    return m;
}

device_info_cache_t &device_info_cache() {
    static device_info_cache_t cache;
    return cache;
}

// Returns true if found, false otherwise.
bool device_info_cache_get(std::shared_ptr<compute::device_info_t> *result,
        impl::engine_t *engine) {
    utils::lock_read_t lock(device_info_cache_mutex());
    auto *intel_engine = utils::downcast<engine_t *>(engine);
    auto it = device_info_cache().find(intel_engine->device_id());
    if (it == device_info_cache().end()) return false;
    if (result) *result = it->second;
    return true;
}

void device_info_cache_set(impl::engine_t *engine,
        const std::shared_ptr<compute::device_info_t> &device_info) {
    utils::lock_write_t lock(device_info_cache_mutex());

    auto *intel_engine = utils::downcast<engine_t *>(engine);
    // Clear the cache to avoid hypothetically large growth.
    const int cache_size_threshold = 1024;
    if (device_info_cache().size() > cache_size_threshold)
        device_info_cache().clear();

    device_info_cache().insert({intel_engine->device_id(), device_info});
}

status_t engine_t::init() {
    return init({});
}

status_t engine_t::init(const std::vector<uint8_t> &cache_blob) {
    // A cache blob is always parsed: a cached device info would let a blob
    // from another build, runtime, or driver pass without verification.
    if (cache_blob.empty() && device_info_cache_get(&device_info_, this))
        return status::success;
    // Since init_device_info that takes a cache blob is only defined for
    // OpenCL we need to do manual dispatching here.
    if (cache_blob.empty())
        CHECK(init_device_info());
    else
        CHECK(init_device_info(cache_blob));
    device_info_cache_set(this, device_info_);

    return status::success;
}

status_t engine_t::serialize_device(serialization_stream_t &sstream) const {
    assert(utils::one_of(runtime_kind(), runtime_kind::ocl, runtime_kind::ze));
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

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

bool dnnl_impl_gpu_intel_mayiuse_ngen_kernels(dnnl::impl::engine_t *engine) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::gpu;

    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    return intel_engine->mayiuse_ngen_kernels();
}

const char *dnnl_impl_gpu_intel_get_isa_name(dnnl::impl::engine_t *engine) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::gpu;

    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *device_info = intel_engine->device_info();
    return intel::compute::to_string(device_info->gpu_arch());
}
