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

#include <unordered_map>

#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel.hpp"

#include "common/utils.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/utils.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/utils.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// Cache for device_info_t objects. Reuse the already initialized
// device_info_t objects to save time on HW detection and nGEN binary
// check.
using device_info_cache_t = std::unordered_map<gpu_utils::device_id_t,
        std::shared_ptr<device_info_t>, gpu_utils::device_id_hash_t>;

utils::rw_mutex_t &device_info_cache_mutex() {
    static utils::rw_mutex_t m;
    return m;
}

device_info_cache_t &device_info_cache() {
    static device_info_cache_t cache;
    return cache;
}

// Returns true if found, false otherwise.
bool device_info_cache_get(
        std::shared_ptr<device_info_t> *result, impl::engine_t *engine) {
    utils::lock_read_t lock(device_info_cache_mutex());
    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto it = device_info_cache().find(compute_engine->device_id());
    if (it == device_info_cache().end()) return false;
    if (result) *result = it->second;
    return true;
}

void device_info_cache_set(impl::engine_t *engine,
        const std::shared_ptr<device_info_t> &device_info) {
    utils::lock_write_t lock(device_info_cache_mutex());

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    // Clear the cache to avoid hypothetically large growth.
    const int cache_size_threshold = 1024;
    if (device_info_cache().size() > cache_size_threshold)
        device_info_cache().clear();

    device_info_cache().insert({compute_engine->device_id(), device_info});
}

status_t compute_engine_t::init() {
    return init({});
}

status_t compute_engine_t::init(const std::vector<uint8_t> &cache_blob) {
    if (device_info_cache_get(&device_info_, this)) return status::success;
    // Since init_device_info that takes a cache blob is only defined for
    // OpenCL we need to do manual dispatching here.
    if (cache_blob.empty())
        CHECK(init_device_info());
    else
        CHECK(init_device_info(cache_blob));
    device_info_cache_set(this, device_info_);

    return status::success;
}

const char *cl_microkernels_check_kernel_code = R""""(
kernel void igc_check() {
    __asm__ volatile(
            ".decl AA0 v_type=G type=ud num_elts=1\n"
            ".decl AA1 v_type=G type=ud num_elts=1\n"
            ".implicit_PSEUDO_INPUT AA0 offset=256 size=4\n"
            ".implicit_PSEUDO_INPUT AA1 offset=256 size=4\n"
            "mov (M1_NM,1) AA0(0,0)<1> AA1(0,0)<0;1,0>\n"
    );
}
)"""";

bool mayiuse_microkernels(const compute_engine_t *engine) {
    auto *device_info = engine->device_info();
    if (device_info->runtime_version() >= xpu::runtime_version_t(24, 22, 29735))
        return true;

    auto mayiuse_mk = [](const compute_engine_t *engine) {
        switch (engine->runtime_kind()) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            case runtime_kind::ocl: {

                auto *ocl_engine
                        = utils::downcast<const ocl::engine_t *>(engine);
                return ocl::try_building(ocl_engine->context(),
                        ocl_engine->device(),
                        cl_microkernels_check_kernel_code);
            }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            case runtime_kind::sycl:
                return sycl::mayiuse_microkernels(
                        utils::downcast<const sycl::engine_t *>(engine));
#endif
            default: return false;
        }
    };
    static std::map<engine_id_t, bool> engine_microkernel_map {
            {engine->engine_id(), mayiuse_mk(engine)}};
    static std::mutex map_mutex;
    std::lock_guard<std::mutex> map_lock(map_mutex);
    auto it = engine_microkernel_map.find(engine->engine_id());
    if (it != std::end(engine_microkernel_map)) return it->second;
    return engine_microkernel_map[engine->engine_id()] = mayiuse_mk(engine);
}

status_t compute_engine_t::create_kernel_from_cache_blob(
        const cache_blob_t &cache_blob, compute::kernel_t &kernel,
        const char *kernel_name) const {
    std::vector<compute::kernel_t> kernels;
    CHECK(create_kernels_from_cache_blob(cache_blob, kernels, {kernel_name}));
    if (kernels.size() != 1) return status::runtime_error;
    kernel = std::move(kernels[0]);
    return status::success;
};

status_t compute_engine_t::create_kernel_bundle(kernel_bundle_t &bundle,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {
    std::vector<kernel_t> kernels;
    CHECK(create_kernels(&kernels, kernel_names, kernel_ctx));
    bundle = kernel_bundle_t(std::move(kernels), kernel_names);
    return status::success;
}

bool compute_engine_t::mayiuse(device_ext_t ext) const {
    return device_info_->has(ext);
}

bool compute_engine_t::mayiuse_ngen_kernels() const {
    return device_info_->mayiuse_ngen_kernels();
}

bool compute_engine_t::mayiuse_sub_group(int size) const {
    return device_info_->mayiuse_sub_group(size);
}

bool compute_engine_t::mayiuse_block_reads_writes_with_sub_group(
        int size) const {
    return size <= 16 ? true
                      : device_info_->gpu_arch() >= compute::gpu_arch_t::xe_hpc;
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

bool dnnl_impl_gpu_mayiuse_ngen_kernels(dnnl::impl::engine_t *engine) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::gpu::intel::compute;

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    return compute_engine->mayiuse_ngen_kernels();
}
