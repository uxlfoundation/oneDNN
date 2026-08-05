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

#include "gpu/intel/compute/ukernels.hpp"

#include "common/verbose.hpp"
#include "gemmstone/microkernel/shim.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/utils.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/utils.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "gpu/intel/ze/engine.hpp"
#include "gpu/intel/ze/utils.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

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

bool mayiuse_microkernels(const engine_t *engine) {
    auto *device_info = engine->device_info();
    if (device_info->runtime_version() >= xpu::runtime_version_t(24, 22, 29735))
        return true;

    auto mayiuse_mk = [](const engine_t *engine) {
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
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
            case runtime_kind::ze: {
                auto *ze_engine = utils::downcast<const ze::engine_t *>(engine);
                return ze::mayiuse_microkernels(ze_engine->device(),
                        ze_engine->context(),
                        cl_microkernels_check_kernel_code);
            }
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

status_t validate_microkernel(const gemmstone::microkernel::Package &package,
        const char *kernel_name) {
    using Status = gemmstone::microkernel::Package::Status;
    const char *reason = nullptr;
    switch (package.status) {
        case Status::Success: return status::success;
        case Status::Pending:
            reason = "was not finalized before validation";
            break;
        case Status::UncertainClobbers:
            reason = "has uncertain register clobbers and is not supported";
            break;
        case Status::UnsupportedHW:
            reason = "is incompatible with the current hardware";
            break;
    }
    VCONDCHECK(primitive, create, check, gpu, reason == nullptr,
            status::unimplemented, "%s microkernel %s", kernel_name, reason);
    return status::runtime_error;
}

void microkernel_shims_t::add(const char *header_name, const char *decorator,
        const gemmstone::microkernel::Package &package) {
    gemmstone::microkernel::ShimOptions options;
    options.subgroupSize = subgroup_size_;
    options.useTileOps = true;
    options.decorator = decorator;
    options.microkernelID = next_id_++;

    kernel_ctx_.add_custom_header(header_name,
            generateShim(package,
                    gemmstone::microkernel::HostLanguage::OpenCL_C, options));
    require_grfs(package.grfMin);
}

void microkernel_shims_t::finalize() {
    if (arch_ >= gpu_arch_t::xe3p && grf_min_ > 256)
        kernel_ctx_.add_option("-cl-intel-512-GRF-per-thread");
    else if (grf_min_ > 128)
        kernel_ctx_.add_option("-cl-intel-256-GRF-per-thread");
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
