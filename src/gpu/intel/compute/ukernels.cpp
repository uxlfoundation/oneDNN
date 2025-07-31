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

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
