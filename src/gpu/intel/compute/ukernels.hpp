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

#ifndef GPU_INTEL_COMPUTE_UKERNELS_HPP
#define GPU_INTEL_COMPUTE_UKERNELS_HPP

#include <algorithm>

#include "common/engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/engine.hpp"

namespace gemmstone {
namespace microkernel {
struct Package; // NOLINT(readability-identifier-naming)
}
} // namespace gemmstone

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

extern const char *cl_microkernels_check_kernel_code;
bool mayiuse_microkernels(const engine_t *engine);

// Validates a finalized microkernel package, emitting a verbose message and
// returning a non-success status if it cannot be used.
status_t validate_microkernel(const gemmstone::microkernel::Package &package,
        const char *kernel_name);

// Embeds microkernel shims, assigning IDs and requesting the needed GRF mode.
class microkernel_shims_t {
public:
    microkernel_shims_t(
            kernel_ctx_t &kernel_ctx, int subgroup_size, gpu_arch_t arch)
        : kernel_ctx_(kernel_ctx), subgroup_size_(subgroup_size), arch_(arch) {}

    void add(const char *header_name, const char *decorator,
            const gemmstone::microkernel::Package &package);

    void require_grfs(int grf_min) { grf_min_ = std::max(grf_min_, grf_min); }

    // Applies the GRF mode required by the packages added so far.
    void finalize();

private:
    kernel_ctx_t &kernel_ctx_;
    int subgroup_size_;
    gpu_arch_t arch_;
    uint32_t next_id_ = 0;
    int grf_min_ = 0;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_COMPUTE_UKERNELS_HPP
