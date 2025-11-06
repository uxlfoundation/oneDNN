/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_KERNEL_HPP
#define GPU_INTEL_CONV_JIT_KERNEL_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"

#include "gpu/intel/conv/jit/config.hpp"
#include "gpu/intel/conv/jit/ir_builder.hpp"
#include "gpu/intel/conv/jit/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

class conv_kernel_t : public ir_kernel_t {
public:
    conv_kernel_t(const config_t &cfg, const kernel_info_t &kernel_info,
            const compute::range_t &local_range, const layout_t &zp_dst)
        : ir_kernel_t(kernel_info.iface("gen_conv"), cfg.options(), local_range,
                {GENERATOR_NAME, GENERATOR_LINE})
        , prb_(cfg.prb())
        , cfg_(cfg) {

        // XXX: BWD_W does 32x32 multiplication in the inner loop which may cause
        // hangs when using with split barrier. Switch to emulation to work around
        // the issue.
        if (prb_.is_bwd_w && options().hw() < ngen::HW::XeHPC)
            force_emulate64();

        ir_utils::debug_profiler_t profile("Conv Kernel Construction Profile");
        // Build IR for the kernel.
        builder_t builder(cfg, kernel_info, zp_dst);
        const stmt_t &body = builder.stmt();
        profile.stamp("Kernel Builder");
        generate_from_ir(
                body, &cfg_.plan().gemm_schedule.kernel_grid_walk_order());
        profile.stop("Generate Assembly");

#ifdef DNNL_DEV_MODE
        gpu_perf_no_trace() << profile;

        gpu_trace() << "Actual register usage:           " << peak_regs();
        int estimated_peak_regs = estimate_register_count(cfg_);
        if (peak_regs() > estimated_peak_regs) {
            gpu_warning() << "conv_kernel_t register usage underestimated: "
                             "estimate = "
                          << estimated_peak_regs
                          << ", actual = " << peak_regs();
        }
#endif
    }

private:
    const problem_t &prb_;
    const config_t &cfg_;
};

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
