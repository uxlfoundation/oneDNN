/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/conv/jit/grf_usage.hpp"

#include "gpu/intel/conv/jit/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

using namespace intel::jit;

void compare(const grf_usage_t &est_usage, const grf_usage_t &ir_usage) {
    std::vector<std::string> headers
            = {"Label", "Estimated regs", "IR regs", "Status"};
    ir_utils::table_t table("GRF usage", headers);
    int est_total = 0;
    int ir_total = 0;
    for (auto label : all_grf_usage_labels()) {
        int est_regs = est_usage.get(label);
        int ir_regs = ir_usage.get(label);
        table << to_string(label) << est_regs << ir_regs;
        table << (ir_regs > est_regs ? "FAIL" : "");
        table << std::endl;
        est_total += est_regs;
        ir_total += ir_regs;
    }
    table << "Total" << est_total << ir_total;
    table << (ir_total > est_total ? "FAIL" : "");
    table << std::endl;
    gpu_trace() << table;
    gpu_trace() << ir_usage.buf_usage();
}

void verify_grf_usage(
        const conv_config_t &cfg, const stmt_t &body, int external_usage) {
    grf_usage_t gpu_info = get_grf_usage(body, cfg.grf_size(), external_usage);
    auto est_info = cfg.plan().grf_usage();
    compare(est_info, gpu_info);
}

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
