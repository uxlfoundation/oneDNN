/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_MATMUL_CPU_MATMUL_PD_HPP
#define CPU_MATMUL_CPU_MATMUL_PD_HPP

#include "common/matmul_pd.hpp"

#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct cpu_matmul_pd_t : public matmul_pd_t {
    using matmul_pd_t::matmul_pd_t;

    bool has_per_n_dst_scales() const {
        const auto &dst_scales = attr()->scales_.get(DNNL_ARG_DST);
        return !dst_scales.has_default_values()
                && dst_scales.get_mask() == dst_qmask_N();
    }

    // NOLINTBEGIN(google-default-arguments)
    status_t attr_scales_ok(const engine_t *engine,
            const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
            const std::vector<int> &supported_qmodes
            = {quantization_mode::static_sazp},
            const std::map<int, std::vector<int>> &extra_masks
            = {}) const override {
        VDISPATCH_MATMUL(
                !has_per_n_dst_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
        CHECK(matmul_pd_t::attr_scales_ok(
                engine, supported_args, supported_qmodes, extra_masks));
        return status::success;
    }
    // NOLINTEND(google-default-arguments)
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
