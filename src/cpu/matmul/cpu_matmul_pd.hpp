/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
    // NOLINTBEGIN(google-default-arguments)
    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) const override {
        bool ok = matmul_pd_t::attr_scales_ok(supported_args);
        const auto &scales = attr()->scales_;
        for (int arg : supported_args) {
            if (scales.has_default_values(arg)) { continue; }
            const auto &g0 = scales.get_group(arg, 0);
            const auto &g1 = scales.get_group(arg, 1);

            // Any group is allowed to be greater than 1 but only one at a
            // time, not both.
            ok = ok
                    && IMPLICATION(!scales.get(arg).has_default_groups(),
                            utils::one_of(1, g0, g1));
        }
        return ok;
    }
    // NOLINTEND(google-default-arguments)
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
