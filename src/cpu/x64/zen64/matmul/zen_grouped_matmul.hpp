/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_MATMUL_ZEN_GROUPED_MATMUL_HPP
#define CPU_X64_ZEN64_MATMUL_ZEN_GROUPED_MATMUL_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/matmul/lowoha_common.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace matmul {

#if DNNL_X64_USE_ZEN
namespace zen_gmm = zendnnl::lowoha::matmul;
#endif

struct zen_grouped_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("zen:grouped_matmul", zen_grouped_matmul_t);

        status_t init(const engine_t *engine);
    };

    zen_grouped_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    // Build the Zen eltwise post-op chain once per primitive; it is
    // expert-independent and reused for every group at execute() time.
    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

#if DNNL_X64_USE_ZEN
    // Pre-built Zen post-op chain, copied into each group's params at
    // execute() time. Eltwise entries are complete; each binary_mul entry
    // carries only its static type, and its per-expert [M_g, N] operand
    // (buff / dims / leading_dim) is patched in at execute() time.
    std::vector<zen_gmm::matmul_post_op> postop_template_;
    // Per binary_mul post-op: slot in postop_template_, oneDNN post-op index
    // (to fetch the src1 buffer), src1 element size (to slice per expert), and
    // whether src1 is grouped (its offsets are validated against dst at exec).
    struct binary_src1_t {
        int chain_idx;
        int po_idx;
        size_t elem_sz;
        bool is_grouped;
    };
    std::vector<binary_src1_t> binary_src1_;
#endif
};

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
