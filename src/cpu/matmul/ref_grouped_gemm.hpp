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

#ifndef CPU_MATMUL_REF_GROUPED_GEMM_HPP
#define CPU_MATMUL_REF_GROUPED_GEMM_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_grouped_gemm_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref_grouped:any", ref_grouped_gemm_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Check for grouped encoding on src and dst
            VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Weights should be dense
            VDISPATCH_MATMUL(
                    !wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Extract grouped encoding
            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            const auto &dst_grouped = dst_d.sparse_desc().grouped_desc;

            // Validate matching number of groups
            VDISPATCH_MATMUL(src_grouped.ngroups == dst_grouped.ngroups,
                    VERBOSE_INCONSISTENT_DIM, "src_ngroups",
                    (int)src_grouped.ngroups, "dst_ngroups",
                    (int)dst_grouped.ngroups);

            // Check data types - support f32 for now
            VDISPATCH_MATMUL(
                    utils::everyone_is(f32, src_type, wei_type, dst_type),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Check offsets are int32
            VDISPATCH_MATMUL(src_d.metadata_type(0) == s32
                            && dst_d.metadata_type(0) == s32,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Check for limited Bias support
            if (with_bias()) {
                memory_desc_wrapper bia_d(weights_md(1));
                VDISPATCH_MATMUL(
                        !bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                        VERBOSE_UNSUPPORTED_BIAS_CFG);
                VDISPATCH_MATMUL(
                        bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
                // Bias shape should be [num_experts, N]
                VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.ngroups,
                        VERBOSE_INCONSISTENT_DIM, "bias_dim[0]",
                        (int)bia_d.dims()[0], "ngroups",
                        (int)src_grouped.ngroups);
                VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                        VERBOSE_INCONSISTENT_DIM, "bias_dim[1]",
                        (int)bia_d.dims()[1], "weights_dim[2]",
                        (int)wei_d.dims()[2]);
            }

            // No scales/post-ops for now
            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            return status::success;
        }
    };

    ref_grouped_gemm_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_MATMUL_REF_GROUPED_GEMM_HPP
