/******************************************************************************
 * Copyright 2025
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
 ******************************************************************************/

#ifndef CPU_RV64_RVV_BINARY_HPP
#define CPU_RV64_RVV_BINARY_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_binary_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <impl::data_type_t date_type>
struct rvv_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;
        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_binary_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;

            VDISPATCH_BINARY(utils::everyone_is(date_type, src_md(0)->data_type,
                                     src_md(1)->data_type, dst_md()->data_type)
                            && platform::has_data_type_support(
                                    src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(IMPLICATION(is_ternary_op(),
                                     platform::has_data_type_support(
                                             src_md(2)->data_type)),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_BINARY(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            const bool layouts_identical
                    = src0_d.similar_to(dst_d, /*with_padding=*/true,
                              /*with_data_type=*/true)
                    && src1_d.similar_to(dst_d, /*with_padding=*/true,
                            /*with_data_type=*/true);

            use_dense_ = src0_d.is_dense(/*with_padding=*/false)
                    && src1_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false)
                    && layouts_identical;
            use_nCspBc_padded_ = !use_dense_
                    && src0_d.blocking_desc().inner_nblks == 1
                    && src1_d.blocking_desc().inner_nblks == 1
                    && dst_d.blocking_desc().inner_nblks == 1
                    && utils::one_of(
                            src0_d.blocking_desc().inner_blks[0], 8, 16)
                    && src0_d.blocking_desc().inner_blks[0]
                            == src1_d.blocking_desc().inner_blks[0]
                    && src0_d.blocking_desc().inner_blks[0]
                            == dst_d.blocking_desc().inner_blks[0]
                    && src0_d.blocking_desc().inner_idxs[0] == 1
                    && src1_d.blocking_desc().inner_idxs[0] == 1
                    && dst_d.blocking_desc().inner_idxs[0] == 1
                    && src0_d.only_padded_dim(1) && src1_d.only_padded_dim(1)
                    && dst_d.only_padded_dim(1) && src0_d.is_dense(true)
                    && src1_d.is_dense(true) && dst_d.is_dense(true);

            VDISPATCH_BINARY(use_dense_ || use_nCspBc_padded_,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            return status::success;
        }

        bool use_dense_, use_nCspBc_padded_;

    private:
        bool check_scales_mask() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};
            return attr_scales_ok(supported_args);
        }
    };

    rvv_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_binary(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_binary(const exec_ctx_t &ctx) const;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_BINARY_HPP