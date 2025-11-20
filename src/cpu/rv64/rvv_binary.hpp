/******************************************************************************
 * Copyright 2025 ZTE Corporation
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

struct rvv_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;
        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_binary_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            const data_type_t d_type = dst_md()->data_type;

            VDISPATCH_BINARY(utils::everyone_is(d_type, src_md(0)->data_type,
                                     src_md(1)->data_type)
                            && platform::has_data_type_support(
                                    src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_BINARY(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_BINARY_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_BINARY(check_layouts(src0_d, src1_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool check_layouts(const memory_desc_wrapper &src0_d,
                const memory_desc_wrapper &src1_d,
                const memory_desc_wrapper &dst_d) const {
            bool plain_dense = src0_d.blocking_desc().inner_nblks == 0
                    && src1_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src0_d.is_dense(/*with_padding=*/false)
                    && src1_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false) && ndims() == 4;
            bool no_broadcast = true;
            if (plain_dense) {
                for (int d = 0; d < ndims(); ++d) {
                    const dim_t a = src0_d.dims()[d];
                    const dim_t b = src1_d.dims()[d];
                    const dim_t c = dst_d.dims()[d];
                    if (!(a == b && a == c)) {
                        no_broadcast = false;
                        break;
                    }
                }
            }
            bool same_layouts = false;
            if (plain_dense && no_broadcast) {
                same_layouts = src0_d.is_plain() && src1_d.is_plain()
                        && dst_d.is_plain()
                        && src0_d.similar_to(src1_d, /*with_strides=*/true,
                                /*with_pads=*/false)
                        && src0_d.similar_to(dst_d, /*with_strides=*/true,
                                /*with_pads=*/false);
            }
            return plain_dense && no_broadcast && same_layouts;
        }
    };

    rvv_binary_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_BINARY_HPP
