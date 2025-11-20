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

#ifndef CPU_RV64_RVV_INNER_PRODUCT_HPP
#define CPU_RV64_RVV_INNER_PRODUCT_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_INNER_PRODUCT(
                    check_types(src_type, wei_type, dst_type, bia_type),
                    VERBOSE_UNSUPPORTED_DT);

            using smask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(smask_t::none),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_INNER_PRODUCT(
                    set_default_params(false) == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper wei_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            VDISPATCH_INNER_PRODUCT(check_layouts(src_d, wei_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);

            nthr_ = dnnl_get_max_threads();

            // Book scratchpad for contiguous packing of src/weights per-thread
            {
                auto scratchpad = scratchpad_registry().registrar();
                const dim_t K
                        = (dim_t)IC() * (dim_t)KD() * (dim_t)KH() * (dim_t)KW();
                const size_t src_elt = types::data_type_size(src_type);
                const size_t wei_elt = types::data_type_size(wei_type);
                scratchpad.book(memory_tracking::names::key_iprod_src_reorder,
                        (size_t)K * (size_t)nthr_, src_elt);
                scratchpad.book(
                        memory_tracking::names::key_iprod_weights_reorder,
                        (size_t)K * (size_t)nthr_, wei_elt);
            }

            return status::success;
        }

        bool check_types(const data_type_t &src_type,
                const data_type_t &wei_type, const data_type_t &dst_type,
                const data_type_t &bia_type) const {
            using namespace data_type;
            const bool dst_ok = utils::one_of(dst_type, f32, s32, s8, u8);
            const bool src_wei_ok = (src_type == f32 && wei_type == f32)
                    || (src_type == s8 && wei_type == s8)
                    || (src_type == u8 && wei_type == s8);
            const bool bia_ok = IMPLICATION(
                    with_bias(), utils::one_of(bia_type, f32, src_type));
            return dst_ok && src_wei_ok && bia_ok
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(wei_type)
                    && platform::has_data_type_support(dst_type);
        }

        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &wei_d,
                const memory_desc_wrapper &dst_d) const {
            // Supported memory layouts: plain, dense, no inner blocks
            const bool plain_dense = src_d.blocking_desc().inner_nblks == 0
                    && wei_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(/*with_padding=*/false)
                    && wei_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false);
            if (!plain_dense) return false;

            // Support only 2D source/weights
            if (ndims() != 2) return false;

            const auto *sd = src_d.dims();
            const auto *wd = wei_d.dims();
            if (sd[0] != MB()) return false; // MB
            if (wd[0] != OC()) return false; // OC
            if (sd[1] != wd[1]) return false; // IC
            for (int i = 2; i < ndims(); ++i) {
                if (sd[i] != wd[i]) return false; // KD/KH/KW...
            }
            if (dst_d.dims()[0] != MB() || dst_d.dims()[1] != OC())
                return false;
            return true;
        }

        int nthr_; // To not exceed the limit in execute used for set up.
    };

    rvv_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_INNER_PRODUCT_HPP
