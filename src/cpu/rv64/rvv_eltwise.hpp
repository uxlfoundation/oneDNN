/*******************************************************************************
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
*******************************************************************************/

#ifndef CPU_RV64_RVV_ELTWISE_HPP
#define CPU_RV64_RVV_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            const data_type_t d_type = dst_md()->data_type;
            using namespace dnnl::impl::data_type;
            bool type_ok = utils::one_of(d_type, f32, s32, s8, u8);
            VDISPATCH_ELTWISE(type_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    src_md()->data_type == d_type, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(check_alg_kind(), VERBOSE_UNSUPPORTED_TAG);

            use_dense_ = src_d.is_dense(true) && dst_d.is_dense(true)
                    && IMPLICATION(!src_d.is_dense() || !dst_d.is_dense(),
                            is_zero_preserved());
            VDISPATCH_ELTWISE(use_dense_, VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool use_dense_;

        bool check_alg_kind() const {
            return utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                    alg_kind::eltwise_square, alg_kind::eltwise_abs,
                    alg_kind::eltwise_sqrt, alg_kind::eltwise_linear,
                    alg_kind::eltwise_clip, alg_kind::eltwise_hardsigmoid,
                    alg_kind::eltwise_hardswish);
        }
    };

    rvv_eltwise_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct rvv_eltwise_bwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        using cpu_eltwise_bwd_pd_t::cpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", rvv_eltwise_bwd_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper src_d(data_md());

            const data_type_t d_type = src_md()->data_type;
            using namespace dnnl::impl::data_type;
            bool type_ok = utils::one_of(d_type, f32, s32, s8, u8);
            VDISPATCH_ELTWISE(type_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(d_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(diff_src_d == diff_dst_d,
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            VDISPATCH_ELTWISE(check_alg_kind(), VERBOSE_UNSUPPORTED_TAG);

            use_dense_ = diff_dst_d.is_dense()
                    || (diff_dst_d.is_dense(true) && is_zero_preserved());
            VDISPATCH_ELTWISE(use_dense_, VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        bool use_dense_;

        bool check_alg_kind() const {
            return utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                    alg_kind::eltwise_square, alg_kind::eltwise_abs,
                    alg_kind::eltwise_sqrt, alg_kind::eltwise_linear,
                    alg_kind::eltwise_clip, alg_kind::eltwise_hardsigmoid,
                    alg_kind::eltwise_hardswish);
        }
    };

    rvv_eltwise_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_ELTWISE_HPP
