/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
* Copyright 2026 Intel Corporation
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

#ifndef CPU_RV64_JIT_UNI_REDUCTION_HPP
#define CPU_RV64_JIT_UNI_REDUCTION_HPP

#include <memory>

#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_reduction_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_reduction_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_uni_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reduction_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            VDISPATCH_REDUCTION(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_REDUCTION(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            conf_.src_type = src_md()->data_type;
            conf_.dst_type = dst_md()->data_type;
            conf_.src_dt_size = types::data_type_size(conf_.src_type);
            conf_.dst_dt_size = types::data_type_size(conf_.dst_type);
            conf_.alg = desc()->alg_kind;

            VDISPATCH_REDUCTION(
                    is_supported_alg(conf_.alg), VERBOSE_BAD_ALGORITHM);
            VDISPATCH_REDUCTION(
                    is_supported_dt_pair(conf_.src_type, conf_.dst_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(
                    IMPLICATION(conf_.src_type == f16 || conf_.dst_type == f16,
                            mayiuse(zvfh)),
                    VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_REDUCTION(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(
                    impl::is_dense_format_kind({src_md(), dst_md()}),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            const memory_desc_wrapper src_mdw(src_md());
            const memory_desc_wrapper dst_mdw(dst_md());

            const format_tag_t src_tag = memory_desc_matches_one_of_tag(
                    *src_md(), x, nc, ncw, nchw, ncdhw);
            const format_tag_t dst_tag = memory_desc_matches_one_of_tag(
                    *dst_md(), x, nc, ncw, nchw, ncdhw);
            VDISPATCH_REDUCTION(
                    src_tag != format_tag::undef && src_tag == dst_tag,
                    VERBOSE_UNSUPPORTED_TAG);

            const int ndims = src_mdw.ndims();
            const auto &src_dims = src_mdw.dims();
            const auto &dst_dims = dst_mdw.dims();

            int reduced_ndims = 0;
            conf_.idle_size = dst_mdw.nelems();
            conf_.reduce_size = 1;
            for (int d = ndims - 1; d >= 0; --d) {
                if (src_dims[d] != dst_dims[d]) {
                    VDISPATCH_REDUCTION(
                            dst_dims[d] == 1, VERBOSE_UNSUPPORTED_TAG);
                    ++reduced_ndims;
                    conf_.reduce_size *= src_dims[d];
                } else {
                    break;
                }
            }

            VDISPATCH_REDUCTION(reduced_ndims != 0,
                    "dimensionality reduction not possible");
            VDISPATCH_REDUCTION(
                    conf_.reduce_size > 0, VERBOSE_EMPTY_TENSOR, "");

            for (int d = 0; d < ndims - reduced_ndims; ++d) {
                VDISPATCH_REDUCTION(
                        src_dims[d] == dst_dims[d], VERBOSE_UNSUPPORTED_TAG);
            }

            return status::success;
        }

        const jit_reduction_conf_t &get_conf() const { return conf_; }

    private:
        static bool is_supported_dt_pair(
                data_type_t src_type, data_type_t dst_type) {
            using namespace data_type;
            return utils::one_of(src_type, f32, f16)
                    && utils::one_of(dst_type, f32, f16);
        }

        static bool is_supported_alg(alg_kind_t alg) {
            using namespace alg_kind;
            return utils::one_of(alg, reduction_sum, reduction_mean,
                    reduction_max, reduction_min);
        }

        jit_reduction_conf_t conf_ = {};
    };

    jit_uni_reduction_t(const pd_t *apd) : primitive_t(apd) {}
    ~jit_uni_reduction_t() override = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_reduction_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
