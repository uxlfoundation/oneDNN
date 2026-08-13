/*******************************************************************************
* Copyright 2026 Léandre Le Duc
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

#ifndef CPU_RV64_SHUFFLE_JIT_UNI_SHUFFLE_HPP
#define CPU_RV64_SHUFFLE_JIT_UNI_SHUFFLE_HPP

#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_exec_types.hpp"

#include "common/shuffle_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_shuffle_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_shuffle_conf_t {
    dim_t mb = 0, c = 0, sp = 0;
    dim_t stride_mb = 0;
    dim_t dt_size = 0;
    dim_t blk_size = 0;
    dim_t nb_blk = 0;
};

struct jit_uni_shuffle_kernel_t;

struct jit_uni_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("jit:rvv", jit_uni_shuffle_t);

        status_t init(const engine_t *engine) {
            UNUSED(engine);
            using namespace dnnl::impl::data_type;
            using namespace format_tag;

            const memory_desc_wrapper src_d(
                    is_fwd() ? src_md() : diff_src_md());
            const memory_desc_wrapper dst_d(
                    is_fwd() ? dst_md() : diff_dst_md());

            const data_type_t dt = src_d.data_type();

            // All datatype are supported, since it's only memory move
            // However, at least the  vector extension is needed
            VDISPATCH_SHUFFLE(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_SHUFFLE(dt == dst_d.data_type(), VERBOSE_INCONSISTENT_DT,
                    "src", "dst");
            VDISPATCH_SHUFFLE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            // We do not make compute op on the elements, only memory manipulation
            // so we support all the data 32/16 sized types.
            VDISPATCH_SHUFFLE(utils::one_of(dt, f32, s32, f16, bf16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SHUFFLE(platform::has_data_type_support(dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SHUFFLE(axis() == 1, VERBOSE_BAD_AXIS);
            VDISPATCH_SHUFFLE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_SHUFFLE(
                    impl::is_dense_format_kind({src_d.md_, dst_d.md_}),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            conf_.mb = MB();
            conf_.c = C();
            conf_.sp = D() * H() * W();
            conf_.dt_size = types::data_type_size(dt);
            conf_.stride_mb = src_d.blocking_desc().strides[0];
            const format_tag_t blocked_tag = memory_desc_matches_one_of_tag(
                    *src_d.md_, nCw4c, nCw8c, nCw16c, nChw4c, nChw8c, nChw16c,
                    nCdhw4c, nCdhw8c, nCdhw16c);
            const format_tag_t nspc_tag = memory_desc_matches_one_of_tag(
                    *src_d.md_, nc, nwc, nhwc, ndhwc);

            VDISPATCH_SHUFFLE(blocked_tag != format_tag::undef
                            || nspc_tag != format_tag::undef,
                    VERBOSE_UNSUPPORTED_TAG);

            if (blocked_tag != format_tag::undef) {
                conf_.blk_size = src_d.blocking_desc().inner_blks[0];
                VDISPATCH_SHUFFLE(
                        conf_.c % conf_.blk_size == 0, VERBOSE_UNSUPPORTED_TAG);
                conf_.nb_blk = conf_.c / conf_.blk_size;
            } else {
                // nspc
                conf_.blk_size = conf_.c;
                conf_.nb_blk = 1;
            }
            // Since offset are uint32_t, we need to ensure they are smaller than UINT32_MAX.
            const dim_t max_off
                    = ((conf_.nb_blk - 1) * conf_.sp * conf_.blk_size
                              + conf_.blk_size - 1)
                    * conf_.dt_size;
            VDISPATCH_SHUFFLE(
                    max_off <= (dim_t)UINT32_MAX, VERBOSE_UNSUPPORTED_TAG);
            return status::success;
        }

        const jit_shuffle_conf_t &get_conf() const { return conf_; }

    private:
        jit_shuffle_conf_t conf_;
    };

    jit_uni_shuffle_t(const pd_t *apd);
    ~jit_uni_shuffle_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_shuffle_kernel_t> kernel_;
    status_t precompute_offsets();
    uint32_t *input_off_ = nullptr;
};
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_SHUFFLE_JIT_UNI_SHUFFLE_HPP
