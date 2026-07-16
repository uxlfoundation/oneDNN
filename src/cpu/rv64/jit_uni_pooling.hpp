/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
* Copyright 2026 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_JIT_UNI_POOLING_HPP
#define CPU_RV64_JIT_UNI_POOLING_HPP

#include <memory>

#include "common/primitive.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Forward pooling. The retained native kernel (jit_uni_pool_ncsp_kernel_t)
// handles the plain nspc and ncsp layouts for both inference and training (f32
// via v, f16 via zvfh; max and avg). The x64/aarch64-style baked kernel
// (jit_uni_pool_kernel_t) handles the blocked (nChw{c_block}c) layout. rv64 has
// no JIT plain<->blocked transpose, so the two layout families never mix.
template <cpu_isa_t isa>
struct jit_uni_pooling_fwd_t : public primitive_t {
    static constexpr data_type_t d_type
            = (isa == zvfh) ? data_type::f16 : data_type::f32;

    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jpp_.isa, ""),
                jit_uni_pooling_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace utils;
            using namespace data_type;
            using namespace prop_kind;

            VDISPATCH_POOLING(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_POOLING(one_of(desc_.prop_kind, forward_inference,
                                      forward_training),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(one_of(desc()->alg_kind, alg_kind::pooling_max,
                                      alg_kind::pooling_avg_include_padding,
                                      alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(everyone_is(d_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            constexpr bool is_f16 = d_type == data_type::f16;
            VDISPATCH_POOLING(
                    IMPLICATION(
                            is_f16, desc()->accum_data_type == data_type::f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(memory_desc_wrapper(dst_md()).is_dense(false),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_POOLING(attr()->has_default_values(
                                      primitive_attr_t::skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            // Resolve post-op binary src1 formats (may be format_any) against dst
            // before the kernels inspect their layout; matches x86/ARM pooling.
            VDISPATCH_POOLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Max training needs the argmax workspace; set it before init_conf,
            // which reads workspace_md() for the index dtype.
            const bool is_training
                    = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            // Plain layouts (nspc/ncsp) use the retained native kernel. It
            // handles forward inference and training for both layouts, f32 and
            // f16, max (max training writes the argmax workspace) and avg. It
            // declines blocked, and f16 max training with a window too large for
            // a 16-bit index, which fall to the baked kernel below.
            {
                status_t st
                        = jit_uni_pool_ncsp_kernel_t<isa, d_type>::init_conf(
                                jpp_, attr_, this);
                if (st == status::success) return status::success;
            }

            // Baked kernel: blocked (all fwd props) and the plain training cases
            // native declined (f16/post-op max training).
            auto scratchpad = scratchpad_registry().registrar();
            status_t st = jit_uni_pool_kernel_t<isa>::init_conf(
                    jpp_, scratchpad, attr_, this);
            VDISPATCH_POOLING(st == status::success, VERBOSE_UNSUPPORTED_TAG);

            // f32 nspc reaching the baked kernel (a case the native kernel
            // declined) issues many narrow per-channel-block calls with little
            // per-element work, so it defers to the nhwc_pooling reduction. f16
            // nspc keeps the baked kernel (its f16<->f32 conversion outweighs the
            // call overhead), and blocked has no reduction fallback, so both
            // stay.
            VDISPATCH_POOLING(
                    !(d_type == data_type::f32
                            && jpp_.tag_kind == jit_pool_tag_kind_t::nspc),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "f32 nspc forward defers to the nhwc_pooling reduction");
            return status::success;
        }

        jit_pool_conf_t jpp_;
    };

    jit_uni_pooling_fwd_t(const pd_t *apd);
    ~jit_uni_pooling_fwd_t() override;

    using data_t = typename prec_traits_t<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    // Baked kernel drivers (blocked/nspc), ported from aarch64.
    void execute_forward_blk(const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx) const;
    void execute_forward_blk_3d(const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_pool_kernel_t<isa>> kernel_;
    std::unique_ptr<jit_uni_pool_ncsp_kernel_t<isa, d_type>> ncsp_kernel_;
    // nspc f32 forward-inference interior (shape-baked ur_w reuse); null else.
    std::unique_ptr<jit_uni_pool_interior_kernel_t<isa, d_type>>
            interior_kernel_;
};

// Backward pooling. Plain layouts (nspc/ncsp) use the native gather kernel
// (jit_uni_pool_bwd_kernel_t); blocked uses the x64/aarch64-style baked kernel.
// Both handle max (via the index workspace) and avg.
template <cpu_isa_t isa>
struct jit_uni_pooling_bwd_t : public primitive_t {
    static constexpr data_type_t d_type
            = (isa == zvfh) ? data_type::f16 : data_type::f32;

    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jpp_.isa, ""),
                jit_uni_pooling_bwd_t);

        status_t init(const engine_t *engine) {
            using namespace utils;
            using namespace data_type;

            VDISPATCH_POOLING(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(one_of(desc()->alg_kind, alg_kind::pooling_max,
                                      alg_kind::pooling_avg_include_padding,
                                      alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(everyone_is(d_type, diff_src_md()->data_type,
                                      diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    platform::has_data_type_support(diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            constexpr bool is_f16 = d_type == data_type::f16;
            VDISPATCH_POOLING(
                    IMPLICATION(
                            is_f16, desc()->accum_data_type == data_type::f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            if (desc()->alg_kind == alg_kind::pooling_max) {
                init_default_ws();
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            // The native gather backward kernel is used only for f16 nspc, where
            // its vectorized f16<->f32 conversion outweighs the extra traffic.
            // f32 (both layouts) and f16 ncsp keep the pre-existing dispatch
            // (baked for blocked/nspc, the nhwc/nchw reference otherwise). For a
            // non-nspc f16 layout init_conf still populates jpp_, but the baked
            // path below repopulates it.
            if (d_type == data_type::f16) {
                status_t st = jit_uni_pool_bwd_kernel_t<isa, d_type>::init_conf(
                        jpp_, this);
                if (st == status::success
                        && jpp_.tag_kind == jit_pool_tag_kind_t::nspc)
                    return status::success;
            }

            // Baked kernel: blocked and nspc (pre-existing path).
            auto scratchpad = scratchpad_registry().registrar();
            VDISPATCH_POOLING(jit_uni_pool_kernel_t<isa>::init_conf(
                                      jpp_, scratchpad, attr_, this)
                            == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            // f32 nspc backward is a memory-bound read-modify-write (each input
            // is updated once per covering output window, ~ceil(K/S)^ndims x the
            // diff_src traffic), which loses to the nhwc_pooling gather (each
            // input written once); with no dtype conversion to amortize it, defer
            // to that reference. f16 nspc took the native kernel above, and
            // blocked has no gather fallback, so it stays on the baked kernel.
            VDISPATCH_POOLING(
                    !(d_type == data_type::f32
                            && jpp_.tag_kind == jit_pool_tag_kind_t::nspc),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "f32 nspc backward defers to the nhwc_pooling gather");
            return status::success;
        }

        jit_pool_conf_t jpp_;
    };

    jit_uni_pooling_bwd_t(const pd_t *apd);
    ~jit_uni_pooling_bwd_t() override;

    using data_t = typename prec_traits_t<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    void execute_backward_blk(const data_t *diff_dst, const char *indices,
            data_t *diff_src, const exec_ctx_t &ctx) const;
    void execute_backward_blk_3d(const data_t *diff_dst, const char *indices,
            data_t *diff_src, const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_pool_kernel_t<isa>> kernel_;
    // Native gather backward kernel (nspc/ncsp); null when the baked kernel runs.
    std::unique_ptr<jit_uni_pool_bwd_kernel_t<isa, d_type>> bwd_kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_POOLING_HPP
