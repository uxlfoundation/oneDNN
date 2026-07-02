/*******************************************************************************
* Copyright 2026 openKylin community
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

#ifndef CPU_RV64_JIT_UNI_PRELU_HPP
#define CPU_RV64_JIT_UNI_PRELU_HPP

#include <memory>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_prelu_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// How the weights tensor maps onto src; selected once at init and consumed by
// execute() to pick a kernel + iteration scheme. The two JIT kernels are a flat
// lockstep loop (weights walk 1:1 with src) and a scalar-broadcast loop (one
// f32 weight); the per_oc layouts are decomposed in C++ into runs that one of
// those two kernels handles. The more exotic broadcast strategies (shared_axes,
// per_mb, ...) are left to ref_prelu.
enum class prelu_bcast_t {
    unsupported,
    full, // weights shape == src shape: flat lockstep over the whole tensor
    scalar, // single weight broadcast over the whole tensor
    per_oc_nhwc, // channel innermost (stride[1]==1): lockstep run of length C
    per_oc_nchw, // channel-major plain: scalar weight per (n, c) spatial plane
    per_oc_blocked, // nChw{8,16}c: lockstep run of length blk per channel block
};

// Standalone PReLU forward: dst = max(0, src) + weights * min(0, src). A thin
// VLA JIT wrapper mirroring rv64/jit_uni_eltwise.cpp. The kernel computes in
// f32; f16 (zvfh) and bf16 (zvfbfwma) are converted at the load/store boundary
// with the native widening/narrowing FP-convert instructions (vfwcvt/vfncvt
// for f16, vfwcvtbf16/vfncvtbf16 for bf16, the latter from Zvfbfmin which
// Zvfbfwma implies). Backward (the diff_weights reduction) is left to ref.
struct jit_uni_prelu_fwd_kernel_t;

struct jit_uni_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:rvv", jit_uni_prelu_fwd_t);

        status_t init(engine_t *engine) {
            UNUSED(engine);
            using namespace dnnl::impl::data_type;

            const data_type_t dt = src_md(0)->data_type;
            const data_type_t wdt = weights_md(0)->data_type;

            VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
            // Single JIT impl; the convert ISA is chosen internally from the
            // data type. The base vector extension is always required; f16/bf16
            // additionally need their convert ISA (zvfh / zvfbfwma), gated by
            // has_data_type_support below so we never emit an unavailable
            // convert. src/dst share a dtype; weights may differ.
            VDISPATCH_PRELU(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_PRELU(
                    utils::one_of(dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(
                    utils::one_of(wdt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(platform::has_data_type_support(dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(platform::has_data_type_support(wdt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(dst_md(0)->data_type == dt, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_PRELU(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));
            VDISPATCH_PRELU(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");

            bcast_ = classify_bcast(src_d, weights_md(0));
            VDISPATCH_PRELU(bcast_ != prelu_bcast_t::unsupported,
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }

        prelu_bcast_t bcast_ = prelu_bcast_t::unsupported;

    private:
        // Mirrors x64 prelu::get_bcast_type for the layout split, but keyed off
        // the common broadcasting-strategy helper that ref also uses.
        static prelu_bcast_t classify_bcast(const memory_desc_wrapper &src_d,
                const memory_desc_t *weights_md) {
            const memory_desc_wrapper weights_d(weights_md);
            const auto strategy
                    = get_rhs_arg_broadcasting_strategy(*weights_md, src_d);

            // Flat walks need a dense src (no internal gaps); per_oc walks need
            // a known plain layout to compute channel offsets.
            if (strategy == broadcasting_strategy_t::no_broadcast) {
                // Same dims + same physical ordering (ignoring dtype) guarantees
                // the flat lockstep pairing; weights may differ in dtype.
                auto same_layout = [](const memory_desc_wrapper &a,
                                           const memory_desc_wrapper &b) {
                    if (a.ndims() != b.ndims()
                            || a.format_kind() != b.format_kind())
                        return false;
                    if (!utils::array_cmp(a.dims(), b.dims(), a.ndims()))
                        return false;
                    if (!a.is_blocking_desc()) return true;
                    const auto &ab = a.blocking_desc();
                    const auto &bb = b.blocking_desc();
                    return ab.inner_nblks == bb.inner_nblks
                            && utils::array_cmp(
                                    ab.strides, bb.strides, a.ndims())
                            && utils::array_cmp(ab.inner_blks, bb.inner_blks,
                                    ab.inner_nblks)
                            && utils::array_cmp(ab.inner_idxs, bb.inner_idxs,
                                    ab.inner_nblks);
                };
                if (same_layout(src_d, weights_d) && src_d.is_dense(true)
                        && weights_d.is_dense(true))
                    return prelu_bcast_t::full;
                return prelu_bcast_t::unsupported;
            }
            if (strategy == broadcasting_strategy_t::scalar) {
                if (src_d.is_dense(true)) return prelu_bcast_t::scalar;
                return prelu_bcast_t::unsupported;
            }
            // The helper splits channel-wise weights into per_oc (channel
            // innermost, e.g. NHWC) and per_oc_spatial (channel-major plain,
            // e.g. NCHW); the stride check below maps either onto the right run
            // scheme, so accept both.
            if (utils::one_of(strategy, broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial)) {
                if (src_d.ndims() < 2) return prelu_bcast_t::unsupported;
                // Channel-blocked formats (nChw{8,16}c): exactly one inner block
                // on the channel dim (idx 1). Require src and weights blocked
                // alike; padding is expected and handled via padded_dims.
                auto chan_block = [](const memory_desc_wrapper &md) -> dim_t {
                    if (!md.is_blocking_desc()) return 0;
                    const auto &b = md.blocking_desc();
                    if (b.inner_nblks != 1 || b.inner_idxs[0] != 1) return 0;
                    return b.inner_blks[0];
                };
                const dim_t sblk = chan_block(src_d),
                            wblk = chan_block(weights_d);
                if (sblk > 0 && sblk == wblk && src_d.is_dense(true))
                    return prelu_bcast_t::per_oc_blocked;
                // Plain layouts: the per_oc offset math (i*C runs / pl*SP planes)
                // assumes a gapless, unpadded plain layout.
                const bool no_padding = utils::array_cmp(
                        src_d.dims(), src_d.padded_dims(), src_d.ndims());
                if (!src_d.is_plain() || !src_d.is_dense() || !no_padding)
                    return prelu_bcast_t::unsupported;
                const auto &strides = src_d.blocking_desc().strides;
                if (strides[1] == 1) return prelu_bcast_t::per_oc_nhwc;
                bool channel_major = strides[0] >= strides[1];
                for (int d = 2; d < src_d.ndims(); ++d)
                    channel_major = channel_major && strides[1] >= strides[d];
                if (channel_major) return prelu_bcast_t::per_oc_nchw;
                return prelu_bcast_t::unsupported;
            }
            return prelu_bcast_t::unsupported;
        }
    };

    jit_uni_prelu_fwd_t(const pd_t *apd);
    ~jit_uni_prelu_fwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_prelu_fwd_kernel_t> kernel_;
    std::unique_ptr<jit_uni_prelu_fwd_kernel_t> scalar_kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_PRELU_HPP
