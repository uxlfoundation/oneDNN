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

#include "cpu/x64/zen64/matmul/zen_lowp_grouped_matmul.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#if DNNL_X64_USE_ZEN
#include "cpu/x64/zen64/common/zen_format_tag.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "zendnnl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace matmul {

DEFINE_PD_CREATE(zen_lowp_grouped_matmul_t::pd_t)

using namespace data_type;
using namespace dnnl::impl::cpu::matmul;

status_t zen_lowp_grouped_matmul_t::pd_t::init(const engine_t *engine) {
#if !(DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY)
    MAYBE_UNUSED(engine);
    return status::unimplemented;
#else
    using smask_t = primitive_attr_t::skip_mask_t;

    // CPU engine only.
    VDISPATCH_MATMUL(
            engine->kind() == engine_kind::cpu, VERBOSE_BAD_ENGINE_KIND);

    // This implementation only supports AMD CPUs.
    VDISPATCH_MATMUL(::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");

    // The int8 dot product needs AVX-512 VNNI (vpdpbusd).
    VDISPATCH_MATMUL(mayiuse(avx512_core_vnni), VERBOSE_UNSUPPORTED_ISA);

    const memory_desc_wrapper src_d(src_md(0));
    const memory_desc_wrapper dst_d(dst_md(0));

    // Grouped src/dst selects the 2Dx3D pattern.
    VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    const dim_t total_M = dst_d.dims()[0];
    const dim_t K = src_d.dims()[1];
    const dim_t N = dst_d.dims()[1];
    const dim_t G = weights_md(0)->dims[0];

    // ZenDNN's group_matmul_direct takes M/N/K and the leading dims
    // (lda == K, ldb == N|K, ldc == N) as int. Larger shapes are
    // declined to the reference implementation.
    const dim_t int_max = std::numeric_limits<int>::max();
    VDISPATCH_MATMUL(
            total_M <= int_max && N <= int_max && K <= int_max && G <= int_max,
            VERBOSE_UNSUPPORTED_FEATURE,
            "dimension > INT_MAX is not supported");

    // ZenDNN addresses every expert with BLAS-style leading dimensions
    // (lda = K, ldb = N|K, ldc = N) over unit-stride inner dimensions. Read the
    // actual descriptor strides and require they are exactly equivalent to that
    // lda/ldb/ldc triple; padded rows or non-unit inner strides cannot be
    // expressed as a leading dimension, so such layouts are declined and left
    // to the reference grouped implementation (no padded-stride support here).
    const auto &src_str = src_d.strides();
    const auto &dst_str = dst_d.strides();
    // src [total_M, K]: K contiguous and row stride == lda == K.
    VDISPATCH_MATMUL(
            src_str[1] == 1 && src_str[0] == K, VERBOSE_UNSUPPORTED_TAG);
    // dst [total_M, N]: N contiguous and row stride == ldc == N.
    VDISPATCH_MATMUL(
            dst_str[1] == 1 && dst_str[0] == N, VERBOSE_UNSUPPORTED_TAG);

    // Symmetric static int8: s8 src, s8 or s4 (W4A8) wei, f32/bf16 dst, s32
    // accumulation. ZenDNN runs a u8 source on a per-tensor kernel that cannot
    // apply per-token scales, and its sym-quant kernel has no f16 destination,
    // so both are declined to the reference.
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const auto dst_dt = dst_md(0)->data_type;
    VDISPATCH_MATMUL(src_dt == s8 && utils::one_of(wei_dt, s8, s4)
                    && utils::one_of(dst_dt, f32, bf16),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            desc()->accum_data_type == s32, VERBOSE_UNSUPPORTED_DT_CFG);

    if (with_bias()) {
        VDISPATCH_MATMUL(utils::one_of(weights_md(1)->data_type, f32, bf16),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
        // execute() reads expert g at offset0 + g * N, so the bias must be a
        // dense [G, N] (a submemory view is fine). format_any is resolved to
        // that layout so a queried descriptor can be allocated.
        if (memory_desc_wrapper(bias_md_).format_any()) {
            VDISPATCH_MATMUL_SC(
                    memory_desc_init_by_tag(bias_md_, format_tag::ab),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
        }
        const memory_desc_wrapper bia_d(bias_md_);
        VDISPATCH_MATMUL(
                bia_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_BIAS_CFG);
        const auto &bia_str = bia_d.strides();
        VDISPATCH_MATMUL(bia_d.ndims() == 2 && bia_d.dims()[0] == G
                        && bia_d.dims()[1] == N && bia_str[1] == 1
                        && bia_str[0] == N,
                VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    // Only post-ops and scales (including the scale data type and groups) may
    // be non-default; zero-points are declined since the scheme is symmetric.
    VDISPATCH_MATMUL(attr()->has_default_values(smask_t::post_ops
                                     | smask_t::scales_groups
                                     | smask_t::scales_data_type,
                             dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Post-op subset ZenDNN can express:
    //   * eltwise with unit scale: relu (alpha == 0 is ReLU, any other alpha
    //     the leaky ReLU slope), gelu_tanh, gelu_erf, tanh, logistic, swish,
    //     mish (s8 weights only), and clip / clip_v2 (both
    //     min(max(x, alpha), beta) forward).
    //   * binary_mul with a full per-row [total_M, N] src1 (dense plain-ab or
    //     grouped) -> ZenDNN dense MATRIX_MUL, sliced per expert. Per-group
    //     [G, 1] and per-row-scalar [total_M, 1] shapes are declined to the
    //     reference grouped implementation.
    auto check_postops = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &e = po.entry_[i];
            using namespace alg_kind;
            if (e.is_eltwise()) {
                if (e.eltwise.scale != 1.f) return false;
                if (!utils::one_of(e.eltwise.alg, eltwise_relu,
                            eltwise_gelu_tanh, eltwise_gelu_erf, eltwise_tanh,
                            eltwise_logistic, eltwise_swish, eltwise_mish,
                            eltwise_clip, eltwise_clip_v2))
                    return false;
                // ZenDNN may run s4 weights on a W4A8 kernel that does not
                // support mish.
                if (e.eltwise.alg == eltwise_mish && wei_dt == s4) return false;
                // ZenDNN reads a NaN alpha as "use the default" (slope 0 for
                // leaky ReLU, 1 for swish) and refuses non-finite clip bounds.
                const bool is_clip = utils::one_of(
                        e.eltwise.alg, eltwise_clip, eltwise_clip_v2);
                if ((is_clip
                            || utils::one_of(
                                    e.eltwise.alg, eltwise_relu, eltwise_swish))
                        && !std::isfinite(e.eltwise.alpha))
                    return false;
                if (is_clip && !std::isfinite(e.eltwise.beta)) return false;
            } else if (e.is_binary()) {
                if (e.binary.alg != binary_mul) return false;
                const auto &s1 = e.binary.src1_desc;
                if (!utils::one_of(s1.data_type, f32, bf16)) return false;
                const memory_desc_wrapper s1_d(s1);
                if (s1_d.format_any()) return false;
                if (s1.ndims != 2) return false;
                // Only the full per-row [total_M, N] operand maps to Zen's
                // dense MATRIX_MUL; other shapes go to the reference.
                if (s1.dims[0] != total_M || s1.dims[1] != N) return false;
                // With total_M == G and N == 1, the grouped API reads a
                // [G, 1] src1 per expert, but execute() slices it per row.
                if (total_M == G && N == 1) return false;
                if (s1_d.is_grouped_desc()) {
                    const auto &s1_str = s1_d.strides();
                    if (s1_str[1] != 1 || s1_str[0] != N) return false;
                } else if (!s1_d.matches_one_of_tag(format_tag::ab)) {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_postops(), VERBOSE_UNSUPPORTED_POSTOP);
    // ZenDNN supports at most 8 post-op entries per call, counting the bias,
    // and does not check longer chains, so they are declined.
    constexpr int max_zen_post_ops = 8;
    VDISPATCH_MATMUL(
            attr()->post_ops_.len() + (with_bias() ? 1 : 0) <= max_zen_post_ops,
            VERBOSE_UNSUPPORTED_POSTOP);

    CHECK(init_int8_scales(engine));

    // Packed weights use the sym-quant layout, selected by a K group size
    // (sym_group_size > 0) matching the weight scales: K for per-column
    // scales, gK for K-blocked scales {gK, 1}. format_any advertises one
    // zen_packed slice per expert; a concrete zen_packed buffer must match the
    // source dtype, orientation, and group size. Plain abc/acb weights are
    // packed by the backend per call.
    const auto &wei_scales = attr()->scales_.get(DNNL_ARG_WEIGHTS);
    const int sym_group_size = wei_scales.has_default_groups()
            ? static_cast<int>(K)
            : static_cast<int>(wei_scales.get_group(-2));

    const bool wei_any = memory_desc_wrapper(weights_md(0)).format_any();
    const bool wei_packed = is_zen_packed(*weights_md(0));
    if (wei_any) {
        VDISPATCH_MATMUL_SC(init_zen_packed_md(weights_md_, src_dt, K, N,
                                    /*batch=*/G, /*weights_transposed=*/false,
                                    sym_group_size),
                VERBOSE_UNSUPPORTED_TAG);
    } else if (wei_packed) {
        const auto &zpd = memory_desc_wrapper(weights_md(0)).zen_packed_desc();
        VDISPATCH_MATMUL(zpd.gemm_src_dt == src_dt
                        && zpd.sym_group_size == sym_group_size
                        && !zpd.weights_transposed,
                VERBOSE_INCONSISTENT_MDS, "weights", "packed-prepack-metadata");
    } else if (wei_dt == s4) {
        // Plain s4 weights must be abc with an even K*N, so each expert's
        // slice starts on a byte boundary. acb would pack the 4-bit pairs
        // along K, which this path does not support.
        VDISPATCH_MATMUL(memory_desc_wrapper(weights_md(0))
                                 .matches_one_of_tag(format_tag::abc),
                VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_MATMUL((K * N) % 2 == 0, VERBOSE_UNSUPPORTED_TAG);
    } else {
        // execute() derives transB/ldb and the per-expert slice span from an
        // exact abc/acb layout (unit inner stride, K*N contiguous per expert).
        VDISPATCH_MATMUL(
                memory_desc_wrapper(weights_md(0))
                        .matches_one_of_tag(format_tag::abc, format_tag::acb),
                VERBOSE_UNSUPPORTED_TAG);
    }
    // execute() adds offset0 (a submemory view) to the weight base. An s4 view
    // must start on a byte boundary.
    VDISPATCH_MATMUL(
            IMPLICATION(wei_dt == s4,
                    memory_desc_wrapper(weights_md(0)).offset0() % 2 == 0),
            VERBOSE_UNSUPPORTED_TAG);

    return status::success;
#endif // DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY
}

status_t zen_lowp_grouped_matmul_t::pd_t::init_int8_scales(
        const engine_t *engine) {
    MAYBE_UNUSED(engine);
#if !(DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY)
    return status::unimplemented;
#else
    // Both scales are required: the source scale selects ZenDNN's sym-quant
    // kernel and the weight scale dequantizes the s32 accumulator into the
    // f32/bf16 destination. Destination scales are not supported for grouped
    // matmul.
    //
    // Accepted pairs:
    //   per-token src (mask src_qmask_M()) with per-expert per-column wei
    //     (mask wei_qmask_N()),
    //   per-token src with K-blocked wei, groups {gK, 1}
    //     (mask wei_qmask_K() | wei_qmask_N()), and
    //   K-blocked src, groups {1, gK} (mask src_qmask_M() | src_qmask_K()),
    //     with K-blocked wei of the same gK.
    // ZenDNN takes one K group size per call from the weight scale, so a
    // K-blocked source must use the weight group size.
    const auto &scales = attr()->scales_;

    // execute() reads precomputed static scales only, so only the default
    // quantization mode is accepted.
    VDISPATCH_MATMUL(
            scales.has_default_qmode(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(!scales.has_default_values(DNNL_ARG_SRC)
                    && !scales.has_default_values(DNNL_ARG_WEIGHTS),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(scales.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    // ZenDNN takes f32 or bf16 scales, and its sym-quant kernel requires the
    // same scale data type for the source and the weights.
    const auto src_scale_dt = scales.get_data_type(DNNL_ARG_SRC);
    VDISPATCH_MATMUL(utils::one_of(src_scale_dt, f32, bf16)
                    && scales.get_data_type(DNNL_ARG_WEIGHTS) == src_scale_dt,
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    const bool src_blocked = !scales.get(DNNL_ARG_SRC).has_default_groups();
    const bool wei_blocked = !scales.get(DNNL_ARG_WEIGHTS).has_default_groups();
    const int src_mask = scales.get_mask(DNNL_ARG_SRC);
    const int wei_mask = scales.get_mask(DNNL_ARG_WEIGHTS);
    VDISPATCH_MATMUL(src_mask
                            == (src_blocked ? src_qmask_M() | src_qmask_K()
                                            : src_qmask_M())
                    && wei_mask
                            == (wei_blocked ? wei_qmask_K() | wei_qmask_N()
                                            : wei_qmask_N()),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // src groups are {gM, gK}, wei groups are {gK, gN}; only blocking along K
    // is representable. A per-column weight scale is one group spanning K,
    // and a blocked source must use the weight gK (see above).
    const dim_t wei_gK
            = wei_blocked ? scales.get_group(DNNL_ARG_WEIGHTS, -2) : K();
    VDISPATCH_MATMUL(IMPLICATION(wei_blocked,
                             scales.get_group(DNNL_ARG_WEIGHTS, -1) == 1),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(
            IMPLICATION(src_blocked,
                    wei_blocked && scales.get_group(DNNL_ARG_SRC, -2) == 1
                            && scales.get_group(DNNL_ARG_SRC, -1) == wei_gK),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    // ZenDNN requires the weight K group, including a per-column group
    // spanning K, to be a multiple of 4 for both s8 and s4 weights.
    VDISPATCH_MATMUL(wei_gK % 4 == 0 && K() % wei_gK == 0,
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    return status::success;
#endif // DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY
}

status_t zen_lowp_grouped_matmul_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
#if DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY

    const auto &po = pd()->attr()->post_ops_;
    postop_template_.clear();
    postop_template_.reserve(po.len());
    binary_src1_.clear();

    using pot = zendnnl::ops::post_op_type_t;
    using zd = zendnnl::common::data_type_t;

    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        zen_lowp_gmm::matmul_post_op lpo {};
        if (e.is_binary()) {
            lpo.po_type = pot::binary_mul;
            switch (e.binary.src1_desc.data_type) {
                case f32: lpo.dtype = zd::f32; break;
                case bf16: lpo.dtype = zd::bf16; break;
                default: return status::runtime_error;
            }
            const memory_desc_wrapper s1_d(e.binary.src1_desc);
            binary_src1_.push_back({static_cast<int>(postop_template_.size()),
                    i, types::data_type_size(e.binary.src1_desc.data_type),
                    s1_d.is_grouped_desc(), s1_d.offset0()});
        } else {
            switch (e.eltwise.alg) {
                case alg_kind::eltwise_relu:
                    lpo.po_type = e.eltwise.alpha == 0.f ? pot::relu
                                                         : pot::leaky_relu;
                    break;
                case alg_kind::eltwise_clip:
                case alg_kind::eltwise_clip_v2: lpo.po_type = pot::clip; break;
                case alg_kind::eltwise_gelu_tanh:
                    lpo.po_type = pot::gelu_tanh;
                    break;
                case alg_kind::eltwise_gelu_erf:
                    lpo.po_type = pot::gelu_erf;
                    break;
                case alg_kind::eltwise_tanh: lpo.po_type = pot::tanh; break;
                case alg_kind::eltwise_logistic:
                    lpo.po_type = pot::sigmoid;
                    break;
                case alg_kind::eltwise_swish: lpo.po_type = pot::swish; break;
                case alg_kind::eltwise_mish: lpo.po_type = pot::mish; break;
                default: return status::runtime_error;
            }
            lpo.alpha = e.eltwise.alpha;
            lpo.beta = e.eltwise.beta;
        }
        postop_template_.push_back(lpo);
    }

    // ---- Static int8 scale metadata ----
    // Data type, element size, and K-group count are expert-independent, so
    // resolve them once here. The buffers live in the execution context and
    // are sliced per expert at execute(). `gk_idx` is where this argument's
    // K group size sits in its group list: last for the source ({gM, gK}),
    // second to last for the weights ({gK, gN}).
    const auto &scales = pd()->attr()->scales_;
    auto setup_scale
            = [&](int arg, int gk_idx, zen_lowp_grouped_scale_info_t &s) {
        if (scales.has_default_values(arg)) return;
        const auto sc_dt = scales.get_data_type(arg);
        s.present = true;
        s.dt = to_zen_dt(sc_dt);
        s.elem_sz = types::data_type_size(sc_dt);
        const dim_t gK = scales.get(arg).has_default_groups()
                ? 1
                : scales.get_group(arg, gk_idx);
        s.ngroups_k = gK > 1 ? pd()->K() / gK : 1;
    };
    setup_scale(DNNL_ARG_SRC, -1, src_scale_);
    setup_scale(DNNL_ARG_WEIGHTS, -2, wei_scale_);
#endif
    return status::success;
}

status_t zen_lowp_grouped_matmul_t::execute(const exec_ctx_t &ctx) const {
#if !(DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY)
    MAYBE_UNUSED(ctx);
    return status::unimplemented;
#else
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t group_count = src_grouped.group_count;
    const dim_t K = wei_d.dims()[wei_d.ndims() - 2];
    const dim_t N = wei_d.dims()[wei_d.ndims() - 1];
    const dim_t total_M = src_d.dims()[0];

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto dst_dt = pd()->dst_md()->data_type;

    const bool with_bias = pd()->with_bias();
    const auto bia_dt
            = with_bias ? pd()->weights_md(1)->data_type : data_type::undef;

    const char *src_data = CTX_IN_MEM(const char *, DNNL_ARG_SRC, 0);
    const int32_t *src_offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    // CTX_IN_MEM is the allocation base. A weight or bias submemory keeps its
    // first element at offset0.
    const char *wei_data = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS)
            + types::elements_to_bytes(
                    wei_dt, static_cast<size_t>(wei_d.offset0()));
    char *dst_data = CTX_OUT_MEM(char *, DNNL_ARG_DST, 0);
    const int32_t *dst_offsets = CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);
    const char *bias_data = nullptr;
    if (with_bias) {
        const memory_desc_wrapper bia_d(pd()->weights_md(1));
        bias_data = CTX_IN_MEM(const char *, DNNL_ARG_BIAS)
                + static_cast<size_t>(bia_d.offset0())
                        * types::data_type_size(bia_dt);
    }

    // Static int8 quantization scales: source scales over the concatenated
    // [total_M, gk] rows and weight scales laid out as a dense [G, gk, N]
    // tensor. Both are sliced per expert below.
    const char *src_scales
            = CTX_IN_MEM(const char *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const char *wei_scales
            = CTX_IN_MEM(const char *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);

    // Attach one expert's slice of a shared quantization buffer, starting
    // `offset_elems` elements into it.
    auto set_quant
            = [](zen_lowp_gmm::matmul_quantization_params_t::matmul_quant_t &q,
                      const zen_lowp_grouped_scale_info_t &s, const char *base,
                      dim_t offset_elems, const std::vector<int64_t> &dims) {
        if (!s.present) return;
        q.buff = base + static_cast<size_t>(offset_elems) * s.elem_sz;
        q.dt = s.dt;
        q.dims = dims;
    };

    const size_t src_dsz = types::data_type_size(src_dt);
    const size_t dst_dsz = types::data_type_size(dst_dt);
    const size_t bia_dsz = with_bias ? types::data_type_size(bia_dt) : 0;

    // Weight layout. Pre-packed (Zen packed) weights are consumed as AOCL
    // blocked (sym-quant): untransposed, ldb = N, one fixed-size packed slot
    // per expert. Plain abc -> B is [K, N] row-major (ldb = N); acb -> [N, K]
    // transposed view (ldb = K), one K*N element slice per expert.
    const bool wei_packed = is_zen_packed(*pd()->weights_md(0));
    bool transB;
    int ldb;
    size_t wei_slice_bytes;
    if (wei_packed) {
        transB = false;
        ldb = static_cast<int>(N);
        wei_slice_bytes = pd()->weights_md(0)
                                  ->format_desc.zen_packed_desc.per_slice_size;
    } else {
        // With N == 1, abc and acb have the same bytes, but an acb descriptor
        // keeps an N stride of K, so treat it as untransposed.
        const dim_t wei_stride_n = wei_d.blocking_desc().strides[2];
        transB = N > 1 && wei_stride_n != 1;
        ldb = transB ? static_cast<int>(K) : static_cast<int>(N);
        // s4 packs two elements per byte, so the slice size is in bytes, not
        // elements * data_type_size().
        wei_slice_bytes = types::elements_to_bytes(
                wei_dt, static_cast<size_t>(K) * static_cast<size_t>(N));
    }

    // Build per-group (per-expert) descriptors for ZenDNN,
    // skipping empty groups (M == 0).
    std::vector<char> layouts;
    std::vector<bool> transAs, transBs, wconst;
    std::vector<int> Ms, Ns, Ks, ldas, ldbs, ldcs;
    std::vector<float> alphas, betas;
    std::vector<const void *> srcs, weis, biases;
    std::vector<void *> dsts;
    std::vector<zen_lowp_gmm::matmul_params> params;

    using zd = zendnnl::common::data_type_t;
    const zd zsrc = to_zen_dt(src_dt);
    const zd zwei = to_zen_dt(wei_dt);
    const zd zdst = to_zen_dt(dst_dt);
    const zd zbia = with_bias ? to_zen_dt(bia_dt) : zd::none;

    // Per binary_mul post-op: base pointer of the full [total_M, N] src1
    // operand (dense or grouped-concatenated, row-major). Each expert slices
    // its [M_g, N] block at the dst row offset.
    std::vector<const char *> bin_src1_base(binary_src1_.size());
    std::vector<const int32_t *> bin_src1_offsets(binary_src1_.size(), nullptr);
    for (size_t k = 0; k < binary_src1_.size(); k++) {
        const int po_arg
                = DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_src1_[k].po_idx)
                | DNNL_ARG_SRC_1;
        // CTX_IN_MEM is the allocation base. A dense ab (or grouped) submemory
        // keeps its selected rows at offset0, so add that before per-expert
        // slicing or the slice would start at the parent's earlier rows.
        bin_src1_base[k] = CTX_IN_MEM(const char *, po_arg)
                + static_cast<size_t>(binary_src1_[k].offset0)
                        * binary_src1_[k].elem_sz;
        if (binary_src1_[k].is_grouped)
            bin_src1_offsets[k] = CTX_IN_MEM(const int32_t *, po_arg, 1);
    }

    for (dim_t g = 0; g < group_count; ++g) {
        const dim_t start = (g == 0) ? 0 : src_offsets[g - 1];
        const dim_t end = src_offsets[g];
        const dim_t dst_start = (g == 0) ? 0 : dst_offsets[g - 1];
        const dim_t dst_end = dst_offsets[g];

        if (start < 0 || end > total_M || end < start || dst_start < 0
                || dst_end > total_M || dst_end < dst_start
                || (dst_end - dst_start) != (end - start))
            return status::invalid_arguments;

        for (size_t k = 0; k < binary_src1_.size(); k++) {
            if (!bin_src1_offsets[k]) continue;
            const dim_t bin_start = (g == 0) ? 0 : bin_src1_offsets[k][g - 1];
            const dim_t bin_end = bin_src1_offsets[k][g];
            if (bin_start != dst_start || bin_end != dst_end)
                return status::invalid_arguments;
        }

        const dim_t M = end - start;
        if (M == 0) continue; // no rows routed to this expert

        layouts.push_back('r');
        transAs.push_back(false);
        transBs.push_back(transB);
        Ms.push_back(static_cast<int>(M));
        Ns.push_back(static_cast<int>(N));
        Ks.push_back(static_cast<int>(K));
        alphas.push_back(1.0f);
        betas.push_back(0.0f);
        ldas.push_back(static_cast<int>(K));
        ldbs.push_back(ldb);
        ldcs.push_back(static_cast<int>(N));
        // Pre-packed weights are produced once by zen_reorder_t and are stable,
        // so mark them const to enable ZenDNN's weight cache (packed once,
        // reused across calls). Plain weights are not guaranteed immutable
        // across executes, so they must not be cached by pointer.
        wconst.push_back(wei_packed);

        srcs.push_back(src_data + static_cast<size_t>(start) * K * src_dsz);
        weis.push_back(wei_data + static_cast<size_t>(g) * wei_slice_bytes);
        dsts.push_back(dst_data + static_cast<size_t>(dst_start) * N * dst_dsz);
        biases.push_back(with_bias
                        ? static_cast<const void *>(bias_data
                                  + static_cast<size_t>(g) * N * bia_dsz)
                        : nullptr);

        zen_lowp_gmm::matmul_params p {};
        p.dtypes.src = zsrc;
        p.dtypes.wei = zwei;
        p.dtypes.dst = zdst;
        p.dtypes.bias = zbia;
        // s4 weights select ZenDNN's W4A8 path with an s8 compute type; s8
        // weights leave it unset. Both accumulate in s32, and the source is
        // already quantized, so no dynamic quantization is requested.
        p.dtypes.compute = wei_dt == s4 ? zd::s8 : zd::none;
        if (wei_packed) {
            // The buffer is in the AOCL blocked layout written by
            // zen_reorder_t for this sym_group_size; the algo must match it.
            p.mem_format_b = 'r';
            p.lowoha_algo = zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;
        }
        // Source scales are laid out [total_M, gk_src], so this expert's rows
        // [start, end) begin at `start * gk_src`. A single-row expert has a
        // {1, 1} source scale; ZenDNN still uses the sym-quant kernel because
        // the weight scale is 2D.
        const dim_t gk_src = src_scale_.ngroups_k;
        set_quant(p.quant_params.src_scale, src_scale_, src_scales,
                start * gk_src, {M, gk_src});
        // Weight scales are laid out [G, gk_wei, N], so this expert owns the
        // [gk_wei, N] block at `g * gk_wei * N`. Both reduce to the ungrouped
        // per-column case at gk_wei == 1.
        const dim_t gk_wei = wei_scale_.ngroups_k;
        set_quant(p.quant_params.wei_scale, wei_scale_, wei_scales,
                g * gk_wei * N, {gk_wei, N});
        p.postop_ = postop_template_;
        // Patch this expert's dense [M_g, N] src1 slice into each binary_mul
        // entry (ZenDNN applies it as a per-element MATRIX_MUL).
        for (size_t k = 0; k < binary_src1_.size(); k++) {
            auto &lpo = p.postop_[binary_src1_[k].chain_idx];
            lpo.dims = {M, N};
            lpo.leading_dim = static_cast<int>(N);
            lpo.buff = const_cast<char *>(bin_src1_base[k]
                    + static_cast<size_t>(dst_start) * static_cast<size_t>(N)
                            * binary_src1_[k].elem_sz);
        }
        params.push_back(p);
    }

    // Nothing to compute (all experts empty).
    if (params.empty()) return status::success;

    const auto st = zen_lowp_gmm::group_matmul_direct(layouts, transAs, transBs,
            Ms, Ns, Ks, alphas, srcs, ldas, weis, ldbs, biases, betas, dsts,
            ldcs, wconst, params);

    return to_dnnl_status(st);
#endif // DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY
}

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
