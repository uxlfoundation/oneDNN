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

#include "cpu/x64/zen64/matmul/zen_grouped_matmul.hpp"

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

using namespace data_type;
using namespace dnnl::impl::cpu::matmul;

#if DNNL_X64_USE_ZEN
namespace zen_gmm = zendnnl::lowoha::matmul;
#endif

status_t zen_grouped_matmul_t::pd_t::init(const engine_t *engine) {
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

    // This implementation requires AVX-512-CORE.
    VDISPATCH_MATMUL(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    const memory_desc_wrapper src_d(src_md(0));
    const memory_desc_wrapper dst_d(dst_md(0));

    // Grouped src/dst selects the 2Dx3D pattern.
    VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const dim_t total_M = dst_d.dims()[0];
    const dim_t K = src_d.dims()[1];
    const dim_t N = dst_d.dims()[1];

    // ZenDNN's group_matmul_direct takes M/N/K and the leading dims
    // (lda == K, ldb == N|K, ldc == N) as int. larger shapes are
    // declined to the reference implementation.
    const dim_t int_max = std::numeric_limits<int>::max();
    VDISPATCH_MATMUL(total_M <= int_max && N <= int_max && K <= int_max,
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
    // Weights must be dense, abc or acb format is already enforced by the
    // common grouped gate (grouped_matmul_desc_init).

    // Supported floating-point configurations (aligned with zen_matmul_t):
    //  1. uniform f32  (f32 src/wei/dst)
    //  2. uniform bf16 (bf16 src/wei/dst)
    //  3. bf16 mixed   (bf16 src/wei, f32 dst)
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const auto dst_dt = dst_md(0)->data_type;
    const bool all_f32 = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool all_bf16 = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
    const bool bf16_mixed
            = utils::everyone_is(bf16, src_dt, wei_dt) && dst_dt == f32;
    VDISPATCH_MATMUL(utils::one_of(true, all_f32, all_bf16, bf16_mixed),
            VERBOSE_UNSUPPORTED_DT_CFG);

    if (with_bias()) {
        const auto bia_dt = weights_md(1)->data_type;
        const bool bia_dt_ok = IMPLICATION(all_f32, bia_dt == f32)
                && IMPLICATION(all_bf16 || bf16_mixed,
                        utils::one_of(bia_dt, bf16, f32));
        VDISPATCH_MATMUL(bia_dt_ok, VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    // Only post-ops may be non-default; this also rejects scales / zero-points
    // (left to the reference grouped implementation).
    VDISPATCH_MATMUL(attr()->has_default_values(smask_t::post_ops, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Post-op subset ZenDNN can express:
    //   * eltwise with unit scale: relu (alpha==0), gelu_tanh, gelu_erf, tanh,
    //     logistic, swish.
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
                            eltwise_logistic, eltwise_swish))
                    return false;
                // Zen maps eltwise_relu to plain ReLU (slope 0).
                if (e.eltwise.alg == eltwise_relu && e.eltwise.alpha != 0.f)
                    return false;
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

    return status::success;
#endif // DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY
}

status_t zen_grouped_matmul_t::init(engine_t *engine) {
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
        zen_gmm::matmul_post_op lpo {};
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
                    s1_d.is_grouped_desc()});
        } else {
            switch (e.eltwise.alg) {
                case alg_kind::eltwise_relu: lpo.po_type = pot::relu; break;
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
                default: return status::runtime_error;
            }
            lpo.alpha = e.eltwise.alpha;
            lpo.beta = e.eltwise.beta;
        }
        postop_template_.push_back(lpo);
    }
#endif
    return status::success;
}

status_t zen_grouped_matmul_t::execute(const exec_ctx_t &ctx) const {
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
    const char *wei_data = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    char *dst_data = CTX_OUT_MEM(char *, DNNL_ARG_DST, 0);
    const int32_t *dst_offsets = CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);
    const char *bias_data
            = with_bias ? CTX_IN_MEM(const char *, DNNL_ARG_BIAS) : nullptr;

    const size_t src_dsz = types::data_type_size(src_dt);
    const size_t wei_dsz = types::data_type_size(wei_dt);
    const size_t dst_dsz = types::data_type_size(dst_dt);
    const size_t bia_dsz = with_bias ? types::data_type_size(bia_dt) : 0;

    // Weight layout: abc -> B is [K, N] row-major (no transpose, ldb = N);
    // acb -> B stored as [N, K] (transposed view, ldb = K).
    const dim_t wei_stride_n = wei_d.blocking_desc().strides[2];
    const bool transB = (wei_stride_n != 1);
    const int ldb = transB ? static_cast<int>(K) : static_cast<int>(N);

    // Build per-group (per-expert) descriptors for ZenDNN,
    // skipping empty groups (M == 0).
    std::vector<char> layouts;
    std::vector<bool> transAs, transBs, wconst;
    std::vector<int> Ms, Ns, Ks, ldas, ldbs, ldcs;
    std::vector<float> alphas, betas;
    std::vector<const void *> srcs, weis, biases;
    std::vector<void *> dsts;
    std::vector<zen_gmm::matmul_params> params;

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
        bin_src1_base[k] = CTX_IN_MEM(const char *, po_arg);
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
        // is_weights_const=false: oneDNN's matmul contract does not guarantee
        // the weights buffer is immutable/stable across executes, so we must
        // not let ZenDNN cache the reordered weights keyed by pointer.
        wconst.push_back(false);

        srcs.push_back(src_data + static_cast<size_t>(start) * K * src_dsz);
        weis.push_back(wei_data + static_cast<size_t>(g) * K * N * wei_dsz);
        dsts.push_back(dst_data + static_cast<size_t>(dst_start) * N * dst_dsz);
        biases.push_back(with_bias
                        ? static_cast<const void *>(bias_data
                                  + static_cast<size_t>(g) * N * bia_dsz)
                        : nullptr);

        zen_gmm::matmul_params p {};
        p.dtypes.src = zsrc;
        p.dtypes.wei = zwei;
        p.dtypes.dst = zdst;
        p.dtypes.bias = zbia;
        p.dtypes.compute = zd::f32;
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

    const auto st = zen_gmm::group_matmul_direct(layouts, transAs, transBs, Ms,
            Ns, Ks, alphas, srcs, ldas, weis, ldbs, biases, betas, dsts, ldcs,
            wconst, params);

    return to_dnnl_status(st);
#endif // DNNL_X64_USE_ZEN && DNNL_EXPERIMENTAL_GROUPED_MEMORY
}

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
