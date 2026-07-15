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

#include "cpu/x64/zen64/matmul/zen_matmul.hpp"

#include <assert.h>
#include <limits>

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/matmul/gemm_based_common.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/zen64/common/zen_format_tag.hpp"

#if DNNL_X64_USE_ZEN
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
using namespace zen_matmul;
#endif

status_t zen_matmul_t::pd_t::init(engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;

#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // CPU engine only.
    VDISPATCH_MATMUL(
            engine->kind() == engine_kind::cpu, VERBOSE_BAD_ENGINE_KIND);

    // Dense format only (no sparse).
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // AMD-only vendor gate via xbyak (portable across GCC/Clang/MSVC).
    VDISPATCH_MATMUL(::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");

    // Zen matmul requires AVX-512 core support regardless of data type.
    // Note: no separate avx512_core_bf16 check is needed here. On AMD CPUs,
    // AVX-512 first appeared with Zen 4, which shipped avx512_core and
    // avx512_core_bf16 together (and every later Zen generation does too).
    // Since avx512_core_bf16 is a superset of avx512_core, any AMD CPU that
    // satisfies avx512_core also supports avx512_core_bf16, so gating on
    // avx512_core alone is sufficient for the bf16 paths as well.
    VDISPATCH_MATMUL(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    // ---- Memory descriptor data types ----
    const auto src_dt = src_md(0)->data_type;
    const auto wei_dt = weights_md(0)->data_type;
    const auto dst_dt = dst_md(0)->data_type;

    // 2D only -- batched (3D+) matmul not yet supported.
    VDISPATCH_MATMUL(ndims() == 2, VERBOSE_BAD_NDIMS, "dst", ndims());

    // No zero-dim tensors.
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    // ---- Datatype validation (aligned with Zen support) ----
    // Supported configurations:
    //  1. Uniform f32:  f32 src, f32 wei, f32 dst
    //  2. Uniform bf16: bf16 src, bf16 wei, bf16 dst
    //  3. bf16 mixed:   bf16 src, bf16 wei, f32 dst
    // Explicitly unsupported: f32 src with bf16 dst.
    const bool all_f32 = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool all_bf16 = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
    const bool bf16_mixed
            = utils::everyone_is(bf16, src_dt, wei_dt) && dst_dt == f32;
    VDISPATCH_MATMUL(utils::one_of(true, all_f32, all_bf16, bf16_mixed),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            desc()->accum_data_type == f32, VERBOSE_UNSUPPORTED_DT_CFG);

    // ---- Bias validation ----
    // Zen supports bias with matching/compatible dtypes;
    // bias must follow 1xN broadcast pattern.
    auto check_bias = [&]() -> bool {
        if (!with_bias()) return true;
        const auto bia_dt = weights_md(1)->data_type;
        const bool bia_dt_ok = IMPLICATION(all_f32, bia_dt == f32)
                && IMPLICATION(all_bf16 || bf16_mixed,
                        utils::one_of(bia_dt, bf16, f32));
        return bia_dt_ok && is_bias_1xN();
    };
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    // ---- Attribute validation ----
    // For f32/bf16: post-ops (eltwise + binary + sum) are supported;
    // fpmath_mode, scales and zero-points must be default.
    VDISPATCH_MATMUL(attr()->has_default_values(
                             smask_t::post_ops | smask_t::sum_dt, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Sum-consistency check (catches sum.dt != dst_dt precision bugs).
    VDISPATCH_MATMUL(
            attr()->post_ops_.check_sum_consistency(dst_dt, /*is_int8=*/false),
            VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Post-ops validation ----
    // Zen supports: sum, eltwise (relu, gelu_tanh, gelu_erf, tanh,
    // sigmoid/logistic, swish), binary (add, mul).
    auto check_postops = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (entry.is_sum(/*require_scale_one=*/false,
                        /*require_zp_zero=*/true)) {
                // Sum maps to plain beta accumulation and must be the very
                // first post-op; at any later position Zen has already
                // consumed the destination, so it cannot be honored.
                if (i != 0) return false;
                // Sum maps to plain beta accumulation, which reads the
                // destination in its native dtype. A sum.dt that differs
                // from dst_dt asks the destination bytes to be
                // reinterpreted as that dtype before accumulation (e.g.
                // f32 dst read as s32); Zen cannot honor that, so only
                // accept a default sum.dt or one matching dst_dt.
                if (!utils::one_of(entry.sum.dt, data_type::undef, dst_dt))
                    return false;
                continue;
            } else if (entry.is_eltwise()) {
                if (entry.eltwise.scale != 1.f) return false;
                using namespace alg_kind;
                if (!utils::one_of(entry.eltwise.alg, eltwise_relu,
                            eltwise_gelu_tanh, eltwise_gelu_erf, eltwise_tanh,
                            eltwise_logistic, eltwise_swish))
                    return false;
                // Zen maps eltwise_relu to a plain ReLU (slope 0); it cannot
                // honor a leaky-relu negative slope, so reject alpha != 0.
                if (entry.eltwise.alg == eltwise_relu
                        && entry.eltwise.alpha != 0.f)
                    return false;
            } else if (entry.is_binary()) {
                using namespace alg_kind;
                if (!utils::one_of(entry.binary.alg, binary_add, binary_mul))
                    return false;
                const auto src1_dt = entry.binary.src1_desc.data_type;
                if (!utils::one_of(src1_dt, f32, bf16)) return false;
            } else {
                // Unsupported post-op kind.
                return false;
            }
        }
        return true;
    };
    VDISPATCH_MATMUL(check_postops(), VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Scales / zero-points validation ----
    VDISPATCH_MATMUL(attr()->scales_.has_default_values(),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // Zen matmul_direct uses int for M/N/K; reject runtime dims/strides
    // before set_default_formats() to avoid undefined behavior.
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Zen f32/bf16 matmul: prepack path. When the framework leaves the
    // weights layout open (format_any) we advertise the dedicated opaque
    // `format_kind::zen_packed` weights format. The bytes are produced by
    // zen_reorder_t (the Zen backend packer) and consumed directly by the
    // backend (mem_format_b='r'), so no oneDNN blocked layout is involved.
    const bool wei_format_any = memory_desc_wrapper(weights_md(0)).format_any();

    // The caller may pass back a weights descriptor that is already in the
    // opaque zen_packed format (e.g. one obtained from a previous query),
    // in which case it must be accepted as-is: no re-packing is needed and
    // the plain-weights paths below (GEMM-format check, matmul_helper_t::ldb)
    // must be skipped since the descriptor has no blocking_desc.
    const bool wei_already_packed = zen::is_zen_packed(*weights_md(0));

    // Resolve format_any memory descriptors to concrete (dense) formats first.
    // For the prepack path this gives the weights a plain blocked layout
    // (dims/padded_dims set), which we then convert in-place to the opaque
    // packed format.
    // An already-packed (opaque) descriptor is not format_any, so
    // set_default_formats() leaves it untouched.
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    bool wei_zen_packed = wei_already_packed;
    if (wei_format_any && (wei_dt == bf16 || wei_dt == f32)) {
        VDISPATCH_MATMUL_SC(
                zen::init_zen_packed_md(weights_md_, src_dt, K(), N()),
                VERBOSE_UNSUPPORTED_TAG);
        wei_zen_packed = true;
    }

    // The Zen prepacked path stores weights in the opaque zen_packed
    // format that gemm_based's plain-weights compatibility check rejects; the
    // backend consumes the packed buffer directly (mem_format_b='r'), so the
    // gemm-format check only applies to the plain weights path.
    VDISPATCH_MATMUL(
            wei_zen_packed || gemm_based::check_gemm_compatible_formats(*this),
            VERBOSE_INCOMPATIBLE_GEMM_FMT);

    // Destination must be plain row-major ab layout.
    const memory_desc_wrapper dst_d(dst_md(0));
    VDISPATCH_MATMUL(
            dst_d.matches_one_of_tag(format_tag::ab), VERBOSE_UNSUPPORTED_TAG);

    VDISPATCH_MATMUL(!::dnnl::impl::cpu::x64::binary_injector::
                             any_binary_postop_rhs_with_ternary_scalar_bcast(
                                     attr()->post_ops_, dst_d),
            VERBOSE_UNSUPPORTED_POSTOP);

    // Resolve format_tag::any on binary post-op src1 memory descriptors.
    VDISPATCH_MATMUL(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);

    // ---- Binary post-op shape/format validation ----
    // Supported src1 layouts for binary post-ops via Zen (both ab,
    // row-major dense):
    //   * per_tensor : src1 dims match dst exactly (no broadcast)
    //   * per_channel: broadcast over every dim except the last (N/OC)
    //                  dim, i.e. the broadcast dims have extent 1 while
    //                  the channel dim is full-size.
    auto check_binary_postop_formats = [&]() -> bool {
        const auto &po = attr()->post_ops_;
        for (int i = 0; i < po.len(); i++) {
            const auto &entry = po.entry_[i];
            if (!entry.is_binary()) continue;

            const auto &src1_desc = entry.binary.src1_desc;
            const auto *dst = dst_md(0);

            // Each dim must either match dst (full) or be a unit-extent
            // broadcast (per_tensor => all full, per_channel => only the
            // channel dim full). The channel dim (last) must be full-size.
            if (src1_desc.ndims != dst->ndims) return false;
            const int channel_dim = src1_desc.ndims - 1;
            for (int d = 0; d < src1_desc.ndims; d++) {
                const bool full = src1_desc.dims[d] == dst->dims[d];
                const bool bcast = src1_desc.dims[d] == 1;
                if (!(full || bcast)) return false;
                if (d == channel_dim && !full) return false;
            }

            // ab format: plain row-major dense.
            const memory_desc_wrapper src1_mdw(src1_desc);
            if (!src1_mdw.is_plain()) return false;
            const auto &strides = src1_mdw.blocking_desc().strides;
            if (strides[1] != 1 || strides[0] != src1_desc.dims[1])
                return false;
        }
        return true;
    };
    VDISPATCH_MATMUL(check_binary_postop_formats(), VERBOSE_UNSUPPORTED_POSTOP);

    // Zen matmul_direct uses int for M/N/K and leading dimensions.
    // Reject descriptors whose dimensions or strides exceed INT_MAX.
    const matmul_helper_t helper(memory_desc_wrapper(src_md(0)),
            memory_desc_wrapper(weights_md(0)), memory_desc_wrapper(dst_md(0)));
    const dim_t int_max = std::numeric_limits<int>::max();
    // For the packed path the weights md is opaque (no blocking_desc), and the
    // backend uses ldb = N; otherwise read ldb from the plain weights strides.
    const dim_t wei_ldb = wei_zen_packed ? N() : helper.ldb();
    const bool fits_zen_int_api = helper.M() <= int_max && helper.N() <= int_max
            && helper.K() <= int_max && helper.lda() <= int_max
            && wei_ldb <= int_max && helper.ldc() <= int_max;
    VDISPATCH_MATMUL(fits_zen_int_api, VERBOSE_UNSUPPORTED_FEATURE,
            "dimension/stride > INT_MAX is not supported");

    return status::success;
#endif // DNNL_X64_USE_ZEN
}

status_t zen_matmul_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
#if DNNL_X64_USE_ZEN
    // Build Zen matmul_post_op chain directly from oneDNN attributes.
    // Static parts (type, alpha, beta, dtype, dims) are set here; only
    // binary buffer pointers are patched at execute() time.
    //
    // The chain is owned by the primitive (not pd_t) so that pd_t remains
    // cheaply-copyable by the framework's primitive cache, matching the
    // brgemm_matmul_t convention.
    const auto &po = pd()->attr()->post_ops_;
    zen_postop_.clear();
    zen_postop_.reserve(po.len());
    postop_indices_.clear();
    postop_indices_.reserve(po.len());
    beta_ = 0.f;

    using pot = zendnnl::ops::post_op_type_t;
    using zd = zendnnl::common::data_type_t;

    for (int i = 0; i < po.len(); i++) {
        const auto &entry = po.entry_[i];
        if (entry.is_sum(/*require_scale_one=*/false,
                    /*require_zp_zero=*/true)) {
            // Sum maps to Zen beta (C = alpha*A*B + beta*C).
            beta_ = entry.sum.scale;
            continue; // not a Zen post-op entry
        }
        matmul_post_op lpo {};
        if (entry.is_eltwise()) {
            switch (entry.eltwise.alg) {
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
            lpo.alpha = entry.eltwise.alpha;
            lpo.beta = entry.eltwise.beta;
        } else if (entry.is_binary()) {
            lpo.po_type = (entry.binary.alg == alg_kind::binary_add)
                    ? pot::binary_add
                    : pot::binary_mul;
            switch (entry.binary.src1_desc.data_type) {
                case f32: lpo.dtype = zd::f32; break;
                case bf16: lpo.dtype = zd::bf16; break;
                default: return status::runtime_error;
            }
            const auto &src1_desc = entry.binary.src1_desc;
            lpo.dims.assign(src1_desc.dims, src1_desc.dims + src1_desc.ndims);
            // buff pointer will be patched at execute() time
        }
        zen_postop_.push_back(lpo);
        postop_indices_.push_back(i);
    }
#endif // DNNL_X64_USE_ZEN
    return status::success;
}

// ================================================================
// Zen helpers and wrapper (translation-unit local).
// ================================================================
#if DNNL_X64_USE_ZEN
namespace {

// Unified Zen direct MatMul launcher.
// - Each tensor (src, wei, dst, bias) gets its own data type from the
//   oneDNN memory descriptors, supporting mixed-precision configs
//   (e.g. bf16 src/wei -> f32 dst, or f32 bias with bf16 compute).
// - transA/transB, lda/ldb/ldc are derived from matmul_helper_t.
// - Post-ops are pre-built at zen_matmul_t::init(engine_t*); only
//   binary buffer pointers are patched here from the execution context.
status_t zen_matmul_direct(data_type_t src_dt, data_type_t wei_dt,
        data_type_t dst_dt, data_type_t bia_dt, const void *A, const void *B,
        void *C, const void *bias, dim_t M, dim_t N, dim_t K, dim_t lda,
        dim_t ldb, dim_t ldc, char transA, char transB, char mem_format_b,
        const std::vector<matmul_post_op> &cached_postops,
        const std::vector<int> &cached_postop_po_indices, float cached_beta,
        const exec_ctx_t &ctx) {
    using zd = zendnnl::common::data_type_t;

    matmul_batch_params_t batch {};
    batch.Batch_A = 1;
    batch.Batch_B = 1;
    batch.batch_stride_src = -1;
    batch.batch_stride_wei = -1;
    batch.batch_stride_dst = -1;

    matmul_params params {};
    params.dtypes.src = to_zen_dt(src_dt);
    params.dtypes.wei = to_zen_dt(wei_dt);
    params.dtypes.dst = to_zen_dt(dst_dt);
    params.dtypes.bias = (bias ? to_zen_dt(bia_dt) : zd::none);
    params.dtypes.compute = zd::f32; // always accumulate in f32

    // 'r' = pre-packed, 'n' = plain weights;
    params.mem_format_b = mem_format_b;

    const char layout = 'r'; // row-major
    const bool trans_a = (transA != 'N');
    const bool trans_b = (transB != 'N');
    const float alpha = 1.f;
    const bool is_weights_const = (mem_format_b == 'r');

    // Copy pre-built post-ops and patch binary buffer pointers (only
    // available at execute time from the execution context).
    params.postop_ = cached_postops;
    for (size_t j = 0; j < params.postop_.size(); j++) {
        auto &lpo = params.postop_[j];
        if (lpo.po_type == zendnnl::ops::post_op_type_t::binary_add
                || lpo.po_type == zendnnl::ops::post_op_type_t::binary_mul) {
            lpo.buff = const_cast<void *>(CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(cached_postop_po_indices[j])
                            | DNNL_ARG_SRC_1));
        }
    }

    const auto st = matmul_direct(layout, trans_a, trans_b, (int)M, (int)N,
            (int)K, alpha, A, (int)lda, B, (int)ldb, bias, cached_beta, C,
            (int)ldc, is_weights_const, batch, params);

    // Defensive: scrub binary buffer pointers before the local `params`
    // (and its `params.postop_` vector) destructs. The buffers belong to
    // the oneDNN exec_ctx, not to matmul_post_op; leaving them set would
    // be a double-free hazard if matmul_post_op ever gained an owning dtor.
    for (auto &lpo : params.postop_)
        lpo.buff = nullptr;

    return to_dnnl_status(st);
}

} // anonymous namespace
#endif // DNNL_X64_USE_ZEN

status_t zen_matmul_t::execute_body(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // Build memory_desc_wrappers (needed by matmul_helper_t).
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    // Use matmul_helper_t to derive M, N, K, transA/B, lda/b/c (handles
    // arbitrary layouts and transpositions correctly).
    matmul_helper_t helper(src_d, weights_d, dst_d);

    // M, N, K and the src/dst leading dims come from src/dst descriptors and
    // are always valid. transB/ldb depend on the weights layout and are
    // resolved below (the packed weights md is opaque, so it has no
    // blocking_desc for matmul_helper_t to read).
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const char transA = helper.transA();
    const dim_t lda = helper.lda();
    const dim_t ldc = helper.ldc();

    // If weights are Zen pre-packed (opaque format_kind::zen_packed,
    // produced by zen_reorder_t), the backend expects:
    //   - mem_format_b = 'r'  (pre-packed)
    //   - transB       = 'N'  (logical orientation is K×N, matching trans='n')
    //   - ldb          = N    (plain K×N leading dim)
    // For plain `ab`/`ba` weights we use the helper-derived values and let
    // the backend pack them itself with mem_format_b='n'.
    const bool wei_is_zen_packed = is_zen_packed(*pd()->weights_md(0));
    char mem_format_b = 'n';
    char transB;
    dim_t ldb;
    if (wei_is_zen_packed) {
        mem_format_b = 'r';
        transB = 'N';
        ldb = N;
    } else {
        transB = helper.transB();
        ldb = helper.ldb();
    }

    // pd_t::init() rejects dimensions/strides above INT_MAX; assert here as a
    // defensive invariant before casting to the int-based Zen API.
    assert(M <= std::numeric_limits<int>::max()
            && N <= std::numeric_limits<int>::max()
            && K <= std::numeric_limits<int>::max()
            && lda <= std::numeric_limits<int>::max()
            && ldb <= std::numeric_limits<int>::max()
            && ldc <= std::numeric_limits<int>::max());

    // Per-tensor data types (may differ for mixed-precision configs).
    const auto src_dt = src_d.data_type();
    const auto wei_dt = weights_d.data_type();
    const auto dst_dt = dst_d.data_type();
    const auto bia_dt = pd()->with_bias() ? pd()->weights_md(1)->data_type
                                          : data_type::undef;

    VDEBUGINFO(2, primitive, matmul,
            "zen matmul: M=%ld N=%ld K=%ld transA=%c transB=%c lda=%ld "
            "ldb=%ld ldc=%ld src_dt=%d wei_dt=%d dst_dt=%d",
            (long)M, (long)N, (long)K, transA, transB, (long)lda, (long)ldb,
            (long)ldc, (int)src_dt, (int)wei_dt, (int)dst_dt);

    // Extract raw pointers.
    const void *A = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *B = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *C = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    // Dispatch to Zen with per-tensor data types and layout info.
    // Post-ops were pre-built at primitive init(); only binary buffer
    // pointers are patched here from the execution context.
    return zen_matmul_direct(src_dt, wei_dt, dst_dt, bia_dt, A, B, C, bias, M,
            N, K, lda, ldb, ldc, transA, transB, mem_format_b, zen_postop_,
            postop_indices_, beta_, ctx);
#endif // DNNL_X64_USE_ZEN
}

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
