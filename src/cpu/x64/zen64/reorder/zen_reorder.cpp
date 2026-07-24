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

#include "cpu/x64/zen64/reorder/zen_reorder.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include <cstdint>
#include <limits>

#include "cpu/x64/cpu_isa_traits.hpp" // cpu().has(tAMD)
#include "cpu/x64/zen64/common/zen_format_tag.hpp" // is_zen_packed

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "zendnnl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace reorder {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;

#if DNNL_X64_USE_ZEN
namespace {

using zd = zendnnl::common::data_type_t;
using zendnnl::ops::matmul_algo_t;

// Zen weight prepack via reorder_direct (is_prepack=true); see
// ZenDNN/zendnnl/src/lowoha_operators/reorder/lowoha_reorder.hpp.
status_t zen_weight_prepack(const void *src, void *dst, zd wei_dt, int64_t K,
        int64_t N, int64_t ldb, bool transposed) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = wei_dt;
    rp.prepack.src_dtype = wei_dt;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = transposed;

    return to_dnnl_status(
            zendnnl::lowoha::reorder::reorder_direct(src, dst, rp));
}

// Plain f32 -> bf16 element conversion (standard reorder_direct path).
// Writes K rows of N elements contiguously into `dst` (row-major).
//
// `ldb` is the source leading dim (in elements), read from the actual src
// strides so a padded leading dim is honored:
//   * ab (row-major): src_strides = {ldb, 1}   (ldb == N for dense, N+pad if padded)
//   * ba (col-major): src_strides = {1, ldb}    (ldb == K for dense, K+pad if padded)
// The destination is always written contiguous (lowoha reorder ignores
// dst_strides), so dst ends up in `ab` (K-major) regardless of src.
status_t f32_to_bf16_plain(const void *src, void *dst, int64_t K, int64_t N,
        int64_t ldb, bool src_is_ab) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.src_dtype = zd::f32;
    rp.dst_dtype = zd::bf16;
    rp.src_shape = {K, N};
    rp.dst_shape = {K, N};
    if (src_is_ab)
        rp.src_strides = {ldb, 1};
    else
        rp.src_strides = {1, ldb};

    return to_dnnl_status(
            zendnnl::lowoha::reorder::reorder_direct(src, dst, rp));
}

} // namespace
#endif

status_t zen_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    VDISPATCH_REORDER_IC(src_engine->kind() == engine_kind::cpu
                    && dst_engine->kind() == engine_kind::cpu,
            VERBOSE_UNSUPPORTED_FEATURE, "non-CPU engine");

    VDISPATCH_REORDER_IC(
            ::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD),
            "This implementation only supports AMD CPUs");

    // Zen weight prepack requires AVX-512 core support regardless of data type.
    VDISPATCH_REORDER_IC(mayiuse(avx512_core), VERBOSE_UNSUPPORTED_ISA);

    const memory_desc_wrapper id(src_md_), od(dst_md_);

    // 2D weight slice, or 3D batched weights (one (K, N) slice per batch).
    VDISPATCH_REORDER_IC(
            utils::one_of(id.ndims(), 2, 3) && id.ndims() == od.ndims(),
            VERBOSE_BAD_NDIMS, "src/dst", id.ndims());

    const int ndims = id.ndims();
    const bool batched = ndims == 3;

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();
    // Supported dtype combos:
    //   bf16 -> bf16  : reorder_direct prepack (Zen blocked algo)
    //   f32  -> f32   : reorder_direct prepack (Zen blocked algo)
    //   f32  -> bf16  : f32->bf16 plain reorder_direct, then bf16 prepack
    //                   (avoids the backend f32->bf16 fringe-N bug; see execute)
    const bool dt_ok = (type_i == data_type::bf16 && type_o == data_type::bf16)
            || (type_i == data_type::f32 && type_o == data_type::f32)
            || (type_i == data_type::f32 && type_o == data_type::bf16);
    VDISPATCH_REORDER_IC(dt_ok, VERBOSE_UNSUPPORTED_DT);

    // Dispatch trigger: only fire when the dst uses the dedicated opaque
    // Zen packed format; otherwise let the regular reorder list handle it.
    VDISPATCH_REORDER_IC(
            is_zen_packed(dst_md_), VERBOSE_UNSUPPORTED_FORMAT_KIND);

    // The dst is the opaque packed format (no oneDNN blocked layout), so there
    // is no blocked-layout / K-alignment / zero-padding requirement to validate
    // here. The recorded buffer size is cross-checked against the backend's
    // packed size further below.

    VDISPATCH_REORDER_IC(
            attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    // The src is a plain blocked layout; the dst is the opaque packed format
    // (not a blocking_desc).
    VDISPATCH_REORDER_IC(
            id.is_blocking_desc(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    VDISPATCH_REORDER_IC(!id.has_runtime_dims_or_strides()
                    && !od.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VDISPATCH_REORDER_IC(!id.has_zero_dim() && !od.has_zero_dim(),
            VERBOSE_BAD_DIM, "src/dst", 0);

    // Each (K, N) slice of src must be plain row-major (`ab`/`abc`) or
    // col-major (`ba`/`acb`). The packer takes an explicit leading dim (derived
    // from the actual strides in execute()), so a padded leading dim on one
    // axis is supported -- an exact tag check would reject those layouts. Match
    // the matmul contract: require a plain layout with one of the inner (K, N)
    // axes contiguous and no zero strides. For 3D also require the batch dim to
    // be outermost (largest stride) so the per-batch stride is the true slice
    // span (execute() advances src by strides[0] per batch).
    const auto &src_strides = id.blocking_desc().strides;
    const bool inner_contig
            = src_strides[ndims - 1] == 1 || src_strides[ndims - 2] == 1;
    bool no_zero_stride = true;
    for (int i = 0; i < ndims; i++)
        no_zero_stride = no_zero_stride && src_strides[i] != 0;
    VDISPATCH_REORDER_IC(id.is_plain() && inner_contig && no_zero_stride,
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_REORDER_IC(!batched
                    || (src_strides[0] >= src_strides[1]
                            && src_strides[0] >= src_strides[2]),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    // src and dst logical dims must agree (oneDNN reorder API contract).
    for (int i = 0; i < ndims; i++)
        VDISPATCH_REORDER_IC(id.dims()[i] == od.dims()[i],
                VERBOSE_INCONSISTENT_DIM, "src", i, "dst", i);

    // Logical (K, N) of one slice and the batch count.
    const dim_t K = od.dims()[ndims - 2];
    const dim_t N = od.dims()[ndims - 1];
    const dim_t batch = batched ? od.dims()[0] : 1;

    // execute() collapses a K==1 slice to contiguous row-major (`ab`) since a
    // dense single row is layout-agnostic. A padded col-major (`acb`/`ba`) row
    // (N stride != 1) is *not* contiguous, so that assumption would misread it;
    // decline it here and let the reference reorder serve the (pathological)
    // case instead.
    VDISPATCH_REORDER_IC(K != 1 || src_strides[ndims - 1] == 1,
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    // zen_weight_prepack takes int64_t K/N/ldb but the matmul that consumes the
    // packed buffer drives them through the int Zen API; reject oversized slices
    // up front so packing and matmul agree on what is representable.
    const dim_t int_max = std::numeric_limits<int>::max();
    VDISPATCH_REORDER_IC(K <= int_max && N <= int_max && batch <= int_max,
            VERBOSE_UNSUPPORTED_FEATURE,
            "dimension > INT_MAX is not supported");

    // Cross-check the recorded packed sizes against the backend-reported
    // per-slice size to turn a silent overrun into a clean dispatch failure.
    // The opaque dst carries per_slice_size and size == per_slice_size * batch.
    const auto &zpd = od.zen_packed_desc();
    const dim_t expected_per_slice = zen_packed_bytes(K, N, type_o);
    VDISPATCH_REORDER_IC(expected_per_slice > 0
                    && zpd.per_slice_size
                            == static_cast<size_t>(expected_per_slice),
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-slice-size");

    // batch is validated >= 1 (no zero dims) and <= INT_MAX above. Guard the
    // per_slice_size * batch product against size_t overflow before comparing:
    // a wrap could otherwise make an undersized buffer pass this check and lead
    // to out-of-bounds writes during packing. Mirrors init_zen_packed_md().
    const size_t batch_sz = static_cast<size_t>(batch);
    VDISPATCH_REORDER_IC(zpd.per_slice_size <= SIZE_MAX / batch_sz,
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-size-overflow");
    VDISPATCH_REORDER_IC(
            zpd.size == zpd.per_slice_size * batch_sz && od.size() == zpd.size,
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-size");

    // The f32 -> bf16 prepack path needs a K*N bf16 conversion buffer. Book it
    // on the primitive scratchpad (declared here, consumed in execute() via the
    // grantor) so execution stays allocation-free.
    if (type_i == data_type::f32 && type_o == data_type::bf16) {
        // One (K, N) bf16 slice; reused across batches in execute().
        const size_t conv_bytes = static_cast<size_t>(K)
                * static_cast<size_t>(N) * sizeof(int16_t);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_space, conv_bytes,
                /*data_size=*/1, /*alignment=*/64);
    }

    return status::success;
#endif
}

status_t zen_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    using namespace status;

    VDISPATCH_REORDER_IC(impl::is_dense_format_kind({src_md, dst_md}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return out_of_memory;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());
    return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd.release());
}

status_t zen_reorder_t::execute(const exec_ctx_t &ctx) const {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    // src is a plain 2D (K, N) slice or a 3D batched [B, K, N] tensor, each
    // slice plain `ab`/`ba` (validated by pd_t::init). Logical per-slice dims
    // are always (K, N); the batch dim (if any) is outermost.
    const int ndims = src_d.ndims();
    const bool batched = ndims == 3;
    const int64_t K = src_d.dims()[ndims - 2];
    const int64_t N = src_d.dims()[ndims - 1];
    const int64_t batch = batched ? src_d.dims()[0] : 1;

    // Detect transpose from the slice's last-two-dim strides:
    //   ab -> strides[..]={ldb, 1}  -> trans='n', ldb = row stride
    //   ba -> strides[..]={1, ldb}  -> trans='t', ldb = col stride
    // When K==1 (degenerate row) and the row is contiguous (N stride == 1),
    // both layouts coincide, so treat the src as `ab` (trans='n'); the byte
    // stream is identical. A padded (non-contiguous) K==1 row is rejected in
    // pd_t::init(), so it never reaches here.
    const auto &src_strides = src_d.blocking_desc().strides;
    const bool src_is_ab = (src_strides[ndims - 1] == 1) || (K == 1);
    const bool transposed = !src_is_ab;

    // The leading dim is read from the actual inner stride (not the logical
    // extent), so a padded leading dim is honored: ab -> row stride
    // (strides[ndims-2]); ba -> col stride (strides[ndims-1]).
    // Exception: when K==1 the slice is a single row and both layouts coincide
    // (src_is_ab is forced true above), but the K-dimension stride is then
    // degenerate (can be 1 for an `acb` source) rather than the row length, so
    // fall back to the logical N -- the packer requires ldb >= N.
    const int64_t ldb = src_is_ab ? (K == 1 ? N : src_strides[ndims - 2])
                                  : src_strides[ndims - 1];

    const auto src_dt = src_d.data_type();
    const auto dst_dt = dst_d.data_type();

    // Per-batch advance: src by its leading-dim stride (elements), dst by one
    // fixed-size packed slot (per_slice_size bytes).
    const size_t src_elem = src_d.data_type_size();
    const size_t src_slice_bytes
            = batched ? static_cast<size_t>(src_strides[0]) * src_elem : 0;
    const size_t dst_slice_bytes = dst_d.zen_packed_desc().per_slice_size;

    const auto *src_base = CTX_IN_MEM(const uint8_t *, DNNL_ARG_FROM);
    auto *dst_base = CTX_OUT_MEM(uint8_t *, DNNL_ARG_TO);

    // f32 -> bf16 prepack needs a per-slice bf16 conversion buffer (booked in
    // pd_t::init), reused across batches so execute() stays allocation-free.
    void *conv = nullptr;
    if (src_dt == data_type::f32 && dst_dt == data_type::bf16) {
        conv = ctx.get_scratchpad_grantor().get<int16_t>(
                memory_tracking::names::key_reorder_space);
        if (conv == nullptr) return status::out_of_memory;
    }

    // Pack one (K, N) slice from `src` into `dst`.
    auto prepack_slice = [&](const void *src, void *dst) -> status_t {
        if (src_dt == data_type::bf16 && dst_dt == data_type::bf16)
            return zen_weight_prepack(
                    src, dst, zd::bf16, K, N, ldb, transposed);

        if (src_dt == data_type::f32 && dst_dt == data_type::f32)
            return zen_weight_prepack(src, dst, zd::f32, K, N, ldb, transposed);

        if (src_dt == data_type::f32 && dst_dt == data_type::bf16) {
            // Convert f32 -> plain bf16 (contiguous `ab`), then prepack that
            // bf16 slice into the Zen blocked layout. The conversion avoids the
            // backend's f32->bf16 fringe-N bug (see header note).
            status_t st = f32_to_bf16_plain(src, conv, K, N, ldb, src_is_ab);
            if (st == success)
                st = zen_weight_prepack(conv, dst, zd::bf16, K, N,
                        /*ldb=*/N, /*transposed=*/false);
            return st;
        }

        return status::unimplemented;
    };

    for (int64_t b = 0; b < batch; b++) {
        const void *src_slice
                = src_base + static_cast<size_t>(b) * src_slice_bytes;
        void *dst_slice = dst_base + static_cast<size_t>(b) * dst_slice_bytes;
        CHECK(prepack_slice(src_slice, dst_slice));
    }

    return status::success;
#endif
}

} // namespace reorder
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
