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
// When src is `ba` (col-major), src_strides={1,K} describes the source;
// the destination is always written contiguous (lowoha reorder ignores
// dst_strides), so dst ends up in `ab` (K-major) regardless of src.
status_t f32_to_bf16_plain(
        const void *src, void *dst, int64_t K, int64_t N, bool src_is_ab) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.src_dtype = zd::f32;
    rp.dst_dtype = zd::bf16;
    rp.src_shape = {K, N};
    rp.dst_shape = {K, N};
    if (!src_is_ab) rp.src_strides = {1, K};

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

    VDISPATCH_REORDER_IC(id.ndims() == 2 && od.ndims() == 2, VERBOSE_BAD_NDIMS,
            "src/dst", id.ndims());

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

    // src must be plain `ab` or `ba` (the only two 2D row/col-major layouts
    // the packer accepts via the `trans` parameter).
    const auto src_tag = id.matches_one_of_tag(ab, ba);
    VDISPATCH_REORDER_IC(
            src_tag != format_tag::undef, VERBOSE_UNSUPPORTED_TAG_S, "src");

    // src and dst logical (K, N) must agree (oneDNN reorder API contract).
    VDISPATCH_REORDER_IC(
            id.dims()[0] == od.dims()[0] && id.dims()[1] == od.dims()[1],
            VERBOSE_INCONSISTENT_DIM, "src", 0, "dst", 0);

    // This reorder packs a single 2D (K, N) weight slice only; batched
    // (batch > 1) packed descriptors are not supported. A batched dst encodes
    // size = per_slice_size * batch, so require the two to be equal to reject
    // any batched packed buffer up front.
    const auto &zpd = od.zen_packed_desc();
    VDISPATCH_REORDER_IC(zpd.per_slice_size == zpd.size,
            VERBOSE_UNSUPPORTED_TAG_S, "dst (batched packed)");

    // Reject any dst whose buffer size disagrees with the
    // backend-reported packed size to turn a silent overrun into a clean
    // dispatch failure. ndims()==2 here, so size() == per-slice size.
    const dim_t expected_packed_bytes
            = zen_packed_bytes(od.dims()[0], od.dims()[1], type_o);
    VDISPATCH_REORDER_IC(expected_packed_bytes > 0
                    && od.size() == static_cast<size_t>(expected_packed_bytes),
            VERBOSE_INCONSISTENT_MDS, "dst", "packed-size");

    // The f32 -> bf16 prepack path needs a K*N bf16 conversion buffer. Book it
    // on the primitive scratchpad (declared here, consumed in execute() via the
    // grantor) so execution stays allocation-free.
    if (type_i == data_type::f32 && type_o == data_type::bf16) {
        const size_t conv_bytes = static_cast<size_t>(id.dims()[0])
                * static_cast<size_t>(id.dims()[1]) * sizeof(int16_t);
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

    // The src has been validated as a 2D plain `ab` or `ba` layout by
    // pd_t::init. Logical dims are always (K, N).
    const int64_t K = src_d.dims()[0];
    const int64_t N = src_d.dims()[1];

    // Detect transpose from the src strides:
    //   ab -> strides = {N, 1}  -> trans='n', ldb=N
    //   ba -> strides = {1, K}  -> trans='t', ldb=K
    // When K==1 (degenerate row), both layouts coincide and we treat the
    // src as `ab` (trans='n') since the byte stream is identical.
    const auto &src_strides = src_d.blocking_desc().strides;
    const bool src_is_ab = (src_strides[1] == 1) || (src_d.dims()[0] == 1);
    const bool transposed = !src_is_ab;

    const void *src_ptr = CTX_IN_MEM(const void *, DNNL_ARG_FROM);
    void *dst_ptr = CTX_OUT_MEM(void *, DNNL_ARG_TO);

    const auto src_dt = src_d.data_type();
    const auto dst_dt = dst_d.data_type();

    // ab -> ldb=N (trans='n'); ba -> ldb=K (trans='t').
    const int64_t ldb = src_is_ab ? N : K;

    if (src_dt == data_type::bf16 && dst_dt == data_type::bf16)
        return zen_weight_prepack(
                src_ptr, dst_ptr, zd::bf16, K, N, ldb, transposed);

    if (src_dt == data_type::f32 && dst_dt == data_type::f32)
        return zen_weight_prepack(
                src_ptr, dst_ptr, zd::f32, K, N, ldb, transposed);

    if (src_dt == data_type::f32 && dst_dt == data_type::bf16) {
        // Mixed-precision prepack: convert f32 -> plain bf16 (contiguous `ab`),
        // then prepack that bf16 into the Zen blocked layout. The conversion
        // scratch is required because f32_to_bf16_plain writes a new bf16
        // buffer; it is taken from the primitive scratchpad (booked in
        // pd_t::init), so execute() performs no heap allocation.
        const auto &scratchpad = ctx.get_scratchpad_grantor();
        void *conv = scratchpad.get<int16_t>(
                memory_tracking::names::key_reorder_space);
        if (conv == nullptr) return status::out_of_memory;

        status_t st = f32_to_bf16_plain(src_ptr, conv, K, N, src_is_ab);
        if (st == success) {
            // conv is in `ab` (K-major) layout regardless of original src.
            st = zen_weight_prepack(conv, dst_ptr, zd::bf16, K, N,
                    /*ldb=*/N, /*transposed=*/false);
        }

        return st;
    }

    return status::unimplemented;
#endif
}

} // namespace reorder
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
