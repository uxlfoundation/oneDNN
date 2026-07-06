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

#ifndef CPU_X64_ZEN64_COMMON_ZEN_FORMAT_TAG_HPP
#define CPU_X64_ZEN64_COMMON_ZEN_FORMAT_TAG_HPP

#include <cstdint> // SIZE_MAX

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp" // reorder_params_t
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp" // weight_prepack_size
#include "zendnnl.hpp" // zendnnl::common::data_type_t
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {

// Zen f32/bf16 matmul has two weight paths. On the prepacked path (selected
// only when the weights md is format_any) the weights md is converted to the
// dedicated opaque `format_kind::zen_packed`; the bytes are produced by the
// Zen backend packer (via zen_reorder_t) and the matmul backend consumes
// the buffer directly with mem_format_b='r'. On the plain-weights path the
// weights keep their `ab`/`ba` layout (a regular blocked format) and the
// backend packs them itself at execute.
//
// The opaque format carries its own size in `zen_packed_desc_t::size`, which
// is what `memory_desc_wrapper::size()` reports, so frameworks allocate the
// exact number of bytes the packer needs -- there is no oneDNN blocked
// "envelope" tag and no zero-padding contract to satisfy.

#if DNNL_X64_USE_ZEN
// Map a zendnnl status enum to a oneDNN status. ZenDNN exposes several
// per-component status enums (e.g. zendnnl::error_handling::status_t for the
// matmul path, zendnnl::lowoha::reorder::status_t for the reorder path); they
// all share the `success` / `unimplemented` enumerators, so a single template
// covers every Zen call site and keeps the mapping from drifting apart.
// Instantiated where the concrete enum type is complete (the matmul / reorder
// translation units), so this header need not name any specific enum.
template <typename zen_status_t>
inline status_t to_dnnl_status(zen_status_t st) {
    if (st == zen_status_t::success) return status::success;
    if (st == zen_status_t::unimplemented) return status::unimplemented;
    return status::runtime_error;
}

// oneDNN -> Zen (zendnnl) data type mapping. Shared by the Zen matmul and
// reorder paths (and the prepack size query below).
inline zendnnl::common::data_type_t to_zen_dt(data_type_t dt) {
    using zd = zendnnl::common::data_type_t;
    switch (dt) {
        case data_type::f32: return zd::f32;
        case data_type::bf16: return zd::bf16;
        case data_type::f16: return zd::f16;
        case data_type::u8: return zd::u8;
        case data_type::s8: return zd::s8;
        case data_type::s32: return zd::s32;
        case data_type::s4: return zd::s4;
        default: return zd::none;
    }
}

// Authoritative packed-buffer size (bytes) for the Zen backend,
// queried directly from the backend via weight_prepack_size() so it always
// matches what the packer (zen_reorder_t) actually writes. The size is
// layout-invariant w.r.t. ldb/transpose, so the query uses the canonical
// row-major, untransposed form (ldb = N, transposed = false).
inline dim_t zen_prepack_size(
        data_type_t wei_dt, data_type_t src_dt, dim_t K, dim_t N) {
    using zd = zendnnl::common::data_type_t;
    const zd zwei = to_zen_dt(wei_dt);
    const zd zsrc = to_zen_dt(src_dt);
    if (zwei == zd::none || zsrc == zd::none || K <= 0 || N <= 0) return 0;

    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = zwei;
    rp.prepack.src_dtype = zsrc;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = N;
    rp.prepack.transposed = false;
    rp.prepack.sym_group_size = 0;
    return static_cast<dim_t>(
            zendnnl::lowoha::reorder::weight_prepack_size(rp));
}
#endif

// Zen packer required buffer size (bytes) for weights of type `dt`, with the
// matmul source data type equal to the weight type (the f32/bf16 Zen matmul
// configs). The exact size is obtained from the backend's weight_prepack_size()
// API; returns 0 for unsupported types or when Zen is disabled.
inline dim_t zen_packed_bytes(dim_t K, dim_t N, data_type_t dt) {
#if DNNL_X64_USE_ZEN
    return zen_prepack_size(dt, dt, K, N);
#else
    MAYBE_UNUSED(K);
    MAYBE_UNUSED(N);
    MAYBE_UNUSED(dt);
    return 0;
#endif
}

// True when the weights memory descriptor uses the dedicated Zen packed
// opaque format. Execute uses mem_format_b='r', transB='N', ldb=N for such
// weights.
inline bool is_zen_packed(const memory_desc_t &md) {
    return md.format_kind == format_kind::zen_packed;
}

// Convert a (resolved, plain) weights memory descriptor in-place to the Zen
// packed opaque format. `dims`, `padded_dims` and `data_type` are preserved
// (they must already be set, e.g. by set_default_formats()); only
// `format_kind` and `format_desc` are overwritten.
//
// gemm_src_dt is the matmul source/compute data type. For the supported Zen
// configs (uniform f32, uniform bf16, bf16 src/wei -> f32 dst) it equals the
// weights compute type; it is recorded so packed descriptors for different
// GEMM source types stay distinct in the primitive cache.
inline status_t init_zen_packed_md(memory_desc_t &weights_md,
        data_type_t gemm_src_dt, dim_t K, dim_t N, dim_t batch = 1) {
    const dim_t per_slice = zen_packed_bytes(K, N, weights_md.data_type);
    if (per_slice <= 0) return status::unimplemented;

    // Guard the total-size multiplication: a non-positive batch or a
    // per_slice * batch product that wraps size_t would yield an undersized
    // zen_packed_desc.size, causing out-of-bounds writes during
    // prepack/reorder. Reject both before mutating the descriptor.
    if (batch <= 0) return status::invalid_arguments;
    const size_t per_slice_sz = static_cast<size_t>(per_slice);
    const size_t batch_sz = static_cast<size_t>(batch);
    if (per_slice_sz > SIZE_MAX / batch_sz) return status::invalid_arguments;

    weights_md.format_kind = format_kind::zen_packed;
    weights_md.format_desc.zen_packed_desc.per_slice_size = per_slice_sz;
    weights_md.format_desc.zen_packed_desc.size = per_slice_sz * batch_sz;
    weights_md.format_desc.zen_packed_desc.gemm_src_dt = gemm_src_dt;
    return status::success;
}

} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
