/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_UTILS_HPP
#define GPU_INTEL_GEMM_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/nstl.hpp"
#include "common/opdesc.hpp"
#include "common/utils.hpp"
#include "gpu/intel/gemm/exec_types.hpp"

namespace dnnl {
namespace impl {

static inline status_t check_gemm_input(char transa, char transb, int m, int n,
        int k, int lda, int ldb, int ldc, float alpha, float beta) {
    using namespace status;
    bool consistency = true && utils::one_of(transa, 'T', 't', 'N', 'n')
            && utils::one_of(transb, 'T', 't', 'N', 'n') && m >= 0 && n >= 0
            && k >= 0;
    if (!consistency) return invalid_arguments;
    bool isTransA = utils::one_of(transa, 'T', 't');
    bool isTransB = utils::one_of(transb, 'T', 't');
    int nrowA = isTransA ? k : m;
    int nrowB = isTransB ? n : k;
    consistency = true && lda >= nstl::max(1, nrowA)
            && ldb >= nstl::max(1, nrowB) && ldc >= nstl::max(1, m);
    if (!consistency) return invalid_arguments;

    return success;
}

static inline status_t check_gemm_x8x8s32_input(char offsetc, char transa,
        char transb, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta) {
    using namespace status;
    if (!utils::one_of(offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return invalid_arguments;
    return check_gemm_input(
            transa, transb, m, n, k, lda, ldb, ldc, alpha, beta);
}

// This function makes a 2d tensor from an nd tensor.
// the 2d tensor just collapes dims[1...ndims-1] from the nd tensor
// The only reason we do not use reshape here is that we want to allow
// fusing blocked dimensions and padded dimensions.
static inline status_t init_2d_desc(memory_desc_t *md_2d,
        const memory_desc_t *md_nd, bool transpose_dims = false) {
    auto p_dims = md_nd->padded_dims;
    auto blk = md_nd->format_desc.blocking;
    auto strides = blk.strides;

    // we assume that the innermost dimension always has stride 1
    assert(IMPLICATION(blk.inner_nblks == 0,
            utils::array_min(strides, md_nd->ndims) == 1));

    // TODO: add checks to see if the memory descriptor can be 2d-fied
    // TODO: change signature to specifiy at which dimension shall we 2d-fy (currently 1st)
    auto p_dim1 = utils::array_product(p_dims + 1, md_nd->ndims - 1);
    auto stride1 = blk.inner_nblks == 0
            ? utils::array_min(strides + 1, md_nd->ndims - 1)
            : 1;

    if (transpose_dims) {
        dnnl_dims_t dims_2d = {p_dim1, p_dims[0]};
        dnnl_dims_t strides_2d = {stride1, strides[0]};
        return memory_desc_init_by_strides(
                *md_2d, 2, dims_2d, md_nd->data_type, strides_2d);
    } else {
        dnnl_dims_t dims_2d = {p_dims[0], p_dim1};
        dnnl_dims_t strides_2d = {strides[0], stride1};
        return memory_desc_init_by_strides(
                *md_2d, 2, dims_2d, md_nd->data_type, strides_2d);
    }
}

static inline status_t create_2d_desc(memory_desc_t *md_2d, int d0, int d1,
        data_type_t dt, transpose_t trans, int ld) {
    dnnl_dims_t dims_2d = {d0, d1};
    if (trans == transpose::notrans) {
        dnnl_dims_t strides_2d = {ld, 1};
        return memory_desc_init_by_strides(*md_2d, 2, dims_2d, dt, strides_2d);
    } else {
        dnnl_dims_t strides_2d = {1, ld};
        return memory_desc_init_by_strides(*md_2d, 2, dims_2d, dt, strides_2d);
    }
}

namespace gpu {
namespace intel {
namespace gemm {

// Memory-desc introspection helpers used by SDPA / grouped_micro_gemm /
// gated_mlp to compute kernel-A/B leading dims and transposition flags.
// Bit-identical to the former gemm_desc_t::get_trans / get_ld static methods,
// preserved so callers keep their original layout decisions.
//
// KNOWN PRE-EXISTING BUG (intentionally preserved for bit-compat with base):
// the `last_dim != 1` short-circuit silently mis-classifies degenerate
// single-column/row matrices (e.g. [m, 1] strides [1, m] returns notrans
// even though the n-axis carries the larger stride). See CLAUDE.md gotcha
// #10 — the internal copy in jit_gemm_pd_t::get_trans was fixed there by
// dropping the `last_dim != 1` gate. Do NOT copy this helper assuming it's
// canonical; if you need correct behavior on degenerate shapes, use the
// internal jit_gemm_pd_t::get_trans / get_ld directly.
static inline transpose_t get_md_trans(const memory_desc_t &md) {
    if (!md.ndims) return transpose::notrans;

    using namespace data_type;
    const bool is_4bit = utils::one_of(md.data_type, f4_e2m1, f4_e3m0, s4, u4);
    dim_t last_dim = md.dims[md.ndims - 1];
    auto strides = md.format_desc.blocking.strides;
    dim_t notranspose_ld
            = md.dims[md.ndims - 2] > 1 ? strides[md.ndims - 2] : last_dim;
    if (is_4bit && notranspose_ld % 2 != 0) return transpose::trans;

    return last_dim != 1 && strides[md.ndims - 1] != 1 ? transpose::trans
                                                       : transpose::notrans;
}

static inline dim_t get_md_ld(const memory_desc_t &md) {
    auto strides = md.format_desc.blocking.strides;
    assert(md.dims[md.ndims - 1] == 1 || strides[md.ndims - 1] == 1
            || md.dims[md.ndims - 2] == 1 || strides[md.ndims - 2] == 1);
    switch (get_md_trans(md)) {
        case transpose::trans:
            return md.dims[md.ndims - 1] > 1 ? strides[md.ndims - 1]
                                             : md.dims[md.ndims - 2];
        default:
            return md.dims[md.ndims - 2] > 1 ? strides[md.ndims - 2]
                                             : md.dims[md.ndims - 1];
    }
}

} // namespace gemm
} // namespace intel
} // namespace gpu

static inline bool is_md_gemm_compatible_plain_format(
        const memory_desc_t *md, bool is_dst = false) {

    if (md->format_kind != format_kind::blocked) return false;

    auto &blk_desc = md->format_desc.blocking;

    if (blk_desc.inner_nblks != 0) return false;

    return (md->dims[md->ndims - 1] == 1
                   || blk_desc.strides[md->ndims - 1] == 1)
            || (!is_dst
                    && (md->dims[md->ndims - 2] == 1
                            || blk_desc.strides[md->ndims - 2] == 1));
}

} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_GEMM_UTILS_HPP
