/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVV_WEIGHTS_LAYOUT_HPP
#define CPU_RV64_RVV_WEIGHTS_LAYOUT_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

// The layout of weights a kernel reads as its A operand: the vectors run
// along one dimension (M), which is innermost, and the panel of an M block
// is contiguous. The M block is the one the kernel was tuned to.
namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Initializes `md` (its logical dimensions set) with its dimensions nested as
// `order` (outermost first) and `m_dim` innermost: blocked by `m_block`, M
// padded to it, or plain dense when M is shorter than one block
inline status_t init_m_blocked_md(
        memory_desc_t &md, int m_dim, dim_t m_block, const int *order) {
    const int ndims = md.ndims;
    const bool blocked = md.dims[m_dim] >= m_block;

    // The strides only give the nesting, from the largest
    blocking_desc_t blk = {};
    int rank = ndims;
    for (int i = 0; i < ndims; ++i) {
        if (!blocked && order[i] == m_dim) continue;
        blk.strides[order[i]] = rank--;
    }
    if (blocked) {
        blk.inner_nblks = 1;
        blk.inner_blks[0] = m_block;
        blk.inner_idxs[0] = m_dim;
    } else {
        blk.strides[m_dim] = rank;
    }
    return memory_desc_init_by_blocking_desc(md, blk);
}

// The M tile `md` is blocked by, 0 when M is innermost and plain. False when
// M is neither innermost nor the only inner block
inline bool m_layout_tile(
        const memory_desc_wrapper &md, int m_dim, dim_t &tile) {
    if (!md.is_blocking_desc() || !md.is_dense(true)) return false;
    const blocking_desc_t &blk = md.blocking_desc();
    tile = 0;
    if (blk.inner_nblks == 0) return blk.strides[m_dim] == 1;
    if (blk.inner_nblks != 1 || blk.inner_idxs[0] != m_dim) return false;
    tile = blk.inner_blks[0];
    return true;
}

// The strides, in elements, with which the kernel reads A from `md`: between
// K rows (`lda`) and between M blocks of `m_block` rows (`block_stride`).
// False when M is neither innermost nor blocked by `m_block`
inline bool m_blocked_strides(const memory_desc_wrapper &md, int m_dim,
        int k_dim, dim_t m_block, dim_t &lda, dim_t &block_stride) {
    if (!md.is_blocking_desc() || !md.is_dense(true)) return false;
    const blocking_desc_t &blk = md.blocking_desc();
    if (blk.inner_nblks == 0) {
        if (blk.strides[m_dim] != 1) return false;
        lda = blk.strides[k_dim];
        block_stride = m_block;
        return true;
    }
    if (blk.inner_nblks != 1 || blk.inner_idxs[0] != m_dim
            || blk.inner_blks[0] != m_block)
        return false;
    lda = blk.strides[k_dim];
    block_stride = blk.strides[m_dim];
    return lda == m_block;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
