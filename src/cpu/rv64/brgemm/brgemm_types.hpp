/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_RV64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

enum brgemm_batch_kind_t {
    brgemm_batch_kind_undef = 0,
    brgemm_addr = 1,
    brgemm_offs = 2,
    brgemm_strd = 3,
};

enum brgemm_layout_t {
    brgemm_layout_undef = 0,
    brgemm_col_major = 1,
    brgemm_row_major = 2,
};

struct brgemm_strides_t {
    dim_t stride_a;
    dim_t stride_b;
};

struct brgemm_batch_element_t {
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
};

// K-blocking tile size: the K rows of A one call reads per M tile.
static constexpr int BRGEMM_BK = 256;

// BRGEMM descriptor: configured at init time, constants baked into JIT kernel.
struct brgemm_desc_t {
    dim_t bcast_dim; // M
    dim_t load_dim; // N (informational; actual N is a runtime parameter)
    dim_t reduce_dim; // K

    dim_t LDA, LDB, LDC;

    float alpha, beta;

    cpu_isa_t isa_impl;
    data_type_t dt_a, dt_b, dt_c;

    int typesize_A, typesize_B, typesize_C;

    brgemm_batch_kind_t type;
    brgemm_layout_t layout;
    dim_t stride_a, stride_b;

    // Blocking parameters (computed by brgemm_desc_init).
    // The blocks rvjit tuned, which the kernel is pinned to
    rvjit_matmul_blocks_t blocks;
    int bd_block; // M tile: blocks.m_block(), or the weights' tile
    bool wide_tile; // the weights may be laid out by a wider tile
    int wei_tile; // the tile the weights are laid out by, when it is given
    dim_t n_bound; // upper bound of the N of a call, when known (0: none)
    int bdb; // number of full bd_block tiles in M
    int bdb_tail; // remaining M rows
    // Elements of A between bd_block tiles: bd_block when A is contiguous
    // along M (set by brgemm_desc_init), more for M-blocked weights
    dim_t stride_a_m_block;

    bool is_f32;
    bool is_int8;

    bool stores_f16() const { return dt_c == data_type::f16; }

    dim_t get_k_block() const {
        // Round f16 output only once after the full f32 accumulation;
        // other outputs retain cache-friendly K blocking.
        return stores_f16() ? reduce_dim : BRGEMM_BK;
    }
};

// Runtime parameters passed to the JIT micro-kernel for one M-tile.
struct brgemm_kernel_params_t {
    const void *ptr_A; // offset 0
    const void *ptr_B; // offset 8
    void *ptr_C; // offset 16
    dim_t N; // offset 24: number of output columns
    dim_t M; // offset 32: actual rows in this tile
    dim_t K; // offset 40: reduction dimension (runtime, for K-blocking)
    float beta; // offset 48: 0.0f or 1.0f
    const void
            *ptr_bias; // offset 56: bias vector (length M), nullptr if unused
    uint64_t rt_flags; // set by the rvjit kernel: the features of this call
};

// Abstract JIT kernel base.
struct brgemm_kernel_t {
    virtual ~brgemm_kernel_t() = default;
    virtual status_t create_kernel() = 0;
    virtual void operator()(brgemm_kernel_params_t *) const = 0;
    virtual const brgemm_desc_t &get_brg() const = 0;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_BRGEMM_BRGEMM_TYPES_HPP
