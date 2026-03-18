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

// K-blocking tile size.  Chosen so that the A working set per M-tile
// fits comfortably in the 32 KB L1D cache:
//   bd_block(8) * BK(256) * 4 bytes = 8 KB.
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
    int bd_block; // M tile per vector (LMUL-dependent)
    int bdb; // number of full bd_block tiles in M
    int bdb_tail; // remaining M rows

    int n_step; // N columns processed per inner iteration (4)
    int rd_block; // K unroll factor (4)
    int rdb; // K / rd_block
    int rdb_tail; // K % rd_block

    bool is_f32;
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
