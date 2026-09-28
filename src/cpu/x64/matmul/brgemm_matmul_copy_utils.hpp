/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_MATMUL_BRGEMM_MATMUL_COPY_UTILS_HPP
#define CPU_X64_MATMUL_BRGEMM_MATMUL_COPY_UTILS_HPP

#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

struct jit_brgemm_matmul_copy_b_t {
    struct ctx_t {
        const void *src = nullptr;
        const void *tr_src = nullptr;
        const void *compensation_ptr = nullptr;
        const void *zp_a_compensation_ptr = nullptr;
        const void *zp_a_neg_value_ptr = nullptr;
        const void *zp_b_value_ptr = nullptr;
        const void *src_scales_ptr = nullptr;
        const void *wei_scales_ptr = nullptr;

        dim_t current_K_start = 0;
        dim_t current_K_iters = 0;
        dim_t current_K_pad = 0;
        dim_t current_N_blk = 0;
        dim_t dynamic_src_stride = 0;
    };

    virtual void operator()(const ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_b_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_b_t() = default;

    const brgemm_matmul_conf_t *conf_;
};

struct jit_brgemm_matmul_copy_a_t {
    struct ctx_t {
        const void *src = nullptr;
        const void *tr_src = nullptr;
        const void *zp_b_compensation_buffer_ptr = nullptr;
        const void *zp_a_compensation_result_ptr = nullptr;
        const void *zp_b_neg_val_ptr = nullptr;
        const void *zp_ab_comp_ptr = nullptr;

        dim_t current_K_start = 0;
        dim_t current_K_blk = 0;
        dim_t current_M_blk = 0;
        dim_t dynamic_src_ld = 0;
    };

    virtual void operator()(const ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_a_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_a_t() = default;

    const brgemm_matmul_conf_t *conf_;
};

// Repacks the e8m0 block scales of A from the user layout [M][K / group_size]
// into the blocked layout the MXFP8 micro-kernel reads. See the layout comment
// on jit_brgemm_matmul_copy_a_scales_impl_t, which is the single source of
// truth for it.
struct jit_brgemm_matmul_copy_a_scales_t {
    struct ctx_t {
        // Base of the user scales, already advanced to the (M_blk, K_blk)
        // block by the caller.
        const void *src_scales = nullptr;
        // Base of the repacked slab of this (M_blk, K_blk) block.
        const void *tr_src_scales = nullptr;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_a_scales_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_a_scales_t() = default;

    const brgemm_matmul_conf_t *conf_;
};

// Relayouts the MXFP8 e8m0 dst scales of one (M_blk, N_blk) block from the
// staging buffer written by the micro-kernel into the user layout
// [M][N / group_size]. See the layout comment on
// jit_brgemm_matmul_copy_dst_scales_impl_t.
struct jit_brgemm_matmul_copy_dst_scales_t {
    struct ctx_t {
        // Base of the user dst scales, already advanced to the block by the
        // caller. Written by the kernel.
        const void *d_scales = nullptr;
        // Base of the per-thread staging buffer of the block.
        const void *tr_d_scales = nullptr;
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_matmul_copy_dst_scales_t(const brgemm_matmul_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_matmul_copy_dst_scales_t() = default;

    const brgemm_matmul_conf_t *conf_;
};

// Index of the tail flavor of the MX scales copy kernels:
// [is_M_tail | (is_other_tail << 1)], `other` being K for the A scales and N
// for the dst scales.
inline int mx_scales_kernel_idx(bool is_M_tail, bool is_other_tail) {
    return (is_M_tail ? 1 : 0) | (is_other_tail ? 2 : 0);
}

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf);

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf);

// Creates the four tail flavors of the copy-A-scales kernel, indexed as
// [is_M_tail | (is_K_tail << 1)]. Entries whose tail does not exist are left
// null.
status_t create_brgemm_matmul_copy_a_scales(
        std::unique_ptr<jit_brgemm_matmul_copy_a_scales_t>
                *copy_A_scales_kernel,
        const brgemm_matmul_conf_t *conf);

// Creates the four tail flavors of the dst scales relayout kernel, indexed as
// mx_scales_kernel_idx(is_M_tail, is_N_tail). Entries whose tail does not
// exist are left null.
status_t create_brgemm_matmul_copy_d_scales(
        std::unique_ptr<jit_brgemm_matmul_copy_dst_scales_t>
                *copy_D_scales_kernel,
        const brgemm_matmul_conf_t *conf);

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
