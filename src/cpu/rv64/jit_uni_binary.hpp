/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 Intel Corporation
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

#ifndef CPU_RV64_JIT_UNI_BINARY_HPP
#define CPU_RV64_JIT_UNI_BINARY_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_eltwise_pd.hpp"

#include "cpu/rv64/jit_uni_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct binary_kernel_t;

using op_t = binary_op_t;
using bcast_t = binary_bcast_t;

// Standalone binary primitive: a VLA JIT wrapper that computes
// (scale0*src0) OP (scale1*src1) in f32, applies the sum + eltwise + binary
// post-op chain, and stores to dst (converted at the boundary), mirroring the
// x64/aarch64 jit_uni_binary_t contract. Supports:
//   - f32/f16/bf16/s32/s8/u8 tensors (f16 needs zvfh, bf16 needs
//     zvfbfwma); src0 and dst share one memory desc unless dst is int8,
//     and src1's dtype may differ
//   - the x64 broadcast surface: full-tensor, scalar, per-batch/per-C/per-W
//     pattern families over plain (ncx/nxc) and single-inner-block
//     (nChw4c/8c/16c, incl. a padded channel tail with a zero-preserving alg)
//     dst, plus src0/src1 different plain layouts (nchw:nhwc, strided load)
//   - per-tensor src0/src1 scales
//   - a post-op chain: same-parameter sums at any positions (applied
//     in-kernel) plus any number of eltwise ops (incl. log/soft_relu/gelu_erf,
//     which fit the available aux budget) and binary ops
//     (f32/f16/bf16/s32/s8/u8 rhs; scalar / per_element / per_oc /
//     per_oc_spatial / per_w broadcast),
//     including the select (ternary) binary post-op with a dst-shaped
//     condition
//   - ternary select (dst = src2 ? src0 : src1) folded into the same f32
//     path, so it also supports broadcast, scales, and post-ops; src2 is s8
// A single class handles every case internally (RVV has one vector isa); like
// x64 a second tail-kernel instance covers the blocked channel tail.
struct jit_uni_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_binary_t);

        status_t init(const engine_t *engine);

        jit_binary_conf_t get_conf() const { return conf_; }

    private:
        op_t get_op_type(const memory_desc_wrapper &src0_d);
        bool is_only_dim0_bcasted(const dims_t &bcast_dims, const int ndims);
        bcast_t get_bcast_type(
                const memory_desc_wrapper &src1_d, const dims_t &bcast_dims);

        // alg_preserves_zero returns true if operation preserves zero in case
        // of both inputs contain zero.
        bool alg_preserves_zero() const;
        bool check_scales_mask() const;
        bool is_bcast_pattern(const dims_t &bcast_dims, const dim_t ndims,
                const dim_t N_bcast, const dim_t C_bcast,
                const dim_t W_bcast) const;
        bool is_bcast_allowed(const int ndims) const;
        bool is_format_non_blocked(const memory_desc_wrapper &mdw) const;
        bool is_different_layouts_allowed(const memory_desc_wrapper &src0_d,
                const memory_desc_wrapper &src1_d) const;
        bool is_applicable();
        bool init_generic_conf();

        jit_binary_conf_t conf_;
    };

    jit_uni_binary_t(const pd_t *apd);

    ~jit_uni_binary_t() override = default;

    status_t init(engine_t *engine) override;

    using data_t = int8_t;

    // Same driver-strategy split as x64/aarch64, chosen by op_type x bcast_type.
    // The scale pointers x64 forwards to the kernel are dereferenced up front on
    // rv64 and passed by value. Under blocked_oc_tail the last channel block is
    // dispatched to kernel_tail_ (x64 shape), which stores the valid lanes of
    // each block and zeroes the padded tail in-kernel.
    void execute_no_bcast_strategy(const data_t *src0, const data_t *src1,
            const data_t *src2, data_t *dst, float scale0, float scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const bcast_t bcast_type) const;
    void execute_bcast_per_batch_strategy(const data_t *src0,
            const data_t *src1, const data_t *src2, data_t *dst, float scale0,
            float scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec) const;
    void execute_bcast_per_c_strategy(const data_t *src0, const data_t *src1,
            const data_t *src2, data_t *dst, float scale0, float scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const op_t op_type, const bcast_t bcast_type,
            const bool blocked_oc_tail) const;
    void execute_bcast_per_w_strategy(const data_t *src0, const data_t *src1,
            const data_t *src2, data_t *dst, float scale0, float scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const op_t op_type, const bool blocked_oc_tail) const;
    void execute_generic_strategy(const data_t *src0, const data_t *src1,
            const data_t *src2, data_t *dst, float scale0, float scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec) const;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    static bool post_ops_ok(const primitive_attr_t *attr,
            const memory_desc_wrapper &src0_d, const memory_desc_wrapper &dst_d,
            const bool is_src_different_layouts, const cpu_isa_t isa,
            const bool use_generic_strategy);

    std::unique_ptr<binary_kernel_t> kernel_;
    // used only in bcast_c_blocked strategy if tail exists (the plain-shape
    // element tail needs no second kernel: vsetvli subsumes it)
    std::unique_ptr<binary_kernel_t> kernel_tail_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
