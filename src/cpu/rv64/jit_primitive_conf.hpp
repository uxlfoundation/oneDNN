/*******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_JIT_PRIMITIVE_CONF_HPP
#define CPU_RV64_JIT_PRIMITIVE_CONF_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Memory layout family handled by the pooling kernel. Both are plain (no
// blocked tags): nspc vectorizes along C, ncsp vectorizes along OW.
enum class jit_pool_tag_kind_t { undef, nspc, ncsp };

// Centralized pooling configuration, populated by
// jit_uni_pool_kernel_t::init_conf and consumed by the driver and the kernel.
// Mirrors the role of aarch64/x64 jit_pool_conf_t (kept lean: no c_block /
// blocked-format / transpose fields, which the RVV path does not use).
struct jit_pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad;
    alg_kind_t alg;
    bool is_backward;
    data_type_t src_dt;
    data_type_t dst_dt;
    int dt_size;
    jit_pool_tag_kind_t tag_kind;
    cpu_isa_t isa;
    bool with_postops;
    bool with_relu; // f32 single-ReLU flag; feeds empty_window_value() only.
            // The post-op chain is fused via fuse_eltwise/fuse_binary.
    bool fuse_eltwise; // eltwise post-op chain fused in-kernel via the injector
            // (f32 and f16 eltwise-only)
    bool fuse_binary; // f32 binary post-op fused in-kernel (channel-vec path)
    float relu_alpha;
    post_ops_t post_ops;
    int nthr;
    // Output-width unroll for the shape-baked interior kernel (number of OW
    // positions whose accumulators share loaded inputs, ARM max_step style).
    int ur_w;
};

// Per-call arguments for the agnostic (runtime-shape) kernel. Unlike aarch64/x64
// — which bake the pooling window into the generated code — this kernel is
// shape-agnostic and receives the window bounds and strides here, so one routine
// per instantiation serves ncsp, OW==1, the nspc boundary columns, and (for f16)
// the whole nspc row. (The f32 nspc interior uses the shape-baked kernel with
// jit_uni_pool_interior_args_t.)
struct jit_uni_pooling_args_t {
    const void *src;
    void *dst;
    dim_t channels;
    dim_t id_start, ih_start, iw_start;
    dim_t id_end, ih_end, iw_end;
    dim_t inW_stride; // IW * C (elements) for nspc, IW for ncsp
    dim_t inD_stride; // IH * IW * C (elements) for nspc, IH*IW for ncsp
    dim_t w_spatial_byte_stride; // C*dt_size for nspc, 1*dt_size for ncsp
    float init_val; // -FLT_MAX (max) / 0 (avg); the f16 kernel bakes its own
    float scale_val; // 1.0/count for avg, unused for max
    float relu_alpha;
    bool with_relu;
    dim_t src_vec_byte_stride; // byte stride between vector elements in src
    dim_t dst_vec_byte_stride; // byte stride between vector elements in dst
    // Binary post-op rhs (src1) base for this call. For full-dst per-element it
    // points at the element matching this position's channel 0; for per-oc /
    // per-tensor it points at the channel-0 / scalar value. Unused otherwise.
    const void *post_op_rhs = nullptr;
    // f16 generic path row unrolling: process n_pos output positions in one
    // call (adjacent positions share the window, so only the base pointers
    // advance). The f32 agnostic kernel ignores these fields and always handles
    // a single output position; f32 nspc interior reuse is handled by
    // jit_uni_pool_interior_args_t instead.
    dim_t n_pos = 1;
    dim_t pos_src_byte_stride = 0;
    dim_t pos_dst_byte_stride = 0;
};

// Per-call arguments for the shape-baked interior kernel. The unroll structure
// (kw, stride_w, ur_w), the algorithm, init value, avg_include scale, and the
// fused eltwise post-op chain are baked into the generated code. Element strides
// are passed (not baked) so the kernel stays correct for tensors whose strides
// exceed 32 bits; the kernel derives the per-block strides from w_stride.
struct jit_uni_pool_interior_args_t {
    const void *src; // channel 0 of (id_start, ih_start, iw_block_start)
    void *dst; // channel 0 of the first output column in the block
    dim_t channels; // remaining channels (VLA tail handled by vsetvli)
    dim_t kh_count; // valid kernel rows  = ih_end - ih_start
    dim_t kd_count; // valid kernel planes = id_end - id_start
    dim_t n_blocks; // number of ur_w output blocks to process in this call
    dim_t w_stride; // byte stride between W positions (= C * dt_size)
    dim_t inW_stride; // byte stride between H rows (= iw * C * dt_size)
    dim_t inD_stride; // byte stride between D planes (= ih * iw * C * dt_size)
    float scale_val; // 1 / window for avg_exclude (ignored otherwise)
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int iw, ih, id;
    int ow, oh, od;
    int os, is;
    int kw, kh, kd;
    int stride_w, stride_h, stride_d;
    int t_pad, l_pad, f_pad;

    int ic_block, oc_block;
    int load_block, reduce_block;
    int bcast_block;

    dim_t load_dim, bcast_dim, reduce_dim;

    int simd_w;
    int ur, ur_tail;
    int load_loop_blk;
    int reduce_loop_unroll;
    int nthr;
    int nb_bcast, nb_load, nb_reduce, load_grp_count;
    int nb_load_blocking, nb_load_blocking_max;
    int nb_bcast_blocking, nb_bcast_blocking_max;
    int nb_reduce_blocking;

    dim_t reduce_loop_bcast_step;
    int reduce_loop_load_step;
    int bcast_loop_bcast_step;
    int bcast_loop_output_step;
    int load_loop_load_step;
    int load_loop_iter_step;

    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_dw_conv;

    // src/weights dtype: f32, bf16 (Zvfbfwma), or f16 (Zvfh).
    data_type_t src_dt;

    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;

    format_tag_t src_tag, wei_tag, dst_tag;
};

struct jit_1x1_conv_args_t {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t first_last_flag;
};

enum {
    FLAG_REDUCE_FIRST = 1 << 0,
    FLAG_REDUCE_LAST = 1 << 1,
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
