/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

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
