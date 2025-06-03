/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_INTERNAL_ATTRS_HPP
#define GRAPH_BACKEND_DNNL_INTERNAL_ATTRS_HPP

#include <string>

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace op_attr {

using namespace dnnl::impl::graph::op_attr;

// internal attributes: bool
const op_attr_t canonicalized = 0x10000;
const op_attr_t change_layout = 0x10001;
const op_attr_t is_constant = 0x10002;
const op_attr_t is_convtranspose = 0x10003;
const op_attr_t is_training = 0x10004;
const op_attr_t fwd_alg_kind = 0x10005;
const op_attr_t fuse_relu = 0x10006;
const op_attr_t with_bias = 0x10007;
const op_attr_t with_runtime_scales = 0x10009;
const op_attr_t with_runtime_zps = 0x10000a;
const op_attr_t with_runtime_src_zps = 0x1000b;
const op_attr_t with_runtime_dst_zps = 0x1000c;
const op_attr_t is_bias_add = 0x1000d;
const op_attr_t with_sum = 0x1000e;
const op_attr_t keep_dst_layout = 0x1000f;
const op_attr_t with_scale = 0x10010;
const op_attr_t is_invert_scale = 0x10011;
const op_attr_t mask_type = 0x10012;
const op_attr_t with_q_scale = 0x10013;
const op_attr_t with_q_zp = 0x10014;
const op_attr_t with_k_scale = 0x10015;
const op_attr_t with_k_zp = 0x10016;
const op_attr_t with_v_scale = 0x10017;
const op_attr_t with_v_zp = 0x10018;
const op_attr_t with_a_scale = 0x1001a;
const op_attr_t with_a_zp = 0x1001b;
const op_attr_t with_o_scale = 0x1001c;
const op_attr_t with_o_zp = 0x1001d;


// int64_t
const op_attr_t alg_kind = 0x10100;
const op_attr_t fusion_info_key = 0x10103;
const op_attr_t group_mask = 0x10104;
const op_attr_t data_type = 0x10105;
const op_attr_t axis_row = 0x10106;
const op_attr_t axis_col = 0x10107;
const op_attr_t fusion_info_keys = 0x10108; // used for sdpa
const op_attr_t q_mask = 0x10109;
const op_attr_t k_mask = 0x1010a;
const op_attr_t v_mask = 0x1010b;
const op_attr_t a_mask = 0x1010c;
const op_attr_t o_mask = 0x1010d;


// string
const op_attr_t dw_type = 0x10201;
const op_attr_t kind = 0x10204;

// float
const op_attr_t p = 0x10300;

// vector of int64_t
const op_attr_t dst_zps = 0x10400;
const op_attr_t src_zps = 0x10401;
const op_attr_t permutation = 0x10402;
const op_attr_t k_group_shape = 0x10403;
const op_attr_t v_group_shape = 0x10404;

static inline std::string internal_attr2str(op_attr_t attr) {
#define CASE(a) \
    case (a): return #a

    switch (attr) {
        CASE(canonicalized);
        CASE(change_layout);
        CASE(is_constant);
        CASE(is_convtranspose);
        CASE(is_training);
        CASE(fwd_alg_kind);
        CASE(fuse_relu);
        CASE(with_bias);
        CASE(with_runtime_scales);
        CASE(with_runtime_zps);
        CASE(with_runtime_src_zps);
        CASE(with_runtime_dst_zps);
        CASE(is_bias_add);
        CASE(with_sum);
        CASE(keep_dst_layout);
        CASE(with_scale);
        CASE(is_invert_scale);
        CASE(mask_type);
        CASE(with_q_scale);
        CASE(with_q_zp);
        CASE(with_k_scale);
        CASE(with_k_zp);
        CASE(with_v_scale);
        CASE(with_v_zp);
        CASE(with_a_scale);
        CASE(with_a_zp);
        CASE(with_o_scale);
        CASE(with_o_zp);
        CASE(q_mask);
        CASE(k_mask);
        CASE(v_mask);
        CASE(a_mask);
        CASE(o_mask);
        CASE(alg_kind);
        CASE(fusion_info_key);
        CASE(axis_row);
        CASE(axis_col);
        CASE(fusion_info_keys);
        CASE(dw_type);
        CASE(kind);
        CASE(p);
        CASE(dst_zps);
        CASE(src_zps);
        CASE(permutation);
        CASE(k_group_shape);
        CASE(v_group_shape);
        default: return "undefined_attr";
    }
#undef CASE
}

} // namespace op_attr
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
