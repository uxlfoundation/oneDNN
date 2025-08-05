/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_BINARY_CONFIG_HPP
#define GPU_INTEL_BINARY_CONFIG_HPP

#include "gpu/gpu_binary_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace binary {

using pd_t = gpu_binary_pd_t;

struct conf_t {
    int ndims, nvect;
    bool use_unroll_16b, src0_unroll_16b;
    bool is_plain_layout;
    bool plain_to_ABcd4a4b;
    bool isXa16b;
    data_type_t src0_data_type;
    data_type_t src1_data_type;
    data_type_t src2_data_type;
    data_type_t dst_data_type;
    alg_kind_t alg;
    // bool is_ne;
    bool is_tensor_op;
    compute::dispatch_t dispatch;
    int mb_block;
    int has_tail;
    int dim0[MAX_NDIMS];
    int src0_bcast_dims[MAX_NDIMS];
    int src1_bcast_dims[MAX_NDIMS];
    int src2_bcast_dims[MAX_NDIMS];
    bool is_dense;
    bool is_same_md;
    bool same_src_dt;
    bool with_binary_post_op;
    bool is_src1_broadcast;
    bool is_src0_blocked;

    memory_desc_info_t src0_md_info;
    memory_desc_info_t src1_md_info;
    memory_desc_info_t src2_md_info;
    memory_desc_info_t dst_md_info;

    attr_info_t attr_info;
};

} // namespace binary
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
