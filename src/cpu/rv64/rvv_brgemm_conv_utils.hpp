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

#ifndef CPU_RV64_RVV_BRGEMM_CONV_UTILS_HPP
#define CPU_RV64_RVV_BRGEMM_CONV_UTILS_HPP

#include "common/c_types_map.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// BRGEMM convolution configuration
struct brgemm_conv_conf_t {
    cpu_isa_t isa;
    prop_kind_t prop_kind;
    int ndims;
    int mb;
    int ngroups, ic, oc; // IC/OC per group
    int id, ih, iw, od, oh, ow;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    format_tag_t src_tag, wei_tag, dst_tag;
    bool with_bias;
    bool with_sum;
    data_type_t src_dt, dst_dt, wei_dt, bia_dt;
    int nthr;
};

namespace brgemm_convolution_utils {

status_t init_conf(brgemm_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

} // namespace brgemm_convolution_utils

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
