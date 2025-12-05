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

#ifndef GPU_INTEL_IP_CONFIG_HPP
#define GPU_INTEL_IP_CONFIG_HPP

#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ip {

using pd_t = inner_product_pd_t;
using fwd_pd_t = gpu_inner_product_fwd_pd_t;
using bwd_data_pd_t = gpu_inner_product_bwd_data_pd_t;
using bwd_weights_pd_t = gpu_inner_product_bwd_weights_pd_t;
using desc_t = inner_product_desc_t;

// Inner Product
struct conf_t {
    dim_idx_t ndims;
    dim_idx_t src_ndims, wei_ndims, dst_ndims;
    dim_t mb, oc, ic, ic_total;
    dim_t id, ih, iw, od, oh, ow;
    dim_t kd, kh, kw;
    bool with_bias, has_spatial;
    bool is_forward, is_backward_data, is_backward_weights;
    compute::dispatch_t dispatch;
    bool reorder_dst = false;
    bool require_stateless_addressing;

    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;

    attr_info_t attr_info;
};

} // namespace ip
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
