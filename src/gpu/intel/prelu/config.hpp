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

#ifndef GPU_INTEL_PRELU_CONFIG_HPP
#define GPU_INTEL_PRELU_CONFIG_HPP

#include "common/c_types_map.hpp"
#include "gpu/gpu_prelu_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace prelu {

using pd_t = prelu_pd_t;
using fwd_pd_t = gpu_prelu_fwd_pd_t;
using bwd_pd_t = gpu_prelu_bwd_pd_t;

struct conf_t {
    bool is_forward;
    bool reduce_diff_weights;
    bool require_stateless_addressing;
    compute::dispatch_t dispatch;

    attr_info_t attr_info;
    memory_desc_info_t src_md_info;
    memory_desc_info_t wei_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t diff_src_md_info;
    memory_desc_info_t diff_wei_md_info;
};

} // namespace prelu
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
