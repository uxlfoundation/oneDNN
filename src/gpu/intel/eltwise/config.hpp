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

#ifndef GPU_INTEL_ELTWISE_CONFIG_HPP
#define GPU_INTEL_ELTWISE_CONFIG_HPP

#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace eltwise {

using pd_t = eltwise_pd_t;
using fwd_pd_t = gpu_eltwise_fwd_pd_t;
using bwd_pd_t = gpu_eltwise_bwd_pd_t;

struct conf_t {
    int ndims;
    int vector_size;
    bool with_zero_padding;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    bool require_stateless_addressing;
    int work_group_size;
    int sub_group_size;
    compute::dispatch_t dispatch;
    memory_desc_info_t data_md_info;
    memory_desc_info_t data_diff_md_info;

    attr_info_t attr_info;
};

} // namespace eltwise
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
