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

#ifndef GPU_INTEL_SHUFFLE_CONFIG_HPP
#define GPU_INTEL_SHUFFLE_CONFIG_HPP

#include "gpu/gpu_shuffle_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace shuffle {

using pd_t = gpu_shuffle_pd_t;

struct conf_t {
    data_type_t data_type;
    dim_idx_t axis;
    dim_t transpose_row;
    dim_t transpose_col;
    compute::dispatch_t dispatch;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

} // namespace shuffle
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
