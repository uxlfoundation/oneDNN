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

#ifndef GPU_INTEL_OCL_MICRO_SDPA_CONFIGS_HPP
#define GPU_INTEL_OCL_MICRO_SDPA_CONFIGS_HPP

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq; // Subgroup tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs; // Subgroup tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq; // Workgroup configuration for K*Q GEMM
    int wg_m_vs, wg_n_vs; // Workgroup configuration for V*S GEMM
};

sdpa_config_t *choose_config_xehpg_fma(
        dim_t head_size, dim_t seq, bool thin_q, bool quantized);

sdpa_config_t *choose_config_xehpg(
        dim_t head_size, dim_t seq, bool thin_q, bool quantized);

sdpa_config_t *choose_config_xehpc(dim_t head_size, dim_t seq, bool thin_q,
        bool quantized, bool is_integrated);

sdpa_config_t *choose_config_xe2(dim_t head_size, dim_t seq, bool thin_q,
        bool quantized, bool is_integrated);

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
