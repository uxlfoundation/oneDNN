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
#include "gemmstone/microkernel_provider.hpp"

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

dim_t round_up_seq_interval(dim_t seq, compute::gpu_arch_t arch);

// serializable options for microkernel configuration
// follows reduced subset of structs from gemmstone that
// are used for ukernel generation
struct ukernel_serialized_opts_t
    : trivially_serializable_t<ukernel_serialized_opts_t> {
    bool localB, slmPtr, scaleA, offsetA;
    uint8_t padding[4] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_opts_t);

struct ukernel_serialized_hwinfo_t
    : trivially_serializable_t<ukernel_serialized_hwinfo_t> {
    uint32_t gmdid;
    uint32_t euCount;
    bool systolicAvailable;
    uint8_t padding[7] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_hwinfo_t);

struct ukernel_serialized_sizes_t
    : trivially_serializable_t<ukernel_serialized_sizes_t> {
    int64_t batch = 0;
    int64_t m = 0, n = 0, k = 0;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_sizes_t);

struct ukernel_serialized_problem_t
    : trivially_serializable_t<ukernel_serialized_problem_t> {
    uint32_t Ta_ext, Tb_ext, Tc_ext;
    uint32_t Ta, Tb, Tc, Ts;

    int A_layout;
    // default values set for optional quantization variables
    // to avoid missing hash due to uninitialized values
    uint32_t Ta_scale = static_cast<uint32_t>(gemmstone::Type::invalid);
    uint32_t Tao = static_cast<uint32_t>(gemmstone::Type::invalid);
    int A_scale_layout = static_cast<int>(gemmstone::MatrixLayout::N);

    int AO_layout = static_cast<int>(gemmstone::MatrixLayout::N);
    int aoPtrDims = 0;
    int aqGroupM = 1;
    int aqGroupK = 1;

    int B_layout;
    int C_layout;

    uint16_t B_tileR;
    uint16_t B_tileC;
    uint8_t B_crosspack;

    uint8_t A_alignment;
    uint8_t A_scale_alignment = 0;
    uint8_t AO_alignment = 0;
    uint8_t B_alignment;

    bool with_scales, with_zp, with_common_scales;
    uint8_t padding[4] = {0};
    int asPtrDims = -1;
    int aOffset;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_problem_t);

struct micro_sdpa_ukernel_params_t
    : trivially_serializable_t<micro_sdpa_ukernel_params_t> {
    int unroll_m_kq, unroll_n_kq;
    int unroll_m_vs, unroll_n_vs;
    int wg_m_kq, wg_n_kq;
    int wg_m_vs, wg_n_vs;

    ukernel_serialized_hwinfo_t hwinfo;
    ukernel_serialized_problem_t problem_kq;
    ukernel_serialized_problem_t problem_vs;
    ukernel_serialized_opts_t opts_kq;
    ukernel_serialized_opts_t opts_vs;
    ukernel_serialized_sizes_t sizes_kq, sizes_vs;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_sdpa_ukernel_params_t);

void deserialize_config_to_gemmstone(gemmstone::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        micro::GEMMProtocol::Options &opts_kq,
        micro::GEMMProtocol::Options &opts_vs, gemmstone::SizeParams &sizes_kq,
        gemmstone::SizeParams &sizes_vs,
        const micro_sdpa_ukernel_params_t &ukernel_config);

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
