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

#include <string>
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

enum class sdpa_property : int {
    none = 0x0,
    second_token = 0x1,
    quantized = 0x2,
    integrated = 0x4,
    fma = 0x8,
    f32 = 0x10,
};

sdpa_property operator|(sdpa_property a, sdpa_property b);
sdpa_property operator&(sdpa_property a, sdpa_property b);
sdpa_property operator^(sdpa_property a, sdpa_property b);
sdpa_property &operator|=(sdpa_property &a, sdpa_property b);
sdpa_property &operator&=(sdpa_property &a, sdpa_property b);
sdpa_property &operator^=(sdpa_property &a, sdpa_property b);

struct config_query_t {
    static constexpr int any = -1;
    compute::gpu_arch_t arch;
    int head_size;
    int seq_len = any;
    sdpa_property property = sdpa_property::none;

    config_query_t(compute::gpu_arch_t arch_, int head_size_,
            int seq_len_ = any, sdpa_property property_ = sdpa_property::none)
        : arch(arch_)
        , head_size(head_size_)
        , seq_len(seq_len_)
        , property(property_) {}
};

struct config_criteria_t {
    static constexpr int any = -1;
    compute::gpu_arch_t arch;
    int head_size;
    int seq_len = any;
    sdpa_property property = sdpa_property::none;
    config_criteria_t(compute::gpu_arch_t a, int hs)
        : arch(a), head_size(hs), seq_len(any), property(sdpa_property::none) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, int sq)
        : arch(a), head_size(hs), seq_len(sq), property(sdpa_property::none) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, sdpa_property prop)
        : arch(a), head_size(hs), seq_len(any), property(prop) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, int sq, sdpa_property prop)
        : arch(a), head_size(hs), seq_len(sq), property(prop) {}
};

struct config_record_t {
    config_criteria_t criteria;
    sdpa_config_t config;
};

std::ostream &operator<<(std::ostream &s, const config_query_t &q);
std::ostream &operator<<(std::ostream &s, const config_criteria_t &c);
std::ostream &operator<<(std::ostream &s, const sdpa_config_t &c);

bool operator==(const config_record_t &key, const config_query_t &query);
bool operator<(const config_criteria_t &lhs, const config_criteria_t &rhs);
bool operator<(const config_record_t &lhs, const config_record_t &rhs);

sdpa_config_t *choose_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32);
dim_t round_up_seq_interval(dim_t seq, compute::gpu_arch_t arch);

dim_t nearest_conf_seq_interval(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32);

// serializable options for microkernel configuration
// follows reduced subset of structs from gemmstone that
// are used for ukernel generation
struct ukernel_serialized_opts_t
    : trivially_serializable_t<ukernel_serialized_opts_t> {

    ukernel_serialized_opts_t() = default;
    ukernel_serialized_opts_t(micro::GEMMProtocol::Options opts)
        : localB(opts.localB)
        , slmPtr(opts.slmPtr)
        , scaleA(opts.scaleA)
        , offsetA(opts.offsetA) {}
    bool localB, slmPtr, scaleA, offsetA;
    uint8_t padding[4] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_opts_t);
static_assert(sizeof(ukernel_serialized_opts_t) == 8,
        "Expected sizeof(ukernel_serialized_opts_t) == 8");

struct ukernel_serialized_hwinfo_t
    : trivially_serializable_t<ukernel_serialized_hwinfo_t> {

    ukernel_serialized_hwinfo_t() = default;
    ukernel_serialized_hwinfo_t(gemmstone::HWInformation &hwInfo)
        : gmdid(static_cast<uint32_t>(hwInfo.gmdid))
        , euCount(static_cast<uint32_t>(hwInfo.euCount))
        , systolicAvailable(hwInfo.systolicAvailable) {}

    uint32_t gmdid;
    uint32_t euCount;
    bool systolicAvailable;
    uint8_t padding[7] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_hwinfo_t);
static_assert(sizeof(ukernel_serialized_hwinfo_t) == 16,
        "Expected sizeof(ukernel_serialized_hwinfo_t) == 16");

struct ukernel_serialized_sizes_t
    : trivially_serializable_t<ukernel_serialized_sizes_t> {

    ukernel_serialized_sizes_t() = default;
    ukernel_serialized_sizes_t(gemmstone::SizeParams &sizes)
        : batch(sizes.batch), m(sizes.m), n(sizes.n), k(sizes.k) {}
    int64_t batch = 0;
    int64_t m = 0, n = 0, k = 0;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_sizes_t);

struct ukernel_serialized_problem_t
    : trivially_serializable_t<ukernel_serialized_problem_t> {

    ukernel_serialized_problem_t() = default;
    ukernel_serialized_problem_t(gemmstone::GEMMProblem &problem)
        : Ta_ext(static_cast<uint32_t>(problem.Ta_ext))
        , Tb_ext(static_cast<uint32_t>(problem.Tb_ext))
        , Tc_ext(static_cast<uint32_t>(problem.Tc_ext))
        , Ta(static_cast<uint32_t>(problem.Ta))
        , Tb(static_cast<uint32_t>(problem.Tb))
        , Tc(static_cast<uint32_t>(problem.Tc))
        , Ts(static_cast<uint32_t>(problem.Ts))
        , A_layout(static_cast<int>(problem.A.layout))
        , B_layout(static_cast<int>(problem.B.layout))
        , C_layout(static_cast<int>(problem.C.layout))
        , B_tileR(problem.B.tileR)
        , B_tileC(problem.B.tileC)
        , B_crosspack(problem.B.crosspack)
        , A_alignment(problem.A.alignment)
        , A_scale_alignment(problem.A_scale.alignment)
        , AO_alignment(problem.AO.alignment)
        , B_alignment(problem.B.alignment)
        , asPtrDims(problem.asPtrDims)
        , aOffset(static_cast<int>(problem.aOffset))
        , Ta_scale(static_cast<uint32_t>(problem.Ta_scale))
        , A_scale_layout(static_cast<int>(problem.A_scale.layout))
        , Tao(static_cast<uint32_t>(problem.Tao))
        , AO_layout(static_cast<int>(problem.AO.layout))
        , aoPtrDims(problem.aoPtrDims)
        , aqGroupM(problem.aqGroupM)
        , aqGroupK(problem.aqGroupK) {}

    uint32_t Ta_ext, Tb_ext, Tc_ext;
    uint32_t Ta, Tb, Tc, Ts;

    int A_layout;
    int B_layout;
    int C_layout;

    uint16_t B_tileR;
    uint16_t B_tileC;
    uint8_t B_crosspack;

    uint8_t A_alignment;
    uint8_t A_scale_alignment;
    uint8_t AO_alignment;
    uint8_t B_alignment;
    // trivially serializable classes require alignment to 8-byte boundaries
    // padding0 bumps class size from 49->56 bytes so uint8_t arguments
    // related to alignment can be grouped together rather than placed at the end of the struct
    uint8_t padding0[7] = {0};

    int asPtrDims;
    int aOffset;

    uint32_t Ta_scale;
    int A_scale_layout;
    uint32_t Tao;

    int AO_layout;
    int aoPtrDims;
    int aqGroupM;
    int aqGroupK;

    bool with_scales, with_zp, with_common_scales;
    uint8_t padding1[1] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_problem_t);
static_assert(sizeof(ukernel_serialized_problem_t) == 96,
        "Expected sizeof(ukernel_serialized_problem_t) == 96");

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
