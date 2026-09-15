//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if defined(__aarch64__)

#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_ffhybrid_fp16fp32_mla_6x16.hpp"
#include "kernels/a64_hybrid_fp16fp32_mla_6x16.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.hpp"
#include "kernels/sve_ffhybrid_fp16fp32_mla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp16fp32_mla_6x4VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<__fp16, __fp16, float> gemm_fp16fp32_methods[] =
{
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL, __fp16, __fp16, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL, __fp16, __fp16, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL, __fp16, __fp16, float>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL, __fp16, __fp16, float>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL, __fp16, __fp16, float>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL, __fp16, __fp16, float>(args); }
},
{
    "sve_hybrid_fp16fp32_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16fp32_mla_6x4VL, __fp16, __fp16, float>(args); }
},
{
    "sve_ffhybrid_fp16fp32_mla_6x4VL",
    KernelWeightFormat::VL1VL_BL16,
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16fp32_mla_6x4VL, __fp16, float>(args); }
},
{
    "a64_hybrid_fp16fp32_mla_6x16",
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp16fp32_mla_6x16, __fp16, __fp16, float>(args); }
},
{
    "a64_ffhybrid_fp16fp32_mla_6x16",
    KernelWeightFormat::VL128_BL16,
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16fp32_mla_6x16, __fp16, float>(args); }
},
{
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return !args._ci->has_fp16(); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, __fp16, __fp16, float>(args); }
},
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<__fp16, __fp16, float> *gemm_implementation_list<__fp16, __fp16, float>() {
    return gemm_fp16fp32_methods;
}

template UniqueGemmCommon<__fp16, __fp16, float> gemm<__fp16, __fp16, float, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<__fp16, __fp16, float, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16, __fp16, float, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__)
