//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if defined(__aarch64__)

#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_pretransposed.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_ffhybrid_fp16_mla_6x32.hpp"
#include "kernels/a64_ffhybrid_fp16fp32fp16_mla_6x16.hpp"
#include "kernels/a64_ffinterleaved_fp16_mla_8x24.hpp"
#include "kernels/a64_hgemm_8x24.hpp"
#include "kernels/a64_hybrid_fp16_mla_6x32.hpp"
#include "kernels/a64_hybrid_fp16fp32fp16_mla_6x16.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/sme2_gemv_fp16_mla_16VL.hpp"
#include "kernels/sme2_gemv_fp16fp32fp16_dot_16VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16_mopa_1VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16_mopa_2VLx1VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL.hpp"
#include "kernels/sme_gemv_fp16_mla_8VL.hpp"
#include "kernels/sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL.hpp"
#include "kernels/sve_ffhybrid_fp16_mla_6x4VL.hpp"
#include "kernels/sve_ffhybrid_fp16fp32fp16_mla_6x4VL.hpp"
#include "kernels/sve_ffinterleaved_fp16_mla_8x3VL.hpp"
#include "kernels/sve_hybrid_fp16_mla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp16fp32fp16_mla_6x4VL.hpp"
#include "kernels/sve_interleaved_fp16_mla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<__fp16, __fp16, __fp16> gemm_fp16_methods[] =
{
{
    "sme2_interleaved_nomerge_fp16_mopa_1VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2_f16f16() && args._fast_mode && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<__fp16>();
                               return args._Msize <= VL; },  // TODO Check this
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16_mopa_1VLx2VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_interleaved_nomerge_fp16_mopa_2VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2_f16f16() && args._fast_mode && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp16_mopa_2VLx1VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_gemv_fp16_mla_16VL",
    [](const GemmArgs &args) { return args._ci->has_sme2_f16f16() && args._fast_mode && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp16_mla_16VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_gemv_fp16fp32fp16_dot_16VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp16fp32fp16_dot_16VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme_gemv_fp16_mla_8VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._fast_mode && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme_gemv_fp16_mla_8VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f16f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL, __fp16, __fp16, __fp16>(args); }
},
{
    "sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f16f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMergeFloatAcc<cls_sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL, __fp16, __fp16, __fp16>(args); }
},
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "sve_hybrid_fp16_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16_mla_6x4VL, __fp16, __fp16, __fp16>(args); }
),
{
    "sve_hybrid_fp16fp32fp16_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    [](const GemmArgs &args) { return !args._fast_mode; },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp16fp32fp16_mla_6x4VL, __fp16, __fp16, __fp16>(args); }
},
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "sve_interleaved_fp16_mla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize > 4) && args._fast_mode; },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "sve_ffinterleaved_fp16_mla_8x3VL",
    KernelWeightFormat::VL1VL_BL16,
    [](const GemmArgs &args) { return args._ci->has_sve() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp16_mla_8x3VL, __fp16, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "sve_ffhybrid_fp16_mla_6x4VL",
    KernelWeightFormat::VL1VL_BL16,
    [](const GemmArgs &args) { return args._ci->has_sve() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16_mla_6x4VL, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16_mla_6x4VL, __fp16, __fp16>(args); }
),
{
    "sve_ffhybrid_fp16fp32fp16_mla_6x4VL",
    KernelWeightFormat::VL1VL_BL16,
    [](const GemmArgs &args) { return args._ci->has_sve2(); },
    [](const GemmArgs &args) { return !args._fast_mode; },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp16fp32fp16_mla_6x4VL, __fp16, __fp16>(args); }
},
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "a64_hybrid_fp16_mla_6x32",
    [](const GemmArgs &args) { return args._ci->has_fp16() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp16_mla_6x32, __fp16, __fp16, __fp16>(args); }
),
{
    "a64_hybrid_fp16fp32fp16_mla_6x16",
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    [](const GemmArgs &args) { return !args._fast_mode; },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp16fp32fp16_mla_6x16, __fp16, __fp16, __fp16>(args); }
},
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "a64_hgemm_8x24",
    [](const GemmArgs &args) { return args._ci->has_fp16() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_hgemm_8x24, __fp16, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "a64_ffinterleaved_fp16_mla_8x24",
    KernelWeightFormat::VL128_BL16,
    [](const GemmArgs &args) { return args._ci->has_fp16() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp16_mla_8x24, __fp16, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp16_mla_8x24, __fp16, __fp16, __fp16>(args); }
),
GemmImplementation<__fp16, __fp16, __fp16>::with_estimate(
    "a64_ffhybrid_fp16_mla_6x32",
    KernelWeightFormat::VL128_BL16,
    [](const GemmArgs &args) { return args._ci->has_fp16() && args._fast_mode; },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16_mla_6x32, __fp16, __fp16>::estimate_cycles<__fp16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16_mla_6x32, __fp16, __fp16>(args); }
),
{
    "a64_ffhybrid_fp16fp32fp16_mla_6x16",
    KernelWeightFormat::VL128_BL16,
    [](const GemmArgs &args) { return args._ci->has_fhm(); },
    [](const GemmArgs &args) { return !args._fast_mode; },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp16fp32fp16_mla_6x16, __fp16, __fp16>(args); }
},
{
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return !args._ci->has_fp16(); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, __fp16, __fp16, __fp16>(args); }
},
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<__fp16, __fp16, __fp16> *gemm_implementation_list<__fp16, __fp16, __fp16>() {
    return gemm_fp16_methods;
}

template UniqueGemmCommon<__fp16, __fp16, __fp16> gemm<__fp16, __fp16, __fp16, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<__fp16, __fp16, __fp16, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16, __fp16, __fp16, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__)
