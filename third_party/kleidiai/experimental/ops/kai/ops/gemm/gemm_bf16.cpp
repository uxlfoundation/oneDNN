//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_pretransposed.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#ifdef __aarch64__
#include "kernels/a64_ffhybrid_bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_ffinterleaved_bf16fp32_dot_8x12.hpp"
#include "kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_hybrid_bf16fp32_dot_6x16.hpp"
#include "kernels/a64_hybrid_bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_bf16fp32_dot_8x12.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/sme2_gemv_bf16fp32_dot_16VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL.hpp"
#include "kernels/sme_gemv_bf16fp32_dot_8VL.hpp"
#include "kernels/sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL.hpp"
#include "kernels/sve_ffhybrid_bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_ffinterleaved_bf16fp32_dot_8x3VL.hpp"
#include "kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL.hpp"
#include "kernels/sve_hybrid_bf16fp32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_8x3VL.hpp"
#endif // __aarch64__
#ifdef __arm__
#include "kernels/a32_sgemm_8x6.hpp"
#endif // __arm__

namespace kai {
namespace ops {

static const GemmImplementation<bfloat16, bfloat16, float> gemm_bf16_methods[] =
{
#ifdef __aarch64__
{
    "sme2_gemv_bf16fp32_dot_16VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_bf16fp32_dot_16VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme_gemv_bf16fp32_dot_8VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme_gemv_bf16fp32_dot_8VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL, bfloat16, bfloat16, float>(args); }
},
{
    "sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_b16f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL, bfloat16, bfloat16, float>(args); }
},
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_interleaved_bf16fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_hybrid_bf16fp32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_mmla_6x4VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_hybrid_bf16fp32_dot_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_interleaved_bf16fp32_dot_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svebf16() && (args._Ksize>2); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
    KernelWeightFormat::VL2VL_BL64,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_ffinterleaved_bf16fp32_dot_8x3VL",
    KernelWeightFormat::VL1VL_BL32,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_dot_8x3VL, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "sve_ffhybrid_bf16fp32_mmla_6x4VL",
    KernelWeightFormat::VL2VL_BL64,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_bf16fp32_mmla_6x4VL, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_hybrid_bf16fp32_mmla_6x16",
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_mmla_6x16, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_interleaved_bf16fp32_mmla_8x12",
    [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_hybrid_bf16fp32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_interleaved_bf16fp32_dot_8x12",
    [](const GemmArgs &args) { return args._ci->has_bf16() && (args._Ksize>2); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_ffinterleaved_bf16fp32_mmla_8x12",
    KernelWeightFormat::VL256_BL64,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_ffhybrid_bf16fp32_mmla_6x16",
    KernelWeightFormat::VL256_BL64,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_bf16fp32_mmla_6x16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_ffinterleaved_bf16fp32_dot_8x12",
    KernelWeightFormat::VL128_BL32,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_dot_8x12, bfloat16, bfloat16, float>(args); }
),
GemmImplementation<bfloat16, bfloat16, float>::with_estimate(
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, bfloat16, float>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, bfloat16, float>(args); }
),
#endif // __aarch64__
#ifdef __arm__
{
    "a32_sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a32_sgemm_8x6, bfloat16, bfloat16, float>(args); }
},
#endif // __arm__
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<bfloat16, bfloat16, float> *gemm_implementation_list<bfloat16, bfloat16, float>() {
    return gemm_bf16_methods;
}

template UniqueGemmCommon<bfloat16, bfloat16, float> gemm<bfloat16, bfloat16, float, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<bfloat16, bfloat16, float, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<bfloat16, bfloat16, float, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai
