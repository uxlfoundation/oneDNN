//
// SPDX-FileCopyrightText: Copyright 2017-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_batched.hpp"
#include "gemv_pretransposed.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#ifdef __aarch64__
#include "kernels/a64_ffhybrid_fp32_mla_6x16.hpp"
#include "kernels/a64_ffhybrid_fp32bf16fp32_mmla_4x24.hpp"
#include "kernels/a64_ffhybrid_fp32bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_ffinterleaved_fp32_mla_8x12.hpp"
#include "kernels/a64_hybrid_fp32_mla_4x24.hpp"
#include "kernels/a64_hybrid_fp32_mla_6x16.hpp"
#include "kernels/a64_hybrid_fp32_mla_8x4.hpp"
#include "kernels/a64_hybrid_fp32bf16fp32_mmla_4x24.hpp"
#include "kernels/a64_hybrid_fp32bf16fp32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/a64_sgemm_8x6.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_6x4.hpp"
#include "kernels/a64_smallK_hybrid_fp32_mla_8x4.hpp"
#include "kernels/sme2_gemv_fp32_mla_16VL.hpp"
#include "kernels/sme2_gemv_fp32bf16fp32_dot_16VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_fp32_mopa_4VLx1VL.hpp"
#include "kernels/sme_gemv_fp32_mla_8VL.hpp"
#include "kernels/sme_gemv_fp32bf16fp32_dot_8VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_fp32_mopa_4VLx1VL.hpp"
#include "kernels/sve_ffhybrid_fp32_mla_6x4VL.hpp"
#include "kernels/sve_ffhybrid_fp32bf16fp32_mmla_4x6VL.hpp"
#include "kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL.hpp"
#include "kernels/sve_ffinterleaved_fp32_mla_8x3VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_6x4VL.hpp"
#include "kernels/sve_hybrid_fp32_mla_8x1VL.hpp"
#include "kernels/sve_hybrid_fp32bf16fp32_mmla_4x6VL.hpp"
#include "kernels/sve_hybrid_fp32bf16fp32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_8x3VL.hpp"
#include "kernels/sve_interleaved_fp32_mla_8x3VL.hpp"
#include "kernels/sve_interleaved_fp32_mmla_8x3VL.hpp"
#endif // __aarch64__
#ifdef __arm__
#include "kernels/a32_sgemm_8x6.hpp"
#endif // __arm__

namespace kai {
namespace ops {

static const GemmImplementation<float, float, float> gemm_fp32_methods[] =
{
{
    "gemv_batched",
    [](const GemmArgs &args) { return args._Msize==1 && args._nbatches>1 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemvBatched<float, float>(args); }
},
#ifdef __aarch64__
GemmImplementation<float, float, float>::with_estimate(
    "a64_interleaved_bf16fp32_mmla_8x12",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_hybrid_fp32bf16fp32_mmla_6x16",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_6x16, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_hybrid_fp32bf16fp32_mmla_4x24",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32bf16fp32_mmla_4x24, float, float, float>(args); }
),
{
    "sme2_gemv_fp32bf16fp32_dot_16VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp32bf16fp32_dot_16VL, float, float, float>(args); }
},
{
    "sme_gemv_fp32bf16fp32_dot_8VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme_gemv_fp32bf16fp32_dot_8VL, float, float, float>(args); }
},
{
    "sme2_gemv_fp32_mla_16VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme2_gemv_fp32_mla_16VL, float, float, float>(args); }
},
{
    "sme_gemv_fp32_mla_8VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._Msize==1 && args._nbatches==1 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemvPretransposed<cls_sme_gemv_fp32_mla_8VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_1VLx4VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_4VLx1VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_sme2() && args._ci->has_sme_b16f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL, float, float, float>(args); }
},
{
    "sme2_interleaved_nomerge_fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_fp32_mopa_2VLx2VL, float, float, float>(args); }
},
{
    "sme_interleaved_nomerge_fp32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp32_mopa_1VLx4VL, float, float, float>(args); }
},
{
    "sme_interleaved_nomerge_fp32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<float>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp32_mopa_4VLx1VL, float, float, float>(args); }
},
{
    "sme_interleaved_nomerge_fp32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_f32f32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp32_mopa_2VLx2VL, float, float, float>(args); }
},
GemmImplementation<float, float, float>::with_estimate(
    "sve_interleaved_bf16fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_hybrid_fp32bf16fp32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_6x4VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_hybrid_fp32bf16fp32_mmla_4x6VL",
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32bf16fp32_mmla_4x6VL, float, float, float>(args); }
),
{
    "sve_interleaved_fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svef32mm() && (args._Ksize>4); },
    [](const GemmArgs &args) { return !(args._fast_mode && args._ci->has_svebf16()); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mmla_8x3VL, float, float, float>(args); }
},
{
    "sve_hybrid_fp32_mla_8x1VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return (args._Nsize < 12); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_8x1VL, float, float, float>(args); }
},
GemmImplementation<float, float, float>::with_estimate(
    "sve_hybrid_fp32_mla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_interleaved_fp32_mla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_fp32_mla_8x3VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
    KernelWeightFormat::VL2VL_BL64_BF16,
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_ffhybrid_fp32bf16fp32_mmla_4x6VL",
    KernelWeightFormat::VL2VL_BL64_BF16,
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32bf16fp32_mmla_4x6VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32bf16fp32_mmla_4x6VL, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_ffinterleaved_fp32_mla_8x3VL",
    KernelWeightFormat::VL1VL_BL32,
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp32_mla_8x3VL, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_fp32_mla_8x3VL, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "sve_ffhybrid_fp32_mla_6x4VL",
    KernelWeightFormat::VL1VL_BL32,
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32_mla_6x4VL, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_sve_ffhybrid_fp32_mla_6x4VL, float, float>(args); }
),
{
    "a64_sgemm_8x6",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A35; },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x6, float, float, float>(args); }
},
{
    "a64_smallK_hybrid_fp32_mla_8x4",
    [](const GemmArgs &args) { return args._Ksize <= 8 && (args._Nsize % 4)==0 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_8x4, float, float, float>(args); }
},
{
    "a64_smallK_hybrid_fp32_mla_6x4",
    [](const GemmArgs &args) { return (args._Ksize > 8 && args._Ksize <= 16) && (args._Nsize % 4)==0 && !args._indirect_input && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_fp32_mla_6x4, float, float, float>(args); }
},
{
    "a64_hybrid_fp32_mla_8x4",
    nullptr,
    [](const GemmArgs &args) { return (args._Nsize < 12); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_8x4, float, float, float>(args); }
},
GemmImplementation<float, float, float>::with_estimate(
    "a64_hybrid_fp32_mla_4x24",
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_4x24, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_hybrid_fp32_mla_6x16",
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_fp32_mla_6x16, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_sgemm_8x12",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_sgemm_8x12, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_ffinterleaved_bf16fp32_mmla_8x12",
    KernelWeightFormat::VL256_BL64_BF16,
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_ffhybrid_fp32bf16fp32_mmla_4x24",
    KernelWeightFormat::VL256_BL64_BF16,
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_4x24, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_ffhybrid_fp32bf16fp32_mmla_6x16",
    KernelWeightFormat::VL256_BL64_BF16,
    [](const GemmArgs &args) { return args._fast_mode && args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_ffinterleaved_fp32_mla_8x12",
    KernelWeightFormat::VL128_BL32,
    nullptr,
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp32_mla_8x12, float, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_fp32_mla_8x12, float, float, float>(args); }
),
GemmImplementation<float, float, float>::with_estimate(
    "a64_ffhybrid_fp32_mla_6x16",
    KernelWeightFormat::VL128_BL32,
    nullptr,
    [](const GemmArgs &args) { return GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32_mla_6x16, float, float>::estimate_cycles<float>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirectFixedFormat<cls_a64_ffhybrid_fp32_mla_6x16, float, float>(args); }
),
#endif // __aarch64__
#ifdef __arm__
{
    "a32_sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a32_sgemm_8x6, float, float, float>(args); }
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
const GemmImplementation<float, float, float> *gemm_implementation_list<float, float, float>() {
    return gemm_fp32_methods;
}

template UniqueGemmCommon<float, float, float> gemm<float, float, float, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<float, float, float, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<float, float, float, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai
