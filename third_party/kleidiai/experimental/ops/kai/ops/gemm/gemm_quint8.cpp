//
// SPDX-FileCopyrightText: Copyright 2019-2020, 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "gemm_hybrid_indirect.hpp"
#include "gemm_hybrid_quantized.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_pretransposed.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_gemm_u16_8x12.hpp"
#include "kernels/a64_gemm_u8_4x4.hpp"
#include "kernels/a64_gemm_u8_8x12.hpp"
#include "kernels/a64_hybrid_u8qa_dot_4x16.hpp"
#include "kernels/a64_hybrid_u8qa_mmla_4x16.hpp"
#include "kernels/a64_hybrid_u8u32_dot_6x16.hpp"
#include "kernels/a64_hybrid_u8u32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_u8u32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_8x4.hpp"
#include "kernels/sme2_gemv_u8qa_dot_16VL.hpp"
#include "kernels/sme2_interleaved_nomerge_u8q_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_u8q_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_u8q_mopa_4VLx1VL.hpp"
#include "kernels/sme_gemv_u8qa_dot_8VL.hpp"
#include "kernels/sme_interleaved_nomerge_u8q_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_u8q_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_u8q_mopa_4VLx1VL.hpp"
#include "kernels/sve_hybrid_u8qa_dot_4x4VL.hpp"
#include "kernels/sve_hybrid_u8qa_mmla_4x4VL.hpp"
#include "kernels/sve_hybrid_u8u32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_u8u32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_u8u32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_u8u32_mmla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32> gemm_quint8_methods[] =
{
{
    "sme2_gemv_u8qa_dot_16VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme2() && quant_hybrid_asymmetric(qp) && args._Msize == 1 && !args._indirect_input && args._nbatches == 1;  },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemvPretransposed<cls_sme2_gemv_u8qa_dot_16VL, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
},
{
    "sme_gemv_u8qa_dot_8VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme() && quant_hybrid_asymmetric(qp) && args._Msize == 1 && !args._indirect_input && args._nbatches == 1;  },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemvPretransposed<cls_sme_gemv_u8qa_dot_8VL, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
},
{
    "sme2_interleaved_nomerge_u8q_mopa_1VLx4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    [](const GemmArgs &args, const Requantize32 &) { const auto VL = sme::get_vector_length<uint32_t>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme2_interleaved_nomerge_u8q_mopa_1VLx4VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "sme2_interleaved_nomerge_u8q_mopa_4VLx1VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    [](const GemmArgs &args, const Requantize32 &) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme2_interleaved_nomerge_u8q_mopa_4VLx1VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "sme2_interleaved_nomerge_u8q_mopa_2VLx2VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme2_interleaved_nomerge_u8q_mopa_2VLx2VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "sme_interleaved_nomerge_u8q_mopa_1VLx4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    [](const GemmArgs &args, const Requantize32 &) { const auto VL = sme::get_vector_length<uint32_t>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme_interleaved_nomerge_u8q_mopa_1VLx4VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "sme_interleaved_nomerge_u8q_mopa_4VLx1VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    [](const GemmArgs &args, const Requantize32 &) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme_interleaved_nomerge_u8q_mopa_4VLx1VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "sme_interleaved_nomerge_u8q_mopa_2VLx2VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && ((qp.per_channel_requant && (qp.per_channel_left_shifts == nullptr)) || (!qp.per_channel_requant && (qp.per_layer_left_shift == 0)));},
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedPretransposedNoMergeQuantizedInline<cls_sme_interleaved_nomerge_u8q_mopa_2VLx2VL, uint8_t, uint8_t, uint8_t>(args, qp); }
},
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8qa_mmla_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_asymmetric(qp) && args._ci->has_sve2() && args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8qa_mmla_4x4VL, uint8_t, uint8_t, uint8_t, Requantize32>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8qa_mmla_4x4VL, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_interleaved_u8u32_mmla_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8u32_mmla_6x4VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8qa_dot_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sve2() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8qa_dot_4x4VL, uint8_t, uint8_t, uint8_t, Requantize32>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8qa_dot_4x4VL, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8u32_dot_6x4VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_sve(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "sve_interleaved_u8u32_dot_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_sve() && (args._Ksize>4); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8qa_mmla_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_i8mm() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8qa_mmla_4x16, uint8_t, uint8_t, uint8_t, Requantize32>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8qa_mmla_4x16, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_interleaved_u8u32_mmla_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm() && (args._Ksize>8); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8u32_mmla_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint8_t, uint8_t, Requantize32, true>(args, qp); }
),
{
    "a64_smallK_hybrid_u8u32_dot_8x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input; },
    [](const GemmArgs &args, const Requantize32 &) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_u8u32_dot_8x4, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "a64_smallK_hybrid_u8u32_dot_6x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input; },
    [](const GemmArgs &args, const Requantize32 &) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_u8u32_dot_6x4, uint8_t, uint8_t, uint8_t>(args, qp); }
},
{
    "a64_gemm_u16_8x12",
    nullptr,
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Msize > 4; },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u16_8x12, uint8_t, uint8_t, uint8_t>(args, qp); },
},
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8qa_dot_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_dotprod() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8qa_dot_4x16, uint8_t, uint8_t, uint8_t, Requantize32>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8qa_dot_4x16, uint8_t, uint8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8u32_dot_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_gemm_u8_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_a64_gemm_u8_8x12, uint8_t, uint8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u8_8x12, uint8_t, uint8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32>::with_estimate(
    "a64_gemm_u8_4x4",
    nullptr,
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_a64_gemm_u8_4x4, uint8_t, uint8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u8_4x4, uint8_t, uint8_t, uint8_t>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, uint8_t, uint8_t, Requantize32> *gemm_implementation_list<uint8_t, uint8_t, uint8_t, Requantize32>() {
    return gemm_quint8_methods;
}

template UniqueGemmCommon<uint8_t, uint8_t, uint8_t> gemm<uint8_t, uint8_t, uint8_t, Requantize32>(const GemmArgs &, const Requantize32 &);
template bool has_opt_gemm<uint8_t, uint8_t, uint8_t, Requantize32>(WeightFormat &, const GemmArgs &, const Requantize32 &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint8_t, uint8_t, Requantize32>(const GemmArgs &, const Requantize32 &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
