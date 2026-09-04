//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_hybrid_s8s32_dot_6x16.hpp"
#include "kernels/a64_hybrid_s8s32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_8x4.hpp"
#include "kernels/sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8s32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8s32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8s32_mopa_4VLx1VL.hpp"
#include "kernels/sve_hybrid_s8s32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_s8s32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<int8_t, int8_t, int32_t> gemm_s8_methods[] =
{
{
    "sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Nsize >= 8*VL || args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL, int8_t, int8_t, int32_t>(args); }
},
{
    "sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL, int8_t, int8_t, int32_t>(args); }
},
{
    "sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL, int8_t, int8_t, int32_t>(args); }
},
{
    "sme_interleaved_nomerge_s8s32_mopa_1VLx4VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_s8s32_mopa_1VLx4VL, int8_t, int8_t, int32_t>(args); }
},
{
    "sme_interleaved_nomerge_s8s32_mopa_4VLx1VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args) { const auto VL = sme::get_vector_length<int32_t>();
                               return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_s8s32_mopa_4VLx1VL, int8_t, int8_t, int32_t>(args); }
},
{
    "sme_interleaved_nomerge_s8s32_mopa_2VLx2VL",
    [](const GemmArgs &args) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_s8s32_mopa_2VLx2VL, int8_t, int8_t, int32_t>(args); }
},
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "sve_hybrid_s8s32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_s8s32_mmla_6x4VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_mmla_6x4VL, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "sve_hybrid_s8s32_dot_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && args._Ksize>=16; },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args) { return args._ci->has_i8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "a64_hybrid_s8s32_mmla_6x16",
    [](const GemmArgs &args) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_s8s32_mmla_6x16, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_s8s32_mmla_6x16, int8_t, int8_t, int32_t>(args); }
),
{
    "a64_smallK_hybrid_s8s32_dot_8x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input && !args._accumulate; },
    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_8x4, int8_t, int8_t, int32_t>(args); }
},
{
    "a64_smallK_hybrid_s8s32_dot_6x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input && !args._accumulate; },
    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_6x4, int8_t, int8_t, int32_t>(args); }
},
{
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && ((args._Msize > 28) || ((args._Msize % 8) > 4)); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s16_8x12, int8_t, int8_t, int32_t>(args); },
},
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "a64_hybrid_s8s32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_gemm_s8_8x12, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s8_8x12, int8_t, int8_t, int32_t>(args); }
),
GemmImplementation<int8_t, int8_t, int32_t>::with_estimate(
    "a64_gemm_s8_4x4",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_gemm_s8_4x4, int8_t, int8_t, int32_t>::estimate_cycles<int32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s8_4x4, int8_t, int8_t, int32_t>(args); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int8_t, int8_t, int32_t> *gemm_implementation_list<int8_t, int8_t, int32_t>() {
    return gemm_s8_methods;
}

template UniqueGemmCommon<int8_t, int8_t, int32_t> gemm<int8_t, int8_t, int32_t, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<int8_t, int8_t, int32_t, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, int32_t, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
