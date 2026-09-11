//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#include "kernels/sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL.hpp"
#include "kernels/sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL.hpp"
#include "kernels/sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8qfp32_mopa_2VLx2VL.hpp"
#include "kernels/sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL.hpp"
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<int8_t, int8_t, float, DequantizeFloat> gemm_s8fp32_methods[] =
{
{
    "sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                        return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL, int8_t, int8_t, float>(args, dq); }
},
{
    "sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                        return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL, int8_t, int8_t, float>(args, dq); }
},
{
    "sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme2() && args._ci->has_sme_i8i32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL, int8_t, int8_t, float>(args, dq); }
},
{
    "sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                        return args._Msize <= VL || (2*VL < args._Msize && args._Msize <= 3*VL); },
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL, int8_t, int8_t, float>(args, dq); }
},
{
    "sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    [](const GemmArgs &args, const DequantizeFloat &) { const auto VL = sme::get_vector_length<float>();
                                                        return args._Nsize <= VL || (2*VL < args._Nsize && args._Nsize <= 3*VL); },
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL, int8_t, int8_t, float>(args, dq); }
},
{
    "sme_interleaved_nomerge_s8qfp32_mopa_2VLx2VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sme() && args._ci->has_sme_i8i32() && !args._accumulate; },
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &dq) { return new GemmInterleavedNoMergeDequantized<cls_sme_interleaved_nomerge_s8qfp32_mopa_2VLx2VL, int8_t, int8_t, float>(args, dq); }
},
GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t, float>(args, qp); }
),
GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sve(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, float>(args, qp); }
),
GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, float>(args, qp); }
),
{
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->get_cpu_model() == CPUModel::A53 && ((args._Msize > 28) || ((args._Msize % 8) > 4)); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s16_8x12, int8_t, int8_t, float>(args, qp); }
},
GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, float>(args, qp); }
),
GemmImplementation<int8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "a64_gemm_s8_4x4",
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_gemm_s8_4x4, int8_t, int8_t, float>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s8_4x4, int8_t, int8_t, float>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int8_t, int8_t, float, DequantizeFloat> *gemm_implementation_list<int8_t, int8_t, float, DequantizeFloat>() {
    return gemm_s8fp32_methods;
}

template UniqueGemmCommon<int8_t, int8_t, float> gemm<int8_t, int8_t, float, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);
template bool has_opt_gemm<int8_t, int8_t, float, DequantizeFloat>(WeightFormat &, const GemmArgs &, const DequantizeFloat &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, float, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
