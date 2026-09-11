//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SVE

namespace kai {
namespace ops {

static const GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat> gemm_s8fp16_methods[] =
{
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t,int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t,int8_t, __fp16>(args, qp); }
),
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sve(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t,int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, __fp16>(args, qp); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, __fp16>(args, qp); }
),
{
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->get_cpu_model() == CPUModel::A53 && ((args._Msize > 28) || ((args._Msize % 8) > 4)); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s16_8x12, int8_t, int8_t, __fp16>(args, qp); }
},
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t,  __fp16>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat> *gemm_implementation_list<int8_t, int8_t, __fp16, DequantizeFloat>() {
    return gemm_s8fp16_methods;
}

template UniqueGemmCommon<int8_t, int8_t, __fp16> gemm<int8_t, int8_t, __fp16, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);
template bool has_opt_gemm<int8_t, int8_t, __fp16, DequantizeFloat>(WeightFormat &, const GemmArgs &, const DequantizeFloat &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, __fp16, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
