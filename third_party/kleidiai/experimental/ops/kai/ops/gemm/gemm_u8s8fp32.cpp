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

#include "kernels/a64_interleaved_u8s8s32_mmla_8x12.hpp"
#include "kernels/sve_interleaved_u8s8s32_mmla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> gemm_u8s8fp32_methods[] =
{
GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "sve_interleaved_u8s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
    "a64_interleaved_u8s8s32_mmla_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> *gemm_implementation_list<uint8_t, int8_t, float, DequantizeFloat>() {
    return gemm_u8s8fp32_methods;
}

template UniqueGemmCommon<uint8_t, int8_t, float> gemm<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);
template bool has_opt_gemm<uint8_t, int8_t, float, DequantizeFloat>(WeightFormat &, const GemmArgs &, const DequantizeFloat &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
