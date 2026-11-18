//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_hybrid_u8u32_dot_6x16.hpp"
#include "kernels/a64_hybrid_u8u32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_u16u32_mla_8x12.hpp"
#include "kernels/a64_interleaved_u8u32_dot_8x12.hpp"
#include "kernels/a64_interleaved_u8u32_mmla_8x12.hpp"
#include "kernels/sve_hybrid_u8u32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_u8u32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_u8u32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_u8u32_mmla_8x3VL.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<uint8_t, uint8_t, uint32_t> gemm_u8_methods[] =
{
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "sve_hybrid_u8u32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "sve_interleaved_u8u32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "sve_hybrid_u8u32_dot_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "sve_interleaved_u8u32_dot_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "a64_interleaved_u8u32_mmla_8x12",
    [](const GemmArgs &args) { return args._ci->has_i8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "a64_hybrid_u8u32_mmla_6x16",
    [](const GemmArgs &args) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint8_t, uint32_t>(args); }
),
{
    "a64_interleaved_u16u32_mla_8x12",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Msize > 4; },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_u16u32_mla_8x12, uint8_t, uint8_t, uint32_t>(args); },
},
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "a64_hybrid_u8u32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint8_t, uint32_t>::with_estimate(
    "a64_interleaved_u8u32_dot_8x12",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_u8u32_dot_8x12, uint8_t, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_u8u32_dot_8x12, uint8_t, uint8_t, uint32_t>(args); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, uint8_t, uint32_t> *gemm_implementation_list<uint8_t, uint8_t, uint32_t>() {
    return gemm_u8_methods;
}

template UniqueGemmCommon<uint8_t, uint8_t, uint32_t> gemm<uint8_t, uint8_t, uint32_t, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<uint8_t, uint8_t, uint32_t, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint8_t, uint32_t, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai
