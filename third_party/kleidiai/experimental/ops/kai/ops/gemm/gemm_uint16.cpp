//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

#include "kernels/a64_gemm_u16_8x12.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<uint16_t, uint16_t, uint32_t> gemm_u16_methods[] =
{
{
    "a64_gemm_u16_8x12",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_u16_8x12, uint16_t, uint16_t, uint32_t>(args); }
},
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint16_t, uint16_t, uint32_t> *gemm_implementation_list<uint16_t, uint16_t, uint32_t>() {
    return gemm_u16_methods;
}

template UniqueGemmCommon<uint16_t, uint16_t, uint32_t> gemm<uint16_t, uint16_t, uint32_t, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<uint16_t, uint16_t, uint32_t, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<uint16_t, uint16_t, uint32_t, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
