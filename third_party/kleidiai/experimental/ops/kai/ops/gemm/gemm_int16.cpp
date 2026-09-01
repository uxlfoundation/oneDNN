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

#include "kernels/a64_gemm_s16_8x12.hpp"

namespace kai {
namespace ops {

static const GemmImplementation<int16_t, int16_t, int32_t> gemm_s16_methods[] =
{
{
    "a64_gemm_s16_8x12",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s16_8x12, int16_t, int16_t, int32_t>(args); }
},
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int16_t, int16_t, int32_t> *gemm_implementation_list<int16_t, int16_t, int32_t>() {
    return gemm_s16_methods;
}

template UniqueGemmCommon<int16_t, int16_t, int32_t> gemm<int16_t, int16_t, int32_t, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<int16_t, int16_t, int32_t, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<int16_t, int16_t, int32_t, Nothing>(const GemmArgs &, const Nothing &);

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
