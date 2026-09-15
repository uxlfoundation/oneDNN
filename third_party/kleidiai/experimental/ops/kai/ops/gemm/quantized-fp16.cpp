//
// SPDX-FileCopyrightText: Copyright 2019, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai/ops/gemm/kai_ops.hpp"
#include "common_internal/utils.hpp"
#include "common_internal/quantized.hpp"

#include <arm_neon.h>

namespace kai {
namespace ops {

// FP16 dequantize
template<>
void dequantize_block_32<__fp16>(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                                 const int32_t * in_ptr, unsigned int in_stride, __fp16 *out_ptr, unsigned int out_stride,
                                 const __fp16 * bias_ptr, bool not_first_pass, const Activation &act)
{
    const float32x4_t vscale = vdupq_n_f32(qp.scale);
    float maxval = std::numeric_limits<float>::infinity();
    float minval = -std::numeric_limits<float>::infinity();

    switch(act.type) {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            maxval = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            minval = 0;
            break;
    }

    const float16x8_t vmin = vdupq_n_f16(static_cast<__fp16>(minval));
    const float16x8_t vmax = vdupq_n_f16(static_cast<__fp16>(maxval));

    for(unsigned int row=0; row<height; row++) {
        unsigned int col=0;
        if (width >= 8) {
            for(; col <= (width-8); col+=8) {
                const int32x4_t vin0 = vld1q_s32(in_ptr + col + (row * in_stride));
                const int32x4_t vin1 = vld1q_s32(in_ptr + col + 4 + (row * in_stride));

                float32x4_t vdeq0 = vmulq_f32(vcvtq_f32_s32(vin0), vscale);
                float32x4_t vdeq1 = vmulq_f32(vcvtq_f32_s32(vin1), vscale);

                if(bias_ptr) {
                    const float16x8_t bin = vld1q_f16(bias_ptr + col);
                    const float32x4_t bin0 = vcvt_f32_f16(vget_low_f16(bin));
                    const float32x4_t bin1 = vcvt_f32_f16(vget_high_f16(bin));

                    vdeq0 = vaddq_f32(vdeq0, bin0);
                    vdeq1 = vaddq_f32(vdeq1, bin1);
                }

                if(not_first_pass) {
                    const float16x8_t in = vld1q_f16(out_ptr + col + (row * out_stride));
                    const float32x4_t in0 = vcvt_f32_f16(vget_low_f16(in));
                    const float32x4_t in1 = vcvt_f32_f16(vget_high_f16(in));

                    vdeq0 = vaddq_f32(vdeq0, in0);
                    vdeq1 = vaddq_f32(vdeq1, in1);
                }

                float16x8_t vdeq16 = vcombine_f16(vcvt_f16_f32(vdeq0), vcvt_f16_f32(vdeq1));

                vdeq16 = vminq_f16(vmaxq_f16(vdeq16, vmin), vmax);
                vst1q_f16(out_ptr + col + (row * out_stride), vdeq16);
            }
        }
        // left-over elements
        for(; col < width; ++col) {
            const int32_t val = *(in_ptr + (row * in_stride) + col);
            float res = static_cast<float>(val * qp.scale);
            if(bias_ptr) {
                res += static_cast<float>(*(bias_ptr + col));
            }
            if(not_first_pass) {
                res += *(out_ptr + (row * out_stride) + col);
            }
            res = std::min(std::max(res, minval), maxval);
            *(out_ptr + (row * out_stride) + col) = static_cast<__fp16>(res);
        }
    }
}

}  // namespace ops
}  // namespace kai
