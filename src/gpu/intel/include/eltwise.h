/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_INCLUDE_ELTWISE_H
#define GPU_INTEL_INCLUDE_ELTWISE_H

#include "gpu/intel/include/dnnl_interop.h"
#include "gpu/intel/include/types.h"

#if DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef DATA_MAX
#if DT_F16 == 1
#define DATA_MAX HALF_MAX
#elif DT_S8 == 1
#define DATA_MAX CHAR_MAX
#elif DT_U8 == 1
#define DATA_MAX UCHAR_MAX
#else
#define DATA_MAX FLT_MAX
#endif
#endif

#if DT_F64 == 1
#define POST_OP_LITERAL(x) x
#else
#define POST_OP_LITERAL(x) x##f
#endif

#define ELT_T POST_OP_DATA_T
#include "gpu/intel/include/eltwise_fwd_body.hxx"
#undef ELT_T

#ifdef ELTWISE_VECTOR_API
// float1 has no builtin overloads, so it defers to the scalar form.
#include "gpu/intel/include/generic_vector_ops.h"

float1 __attribute__((overloadable)) fwd_eltwise_common(
        int eltwise_alg, float1 x, float alpha_, float beta_, float scale_) {
    x[0] = fwd_eltwise_common(eltwise_alg, x[0], alpha_, beta_, scale_);
    return x;
}

#define ELT_T float2
#include "gpu/intel/include/eltwise_fwd_body.hxx"
#undef ELT_T
#define ELT_T float4
#include "gpu/intel/include/eltwise_fwd_body.hxx"
#undef ELT_T
#define ELT_T float8
#include "gpu/intel/include/eltwise_fwd_body.hxx"
#undef ELT_T
#define ELT_T float16
#include "gpu/intel/include/eltwise_fwd_body.hxx"
#undef ELT_T
#endif

#define ELT_T POST_OP_DATA_T
#include "gpu/intel/include/eltwise_bwd_body.hxx"
#undef ELT_T

float fwd_eltwise(POST_OP_DATA_T x, float alpha_, float beta_, float scale_) {
#ifdef ELTWISE_ALG
    return fwd_eltwise_common(ELTWISE_ALG, x, alpha_, beta_, scale_);
#else
    return x;
#endif
}

float bwd_eltwise(
        POST_OP_DATA_T x, POST_OP_DATA_T y, float alpha_, float beta_) {
#ifdef ELTWISE_ALG
    switch (ELTWISE_ALG) {
        case eltwise_relu: return relu_bwd(x, y, alpha_); break;
        case eltwise_linear: return linear_bwd(x, alpha_); break;
        case eltwise_soft_relu: return soft_relu_bwd(x, y, alpha_); break;
        case eltwise_mish: return mish_bwd(x, y); break;
        case eltwise_logistic: return logistic_bwd(x, y); break;
        case eltwise_tanh: return tanh_bwd(x, y); break;
        case eltwise_elu: return elu_bwd(x, y, alpha_); break;
        case eltwise_square: return square_bwd(x, y); break;
        case eltwise_sqrt: return sqrt_bwd(x, y); break;
        case eltwise_abs: return abs_bwd(x, y); break;
        case eltwise_exp: return exp_bwd(x, y); break;
        case eltwise_gelu_tanh: return gelu_tanh_bwd(x, y); break;
        case eltwise_swish: return swish_bwd(x, y, alpha_); break;
        case eltwise_log: return log_bwd(x, y); break;
        case eltwise_clip: return clip_bwd(x, y, alpha_, beta_); break;
        case eltwise_clip_v2: return clip_v2_bwd(x, y, alpha_, beta_); break;
        case eltwise_pow: return pow_bwd(x, y, alpha_, beta_); break;
        case eltwise_gelu_erf: return gelu_erf_bwd(x, y); break;
        case eltwise_hardswish:
            return hardswish_bwd(x, y, alpha_, beta_);
            break;
        case eltwise_hardsigmoid:
            return hardsigmoid_bwd(x, y, alpha_, beta_);
            break;
        case eltwise_relu_dst: return relu_bwd_use_dst(x, y, alpha_); break;
        case eltwise_logistic_dst: return logistic_bwd_use_dst(x, y); break;
        case eltwise_tanh_dst: return tanh_bwd_use_dst(x, y); break;
        case eltwise_elu_dst: return elu_bwd_use_dst(x, y, alpha_); break;
        case eltwise_sqrt_dst: return sqrt_bwd_use_dst(x, y); break;
        case eltwise_exp_dst: return exp_bwd_use_dst(x, y); break;
        case eltwise_clip_v2_dst:
            return clip_v2_bwd_use_dst(x, y, alpha_, beta_);
            break;

        default: return x; break;
    }
#else
    return x;
#endif
}

#endif // GPU_INTEL_INCLUDE_ELTWISE_H
