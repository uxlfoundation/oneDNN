/*******************************************************************************
 * Copyright 2025 Intel Corporation
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
#ifndef GPU_INTEL_INCLUDE_DNNL_INTEROP_H
#define GPU_INTEL_INCLUDE_DNNL_INTEROP_H

// Intended for use in both OpenCL C and C++ code. This is largely intended to
// enable validation of the portion of the oneDNN C API that is required within
// OpenCL kernels.

#ifdef __OPENCL_VERSION__
#define CONSTANT __constant
#else
#define CONSTANT constexpr
#endif

// Eltwise algorithm kinds
CONSTANT int eltwise_relu = 0x20;
CONSTANT int eltwise_tanh = 0x21;
CONSTANT int eltwise_elu = 0x22;
CONSTANT int eltwise_square = 0x23;
CONSTANT int eltwise_abs = 0x24;
CONSTANT int eltwise_sqrt = 0x25;
CONSTANT int eltwise_linear = 0x26;
CONSTANT int eltwise_soft_relu = 0x27;
CONSTANT int eltwise_hardsigmoid = 0x28;
CONSTANT int eltwise_logistic = 0x29;
CONSTANT int eltwise_exp = 0x2a;
CONSTANT int eltwise_gelu_tanh = 0x2b;
CONSTANT int eltwise_swish = 0x2c;
CONSTANT int eltwise_log = 0x2d;
CONSTANT int eltwise_clip = 0x2e;
CONSTANT int eltwise_clip_v2 = 0x2f;
CONSTANT int eltwise_pow = 0x30;
CONSTANT int eltwise_gelu_erf = 0x31;
CONSTANT int eltwise_round = 0x32;
CONSTANT int eltwise_mish = 0x33;
CONSTANT int eltwise_hardswish = 0x34;

CONSTANT int eltwise_relu_dst = 0x100;
CONSTANT int eltwise_tanh_dst = 0x101;
CONSTANT int eltwise_elu_dst = 0x102;
CONSTANT int eltwise_sqrt_dst = 0x103;
CONSTANT int eltwise_logistic_dst = 0x104;
CONSTANT int eltwise_exp_dst = 0x105;
CONSTANT int eltwise_clip_v2_dst = 0x106;

CONSTANT int binary_add = 0x1fff0;
CONSTANT int binary_mul = 0x1fff1;
CONSTANT int binary_max = 0x1fff2;
CONSTANT int binary_min = 0x1fff3;
CONSTANT int binary_div = 0x1fff4;
CONSTANT int binary_sub = 0x1fff5;
CONSTANT int binary_ge = 0x1fff6;
CONSTANT int binary_gt = 0x1fff7;
CONSTANT int binary_le = 0x1fff8;
CONSTANT int binary_lt = 0x1fff9;
CONSTANT int binary_eq = 0x1fffa;
CONSTANT int binary_ne = 0x1fffb;
CONSTANT int binary_select = 0x1fffc;

#ifndef __OPENCL_VERSION__
#include "dnnl_types.h"

// Eltwise algorithm kinds
static_assert(eltwise_relu == dnnl_eltwise_relu, "dnnl API mismatch");
static_assert(eltwise_tanh == dnnl_eltwise_tanh, "dnnl API mismatch");
static_assert(eltwise_elu == dnnl_eltwise_elu, "dnnl API mismatch");
static_assert(eltwise_square == dnnl_eltwise_square, "dnnl API mismatch");
static_assert(eltwise_abs == dnnl_eltwise_abs, "dnnl API mismatch");
static_assert(eltwise_sqrt == dnnl_eltwise_sqrt, "dnnl API mismatch");
static_assert(eltwise_linear == dnnl_eltwise_linear, "dnnl API mismatch");
static_assert(eltwise_soft_relu == dnnl_eltwise_soft_relu, "dnnl API mismatch");
static_assert(
        eltwise_hardsigmoid == dnnl_eltwise_hardsigmoid, "dnnl API mismatch");
static_assert(eltwise_logistic == dnnl_eltwise_logistic, "dnnl API mismatch");
static_assert(eltwise_exp == dnnl_eltwise_exp, "dnnl API mismatch");
static_assert(eltwise_gelu_tanh == dnnl_eltwise_gelu_tanh, "dnnl API mismatch");
static_assert(eltwise_swish == dnnl_eltwise_swish, "dnnl API mismatch");
static_assert(eltwise_log == dnnl_eltwise_log, "dnnl API mismatch");
static_assert(eltwise_clip == dnnl_eltwise_clip, "dnnl API mismatch");
static_assert(eltwise_clip_v2 == dnnl_eltwise_clip_v2, "dnnl API mismatch");
static_assert(eltwise_pow == dnnl_eltwise_pow, "dnnl API mismatch");
static_assert(eltwise_gelu_erf == dnnl_eltwise_gelu_erf, "dnnl API mismatch");
static_assert(eltwise_round == dnnl_eltwise_round, "dnnl API mismatch");
static_assert(eltwise_mish == dnnl_eltwise_mish, "dnnl API mismatch");
static_assert(eltwise_hardswish == dnnl_eltwise_hardswish, "dnnl API mismatch");

static_assert(eltwise_relu_dst == dnnl_eltwise_relu_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_tanh_dst == dnnl_eltwise_tanh_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_elu_dst == dnnl_eltwise_elu_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_sqrt_dst == dnnl_eltwise_sqrt_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_logistic_dst == dnnl_eltwise_logistic_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_exp_dst == dnnl_eltwise_exp_use_dst_for_bwd,
        "dnnl API mismatch");
static_assert(eltwise_clip_v2_dst == dnnl_eltwise_clip_v2_use_dst_for_bwd,
        "dnnl API mismatch");

// Binary algorithm kinds
static_assert(binary_add == dnnl_binary_add, "dnnl API mismatch");
static_assert(binary_mul == dnnl_binary_mul, "dnnl API mismatch");
static_assert(binary_max == dnnl_binary_max, "dnnl API mismatch");
static_assert(binary_min == dnnl_binary_min, "dnnl API mismatch");
static_assert(binary_div == dnnl_binary_div, "dnnl API mismatch");
static_assert(binary_sub == dnnl_binary_sub, "dnnl API mismatch");
static_assert(binary_ge == dnnl_binary_ge, "dnnl API mismatch");
static_assert(binary_gt == dnnl_binary_gt, "dnnl API mismatch");
static_assert(binary_le == dnnl_binary_le, "dnnl API mismatch");
static_assert(binary_lt == dnnl_binary_lt, "dnnl API mismatch");
static_assert(binary_eq == dnnl_binary_eq, "dnnl API mismatch");
static_assert(binary_ne == dnnl_binary_ne, "dnnl API mismatch");
static_assert(binary_select == dnnl_binary_select, "dnnl API mismatch");

#endif
#endif
