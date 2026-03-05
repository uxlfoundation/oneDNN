/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/types.h"
#include "gpu/intel/include/types_interop.h"

#ifdef USE_GENERAL_KERNEL
#if ONE_REDUCTION_PER_SUBGROUP == 1
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
#endif
__kernel void
reusable_softmax_fwd_generic(__global SRC_DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale,
        dim_t softmax_axis_size, dim_t softmax_axis_stride,
        dim_t softmax_axis_chunk_size, dispatch_gws_rt_params_t gws_params) {
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    FLT_ACC_DATA_T max_ = min_val((FLT_ACC_DATA_T)0);
    FLT_ACC_DATA_T denom_ = zero_val((FLT_ACC_DATA_T)0);

    const size_t begin
            = GWS_GET_OFF_NAMED(SRC, DEFAULT_DISPATCHER_SUFFIX, gws_params);

#if MANY_REDUCTIONS_PER_WORKGROUP == 1
    const size_t end = begin + softmax_axis_stride * softmax_axis_size;
#else
    const size_t axis_begin = GWS_GET_OFF_NAMED(
            ORIGINAL, DEFAULT_DISPATCHER_SUFFIX, gws_params);
    const size_t end = min(axis_begin + softmax_axis_stride * softmax_axis_size,
            begin + softmax_axis_stride * softmax_axis_chunk_size);
#endif

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        max_ = max(max_, load(max_, src, c));
    }
    if (USE_WORKGROUP_REDUCTION) { max_ = work_group_reduce_max(max_); }
    if (USE_SUBGROUP_REDUCTION) { max_ = sub_group_reduce_max(max_); }
    max_ = isfinite(max_) ? max_ : -FLT_MAX;

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        denom_ += exp(load(denom_, src, c) - max_);
    }
    if (USE_WORKGROUP_REDUCTION) { denom_ = work_group_reduce_add(denom_); }
    if (USE_SUBGROUP_REDUCTION) { denom_ = sub_group_reduce_add(denom_); }

    if (LOGSOFTMAX) {
        denom_ = log(denom_);
    } else if (SOFTMAX_INF_AS_ZERO && denom_ == 0.f) {
        denom_ = 1.0f;
    } else {
        denom_ = 1.0f / denom_;
    }

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        FLT_ACC_DATA_T src_val = load(src_val, src, c);
        FLT_ACC_DATA_T unscaled = LOGSOFTMAX ? src_val - max_ - denom_
                                             : exp(src_val - max_) * denom_;

        float scale = 1.0f;
        if (src_scale) { scale = *src_scale; }
        if (dst_scale) { scale /= *dst_scale; }

        write(dst + (c - begin), unscaled * scale);
    }
}
#endif

#define VECT_SIZE 8

#if DST_DT_F64
#define COMMON_DATA_MAX DBL_MAX
#else
#define COMMON_DATA_MAX FLT_MAX
#endif

#ifdef USE_VECTORIZED_KERNEL
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
reusable_softmax_fwd_generic(__global DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale,
        dim_t softmax_axis_size, dim_t softmax_axis_stride,
        dim_t softmax_axis_chunk_size, dispatch_gws_rt_params_t gws_params) {

    float scale = 1.0f;
    if (src_scale) { scale = *src_scale; }
    if (dst_scale) { scale /= *dst_scale; }

    const off_t linear_thread_id = get_global_id(0);
    const off_t data_off
            = (linear_thread_id / SUBGROUP_SIZE) * softmax_axis_size;
    global DATA_T *src_backup = src;
    src += data_off;

    VECT_FLOAT_T dk;
    float max_ = -INFINITY;
    float denom_ = 0.0f;
    const bool has_tail = softmax_axis_size % (SUBGROUP_SIZE * VECT_SIZE);
    int last_buf = div_up(softmax_axis_size, (SUBGROUP_SIZE * VECT_SIZE));
    if (has_tail) last_buf--;

    const off_t idx_end = has_tail
            ? softmax_axis_size - (SUBGROUP_SIZE * VECT_SIZE + 1)
            : softmax_axis_size;

    for (off_t idx = 0; idx < idx_end; idx += SUBGROUP_SIZE * VECT_SIZE) {
        dk = block_load(dk, src + idx);
        for (off_t i = 0; i < VECT_SIZE; ++i) {
            max_ = max(dk[i], max_);
        }
    }

    if (has_tail) {
        const off_t idx_beg = last_buf * SUBGROUP_SIZE * VECT_SIZE
                + get_sub_group_local_id();
        const off_t idx_end = idx_beg + SUBGROUP_SIZE * VECT_SIZE;
        for (off_t idx = idx_beg; idx < idx_end; idx += SUBGROUP_SIZE) {
            float d = (idx < softmax_axis_size ? load(d, src, idx)
                                               : -COMMON_DATA_MAX);
            max_ = max(d, max_);
        }
    }

    max_ = sub_group_reduce_max(max_);
    max_ = isfinite(max_) ? max_ : -FLT_MAX;

    for (off_t idx = 0; idx < idx_end; idx += SUBGROUP_SIZE * VECT_SIZE) {
        dk = block_load(dk, src + idx);
        dk = exp(dk - max_);
        for (off_t i = 0; i < VECT_SIZE; ++i)
            denom_ += dk[i];
    }

    if (has_tail) {
        const off_t idx_beg = last_buf * SUBGROUP_SIZE * VECT_SIZE
                + get_sub_group_local_id();
        const off_t idx_end = idx_beg + SUBGROUP_SIZE * VECT_SIZE;
        for (off_t idx = idx_beg; idx < idx_end; idx += SUBGROUP_SIZE) {
            if (idx < softmax_axis_size)
                denom_ += exp(load(denom_, src, idx) - max_);
        }
    }
    denom_ = sub_group_reduce_add(denom_);

    if (LOGSOFTMAX) {
        denom_ = log(denom_);
    } else if (SOFTMAX_INF_AS_ZERO && denom_ == 0.f) {
        denom_ = 1.0f;
    } else {
        denom_ = 1.0f / denom_;
    }

    dst += data_off;

    for (off_t idx = 0; idx < idx_end; idx += SUBGROUP_SIZE * VECT_SIZE) {
        dk = block_load(dk, src + idx);
        dk = LOGSOFTMAX ? dk - max_ - denom_ : exp(dk - max_) * denom_;
        VECT_FLOAT_T result = scale * dk;
        block_write(dst + idx, &result);
    }

    if (has_tail) {
        const off_t idx_beg = last_buf * SUBGROUP_SIZE * VECT_SIZE
                + get_sub_group_local_id();
        const off_t idx_end = idx_beg + SUBGROUP_SIZE * VECT_SIZE;
        for (off_t idx = idx_beg; idx < idx_end; idx += SUBGROUP_SIZE) {
            if (idx < softmax_axis_size) {
                float dk = load(dk, src, idx);
                dk = LOGSOFTMAX ? dk - max_ - denom_ : exp(dk - max_) * denom_;
                write(dst + idx, scale * dk);
            }
        }
    }
}
#endif

#ifdef USE_SMALL_KERNEL
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
reusable_softmax_fwd_generic(__global DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale,
        dim_t softmax_axis_size, dim_t softmax_axis_stride,
        dim_t softmax_axis_chunk_size, dispatch_gws_rt_params_t gws_params) {
    float scale = 1.0f;
    if (src_scale) { scale = *src_scale; }
    if (dst_scale) { scale /= *dst_scale; }
    const off_t data_off
            = (get_global_id(0) / SUBGROUP_SIZE) * softmax_axis_size;
    float d;
    float max_ = -INFINITY;
    float denom_ = 0.0f;
    src += data_off;

    const off_t off = get_sub_group_local_id();

    d = (off < softmax_axis_size ? load(d, src, off) : -INFINITY);
    max_ = sub_group_reduce_max(d);
    max_ = isfinite(max_) ? max_ : -FLT_MAX;

    if (off < softmax_axis_size) denom_ += exp(d - max_);

    denom_ = sub_group_reduce_add(denom_);
    if (LOGSOFTMAX) {
        denom_ = log(denom_);
    } else if (SOFTMAX_INF_AS_ZERO && denom_ == 0.f) {
        denom_ = 1.0f;
    } else {
        denom_ = 1.0f / denom_;
    }
    dst += data_off;

    if (off < softmax_axis_size) {
        float from_src = d;
        float thing = LOGSOFTMAX ? from_src - max_ - denom_
                                 : exp(from_src - max_) * denom_;
        write(dst + off, scale * thing);
    }
}
#endif
