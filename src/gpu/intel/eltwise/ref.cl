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
#include "gpu/intel/include/eltwise.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types_interop.h"

#define GWS_GET_THREAD_ID(index) (get_global_id(index) + offset.array[index])

#if IS_FWD
__kernel void ref_eltwise_fwd(__global SRC_DATA_T *src,
        __global DST_DATA_T *dst, float alpha, float beta,
        dispatch_gws_rt_params_t gws_params, int64x3_t offset POST_OP_ARGS) {

    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    float tmp_s = load(tmp_s, src);
    tmp_s = fwd_eltwise(tmp_s, alpha, beta, 1.0f);

    float dst_data = 0;
#if WITH_SUM
    load(&dst_data, dst);
#endif
    APPLY_POST_OPS_SERIAL(tmp_s, dst_data, GWS_GET_OFF(A, gws_params),
            GWS_GET_OFF(B, gws_params), GWS_GET_OFF(C, gws_params),
            GWS_GET_OFF(D, gws_params), GWS_GET_OFF(E, gws_params),
            GWS_GET_OFF(F, gws_params));

    write(dst, tmp_s);
}

#else // #if IS_FWD

#if DT_F64 == 1 || DT_F32 == 1 || DT_BF16 == 1 || DT_F16 == 1

__kernel void ref_eltwise_bwd(__global SRC_DATA_T *src,
        __global DIFF_SRC_DATA_T *diff_src, __global DIFF_DST_DATA_T *diff_dst,
        float alpha, float beta, dispatch_gws_rt_params_t gws_params,
        int64x3_t offset) {
    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    diff_src = GWS_GET_BUFFER_POS(DIFF_SRC, gws_params, diff_src);
    diff_dst = GWS_GET_BUFFER_POS(DIFF_DST, gws_params, diff_dst);

    POST_OP_DATA_T tmp_dd = load(tmp_dd, diff_dst);
    POST_OP_DATA_T tmp_s = load(tmp_s, src);
    write(diff_src, bwd_eltwise(tmp_dd, tmp_s, alpha, beta));
}
#endif

#endif
