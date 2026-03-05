/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#define ALG_AVG (ALG_AVG_NP || ALG_AVG_P)

#if IS_FWD
KERNEL_ATTR
__kernel void xe_global_pooling_fwd(
        __global DATA_T *src, __global int *ws, __global DST_DATA_T *dst) {

    if (GWS_OVERFLOW) return;

    const off_t mb = get_global_id(0) / C;
    const off_t oc = get_global_id(0) % C;

    const off_t dst_off = DST_OFF(mb, oc, 0, 0, 0);

#if ALG_MAX
    float dst_val = load(dst_val, src, SRC_OFF(mb, oc, 0, 0, 0));
#if IS_TRAINING
    off_t max_idx = 0;
#endif
#else
    float dst_val = 0.f;
#endif

    for (off_t id = 0; id < ID; id++) {
        for (off_t ih = 0; ih < IH; ih++) {
            for (off_t iw = 0; iw < IW; iw++) {
                off_t src_off = SRC_OFF(mb, oc, id, ih, iw);
                float val = load(val, src, src_off);
#if ALG_MAX
                if (val > dst_val) {
                    dst_val = val;
#if IS_TRAINING
                    max_idx = id * IH * IW + ih * IW + iw;
#endif
                }
#else
                dst_val += val;
#endif
            }
        }
    }

#if ALG_MAX
    write(dst + dst_off, dst_val);
#if IS_TRAINING
    ws[dst_off] = max_idx;
#endif
#else
    write(dst + dst_off, dst_val / convert_float(ID * IH * IW));
#endif
}
#endif // IS_FWD

#if IS_BWD

KERNEL_ATTR
__kernel void xe_global_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DATA_T *diff_dst) {

    if (GWS_OVERFLOW) return;

    const off_t mb = GWS_GET_MB();
#if IS_VECTORIZED
    const off_t c = GWS_GET_C() + get_sub_group_local_id();
#else
    const off_t c = GWS_GET_C();
#endif
    const off_t spatial = GWS_GET_SPATIAL();

    const bool is_in_padded_area = NEED_ZERO_PADDING && (mb >= MB || c >= C);
    const off_t dst_off = DST_OFF(mb, c, 0, 0, 0);
#if ALG_AVG
    // Read dst value only once
    const DATA_T dst_val = diff_dst[dst_off];
#endif // ALG_AVG
    int ws_val = ws[dst_off];
    for (off_t sp_idx = spatial;
            sp_idx < min(spatial + SPATIAL_CHUNK, (off_t)SPATIAL_DIM);
            sp_idx++) {
        const off_t iw = sp_idx % IW;
        const off_t ih = ((sp_idx - iw) % (IH * IW)) / IW;
        const off_t id = (sp_idx - iw - ih * IW) / (IH * IW);
        float val_to_write;
        if (is_in_padded_area)
            val_to_write = 0.f;
        else {
#if ALG_MAX
            // Read dst value only in case it's going to be used
            const off_t current_input_idx = id * IH * IW + ih * IW + iw;
            if (current_input_idx == ws_val) {
                load(&val_to_write, diff_dst, dst_off);
            } else {
                val_to_write = 0.f;
            }
#else // ALG_MAX
            val_to_write = into_float(dst_val) / SPATIAL_DIM;
#endif // ALG_MAX
        }
        const off_t src_off = SRC_OFF(mb, GWS_GET_C(), id, ih, iw);
#if IS_VECTORIZED
        block_write(&diff_src[src_off], &val_to_write);
#else
        write(diff_src + src_off, val_to_write);
#endif
    }
}
#endif // IS_BWD
