/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#define DT_UNDEF 1
#include "gpu/intel/include/types.h"
#include "gpu/intel/include/types_interop.h"

__kernel void subbyte_pack(__global uchar *restrict src,
        __global uchar *restrict dst, int64x6_t dims, int64x6_t strides,
        int64x3_t offset) {
    const int elems_per_byte = 8 / SUBBYTE_PACK_BITS;

    const off_t i0 = (get_global_id(0) + offset.array[0]) * elems_per_byte;
    off_t off = i0;
    if (SUBBYTE_PACK_NDIMS > 1)
        off += (get_global_id(1) + offset.array[1]) * strides.array[1];

    if (SUBBYTE_PACK_NDIMS > 2) {
        off_t rem = get_global_id(2) + offset.array[2];
        for (int d = 2; d < SUBBYTE_PACK_NDIMS; ++d) {
            off += (rem % dims.array[d]) * strides.array[d];
            rem /= dims.array[d];
        }
    }

    uchar packed = 0;
    for (int k = 0; k < elems_per_byte; ++k)
        if (i0 + k < dims.array[0])
            packed |= (uchar)(src[off + k] << (k * SUBBYTE_PACK_BITS));
    dst[off / elems_per_byte] = packed;
}
