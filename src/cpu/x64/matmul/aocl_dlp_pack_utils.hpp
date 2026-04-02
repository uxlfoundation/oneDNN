/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_MATMUL_AOCL_DLP_PACK_UTILS_HPP
#define CPU_X64_MATMUL_AOCL_DLP_PACK_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"

#include <aocl_dlp.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {
namespace aocl_dlp_pack_utils {

// Returns the packed buffer size for matrix B (weights) using the
// appropriate AOCL-DLP get_reorder_buf_size API.
// src_dt/wei_dt are the source/weight data types from the matmul descriptor.
// K and N are the matrix dimensions.
// sym_quant: if true, use the symmetric quantization variant (s8s8 only).
// Returns 0 on failure (unsupported type combination).
inline size_t get_packed_b_size(data_type_t src_dt, data_type_t wei_dt,
        dim_t K, dim_t N, bool sym_quant = false) {
    const char order = 'R';
    const char trans = 'N';
    const char mat_type = 'B';
    dlp_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));

    msz_t buf_size = 0;

    using namespace data_type;
    if (sym_quant && src_dt == s8 && wei_dt == s8) {
        buf_size = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(order,
                trans, mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                nullptr, &metadata);
    } else if (src_dt == f32 && wei_dt == f32) {
        buf_size = aocl_get_reorder_buf_size_f32f32f32of32(order, trans,
                mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                &metadata);
    } else if (src_dt == bf16 && wei_dt == bf16) {
        buf_size = aocl_get_reorder_buf_size_bf16bf16f32of32(order, trans,
                mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                &metadata);
    } else if (src_dt == f16 && wei_dt == f16) {
        buf_size = aocl_get_reorder_buf_size_f16f16f16of16(order, trans,
                mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                &metadata);
    } else if (src_dt == u8 && wei_dt == s8) {
        buf_size = aocl_get_reorder_buf_size_u8s8s32os32(order, trans,
                mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                &metadata);
    } else if (src_dt == s8 && wei_dt == s8) {
        buf_size = aocl_get_reorder_buf_size_s8s8s32os32(order, trans,
                mat_type, static_cast<md_t>(K), static_cast<md_t>(N),
                &metadata);
    }

    return static_cast<size_t>(buf_size);
}

// Sets up weights_md as an aocl_dlp_packed format descriptor.
// The original dims/data_type are preserved; only format_kind and
// format_desc are changed. batch is the total number of K×N slices.
inline status_t init_packed_b_md(memory_desc_t &weights_md,
        data_type_t src_dt, dim_t K, dim_t N, dim_t batch = 1,
        bool sym_quant = false) {
    const size_t per_slice
            = get_packed_b_size(src_dt, weights_md.data_type, K, N, sym_quant);
    if (per_slice == 0) return status::unimplemented;

    weights_md.format_kind = format_kind::aocl_dlp_packed;
    weights_md.format_desc.aocl_dlp_packed_desc.size = per_slice * batch;
    weights_md.format_desc.aocl_dlp_packed_desc.per_slice_size = per_slice;
    weights_md.format_desc.aocl_dlp_packed_desc.gemm_src_dt = src_dt;
    return status::success;
}

} // namespace aocl_dlp_pack_utils
} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_MATMUL_AOCL_DLP_PACK_UTILS_HPP
