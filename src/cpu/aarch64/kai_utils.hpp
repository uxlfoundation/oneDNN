/*******************************************************************************
* Copyright 2023-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_UTILS_HPP
#define CPU_AARCH64_KAI_UTILS_HPP

#include <cmath>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/utils.hpp"

#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"
#include "kai/ops/newgemm_lib.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace kai_utils {

inline CPUInfo *get_cpu_info() {
    static CPUInfo ci;
    return &ci;
}

inline int interleave_by(const kai::ops::WeightFormat wf) {
    return (static_cast<int>(wf) >> 8) & 0xFFF;
}

inline int block_by(const kai::ops::WeightFormat wf) {
    return (static_cast<int>(wf) >> 20) & 0xF;
}

inline bool is_fixed_format(const kai::ops::WeightFormat &wf) {
    return wf != kai::ops::WeightFormat::UNSPECIFIED
            && wf != kai::ops::WeightFormat::ANY;
}

inline bool use_bf16_fast_math(const primitive_attr_t *attr, data_type_t src_dt,
        data_type_t weights_dt, data_type_t dst_dt) {
    return utils::everyone_is(data_type::f32, src_dt, weights_dt, dst_dt)
            && utils::one_of(
                    attr->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
}

inline std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm(
        const kai::ops::GemmArgs &base_args, kai::ops::GemmConfig *cfg,
        data_type_t src_dt, data_type_t weights_dt, data_type_t dst_dt) {
    using namespace data_type;
    using namespace kai::ops;

    GemmArgs args(base_args);
    args._cfg = cfg;

    const auto swd_dt = [&](data_type_t s, data_type_t w, data_type_t d) {
        return src_dt == s && weights_dt == w && dst_dt == d;
    };

    if (swd_dt(f32, f32, f32)) return gemm<float, float, float>(args, {});

    if (swd_dt(bf16, bf16, f32))
        return gemm<bfloat16, bfloat16, float>(args, {});
    if (swd_dt(bf16, bf16, bf16))
        return gemm<bfloat16, bfloat16, bfloat16>(args, {});

    if (swd_dt(f16, f16, f16)) return gemm<__fp16, __fp16, __fp16>(args, {});
    if (swd_dt(f16, f16, f32)) return gemm<__fp16, __fp16, float>(args, {});

    if (swd_dt(s8, s8, s32))
        return gemm<int8_t, int8_t, int32_t, Nothing>(args, {});

    return nullptr;
}

inline std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm_dequant(
        const kai::ops::GemmArgs &base_args, kai::ops::GemmConfig *cfg,
        data_type_t src_dt, data_type_t weights_dt, data_type_t dst_dt,
        const kai::ops::DequantizeFloat &dequant) {
    using namespace data_type;
    using namespace kai::ops;

    GemmArgs args(base_args);
    args._cfg = cfg;

    const auto swd_dt = [&](data_type_t s, data_type_t w, data_type_t d) {
        return src_dt == s && weights_dt == w && dst_dt == d;
    };

    if (swd_dt(s8, s8, f32))
        return gemm<int8_t, int8_t, float, DequantizeFloat>(args, dequant);
    if (swd_dt(u8, s8, f32))
        return gemm<uint8_t, int8_t, float, DequantizeFloat>(args, dequant);

    return nullptr;
}

inline unsigned int split_window_2d(
        unsigned int max_threads, const kai::ops::ndrange_t &window_size) {
    const auto row_blocks = window_size.get_size(0);
    const auto col_blocks = window_size.get_size(1);

    if (col_blocks == 1) return max_threads;

    const float ratio = row_blocks / static_cast<float>(col_blocks);
    const auto ideal_row_parts = static_cast<unsigned int>(
            std::round(std::sqrt(max_threads * ratio)));

    if (ideal_row_parts == 0) return 1;

    for (unsigned int adj = 0; adj < ideal_row_parts; ++adj) {
        const unsigned int round_down = ideal_row_parts - adj;
        if (round_down > 0 && max_threads % round_down == 0) return round_down;

        const unsigned int round_up = ideal_row_parts + adj;
        if (max_threads % round_up == 0) return round_up;
    }

    return 1;
}

inline void weight_format_to_memory_desc(memory_desc_t &md,
        kai::ops::WeightFormat wf, dim_t i_dim, dim_t o_dim,
        const std::vector<dim_t> &spatial_dims,
        const std::vector<dim_t> &batch_dims = {}) {
    md.format_kind = format_kind::blocked;
    md.format_desc.blocking = blocking_desc_t {};

    const int interleaved_by_ = interleave_by(wf);
    const int block_by_ = block_by(wf);

    md.format_desc.blocking.strides[i_dim] = interleaved_by_ * block_by_;
    md.padded_dims[i_dim] = utils::rnd_up(md.dims[i_dim], block_by_);

    dim_t outer_stride = interleaved_by_ * md.padded_dims[i_dim];
    dim_t dense_size = md.padded_dims[i_dim];

    for (dim_t sd : spatial_dims) {
        md.format_desc.blocking.strides[sd] = outer_stride;
        outer_stride *= md.padded_dims[sd];
        dense_size *= md.padded_dims[sd];
    }

    md.format_desc.blocking.strides[o_dim] = outer_stride;
    md.padded_dims[o_dim] = utils::rnd_up(md.dims[o_dim], interleaved_by_);
    dense_size *= md.padded_dims[o_dim];

    dim_t batch_stride = dense_size;
    for (dim_t bd : batch_dims) {
        md.format_desc.blocking.strides[bd] = batch_stride;
        batch_stride *= md.padded_dims[bd];
    }

    if (interleaved_by_ > 1) {
        md.format_desc.blocking.inner_nblks = 1 + (block_by_ > 1);
        md.format_desc.blocking.inner_idxs[0] = o_dim;
        md.format_desc.blocking.inner_blks[0] = interleaved_by_;
        if (block_by_ > 1) {
            md.format_desc.blocking.inner_nblks = 2;
            md.format_desc.blocking.inner_idxs[1] = i_dim;
            md.format_desc.blocking.inner_blks[1] = block_by_;
        }
    }
}

} // namespace kai_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
