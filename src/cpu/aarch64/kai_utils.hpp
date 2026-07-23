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

inline bool is_bf16_weight_format(const kai::ops::WeightFormat &wf) {
    return ((static_cast<int>(wf) >> 4) & 0xF) == 1;
}

inline bool use_fast_mode(
        const memory_desc_t &src_md, const primitive_attr_t &attr) {

    // KleidiAI ops fast_mode_ means
    // - If src, weight, dst are f16 then we can use f16 accumulation
    // - If src and weight are f32 or bf16 then we can cast them to bf16
    if (src_md.data_type == data_type::f16) {
        return utils::one_of(attr.acc_mode_, accumulation_mode::relaxed,
                accumulation_mode::any, accumulation_mode::f16);
    } else if (utils::one_of(
                       src_md.data_type, data_type::bf16, data_type::f32)) {
        return utils::one_of(
                attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
    }
    return false;
}

inline bool is_bf16(const memory_desc_t &src_md,
        const memory_desc_t &weights_md, const primitive_attr_t &attr) {
    return (src_md.data_type == data_type::bf16
                   && weights_md.data_type == data_type::bf16)
            || utils::one_of(
                    attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
}

struct post_ops_fusion_t {
    int fallback_start_index = 0;
    bool accumulate = false;
    kai::ops::Activation activation {};

    bool has_fallback(const post_ops_t &post_ops) const {
        return post_ops.len() > fallback_start_index;
    }
};

// Create a post ops fusion struct which describes what a kai op will fuse
// into the kernel itself and what will go to fallback. Note that this cannot fail
// because the worst case is that everything goess to fallback.
inline post_ops_fusion_t create_post_ops_fusion(
        const post_ops_t &post_ops, bool allow_sum_fusion) {
    post_ops_fusion_t fusion = {};

    if (post_ops.len() == 0) return fusion;

    if (post_ops.entry_[0].is_sum()) {
        if (!allow_sum_fusion) return fusion;
        fusion.accumulate = true;
        fusion.fallback_start_index = 1;
    }

    if (post_ops.len() <= fusion.fallback_start_index) return fusion;

    const auto &po = post_ops.entry_[fusion.fallback_start_index];
    if (!po.is_eltwise(true)) return fusion;

    if (po.eltwise.alg == alg_kind::eltwise_relu && po.eltwise.alpha == 0.0f) {
        fusion.activation
                = kai::ops::Activation(kai::ops::Activation::Type::ReLU);
        fusion.fallback_start_index += 1;
    } else if (utils::one_of(po.eltwise.alg, alg_kind::eltwise_clip,
                       alg_kind::eltwise_clip_v2)
            && po.eltwise.alpha == 0.0f && po.eltwise.beta > 0.0f) {
        fusion.activation = kai::ops::Activation(
                kai::ops::Activation::Type::BoundedReLU, po.eltwise.beta, 0.0f);
        fusion.fallback_start_index += 1;
    }

    return fusion;
}

inline int num_sum_post_ops(const post_ops_t &po) {
    int count = 0;
    for (int i = 0; i < po.len(); ++i) {
        if (po.entry_[i].is_sum()) count++;
    }
    return count;
}

inline std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm(
        const kai::ops::GemmArgs &base_args, kai::ops::GemmConfig *cfg,
        data_type_t src_dt, data_type_t weights_dt, data_type_t dst_dt,
        int max_threads = 0) {
    using namespace data_type;
    using namespace kai::ops;

    GemmArgs args(base_args);
    if (max_threads > 0) args._maxthreads = max_threads;
    args._cfg = cfg;

    const auto dt = [&](data_type_t s, data_type_t w, data_type_t d) {
        return src_dt == s && weights_dt == w && dst_dt == d;
    };

    if (dt(f32, f32, f32)) return gemm<float, float, float>(args, {});

    if (dt(bf16, bf16, f32)) return gemm<bfloat16, bfloat16, float>(args, {});
    if (dt(bf16, bf16, bf16))
        return gemm<bfloat16, bfloat16, bfloat16>(args, {});

    if (dt(f16, f16, f16)) return gemm<__fp16, __fp16, __fp16>(args, {});
    if (dt(f16, f16, f32)) return gemm<__fp16, __fp16, float>(args, {});

    if (dt(s8, s8, s32))
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

    // Note that we shouldn't do fixed format for attr fastmath, because we end up modifying
    // the datatype as well as the format. This is technically not allowed in the documented API
    // but practically it does work because reorder accepts it. To maintain parity with ACL
    // we will leave this in for now, because it will require framework level changes
    // to untangle
    if (is_bf16_weight_format(wf)) md.data_type = data_type::bf16;
}

inline bool memory_desc_matches_weight_format(const memory_desc_t &md,
        kai::ops::WeightFormat wf, dim_t i_dim, dim_t o_dim,
        const std::vector<dim_t> &spatial_dims,
        const std::vector<dim_t> &batch_dims = {}) {
    memory_desc_t expected = md;
    weight_format_to_memory_desc(
            expected, wf, i_dim, o_dim, spatial_dims, batch_dims);

    return expected == md;
}

} // namespace kai_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
