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

#ifndef GPU_INTEL_RNN_CONFIG_HPP
#define GPU_INTEL_RNN_CONFIG_HPP

#include "common/serialization.hpp"
#include "gpu/gpu_rnn_pd.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace rnn {

using pd_t = rnn_pd_t;
using fwd_pd_t = gpu_rnn_fwd_pd_t;
using bwd_pd_t = gpu_rnn_bwd_pd_t;
using desc_t = rnn_desc_t;

struct offsets_t {
    strides_t<3> src_layer;
    strides_t<4> src_iter;
    strides_t<4> src_iter_c;
    strides_t<5> weights_layer;
    strides_t<5> weights_iter;
    dim_t weights_layer_comp_off;
    dim_t weights_iter_comp_off;
    strides_t<4> bias;
    strides_t<3> dst_layer;
    strides_t<4> dst_iter;
    strides_t<4> dst_iter_c;
    strides_t<3> diff_src_layer;
    strides_t<4> diff_src_iter;
    strides_t<4> diff_src_iter_c;
    strides_t<5> diff_weights_layer;
    strides_t<5> diff_weights_iter;
    strides_t<4> diff_bias;
    strides_t<3> diff_dst_layer;
    strides_t<4> diff_dst_iter;
    strides_t<4> diff_dst_iter_c;
};

namespace utils {

using namespace impl::utils;

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum data_type_conf_t {
    all_f32,
    all_f16,
    all_bf16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8
};

} // namespace utils

struct ocl_conf_t {
    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {

        compute::kernel_ctx_t kernel_ctx;
        CHECK(init_kernel_ctx(kernel_ctx));
        return engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
    }
    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names = {"rnn_bias_prepare",
                "simple_rnn_copy_init_layer", "simple_rnn_copy_init_iter",
                "simple_rnn_copy_res_layer", "simple_rnn_copy_res_iter",
                "simple_rnn_elemwise_fwd", "simple_rnn_elemwise_bwd",
                "simple_rnn_cell_fwd"};
        return names;
    }

#if __cplusplus >= 202002L
    bool operator==(const ocl_conf_t &) const = default;
#endif
    serialization_stream_t serialize() const {
        DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ocl_conf_t);
        return serialization_stream_t(*this);
    }

    static ocl_conf_t deserialize(const serialization_stream_t &s) {
        ocl_conf_t t {};
        deserializer_t d(s);
        d.pop(t);
        return t;
    }

    status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

    int threads_per_eu = 0;
    int subgroup_size = 0;
    int cell_kind = 0;
    int activation_kind = 0;
    int direction_kind = 0;

    data_type_t src_dt = data_type::undef;
    data_type_t src_c_dt = data_type::undef;
    data_type_t wei_dt = data_type::undef;
    data_type_t bia_dt = data_type::undef;
    data_type_t dst_dt = data_type::undef;
    data_type_t dst_c_dt = data_type::undef;
    data_type_t acc_dt = data_type::undef;
    data_type_t aux_dt = data_type::undef;
    data_type_t ws_state_dt = data_type::undef;
    data_type_t input_dt = data_type::undef;
    data_type_t output_dt = data_type::undef;
    data_type_t diff_dt = data_type::undef;
    uint8_t pad[4] = {};

    struct inner_layouts_t {
#if __cplusplus >= 202002L
        bool operator==(const inner_layouts_t &) const = default;
#endif
        block_layout_t src_layer;
        block_layout_t src_iter;
        block_layout_t src_iter_c;
        block_layout_t weights_layer;
        block_layout_t weights_iter;
        block_layout_t bias;
        block_layout_t dst_layer;
        block_layout_t dst_iter;
        block_layout_t dst_iter_c;
        block_layout_t diff_src_layer;
        block_layout_t diff_src_iter;
        block_layout_t diff_src_iter_c;
        block_layout_t diff_weights_layer;
        block_layout_t diff_weights_iter;
        block_layout_t diff_bias;
        block_layout_t diff_dst_layer;
        block_layout_t diff_dst_iter;
        block_layout_t diff_dst_iter_c;
    };

    inner_layouts_t inner_layouts = {};

    int wei_qparam_mask = 0;

    int elemwise_bwd_batch_block = 0;
    bool need_bias_atomic_reduce = false;
    bool with_bias = false;
    bool with_src_iter = false;
    bool with_src_iter_c = false;
    bool with_dst_iter = false;
    bool with_dst_iter_c = false;
    bool is_fwd = false;
    bool copy_bias = false;
    bool is_int8 = false;
    bool is_testmode = false;
    bool is_training = false;
    bool recompute_gates = false;
    bool copy_src_layer = false;
    bool copy_diff_dst_layer = false;
    bool copy_diff_src_layer;
    bool deterministic = false;
    bool require_stateless_addressing = true;
    uint8_t pad2[7] = {};
    struct comp_conf_t {
        bool is_enabled = false;
        bool compute_gemm_layer = false;
        bool gemm_layer_k_tail = false;
        bool compute_gemm_iter = false;
        bool gemm_iter_k_tail = false;
        bool dhc_tail = false;
        bool mb_tail = false;
        bool enable_iter_block = false;
        int dhc_thr = 0;
        int dhc_tg = 0;
        int mb_thr = 0;
        int mb_tg = 0;
#if __cplusplus >= 202002L
        bool operator==(const comp_conf_t &) const = default;
#endif
    } cell_comp;
};

struct conf_t {
    utils::execution_direction_t exec_dir;
    utils::data_type_conf_t dt_conf;
    dim_t n_layer, n_iter, n_dir, n_gates, n_states;
    dim_t mb;
    dim_t slc, sic, dhc, dlc, wic;

    dim_t gates_ld, gates_ws_ld, arch_ld;

    dim_t n_parts_weights_layer, parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    dim_t n_parts_weights_iter, parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    dim_t n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    dim_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS],
            part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    // Size of packed data in bytes
    dim_t weights_layer_comp_offset, weights_layer_pack_size,
            weights_iter_comp_offset, weights_iter_pack_size;

    struct {
        bool gemm_layer;
        bool gemm_iter;
    } cell_fusion;

    dim_t dhc_loop;
    dim_t iter_loop;

    bool copy_bias;
    dim_t states_ws_ld, scratch_diff_states_ld;
    bool is_fwd, is_training, is_lbr, is_int8, is_testmode, is_vanilla_gru;
    bool use_workspace;
    bool recompute_gates;
    bool copy_src_layer;
    bool copy_diff_dst_layer;
    bool copy_diff_src_layer;

    // for test mode (--skip_nonliner=true of benchdnn)
    float tm_cscale;
    dim_t tm_ngates;

    // Size of workspace for each tensor in bytes
    dim_t ws_states_cell_size, ws_c_states_cell_size, ws_gates_cell_size;
    dim_t ws_gates_size, ws_states_size, ws_c_states_size,
            scratch_diff_states_size, scratch_cell_size, scratch_dhG1_size,
            ws_grid_comp_size, ws_per_cell, ws_bias_size;

    dim_t ws_gates_offset;
    dim_t ws_states_offset;
    dim_t ws_grid_comp_offset;
    dim_t ws_c_state_offset;
    dim_t ws_bias_offset;

    bool merge_gemm_iter, merge_gemm_layer, use_gemm, use_layer_packed_gemm,
            use_iter_packed_gemm;

    // Element size of each workspace part in bytes
    dim_t ws_gates_elsz, ws_states_elsz, ws_grid_comp_elsz, ws_bias_elsz;

    dim_t n_iter_scratch_gates;
    dim_t scratch_gates_size, scratch_gates_elsz, scratch_gates_ld;
    dim_t scratch_diff_gates_size, scratch_diff_gates_elsz,
            scratch_diff_gates_ld;

    data_type_t acc_data_type;
    dim_t acc_data_type_elsz;
    data_type_t aux_data_type;
    data_type_t input_data_type;
    data_type_t output_data_type;
    data_type_t src_data_type;
    data_type_t dst_data_type;
    data_type_t diff_data_type;
    data_type_t wei_layer_type;
    data_type_t wei_iter_type;
    data_type_t bias_data_type;
};

struct reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;
    compute::dispatch_t dispatch;
    int sub_group_size;
    int mask;
    size_t scales_count;
};

} // namespace rnn
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
