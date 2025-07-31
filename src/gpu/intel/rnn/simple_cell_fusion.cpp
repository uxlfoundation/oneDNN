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

#include "gpu/intel/rnn/simple_cell_fusion.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace rnn {

using namespace rnn_utils;
using namespace dnnl::impl::utils;

template <size_t out_ndims, size_t in_ndims>
strides_t<out_ndims> inner(const strides_t<in_ndims> &s) {
    static_assert(in_ndims >= out_ndims,
            "The output strides are expected to be smaller than the input "
            "strides");
    strides_t<out_ndims> ret;
    for (size_t i = 0; i < out_ndims; i++) {
        ret[i] = s[i + in_ndims - out_ndims];
    }
    return ret;
}

status_t compute_cell_fwd(const exec_ctx_t &ctx,
        const compute::kernel_t &kernel, dim_t lay, dim_t dir, dim_t iter,
        const workspace_t &workspace, const user_data_t user_data,
        const sub_buffer_t &weights_layer, const sub_buffer_t &weights_iter,
        const sub_buffer_t &cell_layer, const strides_t<4> &cell_layer_strides,
        const sub_buffer_t &cell_iter, const strides_t<4> &cell_iter_strides,
        const sub_buffer_t &scratch_gates,
        const strides_t<2> &scratch_gates_strides,
        const memory_storage_t &scratch_cell, float alpha,
        const memory_storage_t *tm_scales, const conf_t &conf,
        const ocl_conf_t &ocl_conf, const rnn_offsets_t &offsets) {
    auto &cell_conf = ocl_conf.cell_comp;
    const size_t dhc = conf.dhc;
    const size_t dhc_thr = cell_conf.dhc_thr;
    const size_t dhc_tg = cell_conf.dhc_tg;
    const size_t dhc_loop = utils::rnd_up(conf.dhc_loop, dhc_thr * dhc_tg);

    gpu_assert(dhc_tg % ocl_conf.subgroup_size == 0);

    const size_t mb = conf.mb;
    const size_t batch_tg = cell_conf.mb_tg;
    const size_t batch_thr = cell_conf.mb_thr;
    const size_t batch_local = batch_thr * batch_tg;
    compute::nd_range_t nd_range {
            {utils::div_up(dhc, dhc_loop) * dhc_tg,
                    utils::div_up(mb, batch_local) * batch_tg},
            {dhc_tg, batch_tg}};

    auto gates = workspace.gates(lay, dir, iter);
    auto gates_strides = workspace.gates_strides();
    auto states = workspace.states(lay, dir, iter);
    auto states_strides = workspace.states_strides();
    auto bias = user_data.bias(lay, dir);
    auto c_states_t_l = ocl_conf.cell_kind == alg_kind::vanilla_lstm
            ? workspace.c_states(lay, dir, iter)
            : sub_buffer_t();
    auto c_states_tm1_l = ocl_conf.cell_kind == alg_kind::vanilla_lstm
            ? workspace.c_states(lay, dir, iter - 1)
            : sub_buffer_t();
    auto h_states_tm_l = workspace.states(lay, dir, iter - 1);
    auto ws_grid = workspace.grid_comp(lay, dir, iter);

    arg_list_t arg_list;
    arg_list.append(weights_layer, ocl_conf.wei_dt);
    arg_list.append(offsets.weights_layer);
    arg_list.append(weights_iter, ocl_conf.wei_dt);
    arg_list.append(offsets.weights_iter);
    arg_list.append(cell_layer, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(cell_layer_strides));
    arg_list.append(cell_iter, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(cell_iter_strides));
    arg_list.append(gates, ocl_conf.aux_dt);
    arg_list.append(inner<2>(gates_strides));
    arg_list.append(states, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(states_strides));
    arg_list.append(scratch_cell);

    if (ocl_conf.cell_kind == alg_kind::vanilla_lstm) {
        arg_list.append(c_states_t_l, ocl_conf.aux_dt);
        arg_list.append(c_states_tm1_l, ocl_conf.aux_dt);
        arg_list.append(conf.tm_cscale);
    }

    if (ocl_conf.cell_kind == alg_kind::lbr_gru) {
        arg_list.append(h_states_tm_l, ocl_conf.ws_state_dt);
        arg_list.append(ws_grid, ocl_conf.aux_dt);
    }

    if (!(cell_conf.compute_gemm_layer && cell_conf.compute_gemm_iter)
            || (ocl_conf.cell_kind == alg_kind::lbr_gru)) {
        arg_list.append(scratch_gates, ocl_conf.aux_dt);
        arg_list.append(scratch_gates_strides);
    }

    if (cell_conf.enable_iter_block) { arg_list.append(conf.iter_loop); }

    arg_list.append(bias, ocl_conf.bia_dt);
    arg_list.append(alpha);
    arg_list.append(get_storage(tm_scales));
    arg_list.append(conf.mb);
    arg_list.append(conf.dhc);
    arg_list.append(conf.slc);
    arg_list.append(conf.sic);

    arg_list.append(into<dim_t>(dhc_loop));

    return primitive_t::parallel_for(ctx, nd_range, kernel, arg_list.args);
}

} // namespace rnn
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
