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

// Common for RNN and LSTM cell execution

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

status_t ref_rnn_fwd_t::cell_execution(const cell_ctx_t &cell_struct) {

    auto cell_layer = cell_struct.workspace.states_range(cell_struct.lay,
            cell_struct.lay, cell_struct.dir, cell_struct.dir, cell_struct.iter,
            cell_struct.iter);

    auto iter_off = cell_struct.iter == 0
            ? (-1 * (cell_struct.rnn.n_dir - 1) * (cell_struct.rnn.n_iter + 1))
                    - 1
            : cell_struct.iter - 1;
    auto cell_iter = cell_struct.workspace.states_range(cell_struct.lay + 1,
            cell_struct.lay + 1, cell_struct.dir, cell_struct.dir, iter_off,
            iter_off);

    auto scratch_gates = cell_struct.scratch.gates(0);

    auto wei_layer
            = cell_struct.user_data.wei_layer(cell_struct.lay, cell_struct.dir);
    auto wei_iter
            = cell_struct.user_data.wei_iter(cell_struct.lay, cell_struct.dir);

    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, wei_layer,
            cell_layer, scratch_gates, matmul_layer_fwd));

    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, wei_iter,
            cell_iter, scratch_gates, matmul_iter_fwd));

    CHECK(rnn_bias(cell_struct.ctx, cell_struct.rnn.mb, cell_struct.rnn.dhc,
            cell_struct.iter, cell_struct.lay, cell_struct.dir,
            cell_struct.workspace, cell_struct.scratch, cell_struct.user_data));

    return status::success;
}

status_t ref_rnn_bwd_t::cell_execution(const cell_ctx_t &cell_struct) {

    auto wei_layer = cell_struct.user_data.wei_layer(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir);
    auto wei_iter = cell_struct.user_data.wei_iter(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir);

    auto ws_gates = cell_struct.workspace.gates(
            cell_struct.rnn.n_layer - cell_struct.lay, cell_struct.dir,
            cell_struct.rnn.n_iter - cell_struct.iter - 1);

    // take into account reading first layer from end of state
    // subsequent layers at written at forward pass'
    // last layer step location in state
    auto dsl_lay_off = cell_struct.iter == 0
            ? cell_struct.rnn.n_layer * 2 - cell_struct.lay - 1
            : cell_struct.rnn.n_layer - cell_struct.lay - 1;
    // take into account for bidirectional case
    // any timesteps after first one
    // write out from bottom up of timestep state location
    auto dsl_iter_offset = cell_struct.iter == 0 ? 0
            : cell_struct.dir == 1
            ? cell_struct.iter
            : cell_struct.rnn.n_iter - cell_struct.iter + 1;
    auto diff_cell_layer = cell_struct.scratch.diff_states(
            dsl_lay_off, cell_struct.dir, dsl_iter_offset);

    // account for bidirectional cases needing to jumping over
    //  n_iter state blocks
    auto dir_off = cell_struct.dir == 1
            ? cell_struct.iter + 1
            : cell_struct.rnn.n_iter - cell_struct.iter;
    auto diff_cell_iter = cell_struct.scratch.diff_states(
            cell_struct.rnn.n_layer - cell_struct.lay, cell_struct.dir,
            dir_off);

    auto wei_cell_layer = cell_struct.workspace.states_range(
            cell_struct.rnn.n_layer - 1 - cell_struct.lay,
            cell_struct.rnn.n_layer - 1 - cell_struct.lay, cell_struct.dir,
            cell_struct.dir, cell_struct.rnn.n_iter - cell_struct.iter - 1,
            cell_struct.rnn.n_iter - cell_struct.iter - 1);

    auto wci_offset = cell_struct.rnn.n_iter - cell_struct.iter - 2;

    if (cell_struct.rnn.n_dir == 2) {
        wci_offset = cell_struct.rnn.n_iter - cell_struct.iter - 1 == 0
                ? cell_struct.rnn.n_iter - cell_struct.iter
                        - cell_struct.rnn.n_iter - 3
                : cell_struct.rnn.n_iter - cell_struct.iter - 2;
    }

    auto wei_cell_iter = cell_struct.workspace.states_range(
            cell_struct.rnn.n_layer - cell_struct.lay,
            cell_struct.rnn.n_layer - cell_struct.lay, cell_struct.dir,
            cell_struct.dir, wci_offset, wci_offset);

    auto diff_gates = cell_struct.scratch.diff_gates(0);

    CHECK(rnn_bias(cell_struct.ctx, cell_struct.rnn.mb, cell_struct.rnn.dhc,
            cell_struct.iter, cell_struct.lay, cell_struct.dir,
            cell_struct.rnn.n_layer, diff_cell_layer, diff_cell_iter,
            cell_struct.user_data, ws_gates, diff_gates));

    auto dsi_offset = cell_struct.dir == 1
            ? cell_struct.iter + 1
            : cell_struct.rnn.n_iter - cell_struct.iter;

    auto diff_states_layer = cell_struct.scratch.diff_states(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir,
            dsi_offset, 0);

    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, wei_iter,
            diff_gates, diff_states_layer, matmul_iter_bwd));

    auto diff_states_iter = cell_struct.scratch.diff_states(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir,
            dsi_offset, 1);

    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, wei_layer,
            diff_gates, diff_states_iter, matmul_layer_bwd));

    auto diff_wei_layer = cell_struct.user_data.diff_wei_layer(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir);
    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, diff_gates,
            wei_cell_layer, diff_wei_layer, matmul_diff_wei_layer));

    auto diff_wei_iter = cell_struct.user_data.diff_wei_iter(
            cell_struct.rnn.n_layer - cell_struct.lay - 1, cell_struct.dir);
    CHECK(matmul_primitive(cell_struct.engine, cell_struct.ctx, diff_gates,
            wei_cell_iter, diff_wei_iter, matmul_diff_wei_iter));

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
