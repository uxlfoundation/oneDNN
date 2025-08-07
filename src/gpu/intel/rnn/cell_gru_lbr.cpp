/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

/*
 * Cell execution GRU with linear before reset
 */

#include "gpu/intel/rnn/grid.hpp"
#include "gpu/intel/rnn/simple_cell_fusion.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace rnn {

using namespace utils;

template cell_execution_sig(simple_fwd_t::cell_execution_gru_lbr);
template cell_execution_sig(simple_bwd_t::cell_execution_gru_lbr);

template <prop_kind_t aprop>
cell_execution_sig((simple_common_t<aprop>::cell_execution_gru_lbr)) {
    const conf_t &conf = this->pd()->conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const offsets_t &offsets = this->pd()->off;

    const bool use_cell = ocl_conf.cell_comp.is_enabled;

    strides_t<4> user_layer_strides {[&]() {
        auto s = user_data.src_layer_strides(dir);
        return strides_t<4> {0, 0, s[0], s[1]};
    }()};

    auto cell_layer = !conf.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states(lay - 1, dir, iter);
    auto &cell_layer_strides = !conf.copy_src_layer && lay == 0
            ? user_layer_strides
            : workspace.states_strides();
    auto cell_iter = workspace.states(lay, dir, iter - 1);
    auto cell_iter_strides = workspace.states_strides();

    auto gemm_cell_layer_fwd = !conf.copy_src_layer && lay == 0
            ? gemm_layer_fwd_src
            : gemm_layer_fwd;
    auto gemm_diff_wei_cell_layer = !conf.copy_src_layer && lay == 0
            ? gemm_diff_wei_layer_src
            : gemm_diff_wei_layer;

    auto scratch_gates = scratch.gates(iter);
    strides_t<2> scratch_gates_strides
            = {scratch.calc_off_gates(1), conf.scratch_gates_ld};
    auto &scratch_cell = scratch.cell() ? *scratch.cell()
                                        : memory_storage_t::empty_storage();

    auto wei_layer = user_data.wei_layer(lay, dir);
    auto wei_iter = user_data.wei_iter(lay, dir);

    if (aprop == prop_kind::forward) {
        // call made when cell execution is enabled
        if (!conf.merge_gemm_layer && !conf.cell_fusion.gemm_layer)
            CHECK(gemm_primitive(engine, ctx, wei_layer, cell_layer,
                    scratch_gates, gemm_cell_layer_fwd));

        if (!conf.cell_fusion.gemm_iter)
            CHECK(gemm_primitive(engine, ctx, wei_iter, cell_iter, scratch_cell,
                    gemm_iter_fwd));

        if (!use_cell) {
            CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, conf.dhc,
                    conf.mb, 1, user_data, workspace, scratch_gates, {},
                    scratch_cell, {}, {}, {}, 0, tm_scales, diff_bias));
        } else {

            CHECK(compute_cell_fwd(ctx, kernels_[kernel_id::cell_fwd], lay, dir,
                    iter, workspace, user_data, wei_layer, wei_iter, cell_layer,
                    cell_layer_strides, cell_iter, cell_iter_strides,
                    scratch_gates, scratch_gates_strides, scratch_cell,
                    pd()->desc()->alpha, tm_scales, conf, ocl_conf, offsets));
        }
    } else {
        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_layer
                = !conf.copy_diff_dst_layer && lay + 1 == conf.n_layer
                ? user_data.diff_dst_layer(dir, iter)
                : scratch.diff_states(lay + 1, dir, conf.n_states, iter);
        auto diff_states_layer_ld
                = !conf.copy_diff_dst_layer && lay + 1 == conf.n_layer
                ? offsets.diff_dst_layer[1]
                : conf.scratch_diff_states_ld;

        auto diff_states = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states1 = !conf.copy_diff_src_layer && lay == 0
                ? user_data.diff_src_layer(dir, iter)
                : scratch.diff_states(lay, dir, conf.n_states, iter);

        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, conf.dhc, conf.mb,
                ocl_conf.elemwise_bwd_batch_block, user_data, workspace,
                scratch_gates, diff_gates, scratch_cell, diff_states,
                diff_states_iter, diff_states_layer, diff_states_layer_ld,
                tm_scales, diff_bias));

        if (!conf.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_layer,
                    user_data.diff_wei_layer(lay, dir),
                    gemm_diff_wei_cell_layer));

            auto gemm_layer_cell_bwd = !conf.copy_diff_src_layer && lay == 0
                    ? gemm_layer_bwd_src
                    : gemm_layer_bwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, diff_gates,
                    diff_states1, gemm_layer_cell_bwd));
        }

        CHECK(gemm_primitive(engine, ctx, wei_iter, scratch_cell, diff_states,
                gemm_iter_bwd));

        CHECK(gemm_primitive(engine, ctx, scratch_cell, cell_iter,
                user_data.diff_wei_iter(lay, dir), gemm_diff_wei_iter));
    }
    return status::success;
}

} // namespace rnn
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
