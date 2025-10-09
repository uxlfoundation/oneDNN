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

#include "gpu/intel/rnn/utils.hpp"

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace rnn {
namespace utils {

using namespace intel::gpu_utils;
using namespace intel::compute;
using namespace prop_kind;
using namespace data_type;

bool is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

void init_conf(conf_t &conf, const desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &bias_d, data_type_t acc_data_t,
        const device_info_t &device_info) {

    bool is_xe_hpc = device_info.gpu_arch() == gpu_arch_t::xe_hpc;
    conf = utils::zero<decltype(conf)>();
    conf.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    conf.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    conf.is_lbr = rd.cell_kind == dnnl_lbr_gru;
    conf.is_vanilla_gru = rd.cell_kind == dnnl_vanilla_gru;
    conf.arch_ld = is_xe_hpc ? 128 : 64;

    switch (rd.direction) {
        case dnnl_unidirectional_left2right: conf.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: conf.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: conf.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: conf.exec_dir = bi_sum; break;
        default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        conf.dt_conf = all_f32;
    else if (everyone_is(bf16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        conf.dt_conf = all_bf16;
    else if (everyone_is(f16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        conf.dt_conf = all_f16;
    else if (dst_layer_d.data_type() == u8) {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            conf.dt_conf = u8u8u8u8;
        else
            conf.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            conf.dt_conf = u8u8u8f32;
        else
            conf.dt_conf = f32u8f32f32;
    }
    conf.is_int8 = !one_of(conf.dt_conf, all_f32, all_f16, all_bf16);

    conf.aux_data_type
            = acc_data_t == data_type::f16 ? data_type::f16 : data_type::f32;
    conf.diff_data_type = diff_dst_layer_d.data_type();

    conf.acc_data_type = acc_data_t;
    conf.acc_data_type_elsz = types::data_type_size(acc_data_t);

    conf.wei_layer_type = weights_layer_d.data_type();
    conf.wei_iter_type = weights_iter_d.data_type();

    conf.n_layer = weights_layer_d.dims()[0];
    conf.n_iter = src_layer_d.dims()[0];
    conf.n_dir = weights_layer_d.dims()[1];
    conf.n_gates = weights_layer_d.dims()[3];
    conf.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    conf.n_bias = conf.n_gates + conf.is_lbr;
    conf.mb = src_layer_d.dims()[1];
    conf.sic = weights_iter_d.dims()[2];
    conf.slc = weights_layer_d.dims()[2];
    conf.dhc = weights_layer_d.dims()[4];
    conf.dlc = dst_layer_d.dims()[2];
    conf.wic = nstl::max(conf.slc, nstl::max(conf.sic, conf.dhc));

    conf.gates_ld = conf.dhc * conf.n_gates;

    // Set the correct number of weights parts
    conf.n_parts_weights_layer = 1;
    conf.parts_weights_layer[0] = conf.n_gates;
    conf.parts_weights_layer[1] = 0;

    //there are two parts for VANILLA GRU weights iteration
    conf.n_parts_weights_iter = conf.is_vanilla_gru ? 2 : 1;
    conf.parts_weights_iter[0] = conf.is_vanilla_gru ? 2 : conf.n_gates;
    conf.parts_weights_iter[1] = conf.is_vanilla_gru ? 1 : 0;

    conf.n_parts_bias = 1;
    conf.parts_bias[0] = conf.n_bias;
    conf.parts_bias[1] = 0;

    bool is_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::lbr_gru);

    // Decide if to merge gemm across iterations or layers
    auto dst_layer_is_trivial_stride = dst_layer_d.dims()[0] <= 1
            || dst_layer_d.dims()[1] <= 1
            || (dst_layer_d.blocking_desc().strides[0]
                    == (dst_layer_d.blocking_desc().strides[1] * conf.mb));

    // Does not account for alignment striding
    dim_t merge_scratch_size_estimate = conf.gates_ld * conf.mb * conf.n_iter;
    bool is_small_scratch = merge_scratch_size_estimate < 256 * 1024 * 1024;
    conf.merge_gemm_layer = dev_getenv("merge_gemm_layer",
            is_small_scratch); // Avoid excessive memory usage
    conf.merge_gemm_iter = dev_getenv("merge_gemm_iter",
            is_small_scratch && dst_layer_is_trivial_stride
                    && !(conf.is_fwd || is_gru));

    if (conf.is_fwd) {
        bool can_fuse_gemm = !conf.is_int8
                && conf.wei_iter_type == conf.wei_layer_type && conf.is_fwd
                && utils::one_of(rd.cell_kind, alg_kind::vanilla_rnn,
                        alg_kind::vanilla_lstm, alg_kind::lbr_gru);

        // Poor implementation performance if dhc % subgroup_size != 0
        bool tail_dhc = conf.dhc % device_info.min_subgroup_size() != 0;
        // Since RNN cells may result in very small workloads the CPU overhead
        // to dispatch kernels may be significant. As such, if the work per eu
        // is too small, we need to fuse kernel operations to reduce CPU
        // workload.
        dim_t fuse_gemm_limit = [&]() {
            const dim_t work_threshold = tail_dhc ? 512 : 1024;
            return work_threshold * device_info.eu_count()
                    * device_info.max_subgroup_size(conf.acc_data_type);
        }();

        // For large enough k dimension, parallelization in external gemm
        // kernels is more performant.
        int min_k = nstl::min((int)conf.sic, (int)conf.slc);
        int ideal_k = math::lcm((int)device_info.min_subgroup_size(), min_k);
        if (rd.cell_kind == alg_kind::vanilla_lstm) {
            min_k = (min_k <= 256) ? 160 : 256;
        } else {
            min_k = device_info.mayiuse_systolic() ? 64 : 256;
        }
        dim_t k_limit = tail_dhc ? 50 : nstl::min(min_k, ideal_k);

        // The fused gemm implementation assumes the dst channel dimension is
        // dense
        auto is_dense_dst_c = [](const memory_desc_wrapper &md) {
            if (md.format_kind() == format_kind::any) return true;
            if (md.format_kind() != format_kind::blocked) return false;
            if (md.dims()[4] == 1) return true;
            if (md.blocking_desc().strides[4] == 1) return true;
            return false;
        };

        conf.cell_fusion.gemm_iter
                = dev_getenv("fuse_gemm_iter",
                          !conf.merge_gemm_iter
                                  && conf.dhc * conf.sic * conf.mb
                                                  * conf.n_gates
                                          < fuse_gemm_limit
                                  && conf.sic <= k_limit
                                  && is_dense_dst_c(weights_layer_d))
                && can_fuse_gemm;
        conf.cell_fusion.gemm_layer
                = dev_getenv("fuse_gemm_layer",
                          conf.cell_fusion.gemm_iter && !conf.merge_gemm_layer
                                  && conf.dhc * conf.slc * conf.mb
                                                  * conf.n_gates
                                          < fuse_gemm_limit
                                  && conf.slc <= k_limit
                                  && is_dense_dst_c(weights_iter_d))
                && can_fuse_gemm;

        // Currently, external gemm_iter always accumulates in C. As such,
        // external gemm_layer is required to initialize the memory.
        gpu_assert(IMPLICATION(
                conf.cell_fusion.gemm_layer, conf.cell_fusion.gemm_iter));

        bool can_iter_loop = conf.cell_fusion.gemm_iter
                && (conf.merge_gemm_layer || conf.cell_fusion.gemm_layer);

        const int loop_all = 0;
        conf.iter_loop = dev_getenv("iter_loop", can_iter_loop ? loop_all : 1);
        if (conf.iter_loop == loop_all) conf.iter_loop = conf.n_iter;

        conf.dhc_loop = dev_getenv("dhc_loop", conf.iter_loop ? loop_all : 1);
        if (conf.dhc_loop == loop_all) conf.dhc_loop = conf.dhc;

        // A synchronization point is required after cell computation on along
        // the dhc dimension. This requires dhc to be calculated on one thread
        // group.
        gpu_assert(IMPLICATION(conf.iter_loop, conf.dhc_loop == conf.dhc));
    } else {
        conf.iter_loop = 1;
        conf.dhc_loop = 1;
    }

    // Decide to copy bias
    conf.copy_bias = conf.is_int8;

    conf.use_workspace = conf.is_training;

    conf.src_data_type = src_layer_d.data_type();
    conf.input_data_type = src_iter_d.data_type();
    conf.bias_data_type = bias_d.data_type();
    conf.dst_data_type = dst_layer_d.data_type();
    conf.output_data_type = dst_iter_d.data_type();

    // Assign types for optional parameters for improved kernel reuse.
    if (conf.input_data_type == data_type::undef)
        conf.input_data_type = conf.src_data_type;
    if (conf.output_data_type == data_type::undef)
        conf.output_data_type = conf.dst_data_type;
}

void init_test_mode(conf_t &conf, const primitive_attr_t &attr) {
    conf.is_testmode = attr.rnn_tparams_.test_mode_;
    conf.tm_ngates = attr.rnn_tparams_.ngates_;
    conf.tm_cscale = attr.rnn_tparams_.cscale_;
}

void set_conf(conf_t &conf, const desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &diff_src_layer_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    const bool is_fwd = conf.is_fwd;
    const bool is_bwd = !conf.is_fwd;

    dim_t aux_elsz = into<dim_t>(types::data_type_size(conf.aux_data_type));
    conf.ws_states_elsz = types::data_type_size(conf.src_data_type);

    conf.scratch_gates_elsz = types::data_type_size(conf.acc_data_type);
    conf.scratch_diff_gates_elsz
            = is_bwd ? types::data_type_size(conf.src_data_type) : 0;

    // Set workspace sizes to store:
    // states to copmute a pass
    // diff states to copmute bwd pass (training only)
    // intermediate results from the gates
    conf.states_ws_ld = get_good_ld(conf.arch_ld,
            nstl::max(conf.slc, nstl::max(conf.sic, conf.dhc)),
            conf.ws_states_elsz);
    conf.gates_ws_ld = get_good_ld(conf.arch_ld, conf.gates_ld, aux_elsz);
    // Disable associativity check on some large problems to reduce memory
    // usage. Can be removed when further improvements are made to
    // copy_diff_src_layer
    conf.scratch_diff_states_ld = get_good_ld(conf.arch_ld,
            nstl::max(conf.slc, nstl::max(conf.sic, conf.dhc)), sizeof(float),
            utils::everyone_is(conf.slc, conf.sic, conf.dhc)
                    && conf.n_layer * conf.n_dir * conf.n_iter * conf.mb
                            > 128 * 1024);

    conf.scratch_gates_ld
            = get_good_ld(conf.arch_ld, conf.gates_ld, conf.scratch_gates_elsz);
    conf.scratch_diff_gates_ld = is_bwd
            ? get_good_ld(
                    conf.arch_ld, conf.gates_ld, conf.scratch_diff_gates_elsz)
            : 0;

    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;

    bool require_copy_src_layer = [&]() {
        auto &strides = src_layer_d.strides();
        auto &pdims = src_layer_d.padded_dims();
        auto dt_size = types::data_type_size(src_layer_d.data_type());

        // The GEMM interface assumes input buffers are well aligned. We need to
        // implement a way to avoid kernels relying on this alignment to remove
        // this restriction.
        if (pdims[0] > 1 && (strides[0] * dt_size) % 8) return true;

        if (conf.merge_gemm_layer) {
            // GEMM inputs are represented as 2d inputs. As such, the merged
            // dimension need to be dense. This restriction could be removed by
            // using batched GEMM with appropriate strides instead.
            constexpr int iter_dim = 0, mb_dim = 1;
            if (pdims[iter_dim] > 1 && pdims[mb_dim] > 1
                    && (strides[iter_dim] != strides[mb_dim] * conf.mb))
                return true;
            if (conf.exec_dir != utils::l2r) return true;
        }

        // Bug workaround, likely related to the undefined mb stride
        if (pdims[1] == 1) return true;

        return false;
    }();

    bool prefer_copy_src_layer = [&]() {
        auto &strides = src_layer_d.strides();
        auto &pdims = src_layer_d.padded_dims();
        auto dt_size = types::data_type_size(src_layer_d.data_type());

        // Data is already well aligned. Copying does not provide benefit
        if (pdims[1] == 1 || strides[1] == conf.gates_ws_ld
                || (strides[1] % 64 == 0))
            return false;

        // Better to rely on GEMM to emit reorder if it is necessary if there is
        // limited data reuse
        const dim_t data_reuse = conf.n_dir * (conf.is_training ? 2 : 1);
        if (data_reuse < 2) return false;

        // Prefer lower memory usage
        if (src_layer_d.nelems(true) * dt_size >= 1024 * 1024 * 1024)
            return false;

        return true;
    }();

    bool copy_src_layer = dev_getenv("copy_src_layer", prefer_copy_src_layer)
            || require_copy_src_layer;

    bool require_copy_diff_dst_layer = [&]() {
        if (is_fwd) return false;

        auto &strides = diff_dst_layer_d.strides();
        auto &pdims = diff_dst_layer_d.padded_dims();
        auto dt_size = types::data_type_size(diff_dst_layer_d.data_type());

        // The GEMM interface assumes input buffers are well aligned. We need to
        // implement a way to avoid kernels relying on this alignment to remove
        // this restriction.
        if (pdims[0] > 1 && (strides[0] * dt_size) % 8) return true;

        if (conf.merge_gemm_layer) {
            // GEMM inputs are represented as 2d inputs. As such, the merged
            // dimension need to be dense. This restriction could be removed by
            // using batched GEMM with appropriate strides instead.
            constexpr int iter_dim = 0, mb_dim = 1;
            if (pdims[iter_dim] > 1 && pdims[mb_dim] > 1
                    && (strides[iter_dim] != strides[mb_dim] * conf.mb))
                return true;
            if (conf.exec_dir != utils::r2l) return true;
        }

        // Bug workaround, likely related to the undefined mb stride
        if (pdims[1] == 1) return true;

        return false;
    }();
    bool copy_diff_dst_layer = dev_getenv("copy_diff_dst_layer", false)
            || require_copy_diff_dst_layer;

    bool require_copy_diff_src_layer = [&]() {
        if (is_fwd) return false;

        // Unimplemented
        if (conf.merge_gemm_iter || conf.merge_gemm_layer) return true;

        // Direction need to be summed together. This requires generating new
        // GEMM kernels to perform a sum accumulation for the final accumulation.
        if (conf.n_dir > 1) return true;

        return false;
    }();
    bool copy_diff_src_layer = dev_getenv("copy_diff_src_layer", false)
            || require_copy_diff_src_layer;
    conf.copy_src_layer = copy_src_layer;
    conf.copy_diff_dst_layer = copy_diff_dst_layer;
    conf.copy_diff_src_layer = copy_diff_src_layer;
    conf.ws_states_cell_size
            = conf.mb * conf.states_ws_ld * conf.ws_states_elsz;
    conf.ws_states_size = (copy_src_layer ? conf.n_layer + 1 : conf.n_layer)
            * conf.n_dir * (conf.n_iter + 1) * conf.ws_states_cell_size;

    // we do not need a good ld for iter_c as it is not involved in GEMM
    // for now reverting it back to what it was originally
    // TODO: seprate diff_c_offsets from diff-states & seprate h- and c- off
    conf.ws_c_states_cell_size
            = is_lstm ? conf.mb * conf.states_ws_ld * aux_elsz : 0;
    conf.ws_c_states_size = is_lstm ? conf.n_layer * conf.n_dir
                    * (conf.n_iter + 1) * conf.ws_c_states_cell_size
                                    : 0;

    auto scratch_diff_n_states
            = copy_diff_dst_layer || copy_diff_src_layer || conf.n_layer != 1
            ? conf.n_states + 1
            : conf.n_states;
    // conf.n_layer > 1 is currently required due to copy_{init,res}_iter
    bool have_result_layer = copy_diff_src_layer || conf.n_layer > 1;
    auto scratch_diff_n_layer
            = conf.n_layer - 1 + copy_diff_dst_layer + have_result_layer;

    // Due to the grid iteration used, if no full layers are required, only use
    // 2 cells, one for the previous iteration and one for the current
    // iteration.
    auto scratch_diff_n_cells = is_bwd
            ? (scratch_diff_n_layer > 0
                              ? scratch_diff_n_layer * (conf.n_iter + 1)
                              : 2)
                    * scratch_diff_n_states * conf.n_dir
            : 0;
    conf.scratch_diff_states_size = scratch_diff_n_cells * conf.mb
            * conf.scratch_diff_states_ld * aux_elsz;

    conf.ws_gates_cell_size = conf.mb * conf.gates_ws_ld * aux_elsz;
    conf.ws_gates_size = conf.is_training ? (conf.n_layer * conf.n_dir
                                 * conf.n_iter * conf.ws_gates_cell_size)
                                          : 0;

    // Reduce workspace memory by recomputing gates for bwd
    // TODO: Extend this optimization to other alg_kind.
    bool supports_recompute_gates
            = utils::one_of(rd.cell_kind, alg_kind::vanilla_lstm,
                      alg_kind::vanilla_rnn)
            && conf.is_training;
    bool prefer_recompute_gates = conf.ws_gates_size >= 512 * 1024 * 1024;
    conf.recompute_gates = dev_getenv("recompute_gates", prefer_recompute_gates)
            && supports_recompute_gates;
    if (conf.recompute_gates) conf.ws_gates_size = 0;

    conf.n_iter_scratch_gates
            = (conf.merge_gemm_layer || conf.merge_gemm_iter) ? conf.n_iter : 1;

    // To reduce memory usage, use scratch_diff_gates in place of scratch_gates
    // when the layout is the same, i.e. they have the same data type size.
    bool need_scratch_gates = is_fwd
            || (conf.recompute_gates
                    && conf.scratch_gates_elsz != conf.scratch_diff_gates_elsz);
    conf.scratch_gates_size = need_scratch_gates ? conf.n_iter_scratch_gates
                    * conf.mb * conf.scratch_gates_ld * conf.scratch_gates_elsz
                                                 : 0;
    conf.scratch_diff_gates_size = is_bwd ? conf.n_iter_scratch_gates * conf.mb
                    * conf.scratch_diff_gates_ld * conf.scratch_diff_gates_elsz
                                          : 0;
    conf.scratch_dhG1_size = (rd.cell_kind == alg_kind::vanilla_gru && is_bwd)
            ? conf.mb * conf.scratch_diff_states_ld * sizeof(float)
            : 0;
    conf.ws_bias_size
            = conf.n_layer * conf.n_dir * conf.n_bias * conf.dhc * aux_elsz;

    // For intermediate step in post-gemm fwd lbr gru
    conf.scratch_cell_size = [&]() {
        if (conf.is_lbr && is_fwd) {
            return conf.mb * conf.scratch_gates_ld * conf.scratch_gates_elsz;
        } else if (conf.is_lbr && is_bwd) {
            return conf.mb * conf.scratch_diff_gates_ld
                    * conf.scratch_diff_gates_elsz;
        } else if (rd.cell_kind == alg_kind::vanilla_gru && is_bwd) {
            return conf.mb * conf.states_ws_ld * conf.ws_states_elsz;
        } else {
            return static_cast<dim_t>(0);
        }
    }();

    // Used for storing the intermediate value from fwd pass in training lbr gru
    conf.ws_per_cell = conf.is_lbr * conf.mb * conf.dhc * aux_elsz;
    conf.ws_grid_comp_size = conf.is_lbr * conf.is_training * conf.n_layer
            * conf.n_dir * conf.n_iter * conf.ws_per_cell;

    set_workspace_offsets(conf, conf.ws_gates_offset, conf.ws_states_offset,
            conf.ws_c_state_offset, conf.ws_grid_comp_offset,
            conf.ws_bias_offset);
}

dim_t get_good_ld(
        dim_t arch_ld, dim_t dim, dim_t sizeof_dt, bool ignore_assoc) {
    // Leading dimension for matrices has 64-byte or 128-byte alignment (PVC-A)
    dim_t ld = rnd_up(dim, arch_ld / sizeof_dt);
    // Further alignment is associated with 8-way associativity of L1-cache
    return (ld % 256 == 0) && !ignore_assoc ? ld + arch_ld / sizeof_dt : ld;
}

dim_t set_workspace_offsets(const conf_t &conf, dim_t &ws_gates_offset,
        dim_t &ws_states_offset, dim_t &ws_c_states_offset,
        dim_t &ws_grid_comp_offset, dim_t &ws_bias_offset) {

    const dim_t page_size = 4096;
    dim_t current_offset = 0;

#define register_space(a) \
    do { \
        current_offset = utils::rnd_up(current_offset, page_size); \
        CONCAT2(a, _offset) = current_offset; \
        current_offset += conf.CONCAT2(a, _size); \
    } while (false)

    // Mandatory workspaces: go to workspace if use_workspace, scratchpad
    // otherwise assumes the workspace base pointer is page aligned
    register_space(ws_gates);
    register_space(ws_states);
    register_space(ws_c_states);
    register_space(ws_grid_comp);

    ws_bias_offset = 0;
    if (conf.copy_bias) { register_space(ws_bias); }
    return current_offset;
}

dim_t get_workspace_size(const conf_t &conf) {
    dim_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_grid_comp_offset, ws_bias_offset;
    return set_workspace_offsets(conf, ws_gates_offset, ws_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, ws_bias_offset);
}

status_t set_good_strides(
        dim_t ld_, memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;
    using namespace format_tag;

    if (tag == ldigo) {
        strides[2] = utils::get_good_ld(
                ld_, strides[2], types::data_type_size(weights_md.data_type));
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[4] = utils::get_good_ld(
                ld_, strides[4], types::data_type_size(weights_md.data_type));
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t set_weights_desc(memory_desc_t &weights_md, const conf_t &conf) {
    using namespace format_tag;
    if (weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, conf.is_fwd ? ldigo : ldgoi));

        // Adjust strides for good leading dimension in GEMM
        CHECK(set_good_strides(
                conf.arch_ld, weights_md, conf.is_fwd ? ldigo : ldgoi));

        // set we need extra memory
        if (conf.is_fwd && conf.is_int8) {
            weights_md.extra.flags = memory_extra_flags::rnn_u8s8_compensation;
            weights_md.extra.compensation_mask = 27; // ldigo 11011;
        }
        return status::success;
    } else if (weights_md.format_kind != format_kind::blocked) {
        // This implementation only supports blocked memory
        return status::unimplemented;
    } else if (conf.is_fwd && conf.is_int8) {
        // Int8 RNN requires extra memory on weights buffers for the
        // compensations
        return status::unimplemented;
    }
    return status::success;
}

const memory_storage_t &get_storage(const memory_storage_t *storage) {
    return storage ? *storage : memory_storage_t::empty_storage();
}
const memory_storage_t &get_storage(
        const std::unique_ptr<memory_storage_t> &storage) {
    return get_storage(storage.get());
}

} // namespace utils
} // namespace rnn
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
