/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP
#define GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline int off_ker_bias(int dhc, int i0, int i1, int n_gates) {
    return i0 * dhc + i1;
}

inline int cell_ws_state(int states_ws_ld, int i, int j) {
    return i * states_ws_ld + j;
}

inline int cell_scratch_mem(
        int scratch_gates_ld, int dhc, int i, int n, int j) {
    return i * scratch_gates_ld + n * dhc + j;
}

struct ref_rnn_copy_fwd_t {
    ref_rnn_copy_fwd_t(const sycl_rnn_copy_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src,
            xpu::sycl::out_memory_arg_t &dst)
        : src_ {src}, dst_ {dst}, conf_ {conf} {}

    void operator()(::sycl::nd_item<3> item) const {
        const dim_t tl = item.get_global_id(0) // timestep/layer
                / (conf_.layer ? 1 : conf_.n_dir);
        dim_t dir = conf_.layer
                ? 0
                : item.get_global_id(0) % conf_.n_dir; // direction
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (dir >= conf_.n_dir || n >= conf_.batch || c >= conf_.range) return;

        dim_t src_offset = 0;
        dim_t dst_offset = 0;
        if (conf_.layer) { // layer
            if (tl >= conf_.n_iter) return;
            if (conf_.to_state) { // init
                if (conf_.l2r) { // l2r
                    src_offset = conf_.src_md.off(tl, n, c);
                    dst_offset = conf_.dst_md.off(0, dir, tl, n, c);
                    do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    dir = 1;
                }
                if (conf_.r2l) { // r2l
                    src_offset = conf_.src_md.off(tl, n, c);
                    dst_offset = conf_.dst_md.off(
                            0, conf_.n_dir - 1, conf_.n_iter - tl - 1, n, c);
                    do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                }
            } else { // res
                if (conf_.l2r) {
                    dst_offset = conf_.dst_md.off(tl, n, dir * conf_.range + c);
                    src_offset = conf_.src_md.off(conf_.n_layer, dir, tl, n, c);
                    do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    dir = 1;
                }
                if (conf_.r2l) {
                    dst_offset = conf_.dst_md.off(tl, n, dir * conf_.range + c);
                    src_offset = conf_.src_md.off(
                            conf_.n_layer, dir, conf_.n_iter - tl - 1, n, c);
                    if (conf_.sum) {
                        dst_offset = conf_.dst_md.off(tl, n, c);
                        auto src = load_float_value(
                                src_md().data_type(), src_ptr(), src_offset);
                        auto dst = load_float_value(conf_.dst_md.data_type(),
                                dst_ptr(), dst_offset);
                        store_float_value(dst_md().data_type(), src + dst,
                                dst_ptr(), dst_offset);
                    } else {
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    }
                }
            }
        } else { // iter
            if (tl >= conf_.n_layer) return;
            if (conf_.to_state) { // init
                src_offset = conf_.src_md.off(tl, dir, n, c);
                dst_offset = conf_.dst_md.off(tl, dir, conf_.n_iter, n, c);
                do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());

            } else { // res
                src_offset
                        = conf_.src_md.off(tl + 1, dir, conf_.n_iter - 1, n, c);
                dst_offset = conf_.dst_md.off(tl, dir, n, c);
                do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
            }
        }
    }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_conf_t conf_;

    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    void do_copy(
            dim_t src_offset, dim_t dst_offset, void *from, void *to) const {
        if (src_ptr()) {
            auto src = load_float_value(src_md().data_type(), from, src_offset);
            if (dst_ptr()) {
                store_float_value(dst_md().data_type(), src, to, dst_offset);
            }
        } else {
            if (dst_ptr()) {
                store_float_value(dst_md().data_type(), 0.0f, to, dst_offset);
            }
        }
    }
};

struct ref_rnn_copy_bwd_t {
    ref_rnn_copy_bwd_t(const sycl_rnn_copy_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src,
            xpu::sycl::out_memory_arg_t &dst)
        : src_ {src}, dst_ {dst}, conf_ {conf} {}

    void operator()(::sycl::nd_item<3> item) const {
        const dim_t tl = item.get_global_id(0) // timestep/layer
                / (conf_.layer ? 1 : conf_.n_dir);
        dim_t dir = conf_.layer
                ? 0
                : item.get_global_id(0) % conf_.n_dir; // direction
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (dir >= conf_.n_dir || n >= conf_.batch || c >= conf_.range) return;

        dim_t src_offset = 0;
        dim_t dst_offset = 0;
        if (conf_.layer) {
            if (tl >= conf_.n_iter) return;
            if (conf_.to_state) { // init
                if (conf_.l2r && conf_.r2l) {
                    if (conf_.sum) { // sum
                        src_offset = conf_.src_md.off(tl, n, c);
                        dst_offset = conf_.dst_md.off(
                                conf_.n_layer, 0, tl + 1, 0, n, c);
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                        dst_offset = conf_.dst_md.off(
                                conf_.n_layer, 1, tl + 1, 0, n, c);
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    } else { // concat
                        src_offset = conf_.src_md.off(tl, n, c);
                        dst_offset = conf_.dst_md.off(
                                conf_.n_layer, 0, tl + 1, 0, n, c);
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                        src_offset = conf_.src_md.off(tl, n, c + conf_.range);

                        dst_offset = conf_.dst_md.off(
                                conf_.n_layer, 1, tl + 1, 0, n, c);
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    }
                } else if (conf_.l2r && !conf_.r2l) { // l2r
                    src_offset = conf_.src_md.off(tl, n, c);
                    dst_offset = conf_.dst_md.off(
                            conf_.n_layer, dir, tl + 1, 0, n, c);
                    do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                } else if (!conf_.l2r && conf_.r2l) { // r2l
                    src_offset = conf_.src_md.off(conf_.n_iter - tl - 1, n, c);
                    dst_offset = conf_.dst_md.off(
                            conf_.n_layer, 0, tl + 1, 0, n, c);
                    do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                }
            } else { // res
                if (conf_.l2r) {
                    src_offset = conf_.src_md.off(0, 0, tl + 1, 1, n, c);
                    dst_offset = conf_.dst_md.off(tl, n, c);
                    if (c > conf_.states_ws_ld) {
                        store_float_value(conf_.src_md.data_type(), 0,
                                dst_ptr(), dst_offset);
                    } else if (c < conf_.dst_md.dims()[2]) {
                        do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
                    }
                }
                if (conf_.r2l) {
                    src_offset = conf_.src_md.off(
                            0, 0, conf_.n_iter - tl, 1, n, c);
                    dst_offset = conf_.dst_md.off(tl, n, c);
                    if (conf_.l2r) { // sum,concat
                        src_offset = conf_.src_md.off(0, 1, tl + 1, 1, n, c);
                        dst_offset = conf_.dst_md.off(tl, n, c);
                        auto src = load_float_value(conf_.src_md.data_type(),
                                src_ptr(), src_offset);
                        auto dst = load_float_value(conf_.src_md.data_type(),
                                dst_ptr(), dst_offset);

                        if (c > conf_.states_ws_ld) {
                            store_float_value(conf_.src_md.data_type(), 0,
                                    dst_ptr(), dst_offset);
                        } else if (c < conf_.dst_md.dims()[2]) {
                            store_float_value(conf_.src_md.data_type(),
                                    src + dst, dst_ptr(), dst_offset);
                        }
                    } else {
                        if (c > conf_.states_ws_ld) {
                            store_float_value(conf_.src_md.data_type(), 0,
                                    dst_ptr(), dst_offset);
                        } else if (c < conf_.dst_md.dims()[2]) {
                            do_copy(src_offset, dst_offset, src_ptr(),
                                    dst_ptr());
                        }
                    }
                }
            }
        }

        else { // iter
            if (tl >= conf_.n_layer || n >= conf_.batch || c >= conf_.range)
                return;
            if (conf_.to_state) { // init
                src_offset = conf_.src_md.off(tl, dir, n, c);
                dst_offset
                        = conf_.dst_md.off(conf_.n_layer + tl, dir, 0, 0, n, c);
                do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
            } else { // res
                src_offset = conf_.src_md.off(
                        tl, dir, dir == 0 ? 1 : conf_.n_iter, 0, n, c);
                dst_offset = conf_.dst_md.off(tl, dir, n, c);
                do_copy(src_offset, dst_offset, src_ptr(), dst_ptr());
            }
        }
    }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_conf_t conf_;

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    void do_copy(
            dim_t src_offset, dim_t dst_offset, void *from, void *to) const {
        if (from) {
            auto src = load_float_value(
                    conf_.src_md.data_type(), from, src_offset);
            if (dst_ptr()) {
                store_float_value(
                        conf_.src_md.data_type(), src, to, dst_offset);
            }
        } else {
            if (dst_ptr()) {
                store_float_value(
                        conf_.src_md.data_type(), 0.0f, to, dst_offset);
            }
        }
    }
};

struct ref_rnn_bias_fwd {
    ref_rnn_bias_fwd(const sycl_rnn_bias_fwd_conf_t &conf,
            const xpu::sycl::inout_memory_arg_t &gates_base,
            const xpu::sycl::in_memory_arg_t &bias,
            const xpu::sycl::out_memory_arg_t &states_base)
        : gates_ {gates_base}
        , bias_ {bias}
        , states_ {states_base}
        , conf_ {conf} {}
    void operator()(::sycl::nd_item<3> item) const {

        const int b = item.get_global_id(1);
        const int c = item.get_global_id(0);

        if (b >= conf_.batch || c >= conf_.dhc) return;

        auto gates = gates_ptr();
        auto bias = bias_ptr();
        auto states = states_ptr();

        auto gates_offset = gates_data_offset(b, c);
        auto bias_offset = bias_data_offset(b, c);
        auto states_offset = states_data_offset(b, c);

        auto gates_val
                = load_float_value(conf_.states_data_type, gates, gates_offset);
        auto bias_val = load_float_value(conf_.bias_type, bias, bias_offset);

        auto g = compute_gates(gates_val, bias_val);

        store_float_value(conf_.states_data_type, g, states, states_offset);
        store_float_value(conf_.gates_type, g, gates, gates_offset);
    }

    inline dim_t gates_data_offset(int b, int c) const {
        return cell_scratch_mem(conf_.gates_ws_ld, conf_.dhc, b, 0, c);
    }

    inline dim_t bias_data_offset(int b, int c) const {
        return off_ker_bias(conf_.dhc, 0, c, 0);
    }

    inline dim_t states_data_offset(int b, int c) const {
        return cell_ws_state(conf_.states_ws_ld, b, c);
    }

    float compute_gates(float in_val, float bias_val) const {
        switch (conf_.activation_kind) {
            case alg_kind::eltwise_relu:
                return (float)(math::relu_fwd(
                        (float)(in_val + bias_val), conf_.alpha));
            case alg_kind::eltwise_tanh:
                return (float)(math::tanh_fwd((float)(in_val + bias_val)));
            case alg_kind::eltwise_logistic:
                return (float)(math::logistic_fwd((float)(in_val + bias_val)));
            default: return 0;
        }
    }

    void *gates_ptr() const { return gates_.get_pointer(); }
    void *states_ptr() const { return states_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }

    xpu::sycl::inout_memory_arg_t gates_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::out_memory_arg_t states_;
    sycl_rnn_bias_fwd_conf_t conf_;
};

struct ref_rnn_bias_bwd {
    ref_rnn_bias_bwd(const sycl_rnn_bias_bwd_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &ws_gates_base,
            const xpu::sycl::in_memory_arg_t &diff_lay_base,
            const xpu::sycl::in_memory_arg_t &diff_iter_base,
            const xpu::sycl::out_memory_arg_t &scratch_diff_gates_base,
            const xpu::sycl::inout_memory_arg_t &diff_bias_base)
        : ws_gates_(ws_gates_base)
        , diff_lay_(diff_lay_base)
        , diff_iter_(diff_iter_base)
        , scratch_diff_gates_(scratch_diff_gates_base)
        , diff_bias_(diff_bias_base)
        , conf_ {conf} {}
    void operator()(::sycl::nd_item<3> item) const {
        const int b = item.get_global_id(1);
        const int c = item.get_global_id(0);

        if (b >= conf_.batch || c >= conf_.dhc) return;

        auto diff_state_l = diff_lay_ptr();
        auto diff_state_i = diff_iter_ptr();

        auto scratch_diff_gates = scratch_diff_gates_ptr();
        auto diff_bias = diff_bias_ptr();

        auto diff_state_l_off = cell_scratch_diff_states(b, c);
        auto diff_state_iter_off = cell_scratch_diff_states(b, c);

        auto diff_bias_off = bias_data_offset(0, c);

        auto scratch_diff_gates_off = cell_offset(b, c);

        float dsl_val = load_float_value(
                conf_.diff_states_type, diff_state_l, diff_state_l_off);
        float dsi_val = load_float_value(
                conf_.diff_states_type, diff_state_i, diff_state_iter_off);

        float dH = dsl_val + dsi_val;

        auto ws_gates = ws_gates_ptr();
        auto ws_gates_off = cell_ws_gates(b, c);

        float g = load_float_value(conf_.gates_type, ws_gates, ws_gates_off);

        float tmp = dH * activation_bwd(g);

        store_float_value(conf_.diff_gates_type, tmp, scratch_diff_gates,
                scratch_diff_gates_off);

        ::sycl::atomic_ref<float, ::sycl::memory_order::relaxed,
                ::sycl::memory_scope::device,
                ::sycl::access::address_space::global_space>
                atomic_bias_out(
                        reinterpret_cast<float *>(diff_bias)[diff_bias_off]);
        atomic_bias_out.fetch_add(tmp);
    }

    inline dim_t bias_data_offset(int b, int c) const {
        return off_ker_bias(conf_.dhc, 0, c, 0);
    }

    inline dim_t cell_scratch_diff_states(int i4, int i5) const {
        return (i4 * conf_.scratch_diff_states_ld + i5);
    }
    inline dim_t cell_offset(int b, int c) const {
        return cell_scratch_mem(conf_.gates_ws_ld, conf_.dhc, b, 0, c);
    }

    inline dim_t cell_ws_gates(int b, int c) const {
        return (b * conf_.states_ws_ld + c);
    }

    inline float activation_bwd(const float &in_val) const {
        switch (conf_.activation_kind) {
            case alg_kind::eltwise_relu:
                return math::relu_bwd_use_dst(1.0f, in_val, conf_.alpha);
            case alg_kind::eltwise_tanh:
                return math::tanh_bwd_use_dst(1.0f, in_val);
            case alg_kind::eltwise_logistic:
                return math::logistic_use_dst(1.0f, in_val);
            default: return 0;
        }
    }

    void *ws_gates_ptr() const { return ws_gates_.get_pointer(); }
    void *diff_lay_ptr() const { return diff_lay_.get_pointer(); }
    void *diff_iter_ptr() const { return diff_iter_.get_pointer(); }
    void *scratch_diff_gates_ptr() const {
        return scratch_diff_gates_.get_pointer();
    }
    void *diff_bias_ptr() const { return diff_bias_.get_pointer(); }

    xpu::sycl::in_memory_arg_t ws_gates_;
    xpu::sycl::in_memory_arg_t diff_lay_;
    xpu::sycl::in_memory_arg_t diff_iter_;
    xpu::sycl::out_memory_arg_t scratch_diff_gates_;
    xpu::sycl::inout_memory_arg_t diff_bias_;
    sycl_rnn_bias_bwd_conf_t conf_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
