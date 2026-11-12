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

#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_1_fwd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
status_t
jit_uni_gru_cell_postgemm_part1_fwd_t<isa, src_data_t, scratch_data_t>::init(
        data_type_t sdt) {
    CHECK(jit_uni_rnn_postgemm_t::init(src_data_t));
    // no need to save state of registers
    // (unless emulating bf16 support)
    const bool save_state
            = src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16);
    // we use rax for both constant tables as they use the same table
    CHECK(safe_ptr_assign(sigmoid_injector_,
            new injector_t(this, alg_kind::eltwise_logistic, 0.0f, 0.0f, 1.0f,
                    data_type::f32, save_state, rax)));
    return create_kernel();
}

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
void jit_uni_gru_cell_postgemm_part1_fwd_t<isa, src_data_t,
        scratch_data_t>::generate() {
    using namespace Xbyak;
    const auto is_training
            = pd_->desc()->prop_kind == prop_kind::forward_training;

    const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
    float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

    // Register map
    const Reg64 loop_cnt(rbx); // loop counter

    // We start code generations here
    preamble();

    // extract addresses passed as parameter
    const auto addr_ws_gates_reg = abi_param1;
    const auto addr_scratch_gates_reg = abi_param2;
    const auto addr_bias_reg = abi_param3;
    const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
    const auto addr_states_t_l_copy_reg = r10;
    const auto addr_states_tm1_l_reg = r11;
    // Here we cannot use rbp to have initial stack pointer so we
    // use rsp and offset it with the size of pushed registers in
    // preamble
    const auto base_args = get_stack_params_address();
    mov(addr_states_t_l_copy_reg, ptr[base_args]);
    mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
#else
    const auto addr_states_t_l_copy_reg = abi_param5;
    const auto addr_states_tm1_l_reg = abi_param6;
#endif
    // helper lambda to address the gates and biases
    const auto sg_addr = [&](int i, int j) {
        return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size
                + j * vlen];
    };
    const auto wg_addr = [&](int i, int j) {
        return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size
                + j * vlen_dst];
    };
    const auto B_addr = [&](int i, int j) {
        return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_ + j * vlen];
    };

    const dim_t loop_len = rnn_.dhc;
    const int loop_tail = static_cast<int>(loop_len % vlen_elems);
    // initialize registers with addresses and constants
    init_regs(weights_scales, vlen, loop_tail);

    // both sigmoid and tanh use the same table so load address just once in rax
    sigmoid_injector_->load_table_addr();

    const dim_t nb_loop_len = loop_len / vlen_elems;
    int loop_ur_val = 1;
    const bool is_brgemm = rnn_.is_brgemm && !rnn_.unfused_post_gemm;
    if (is_brgemm) {
#ifdef _WIN32
        mov(loop_cnt, ptr[base_args + 40]);
#else
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(loop_cnt, ptr[base_args + 24]);
#endif
    } else {
        for (loop_ur_val = loop_ur_max; loop_ur_val > 1; --loop_ur_val)
            if (nb_loop_len % loop_ur_val == 0) break;

        mov(loop_cnt, loop_len);
    }
    const int loop_ur = loop_ur_val;

    auto compute_loop = [&](int current_vlen_elem, int current_loop_unroll) {
        const auto current_vlen = current_vlen_elem * scratch_dt_size;
        Label loop_start_label;
        L(loop_start_label);
        {
            for (int loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                    ++loop_ur_idx) {
                const Vmm G0(G0_idx(loop_ur_idx));
                const Vmm G1(G1_idx(loop_ur_idx));
                // batch these operations in order to combine calls to injector:
                //      Compute gate 0: G0 = sigmoid(G0 + b0)
                //      Compute gate 1: G1 = sigmoid(G1 + b1)

                // load gates from scratchpad
                load(G0, sg_addr(0, loop_ur_idx), scratch_data_t, current_vlen);
                load(G1, sg_addr(1, loop_ur_idx), scratch_data_t, current_vlen);

                // dequantize gates from s32 to f32 if needed
                deq_w(src_data_t, G0, tmp1_vmm, tmp2_vmm,
                        0 * rnn_.dhc + loop_ur_idx * vlen_qscale, mask,
                        current_vlen);
                deq_w(src_data_t, G1, tmp1_vmm, tmp2_vmm,
                        1 * rnn_.dhc + loop_ur_idx * vlen_qscale, mask,
                        current_vlen);

                // apply bias
                to_float(tmp1_vmm, B_addr(0, loop_ur_idx), rnn_.bias_dt,
                        current_vlen);
                compute_vaddps(G0, G0, tmp1_vmm, current_vlen);
                to_float(tmp2_vmm, B_addr(1, loop_ur_idx), rnn_.bias_dt,
                        current_vlen);
                compute_vaddps(G1, G1, tmp2_vmm, current_vlen);
            }

            // Compute sigmoid of unrolled G0 and G1 regs together
            // (this allows to not save any registers during eltwise)
            injector_utils::vmm_index_set_t vmm_idxs;
            for (int loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                    ++loop_ur_idx) {
                vmm_idxs.emplace(G0_idx(loop_ur_idx));
                vmm_idxs.emplace(G1_idx(loop_ur_idx));
            }
            sigmoid_injector_->compute_vector_range(vmm_idxs);

            for (int loop_ur_idx = 0; loop_ur_idx < current_loop_unroll;
                    ++loop_ur_idx) {
                const Vmm G0(G0_idx(loop_ur_idx));
                const Vmm G1(G1_idx(loop_ur_idx));
                // store G0 for use in postgemm_part2
                store(sg_addr(0, loop_ur_idx), G0, scratch_data_t,
                        current_vlen);

                // if training we write back the gates
                if (is_training) {
                    to_src(wg_addr(1, loop_ur_idx), G1, src_data_t,
                            current_vlen);
                    to_src(wg_addr(0, loop_ur_idx), G0, src_data_t,
                            current_vlen);
                }

                // states_t_l = states_tm1_l * G1
                to_float(tmp1_vmm,
                        ptr[addr_states_tm1_l_reg + loop_ur_idx * vlen_dst],
                        src_data_t, current_vlen);
                compute_vmulps(G1, G1, tmp1_vmm, current_vlen);
                to_src(ptr[addr_states_t_l_reg + loop_ur_idx * vlen_dst], G1,
                        src_data_t, current_vlen);
                // if states_t_l_copy is a non null ptr, we write the output
                // to both tensors
                Label loop_inc_regs;
                cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
                jle(loop_inc_regs);
                // As to_src is called with write_only=true it's important
                // for xf16 src_dt to execute just after to_src method with
                // write_only=false for the same Vmm
                to_src(ptr[addr_states_t_l_copy_reg + loop_ur_idx * vlen_dst],
                        G1, src_data_t, current_vlen, true);
                L(loop_inc_regs);
            }

            if (current_vlen_elem != loop_tail) {
                // increment address pointers
                const auto current_gate_size = current_vlen == vlen
                        ? vlen_dst * current_loop_unroll
                        : gate_dt_size;
                const auto current_states_size = current_vlen == vlen
                        ? vlen_dst * current_loop_unroll
                        : hstate_dt_size;

                add(addr_scratch_gates_reg, current_vlen * current_loop_unroll);
                add(addr_bias_reg,
                        current_vlen == vlen ? vlen_bias_ * current_loop_unroll
                                             : bias_dt_size_);
                add(addr_states_t_l_reg, current_states_size);
                add(addr_states_t_l_copy_reg, current_states_size);
                add(addr_states_tm1_l_reg, current_states_size);
                if (is_training) add(addr_ws_gates_reg, current_gate_size);
                const int qscale_shift = current_vlen == vlen
                        ? current_vlen * current_loop_unroll
                        : qscale_dt_size;
                inc_regs(mask, qscale_shift);

                // increment loop counter
                sub(loop_cnt, current_vlen_elem * current_loop_unroll);
                cmp(loop_cnt, current_vlen_elem * current_loop_unroll);
                jge(loop_start_label);
            }
        }
    };

    // vector processing
    if (loop_len >= vlen_elems) {
        Label tail_processing_or_exit_label;
        if (is_brgemm) {
            cmp(loop_cnt, vlen_elems * loop_ur);
            jl(tail_processing_or_exit_label, T_NEAR);
        }
        compute_loop(vlen_elems, loop_ur);
        L(tail_processing_or_exit_label);
    }

    // tail processing
    if (loop_tail > 0) {
        Label exit_label;
        if (is_brgemm) {
            cmp(loop_cnt, 0);
            jle(exit_label, T_NEAR);
        }

        compute_loop(is_avx512 ? loop_tail : 1, 1);
        L(exit_label);
    }

    postamble();

    // Again, only one table is needed and shared between sigmoid and tanh
    sigmoid_injector_->prepare_table(true);
    init_table(vlen);
}

template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx512_core,
        data_type::f32, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx512_core,
        data_type::bf16, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx512_core,
        data_type::f16, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx512_core,
        data_type::u8, data_type::s32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx512_core,
        data_type::s8, data_type::s32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx2, data_type::f32,
        data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx2, data_type::bf16,
        data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx2, data_type::f16,
        data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx2, data_type::u8,
        data_type::s32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd_t<avx2, data_type::s8,
        data_type::s32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
