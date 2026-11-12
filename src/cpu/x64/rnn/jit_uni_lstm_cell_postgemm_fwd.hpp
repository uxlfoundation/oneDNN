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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "common/utils.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_fwd_t
    : public jit_uni_rnn_postgemm_t,
      public jit_uni_lstm_cell_postgemm_t<isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd_t)

    jit_uni_lstm_cell_postgemm_fwd_t(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm_t(rnn, pd, jit_name())
        , jit_uni_lstm_cell_postgemm_t<isa>(this,
                  get_last_preserved_vmm_idx(1) + 1,
                  // usage of jit_uni_rnn_postgemm_t::bf16_emu_ to identify bf16
                  // emulation case is illegal here, it's created in
                  // jit_uni_rnn_postgemm_t::init(), not in constructor, so
                  // jit_uni_rnn_postgemm_t::bf16_emu_ = nullptr always on this
                  // stage
                  src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
    }

    ~jit_uni_lstm_cell_postgemm_fwd_t() override = default;

    status_t init(data_type_t sdt) override;

protected:
    using injector_t = typename jit_uni_lstm_cell_postgemm_t<isa>::injector_t;
    using Vmm = typename jit_uni_lstm_cell_postgemm_t<isa>::Vmm;

    std::unique_ptr<injector_t> sigmoid_injector_;
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    static constexpr int vlen_ = cpu_isa_traits_t<isa>::vlen;
    static constexpr int qscale_dt_size = sizeof(float);
    static constexpr int weights_peephole_dt_size_ = sizeof(float);
    const int vlen_dst_ = vlen_
            / static_cast<int>(
                    sizeof(float) / types::data_type_size(src_data_t));
    const int vlen_bias_
            = vlen_ / static_cast<int>(sizeof(float) / bias_dt_size_);
    const int vlen_c_states_
            = vlen_ / static_cast<int>(sizeof(float) / cstate_dt_size_);
    const int hstate_dt_size_
            = static_cast<int>(types::data_type_size(src_data_t));
    const int gate_dt_size_
            = static_cast<int>(types::data_type_size(src_data_t));
    const int scratch_dt_size_
            = static_cast<int>(types::data_type_size(scratch_data_t));
    int get_vmm_idx(int unroll_idx, int type_shift) const {
        const int preserved_vmm_start_idx = 1;
        // G0, G1, G2, G3, c_states;
        const int num_preserved_regs_for_loop_iter = 5;
        assert(type_shift < num_preserved_regs_for_loop_iter);
        const int unroll_idx_start = preserved_vmm_start_idx
                + num_preserved_regs_for_loop_iter * unroll_idx;
        return unroll_idx_start + type_shift;
    }

    int G0_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 0); }
    int G1_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 1); }
    int G2_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 2); }
    int G3_idx(int unroll_idx) const { return get_vmm_idx(unroll_idx, 3); }
    int c_states_idx(int unroll_idx) const {
        return get_vmm_idx(unroll_idx, 4);
    }
    int get_last_preserved_vmm_idx(int current_loop_unroll) const {
        return c_states_idx(current_loop_unroll - 1);
    }

    dim_t scale_off(int gate_idx, int unroll_idx) const {
        const int vlen_qscale_elem = vlen_ / qscale_dt_size;
        return gate_idx * rnn_.dhc + unroll_idx * vlen_qscale_elem;
    }

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
