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

#ifndef CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_RNN_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_rnn_cell_postgemm_fwd_t : public jit_uni_rnn_postgemm_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_rnn_cell_postgemm_fwd_t)

    using injector_t
            = jit_uni_eltwise_injector_t<typename cpu_isa_traits_t<isa>::Vmm>;

    jit_uni_rnn_cell_postgemm_fwd_t(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm_t(rnn, pd, jit_name()) {}

    status_t init(data_type_t sdt) override;

protected:
    std::unique_ptr<injector_t> injector_;

    // register size in bytes
    using Vmm = typename cpu_isa_traits_t<isa>::Vmm;
    static constexpr int vlen = cpu_isa_traits_t<isa>::vlen;
    static constexpr int cstate_dt_size = sizeof(float);
    static constexpr int qscale_dt_size = sizeof(float);

    const int vlen_dst = vlen
            / static_cast<int>(
                    sizeof(float) / types::data_type_size(src_data_t));
    const int vlen_bias
            = vlen / static_cast<int>(sizeof(float) / bias_dt_size_);
    const int hstate_dt_size
            = static_cast<int>(types::data_type_size(src_data_t));
    const int gate_dt_size
            = static_cast<int>(types::data_type_size(src_data_t));
    const int scratch_dt_size
            = static_cast<int>(types::data_type_size(scratch_data_t));

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
