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

#include "cpu/rnn/ref_rnn_impl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <>
rnn_gemm_sig(ref_rnn_fwd_u8s8_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    int32_t offsetc = 0;
    return gemm_s8u8s32_compute("P", "N", "F", &m, &n, &k, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, &offsetc);
}

template <>
rnn_gemm_sig(ref_rnn_fwd_s8s8_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    int32_t offsetc = 0;
    return gemm_s8s8s32_compute("P", "N", "F", &m, &n, &k, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, &offsetc);
}

RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_s8s8_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_s8s8_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_s8s8_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_s8s8_t)

/* Fix for MSVS warning C4661 */
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_u8s8_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_u8s8_t::merged_layer_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_s8s8_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_s8s8_t::merged_layer_brgemm);

template struct ref_rnn_common_t<prop_kind::forward, data_type::u8,
        data_type::s8, data_type::s32>;
template struct ref_rnn_common_t<prop_kind::forward, data_type::s8,
        data_type::s8, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
