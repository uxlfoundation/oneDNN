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
rnn_gemm_sig((ref_rnn_fwd_f32_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC, nullptr, pd()->rnn_.force_nocopy);
}

template <>
rnn_gemm_sig((ref_rnn_bwd_f32_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC, nullptr, pd()->rnn_.force_nocopy);
}

template <>
rnn_gemm_sig(ref_rnn_fwd_f32_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return sgemm_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig(ref_rnn_bwd_f32_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return sgemm_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_INIT_LAYER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_INIT_ITER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_RES_LAYER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_RES_ITER_BWD(ref_rnn_common_bwd_f32_t)

/* Fix for MSVS warning C4661 */
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f32_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f32_t::merged_layer_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_bwd_f32_t::merged_layer_execution_ref);

template struct ref_rnn_common_t<prop_kind::forward, data_type::f32,
        data_type::f32, data_type::f32>;
template struct ref_rnn_common_t<prop_kind::backward, data_type::f32,
        data_type::f32, data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
