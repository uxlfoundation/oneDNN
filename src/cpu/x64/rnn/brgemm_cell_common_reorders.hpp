/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_RNN_BRGEMM_CELL_COMMON_REORDERS_HPP
#define CPU_X64_RNN_BRGEMM_CELL_COMMON_REORDERS_HPP

#include "cpu/x64/jit_brgemm_transpose_utils.hpp"
#include "cpu/x64/rnn/jit_brgemm_transpose_single_row.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {
struct rnn_conf_t;
}
namespace x64 {
struct src_layer_iter_transpose_t {
    src_layer_iter_transpose_t(const dim_t src_ld, const dim_t dst_ld,
            const dim_t rows, const dim_t cols,
            jit_brgemm_trans_src_t *const kernel_transpose);

    template <typename Dt>
    void execute(const Dt *src, Dt *dst) const;

private:
    const dim_t src_ld_;
    const dim_t dst_ld_;
    const dim_t src_rows_;
    const dim_t src_cols_;
    jit_brgemm_trans_src_t *const kernel_transpose_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
