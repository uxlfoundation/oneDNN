/*
*******************************************************************************
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
*******************************************************************************
*/

#ifndef CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP
#define CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP

#include <memory>

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdpa_brgemm_copy {

// Packs a BRGEMM B operand [K x N] with the BRGEMM matmul copy-B kernels.
// N is split into panel_n-wide panels, each stored as an independent
// [packed_k(K, dt) x panel_n] matrix; see panel_offset(). A BRGEMM reading a
// packed panel uses ldb = panel_n and an N block of at most panel_n. execute
// positions panels explicitly; the copy kernels use LDB2 == 0.
class b_packer_t {
public:
    // N width of one packed panel: the copy kernel's wei_n_blk and N_blk, so
    // the kernel writes one panel per N iteration. 64 matches the BRGEMM
    // matmul default B block (get_default_n_block()); the copy kernels also
    // support 16, 32 and 48. This is the single source of the panel width for
    // all users of the packed layout.
    static constexpr dim_t panel_n = 64;

    static dim_t packed_ld(dim_t N);
    static dim_t k_block(data_type_t dt);
    static dim_t packed_k(dim_t K, data_type_t dt);
    // Element offset of logical B[k, n], where padded_k = packed_k(K, dt) and
    // v is the VNNI granularity.
    static std::size_t panel_offset(dim_t k, dim_t n, dim_t padded_k, dim_t v);

    status_t init(dim_t K, dim_t N, dim_t src_k_stride, dim_t src_n_stride,
            data_type_t dt);

    // Panel height of this packer's destination.
    dim_t padded_k() const;
    // Maximum number of K rows processed by one copy-kernel call.
    dim_t k_step() const;

    void execute(const void *src, void *dst) const;
    // Packs rows [k_begin, k_end) of the panel starting at column n0. Range
    // bounds must be multiples of k_step(), except k_end == K, so that only the
    // last range pads K.
    void execute_panel_range(const void *src, void *dst, dim_t n0,
            dim_t k_begin, dim_t k_end) const;

private:
    // The copy kernel retains a pointer to this configuration.
    matmul::brgemm_matmul_conf_t conf_;
    std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t> kernel_;
    dim_t K_ = 0;
    dim_t N_ = 0;
    dim_t src_ld_ = 0;
    dim_t k_pad_block_ = 0;
    dim_t k_vnni_ = 0;
    data_type_t dt_ = data_type::undef;
    format_tag_t src_tag_ = format_tag::undef;
};

} // namespace sdpa_brgemm_copy
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP
