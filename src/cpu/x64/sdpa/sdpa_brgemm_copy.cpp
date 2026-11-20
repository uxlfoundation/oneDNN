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

#include "cpu/x64/sdpa/sdpa_brgemm_copy.hpp"

#include <cassert>

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdpa_brgemm_copy {

dim_t b_packer_t::packed_ld(dim_t N) {
    return utils::rnd_up(N, panel_n);
}

dim_t b_packer_t::k_block(data_type_t dt) {
    return matmul::get_wei_k_blk(dt);
}

dim_t b_packer_t::packed_k(dim_t K, data_type_t dt) {
    return utils::rnd_up(K, k_block(dt));
}

std::size_t b_packer_t::panel_offset(
        dim_t k, dim_t n, dim_t padded_k, dim_t v) {
    return static_cast<std::size_t>(n / panel_n) * padded_k * panel_n
            + static_cast<std::size_t>(k / v) * panel_n * v
            + static_cast<std::size_t>(n % panel_n) * v + k % v;
}

status_t b_packer_t::init(dim_t K, dim_t N, dim_t src_k_stride,
        dim_t src_n_stride, data_type_t dt) {
    if (K <= 0 || N <= 0 || src_k_stride <= 0 || src_n_stride <= 0)
        return status::invalid_arguments;
    if (src_k_stride != 1 && src_n_stride != 1) return status::unimplemented;

    K_ = K;
    N_ = N;
    dt_ = dt;
    if (src_n_stride == 1) {
        src_tag_ = format_tag::ab;
        src_ld_ = src_k_stride;
    } else {
        src_tag_ = format_tag::ba;
        src_ld_ = src_n_stride;
    }
    k_vnni_ = data_type_vnni_granularity(dt_);
    if (k_vnni_ <= 0) return status::invalid_arguments;

    CHECK(matmul::init_conf(
            conf_, 1, 0, K_, N_, src_ld_, panel_n, dt_, dt_, src_tag_));
    k_pad_block_ = k_block(dt_);
    return matmul::create_brgemm_matmul_copy_b(kernel_, &conf_);
}

dim_t b_packer_t::padded_k() const {
    return packed_k(K_, dt_);
}

dim_t b_packer_t::k_step() const {
    return conf_.K_blk;
}

void b_packer_t::execute(const void *src, void *dst) const {
    for (dim_t n0 = 0; n0 < N_; n0 += panel_n)
        execute_panel_range(src, dst, n0, 0, K_);
}

void b_packer_t::execute_panel_range(const void *src, void *dst, dim_t n0,
        dim_t k_begin, dim_t k_end) const {
    assert(n0 % panel_n == 0);
    assert(k_begin % k_step() == 0);
    assert(k_end == K_ || k_end % k_step() == 0);
    const auto src_size = types::data_type_size(dt_);
    const auto dst_size = types::data_type_size(conf_.wei_dt);
    const auto *src_bytes = static_cast<const char *>(src);
    auto *dst_bytes = static_cast<char *>(dst);

    const dim_t n = N_ - n0 < panel_n ? N_ - n0 : panel_n;
    for (dim_t k0 = k_begin; k0 < k_end; k0 += conf_.K_blk) {
        const dim_t k_iters
                = nstl::min(conf_.K_blk, nstl::min(K_ - k0, k_end - k0));
        const dim_t k_remainder = k_iters % k_pad_block_;
        const dim_t k_pad = k_remainder == 0
                ? 0
                : k_pad_block_ - utils::rnd_up(k_remainder, k_vnni_);
        const dim_t src_offset = src_tag_ == format_tag::ab ? k0 * src_ld_ + n0
                                                            : n0 * src_ld_ + k0;
        const dim_t dst_offset
                = (k0 / k_vnni_ * panel_n * k_vnni_
                          + n0 / panel_n * packed_k(K_, dt_) * panel_n)
                * dst_size;

        auto ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
        ctx.src = src_bytes + src_offset * src_size;
        ctx.tr_src = dst_bytes + dst_offset;
        ctx.current_K_start = k0;
        ctx.current_K_iters = k_iters;
        ctx.current_K_pad = k_pad;
        ctx.current_N_blk = n;
        (*kernel_)(&ctx);
    }
}

} // namespace sdpa_brgemm_copy
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
