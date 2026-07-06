/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP
#define CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP

#include "common/c_types_map.hpp"

#include <atomic>
#include <cstddef>

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvjit/rvjit_rvv.hpp"
#include "xbyak_riscv/xbyak_riscv_util.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

extern std::atomic<dim_t> rvv_gemm_f32_m_unroll;
extern std::atomic<dim_t> rvv_gemm_f32_n_unroll;

template <typename T, bool isTransA, bool isTransB>
struct gemm_traits_t {};

template <bool isTransA, bool isTransB>
struct gemm_traits_t<float, isTransA, isTransB> {
    // m is determined by VLEN at runtime via get_m_unroll_factor()
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 256;
    static constexpr dim_t BK = isTransB ? 96 : 256;
};

template <typename T>
struct gemm_utils_traits;

template <>
struct gemm_utils_traits<float> {
    // m = VLEN / 32 * LMUL, where LMUL = 4 for f32
    // VLEN=128 -> m=16, VLEN=256 -> m=32, VLEN=512 -> m=64
    static dim_t get_m_unroll_factor() {
        dim_t m = rvv_gemm_f32_m_unroll.load(std::memory_order_relaxed);
        if (m == 0) {
            const uint32_t vlen = Xbyak_riscv::CPU::getInstance().getVlen();
            m = static_cast<dim_t>(vlen / 32 * 4);
            rvv_gemm_f32_m_unroll.store(m, std::memory_order_relaxed);
        }
        return m;
    }

    // n is the JIT kernel's hardware/model-derived N-tile ceiling (the same
    // formula rvv_matmul_engine_t::configure() uses internally) — the
    // driver's N-tiling must match it exactly, since the GEMM kernel
    // dispatches N via a plain switch_ (no internal unroll loop), so it can
    // only ever process up to this many columns in a single call.
    static dim_t get_n_unroll_factor() {
        dim_t n = rvv_gemm_f32_n_unroll.load(std::memory_order_relaxed);
        if (n == 0) {
            const auto vpu = rv64_rvjit_model().vpu;
            n = static_cast<dim_t>(
                    vpu.max_n_accumulators(rvjit::SEW::e32, rvjit::SEW::e32));
            rvv_gemm_f32_n_unroll.store(n, std::memory_order_relaxed);
        }
        return n;
    }
};

// Sum the m*n values from p_src into p_dst, assuming the two-dimensional
// arrays have leading dimensions ld_src and ld_dst, respectively
template <typename data_t>
void sum_two_matrices(dim_t m, dim_t n, data_t *__restrict p_src, dim_t ld_src,
        data_t *__restrict p_dst, dim_t ld_dst) {

    for (dim_t j = 0; j < n; j++) {
        for (dim_t i = 0; i < m; i++) {
            p_dst[i + j * ld_dst] += p_src[i + j * ld_src];
        }
    }
}

void calc_nthr_nocopy_rvv(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK);

void partition_unit_diff(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP
