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
#include "cpu/rv64/rvv_matmul.hpp"
#include "common/dnnl_thread.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

template <size_t I = 0, typename... Tp>
inline void unroll_fma_rowmajor(std::tuple<Tp...> &t, const float *s_ptr,
        const vfloat32m1_t &w_vec, dim_t K, dim_t k, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        std::get<I>(t) = __riscv_vfmacc_vf_f32m1(
                std::get<I>(t), s_ptr[I * K + k], w_vec, vl);
        unroll_fma_rowmajor<I + 1, Tp...>(t, s_ptr, w_vec, K, k, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_store(
        std::tuple<Tp...> &t, float *const *dst_ptrs, dim_t n, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        __riscv_vse32_v_f32m1(dst_ptrs[I] + n, std::get<I>(t), vl);
        unroll_store<I + 1, Tp...>(t, dst_ptrs, n, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_add_scalar_bias(
        std::tuple<Tp...> &t, float bias_scalar, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        std::get<I>(t)
                = __riscv_vfadd_vf_f32m1(std::get<I>(t), bias_scalar, vl);
        unroll_add_scalar_bias<I + 1, Tp...>(t, bias_scalar, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_add_scalar_bias_array(
        std::tuple<Tp...> &t, const float *bias_per_acc, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        std::get<I>(t)
                = __riscv_vfadd_vf_f32m1(std::get<I>(t), bias_per_acc[I], vl);
        unroll_add_scalar_bias_array<I + 1, Tp...>(t, bias_per_acc, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_add_vector_bias_ptrs(
        std::tuple<Tp...> &t, const float *const *bias_ptrs, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_ptrs[I], vl);
        std::get<I>(t) = __riscv_vfadd_vv_f32m1(std::get<I>(t), bias_vec, vl);
        unroll_add_vector_bias_ptrs<I + 1, Tp...>(t, bias_ptrs, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_apply_postops(
        std::tuple<Tp...> &t, const rvv_postops_t &postops_handler, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        std::get<I>(t) = postops_handler.apply(std::get<I>(t), vl);
        unroll_apply_postops<I + 1, Tp...>(t, postops_handler, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_fma_contiguous(std::tuple<Tp...> &t,
        const float *const *src_rows, const float *weights_col, dim_t k,
        size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        vfloat32m1_t src_vec = __riscv_vle32_v_f32m1(src_rows[I] + k, vl);
        vfloat32m1_t w_vec = __riscv_vle32_v_f32m1(weights_col + k, vl);
        std::get<I>(t)
                = __riscv_vfmacc_vv_f32m1(std::get<I>(t), src_vec, w_vec, vl);
        unroll_fma_contiguous<I + 1, Tp...>(t, src_rows, weights_col, k, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_init(std::tuple<Tp...> &t, size_t vl) {
    if constexpr (I < sizeof...(Tp)) {
        std::get<I>(t) = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        unroll_init<I + 1, Tp...>(t, vl);
    }
}

template <size_t I = 0, typename... Tp>
inline void unroll_reduce(const std::tuple<Tp...> &t, float *results, size_t vl,
        vfloat32m1_t zero_scalar) {
    if constexpr (I < sizeof...(Tp)) {
        vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(
                std::get<I>(t), zero_scalar, vl);
        results[I] = __riscv_vfmv_f_s_f32m1_f32(sum);
        unroll_reduce<I + 1, Tp...>(t, results, vl, zero_scalar);
    }
}

template <size_t I = 0>
inline void unroll_add_bias_scalar(float *results, float bias_val) {
    if constexpr (I < 12) {
        results[I] += bias_val;
        unroll_add_bias_scalar<I + 1>(results, bias_val);
    }
}

template <size_t I = 0>
inline void unroll_add_bias_array(float *results, const float *bias_vals) {
    if constexpr (I < 12) {
        results[I] += bias_vals[I];
        unroll_add_bias_array<I + 1>(results, bias_vals);
    }
}

void rvv_matmul_colmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    // MR = 12: Number of matrix rows processed simultaneously
    // - Uses 12 of 32 available RVV vector registers for accumulators
    // - Provides good instruction-level parallelism to hide FMA latency
    // - Leaves sufficient registers (~16-20) for temporaries and compiler optimization
    constexpr dim_t MR = 12;

    int bias_ndims = 0;
    std::vector<size_t> bias_strides;
    const dim_t *bias_dims = nullptr;
    if (bias) {
        bias_ndims = bias_d.ndims();
        bias_dims = bias_d.dims();
        bias_strides.resize(bias_ndims);
        if (bias_ndims > 0) {
            bias_strides[bias_ndims - 1] = 1;
            for (int d = bias_ndims - 2; d >= 0; --d)
                bias_strides[d]
                        = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
        }
    }

    parallel_nd(batch, M / MR, [&](dim_t b, dim_t m_blk) {
        dim_t m = m_blk * MR;

        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2)
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            dst_idx_prefix[ndims - 2] = m;
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        } else {
            dst_idx_prefix[ndims - 2] = m;
        }

        std::array<size_t, MR> base_bias_offs {};
        if (bias) {
            const int dst_ndims = dst_d.ndims();
            for (int i = 0; i < (int)MR; ++i) {
                size_t base_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx;
                    if (dst_dim_idx == (ndims - 2)) {
                        idx = (bias_dims[d] == 1) ? 0 : (m + i);
                    } else {
                        idx = (bias_dims[d] == 1) ? 0
                                                  : dst_idx_prefix[dst_dim_idx];
                    }
                    base_off += idx * bias_strides[d];
                }
                base_bias_offs[i] = base_off;
            }
        }

        const float *src_b = src + (size_t)b * M * K;
        float *dst_b = dst + (size_t)b * M * N;
        const float *weights_b = weights + weights_batch_offset;

        const float *src_rows[MR];
        for (int i = 0; i < (int)MR; ++i) {
            src_rows[i] = src_b + (m + i) * K;
        }

        for (dim_t n = 0; n < N; n++) {
            const float *weights_col = weights_b + n * K;

            vfloat32m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8,
                    acc9, acc10, acc11;
            auto acc_tuple = std::tie(acc0, acc1, acc2, acc3, acc4, acc5, acc6,
                    acc7, acc8, acc9, acc10, acc11);
            static_assert(std::tuple_size<decltype(acc_tuple)>::value == MR,
                    "Must match MR");

            size_t init_vl = __riscv_vsetvl_e32m1(K);
            unroll_init(acc_tuple, init_vl);

            for (dim_t k = 0; k < K;) {
                size_t vl = __riscv_vsetvl_e32m1(K - k);
                unroll_fma_contiguous(acc_tuple, src_rows, weights_col, k, vl);
                k += vl;
            }

            float results[MR];
            size_t vl_reduce = __riscv_vsetvl_e32m1(K);
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl_reduce);
            unroll_reduce(acc_tuple, results, vl_reduce, zero);

            if (bias) {
                if (bias_d.nelems() == 1) {
                    unroll_add_bias_scalar(results, bias[0]);
                } else if (bias_dims[bias_ndims - 1] == 1) {
                    float bias_per_row[MR];
                    for (int i = 0; i < (int)MR; ++i) {
                        bias_per_row[i] = bias[base_bias_offs[i]];
                    }
                    unroll_add_bias_array(results, bias_per_row);
                } else {
                    float bias_vals[MR];
                    for (int i = 0; i < (int)MR; ++i) {
                        bias_vals[i] = bias[base_bias_offs[i] + n];
                    }
                    unroll_add_bias_array(results, bias_vals);
                }
            }

            for (int off = 0; off < (int)MR;) {
                size_t vl_chunk = __riscv_vsetvl_e32m1((size_t)MR - off);
                vfloat32m1_t res_vec
                        = __riscv_vle32_v_f32m1(results + off, vl_chunk);
                res_vec = postops_handler.apply(res_vec, vl_chunk);
                __riscv_vse32_v_f32m1(results + off, res_vec, vl_chunk);

                off += (int)vl_chunk;
            }

            for (int i = 0; i < (int)MR; ++i) {
                dst_b[(m + i) * N + n] = results[i];
            }
        }
    });

    const dim_t m_tail_start = M - (M % MR);
    parallel_nd(batch, M % MR, [&](dim_t b, dim_t m_off) {
        dim_t m = m_tail_start + m_off;

        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        size_t base_bias_off = 0;
        if (bias) {
            const int dst_ndims = dst_d.ndims();
            for (int d = 0; d < bias_ndims - 1; ++d) {
                int dst_dim_idx = d + (dst_ndims - bias_ndims);
                dim_t idx
                        = (bias_dims[d] == 1) ? 0 : dst_idx_prefix[dst_dim_idx];
                base_bias_off += idx * bias_strides[d];
            }
        }

        const float *src_row = src + (size_t)b * M * K + m * K;
        float *dst_row = dst + (size_t)b * M * N + m * N;
        const float *weights_ptr = weights + weights_batch_offset;

        const size_t max_vl = __riscv_vsetvlmax_e32m1();
        std::vector<float> temp_results(max_vl);

        for (dim_t n = 0; n < N;) {
            size_t vl_n = __riscv_vsetvl_e32m1(N - n);

            for (size_t i = 0; i < vl_n; ++i) {
                const float *weights_col = weights_ptr + (n + i) * K;
                vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);

                for (dim_t k = 0; k < K;) {
                    size_t vl_k = __riscv_vsetvl_e32m1(K - k);

                    vfloat32m1_t src_vec
                            = __riscv_vle32_v_f32m1(src_row + k, vl_k);
                    vfloat32m1_t w_vec
                            = __riscv_vle32_v_f32m1(weights_col + k, vl_k);

                    acc = __riscv_vfredosum_vs_f32m1_f32m1(
                            __riscv_vfmul_vv_f32m1(src_vec, w_vec, vl_k), acc,
                            vl_k);

                    k += vl_k;
                }

                temp_results[i] = __riscv_vfmv_f_s_f32m1_f32(acc);
            }

            vfloat32m1_t result_vec
                    = __riscv_vle32_v_f32m1(temp_results.data(), vl_n);

            if (bias) {
                if (bias_d.nelems() == 1) {
                    result_vec
                            = __riscv_vfadd_vf_f32m1(result_vec, bias[0], vl_n);
                } else if (bias_dims[bias_ndims - 1] == 1) {
                    result_vec = __riscv_vfadd_vf_f32m1(
                            result_vec, bias[base_bias_off], vl_n);
                } else {
                    const float *bias_ptr = bias + base_bias_off + n;
                    vfloat32m1_t bias_vec
                            = __riscv_vle32_v_f32m1(bias_ptr, vl_n);
                    result_vec = __riscv_vfadd_vv_f32m1(
                            result_vec, bias_vec, vl_n);
                }
            }

            result_vec = postops_handler.apply(result_vec, vl_n);
            __riscv_vse32_v_f32m1(dst_row + n, result_vec, vl_n);

            n += vl_n;
        }
    });
}

void rvv_matmul_rowmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    // MR = 12: Number of matrix rows processed simultaneously
    // - Uses 12 of 32 available RVV vector registers for accumulators
    // - Provides good instruction-level parallelism to hide FMA latency
    // - Leaves sufficient registers (~16-20) for temporaries and compiler optimization
    constexpr dim_t MR = 12;

    int bias_ndims = 0;
    std::vector<size_t> bias_strides;
    const dim_t *bias_dims = nullptr;
    if (bias) {
        bias_ndims = bias_d.ndims();
        bias_dims = bias_d.dims();
        bias_strides.resize(bias_ndims);
        bias_strides[bias_ndims - 1] = 1;
        for (int d = bias_ndims - 2; d >= 0; --d)
            bias_strides[d] = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
    }

    parallel_nd(batch, M / MR, [&](dim_t b, dim_t m_blk) {
        dim_t m = m_blk * MR;

        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            dst_idx_prefix[ndims - 2] = m;
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        } else {
            dst_idx_prefix[ndims - 2] = m;
        }

        std::array<size_t, MR> base_bias_offs {};
        if (bias) {
            const int dst_ndims = dst_d.ndims();
            for (int i = 0; i < (int)MR; ++i) {
                size_t base_off = 0;
                for (int d = 0; d < bias_ndims - 1; ++d) {
                    int dst_dim_idx = d + (dst_ndims - bias_ndims);
                    dim_t idx;
                    if (dst_dim_idx == (ndims - 2)) {
                        idx = (bias_dims[d] == 1) ? 0 : (m + i);
                    } else {
                        idx = (bias_dims[d] == 1) ? 0
                                                  : dst_idx_prefix[dst_dim_idx];
                    }
                    base_off += idx * bias_strides[d];
                }
                base_bias_offs[i] = base_off;
            }
        }

        const float *src_b = src + (size_t)b * M * K;
        float *dst_b = dst + (size_t)b * M * N;
        const float *weights_b = weights + weights_batch_offset;

        for (dim_t n = 0; n < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n);

            vfloat32m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8,
                    acc9, acc10, acc11;
            auto acc_tuple = std::tie(acc0, acc1, acc2, acc3, acc4, acc5, acc6,
                    acc7, acc8, acc9, acc10, acc11);
            static_assert(std::tuple_size<decltype(acc_tuple)>::value == MR,
                    "Must match MR");

            unroll_init(acc_tuple, vl);

            const float *s_ptr_base = src_b + m * K;
            float *d_ptr_base = dst_b + m * N + n;

            for (dim_t k = 0; k < K; ++k) {
                const float *w_ptr = weights_b + k * N + n;
                vfloat32m1_t w_vec = __riscv_vle32_v_f32m1(w_ptr, vl);
                unroll_fma_rowmajor(acc_tuple, s_ptr_base, w_vec, K, k, vl);
            }

            if (bias) {
                if (bias_d.nelems() == 1) {
                    unroll_add_scalar_bias(acc_tuple, bias[0], vl);
                } else {
                    if (bias_dims[bias_ndims - 1] == 1) {
                        float per_acc_scalars[MR];
                        for (int i = 0; i < (int)MR; ++i) {
                            per_acc_scalars[i] = bias[base_bias_offs[i]];
                        }
                        unroll_add_scalar_bias_array(
                                acc_tuple, per_acc_scalars, vl);
                    } else {
                        const float *bias_ptrs_arr[MR];
                        for (int i = 0; i < (int)MR; ++i) {
                            bias_ptrs_arr[i] = bias + base_bias_offs[i] + n;
                        }
                        unroll_add_vector_bias_ptrs(
                                acc_tuple, bias_ptrs_arr, vl);
                    }
                }
            }

            unroll_apply_postops(acc_tuple, postops_handler, vl);

            float *dst_ptrs[MR];
            for (int i = 0; i < (int)MR; ++i) {
                dst_ptrs[i] = d_ptr_base + i * N;
            }
            unroll_store(acc_tuple, dst_ptrs, 0, vl);

            n += vl;
        }
    });

    const dim_t m_tail_start = M - (M % MR);
    parallel_nd(batch, M % MR, [&](dim_t b, dim_t m_off) {
        dim_t m = m_tail_start + m_off;

        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        size_t base_bias_off = 0;
        if (bias) {
            const int dst_ndims = dst_d.ndims();
            for (int d = 0; d < bias_ndims - 1; ++d) {
                int dst_dim_idx = d + (dst_ndims - bias_ndims);
                dim_t idx
                        = (bias_dims[d] == 1) ? 0 : dst_idx_prefix[dst_dim_idx];
                base_bias_off += idx * bias_strides[d];
            }
        }

        const float *src_row = src + (size_t)b * M * K + (size_t)m * K;
        float *dst_row = dst + (size_t)b * M * N + (size_t)m * N;
        const float *weights_ptr = weights + weights_batch_offset;

        for (dim_t n = 0; n < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (dim_t k = 0; k < K; ++k) {
                acc = __riscv_vfmacc_vf_f32m1(acc, src_row[k],
                        __riscv_vle32_v_f32m1(weights_ptr + k * N + n, vl), vl);
            }

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                } else {
                    if (bias_dims[bias_ndims - 1] == 1) {
                        acc = __riscv_vfadd_vf_f32m1(
                                acc, bias[base_bias_off], vl);
                    } else {
                        const float *bias_ptr = bias + base_bias_off + n;
                        vfloat32m1_t bias_vec
                                = __riscv_vle32_v_f32m1(bias_ptr, vl);
                        acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                    }
                }
            }

            acc = postops_handler.apply(acc, vl);
            __riscv_vse32_v_f32m1(dst_row + n, acc, vl);
            n += vl;
        }
    });
}

rvv_matmul_t::rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_matmul_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    if (pd()->is_col_major(weights_d)) {
        rvv_matmul_colmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    } else {
        rvv_matmul_rowmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    }

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl